// A seventh made-up teleop-walking-benchmark adapter for gaitkeeper's tests, shaped like the
// benchmark's handoff: the current frame in front of a history that holds it too, a command
// zeroed below some size, a two-leg clock that reads phase zero then (running on underneath),
// and roll and pitch computed from gravity.
namespace toy7 {

constexpr int NUM_ACTIONS = 12, HIST = 3;
constexpr int FRAME = 2 + 1 + 1 + 4 + 3 + 2 + 12 + 12 + 12;
constexpr int NUM_OBS = FRAME * (1 + HIST);
constexpr float DT = 0.02f, PERIOD = 0.9f, STAND = 0.15f;

__device__ const float D_DEFAULT[NUM_ACTIONS] = {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
                                                 -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f};
const float KPS[NUM_ACTIONS] = {100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20};
const float KDS[NUM_ACTIONS] = {2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f};
const policy_api::Limits LIMITS = {-1.0, 1.0, 1.0, 1.0, 0.0};

__global__ void k_obs(const float* q, const float* dq, const float* gyro, const float* gravity,
                      const float* cmd, const float* last, float* hist, float* obs, float phase,
                      int primed, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const float* c = cmd + env * 3;
  const bool still = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) < STAND;
  float f[FRAME];
  int k = 0;
  f[k++] = still ? 0.0f : c[0];
  f[k++] = still ? 0.0f : c[1];
  f[k++] = 0.75f;
  f[k++] = still ? 0.0f : c[2];
  const float l = still ? 0.0f : phase, r = still ? 0.0f : fmodf(phase + 0.5f, 1.0f);
  f[k++] = sinf(2.0f * float(M_PI) * l);
  f[k++] = cosf(2.0f * float(M_PI) * l);
  f[k++] = sinf(2.0f * float(M_PI) * r);
  f[k++] = cosf(2.0f * float(M_PI) * r);
  for (int i = 0; i < 3; ++i) f[k++] = gyro[env * 3 + i] * 0.25f;
  const float* g = gravity + env * 3;
  f[k++] = atan2f(-g[1], -g[2]);
  f[k++] = asinf(fminf(fmaxf(g[0], -1.0f), 1.0f));
  for (int j = 0; j < NUM_ACTIONS; ++j) f[k++] = q[env * POLICY_NUM_MOTOR + j] - D_DEFAULT[j];
  for (int j = 0; j < NUM_ACTIONS; ++j) f[k++] = dq[env * POLICY_NUM_MOTOR + j] * 0.05f;
  for (int j = 0; j < NUM_ACTIONS; ++j) f[k++] = last[env * NUM_ACTIONS + j];
  float* h = hist + env * HIST * FRAME;
  if (primed) {
    for (int t = 0; t + 1 < HIST; ++t)
      for (int i = 0; i < FRAME; ++i) h[t * FRAME + i] = h[(t + 1) * FRAME + i];
    for (int i = 0; i < FRAME; ++i) h[(HIST - 1) * FRAME + i] = f[i];
  } else {
    for (int t = 0; t < HIST; ++t)
      for (int i = 0; i < FRAME; ++i) h[t * FRAME + i] = f[i];
  }
  float* o = obs + env * NUM_OBS;
  for (int i = 0; i < FRAME; ++i) o[i] = f[i];
  for (int t = 0; t < HIST; ++t)
    for (int i = 0; i < FRAME; ++i) o[FRAME * (1 + t) + i] = h[t * FRAME + i];
}

__global__ void k_act(const float* act, const float* arm_pose, float* last, float* q_target,
                      int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    last[env * NUM_ACTIONS + i] = act[env * NUM_ACTIONS + i];
    q_target[env * POLICY_NUM_MOTOR + i] = D_DEFAULT[i] + 0.25f * act[env * NUM_ACTIONS + i];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr, *d_hist = nullptr;
  int envs = 0;
  long long step_index = 0;
  bool primed = false;
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last, (void*)d_hist})
      if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make("policies/toy7/model.onnx", n, NUM_OBS, NUM_ACTIONS);
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, NUM_OBS);
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_last, NUM_ACTIONS);
    zeros(&d_hist, HIST * FRAME);
  }
  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const float phase = float(fmod(double(step_index) * double(DT) / double(PERIOD), 1.0));
    k_obs<<<blocks, threads>>>(c.motor_q, c.motor_dq, c.gyro, c.gravity, c.cmd, d_last, d_hist,
                               d_obs, phase, primed ? 1 : 0, envs);
    primed = true;
    ++step_index;
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_act<<<blocks, threads>>>(d_act, c.arm_pose, d_last, c.q_target, envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACTIONS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "toy7"; }
};

}
