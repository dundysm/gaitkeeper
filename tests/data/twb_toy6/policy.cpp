// A sixth made-up teleop-walking-benchmark adapter for gaitkeeper's tests, shaped like the
// benchmark's asap: a walk latch on the task that gates the command, feeds a flag and stops
// the clock at phase zero; the harness's arm targets observed; and the current frame split
// around a history of the two frames before it, term by term, newest first.
namespace toy6 {

constexpr int NUM_ACTIONS = 12, OWNED = 15, ARMS = 3, HIST = 2;
constexpr int FRAME = 12 + 3 + 1 + 1 + 2 + 1 + 1 + 15 + 3 + ARMS + 1;
constexpr int HEAD = 12 + 3 + 1 + 1 + 2 + 1 + 1 + 15;
constexpr int NUM_OBS = FRAME + HIST * FRAME;
constexpr float DT = 0.02f, PERIOD = 0.8f, HEIGHT = 1.5f;
constexpr float ENTER_D = 0.1f, ENTER_Y = 0.12f, EXIT_D = 0.05f, EXIT_Y = 0.06f;
__device__ const int SEG[11][2] = {{0, 12}, {12, 15}, {15, 16}, {16, 17}, {17, 19}, {19, 20},
                                   {20, 21}, {21, 36}, {36, 39}, {39, 42}, {42, 43}};

__device__ const float D_DEFAULT[OWNED] = {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
                                           -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, 0, 0, 0};
const float KPS[OWNED] = {100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 400, 400, 400};
const float KDS[OWNED] = {2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 5, 5, 5};
const policy_api::Limits LIMITS = {-1.0, 1.0, 0.8, 0.8, 0.0};

__global__ void k_obs(const float* q, const float* dq, const float* gyro, const float* gravity,
                      const float* cmd, const float* task, const float* arm_pose,
                      const float* last, unsigned char* latch, float* hist, float* obs,
                      float time_s, int first, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const float* c = cmd + env * 3;
  const float dist = task[env * 4], yaw = fabsf(task[env * 4 + 1]);
  if (latch[env]) {
    if (dist < EXIT_D && yaw < EXIT_Y) latch[env] = 0;
  } else if (dist > ENTER_D || yaw > ENTER_Y) {
    latch[env] = 1;
  }
  const bool walk = latch[env] != 0;
  const float phase = walk ? fmodf(time_s, PERIOD) / PERIOD : 0.0f;
  float f[FRAME];
  int k = 0;
  for (int i = 0; i < NUM_ACTIONS; ++i) f[k++] = last[env * NUM_ACTIONS + i];
  for (int i = 0; i < 3; ++i) f[k++] = gyro[env * 3 + i] * 0.25f;
  f[k++] = walk ? c[2] : 0.0f;
  f[k++] = HEIGHT;
  f[k++] = walk ? c[0] : 0.0f;
  f[k++] = walk ? c[1] : 0.0f;
  f[k++] = walk ? 1.0f : 0.0f;
  f[k++] = cosf(2.0f * float(M_PI) * phase);
  for (int j = 0; j < OWNED; ++j) f[k++] = q[env * POLICY_NUM_MOTOR + j] - D_DEFAULT[j];
  for (int i = 0; i < 3; ++i) f[k++] = gravity[env * 3 + i];
  for (int i = 0; i < ARMS; ++i) f[k++] = arm_pose[env * POLICY_NUM_MOTOR + OWNED + i];
  f[k++] = sinf(2.0f * float(M_PI) * phase);
  float* h = hist + env * HIST * FRAME;
  if (first)
    for (int t = 0; t < HIST; ++t)
      for (int i = 0; i < FRAME; ++i) h[t * FRAME + i] = f[i];
  float* o = obs + env * NUM_OBS;
  int at = 0;
  for (int i = 0; i < HEAD; ++i) o[at++] = f[i];
  for (int s = 0; s < 11; ++s)
    for (int t = 0; t < HIST; ++t)
      for (int i = SEG[s][0]; i < SEG[s][1]; ++i) o[at++] = h[t * FRAME + i];
  for (int i = HEAD; i < FRAME; ++i) o[at++] = f[i];
  for (int t = HIST - 1; t > 0; --t)
    for (int i = 0; i < FRAME; ++i) h[t * FRAME + i] = h[(t - 1) * FRAME + i];
  for (int i = 0; i < FRAME; ++i) h[i] = f[i];
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
  for (int i = NUM_ACTIONS; i < OWNED; ++i) q_target[env * POLICY_NUM_MOTOR + i] = D_DEFAULT[i];
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr, *d_hist = nullptr;
  unsigned char* d_latch = nullptr;
  int envs = 0;
  long long ticks = 0;
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last, (void*)d_hist, (void*)d_latch})
      if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make("policies/toy6/model.onnx", n, NUM_OBS, NUM_ACTIONS);
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, NUM_OBS);
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_last, NUM_ACTIONS);
    zeros(&d_hist, HIST * FRAME);
    cudaMalloc(&d_latch, size_t(n));
    cudaMemset(d_latch, 0, size_t(n));
  }
  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const float time_s = float(ticks) * DT;
    const int first = ticks == 0 ? 1 : 0;
    ++ticks;
    k_obs<<<blocks, threads>>>(c.motor_q, c.motor_dq, c.gyro, c.gravity, c.cmd, c.task,
                               c.arm_pose, d_last, d_latch, d_hist, d_obs, time_s, first, envs);
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_act<<<blocks, threads>>>(d_act, c.arm_pose, d_last, c.q_target, envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "toy6"; }
};

}
