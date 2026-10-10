// A fourth made-up teleop-walking-benchmark adapter for gaitkeeper's tests, shaped like the
// benchmark's falcon: it passes its command only while the command is nonzero and a warm-up
// has passed, feeds that as a flag, runs its gait clock only then (cos and sin apart), and
// rescales the command's direction to a speed set by the distance to the waypoint.
namespace toy4 {

constexpr int NUM_ACTIONS = 12;
constexpr int OBS_DIM = 12 + 3 + 3 + 1 + 2 + 1 + 12 + 12 + 3 + 1;
constexpr float ACTION_SCALE = 0.25f, PERIOD = 0.8f, DT = 0.02f;
constexpr double WARMUP_S = 0.5;
constexpr float POS_P = 2.0f, SPEED_NORM = 0.9f, VX_MIN = -0.6f, VX_MAX = 0.9f, VY_ABS = 0.5f;

__device__ const float D_DEFAULT[NUM_ACTIONS] = {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
                                                 -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f};
const float KPS[NUM_ACTIONS] = {100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20};
const float KDS[NUM_ACTIONS] = {2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f};
const policy_api::Limits LIMITS = {VX_MIN, VX_MAX, VY_ABS, 0.8, SPEED_NORM};

__global__ void k_obs(const float* q, const float* dq, const float* gyro, const float* gravity,
                      const float* cmd, const float* task, const float* last, float* clock,
                      float* obs, int warm, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const float* c = cmd + env * 3;
  float vx = c[0], vy = c[1];
  const float wz = c[2];
  const bool walking = vx != 0.0f || vy != 0.0f || wz != 0.0f;
  const float speed = sqrtf(vx * vx + vy * vy);
  if (speed > 1e-9f) {
    const float s = fminf(POS_P * task[env * 4], SPEED_NORM) / speed;
    vx = fminf(fmaxf(vx * s, VX_MIN), VX_MAX);
    vy = fminf(fmaxf(vy * s, -VY_ABS), VY_ABS);
  }
  const bool go = walking && warm != 0;
  if (go) clock[env] += DT;
  const float phase = fmodf(clock[env], PERIOD) / PERIOD;
  float* o = obs + env * OBS_DIM;
  int k = 0;
  for (int i = 0; i < NUM_ACTIONS; ++i) o[k++] = last[env * NUM_ACTIONS + i];
  for (int i = 0; i < 3; ++i) o[k++] = gyro[env * 3 + i] * 0.25f;
  o[k++] = go ? wz : 0.0f;
  o[k++] = go ? vx : 0.0f;
  o[k++] = go ? vy : 0.0f;
  o[k++] = go ? 1.0f : 0.0f;
  o[k++] = 0.0f;
  o[k++] = 0.0f;
  o[k++] = cosf(2.0f * float(M_PI) * phase);
  for (int i = 0; i < NUM_ACTIONS; ++i) o[k++] = q[env * POLICY_NUM_MOTOR + i] - D_DEFAULT[i];
  for (int i = 0; i < NUM_ACTIONS; ++i) o[k++] = dq[env * POLICY_NUM_MOTOR + i] * 0.05f;
  for (int i = 0; i < 3; ++i) o[k++] = gravity[env * 3 + i];
  o[k++] = sinf(2.0f * float(M_PI) * phase);
}

__global__ void k_act(const float* act, const float* arm_pose, float* last, float* q_target,
                      int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    last[env * NUM_ACTIONS + i] = act[env * NUM_ACTIONS + i];
    q_target[env * POLICY_NUM_MOTOR + i] = D_DEFAULT[i] + ACTION_SCALE * act[env * NUM_ACTIONS + i];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr, *d_clock = nullptr;
  int envs = 0;
  long long ticks = 0;
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last, (void*)d_clock})
      if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make("policies/toy4/model.onnx", n, OBS_DIM, NUM_ACTIONS);
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, OBS_DIM);
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_last, NUM_ACTIONS);
    zeros(&d_clock, 1);
  }
  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const int warm = double(ticks) * double(DT) >= WARMUP_S ? 1 : 0;
    ++ticks;
    k_obs<<<blocks, threads>>>(c.motor_q, c.motor_dq, c.gyro, c.gravity, c.cmd, c.task, d_last,
                               d_clock, d_obs, warm, envs);
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_act<<<blocks, threads>>>(d_act, c.arm_pose, d_last, c.q_target, envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACTIONS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "toy4"; }
};

}
