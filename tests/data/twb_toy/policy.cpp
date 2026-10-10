// A made-up teleop-walking-benchmark adapter for gaitkeeper's tests: legs and waist in a
// permuted order, the command fed as [wz, vx, vy], a constant height slot, a gated clock
// advanced before use, actions clipped to +-5.
namespace toy {

constexpr int NUM_ACTIONS = 15;
constexpr int NUM_OBS = 3 + 3 + 3 + 1 + 2 + 3 * NUM_ACTIONS;
constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_CLIP = 5.0f;
constexpr float GAIT_PERIOD = 0.8f;
constexpr float CONTROL_DT = 0.02f;

__device__ const int D_MOTOR[NUM_ACTIONS] = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 12, 13, 14};
__device__ const float D_DEFAULT[NUM_ACTIONS] = {-0.1f, -0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f,
                                                 -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
const float KPS[NUM_ACTIONS] = {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 200, 200, 200};
const float KDS[NUM_ACTIONS] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5};
const policy_api::Limits LIMITS = {-0.5, 1.0, 0.3, 0.5, 0.0};

__global__ void k_obs(const float* q, const float* dq, const float* gyro, const float* gravity,
                      const float* cmd, const float* last, float phase, float* obs, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  float* o = obs + env * NUM_OBS;
  for (int k = 0; k < 3; ++k) {
    o[k] = gyro[env * 3 + k] * 0.25f;
    o[3 + k] = gravity[env * 3 + k];
  }
  const float* c = cmd + env * 3;
  o[6] = c[2];
  o[7] = c[0];
  o[8] = c[1];
  o[9] = 0.75f;
  const bool on = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) >= 0.1f;
  o[10] = on ? sinf(2.0f * float(M_PI) * phase) : 0.0f;
  o[11] = on ? cosf(2.0f * float(M_PI) * phase) : 0.0f;
  for (int k = 0; k < NUM_ACTIONS; ++k) {
    const int m = D_MOTOR[k];
    o[12 + k] = q[env * POLICY_NUM_MOTOR + m] - D_DEFAULT[k];
    o[12 + NUM_ACTIONS + k] = dq[env * POLICY_NUM_MOTOR + m] * 0.05f;
    o[12 + 2 * NUM_ACTIONS + k] = last[env * NUM_ACTIONS + k];
  }
}

__global__ void k_act(const float* act, const float* arm_pose, float* last, float* q_target,
                      int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int k = 0; k < NUM_ACTIONS; ++k) {
    const float a = fminf(fmaxf(act[env * NUM_ACTIONS + k], -ACTION_CLIP), ACTION_CLIP);
    last[env * NUM_ACTIONS + k] = a;
    q_target[env * POLICY_NUM_MOTOR + D_MOTOR[k]] = D_DEFAULT[k] + a * ACTION_SCALE;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  double phase = 0.0;
  int envs = 0;
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make("policies/toy/model.onnx", n, NUM_OBS, NUM_ACTIONS);
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
  }
  void step(const policy_api::Ctx& c) override {
    phase = std::fmod(phase + CONTROL_DT / GAIT_PERIOD, 1.0);
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_obs<<<blocks, threads>>>(c.motor_q, c.motor_dq, c.gyro, c.gravity, c.cmd, d_last,
                               float(phase), d_obs, envs);
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_act<<<blocks, threads>>>(d_act, c.arm_pose, d_last, c.q_target, envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACTIONS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "toy"; }
};

}
