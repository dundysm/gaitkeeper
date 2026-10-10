// A fifth made-up teleop-walking-benchmark adapter for gaitkeeper's tests, shaped like the
// benchmark's openwbt: a recurrent graph and clock inputs per foot as walk-these-ways builds
// them (each foot's phase warped so its stance fills half a cycle), held at one phase while
// the command is exactly zero.
namespace toy5 {

constexpr int NUM_ACTIONS = 12;
constexpr int NUM_OBS = 3 + 3 + 3 + 12 + 12 + 12 + 2;
constexpr int HIDDEN = 8;
constexpr double DT = 0.02, FREQ = 1.25, OFFSET = 0.5, RATIO = 0.55, START = 0.2, HOLD = 0.35;

__device__ const float D_DEFAULT[NUM_ACTIONS] = {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
                                                 -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f};
const float KPS[15] = {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 300, 300, 300};
const float KDS[15] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 3, 3, 3};
const policy_api::Limits LIMITS = {-0.3, 0.3, 0.3, 0.3, 0.0};

__global__ void k_obs(const float* q, const float* dq, const float* gyro, const float* gravity,
                      const float* cmd, const float* last, double* gait, float* obs, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const float* c = cmd + env * 3;
  double g = fmod(gait[env] + DT * FREQ, 1.0);
  double foot[2] = {fmod(g + OFFSET, 1.0), g};
  if (c[0] == 0.0f && c[1] == 0.0f && c[2] == 0.0f) {
    g = HOLD;
    foot[0] = HOLD;
    foot[1] = HOLD;
  }
  gait[env] = g;
  float* o = obs + env * NUM_OBS;
  for (int k = 0; k < 3; ++k) {
    o[k] = c[k] * 2.0f;
    o[3 + k] = gravity[env * 3 + k];
    o[6 + k] = gyro[env * 3 + k] * 0.25f;
  }
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    o[9 + j] = q[env * POLICY_NUM_MOTOR + j] - D_DEFAULT[j];
    o[21 + j] = dq[env * POLICY_NUM_MOTOR + j] * 0.05f;
    o[33 + j] = last[env * NUM_ACTIONS + j];
  }
  for (int i = 0; i < 2; ++i) {
    const double x = foot[i];
    const double w = x < RATIO ? 0.5 * x / RATIO : 0.5 + 0.5 * (x - RATIO) / (1.0 - RATIO);
    o[45 + i] = float(sin(2.0 * M_PI * w));
  }
}

__global__ void k_act(const float* act, const float* arm_pose, float* last, float* q_target,
                      int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    last[env * NUM_ACTIONS + j] = act[env * NUM_ACTIONS + j];
    q_target[env * POLICY_NUM_MOTOR + j] = D_DEFAULT[j] + 0.25f * act[env * NUM_ACTIONS + j];
  }
  for (int j = 12; j < 15; ++j) q_target[env * POLICY_NUM_MOTOR + j] = 0.0f;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr, *d_h_in = nullptr, *d_h_out = nullptr;
  double* d_gait = nullptr;
  int envs = 0;
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last, (void*)d_h_in, (void*)d_h_out,
                    (void*)d_gait})
      if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/toy5/model.onnx", n,
        {{"obs", {-1, NUM_OBS}}, {"h_in", {1, -1, HIDDEN}}},
        {{"actions", {-1, NUM_ACTIONS}}, {"h_out", {1, -1, HIDDEN}}});
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, NUM_OBS);
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_last, NUM_ACTIONS);
    zeros(&d_h_in, HIDDEN);
    zeros(&d_h_out, HIDDEN);
    cudaMalloc(&d_gait, size_t(n) * sizeof(double));
    std::vector<double> g(size_t(n), START);
    cudaMemcpy(d_gait, g.data(), g.size() * sizeof(double), cudaMemcpyHostToDevice);
  }
  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_obs<<<blocks, threads>>>(c.motor_q, c.motor_dq, c.gyro, c.gravity, c.cmd, d_last, d_gait,
                               d_obs, envs);
    const float* in[2] = {d_obs, d_h_in};
    float* out[2] = {d_act, d_h_out};
    policy_api::engine_run(*engine, in, out, envs);
    std::swap(d_h_in, d_h_out);
    k_act<<<blocks, threads>>>(d_act, c.arm_pose, d_last, c.q_target, envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return 15; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "toy5"; }
};

}
