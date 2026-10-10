// A third made-up teleop-walking-benchmark adapter for gaitkeeper's tests, shaped like the
// benchmark's homie and gr00t_wbc: variants made by name, two graphs picked by the command, a
// waist the port holds and observes, no observation until the second step, and a command the
// port makes from the task (its direction, at a speed set by the distance, and its own yaw).
namespace toy3 {

constexpr int NUM_ACTIONS = 12;
constexpr int OWNED = 15;
constexpr int FRAME = 3 + 1 + 3 + 3 + 13 + 13 + 12;
constexpr int HISTORY = 2;
constexpr float ACTION_SCALE = 0.25f;
constexpr float POS_P = 0.8f, SPEED = 0.4f, VX_MIN = -0.25f, VX_MAX = 0.4f, VY_ABS = 0.25f;
constexpr float YAW_P = 1.2f, YAW_ABS = 0.8f, NEAR = 0.35f, FAR = 1.0f, SWITCH = 0.05f;
constexpr float TWO_PI = 2.0f * float(M_PI);

struct Variant {
  const char* name;
  float height;
};
constexpr Variant VARIANTS[] = {{"toy3_low", 0.7f}, {"toy3_high", 0.78f}};

__device__ const float D_DEFAULT[NUM_ACTIONS] = {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
                                                 -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f};
const float KPS[OWNED] = {150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 300, 300, 300};
const float KDS[OWNED] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5};
const policy_api::Limits LIMITS = {-0.4f, 0.4f, 0.4f, YAW_ABS, SPEED};

__device__ void drive(const float* c, const float* t, float* d) {
  d[0] = c[0];
  d[1] = c[1];
  d[2] = c[2];
  const float dist = t[0], yaw_err = t[1];
  const bool moving = c[0] != 0.0f || c[1] != 0.0f;
  float bearing = 0.0f, w = 0.0f;
  if (moving) {
    bearing = atan2f(c[1], c[0]);
    w = fminf(fmaxf((dist - NEAR) / (FAR - NEAR), 0.0f), 1.0f);
    const float s = fminf(POS_P * dist, SPEED);
    d[0] = fminf(fmaxf(s * cosf(bearing), VX_MIN), VX_MAX);
    d[1] = fminf(fmaxf(s * sinf(bearing), -VY_ABS), VY_ABS);
  }
  if (w > 0.0f) {
    const float aim = remainderf(yaw_err + w * remainderf(bearing - yaw_err, TWO_PI), TWO_PI);
    d[2] = fminf(fmaxf(YAW_P * aim, -YAW_ABS), YAW_ABS);
  } else if (c[2] != 0.0f) {
    d[2] = fminf(fmaxf(YAW_P * yaw_err, -YAW_ABS), YAW_ABS);
  }
}

__global__ void k_obs(const float* q, const float* dq, const float* gyro, const float* gravity,
                      const float* cmd, const float* task, const float* last, float* obs,
                      float height, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  float d[3];
  drive(cmd + env * 3, task + env * 4, d);
  float f[FRAME];
  int k = 0;
  f[k++] = d[0] * 2.0f;
  f[k++] = d[1] * 2.0f;
  f[k++] = d[2] * 0.25f;
  f[k++] = height;
  for (int i = 0; i < 3; ++i) f[k++] = gyro[env * 3 + i] * 0.5f;
  for (int i = 0; i < 3; ++i) f[k++] = gravity[env * 3 + i];
  for (int i = 0; i < 13; ++i)
    f[k++] = q[env * POLICY_NUM_MOTOR + i] - (i < NUM_ACTIONS ? D_DEFAULT[i] : 0.0f);
  for (int i = 0; i < 13; ++i) f[k++] = dq[env * POLICY_NUM_MOTOR + i] * 0.05f;
  for (int i = 0; i < NUM_ACTIONS; ++i) f[k++] = last[env * NUM_ACTIONS + i];
  float* o = obs + env * FRAME * HISTORY;
  for (int i = 0; i < FRAME * (HISTORY - 1); ++i) o[i] = o[i + FRAME];
  for (int i = 0; i < FRAME; ++i) o[FRAME * (HISTORY - 1) + i] = f[i];
}

__global__ void k_act(const float* walk, const float* stand, const float* cmd,
                      const float* arm_pose, float* last, float* q_target, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  const float* c = cmd + env * 3;
  const float norm = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  const float* a = (norm > SWITCH ? walk : stand) + env * NUM_ACTIONS;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    last[env * NUM_ACTIONS + i] = a[i];
    q_target[env * POLICY_NUM_MOTOR + i] = D_DEFAULT[i] + ACTION_SCALE * a[i];
  }
  for (int j = NUM_ACTIONS; j < OWNED; ++j) q_target[env * POLICY_NUM_MOTOR + j] = 0.0f;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> walk, stand;
  float *d_obs = nullptr, *d_walk = nullptr, *d_stand = nullptr, *d_last = nullptr;
  int envs = 0;
  bool primed = false;
  const Variant variant;
  explicit Policy(const Variant& v) : variant(v) {}
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_walk, (void*)d_stand, (void*)d_last})
      if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    // a port that writes beside the benchmark (a patched model, say) and reads it back
    std::filesystem::create_directories("build/trt");
    std::ofstream("build/trt/toy3.txt") << "toy3";
    if (!std::ifstream("build/trt/toy3.txt")) throw std::runtime_error("toy3: no build/trt");
    walk = policy_api::engine_make("policies/toy3/model_walk.onnx", n, FRAME * HISTORY, NUM_ACTIONS);
    stand = policy_api::engine_make("policies/toy3/model_stand.onnx", n, FRAME * HISTORY, NUM_ACTIONS);
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, FRAME * HISTORY);
    zeros(&d_walk, NUM_ACTIONS);
    zeros(&d_stand, NUM_ACTIONS);
    zeros(&d_last, NUM_ACTIONS);
  }
  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    if (primed)
      k_obs<<<blocks, threads>>>(c.motor_q, c.motor_dq, c.gyro, c.gravity, c.cmd, c.task,
                                 d_last, d_obs, variant.height, envs);
    primed = true;
    policy_api::engine_run(*walk, d_obs, d_walk, envs);
    policy_api::engine_run(*stand, d_obs, d_stand, envs);
    k_act<<<blocks, threads>>>(d_walk, d_stand, c.cmd, c.arm_pose, d_last, c.q_target, envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return variant.name; }
};

std::vector<std::string> names() {
  std::vector<std::string> all;
  for (const Variant& v : VARIANTS) all.emplace_back(v.name);
  return all;
}

std::unique_ptr<policy_api::Policy> make(const std::string& name) {
  for (const Variant& v : VARIANTS)
    if (name == v.name) return std::make_unique<Policy>(v);
  return nullptr;
}

}
