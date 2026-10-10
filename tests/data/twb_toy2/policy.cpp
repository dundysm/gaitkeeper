// A second made-up teleop-walking-benchmark adapter for gaitkeeper's tests, shaped like a
// port that keeps its own state: it observes the action from two steps back, joint
// velocities by difference of positions, a clock whose period follows the command speed
// (held while standing), zero padding in every frame of a three-frame history, and a
// command it steers itself from the harness's task (a waypoint follower).
namespace toy2 {

constexpr int NUM_ACTIONS = 12;
constexpr int FRAME = 12 + 3 + 1 + 12 + 12 + 3 + 2 + 3 + 2;
constexpr int HISTORY = 3;
constexpr float CONTROL_DT = 0.02f;
constexpr float ACTION_SCALE = 0.4f;
constexpr float SLOW = 1.0f, FAST = 0.6f, PMIN = 0.5f, CAP = 1.0f, START = 0.2f, SPAN = 0.5f;
constexpr float STAND = 0.15f;
constexpr float FAR = 2.0f, NEAR = 0.5f, WALK = 0.6f, WALK_P = 1.2f, YAW_P = 1.0f;
constexpr float VY_ABS = 0.25f, YAW_ABS = 0.5f;

__device__ const float D_DEFAULT[NUM_ACTIONS] = {-0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,
                                                 -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f};
const float KPS[15] = {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40, 300, 300, 300};
const float KDS[15] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5};
const policy_api::Limits LIMITS = {-0.6f, 0.9f, VY_ABS, YAW_ABS, 0.6f};

__device__ inline float wrap(float a) { return a - 2.0f * float(M_PI) * rintf(a / (2.0f * float(M_PI))); }

__device__ inline float period_of(float s) {
  const float t = (fminf(s, CAP) - START) / SPAN;
  return fmaxf(PMIN, SLOW + (FAST - SLOW) * fmaxf(t, 0.0f));
}

__device__ void steer(const float* c, const float* t, float* out) {
  out[0] = c[0]; out[1] = c[1]; out[2] = c[2];
  const float dist = t[0], yaw_err = t[1];
  if (!(dist > 0.0f)) return;
  const bool pos_ok = sqrtf(c[0] * c[0] + c[1] * c[1]) < 1e-9f;
  const bool yaw_ok = fabsf(c[2]) < 1e-9f;
  float bearing = 0.0f, w = 0.0f;
  if (!pos_ok) {
    bearing = atan2f(t[3], t[2]);
    w = fminf(fmaxf((dist - NEAR) / (FAR - NEAR), 0.0f), 1.0f);
    const float speed = fmaxf(cosf(bearing), 0.0f) * fminf(WALK_P * dist, WALK);
    out[0] = speed;
    out[1] = fminf(fmaxf(speed * sinf(bearing), -VY_ABS), VY_ABS);
  }
  const float aim = w > 0.0f ? wrap(yaw_err + w * wrap(bearing - yaw_err)) : (yaw_ok ? 0.0f : yaw_err);
  out[2] = fminf(fmaxf(YAW_P * aim, -YAW_ABS), YAW_ABS);
}

__global__ void k_obs(const float* q, const float* gyro, const float* gravity, const float* cmd,
                      const float* task, const float* lag2, const float* phase, float* prev_q,
                      float* drive, float* obs, int first, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  float d[3];
  steer(cmd + env * 3, task + env * 4, d);
  for (int k = 0; k < 3; ++k) drive[env * 3 + k] = d[k];
  float f[FRAME];
  for (int i = 0; i < FRAME; ++i) f[i] = 0.0f;
  for (int i = 0; i < NUM_ACTIONS; ++i) f[i] = lag2[env * NUM_ACTIONS + i];
  for (int k = 0; k < 3; ++k) f[12 + k] = d[k];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const float qi = q[env * POLICY_NUM_MOTOR + i];
    f[16 + i] = qi - D_DEFAULT[i];
    f[28 + i] = first ? 0.0f : (qi - prev_q[env * NUM_ACTIONS + i]) / CONTROL_DT;
    prev_q[env * NUM_ACTIONS + i] = qi;
  }
  for (int k = 0; k < 3; ++k) f[40 + k] = gravity[env * 3 + k];
  f[43] = sinf(2.0f * float(M_PI) * phase[env]);
  f[44] = cosf(2.0f * float(M_PI) * phase[env]);
  for (int k = 0; k < 3; ++k) f[45 + k] = gyro[env * 3 + k] * 0.25f;
  float* o = obs + env * FRAME * HISTORY;
  if (first) {
    for (int h = 0; h < HISTORY; ++h)
      for (int i = 0; i < FRAME; ++i) o[h * FRAME + i] = f[i];
  } else {
    for (int i = 0; i < FRAME * (HISTORY - 1); ++i) o[i] = o[i + FRAME];
    for (int i = 0; i < FRAME; ++i) o[FRAME * (HISTORY - 1) + i] = f[i];
  }
}

__global__ void k_act(const float* act, const float* arm_pose, const float* drive, float* lag1,
                      float* lag2, float* phase, float* q_target, int first, int envs) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int i = 0; i < NUM_ACTIONS; ++i)
    q_target[env * POLICY_NUM_MOTOR + i] = D_DEFAULT[i] + ACTION_SCALE * act[env * NUM_ACTIONS + i];
  const float* d = drive + env * 3;
  const float s = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (!first && s >= STAND) phase[env] = fmodf(phase[env] + CONTROL_DT / period_of(s), 1.0f);
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    lag2[env * NUM_ACTIONS + i] = lag1[env * NUM_ACTIONS + i];
    lag1[env * NUM_ACTIONS + i] = act[env * NUM_ACTIONS + i];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_l1 = nullptr, *d_l2 = nullptr;
  float *d_pq = nullptr, *d_ph = nullptr, *d_drive = nullptr;
  int envs = 0;
  long long ticks = 0;
  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_l1, (void*)d_l2, (void*)d_pq,
                    (void*)d_ph, (void*)d_drive})
      if (p) cudaFree(p);
  }
  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make("policies/toy2/model.onnx", n, FRAME * HISTORY, NUM_ACTIONS);
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, FRAME * HISTORY);
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_l1, NUM_ACTIONS);
    zeros(&d_l2, NUM_ACTIONS);
    zeros(&d_pq, NUM_ACTIONS);
    zeros(&d_ph, 1);
    zeros(&d_drive, 3);
  }
  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const int first = ticks == 0 ? 1 : 0;
    ++ticks;
    k_obs<<<blocks, threads>>>(c.motor_q, c.gyro, c.gravity, c.cmd, c.task, d_l2, d_ph, d_pq,
                               d_drive, d_obs, first, envs);
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_act<<<blocks, threads>>>(d_act, c.arm_pose, d_drive, d_l1, d_l2, d_ph, c.q_target, first,
                               envs);
  }
  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACTIONS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "toy2"; }
};

}
