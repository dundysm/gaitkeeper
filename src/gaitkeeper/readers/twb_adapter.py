"""Run a teleop-walking-benchmark policy adapter (policies/<name>/policy.cpp) on the CPU.

The benchmark's adapters are CUDA kernels behind a small C++ interface (policy_api::Policy:
``init``, ``step(Ctx)``, ``kp``, ``kd``, ``owned``, ``limits``). Rather than parse them, this
module compiles one with a shim that runs its kernels as plain loops over the launch grid
and replaces the TensorRT engine with a recorder: ``engine_run`` keeps the observation it
is given and returns an action the caller chose. ``probe.py`` drives the result to read the
adapter's observation layout, action map and constants by experiment.

Needs a C++17 compiler (``g++`` or ``clang++``, or ``$CXX``). The compiled library is cached
under the gaitkeeper cache directory, keyed by the adapter's source.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

NUM_MOTOR = 29

_PROLOGUE = r"""
#define _USE_MATH_DEFINES
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
using std::isfinite;

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __constant__
struct dim3 {
  unsigned x = 1, y = 1, z = 1;
  dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {}
};
static thread_local dim3 blockIdx(0, 0, 0), threadIdx(0, 0, 0), blockDim, gridDim;
#define GK_LAUNCH(B, T, ...)                                                   \
  do {                                                                         \
    const dim3 gk_b = dim3(B), gk_t = dim3(T);                                 \
    gridDim = gk_b;                                                            \
    blockDim = gk_t;                                                           \
    for (unsigned bz = 0; bz < gk_b.z; ++bz)                                   \
      for (unsigned by = 0; by < gk_b.y; ++by)                                 \
        for (unsigned bx = 0; bx < gk_b.x; ++bx)                               \
          for (unsigned tz = 0; tz < gk_t.z; ++tz)                             \
            for (unsigned ty = 0; ty < gk_t.y; ++ty)                           \
              for (unsigned tx = 0; tx < gk_t.x; ++tx) {                       \
                blockIdx = dim3(bx, by, bz);                                   \
                threadIdx = dim3(tx, ty, tz);                                  \
                __VA_ARGS__;                                                   \
              }                                                                \
  } while (0)

typedef int cudaError_t;
typedef void* cudaStream_t;
enum cudaMemcpyKind {
  cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice, cudaMemcpyDefault
};
template <class T> static int cudaMalloc(T** p, size_t n) {
  *p = (T*)std::calloc(n ? n : 1, 1);
  return 0;
}
static int cudaFree(void* p) { std::free(p); return 0; }
static int cudaMemset(void* p, int v, size_t n) { std::memset(p, v, n); return 0; }
static int cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
  std::memmove(d, s, n);
  return 0;
}
static int cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind, cudaStream_t = 0) {
  std::memmove(d, s, n);
  return 0;
}
static int cudaDeviceSynchronize() { return 0; }
static int cudaGetLastError() { return 0; }
template <class T> static T __ldg(const T* p) { return *p; }

namespace policy_api {
inline constexpr int NUM_MOTOR = 29;
struct Limits {
  double vx_min, vx_max, vy_abs, yaw_rate_abs, speed_norm;
  double pos_reached_enter_m = 0.10;
  double pos_reached_exit_m = 0.20;
  double yaw_reached_enter_rad = 0.05;
  double yaw_reached_exit_rad = 0.12;
  double walk_kp_pos = 1.5;
  double walk_kp_yaw = 1.5;
};
struct Ctx {
  const float* motor_q;
  const float* motor_dq;
  const float* gyro;
  const float* base_lin_vel;
  const float* gravity;
  const float* cmd;
  const float* task;
  const float* arm_pose;
  const float* base_quat;
  float* q_target;
};
struct Policy {
  virtual ~Policy() = default;
  virtual void init(int envs) = 0;
  virtual void step(const Ctx& c) = 0;
  virtual const float* kp() const = 0;
  virtual const float* kd() const = 0;
  virtual int owned() const = 0;
  virtual Limits limits() const = 0;
  virtual const char* name() const = 0;
};
struct TensorSpec {
  std::string name;
  std::vector<int> shape;
};
struct Engine {
  std::string path;
  std::vector<int> in_sizes, out_sizes;  // per batch row
  std::vector<std::vector<float>> seen;  // inputs of the last run, concatenated per input
  int runs = 0;
  int index = 0;
};
static std::vector<Engine*> g_engines;
static std::vector<float> g_action;  // returned by every engine as its first output
static std::map<int, std::vector<float>> g_engine_action;  // ... unless set for that engine

static int per_row(const std::vector<int>& shape) {
  int n = 1;
  for (int d : shape) n *= d < 0 ? 1 : d;
  return n;
}
inline std::shared_ptr<Engine> engine_make(const std::string& path, int, int obs_dim, int act_dim) {
  auto e = std::make_shared<Engine>();
  e->path = path;
  e->in_sizes = {obs_dim};
  e->out_sizes = {act_dim};
  e->index = int(g_engines.size());
  g_engines.push_back(e.get());
  return e;
}
inline std::shared_ptr<Engine> engine_make(
    const std::string& path, int, const std::vector<TensorSpec>& in,
    const std::vector<TensorSpec>& out) {
  auto e = std::make_shared<Engine>();
  e->path = path;
  for (auto& t : in) e->in_sizes.push_back(per_row(t.shape));
  for (auto& t : out) e->out_sizes.push_back(per_row(t.shape));
  e->index = int(g_engines.size());
  g_engines.push_back(e.get());
  return e;
}
inline void engine_run(Engine& e, const float* const* in, float* const* out, int batch) {
  e.seen.clear();
  for (size_t k = 0; k < e.in_sizes.size(); ++k)
    e.seen.emplace_back(in[k], in[k] + size_t(batch) * e.in_sizes[k]);
  auto own = g_engine_action.find(e.index);
  const std::vector<float>& act = own != g_engine_action.end() ? own->second : g_action;
  for (size_t k = 0; k < e.out_sizes.size(); ++k) {
    const size_t n = size_t(batch) * e.out_sizes[k];
    for (size_t i = 0; i < n; ++i)
      out[k][i] = k == 0 && i < act.size() ? act[i] : 0.0f;
  }
  ++e.runs;
}
inline void engine_run(Engine& e, const float* obs, float* act, int batch) {
  const float* in[1] = {obs};
  float* out[1] = {act};
  engine_run(e, in, out, batch);
}
}  // namespace policy_api
constexpr int POLICY_NUM_MOTOR = policy_api::NUM_MOTOR;
#ifndef NV_TENSORRT_MAJOR
#define NV_TENSORRT_MAJOR 10
#define NV_TENSORRT_MINOR 0
#endif
"""

_EPILOGUE = r"""
static policy_api::Policy* g_policy = nullptr;
static std::string g_variant;
template <class T> static policy_api::Policy* gk_default() {
  if constexpr (std::is_default_constructible_v<T>) {
    return new T();
  } else {
    throw std::runtime_error("the policy is made by variant: name one of names()");
  }
}
static policy_api::Policy* gk_make() {
#ifdef GK_FACTORY
  if (!g_variant.empty()) {
    auto p = GK_NS::make(g_variant);
    if (!p) throw std::runtime_error("no variant '" + g_variant + "' in names()");
    return p.release();
  }
#endif
  return gk_default<GK_CLASS>();
}
extern "C" {
void gk_set_variant(const char* v) { g_variant = v ? v : ""; }
int gk_variants(char* out, int cap) {
  std::string all;
#ifdef GK_FACTORY
  for (const auto& n : GK_NS::names()) all += n + "\n";
#endif
  if (out && cap > 0) {
    std::strncpy(out, all.c_str(), size_t(cap) - 1);
    out[cap - 1] = 0;
  }
  return int(all.size());
}
int gk_new() {
  delete g_policy;
  g_policy = nullptr;
  policy_api::g_engines.clear();
  try {
    g_policy = gk_make();
    g_policy->init(1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gk_new: %s\n", e.what());
    return -1;
  }
  return int(policy_api::g_engines.size());
}
void gk_set_action(const float* a, int n) {
  policy_api::g_action.assign(a, a + n);
  policy_api::g_engine_action.clear();
}
void gk_set_engine_action(int k, const float* a, int n) {
  policy_api::g_engine_action[k].assign(a, a + n);
}
void gk_step(const float* q, const float* dq, const float* gyro, const float* lin_vel,
             const float* gravity, const float* cmd, const float* task, const float* arm_pose,
             const float* quat, float* q_target) {
  const policy_api::Ctx c{q, dq, gyro, lin_vel, gravity, cmd, task, arm_pose, quat, q_target};
  g_policy->step(c);
}
int gk_engines() { return int(policy_api::g_engines.size()); }
int gk_engine_runs(int k) { return policy_api::g_engines[k]->runs; }
int gk_engine_io(int k, int* in_sizes, int* out_sizes, int cap) {
  auto* e = policy_api::g_engines[k];
  for (size_t i = 0; i < e->in_sizes.size() && int(i) < cap; ++i) in_sizes[i] = e->in_sizes[i];
  for (size_t i = 0; i < e->out_sizes.size() && int(i) < cap; ++i) out_sizes[i] = e->out_sizes[i];
  return int(e->in_sizes.size()) * 100 + int(e->out_sizes.size());
}
const char* gk_engine_path(int k) { return policy_api::g_engines[k]->path.c_str(); }
int gk_seen(int k, int input, float* out, int cap) {
  auto* e = policy_api::g_engines[k];
  if (input >= int(e->seen.size())) return 0;
  const auto& v = e->seen[input];
  const int n = std::min(int(v.size()), cap);
  std::memcpy(out, v.data(), n * sizeof(float));
  return int(v.size());
}
void gk_gains(float* kp, float* kd, int n) {
  const float* p = g_policy->kp();
  const float* d = g_policy->kd();
  for (int i = 0; i < n; ++i) {
    kp[i] = p ? p[i] : NAN;
    kd[i] = d ? d[i] : NAN;
  }
}
int gk_owned() { return g_policy->owned(); }
void gk_limits(double* out) {
  const auto l = g_policy->limits();
  const double v[11] = {l.vx_min, l.vx_max, l.vy_abs, l.yaw_rate_abs, l.speed_norm,
                        l.pos_reached_enter_m, l.pos_reached_exit_m, l.yaw_reached_enter_rad,
                        l.yaw_reached_exit_rad, l.walk_kp_pos, l.walk_kp_yaw};
  std::memcpy(out, v, sizeof v);
}
const char* gk_name() { return g_policy->name(); }
}
"""

LIMIT_FIELDS = (
    "vx_min",
    "vx_max",
    "vy_abs",
    "yaw_rate_abs",
    "speed_norm",
    "pos_reached_enter_m",
    "pos_reached_exit_m",
    "yaw_reached_enter_rad",
    "yaw_reached_exit_rad",
    "walk_kp_pos",
    "walk_kp_yaw",
)


def _launches(src: str) -> str:
    """``kernel<<<B, T>>>(args);`` -> ``GK_LAUNCH((B), (T), kernel(args));``"""
    out, i = [], 0
    while True:
        j = src.find("<<<", i)
        if j < 0:
            out.append(src[i:])
            return "".join(out)
        m = re.search(r"([\w:]+)\s*$", src[i:j])
        if not m:
            raise ValueError("a kernel launch without a kernel name")
        k0 = i + m.start(1)
        end = src.index(">>>", j)
        cfg = src[j + 3 : end]
        depth, parts, cur = 0, [], ""
        for ch in cfg:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        parts.append(cur)
        p = src.index("(", end)
        depth, q = 0, p
        while True:
            if src[q] == "(":
                depth += 1
            elif src[q] == ")":
                depth -= 1
                if depth == 0:
                    break
            q += 1
        call = src[k0:j] + src[p : q + 1]
        out.append(src[i:k0])
        out.append(f"GK_LAUNCH(({parts[0].strip()}), ({parts[1].strip()}), {call})")
        i = q + 1


# A port with variants (gr00t_wbc_h066_p012, ...) is made by name: names() lists them and
# make(name) builds one, as the harness's policy_names() and make_policy() do.
_FACTORY = re.compile(
    r"std::unique_ptr\s*<\s*policy_api::Policy\s*>\s+make\s*\(\s*const\s+std::string\s*&"
)


def _namespace(src: str) -> str:
    m = re.search(r"^\s*namespace\s+(\w+)\s*\{", src, re.M)
    if not m:
        raise ValueError("adapter has no namespace")
    return m.group(1)


def _compiler() -> str:
    for c in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if c and shutil.which(c):
            return c
    raise RuntimeError("reading a benchmark adapter needs a C++17 compiler (g++ or clang++)")


def _cache_dir() -> Path:
    base = os.environ.get("GAITKEEPER_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "gaitkeeper"
    )
    d = Path(base) / "twb_adapters"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(policy_cpp: str | Path, cls: str = "Policy") -> Path:
    """Compile the adapter with the CPU shim; returns the shared library's path."""
    src = Path(policy_cpp).read_text()
    ns = _namespace(src)
    body = _launches(src)
    defs = f"\n#define GK_CLASS {ns}::{cls}\n#define GK_NS {ns}\n"
    if _FACTORY.search(src) and re.search(r"\bnames\s*\(\s*\)\s*\{", src):
        defs += "#define GK_FACTORY 1\n"
    code = _PROLOGUE + "\n" + body + defs + _EPILOGUE
    key = hashlib.sha256((code + "v1").encode()).hexdigest()[:16]
    d = _cache_dir()
    lib = d / f"{ns}_{cls}_{key}.so"
    if lib.exists():
        return lib
    # Compile under names unique to this process, then move the library into place in one
    # step: a concurrent build or load never sees a partly written file, and a compile that
    # is killed leaves only a temporary that is never reused.
    tag = f".{os.getpid()}.{uuid.uuid4().hex[:8]}"
    cpp = d / f"{ns}_{cls}_{key}{tag}.cpp"
    tmp = d / f"{ns}_{cls}_{key}{tag}.so.part"
    cpp.write_text(code)
    cmd = [_compiler(), "-std=c++17", "-O1", "-shared", "-fPIC", "-w", str(cpp), "-o", str(tmp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            lines = [ln for ln in r.stderr.splitlines() if "error" in ln][:8]
            raise RuntimeError(f"compiling {policy_cpp} for the CPU failed:\n" + "\n".join(lines))
        os.replace(tmp, lib)
    finally:
        cpp.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return lib


def preprocessed(policy_cpp: str | Path) -> str:
    """The adapter's source with its macros expanded (for reading constant arrays)."""
    src = Path(policy_cpp).read_text()
    r = subprocess.run(
        [_compiler(), "-E", "-P", "-x", "c++", "-"], input=src, capture_output=True, text=True
    )
    return r.stdout if r.returncode == 0 else src


_F = ctypes.POINTER(ctypes.c_float)


def _p(a: np.ndarray):
    return a.ctypes.data_as(_F)


def _sandbox(policy_cpp: Path) -> Path:
    """A working directory for the adapter, as the harness runs it from the benchmark's root:
    ``policies`` links to the checkout's, and anything the port writes (a patched model under
    ``build/``, say) lands here, not in the checkout."""
    root = policy_cpp.resolve().parent.parent.parent
    key = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    d = _cache_dir().parent / "twb_roots" / key
    (d / "build" / "trt").mkdir(parents=True, exist_ok=True)
    link = d / "policies"
    if not link.exists():
        try:
            link.symlink_to(root / "policies", target_is_directory=True)
        except FileExistsError:
            pass
    return d


def _variants_of(lib) -> list[str]:
    lib.gk_variants.restype = ctypes.c_int
    n = lib.gk_variants(None, 0)
    buf = ctypes.create_string_buffer(n + 1)
    lib.gk_variants(buf, n + 1)
    return [v for v in buf.value.decode().split("\n") if v]


def variants(policy_cpp: str | Path, cls: str = "Policy") -> list[str]:
    """The names a port with variants is made by (its ``names()``); empty for a plain port."""
    return _variants_of(ctypes.CDLL(str(build(policy_cpp, cls))))


class Adapter:
    """One compiled adapter. ``reset`` makes a fresh policy object; ``step`` runs it once on
    the given inputs and returns the motor targets and what each engine was given.
    ``variant`` names the one to make when the port has several (its ``names()``)."""

    def __init__(self, policy_cpp: str | Path, cls: str = "Policy", variant: str | None = None):
        self.path = Path(policy_cpp)
        self.lib = ctypes.CDLL(str(build(policy_cpp, cls)))
        L = self.lib
        self.variants = _variants_of(L)
        if self.variants and variant is None:
            raise ValueError(
                f"{policy_cpp} has variants; name one with variant=: {', '.join(self.variants)}"
            )
        if variant is not None and variant not in self.variants:
            raise ValueError(
                f"{policy_cpp}: no variant {variant!r}"
                + (f" (it has {', '.join(self.variants)})" if self.variants else "")
            )
        self.variant = variant
        L.gk_set_variant.argtypes = [ctypes.c_char_p]
        L.gk_set_variant((variant or "").encode())
        self._root = _sandbox(self.path)
        L.gk_new.restype = ctypes.c_int
        L.gk_name.restype = ctypes.c_char_p
        L.gk_engine_path.restype = ctypes.c_char_p
        L.gk_engine_path.argtypes = [ctypes.c_int]
        L.gk_owned.restype = ctypes.c_int
        self.reset()
        self.name = L.gk_name().decode()
        kp, kd = np.zeros(NUM_MOTOR, np.float32), np.zeros(NUM_MOTOR, np.float32)
        L.gk_gains(_p(kp), _p(kd), NUM_MOTOR)
        self.kp_raw, self.kd_raw = kp.astype(float), kd.astype(float)
        self.owned = int(L.gk_owned())
        lim = np.zeros(11, np.float64)
        L.gk_limits(lim.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        self.limits = dict(zip(LIMIT_FIELDS, map(float, lim)))
        self.engines = []
        for k in range(L.gk_engines()):
            ins, outs = (ctypes.c_int * 16)(), (ctypes.c_int * 16)()
            code = L.gk_engine_io(k, ins, outs, 16)
            self.engines.append(
                {
                    "path": L.gk_engine_path(k).decode(),
                    "inputs": list(ins[: code // 100]),
                    "outputs": list(outs[: code % 100]),
                }
            )
        if not self.engines:
            raise RuntimeError(f"{policy_cpp}: init made no engine")
        self.act_dim = self.engines[0]["outputs"][0]
        self.obs_dim = self.engines[0]["inputs"][0]

    def reset(self) -> None:
        here = os.getcwd()
        os.chdir(self._root)
        try:
            ok = self.lib.gk_new() >= 0
        finally:
            os.chdir(here)
        if not ok:
            raise RuntimeError(f"{self.path}: the policy object could not be made")

    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray | None = None,
        gyro=(0, 0, 0),
        lin_vel=(0, 0, 0),
        gravity=(0, 0, -1),
        cmd=(0, 0, 0),
        action: np.ndarray | None = None,
        arm_pose: np.ndarray | None = None,
        quat=(1, 0, 0, 0),
        task: np.ndarray | None = None,
        engine_actions: dict[int, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """One step. Every engine returns ``action``, except those ``engine_actions`` names."""
        f = lambda v, n: np.ascontiguousarray(  # noqa: E731
            np.zeros(n) if v is None else v, dtype=np.float32
        )
        a = f(action, self.act_dim)
        self.lib.gk_set_action(_p(a), len(a))
        for k, ea in (engine_actions or {}).items():
            e = f(ea, self.engines[k]["outputs"][0])
            self.lib.gk_set_engine_action(int(k), _p(e), len(e))
        tgt = np.zeros(NUM_MOTOR, np.float32)
        args = [
            f(q, NUM_MOTOR),
            f(dq, NUM_MOTOR),
            f(gyro, 3),
            f(lin_vel, 3),
            f(gravity, 3),
            f(cmd, 3),
            f(task, 64),
            f(arm_pose, NUM_MOTOR),
            f(quat, 4),
        ]
        self.lib.gk_step(*[_p(x) for x in args], _p(tgt))
        seen = []
        for k in range(len(self.engines)):
            n = self.engines[k]["inputs"][0]
            buf = np.zeros(max(n, 1), np.float32)
            self.lib.gk_seen(k, 0, _p(buf), len(buf))
            seen.append(buf.astype(float))
        return tgt.astype(float), seen
