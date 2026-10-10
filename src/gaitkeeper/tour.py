"""Closed-loop waypoint tours: the command comes from where the robot is, not from a clock.

``WaypointTour`` turns a list of waypoints (relative to the pose the tour starts from) into
velocity commands each policy step, the way a teleoperation layer or a benchmark harness
does: a proportional controller on the position and heading error in the body frame, with
hysteresis on "reached", then clamped to the policy's command limits. It records the error
at the end of each waypoint's time slot.

``bench_waypoints``, ``bench_arm_walk`` and ``bench_punches`` build the tour, the arm motion
and the punches of rhoyn's teleop-walking-benchmark (main.cpp): 18 waypoints, 5 s each, in a 0.75 m disc around
(0.3, 0); arms walking at random around STANCE; a punch on a random link every 5 s whose
force ceiling ramps from 1/3 to 1 of 500 N over 60 s. The draws follow the same
distributions with numpy's generator, not the benchmark's own random stream, so a seed here
is not the same run as the same seed there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .runner import External, Push


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


@dataclass
class WaypointTour:
    waypoints: list[tuple[float, float, float]]  # (x, y, yaw) relative to the start pose
    point_s: float = 5.0
    start_s: float = 0.0
    kp_pos: float = 1.5
    kp_yaw: float = 1.5
    vx: tuple[float, float] = (-0.5, 1.0)
    vy_abs: float = 0.3
    wz_abs: float = 0.2
    speed_norm: float = 0.0
    pos_enter: float = 0.10
    pos_exit: float = 0.20
    yaw_enter: float = 0.05
    yaw_exit: float = 0.12
    _anchor: tuple[float, float, float] | None = field(default=None, repr=False)
    _index: int = field(default=-1, repr=False)
    _pos_ok: bool = field(default=False, repr=False)
    _yaw_ok: bool = field(default=False, repr=False)
    _last: tuple[float, float] | None = field(default=None, repr=False)
    _ends: list[tuple[float, float]] = field(default_factory=list, repr=False)
    # The harness's task for the step just commanded (main.cpp h_task): distance to the
    # waypoint, yaw error, and the waypoint in the body frame; zeros with no waypoint.
    task: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __call__(self, time: float, base: np.ndarray) -> np.ndarray:
        """``base`` is the free joint's qpos: position, then quaternion wxyz."""
        elapsed = time - self.start_s
        self.task = (0.0, 0.0, 0.0, 0.0)
        if elapsed < 0:
            return np.zeros(3)
        x, y, yaw = float(base[0]), float(base[1]), _yaw(base[3:7])
        if self._anchor is None:
            self._anchor = (x, y, yaw)
        index = int(elapsed / self.point_s)
        if index != self._index:
            if self._index >= 0 and self._last is not None:
                self._ends.append(self._last)
            self._index, self._pos_ok, self._yaw_ok = index, False, False
        if index >= len(self.waypoints):
            return np.zeros(3)
        ax, ay, ayaw = self._anchor
        wx, wy, wyaw = self.waypoints[index]
        tx = ax + wx * math.cos(ayaw) - wy * math.sin(ayaw)
        ty = ay + wx * math.sin(ayaw) + wy * math.cos(ayaw)
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        yaw_err = math.remainder(ayaw + wyaw - yaw, 2 * math.pi)
        self._last = (dist, yaw_err)
        self._pos_ok = dist <= self.pos_exit if self._pos_ok else dist < self.pos_enter
        self._yaw_ok = (
            abs(yaw_err) <= self.yaw_exit if self._yaw_ok else abs(yaw_err) < self.yaw_enter
        )
        c, s = math.cos(yaw), math.sin(yaw)
        self.task = (dist, yaw_err, c * dx + s * dy, -s * dx + c * dy)
        cmd = np.zeros(3)
        if not self._pos_ok:
            cmd[0] = self.kp_pos * (c * dx + s * dy)
            cmd[1] = self.kp_pos * (-s * dx + c * dy)
        if not self._yaw_ok:
            cmd[2] = self.kp_yaw * yaw_err
        norm = math.hypot(cmd[0], cmd[1])
        if self.speed_norm > 0 and norm > self.speed_norm:
            cmd[:2] *= self.speed_norm / norm
        cmd[0] = min(max(cmd[0], self.vx[0]), self.vx[1])
        cmd[1] = min(max(cmd[1], -self.vy_abs), self.vy_abs)
        cmd[2] = min(max(cmd[2], -self.wz_abs), self.wz_abs)
        return cmd

    def report(self, fell_at: float | None) -> dict[str, Any]:
        """Error at the end of each waypoint slot that finished before a fall, as the
        benchmark scores it: centimetres and degrees, averaged over finished slots."""
        ends = list(self._ends)
        finished = len(self.waypoints) if fell_at is None else len(ends)
        if fell_at is None and self._last is not None and len(ends) < len(self.waypoints):
            ends.append(self._last)
        ends = ends[:finished]
        pos = [d * 100 for d, _ in ends]
        yaw = [abs(e) * 180 / math.pi for _, e in ends]
        return {
            "outcome": "complete" if fell_at is None else "fell",
            "survival_s": None if fell_at is None else fell_at,
            "targets": len(ends),
            "pos_err_cm": float(np.mean(pos)) if pos else math.nan,
            "yaw_err_deg": float(np.mean(yaw)) if yaw else math.nan,
            "reached": sum(1 for d, e in ends if d < self.pos_enter and abs(e) < self.yaw_enter),
        }


# -- rhoyn/teleop-walking-benchmark ----------------------------------------------------------

BENCH_WAYPOINTS = 18
BENCH_POINT_S = 5.0
BENCH_CENTRE = (0.3, 0.0)
BENCH_RADIUS_M = 0.75
BENCH_FORCE_MAX_N = 500.0
BENCH_FORCE_SCALE_MIN = 0.5
BENCH_RAMP_FLOOR = 1.0 / 3.0
BENCH_RAMP_S = 60.0
BENCH_PUNCH_DELAY_S = 0.1
BENCH_PUNCH_S = 0.08
BENCH_ARM_STEP_RAD = 0.06
BENCH_ARM_WINDOW_S = 0.5
BENCH_ARM_MIRROR = (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
BENCH_ARM_LEFT = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
BENCH_ARM_RIGHT = tuple(n.replace("left_", "right_") for n in BENCH_ARM_LEFT)
# The harness's starting pose for the arms (STANCE, MuJoCo motor order 15..28).
BENCH_ARM_STANCE_LEFT = (0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0)


def bench_waypoints(seed: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng([seed, 2])
    out = []
    for _ in range(BENCH_WAYPOINTS):
        r = BENCH_RADIUS_M * math.sqrt(rng.uniform())
        b = rng.uniform(0, 2 * math.pi)
        yaw = math.remainder(rng.uniform(0, 2 * math.pi), 2 * math.pi)
        out.append((BENCH_CENTRE[0] + r * math.cos(b), BENCH_CENTRE[1] + r * math.sin(b), yaw))
    return out


def bench_punches(seed: int, bodies: list[str]) -> list[Push]:
    """One punch per waypoint slot, on a random link, random direction on the sphere."""
    rng = np.random.default_rng([seed, 3])
    out = []
    for i in range(BENCH_WAYPOINTS):
        t = BENCH_PUNCH_DELAY_S + BENCH_POINT_S * i
        body = bodies[int(rng.integers(len(bodies)))]
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        climbed = min((t - BENCH_PUNCH_DELAY_S) / BENCH_RAMP_S, 1.0)
        ceiling = BENCH_FORCE_MAX_N * (BENCH_RAMP_FLOOR + (1 - BENCH_RAMP_FLOOR) * climbed)
        f = ceiling * rng.uniform(BENCH_FORCE_SCALE_MIN, 1.0)
        out.append(Push(t, "force", tuple(float(x) for x in d * f), body, BENCH_PUNCH_S))
    return out


def bench_arm_walk(
    seed: int, seconds: float, limits: dict[str, tuple[float, float]], dt: float = 0.02
) -> dict[str, list[list[float]]]:
    """Arm targets as {joint: [[t, value], ...]}: each 0.5 s the left arm's target moves by
    25 steps of up to 0.06 rad per joint, inside the joint limits, and the right arm
    mirrors it. The benchmark also redraws a target that puts a hand in front of the body
    or above the head; that check is not reproduced."""
    rng = np.random.default_rng([seed, 4])
    ticks = int(BENCH_ARM_WINDOW_S / dt + 0.5)
    cur = np.array(BENCH_ARM_STANCE_LEFT, float)
    lo = np.array([limits[n][0] for n in BENCH_ARM_LEFT])
    hi = np.array([limits[n][1] for n in BENCH_ARM_LEFT])
    times, poses = [0.0], [cur.copy()]
    t = 0.0
    while t < seconds:
        for _ in range(ticks):
            cur = np.clip(cur + BENCH_ARM_STEP_RAD * rng.uniform(-1, 1, len(cur)), lo, hi)
        t += BENCH_ARM_WINDOW_S
        times.append(t)
        poses.append(cur.copy())
    out: dict[str, list[list[float]]] = {}
    for i, n in enumerate(BENCH_ARM_LEFT):
        out[n] = [[tt, float(p[i])] for tt, p in zip(times, poses)]
    for i, n in enumerate(BENCH_ARM_RIGHT):
        out[n] = [[tt, float(BENCH_ARM_MIRROR[i] * p[i])] for tt, p in zip(times, poses)]
    return out


def bench_arms_external(
    seed: int,
    seconds: float,
    limits: dict[str, tuple[float, float]],
    kp: dict[str, float],
    kd: dict[str, float],
    obs: str = "real",
) -> External:
    walk = bench_arm_walk(seed, seconds, limits)
    return External(list(walk), drive="trajectory", pose=walk, kp=kp, kd=kd, obs=obs)


# -- running a tour ------------------------------------------------------------------------

BENCH_ARMS = BENCH_ARM_LEFT + BENCH_ARM_RIGHT
_WN = 10 * 2.0 * math.pi  # the benchmark's holding servo: 10 Hz, damping ratio 2
_ZETA = 2.0


def armature_gains(m: Any, joints: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """The benchmark's gains for joints it holds: kp = armature * wn^2, kd = 2 zeta armature wn."""
    import mujoco

    kp, kd = {}, {}
    for j in joints:
        arm = float(
            m.dof_armature[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]]
        )
        kp[j], kd[j] = arm * _WN * _WN, 2 * _ZETA * arm * _WN
    return kp, kd


@dataclass
class TourOptions:
    waypoints: list[tuple[float, float, float]] | None = None  # None: the benchmark's draws
    point_s: float = BENCH_POINT_S
    arms: str = "policy"  # "policy", "hold" (at STANCE) or "walk" (the benchmark's random walk)
    arms_obs: str = (
        "real"  # how the policy sees arms it does not drive; "contract": as its externals say
    )
    hold_gains: str = "armature"  # "armature" (the benchmark's) or "policy" (the contract's)
    punches: str = "none"  # "none" or "benchmark"
    backend: str | None = None


def _tour_job(job: tuple) -> dict[str, Any]:
    import mujoco

    from .contract import Contract
    from .policy import OnnxPolicy
    from .runner import RunConfig, Runner

    contract_d, mjcf, policy_path, seed, o = job
    c = Contract.from_dict(contract_d)
    rec = c.get("policy_io.graph.recurrent", None) or None
    r = Runner(c, mjcf, OnnxPolicy(policy_path, rec))
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    lim = c.get("policy_io.commands.base_velocity.limit", None) or {}
    wps = o.waypoints if o.waypoints is not None else bench_waypoints(seed)
    tour = WaypointTour(
        list(wps),
        point_s=o.point_s,
        vx=tuple(lim.get("vx", (-0.5, 1.0))),
        vy_abs=max(abs(x) for x in lim.get("vy", (-0.3, 0.3))),
        wz_abs=max(abs(x) for x in lim.get("wz", (-0.2, 0.2))),
    )
    seconds = len(wps) * o.point_s
    ext = None
    un_traj = None
    if o.arms != "policy":
        arms = [j for j in r.names if j in BENCH_ARMS]
        # The contract's own externals (a port's held waist, say) stay; the arms are replaced.
        base = [e for e in r.externals(RunConfig())]
        contract_obs = {j: e.obs for e in base for j in e.joints}
        ext = []
        for e in base:
            rest = [j for j in e.joints if j not in BENCH_ARMS]
            if rest:
                ext.append(External(rest, e.drive, e.pose, e.kp, e.kd, e.obs))
        if o.hold_gains == "armature":
            kp, kd = armature_gains(m, arms)
        else:
            kp = {j: float(r.kp[r.names.index(j)]) for j in arms}
            kd = {j: float(r.kd[r.names.index(j)]) for j in arms}
        stance = dict(zip(BENCH_ARM_LEFT, BENCH_ARM_STANCE_LEFT))
        stance.update(
            {
                rj: BENCH_ARM_MIRROR[i] * BENCH_ARM_STANCE_LEFT[i]
                for i, rj in enumerate(BENCH_ARM_RIGHT)
            }
        )
        groups: dict[str, list[str]] = {}
        for j in arms:
            obs = contract_obs.get(j, "real") if o.arms_obs == "contract" else o.arms_obs
            groups.setdefault(obs, []).append(j)
        walk = None
        if o.arms == "walk":
            limits = {
                j: tuple(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
                for j in BENCH_ARMS
            }
            walk = bench_arm_walk(seed, seconds, limits)
            # Arms the policy does not list (a legs-only policy) are the harness's already:
            # they follow the walk through control.unlisted.
            unlisted = c.get("control.unlisted", None)
            if unlisted:
                un_traj = {j: walk[j] for j in BENCH_ARMS if j not in r.names}
        for obs, js in groups.items():
            if walk is not None:
                ext.append(External(js, "trajectory", {j: walk[j] for j in js}, kp, kd, obs))
            else:
                ext.append(External(js, "hold", {j: stance[j] for j in js}, kp, kd, obs))
    pushes = []
    if o.punches == "benchmark":
        bodies = [m.body(m.jnt_bodyid[m.actuator_trnid[a, 0]]).name for a in range(m.nu)]
        pushes = bench_punches(seed, bodies)
    res = r.run(
        RunConfig(
            backend=o.backend,
            seconds=seconds,
            command_source=tour,
            external=ext,
            pushes=pushes,
            seed=seed,
            unlisted_trajectory=un_traj,
        )
    )
    return {"seed": seed, **(res.task or {})}


def run_tour(
    contract: Any,
    mjcf: str,
    policy_path: str,
    seeds: list[int],
    options: TourOptions,
    workers: int | None = None,
) -> dict[str, Any]:
    """Run the tour once per seed. Returns per-seed reports and their summary."""
    from concurrent.futures import ProcessPoolExecutor

    from .runner import pool_context

    jobs = [(contract.to_dict(), str(mjcf), str(policy_path), s, options) for s in seeds]
    if workers and workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(workers, mp_context=pool_context()) as ex:
            runs = list(ex.map(_tour_job, jobs))
    else:
        runs = [_tour_job(j) for j in jobs]
    n_wp = len(options.waypoints) if options.waypoints is not None else BENCH_WAYPOINTS
    full = n_wp * options.point_s
    surv = [r["survival_s"] if r.get("survival_s") is not None else full for r in runs]
    pos = [r["pos_err_cm"] for r in runs if not math.isnan(r.get("pos_err_cm", math.nan))]
    yaw = [r["yaw_err_deg"] for r in runs if not math.isnan(r.get("yaw_err_deg", math.nan))]
    return {
        "runs": runs,
        "seconds": full,
        "mean_survival_s": float(np.mean(surv)),
        "complete": sum(r.get("outcome") == "complete" for r in runs),
        "pos_err_cm": float(np.mean(pos)) if pos else math.nan,
        "yaw_err_deg": float(np.mean(yaw)) if yaw else math.nan,
        "options": {k: v for k, v in options.__dict__.items() if k != "waypoints"},
    }
