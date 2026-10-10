"""Observation term library.

Each term is rebuilt from raw simulator state and the contract, never from the
values a harness derived itself. Arrays are vectorized over steps (leading
axis T). Quaternions are (w, x, y, z).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GRAVITY_W = np.array([0.0, 0.0, -1.0])


# -- rotation helpers ---------------------------------------------------------


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz unit quaternions to (..., 3, 3) rotation matrices (body to world)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty(q.shape[:-1] + (3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def yaw_of(q: np.ndarray) -> np.ndarray:
    r = quat_to_mat(q)
    return np.arctan2(r[..., 1, 0], r[..., 0, 0])


# -- raw state ----------------------------------------------------------------


@dataclass
class StateLayout:
    """How the simulator's generalized state is laid out (from the trace, not the contract)."""

    joint_names: list[str]  # hinge joints in qpos order after the free joint
    free_joint: bool = True
    quat_order: str = "wxyz"
    ang_vel_frame: str = "body"  # frame of qvel[3:6]; MuJoCo free joints use the body frame
    lin_vel_frame: str = "world"

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> StateLayout:
        s = meta["state_layout"]
        return cls(
            joint_names=list(s["joint_names"]),
            free_joint=s.get("free_joint", True),
            quat_order=s.get("quat_order", "wxyz"),
            ang_vel_frame=s.get("ang_vel_frame", "body"),
            lin_vel_frame=s.get("lin_vel_frame", "world"),
        )

    def to_meta(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.joint_names),
            "free_joint": self.free_joint,
            "quat_order": self.quat_order,
            "ang_vel_frame": self.ang_vel_frame,
            "lin_vel_frame": self.lin_vel_frame,
        }


@dataclass
class RawState:
    """Named raw state over T steps at the instants observations were built."""

    root_quat: np.ndarray  # (T, 4) wxyz
    ang_vel_body: np.ndarray  # (T, 3) root angular velocity, root body frame
    joint_pos: np.ndarray  # (T, J) in layout.joint_names order
    joint_vel: np.ndarray  # (T, J)
    joint_names: list[str]
    command: np.ndarray  # (T, 3)
    episode_step: np.ndarray  # (T,) steps since the last reset
    reset: np.ndarray  # (T,) True when the observation follows a reset
    prev_action: np.ndarray  # (T, A) previous raw policy output (zeros after reset)
    action: np.ndarray | None = None  # (T, A) raw policy output at each step, when known
    lin_vel_world: np.ndarray | None = None  # (T, 3) root linear velocity, world frame

    @classmethod
    def from_arrays(
        cls,
        qpos: np.ndarray,
        qvel: np.ndarray,
        layout: StateLayout,
        command: np.ndarray,
        episode_step: np.ndarray,
        reset: np.ndarray,
        action: np.ndarray,
    ) -> RawState:
        qpos = np.asarray(qpos, dtype=np.float64)
        qvel = np.asarray(qvel, dtype=np.float64)
        if not layout.free_joint:
            raise ValueError("fixed-base layouts are not supported")
        quat = qpos[:, 3:7]
        if layout.quat_order == "xyzw":
            quat = quat[:, [3, 0, 1, 2]]
        w = qvel[:, 3:6]
        v = qvel[:, 0:3]
        if layout.lin_vel_frame == "body":
            v = np.einsum("tij,tj->ti", quat_to_mat(quat), v)
        if layout.ang_vel_frame == "world":
            w = np.einsum("tji,tj->ti", quat_to_mat(quat), w)
        reset = np.asarray(reset, dtype=bool)
        action = np.asarray(action, dtype=np.float64)
        prev = np.zeros_like(action)
        prev[1:] = action[:-1]
        prev[reset] = 0.0
        return cls(
            root_quat=quat,
            ang_vel_body=w,
            joint_pos=qpos[:, 7:],
            joint_vel=qvel[:, 6:],
            joint_names=list(layout.joint_names),
            command=np.asarray(command, dtype=np.float64),
            episode_step=np.asarray(episode_step),
            reset=reset,
            prev_action=prev,
            action=action,
            lin_vel_world=v,
        )

    def joint_index(self, names: list[str]) -> np.ndarray:
        idx = {n: i for i, n in enumerate(self.joint_names)}
        missing = [n for n in names if n not in idx]
        if missing:
            raise KeyError(f"joints not in the simulator state: {missing}")
        return np.array([idx[n] for n in names])


# -- terms ----------------------------------------------------------------------

TermFn = Callable[[RawState, dict[str, Any], "TermContext"], np.ndarray]


@dataclass
class TermContext:
    joint_names: list[str]  # policy joint order
    default_joint_pos: np.ndarray  # policy order
    policy_dt: float
    imu_rotation_in_root: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    imu_frame: str = "body"


def base_ang_vel(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    if ctx.imu_frame == "world":
        return np.einsum("tij,tj->ti", quat_to_mat(s.root_quat), s.ang_vel_body)
    r_imu = quat_to_mat(np.asarray(ctx.imu_rotation_in_root)[None])[0]
    return s.ang_vel_body @ r_imu  # r_imu^T w for each row


def base_lin_vel(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    """Root linear velocity in the root frame, as Isaac Lab's and mjlab's ``base_lin_vel``
    read it from the simulator. A real robot has no such sensor."""
    if s.lin_vel_world is None:
        raise ValueError("base_lin_vel needs the root's linear velocity, which this state lacks")
    return np.einsum("tji,tj->ti", quat_to_mat(s.root_quat), s.lin_vel_world)


def projected_gravity(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    r = quat_to_mat(s.root_quat)
    return np.einsum("tji,j->ti", r, GRAVITY_W)


def velocity_commands(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    return s.command[:, :3].copy()


def _accumulated_phase(steps: np.ndarray, policy_dt: float, period: float) -> np.ndarray:
    """Phase from a float32 accumulator advanced by dt/period each step, as the
    Unitree deploy runtime keeps it (observations.h gait_phase)."""
    f = np.float32
    delta = f(policy_dt) * (f(1.0) / f(period))
    n = int(steps.max()) + 1 if steps.size else 1
    seq = np.empty(n, dtype=f)
    g = f(0.0)
    for i in range(n):
        seq[i] = g
        g = f(np.fmod(f(g + delta), f(1.0)))
    return seq[steps].astype(np.float64)


def gait_phase(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    period = float(p["period"])
    # clock_offset_steps: the phase clock runs this many policy steps ahead of the
    # episode counter (2 in the Unitree deploy runtime, 0 in training).
    k = s.episode_step.astype(np.int64) + int(p.get("clock_offset_steps", 0))
    arith = p.get("arithmetic", "float32")
    if arith == "float32":
        # The reference evaluates (step * dt) % period / period * pi * 2 in float32;
        # at 20 s the argument's rounding alone is about 2e-5, so mirror it.
        f = np.float32
        t32 = k.astype(f) * f(ctx.policy_dt)
        arg = (np.fmod(t32, f(period)) / f(period)) * f(np.pi) * f(2.0)
        out = np.stack([np.sin(arg), np.cos(arg)], axis=1).astype(np.float64)
    elif arith == "float32_accumulate":
        g = _accumulated_phase(k, ctx.policy_dt, period)
        out = np.stack([np.sin(g * 2 * np.pi), np.cos(g * 2 * np.pi)], axis=1)
        out = out.astype(np.float32).astype(np.float64)
    else:
        ph = np.mod(k.astype(np.float64) * ctx.policy_dt, period) / period
        out = np.stack([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], axis=1)
    thr = p.get("stand_threshold")
    if thr is not None:
        out[np.linalg.norm(s.command[:, :3], axis=1) < float(thr)] = 0.0
    return out


def gait_phase_legs(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    """A two-leg clock: [sin, sin, cos, cos] of the left and right phases, the right leg half
    a period behind, so [s, -s, c, -c]. Used by ClOBOT's G1 policy (gait_phase_legs, period
    1.0); the clock runs from the episode start and is never zeroed."""
    period = float(p["period"])
    offset = float(p.get("offset", 0.5))
    k = s.episode_step.astype(np.float64) + int(p.get("clock_offset_steps", 0))
    ph = np.mod(k * ctx.policy_dt, period) / period
    a, b = 2 * np.pi * ph, 2 * np.pi * (ph + offset)
    return np.stack([np.sin(a), np.sin(b), np.cos(a), np.cos(b)], axis=1)


def joint_pos_rel(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    return s.joint_pos[:, s.joint_index(ctx.joint_names)] - ctx.default_joint_pos[None]


def joint_vel_rel(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    return s.joint_vel[:, s.joint_index(ctx.joint_names)].copy()


def last_action(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    return s.prev_action.copy()


def constant(s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    """Fixed values a harness feeds (a height command, a zeroed slot)."""
    v = np.asarray(p["value"], dtype=np.float64).reshape(-1)
    return np.repeat(v[None], len(s.episode_step), axis=0)


# -- stateful terms -------------------------------------------------------------------
# A term whose value depends on earlier steps keeps a state from one step to the next. Each is
# written once, as a step function over a single row; over a whole trace it is scanned row by
# row (restarting at every reset), and the closed loop calls the same step, so the two cannot
# differ.


def _row(s: RawState, t: int) -> RawState:
    return RawState(
        root_quat=s.root_quat[t : t + 1],
        ang_vel_body=s.ang_vel_body[t : t + 1],
        joint_pos=s.joint_pos[t : t + 1],
        joint_vel=s.joint_vel[t : t + 1],
        joint_names=s.joint_names,
        command=s.command[t : t + 1],
        episode_step=s.episode_step[t : t + 1],
        reset=s.reset[t : t + 1],
        prev_action=s.prev_action[t : t + 1],
        action=None if s.action is None else s.action[t : t + 1],
        lin_vel_world=None if s.lin_vel_world is None else s.lin_vel_world[t : t + 1],
    )


def _action_lag_step(state: Any, s: RawState, p: dict[str, Any], ctx: TermContext):
    """The raw action ``lag`` steps back (zeros before the episode has that many). ``lag`` 1
    is ``prev_action``; a port that observes an older action keeps the ones in between."""
    lag = int(p.get("lag", 1))
    past = list(state or [])
    out = past[-(lag - 1)] if lag > 1 and len(past) >= lag - 1 else np.zeros_like(s.prev_action[0])
    if lag == 1:
        out = s.prev_action[0]
    past.append(s.prev_action[0].copy())
    return np.asarray(out, dtype=np.float64), past[-max(lag - 1, 1) :]


def _joint_vel_diff_step(state: Any, s: RawState, p: dict[str, Any], ctx: TermContext):
    """Joint velocity as the change in measured position over one policy step, zero at the
    first step after a reset (a port that never reads the simulator's velocity)."""
    q = s.joint_pos[0, s.joint_index(ctx.joint_names)]
    dt = float(p.get("dt", ctx.policy_dt))
    out = np.zeros_like(q) if state is None else (q - state) / dt
    if p.get("arithmetic") == "float32":
        f = np.float32
        out = (
            np.zeros_like(q)
            if state is None
            else ((q.astype(f) - state.astype(f)) / f(dt)).astype(np.float64)
        )
    return out, q.copy()


def speed_period(speed: float, knots: list[list[float]]) -> float:
    """The gait period at a command speed: linear between knots, flat outside them."""
    xs = [float(k[0]) for k in knots]
    ys = [float(k[1]) for k in knots]
    return float(np.interp(speed, xs, ys))


def _command_speed(cmd: np.ndarray, how: str) -> float:
    f = np.float32
    c = np.asarray(cmd[:3], dtype=f)
    if how == "norm3":
        return float(np.sqrt(f(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])))
    if how == "planar":
        return float(np.sqrt(f(c[0] * c[0] + c[1] * c[1])))
    if how == "vx":
        return float(abs(c[0]))
    raise ValueError(f"gait_phase_speed: unknown speed {how!r}")


def _gait_phase_speed_step(state: Any, s: RawState, p: dict[str, Any], ctx: TermContext):
    """A clock whose rate follows the command: [sin, cos] of a phase that advances by
    dt / period(speed) after each step while the command speed is at least ``stand_speed``
    (not on the first step of an episode), in float32 as a port keeps it. ``period_knots``
    gives the period against speed."""
    f = np.float32
    first = state is None
    ph = f(0.0) if first else f(state)
    a = f(2.0) * f(np.pi) * ph
    out = np.array([np.sin(a), np.cos(a)], dtype=f).astype(np.float64)
    speed = _command_speed(s.command[0], p.get("speed", "norm3"))
    advance = (not first or p.get("advance_first", False)) and speed >= float(
        p.get("stand_speed", 0.0)
    )
    if advance:
        period = f(speed_period(speed, p["period_knots"]))
        ph = f(np.fmod(f(ph + f(ctx.policy_dt) / period), f(1.0)))
    return out, ph


STATEFUL: dict[str, Callable[..., tuple[np.ndarray, Any]]] = {
    "joint_vel_diff": _joint_vel_diff_step,
    "gait_phase_speed": _gait_phase_speed_step,
}


def _stateful(term: dict[str, Any]):
    p = term.get("params", {}) or {}
    if term["id"] == "last_action" and int(p.get("lag", 1)) > 1:
        return _action_lag_step
    return STATEFUL.get(term["id"])


def _scan(fn, s: RawState, p: dict[str, Any], ctx: TermContext) -> np.ndarray:
    out, state = [], None
    for t in range(len(s.episode_step)):
        if t == 0 or bool(s.reset[t]):
            state = None
        v, state = fn(state, _row(s, t), p, ctx)
        out.append(v)
    return np.array(out)


def _scan_term(fn):
    return lambda s, p, ctx: _scan(fn, s, p, ctx)


TERMS: dict[str, TermFn] = {
    "base_ang_vel": base_ang_vel,
    "base_lin_vel": base_lin_vel,
    "projected_gravity": projected_gravity,
    "velocity_commands": velocity_commands,
    "gait_phase": gait_phase,
    "gait_phase_legs": gait_phase_legs,
    "joint_pos_rel": joint_pos_rel,
    "joint_vel_rel": joint_vel_rel,
    "last_action": last_action,
    "constant": constant,
    "joint_vel_diff": _scan_term(_joint_vel_diff_step),
    "gait_phase_speed": _scan_term(_gait_phase_speed_step),
}


def term_key(t: dict[str, Any]) -> str:
    """A term's name within its group: ``source_name`` when given (two entries may share an
    id, for example a command split around other terms), else its id."""
    return str(t.get("source_name") or t["id"])


def _select(x: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """``params.index``: the term's elements in this order (a permutation or a subset)."""
    idx = p.get("index") if p else None
    return x if idx is None else x[:, [int(i) for i in idx]]


# -- assembling an observation -----------------------------------------------------


def apply_clip_scale(
    x: np.ndarray, term: dict[str, Any], clip_then_scale: bool = True
) -> np.ndarray:
    clip = term.get("clip")
    scale = np.asarray(term.get("scale", 1.0), dtype=np.float64)
    if clip is not None and clip_then_scale:
        x = np.clip(x, clip[0], clip[1])
    x = x * scale
    if clip is not None and not clip_then_scale:
        x = np.clip(x, clip[0], clip[1])
    return x


def stack_history(
    x: np.ndarray, reset: np.ndarray, length: int, init: str, order: str
) -> np.ndarray:
    """(T, d) per-step values to (T, length, d) history windows.

    ``init`` says what fills the buffer at a reset: "repeat_first" (the first
    value) or "zeros". ``order`` is "oldest_first" or "newest_first".
    """
    t_len, d = x.shape
    out = np.empty((t_len, length, d))
    buf = np.zeros((length, d))
    for t in range(t_len):
        if t == 0 or reset[t]:
            buf[:] = x[t] if init == "repeat_first" else 0.0
            if init == "zeros":
                buf[-1] = x[t]
        else:
            buf = np.roll(buf, -1, axis=0)
            buf[-1] = x[t]
        out[t] = buf if order == "oldest_first" else buf[::-1]
    return out


def term_values(
    s: RawState, terms: list[dict[str, Any]], ctx: TermContext, clip_then_scale: bool = True
) -> dict[str, np.ndarray]:
    """Per-step value of each term after clip and scale, before history. The group's
    ``clip_then_scale`` says the order (Isaac Lab clips first; legged_gym scales first)."""
    out = {}
    for term in terms:
        fn = TERMS.get(term["id"])
        if fn is None:
            raise KeyError(f"no term '{term['id']}' in the library")
        p = term.get("params", {}) or {}
        st = _stateful(term)
        raw = _scan(st, s, p, ctx) if st is not None else fn(s, p, ctx)
        out[term_key(term)] = apply_clip_scale(_select(raw, p), term, clip_then_scale)
    return out


def assemble(
    values: dict[str, np.ndarray],
    terms: list[dict[str, Any]],
    reset: np.ndarray,
    history: dict[str, Any],
) -> tuple[np.ndarray, dict[str, slice]]:
    """Concatenate terms into the policy input. Returns (obs, slice per term).

    For term-major layout, each term's slice holds its whole history window.
    For time-major layout, slices are not contiguous, so the returned slices
    index the term-major view and the caller reorders through ``layout_index``.
    """
    length = int(history.get("length", 1))
    init = history.get("init", "repeat_first")
    order = history.get("order", "oldest_first")
    layout = history.get("layout", "term_major")
    windows = {
        term_key(t): stack_history(values[term_key(t)], reset, length, init, order) for t in terms
    }
    t_len = reset.shape[0]
    if layout == "term_major":
        parts = [windows[term_key(t)].reshape(t_len, -1) for t in terms]
    elif layout == "time_major":
        parts = [windows[term_key(t)][:, k, :] for k in range(length) for t in terms]
    else:
        raise ValueError(f"unknown history layout {layout!r}")
    obs = np.concatenate(parts, axis=1)
    return obs, term_slices(terms, history)


def term_slices(terms: list[dict[str, Any]], history: dict[str, Any]) -> dict[str, np.ndarray]:
    """Column indices of each term (all history slots) in the assembled observation."""
    length = int(history.get("length", 1))
    layout = history.get("layout", "term_major")
    dims = [int(t["dim"]) for t in terms]
    cols: dict[str, list[int]] = {term_key(t): [] for t in terms}
    pos = 0
    if layout == "term_major":
        for t, d in zip(terms, dims):
            cols[term_key(t)] = list(range(pos, pos + d * length))
            pos += d * length
    else:
        for _k in range(length):
            for t, d in zip(terms, dims):
                cols[term_key(t)].extend(range(pos, pos + d))
                pos += d
    return {k: np.array(v) for k, v in cols.items()}


def context_from_contract(contract: Any) -> TermContext:
    names = contract.get("policy_io.joints.names")
    default = contract.get("control.default_joint_pos")
    imu = contract.get("policy_io.imu", {}) or {}
    return TermContext(
        joint_names=list(names),
        default_joint_pos=np.array([default[n] for n in names], dtype=np.float64),
        policy_dt=float(contract.get("timing.policy_dt")),
        imu_rotation_in_root=np.asarray(
            imu.get("rotation_in_root") or [1.0, 0, 0, 0], dtype=np.float64
        ),
        imu_frame=imu.get("frame", "body"),
    )


def build_observation(
    s: RawState, contract: Any
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Rebuild the policy input from raw state. Returns (obs, per-term values, term columns)."""
    group = contract.get("policy_io.observation_groups.policy")
    terms = group["terms"]
    ctx = context_from_contract(contract)
    values = term_values(s, terms, ctx, bool(group.get("clip_then_scale", True)))
    obs, _ = assemble(values, terms, s.reset, group.get("history", {}))
    return obs, values, term_slices(terms, group.get("history", {}))


class ObservationBuilder:
    """Build the policy input one step at a time, as a closed loop needs it.

    Gives the same vectors as ``build_observation`` over a whole trace (tested).
    """

    def __init__(self, contract: Any):
        group = contract.get("policy_io.observation_groups.policy")
        self.terms = group["terms"]
        self.clip_then_scale = bool(group.get("clip_then_scale", True))
        self.history = dict(group.get("history", {}) or {})
        self.length = int(self.history.get("length", 1))
        self.ctx = context_from_contract(contract)
        self.buffers: dict[str, np.ndarray] = {}
        self.states: dict[str, Any] = {}

    def step(
        self,
        root_quat: np.ndarray,
        ang_vel_body: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        joint_names: list[str],
        command: np.ndarray,
        episode_step: int,
        prev_action: np.ndarray,
        lin_vel_world: np.ndarray | None = None,
    ) -> np.ndarray:
        reset = episode_step == 0
        s = RawState(
            root_quat=np.asarray(root_quat, dtype=np.float64)[None],
            ang_vel_body=np.asarray(ang_vel_body, dtype=np.float64)[None],
            joint_pos=np.asarray(joint_pos, dtype=np.float64)[None],
            joint_vel=np.asarray(joint_vel, dtype=np.float64)[None],
            joint_names=joint_names,
            command=np.asarray(command, dtype=np.float64)[None],
            episode_step=np.array([episode_step]),
            reset=np.array([reset]),
            prev_action=(np.zeros_like(prev_action) if reset else np.asarray(prev_action))[None],
            lin_vel_world=None if lin_vel_world is None else np.asarray(lin_vel_world, float)[None],
        )
        values = {}
        for term in self.terms:
            k, p = term_key(term), term.get("params", {}) or {}
            st = _stateful(term)
            if st is None:
                values.update(term_values(s, [term], self.ctx, self.clip_then_scale))
                continue
            v, self.states[k] = st(None if reset else self.states.get(k), s, p, self.ctx)
            values[k] = apply_clip_scale(_select(v[None], p), term, self.clip_then_scale)
        init = self.history.get("init", "repeat_first")
        for t in self.terms:
            x = values[term_key(t)][0]
            buf = self.buffers.get(term_key(t))
            if reset or buf is None:
                buf = np.repeat(x[None], self.length, axis=0)
                if init == "zeros":
                    buf[:-1] = 0.0
            else:
                buf = np.roll(buf, -1, axis=0)
                buf[-1] = x
            self.buffers[term_key(t)] = buf
        order = self.history.get("order", "oldest_first")
        win = {k: (b if order == "oldest_first" else b[::-1]) for k, b in self.buffers.items()}
        if self.history.get("layout", "term_major") == "term_major":
            parts = [win[term_key(t)].reshape(-1) for t in self.terms]
        else:
            parts = [win[term_key(t)][k] for k in range(self.length) for t in self.terms]
        return np.concatenate(parts)
