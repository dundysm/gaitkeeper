"""Read a teleop-walking-benchmark adapter by experiment, then write its contracts.

``probe(adapter)`` runs the compiled adapter (``twb_adapter.Adapter``) on chosen inputs and
reads off what it does:

* the action map: which motor each action element moves, by how much (the scale), from
  where (the offset), and where it stops (a clip);
* the observation layout: which input each observation element follows and with what gain
  (base_ang_vel, projected_gravity, velocity_commands, base_lin_vel, joint_pos_rel,
  joint_vel_rel, last_action), per joint whether the position is measured or faked from the
  last action, and the default pose the positions are taken relative to;
* the history: length, layout, order and how it is filled after a reset (or that the port
  feeds zeros until its second step);
* a gait clock: its period, its phase at the first step, and the command norm below which
  it is zeroed; a clock whose period follows the command speed; a two-leg clock held while
  standing; a clock that runs only while the port passes its command; clock inputs per foot
  with the stance warped (walk-these-ways);
* what the port does to the command: its own steering from the harness's task (a waypoint
  follower, or the command's direction at a speed set by the distance), and a gate that
  passes it only while it is nonzero (after a warm-up), above some size, or once the
  waypoint is far enough (a walk latch);
* motors the port observes but no action drives (a held waist, the harness's arms), and the
  harness's arm targets when the port observes those;
* a layout no single history window describes (the current frame split around earlier
  ones, or repeated in front of a history that holds it): read one frame at a time, the
  layout stated as [term, lag] chunks;
* a port with variants (made by name), and one that runs a walking and a standing graph
  picked by the command;
* the constants behind the interface: gains, ``owned()`` and the command limits.

Everything is measured as a finite difference around the benchmark's stance, so a gain is
exact to float32 rounding wherever the adapter is linear in that input. What the probe cannot
explain (an observation element that follows nothing it varies, a clock it does not know)
is reported, and no contract is written for that adapter.

``contracts(probe_result, mjcf)`` turns a result into two contracts: ``trained`` (every
policy joint driven and observed for real) and ``port`` (what the benchmark runs: motors from
``owned()`` on held by the harness at its stance with armature gains, observed as the adapter
observes them).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .twb_adapter import NUM_MOTOR, Adapter, preprocessed

# The harness (main.cpp): the pose it starts from and holds unowned motors at.
STANCE = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, 0.0, 0.0, 0.0]
    + [0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
)
POLICY_DT = 0.02  # main.cpp PERIOD_S
# main.cpp (rhoyn/teleop-walking-benchmark@4ed23c2 on): the harness applies the policy's kp()
# and kd() to every leg and waist motor (below ARM_LEFT_FIRST), and to the arms only when the
# policy owns all 29 motors; other arm motors keep the harness's armature gains.
ARM_LEFT_FIRST = 15
DELTA = 1e-2
WARM = 100  # 2 s: past start-up ramps (legged_rl_lab scales its actions in over 0.8 s)
LAGS = 16
BASE_CMD = (0.4, 0.0, 0.0)  # moving, so a clock gated on the command runs
SCALARS = {
    "gyro": ("base_ang_vel", 3),
    "lin_vel": ("base_lin_vel", 3),
    "gravity": ("projected_gravity", 3),
    "cmd": ("velocity_commands", 3),
}
JOINTS = {"q": "joint_pos_rel", "dq": "joint_vel_rel", "action": "last_action"}
# Joint orders a policy is commonly trained in, as MuJoCo/SDK motor indices.
KNOWN_ORDERS = {
    "sdk": list(range(NUM_MOTOR)),
    "isaaclab": [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17]
    + [24, 18, 25, 19, 26, 20, 27, 21, 28],
}
TOL = 1e-5


class ProbeError(ValueError):
    pass


@dataclass
class ProbeResult:
    name: str
    obs_dim: int
    act_dim: int
    owned: int
    limits: dict[str, float]
    kp: list[float]  # per motor, valid below `owned` (and where `gains_len` covers)
    kd: list[float]
    gains_len: int
    p2m: list[int]  # policy joint k -> motor
    p2m_from: list[str]  # "action", "obs", or the name of a known order
    action_scale: list[float]
    action_offset: list[float | None]
    action_clip: list | None  # per policy joint [lo, hi] of the motor target, if bounded
    discarded: list[int]  # policy joints whose action moves no motor
    terms: list[dict[str, Any]]
    history: dict[str, Any]
    default_pose: list[float | None]  # per policy joint, from joint_pos_rel
    joint_obs: list[str]  # per policy joint: "real", "echo_action" or "default"
    clock: dict[str, Any] | None
    engines: list[dict[str, Any]]
    hold_target: list[float] = field(
        default_factory=list
    )  # per motor, the port's target at zero action
    findings: list[str] = field(default_factory=list)
    action_lag: int = 1  # the observed action is this many steps old
    shaping: dict[str, Any] | None = None  # the port's own command from the harness's task
    observed_extra: list[int] = field(default_factory=list)  # motors observed, not driven
    switch: dict[str, Any] | None = None  # two graphs picked by the command
    gate: dict[str, Any] | None = None  # when the port passes its command at all
    extra_default: list[float] = field(default_factory=list)  # their joint_pos_rel offsets
    target_default: dict[int, float] = field(default_factory=dict)  # joint_target_rel offsets

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _inputs(cmd=BASE_CMD) -> dict[str, np.ndarray]:
    return {
        "q": STANCE.copy(),
        "dq": np.zeros(NUM_MOTOR),
        "gyro": np.zeros(3),
        "lin_vel": np.zeros(3),
        "gravity": np.array([0.0, 0.0, -1.0]),
        "cmd": np.array(cmd, float),
        "arm": STANCE.copy(),  # the harness's arm targets
    }


def _run(
    a: Adapter,
    steps: int,
    at: int | None = None,
    family: str | None = None,
    element: int = 0,
    delta: float = DELTA,
    cmd=BASE_CMD,
    task=None,
) -> tuple[np.ndarray, np.ndarray]:
    a.reset()
    obs, tgt = [], []
    for t in range(steps):
        x = _inputs(cmd)
        act = np.zeros(a.act_dim)
        if t == at and family is not None:
            if family == "action":
                act[element] = delta
            else:
                x[family][element] += delta
        q_t, seen = a.step(
            x["q"],
            x["dq"],
            x["gyro"],
            x["lin_vel"],
            x["gravity"],
            x["cmd"],
            action=act,
            arm_pose=x["arm"],
            task=task,
        )
        obs.append(seen[0][: a.obs_dim])
        tgt.append(q_t)
    return np.array(obs), np.array(tgt)


def _nz(v: np.ndarray, tol: float = TOL) -> dict[int, float]:
    return {int(i): float(v[i]) for i in np.flatnonzero(np.abs(v) > tol)}


def _round(x: float, sig: int = 5) -> float:
    if x == 0 or not math.isfinite(x):
        return x
    return float(round(x, sig - 1 - int(math.floor(math.log10(abs(x))))))


def _probe(a: Adapter) -> ProbeResult:
    n, steps = a.act_dim, WARM + LAGS + 1
    findings: list[str] = []
    base_obs, base_tgt = _run(a, steps)
    varying = np.flatnonzero(np.abs(np.diff(base_obs[WARM - 4 :], axis=0)).max(axis=0) > TOL)

    # -- responses: family, element -> per lag {obs index: gain}
    resp: dict[tuple[str, int], list[dict[int, float]]] = {}
    tresp: dict[int, dict[int, float]] = {}
    sizes = {"q": NUM_MOTOR, "dq": NUM_MOTOR, "action": n} | {k: 3 for k in SCALARS}
    sizes["arm"] = NUM_MOTOR  # the harness's arm targets: does the port observe them?
    for fam, size in sizes.items():
        for e in range(ARM_LEFT_FIRST if fam == "arm" else 0, size):
            o, t = _run(a, steps, WARM, fam, e)
            d = (o[WARM:] - base_obs[WARM:]) / DELTA
            resp[(fam, e)] = [_nz(d[k]) for k in range(len(d))]
            if fam == "action":
                tresp[e] = _nz((t[WARM] - base_tgt[WARM]) / DELTA)

    # -- action map
    p2m: list[int | None] = [None] * n
    p2m_from = ["?"] * n
    scale = [math.nan] * n
    offset: list[float | None] = [None] * n
    for k in range(n):
        r = tresp[k]
        if len(r) == 1:
            m, g = next(iter(r.items()))
            p2m[k], p2m_from[k], scale[k] = m, "action", _round(g)
            offset[k] = _round(float(base_tgt[WARM][m]) - 0.0)
        elif len(r) > 1:
            findings.append(f"action {k} moves {len(r)} motors {sorted(r)}: not a joint target")
    discarded = [k for k in range(n) if not tresp[k]]
    k0 = next((k for k in range(n) if p2m_from[k] == "action"), None)
    if k0 is not None:
        _, t0 = _run(a, 1, 0, "action", k0)
        g0 = (t0[0][p2m[k0]] - base_tgt[0][p2m[k0]]) / DELTA
        if abs(g0 - scale[k0]) > 1e-4 * max(1.0, abs(scale[k0])):
            findings.append(
                f"the action gain at the first step is {g0:.4g}, {scale[k0]:.4g} after "
                f"{WARM * POLICY_DT:g} s: the port ramps its actions in"
            )

    # The action is observed one step later (last_action), or more when a port keeps an
    # older one: the first lag any observation element responds to an action at.
    alag = min(
        (
            next((j for j in range(1, len(resp[("action", e)])) if resp[("action", e)][j]), 99)
            for e in range(n)
        ),
        default=1,
    )
    alag = 1 if alag == 99 else alag
    if alag > 1:
        findings.append(f"the port observes the action from {alag} steps back")

    # -- joint blocks at the newest frame (state: lag 0; action: its lag)
    def newest(fam: str, e: int) -> dict[int, float]:
        lag = alag if fam == "action" else 0
        return resp[(fam, e)][lag]

    def is_diff(m: int, i: int, g: float) -> bool:
        """A position response that reverses the next step, in place: (q - q_prev) / dt."""
        back = resp[("q", m)][1].get(i) if len(resp[("q", m)]) > 1 else None
        return back is not None and abs(back + g) <= 1e-3 * abs(g)

    # A motor seen through joint_pos_rel also names its policy joint, for actions that move
    # nothing: the block start comes from the joints the action map does place.
    def block_start(fam: str) -> int | None:
        votes: Counter[int] = Counter()
        for k in range(n):
            e = k if fam == "action" else p2m[k]
            if e is None:
                continue
            for i, g in newest(fam, e).items():
                if fam == "q" and is_diff(e, i, g):
                    continue
                votes[i - k] += 1
        return votes.most_common(1)[0][0] if votes else None

    starts = {fam: block_start(fam) for fam in JOINTS}
    if starts["q"] is not None:
        for m in range(NUM_MOTOR):
            if m in p2m:
                continue
            for i, g in newest("q", m).items():
                if is_diff(m, i, g):
                    continue
                k = i - starts["q"]
                if 0 <= k < n and p2m[k] is None:
                    p2m[k], p2m_from[k] = m, "obs"
    missing = [k for k in range(n) if p2m[k] is None]
    if missing:
        known = {k: m for k, m in enumerate(p2m) if m is not None}
        for oname, order in KNOWN_ORDERS.items():
            if len(order) >= n and all(order[k] == m for k, m in known.items()):
                for k in missing:
                    p2m[k], p2m_from[k] = order[k], oname
                findings.append(
                    f"policy joints {missing} move no motor and are not measured: placed by the "
                    f"{oname} joint order, which fits the {len(known)} placed joints"
                )
                break
        else:
            raise ProbeError(
                f"{a.name}: policy joints {missing} move no motor and fit no known joint order"
            )
    if len(set(p2m)) != n or max(p2m) >= NUM_MOTOR:  # type: ignore[type-var]
        raise ProbeError(f"{a.name}: joint map {p2m} repeats or leaves the 29 motors")
    p2m_i = [int(m) for m in p2m]  # type: ignore[arg-type]

    # -- history: follow one element back in time. Inputs return to the baseline after
    # the perturbed step, so at lag j only the slot holding the frame j steps old responds.
    hist_fam = next((f for f in ("gyro", "gravity", "cmd", "lin_vel") if newest(f, 0)), None)
    he = 0
    if hist_fam is None:
        he = next((m for m in p2m_i if newest("q", m)), None)
        if he is None:
            raise ProbeError(
                f"{a.name}: the observation follows none of gyro, gravity, command or joint "
                "positions (a reference-motion or latent policy)"
            )
        hist_fam = "q"
    lag0 = alag if hist_fam == "action" else 0
    i0 = next(iter(newest(hist_fam, he)))
    age_of: dict[int, int] = {}
    slots = []
    for lag in range(lag0 + 1, len(resp[(hist_fam, he)])):
        r_ = list(resp[(hist_fam, he)][lag])
        if not r_:
            break
        slots.append(min(r_, key=lambda j: abs(j - i0)))
    H = 1 + len(slots)
    history: dict[str, Any] = {
        "length": H,
        "layout": "term_major",
        "order": "oldest_first",
        "init": "repeat_first",  # one frame: nothing to fill
    }
    if slots:
        d = i0 - slots[0]
        hdim = {"q": n, "dq": n}.get(hist_fam, 3)
        history["layout"] = "term_major" if abs(d) == hdim else "time_major"
        history["order"] = "oldest_first" if d > 0 else "newest_first"
        o, _ = _run(a, 1, 0, hist_fam, he)
        first = _nz((o[0] - base_obs[0]) / DELTA)
        history["init"] = "repeat_first" if len(first) > 1 else "zeros"
    if not (np.abs(base_obs[0]) > TOL).any() and (np.abs(base_obs[1]) > TOL).any():
        # nothing at the first step, though gravity and the stance are fed: the port builds
        # its first observation a step late and the policy sees zeros
        history["first_frame"] = "zeros"
        findings.append("the port feeds zeros at the first step (it builds no observation yet)")
    # the age of every probed element: the first lag it responds at
    for (fam, _e), lags in resp.items():
        l0 = alag if fam == "action" else 0
        for lag in range(l0, len(lags)):
            for i in lags[lag]:
                age_of.setdefault(i, lag - l0)
                age_of[i] = min(age_of[i], lag - l0)

    # -- what each newest-frame element follows: (term id, element, gain)
    desc: dict[int, tuple[str, int, float]] = {}
    joint_obs = ["real"] * n
    default_pose: list[float | None] = [None] * n
    echo_gain: dict[int, float] = {}
    kq = {m: k for k, m in enumerate(p2m_i)}

    def put(i: int, d_: tuple[str, int, float]) -> None:
        if i in desc and desc[i][:2] != d_[:2]:
            raise ProbeError(f"{a.name}: obs {i} follows both {desc[i][:2]} and {d_[:2]}")
        desc[i] = d_

    for fam, (tid, dim) in SCALARS.items():
        for e in range(dim):
            for i, g in newest(fam, e).items():
                put(i, (tid, e, _round(g)))
    for i, d_ in _gravity_angles(a, desc, base_obs, findings).items():
        desc[i] = d_
    # A motor no action drives can still be observed (a waist the port holds, arms the
    # harness holds): it joins the joint terms after the policy's joints, as element n + j.
    extra: list[int] = []
    extra_default: dict[int, float] = {}
    for m in range(NUM_MOTOR):
        for fam, tid in (("q", "joint_pos_rel"), ("dq", "joint_vel_rel")):
            for i, g in newest(fam, m).items():
                if m not in kq:
                    if m not in extra:
                        extra.append(m)
                    k = n + extra.index(m)
                else:
                    k = kq[m]
                if fam == "q" and is_diff(m, i, g):
                    put(i, ("joint_vel_diff", k, _round(g * POLICY_DT)))
                    continue
                put(i, (tid, k, _round(g)))
                if fam == "q" and k < n:
                    default_pose[k] = _round(STANCE[m] - base_obs[WARM][i] / g)
                elif fam == "q":
                    extra_default[m] = _round(STANCE[m] - base_obs[WARM][i] / g)
    target_default: dict[int, float] = {}
    for m in range(ARM_LEFT_FIRST, NUM_MOTOR):
        for i, g in newest("arm", m).items():
            if m in kq:
                raise ProbeError(f"{a.name}: obs {i} follows the harness's target for motor {m}")
            if m not in extra:
                extra.append(m)
            put(i, ("joint_target_rel", n + extra.index(m), _round(g)))
            target_default[m] = _round(STANCE[m] - base_obs[WARM][i] / g)
    if target_default:
        findings.append(
            f"the port observes the harness's targets for motor(s) {sorted(target_default)} "
            "(joint_target_rel)"
        )
    if extra:
        findings.append(
            f"the port observes motor(s) {extra} that no action drives (held by the port or "
            "the harness): read after the policy's joints"
        )
    sq = starts["q"]
    for k in range(n):
        for i, g in newest("action", k).items():
            if i in desc:
                continue
            if (
                sq is not None
                and i - sq == k
                and ("joint_pos_rel", k) not in {v[:2] for v in desc.values()}
            ):
                joint_obs[k] = "echo_action"
                echo_gain[k] = _round(g)
                put(i, ("joint_pos_rel", k, 1.0))
            else:
                put(i, ("last_action", k, _round(g)))

    # A port that makes its own command from the harness's task may not pass the command
    # through when there is no waypoint (the probe's runs above): its command elements answer
    # the task instead. Read them, and how the port shapes its command, from runs with one.
    task_old: set[int] = set()
    shaping = None
    if sorted(d_[1] for d_ in desc.values() if d_[0] == "velocity_commands") != [0, 1, 2]:
        tc = _task_command(a, desc, H, findings)
        if tc is not None:
            cdesc, task_old, shaping = tc
            for i in [i for i, d_ in desc.items() if d_[0] == "velocity_commands"]:
                desc.pop(i)
            for i, d_ in cdesc.items():
                put(i, d_)

    # A port that passes its command only while told to walk (and maybe not at first): a
    # flag that follows whether the command is zero, and a clock that runs only then.
    gate = None
    cand = [i for i in range(a.obs_dim) if i not in desc and i not in age_of and i not in task_old]
    tg = _gate(a, cand, H, history, findings)
    if tg is not None:
        gate, gdesc, gold = tg
        for i, d_ in gdesc.items():
            put(i, d_)
        task_old |= gold
    elif shaping is None:
        gate = _norm_gate(a, desc, findings)

    # clock elements and their ages; constants
    # (a clock whose rate follows the command responds to it a step later, so it can carry
    # an age; what matters is that nothing above describes it)
    vary = [int(i) for i in varying if int(i) not in desc]
    newest_clock = _newest_copies(base_obs, vary, H)
    clock = None
    if newest_clock:
        clock, cdesc = _clock(a, newest_clock, findings, gate)
        for i, d_ in cdesc.items():
            put(i, d_)
    explained_old = {i for i, ag in age_of.items() if ag > 0} | (set(vary) - set(newest_clock))
    explained_old |= task_old
    F = a.obs_dim // H

    def old_slot(i: int) -> bool:
        sign = 1 if history["order"] == "oldest_first" else -1
        step = n if history["layout"] == "term_major" else F
        for s0 in starts.values():
            if s0 is None:
                continue
            for j in range(1, H):
                lo = s0 - sign * j * step
                if lo <= i < lo + n:
                    return True
        return False

    rest = [
        i for i in range(a.obs_dim) if i not in desc and i not in explained_old and not old_slot(i)
    ]
    common = {}
    for tid in ("joint_pos_rel", "joint_vel_rel", "last_action"):
        g = [v[2] for v in desc.values() if v[0] == tid]
        common[tid] = Counter(g).most_common(1)[0][0] if g else 1.0
    sd = starts["dq"]
    consts: dict[int, float] = {}
    for i in rest:
        v = float(base_obs[WARM][i])
        if abs(v) < TOL and sq is not None and 0 <= i - sq < n and joint_obs[i - sq] == "real":
            k = i - sq
            if not any(d_[:2] == ("joint_pos_rel", k) for d_ in desc.values()):
                joint_obs[k] = "default"
                put(i, ("joint_pos_rel", k, common["joint_pos_rel"]))
                continue
        if abs(v) < TOL and sd is not None and 0 <= i - sd < n and joint_obs[i - sd] != "real":
            put(i, ("joint_vel_rel", i - sd, common["joint_vel_rel"]))
            continue
        consts[i] = _round(v, 6)
        put(i, ("constant", 0, 1.0))
    if consts:
        a.reset()
        x = _inputs()
        for _ in range(H + 2):  # every history slot filled, whatever the port does at start
            _, seen_q = a.step(
                x["q"],
                x["dq"],
                x["gyro"],
                x["lin_vel"],
                x["gravity"],
                x["cmd"],
                arm_pose=STANCE,
                quat=(0.0, 0.0, 0.0, 1.0),
            )
        moved = [i for i in consts if abs(seen_q[0][i] - consts[i]) > 1e-4]
        if moved:
            raise ProbeError(
                f"{a.name}: obs {moved} follow base_quat directly (not gravity or gyro); "
                "gaitkeeper builds no such term"
                + (
                    f"; at the upright quaternion (wxyz, as the harness passes it) they read "
                    f"{[consts[i] for i in moved]}"
                )
            )
    if consts and H > 1:
        # A constant slot repeats in every frame of a time-major history: keep the newest
        # frame's copy as the term; the older copies must hold the same value.
        if history["layout"] != "time_major":
            raise ProbeError(
                f"{a.name}: constant observation elements in a term-major history: "
                f"{sorted(consts)[:6]}"
            )
        lo = (H - 1) * F if history["order"] == "oldest_first" else 0
        newest_consts = {i: v for i, v in consts.items() if lo <= i < lo + F}
        for i, v in consts.items():
            j = lo + (i % F)
            if j not in newest_consts or abs(newest_consts[j] - v) > 1e-6:
                raise ProbeError(
                    f"{a.name}: obs {i} is constant ({v:g}) but its newest copy {j} is not"
                )
        for i in set(consts) - set(newest_consts):
            desc.pop(i, None)
        consts = newest_consts
    if consts:
        findings.append(
            f"{len(consts)} observation element(s) are constants the port feeds: "
            + ", ".join(f"[{i}]={v:g}" for i, v in list(consts.items())[:8])
            + (" ..." if len(consts) > 8 else "")
        )
    lost = [k for k in range(n) if joint_obs[k] == "real" and default_pose[k] is None]
    zero_vel = [
        k
        for k in range(n)
        if joint_obs[k] == "real"
        and not any(d_[:2] == ("joint_vel_rel", k) for d_ in desc.values())
        and any(d_[0] == "joint_vel_rel" for d_ in desc.values())
    ]
    if zero_vel:
        findings.append(f"joint_vel_rel of policy joints {zero_vel} is not observed")

    # -- terms: runs of the newest frame that follow one term
    full_dim = {"joint_pos_rel": n, "joint_vel_rel": n, "last_action": n, "gait_phase": 2}
    full_dim |= {"joint_vel_diff": n, "gait_phase_speed": 2}
    full_dim |= {"gait_phase_legs": 4} | {tid: d for tid, d in SCALARS.values()}
    full_dim |= {"gait_phase_gated": 2, "command_gate": 1, "gravity_euler": 2}
    full_dim |= {"joint_target_rel": n}
    full_dim |= {"gait_phase_feet": len((clock or {}).get("offsets", []))}
    order = sorted(desc)
    terms: list[dict[str, Any]] = []
    for i in order:
        tid, e, g = desc[i]
        last = terms[-1] if terms else None
        if (
            last is not None
            and last["id"] == tid
            and (tid == "constant" or e not in last["elements"])
        ):
            last["elements"].append(e)
            last["scale"].append(g)
            last["values"].append(consts.get(i))
            continue
        terms.append(
            {"id": tid, "elements": [e], "scale": [g], "values": [consts.get(i)], "start": i}
        )
    seen_ids: Counter[str] = Counter()
    for t in terms:
        seen_ids[t["id"]] += 1
        t["source_name"] = t["id"] if seen_ids[t["id"]] == 1 else f"{t['id']}_{seen_ids[t['id']]}"
        t["dim"] = len(t["elements"])
        if t["id"] == "constant":
            t["value"] = t.pop("values")
            t["scale"] = [1.0] * t["dim"]
            t.pop("elements")
            continue
        t.pop("values")
        full = full_dim[t["id"]]
        if (
            t["id"] in ("joint_pos_rel", "joint_vel_rel", "joint_vel_diff", "joint_target_rel")
            and max(t["elements"]) >= n
        ):
            t["extra"] = list(extra)
            full = n + len(extra)
        if t["elements"] != list(range(full)):
            t["index"] = t["elements"]
        t.pop("elements")
    if sum(t["dim"] for t in terms) * H != a.obs_dim:
        raise ProbeError(
            f"{a.name}: {sum(t['dim'] for t in terms)} elements per frame x {H} frames != "
            f"obs {a.obs_dim}"
        )
    del lost

    # -- target bounds: drive each placed joint far each way
    bounds: list[list[float] | None] = [None] * n
    for k in range(n):
        if p2m_from[k] != "action":
            continue
        lo_hi = []
        for sign in (-1.0, 1.0):
            _, t_ = _run(a, WARM + 1, WARM, "action", k, delta=sign * 1000.0)
            lo_hi.append(float(t_[WARM][p2m_i[k]]))
        bounds[k] = sorted(lo_hi)
    clip = None
    pending_raw_clip = None
    clipped = [
        k
        for k in range(n)
        if bounds[k] is not None and (bounds[k][1] - bounds[k][0]) < 0.999 * 2000 * abs(scale[k])
    ]
    if clipped:
        raw = {
            _round(
                max(abs(bounds[k][0] - offset[k]), abs(bounds[k][1] - offset[k])) / abs(scale[k]), 4
            )
            for k in range(n)
            if bounds[k] is not None and offset[k] is not None
        }
        if len(raw) == 1:
            c_raw = next(iter(raw))
            findings.append(f"actions are clipped to +-{c_raw:g} before scaling")
        else:
            c_raw = None
        clip = [[_round(b_[0]), _round(b_[1])] if b_ is not None else None for b_ in bounds]
        pending_raw_clip = c_raw
    static = _static_arrays(preprocessed(a.path), n)
    if any(default_pose[k] is None for k in range(n)):
        measured = [math.nan if d_ is None else d_ for d_ in default_pose]
        for k in range(n):  # the action offset is the default where both exist
            if default_pose[k] is None and offset[k] is not None:
                measured[k] = offset[k]
        fill = _match_array(static, measured, p2m_i, "DEFAULT")
        if fill is not None:
            for k in range(n):
                if default_pose[k] is None:
                    default_pose[k] = _round(fill[k])
            findings.append(
                f"default pose of unmeasured joints from the source array {fill.name}, which "
                "matches every measured default"
            )
    for k in discarded:
        if k in echo_gain:
            scale[k] = echo_gain[k]
        offset[k] = default_pose[k]
    if any(not math.isfinite(scale[k]) for k in discarded):
        fill = _match_array(static, scale, p2m_i, "SCALE")
        if fill is None:
            fill = _match_scalar(preprocessed(a.path), scale, "ACTION_SCALE", n)
        if fill is not None:
            for k in discarded:
                if not math.isfinite(scale[k]):
                    scale[k] = fill[k]
            findings.append(
                f"scales of the discarded actions from the source array {fill.name}, which "
                "matches every measured scale"
            )
    for k in range(n):
        if default_pose[k] is None:
            default_pose[k] = offset[k] if offset[k] is not None else float(STANCE[p2m_i[k]])
    if clip is not None and pending_raw_clip is not None:
        for k in range(n):
            if clip[k] is None and offset[k] is not None and math.isfinite(scale[k]):
                lim = abs(scale[k]) * pending_raw_clip
                clip[k] = [_round(offset[k] - lim), _round(offset[k] + lim)]
    if discarded:
        findings.append(
            f"{len(discarded)} action element(s) move no motor (the port discards them): "
            + (
                "scales from the faked joint_pos_rel"
                if echo_gain
                else "their scales are read, not measured"
            )
        )
    if any(not math.isfinite(x) for x in scale):
        raise ProbeError(f"{a.name}: action scales of {discarded} cannot be measured or read")
    kp = np.zeros(NUM_MOTOR)
    kd = np.zeros(NUM_MOTOR)
    kp[:], kd[:] = a.kp_raw, a.kd_raw
    return ProbeResult(
        name=a.name,
        obs_dim=a.obs_dim,
        act_dim=n,
        owned=a.owned,
        limits=a.limits,
        kp=[_round(float(x), 7) for x in kp],
        kd=[_round(float(x), 7) for x in kd],
        gains_len=_gains_len(a),
        p2m=p2m_i,
        p2m_from=p2m_from,
        action_scale=scale,
        action_offset=offset,
        action_clip=clip,
        discarded=discarded,
        terms=terms,
        history=history,
        default_pose=default_pose,
        joint_obs=joint_obs,
        clock=clock,
        engines=a.engines,
        hold_target=[_round(float(x), 6) for x in base_tgt[WARM]],
        findings=findings,
        action_lag=alag,
        shaping=shaping if shaping is not None else _shaping(a, desc, findings),
        observed_extra=extra,
        extra_default=[float(extra_default.get(m, 0.0)) for m in extra],
        target_default={int(m): float(v) for m, v in target_default.items()},
        switch=_switch(a, p2m_i, p2m_from, findings),
        gate=gate,
    )


class _TaskDefault:
    """An adapter run with a waypoint whenever the caller gives none (one far enough to open
    a port's walk latch), so its command passes as an ungated port's does."""

    def __init__(self, a: Adapter, task: np.ndarray):
        self._a = a
        self._task = np.r_[task, np.zeros(60)]

    def __getattr__(self, k: str) -> Any:
        return getattr(self._a, k)

    def step(self, *args: Any, task: Any = None, **kw: Any) -> tuple[np.ndarray, list[np.ndarray]]:
        return self._a.step(*args, task=self._task if task is None else task, **kw)


FAR_TASK = np.array([1.0, 0.3, math.cos(0.2), math.sin(0.2)])


def _latch_task(a: Adapter) -> np.ndarray | None:
    """A task under which a port passes its command when it passes none without a waypoint
    (a walk latch on the task; asap): the far task, or None when that is not the port's way."""

    def responses(task) -> list[dict[int, float]]:
        base, _ = _run(a, WARM + 1, task=task)
        out = []
        for e in range(3):
            o, _ = _run(a, WARM + 1, WARM, "cmd", e, task=task)
            out.append(_nz((o[WARM] - base[WARM]) / DELTA))
        return out

    if all(responses(None)):
        return None
    far = responses(np.r_[FAR_TASK, np.zeros(60)])
    if not all(far):
        return None
    other = responses(np.r_[[1.5, -0.4, 1.5 * math.cos(-0.5), 1.5 * math.sin(-0.5)], np.zeros(60)])
    same = all(
        set(f) == set(o_) and all(abs(f[i] - o_[i]) < 1e-6 for i in f) for f, o_ in zip(far, other)
    )
    return FAR_TASK if same else None


def _with_latch(a: Adapter, r: ProbeResult, findings: list[str]) -> ProbeResult:
    """A reading made with the latch held open, completed: the latch's thresholds (a gate on
    the task), the flag it feeds, and the clocks it stops at phase zero."""
    cmd_el = {}
    for t in r.terms:
        if t["id"] == "velocity_commands":
            for k, e in enumerate(t.get("index", range(t["dim"]))):
                cmd_el[e] = (t["start"] + k, t["scale"][k])
    if 0 not in cmd_el:
        raise ProbeError(f"{a.name}: a walk latch on the task, but no vx command element")
    i_vx, g_vx = cmd_el[0]

    def latched_after(tasks: list) -> bool:
        a.reset()
        x = _inputs()
        for tk in tasks:
            _, seen = a.step(
                x["q"],
                x["dq"],
                x["gyro"],
                x["lin_vel"],
                x["gravity"],
                x["cmd"],
                arm_pose=STANCE,
                task=np.r_[tk, np.zeros(60)],
            )
        return abs(seen[0][i_vx] - g_vx * BASE_CMD[0]) < 1e-6

    def bisect(f, lo: float, hi: float) -> float:
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if f(mid) else (mid, hi)
        return _round(hi, 4)

    zero = np.zeros(4)
    on = [FAR_TASK] * 3
    enter_d = bisect(lambda d: latched_after([np.array([d, 0.0, d, 0.0])]), 0.0, 1.0)
    enter_y = bisect(lambda y: latched_after([np.array([0.0, y, 0.0, 0.0])]), 0.0, 1.0)
    # once open, it stays open until the waypoint is within both exit thresholds: the
    # bisections find the first value that opens it (enter) and the first that keeps it open
    exit_d = bisect(lambda d: latched_after(on + [np.array([d, 0.0, d, 0.0])]), 0.0, enter_d)
    exit_y = bisect(lambda y: latched_after(on + [np.array([0.0, y, 0.0, 0.0])]), 0.0, enter_y)
    gate = {
        "on": "task_latch",
        "enter": {"dist": enter_d, "yaw": enter_y},
        "exit": {"dist": exit_d, "yaw": exit_y},
    }
    if latched_after([zero]):
        raise ProbeError(f"{a.name}: the command passes with no waypoint after all")
    # the flag: constants under the open latch that read otherwise when it is shut
    shut, _ = _run(a, WARM + 1)
    opened, _ = _run(a, WARM + 1, task=np.r_[FAR_TASK, np.zeros(60)])
    terms = []
    for t in r.terms:
        if t["id"] != "constant":
            terms.append(t)
            continue
        idx = [t["start"] + k for k in range(t["dim"])]
        flag = [abs(opened[WARM][i] - shut[WARM][i]) > TOL for i in idx]
        run_start = 0
        for k in range(1, t["dim"] + 1):
            if k < t["dim"] and flag[k] == flag[run_start]:
                continue
            sub = list(range(run_start, k))
            if flag[run_start]:
                for j in sub:
                    if abs(shut[WARM][idx[j]]) > TOL:
                        raise ProbeError(
                            f"{a.name}: obs {idx[j]} is a flag that is not 0 when shut"
                        )
                terms.extend(
                    {
                        "id": "command_gate",
                        "dim": 1,
                        "scale": [_round(t["value"][j])],
                        "start": idx[j],
                        "source_name": "command_gate",
                    }
                    for j in sub
                )
            else:
                terms.append(
                    dict(
                        t,
                        dim=len(sub),
                        value=[t["value"][j] for j in sub],
                        scale=[1.0] * len(sub),
                        start=idx[sub[0]],
                    )
                )
            run_start = k
    # unique names, as the reading names them
    seen_ids: Counter[str] = Counter()
    for t in terms:
        seen_ids[t["id"]] += 1
        t["source_name"] = t["id"] if seen_ids[t["id"]] == 1 else f"{t['id']}_{seen_ids[t['id']]}"
    r.terms = terms
    # a clock the latch stops reads phase zero while shut
    if r.clock and r.clock.get("id") == "gait_phase":
        for t in r.terms:
            if t["id"] != "gait_phase":
                continue
            for k, e in enumerate(t.get("index", range(t["dim"]))):
                want = 0.0 if e == 0 else t["scale"][k]
                if abs(shut[WARM][t["start"] + k] - want) > 1e-5:
                    raise ProbeError(f"{a.name}: a clock the walk latch stops, not at phase zero")
        r.clock["gate"] = "zero_phase"
    r.gate = gate
    findings.append(
        "the port passes its command only once the waypoint is farther than "
        f"{enter_d:g} m or {enter_y:g} rad off, until it is within {exit_d:g} m and "
        f"{exit_y:g} rad (a walk latch on the task)"
    )
    return r


def probe(a: Adapter) -> ProbeResult:
    """Read an adapter (see the module's docstring). A port that passes its command only
    once a waypoint is far enough is read with one, then its latch is measured."""
    lt = _latch_task(a)
    if lt is None:
        return _probe(a)
    r = _probe(_TaskDefault(a, lt))  # type: ignore[arg-type]
    return _with_latch(a, r, r.findings)


def _shaping(
    a: Adapter, desc: dict[int, tuple[str, int, float]], findings: list[str]
) -> dict[str, Any] | None:
    """Whether the port turns the harness's task into its own command, and if so the
    parameters of the waypoint follower it uses (commands.waypoint_follow), measured."""
    el = {d_[1]: (i, d_[2]) for i, d_ in desc.items() if d_[0] == "velocity_commands"}
    if sorted(el) != [0, 1, 2]:
        return None

    def seen(cmd, task) -> np.ndarray:
        o, _ = _run(a, 3, cmd=cmd, task=np.r_[task, np.zeros(60)])
        return np.array([o[2][el[e][0]] / el[e][1] for e in range(3)])

    c0 = (0.4, 0.0, 0.1)
    probes = [(0.2, 0.1, 0.2, 0.0), (3.0, 0.3, 2.0, 1.5), (0.6, -0.4, 0.3, -0.5)]
    if all(np.allclose(seen(c0, tk), c0, atol=1e-6) for tk in probes):
        return None
    walk_p = seen(c0, (0.05, 0.0, 0.05, 0.0))[0] / 0.05
    walk_speed = seen(c0, (50.0, 0.0, 50.0, 0.0))[0]
    yaw_p = seen(c0, (0.01, 0.05, 0.01, 0.0))[2] / 0.05
    yaw_rate_abs = abs(seen(c0, (0.01, 3.0, 0.01, 0.0))[2])
    vy_abs = abs(seen(c0, (50.0, 0.0, 50.0 * math.cos(1.2), 50.0 * math.sin(1.2)))[1])
    vy_free = walk_speed * math.cos(1.2) * math.sin(1.2)
    if abs(vy_abs - vy_free) < 1e-6:  # did not bind: the port's command limit
        vy_abs = float(a.limits.get("vy_abs", vy_abs))
    # facing weight against distance: bearing b, yaw error 0 -> yaw rate yaw_p * w(d) * b
    b = 0.2

    def face_w(d: float) -> float:
        wz = seen(c0, (d, 0.0, d * math.cos(b), d * math.sin(b)))[2]
        return wz / (yaw_p * b)

    lo, hi = 0.0, 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if face_w(mid) < 1e-6 else (lo, mid)
    near = hi
    lo, hi = near, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if face_w(mid) < 1 - 1e-6 else (lo, mid)
    far = hi
    params = {
        "walk_p": _round(walk_p),
        "walk_speed": _round(walk_speed),
        "yaw_p": _round(yaw_p),
        "yaw_rate_abs": _round(yaw_rate_abs),
        "vy_abs": _round(vy_abs),
        "face_near_m": _round(near, 4),
        "face_far_m": _round(far, 4),
    }
    findings.append(
        "the port steers from the harness's task, not its velocity command: a waypoint "
        "follower with " + ", ".join(f"{k} {v:g}" for k, v in params.items())
    )
    return {"kind": "waypoint_follow", "params": params}


def _gravity_angles(
    a: Adapter, desc: dict[int, tuple[str, int, float]], base_obs: np.ndarray, findings: list[str]
) -> dict[int, tuple[str, int, float]]:
    """Elements that follow gravity but not linearly: roll atan2(-g_y, -g_z) and pitch
    asin(g_x) computed from it (gravity_euler). Returns their new descriptions."""
    els = {i: d_ for i, d_ in desc.items() if d_[0] == "projected_gravity"}
    if not els:
        return {}
    tilts = [np.array([0.35, -0.3, -0.9]), np.array([-0.4, 0.5, -0.8]), np.array([0.2, 0.6, -0.75])]
    seen = []
    for g in tilts:
        g = g / np.linalg.norm(g)
        a.reset()
        x = _inputs()
        for _ in range(3):
            _, sv = a.step(x["q"], x["dq"], x["gyro"], x["lin_vel"], g, x["cmd"], arm_pose=STANCE)
        seen.append((g, sv[0][: a.obs_dim]))
    out = {}
    for i, (_tid, e, gain) in els.items():
        lin = [base_obs[2][i] + gain * (g[e] - (-1.0 if e == 2 else 0.0)) for g, _ in seen]
        if max(abs(v[i] - L) for (_, v), L in zip(seen, lin)) < 1e-5:
            continue
        feats = {
            0: [math.atan2(-g[1], -g[2]) for g, _ in seen],
            1: [math.asin(max(-1.0, min(1.0, g[0]))) for g, _ in seen],
        }
        hit = None
        for k, f in feats.items():
            vals = [v[i] for _, v in seen]
            sc = vals[0] / f[0] if abs(f[0]) > 1e-9 else 0.0
            if abs(sc) > 1e-6 and all(abs(v - sc * fv) < 1e-5 for v, fv in zip(vals, f)):
                hit = (k, _round(sc))
                break
        if hit is None:
            raise ProbeError(f"{a.name}: obs {i} follows gravity, but not as gaitkeeper builds it")
        out[i] = ("gravity_euler", hit[0], hit[1])
    if out:
        findings.append(
            f"obs {sorted(out)} are roll and pitch computed from gravity (atan2, asin), not "
            "gravity itself"
        )
    return out


def _norm_gate(
    a: Adapter, desc: dict[int, tuple[str, int, float]], findings: list[str]
) -> dict[str, Any] | None:
    """A port that zeroes its command below some size, with no flag for it (handoff): the
    threshold, and whether it is the full or the planar norm. None when small commands
    pass."""
    el = {d_[1]: (i, d_[2]) for i, d_ in desc.items() if d_[0] == "velocity_commands"}
    if 0 not in el:
        return None
    i0, g0 = el[0]

    def passes(cmd) -> bool:
        o, _ = _run(a, WARM + 1, cmd=cmd)
        return abs(o[WARM][i0] - g0 * cmd[0]) < 1e-6 and abs(o[WARM][i0]) > 1e-9

    if passes((1e-3, 0.0, 0.0)):
        return None
    lo, hi = 0.0, 1.0
    if not passes((hi, 0.0, 0.0)):
        return None
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if not passes((mid, 0.0, 0.0)) else (lo, mid)
    thr = _round(hi, 4)
    # with vx just under the threshold, a yaw rate that lifts the full norm over it
    norm = "norm3" if passes((thr * 0.9, 0.0, thr * 0.6)) else "planar"
    findings.append(
        f"the port zeroes its command while its {'planar ' if norm == 'planar' else ''}norm "
        f"is below {thr:g}"
    )
    return {"on": "command_norm", "threshold": thr, "norm": norm}


def _gate(
    a: Adapter, cand: list[int], H: int, history: dict[str, Any], findings: list[str]
) -> tuple[dict[str, Any], dict[int, tuple[str, int, float]], set[int]] | None:
    """A flag the port feeds while it passes the command (1 walking, 0 standing): which
    elements, when it opens (any nonzero command), and how long it stays shut after a
    start. None when no unexplained element follows whether the command is zero."""
    r1, _ = _run(a, WARM + 1)
    r0, _ = _run(a, WARM + 1, cmd=(0.0, 0.0, 0.0))
    flags = [
        i
        for i in cand
        if abs(r1[WARM][i] - r0[WARM][i]) > TOL
        and np.ptp(r1[WARM - 3 :, i]) < TOL
        and np.ptp(r0[WARM - 3 :, i]) < TOL
    ]
    if not flags:
        return None
    F = a.obs_dim // H
    old: set[int] = set()
    if H > 1:
        if history["layout"] != "time_major":
            raise ProbeError(f"{a.name}: a command flag {flags} in a term-major history")
        lo = (H - 1) * F if history["order"] == "oldest_first" else 0
        old = {i for i in flags if not lo <= i < lo + F}
        flags = [i for i in flags if lo <= i < lo + F]
        if not flags:
            return None
    gdesc = {}
    for i in flags:
        on, off = float(r1[WARM][i]), float(r0[WARM][i])
        if abs(off) > TOL:
            raise ProbeError(
                f"{a.name}: obs {i} reads {off:g} at a zero command and {on:g} when moving: a "
                "flag gaitkeeper's command_gate does not build (it is 0 when shut)"
            )
        gdesc[i] = ("command_gate", 0, _round(on))
    i0 = flags[0]
    on_at = np.flatnonzero(np.abs(r1[:, i0] - r1[WARM, i0]) < TOL)
    warm = int(on_at[0]) if len(on_at) else 0
    for c in ((1e-3, 0.0, 0.0), (0.0, 1e-3, 0.0), (0.0, 0.0, 1e-3)):
        r, _ = _run(a, WARM + 1, cmd=c)
        if abs(r[WARM][i0] - r1[WARM][i0]) > TOL:
            raise ProbeError(
                f"{a.name}: the port's walk flag needs a command above some size: a gate "
                "gaitkeeper does not build"
            )
    spec: dict[str, Any] = {"on": "command_nonzero"}
    if warm:
        spec["warmup_s"] = _round(warm * POLICY_DT, 6)
    findings.append(
        "the port passes its command only while it is nonzero"
        + (f" and {warm * POLICY_DT:g} s after a start" if warm else "")
        + f", and feeds that as a flag (obs {flags})"
    )
    return spec, gdesc, old


def _gated_clock(
    a: Adapter, idx: list[int], gate: dict[str, Any], findings: list[str]
) -> tuple[dict[str, Any], dict[int, tuple[str, int, float]]]:
    """A [sin, cos] clock that runs only while the command gate is open (falcon): its
    period, and whether it advances before its first reading."""
    warm = int(round(float(gate.get("warmup_s", 0.0)) / POLICY_DT))
    steps = 240
    o, _ = _run(a, warm + steps)
    if np.ptp(o[: max(warm, 1), idx], axis=0).max() > TOL:
        raise ProbeError(f"{a.name}: clock elements {idx} run before the gate opens")
    if len(idx) != 2:
        raise ProbeError(f"{a.name}: a gated clock of {len(idx)} elements")
    err, w, coef = _fit_rate(o[warm:, idx], np.arange(steps, dtype=float))
    if err > 1e-6 * steps:
        raise ProbeError(f"{a.name}: clock elements {idx} are not sinusoids of one rate")
    period = _round(2 * math.pi * POLICY_DT / w, 4)
    w = 2 * math.pi * POLICY_DT / period
    amp = np.hypot(coef[0], coef[1])
    phi = np.arctan2(coef[1], coef[0])
    first = None
    for adv, lead in ((True, w), (False, 0.0)):
        roles = []
        for p_ in phi:
            r = math.remainder(p_ - lead, 2 * math.pi)
            roles.append(0 if abs(r) < 2e-3 else 1 if abs(r - math.pi / 2) < 2e-3 else -1)
        if sorted(roles) == [0, 1]:
            first = adv
            break
    if first is None:
        raise ProbeError(
            f"{a.name}: gated clock phases {np.round(phi, 3).tolist()} fit no [sin, cos]"
        )
    # it holds while the gate is shut
    a.reset()
    x = _inputs()
    rows = []
    for t in range(warm + 30):
        c = BASE_CMD if t < warm + 20 else (0.0, 0.0, 0.0)
        _, seen = a.step(x["q"], x["dq"], x["gyro"], x["lin_vel"], x["gravity"], c, arm_pose=STANCE)
        rows.append(seen[0][: a.obs_dim][idx])
    if np.ptp(np.array(rows[warm + 21 :]), axis=0).max() > TOL:
        raise ProbeError(f"{a.name}: the clock runs while the gate is shut")
    findings.append(
        f"a gait clock of period {period:g} s that runs only while the gate is open"
        + (", advancing before it is read" if first else "")
    )
    clock = {"id": "gait_phase_gated", "period": period, "advance_first": first}
    # an amplitude a fit leaves a hair off a round value is that value
    amp = [round(float(A), 3) if abs(A - round(A, 3)) < 5e-5 else _round(float(A)) for A in amp]
    desc = {i: ("gait_phase_gated", r, A) for i, r, A in zip(idx, roles, amp)}
    return clock, desc


def _switch(
    a: Adapter, p2m: list[int], p2m_from: list[str], findings: list[str]
) -> dict[str, Any] | None:
    """A port with several graphs of one shape that picks one each step by the command (a
    walking and a standing policy): which one drives at each command, and the threshold."""
    E = len(a.engines)
    if E < 2:
        return None
    io = {(tuple(e["inputs"][:1]), tuple(e["outputs"][:1])) for e in a.engines}
    k0 = next((k for k in range(a.act_dim) if p2m_from[k] == "action"), None)
    if len(io) != 1 or k0 is None:
        findings.append(f"the port runs {E} graphs of different shapes: only the first is read")
        return None

    def used(cmd) -> list[int]:
        out = []
        for k in range(E):
            for probe_k in (None, k):
                a.reset()
                x = _inputs(cmd)
                for t in range(3):
                    ea = {j: np.zeros(a.act_dim) for j in range(E)}
                    if t == 2 and probe_k is not None:
                        ea[probe_k][k0] = 0.5
                    tgt, seen = a.step(
                        x["q"],
                        x["dq"],
                        x["gyro"],
                        x["lin_vel"],
                        x["gravity"],
                        x["cmd"],
                        arm_pose=STANCE,
                        engine_actions=ea,
                    )
                if probe_k is None:
                    ref = tgt[p2m[k0]]
                elif abs(tgt[p2m[k0]] - ref) > 1e-6:
                    out.append(k)
            if len(seen) > 1 and any(not np.allclose(sv, seen[0]) for sv in seen[1:]):
                findings.append(
                    "the port's graphs see different observations: only the first is read"
                )
                return []
        return out

    hi, lo = used(BASE_CMD), used((0.0, 0.0, 0.0))
    if len(hi) != 1 or len(lo) != 1 or hi == lo:
        if not (len(hi) == 1 and hi == lo):
            findings.append(
                f"the port's graphs drive at commands gaitkeeper cannot read ({hi}, {lo})"
            )
        return None
    kw, kb = hi[0], lo[0]
    l_, h_ = 0.0, float(np.linalg.norm(BASE_CMD))
    for _ in range(30):
        mid = (l_ + h_) / 2
        l_, h_ = (mid, h_) if used((mid, 0.0, 0.0)) == [kb] else (l_, mid)
    thr = _round(h_, 4)
    by = "command_norm" if used((0.0, 0.0, thr * 1.05)) == [kw] else "command_planar_norm"
    if used((0.0, thr * 1.05, 0.0)) != [kw]:
        findings.append("the port picks its graph by vx alone: read as the command's norm")
    name = lambda k: Path(a.engines[k]["path"]).name  # noqa: E731
    findings.append(
        f"the port runs {name(kw)} while the command's {'planar ' if by != 'command_norm' else ''}"
        f"norm exceeds {thr:g}, else {name(kb)}"
    )
    return {"by": by, "threshold": thr, "above": name(kw), "below": name(kb)}


def _task_command(
    a: Adapter, desc: dict[int, tuple[str, int, float]], H: int, findings: list[str]
) -> tuple[dict[int, tuple[str, int, float]], set[int], dict[str, Any]] | None:
    """The command elements of a port that steers by the task: they follow the task, and the
    command only for its direction (commands.speed_to_distance). Returns their descriptions,
    the older history copies, and the shaping with its parameters measured; None when no
    element follows the task."""
    # a near waypoint: speed below any cap, a facing weight partway (both answer the task)
    c0, t0 = (0.3, 0.2, 0.25), (0.4, 0.3, 0.4 * math.cos(0.4), 0.4 * math.sin(0.4))
    steps = WARM + H + 1

    def task_run(k: int | None) -> np.ndarray:
        t_ = np.array(t0)
        a.reset()
        rows = []
        x = _inputs(c0)
        for s_ in range(steps):
            tk = t_.copy()
            if s_ == WARM and k is not None:
                tk[k] += DELTA
            _, seen_ = a.step(
                x["q"],
                x["dq"],
                x["gyro"],
                x["lin_vel"],
                x["gravity"],
                x["cmd"],
                arm_pose=STANCE,
                task=np.r_[tk, np.zeros(60)],
            )
            rows.append(seen_[0][: a.obs_dim])
        return np.array(rows)

    base = task_run(None)
    newest_, old = set(), set()
    for k in range(4):
        d = np.abs(task_run(k)[WARM:] - base[WARM:]) > TOL
        newest_ |= set(np.flatnonzero(d[0]).tolist())
        for lag in range(1, len(d)):
            old |= set(np.flatnonzero(d[lag]).tolist()) - newest_
    els = sorted(i for i in newest_ if i not in desc or desc[i][0] == "velocity_commands")
    if not els:
        return None
    # command elements the task leaves alone (a yaw rate passed through) are read already
    els = sorted(set(els) | {i for i, d_ in desc.items() if d_[0] == "velocity_commands"})

    def seen(cmd, task) -> np.ndarray:
        o, _ = _run(a, WARM + 1, cmd=cmd, task=np.r_[task, np.zeros(60)])
        return o[WARM][els]

    near0 = (0.2, 0.0, 0.2, 0.0)
    A, B = seen((0.3, 0.0, 0.0), near0), seen((0.0, 0.3, 0.0), near0)
    C = seen((0.0, 0.0, 0.3), (0.0, 0.1, 0.0, 0.0))
    role = {}
    for j in range(len(els)):
        if abs(A[j]) > TOL and abs(B[j]) < 1e-5:
            role[0] = j
        elif abs(B[j]) > TOL and abs(A[j]) < 1e-5:
            role[1] = j
        elif abs(C[j]) > TOL:
            role[2] = j
    if sorted(role) != [0, 1, 2] or len(els) != 3:
        raise ProbeError(
            f"{a.name}: obs {els} follow the harness's task, but not as a velocity command "
            "gaitkeeper can shape"
        )
    jx, jy, jz = role[0], role[1], role[2]
    if abs(seen((0.6, 0.0, 0.0), near0)[jx] - A[jx]) > 1e-5:
        raise ProbeError(
            f"{a.name}: the port's command follows the task and the command's size: no "
            "shaping gaitkeeper knows"
        )
    lim = a.limits
    cap_ref = lim["speed_norm"] if lim.get("speed_norm", 0) > 0 else 1.0
    b = 0.3
    far = (50.0, 0.0, 50.0, 0.0)
    diag = seen((0.3 * math.cos(b), 0.3 * math.sin(b), 0.0), far)
    sx = diag[jx] / math.cos(b) / cap_ref
    sy = diag[jy] / math.sin(b) / cap_ref
    pos_p = seen((0.3, 0.0, 0.0), (0.1, 0.0, 0.1, 0.0))[jx] / (sx * 0.1)
    cap = cap_ref
    vx_hi = seen((0.3, 0.0, 0.0), far)[jx] / sx
    vx_lo = seen((-0.3, 0.0, 0.0), far)[jx] / sx
    vy_abs = seen((0.0, 0.3, 0.0), far)[jy] / sy
    vx_hi = vx_hi if vx_hi < cap - 1e-5 else max(cap, lim["vx_max"])
    vx_lo = vx_lo if vx_lo > -cap + 1e-5 else min(-cap, lim["vx_min"])
    vy_abs = vy_abs if vy_abs < cap - 1e-5 else max(cap, lim["vy_abs"])
    params: dict[str, Any] = {
        "pos_p": _round(pos_p),
        "speed_cap": _round(cap),
        "vx": [_round(vx_lo), _round(vx_hi)],
        "vy_abs": _round(vy_abs),
    }
    # the yaw rate: passed through, or the port's own (toward the yaw target, facing the
    # direction of travel when far)
    p1 = seen((0.3, 0.0, 0.2), (0.2, 0.1, 0.2, 0.0))[jz]
    p2 = seen((0.3, 0.0, 0.2), (0.2, 0.5, 0.2, 0.0))[jz]
    p3 = seen((0.3, 0.0, 0.4), (0.2, 0.1, 0.2, 0.0))[jz]
    if abs(p1 - p2) < 1e-6 and abs(p3 - 2 * p1) < 1e-5:
        sz = p1 / 0.2
        params["yaw"] = "pass"
    else:
        wz_ref = lim["yaw_rate_abs"] if lim.get("yaw_rate_abs", 0) > 0 else 1.0
        sz = seen((0.0, 0.0, 0.3), (0.0, 3.0, 0.0, 0.0))[jz] / wz_ref
        yaw_p = C[jz] / (sz * 0.1)
        bb = 0.2

        def face_w(d: float) -> float:
            v = seen((0.3 * math.cos(bb), 0.3 * math.sin(bb), 0.0), (d, 0.0, d, 0.0))[jz]
            return v / (sz * yaw_p * bb)

        lo, hi = 0.0, 10.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if face_w(mid) < 1e-6 else (lo, mid)
        near = hi
        lo, hi = near, 20.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if face_w(mid) < 1 - 1e-6 else (lo, mid)
        params["yaw"] = {
            "yaw_p": _round(yaw_p),
            "yaw_rate_abs": _round(wz_ref),
            "face_near_m": _round(near, 4),
            "face_far_m": _round(hi, 4),
        }
    cdesc = {
        els[jx]: ("velocity_commands", 0, _round(sx)),
        els[jy]: ("velocity_commands", 1, _round(sy)),
        els[jz]: ("velocity_commands", 2, _round(sz)),
    }
    findings.append(
        "the port makes its own command from the harness's task: the command's direction at "
        f"min({params['pos_p']:g} x distance, {params['speed_cap']:g}) m/s, yaw "
        + (
            "as commanded"
            if params["yaw"] == "pass"
            else "toward the target, facing travel when far"
        )
        + f" (speeds factored assuming the port caps at limits() speed_norm {cap_ref:g})"
    )
    return cdesc, old, {"kind": "speed_to_distance", "params": params}


class _Arr(list):
    name = ""


def _static_arrays(src: str, n: int) -> list[_Arr]:
    """Float arrays the source declares with n (or 29) literal entries."""
    import re

    out = []
    pat = r"(?:__device__\s+)?const\s+float\s+(\w+)\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}"
    for m in re.finditer(pat, src):
        try:
            vals = [float(x.strip().rstrip("fF")) for x in m.group(2).split(",") if x.strip()]
        except ValueError:
            continue
        if len(vals) in (n, NUM_MOTOR):
            a = _Arr(vals)
            a.name = m.group(1)
            out.append(a)
    return out


def _match_array(arrs: list[_Arr], measured: list[float], p2m: list[int], key: str):
    """The one array named like ``key`` that agrees with every measured value, in policy or
    motor order; returned in policy order."""
    hits = []
    for arr in arrs:
        if key not in arr.name.upper():
            continue
        for order in ("policy", "motor"):
            if order == "policy" and len(arr) != len(measured):
                continue
            if order == "motor" and len(arr) != NUM_MOTOR:
                continue
            vals = list(arr) if order == "policy" else [arr[m] for m in p2m]
            ok = all(
                abs(v - x) <= 1e-4 * max(1.0, abs(x))
                for v, x in zip(vals, measured)
                if math.isfinite(x)
            )
            if ok:
                out = _Arr(vals)
                out.name = arr.name
                hits.append(out)
    distinct = {tuple(h) for h in hits}
    return hits[0] if len(distinct) == 1 else None


def _match_scalar(src: str, measured: list[float], key: str, n: int):
    """A scalar constant named like ``key`` equal to every measured value."""
    import re

    vals = {x for x in measured if math.isfinite(x)}
    if len(vals) != 1:
        return None
    v = next(iter(vals))
    for m in re.finditer(r"constexpr\s+(?:float|double)\s+(\w+)\s*=\s*([-+\d.eE]+)[fF]?\s*;", src):
        if key in m.group(1).upper() and abs(float(m.group(2)) - v) <= 1e-6 * max(1.0, abs(v)):
            out = _Arr([float(m.group(2))] * n)
            out.name = m.group(1)
            return out
    return None


def _gains_len(a: Adapter) -> int:
    """How many entries the adapter's kp() array has: owned() unless its declaration says."""
    import re

    src = preprocessed(a.path)
    m = re.search(r"kp\(\)\s*const\s*override\s*\{\s*return\s+(\w+)", src)
    if not m:
        return a.owned
    sym = m.group(1)
    d = re.search(rf"\b{sym}\s*\[\s*(\w+)\s*\]", src) or re.search(
        rf"std::array\s*<\s*float\s*,\s*(\w+)\s*>\s*{sym}\b", src
    )
    if not d:
        return a.owned
    size = d.group(1)
    if size.isdigit():
        return int(size)
    c = re.search(rf"constexpr\s+int\s+{size}\s*=\s*(\d+)\s*;", src)
    if c:
        return int(c.group(1))
    return NUM_MOTOR if size in ("NUM_MOTOR", "POLICY_NUM_MOTOR") else a.owned


def _newest_copies(obs: np.ndarray, idx: list[int], H: int) -> list[int]:
    """Of varying elements in a history, the newest: those that are no lagged copy of
    another (an element j steps old holds what another held j steps before)."""
    if H == 1 or not idx:
        return list(idx)
    older = set()
    x = obs[WARM - 4 :]
    for i in idx:
        for i2 in idx:
            if i2 == i:
                continue
            for j in range(1, H):
                if np.allclose(x[j:, i], x[:-j, i2], atol=1e-6):
                    older.add(i)
    return [i for i in idx if i not in older]


def _fit_rate(x: np.ndarray, t: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least-squares fit of sin(w t + phi) to each column of x: (residual, w, coef)."""
    ref = int(np.argmax(x.var(axis=0)))
    spec = np.abs(np.fft.rfft(x[:, ref] - x[:, ref].mean()))
    k = int(np.argmax(spec[1:]) + 1)
    w0 = 2 * math.pi * k / len(t)
    best = None
    for w in np.linspace(w0 * 0.8, w0 * 1.2, 4001):
        B = np.stack([np.sin(w * t), np.cos(w * t)], axis=1)
        coef, *_ = np.linalg.lstsq(B, x, rcond=None)
        err = float(np.sum((B @ coef - x) ** 2))
        if best is None or err < best[0]:
            best = (err, float(w), coef)
    return best  # type: ignore[return-value]


def _speed_clock(
    a: Adapter, idx: list[int], findings: list[str]
) -> tuple[dict[str, Any], dict[int, tuple[str, int, float]]] | None:
    """A [sin, cos] clock whose rate follows the command speed, gated below a speed (zealot).
    Returns None when the rate does not depend on the speed."""
    steps = 240
    t = np.arange(steps, dtype=float)
    fits = []
    for vx in (0.4, 0.9):
        o, _ = _run(a, steps, cmd=(vx, 0.0, 0.0))
        fits.append((o, _fit_rate(o[2:, idx], t[2:])))
    (o, (err, w, coef)), (_, (err2, w2, _)) = fits
    if len(idx) != 2 or err > 1e-6 * steps or err2 > 1e-6 * steps:
        return None
    if abs(w2 - w) <= 1e-4 * w:
        return None
    amp = np.hypot(coef[0], coef[1])
    phi = np.arctan2(coef[1], coef[0])
    x = o[:, idx]
    advance_first = bool(np.abs(x[1] - x[0]).max() > TOL)
    lead = 0.0 if advance_first else w  # phase (t - 1) * delta when the first step holds
    roles = []
    for p_ in phi:
        r = math.remainder(p_ + lead, 2 * math.pi)
        if abs(r) < 2e-3:
            roles.append(0)
        elif abs(r - math.pi / 2) < 2e-3:
            roles.append(1)
        else:
            raise ProbeError(
                f"{a.name}: speed clock phases {np.round(phi, 3).tolist()} fit no [sin, cos]"
            )
    if sorted(roles) != [0, 1]:
        raise ProbeError(f"{a.name}: speed clock elements are not one [sin, cos] pair")
    i_s, i_c = (idx[roles.index(0)], idx[roles.index(1)])
    a_s, a_c = (float(amp[roles.index(0)]), float(amp[roles.index(1)]))

    def rate(cmd: tuple[float, float, float], span: int = 20) -> float:
        """Phase advanced per step, averaged over ``span`` steps (unwrapped)."""
        o_, _ = _run(a, 3 + span, cmd=cmd)
        ang = np.unwrap(
            [math.atan2(o_[k, i_s] / a_s, o_[k, i_c] / a_c) for k in range(2, 3 + span)]
        )
        return float((ang[-1] - ang[0]) / (2 * math.pi) / span)

    # which speed: the full command norm, the planar norm or vx alone
    r_x, r_w = rate((0.5, 0.0, 0.0)), rate((0.0, 0.0, 0.5))
    r_xy = rate((0.3, 0.4, 0.0))
    if abs(r_w - r_x) < 1e-7 and abs(r_xy - r_x) < 1e-7:
        how = "norm3"
    elif r_w < 1e-9 and abs(r_xy - r_x) < 1e-7:
        how = "planar"
    elif r_w < 1e-9:
        how = "vx"
    else:
        raise ProbeError(f"{a.name}: the clock's rate follows no command speed gaitkeeper knows")
    # the speed below which it holds
    lo, hi = 0.0, 0.5
    if rate((lo, 0.0, 0.0)) > 1e-9:
        stand = 0.0
    else:
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if rate((mid, 0.0, 0.0)) < 1e-9 else (lo, mid)
        stand = _round(hi, 4)
    # the period against speed, as knots of a piecewise-linear curve
    speeds = np.round(np.arange(stand, 2.0 + 1e-9, 0.01), 6)
    speeds[0] = stand
    period = np.array([POLICY_DT / rate((float(s), 0.0, 0.0)) for s in speeds])
    knots = _knots(speeds, period)
    for s_, p_ in ((0.37, None), (1.13, None)):
        got = POLICY_DT / rate((s_, 0.0, 0.0))
        want = float(np.interp(s_, [k_[0] for k_ in knots], [k_[1] for k_ in knots]))
        if abs(got - want) > 1e-3 * want:
            raise ProbeError(f"{a.name}: the clock's period against speed is not piecewise linear")
        del p_
    clock = {
        "id": "gait_phase_speed",
        "period_knots": knots,
        "stand_speed": stand,
        "speed": how,
        "advance_first": advance_first,
    }
    findings.append(
        f"a gait clock whose period follows the command speed ({how}): "
        + ", ".join(f"{p:g} s at {s:g}" for s, p in knots)
        + f"; it holds below {stand:g}"
    )
    desc = {i_s: ("gait_phase_speed", 0, _round(a_s)), i_c: ("gait_phase_speed", 1, _round(a_c))}
    return clock, desc


def _knots(x: np.ndarray, y: np.ndarray, tol: float = 2e-4) -> list[list[float]]:
    """Corners of a piecewise-linear curve sampled at x: lines fitted to the straight runs,
    intersected, so a corner between samples is placed where the lines meet."""
    keep = [0, len(x) - 1]

    def rdp(i0: int, i1: int) -> None:
        if i1 - i0 < 2:
            return
        line = y[i0] + (y[i1] - y[i0]) * (x[i0 + 1 : i1] - x[i0]) / (x[i1] - x[i0])
        d = np.abs(y[i0 + 1 : i1] - line)
        j = int(np.argmax(d))
        if d[j] > tol:
            keep.append(i0 + 1 + j)
            rdp(i0, i0 + 1 + j)
            rdp(i0 + 1 + j, i1)

    rdp(0, len(x) - 1)
    ks = sorted(set(keep))
    # fit each straight run on its interior (a corner falls between samples, so the samples
    # next to it may sit on either line), skip runs too short to be one, then meet neighbours
    lines = []
    for a_, b_ in zip(ks, ks[1:]):
        lo, hi = (a_ + 1, b_ - 1) if b_ - a_ > 3 else (a_, b_)
        if hi - lo < 2 and len(ks) > 2:
            continue
        lines.append(np.polyfit(x[lo : hi + 1], y[lo : hi + 1], 1))
    pts = [[float(x[0]), float(np.polyval(lines[0], x[0]))]]
    for l1, l2 in zip(lines, lines[1:]):
        if abs(l1[0] - l2[0]) < 1e-9:
            continue
        xc = (l2[1] - l1[1]) / (l1[0] - l2[0])
        pts.append([float(xc), float(np.polyval(l1, xc))])
    pts.append([float(x[-1]), float(np.polyval(lines[-1], x[-1]))])
    return [[round(p[0], 4), _round(p[1], 4)] for p in pts]


def _feet_phase(x: np.ndarray, delta: float, ratio: float) -> tuple[float, float]:
    """The phase at the first sample of sin(2 pi warp(phase + t delta)) fitted to x: (phase,
    largest residual)."""
    from ..terms import warp_stance

    t = np.arange(len(x), dtype=float)

    def errs(grid: np.ndarray) -> np.ndarray:
        ph = np.mod(grid[:, None] + t[None, :] * delta, 1.0)
        return np.abs(np.sin(2 * np.pi * warp_stance(ph, ratio)) - x[None, :]).max(axis=1)

    g = np.arange(0.0, 1.0, 1e-3)
    a0 = float(g[int(np.argmin(errs(g)))])
    g = a0 + np.linspace(-2e-3, 2e-3, 4001)
    e = errs(g)
    j = int(np.argmin(e))
    return float(np.mod(g[j], 1.0)), float(e[j])


def _feet_clock(
    a: Adapter, idx: list[int], findings: list[str]
) -> tuple[dict[str, Any], dict[int, tuple[str, int, float]]] | None:
    """Clock inputs per foot, sin(2 pi warp(phase)) with the stance part of the cycle warped
    to its first half (walk-these-ways; openwbt): the gait frequency, the stance ratio, each
    foot's offset, the index at the start, and what it does while the command is zero.
    None when the elements are not such a clock."""
    T = 400
    o, _ = _run(a, T)
    x = o[:, idx]
    deltas, ratios = [], []
    for k in range(len(idx)):
        xk = x[:, k]
        up = [t + xk[t] / (xk[t] - xk[t + 1]) for t in range(T - 1) if xk[t] < 0 <= xk[t + 1]]
        if len(up) < 3:
            return None
        deltas.append((len(up) - 1) / (up[-1] - up[0]))
        ratios.append(float(np.mean(xk > 0)))
    f0, r0 = float(np.mean(deltas)) / POLICY_DT, float(np.mean(ratios))
    best = None
    for f_ in np.round(f0, 2) + np.arange(-0.03, 0.031, 0.01):
        for r_ in np.round(r0, 2) + np.arange(-0.03, 0.031, 0.01):
            if not 0.05 < r_ < 0.95:
                continue
            fits = [_feet_phase(x[:, k], f_ * POLICY_DT, r_) for k in range(len(idx))]
            e = max(f[1] for f in fits)
            if best is None or e < best[0]:
                best = (e, round(float(f_), 4), round(float(r_), 4), [f[0] for f in fits])
    if best is None or best[0] > 1e-4:
        return None
    _, freq, ratio, a_ph = best
    delta = freq * POLICY_DT
    clock: dict[str, Any] = {"id": "gait_phase_feet", "frequency": freq, "stance_ratio": ratio}
    z, _ = _run(a, 4, cmd=(0.0, 0.0, 0.0))
    holds = np.ptp(z[1:, idx], axis=0).max() < TOL
    if holds:
        zs, _ = _run(a, 4, cmd=(1e-4, 0.0, 0.0))
        if np.ptp(zs[1:, idx], axis=0).max() < TOL:
            raise ProbeError(f"{a.name}: a foot clock that holds below some command size")

        def inverse(v: float) -> list[float]:
            w1 = (math.asin(max(-1.0, min(1.0, v))) / (2 * math.pi)) % 1.0
            out = []
            for w_ in (w1, (0.5 - w1) % 1.0):
                out.append(2 * ratio * w_ if w_ < 0.5 else ratio + (w_ - 0.5) * 2 * (1 - ratio))
            return out

        cands = [inverse(float(v)) for v in z[2, idx]]
        common = [h for h in cands[0] if all(min(abs(h - c) for c in cs) < 1e-4 for cs in cands)]
        if not common:
            raise ProbeError(f"{a.name}: the feet hold different phases while standing")
        # stand for 5 steps, then walk: the phases it restarts at give each foot's offset
        a.reset()
        rows = []
        x_ = _inputs()
        for t in range(5 + T):
            c = (0.0, 0.0, 0.0) if t < 5 else BASE_CMD
            _, seen = a.step(
                x_["q"], x_["dq"], x_["gyro"], x_["lin_vel"], x_["gravity"], c, arm_pose=STANCE
            )
            rows.append(seen[0][: a.obs_dim][idx])
        rx = np.array(rows)[5:]
        b_ph = [_feet_phase(rx[:, k], delta, ratio)[0] for k in range(len(idx))]

        def offs(h: float) -> list[float]:
            return [round((b - h - delta) % 1.0, 4) % 1.0 for b in b_ph]

        # two holds read the same while standing (sin is even about its peak): either fits,
        # so take the one that puts a foot on the gait index itself (an offset of zero)
        hold = round(min(common, key=lambda h: min(min(o, 1 - o) for o in offs(h))), 4)
        offsets = offs(hold)
        clock["stand"] = {"hold": hold}
    else:
        offsets = [round((ak - a_ph[-1]) % 1.0, 4) % 1.0 for ak in a_ph]
    starts = [(ak - delta - off) % 1.0 for ak, off in zip(a_ph, offsets)]
    if max(starts) - min(starts) > 1e-3 and not (max(starts) > 0.999 and min(starts) < 0.001):
        raise ProbeError(f"{a.name}: the feet clocks do not share one gait index")
    clock["offsets"] = offsets
    clock["start"] = round(starts[-1], 4) % 1.0
    findings.append(
        f"clock inputs per foot (walk-these-ways): {freq:g} Hz, stance ratio {ratio:g}, "
        f"offsets {offsets}, from index {clock['start']:g}"
        + (f"; held at {clock['stand']['hold']:g} while the command is zero" if holds else "")
    )
    desc = {i: ("gait_phase_feet", k, 1.0) for k, i in enumerate(idx)}
    return clock, desc


def _legs_stand(
    a: Adapter, idx: list[int], el: list[int], amp: np.ndarray, findings: list[str]
) -> dict[str, Any]:
    """A two-leg clock that holds while the command says stand: the phases it holds at, the
    phases it restarts from, and the thresholds on |(vx, vy)| and |wz| below which it holds."""
    roles = {
        e: (i, float(A)) for i, e, A in zip(idx, el, amp)
    }  # 0 sin a, 1 sin b, 2 cos a, 3 cos b

    def phases(row: np.ndarray) -> list[float]:
        out = []
        for s_, c_ in ((0, 2), (1, 3)):
            i_s, a_s = roles[s_]
            i_c, a_c = roles[c_]
            out.append(_round(math.atan2(row[i_s] / a_s, row[i_c] / a_c) / (2 * math.pi) % 1.0, 6))
        return out

    def holding(cmd) -> bool:
        o_, _ = _run(a, 4, cmd=cmd)
        return bool(np.abs(o_[3, idx] - o_[2, idx]).max() < TOL)

    z, _ = _run(a, 3, cmd=(0.0, 0.0, 0.0))
    hold = phases(z[2])

    def restart(k: int) -> list[float]:
        """Stand k steps, then move: the phases at the first moving step."""
        a.reset()
        x = _inputs()
        for t in range(k + 1):
            c = (0.0, 0.0, 0.0) if t < k else BASE_CMD
            _, seen = a.step(
                x["q"], x["dq"], x["gyro"], x["lin_vel"], x["gravity"], c, arm_pose=STANCE
            )
        return phases(seen[0][: a.obs_dim])

    resume, later = restart(3), restart(7)
    if any(abs(math.remainder(b - r, 1.0)) > 1e-4 for r, b in zip(resume, later)):
        # the clock ran on while it read zero: masked, not held (handoff)
        if any(abs(math.remainder(h, 1.0)) > 1e-4 for h in hold):
            raise ProbeError(f"{a.name}: a two-leg clock masked at phases {hold} while standing")
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if holding((mid, 0.0, 0.0)) else (lo, mid)
        thr = _round(hi, 4)
        norm = "planar" if holding((thr * 0.9, 0.0, thr * 0.6)) else "norm3"
        findings.append(
            f"the two-leg clock reads phase zero while the command's "
            f"{'planar ' if norm == 'planar' else ''}norm is below "
            f"{thr:g}, running on underneath"
        )
        return {"mode": "zero_phase", "norm": norm, "threshold": thr}
    eps = []
    for axis in (0, 2):
        lo, hi = 0.0, 0.5
        for _ in range(30):
            mid = (lo + hi) / 2
            cmd = [0.0, 0.0, 0.0]
            cmd[axis] = mid
            lo, hi = (mid, hi) if holding(tuple(cmd)) else (lo, mid)
        eps.append(_round(hi, 3))
    findings.append(
        f"the two-leg clock holds at {hold} (cycles) while |(vx, vy)| < {eps[0]:g} and "
        f"|wz| < {eps[1]:g}, and restarts at {resume}"
    )
    return {"eps_planar": eps[0], "eps_yaw": eps[1], "hold": hold, "resume": resume}


def _clock(
    a: Adapter, idx: list[int], findings: list[str], gate: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[int, tuple[str, int, float]]]:
    """Fit sin(w t + phi_i) to each clock element and name it as an element of gait_phase
    ([sin, cos] of one phase) or gait_phase_legs ([sin a, sin b, cos a, cos b], b half a
    period on)."""
    if gate is not None and gate.get("on") == "command_nonzero":
        return _gated_clock(a, idx, gate, findings)
    sc = _speed_clock(a, idx, findings)
    if sc is not None:
        return sc
    steps = 240
    o, _ = _run(a, steps)
    x = o[:, idx]
    ref = int(np.argmax(x.var(axis=0)))
    spec = np.abs(np.fft.rfft(x[:, ref] - x[:, ref].mean()))
    k = int(np.argmax(spec[1:]) + 1)
    w0 = 2 * math.pi * k / steps
    best = None
    t = np.arange(steps)
    for w in np.linspace(w0 * 0.9, w0 * 1.1, 4001):
        B = np.stack([np.sin(w * t), np.cos(w * t)], axis=1)
        coef, res, *_ = np.linalg.lstsq(B, x, rcond=None)
        err = float(np.sum((B @ coef - x) ** 2))
        if best is None or err < best[0]:
            best = (err, w, coef)
    err, w, coef = best  # type: ignore[misc]
    if err > 1e-6 * steps * len(idx):
        fc = _feet_clock(a, idx, findings)
        if fc is not None:
            return fc
        raise ProbeError(f"{a.name}: clock elements {idx} are not sinusoids of one rate")
    amp = np.hypot(coef[0], coef[1])
    phi = np.arctan2(coef[1], coef[0])  # x = A sin(w t + phi)
    period = _round(2 * math.pi * POLICY_DT / w, 4)
    w = 2 * math.pi * POLICY_DT / period
    n_per = max(int(round(period / POLICY_DT)), 1)

    def assign(psis: list[float], off: int) -> list[int] | None:
        out = []
        for p_ in phi:
            hit = [
                j
                for j, ps in enumerate(psis)
                if abs(math.remainder(p_ - (w * off + ps), 2 * math.pi)) < 2e-3
            ]
            if not hit:
                return None
            out.append(hit[0])
        return out

    kinds = [
        ("gait_phase", [0.0, math.pi / 2]),
        ("gait_phase_legs", [0.0, math.pi, math.pi / 2, 3 * math.pi / 2]),
    ]
    for kind, psis in kinds:
        for off in range(n_per):
            el = assign(psis, off)
            if el is None:
                continue
            clock: dict[str, Any] = {"id": kind, "period": period, "clock_offset_steps": off}
            z, _ = _run(a, 3, cmd=(0.0, 0.0, 0.0))
            if np.abs(z[:, idx]).max() < TOL:
                lo, hi = 0.0, float(np.linalg.norm(BASE_CMD))
                for _ in range(24):
                    mid = (lo + hi) / 2
                    r, _ = _run(a, 2, cmd=(mid, 0.0, 0.0))
                    lo, hi = (mid, hi) if np.abs(r[:, idx]).max() < TOL else (lo, mid)
                clock["stand_threshold"] = _round(hi, 3)
                r, _ = _run(a, 2, cmd=(0.0, hi * 1.2, 0.0))
                if np.abs(r[:, idx]).max() < TOL:
                    findings.append(
                        "the clock gate ignores vy: not the command norm gaitkeeper uses"
                    )
                if kind == "gait_phase_legs":
                    findings.append(
                        "a gated two-leg clock: gaitkeeper's gait_phase_legs has no gate"
                    )
            elif kind == "gait_phase_legs" and np.abs(z[2, idx] - z[1, idx]).max() < TOL:
                clock["stand"] = _legs_stand(a, idx, el, amp, findings)
            desc = {i: (kind, int(e), _round(float(A))) for i, e, A in zip(idx, el, amp)}
            return clock, desc
    raise ProbeError(f"{a.name}: clock phases {np.round(phi, 3).tolist()} fit no known clock term")


# -- contracts --------------------------------------------------------------------------


def contracts(r: ProbeResult, mjcf: str | Path, source: str | Path) -> dict[str, Any]:
    """The trained and port contracts of a probe result (``{"trained": c, "port": c}``)."""
    import mujoco

    from ..contract import SCHEMA, Contract
    from ..tables import SDK_TABLES
    from ..tour import armature_gains

    table_name, table = SDK_TABLES["unitree_g1_29dof"]
    m = mujoco.MjModel.from_xml_path(str(mjcf))
    names = [table[i] for i in r.p2m]
    allm = [table[i] for i in range(NUM_MOTOR)]
    hold_kp, hold_kd = armature_gains(m, allm)
    stance = {allm[i]: float(STANCE[i]) for i in range(NUM_MOTOR)}

    def harness_gains(mi: int) -> tuple[float, float]:
        """The gains the harness applies to motor ``mi`` (main.cpp, after the crane)."""
        nm = allm[mi]
        if (mi < ARM_LEFT_FIRST or r.owned == NUM_MOTOR) and mi < r.gains_len:
            return float(r.kp[mi]), float(r.kd[mi])
        return hold_kp[nm], hold_kd[nm]

    detail = f"probed from {source} (gaitkeeper.readers.twb_probe)"
    out = {}
    for variant in ("trained", "port"):
        c = Contract({"schema": SCHEMA})
        c.set(
            "source",
            {
                "format": "twb_port",
                "framework": {"name": None, "version": None},
                "files": [{"path": str(source)}],
                "robot": "unitree_g1_29dof",
                "note": f"{r.name}: {variant} variant",
            },
            "file",
        )
        c.set("timing.policy_dt", POLICY_DT, "file", "main.cpp PERIOD_S")
        c.set("timing.sim_dt", None, "unknown", "the target model's step")
        c.set("timing.decimation", None, "unknown", "")
        c.set("timing.order", ["obs", "infer", "target", "pd", "step"], "default", "")
        c.set("timing.target_hold", "zoh", "default", "")
        c.set(
            "policy_io.joints",
            {"names": names, "table": table_name, "joint_ids_map": r.p2m},
            "file",
            detail + "; joint map from " + ", ".join(sorted(set(r.p2m_from))),
        )
        terms = []
        for t in r.terms:
            term = {
                "id": t["id"],
                "source_name": t["source_name"],
                "dim": t["dim"],
                "scale": t["scale"],
                "clip": None,
                "params": {},
            }
            if t["id"] == "velocity_commands":
                term["params"] = {"command_name": "base_velocity"}
            elif t["id"] == "last_action":
                term["params"] = {"reset": "zeros"}
                if r.action_lag > 1:
                    term["params"]["lag"] = r.action_lag
            elif t["id"] == "gait_phase_speed":
                ck = r.clock or {}
                term["params"] = {
                    k: ck[k] for k in ("period_knots", "stand_speed", "speed", "advance_first")
                }
            elif t["id"] == "gait_phase":
                ck = r.clock or {}
                term["params"] = {
                    "arithmetic": "float64",
                    "period": ck["period"],
                    "clock_offset_steps": ck.get("clock_offset_steps", 0),
                }
                if "stand_threshold" in ck:
                    term["params"]["stand_threshold"] = ck["stand_threshold"]
                if ck.get("gate"):
                    term["params"]["gate"] = ck["gate"]
            elif t["id"] == "gait_phase_feet":
                ck = r.clock or {}
                term["params"] = {
                    k: ck[k]
                    for k in ("frequency", "stance_ratio", "offsets", "start", "stand")
                    if k in ck
                }
            elif t["id"] == "gait_phase_gated":
                ck = r.clock or {}
                term["params"] = {"period": ck["period"], "advance_first": ck["advance_first"]}
            elif t["id"] == "gait_phase_legs":
                ck = r.clock or {}
                term["params"] = {"period": ck["period"]}
                if ck.get("clock_offset_steps"):
                    term["params"]["clock_offset_steps"] = ck["clock_offset_steps"]
                if ck.get("stand"):
                    term["params"]["stand"] = ck["stand"]
            elif t["id"] == "constant":
                term["params"] = {"value": t["value"]}
            if t.get("extra"):
                ex = [allm[mi] for mi in t["extra"]]
                term["params"]["joints"] = names + ex
                if t["id"] == "joint_pos_rel":
                    term["params"]["default"] = {
                        allm[mi]: float(d_) for mi, d_ in zip(r.observed_extra, r.extra_default)
                    }
                if t["id"] == "joint_target_rel":
                    term["params"]["default"] = {
                        allm[mi]: float(r.target_default.get(mi, 0.0)) for mi in r.observed_extra
                    }
            if "index" in t:
                term["params"]["index"] = t["index"]
            terms.append(term)
        hist = {
            k: r.history[k]
            for k in ("length", "layout", "order", "init", "first_frame", "chunks")
            if k in r.history
        }
        c.set(
            "policy_io.observation_groups",
            {"policy": {"terms": terms, "history": hist, "clip_then_scale": True}},
            "file",
            detail,
        )
        lim = r.limits
        c.set(
            "policy_io.commands",
            {
                "base_velocity": {
                    "limit": {
                        "vx": [lim["vx_min"], lim["vx_max"]],
                        "vy": [-lim["vy_abs"], lim["vy_abs"]],
                        "wz": [-lim["yaw_rate_abs"], lim["yaw_rate_abs"]],
                    },
                    "trained": None,
                    "heading": "off",
                }
                | ({"shaping": r.shaping} if r.shaping else {})
                | ({"gate": r.gate} if r.gate else {})
            },
            "file",
            detail
            + ": limits()"
            + ("; the port's own steering from the task" if r.shaping else ""),
        )
        c.set(
            "policy_io.graph",
            {
                "inputs": [{"name": "obs", "shape": [1, r.obs_dim]}],
                "outputs": [{"name": "actions", "shape": [1, r.act_dim]}],
                "recurrent": [],
            }
            | ({"switch": r.switch} if r.switch else {}),
            "file",
            detail + ("; the graph each command runs, measured" if r.switch else ""),
        )
        c.set(
            "control.default_joint_pos",
            dict(zip(names, map(float, r.default_pose))),
            "file",
            detail + ": joint_pos_rel at the stance",
        )
        sc = dict(zip(names, map(float, r.action_scale)))
        off = {
            nm: float(o if o is not None else d)
            for nm, o, d in zip(names, r.action_offset, r.default_pose)
        }
        clip = r.action_clip
        c.set(
            "control.actions.joint_pos",
            {
                "scale": sc,
                "offset": off,
                "clip": [
                    b_ if b_ is not None else [off[nm] - 1e3, off[nm] + 1e3]
                    for nm, b_ in zip(names, clip)
                ]
                if clip
                else None,
                "clip_stage": "processed" if clip else "none",
            },
            "file",
            detail + ": action map",
        )
        kp, kd = {}, {}
        for nm, mi in zip(names, r.p2m):
            known = mi < r.owned or (mi < r.gains_len and r.kp[mi] > 0)
            kp[nm] = r.kp[mi] if known else hold_kp[nm]
            kd[nm] = r.kd[mi] if known else hold_kd[nm]
            if variant == "port" and mi >= r.owned:
                kp[nm], kd[nm] = harness_gains(mi)
        c.set(
            "control.actuators",
            {
                "kp": kp,
                "kd": kd,
                "kind": "explicit_pd",
                "pd_period": "sim_step",
                "integrator": "implicitfast",
                "torque_limit_at": "actuator_force",
            },
            "file",
            detail + ": kp(), kd(); past owned() the harness's gains (kp() for legs and "
            "waist, armature gains for arms)",
        )
        ext = []
        if variant == "port":
            # The harness holds motors from owned() on at its stance (with kp() on legs and
            # waist, armature gains on arms); the adapter itself holds owned motors whose
            # action it discards, at its own target with the policy's gains.
            held = [k for k, mi in enumerate(r.p2m) if mi >= r.owned or k in r.discarded]
            groups: dict[tuple[str, bool], list[int]] = {}
            for k in held:
                groups.setdefault((r.joint_obs[k], r.p2m[k] >= r.owned), []).append(k)
            for (obs, by_harness), ks in groups.items():
                js = [names[k] for k in ks]
                ext.append(
                    {
                        "joints": js,
                        "drive": "hold",
                        "pose": {
                            j: stance[j] if by_harness else float(r.hold_target[r.p2m[k]])
                            for j, k in zip(js, ks)
                        },
                        "kp": {
                            j: harness_gains(r.p2m[k])[0] if by_harness else kp[j]
                            for j, k in zip(js, ks)
                        },
                        "kd": {
                            j: harness_gains(r.p2m[k])[1] if by_harness else kd[j]
                            for j, k in zip(js, ks)
                        },
                        "obs": obs,
                    }
                )
        held_names = {j for e in ext for j in e["joints"]}
        c.set(
            "control.ownership",
            {
                "owned": "all" if not ext else [nm for nm in names if nm not in held_names],
                "external": ext,
            },
            "file",
            detail + (": owned()" if variant == "port" else ": every policy joint, as trained"),
        )
        rest = [nm for nm in allm if nm not in names]
        if rest:
            # Motors below owned() that the policy does not list are the adapter's to hold
            # (at its own target, with kp()); the rest the harness holds.
            mi = {nm: allm.index(nm) for nm in rest}
            own = {nm for nm in rest if mi[nm] < r.owned and mi[nm] < r.gains_len}
            c.set(
                "control.unlisted",
                {
                    "pose": {
                        nm: float(r.hold_target[mi[nm]]) if nm in own else stance[nm] for nm in rest
                    },
                    "kp": {
                        nm: r.kp[mi[nm]] if nm in own else harness_gains(mi[nm])[0] for nm in rest
                    },
                    "kd": {
                        nm: r.kd[mi[nm]] if nm in own else harness_gains(mi[nm])[1] for nm in rest
                    },
                    "note": "motors the policy does not list: below owned() the adapter holds "
                    "them at its target with kp(); the harness holds the rest at its stance, "
                    "with kp() on legs and waist and armature gains on arms",
                },
                "file",
                detail + ": targets at zero action, kp(), owned()",
            )
        for k in ("effort_limit", "velocity_limit", "armature", "joint_friction", "joint_damping"):
            c.set(f"model.{k}", None, "unknown", "from the target model")
        c.set("evidence", {"level": "L0", "golden": None}, "default", "")
        out[variant] = c
    return out


class _FrameView:
    """An adapter seen through one frame of its observation (the elements that are no
    older copy of another): what ``probe`` reads when the layout around it is irregular."""

    def __init__(self, a: Adapter, frame: list[int]):
        self._a = a
        self._frame = np.array(frame, dtype=int)
        self.obs_dim = len(frame)
        self.engines = [
            dict(e, inputs=[len(frame)] + list(e["inputs"][1:]))
            if e["inputs"][:1] == [a.obs_dim]
            else e
            for e in a.engines
        ]

    def __getattr__(self, k: str) -> Any:
        return getattr(self._a, k)

    def step(self, *args: Any, **kw: Any) -> tuple[np.ndarray, list[np.ndarray]]:
        tgt, seen = self._a.step(*args, **kw)
        n = self._a.obs_dim
        return tgt, [sv[:n][self._frame] if len(sv) >= n else sv for sv in seen]


def _random_trace(a: Adapter, steps: int, seed: int = 1) -> np.ndarray:
    """The observation over ``steps`` steps of random inputs, a random action each step."""
    rng = np.random.default_rng(seed)
    a.reset()
    rows = []
    lim = a.limits
    vx = (lim["vx_min"], lim["vx_max"]) if lim["vx_max"] > lim["vx_min"] else (-0.5, 0.5)
    vy = lim["vy_abs"] or 0.3
    wz = lim["yaw_rate_abs"] or 0.5
    for t in range(steps):
        g = np.array([0.0, 0.0, -1.0]) + rng.normal(0, 0.15, 3)
        cmd = np.array([rng.uniform(*vx), rng.uniform(-vy, vy), rng.uniform(-wz, wz)])
        if t % 7 == 3:
            cmd[:] = 0.0  # standing now and then: clocks and gates that stop, stop
        d_ = float(rng.uniform(0.0, 2.0))
        b_ = float(rng.uniform(-math.pi, math.pi))
        task = [d_, rng.uniform(-1.0, 1.0), d_ * math.cos(b_), d_ * math.sin(b_)]
        if t % 9 in (4, 5):
            task = [0.0, 0.0, 0.0, 0.0]  # at the waypoint now and then: latches shut
        arm = STANCE.copy()
        arm[ARM_LEFT_FIRST:] += rng.normal(0, 0.1, NUM_MOTOR - ARM_LEFT_FIRST)
        _, seen = a.step(
            STANCE + rng.normal(0, 0.1, NUM_MOTOR),
            rng.normal(0, 1.0, NUM_MOTOR),
            rng.normal(0, 0.5, 3),
            rng.normal(0, 0.5, 3),
            g / np.linalg.norm(g),
            cmd,
            action=rng.normal(0, 0.5, a.act_dim),
            arm_pose=arm,
            task=np.r_[task, np.zeros(60)],
        )
        rows.append(seen[0][: a.obs_dim])
    return np.array(rows)


def _layout_map(a: Adapter, max_lag: int = LAGS) -> dict[str, Any] | None:
    """Which elements of the observation are older copies of others: every element that is
    no copy is the frame; every other is (frame element, lag). Read from random inputs, where
    a copy repeats its source exactly, lagged. None when nothing is a copy."""
    T = WARM + max_lag + 40
    obs = _random_trace(a, T)
    D = obs.shape[1]
    varying = np.ptp(obs[T - 40 :], axis=0) > 1e-6
    matches: dict[int, list[tuple[int, int]]] = {i: [] for i in range(D)}
    # A copy holds its source's value ``lag`` steps back over the whole trace, and before the
    # trace has that many steps the first value or zero: a clock that only repeats itself a
    # period on fails at the start.
    for lag in range(0, max_lag + 1):
        A, B = obs[lag:], obs[: T - lag]
        order = np.argsort(B[-1])
        b_last = B[-1][order]
        for i in np.flatnonzero(varying):
            lo = np.searchsorted(b_last, A[-1, i] - 1e-6)
            hi = np.searchsorted(b_last, A[-1, i] + 1e-6)
            for j in order[lo:hi]:
                if (lag == 0 and j >= i) or not varying[j]:
                    continue
                if np.abs(A[:, i] - B[:, j]).max() >= 1e-6:
                    continue
                head = obs[:lag, i]
                if lag and not (np.abs(head - obs[0, j]).max() < 1e-6 or np.abs(head).max() < 1e-9):
                    continue
                matches[int(i)].append((int(j), lag))
    copies = {i for i, m in matches.items() if m}
    if not copies:
        return None

    # each copy's source in the frame, at the smallest total lag (relaxed to a fixed point).
    # A periodic signal (a clock) also matches itself half a period on, so a ring of copies
    # can have no source: its first element is then the frame's.
    src: dict[int, tuple[int, int]] = {i: (i, 0) for i in range(D) if i not in copies}
    while True:
        changed = True
        while changed:
            changed = False
            for i in copies:
                if src.get(i, (0, 1))[1] == 0:
                    continue
                for j, lag in matches[i]:
                    if j in src and (i not in src or src[j][1] + lag < src[i][1]):
                        src[i] = (src[j][0], src[j][1] + lag)
                        changed = True
        missing = sorted(copies - set(src))
        if not missing:
            break
        src[missing[0]] = (missing[0], 0)
    frame = sorted(i for i, (j, lag) in src.items() if j == i and lag == 0)
    fpos = {i: k for k, i in enumerate(frame)}
    elem = [(fpos[src[i][0]], src[i][1]) for i in range(D)]
    # what a lagged element holds before the episode has that many steps
    init = "repeat_first"
    for i, (k, lag) in enumerate(elem):
        if lag > 0:
            if abs(obs[0, i]) < 1e-9 and abs(obs[0, frame[k]]) > 1e-6:
                init = "zeros"
            break
    return {"frame": frame, "elements": elem, "init": init}


def probe_any(a: Adapter) -> ProbeResult:
    """``probe``, and when its layout reading fails, a reading of one frame of the
    observation with the layout around it stated element by element (history.chunks)."""
    try:
        return probe(a)
    except ProbeError as e:
        first = e
    try:
        return probe_frames(a)
    except ProbeError as e:
        if "nothing in the observation is a copy" in str(e):
            raise first from None
        raise ProbeError(
            f"{e} (read as one frame of an irregular layout; as laid out: {first})"
        ) from None


def probe_frames(a: Adapter) -> ProbeResult:
    """Read one frame of the observation (the elements that are no older copy of another),
    and state the layout around it element by element: [term, lag] chunks."""
    lay = _layout_map(a)
    if lay is None:
        raise ProbeError(f"{a.name}: nothing in the observation is a copy of another element")
    r = probe(_FrameView(a, lay["frame"]))  # type: ignore[arg-type]
    if r.history.get("length", 1) != 1:
        raise ProbeError(f"{a.name}: one frame of the observation still holds a history")
    # each frame position's term and its place in the term; constants are regrouped below
    where: dict[int, tuple[str, int]] = {}
    const_at: dict[int, float] = {}
    for t in r.terms:
        for k in range(t["dim"]):
            if t["id"] == "constant":
                const_at[t["start"] + k] = float(t["value"][k])
            else:
                where[t["start"] + k] = (t["source_name"], k)
    terms = [t for t in r.terms if t["id"] != "constant"]
    dims = {t["source_name"]: t["dim"] for t in terms}
    consts: dict[tuple, str] = {}
    chunks: list[list[Any]] = []
    i = 0
    D = len(lay["elements"])
    while i < D:
        p, lag = lay["elements"][i]
        if p in const_at:
            # a run of constants in the full observation is one constant term (one per value)
            run = []
            while i < D and lay["elements"][i][0] in const_at:
                run.append(const_at[lay["elements"][i][0]])
                i += 1
            key = consts.get(tuple(run))
            if key is None:
                key = f"constant_{len(consts) + 1}" if consts else "constant"
                consts[tuple(run)] = key
                terms.append(
                    {
                        "id": "constant",
                        "source_name": key,
                        "dim": len(run),
                        "scale": [1.0] * len(run),
                        "value": run,
                        "start": -1,
                    }
                )
                dims[key] = len(run)
            chunks.append([key, 0])
            continue
        key, k = where[p]
        if k != 0:
            raise ProbeError(f"{a.name}: obs {i} starts {key} at its element {k}")
        n = dims[key]
        for j in range(n):
            pj, lj = lay["elements"][i + j] if i + j < D else (None, None)
            if pj is None or where.get(pj) != (key, j) or lj != lag:
                raise ProbeError(f"{a.name}: obs {i + j} breaks {key} at lag {lag}")
        chunks.append([key, lag])
        i += n
    r.terms = terms
    r.history = {
        "length": 1,
        "layout": "term_major",
        "order": "oldest_first",
        "init": lay["init"],
        "chunks": chunks,
    } | ({"first_frame": r.history["first_frame"]} if "first_frame" in r.history else {})
    r.obs_dim = a.obs_dim
    r.findings.append(
        f"the observation is one frame of {len(lay['frame'])} elements laid out "
        f"{len(chunks)} times by term and age (history.chunks, lags up to "
        f"{max(c[1] for c in chunks)})"
    )
    return r


def read_twb_adapter(
    policy_cpp: str | Path, mjcf: str | Path, cls: str = "Policy", variant: str | None = None
):
    """Compile, probe, write contracts and check them: returns (probe result, {"trained",
    "port"}, verification). The port contract is checked by building the observation both
    ways on random inputs; a mismatch is a term the probe misread or gaitkeeper lacks.
    ``variant`` names the one to read when the port has several (its ``names()``)."""
    a = Adapter(policy_cpp, cls, variant)
    r = probe_any(a)
    cs = contracts(r, mjcf, policy_cpp)
    v = verify(a, cs["port"])
    v["ok"] = v["max_abs"] < 1e-4
    if not v["ok"] and not r.history.get("chunks"):
        # a layout that only looked like one history window: read it frame by frame
        try:
            r2 = probe_frames(a)
            cs2 = contracts(r2, mjcf, policy_cpp)
            v2 = verify(a, cs2["port"])
            if v2["max_abs"] < v["max_abs"]:
                r, cs, v = r2, cs2, v2
                v["ok"] = v["max_abs"] < 1e-4
        except ProbeError:
            pass
    for c in cs.values():
        note = c.get("source.note")
        c.data["source"]["note"] = note + (
            "; port observation matches the adapter's on random inputs"
            if v["ok"]
            else f"; observation DIFFERS from the adapter's by up to {v['max_abs']:.3g}"
        )
    return r, cs, v


def verify(a: Adapter, port: Any, steps: int = 80, seed: int = 0) -> dict[str, Any]:
    """Feed the adapter and gaitkeeper's observation builder the same random inputs (the
    port contract's view: held joints observed as it says) and compare what the policy
    would see. Returns the largest error overall and per term."""
    from ..commands import CommandGate, policy_command
    from ..terms import ObservationBuilder, quat_to_mat, term_slices

    rng = np.random.default_rng(seed)
    names = list(port.get("policy_io.joints.names"))
    p2m = list(port.get("policy_io.joints.joint_ids_map"))
    n = len(names)
    act = port.get("control.actions.joint_pos")
    off = np.array([act["offset"][nm] for nm in names])
    sc = np.array([act["scale"][nm] for nm in names])
    default = np.array([port.get("control.default_joint_pos")[nm] for nm in names])
    obs_mode = {}
    for e in (port.get("control.ownership", None) or {}).get("external", []) or []:
        for j in e["joints"]:
            obs_mode[j] = e.get("obs", "real")
    lim = port.get("policy_io.commands.base_velocity.limit")
    shaping = port.get("policy_io.commands.base_velocity.shaping", None)
    gate_spec = port.get("policy_io.commands.base_velocity.gate", None)
    gate = CommandGate(gate_spec, POLICY_DT) if gate_spec else None
    if gate is not None:  # run well past the warm-up, so the gate opens and shuts
        steps = max(steps, int(round(gate.warmup / POLICY_DT)) + 80)
    builder = ObservationBuilder(port)
    from ..tables import SDK_TABLES

    table = SDK_TABLES["unitree_g1_29dof"][1]
    ex_m = [table.index(j) for j in builder.extra_joints]
    obs_names = names + builder.extra_joints
    group = port.get("policy_io.observation_groups.policy")
    cols = term_slices(group["terms"], group.get("history", {}))
    a.reset()
    prev = np.zeros(n)
    worst = np.zeros(a.obs_dim)
    for t in range(steps):
        q = STANCE + rng.normal(0, 0.1, NUM_MOTOR)
        dq = rng.normal(0, 1.0, NUM_MOTOR)
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        ang = rng.uniform(0, 0.4)
        quat = np.r_[math.cos(ang / 2), math.sin(ang / 2) * axis]
        R = quat_to_mat(quat[None])[0]
        gyro = rng.normal(0, 0.5, 3)
        v_w = rng.normal(0, 0.5, 3)
        cmd = np.array([rng.uniform(*lim["vx"]), rng.uniform(*lim["vy"]), rng.uniform(*lim["wz"])])
        if t % 7 == 3:
            cmd[:] = 0.0  # a standing step, for gated clocks
        if t % 11 == 5:
            cmd[:2] = 0.0  # position reached, heading not
        # the harness's task: no waypoint on some steps, else a random one
        d_ = 0.0 if t % 5 == 0 else float(rng.uniform(0.0, 3.0))
        b_ = float(rng.uniform(-math.pi, math.pi))
        task = np.array([d_, rng.uniform(-math.pi, math.pi), d_ * math.cos(b_), d_ * math.sin(b_)])
        action = np.clip(rng.normal(0, 0.5, n), -1.0, 1.0)
        arm = STANCE.copy()
        arm[ARM_LEFT_FIRST:] += rng.normal(0, 0.1, NUM_MOTOR - ARM_LEFT_FIRST)
        _, seen = a.step(
            q,
            dq,
            gyro,
            R.T @ v_w,
            R.T @ np.array([0.0, 0.0, -1.0]),
            cmd,
            action=action,
            arm_pose=arm,
            quat=quat,
            task=np.r_[task, np.zeros(60)],
        )
        qj, vj = q[p2m].copy(), dq[p2m].copy()
        for k, nm in enumerate(names):
            mode = obs_mode.get(nm, "real")
            if mode == "echo_action":
                qj[k], vj[k] = off[k] + sc[k] * prev[k], 0.0
            elif mode == "default":
                qj[k], vj[k] = default[k], 0.0
        seen_cmd = policy_command(cmd, task, shaping, gate, t)
        qj, vj = np.r_[qj, q[ex_m]], np.r_[vj, dq[ex_m]]
        tgt = np.r_[np.full(len(names), np.nan), arm[ex_m]]
        mine = builder.step(quat, gyro, qj, vj, obs_names, seen_cmd, t, prev, v_w, tgt)
        worst = np.maximum(worst, np.abs(mine - seen[0][: a.obs_dim]))
        prev = action
    per_term = {k: float(worst[c].max()) for k, c in cols.items()}
    return {"max_abs": float(worst.max()), "per_term": per_term}
