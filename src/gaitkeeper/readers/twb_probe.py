"""Read a teleop-walking-benchmark adapter by experiment, then write its contracts.

``probe(adapter)`` runs the compiled adapter (``twb_adapter.Adapter``) on chosen inputs and
reads off what it does:

* the action map: which motor each action element moves, by how much (the scale), from
  where (the offset), and where it stops (a clip);
* the observation layout: which input each observation element follows and with what gain
  (base_ang_vel, projected_gravity, velocity_commands, base_lin_vel, joint_pos_rel,
  joint_vel_rel, last_action), per joint whether the position is measured or faked from the
  last action, and the default pose the positions are taken relative to;
* the history: length, layout, order and how it is filled after a reset;
* a gait clock: its period, its phase at the first step, and the command norm below which
  it is zeroed;
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
            arm_pose=STANCE,
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


def probe(a: Adapter) -> ProbeResult:
    n, steps = a.act_dim, WARM + LAGS + 1
    findings: list[str] = []
    base_obs, base_tgt = _run(a, steps)
    varying = np.flatnonzero(np.abs(np.diff(base_obs[WARM - 4 :], axis=0)).max(axis=0) > TOL)

    # -- responses: family, element -> per lag {obs index: gain}
    resp: dict[tuple[str, int], list[dict[int, float]]] = {}
    tresp: dict[int, dict[int, float]] = {}
    sizes = {"q": NUM_MOTOR, "dq": NUM_MOTOR, "action": n} | {k: 3 for k in SCALARS}
    for fam, size in sizes.items():
        for e in range(size):
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
    for m in range(NUM_MOTOR):
        for fam, tid in (("q", "joint_pos_rel"), ("dq", "joint_vel_rel")):
            for i, g in newest(fam, m).items():
                if m not in kq:
                    raise ProbeError(f"{a.name}: obs {i} follows motor {m}, which no action drives")
                k = kq[m]
                if fam == "q" and is_diff(m, i, g):
                    put(i, ("joint_vel_diff", k, _round(g * POLICY_DT)))
                    continue
                put(i, (tid, k, _round(g)))
                if fam == "q":
                    default_pose[k] = _round(STANCE[m] - base_obs[WARM][i] / g)
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

    # clock elements and their ages; constants
    # (a clock whose rate follows the command responds to it a step later, so it can carry
    # an age; what matters is that nothing above describes it)
    vary = [int(i) for i in varying if int(i) not in desc]
    newest_clock = _newest_copies(base_obs, vary, H)
    clock = None
    if newest_clock:
        clock, cdesc = _clock(a, newest_clock, findings)
        for i, d_ in cdesc.items():
            put(i, d_)
    explained_old = {i for i, ag in age_of.items() if ag > 0} | (set(vary) - set(newest_clock))
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
        for _ in range(2):
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
        if t["elements"] != list(range(full_dim[t["id"]])):
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
        shaping=_shaping(a, desc, findings),
    )


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


def _clock(
    a: Adapter, idx: list[int], findings: list[str]
) -> tuple[dict[str, Any], dict[int, tuple[str, int, float]]]:
    """Fit sin(w t + phi_i) to each clock element and name it as an element of gait_phase
    ([sin, cos] of one phase) or gait_phase_legs ([sin a, sin b, cos a, cos b], b half a
    period on)."""
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
            elif t["id"] == "gait_phase_legs":
                ck = r.clock or {}
                term["params"] = {"period": ck["period"]}
                if ck.get("clock_offset_steps"):
                    term["params"]["clock_offset_steps"] = ck["clock_offset_steps"]
            elif t["id"] == "constant":
                term["params"] = {"value": t["value"]}
            if "index" in t:
                term["params"]["index"] = t["index"]
            terms.append(term)
        hist = {k: r.history[k] for k in ("length", "layout", "order", "init")}
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
            },
            "file",
            detail,
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


def read_twb_adapter(policy_cpp: str | Path, mjcf: str | Path, cls: str = "Policy"):
    """Compile, probe, write contracts and check them: returns (probe result, {"trained",
    "port"}, verification). The port contract is checked by building the observation both
    ways on random inputs; a mismatch is a term the probe misread or gaitkeeper lacks."""
    a = Adapter(policy_cpp, cls)
    r = probe(a)
    cs = contracts(r, mjcf, policy_cpp)
    v = verify(a, cs["port"])
    v["ok"] = v["max_abs"] < 1e-4
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
    from ..commands import shape as shape_command
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
    builder = ObservationBuilder(port)
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
        _, seen = a.step(
            q,
            dq,
            gyro,
            R.T @ v_w,
            R.T @ np.array([0.0, 0.0, -1.0]),
            cmd,
            action=action,
            arm_pose=STANCE,
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
        seen_cmd = shape_command(cmd, task, shaping)
        mine = builder.step(quat, gyro, qj, vj, names, seen_cmd, t, prev, v_w)
        worst = np.maximum(worst, np.abs(mine - seen[0][: a.obs_dim]))
        prev = action
    per_term = {k: float(worst[c].max()) for k, c in cols.items()}
    return {"max_abs": float(worst.max()), "per_term": per_term}
