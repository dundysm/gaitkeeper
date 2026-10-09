"""Boundary D: the inverse dynamics residual (plan section 7.5, Appendix B).

For every recorded physics step: set ``qpos`` and ``qvel`` from the trace, set
``qacc`` by finite difference of the next row's velocity, run MuJoCo inverse
dynamics with the discrete flag, and subtract the torque the contract
predicts, ``clip(kp (target - q) - kd v, +-limit)``. The trace's own torque
channel is never used. What remains is force the analysis model cannot
explain.

The analysis model is the target with the contract's drive (kind, gains,
limits) at the trace's sim step under ``implicitfast``. Its passive
parameters are the target's own, since those are what D measures; the
contract's values are listed next to them.

Clean steps (per joint): no contact between the world and any body in the
joint's subtree (so swing for a leg joint, every step for arms and waist), the
drive not near its effort limit (``kp |e| + kd |v| < 0.9 limit``), and both rows
of the finite difference in one episode.

Detection names a chain, never a parameter: a chain is above the floor when a
joint's clean-step RMS exceeds ``FLOOR_MARGIN`` times the floor calibrated for
the trace's source engine. The root residual (force on the floating base) is
reported on its own. Parameter fits ``r = dI a + dB v + c`` and the armature
search used when the analysis model has joint friction are research output,
off by default and never part of a verdict.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contract import Contract
from .models import LoadedModel, bare, find_id, load_model
from .trace import Trace

FLOOR_MARGIN = 3.0
NEAR_LIMIT = 0.9
CONTACT_MARGIN = 0.005  # m: a body this close to the world counts as touching
CONTACT_DILATE = 2  # physics rows either side of a contact also count as stance
FLOORS_PATH = Path(__file__).parent / "data" / "floors.json"
FIT_MIN_ROWS = 200
FIT_HELD_OUT_R2 = 0.5
FIT_SPLIT_AGREE = 0.5  # halves must agree within this fraction of the larger dI
CONFOUND_CORR = 0.9
SEARCH_PASSES = 4
ARMATURE_GRID = np.round(np.arange(-0.03, 0.0301, 0.001), 4)


def chain_of(name: str) -> str:
    """Humanoid chain from a joint name; ``other`` when no rule matches."""
    n = bare(name).lower()
    side = "left " if n.startswith("left") else ("right " if n.startswith("right") else "")
    if any(k in n for k in ("hip", "knee", "ankle")):
        return f"{side}leg".strip()
    if any(k in n for k in ("shoulder", "elbow", "wrist", "hand")):
        return f"{side}arm".strip()
    if any(k in n for k in ("waist", "torso")):
        return "waist"
    return "other"


def engine_key(meta: dict[str, Any]) -> str:
    """Name of the engine that produced a trace, for the floor table."""
    if meta.get("engine"):
        return str(meta["engine"])
    fw = meta.get("framework") or {}
    if fw.get("name") == "mjlab":
        return f"mjlab {fw.get('version')} / mujoco_warp {fw.get('mujoco_warp')}"
    return str(fw.get("name") or "unknown")


def load_floors(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else FLOORS_PATH
    return json.loads(p.read_text()).get("engines", {}) if p.exists() else {}


# -- analysis model -------------------------------------------------------------------------------


@dataclass
class Analysis:
    model: mujoco.MjModel
    names: list[str]  # contract joint names (bare)
    jid: np.ndarray
    qadr: np.ndarray
    dadr: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    limit: np.ndarray
    limit_from: str
    source: str
    notes: list[str] = field(default_factory=list)


def analysis_model(
    target: str | Path | mujoco.MjModel, contract: Contract, sim_dt: float
) -> Analysis:
    """The target with the contract's drive at the trace's sim step. ``target``
    is a path (MJCF, compiled model or trace directory) or a model, copied."""
    if isinstance(target, mujoco.MjModel):
        lm = LoadedModel(copy.copy(target), "given model", True)
    else:
        lm = load_model(target)
    m = lm.model
    names = [bare(n) for n in contract.get("policy_io.joints.names")]
    jid = np.array([find_id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names])
    if (jid < 0).any():
        raise KeyError(
            f"joints not in the analysis model: {[n for n, j in zip(names, jid) if j < 0]}"
        )
    kpd, kdd = contract.get("control.actuators.kp"), contract.get("control.actuators.kd")
    kp = np.array([float(kpd[n]) for n in names])
    kd = np.array([float(kdd[n]) for n in names])
    el = contract.get("model.effort_limit", None)
    if isinstance(el, dict) and all(el.get(n) for n in names):
        lim, lim_from = np.array([float(el[n]) for n in names]), "contract"
    else:
        from .runner import model_torque_limits

        aid0 = _actuator_of(m, jid)
        lim, _ = model_torque_limits(m, jid, aid0)
        lim_from = "target model (contract states none)"
    m.opt.timestep = float(sim_dt)
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    aid = _actuator_of(m, jid)
    for i in range(len(names)):
        a, j = int(aid[i]), int(jid[i])
        if a < 0:
            raise KeyError(f"no actuator on {names[i]}")
        m.jnt_actfrclimited[j] = 0
        m.actuator_trntype[a] = mujoco.mjtTrn.mjTRN_JOINT
        m.actuator_gear[a, :] = 0.0
        m.actuator_gear[a, 0] = 1.0
        m.actuator_dyntype[a] = mujoco.mjtDyn.mjDYN_NONE
        m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_gainprm[a, 0] = kp[i]
        m.actuator_biasprm[a, :3] = [0.0, -kp[i], -kd[i]]
        m.actuator_ctrllimited[a] = 0
        m.actuator_forcelimited[a] = 1 if np.isfinite(lim[i]) else 0
        if np.isfinite(lim[i]):
            m.actuator_forcerange[a] = [-lim[i], lim[i]]
    notes = list(lm.notes)
    notes.append(
        f"analysis model {lm.source} at the trace's step {sim_dt * 1e3:.4g} ms, implicitfast, "
        f"contract drive (kp, kd), limits from {lim_from}"
    )
    return Analysis(
        m,
        names,
        jid,
        m.jnt_qposadr[jid].copy(),
        m.jnt_dofadr[jid].copy(),
        kp,
        kd,
        lim,
        lim_from,
        lm.source,
        notes,
    )


def _actuator_of(m: mujoco.MjModel, jid: np.ndarray) -> np.ndarray:
    act = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            act.setdefault(int(m.actuator_trnid[a, 0]), a)
    return np.array([act.get(int(j), -1) for j in jid])


def passive_comparison(an: Analysis, contract: Contract) -> list[str]:
    """Target passive values next to the contract's, per field, where they differ."""
    out = []
    m = an.model
    for path, fld, label in (
        ("model.armature", "dof_armature", "armature"),
        ("model.joint_damping", "dof_damping", "damping"),
        ("model.joint_friction", "dof_frictionloss", "frictionloss"),
    ):
        ref = contract.get(path, None)
        if not isinstance(ref, dict):
            out.append(f"{label}: the contract states none")
            continue
        diffs = []
        for n, d in zip(an.names, an.dadr):
            if ref.get(n) is None:
                continue
            t, c = float(getattr(m, fld)[d]), float(ref[n])
            if abs(t - c) > 1e-9 + 1e-6 * abs(c):
                diffs.append((n, t, c))
        if diffs:
            span = ", ".join(f"{n} {t:.4g} vs {c:.4g}" for n, t, c in diffs[:4])
            more = f" and {len(diffs) - 4} more" if len(diffs) > 4 else ""
            out.append(
                f"{label}: target differs from the contract on {len(diffs)} joints ({span}{more})"
            )
        else:
            out.append(f"{label}: target equals the contract")
    return out


# -- physics rows ----------------------------------------------------------------------------------


@dataclass
class Rows:
    qpos: np.ndarray  # (S, nq) in the analysis model's layout
    qvel: np.ndarray
    ctrl: np.ndarray  # (S, J) targets in contract joint order
    valid: np.ndarray  # (S - 1,) rows s and s + 1 are consecutive steps of one episode
    h: float
    xfrc: np.ndarray | None  # (S, 3) world force
    xfrc_body: np.ndarray | None  # (S,) analysis body id, -1 none
    missing_joints: list[str]


def physics_rows(trace: Trace, an: Analysis) -> Rows | str:
    """Trace physics rows mapped into the analysis model, or why they cannot be."""
    P = {k[2:]: v for k, v in trace.arrays.items() if k.startswith("p/")}
    need = ("qpos", "qvel", "ctrl", "step", "substep")
    if any(k not in P for k in need):
        return "no physics-rate channels (p/qpos, p/qvel, p/ctrl, p/step, p/substep)"
    h = trace.meta.get("sim_dt")
    if not h:
        return "trace meta has no sim_dt"
    h = float(h)
    m = an.model
    lay = trace.meta.get("state_layout", {})
    tnames = [bare(n) for n in lay.get("joint_names", [])]
    src_q = np.asarray(P["qpos"], dtype=np.float64)
    src_v = np.asarray(P["qvel"], dtype=np.float64)
    if len(tnames) != src_q.shape[1] - 7:
        return "state layout joint names do not match the physics rows"
    S = len(src_q)
    qpos = np.tile(m.qpos0, (S, 1))
    qvel = np.zeros((S, m.nv))
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free:
        return "analysis model has no floating base"
    fq, fv = int(m.jnt_qposadr[free[0]]), int(m.jnt_dofadr[free[0]])
    qpos[:, fq : fq + 7] = src_q[:, :7]
    qvel[:, fv : fv + 6] = src_v[:, :6]
    missing = []
    for k, n in enumerate(tnames):
        j = find_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        if j < 0:
            missing.append(n)
            continue
        qpos[:, m.jnt_qposadr[j]] = src_q[:, 7 + k]
        qvel[:, m.jnt_dofadr[j]] = src_v[:, 6 + k]
    cn = [bare(n) for n in trace.meta.get("target_joint_names", tnames)]
    ctrl_src = np.asarray(P["ctrl"], dtype=np.float64)
    try:
        ctrl = ctrl_src[:, [cn.index(n) for n in an.names]]
    except ValueError:
        return "physics targets do not name every contract joint"
    step, sub = np.asarray(P["step"]).astype(int), np.asarray(P["substep"]).astype(int)
    dec = int(trace.meta.get("decimation") or (sub.max() + 1))
    flat = step * dec + sub
    valid = np.diff(flat) == 1
    reset = trace.get("reset")
    if reset is not None:
        for s in np.flatnonzero(valid):
            if sub[s + 1] == 0 and step[s + 1] < len(reset) and reset[step[s + 1]]:
                valid[s] = False
    xfrc = xb = None
    if "xfrc" in P and np.any(P["xfrc"]):
        xfrc = np.asarray(P["xfrc"], dtype=np.float64)
        if "xfrc_body" in P:
            bn = trace.meta.get("body_names")
            xb = np.array(
                [
                    -1 if b < 0 else find_id(m, mujoco.mjtObj.mjOBJ_BODY, bare(bn[b]) if bn else "")
                    for b in np.asarray(P["xfrc_body"]).astype(int)
                ]
            )
        else:
            bodies = {p[2] for p in trace.meta.get("pushes") or []}
            if len(bodies) != 1:
                return "pushes on more than one body without a per-row body channel"
            bid = find_id(m, mujoco.mjtObj.mjOBJ_BODY, bodies.pop())
            xb = np.where(np.abs(xfrc).sum(1) > 0, bid, -1)
        if (xb[np.abs(xfrc).sum(1) > 0] < 0).any():
            return "a pushed body is not in the analysis model"
    return Rows(qpos, qvel, ctrl, valid, h, xfrc, xb, missing)


# -- residual ---------------------------------------------------------------------------------------


@dataclass
class Series:
    r: np.ndarray  # (S-1, J) joint residual, N m
    a: np.ndarray  # (S-1, J) finite-difference acceleration
    v: np.ndarray  # (S-1, J)
    root: np.ndarray  # (S-1, 6) base residual: force N, torque N m
    clean: np.ndarray  # (S-1, J)
    stance: np.ndarray  # (S-1, J) contact in the joint's subtree
    near: np.ndarray  # (S-1, J) drive near its effort limit
    valid: np.ndarray  # (S-1,)


def _subtree_bodies(m: mujoco.MjModel, body: int) -> np.ndarray:
    inside = np.zeros(m.nbody, dtype=bool)
    inside[body] = True
    for b in range(body + 1, m.nbody):
        if inside[m.body_parentid[b]]:
            inside[b] = True
    return inside


def residual_series(an: Analysis, rows: Rows) -> Series:
    m = an.model
    d = mujoco.MjData(m)
    mc = mujoco.MjModel.__copy__(m)
    mc.geom_margin[:] += CONTACT_MARGIN
    dc = mujoco.MjData(mc)
    S = len(rows.qpos) - 1
    J = len(an.names)
    R = np.zeros((S, J))
    root = np.zeros((S, 6))
    touch = np.zeros((S, m.nbody), dtype=bool)
    aid = _actuator_of(m, an.jid)
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    fv = int(m.jnt_dofadr[free])
    qx = np.zeros(m.nv)
    h = rows.h
    static = m.body_weldid == 0  # the world and bodies welded to it (terrain)
    for s in range(S):
        if not rows.valid[s]:
            continue
        d.qpos[:] = rows.qpos[s]
        d.qvel[:] = rows.qvel[s]
        d.qacc[:] = (rows.qvel[s + 1] - rows.qvel[s]) / h
        d.ctrl[aid] = rows.ctrl[s]
        mujoco.mj_inverse(m, d)
        f = d.qfrc_inverse.copy()
        if rows.xfrc is not None and rows.xfrc_body[s] >= 0:
            qx[:] = 0.0
            b = int(rows.xfrc_body[s])
            mujoco.mj_applyFT(m, d, rows.xfrc[s], np.zeros(3), d.xipos[b], b, qx)
            f -= qx
        q, v = rows.qpos[s, an.qadr], rows.qvel[s, an.dadr]
        tau = np.clip(an.kp * (rows.ctrl[s] - q) - an.kd * v, -an.limit, an.limit)
        R[s] = f[an.dadr] - tau
        root[s] = f[fv : fv + 6]
        dc.qpos[:] = rows.qpos[s]
        mujoco.mj_kinematics(mc, dc)
        mujoco.mj_collision(mc, dc)
        if dc.ncon:
            g1, g2 = dc.contact.geom1[: dc.ncon], dc.contact.geom2[: dc.ncon]
            b1, b2 = mc.geom_bodyid[g1], mc.geom_bodyid[g2]
            w1, w2 = static[b1], static[b2]
            near_c = dc.contact.dist[: dc.ncon] < CONTACT_MARGIN
            touch[s, b2[w1 & ~w2 & near_c]] = True
            touch[s, b1[w2 & ~w1 & near_c]] = True
    # dilate contact in time
    k = CONTACT_DILATE
    if k:
        t = touch.copy()
        for o in range(1, k + 1):
            t[o:] |= touch[:-o]
            t[:-o] |= touch[o:]
        touch = t
    stance = np.zeros((S, J), dtype=bool)
    for i, j in enumerate(an.jid):
        sub = _subtree_bodies(m, int(m.jnt_bodyid[j]))
        stance[:, i] = touch[:, sub].any(1)
    v = rows.qvel[:S][:, an.dadr]
    e = rows.ctrl[:S] - rows.qpos[:S][:, an.qadr]
    near = np.abs(an.kp * e) + an.kd * np.abs(v) >= NEAR_LIMIT * an.limit
    a = (rows.qvel[1:] - rows.qvel[:-1])[:, an.dadr] / h
    clean = rows.valid[:, None] & ~stance & ~near
    return Series(R, a, v, root, clean, stance, near, rows.valid.copy())


# -- detection --------------------------------------------------------------------------------------


@dataclass
class JointStat:
    name: str
    chain: str
    n_clean: int
    rms: float
    max_abs: float
    floor: float | None
    ratio: float | None


@dataclass
class DResult:
    status: str  # "at_floor" | "above_floor" | "not_measured" | "uncalibrated"
    engine: str
    joints: list[JointStat] = field(default_factory=list)
    chains: dict[str, dict[str, Any]] = field(default_factory=dict)
    root: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    passive: list[str] = field(default_factory=list)
    fits: list[dict[str, Any]] | None = None
    reason: str = ""

    @property
    def above_chains(self) -> list[str]:
        out = [c for c, v in self.chains.items() if v["above"]]
        if self.root.get("above"):
            out.append("root/contact")
        return out

    def lines(self) -> list[str]:
        head = {
            "at_floor": "D at floor",
            "above_floor": "D above floor on " + ", ".join(self.above_chains),
            "not_measured": "D not measured",
            "uncalibrated": "D uncalibrated",
        }[self.status]
        out = [
            f"{head} (source engine {self.engine})" + (f": {self.reason}" if self.reason else "")
        ]
        for c, v in self.chains.items():
            fl = "uncalibrated" if v["ratio"] is None else f"{v['ratio']:.3g}x floor"
            out.append(
                f"  {c:<10} worst {v['worst']:<28} clean RMS {v['rms']:.4g} N m ({fl})"
                + ("  ABOVE" if v["above"] else "")
            )
        if self.root:
            r = self.root
            fl = "" if r.get("ratio") is None else f" ({r['ratio']:.3g}x floor)"
            out.append(
                f"  root       RMS force {r['force_rms']:.4g} N, torque {r['torque_rms']:.4g} N m{fl}"
                + ("  ABOVE" if r.get("above") else "")
            )
        out.extend(f"  {p}" for p in self.passive)
        out.extend(f"  note: {n}" for n in self.notes)
        if self.fits is not None:
            out.append("  parameter fits (research, never part of a verdict):")
            for f in self.fits:
                out.append(f"    {f['text']}")
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "engine": self.engine,
            "above": self.above_chains,
            "chains": self.chains,
            "root": self.root,
            "joints": [j.__dict__ for j in self.joints],
            "notes": self.notes,
            "passive": self.passive,
            "fits": self.fits,
            "reason": self.reason,
        }


def joint_stats(an: Analysis, se: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    J = len(an.names)
    n = se.clean.sum(0)
    rms = np.array(
        [
            math.sqrt(float(np.mean(se.r[se.clean[:, i], i] ** 2))) if n[i] else math.nan
            for i in range(J)
        ]
    )
    mx = np.array(
        [float(np.abs(se.r[se.clean[:, i], i]).max()) if n[i] else math.nan for i in range(J)]
    )
    return n, rms, mx


def root_stats(se: Series) -> tuple[float, float]:
    rr = se.root[se.valid]
    if not len(rr):
        return math.nan, math.nan
    return (
        float(np.sqrt(np.mean(np.sum(rr[:, :3] ** 2, 1)))),
        float(np.sqrt(np.mean(np.sum(rr[:, 3:] ** 2, 1)))),
    )


def dynamics_residual(
    trace: Trace,
    contract: Contract,
    target: str | Path,
    floors: dict[str, Any] | None = None,
    engine: str | None = None,
    fits: bool = False,
) -> DResult:
    eng = engine or engine_key(trace.meta)
    h = trace.meta.get("sim_dt")
    if not h or "p/qpos" not in trace.arrays:
        return DResult("not_measured", eng, reason="no physics-rate channels in the trace")
    an = analysis_model(target, contract, float(h))
    rows = physics_rows(trace, an)
    if isinstance(rows, str):
        return DResult("not_measured", eng, reason=rows)
    se = residual_series(an, rows)
    floors = load_floors() if floors is None else floors
    fl = floors.get(eng)
    n, rms, mx = joint_stats(an, se)
    fr, tr = root_stats(se)
    res = DResult("at_floor", eng, notes=list(an.notes), passive=passive_comparison(an, contract))
    if rows.missing_joints:
        res.notes.append(f"trace joints not in the analysis model: {rows.missing_joints}")
    for i, nm in enumerate(an.names):
        f = (fl or {}).get("joint_rms", {}).get(nm)
        ratio = float(rms[i] / f) if (f and np.isfinite(rms[i])) else None
        res.joints.append(
            JointStat(nm, chain_of(nm), int(n[i]), float(rms[i]), float(mx[i]), f, ratio)
        )
    for js in res.joints:
        c = res.chains.setdefault(
            js.chain,
            {"above": False, "worst": js.name, "rms": js.rms, "ratio": js.ratio, "n_clean": 0},
        )
        c["n_clean"] += js.n_clean
        key = js.ratio if js.ratio is not None else -1.0
        cur = c["ratio"] if c["ratio"] is not None else -1.0
        if key > cur or (c["ratio"] is None and js.ratio is None and js.rms > c["rms"]):
            c.update(worst=js.name, rms=js.rms, ratio=js.ratio)
        if js.ratio is not None and js.ratio > FLOOR_MARGIN:
            c["above"] = True
    res.root = {"force_rms": fr, "torque_rms": tr, "ratio": None, "above": False}
    if fl:
        rf, rt = fl.get("root_force_rms"), fl.get("root_torque_rms")
        if rf and rt:
            ratio = max(fr / rf, tr / rt)
            res.root.update(ratio=float(ratio), above=bool(ratio > FLOOR_MARGIN))
    if not fl:
        res.status = "uncalibrated"
        res.reason = f"no floor calibrated for {eng}; PHYSICS is blocked until it is"
    elif res.above_chains:
        res.status = "above_floor"
    if any(js.ratio is None for js in res.joints) and fl:
        res.notes.append(
            "no floor for: " + ", ".join(js.name for js in res.joints if js.ratio is None)
        )
    if fits:
        res.fits = parameter_fits(an, rows, se)
    return res


# -- floor calibration ------------------------------------------------------------------------------


def calibrate(traces: list[Trace], contract_of, model_of) -> dict[str, Any]:
    """Floor from traces analysed against their own model: per joint the largest
    clean-step RMS over the traces, and the root residual RMS."""
    joint: dict[str, float] = {}
    jmax: dict[str, float] = {}
    rf = rt = 0.0
    used = []
    for tr in traces:
        c = contract_of(tr)
        an = analysis_model(model_of(tr), c, float(tr.meta["sim_dt"]))
        rows = physics_rows(tr, an)
        if isinstance(rows, str):
            raise ValueError(rows)
        se = residual_series(an, rows)
        n, rms, mx = joint_stats(an, se)
        for i, nm in enumerate(an.names):
            if n[i] >= FIT_MIN_ROWS:
                joint[nm] = max(joint.get(nm, 0.0), float(rms[i]))
                jmax[nm] = max(jmax.get(nm, 0.0), float(mx[i]))
        f, t = root_stats(se)
        rf, rt = max(rf, f), max(rt, t)
        used.append(str(tr.path) if tr.path else tr.meta.get("engine", "trace"))
    return {
        "joint_rms": joint,
        "joint_max": jmax,
        "root_force_rms": rf,
        "root_torque_rms": rt,
        "calibrated_on": used,
        "analysis_mujoco": [mujoco.__version__],
        "margin": FLOOR_MARGIN,
    }


# -- parameter fits (research) -----------------------------------------------------------------------


def _lsq(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    co, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ co
    tot = float(np.sum((y - y.mean()) ** 2)) or 1e-30
    return co, 1.0 - float(np.sum(res**2)) / tot


def fit_joint(r: np.ndarray, a: np.ndarray, v: np.ndarray) -> dict[str, Any]:
    """``r = dI a + dB v + c`` with a split-half check and a confounding check."""
    n = len(r)
    if n < FIT_MIN_ROWS:
        return {"status": "abstain", "why": f"{n} clean rows"}
    X = np.c_[a, v, np.ones(n)]
    co, r2 = _lsq(X, r)
    half = n // 2
    c1, _ = _lsq(X[:half], r[:half])
    pred = X[half:] @ c1
    tot = float(np.sum((r[half:] - r[half:].mean()) ** 2)) or 1e-30
    r2_held = 1.0 - float(np.sum((r[half:] - pred) ** 2)) / tot
    c2, _ = _lsq(X[half:], r[half:])
    corr = float(np.corrcoef(a, v)[0, 1]) if np.std(a) > 0 and np.std(v) > 0 else 1.0
    out = {
        "dI": float(co[0]),
        "dB": float(co[1]),
        "c": float(co[2]),
        "r2": r2,
        "r2_held_out": r2_held,
        "dI_halves": [float(c1[0]), float(c2[0])],
        "corr_a_v": corr,
    }
    why = []
    if abs(corr) > CONFOUND_CORR:
        why.append(f"acceleration and velocity confounded (corr {corr:+.2f})")
    if r2_held < FIT_HELD_OUT_R2:
        why.append(f"held-out R^2 {r2_held:.2f}")
    big = max(abs(c1[0]), abs(c2[0]))
    if big > 0 and abs(c1[0] - c2[0]) > FIT_SPLIT_AGREE * big:
        why.append(f"halves disagree on dI ({c1[0]:+.4f}, {c2[0]:+.4f})")
    out["status"] = "abstain" if why else "fit"
    out["why"] = "; ".join(why)
    return out


def _search_pass(
    an: Analysis, rows: Rows, se: Series, joints: list[int], start: np.ndarray
) -> dict[int, np.ndarray]:
    """Cost per joint for every grid shift applied to all searched joints at once,
    around the armature in ``start``."""
    m = an.model
    S = len(se.r)
    sel = np.flatnonzero(se.clean[:, joints].any(1))
    cost = {j: [] for j in joints}
    d = mujoco.MjData(m)
    aid = _actuator_of(m, an.jid)
    for delta in ARMATURE_GRID:
        ok = {j: start[j] + delta > 0 for j in joints}  # armature must stay positive
        if not any(ok.values()):
            for j in joints:
                cost[j].append(math.inf)
            continue
        for j in joints:
            # a joint that cannot take this shift keeps its start value
            m.dof_armature[an.dadr[j]] = start[j] + delta if ok[j] else start[j]
        r = np.zeros((S, len(an.names)))
        for s in sel:
            d.qpos[:] = rows.qpos[s]
            d.qvel[:] = rows.qvel[s]
            d.qacc[:] = (rows.qvel[s + 1] - rows.qvel[s]) / rows.h
            d.ctrl[aid] = rows.ctrl[s]
            mujoco.mj_inverse(m, d)
            q, v = rows.qpos[s, an.qadr], rows.qvel[s, an.dadr]
            tau = np.clip(an.kp * (rows.ctrl[s] - q) - an.kd * v, -an.limit, an.limit)
            r[s] = d.qfrc_inverse[an.dadr] - tau
        for j in joints:
            if not ok[j]:
                cost[j].append(math.inf)
                continue
            k = se.clean[:, j]
            X = np.c_[se.v[k, j], np.ones(k.sum())]
            co, *_ = np.linalg.lstsq(X, r[k, j], rcond=None)
            cost[j].append(float(np.sqrt(np.mean((r[k, j] - X @ co) ** 2))))
    return {j: np.array(c) for j, c in cost.items()}


def armature_search(
    an: Analysis, rows: Rows, se: Series, joints: list[int]
) -> dict[int, dict[str, Any]]:
    """Joint friction makes the inferred constraint force depend on inertia, so
    the linear fit fails; search the analysis armature instead (plan 7.5).

    The discrete inverse couples joints through the mass matrix, so a shift on
    one joint moves the residual of the others a little. The search therefore
    repeats around each joint's current best until no joint moves."""
    m = an.model
    base = m.dof_armature[an.dadr].copy()
    cur = base.copy()
    out: dict[int, dict[str, Any]] = {}
    try:
        for _ in range(SEARCH_PASSES):
            cost = _search_pass(an, rows, se, joints, cur)
            moved = False
            nxt = cur.copy()
            for j in joints:
                c = cost[j]
                ok = np.isfinite(c)
                b = int(np.argmin(np.where(ok, c, np.inf)))
                idx = np.flatnonzero(ok)
                edge = min(c[idx[0]], c[idx[-1]])
                nxt[j] = cur[j] + ARMATURE_GRID[b]
                moved |= bool(ARMATURE_GRID[b] != 0)
                out[j] = {
                    "dI": float(base[j] - nxt[j]),
                    "min_rms": float(c[b]),
                    "sharpness": float(edge / c[b]) if c[b] > 0 else math.inf,
                    "at_grid_edge": b in (int(idx[0]), int(idx[-1])),
                }
            cur = nxt
            if not moved:
                break
    finally:
        m.dof_armature[an.dadr] = base
    return out


def parameter_fits(an: Analysis, rows: Rows, se: Series) -> list[dict[str, Any]]:
    fric = [i for i in range(len(an.names)) if an.model.dof_frictionloss[an.dadr[i]] > 0]
    search = armature_search(an, rows, se, fric) if fric else {}
    out = []
    for i, nm in enumerate(an.names):
        k = se.clean[:, i]
        if i in search:
            s = search[i]
            ok = not s["at_grid_edge"] and s["sharpness"] > 1.5
            f = {
                "joint": nm,
                "method": "armature search (joint friction)",
                **s,
                "status": "fit" if ok else "abstain",
            }
            f["text"] = (
                f"{nm}: armature search dI {s['dI']:+.4f} (sharpness {s['sharpness']:.1f})"
                + ("" if ok else "; abstain: minimum not sharp or at the grid edge")
            )
        else:
            f = {
                "joint": nm,
                "method": "linear fit",
                **fit_joint(se.r[k, i], se.a[k, i], se.v[k, i]),
            }
            if f["status"] == "fit":
                f["text"] = (
                    f"{nm}: dI {f['dI']:+.5f} dB {f['dB']:+.4f} c {f['c']:+.4f} "
                    f"(R^2 {f['r2']:.3f}, held out {f['r2_held_out']:.3f})"
                )
            else:
                f["text"] = f"{nm}: parameter not identifiable ({f['why']})"
        out.append(f)
    return out
