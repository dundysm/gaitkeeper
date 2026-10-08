"""The physics counterfactual for sources whose model is recorded (plan section 4).

The closed loop runs on the recorder's model and on the target, same policy,
contract, backend, scenario and seeds. Only the model differs. When the two
fail on significantly different numbers of seeds (Fisher's exact test), the
model is the cause, and swapping parameter groups from the source into the
target localizes it: a group is named when swapping it alone significantly
reduces the target's failures. The scenario is the golden trace's own schedule
and pushes, judged on the segments the source itself tracked (L2 bar, plan
7.2), so a limitation the source shows is never counted against the target.

Groups are swapped by element name (bare names, so a recorder's ``robot/``
prefix does not matter): armature, joint damping and friction, joint ranges,
effort limits, mass and inertia, body placement, contact parameters, time
step and solver. Collision geometry and elements present in one model only
cannot be swapped; when no group and not all groups together restore the
outcome, that remainder is what the report names.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contract import Contract
from .models import bare, find_id, load_model
from .runner import Push, model_torque_limits
from .task import AXES, Segment, TaskOutcome, TaskSpec, rejudge, run_task
from .terms import quat_to_mat
from .trace import Trace

CF_ALPHA = 0.01  # outcome change: two-sided Fisher p below this
LOC_ALPHA = 0.05  # a group restores: one-sided Fisher p below this against the target
SEEDS = tuple(range(1, 13))


# -- statistics -----------------------------------------------------------------------------------


def _hyper(k: int, n1: int, n2: int, K: int) -> float:
    return math.comb(n1, k) * math.comb(n2, K - k) / math.comb(n1 + n2, K)


def fisher(f1: int, n1: int, f2: int, n2: int, side: str = "two") -> float:
    """Fisher's exact test on failures f1 of n1 against f2 of n2. ``side`` "less":
    the probability of f1 this low or lower given the margins."""
    K = f1 + f2
    lo, hi = max(0, K - n2), min(K, n1)
    probs = {k: _hyper(k, n1, n2, K) for k in range(lo, hi + 1)}
    if side == "less":
        return float(sum(p for k, p in probs.items() if k <= f1))
    p0 = probs[f1]
    return float(min(1.0, sum(p for p in probs.values() if p <= p0 * (1 + 1e-9))))


# -- the source's own scenario --------------------------------------------------------------------


def golden_scenario(trace: Trace, policy_dt: float | None = None) -> TaskSpec:
    """The trace's schedule and pushes, judged where the source met the bar."""
    meta = trace.meta
    if not meta.get("schedule"):
        raise ValueError("the trace records no command schedule")
    dt = float(policy_dt or meta.get("policy_dt") or 0.02)
    sched = [(float(r[0]), (float(r[1]), float(r[2]), float(r[3]))) for r in meta["schedule"]]
    seconds = trace.n_steps * dt
    pushes = [
        Push(float(p[0]), "force", tuple(float(x) for x in p[3]), str(p[2]), float(p[1]))
        for p in meta.get("pushes") or []
    ]
    spec = TaskSpec("golden schedule", seconds, sched, pushes)
    q, v = np.asarray(trace["qpos"], float), np.asarray(trace["qvel"], float)
    rot = quat_to_mat(q[:, 3:7])
    vb = np.einsum("tji,tj->ti", rot, v[:, :3])
    vel = np.c_[vb[:, 0], vb[:, 1], v[:, 5]]
    segs = []
    for sg in spec.segment_list():
        k0, k1 = int(round((sg.t0 + (sg.t1 - sg.t0) / 2) / dt)), int(round(sg.t1 / dt))
        a = vel[k0:k1].mean(0)
        judged = tuple(
            bool(c == 0 or (a[i] * c > 0 and abs(a[i]) >= 0.5 * abs(c)))
            for i, c in enumerate(sg.cmd)
        )
        stand = all(c == 0 for c in sg.cmd) and math.hypot(a[0], a[1]) < 0.05
        segs.append(Segment(sg.t0, sg.t1, sg.cmd, judged, stand))
    spec.segments = segs
    return spec


def source_tracking_text(spec: TaskSpec) -> list[str]:
    out = []
    for sg in spec.segment_list():
        held = [AXES[i] for i, c in enumerate(sg.cmd) if c != 0 and sg.judged[i]]
        dropped = [AXES[i] for i, c in enumerate(sg.cmd) if c != 0 and not sg.judged[i]]
        if dropped:
            out.append(
                f"  {sg.label()}: not judged on {', '.join(dropped)} (the source misses the bar)"
            )
        elif not held and not sg.judge_stand:
            out.append(f"  {sg.label()}: standing not judged (the source drifts)")
    return out


# -- parameter groups -----------------------------------------------------------------------------


def _joint_pairs(dst: mujoco.MjModel, src: mujoco.MjModel) -> list[tuple[int, int]]:
    out = []
    for j in range(dst.njnt):
        if dst.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        k = find_id(src, mujoco.mjtObj.mjOBJ_JOINT, bare(dst.joint(j).name))
        if k >= 0:
            out.append((j, k))
    return out


def _body_pairs(dst: mujoco.MjModel, src: mujoco.MjModel) -> list[tuple[int, int]]:
    out = []
    for b in range(1, dst.nbody):
        k = find_id(src, mujoco.mjtObj.mjOBJ_BODY, bare(dst.body(b).name))
        if k > 0:
            out.append((b, k))
    return out


def _g_armature(dst, src):
    for j, k in _joint_pairs(dst, src):
        dst.dof_armature[dst.jnt_dofadr[j]] = src.dof_armature[src.jnt_dofadr[k]]


def _g_passive(dst, src):
    for j, k in _joint_pairs(dst, src):
        a, b = dst.jnt_dofadr[j], src.jnt_dofadr[k]
        dst.dof_damping[a] = src.dof_damping[b]
        dst.dof_frictionloss[a] = src.dof_frictionloss[b]


def _g_ranges(dst, src):
    for j, k in _joint_pairs(dst, src):
        dst.jnt_range[j] = src.jnt_range[k]
        dst.jnt_limited[j] = src.jnt_limited[k]


def _g_limits(dst, src):
    pairs = _joint_pairs(dst, src)
    act_d = {int(dst.actuator_trnid[a, 0]): a for a in range(dst.nu)}
    act_s = {int(src.actuator_trnid[a, 0]): a for a in range(src.nu)}
    jd = np.array([j for j, _ in pairs if j in act_d])
    js = np.array([k for j, k in pairs if j in act_d])
    lim, _ = model_torque_limits(src, js, np.array([act_s.get(int(k), -1) for k in js]))
    for j, value in zip(jd, lim):
        a = act_d[int(j)]
        dst.jnt_actfrclimited[j] = 0
        if np.isfinite(value):
            dst.actuator_forcelimited[a] = 1
            dst.actuator_forcerange[a] = [-value, value]
        else:
            dst.actuator_forcelimited[a] = 0


def _g_inertia(dst, src):
    for b, k in _body_pairs(dst, src):
        dst.body_mass[b] = src.body_mass[k]
        dst.body_inertia[b] = src.body_inertia[k]
        dst.body_ipos[b] = src.body_ipos[k]
        dst.body_iquat[b] = src.body_iquat[k]


def _g_placement(dst, src):
    for b, k in _body_pairs(dst, src):
        dst.body_pos[b] = src.body_pos[k]
        dst.body_quat[b] = src.body_quat[k]


_GEOM_CONTACT = (
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_solmix",
    "geom_margin",
    "geom_gap",
)


def _collision_geoms(m: mujoco.MjModel, body: int) -> list[int]:
    return [
        g
        for g in range(m.ngeom)
        if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]


def _static_collision(m: mujoco.MjModel) -> list[int]:
    return [
        g
        for g in range(m.ngeom)
        if m.body_weldid[m.geom_bodyid[g]] == 0 and (m.geom_contype[g] or m.geom_conaffinity[g])
    ]


def _copy_contact(dst, src, gd: list[int], gs: list[int]) -> None:
    if not gd or not gs:
        return
    for f in _GEOM_CONTACT:
        getattr(dst, f)[gd] = getattr(src, f)[gs[0]]
    dst.geom_condim[gd] = src.geom_condim[gs[0]]
    dst.geom_priority[gd] = src.geom_priority[gs[0]]


def _g_contact(dst, src):
    for b, k in _body_pairs(dst, src):
        _copy_contact(dst, src, _collision_geoms(dst, b), _collision_geoms(src, k))
    _copy_contact(dst, src, _static_collision(dst), _static_collision(src))
    dst.opt.cone = src.opt.cone
    dst.opt.impratio = src.opt.impratio


def _g_step(dst, src):
    for o in (
        "timestep",
        "iterations",
        "ls_iterations",
        "solver",
        "tolerance",
        "ls_tolerance",
        "noslip_iterations",
    ):
        setattr(dst.opt, o, getattr(src.opt, o))


def _part(fields: tuple[str, ...]):
    def f(dst, src):
        for b, k in _body_pairs(dst, src):
            for x in fields:
                getattr(dst, x)[b] = getattr(src, x)[k]

    return f


# Second level, tried inside a group that restores the outcome.
PARTS: dict[str, dict[str, Callable[[mujoco.MjModel, mujoco.MjModel], None]]] = {
    "mass and inertia": {
        "body masses": _part(("body_mass",)),
        "centers of mass": _part(("body_ipos",)),
        "rotational inertia": _part(("body_inertia", "body_iquat")),
    },
}

GROUPS: dict[str, Callable[[mujoco.MjModel, mujoco.MjModel], None]] = {
    "armature": _g_armature,
    "joint damping and friction": _g_passive,
    "joint ranges": _g_ranges,
    "effort limits": _g_limits,
    "mass and inertia": _g_inertia,
    "body placement": _g_placement,
    "contact parameters": _g_contact,
    "time step and solver": _g_step,
}


def group_differs(name: str, dst: mujoco.MjModel, src: mujoco.MjModel) -> bool:
    import copy

    probe = copy.copy(dst)
    GROUPS[name](probe, src)
    for f in dir(dst):
        if f.startswith("_"):
            continue
        try:
            x, y = getattr(dst, f), getattr(probe, f)
        except Exception:
            continue
        if isinstance(x, np.ndarray) and x.dtype.kind in "fi" and x.size and x.shape == y.shape:
            if not np.allclose(x, y, rtol=1e-9, atol=1e-12):
                return True
    for o in ("timestep", "iterations", "ls_iterations", "solver", "tolerance", "cone", "impratio"):
        if getattr(dst.opt, o) != getattr(probe.opt, o):
            return True
    return False


_SRC_CACHE: dict[str, mujoco.MjModel] = {}


def _source_model(path: str) -> mujoco.MjModel:
    if path not in _SRC_CACHE:
        _SRC_CACHE.clear()
        _SRC_CACHE[path] = load_model(path, cache=False).model
    return _SRC_CACHE[path]


def swap_edit(
    m: mujoco.MjModel, source: str, groups: tuple[str, ...], base_edit: Any = None
) -> None:
    """Model edit for the runner: the target's own edit, then the source's groups."""
    if base_edit is not None:
        base_edit(m)
    src = _source_model(source)
    for g in groups:
        if g in GROUPS:
            GROUPS[g](m, src)
        else:
            parent, part = g.split(": ", 1)
            PARTS[parent][part](m, src)


_FIELDS = {
    "mass and inertia": {"body_mass", "body_ipos", "body_inertia"},
    "mass and inertia: body masses": {"body_mass"},
    "mass and inertia: centers of mass": {"body_ipos"},
    "mass and inertia: rotational inertia": {"body_inertia"},
    "contact parameters": {"geom_friction"},
}


def _randomization_note(cf: Counterfactual, source: str) -> None:
    """Say so when what restores the outcome is part of the source's randomization draw."""
    p = Path(source) / "dr.json"
    if not p.exists():
        return
    drawn = {k for k, v in json.loads(p.read_text()).items() if v is not None}
    for g in [x for x in cf.groups + cf.parts if x.restores]:
        hit = _FIELDS.get(g.name, set()) & drawn
        if hit:
            cf.notes.append(
                f"{g.name}: the source's values include its startup randomization draw "
                f"({', '.join(sorted(hit))} in dr.json); the source is one sample of the "
                f"training distribution, not its nominal model"
            )


# -- the counterfactual ---------------------------------------------------------------------------


@dataclass
class GroupResult:
    name: str
    failed: int
    p_vs_target: float
    restores: bool


@dataclass
class Counterfactual:
    scenario: str
    seeds: int
    source: TaskOutcome
    target: TaskOutcome
    p: float
    changes: bool
    groups: list[GroupResult] = field(default_factory=list)
    all_groups: GroupResult | None = None
    parts: list[GroupResult] = field(default_factory=list)
    differing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def localized(self) -> list[str]:
        return [g.name for g in self.groups if g.restores]

    def conclusion(self) -> str:
        if not self.changes:
            return "the outcome does not change with the model"
        if self.localized:
            parts = [g.name.split(": ", 1)[1] for g in self.parts if g.restores]
            return (
                "the outcome follows "
                + ", ".join(self.localized)
                + (f" (within it: {', '.join(parts)})" if parts else "")
            )
        if self.all_groups and self.all_groups.restores:
            return "no single group restores the outcome; all swappable groups together do"
        return (
            "no swappable group restores the outcome: the difference is in what cannot be "
            "swapped (collision geometry, elements in one model only)"
        )

    def lines(self) -> list[str]:
        n = self.seeds
        out = [
            f"Counterfactual ({self.scenario}, {n} seeds, same policy, contract, backend):",
            f"  source model {self.source.model}: misses the bar on {self.source.n_failed}/{n} "
            f"{_kinds(self.source)}",
            f"  target model {self.target.model}: misses the bar on {self.target.n_failed}/{n} "
            f"{_kinds(self.target)}",
            f"  Fisher exact p = {self.p:.3g}: "
            + (
                "outcome changes with the model" if self.changes else f"no change at p < {CF_ALPHA}"
            ),
        ]
        if self.changes:
            out.append(f"  groups that differ: {', '.join(self.differing) or 'none'}")
            for g in self.groups:
                out.append(
                    f"    target + source {g.name}: misses on {g.failed}/{n} (p = {g.p_vs_target:.3g})"
                    + ("  RESTORES" if g.restores else "")
                )
            for g in self.parts:
                out.append(
                    f"    target + source {g.name}: misses on {g.failed}/{n} (p = {g.p_vs_target:.3g})"
                    + ("  RESTORES" if g.restores else "")
                )
            if self.all_groups:
                g = self.all_groups
                out.append(
                    f"    target + all groups: misses on {g.failed}/{n} (p = {g.p_vs_target:.3g})"
                    + ("  RESTORES" if g.restores else "")
                )
            out.append(f"  conclusion: {self.conclusion()}")
        out.extend(f"  note: {x}" for x in self.notes)
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "seeds": self.seeds,
            "source": {
                "model": self.source.model,
                "failed": self.source.n_failed,
                "kinds": self.source.kinds(),
            },
            "target": {
                "model": self.target.model,
                "failed": self.target.n_failed,
                "kinds": self.target.kinds(),
            },
            "p": self.p,
            "changes": self.changes,
            "differing": self.differing,
            "groups": [g.__dict__ for g in self.groups],
            "all_groups": self.all_groups.__dict__ if self.all_groups else None,
            "parts": [g.__dict__ for g in self.parts],
            "localized": self.localized,
            "conclusion": self.conclusion(),
            "notes": self.notes,
        }


def _kinds(o: TaskOutcome) -> str:
    k = o.kinds()
    return "(" + ", ".join(f"{a} {b}" for a, b in sorted(k.items())) + ")" if k else ""


def counterfactual(
    trace: Trace,
    contract: Contract,
    onnx: str,
    target: str,
    source: str | None = None,
    seeds: tuple[int, ...] = SEEDS,
    backend: str | None = None,
    spec: TaskSpec | None = None,
    target_edit: Any = None,
    localize: bool = True,
    workers: int | None = None,
) -> Counterfactual:
    """``source`` defaults to the trace directory (the recorder's model)."""
    source = str(source or trace.path)
    if not source or not (Path(source) / "manifest.json").exists() and not Path(source).is_file():
        raise ValueError("the source model is not recorded with this trace")
    spec = spec or golden_scenario(trace)
    kw = {"backend": backend, "workers": workers}
    src = run_task(contract, source, onnx, spec, seeds, **kw)
    # Segments the source misses in this runner on any seed are its own limitation: not judged.
    segs, dropped = [], []
    for i, sg in enumerate(spec.segment_list()):
        bad = {k for s in src.seeds for k, _ in s.segments[i].problems}
        if bad:
            dropped.append(f"{sg.label()} ({', '.join(sorted(bad))} on the source)")
            sg = Segment(sg.t0, sg.t1, sg.cmd, (False, False, False), False)
        segs.append(sg)
    src = rejudge(src, segs)
    spec = TaskSpec(**{**spec.__dict__, "segments": segs})
    tgt = run_task(contract, target, onnx, spec, seeds, model_edit=target_edit, **kw)
    n = len(seeds)
    p = fisher(tgt.n_failed, n, src.n_failed, n)
    cf = Counterfactual(spec.name, n, src, tgt, p, p < CF_ALPHA)
    if dropped:
        cf.notes.append("not judged, the source misses them in this runner: " + "; ".join(dropped))
    if not cf.changes or not localize:
        return cf
    if tgt.n_failed < src.n_failed:
        cf.notes.append("the target does better than the source; groups are not localized")
        return cf
    dst_model = load_model(target).model
    if target_edit is not None:
        target_edit(dst_model)
    src_model = load_model(source).model
    cf.differing = [g for g in GROUPS if group_differs(g, dst_model, src_model)]
    del dst_model, src_model
    for g in cf.differing:
        edit = partial(swap_edit, source=source, groups=(g,), base_edit=target_edit)
        o = run_task(contract, target, onnx, spec, seeds, model_edit=edit, **kw)
        pg = fisher(o.n_failed, n, tgt.n_failed, n, side="less")
        cf.groups.append(GroupResult(g, o.n_failed, pg, pg < LOC_ALPHA))
    for g in list(cf.localized):
        for part in PARTS.get(g, {}):
            name = f"{g}: {part}"
            edit = partial(swap_edit, source=source, groups=(name,), base_edit=target_edit)
            o = run_task(contract, target, onnx, spec, seeds, model_edit=edit, **kw)
            pg = fisher(o.n_failed, n, tgt.n_failed, n, side="less")
            cf.parts.append(GroupResult(name, o.n_failed, pg, pg < LOC_ALPHA))
    _randomization_note(cf, source)
    if not cf.localized and len(cf.differing) > 1:
        edit = partial(swap_edit, source=source, groups=tuple(cf.differing), base_edit=target_edit)
        o = run_task(contract, target, onnx, spec, seeds, model_edit=edit, **kw)
        pg = fisher(o.n_failed, n, tgt.n_failed, n, side="less")
        cf.all_groups = GroupResult("all", o.n_failed, pg, pg < LOC_ALPHA)
    return cf
