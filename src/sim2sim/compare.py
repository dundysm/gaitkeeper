"""Boundary comparator: B (observation to action), A (state to observation),
C (action to target and effort), against a recorded trace.

Rules (plan sections 4 and 7.3):

* References come from the trace's raw simulator state and the contract,
  never from values a harness derived itself.
* Alignment is checked before values: a one-step shift is a timing error.
* Tolerances come from named classes; they are never widened to pass.
* A mismatch is named only when exactly one of the simplest explanations fits.
  Several equally simple fits are reported as ambiguous (abstention).
* A pass is "separated" only when the trace distinguishes the contract from
  every known alternative. Otherwise the property is reported as not covered,
  and the boundary cannot be verified on this trace (under-excited).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import Contract
from .tables import TABLES
from .terms import (
    RawState,
    StateLayout,
    TermContext,
    apply_clip_scale,
    assemble,
    context_from_contract,
    term_slices,
    term_values,
)
from .trace import Trace

EXACT_REL = 1e-5  # float32 pipeline: about 80 ulp at magnitude 1

TOLERANCE_CLASSES = {
    "exact": "1e-5 x max(1, |value|): same formula on float32 inputs",
    "export_rounding": "exact, plus contract numbers known only to half a unit in their last printed digit, "
    "propagated through the formula",
    "float32_affine": "exact on the largest product in the affine actuator law (kp |ctrl|, kp |q|), since "
    "float32 cancellation scales with it",
}


# -- results -------------------------------------------------------------------------


@dataclass
class TermResult:
    term: str
    boundary: str
    status: str  # pass | fail | not_covered
    max_abs: float
    rms: float
    max_tol: float
    worst_ratio: float  # max |err| / tol
    tol_class: str
    n_rows: int
    held_out_max_abs: float = 0.0
    pattern: str | None = None
    ambiguous: list[str] = field(default_factory=list)
    not_separated: list[str] = field(default_factory=list)
    first_bad_step: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryResult:
    name: str
    status: str  # pass | fail | not_covered | not_checked
    terms: list[TermResult] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Report:
    boundaries: dict[str, BoundaryResult]
    verdict: str
    evidence: str
    label: str | None
    excitation: list[dict[str, Any]]
    findings: list[str]

    def summary(self) -> str:
        lines = [
            f"verdict {self.verdict} | evidence {self.evidence}"
            + (f" | {self.label}" if self.label else "")
        ]
        for b in self.boundaries.values():
            lines.append(
                f"  {b.name}: {b.status}" + (f"  [{'; '.join(b.patterns)}]" if b.patterns else "")
            )
            for n in b.notes:
                lines.append(f"    note: {n}")
        for f in self.findings:
            lines.append(f"  finding: {f}")
        return "\n".join(lines)


# -- candidates ------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    complexity: int  # number of free parameters; structural alternatives count 1
    predict: Callable[[], np.ndarray | None]
    structural: bool = True  # a named alternative usable for the separation test
    fitted: dict[str, Any] = field(default_factory=dict)
    tol: Callable[[np.ndarray], np.ndarray] | None = (
        None  # when the candidate's own inputs set the tolerance
    )


def _within(o: np.ndarray, e: np.ndarray, tol: np.ndarray) -> np.ndarray:
    return np.abs(o - e) <= tol


def _split(n: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n)
    return idx[idx % 2 == 0], idx[idx % 2 == 1]  # fit rows, held-out rows


def _fit_scale(
    o: np.ndarray, e: np.ndarray, rows: np.ndarray, per_column: bool
) -> np.ndarray | None:
    oo, ee = o[rows], e[rows]
    if per_column:
        den = (ee * ee).sum(0)
        if np.any(den < 1e-12):
            return None
        return (oo * ee).sum(0) / den
    den = float((ee * ee).sum())
    return None if den < 1e-12 else np.array(float((oo * ee).sum()) / den)


def _fit_affine(
    o: np.ndarray, e: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    a, b = [], []
    for j in range(e.shape[1]):
        x = e[rows, j]
        if np.ptp(x) < 1e-9:
            return None
        A = np.stack([x, np.ones_like(x)], 1)
        sol, *_ = np.linalg.lstsq(A, o[rows, j], rcond=None)
        a.append(sol[0])
        b.append(sol[1])
    return np.array(a), np.array(b)


def _match_columns(o: np.ndarray, e: np.ndarray, tol: np.ndarray) -> list[int] | None:
    """Permutation p with o[:, i] == e[:, p[i]] within tolerance, if one exists uniquely."""
    d = e.shape[1]
    p = []
    for i in range(d):
        ok = [
            j
            for j in range(d)
            if np.all(np.abs(o[:, i] - e[:, j]) <= np.maximum(tol[:, i], tol[:, j]))
        ]
        if len(ok) != 1:
            return None
        p.append(ok[0])
    return p if sorted(p) == list(range(d)) else None


def _shift(x: np.ndarray, lag: int, reset: np.ndarray | None = None) -> np.ndarray:
    """Row k gets x[k - lag]; rows without a source repeat the first available row."""
    out = np.empty_like(x)
    if lag >= 0:
        out[lag:] = x[: len(x) - lag]
        out[:lag] = x[0]
    else:
        out[:lag] = x[-lag:]
        out[lag:] = x[-1]
    return out


def _known_permutation_name(perm_names: list[str], names: list[str]) -> str | None:
    for tname, table in TABLES.items():
        if set(table) != set(names) or len(table) != len(names):
            continue
        if perm_names == table:
            return f"values in {tname} order written into policy-order slots (remap skipped)"
        pos = {n: i for i, n in enumerate(table)}
        if perm_names == [names[pos[n]] for n in names]:
            return f"policy-order values sent in {tname} order (remap skipped on output)"
    return None


def _classify(
    o: np.ndarray,
    e: np.ndarray,
    tol_fn: Callable[[np.ndarray], np.ndarray],
    candidates: list[Candidate],
) -> tuple[str | None, list[str], dict[str, Any]]:
    """Return (named pattern, ambiguous list, fitted values) for a failing column block."""
    fits = []
    for c in candidates:
        pred = c.predict()
        if pred is None:
            continue
        if np.all(_within(o, pred, (c.tol or tol_fn)(pred))):
            fits.append(c)
    if not fits:
        return "none fits", [], {}
    best = min(c.complexity for c in fits)
    top = [c for c in fits if c.complexity == best]
    names = sorted({c.name for c in top})
    if len(names) == 1:
        return names[0], [], top[0].fitted
    return None, names, {}


def _generic_candidates(
    o: np.ndarray, e: np.ndarray, fit_rows: np.ndarray, lagged: Callable[[int], np.ndarray]
) -> list[Candidate]:
    d = e.shape[1]
    cands: list[Candidate] = []
    s = _fit_scale(o, e, fit_rows, per_column=False)
    if s is not None:
        sv = float(s)
        name = "sign flip" if abs(sv + 1) < 1e-3 else f"constant scale x{sv:.4g}"
        cands.append(
            Candidate(name, 1, lambda sv=sv: e * sv, structural=False, fitted={"scale": sv})
        )
    sc = _fit_scale(o, e, fit_rows, per_column=True)
    if sc is not None and d > 1:
        bad = np.where(np.abs(sc - 1) > 1e-4)[0]
        cands.append(
            Candidate(
                f"per-column scale on columns {bad.tolist()}",
                max(len(bad), 1) + 1,
                lambda sc=sc: e * sc[None],
                structural=False,
                fitted={"scale": sc.tolist(), "columns": bad.tolist()},
            )
        )
    off = (o[fit_rows] - e[fit_rows]).mean(0)
    bad = np.where(np.abs(off) > 1e-6)[0]
    cands.append(
        Candidate(
            f"constant offset on columns {bad.tolist()}",
            max(len(bad), 1) + 1,
            lambda off=off: e + off[None],
            structural=False,
            fitted={"offset": off.tolist(), "columns": bad.tolist()},
        )
    )
    aff = _fit_affine(o, e, fit_rows)
    if aff is not None:
        a, b = aff
        cands.append(
            Candidate(
                "per-column affine",
                2 * d,
                lambda a=a, b=b: e * a[None] + b[None],
                structural=False,
                fitted={"scale": a.tolist(), "offset": b.tolist()},
            )
        )
    cmax = float(np.abs(o).max())
    if cmax > 0 and np.abs(e).max() > cmax * (1 + 1e-4):
        cands.append(
            Candidate(
                f"clipped at +-{cmax:.4g}",
                1,
                lambda c=cmax: np.clip(e, -c, c),
                structural=False,
                fitted={"clip": cmax},
            )
        )
    cands.append(Candidate("zero (term not filled)", 1, lambda: np.zeros_like(e)))
    for lag in (1, -1):
        cands.append(Candidate(f"term shifted by {lag} step", 1, lambda lag=lag: lagged(lag)))
    return cands


# -- boundary B ------------------------------------------------------------------------------


def check_b(trace: Trace, policy: Any) -> BoundaryResult:
    if policy is None:
        return BoundaryResult("B", "not_checked", notes=["no policy file given"])
    obs = trace["obs"]
    if obs.shape[1] != policy.n_in:
        return BoundaryResult(
            "B", "not_checked", notes=[f"policy takes {policy.n_in} inputs, log has {obs.shape[1]}"]
        )
    a = policy(obs)
    rec = trace["action"].astype(np.float64)
    err = np.abs(a - rec)
    tol = EXACT_REL * np.maximum(1.0, np.abs(rec))
    ok = err <= tol
    _, held = _split(len(rec))
    tr = TermResult(
        "action",
        "B",
        "pass" if ok.all() else "fail",
        float(err.max()),
        float(np.sqrt((err**2).mean())),
        float(tol.max()),
        float((err / tol).max()),
        "exact",
        len(rec),
        float(err[held].max()),
    )
    if not ok.all():
        tr.first_bad_step = int(np.where(~ok.all(1))[0][0])
        tr.pattern = (
            "policy output differs on recorded inputs (wrong file, missing normalizer or state)"
        )
    return BoundaryResult("B", tr.status, [tr])


# -- boundary A ------------------------------------------------------------------------------------


def raw_state(trace: Trace) -> RawState:
    layout = StateLayout.from_meta(trace.meta)
    ep = trace.get("episode_step")
    if ep is None:
        ep = np.zeros(trace.n_steps, int)
        c = 0
        for k, r in enumerate(trace["reset"]):
            c = 0 if r else c + 1
            ep[k] = c
    return RawState.from_arrays(
        trace["qpos"], trace["qvel"], layout, trace["command"], ep, trace["reset"], trace["action"]
    )


def _term_tol(term: dict[str, Any], contract: Contract, e: np.ndarray) -> np.ndarray:
    tol = EXACT_REL * np.maximum(1.0, np.abs(e))
    if term["id"] == "joint_pos_rel":
        tol = tol + contract.resolution("control.default_joint_pos")
    return tol


def _alt_obs(
    state: RawState,
    contract: Contract,
    terms: list[dict[str, Any]],
    history: dict[str, Any],
    ctx: TermContext,
) -> np.ndarray:
    values = term_values(state, terms, ctx)
    obs, _ = assemble(values, terms, state.reset, history)
    return obs


def check_a(trace: Trace, contract: Contract) -> BoundaryResult:
    group = contract.get("policy_io.observation_groups.policy")
    terms, history = group["terms"], dict(group.get("history", {}))
    ctx = context_from_contract(contract)
    s = raw_state(trace)
    values = term_values(s, terms, ctx)
    E, _ = assemble(values, terms, s.reset, history)
    O = trace["obs"].astype(np.float64)  # noqa: E741
    if O.shape != E.shape:
        return BoundaryResult(
            "A", "fail", patterns=[f"observation size {O.shape[1]}, contract builds {E.shape[1]}"]
        )
    cols = term_slices(terms, history)

    def tol_for(h: dict[str, Any], vals: dict[str, np.ndarray] = values) -> np.ndarray:
        # Tolerances follow their values through the history layout.
        tv = {t["id"]: _term_tol(t, contract, vals[t["id"]]) for t in terms}
        return assemble(tv, terms, s.reset, h)[0]

    tol_full = tol_for(history)
    fit_rows, held = _split(len(O))
    res = BoundaryResult("A", "pass")

    def term_ok(Ex: np.ndarray, tid: str, tol: np.ndarray = tol_full) -> bool:
        c = cols[tid]
        return bool(np.all(np.abs(O[:, c] - Ex[:, c]) <= tol[:, c]))

    failing = [t["id"] for t in terms if not term_ok(E, t["id"])]

    # Group-level explanations first: timing, then history structure.
    group_cands: list[tuple[str, np.ndarray, np.ndarray]] = []
    for lag in (1, -1, 2):
        group_cands.append(
            (
                f"observation built {abs(lag)} step {'late' if lag > 0 else 'early'} "
                f"(timing, not a term bug)",
                _shift(E, lag),
                _shift(tol_full, lag),
            )
        )
    lag_state = _state(
        s,
        **{
            k: _shift(getattr(s, k), 1)
            for k in ("root_quat", "ang_vel_body", "joint_pos", "joint_vel")
        },
    )
    group_cands.append(
        (
            "simulator state read 1 step late, command and last action current (timing)",
            _alt_obs(lag_state, contract, terms, history, ctx),
            tol_full,
        )
    )
    L = int(history.get("length", 1))
    if L > 1:
        for key, alt in (
            ("layout", {"term_major": "time_major", "time_major": "term_major"}),
            ("order", {"oldest_first": "newest_first", "newest_first": "oldest_first"}),
            ("init", {"repeat_first": "zeros", "zeros": "repeat_first"}),
        ):
            h2 = dict(history)
            h2[key] = alt[history.get(key, list(alt)[0])]
            group_cands.append(
                (
                    f"history {key} is {h2[key]} (contract: {history.get(key)})",
                    _alt_obs(s, contract, terms, h2, ctx),
                    tol_for(h2),
                )
            )
    group_hit = None
    if failing:
        hits = [
            name for name, Ex, tl in group_cands if all(term_ok(Ex, t["id"], tl) for t in terms)
        ]
        if len(hits) == 1:
            group_hit = hits[0]
        elif len(hits) > 1:
            res.status = "fail"
            res.patterns.append(f"ambiguous: {hits}")

    for t in terms:
        tid = t["id"]
        c = cols[tid]
        o, e, tol = O[:, c], E[:, c], tol_full[:, c]
        err = np.abs(o - e)
        tr = TermResult(
            tid,
            "A",
            "pass",
            float(err.max()),
            float(np.sqrt((err**2).mean())),
            float(tol.max()),
            float((err / tol).max()),
            "export_rounding" if tid == "joint_pos_rel" else "exact",
            len(o),
            float(err[held].max()),
        )
        alts = _term_alternatives(tid, t, s, contract, ctx, terms, history, cols)
        if tid in failing:
            tr.status = "fail"
            bad = ~(err <= tol).all(1)
            tr.first_bad_step = int(np.where(bad)[0][0])
            if group_hit:
                tr.pattern = group_hit
            else:

                def lagged(lag: int, e=e) -> np.ndarray:
                    return _shift(e, lag)

                cands = [Candidate(n, 1, (lambda p=p: p)) for n, p in alts] + _generic_candidates(
                    o, e, fit_rows, lagged
                )
                if tid in ("joint_pos_rel", "joint_vel_rel", "last_action"):
                    per_slot = o.shape[1] == len(ctx.joint_names)
                    base = (
                        ctx.default_joint_pos[None]
                        if (tid == "joint_pos_rel" and per_slot)
                        else 0.0
                    )
                    for pc in _permutation_candidates(
                        o + base, e + base, tol, ctx.joint_names, per_slot_dim=len(ctx.joint_names)
                    ):
                        f = pc.predict
                        pc.predict = lambda f=f, base=base: f() - base
                        cands.append(pc)

                def tfn(pred: np.ndarray, t=t) -> np.ndarray:
                    return _term_tol(t, contract, pred)

                tr.pattern, tr.ambiguous, tr.detail = _classify(o, e, tfn, cands)
        else:
            for name, pred in alts:
                if np.all(np.abs(o - pred) <= _term_tol(t, contract, pred)):
                    tr.not_separated.append(name)
            if tr.not_separated:
                tr.status = "not_covered"
        res.terms.append(tr)

    if group_hit:
        res.status = "fail"
        res.patterns.append(group_hit)
    elif any(tr.status == "fail" for tr in res.terms):
        res.status = "fail"
        for tr in res.terms:
            if tr.status == "fail":
                res.patterns.append(f"{tr.term}: {tr.pattern or 'ambiguous ' + str(tr.ambiguous)}")
    elif any(tr.status == "not_covered" for tr in res.terms):
        res.status = "not_covered"
        for tr in res.terms:
            if tr.not_separated:
                res.notes.append(
                    f"{tr.term}: trace cannot separate the contract from {tr.not_separated}"
                )
    return res


def _term_alternatives(
    tid: str,
    term: dict[str, Any],
    s: RawState,
    contract: Contract,
    ctx: TermContext,
    terms: list[dict[str, Any]],
    history: dict[str, Any],
    cols: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Named alternative implementations a harness might have used, as full term columns."""
    alts: list[tuple[str, dict[str, Any] | None, TermContext | None, RawState | None]] = []
    if tid == "base_ang_vel":
        alts.append(
            ("angular velocity in the world frame", None, _ctx(ctx, imu_frame="world"), None)
        )
    if tid == "projected_gravity":
        q = s.root_quat
        s2 = _state(s, root_quat=q[:, [1, 2, 3, 0]])  # (x, y, z, w) read as (w, x, y, z)
        alts.append(("quaternion read as xyzw", None, None, s2))
    if tid == "gait_phase":
        p = dict(term.get("params", {}))
        p["stand_threshold"] = None
        alts.append(("phase without the stand rule", {**term, "params": p}, None, None))
        s2 = _state(s, episode_step=s.episode_step + 1)
        alts.append(("phase clock one step ahead", None, None, s2))
    if tid == "last_action":
        if s.action is not None:
            prev = np.zeros_like(s.prev_action)
            prev[1:] = s.action[:-1]
            alts.append(
                ("last action not zeroed at reset", None, None, _state(s, prev_action=prev))
            )
        sc = contract.get("control.actions.joint_pos.scale")
        off = contract.get("control.actions.joint_pos.offset")
        if isinstance(sc, dict) and isinstance(off, dict):
            scv = np.array([sc[n] for n in ctx.joint_names])
            offv = np.array([off[n] for n in ctx.joint_names])
            alts.append(
                (
                    "processed action instead of raw",
                    None,
                    None,
                    _state(
                        s, prev_action=np.where(s.reset[:, None], 0.0, s.prev_action * scv + offv)
                    ),
                )
            )
    if tid == "velocity_commands":
        alts.append(
            ("command not fed (zeros)", None, None, _state(s, command=np.zeros_like(s.command)))
        )
    out = []
    for name, t2, c2, s2 in alts:
        tt = t2 or term
        vals = {
            tt["id"]: apply_clip_scale(
                _TERM_FN(tt["id"])(s2 or s, tt.get("params", {}), c2 or ctx), tt
            )
        }
        win, _ = assemble(vals, [tt], (s2 or s).reset, history)
        out.append((name, win))
    for tname, table in TABLES.items():
        if (
            tid in ("joint_pos_rel", "joint_vel_rel", "last_action")
            and set(table) == set(ctx.joint_names)
            and table != ctx.joint_names
        ):
            idx = [ctx.joint_names.index(n) for n in table]
            if tid == "joint_pos_rel":
                # A skipped remap permutes measured positions; the default is still subtracted per slot.
                ji = s.joint_index(ctx.joint_names)
                s2 = _state(s, joint_pos=s.joint_pos.copy())
                s2.joint_pos[:, ji] = s.joint_pos[:, ji][:, idx]
                vals = term_values(s2, [term], ctx)[tid]
            else:
                vals = term_values(s, [term], ctx)[tid][:, idx]
            win, _ = assemble({tid: vals}, [term], s.reset, history)
            out.append(
                (
                    f"joint permutation: values in {tname} order written into policy-order slots "
                    f"(remap skipped)",
                    win,
                )
            )
    return out


def _TERM_FN(tid: str):
    from .terms import TERMS

    return TERMS[tid]


def _ctx(ctx: TermContext, **kw: Any) -> TermContext:
    d = dict(ctx.__dict__)
    d.update(kw)
    return TermContext(**d)


def _state(s: RawState, **kw: Any) -> RawState:
    d = dict(s.__dict__)
    d.update(kw)
    return RawState(**d)


def _permutation_candidates(
    o: np.ndarray, e: np.ndarray, tol: np.ndarray, names: list[str], per_slot_dim: int
) -> list[Candidate]:
    if o.shape[1] != per_slot_dim:
        return []  # history-stacked joint terms: permutation search per slot is not implemented
    p = _match_columns(o, e, tol)
    if p is None or p == list(range(len(p))):
        return []
    perm_names = [names[j] for j in p]
    known = _known_permutation_name(perm_names, names)
    moved = [names[i] for i in range(len(p)) if p[i] != i]
    if known:
        return [
            Candidate(
                f"joint permutation: {known}", 1, lambda p=p: e[:, p], fitted={"slots": perm_names}
            )
        ]
    return [
        Candidate(
            f"joint permutation on {moved}",
            len(moved),
            lambda p=p: e[:, p],
            structural=False,
            fitted={"slots": perm_names},
        )
    ]


# -- boundary C -----------------------------------------------------------------------------------------


def _vec(contract: Contract, path: str, names: list[str]) -> np.ndarray | None:
    v = contract.get(path, None)
    if not isinstance(v, dict):
        return None
    try:
        return np.array([float(v[n]) for n in names])
    except (KeyError, TypeError):
        return None


def check_c(trace: Trace, contract: Contract) -> BoundaryResult:
    res = BoundaryResult("C", "pass")
    tgt_unc = 0.0
    pnames = list(contract.get("policy_io.joints.names"))
    scale = _vec(contract, "control.actions.joint_pos.scale", pnames)
    offset = _vec(contract, "control.actions.joint_pos.offset", pnames)
    if offset is None:
        offset = _vec(contract, "control.default_joint_pos", pnames)
    if scale is None or offset is None:
        return BoundaryResult("C", "not_checked", notes=["contract lacks action scale or offset"])
    rs, ro = (
        contract.resolution("control.actions.joint_pos.scale"),
        contract.resolution("control.actions.joint_pos.offset"),
    )
    raw = trace["action"].astype(np.float64)
    # Which target channel: physics-rate ctrl for golden traces, control-rate target for harness logs.
    if trace.physics("ctrl") is not None:
        rows_step = trace.physics("step").astype(int)
        tgt = trace.physics("ctrl").astype(np.float64)
        tnames = list(trace.meta["target_joint_names"])
        q = trace.physics("qpos").astype(np.float64)[:, 7:]
        qd = trace.physics("qvel").astype(np.float64)[:, 6:]
        eff = trace.physics("effort")
        rate = "physics"
    elif trace.get("target") is not None:
        rows_step = np.arange(trace.n_steps)
        tgt = trace["target"].astype(np.float64)
        tnames = list(trace.meta.get("target_joint_names", pnames))
        q = trace["qpos"].astype(np.float64)[:, 7:]
        qd = trace["qvel"].astype(np.float64)[:, 6:]
        eff = trace.get("effort")
        rate = "control"
    else:
        tgt = None
        rows_step = np.arange(trace.n_steps)
        tnames = list(trace.meta.get("effort_joint_names", pnames))
        q = trace["qpos"].astype(np.float64)[:, 7:]
        qd = trace["qvel"].astype(np.float64)[:, 6:]
        eff = trace.get("effort")
        rate = "control"
    layout = StateLayout.from_meta(trace.meta)
    qidx = [layout.joint_names.index(n) for n in tnames]
    q, qd = q[:, qidx], qd[:, qidx]
    col = [pnames.index(n) for n in tnames]  # target column j holds policy joint col[j]
    sc, of = scale[col], offset[col]
    raw_rows = raw[rows_step][:, col]

    def expected(r: np.ndarray) -> np.ndarray:
        return r * sc[None] + of[None]

    def tol_t(r: np.ndarray, e: np.ndarray) -> np.ndarray:
        return EXACT_REL * np.maximum(1.0, np.abs(e)) + np.abs(r) * rs + ro

    fit_rows, held = _split(len(rows_step))
    if tgt is not None:
        E = expected(raw_rows)
        tol = tol_t(raw_rows, E)
        err = np.abs(tgt - E)
        ok = err <= tol
        tr = TermResult(
            "target",
            "C",
            "pass" if ok.all() else "fail",
            float(err.max()),
            float(np.sqrt((err**2).mean())),
            float(tol.max()),
            float((err / tol).max()),
            "export_rounding" if (rs or ro) else "exact",
            len(E),
            float(err[held].max()),
        )
        tr.detail["rate"] = rate
        tr.detail["per_joint_max_abs"] = dict(zip(tnames, err.max(0).tolist()))
        if not ok.all():
            tr.first_bad_step = int(rows_step[np.where(~ok.all(1))[0][0]])

            def lag_raw(lag: int) -> np.ndarray:
                return _shift(raw, lag)[rows_step][:, col]

            cands = []
            for lag in (1, 2):
                rl = lag_raw(lag)
                cands.append(
                    Candidate(
                        f"action applied {lag} policy step late (delay)",
                        1,
                        lambda rl=rl: expected(rl),
                        tol=lambda e, rl=rl: tol_t(rl, e),
                    )
                )
            cands.append(Candidate("default offset not added", 1, lambda: raw_rows * sc[None]))
            cands.append(
                Candidate("target = raw action (no scale, no offset)", 1, lambda: raw_rows.copy())
            )
            # Scale or offset from the other source, when the contract recorded one.
            alt = contract.prov("control.actions.joint_pos.scale").alternatives or {}
            for src, vals in alt.items():
                v = np.array(vals, dtype=np.float64)[col]
                cands.append(
                    Candidate(
                        f"action scale from {src}", 1, lambda v=v: raw_rows * v[None] + of[None]
                    )
                )
            # Raw action clipped before scaling (wrapper clip).
            c = _clip_estimate(tgt, raw_rows, sc, of, rs)
            if c is not None:
                rc = np.clip(raw_rows, -c, c)
                cands.append(
                    Candidate(
                        f"raw action clipped at +-{c:.4g} before scaling",
                        1,
                        lambda rc=rc: expected(rc),
                        fitted={"clip": c},
                        tol=lambda e, rc=rc: tol_t(rc, e),
                    )
                )
            p = _match_columns(tgt, E, tol)
            if p is not None and p != list(range(len(p))):
                perm_names = [tnames[j] for j in p]
                known = _known_permutation_name(perm_names, tnames)
                cands.append(
                    Candidate(
                        f"joint permutation: {known}"
                        if known
                        else f"joint permutation on {[tnames[i] for i in range(len(p)) if p[i] != i]}",
                        1 if known else len(p),
                        lambda p=p: E[:, p],
                        tol=lambda e, p=p: tol_t(raw_rows[:, p], e),
                    )
                )
            sc_fit = _fit_scale(tgt - of[None], raw_rows * sc[None], fit_rows, per_column=True)
            if sc_fit is not None:
                badj = [tnames[j] for j in np.where(np.abs(sc_fit - 1) > 1e-3)[0]]
                cands.append(
                    Candidate(
                        f"action scale differs on {badj}",
                        len(badj) + 1,
                        lambda f=sc_fit: raw_rows * (sc * f)[None] + of[None],
                        structural=False,
                        fitted={"ratio": dict(zip(tnames, sc_fit.tolist()))},
                    )
                )
            tr.pattern, tr.ambiguous, tr.detail["fit"] = _classify(
                tgt, E, lambda e: tol_t(raw_rows, e), cands
            )
        res.terms.append(tr)
        tgt_for_pd = tgt
    else:
        tgt_for_pd = expected(raw_rows)
        tgt_unc = np.abs(raw_rows) * rs + ro  # contract rounding carried into the computed target
        res.notes.append(
            "no target channel; effort checked against targets computed from the contract"
        )

    # PD law, on the recorded target (separates the mapping from the actuator law).
    if eff is not None:
        eff = eff.astype(np.float64)
        kp = _vec(contract, "control.actuators.kp", tnames)
        kd = _vec(contract, "control.actuators.kd", tnames)
        if kp is None or kd is None:
            res.notes.append("contract has no gains; effort not checked")
        else:
            rkp, rkd = (
                contract.resolution("control.actuators.kp"),
                contract.resolution("control.actuators.kd"),
            )
            e_q = tgt_for_pd - q
            pred = kp[None] * e_q - kd[None] * qd
            tol = (
                EXACT_REL
                * np.maximum.reduce(
                    [
                        np.ones_like(pred),
                        np.abs(pred),
                        kp[None] * np.abs(tgt_for_pd),
                        kp[None] * np.abs(q),
                    ]
                )
                + np.abs(e_q) * rkp
                + np.abs(qd) * rkd
                + kp[None] * tgt_unc
            )
            lim = _vec(contract, "model.effort_limit", tnames)
            sat = np.zeros_like(pred, dtype=bool)
            inferred_limit: dict[str, float] = {}
            if lim is not None:
                sat = np.abs(pred) > lim[None]
                pred = np.clip(pred, -lim[None], lim[None])
            else:
                # Limits unknown: a joint whose effort sits at a constant magnitude below the
                # prediction is saturating; infer the limit and exclude those rows.
                for j in range(pred.shape[1]):
                    m = float(np.abs(eff[:, j]).max())
                    s_j = (np.abs(pred[:, j]) > m + tol[:, j]) & (
                        np.abs(np.abs(eff[:, j]) - m) <= 1e-4 * max(1, m)
                    )
                    if s_j.any():
                        sat[:, j] = s_j
                        inferred_limit[tnames[j]] = m
            err = np.abs(eff - pred)
            ok = (err <= tol) | (sat & (lim is None))
            tr = TermResult(
                "effort",
                "C",
                "pass" if ok.all() else "fail",
                float(err[~sat].max(initial=0)),
                float(np.sqrt((err[~sat] ** 2).mean())),
                float(tol.max()),
                float((err / tol)[~sat].max(initial=0)),
                "float32_affine" + ("+export_rounding" if (rkp or rkd) else ""),
                len(pred),
                float(err[held][~sat[held]].max(initial=0)),
            )
            tr.detail["rows_at_limit"] = int(sat.any(1).sum())
            tr.detail["joint_steps_at_limit"] = int(sat.sum())
            if inferred_limit:
                tr.detail["inferred_limits"] = inferred_limit
            if not ok.all():
                tr.first_bad_step = int(np.where(~ok.all(1))[0][0])
                tr.pattern, tr.ambiguous, tr.detail["fit"] = _classify_gains(
                    eff, e_q, qd, kp, kd, tnames, sat, tol
                )
            res.terms.append(tr)
    if any(t.status == "fail" for t in res.terms):
        res.status = "fail"
        res.patterns = [
            f"{t.term}: {t.pattern or 'ambiguous ' + str(t.ambiguous)}"
            for t in res.terms
            if t.status == "fail"
        ]
    return res


def _clip_estimate(
    tgt: np.ndarray, raw: np.ndarray, sc: np.ndarray, of: np.ndarray, rs: float
) -> float | None:
    """Clip level when targets plateau where raw actions exceed it, else None.

    Rows whose recorded target sits clearly inside the raw action's reach are
    clipped; their effective raw value estimates the clip level. A level within
    1 percent of a two-significant-digit number is taken as that number, since
    the effective value carries the contract's scale rounding.
    """
    r_eff = (tgt - of[None]) / sc[None]
    margin = np.abs(raw) * (rs / np.maximum(sc[None], 1e-12)) + 1e-4
    clipped = np.abs(r_eff) < np.abs(raw) - 2 * margin - 1e-3
    if clipped.sum() < 3:
        return None
    c = float(np.median(np.abs(r_eff[clipped])))
    nice = float(f"{c:.2g}")
    return nice if abs(nice - c) <= 0.01 * c else c


def _classify_gains(
    eff: np.ndarray,
    e_q: np.ndarray,
    qd: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    names: list[str],
    sat: np.ndarray,
    tol: np.ndarray,
) -> tuple[str | None, list[str], dict[str, Any]]:
    d = len(names)
    kp_hat, kd_hat = np.zeros(d), np.zeros(d)
    for j in range(d):
        use = ~sat[:, j]
        A = np.stack([e_q[use, j], -qd[use, j]], 1)
        sol, *_ = np.linalg.lstsq(A, eff[use, j], rcond=None)
        kp_hat[j], kd_hat[j] = sol
    fitted = {
        "kp_hat": dict(zip(names, np.round(kp_hat, 4).tolist())),
        "kd_hat": dict(zip(names, np.round(kd_hat, 4).tolist())),
    }
    rel_kp = np.abs(kp_hat - kp) / np.maximum(kp, 1e-9)
    bad = [
        names[j]
        for j in range(d)
        if rel_kp[j] > 1e-2 or abs(kd_hat[j] - kd[j]) > 1e-2 * max(kd[j], 1e-3)
    ]
    if not bad:
        return "none fits (gains match; check targets and state alignment)", [], fitted
    # Do the fitted gains equal the contract's under a permutation?
    p = []
    for j in range(d):
        m = [
            i
            for i in range(d)
            if abs(kp_hat[j] - kp[i]) <= 1e-2 * kp[i]
            and abs(kd_hat[j] - kd[i]) <= 1e-2 * max(kd[i], 1e-3)
        ]
        p.append(m)
    if all(len(m) >= 1 for m in p):
        # Gains are a rearrangement of the contract's values. Name the table if one explains it.
        for tname, table in TABLES.items():
            if set(table) != set(names):
                continue
            # Gains listed in table order, applied in log (joint) order.
            g_kp = np.array([kp[names.index(n)] for n in table])
            # 1 percent: gains are regressed on targets known only to export rounding.
            if np.allclose(g_kp, kp_hat, rtol=1e-2):
                return f"gains bound by index in {tname} order instead of by name", [], fitted
        return f"gains bound to the wrong joints on {bad}", [], fitted
    ratio = kp_hat / kp
    if np.allclose(ratio, ratio[0], rtol=1e-2):
        return f"gains scaled x{ratio[0]:.4g}", [], fitted
    return f"gains differ on {bad}", [], fitted


# -- excitation, verdict --------------------------------------------------------------------------------


def excitation(trace: Trace, contract: Contract) -> list[dict[str, Any]]:
    from .recorders.schedule import excitation_checklist

    s = raw_state(trace)
    eff = trace.physics("effort")
    share = None
    lim = contract.get("model.effort_limit", None)
    if eff is not None and isinstance(lim, dict):
        names = trace.meta["effort_joint_names"]
        lv = np.array([lim.get(n) or np.inf for n in names])
        share = float(np.mean(np.abs(eff) >= lv[None] - 1e-4))
    thr = 0.1
    for t in contract.get("policy_io.observation_groups.policy.terms", []):
        if t["id"] == "gait_phase" and t.get("params", {}).get("stand_threshold") is not None:
            thr = float(t["params"]["stand_threshold"])
    return [
        e.__dict__
        for e in excitation_checklist(
            s.command, s.root_quat, s.ang_vel_body, s.reset, share, stand_threshold=thr
        )
    ]


def verify(trace: Trace, contract: Contract, policy: Any = None) -> Report:
    problems = trace.validate()
    if problems:
        return Report({}, "INVALID_INPUT", "L0", None, [], problems)
    b = check_b(trace, policy)
    a = check_a(trace, contract)
    c = check_c(trace, contract)
    exc = excitation(trace, contract)
    findings: list[str] = []
    label = "SELF_CONSISTENT" if trace.self_consistent_only else None
    statuses = [x.status for x in (b, a, c)]
    if "fail" in statuses:
        verdict = "CONTRACT"
    elif "not_covered" in statuses:
        verdict = "UNDETERMINED"
        findings.append("under-excited: some contract properties are not separated by this trace")
    else:
        verdict = "PASS"
    under = [e["name"] for e in exc if not e["ok"]]
    if under:
        findings.append(f"excitation items missing: {under}")
    unknown = [
        p
        for p in (
            "control.actuators.kind",
            "control.actuators.pd_period",
            "control.actuators.integrator",
        )
        if contract.prov(p).source in ("unknown", "default")
    ]
    if unknown:
        findings.append(f"CONTROLLER_ASSUMED: {unknown} not stated by the contract")
    if label:
        evidence = "L1"
    elif verdict == "PASS" and trace.kind == "golden" and b.status == "pass":
        evidence = "L2"
    else:
        evidence = "L1" if verdict != "INVALID_INPUT" else "L0"
    return Report({"B": b, "A": a, "C": c}, verdict, evidence, label, exc, findings)
