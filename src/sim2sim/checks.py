"""Static and linearized checks of a contract against a target MJCF (plan section 7.7).

Each check returns a ``CheckResult`` with a status (PASS, WARN, FAIL, INFO,
SKIP) and the lines a report prints.

* S16 left/right symmetry of default pose and gains, bound by name.
* S17a scalar PD margin per joint, 2 kd h / I + kp h^2 / I with
  I = 1 / (M^-1)_jj. A semi-implicit Euler PD on one joint is stable iff the
  margin is below 4. Applies to backends that compute torque outside the
  solver (explicit_zoh, python_pd); warns above 4.
* S17b eigenvalues of the one-step closed-loop map (finite-difference Jacobian
  of one step, contacts held, frictionloss zeroed). One slow positive real mode
  is the robot tipping over and is reported as a fall time constant; any other
  mode outside the unit circle (negative, complex, or real growing faster than
  0.1 s) fails.
* S18 scenario commands against trained ranges (limit ranges when no trained
  range is known) and measured dead zones.
* S19 torque limit differences, contract against MJCF, latent or active from
  the torque a walk actually needs.
* S21 provenance summary and every pair of sources that disagree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .contract import Contract
from .runner import Binding, RunConfig, Runner

MARGIN_BOUND = 4.0
SLOW_MODE_TC = 0.1  # s; real positive modes growing slower than this are tipping
DRIFT_TC = 2.0  # s; slower real modes are drift within finite-difference noise (no restoring force)
MODE_TOL = 1e-4  # per step, above which |lambda| counts as outside the unit circle
SYM_TOL = 1e-6


@dataclass
class CheckResult:
    id: str
    status: str
    lines: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    def text(self) -> str:
        head = f"{self.id:<4} {self.status:<5} {self.lines[0] if self.lines else ''}"
        return "\n".join([head] + [f"           {ln}" for ln in self.lines[1:]])


# -- S16 ---------------------------------------------------------------------------------

MIRRORED = ("_roll_", "_yaw_")


def s16_symmetry(c: Contract) -> CheckResult:
    names = list(c.get("policy_io.joints.names"))
    default = c.get("control.default_joint_pos")
    kp = c.get("control.actuators.kp")
    kd = c.get("control.actuators.kd")
    pairs = [
        (n, "right_" + n[5:]) for n in names if n.startswith("left_") and "right_" + n[5:] in names
    ]
    bad: dict[str, list[str]] = {"default": [], "kp": [], "kd": []}
    for left, right in pairs:
        sign = -1.0 if any(t in left for t in MIRRORED) else 1.0
        if abs(default[left] - sign * default[right]) > SYM_TOL:
            bad["default"].append(f"{left[5:]} {default[left]:+.4g} vs {default[right]:+.4g}")
        for key, g in (("kp", kp), ("kd", kd)):
            if abs(g[left] - g[right]) > SYM_TOL * max(1.0, abs(g[left])):
                bad[key].append(f"{left[5:]} {g[left]:.4g} vs {g[right]:.4g}")
    n_bad = {k: len(v) for k, v in bad.items()}
    status = "PASS" if not any(n_bad.values()) else "FAIL"
    lines = [
        f"{len(pairs)} left/right pairs bound by name; mismatched pairs: default {n_bad['default']}, "
        f"kp {n_bad['kp']}, kd {n_bad['kd']} (roll and yaw mirrored in sign)"
    ]
    for k, v in bad.items():
        for item in v:
            lines.append(f"{k}: {item}")
    if status == "FAIL":
        lines.append(
            "an asymmetric default or gain set usually means a list was read in the wrong joint order"
        )
    return CheckResult("S16", status, lines, {"mismatched": n_bad, "pairs": len(pairs)})


# -- S17a ---------------------------------------------------------------------------------


def joint_inertia(m: mujoco.MjModel, d: mujoco.MjData, dadr: np.ndarray) -> np.ndarray:
    """Effective inertia 1 / (M^-1)_jj at the current state."""
    mujoco.mj_forward(m, d)
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, full)
    minv = np.linalg.inv(full)
    return 1.0 / np.diag(minv)[dadr]


def pd_margin(kp: np.ndarray, kd: np.ndarray, inertia: np.ndarray, h: float) -> np.ndarray:
    return 2.0 * kd * h / inertia + kp * h * h / inertia


def sample_configs(
    r: Runner, policy_walk: bool = True, seconds: float = 4.0
) -> list[tuple[str, np.ndarray]]:
    """Standing at the default pose, then snapshots of a native walk at 0.5 m/s."""
    m, d, b = r.build(RunConfig(), "native_implicit")
    r.reset_state(m, d, b, RunConfig(), r.default, np.random.default_rng(0))
    out = [("default pose", d.qpos.copy())]
    if policy_walk and r.policy is not None:
        res = r.run(RunConfig(seconds=seconds, command=(0.5, 0.0, 0.0), record=True))
        q = res.log["qpos"]
        k = len(q)
        for frac in (0.25, 0.5, 0.75, 1.0):
            i = min(k - 1, int(frac * k) - 1)
            out.append((f"walk t={i * r.policy_dt:.2f} s", q[i].copy()))
    return out


def s17a_margin(
    r: Runner, backend: str, configs: list[tuple[str, np.ndarray]] | None = None
) -> CheckResult:
    m, d, b = r.build(RunConfig(), "python_pd")
    configs = configs or sample_configs(r, policy_walk=False)
    hs = {"python_pd": b.timestep}
    if r.sim_dt is not None:
        hs["explicit_zoh"] = r.sim_dt
    inertias = []
    for _, q in configs:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        inertias.append(joint_inertia(m, d, b.dadr))
    inertia = np.min(np.array(inertias), axis=0)  # the worst case over configs
    rows = {}
    for bk, h in hs.items():
        rows[bk] = pd_margin(b.kp, b.kd, inertia, h)
    applies = backend in ("explicit_zoh", "python_pd")
    lines = []
    data = {"inertia_min": inertia, "configs": [c[0] for c in configs], "margins": rows, "h": hs}
    worst = []
    for bk, mg in rows.items():
        order = np.argsort(-mg)[:3]
        top = ", ".join(f"{b.names[i]} {mg[i]:.2f}" for i in order)
        n_over = int((mg >= MARGIN_BOUND).sum())
        lines.append(
            f"{bk} (h {hs[bk] * 1e3:.4g} ms): highest {top}; joints at or above 4: {n_over}"
        )
        worst.append((bk, n_over))
    lines.insert(0, f"scalar margin 2 kd h/I + kp h^2/I over {len(configs)} configs (bound 4)")
    if not applies:
        lines.append(f"backend {backend}: PD inside the solver, margin informational")
        status = "INFO"
    else:
        n_over = dict(worst).get(backend, 0)
        status = "WARN" if n_over else "PASS"
        if n_over:
            lines.append(
                f"{backend}: {n_over} joint(s) unstable as decoupled PD; see S17b for the coupled map"
            )
    return CheckResult("S17a", status, lines, data)


# -- S17b ---------------------------------------------------------------------------------


def settle(
    r: Runner, seconds: float = 0.2, cfg: RunConfig | None = None
) -> tuple[np.ndarray, np.ndarray]:
    res_m, d, b = r.build(cfg or RunConfig(), "native_implicit")
    r.reset_state(res_m, d, b, RunConfig(), r.default, np.random.default_rng(0))
    d.ctrl[b.aid] = r.default
    for _ in range(int(round(seconds / b.timestep))):
        mujoco.mj_step(res_m, d)
    return d.qpos.copy(), d.qvel.copy()


def closed_loop_map(
    r: Runner,
    backend: str,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target: np.ndarray,
    cfg: RunConfig | None = None,
) -> tuple[np.ndarray, float, Binding]:
    """One-step closed-loop Jacobian and the time it spans."""
    cfg = cfg or RunConfig()
    m, d, b = r.build(cfg, backend)
    m.dof_frictionloss[:] = 0.0
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(m, d)
    if backend == "native_implicit":
        d.ctrl[b.aid] = target
    else:
        tau = b.kp * (target - d.qpos[b.qadr]) - b.kd * d.qvel[b.dadr]
        d.ctrl[b.aid] = np.clip(tau, -b.limit, b.limit)
    nx = 2 * m.nv + m.na
    a = np.zeros((nx, nx))
    bm = np.zeros((nx, m.nu))
    mujoco.mjd_transitionFD(m, d, 1e-6, True, a, bm, None, None)
    if backend == "native_implicit":
        return a, b.timestep, b
    k = np.zeros((m.nu, nx))
    tau = b.kp * (target - d.qpos[b.qadr]) - b.kd * d.qvel[b.dadr]
    live = np.abs(tau) < b.limit  # a clipped joint has no feedback
    for i in range(len(b.names)):
        if live[i]:
            k[b.aid[i], b.dadr[i]] = -b.kp[i]
            k[b.aid[i], m.nv + b.dadr[i]] = -b.kd[i]
    if backend == "python_pd":
        return a + bm @ k, b.timestep, b
    n = b.pd_every
    acc = np.eye(nx)
    s = np.zeros((nx, nx))
    for _ in range(n):
        s = s + acc @ bm @ k
        acc = a @ acc
    return acc + s, n * b.timestep, b


def classify_modes(eig: np.ndarray, h: float) -> dict[str, Any]:
    mod = np.abs(eig)
    out = mod > 1.0 + MODE_TOL
    real = np.abs(eig.imag) < 1e-6
    slow, bad, drift = [], [], []
    for e, o, rl in zip(eig, out, real):
        if not o:
            continue
        if rl and e.real > 0:
            tc = h / math.log(e.real)
            if tc > DRIFT_TC:
                drift.append((complex(e), tc))
            else:
                (slow if tc >= SLOW_MODE_TC else bad).append((complex(e), tc))
        else:
            bad.append((complex(e), h / math.log(abs(e))))
    return {
        "outside": int(out.sum()) - len(drift),
        "slow": slow,
        "bad": bad,
        "drift": drift,
        "max_mod": float(mod.max()),
    }


def s17b_modes(r: Runner, backend: str, cfg: RunConfig | None = None) -> CheckResult:
    qpos, qvel = settle(r, cfg=cfg)
    a, h, b = closed_loop_map(r, backend, qpos, qvel, r.default, cfg)
    eig = np.linalg.eigvals(a)
    cl = classify_modes(eig, h)
    lines = [
        f"backend {backend}, map step {h * 1e3:.4g} ms, standing after 0.2 s settle, frictionloss zeroed: "
        f"{cl['outside']} mode(s) outside the unit circle"
    ]
    for e, tc in cl["slow"]:
        lines.append(
            f"slow real mode {e.real:.4f} per step: fall time constant {tc:.2f} s (tipping, not a controller fault)"
        )
    for e, tc in sorted(cl["bad"], key=lambda x: -abs(x[0]))[:6]:
        kind = (
            "negative"
            if abs(e.imag) < 1e-6 and e.real < 0
            else ("complex" if abs(e.imag) >= 1e-6 else "fast real")
        )
        lines.append(
            f"{kind} mode {e.real:+.4g}{e.imag:+.4g}j, |lambda| {abs(e):.4g}, growth time {abs(tc) * 1e3:.3g} ms"
        )
    if cl["drift"]:
        lines.append(
            f"{len(cl['drift'])} near-neutral real mode(s) slower than {DRIFT_TC:g} s (drift, not counted): "
            + ", ".join(f"{e.real:.5f}" for e, _ in cl["drift"])
        )
    if len(cl["bad"]) > 6:
        lines.append(f"... {len(cl['bad']) - 6} more")
    if len(cl["slow"]) > 1:
        lines.append("more than one slow real mode: check the contact set at this pose")
    status = "FAIL" if cl["bad"] else "PASS"
    most_neg = min((e.real for e, _ in cl["bad"] if abs(e.imag) < 1e-6), default=None)
    return CheckResult(
        "S17b",
        status,
        lines,
        {
            "outside": cl["outside"],
            "bad": len(cl["bad"]),
            "slow_tc": [tc for _, tc in cl["slow"]],
            "most_negative": most_neg,
            "max_mod": cl["max_mod"],
        },
    )


# -- S18 ---------------------------------------------------------------------------------

AXES = ("vx", "vy", "wz")


def s18_commands(
    c: Contract,
    scenarios: list[tuple[float, float, float]] | None = None,
    dead_zones: dict[str, tuple[float, float]] | None = None,
) -> CheckResult:
    lim = c.get("policy_io.commands.base_velocity.limit", None) or {}
    trained = c.get("policy_io.commands.base_velocity.trained", None)
    tp = c.prov("policy_io.commands.base_velocity.trained")
    ref = trained or lim
    ref_name = "trained" if trained else "limit (no trained range known)"
    lines = [f"reference range: {ref_name}"]
    status = "PASS"
    for ax in AXES:
        lo_l, hi_l = lim.get(ax, [None, None]) if lim else (None, None)
        t = trained.get(ax) if trained else None
        s = f"{ax}: limit {fmt_range(lim.get(ax) if lim else None)}, trained {fmt_range(t)}"
        if t and lim.get(ax) and (lim[ax][0] < t[0] - 1e-9 or lim[ax][1] > t[1] + 1e-9):
            s += "  (limit allows commands never trained)"
            status = "WARN"
        if dead_zones and ax in dead_zones:
            s += f", dead zone {fmt_range(dead_zones[ax])}"
        lines.append(s)
    if trained and tp.source != "live":
        lines.append(f"trained range from {tp.source}: {tp.detail[:160]}")
    for sc in scenarios or []:
        probs = []
        for ax, v in zip(AXES, sc):
            r = ref.get(ax) if ref else None
            if r and not (r[0] - 1e-9 <= v <= r[1] + 1e-9):
                probs.append(f"{ax} {v:+.2f} outside {ref_name.split()[0]} {fmt_range(r)}")
            if dead_zones and ax in dead_zones and v != 0.0:
                dz = dead_zones[ax]
                if dz[0] < v < dz[1]:
                    probs.append(f"{ax} {v:+.2f} inside dead zone {fmt_range(dz)}")
        if probs:
            status = "FAIL"
            lines.append(
                f"scenario ({sc[0]:+.2f}, {sc[1]:+.2f}, {sc[2]:+.2f}): " + "; ".join(probs)
            )
        else:
            lines.append(f"scenario ({sc[0]:+.2f}, {sc[1]:+.2f}, {sc[2]:+.2f}): inside")
    return CheckResult("S18", status, lines)


def fmt_range(r: Any) -> str:
    if r is None:
        return "unknown"
    return f"[{r[0]:+.2f}, {r[1]:+.2f}]"


# -- S19 ---------------------------------------------------------------------------------


def s19_limits(
    r: Runner, run: Any = None, command: tuple[float, float, float] = (0.5, 0.0, 0.0)
) -> CheckResult:
    c = r.contract
    el = c.get("model.effort_limit", None)
    vl = c.get("model.velocity_limit", None)
    m, d, b = r.build(RunConfig(), "native_implicit")
    model = b.model_limit
    if run is None and r.policy is not None:
        run = r.run(RunConfig(seconds=10.0, command=command))
    lines = []
    if el is None:
        lines.append(
            "the contract states no effort limits; the MJCF limits are used: " + _limit_summary(b)
        )
        return CheckResult("S19", "SKIP", lines, {"model_limit": model})
    contract = np.array([float(el[n]) for n in b.names])
    diff = np.flatnonzero(np.abs(contract - model) > 1e-6 * np.maximum(1.0, contract))
    p = c.prov("model.effort_limit")
    lines.append(
        f"{len(diff)} joint(s) with torque limits that differ (contract from {p.source}, MJCF from "
        f"{', '.join(sorted(set(b.model_limit_from)))})"
    )
    active = []
    for i in diff:
        lo = min(contract[i], model[i])
        peak = run.tau_peak[i] if run is not None else None
        state = "unknown (no policy run)"
        if peak is not None:
            state = "ACTIVE" if peak >= lo - 1e-6 else "latent"
            if state == "ACTIVE":
                active.append(b.names[i])
        lines.append(
            f"{b.names[i]}: contract {contract[i]:g}, MJCF {model[i]:g}, walk peak "
            f"{'n/a' if peak is None else f'{peak:.1f}'} N m: {state}"
        )
    if vl is not None and run is not None:
        vlim = np.array([float(vl[n]) for n in b.names])
        over = np.flatnonzero(run.qd_peak > vlim)
        lines.append(
            f"velocity limits: MuJoCo enforces none; walk exceeds the contract limit on {len(over)} joint(s)"
            + (
                ": " + ", ".join(f"{b.names[i]} {run.qd_peak[i]:.1f}>{vlim[i]:g}" for i in over)
                if len(over)
                else ""
            )
        )
    data = {"differ": [b.names[i] for i in diff], "active": active}
    if active and r.policy is not None:
        alt = r.run(RunConfig(seconds=10.0, command=command, limit_source="contract"))
        lines.append(
            f"second loop under contract limits: {alt.summary()} (MJCF limits: {run.summary()})"
        )
        data["contract_run"] = alt.summary()
    if run is not None:
        lines.append(f"walk at {command}: {run.summary()}")
    status = "WARN" if active else ("INFO" if len(diff) else "PASS")
    if active:
        lines.insert(1, "LIMIT_DIFFERENCE_ACTIVE: " + ", ".join(active))
    return CheckResult("S19", status, lines, data)


def _limit_summary(b: Binding) -> str:
    vals = {}
    for n, v in zip(b.names, b.model_limit):
        key = n.replace("left_", "").replace("right_", "").replace("_joint", "")
        vals[key] = v
    return ", ".join(f"{k} {v:g}" for k, v in vals.items())


# -- S21 ---------------------------------------------------------------------------------


def s21_sources(c: Contract, mjcf: str | None = None) -> CheckResult:
    counts: dict[str, int] = {}
    for p in c.provenance.values():
        counts[p.source] = counts.get(p.source, 0) + 1
    lines = ["provenance: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))]
    names = list(c.get("policy_io.joints.names"))
    disagreements = 0
    for path, p in sorted(c.provenance.items()):
        if not p.alternatives:
            continue
        cur = c.get(path, None)
        alts = dict(p.alternatives)
        vals = {"contract": cur, **alts}
        flat = {k: _flatten(v, names) for k, v in vals.items()}
        keys = list(flat)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, bb = flat[keys[i]], flat[keys[j]]
                if a is None or bb is None or a.shape != bb.shape:
                    continue
                tol = (p.resolution or 0.0) + 1e-9 * np.maximum(1.0, np.abs(a))
                bad = np.flatnonzero(np.abs(a - bb) > tol)
                if keys[i] == "contract" and keys[j] in alts and _same_source(p, keys[j]):
                    continue
                if len(bad):
                    disagreements += 1
                    who = (
                        ", ".join(f"{names[k]} {a[k]:g}/{bb[k]:g}" for k in bad[:6])
                        if len(a) == len(names)
                        else ", ".join(f"[{k}] {a[k]:g}/{bb[k]:g}" for k in bad[:6])
                    )
                    more = f" (+{len(bad) - 6} more)" if len(bad) > 6 else ""
                    lines.append(
                        f"{path}: {keys[i]} vs {keys[j]} differ on {len(bad)}: {who}{more}"
                    )
    if mjcf:
        lines.extend(_mjcf_actuator_note(c, mjcf))
    unknown = [k for k, p in c.provenance.items() if p.source == "unknown"]
    if unknown:
        lines.append("unknown: " + ", ".join(sorted(unknown)))
    status = "WARN" if disagreements else "PASS"
    return CheckResult("S21", status, lines, {"counts": counts, "disagreements": disagreements})


def _same_source(p: Any, alt_key: str) -> bool:
    return alt_key in (p.detail or "") and False


def _flatten(v: Any, names: list[str]) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, dict):
        if all(n in v for n in names):
            try:
                return np.array([float(v[n]) for n in names])
            except (TypeError, ValueError):
                return None
        return None
    try:
        return np.asarray(v, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None


def _mjcf_actuator_note(c: Contract, mjcf: str) -> list[str]:
    m = mujoco.MjModel.from_xml_path(mjcf)
    names = list(c.get("policy_io.joints.names"))
    kinds: dict[str, int] = {}
    kps = []
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jn = m.joint(int(m.actuator_trnid[a, 0])).name
        if jn not in names:
            continue
        if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE:
            kinds["position servo"] = kinds.get("position servo", 0) + 1
            kps.append(float(m.actuator_gainprm[a, 0]))
        else:
            kinds["motor"] = kinds.get("motor", 0) + 1
    s = "MJCF actuators on policy joints: " + ", ".join(f"{k} {v}" for k, v in kinds.items())
    if kps:
        s += f" (kp {min(kps):g} to {max(kps):g}); the runner replaces them with the contract gains"
    else:
        s += "; the runner drives them with the contract gains"
    return [s]


# -- all --------------------------------------------------------------------------------


def run_checks(
    r: Runner,
    backend: str | None = None,
    scenarios: list[tuple[float, float, float]] | None = None,
    dead_zones: dict[str, tuple[float, float]] | None = None,
    which: tuple[str, ...] = ("S16", "S17a", "S17b", "S18", "S19", "S21"),
) -> list[CheckResult]:
    backend = r.choose_backend(backend)
    out = []
    configs = None
    if "S16" in which:
        out.append(s16_symmetry(r.contract))
    if "S17a" in which:
        configs = sample_configs(r)
        out.append(s17a_margin(r, backend, configs))
    if "S17b" in which:
        out.append(s17b_modes(r, backend))
    if "S18" in which:
        out.append(s18_commands(r.contract, scenarios, dead_zones))
    if "S19" in which:
        out.append(s19_limits(r, command=scenarios[0] if scenarios else (0.5, 0.0, 0.0)))
    if "S21" in which:
        out.append(s21_sources(r.contract, r.mjcf))
    return out
