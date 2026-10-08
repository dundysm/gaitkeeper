"""Behavior probes that turn the command response map into an L1 finding
(plan sections 1, 4 and 7.2).

* Standstill inside a dead zone: over the last 5 s of a 15 s run the action
  barely moves (mean per joint std below 0.02), the joints are still (mean
  |qd| below 0.002 rad/s) and no body makes or breaks ground contact. Run
  under two backends, and again after a base velocity kick.
* Yaw while walking: achieved over commanded yaw rate at vx 0.5, inside the
  limit range.
* Pushes: survival over 30 s runs at the trained push level and under force
  punches.
* Physics fragility (optional): the selected scenarios under changed friction,
  contact softness, armature, kp scale and step size, each next to a
  no-policy fall baseline on the same model and backend, S17a and S17b. A
  fall time is compared with the baseline only when S17a does not warn and no
  joint sits at its torque limit. Reported as fragility, never as attribution.

Every number here comes from our runner: evidence L1, under the controller
assumptions printed with it.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from .contract import Contract
from .runner import Push, PushGenerator, RunConfig, Runner

STILL_ACTION_STD = 0.02
STILL_JOINT_SPEED = 0.002  # rad/s
TAIL_S = 5.0
STILL_SECONDS = 15.0
KICK_AT = 5.0
KICK_SECONDS = 20.0
PUSH_SECONDS = 30.0
PUSH_EVERY = 5.0
PUNCH_N = 600.0  # magnitude uniform in [0.5, 1] of this
PUNCH_S = 0.08
AT_LIMIT_FRAC = 0.01  # a joint "sits at its torque limit" above this share of steps
FRAGILITY_SECONDS = 10.0


# -- model variants for the fragility sweep -------------------------------------------


def _hinge_dofs(m: mujoco.MjModel) -> np.ndarray:
    return np.array(
        [m.jnt_dofadr[j] for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE],
        dtype=int,
    )


def _friction(mu: float):
    def f(m: mujoco.MjModel) -> None:
        m.geom_friction[:, 0] = mu

    return f


def _solref(tc: float):
    def f(m: mujoco.MjModel) -> None:
        m.geom_solref[:, 0] = tc

    return f


def _armature(mult: float):
    def f(m: mujoco.MjModel) -> None:
        dofs = _hinge_dofs(m)
        m.dof_armature[dofs] = m.dof_armature[dofs] * mult

    return f


def _step(h: float):
    def f(m: mujoco.MjModel) -> None:
        m.opt.timestep = h

    return f


# name: (model edit or None, kp scale)
VARIANTS: dict[str, tuple[Any, float]] = {
    "nominal": (None, 1.0),
    "friction 0.3": (_friction(0.3), 1.0),
    "friction 2.0": (_friction(2.0), 1.0),
    "contacts softer (solref 0.05)": (_solref(0.05), 1.0),
    "contacts stiffer (solref 0.005)": (_solref(0.005), 1.0),
    "armature 0": (_armature(0.0), 1.0),
    "armature x3": (_armature(3.0), 1.0),
    "kp x0.8": (None, 0.8),
    "kp x1.2": (None, 1.2),
    "step 1 ms": (_step(0.001), 1.0),
    "step 5 ms": (_step(0.005), 1.0),
}


# -- worker ---------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _winit(contract: dict[str, Any], mjcf: str, onnx: str) -> None:
    _W.clear()
    _W.update(contract=contract, mjcf=mjcf, onnx=onnx, runners={})


def _runner(kp_scale: float = 1.0) -> Runner:
    rs = _W["runners"]
    if kp_scale not in rs:
        from .policy import OnnxPolicy

        c = Contract.from_dict(_W["contract"])
        if kp_scale != 1.0:
            kp = dict(c.get("control.actuators.kp"))
            c.set(
                "control.actuators.kp",
                {k: v * kp_scale for k, v in kp.items()},
                "user",
                f"fragility sweep: kp x{kp_scale:g}",
            )
        rs[kp_scale] = Runner(c, _W["mjcf"], OnnxPolicy(_W["onnx"]))
    return rs[kp_scale]


def _cfg(job: dict[str, Any]) -> RunConfig:
    kw = dict(job.get("cfg", {}))
    edit = VARIANTS[job["variant"]][0] if job.get("variant") else None
    if edit is not None:
        kw["model_edit"] = edit
    return RunConfig(**kw)


def stillness(res: Any, policy_dt: float, tail_s: float = TAIL_S) -> dict[str, Any]:
    """Standstill measures over the last ``tail_s`` of a recorded run."""
    out: dict[str, Any] = {
        "fell_at": res.fell_at,
        "speed": float(math.hypot(res.vx_tail, res.vy_tail)),
        "wz": float(res.wz_tail),
    }
    if res.fell_at is not None or res.log is None:
        out["still"] = False
        return out
    n = max(1, int(round(tail_s / policy_dt)))
    act = res.log["action"][-n:]
    qd = res.log["qvel"][-n:][:, res.dadr]
    out["action_std"] = float(act.std(axis=0).mean())
    out["joint_speed"] = float(np.abs(qd).mean())
    if res.contacts is not None and len(res.log["action"]):
        sub = len(res.contacts) // len(res.log["action"])
        c = res.contacts[-n * sub :].astype(np.int8)
        out["contact_switches"] = int(np.abs(np.diff(c, axis=0)).sum())
    else:
        out["contact_switches"] = None
    out["still"] = is_still(out)
    return out


def is_still(s: dict[str, Any]) -> bool:
    return (
        s.get("fell_at") is None
        and s.get("action_std", math.inf) < STILL_ACTION_STD
        and s.get("joint_speed", math.inf) < STILL_JOINT_SPEED
        and s.get("contact_switches") == 0
    )


def _work(job: dict[str, Any]) -> dict[str, Any]:
    kind = job["kind"]
    kp_scale = VARIANTS[job["variant"]][1] if job.get("variant") else 1.0
    r = _runner(kp_scale)
    out = {k: v for k, v in job.items() if k != "cfg"}
    if kind == "check":
        from .checks import s17a_margin, s17b_modes

        cfg = _cfg(job)
        a = s17a_margin(r, job["backend"], cfg=cfg)
        b = s17b_modes(r, job["backend"], cfg=cfg)
        mg = a.data["margins"].get(job["backend"])
        out.update(
            s17a=a.status,
            s17a_over=int((mg >= 4.0).sum()) if mg is not None else 0,
            s17a_max=float(mg.max()) if mg is not None else None,
            s17b=b.status,
            s17b_bad=b.data["bad"],
            s17b_most_negative=b.data["most_negative"],
            tipping_tc=(b.data["slow_tc"][0] if b.data["slow_tc"] else None),
        )
        return out
    cfg = _cfg(job)
    res = r.run(cfg)
    out.update(
        fell_at=res.fell_at,
        achieved=(res.vx_tail, res.vy_tail, res.wz_tail),
        dist=res.dist,
        at_limit=[res.names[i] for i in np.flatnonzero(res.sat_frac > AT_LIMIT_FRAC)],
        sat_max=float(res.sat_frac.max()),
        controller=res.controller,
    )
    if kind == "still":
        out["still"] = stillness(res, r.policy_dt)
    return out


# -- the probes -----------------------------------------------------------------------


@dataclass
class Probes:
    header: str = ""
    backends: list[str] = field(default_factory=list)
    standstill: list[dict[str, Any]] = field(default_factory=list)
    kicks: list[dict[str, Any]] = field(default_factory=list)
    kick_vector: tuple[float, float, float] | None = None
    kick_source: str = ""
    yaw_walk: dict[str, Any] | None = None
    pushes: list[dict[str, Any]] = field(default_factory=list)
    fragility: dict[str, Any] | None = None
    findings: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = []
        if self.standstill:
            out.append(
                f"STANDSTILL inside the dead zone (last {TAIL_S:g} s of {STILL_SECONDS:g} s; still = mean "
                f"action std < {STILL_ACTION_STD:g}, mean |qd| < {STILL_JOINT_SPEED:g} rad/s, no ground "
                "contact made or broken):"
            )
            for row in self.standstill:
                cells = [f"{bk}: {_still_text(row['by'][bk])}" for bk in self.backends]
                out.append(f"  cmd {_cmd(row['cmd'])}  " + "  ".join(cells))
        if self.kicks:
            v = self.kick_vector
            out.append(
                f"  after one base kick ({v[0]:+.2f}, {v[1]:+.2f}) m/s at {KICK_AT:g} s ({self.kick_source}), "
                f"last {TAIL_S:g} s of {KICK_SECONDS:g} s:"
            )
            for row in self.kicks:
                cells = [f"{bk}: {_still_text(row['by'][bk])}" for bk in self.backends]
                out.append(f"  cmd {_cmd(row['cmd'])}  " + "  ".join(cells))
        if self.yaw_walk:
            y = self.yaw_walk
            if y.get("ratios"):
                cells = "  ".join(
                    f"{c:+.2f} -> {a:+.2f}" for c, a in zip(y["commands"], y["achieved"])
                )
                out.append(
                    f"YAW WHILE WALKING at vx {y['vx']:.2f}: tracks {min(y['ratios']):.0%} to "
                    f"{max(y['ratios']):.0%} of the commanded rate inside the limit range ({cells})"
                )
            else:
                out.append(f"YAW WHILE WALKING at vx {y['vx']:.2f}: no row inside the limit range")
        if self.pushes:
            p0 = self.pushes[0]
            out.append(
                f"PUSHES ({_cmd(p0['cmd'])}, {PUSH_SECONDS:g} s runs, {p0['n']} seeds, {p0['backend']}):"
            )
            for p in self.pushes:
                fell = [t for t in p["fell_at"] if t is not None]
                extra = (
                    f" (falls at {min(fell):.1f} to {max(fell):.1f} s, mean {np.mean(fell):.1f} s)"
                    if fell
                    else ""
                )
                out.append(f"  {p['label']}: survived {p['survived']}/{p['n']}{extra}")
        if self.fragility:
            out.extend(_fragility_lines(self.fragility))
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "header": self.header,
            "backends": self.backends,
            "standstill": self.standstill,
            "kicks": self.kicks,
            "kick_vector": self.kick_vector,
            "kick_source": self.kick_source,
            "yaw_walk": self.yaw_walk,
            "pushes": self.pushes,
            "fragility": self.fragility,
            "findings": self.findings,
        }


def _cmd(c: Any) -> str:
    return f"({c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f})"


def _still_text(s: dict[str, Any]) -> str:
    if s.get("fell_at") is not None:
        return f"fell at {s['fell_at']:.2f} s"
    word = "still" if s["still"] else "MOVING"
    return (
        f"{word} (action std {s['action_std']:.4f}, |qd| {s['joint_speed']:.4f}, "
        f"{s['contact_switches']} contact changes, speed {s['speed']:.3f}, wz {s['wz']:+.3f})"
    )


def contract_header(c: Contract) -> str:
    src = c.get("source", None) or {}
    files = ", ".join(f["path"] for f in src.get("files", []) or [])
    n_unknown = sum(1 for p in c.provenance.values() if p.source == "unknown")
    n_default = sum(1 for p in c.provenance.values() if p.source == "default")
    presets = c.get("source.presets", None)
    pre = f" | presets {', '.join(presets)}" if presets else ""
    return (
        f"Contract  {src.get('format', 'unknown')} ({files or 'no files'}) | level L1 | "
        f"{n_unknown + n_default} fields default or unknown ({n_unknown} unknown, {n_default} default){pre}"
    )


def second_backend(primary: str) -> str:
    # python_pd at the MJCF step is how hand-written harnesses run explicit PD
    return "python_pd" if primary != "python_pd" else "native_implicit"


def _dead_commands(env: Any) -> list[tuple[float, float, float]]:
    from .envelope import AX, AX_NAME

    cmds = []
    for key in ("vx", "vy", "wz"):
        d = env.dead.get(key)
        if not d:
            continue
        lim = env.limit.get(AX_NAME[key]) if env.limit else None
        for side in ("pos", "neg"):
            e = d.get(f"{side}_edge")
            if e is None:
                continue
            if lim and not (lim[0] - 1e-9 <= e <= lim[1] + 1e-9):
                # the largest ignored command inside what the interface allows
                inside = [
                    r["cmd"][AX[key]]
                    for r in env.rows[key]
                    if (r["cmd"][AX[key]] > 0) == (side == "pos")
                    and abs(r["cmd"][AX[key]]) <= abs(e)
                    and lim[0] - 1e-9 <= r["cmd"][AX[key]] <= lim[1] + 1e-9
                ]
                if not inside:
                    continue
                e = max(inside, key=abs)
            c = [0.0, 0.0, 0.0]
            c[AX[key]] = float(e)
            cmds.append(tuple(c))
    return cmds


def yaw_walk(env: Any) -> dict[str, Any]:
    lim = env.limit.get("wz") if env.limit else None
    rows = [
        r
        for r in env.rows.get("wz_walk", [])
        if r["fell_at"] is None and (not lim or lim[0] - 1e-9 <= r["cmd"][2] <= lim[1] + 1e-9)
    ]
    rows.sort(key=lambda r: r["cmd"][2])
    return {
        "vx": rows[0]["cmd"][0] if rows else 0.5,
        "commands": [r["cmd"][2] for r in rows],
        "achieved": [r["achieved"][2] for r in rows],
        "ratios": [r["achieved"][2] / r["cmd"][2] for r in rows],
    }


def _kick(c: Contract) -> tuple[tuple[float, float, float], str]:
    te = c.get("model.training_envelope", None) or {}
    p = te.get("pushes") if isinstance(te, dict) else None
    if p and p.get("kind") == "velocity_kick" and p.get("x") and p.get("y"):
        return (float(max(p["x"])), float(max(p["y"])), 0.0), "corner of the trained push range"
    return (0.5, 0.5, 0.0), "trained push level unknown; 0.5 m/s per axis assumed"


def _trained_pushes(c: Contract) -> tuple[float, float, str] | None:
    te = c.get("model.training_envelope", None) or {}
    p = te.get("pushes") if isinstance(te, dict) else None
    if p and p.get("kind") == "velocity_kick" and p.get("x"):
        v = float(max(abs(x) for x in p["x"]))
        every = float(p.get("every_s", PUSH_EVERY))
        return v, every, c.prov("model.training_envelope").source
    return None


def probe(
    env: Any,
    contract: Contract,
    mjcf: str,
    onnx: str,
    backend: str | None = None,
    workers: int | None = None,
    pushes: bool = True,
    push_seeds: int = 10,
    physics: bool = False,
    variants: list[str] | None = None,
) -> Probes:
    r = Runner(contract, mjcf)
    primary = r.choose_backend(backend)
    backends = [primary, second_backend(primary)]
    pr = Probes(header=contract_header(contract), backends=backends)
    kick, kick_src = _kick(contract)
    pr.kick_vector, pr.kick_source = kick, kick_src
    jobs: list[dict[str, Any]] = []
    dead = _dead_commands(env)
    for cmd in dead:
        for bk in backends:
            jobs.append(
                {
                    "kind": "still",
                    "tag": "still",
                    "cmd": cmd,
                    "backend": bk,
                    "cfg": dict(
                        backend=bk, command=cmd, seconds=STILL_SECONDS, record=True, contacts=True
                    ),
                }
            )
            jobs.append(
                {
                    "kind": "still",
                    "tag": "kick",
                    "cmd": cmd,
                    "backend": bk,
                    "cfg": dict(
                        backend=bk,
                        command=cmd,
                        seconds=KICK_SECONDS,
                        record=True,
                        contacts=True,
                        pushes=[Push(KICK_AT, "velocity", kick)],
                    ),
                }
            )
    fwd = next((s["cmd"] for s in env.scenarios if s["name"] == "fwd" and s.get("cmd")), None)
    push_specs = []
    if pushes and fwd:
        tp = _trained_pushes(contract)
        if tp:
            v, every, src = tp
            push_specs.append(
                (
                    f"trained level ({src}): base velocity kicks uniform in +-{v:g} m/s on x and y every {every:g} s",
                    PushGenerator(every_s=every, velocity=v),
                )
            )
        push_specs.append(
            (
                f"force punches {PUNCH_N / 2:g} to {PUNCH_N:g} N for {PUNCH_S:g} s, random body and "
                f"direction, every {PUSH_EVERY:g} s",
                PushGenerator(every_s=PUSH_EVERY, force=PUNCH_N, duration=PUNCH_S),
            )
        )
        for label, gen in push_specs:
            for s in range(1, push_seeds + 1):
                jobs.append(
                    {
                        "kind": "run",
                        "tag": "push",
                        "label": label,
                        "cmd": fwd,
                        "backend": primary,
                        "cfg": dict(
                            backend=primary,
                            command=fwd,
                            seconds=PUSH_SECONDS,
                            seed=s,
                            push_generator=gen,
                        ),
                    }
                )
    scen = [s for s in env.scenarios if s.get("cmd")]
    vnames = variants or list(VARIANTS)
    if physics:
        for v in vnames:
            for bk in backends:
                jobs.append({"kind": "check", "tag": "frag_check", "variant": v, "backend": bk})
                jobs.append(
                    {
                        "kind": "run",
                        "tag": "frag_zero",
                        "variant": v,
                        "backend": bk,
                        "cfg": dict(
                            backend=bk,
                            command=(0.0, 0.0, 0.0),
                            seconds=FRAGILITY_SECONDS,
                            policy_mode="zero",
                        ),
                    }
                )
                for s in scen:
                    jobs.append(
                        {
                            "kind": "run",
                            "tag": "frag_run",
                            "variant": v,
                            "backend": bk,
                            "scenario": s["name"],
                            "cmd": s["cmd"],
                            "cfg": dict(
                                backend=bk, command=tuple(s["cmd"]), seconds=FRAGILITY_SECONDS
                            ),
                        }
                    )
    workers = workers or min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(
        workers, initializer=_winit, initargs=(contract.to_dict(), mjcf, onnx)
    ) as ex:
        res = list(ex.map(_work, jobs, chunksize=1))

    for tag, dest in (("still", pr.standstill), ("kick", pr.kicks)):
        for cmd in dead:
            by = {
                x["backend"]: x["still"]
                for x in res
                if x["tag"] == tag and tuple(x["cmd"]) == tuple(cmd)
            }
            dest.append({"cmd": cmd, "by": by})
    pr.yaw_walk = yaw_walk(env)
    for label, _ in push_specs:
        rs = [x for x in res if x["tag"] == "push" and x["label"] == label]
        pr.pushes.append(
            {
                "label": label,
                "cmd": fwd,
                "backend": primary,
                "n": len(rs),
                "survived": sum(x["fell_at"] is None for x in rs),
                "fell_at": [x["fell_at"] for x in rs],
            }
        )
    if physics:
        pr.fragility = fragility_table(res, vnames, backends, scen)
    return pr


# -- fragility --------------------------------------------------------------------------


def _passes(x: dict[str, Any]) -> bool:
    from .envelope import L1_DRIFT, L1_SAT, L1_SPEED_RATIO

    if x["fell_at"] is not None or x["sat_max"] > L1_SAT:
        return False
    cmd, a = x["cmd"], x["achieved"]
    if all(c == 0 for c in cmd):
        return math.hypot(a[0], a[1]) < L1_DRIFT
    return all(
        c == 0 or (a[i] * c > 0 and abs(a[i]) >= L1_SPEED_RATIO * abs(c)) for i, c in enumerate(cmd)
    )


def fall_comparison(
    runs: list[dict[str, Any]], zero: dict[str, Any], chk: dict[str, Any]
) -> dict[str, Any]:
    """Policy fall times against the no-policy baseline on the same model and backend,
    unless S17a warns or a joint sits at its torque limit (plan section 4)."""
    falls = [x for x in runs if x["fell_at"] is not None]
    reasons = []
    if chk["s17a"] == "WARN":
        reasons.append(f"S17a warns on {chk['s17a_over']} joint(s)")
    limited = sorted({n for x in runs for n in x["at_limit"]})
    if limited:
        reasons.append(
            f"{len(limited)} joint(s) at their torque limit on more than {AT_LIMIT_FRAC:.0%} of steps"
        )
    out: dict[str, Any] = {
        "baseline": zero["fell_at"],
        "tipping_tc": chk["tipping_tc"],
        "suppressed": reasons,
        "falls": [(x["scenario"], x["fell_at"]) for x in falls],
    }
    if falls and not reasons and zero["fell_at"] is not None:
        out["before_baseline"] = [s for s, t in out["falls"] if t < zero["fell_at"]]
    return out


def fragility_table(
    res: list[dict[str, Any]], vnames: list[str], backends: list[str], scen: list[dict[str, Any]]
) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for v in vnames:
        cells[v] = {}
        for bk in backends:
            runs = [
                x
                for x in res
                if x["tag"] == "frag_run" and x["variant"] == v and x["backend"] == bk
            ]
            zero = next(
                x
                for x in res
                if x["tag"] == "frag_zero" and x["variant"] == v and x["backend"] == bk
            )
            chk = next(
                x
                for x in res
                if x["tag"] == "frag_check" and x["variant"] == v and x["backend"] == bk
            )
            cells[v][bk] = {
                "runs": {
                    x["scenario"]: {
                        "fell_at": x["fell_at"],
                        "achieved": x["achieved"],
                        "pass": _passes(x),
                        "at_limit": x["at_limit"],
                    }
                    for x in runs
                },
                "zero_fell_at": zero["fell_at"],
                "check": {k: chk[k] for k in chk if k.startswith(("s17", "tipping"))},
                "falls": fall_comparison(runs, zero, chk),
            }
    loud = []
    for v in vnames:
        if v == "nominal":
            continue
        for bk in backends:
            nom = cells.get("nominal", {}).get(bk)
            cur = cells[v][bk]
            lost = [
                s
                for s, r in cur["runs"].items()
                if not r["pass"] and (nom is None or nom["runs"].get(s, {}).get("pass"))
            ]
            if not lost:
                continue
            ck, nk = cur["check"], (nom["check"] if nom else {})
            numeric = (ck.get("s17a") == "WARN" and nk.get("s17a") != "WARN") or (
                ck.get("s17b") == "FAIL" and nk.get("s17b") != "FAIL"
            )
            loud.append({"variant": v, "backend": bk, "lost": lost, "numerical": numeric})
    return {
        "scenarios": [s["name"] for s in scen],
        "backends": backends,
        "cells": cells,
        "loud": loud,
    }


def _cell(r: dict[str, Any], cmd_axis: int | None) -> str:
    if r["fell_at"] is not None:
        return f"fell {r['fell_at']:.2f}s"
    a = r["achieved"]
    v = a[cmd_axis] if cmd_axis is not None else math.hypot(a[0], a[1])
    return f"{v:+.2f}" + ("" if r["pass"] else "!")


def _fragility_lines(f: dict[str, Any]) -> list[str]:
    out = [
        f"FRAGILITY (L1, {FRAGILITY_SECONDS:g} s per run, seed 0; a fragility, never an attribution; "
        "cells show the commanded axis (fwd + yaw: wz; stand: horizontal drift); ! = misses the L1 bar)"
    ]
    scen = f["scenarios"]
    head = f"  {'variant':<32}{'backend':<17}" + "".join(f"{s:>14}" for s in scen)
    head += f"{'no policy':>12}  S17a, S17b"
    out.append(head)
    for v, by in f["cells"].items():
        for bk, cell in by.items():
            row = f"  {v:<32}{bk:<17}"
            for s in scen:
                r = cell["runs"].get(s)
                axis = None
                if r is not None:
                    # report the axis the scenario commands; standing reports drift
                    axis = _axis_of(s, r)
                row += f"{(_cell(r, axis) if r else '-'):>14}"
            z = cell["zero_fell_at"]
            row += f"{('fell ' + format(z, '.2f') + 's') if z is not None else 'stands':>12}"
            ck = cell["check"]
            s17a = f"{ck['s17a']}" + (
                f" ({ck['s17a_over']} >= 4, max {ck['s17a_max']:.2f})"
                if ck.get("s17a_max") is not None and bk != "native_implicit"
                else ""
            )
            s17b = f"{ck['s17b']}" + (f" ({ck['s17b_bad']} bad modes)" if ck["s17b_bad"] else "")
            tc = ck.get("tipping_tc")
            row += f"  {s17a}, {s17b}" + (f", tipping {tc:.2f} s" if tc else "")
            out.append(row)
    for v, by in f["cells"].items():
        for bk, cell in by.items():
            fc = cell["falls"]
            if not fc["falls"]:
                continue
            first = min(t for _, t in fc["falls"])
            base = fc["baseline"]
            base_txt = (
                f"no-policy baseline {base:.2f} s" if base is not None else "no-policy run stands"
            )
            tc_txt = (
                f", S17b tipping time constant {fc['tipping_tc']:.2f} s" if fc["tipping_tc"] else ""
            )
            if fc["suppressed"]:
                out.append(
                    f"  fall time, {v} / {bk}: first fall {first:.2f} s; comparison suppressed ("
                    + "; ".join(fc["suppressed"])
                    + f"); {base_txt}{tc_txt}"
                )
            else:
                before = fc.get("before_baseline") or []
                rel = (
                    f"falls before the no-policy baseline in {len(before)} scenario(s)"
                    if before
                    else "no fall before the no-policy baseline"
                )
                out.append(
                    f"  fall time, {v} / {bk}: first fall {first:.2f} s; {base_txt}{tc_txt}; {rel}"
                )
    for item in f["loud"]:
        what = ", ".join(item["lost"])
        if item["numerical"]:
            out.append(
                f"LOUD  {item['variant']} under {item['backend']}: misses the bar in {what}. S17a or S17b "
                "fires here and not on the nominal model: the PD loop is numerically unstable on this "
                "model at this step. No trace: UNDETERMINED, with S17a and S17b named; not a contract "
                "finding, not PHYSICS."
            )
        else:
            out.append(
                f"LOUD  {item['variant']} under {item['backend']}: misses the bar in {what} "
                "(fragility at L1, not an attribution)"
            )
    if not f["loud"]:
        out.append("no variant changes a scenario that passes on the nominal model")
    return out


def _axis_of(name: str, r: dict[str, Any]) -> int | None:
    return {"fwd": 0, "back": 0, "lat": 1, "yaw in place": 2, "fwd + yaw": 2}.get(name)
