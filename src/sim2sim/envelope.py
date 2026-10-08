"""Command response map (plan section 7.2).

Sweeps each command axis across the contract range and beyond, plus yaw while
walking, in the closed-loop runner. Achieved speed is the mean over the last
5 s of a 15 s run. A command is in a dead zone when the robot achieves less
than 20% of it. Scenarios for closed-loop work are picked from the map and
held to the L1 bar: survive over three seeded starts, right sign and at least
half the commanded speed, standing drift under 0.05 m/s, no joint at its torque
limit more than 20% of steps.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import Contract
from .runner import RunConfig, Runner

DEAD_RATIO = 0.2
SECONDS = 15.0
L1_SPEED_RATIO = 0.5
L1_DRIFT = 0.05
L1_SAT = 0.2
SEEDS = (1, 2, 3)

_RUNNER: Runner | None = None


def _init(contract: dict[str, Any], mjcf: str, onnx: str) -> None:
    global _RUNNER
    from .policy import OnnxPolicy

    _RUNNER = Runner(Contract.from_dict(contract), mjcf, OnnxPolicy(onnx))


def _one(job: tuple[str, tuple[float, float, float], dict[str, Any]]) -> dict[str, Any]:
    label, cmd, kw = job
    res = _RUNNER.run(RunConfig(command=cmd, seconds=SECONDS, **kw))
    return {
        "label": label,
        "cmd": cmd,
        "fell_at": res.fell_at,
        "achieved": (res.vx_tail, res.vy_tail, res.wz_tail),
        "sat_max": float(res.sat_frac.max()),
        "controller": res.controller,
        "seed": kw.get("seed", 0),
    }


@dataclass
class Envelope:
    rows: dict[str, list[dict[str, Any]]]
    dead: dict[str, dict[str, Any]]
    ref_name: str
    ref: dict[str, list[float]]
    limit: dict[str, list[float]]
    trained: dict[str, list[float]] | None
    scenarios: list[dict[str, Any]] = field(default_factory=list)
    controller: dict[str, Any] | None = None
    findings: list[str] = field(default_factory=list)

    def dead_zones(self) -> dict[str, tuple[float, float]]:
        return {
            ax: (d["neg_edge"] or 0.0, d["pos_edge"] or 0.0)
            for ax, d in self.dead.items()
            if ax != "wz_walk"
        }

    def lines(self) -> list[str]:
        out = []
        if self.controller:
            out.extend(Runner.controller_text(self.controller))
        out.append(
            f"reference range: {self.ref_name}; dead zone = achieved below {DEAD_RATIO:.0%} of commanded"
        )
        for key, title in (
            ("vx", "vx"),
            ("vy", "vy"),
            ("wz", "wz in place"),
            ("wz_walk", "wz at vx 0.50"),
        ):
            cells = []
            for r in self.rows.get(key, []):
                c = r["cmd"][AX[key]]
                a = r["achieved"][AX[key]]
                mark = " FELL" if r["fell_at"] is not None else ""
                tag = _tag(c, AX_NAME[key], self.trained, self.limit)
                cells.append(f"{c:+.2f}{tag}->{a:+.2f}{mark}")
            out.append(f"  {title:<13} " + "  ".join(cells))
        out.append("  (* outside trained range, ** outside limit range)")
        for d in self.dead.values():
            if d["text"]:
                out.append(f"DEAD ZONE  {d['text']}")
        for f in self.findings:
            out.append(f"Finding   {f}")
        if self.scenarios:
            out.append("Scenarios (L1 bar over seeds " + ", ".join(map(str, SEEDS)) + "):")
            for s in self.scenarios:
                out.append(f"  {s['name']:<22} {s['verdict']}  {s['detail']}")
        return out


AX = {"vx": 0, "vy": 1, "wz": 2, "wz_walk": 2}
AX_NAME = {"vx": "vx", "vy": "vy", "wz": "wz", "wz_walk": "wz"}


def _tag(v: float, ax: str, trained: dict | None, limit: dict) -> str:
    lim = limit.get(ax) if limit else None
    if lim and not (lim[0] - 1e-9 <= v <= lim[1] + 1e-9):
        return "**"
    tr = trained.get(ax) if trained else None
    if tr and not (tr[0] - 1e-9 <= v <= tr[1] + 1e-9):
        return "*"
    return ""


def grids(limit: dict[str, list[float]]) -> dict[str, list[float]]:
    def span(ax: str, fine: list[float], coarse_step: float, beyond: float) -> list[float]:
        lo, hi = limit.get(ax, [-1.0, 1.0])
        vals = set()
        for f in fine:
            vals.update((f, -f))
        v = math.floor((lo - beyond) / coarse_step) * coarse_step
        while v <= hi + beyond + 1e-9:
            vals.add(round(v, 3))
            v += coarse_step
        vals.update((lo, hi))
        return sorted(x for x in vals if abs(x) > 1e-9)

    return {
        "vx": span("vx", [0.05, 0.1, 0.15, 0.2, 0.22, 0.25, 0.3, 0.4], 0.25, 0.5),
        "vy": span("vy", [0.05, 0.1, 0.15, 0.2, 0.25, 0.28, 0.3], 0.1, 0.2),
        "wz": span("wz", [0.1, 0.2, 0.3, 0.5], 0.25, 0.5),
        "wz_walk": sorted({0.2, -0.2, 0.5, -0.5, *(limit.get("wz") or [])}),
    }


def _dead(rows: list[dict[str, Any]], i: int, ref: list[float] | None) -> dict[str, Any]:
    """Edges of the dead zone around zero, scanning outward on each side."""
    out: dict[str, Any] = {"pos_edge": None, "pos_first": None, "neg_edge": None, "neg_first": None}
    for side, sign in (("pos", 1), ("neg", -1)):
        rs = sorted((r for r in rows if sign * r["cmd"][i] > 0), key=lambda r: abs(r["cmd"][i]))
        for r in rs:
            c, a = r["cmd"][i], r["achieved"][i]
            if r["fell_at"] is None and a * c > 0 and abs(a) >= DEAD_RATIO * abs(c):
                out[f"{side}_first"] = c
                break
            out[f"{side}_edge"] = c
    return out


def sweep(
    contract: Contract, mjcf: str, onnx: str, backend: str | None = None, workers: int | None = None
) -> Envelope:
    lim = contract.get("policy_io.commands.base_velocity.limit", None) or {}
    trained = contract.get("policy_io.commands.base_velocity.trained", None)
    ref = trained or lim
    g = grids(lim)
    kw = {"backend": backend} if backend else {}
    jobs = []
    for key, vals in g.items():
        for v in vals:
            cmd = [0.0, 0.0, 0.0]
            cmd[AX[key]] = v
            if key == "wz_walk":
                cmd[0] = 0.5
            jobs.append((key, tuple(cmd), kw))
    workers = workers or min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(
        workers, initializer=_init, initargs=(contract.to_dict(), mjcf, onnx)
    ) as ex:
        res = list(ex.map(_one, jobs))
        rows: dict[str, list[dict[str, Any]]] = {k: [] for k in g}
        for r in res:
            rows[r["label"]].append(r)
        env = Envelope(
            rows,
            {},
            "trained" if trained else "limit (no trained range known)",
            ref,
            lim,
            trained,
            controller=res[0]["controller"] if res else None,
        )
        stand = _stand_threshold(contract)
        for key in ("vx", "vy", "wz"):
            d = _dead(rows[key], AX[key], ref.get(key) if ref else None)
            d["text"] = _dead_text(key, d, ref.get(AX_NAME[key]) if ref else None, stand)
            env.dead[key] = d
        env.scenarios = _scenarios(env, ex, kw)
    limited = any(d.get("limited") for d in env.dead.values()) or any(
        s["verdict"] == "FAIL" or s["verdict"] == "NONE" for s in env.scenarios
    )
    if limited:
        env.findings.append(
            "BEHAVIORAL_LIMITATION (L1, this runner, these assumptions; not an attribution)"
        )
    return env


def _stand_threshold(c: Contract) -> float | None:
    for t in c.get("policy_io.observation_groups.policy.terms", []) or []:
        if t["id"] == "gait_phase" and t.get("params", {}).get("stand_threshold") is not None:
            return float(t["params"]["stand_threshold"])
    return None


def _dead_text(key: str, d: dict[str, Any], ref: list[float] | None, stand: float | None) -> str:
    parts = []
    axis = "in-place turning" if key == "wz" else key
    for side, sign in (("pos", 1), ("neg", -1)):
        edge, first = d[f"{side}_edge"], d[f"{side}_first"]
        if edge is None:
            continue
        s = "+" if sign > 0 else "-"
        r = ref[1] if (ref and sign > 0) else (ref[0] if ref else None)
        if first is None or (r is not None and abs(first) > abs(r) + 1e-9):
            parts.append(
                f"no {axis} {s} up to {edge:+.2f} (reference range ends at "
                f"{'unknown' if r is None else f'{r:+.2f}'})"
            )
            d["limited"] = True
        elif stand is not None and abs(edge) < stand:
            parts.append(
                f"{axis} ignored for |cmd| <= {abs(edge):.2f} on the {s} side, below the phase "
                f"term's stand threshold {stand:g} (by design)"
            )
        else:
            parts.append(
                f"{axis} ignored for |cmd| <= {abs(edge):.2f} on the {s} side (first tracked {first:+.2f})"
            )
            d["limited"] = True
    return "; ".join(parts)


def _pick(env: Envelope, key: str, sign: int, target: float) -> float | None:
    """A tracked command inside the reference and limit ranges, closest to target."""
    r = env.ref.get(AX_NAME[key]) if env.ref else None
    best = None
    for row in env.rows[key]:
        c, a = row["cmd"][AX[key]], row["achieved"][AX[key]]
        if (
            sign * c <= 0
            or row["fell_at"] is not None
            or a * c <= 0
            or abs(a) < DEAD_RATIO * abs(c)
        ):
            continue
        if r and not (r[0] - 1e-9 <= c <= r[1] + 1e-9):
            continue
        lim = env.limit.get(AX_NAME[key]) if env.limit else None
        if lim and not (lim[0] - 1e-9 <= c <= lim[1] + 1e-9):
            continue  # a scenario stays inside what the command interface allows
        if best is None or abs(c - target) < abs(best - target):
            best = c
    return best


def _scenarios(env: Envelope, ex: ProcessPoolExecutor, kw: dict[str, Any]) -> list[dict[str, Any]]:
    want: list[tuple[str, tuple[float, float, float] | None]] = [("stand", (0.0, 0.0, 0.0))]
    fx = _pick(env, "vx", 1, 0.5)
    bx = _pick(env, "vx", -1, -0.5)
    ly = _pick(env, "vy", 1, (env.ref.get("vy") or [0, 0.3])[1])
    wz = _pick(env, "wz", 1, (env.ref.get("wz") or [0, 0.2])[1])
    want.append(("fwd", (fx, 0.0, 0.0) if fx else None))
    want.append(("back", (bx, 0.0, 0.0) if bx else None))
    want.append(("lat", (0.0, ly, 0.0) if ly else None))
    if wz:
        want.append(("yaw in place", (0.0, 0.0, wz)))
    else:
        wr = env.ref.get("wz") if env.ref else None
        want.append(("yaw in place", None))
        if fx and wr:
            want.append(("fwd + yaw", (fx, 0.0, wr[1])))
    jobs = []
    for name, cmd in want:
        if cmd is None:
            continue
        for s in SEEDS:
            jobs.append(
                (
                    name,
                    cmd,
                    {
                        **kw,
                        "seed": s,
                        "init_noise": 0.02,
                        "yaw0": float(np.random.default_rng(s).uniform(-math.pi, math.pi)),
                    },
                )
            )
    res = list(ex.map(_one, jobs))
    out = []
    for name, cmd in want:
        if cmd is None:
            out.append(
                {
                    "name": name,
                    "cmd": None,
                    "verdict": "NONE",
                    "detail": "no tracked command inside the reference range",
                }
            )
            continue
        rs = [r for r in res if r["label"] == name]
        probs = []
        for r in rs:
            if r["fell_at"] is not None:
                probs.append(f"seed {r['seed']} fell at {r['fell_at']:.2f} s")
                continue
            a = r["achieved"]
            if name == "stand":
                drift = math.hypot(a[0], a[1])
                if drift >= L1_DRIFT:
                    probs.append(f"seed {r['seed']} drifts {drift:.3f} m/s")
            for i in range(3):
                if cmd[i] != 0 and (a[i] * cmd[i] <= 0 or abs(a[i]) < L1_SPEED_RATIO * abs(cmd[i])):
                    probs.append(
                        f"seed {r['seed']} {('vx', 'vy', 'wz')[i]} {a[i]:+.2f} for {cmd[i]:+.2f}"
                    )
            if r["sat_max"] > L1_SAT:
                probs.append(f"seed {r['seed']} at torque limit {r['sat_max']:.0%} of steps")
        ach = np.mean(
            [r["achieved"] for r in rs if r["fell_at"] is None] or [[math.nan] * 3], axis=0
        )
        detail = f"cmd ({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}) achieved ({ach[0]:+.2f}, {ach[1]:+.2f}, {ach[2]:+.2f})"
        out.append(
            {
                "name": name,
                "cmd": cmd,
                "verdict": "FAIL" if probs else "pass",
                "detail": detail + ("; " + "; ".join(probs) if probs else ""),
            }
        )
    return out
