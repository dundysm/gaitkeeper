"""One report for a policy under a walking benchmark: where along the way from its own setup
to the benchmark's full conditions does it stop walking.

The tour is run at a ladder of stages, each adding one thing the benchmark harness does:

  own        the policy drives every joint it was trained to drive; no punches
  arms_hold  the harness takes the arms and holds them at its stance, with its gains
  arms_walk  the harness walks the arms at random
  punches    and punches a random link every waypoint (the full benchmark)

With a port contract (what a benchmark port actually runs: joints it does not drive, faked
observations, its own gains), the full stack is run on it too, and its values are compared
with the policy's own contract. The survival lost at each step is the report's attribution:
L1 evidence on this runner and model, not a claim about the benchmark's own numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .tour import BENCH_ARMS, TourOptions, run_tour

STAGES: dict[str, dict[str, str]] = {
    "own": {"arms": "policy", "punches": "none"},
    "arms_hold": {"arms": "hold", "punches": "none"},
    "arms_walk": {"arms": "walk", "punches": "none"},
    "punches": {"arms": "walk", "punches": "benchmark"},
}
STAGE_TEXT = {
    "own": "own setup",
    "arms_hold": "harness holds the arms",
    "arms_walk": "harness walks the arms",
    "punches": "plus punches (full benchmark)",
    "port": "port contract, full benchmark",
    "port_quiet": "port contract, arms still, no punches",
}
# A stage that costs less than this much mean survival is not called out.
NOTABLE_S = 5.0


@dataclass
class BenchReport:
    name: str
    seconds: float
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    deviation: list[str] = field(default_factory=list)
    envelope: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def gate(self) -> bool:
        last = self.stages.get("port") or self.stages.get("punches")
        return bool(last) and last["complete"] == len(last["runs"])

    def lines(self) -> list[str]:
        out = [f"Bench: {self.name}; tour {self.seconds:.0f} s, seeds per stage shown"]
        out.append(
            f"  {'stage':<38} {'survival':>12} {'complete':>9} {'pos err':>8} {'yaw err':>8}"
        )
        for key, st in self.stages.items():
            pos = "-" if math.isnan(st["pos_err_cm"]) else f"{st['pos_err_cm']:.0f} cm"
            yaw = "-" if math.isnan(st["yaw_err_deg"]) else f"{st['yaw_err_deg']:.0f} deg"
            out.append(
                f"  {STAGE_TEXT.get(key, key):<38} {st['mean_survival_s']:>7.1f} s   "
                f"{st['complete']:>3d}/{len(st['runs']):<3d}  {pos:>8} {yaw:>8}"
            )
        for n in self.notes:
            out.append(f"Note      {n}")
        for f in self.findings:
            out.append(f"Finding   {f}")
        if self.deviation:
            out.append("")
            out.extend(self.deviation)
        if self.envelope:
            out.append("")
            out.extend(self.envelope)
        out.append("Evidence L1: this runner, this model, these assumptions; not an attribution.")
        return out

    def markdown(self) -> str:
        md = [f"# gaitkeeper bench: {self.name}", ""]
        md.append(f"Tour of {self.seconds:.0f} s per seed. Evidence L1 (this runner and model).")
        md += ["", "| Stage | Mean survival (s) | Complete | Pos err (cm) | Yaw err (deg) |"]
        md.append("|---|---|---|---|---|")
        for key, st in self.stages.items():
            pos = "–" if math.isnan(st["pos_err_cm"]) else f"{st['pos_err_cm']:.0f}"
            yaw = "–" if math.isnan(st["yaw_err_deg"]) else f"{st['yaw_err_deg']:.0f}"
            md.append(
                f"| {STAGE_TEXT.get(key, key)} | {st['mean_survival_s']:.1f} | "
                f"{st['complete']}/{len(st['runs'])} | {pos} | {yaw} |"
            )
        if self.findings or self.notes:
            md += ["", "## Findings", ""]
            md += [f"- {f}" for f in self.findings]
            md += [f"- Note: {n}" for n in self.notes]
        if self.deviation:
            md += ["", "## Port against the policy's own values", "", "```"]
            md += self.deviation + ["```"]
        if self.envelope:
            md += ["", "## Command envelope", "", "```"]
            md += self.envelope + ["```"]
        return "\n".join(md) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "command": "bench",
            "evidence": "L1",
            "name": self.name,
            "seconds": self.seconds,
            "stages": self.stages,
            "findings": self.findings,
            "notes": self.notes,
            "deviation": self.deviation,
            "envelope": self.envelope,
        }


def attribute(stages: dict[str, dict[str, Any]], seconds: float) -> list[str]:
    """Name the steps of the ladder that cost survival, largest first."""
    order = [k for k in STAGES if k in stages]
    steps = []
    for a, b in zip(order, order[1:]):
        lost = stages[a]["mean_survival_s"] - stages[b]["mean_survival_s"]
        steps.append((lost, a, b))
    if "port" in stages and "punches" in stages:
        lost = stages["punches"]["mean_survival_s"] - stages["port"]["mean_survival_s"]
        steps.append((lost, "punches", "port"))
    out = []
    own = stages.get(order[0]) if order else None
    if own and own["mean_survival_s"] < seconds - 1e-6:
        out.append(
            f"falls in its own setup: mean survival {own['mean_survival_s']:.1f} of {seconds:.0f} s "
            "with nothing the harness adds; the port is not the first suspect"
        )
    for lost, a, b in sorted(steps, reverse=True):
        if lost < NOTABLE_S:
            continue
        what = {
            "arms_hold": "taking the arms away from the policy (held at the harness stance, harness gains)",
            "arms_walk": "the random arm walk",
            "punches": "punching",
            "port": "the port's own differences (ownership, gains, observations) over the clean contract",
        }[b]
        out.append(
            f"{what} costs {lost:.1f} s of mean survival "
            f"({stages[a]['mean_survival_s']:.1f} -> {stages[b]['mean_survival_s']:.1f} s)"
        )
    if "port" in stages and "punches" in stages:
        lost = stages["port"]["mean_survival_s"] - stages["punches"]["mean_survival_s"]
        if lost >= NOTABLE_S:
            out.append(
                f"the port survives {lost:.1f} s longer than the clean contract under the same "
                "stack; its changes help here"
            )
    return out


def unlisted_diff(port: Any, own: Any) -> list[str]:
    """How the port holds the joints the policy does not list, against the policy's own
    contract (a legs-only policy trained with the arms held some other way)."""
    a = port.get("control.unlisted", None) or {}
    b = own.get("control.unlisted", None) or {}
    if not a and not b:
        return []
    if not a or not b:
        side = "the port" if a else "the policy's own contract"
        return [f"unlisted joints: only {side} holds them (control.unlisted)"]
    out = []
    for f in ("pose", "kp", "kd"):
        va, vb = a.get(f, {}) or {}, b.get(f, {}) or {}
        names = sorted(set(va) | set(vb)) if isinstance(va, dict) and isinstance(vb, dict) else []
        if not names:
            if va != vb:
                out.append(f"unlisted {f}: port {va} against {vb}")
            continue
        diff = [(n, float(va.get(n, 0.0)), float(vb.get(n, 0.0))) for n in names]
        diff = [(n, x, y) for n, x, y in diff if abs(x - y) > 1e-6 * max(abs(y), 1e-3)]
        if not diff:
            out.append(f"unlisted {f}: same on {len(names)} joint(s)")
            continue
        n, x, y = max(diff, key=lambda r: abs(r[1] - r[2]))
        out.append(
            f"unlisted {f}: {len(diff)} of {len(names)} joint(s) differ; largest {n} "
            f"port {x:.4g} against {y:.4g}"
        )
    return out


def run_bench(
    contract: Any,
    mjcf: str,
    policy_path: str,
    seeds: list[int],
    port: Any | None = None,
    stages: list[str] | None = None,
    waypoints: list[tuple[float, float, float]] | None = None,
    point_s: float = 5.0,
    workers: int | None = None,
    backend: str | None = None,
    name: str = "policy",
    progress: Any = None,
) -> BenchReport:
    stages = list(stages or STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; choose from {list(STAGES)}")
    names = list(contract.get("policy_io.joints.names"))
    arms_listed = [j for j in names if j in BENCH_ARMS]
    unlisted = contract.get("control.unlisted", None)
    notes = []
    if not arms_listed:
        # A legs-only policy: the harness has the arms in every stage already.
        drop = {"arms_hold"} if unlisted else {"arms_hold", "arms_walk"}
        stages = [s for s in stages if s not in drop]
        notes.append(
            "the policy lists no arm joints: the harness holds them in every stage"
            + ("" if unlisted else " (no control.unlisted: they get no torque and cannot walk)")
        )
    rep = BenchReport(name=name, seconds=0.0, notes=notes)

    def one(key: str, c: Any, arms: str, punches: str, arms_obs: str) -> None:
        if progress:
            progress(key)
        o = TourOptions(
            waypoints=waypoints,
            point_s=point_s,
            arms=arms,
            arms_obs=arms_obs,
            hold_gains="armature",
            punches=punches,
            backend=backend,
        )
        r = run_tour(c, mjcf, policy_path, seeds, o, workers)
        rep.seconds = r["seconds"]
        rep.stages[key] = r

    for key in stages:
        st = STAGES[key]
        arms = st["arms"]
        if not arms_listed:
            arms = "walk" if arms == "walk" and unlisted else "policy"
        one(key, contract, arms, st["punches"], "real")
    if port is not None:
        port_arms = [j for j in port.get("policy_io.joints.names") if j in BENCH_ARMS]
        walk = "walk" if (port_arms or port.get("control.unlisted", None)) else "policy"
        one("port_quiet", port, "policy", "none", "contract")
        one("port", port, walk, "benchmark", "contract")
        from .deviation import compare

        try:
            dev = compare(port, contract, reference_name="the policy's own contract")
            lines = dev.lines()
            lines[0] = "Port values against the policy's own contract (the port runs its values):"
            rep.deviation = lines
        except KeyError as e:
            rep.deviation = [f"port against the policy's own contract: not comparable ({e})"]
        rep.deviation += unlisted_diff(port, contract)
        own_ext = (port.get("control.ownership", None) or {}).get("external", []) or []
        for e in own_ext:
            e = e if isinstance(e, dict) else e.__dict__
            rep.deviation.append(
                f"port: harness drives {len(e['joints'])} joint(s) ({e.get('drive', 'hold')}), "
                f"policy sees them as {e.get('obs', 'real')}: {', '.join(e['joints'][:6])}"
                + (" ..." if len(e["joints"]) > 6 else "")
            )
    rep.findings = attribute(rep.stages, rep.seconds)
    if "port_quiet" in rep.stages and "own" in rep.stages:
        lost = rep.stages["own"]["mean_survival_s"] - rep.stages["port_quiet"]["mean_survival_s"]
        if lost >= NOTABLE_S:
            rep.findings.insert(
                0,
                f"the port alone, with arms still and no punches, costs {lost:.1f} s against the "
                "policy's own contract: the port is broken before the benchmark stresses it",
            )
    return rep
