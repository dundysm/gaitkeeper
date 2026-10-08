"""Boundaries B, A, C, then D, then the closed loop, then the verdict (plan 5, 7.3).

``diagnose_trace`` is ``verify`` with a target model: the mapping checks
against the trace, the dynamics residual of the trace under the target, the
golden scenario on the target (the nominal closed loop, judged where the
source met the bar) and, when the source's model is recorded, the model
counterfactual. ``diagnose_task`` is the no-reference path: nominal scenarios
from the envelope and a requested task, both in this runner, L1 at most.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .compare import Report, verify
from .contract import Contract
from .counterfactual import Counterfactual, counterfactual, golden_scenario
from .residual import DResult, dynamics_residual
from .task import TaskOutcome, TaskSpec, run_task
from .trace import Trace
from .verdict import Decision, Evidence, decide


@dataclass
class Diagnosis:
    decision: Decision
    evidence: Evidence
    report: Report | None = None
    d: DResult | None = None
    cf: Counterfactual | None = None
    nominal: TaskOutcome | None = None
    task: TaskOutcome | None = None
    envelope: Any = None
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out: list[str] = []
        if self.report is not None:
            out.append(self.report.summary())
        if self.d is not None:
            out.extend(self.d.lines())
        if self.cf is not None:
            out.extend(self.cf.lines())
        elif self.nominal is not None:
            out.append(
                f"Nominal closed loop ({self.nominal.spec.name}, {len(self.nominal.seeds)} seeds) on "
                f"{self.nominal.model}: misses the bar on {self.nominal.n_failed}"
            )
            out.extend(self.nominal.segment_table())
        if self.task is not None:
            out.append(
                f"Task {self.task.spec.name!r} ({len(self.task.seeds)} seeds) on {self.task.model}: "
                f"misses the bar on {self.task.n_failed}/{len(self.task.seeds)} seeds"
            )
            sp = self.task.spec
            if sp.hold:
                out.append(
                    f"  held by the harness, not the policy: {len(sp.hold)} joints "
                    f"({', '.join(sp.hold[:2])}{', ...' if len(sp.hold) > 2 else ''}); "
                    f"their observations: {sp.unowned_obs}"
                )
            if sp.pushes or sp.push_generator:
                out.append(f"  pushes: {_push_text(sp)}")
            out.extend(self.task.segment_table())
            for s in self.task.seeds:
                falls = [t for k, t in s.problems if k in ("fall", "torque limit")]
                if falls:
                    out.append(f"  seed {s.seed}: {'; '.join(falls)}")
        out.extend(f"note: {n}" for n in self.notes)
        out.extend(self.decision.lines())
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_json(),
            "evidence": self.evidence.__dict__,
            "d": self.d.to_json() if self.d else None,
            "counterfactual": self.cf.to_json() if self.cf else None,
            "nominal_failed": self.nominal.n_failed if self.nominal else None,
            "task": {
                "failed": self.task.n_failed,
                "seeds": len(self.task.seeds),
                "kinds": self.task.kinds(),
                "segments": self.task.segment_table(),
            }
            if self.task
            else None,
            "notes": self.notes,
        }


def _push_text(sp: TaskSpec) -> str:
    g = sp.push_generator
    parts = [f"{np.linalg.norm(p.vector):.0f} N at {p.t:g} s" for p in sp.pushes]
    if g is not None and g.force:
        size = f"{g.force:g} N" if g.fixed else f"{0.5 * g.force:g} to {g.force:g} N"
        parts.append(
            f"{size} {g.direction} for {g.duration:g} s on {g.body} every {g.every_s:g} s "
            f"from {g.first_s:g} s"
        )
    if g is not None and g.velocity:
        parts.append(
            f"base kicks up to {g.velocity:g} m/s every {g.every_s:g} s from {g.first_s:g} s"
        )
    return "; ".join(parts)


def _mapping(rep: Report) -> str | None:
    return {"PASS": "pass", "CONTRACT": "fail", "UNDETERMINED": "not_covered"}.get(rep.verdict)


def _controller_assumed(contract: Contract) -> bool:
    return any(
        contract.prov(p).source in ("unknown", "default")
        for p in (
            "control.actuators.kind",
            "control.actuators.pd_period",
            "control.actuators.integrator",
        )
    )


def diagnose_trace(
    trace: Trace,
    contract: Contract,
    policy: Any = None,
    onnx: str | None = None,
    target: str | None = None,
    seeds: tuple[int, ...] = tuple(range(1, 13)),
    run_counterfactual: bool = True,
    task: TaskSpec | None = None,
    workers: int | None = None,
    target_edit: Any = None,
) -> Diagnosis:
    """``target_edit`` changes the target model after loading (a picklable callable)."""
    rep = verify(trace, contract, policy)
    ref = "golden" if trace.kind == "golden" else "harness"
    ev = Evidence(
        reference=ref,
        mapping=_mapping(rep),
        mapping_findings=[
            f"{k}: {'; '.join(b.patterns) or b.status}"
            for k, b in rep.boundaries.items()
            if b.status == "fail"
        ],
        self_trace=trace.self_consistent_only,
        controller_assumed=_controller_assumed(contract),
        limit_active=any(f.startswith("LIMIT_DIFFERENCE_ACTIVE") for f in rep.findings),
        invalid=rep.findings if rep.verdict == "INVALID_INPUT" else [],
    )
    dg = Diagnosis(decide(ev), ev, rep)
    if ev.mapping != "pass" or target is None or ev.invalid:
        return dg
    tmodel: Any = target
    if target_edit is not None:
        from .models import load_model

        tmodel = load_model(target).model
        target_edit(tmodel)
    dg.d = dynamics_residual(trace, contract, tmodel)
    del tmodel
    ev.d, ev.d_chains = dg.d.status, dg.d.above_chains
    source = (
        trace.path
        if trace.path and any((Path(trace.path) / f).exists() for f in ("model.mjb", "model_xml"))
        else None
    )
    if onnx and trace.meta.get("schedule"):
        if source and run_counterfactual:
            dg.cf = counterfactual(
                trace, contract, onnx, target, seeds=seeds, workers=workers, target_edit=target_edit
            )
            dg.nominal = dg.cf.target
            ev.counterfactual = "changes" if dg.cf.changes else "no_change"
            ev.localized = list(dg.cf.localized)
        else:
            dg.nominal = run_task(
                contract, target, onnx, golden_scenario(trace), seeds, workers=workers
            )
            if not source:
                dg.notes.append("the source's model is not recorded: no model counterfactual")
        ev.nominal = "pass" if dg.nominal.passed else "fail"
    elif onnx:
        dg.notes.append("the trace records no command schedule: nominal closed loop not run")
    if task is not None and onnx:
        dg.task = run_task(
            contract, target, onnx, task, seeds[:3], workers=workers, model_edit=target_edit
        )
        ev.task = "pass" if dg.task.passed else "fail"
        ev.task_kinds = dg.task.kinds()
        if not dg.task.passed and source:
            src = run_task(contract, str(source), onnx, task, seeds[:3], workers=workers)
            ev.source_shows_limitation = bool(set(src.kinds()) & set(dg.task.kinds()))
            dg.notes.append(
                f"task on the source's recorded model: misses the bar on {src.n_failed}/{len(src.seeds)}"
                + (f" ({', '.join(sorted(src.kinds()))})" if src.kinds() else "")
            )
    dg.decision = decide(ev)
    return dg


def diagnose_task(
    contract: Contract,
    model: str,
    onnx: str,
    task: TaskSpec,
    seeds: tuple[int, ...] = (1, 2, 3),
    backend: str | None = None,
    workers: int | None = None,
    envelope: Any = None,
) -> Diagnosis:
    """No reference: nominal from the envelope's scenarios, then the task (L1)."""
    from .envelope import sweep

    env = envelope or sweep(contract, model, onnx, backend=backend, workers=workers)
    scen = [s for s in env.scenarios if s["cmd"] is not None]
    nominal = "fail" if any(s["verdict"] == "FAIL" for s in scen) else ("pass" if scen else None)
    out = run_task(contract, model, onnx, task, seeds, backend=backend, workers=workers)
    unknown = len(contract.unknown_fields())
    ev = Evidence(
        reference="none",
        nominal=nominal,
        task="pass" if out.passed else "fail",
        task_kinds=out.kinds(),
        controller_assumed=bool(out.controller.get("CONTROLLER_ASSUMED")),
        unknown_fields=unknown,
    )
    dg = Diagnosis(decide(ev), ev, task=out, envelope=env)
    return dg
