"""Verdicts, L1 findings, evidence levels and exit codes (plan section 4).

``decide`` is the decision table as code. It takes what was measured
(``Evidence``) and returns one verdict or L1 findings only, the evidence
level, the exit code, what was ruled out, what would settle an
``UNDETERMINED``, and the caveats that must travel with the result.

Rules kept here and nowhere else:

* ``PHYSICS`` needs a golden reference from an independent source, A to C
  passing, a nominal failure, D above a floor calibrated for the source
  engine, and a counterfactual in which the outcome changes with the model
  (or with the limits, for an active limit difference). A residual alone is
  co-occurrence and never enough.
* ``POLICY_UNDER_TASK`` needs A to C passing against a golden reference and
  the source showing the same limitation.
* Without a reference the strongest output is an L1 finding, and a silent
  contract error is never excluded.
* A self trace (written by this runner) never raises the evidence level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EXIT = {
    "PASS": 0,
    "CONTRACT": 1,
    "INVALID_INPUT": 2,
    "PHYSICS": 3,
    "POLICY_UNDER_TASK": 4,
    "UNDETERMINED": 5,
    "UNSUPPORTED": 6,
}
CONFIDENT = ("CONTRACT", "PHYSICS", "POLICY_UNDER_TASK")
LIMITATION_KINDS = ("dead zone", "partial", "fall", "torque limit", "drift")
L1_CAVEAT = (
    "L1: measured in this runner only, under the stated controller assumptions; "
    "a working rollout is not proof of a correct contract"
)


@dataclass
class Evidence:
    reference: str  # "golden" | "harness" | "none"
    mapping: str | None = None  # A to C: "pass" | "fail" | "not_covered" | None (not checked)
    mapping_findings: list[str] = field(default_factory=list)
    nominal: str | None = None  # L1 or L2 bar on the nominal scenarios: "pass" | "fail" | None
    task: str | None = None  # the requested task: "pass" | "fail" | None (none requested)
    task_kinds: dict[str, int] = field(default_factory=dict)  # failure kinds over seeds
    d: str | None = None  # "at_floor" | "above_floor" | "not_measured" | "uncalibrated" | None
    d_chains: list[str] = field(default_factory=list)
    counterfactual: str | None = None  # "changes" | "no_change" | None (not run)
    localized: list[str] = field(default_factory=list)  # groups whose swap restores the outcome
    localized_randomized: bool = False  # the restoring values are the source's randomization draw
    limit_active: bool = False
    limit_counterfactual: str | None = None  # "changes" | "no_change" | None
    source_shows_limitation: bool | None = None
    self_trace: bool = False
    controller_assumed: bool = False
    unknown_fields: int | None = None
    invalid: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


@dataclass
class Decision:
    verdict: str | None  # None: L1 findings only
    findings: list[str]
    evidence: str
    exit_code: int
    row: str
    cause: str | None = None
    ruled_out: list[str] = field(default_factory=list)
    next_step: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.verdict in CONFIDENT

    @property
    def headline(self) -> str:
        if self.verdict:
            return self.verdict
        main = [f for f in self.findings if f in ("TASK_FAILURE_OBSERVED", "BEHAVIORAL_LIMITATION")]
        return " / ".join(main) if main else "no verdict"

    def lines(self) -> list[str]:
        out = [
            f"{'Verdict' if self.verdict else 'Finding'}  {self.headline}  (evidence {self.evidence}, "
            f"exit {self.exit_code})"
        ]
        if self.cause:
            out.append(f"  cause: {self.cause}")
        rest = [f for f in self.findings if f not in self.headline]
        if rest:
            out.append(f"  findings: {', '.join(rest)}")
        out.append(f"  decision table row: {self.row}")
        for r in self.ruled_out:
            out.append(f"  ruled out: {r}")
        for n in self.next_step:
            out.append(f"  next: {n}")
        for c in self.caveats:
            out.append(f"  caveat: {c}")
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "findings": self.findings,
            "evidence": self.evidence,
            "exit_code": self.exit_code,
            "row": self.row,
            "cause": self.cause,
            "ruled_out": self.ruled_out,
            "next": self.next_step,
            "caveats": self.caveats,
        }


def _d_finding(ev: Evidence) -> list[str]:
    if ev.d == "above_floor":
        return [f"D above floor on {', '.join(ev.d_chains)}"]
    return []


def _common(ev: Evidence) -> list[str]:
    out = []
    if ev.controller_assumed:
        out.append("CONTROLLER_ASSUMED")
    if ev.limit_active:
        out.append("LIMIT_DIFFERENCE_ACTIVE")
    return out


def _limitation(ev: Evidence) -> bool:
    return bool(ev.task_kinds) and all(k in LIMITATION_KINDS for k in ev.task_kinds)


def decide(ev: Evidence) -> Decision:
    common = _common(ev)
    if ev.invalid:
        return Decision(
            "INVALID_INPUT", ev.invalid, "L0", EXIT["INVALID_INPUT"], "unreadable input"
        )
    if ev.unsupported:
        return Decision(
            "UNSUPPORTED", ev.unsupported, "L0", EXIT["UNSUPPORTED"], "semantics not representable"
        )

    if ev.reference in ("golden", "harness") and ev.mapping == "fail":
        return Decision(
            "CONTRACT",
            ev.mapping_findings + common,
            "L1",
            EXIT["CONTRACT"],
            f"{ev.reference}: mapping fails",
            cause="; ".join(ev.mapping_findings) or "boundary A, B or C",
        )

    if ev.reference == "golden" and not ev.self_trace:
        return _golden(ev, common)
    if ev.reference in ("golden", "harness"):
        return _harness(ev, common)
    return _none(ev, common)


def _golden(ev: Evidence, common: list[str]) -> Decision:
    if ev.mapping != "pass":
        return Decision(
            "UNDETERMINED",
            ["under-excited: some contract properties are not separated by this trace"] + common,
            "L1",
            EXIT["UNDETERMINED"],
            "golden: mapping not covered",
            next_step=["record a trace with the missing excitation items"],
        )
    d_level = "L3" if ev.d == "at_floor" else "L2"
    if ev.nominal in ("pass", None) and ev.task in ("pass", None):
        cav = [] if ev.nominal else ["closed loop not run: conformance of A to C only"]
        if ev.d in (None, "not_measured"):
            cav.append("D not measured: L3 needs physics-rate channels")
        return Decision(
            "PASS",
            common + _d_finding(ev),
            d_level,
            EXIT["PASS"],
            "golden: pass, pass",
            caveats=cav,
        )
    if ev.nominal in ("pass", None) and ev.task == "fail":
        if ev.source_shows_limitation:
            return Decision(
                "POLICY_UNDER_TASK",
                ["TASK_FAILURE_OBSERVED"] + _d_finding(ev) + common,
                "L2",
                EXIT["POLICY_UNDER_TASK"],
                "golden: pass, pass, task fails, source shows it",
                cause="the source shows the same limitation under the task's commands",
            )
        return Decision(
            None,
            ["TASK_FAILURE_OBSERVED"] + _d_finding(ev) + common,
            "L2",
            EXIT["UNDETERMINED"],
            "golden: pass, pass, task fails",
            ruled_out=["contract mapping (A to C pass against the golden trace)"],
            next_step=[
                "run the task's commands in the source simulator (POLICY_UNDER_TASK if it fails there too)"
            ],
            caveats=["a residual or model difference is co-occurrence, not a cause"],
        )
    # nominal failure
    if ev.d == "above_floor":
        changed = ev.counterfactual == "changes" or ev.limit_counterfactual == "changes"
        if changed:
            groups = list(ev.localized)
            if ev.limit_counterfactual == "changes" and "effort limits" not in groups:
                groups.append("effort limits")
            cause = f"dynamics differ on {', '.join(ev.d_chains)}"
            if groups:
                cause += f"; the outcome follows {', '.join(groups)}"
            return Decision(
                "PHYSICS",
                _d_finding(ev) + common,
                "L2",
                EXIT["PHYSICS"],
                "golden: pass, nominal fails, D above floor, counterfactual changes the outcome",
                cause=cause,
                ruled_out=["contract mapping (A to C pass against the golden trace)"],
                caveats=[
                    "the chain is named from D; a group is named only when swapping it restores the outcome"
                ]
                + (
                    [
                        "the restoring values are the source's startup randomization draw: the "
                        "target differs from this sample of the training distribution, not "
                        "necessarily from the distribution"
                    ]
                    if ev.localized_randomized
                    else []
                ),
            )
        return Decision(
            "UNDETERMINED",
            _d_finding(ev) + common,
            "L2",
            EXIT["UNDETERMINED"],
            "golden: pass, nominal fails, D above floor, no counterfactual or no change",
            ruled_out=["contract mapping (A to C pass against the golden trace)"],
            next_step=["run the model counterfactual (needs the source's recorded model)"]
            if ev.counterfactual is None
            else [
                "the model difference does not change the outcome: look at stance contact, reset and timing"
            ],
        )
    if ev.d in (None, "not_measured", "uncalibrated"):
        why = "uncalibrated for the source engine" if ev.d == "uncalibrated" else "not measured"
        return Decision(
            "UNDETERMINED",
            common,
            "L2",
            EXIT["UNDETERMINED"],
            f"golden: pass, nominal fails, D {why}",
            ruled_out=["contract mapping (A to C pass against the golden trace)"],
            next_step=[
                "E4 calibration for this engine"
                if ev.d == "uncalibrated"
                else "record physics-rate channels"
            ],
        )
    return Decision(
        "UNDETERMINED",
        common,
        "L3",
        EXIT["UNDETERMINED"],
        "golden: pass, nominal fails, D at floor",
        ruled_out=["contract mapping", "dynamics on the measured channels"],
        next_step=["look at stance contact, reset and timing"],
    )


def _harness(ev: Evidence, common: list[str]) -> Decision:
    label = ["SELF_CONSISTENT"] if ev.self_trace else []
    if ev.mapping != "pass":
        return Decision(
            "UNDETERMINED",
            label
            + ["under-excited: some contract properties are not separated by this trace"]
            + common,
            "L1",
            EXIT["UNDETERMINED"],
            "harness log: mapping not covered",
        )
    if ev.nominal == "fail":
        return Decision(
            "UNDETERMINED",
            label + common,
            "L1",
            EXIT["UNDETERMINED"],
            "harness log: conforms, nominal fails",
            ruled_out=["the logged mapping (A and C)"],
            next_step=["a golden trace from the training stack"],
        )
    if ev.task == "fail":
        return Decision(
            None,
            label + ["TASK_FAILURE_OBSERVED"] + common,
            "L1",
            EXIT["UNDETERMINED"],
            "harness log: conforms, task fails",
            ruled_out=["the logged mapping (A and C)"],
            caveats=[L1_CAVEAT],
        )
    return Decision("PASS", label + common, "L1", EXIT["PASS"], "harness log: conforms")


def _none(ev: Evidence, common: list[str]) -> Decision:
    unknown = (
        f"{ev.unknown_fields} contract fields default or unknown"
        if ev.unknown_fields is not None
        else None
    )
    cav = [L1_CAVEAT] + ([unknown] if unknown else [])
    if ev.nominal == "fail":
        return Decision(
            "UNDETERMINED",
            common,
            "L1",
            EXIT["UNDETERMINED"],
            "no reference: nominal fails",
            next_step=["static checks (gaitkeeper check)", "a golden trace or harness log"],
            caveats=cav,
        )
    if ev.nominal is None:
        if ev.task == "fail":
            return Decision(
                "UNDETERMINED",
                ["TASK_FAILURE_OBSERVED"] + common,
                "L1",
                EXIT["UNDETERMINED"],
                "no reference: nominal not run, task fails",
                next_step=["run the nominal scenarios"],
                caveats=cav,
            )
        return Decision(
            "UNDETERMINED", common, "L0", EXIT["UNDETERMINED"], "no reference: nothing run"
        )
    if ev.task in ("pass", None):
        return Decision(
            "PASS",
            common,
            "L1",
            EXIT["PASS"],
            "no reference: nominal passes, task passes or none",
            caveats=cav,
        )
    found = ["TASK_FAILURE_OBSERVED"] + (["BEHAVIORAL_LIMITATION"] if _limitation(ev) else [])
    return Decision(
        None,
        found + common,
        "L1",
        EXIT["UNDETERMINED"],
        "no reference: nominal passes, task fails",
        next_step=[
            "a golden trace from the training stack under the task's commands "
            "(POLICY_UNDER_TASK if the source shows the same limitation)"
        ],
        caveats=cav
        + [
            "a silent contract error is not excluded",
            "no attribution: neither PHYSICS nor CONTRACT can be concluded without a reference",
        ],
    )
