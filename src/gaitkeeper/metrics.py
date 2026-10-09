"""The headline metric: false confident attribution rate (plan section 10).

Over a corpus of cases with known causes, the share of confident outputs
(``CONTRACT``, ``PHYSICS``, ``POLICY_UNDER_TASK``, or a named parameter group)
whose named cause is wrong, reported next to the abstention rate and the
detection rate per boundary.

A case's truth is ``{"kind": k, "boundary": b, "tokens": [...], "groups": [...]}``
with kind one of ``contract``, ``physics``, ``task``, ``none``. A confident
output is wrong when its kind differs from the truth, when a ``CONTRACT``
output's cause misses one of the truth's tokens, or when a ``PHYSICS`` output
names a group the truth does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

KIND_OF = {"CONTRACT": "contract", "PHYSICS": "physics", "POLICY_UNDER_TASK": "task"}


@dataclass
class Case:
    name: str
    truth: dict[str, Any]
    verdict: str | None
    cause: str | None
    groups: list[str]
    boundaries: list[str]  # boundaries the output names as failing
    findings: list[str]


def wrong(c: Case) -> str | None:
    """Why a confident output is wrong, or None when it is right or not confident."""
    kind = KIND_OF.get(c.verdict or "")
    if kind is None and not c.groups:
        return None
    if kind is None:
        kind = "physics"
    t = c.truth
    if kind != t["kind"]:
        return f"named {kind}, truth {t['kind']}"
    if kind == "contract":
        text = c.cause or ""
        missing = [x for x in t.get("tokens", []) if x not in text]
        if missing:
            return f"cause misses {missing}"
    if kind == "physics" and c.groups:
        extra = [g for g in c.groups if g.split(": ")[0] not in t.get("groups", [])]
        if extra:
            return f"names {extra}, truth {t.get('groups')}"
    return None


def score(cases: list[Case]) -> dict[str, Any]:
    confident = [c for c in cases if KIND_OF.get(c.verdict or "") or c.groups]
    bad = [(c.name, wrong(c)) for c in confident if wrong(c)]
    abstain = [c for c in cases if c.verdict == "UNDETERMINED" or c.verdict is None]
    det: dict[str, dict[str, int]] = {}
    for c in cases:
        for b in str(c.truth.get("boundary") or "").split("+"):
            if not b:
                continue
            e = det.setdefault(b, {"cases": 0, "detected": 0})
            e["cases"] += 1
            if (b == "D" and c.verdict == "PHYSICS") or (
                b in c.boundaries and c.verdict == "CONTRACT"
            ):
                e["detected"] += 1
    return {
        "cases": len(cases),
        "confident": len(confident),
        "false_confident": len(bad),
        "false_confident_attribution_rate": (len(bad) / len(confident)) if confident else 0.0,
        "false_confident_cases": bad,
        "abstention_rate": len(abstain) / len(cases) if cases else 0.0,
        "detection": {b: {**v, "rate": v["detected"] / v["cases"]} for b, v in sorted(det.items())},
    }
