"""The decision table of plan section 4, row by row, and its invariants."""

import itertools

import pytest

from gaitkeeper.metrics import Case, score
from gaitkeeper.verdict import EXIT, Evidence, decide

G = {"reference": "golden", "mapping": "pass"}
ROWS = [
    # (row, evidence, verdict, exit code, evidence level, findings that must appear)
    (
        "golden mapping fails",
        dict(reference="golden", mapping="fail", mapping_findings=["C: kp scaled x0.7"]),
        "CONTRACT",
        1,
        "L1",
        [],
    ),
    ("golden pass pass, D at floor", dict(**G, nominal="pass", d="at_floor"), "PASS", 0, "L3", []),
    (
        "golden pass pass, D above",
        dict(**G, nominal="pass", task="pass", d="above_floor", d_chains=["left leg"]),
        "PASS",
        0,
        "L2",
        ["D above floor on left leg"],
    ),
    (
        "golden nominal fails, D above, counterfactual changes",
        dict(
            **G,
            nominal="fail",
            d="above_floor",
            d_chains=["root/contact"],
            counterfactual="changes",
            localized=["contact parameters"],
        ),
        "PHYSICS",
        3,
        "L2",
        [],
    ),
    (
        "golden nominal fails, D above, limits change the outcome",
        dict(
            **G,
            nominal="fail",
            d="above_floor",
            d_chains=["left leg"],
            limit_active=True,
            limit_counterfactual="changes",
        ),
        "PHYSICS",
        3,
        "L2",
        ["LIMIT_DIFFERENCE_ACTIVE"],
    ),
    (
        "golden nominal fails, D above, no counterfactual",
        dict(**G, nominal="fail", d="above_floor", d_chains=["left leg"]),
        "UNDETERMINED",
        5,
        "L2",
        ["D above floor on left leg"],
    ),
    (
        "golden nominal fails, D above, counterfactual no change",
        dict(
            **G, nominal="fail", d="above_floor", d_chains=["left leg"], counterfactual="no_change"
        ),
        "UNDETERMINED",
        5,
        "L2",
        [],
    ),
    (
        "golden nominal fails, D not measured",
        dict(**G, nominal="fail", d="not_measured", counterfactual="changes"),
        "UNDETERMINED",
        5,
        "L2",
        [],
    ),
    (
        "golden nominal fails, D uncalibrated (PhysX before E4)",
        dict(**G, nominal="fail", d="uncalibrated", counterfactual="changes"),
        "UNDETERMINED",
        5,
        "L2",
        [],
    ),
    (
        "golden nominal fails, D at floor",
        dict(**G, nominal="fail", d="at_floor"),
        "UNDETERMINED",
        5,
        "L3",
        [],
    ),
    (
        "golden task fails",
        dict(**G, nominal="pass", task="fail", d="above_floor", d_chains=["waist"]),
        None,
        5,
        "L2",
        ["TASK_FAILURE_OBSERVED", "D above floor on waist"],
    ),
    (
        "golden task fails, source shows it",
        dict(**G, nominal="pass", task="fail", d="at_floor", source_shows_limitation=True),
        "POLICY_UNDER_TASK",
        4,
        "L2",
        ["TASK_FAILURE_OBSERVED"],
    ),
    ("harness mapping fails", dict(reference="harness", mapping="fail"), "CONTRACT", 1, "L1", []),
    ("harness conforms", dict(reference="harness", mapping="pass"), "PASS", 0, "L1", []),
    (
        "harness conforms, task fails",
        dict(reference="harness", mapping="pass", nominal="pass", task="fail"),
        None,
        5,
        "L1",
        ["TASK_FAILURE_OBSERVED"],
    ),
    (
        "harness conforms, nominal fails",
        dict(reference="harness", mapping="pass", nominal="fail"),
        "UNDETERMINED",
        5,
        "L1",
        [],
    ),
    (
        "no reference, pass",
        dict(reference="none", nominal="pass", unknown_fields=7),
        "PASS",
        0,
        "L1",
        [],
    ),
    (
        "no reference, task fails in a dead zone",
        dict(reference="none", nominal="pass", task="fail", task_kinds={"dead zone": 3}),
        None,
        5,
        "L1",
        ["TASK_FAILURE_OBSERVED", "BEHAVIORAL_LIMITATION"],
    ),
    (
        "no reference, task fails otherwise",
        dict(reference="none", nominal="pass", task="fail", task_kinds={"wrong sign": 1}),
        None,
        5,
        "L1",
        ["TASK_FAILURE_OBSERVED"],
    ),
    (
        "no reference, nominal fails",
        dict(reference="none", nominal="fail"),
        "UNDETERMINED",
        5,
        "L1",
        [],
    ),
    (
        "invalid input",
        dict(reference="golden", invalid=["missing key obs"]),
        "INVALID_INPUT",
        2,
        "L0",
        [],
    ),
    (
        "unsupported",
        dict(reference="none", unsupported=["learned actuator net"]),
        "UNSUPPORTED",
        6,
        "L0",
        [],
    ),
]


@pytest.mark.parametrize("row,ev,verdict,code,level,findings", ROWS, ids=[r[0] for r in ROWS])
def test_decision_table_row(row, ev, verdict, code, level, findings):
    d = decide(Evidence(**ev))
    assert d.verdict == verdict, d.lines()
    assert d.exit_code == code
    assert d.evidence == level
    for f in findings:
        assert f in d.findings, (f, d.findings)
    if verdict:
        assert EXIT[verdict] == code


def test_physics_names_a_chain_and_only_groups_that_restore():
    d = decide(
        Evidence(
            **G, nominal="fail", d="above_floor", d_chains=["left leg"], counterfactual="changes"
        )
    )
    assert d.verdict == "PHYSICS"
    assert d.cause == "dynamics differ on left leg"  # no parameter without a restoring swap
    d = decide(
        Evidence(
            **G,
            nominal="fail",
            d="above_floor",
            d_chains=["left leg"],
            counterfactual="changes",
            localized=["mass and inertia"],
        )
    )
    assert "the outcome follows mass and inertia" in d.cause
    assert not any("randomization" in c for c in d.caveats)
    d = decide(
        Evidence(
            **G,
            nominal="fail",
            d="above_floor",
            d_chains=["left leg"],
            counterfactual="changes",
            localized=["mass and inertia"],
            localized_randomized=True,
        )
    )
    assert d.verdict == "PHYSICS" and any("randomization draw" in c for c in d.caveats)


def test_no_reference_carries_the_caveats():
    d = decide(
        Evidence(
            reference="none",
            nominal="pass",
            task="fail",
            task_kinds={"dead zone": 3},
            unknown_fields=7,
        )
    )
    text = " ".join(d.caveats)
    assert "a silent contract error is not excluded" in text
    assert "7 contract fields default or unknown" in text
    assert d.verdict is None and d.headline == "TASK_FAILURE_OBSERVED / BEHAVIORAL_LIMITATION"


def test_self_trace_never_raises_the_level_or_attributes():
    d = decide(
        Evidence(
            **G,
            self_trace=True,
            nominal="fail",
            d="above_floor",
            d_chains=["x"],
            counterfactual="changes",
        )
    )
    assert d.verdict != "PHYSICS" and d.evidence == "L1"
    d = decide(Evidence(**G, self_trace=True, nominal="pass", d="at_floor"))
    assert d.evidence == "L1" and "SELF_CONSISTENT" in d.findings


OPTS = dict(
    reference=["golden", "harness", "none"],
    mapping=[None, "pass", "fail", "not_covered"],
    nominal=[None, "pass", "fail"],
    task=[None, "pass", "fail"],
    d=[None, "at_floor", "above_floor", "not_measured", "uncalibrated"],
    counterfactual=[None, "changes", "no_change"],
    limit_counterfactual=[None, "changes"],
    source_shows_limitation=[None, True],
    self_trace=[False, True],
)


def test_invariants_over_every_combination():
    keys = list(OPTS)
    n = 0
    for combo in itertools.product(*OPTS.values()):
        ev = Evidence(**dict(zip(keys, combo)))
        d = decide(ev)
        n += 1
        if d.verdict == "PHYSICS":
            assert ev.reference == "golden" and not ev.self_trace and ev.mapping == "pass"
            assert ev.nominal == "fail" and ev.d == "above_floor"
            assert "changes" in (ev.counterfactual, ev.limit_counterfactual)
        if d.verdict == "CONTRACT":
            assert ev.mapping == "fail" and ev.reference != "none"
        if d.verdict == "POLICY_UNDER_TASK":
            assert ev.reference == "golden" and ev.source_shows_limitation and ev.mapping == "pass"
        if ev.reference == "none":
            assert d.evidence in ("L0", "L1") and not d.confident
        if ev.self_trace:
            assert d.evidence in ("L0", "L1")
        # AT9: a task-only failure (nominal passes) never yields PHYSICS
        if ev.nominal == "pass" and ev.task == "fail":
            assert d.verdict != "PHYSICS"
        assert d.exit_code in EXIT.values()
    assert n > 10000


def test_false_confident_attribution_metric():
    t_c = {"kind": "contract", "boundary": "C", "tokens": ["kp scaled x0.7"]}
    t_p = {"kind": "physics", "boundary": "D", "groups": ["contact parameters"]}
    cases = [
        Case("c ok", t_c, "CONTRACT", "C: kp scaled x0.7", [], ["C"], []),
        Case("c wrong term", t_c, "CONTRACT", "C: kd scaled x0.5", [], ["C"], []),
        Case("p ok", t_p, "PHYSICS", "dynamics differ", ["contact parameters"], [], []),
        Case("p wrong group", t_p, "PHYSICS", "dynamics differ", ["armature"], [], []),
        Case("task as physics", {"kind": "task"}, "PHYSICS", "x", [], [], []),
        Case("abstain", t_p, "UNDETERMINED", None, [], [], []),
    ]
    s = score(cases)
    assert s["confident"] == 5 and s["false_confident"] == 3
    assert s["false_confident_attribution_rate"] == pytest.approx(0.6)
    assert s["abstention_rate"] == pytest.approx(1 / 6)
    assert s["detection"]["C"]["rate"] == 1.0 and s["detection"]["D"]["rate"] == pytest.approx(
        2 / 3
    )
