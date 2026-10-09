"""The PHYSICS counterfactual end to end: statistics, the golden scenario, group
swapping, positive controls, and the acceptance tests AT1, AT9 and AT15."""

import math
from functools import partial

import pytest
from assets import GOLDEN_A, MENAGERIE_G1, MJLAB_ONNX, UMJ_G1, need
from sources import concat, external, runner_trace, schedule_of

from gaitkeeper import diagnose as D
from gaitkeeper import residual as R
from gaitkeeper.compare import verify
from gaitkeeper.contract import Contract
from gaitkeeper.counterfactual import (
    GROUPS,
    fisher,
    golden_scenario,
    group_differs,
    swap_edit,
)
from gaitkeeper.inject import compose, physics_edit
from gaitkeeper.models import load_model
from gaitkeeper.policy import OnnxPolicy
from gaitkeeper.task import TaskSpec
from gaitkeeper.trace import Trace

SEEDS6 = tuple(range(1, 7))
DEAD_ZONE = TaskSpec(
    "dead zone",
    7.0,
    [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.05, 0.0, 0.0)), (4.0, (0.0, 0.0, 0.08))],
    [],
)


def golden_a():
    need(GOLDEN_A / "golden.npz", GOLDEN_A / "model_patch.json", MJLAB_ONNX)
    return Trace.load(GOLDEN_A), Contract.load(GOLDEN_A / "contract.live.yaml")


def test_fisher_exact():
    assert fisher(0, 12, 0, 12) == 1.0
    assert fisher(8, 12, 0, 12) == pytest.approx(0.0013460, abs=1e-6)
    assert fisher(6, 6, 0, 6) == pytest.approx(2 / math.comb(12, 6), rel=1e-9)
    assert fisher(0, 6, 6, 6, side="less") == pytest.approx(1 / math.comb(12, 6), rel=1e-9)
    assert fisher(1, 12, 0, 12) == 1.0


def test_golden_scenario_judges_only_where_the_source_met_the_bar():
    tr, _ = golden_a()
    segs = {sg.cmd: sg for sg in golden_scenario(tr).segment_list()}
    # the mjlab policy has a 0.05 dead zone by design: the source misses these
    assert segs[(0.05, 0.0, 0.0)].judged[0] is False
    assert segs[(0.0, 0.06, 0.0)].judged[1] is False
    assert segs[(0.0, 0.0, 0.08)].judged[2] is False
    assert all(segs[(0.5, 0.0, 0.0)].judged) and all(segs[(0.0, 0.0, 0.5)].judged)
    assert segs[(0.0, 0.0, 0.0)].judge_stand


def test_group_swap_copies_the_source_values():
    need(GOLDEN_A / "model_patch.json", MENAGERIE_G1)
    src = load_model(GOLDEN_A).model
    assert not [g for g in GROUPS if group_differs(g, src, src)]
    dst = load_model(MENAGERIE_G1).model
    differ = [g for g in GROUPS if group_differs(g, dst, src)]
    assert {"armature", "mass and inertia", "contact parameters"} <= set(differ)
    swap_edit(dst, str(GOLDEN_A), tuple(differ))
    assert not [g for g in GROUPS if group_differs(g, dst, src)]


@pytest.mark.parametrize(
    "edit, groups, part",
    [
        (physics_edit("floor_friction", mu=0.1), ["contact parameters"], None),
        (
            physics_edit("body_mass", body="torso_link", delta=15.0),
            ["mass and inertia"],
            "mass and inertia: body masses",
        ),
    ],
    ids=["floor friction 0.1", "torso +15 kg"],
)
def test_physics_positive_controls(edit, groups, part):
    """A known physics change on the target, contract right: PHYSICS, and the group
    whose swap restores the outcome is the one that was changed."""
    tr, c = golden_a()
    dg = D.diagnose_trace(
        tr, c, OnnxPolicy(MJLAB_ONNX), str(MJLAB_ONNX), str(GOLDEN_A), SEEDS6, target_edit=edit
    )
    dec = dg.decision
    assert dec.verdict == "PHYSICS" and dec.exit_code == 3, dg.lines()
    assert dg.cf.changes and dg.cf.localized == groups
    if part:
        assert [p.name for p in dg.cf.parts if p.restores] == [part]
    assert dg.d.status == "above_floor"


def test_at9_matched_model_and_dead_zone_is_never_physics():
    """AT9: right contract, the source's own model, commands inside the dead zone."""
    tr, c = golden_a()
    dg = D.diagnose_trace(
        tr, c, OnnxPolicy(MJLAB_ONNX), str(MJLAB_ONNX), str(GOLDEN_A), SEEDS6, task=DEAD_ZONE
    )
    dec = dg.decision
    assert dg.d.status == "at_floor" and not dg.cf.changes
    assert set(dg.task.kinds()) == {"dead zone"}
    assert dec.verdict == "POLICY_UNDER_TASK" and dec.exit_code == 4, dg.lines()


def test_at1_altered_physics_is_never_a_contract_finding(tmp_path):
    """AT1: armature 0 and floor friction 2.0, the contract right. Neither a trace
    recorded on that physics nor a target with it yields CONTRACT."""
    tr, c = golden_a()
    need(UMJ_G1)
    edit = compose(physics_edit("armature", value=0.0), physics_edit("floor_friction", mu=2.0))
    dg = D.diagnose_trace(
        tr, c, OnnxPolicy(MJLAB_ONNX), str(MJLAB_ONNX), str(GOLDEN_A), SEEDS6, target_edit=edit
    )
    assert dg.decision.verdict != "CONTRACT" and dg.evidence.mapping == "pass", dg.lines()
    s1 = [
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (0.5, 0.0, 0.0)),
        (4.0, (0.0, 0.4, 0.0)),
        (7.0, (0.0, 0.0, 0.5)),
    ]
    s2 = [(0.0, (0.0, 0.0, 0.0)), (1.0, (-0.4, 0.0, 0.0)), (4.0, (0.6, -0.2, 0.4))]
    eps = [
        runner_trace(c, UMJ_G1, MJLAB_ONNX, s1, 9.0, edit),
        runner_trace(c, UMJ_G1, MJLAB_ONNX, s2, 7.0, edit, seed=1),
    ]
    src = external(concat(*eps), tmp_path / "src", UMJ_G1, edit, schedule_of((s1, 9.0), (s2, 7.0)))
    rep = verify(src, c, OnnxPolicy(MJLAB_ONNX))
    assert rep.verdict == "PASS", rep.summary()


def test_at15_standin_drive_with_a_dead_zone_failure_is_never_physics(tmp_path, monkeypatch):
    """AT15: the source integrates the drive as a position-implicit stand-in, so D
    is far above floor on every chain while behavior is unchanged; the task fails
    for another cause (commands inside the dead zone). PHYSICS must not follow."""
    _, c = golden_a()
    need(UMJ_G1)
    edit = physics_edit("frictionless", timestep=0.005)
    s1 = [
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (0.5, 0.0, 0.0)),
        (3.0, (0.0, 0.4, 0.0)),
        (5.0, (0.0, 0.0, 0.5)),
        (7.0, (0.4, 0.0, 0.5)),
        (9.0, (0.0, 0.0, 0.0)),
    ]
    s2 = [
        (0.0, (0.0, 0.0, 0.0)),
        (1.0, (0.8, -0.3, 0.3)),
        (3.0, (-0.4, 0.0, 0.0)),
        (5.0, (0.0, -0.4, 0.0)),
        (7.0, (0.0, 0.0, 0.0)),
    ]

    def episodes(backend):
        return concat(
            runner_trace(c, UMJ_G1, MJLAB_ONNX, s1, 11.0, edit, backend),
            runner_trace(c, UMJ_G1, MJLAB_ONNX, s2, 9.0, edit, backend, seed=1),
        )

    native = episodes("native_implicit")
    src = external(
        episodes("standin_implicit"),
        tmp_path / "standin",
        UMJ_G1,
        edit,
        schedule_of((s1, 11.0), (s2, 9.0)),
    )
    # A floor for the stand-in source engine: the same runner with the native drive.
    target = load_model(UMJ_G1).model
    edit(target)
    fl = R.calibrate([native], lambda t: c, lambda t: target)
    floors = {src.meta["engine"]: fl}
    monkeypatch.setattr(D, "dynamics_residual", partial(D.dynamics_residual, floors=floors))
    dg = D.diagnose_trace(
        src,
        c,
        OnnxPolicy(MJLAB_ONNX),
        str(MJLAB_ONNX),
        str(UMJ_G1),
        tuple(range(1, 13)),
        task=DEAD_ZONE,
        target_edit=edit,
    )
    ev, dec = dg.evidence, dg.decision
    assert ev.mapping == "pass"
    assert ev.d == "above_floor" and {"left leg", "right leg"} <= set(ev.d_chains)
    assert ev.nominal == "pass" and ev.counterfactual == "no_change"
    assert ev.task == "fail" and set(ev.task_kinds) == {"dead zone"}
    assert dec.verdict != "PHYSICS" and dec.exit_code != 3, dg.lines()
    assert dec.verdict == "POLICY_UNDER_TASK"
    assert any(f.startswith("D above floor") for f in dec.findings)  # reported, not a cause
    assert "dynamics" not in (dec.cause or "")
