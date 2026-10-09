"""E4 on the MuJoCo stand-in for a position-implicit drive (plan Appendix B):
two step sizes separate a by-motor-type armature difference from the drive's
discretization terms; one step size cannot."""

import pytest
from assets import GOLDEN_A, MJLAB_ONNX, UMJ_G1, URL_G1, need
from sources import runner_trace

from gaitkeeper.contract import Contract
from gaitkeeper.e4 import e4
from gaitkeeper.inject import compose, physics_edit
from gaitkeeper.models import load_model

BY_TYPE = {"hip": 0.004, "knee": 0.006, "ankle": 0.001}
SCHED = [
    (0.0, (0.0, 0.0, 0.0)),
    (1.0, (0.5, 0.0, 0.0)),
    (4.0, (0.0, 0.4, 0.0)),
    (7.0, (0.0, 0.0, 0.5)),
    (10.0, (0.6, -0.2, 0.4)),
]


def _source(extra_ankle=0.0):
    edits = [physics_edit("frictionless")]
    edits += [physics_edit("armature", value=v, joints=(k,), add=True) for k, v in BY_TYPE.items()]
    if extra_ankle:
        edits.append(physics_edit("armature", value=extra_ankle, joints=("ankle",), add=True))
    return compose(*edits)


def _traces(c, onnx, steps, ankle=True):
    def tr(h, edit):
        return runner_trace(c, UMJ_G1, onnx, SCHED, 13.0, edit, "standin_implicit", timestep=h)

    target = load_model(UMJ_G1).model
    physics_edit("frictionless")(target)
    stock = [tr(h, _source()) for h in steps]
    return target, stock, (tr(steps[0], _source(0.01)) if ankle else None)


@pytest.fixture(scope="module")
def setup():
    """The unitree_rl_lab G1 with its training (asset) gains: kd / kp differs by motor."""
    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.presets import apply_preset
    from gaitkeeper.readers.unitree_deploy import read_unitree_deploy

    c = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")[0]
    apply_preset(c, "unitree_rl_lab_g1_29dof_velocity@4960b84")
    target, stock, ankle = _traces(c, URL_G1 / "policy.onnx", (0.005, 0.0025))
    return c, target, stock, ankle


def _truth(name):
    return -next((v for k, v in BY_TYPE.items() if k in name), 0.0)


def test_two_step_sizes_recover_the_drive_and_the_real_difference(setup):
    c, target, stock, ankle = setup
    r = e4(stock, c, target, ankle=ankle, ankle_change=0.01)
    assert not r.left_out
    assert r.beta == pytest.approx(-1.0, abs=1e-3) and abs(r.alpha) < 1e-3
    assert r.gamma == pytest.approx(-1.0, abs=1e-3)
    assert r.explained and r.resid_b < 1e-5
    for n, v in r.c.items():
        assert v == pytest.approx(_truth(n), abs=1e-4), n
    assert r.ankle_ok and len(r.ankle) == 4
    for v in r.ankle.values():
        assert v == pytest.approx(-0.01, abs=2e-4)


def test_one_ratio_for_every_joint_needs_a_third_step():
    """mjlab sets every gain from one natural frequency: two steps cannot separate
    h kd from h^2 kp, and E4 says so instead of returning a split; three can."""
    need(UMJ_G1, MJLAB_ONNX, GOLDEN_A / "contract.live.yaml")
    c = Contract.load(GOLDEN_A / "contract.live.yaml")
    target, stock, _ = _traces(c, MJLAB_ONNX, (0.005, 0.0025, 0.002), ankle=False)
    r = e4(stock[:2], c, target)
    assert not r.identifiable and not r.explained and "third step size" in r.lines()[0]
    r = e4(stock, c, target)
    assert r.identifiable and r.explained
    assert r.beta == pytest.approx(-1.0, abs=1e-3) and abs(r.alpha) < 1e-3
    for n, v in r.c.items():
        assert v == pytest.approx(_truth(n), abs=1e-4), n


def test_one_step_size_is_refused(setup):
    c, target, stock, _ = setup
    with pytest.raises(ValueError, match="two step sizes"):
        e4([stock[0], stock[0]], c, target)
