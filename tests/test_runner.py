import numpy as np
import pytest
from assets import UMJ_G1, URL_G1, need

from gaitkeeper.presets import apply_preset
from gaitkeeper.readers.unitree_deploy import read_unitree_deploy
from gaitkeeper.runner import External, Push, RunConfig, Runner, _divisor_step, load_schedule

PRESET = "unitree_rl_lab_g1_29dof_velocity@4960b84"


def test_divisor_step():
    assert _divisor_step(0.005, 0.002) == pytest.approx((0.005 / 3, 3))
    assert _divisor_step(0.005, 0.005) == pytest.approx((0.005, 1))
    assert _divisor_step(0.02, 0.002) == pytest.approx((0.002, 10))


def test_load_schedule(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("t,vx,vy,wz\n0,0,0,0\n2.5,0.5,0,0.1\n")
    assert load_schedule(p) == [(0.0, (0.0, 0.0, 0.0)), (2.5, (0.5, 0.0, 0.1))]
    p = tmp_path / "s.yaml"
    p.write_text("schedule: [[0, 0.2, 0, 0], [1, 0, 0.3, 0]]\n")
    assert load_schedule(p)[1] == (1.0, (0.0, 0.3, 0.0))


@pytest.fixture(scope="module")
def url():
    need(UMJ_G1, URL_G1 / "deploy.yaml")
    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    return c


def _runner(c):
    from gaitkeeper.policy import OnnxPolicy

    return Runner(c, UMJ_G1, OnnxPolicy(URL_G1 / "policy.onnx"))


def test_controller_assumed_without_a_source(url):
    r = _runner(url)
    ctl = r.controller(r.choose_backend(None))
    assert ctl["backend"] == "native_implicit" and ctl["CONTROLLER_ASSUMED"]
    assert set(ctl["assumed"]) == {"kind", "pd_period", "integrator", "torque_limit_at"}
    text = "\n".join(Runner.controller_text(ctl))
    assert text.startswith("CONTROLLER_ASSUMED: backend native_implicit; kind unknown (unknown)")


def test_preset_names_the_controller_and_explicit_zoh_changes_the_step(url):
    c = url.copy()
    apply_preset(c, PRESET)
    r = _runner(c)
    ctl = r.controller("native_implicit")
    assert not ctl["CONTROLLER_ASSUMED"] and ctl["kind"]["from"] == "preset"
    m, d, b = r.build(RunConfig(), "explicit_zoh")
    assert b.timestep == pytest.approx(0.005 / 3) and b.pd_every == 3 and b.substeps == 12
    assert "changed from 2 ms to 1.667 ms" in b.timestep_note
    ctl = r.controller("explicit_zoh", b)
    assert (
        ctl["mismatch"] and ctl["CONTROLLER_ASSUMED"]
    )  # implicit PD trained, explicit backend chosen


def test_limits_from_the_mjcf_and_from_the_contract(url):
    c = url.copy()
    apply_preset(c, PRESET)
    r = _runner(c)
    _, _, b = r.build(RunConfig(), "native_implicit")
    i = b.names.index("left_hip_roll_joint")
    assert b.limit[i] == 88.0 and b.limit_from[i] in (
        "joint actuatorfrcrange",
        "motor ctrlrange x gear",
    )
    _, _, b = r.build(RunConfig(limit_source="contract"), "native_implicit")
    assert b.limit[i] == 139.0


def test_held_joints_stay_at_the_hold_pose_and_pushes_apply(url):
    r = _runner(url)
    arms = [n for n in r.names if "wrist" in n]
    res = r.run(
        RunConfig(
            seconds=2.0,
            external=[External(arms, pose="zero", obs="default")],
            record=True,
            pushes=[
                Push(1.0, "velocity", (0.3, 0.0, 0.0)),
                Push(1.5, "force", (0, 50, 0), "torso_link", 0.1),
            ],
        )
    )
    assert res.survived
    tgt = res.log["target"]
    idx = [r.names.index(n) for n in arms]
    assert np.all(tgt[:, idx] == 0.0)
    assert [p["kind"] for p in res.pushes] == ["velocity", "force"]
    obs = res.log["obs"]
    assert obs.shape[1] == 480
