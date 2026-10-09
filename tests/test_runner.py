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
    # a preset names the controller but does not confirm it
    assert ctl["CONTROLLER_ASSUMED"] and ctl["kind"]["from"] == "preset"
    assert ctl["assumed"] == ["kind", "pd_period", "integrator", "torque_limit_at"]
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


def _legs_only(c, unlisted):
    """The contract restricted to its first 15 policy joints in MuJoCo order (legs, waist)."""
    import copy

    from gaitkeeper.contract import Contract

    d = copy.deepcopy(c.to_dict())
    keep = [
        n
        for n in d["policy_io"]["joints"]["names"]
        if "shoulder" not in n and "elbow" not in n and "wrist" not in n
    ]
    d["policy_io"]["joints"]["names"] = keep
    d["policy_io"]["joints"].pop("joint_ids_map", None)
    ctl = d["control"]
    ctl["default_joint_pos"] = {n: ctl["default_joint_pos"][n] for n in keep}
    for k in ("scale", "offset"):
        ctl["actions"]["joint_pos"][k] = {n: ctl["actions"]["joint_pos"][k][n] for n in keep}
    for k in ("kp", "kd"):
        ctl["actuators"][k] = {n: ctl["actuators"][k][n] for n in keep}
    if unlisted is not None:
        ctl["unlisted"] = unlisted
    return Contract.from_dict(d), keep


def test_unlisted_joints_are_held_only_when_the_contract_says_so(url):
    import mujoco

    arms = {"left_elbow_joint": 0.97, "right_elbow_joint": 0.97}
    c, keep = _legs_only(url, {"pose": arms, "kp": 50.0, "kd": {"left_elbow_joint": 2.0}})
    r = Runner(c, UMJ_G1, None)
    m, d, b = r.build(RunConfig(), "native_implicit")
    assert len(b.unlisted) == 29 - len(keep)
    by_q = {q: (a, pose) for a, q, pose in b.unlisted}
    for nm, v in arms.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
        a, pose = by_q[int(m.jnt_qposadr[j])]
        assert pose == v and m.actuator_gainprm[a, 0] == 50.0
    r.reset_state(m, d, b, RunConfig(), r.default, np.random.default_rng(0))
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")
    assert d.qpos[m.jnt_qposadr[j]] == 0.97
    # Without control.unlisted nothing changes: those joints get no torque, as before.
    c0, _ = _legs_only(url, None)
    _, _, b0 = Runner(c0, UMJ_G1, None).build(RunConfig(), "native_implicit")
    assert b0.unlisted == []
