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


def test_a_push_off_the_centre_of_mass_also_twists(url):
    """The benchmark punches a link at its joint's anchor, not at its centre of mass: the
    same force there adds a torque about it."""
    r = _runner(url)
    spin = []
    for point in (None, (0.0, 0.0, 0.3)):
        res = r.run(
            RunConfig(
                seconds=1.2,
                record=True,
                pushes=[Push(1.0, "force", (60.0, 0.0, 0.0), "torso_link", 0.1, point)],
            )
        )
        assert ("point" in res.pushes[0]) == (point is not None)
        spin.append(np.abs(res.log["qvel"][-1][3:6]).max())
    assert abs(spin[1] - spin[0]) > 0.05


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


def test_unlisted_joints_follow_a_trajectory_when_given(url):
    import mujoco

    from gaitkeeper.contract import Contract

    c, keep = _legs_only(url, {"pose": {}, "kp": 80.0, "kd": 4.0})
    d = c.to_dict()
    for t in d["policy_io"]["observation_groups"]["policy"]["terms"]:
        if t["id"] in ("joint_pos_rel", "joint_vel_rel", "last_action"):
            t["dim"] = len(keep)
            if isinstance(t.get("scale"), list):
                t["scale"] = t["scale"][: len(keep)]
    r = Runner(Contract.from_dict(d), UMJ_G1, None)
    traj = {"left_elbow_joint": [[0.0, 0.0], [1.0, 0.8]]}
    res = r.run(RunConfig(seconds=1.0, policy_mode="zero", unlisted_trajectory=traj, record=True))
    m = mujoco.MjModel.from_xml_path(str(UMJ_G1))
    q = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_elbow_joint")]
    qpos = np.asarray(res.log["qpos"])[:, q]
    assert qpos[0] == pytest.approx(0.0, abs=0.05) and qpos[-1] > 0.4
    with pytest.raises(ValueError, match="not unlisted"):
        r.run(RunConfig(seconds=0.1, policy_mode="zero", unlisted_trajectory={"nope": [[0, 0]]}))


def test_waypoint_follow_shaping():
    import math

    import numpy as np

    from gaitkeeper.commands import shape

    sh = {
        "kind": "waypoint_follow",
        "params": {
            "face_far_m": 1.5,
            "face_near_m": 0.4,
            "walk_speed": 0.5,
            "walk_p": 1.5,
            "yaw_p": 1.2,
            "vy_abs": 0.3,
            "yaw_rate_abs": 0.6,
        },
    }
    cmd = np.array([0.4, 0.1, 0.2])
    assert np.allclose(shape(cmd, (0, 0, 0, 0), sh), cmd)  # no waypoint: passed through
    assert np.allclose(shape(cmd, None, None), cmd)
    # straight ahead and near: walk_p * distance, aim at the yaw error
    out = shape(cmd, (0.2, 0.1, 0.2, 0.0), sh)
    assert np.allclose(out, [0.3, 0.0, 0.12])
    # far and to the left: capped speed, gated by cos(bearing), facing the waypoint
    b = 0.5
    out = shape(cmd, (3.0, 0.0, 3 * math.cos(b), 3 * math.sin(b)), sh)
    assert np.allclose(out, [0.5 * math.cos(b), 0.5 * math.cos(b) * math.sin(b), min(1.2 * b, 0.6)])
    # position reached (planar command zero) and yaw reached: no motion
    assert np.allclose(shape(np.zeros(3), (0.05, 0.02, 0.05, 0.0), sh), np.zeros(3))
