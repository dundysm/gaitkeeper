from pathlib import Path

import numpy as np
import pytest

from gaitkeeper.readers.legged_gym import read_legged_gym, urdf_dof_order

D = Path(__file__).parent / "data" / "legged_gym"


def _read(**kw):
    return read_legged_gym(
        D / "g1_config.py",
        D / "base_config.py",
        urdf=D / "g1_12dof.urdf",
        env_py=D / "g1_env.py",
        **kw,
    )


def test_urdf_dof_order_is_depth_first():
    names = urdf_dof_order(D / "g1_12dof.urdf")
    assert names[:3] == ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint"]
    assert names[6] == "right_hip_pitch_joint" and len(names) == 12


def test_config_classes_inherit_and_gains_match_by_substring():
    c, findings = _read()
    assert findings == []
    kp = c.get("control.actuators.kp")
    assert kp["left_knee_joint"] == 150.0 and kp["right_ankle_roll_joint"] == 40.0
    assert c.get("timing.policy_dt") == pytest.approx(0.02)  # base dt 0.005, decimation 4
    act = c.get("control.actions.joint_pos")
    assert act["scale"]["left_hip_pitch_joint"] == 0.25
    assert act["clip"][0] == pytest.approx([-0.1 - 25.0, -0.1 + 25.0])
    cmd = c.get("policy_io.commands.base_velocity")
    assert cmd["heading"] == "on" and cmd["trained"]["vx"] == [-1.0, 1.0]
    ids = [t["id"] for t in c.get("policy_io.observation_groups.policy.terms")]
    assert ids[0] == "base_ang_vel" and ids[-1] == "gait_phase"
    assert c.get("control.unlisted.kp")["waist_yaw_joint"] == 500.0


def test_observation_is_scaled_then_clipped():
    from gaitkeeper.terms import ObservationBuilder

    c, _ = _read()
    b = ObservationBuilder(c)
    names = c.get("policy_io.joints.names")
    obs = b.step(
        np.array([1.0, 0, 0, 0]),
        np.zeros(3),
        np.zeros(12),
        np.full(12, 3000.0),
        names,
        np.zeros(3),
        0,
        np.zeros(12),
    )
    assert obs[21] == pytest.approx(100.0)  # 3000 * 0.05 = 150, clipped to 100


def test_reader_refuses_an_unknown_layout(tmp_path):
    p = tmp_path / "cfg.py"
    p.write_text(
        (D / "g1_config.py").read_text().replace("num_observations = 47", "num_observations = 50")
    )
    with pytest.raises(ValueError, match="fits neither"):
        read_legged_gym(p, D / "base_config.py", urdf=D / "g1_12dof.urdf")


def _variant(tmp_path, old, new):
    src = (D / "g1_config.py").read_text()
    assert old in src
    p = tmp_path / "g1_config.py"
    p.write_text(src.replace(old, new))
    return read_legged_gym(
        p, D / "base_config.py", urdf=D / "g1_12dof.urdf", env_py=D / "g1_env.py"
    )


def test_overlapping_gain_keys_follow_legged_gym_last_match(tmp_path):
    # legged_robot.py applies every matching key in turn, so the last one wins, and
    # damping is read with the same key.
    c, findings = _variant(
        tmp_path,
        'stiffness = {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100,',
        'stiffness = {"hip": 50, "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100,',
    )
    assert c.get("control.actuators.kp")["left_hip_pitch_joint"] == 100.0
    assert any("left_hip_pitch_joint" in f and "'hip_pitch'" in f for f in findings)


def test_constant_arithmetic_is_read_and_other_expressions_are_refused(tmp_path):
    c, _ = _variant(tmp_path, "action_scale = 0.25", "action_scale = 1 / 4")
    assert c.get("control.actions.joint_pos")["scale"]["left_knee_joint"] == 0.25
    with pytest.raises(ValueError, match="control.action_scale = SCALE"):
        _variant(tmp_path, "action_scale = 0.25", "action_scale = SCALE")
