from pathlib import Path

import pytest

from gaitkeeper.readers.isaaclab_env import read_isaaclab_env, resolve
from gaitkeeper.tables import G1_29_ISAAC

ENV = Path(__file__).parent / "data" / "isaaclab" / "env.yaml"


def test_resolve_matches_whole_names_and_refuses_two_keys():
    names = ["left_knee_joint", "waist_yaw_joint"]
    assert resolve({".*_knee_joint": 0.3}, names, "x", 0.0) == {
        "left_knee_joint": 0.3,
        "waist_yaw_joint": 0.0,
    }
    with pytest.raises(ValueError, match="matches 2 keys"):
        resolve({".*_joint": 1, "left_.*": 2}, names, "x")


def test_reader_reads_training_facts():
    c, findings = read_isaaclab_env(ENV)
    assert c.get("policy_io.joints.names") == list(G1_29_ISAAC)
    assert any("assumed" in f for f in findings) and any("noise" in f for f in findings)
    assert c.get("timing.policy_dt") == pytest.approx(0.02)
    kp, kd = c.get("control.actuators.kp"), c.get("control.actuators.kd")
    assert kp["waist_yaw_joint"] == 200.0 and kp["left_knee_joint"] == 150.0
    assert kp["left_wrist_yaw_joint"] == 40.0 and kd["waist_pitch_joint"] == 5.0
    assert c.get("control.actuators.kind") == "implicit_pd"
    d = c.get("control.default_joint_pos")
    assert d["left_elbow_joint"] == 0.97 and d["waist_yaw_joint"] == 0.0
    g = c.get("policy_io.observation_groups.policy")
    assert g["history"]["length"] == 5 and g["clip_then_scale"] is True
    terms = {t["source_name"]: t for t in g["terms"]}
    assert terms["joint_vel_rel"]["clip"] == [-100.0, 100.0]
    assert terms["gait_phase"]["params"]["clock_offset_steps"] == 0
    cmd = c.get("policy_io.commands.base_velocity")
    assert cmd["trained"]["vx"] == [-0.5, 1.0]  # the curriculum widens lin vel
    assert cmd["trained"]["wz"] == [-0.1, 0.1]  # but not yaw
    assert c.get("model.effort_limit")["left_knee_joint"] == 88.0


def test_reader_takes_the_order_from_a_deploy_yaml(tmp_path):
    from gaitkeeper.tables import G1_29_SDK

    p = tmp_path / "deploy.yaml"
    p.write_text("joint_ids_map: [" + ", ".join(str(i) for i in range(29)) + "]\n")
    c, findings = read_isaaclab_env(ENV, joint_order=p)
    assert c.get("policy_io.joints.names") == list(G1_29_SDK)
    assert not any("assumed" in f for f in findings)


def test_reader_refuses_an_mjlab_config(tmp_path):
    p = tmp_path / "env.yaml"
    p.write_text("decimation: 4\nscene:\n  entities: {}\nobservations:\n  actor: {}\n")
    with pytest.raises(ValueError, match="mjlab"):
        read_isaaclab_env(p)
