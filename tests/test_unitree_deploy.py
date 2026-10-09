import numpy as np
import pytest
import yaml
from assets import UMJLAB_G1, URL_G1, need

from gaitkeeper.presets import apply_preset
from gaitkeeper.readers.unitree_deploy import read_unitree_deploy
from gaitkeeper.tables import G1_29_ISAAC_IDS, G1_29_SDK, H1_SDK, SDK_TABLES


def _yaml(tmp_path, robot_names, jmap, **over):
    n = len(jmap)
    sdk_len = len(robot_names)
    d = {
        "joint_ids_map": jmap,
        "step_dt": 0.02,
        "stiffness": [float(10 + i) for i in range(sdk_len)],  # SDK order
        "damping": [float(i) / 10 for i in range(sdk_len)],
        "default_joint_pos": [float(i) / 100 for i in range(n)],  # policy order
        "commands": {
            "base_velocity": {
                "ranges": {
                    "lin_vel_x": [-0.5, 1.0],
                    "lin_vel_y": [-0.3, 0.3],
                    "ang_vel_z": [-0.2, 0.2],
                }
            }
        },
        "actions": {
            "JointPositionAction": {"scale": [0.25] * n, "offset": [0.0] * n, "clip": None}
        },
        "observations": {
            "base_ang_vel": {"scale": [0.2] * 3, "history_length": 5},
            "projected_gravity": {"scale": [1.0] * 3, "history_length": 5},
            "velocity_commands": {"scale": [1.0] * 3, "history_length": 5},
            "joint_pos_rel": {"scale": [1.0] * n, "history_length": 5},
            "joint_vel_rel": {"scale": [0.05] * n, "history_length": 5},
            "last_action": {"scale": [1.0] * n, "history_length": 5},
        },
    }
    d.update(over)
    p = tmp_path / "deploy.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_g1_names_and_sdk_ordered_gains(tmp_path):
    p = _yaml(tmp_path, G1_29_SDK, list(G1_29_ISAAC_IDS))
    c, findings = read_unitree_deploy(p)
    names = c.get("policy_io.joints.names")
    assert names == [G1_29_SDK[i] for i in G1_29_ISAAC_IDS]
    kp = c.get("control.actuators.kp")
    for sdk_i, n in enumerate(G1_29_SDK):
        assert kp[n] == 10 + sdk_i  # gains follow the SDK index, not the policy index
    default = c.get("control.default_joint_pos")
    assert default[names[3]] == pytest.approx(0.03)  # policy order
    h = c.get("policy_io.observation_groups.policy.history")
    assert h == {
        "length": 5,
        "layout": "term_major",
        "order": "oldest_first",
        "init": "repeat_first",
    }
    assert (
        "observation_manager.h:72"
        in c.prov("policy_io.observation_groups.policy.history.layout").detail
    )
    assert c.get("control.actuators.kind") is None
    assert c.get("policy_io.commands.base_velocity.limit") == {
        "vx": [-0.5, 1.0],
        "vy": [-0.3, 0.3],
        "wz": [-0.2, 0.2],
    }
    assert c.get("policy_io.commands.base_velocity.trained") is None
    assert not findings


def test_gym_history_is_time_major_and_mixed_history_is_unsupported(tmp_path):
    p = _yaml(tmp_path, G1_29_SDK, list(range(29)))
    d = yaml.safe_load(p.read_text())
    d["observations"]["use_gym_history"] = True
    p.write_text(yaml.safe_dump(d))
    c, _ = read_unitree_deploy(p)
    assert c.get("policy_io.observation_groups.policy.history.layout") == "time_major"
    d["observations"]["last_action"]["history_length"] = 1
    p.write_text(yaml.safe_dump(d))
    with pytest.raises(ValueError, match="history lengths differ"):
        read_unitree_deploy(p)


def test_h1_table_skips_the_empty_slot(tmp_path):
    assert len(H1_SDK) == 20 and H1_SDK[9] == ""
    jmap = [i for i in range(20) if i != 9]
    p = _yaml(tmp_path, H1_SDK, jmap)
    c, _ = read_unitree_deploy(p, robot="unitree_h1")
    names = c.get("policy_io.joints.names")
    assert "" not in names and len(names) == 19
    assert c.get("control.actuators.kp")["left_ankle_joint"] == 20.0  # SDK slot 10
    bad = _yaml(tmp_path, H1_SDK, [9] + jmap[1:])
    with pytest.raises(ValueError, match="unused or repeated"):
        read_unitree_deploy(bad, robot="unitree_h1")
    assert set(SDK_TABLES) == {"unitree_g1_29dof", "unitree_h1"}


def test_preset_fills_training_facts_and_keeps_asset_gains_as_alternative():
    need(URL_G1 / "deploy.yaml")
    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    assert c.get("policy_io.graph.inputs")[0]["shape"][-1] == 480
    filled = apply_preset(c, "unitree_rl_lab_g1_29dof_velocity@4960b84")
    assert "control.actuators.kind" in filled
    assert c.get("control.actuators.kind") == "implicit_pd"
    assert c.prov("control.actuators.kind").source == "preset"
    assert c.get("model.effort_limit")["left_hip_roll_joint"] == 139.0
    assert c.get("policy_io.commands.base_velocity.trained")["wz"] == [-0.1, 0.1]
    alt = c.prov("control.actuators.kp").alternatives["asset config"]
    names = c.get("policy_io.joints.names")
    i = names.index("waist_roll_joint")
    assert c.get("control.actuators.kp")["waist_roll_joint"] == 200.0 and alt[i] == 40.0


def test_mjlab_deploy_phase_clock_and_exact_values():
    need(UMJLAB_G1 / "deploy.yaml")
    c, _ = read_unitree_deploy(UMJLAB_G1 / "deploy.yaml", UMJLAB_G1 / "policy.onnx")
    terms = c.get("policy_io.observation_groups.policy.terms")
    gp = next(t for t in terms if t["id"] == "gait_phase")
    assert gp["params"]["clock_offset_steps"] == 2
    assert gp["params"]["stand_threshold"] == 0.1
    assert c.get("control.actions.joint_pos.scale")["left_wrist_pitch_joint"] == 0.07
    assert (
        c.prov("control.actions.joint_pos.scale").resolution is None
    )  # exact: the robot runs 0.07
    assert np.isclose(c.get("control.actuators.kd")["left_wrist_pitch_joint"], 1.1)
