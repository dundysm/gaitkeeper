from pathlib import Path

import numpy as np
import pytest

from gaitkeeper.readers.rl_gym_deploy import read_rl_gym_deploy

CFG = Path(__file__).parent / "data" / "rl_gym" / "g1.yaml"


def _deploy_obs(cfg, q, dq, quat, omega, cmd, action, counter):
    """deploy_mujoco.py's observation, transcribed."""
    qw, qx, qy, qz = quat
    g = np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])
    phase = counter * cfg["simulation_dt"] % 0.8 / 0.8
    return np.concatenate(
        [
            omega * cfg["ang_vel_scale"],
            g,
            cmd * np.array(cfg["cmd_scale"]),
            (q - np.array(cfg["default_angles"])) * cfg["dof_pos_scale"],
            dq * cfg["dof_vel_scale"],
            action,
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)],
        ]
    )


def test_reader_rebuilds_the_deploy_script_observation():
    import yaml

    from gaitkeeper.terms import ObservationBuilder

    cfg = yaml.safe_load(CFG.read_text())
    c, findings = read_rl_gym_deploy(CFG)
    assert c.get("timing.policy_dt") == pytest.approx(0.02)
    assert c.get("policy_io.joints.names")[0] == "left_hip_pitch_joint"
    assert c.get("control.unlisted.kp")["waist_yaw_joint"] == 500.0 and findings
    b = ObservationBuilder(c)
    rng = np.random.default_rng(0)
    names = c.get("policy_io.joints.names")
    prev = np.zeros(12)
    for t in range(30):
        q, dq = rng.normal(0, 0.2, 12), rng.normal(0, 1, 12)
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        quat = np.r_[np.cos(0.1), np.sin(0.1) * ax]
        omega, cmd = rng.normal(0, 0.5, 3), rng.normal(0, 0.5, 3)
        want = _deploy_obs(cfg, q, dq, quat, omega, cmd, prev, (t + 1) * 10)
        got = b.step(quat, omega, q, dq, names, cmd, t, prev)
        np.testing.assert_allclose(got, want, atol=1e-9)
        prev = rng.normal(0, 0.5, 12)


def test_reader_refuses_a_fork_with_another_layout(tmp_path):
    p = tmp_path / "fork.yaml"
    p.write_text(CFG.read_text().replace("num_obs: 47", "num_obs: 90"))
    with pytest.raises(ValueError, match="another observation"):
        read_rl_gym_deploy(p)
