"""The Isaac Lab recorder's CPU parts. The Isaac parts run only on a GPU (docs/GPU_SESSION.md);
these tests make sure the module imports without Isaac and that its pure helpers are right."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from assets import URL_G1, need

from gaitkeeper.recorders import isaaclab as rec
from gaitkeeper.recorders.schedule import (
    RL_LAB_EPISODE_S,
    RL_LAB_PUSHES,
    RL_LAB_SCHEDULE,
    command_at,
    excitation_checklist,
)


def test_module_imports_without_isaac():
    code = (
        "import sys, gaitkeeper.recorders.isaaclab\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('isaaclab', 'isaacsim', 'omni', 'torch', 'unitree_rl_lab', 'carb')]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_assemble_state_uses_the_mujoco_free_joint_layout():
    q = np.array([2.0, 0.0, 0.0, 0.0])  # unnormalized identity
    qpos, qvel = rec.assemble_state(
        [1.0, 2.0, 0.8], q, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], np.arange(29), -np.arange(29)
    )
    assert qpos.shape == (36,) and qvel.shape == (35,)
    np.testing.assert_allclose(qpos[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(qpos[:3], [1.0, 2.0, 0.8])
    np.testing.assert_allclose(qvel[:6], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(qpos[7:], np.arange(29))
    np.testing.assert_allclose(qvel[6:], -np.arange(29))


def test_punches_are_seeded_horizontal_and_spaced():
    a = rec.punches(3, 27.0, 3.0, 34.0, 600.0, 0.1, "torso_link")
    b = rec.punches(3, 27.0, 3.0, 34.0, 600.0, 0.1, "torso_link")
    c = rec.punches(4, 27.0, 3.0, 34.0, 600.0, 0.1, "torso_link")
    assert a == b and a != c
    assert [p[0] for p in a] == [27.0, 30.0, 33.0]
    for _, dur, body, f in a:
        assert dur == 0.1 and body == "torso_link" and f[2] == 0.0
        assert np.hypot(f[0], f[1]) == pytest.approx(600.0)


def _rows(vx, steps=50, envs=4, act_std=0.0, qd=0.0, fell=None):
    rng = np.random.default_rng(0)
    vel = np.zeros((steps, envs, 3))
    vel[..., 0] = vx
    acts = rng.normal(0.0, act_std, (steps, envs, 29)) if act_std else np.zeros((steps, envs, 29))
    jv = np.full((steps, envs, 29), qd)
    return vel, acts, jv, np.zeros(envs, bool) if fell is None else fell


def test_summarize_skips_fallen_envs_and_flags_standstill():
    vel, acts, jv, fell = _rows(0.0)
    vel[:, 0, 0] = 5.0  # a fallen env must not count
    fell = np.array([True, False, False, False])
    s = rec.summarize(vel, acts, jv, fell, tail=20)
    assert s["envs"] == 4 and s["fell"] == 1
    assert s["vx"] == pytest.approx(0.0) and s["standstill"]

    s = rec.summarize(*_rows(0.4, act_std=0.3, qd=1.0), tail=20)
    assert s["vx"] == pytest.approx(0.4) and not s["standstill"]

    s = rec.summarize(*_rows(0.4, fell=np.ones(4, bool)), tail=20)
    assert s == {"envs": 4, "fell": 4}


def test_verdict_lines_name_the_dead_zone_either_way():
    stand = {"envs": 16, "fell": 0, "vx": 0.0, "wz": 0.0, "standstill": True}
    walk = {"envs": 16, "fell": 0, "vx": 0.47, "wz": 0.0, "standstill": False}
    moving = {"envs": 16, "fell": 0, "vx": 0.12, "wz": 0.0, "standstill": False}
    yaw = {"envs": 16, "fell": 0, "vx": 0.0, "wz": 0.15, "standstill": False}
    res = {
        "asset config": [
            ("fwd 0.15 (MuJoCo: stands)", stand),
            ("fwd 0.50 (MuJoCo: walks 0.48)", walk),
            ("yaw 0.2 in place (MuJoCo: 0.02)", yaw),
        ],
        "deploy.yaml": [
            ("fwd 0.15 (MuJoCo: stands)", moving),
            ("fwd 0.50 (MuJoCo: walks 0.48)", walk),
        ],
        "broken": [("fwd 0.50 (MuJoCo: walks 0.48)", {"envs": 16, "fell": 16})],
    }
    text = "\n".join(rec.verdict_lines(res))
    assert "stands still, as in MuJoCo" in text
    assert "moves at 0.12 m/s, unlike MuJoCo" in text
    assert "0.150 rad/s" in text
    assert "broken gains: does not walk" in text
    labels = {lab for lab, _ in rec.CHECK_COMMANDS}
    for rows in res.values():
        assert {lab for lab, _ in rows} <= labels


def test_arm_joints_match_the_demo_harness():
    from gaitkeeper.cli import DEMO_ARMS

    assert rec.ARM_JOINTS == DEMO_ARMS and len(rec.ARM_JOINTS) == 14


def test_rl_lab_schedule_turns_past_90_degrees_at_the_policys_own_turn_rate():
    for _, vx, vy, wz in RL_LAB_SCHEDULE:
        assert -0.5 <= vx <= 1.0 and abs(vy) <= 0.3 and abs(wz) <= 0.2
    # The policy as measured in MuJoCo: 0.13 rad/s for walk + 0.2 yaw, 0.02 in place.
    # The heading must still pass 90 degrees before the episode resets.
    dt = 0.02
    t = np.arange(0, 36.0, dt)
    cmd = np.stack([command_at(RL_LAB_SCHEDULE, x) for x in t])
    walking = np.linalg.norm(cmd[:, :2], axis=1) > 0.1
    rate = np.where(walking, 0.65, 0.1) * cmd[:, 2]
    reset = np.zeros(len(t), bool)
    reset[0] = True
    reset[int(RL_LAB_EPISODE_S / dt)] = True
    yaw = np.zeros(len(t))
    for k in range(1, len(t)):
        yaw[k] = 0.0 if reset[k] else yaw[k - 1] + rate[k - 1] * dt
    quat = np.stack([np.cos(yaw / 2), 0 * yaw, 0 * yaw, np.sin(yaw / 2)], 1)
    angv = np.zeros((len(t), 3))
    angv[:, :2] = 0.1  # stands in for gait motion
    angv[:, 2] = rate
    items = {e.name: e for e in excitation_checklist(cmd, quat, angv, reset, 0.01)}
    assert all(e.ok for e in items.values()), items
    for start, dur, _, _ in RL_LAB_PUSHES:
        assert walking[int(start / dt)] and start + dur < t[-1]


def test_cli_parses_without_isaac(tmp_path):
    a = rec.parse_args(["check", "--onnx", "p.onnx", "--deploy", "d.yaml"])
    assert a.fn is rec.check and a.envs == 16 and a.episode_s > 1e5
    a = rec.parse_args(
        [
            "record",
            "--onnx",
            "p.onnx",
            "--out",
            str(tmp_path),
            "--hold",
            "arms",
            "--sim-dt",
            "0.0025",
        ]
    )
    assert a.fn is rec.record and a.hold.split(",") == rec.ARM_JOINTS and a.sim_dt == 0.0025
    assert a.episode_s == RL_LAB_EPISODE_S and a.gains == "asset"
    a = rec.parse_args(
        ["record", "--onnx", "p.onnx", "--out", "x", "--schedule", "t.yaml", "--seconds", "34"]
    )
    assert a.episode_s == 35.0  # a task tour runs without a timeout reset
    with pytest.raises(SystemExit):
        rec.parse_args(["record", "--onnx", "p.onnx", "--out", "x", "--gains", "deploy"])


def test_deploy_gains_cover_all_29_joints():
    need(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    kp, kd = rec.deploy_gains(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    assert len(kp) == len(kd) == 29
    assert all(v > 0 for v in kp.values()) and all(v >= 0 for v in kd.values())
    assert set(rec.ARM_JOINTS) <= set(kp)


def test_floor_calibration_writes_the_sources_armature_into_the_mjcf(tmp_path):
    from assets import UMJ_G1

    need(UMJ_G1)
    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    import calibrate_floor
    import mujoco

    from gaitkeeper.trace import Trace

    names = ["left_knee_joint", "right_ankle_pitch_joint"]
    (tmp_path / "isaac_model.json").write_text(
        json.dumps({"joint_names": names, "armature": [0.123, 0.0456]})
    )
    tr = Trace({}, {}, "golden", tmp_path)
    m = calibrate_floor.source_model(str(UMJ_G1), tr, frictionless=True)
    for n, v in zip(names, (0.123, 0.0456)):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert m.dof_armature[m.jnt_dofadr[j]] == pytest.approx(v)
    assert (m.dof_frictionloss == 0).all()
