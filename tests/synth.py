"""Synthetic traces and contracts for tests: no simulator, no policy file."""

from __future__ import annotations

import numpy as np

from gaitkeeper.contract import SCHEMA, Contract
from gaitkeeper.tables import G1_29_SDK
from gaitkeeper.terms import StateLayout
from gaitkeeper.trace import Trace

T = 480
DT = 0.02
RESETS = (0, 240)
SCHEDULE = [  # (first row, vx, vy, wz)
    (0, 0.0, 0.0, 0.0),
    (60, 0.05, 0.0, 0.0),
    (90, 0.5, 0.0, 0.0),
    (150, 0.0, 0.0, 0.5),
    (200, 0.0, 0.3, 0.0),
    (240, 0.0, 0.0, 0.0),
    (300, 0.8, -0.2, 0.3),
    (400, 0.0, 0.0, -0.4),
]
TERMS = [
    {"id": "base_ang_vel", "dim": 3, "scale": 1.0, "clip": None, "params": {}},
    {"id": "projected_gravity", "dim": 3, "scale": 1.0, "clip": None, "params": {}},
    {"id": "velocity_commands", "dim": 3, "scale": 1.0, "clip": None, "params": {}},
    {
        "id": "gait_phase",
        "dim": 2,
        "scale": 1.0,
        "clip": None,
        "params": {
            "period": 0.6,
            "stand_threshold": 0.1,
            "clock": "episode_steps_since_reset",
            "arithmetic": "float32",
        },
    },
    {"id": "joint_pos_rel", "dim": 29, "scale": 1.0, "clip": None, "params": {}},
    {"id": "joint_vel_rel", "dim": 29, "scale": 1.0, "clip": None, "params": {}},
    {"id": "last_action", "dim": 29, "scale": 1.0, "clip": None, "params": {"reset": "zeros"}},
]
N_OBS = 3 + 3 + 3 + 2 + 29 * 3


def euler_to_quat(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=1,
    )


def truth_values() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    return {
        "default": np.round(rng.uniform(-0.4, 0.6, 29), 6),
        "scale": rng.choice([0.548, 0.351, 0.439, 0.0745], 29) + rng.uniform(-3e-4, 3e-4, 29),
        "kp": rng.choice([40.179, 99.098, 28.501, 14.251, 16.778], 29)
        + rng.uniform(-3e-4, 3e-4, 29),
        "kd": rng.choice([2.558, 6.309, 1.814, 0.907, 1.068], 29) + rng.uniform(-3e-4, 3e-4, 29),
    }


def make_contract(rounded: bool) -> Contract:
    """The truth (rounded=False) or a files-style contract printed to 3 decimals."""
    v = truth_values()
    names = list(G1_29_SDK)
    r = 0.0005 if rounded else None

    def m(x: np.ndarray, digits: int = 3) -> dict[str, float]:
        return {n: float(np.round(a, digits) if rounded else a) for n, a in zip(names, x)}

    c = Contract({"schema": SCHEMA})
    c.set("timing.policy_dt", DT, "file", "test")
    c.set("policy_io.joints.names", names, "file", "test")
    c.set(
        "policy_io.imu",
        {"frame": "body", "rotation_in_root": [1.0, 0.0, 0.0, 0.0]},
        "default",
        "test",
    )
    c.set(
        "policy_io.observation_groups.policy",
        {
            "terms": TERMS,
            "history": {
                "length": 1,
                "layout": "term_major",
                "order": "oldest_first",
                "init": "repeat_first",
            },
        },
        "file",
        "test",
    )
    c.set("control.default_joint_pos", m(v["default"]), "file", "test", resolution=r)
    yaml_scale = [float(np.round(x, 2)) for x in v["scale"]]
    c.set(
        "control.actions.joint_pos.scale",
        m(v["scale"]),
        "file",
        "test",
        resolution=r,
        alternatives={"onnx": [float(np.round(x, 3)) for x in v["scale"]], "yaml": yaml_scale},
    )
    c.set("control.actions.joint_pos.offset", m(v["default"]), "file", "test", resolution=r)
    c.set("control.actuators.kp", m(v["kp"]), "file", "test", resolution=r)
    c.set("control.actuators.kd", m(v["kd"]), "file", "test", resolution=r)
    return c


class LinearPolicy:
    """Deterministic stand-in for an exported policy."""

    def __init__(self, n_in: int = N_OBS, n_out: int = 29, seed: int = 3) -> None:
        self.n_in = n_in
        self.w = np.random.default_rng(seed).normal(0, 0.3, (n_in, n_out))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(obs, np.float64) @ self.w * 2.0).astype(np.float32) * 1.5


def make_golden() -> Trace:
    """A golden-shaped trace with raw state only; obs and actions are filled by the harness builder."""
    t = np.arange(T) * DT
    cmd = np.zeros((T, 3))
    for i, (row, vx, vy, wz) in enumerate(SCHEDULE):
        end = SCHEDULE[i + 1][0] if i + 1 < len(SCHEDULE) else T
        cmd[row:end] = (vx, vy, wz)
    reset = np.zeros(T, bool)
    reset[list(RESETS)] = True
    ep = np.zeros(T, int)
    for k in range(T):
        ep[k] = 0 if reset[k] else ep[k - 1] + 1
    yaw = np.cumsum(cmd[:, 2]) * DT * 1.6 + 0.3
    quat = euler_to_quat(0.06 * np.sin(3.1 * t), 0.05 * np.sin(2.3 * t + 1.0), yaw)
    rng = np.random.default_rng(11)
    w = np.stack([0.4 * np.sin(5 * t), 0.3 * np.cos(4 * t), cmd[:, 2] + 0.1 * np.sin(7 * t)], 1)
    v = truth_values()
    ph, fr = rng.uniform(0, 6, 29), rng.uniform(2, 9, 29)
    jp = v["default"][None] + 0.3 * np.sin(fr[None] * t[:, None] + ph[None])
    jv = 0.3 * fr[None] * np.cos(fr[None] * t[:, None] + ph[None])
    qpos = np.concatenate([np.zeros((T, 3)), quat, jp], 1).astype(np.float32)
    qvel = np.concatenate([np.zeros((T, 3)), w, jv], 1).astype(np.float32)
    arrays = {
        "obs": np.zeros((T, N_OBS), np.float32),
        "action": np.zeros((T, 29), np.float32),
        "command": cmd.astype(np.float32),
        "qpos": qpos,
        "qvel": qvel,
        "reset": reset,
        "episode_step": ep,
    }
    meta = {"state_layout": StateLayout(list(G1_29_SDK)).to_meta()}
    return Trace(arrays, meta, "golden")
