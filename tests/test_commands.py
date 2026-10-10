"""What a port does to the harness's command before the policy sees it: shaping and gates."""

import math

import numpy as np

from gaitkeeper.commands import CommandGate, policy_command, shape

S2D = {
    "kind": "speed_to_distance",
    "params": {"pos_p": 0.8, "speed_cap": 0.4, "vx": [-0.25, 0.4], "vy_abs": 0.25},
}
FACE = {"yaw_p": 1.2, "yaw_rate_abs": 0.8, "face_near_m": 0.35, "face_far_m": 1.0}


def test_speed_to_distance_keeps_the_direction_and_sets_the_speed():
    # the command's size does not matter, its direction does
    for c in ([0.3, 0.0, 0.1], [0.9, 0.0, 0.1]):
        assert np.allclose(shape(np.array(c), (0.2, 0.0, 0.2, 0.0), S2D), [0.16, 0.0, 0.1])
    b = 0.4
    out = shape(np.array([math.cos(b), math.sin(b), 0.0]), (5.0, 0.0, 0.0, 0.0), S2D)
    assert np.allclose(out, [0.4 * math.cos(b), 0.4 * math.sin(b), 0.0])
    # clamps: backwards to vx[0], sideways to vy_abs
    assert np.allclose(shape(np.array([-1.0, 0, 0]), (5.0, 0, 0, 0), S2D)[0], -0.25)
    assert np.allclose(shape(np.array([0.0, 1.0, 0]), (5.0, 0, 0, 0), S2D)[1], 0.25)
    # a zero planar command stays zero; with no waypoint the speed is zero
    assert np.allclose(shape(np.array([0.0, 0.0, 0.3]), (1.0, 0.2, 0, 0), S2D), [0, 0, 0.3])
    assert np.allclose(shape(np.array([0.5, 0.0, 0.0]), (0.0, 0, 0, 0), S2D), [0, 0, 0])


def test_speed_to_distance_steers_its_own_yaw():
    sh = {"kind": "speed_to_distance", "params": dict(S2D["params"], yaw=FACE)}
    # standing and turning: yaw_p times the yaw error, clamped
    assert np.allclose(shape(np.array([0, 0, 0.3]), (0, 0.1, 0, 0), sh), [0, 0, 0.12])
    assert np.allclose(shape(np.array([0, 0, 0.3]), (0, 3.0, 0, 0), sh), [0, 0, 0.8])
    # turned enough (the harness's yaw command is zero): no yaw
    assert np.allclose(shape(np.array([0, 0, 0.0]), (0, 0.1, 0, 0), sh), [0, 0, 0])
    # walking far: face the direction of travel, near: aim at the yaw error
    b = 0.3
    c = np.array([math.cos(b), math.sin(b), 0.0])
    assert np.isclose(shape(c, (2.0, 0.0, 0, 0), sh)[2], 1.2 * b)
    assert np.isclose(shape(c, (0.2, 0.0, 0, 0), sh)[2], 0.0)
    w = (0.6 - 0.35) / 0.65
    assert np.isclose(shape(c, (0.6, 0.0, 0, 0), sh)[2], 1.2 * w * b)


def test_gate_on_a_nonzero_command_after_a_warmup():
    g = CommandGate({"on": "command_nonzero", "warmup_s": 0.1}, 0.02)
    cmd = np.array([0.0, 0.0, 0.2])
    assert [g(cmd, None, t) for t in range(7)] == [0, 0, 0, 0, 0, 1, 1]
    assert g(np.zeros(3), None, 8) == 0.0
    out = policy_command(np.array([0.4, 0.1, 0.0]), None, None, g, 3)
    assert np.allclose(out, [0, 0, 0, 0])  # gated shut: zeros and the flag
    out = policy_command(np.array([0.4, 0.1, 0.0]), None, None, g, 6)
    assert np.allclose(out, [0.4, 0.1, 0.0, 1.0])


def test_gate_latched_on_the_task():
    g = CommandGate(
        {
            "on": "task_latch",
            "enter": {"dist": 0.08, "yaw": 0.1},
            "exit": {"dist": 0.04, "yaw": 0.05},
        },
        0.02,
    )
    seq = [(0.0, 0.0), (0.06, 0.0), (0.09, 0.0), (0.06, 0.0), (0.03, 0.07), (0.03, 0.02)]
    got = [g(np.zeros(3), (d, y, 0, 0), t) for t, (d, y) in enumerate(seq)]
    assert got == [0, 0, 1, 1, 1, 0]
    assert g(np.zeros(3), (0.06, 0.0, 0, 0), 0) == 0  # a new episode starts unlatched
