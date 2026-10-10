import numpy as np
import pytest

from gaitkeeper.terms import (
    RawState,
    StateLayout,
    TermContext,
    assemble,
    base_ang_vel,
    gait_phase,
    projected_gravity,
    stack_history,
    term_slices,
)


def _state(quat, w=None, cmd=None, ep=None, n_j=2):
    t = len(quat)
    return RawState(
        np.asarray(quat, float),
        np.zeros((t, 3)) if w is None else np.asarray(w, float),
        np.zeros((t, n_j)),
        np.zeros((t, n_j)),
        [f"j{i}" for i in range(n_j)],
        np.zeros((t, 3)) if cmd is None else np.asarray(cmd, float),
        np.arange(t) if ep is None else np.asarray(ep),
        np.zeros(t, bool),
        np.zeros((t, n_j)),
    )


CTX = TermContext(["j0", "j1"], np.zeros(2), 0.02)


def test_projected_gravity_upright_and_rolled():
    g = projected_gravity(_state([[1, 0, 0, 0]]), {}, CTX)
    assert np.allclose(g, [[0, 0, -1]])
    a = np.pi / 2  # roll 90 degrees about x: gravity moves to -y in the body frame
    g = projected_gravity(_state([[np.cos(a / 2), np.sin(a / 2), 0, 0]]), {}, CTX)
    assert np.allclose(g, [[0, -1, 0]], atol=1e-12)


def test_body_and_world_frame_gyro_differ_under_yaw():
    a = np.pi / 2  # yaw 90 degrees
    s = _state([[np.cos(a / 2), 0, 0, np.sin(a / 2)]], w=[[1.0, 0, 0]])
    assert np.allclose(base_ang_vel(s, {}, CTX), [[1, 0, 0]])
    world = TermContext(["j0", "j1"], np.zeros(2), 0.02, imu_frame="world")
    assert np.allclose(base_ang_vel(s, {}, world), [[0, 1, 0]], atol=1e-12)


def test_layout_reads_xyzw_quaternions():
    lay = StateLayout(["j0"], quat_order="xyzw")
    qpos = np.array([[0, 0, 0, 0.0, 0.0, 0.0, 1.0, 0.0]])  # identity in xyzw
    s = RawState.from_arrays(
        qpos,
        np.zeros((1, 7)),
        lay,
        np.zeros((1, 3)),
        np.zeros(1),
        np.ones(1, bool),
        np.zeros((1, 1)),
    )
    assert np.allclose(s.root_quat, [[1, 0, 0, 0]])


def test_gait_phase_stand_rule_and_float32_arithmetic():
    ep = np.array([0, 15, 999])
    cmd = np.array([[0.05, 0, 0], [0.5, 0, 0], [0.5, 0, 0]])
    p = {"period": 0.6, "stand_threshold": 0.1}
    out = gait_phase(_state(np.tile([1.0, 0, 0, 0], (3, 1)), cmd=cmd, ep=ep), p, CTX)
    assert np.all(out[0] == 0)  # below the stand threshold
    assert np.allclose(out[1], [np.sin(np.pi), np.cos(np.pi)], atol=1e-6)
    f64 = gait_phase(
        _state(np.tile([1.0, 0, 0, 0], (3, 1)), cmd=cmd, ep=ep), {**p, "arithmetic": "float64"}, CTX
    )
    assert 1e-6 < np.abs(out[2] - f64[2]).max() < 1e-4  # float32 argument rounding at 20 s


def test_stack_history_orders_and_inits():
    x = np.arange(4, dtype=float)[:, None]
    reset = np.array([True, False, False, True])
    h = stack_history(x, reset, 3, "repeat_first", "oldest_first")
    assert h[2, :, 0].tolist() == [0, 1, 2] and h[3, :, 0].tolist() == [3, 3, 3]
    assert stack_history(x, reset, 3, "repeat_first", "newest_first")[2, :, 0].tolist() == [2, 1, 0]
    assert stack_history(x, reset, 3, "zeros", "oldest_first")[3, :, 0].tolist() == [0, 0, 3]


@pytest.mark.parametrize("layout", ["term_major", "time_major"])
def test_term_slices_index_the_assembled_vector(layout):
    terms = [{"id": "a", "dim": 2}, {"id": "b", "dim": 1}]
    vals = {"a": np.array([[1.0, 2.0], [3.0, 4.0]]), "b": np.array([[9.0], [8.0]])}
    hist = {"length": 2, "layout": layout, "order": "oldest_first", "init": "repeat_first"}
    obs, _ = assemble(vals, terms, np.array([True, False]), hist)
    cols = term_slices(terms, hist)
    assert sorted(np.concatenate(list(cols.values())).tolist()) == list(range(6))
    assert set(obs[1, cols["b"]].tolist()) == {9.0, 8.0}


def test_gait_phase_legs_is_a_two_leg_clock_half_a_period_apart():
    from gaitkeeper.terms import gait_phase_legs

    s = _state([[1, 0, 0, 0]] * 4, ep=np.array([0, 5, 10, 25]))
    ctx = TermContext(["j0", "j1"], np.zeros(2), policy_dt=0.02)
    out = gait_phase_legs(s, {"period": 1.0}, ctx)
    ph = 2 * np.pi * np.array([0, 0.1, 0.2, 0.5])
    np.testing.assert_allclose(out[:, 0], np.sin(ph), atol=1e-12)
    np.testing.assert_allclose(out[:, 1], -np.sin(ph), atol=1e-12)
    np.testing.assert_allclose(out[:, 2], np.cos(ph), atol=1e-12)
    np.testing.assert_allclose(out[:, 3], -np.cos(ph), atol=1e-12)


def test_base_lin_vel_is_the_root_velocity_in_the_root_frame():
    from gaitkeeper.terms import base_lin_vel

    yaw90 = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]  # wxyz, 90 degrees about z
    s = _state([yaw90])
    s.lin_vel_world = np.array([[1.0, 0.0, 0.2]])
    out = base_lin_vel(s, {}, TermContext(["j0", "j1"], np.zeros(2), policy_dt=0.02))
    np.testing.assert_allclose(out, [[0.0, -1.0, 0.2]], atol=1e-12)
    s.lin_vel_world = None
    with pytest.raises(ValueError, match="linear velocity"):
        base_lin_vel(s, {}, TermContext(["j0", "j1"], np.zeros(2), policy_dt=0.02))


def test_raw_state_from_arrays_keeps_the_linear_velocity():
    qpos = np.zeros((2, 9))
    qpos[:, 3] = 1.0
    qvel = np.zeros((2, 8))
    qvel[:, 0] = [0.3, 0.4]
    s = RawState.from_arrays(
        qpos,
        qvel,
        StateLayout(joint_names=["j0", "j1"]),
        np.zeros((2, 3)),
        np.arange(2),
        np.array([True, False]),
        np.zeros((2, 2)),
    )
    np.testing.assert_allclose(s.lin_vel_world[:, 0], [0.3, 0.4])


def _contract_for(terms, history=None):
    from gaitkeeper.contract import Contract

    c = Contract({})
    c.set("policy_io.joints.names", ["j0", "j1"], "file", "t")
    c.set("control.default_joint_pos", {"j0": 0.1, "j1": -0.2}, "file", "t")
    c.set("timing.policy_dt", 0.02, "file", "t")
    c.set(
        "policy_io.observation_groups.policy",
        {"terms": terms, "history": history or {"length": 1}},
        "file",
        "t",
    )
    return c


def _walk(T=40, seed=1):
    rng = np.random.default_rng(seed)
    q = np.cumsum(rng.normal(0, 0.01, (T, 2)), axis=0)
    a = rng.normal(0, 0.5, (T, 2))
    cmd = np.zeros((T, 3))
    cmd[:, 0] = np.where(np.arange(T) % 9 < 6, rng.uniform(0.0, 1.2, T), 0.0)
    reset = np.zeros(T, bool)
    reset[25] = True
    ep = np.zeros(T, int)
    for t in range(1, T):
        ep[t] = 0 if reset[t] else ep[t - 1] + 1
    prev = np.zeros_like(a)
    prev[1:] = a[:-1]
    prev[reset] = 0.0
    s = RawState(
        np.tile([1.0, 0, 0, 0], (T, 1)),
        np.zeros((T, 3)),
        q,
        np.zeros((T, 2)),
        ["j0", "j1"],
        cmd,
        ep,
        reset,
        prev,
        a,
    )
    return s, q, a, cmd, ep, reset


STATEFUL_TERMS = [
    {"id": "last_action", "source_name": "a2", "dim": 2, "params": {"lag": 2}},
    {"id": "joint_vel_diff", "dim": 2},
    {
        "id": "gait_phase_speed",
        "dim": 2,
        "params": {"period_knots": [[0.1, 0.8], [0.74, 0.4]], "stand_speed": 0.1},
    },
    {
        "id": "gait_phase_legs",
        "dim": 4,
        "params": {
            "period": 0.8,
            "stand": {
                "eps_planar": 0.01,
                "eps_yaw": 0.01,
                "hold": [0.25, 0.75],
                "resume": [0.0, 0.5],
            },
        },
    },
]


def test_stateful_terms_follow_their_definitions():
    from gaitkeeper.terms import term_values

    s, q, a, cmd, ep, reset = _walk()
    v = term_values(s, STATEFUL_TERMS, CTX)
    # the action two steps back, zeros until an episode has one
    for t in range(len(ep)):
        want = a[t - 2] if ep[t] >= 2 else np.zeros(2)
        assert np.allclose(v["a2"][t], want)
    # velocity by difference of positions, zero at an episode's first step
    for t in range(len(ep)):
        want = np.zeros(2) if ep[t] == 0 else (q[t] - q[t - 1]) / 0.02
        assert np.allclose(v["joint_vel_diff"][t], want)
    # the clock: no advance at an episode's first step or below stand speed; the period
    # interpolates the knots
    ph = 0.0
    for t in range(len(ep)):
        if ep[t] == 0:
            ph = 0.0
        assert np.allclose(
            v["gait_phase_speed"][t], [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], atol=1e-5
        )
        sp = abs(cmd[t, 0])
        if ep[t] > 0 and sp >= 0.1:
            ph = (ph + 0.02 / np.interp(sp, [0.1, 0.74], [0.8, 0.4])) % 1.0
    # the two-leg clock: held while the command says stand, restarted on the first step after
    standing = False
    for t in range(len(ep)):
        if ep[t] == 0:
            pa, standing = -0.02 / 0.8, False
        pa = (pa + 0.02 / 0.8) % 1.0
        pb = (pa + 0.5) % 1.0
        if abs(cmd[t, 0]) < 0.01:
            pa, pb, standing = 0.25, 0.75, True
        elif standing:
            pa, pb, standing = 0.0, 0.5, False
        A, B = 2 * np.pi * pa, 2 * np.pi * pb
        want = [np.sin(A), np.sin(B), np.cos(A), np.cos(B)]
        assert np.allclose(v["gait_phase_legs"][t], want, atol=1e-6), t
    assert (np.abs(cmd[:, 0]) < 0.01).any() and (np.abs(cmd[:, 0]) >= 0.01).any()


def test_builder_matches_the_whole_trace_for_stateful_terms():
    from gaitkeeper.terms import ObservationBuilder, build_observation

    s, q, a, cmd, ep, reset = _walk()
    c = _contract_for(STATEFUL_TERMS, {"length": 3, "layout": "time_major"})
    whole, _, _ = build_observation(s, c)
    b = ObservationBuilder(c)
    for t in range(len(ep)):
        got = b.step(
            [1.0, 0, 0, 0],
            np.zeros(3),
            q[t],
            np.zeros(2),
            ["j0", "j1"],
            cmd[t],
            int(ep[t]),
            s.prev_action[t],
        )
        assert np.allclose(got, whole[t]), t


def test_joint_terms_can_observe_joints_no_action_drives():
    from gaitkeeper.terms import ObservationBuilder, build_observation

    T = 6
    rng = np.random.default_rng(3)
    q = rng.normal(0, 0.1, (T, 3))
    v = rng.normal(0, 1.0, (T, 3))
    s = RawState(
        np.tile([1.0, 0, 0, 0], (T, 1)),
        np.zeros((T, 3)),
        q,
        v,
        ["j0", "j1", "waist"],
        np.zeros((T, 3)),
        np.arange(T),
        np.zeros(T, bool),
        np.zeros((T, 2)),
    )
    joints = ["j0", "j1", "waist"]
    terms = [
        {"id": "joint_pos_rel", "dim": 3, "params": {"joints": joints, "default": {"waist": 0.05}}},
        {"id": "joint_vel_rel", "dim": 3, "params": {"joints": joints}},
    ]
    c = _contract_for(terms)
    obs, _, _ = build_observation(s, c)
    assert np.allclose(obs[:, :3], q - [0.1, -0.2, 0.05])
    assert np.allclose(obs[:, 3:], v)
    b = ObservationBuilder(c)
    assert b.extra_joints == ["waist"]
    for t in range(T):
        got = b.step([1.0, 0, 0, 0], np.zeros(3), q[t], v[t], joints, np.zeros(3), t, np.zeros(2))
        assert np.allclose(got, obs[t])


def test_a_port_that_builds_its_first_observation_late():
    from gaitkeeper.terms import ObservationBuilder, build_observation

    s, q, a, cmd, ep, reset = _walk(T=30)
    terms = [{"id": "joint_pos_rel", "dim": 2}, {"id": "velocity_commands", "dim": 3}]
    hist = {"length": 3, "layout": "time_major", "init": "zeros", "first_frame": "zeros"}
    c = _contract_for(terms, hist)
    whole, _, _ = build_observation(s, c)
    assert not whole[0].any() and not whole[25].any()  # nothing at an episode's first step
    assert np.allclose(whole[1, :10], 0.0) and np.allclose(whole[1, 10:12], q[1] - [0.1, -0.2])
    b = ObservationBuilder(c)
    for t in range(30):
        got = b.step(
            [1.0, 0, 0, 0],
            np.zeros(3),
            q[t],
            np.zeros(2),
            ["j0", "j1"],
            cmd[t],
            int(ep[t]),
            s.prev_action[t],
        )
        assert np.allclose(got, whole[t]), t


def test_gated_command_terms():
    from gaitkeeper.terms import term_values

    T = 8
    gate = np.array([0, 0, 1, 1, 1, 0, 1, 1], float)
    cmd = np.c_[np.full((T, 3), 0.3) * gate[:, None], gate]
    s = _state(np.tile([1.0, 0, 0, 0], (T, 1)), cmd=cmd)
    terms = [
        {"id": "command_gate", "dim": 1},
        {"id": "gait_phase_gated", "dim": 2, "params": {"period": 0.1, "advance_first": True}},
        {"id": "velocity_commands", "dim": 3},
    ]
    v = term_values(s, terms, CTX)
    assert v["command_gate"][:, 0].tolist() == gate.tolist()
    assert np.allclose(v["velocity_commands"], cmd[:, :3])
    clk, want = 0.0, []
    for g in gate:
        clk += 0.02 * g  # the clock runs only while the gate is open, before it is read
        want.append([np.sin(2 * np.pi * (clk % 0.1) / 0.1), np.cos(2 * np.pi * (clk % 0.1) / 0.1)])
    assert np.allclose(v["gait_phase_gated"], want, atol=1e-5)


def test_feet_clock_warps_stance_and_holds_at_a_zero_command():
    from gaitkeeper.terms import term_values, warp_stance

    T = 30
    cmd = np.full((T, 3), 0.2)
    cmd[10:14] = 0.0
    s = _state(np.tile([1.0, 0, 0, 0], (T, 1)), cmd=cmd)
    p = {
        "frequency": 1.5,
        "stance_ratio": 0.6,
        "offsets": [0.5, 0.0],
        "start": 0.3,
        "stand": {"hold": 0.3},
    }
    v = term_values(s, [{"id": "gait_phase_feet", "dim": 2, "params": p}], CTX)["gait_phase_feet"]
    g, want = 0.3, []
    for t in range(T):
        g = (g + 0.03) % 1.0
        feet = [(g + 0.5) % 1.0, g]
        if not cmd[t].any():
            g, feet = 0.3, [0.3, 0.3]
        want.append(np.sin(2 * np.pi * warp_stance(np.array(feet), 0.6)))
    assert np.allclose(v, want)
    assert np.allclose(warp_stance(np.array([0.0, 0.3, 0.6, 0.8]), 0.6), [0, 0.25, 0.5, 0.75])
