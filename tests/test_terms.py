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
