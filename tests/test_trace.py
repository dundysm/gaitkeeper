import numpy as np

from sim2sim.trace import Trace, check_state_increments


def test_state_increment_check_names_the_velocity_convention():
    h, n = 0.005, 200
    v = np.cumsum(np.random.default_rng(0).normal(0, 0.1, (n, 3)), 0)
    q = np.zeros((n, 3))
    for s in range(1, n):
        q[s] = q[s - 1] + h * v[s]  # semi-implicit Euler: position uses the new velocity
    qpos = np.concatenate([np.zeros((n, 7)), q], 1)
    qvel = np.concatenate([np.zeros((n, 6)), v], 1)
    chk = check_state_increments(qpos, qvel, h, np.ones(n - 1, bool))
    assert chk.convention == "v_next" and chk.max_abs < 1e-12
    qd = np.delete(qpos, 50, axis=0)  # a skipped physics step
    vd = np.delete(qvel, 50, axis=0)
    assert check_state_increments(qd, vd, h, np.ones(n - 2, bool)).convention == "neither"


def test_harness_and_golden_roundtrip(tmp_path, clean_log, harness):
    p = tmp_path / "log.npz"
    clean_log.save(p)
    t = Trace.load(p)
    assert t.kind == "harness" and np.array_equal(t["obs"], clean_log["obs"])
    harness.g.save(tmp_path / "golden")
    g = Trace.load(tmp_path / "golden")
    assert g.kind == "golden" and g.validate() == []


def test_validate_reports_missing_keys():
    t = Trace({"obs": np.zeros((2, 1))}, {}, "golden")
    assert "missing key action" in t.validate()
