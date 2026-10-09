import numpy as np
import synth

from gaitkeeper.terms import (
    ObservationBuilder,
    RawState,
    StateLayout,
    TermContext,
    build_observation,
    gait_phase,
)


def test_streaming_builder_matches_batch(truth):
    g = synth.make_golden()
    c = truth
    actions = np.random.default_rng(1).normal(0, 1, (g.n_steps, 29))
    for layout in ("term_major", "time_major"):
        for init in ("repeat_first", "zeros"):
            cc = c.copy()
            grp = cc.get("policy_io.observation_groups.policy")
            grp["history"] = {"length": 3, "layout": layout, "order": "oldest_first", "init": init}
            cc.set("policy_io.observation_groups.policy", grp, "file", "test")
            s = RawState.from_arrays(
                g["qpos"],
                g["qvel"],
                StateLayout.from_meta(g.meta),
                g["command"],
                g["episode_step"],
                g["reset"],
                actions,
            )
            batch, _, _ = build_observation(s, cc)
            b = ObservationBuilder(cc)
            for k in range(len(batch)):
                o = b.step(
                    s.root_quat[k],
                    s.ang_vel_body[k],
                    s.joint_pos[k],
                    s.joint_vel[k],
                    s.joint_names,
                    s.command[k],
                    int(s.episode_step[k]),
                    s.prev_action[k],
                )
                np.testing.assert_allclose(o, batch[k], atol=1e-12)


def _phase_state(steps, cmd=0.5):
    t = len(steps)
    return RawState(
        np.tile([1.0, 0, 0, 0], (t, 1)),
        np.zeros((t, 3)),
        np.zeros((t, 1)),
        np.zeros((t, 1)),
        ["j"],
        np.tile([cmd, 0, 0], (t, 1)),
        np.asarray(steps),
        np.zeros(t, bool),
        np.zeros((t, 1)),
    )


def test_clock_offset_shifts_the_phase():
    ctx = TermContext(["j"], np.zeros(1), 0.02)
    s = _phase_state(np.arange(100))
    base = gait_phase(s, {"period": 0.6}, ctx)
    lead = gait_phase(s, {"period": 0.6, "clock_offset_steps": 2}, ctx)
    np.testing.assert_allclose(lead[:-2], base[2:], atol=1e-6)


def test_accumulated_phase_tracks_the_product_form():
    ctx = TermContext(["j"], np.zeros(1), 0.02)
    s = _phase_state(np.arange(1000))
    acc = gait_phase(s, {"period": 0.6, "arithmetic": "float32_accumulate"}, ctx)
    exact = gait_phase(s, {"period": 0.6, "arithmetic": "float64"}, ctx)
    d = np.abs(acc - exact).max()
    assert 1e-7 < d < 1e-3  # float32 accumulation drifts, slowly
    assert synth.DT == 0.02
