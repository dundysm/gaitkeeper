"""infer: layout from a trace, abstaining while columns are ambiguous (plan 7.4, AT4)."""

import numpy as np
import pytest
from assets import UMJ_G1, URL_G1, need

from sim2sim.cli import main
from sim2sim.infer import infer
from sim2sim.trace import Trace

WIDTHS = [3, 3, 2, 5]  # terms before the last action
A = 5
H = 3
T = 300


def _signals(seed: int = 0, constant_cols: tuple[int, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    terms = [np.cumsum(rng.normal(0, 0.1, (T, d)), 0) for d in WIDTHS]
    for c in constant_cols:
        terms[0][:, c] = 0.5
    act = rng.normal(0, 1, (T, A))
    return np.concatenate(terms, 1), act


def _encode(sig, act, layout, order, init="repeat_first"):
    prev = np.zeros_like(act)
    prev[1:] = act[:-1]
    frame = np.concatenate([sig, prev], 1)  # newest frame: terms then last action
    widths = WIDTHS + [A]
    starts = np.cumsum([0] + widths)
    slots = []  # slot h (oldest first) holds frame t - (H - 1 - h)
    for h in range(H):
        lag = H - 1 - h
        f = np.empty_like(frame)
        f[lag:] = frame[: T - lag]
        f[:lag] = frame[0] if init == "repeat_first" else 0.0
        slots.append(f)
    if order == "newest_first":
        slots = slots[::-1]
    if layout == "time_major":
        obs = np.concatenate(slots, 1)
    else:
        obs = np.concatenate(
            [
                np.concatenate([s[:, a:b] for s in slots], 1)
                for a, b in zip(starts[:-1], starts[1:])
            ],
            1,
        )
    reset = np.zeros(T, bool)
    reset[0] = True
    arrays = {
        "obs": obs.astype(np.float32),
        "action": act.astype(np.float32),
        "reset": reset,
        "episode_step": np.arange(T),
    }
    return Trace(arrays, {}, "harness")


@pytest.mark.parametrize("layout", ["term_major", "time_major"])
@pytest.mark.parametrize("order", ["oldest_first", "newest_first"])
def test_structure_from_obs_and_action_alone(layout, order):
    sig, act = _signals()
    r = infer(_encode(sig, act, layout, order))
    assert r.history_length == H and r.layout == layout and r.order == order
    assert r.init == "repeat_first" and not r.ambiguous_cols and not r.raw_state
    assert r.last_action_cols == H * A
    if layout == "term_major":
        assert [b.width for b in r.blocks] == WIDTHS + [A]
        assert r.blocks[-1].label == "last_action"
    else:
        assert r.frame_width == sum(WIDTHS) + A
        assert r.blocks[-1].label == "last_action" and r.blocks[-1].width == A
    # Random walks carry no raw label, so the result is partial, never a guess.
    assert r.status == "partial"


def test_zero_init_is_recovered():
    sig, act = _signals()
    r = infer(_encode(sig, act, "term_major", "oldest_first", init="zeros"))
    assert r.init == "zeros"


def test_constant_column_abstains_with_no_boundaries():
    # A constant command makes its history columns copies of each other (AT4).
    sig, act = _signals(constant_cols=(1,))
    r = infer(_encode(sig, act, "term_major", "oldest_first"))
    assert r.status == "abstained" and r.ambiguous_cols
    assert r.blocks == [] and r.layout is None and r.init is None


def test_independent_noise_abstains_without_claiming_a_history_length():
    sig, act = _signals()
    tr = _encode(sig, act, "term_major", "oldest_first")
    rng = np.random.default_rng(1)
    tr.arrays["obs"] = tr["obs"] + rng.uniform(-0.01, 0.01, tr["obs"].shape).astype(np.float32)
    r = infer(tr)
    assert r.status == "abstained" and r.history_length is None and r.blocks == []


def test_raw_state_labels_every_term(clean_log, files):
    r = infer(clean_log)
    terms = files.get("policy_io.observation_groups.policy")["terms"]
    assert r.status == "inferred", "\n".join(r.lines())
    got = [(b.width, b.label) for b in r.blocks]
    want = [t["dim"] for t in terms]
    assert [w for w, _ in got] == want
    labels = [lab for _, lab in got]
    assert labels[:3] == ["base_ang_vel", "projected_gravity", "velocity_commands"]
    assert labels[3].startswith("unit circle pair")
    assert labels[4:] == ["joint_pos_rel", "joint_vel_rel", "last_action"]
    assert r.blocks[4].detail["joint_order"] == "unitree_g1_29dof_sdk"


def test_raw_state_names_a_world_frame_gyro(harness):
    from sim2sim.inject import defects

    d = next(d for d in defects() if d.name == "gyro in the world frame")
    r = infer(d.build(harness))
    assert r.blocks[0].label == "base_ang_vel (world frame)"


def test_cli_exit_codes(tmp_path, clean_log):
    p = tmp_path / "log.npz"
    clean_log.save(p)
    assert main(["infer", str(p), "--json", str(tmp_path / "r.json")]) == 0
    assert main(["infer", "--trace", str(p), "--no-raw"]) == 3  # partial without raw state


def test_closed_loop_trace_recovers_the_rl_lab_layout(tmp_path):
    need(UMJ_G1, URL_G1)
    sched = tmp_path / "s.csv"
    sched.write_text(
        "0,0,0,0\n2,0.5,0,0\n5,0.5,0,0.5\n9,0,0.3,0\n12,-0.4,0,0\n15,0.3,-0.25,-0.5\n"
        "19,0.8,0.1,0.3\n23,0,0,0.8\n26,0.1,0,0\n"
    )
    rec = tmp_path / "t.npz"
    main(
        [
            "run",
            "--deploy", str(URL_G1 / "deploy.yaml"),
            "--onnx", str(URL_G1 / "policy.onnx"),
            "--preset", "unitree_rl_lab_g1_29dof_velocity@4960b84",
            "--mjcf", str(UMJ_G1),
            "--schedule", str(sched),
            "--seconds", "30",
            "--record", str(rec),
        ]
    )  # fmt: skip
    r = infer(Trace.load(rec))
    assert r.status == "inferred", "\n".join(r.lines())
    assert (r.history_length, r.layout, r.order, r.init) == (
        5, "term_major", "oldest_first", "repeat_first"
    )  # fmt: skip
    assert [b.label for b in r.blocks] == [
        "base_ang_vel", "projected_gravity", "velocity_commands",
        "joint_pos_rel", "joint_vel_rel", "last_action",
    ]  # fmt: skip
    assert r.blocks[0].detail["scale"] == 0.2 and r.blocks[4].detail["scale"] == 0.05
    assert r.blocks[3].detail["joint_order"].startswith("unitree_g1_29dof_isaac_bfs")
