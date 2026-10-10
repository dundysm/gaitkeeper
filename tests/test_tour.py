import math

import numpy as np
import pytest

from gaitkeeper.tour import (
    BENCH_ARM_LEFT,
    BENCH_ARM_RIGHT,
    BENCH_CENTRE,
    BENCH_RADIUS_M,
    WaypointTour,
    bench_arm_walk,
    bench_punches,
    bench_waypoints,
)


def _base(x, y, yaw):
    return np.array([x, y, 0.8, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])


def test_tour_commands_toward_the_waypoint_in_the_body_frame_and_clamps():
    t = WaypointTour([(1.0, 0.0, 0.0)], vx=(-0.5, 1.0), vy_abs=0.3, wz_abs=0.2)
    cmd = t(0.0, _base(0, 0, 0))
    np.testing.assert_allclose(cmd, [1.0, 0.0, 0.0])  # 1.5 * 1 m, clamped to vx max
    # Turned 90 degrees left: the same target is now to the robot's right.
    t2 = WaypointTour([(0.0, -0.1, math.pi / 2)], vy_abs=0.3, wz_abs=0.2)
    cmd = t2(0.0, _base(0, 0, 0))
    assert cmd[1] == pytest.approx(-0.15) and cmd[2] == pytest.approx(0.2)


def test_tour_reached_has_hysteresis_and_report_scores_slot_ends():
    t = WaypointTour([(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)], point_s=1.0)
    assert np.allclose(t(0.0, _base(0, 0, 0)), 0)  # at the target: reached, no command
    assert np.allclose(t(0.5, _base(0.15, 0, 0)), 0)  # inside the exit radius: still reached
    assert t(0.6, _base(0.25, 0, 0))[0] < 0  # left it: drives back
    t(1.0, _base(0.45, 0, 0))  # slot 2 starts
    rep = t.report(None)
    assert rep["outcome"] == "complete" and rep["targets"] == 2
    assert rep["pos_err_cm"] == pytest.approx((25 + 5) / 2)
    assert t.report(1.5)["outcome"] == "fell" and t.report(1.5)["targets"] == 1


def test_benchmark_draws_stay_in_their_ranges():
    for x, y, yaw in bench_waypoints(3):
        assert math.hypot(x - BENCH_CENTRE[0], y - BENCH_CENTRE[1]) <= BENCH_RADIUS_M + 1e-9
        assert -math.pi <= yaw <= math.pi
    p = bench_punches(3, ["pelvis", "torso_link"])
    forces = [np.linalg.norm(q.vector) for q in p]
    assert len(p) == 18 and p[0].t == pytest.approx(0.1) and p[1].t == pytest.approx(5.1)
    assert 500 / 6 - 1e-6 <= forces[0] <= 500 / 3 + 1e-6  # floor 1/3 of 500, scale 0.5..1
    assert max(forces) <= 500 + 1e-6
    lims = {n: (-1.0, 1.0) for n in BENCH_ARM_LEFT}
    walk = bench_arm_walk(3, 5.0, lims)
    for left, right, sign in zip(BENCH_ARM_LEFT, BENCH_ARM_RIGHT, (1, -1, -1, 1, -1, 1, -1)):
        lv = np.array(walk[left])[:, 1]
        assert np.all(np.abs(lv) <= 1.0)
        np.testing.assert_allclose(np.array(walk[right])[:, 1], sign * lv)


def test_runner_takes_a_closed_loop_command_and_reports_the_tour():
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.policy import OnnxPolicy
    from gaitkeeper.readers.unitree_deploy import read_unitree_deploy
    from gaitkeeper.runner import RunConfig, Runner

    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    r = Runner(c, UMJ_G1, OnnxPolicy(URL_G1 / "policy.onnx"))
    tour = WaypointTour([(1.0, 0.0, 0.0)], point_s=3.0, vx=(-0.5, 1.0))
    res = r.run(RunConfig(seconds=3.0, command_source=tour))
    assert res.survived and res.dx > 0.5
    assert res.task["outcome"] == "complete" and res.task["targets"] == 1


def test_tour_command_runs_waypoints_from_a_file_and_takes_the_arms(tmp_path, capsys):
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.cli import main

    wp = tmp_path / "wp.yaml"
    wp.write_text("waypoints:\n  - [0.3, 0.0, 0.0]\n  - [0.3, 0.2, 0.5]\n")
    out = tmp_path / "t.json"
    argv = ["tour", "--deploy", str(URL_G1 / "deploy.yaml"), "--onnx", str(URL_G1 / "policy.onnx")]
    argv += ["--mjcf", str(UMJ_G1), "--waypoints", str(wp), "--point-s", "2", "--seeds", "1"]
    argv += ["--arms", "walk", "--json", str(out)]
    code = main(argv)
    text = capsys.readouterr().out
    assert "arms walk (gains armature" in text and "Mean survival" in text
    import json

    d = json.loads(out.read_text())
    assert d["command"] == "tour" and len(d["runs"]) == 1 and d["seconds"] == 4.0
    assert code in (0, 5)
