import shutil
from pathlib import Path

import numpy as np
import pytest

TOY = Path(__file__).parent / "data" / "twb_toy" / "policy.cpp"
needs_cxx = pytest.mark.skipif(
    not (shutil.which("g++") or shutil.which("clang++")), reason="needs a C++ compiler"
)


def test_kernel_launches_become_loops():
    from gaitkeeper.readers.twb_adapter import _launches

    src = "k_obs<<<blocks, dim3(a, b)>>>(x, f(y, z));\nother();"
    assert _launches(src) == "GK_LAUNCH((blocks), (dim3(a, b)), k_obs(x, f(y, z)));\nother();"


def test_constant_and_index_terms():
    from gaitkeeper.terms import RawState, TermContext, term_values

    s = RawState(
        root_quat=np.array([[1.0, 0, 0, 0]]),
        ang_vel_body=np.zeros((1, 3)),
        joint_pos=np.zeros((1, 1)),
        joint_vel=np.zeros((1, 1)),
        joint_names=["j"],
        command=np.array([[0.5, 0.1, 0.2]]),
        episode_step=np.array([0]),
        reset=np.array([True]),
        prev_action=np.zeros((1, 1)),
    )
    ctx = TermContext(["j"], np.zeros(1), 0.02, np.array([1.0, 0, 0, 0]), "body")
    terms = [
        {
            "id": "velocity_commands",
            "source_name": "cmd_yaw_first",
            "dim": 3,
            "scale": 1.0,
            "params": {"index": [2, 0, 1]},
        },
        {
            "id": "constant",
            "source_name": "height",
            "dim": 1,
            "scale": 1.0,
            "params": {"value": [0.75]},
        },
    ]
    v = term_values(s, terms, ctx)
    np.testing.assert_allclose(v["cmd_yaw_first"], [[0.2, 0.5, 0.1]])
    np.testing.assert_allclose(v["height"], [[0.75]])


@needs_cxx
def test_probe_reads_the_toy_adapter():
    from gaitkeeper.readers.twb_adapter import Adapter
    from gaitkeeper.readers.twb_probe import probe

    r = probe(Adapter(TOY))
    assert r.name == "toy" and r.obs_dim == 57 and r.owned == 15
    assert r.p2m == [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 12, 13, 14]
    assert set(r.action_scale) == {0.25} and r.action_offset[6] == pytest.approx(0.3)
    assert r.action_clip[0] == pytest.approx([-0.1 - 1.25, -0.1 + 1.25])
    ids = [(t["source_name"], t.get("index")) for t in r.terms]
    assert ids[:5] == [
        ("base_ang_vel", None),
        ("projected_gravity", None),
        ("velocity_commands", [2, 0, 1]),
        ("constant", None),
        ("gait_phase", None),
    ]
    assert r.terms[3]["value"] == [0.75] and r.terms[0]["scale"] == [0.25] * 3
    assert r.clock == {
        "id": "gait_phase",
        "period": 0.8,
        "clock_offset_steps": 1,
        "stand_threshold": 0.1,
    }
    assert r.kp[:3] == [100.0, 100.0, 100.0] and r.history["length"] == 1
    assert any("constants the port feeds" in f for f in r.findings)


@needs_cxx
def test_toy_contracts_rebuild_the_adapter_observation(tmp_path):
    from assets import UMJ_G1, need

    need(UMJ_G1)
    from gaitkeeper.readers.twb_adapter import Adapter
    from gaitkeeper.readers.twb_probe import read_twb_adapter, verify

    r, cs, v = read_twb_adapter(TOY, UMJ_G1)
    assert v["ok"], v
    port = cs["port"]
    assert port.get("control.ownership.owned") == "all"
    unl = port.get("control.unlisted")
    assert "left_elbow_joint" in unl["pose"] and unl["kp"]["left_elbow_joint"] > 0
    # a wrong scale is caught
    bad = port.copy()
    bad.data["policy_io"]["observation_groups"]["policy"]["terms"][0]["scale"] = [0.2] * 3
    assert verify(Adapter(TOY), bad)["per_term"]["base_ang_vel"] > 1e-3


@needs_cxx
def test_adapter_command_writes_both_contracts(tmp_path, capsys):
    from assets import UMJ_G1, need

    need(UMJ_G1)
    from gaitkeeper.cli import main

    code = main(["adapter", str(TOY), "--mjcf", str(UMJ_G1), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "MATCHES ADAPTER" in out and "velocity_commands[3]*" in out
    assert (tmp_path / "toy.trained.yaml").exists() and (tmp_path / "toy.port.yaml").exists()


def _load_toy(cache: str) -> str:
    import os

    os.environ["GAITKEEPER_CACHE"] = cache
    from gaitkeeper.readers.twb_adapter import Adapter

    Adapter(TOY)
    return "ok"


@needs_cxx
def test_concurrent_builds_never_load_a_partial_library(tmp_path):
    """Several processes building the same adapter into an empty cache all load it."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("spawn")) as ex:
        out = list(ex.map(_load_toy, [str(tmp_path)] * 8))
    assert out == ["ok"] * 8
    left = sorted(p.name for p in (tmp_path / "twb_adapters").iterdir())
    assert len(left) == 1 and left[0].endswith(".so"), left


@needs_cxx
def test_harness_holds_unowned_waist_at_the_policys_gains():
    """main.cpp (benchmark@4ed23c2): legs and waist always take kp()/kd(); arms keep the
    harness's armature gains unless the policy owns all 29 motors."""
    import dataclasses

    from assets import UMJ_G1, need

    need(UMJ_G1)
    from gaitkeeper.readers.twb_adapter import Adapter
    from gaitkeeper.readers.twb_probe import contracts, probe

    r = dataclasses.replace(probe(Adapter(TOY)), owned=12)  # waist now held by the harness
    port = contracts(r, UMJ_G1, TOY)["port"]
    held = {j: e for e in port.get("control.ownership.external") for j in e["joints"]}
    assert held["waist_yaw_joint"]["kp"]["waist_yaw_joint"] == 200.0
    assert held["waist_yaw_joint"]["kd"]["waist_yaw_joint"] == 5.0
    assert held["waist_yaw_joint"]["pose"]["waist_yaw_joint"] == 0.0  # the harness stance
    arm = port.get("control.unlisted")["kp"]["left_elbow_joint"]
    assert 0 < arm < 100  # armature gains, not a policy gain


TOY2 = Path(__file__).parent / "data" / "twb_toy2" / "policy.cpp"


@needs_cxx
def test_probe_reads_a_port_that_keeps_its_own_state():
    """Lagged action, velocity by difference, a speed clock, padding in every frame of a
    time-major history and a waypoint follower, all measured, not assumed."""
    from gaitkeeper.readers.twb_adapter import Adapter
    from gaitkeeper.readers.twb_probe import probe

    r = probe(Adapter(TOY2))
    ids = [t["id"] for t in r.terms]
    assert ids == [
        "last_action",
        "velocity_commands",
        "constant",
        "joint_pos_rel",
        "joint_vel_diff",
        "projected_gravity",
        "gait_phase_speed",
        "base_ang_vel",
        "constant",
    ]
    assert r.action_lag == 2
    assert r.history == {
        "length": 3,
        "layout": "time_major",
        "order": "oldest_first",
        "init": "repeat_first",
    }
    assert r.clock["period_knots"] == [[0.15, 1.0], [0.2, 1.0], [0.825, 0.5], [2.0, 0.5]]
    assert r.clock["stand_speed"] == 0.15 and r.clock["speed"] == "norm3"
    assert r.clock["advance_first"] is False
    assert r.shaping["params"] == {
        "walk_p": 1.2,
        "walk_speed": 0.6,
        "yaw_p": 1.0,
        "yaw_rate_abs": 0.5,
        "vy_abs": 0.25,
        "face_near_m": 0.5,
        "face_far_m": 2.0,
    }
    assert r.action_scale == [0.4] * 12


@needs_cxx
def test_the_stateful_toy_contract_rebuilds_its_observation():
    from assets import UMJ_G1, need

    need(UMJ_G1)
    from gaitkeeper.readers.twb_adapter import Adapter
    from gaitkeeper.readers.twb_probe import read_twb_adapter, verify

    r, cs, v = read_twb_adapter(TOY2, UMJ_G1)
    assert v["ok"], v
    # a wrong steering gain is caught on random tasks
    bad = cs["port"].copy()
    bad.data["policy_io"]["commands"]["base_velocity"]["shaping"]["params"]["walk_p"] = 1.5
    assert verify(Adapter(TOY2), bad)["per_term"]["velocity_commands"] > 1e-3
