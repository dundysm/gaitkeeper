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
    assert code == 0 and "Verified" in out and "velocity_commands[3]*" in out
    assert (tmp_path / "toy.trained.yaml").exists() and (tmp_path / "toy.port.yaml").exists()
