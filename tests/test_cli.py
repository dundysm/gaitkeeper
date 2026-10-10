import json

import numpy as np

from gaitkeeper.cli import main


def test_verify_exit_codes(tmp_path, clean_log, files, harness):
    files.save(tmp_path / "c.yaml")
    clean_log.save(tmp_path / "clean.npz")
    assert (
        main(
            [
                "verify",
                str(tmp_path / "clean.npz"),
                "--contract",
                str(tmp_path / "c.yaml"),
                "--json",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    bad = clean_log.slice_steps(0, clean_log.n_steps)
    bad.arrays["target"] = bad["target"] - np.float32(0.1)
    bad.save(tmp_path / "bad.npz")
    # Plan section 4: CONTRACT exits 1, INVALID_INPUT 2.
    assert main(["verify", str(tmp_path / "bad.npz"), "--contract", str(tmp_path / "c.yaml")]) == 1
    (tmp_path / "junk.npz").write_bytes(b"not a trace")
    assert main(["verify", str(tmp_path / "junk.npz"), "--contract", str(tmp_path / "c.yaml")]) == 2
    assert (tmp_path / "r.json").exists()


def test_inspect_writes_contract(tmp_path, files):
    files.save(tmp_path / "c.yaml")
    assert (
        main(["inspect", "--contract", str(tmp_path / "c.yaml"), "--out", str(tmp_path / "o.yaml")])
        == 0
    )
    assert (tmp_path / "o.yaml").read_text().startswith("schema")


def test_task_on_the_issue_145_setup(tmp_path, capsys):
    """The unitree_rl_lab policy on unitree_mujoco's G1, arms held, a waypoint-like
    tour of small commands and in-place turns, then punches: an L1 finding, a
    behavioral limitation, never PHYSICS or CONTRACT."""
    from assets import UMJ_G1, URL_G1, need

    need(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx", UMJ_G1)
    sched = tmp_path / "tour.csv"
    sched.write_text("0,0,0,0\n2,0.15,0,0\n6,0,0,0.2\n10,0.3,0,0\n14,0,0.1,0\n18,0,0,0\n")
    arms = ",".join(
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
        + ("wrist_roll", "wrist_pitch", "wrist_yaw")
    )
    out = tmp_path / "task.json"
    code = main(
        ["task", "--deploy", str(URL_G1 / "deploy.yaml"), "--onnx", str(URL_G1 / "policy.onnx")]
        + ["--preset", "unitree_rl_lab_g1_29dof_velocity@4960b84", "--mjcf", str(UMJ_G1)]
        + ["--schedule", str(sched), "--seconds", "24", "--hold", arms]
        + ["--push-every", "3", "--push-first", "19", "--push-force", "600"]
        + ["--seeds", "2", "--json", str(out)]
    )
    text = capsys.readouterr().out
    j = json.loads(out.read_text())
    dec = j["decision"]
    assert code == 5 and dec["verdict"] is None, text
    assert dec["headline"] == "TASK_FAILURE_OBSERVED / BEHAVIORAL_LIMITATION"
    assert dec["evidence"] == "L1"
    assert "a silent contract error is not excluded" in dec["caveats"]
    assert set(j["task"]["kinds"]) == {"dead zone", "fall"}
    assert "Finding  TASK_FAILURE_OBSERVED / BEHAVIORAL_LIMITATION  (evidence L1, exit 5)" in text


def test_set_states_a_contract_field_as_user_given(tmp_path, capsys):
    from argparse import Namespace

    import pytest

    from gaitkeeper.cli import _contract
    from gaitkeeper.contract import SCHEMA, Contract

    p = tmp_path / "c.yaml"
    Contract({"schema": SCHEMA}).save(p)
    sets = ["timing.policy_dt=0.02", "model.armature={a: 0.01}"]
    c = _contract(Namespace(contract=str(p), preset=[], set=sets))
    assert c.get("timing.policy_dt") == 0.02 and c.prov("timing.policy_dt").source == "user"
    assert c.get("model.armature") == {"a": 0.01}
    with pytest.raises(SystemExit) as e:
        _contract(Namespace(contract=str(p), preset=[], set=["x.y=1"]))
    assert e.value.code == 2 and "top level" in capsys.readouterr().err


def test_wrong_kind_of_input_exits_2_naming_the_flag(tmp_path, capsys):
    from gaitkeeper.cli import main

    bad = tmp_path / "policy.onnx"
    bad.write_text("a: 1\n")
    scene = tmp_path / "scene.xml"
    scene.write_text("a: 1\n")
    assert main(["doctor", "--onnx", str(bad), "--mjcf", str(scene)]) == 2
    assert "--onnx" in capsys.readouterr().err
    cfg = tmp_path / "cfg.yaml"
    cfg.write_bytes(b"\x8c\x00\xff")
    assert main(["inspect", "--config", str(cfg)]) == 2
    assert "--config" in capsys.readouterr().err


def test_missing_config_is_a_usage_error(tmp_path, capsys):
    import pytest

    from gaitkeeper.cli import main

    with pytest.raises(SystemExit) as e:
        main(["inspect"])
    assert e.value.code == 2 and "--config" in capsys.readouterr().err


def test_policy_of_any_extension_and_scene_directory_are_checked(tmp_path, capsys):
    from gaitkeeper.cli import main

    bad = tmp_path / "policy.bin"
    bad.write_text("a: 1\n")
    assert main(["inspect", "--onnx", str(bad)]) == 2
    assert "--onnx" in capsys.readouterr().err
    assert main(["doctor", "--onnx", str(bad), "--mjcf", str(tmp_path)]) == 2
