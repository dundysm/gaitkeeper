"""Fixture manifest, download with hash checks, and the first-run CLI paths."""

import hashlib
import re
from pathlib import Path

import pytest

from gaitkeeper import fixtures
from gaitkeeper.cli import main


def test_manifest_pins_every_file():
    sets = fixtures.manifest()
    base = {"g1_rl_lab", "g1_rl_mjlab", "g1_unitree_mujoco", "g1_menagerie"}
    assert {k for k, s in sets.items() if s.group is None} == base
    for s in sets.values():
        assert s.host in fixtures.HOSTS
        if s.group == "golden":  # published golden traces live in a dataset, not on GitHub
            assert s.host == "huggingface"
    for s in sets.values():
        assert re.fullmatch(r"[0-9a-f]{40}", s.commit), s.name
        assert s.files and s.license
        for f in s.files.values():
            assert re.fullmatch(r"[0-9a-f]{64}", f["sha256"])
            assert not f["path"].startswith("/")
        if s.scene:
            assert s.scene in s.files
    # The scenes reference only files the set downloads.
    umj = sets["g1_unitree_mujoco"]
    assert sum(k.startswith("meshes/") for k in umj.files) == 36


def _fake_set(tmp_path, monkeypatch, content=b"policy bytes", pinned=None):
    src = tmp_path / "upstream" / "owner" / "repo" / ("a" * 40) / "dir"
    src.mkdir(parents=True)
    (src / "policy.onnx").write_bytes(content)
    fake = fixtures.FixtureSet(
        "fake",
        "owner/repo",
        "a" * 40,
        "a test set",
        "test license",
        {
            "policy.onnx": {
                "path": "dir/policy.onnx",
                "sha256": pinned or hashlib.sha256(content).hexdigest(),
            }
        },
    )
    monkeypatch.setattr(fixtures, "manifest", lambda: {"fake": fake})
    return "file://" + str(tmp_path / "upstream") + "/{repo}/{commit}/{path}"


def test_fetch_downloads_checks_and_skips(tmp_path, monkeypatch):
    url = _fake_set(tmp_path, monkeypatch)
    root = tmp_path / "data"
    logs = []
    s = fixtures.fetch("fake", root, base_url=url, log=logs.append)
    assert s.path("policy.onnx", root).read_bytes() == b"policy bytes"
    assert any("test license" in m for m in logs)
    logs.clear()
    fixtures.fetch("fake", root, base_url=url, log=logs.append)
    assert logs == []  # present with the right hash: nothing downloaded
    assert fixtures.require("fake", root).present(root)


def test_fetch_refuses_a_file_whose_hash_differs(tmp_path, monkeypatch):
    url = _fake_set(tmp_path, monkeypatch, pinned="0" * 64)
    root = tmp_path / "data"
    with pytest.raises(fixtures.FixtureError, match="does not match the pinned"):
        fixtures.fetch("fake", root, base_url=url, log=lambda m: None)
    assert not any((root / "fake").glob("*"))


def test_missing_set_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GAITKEEPER_DATA", str(tmp_path))
    with pytest.raises(fixtures.FixtureError, match="gaitkeeper fetch g1_rl_lab"):
        fixtures.require("g1_rl_lab")
    with pytest.raises(fixtures.FixtureError, match="known"):
        fixtures.get("nope")


def test_demo_without_files_and_no_fetch_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GAITKEEPER_DATA", str(tmp_path))
    assert main(["demo", "--no-fetch"]) == 2
    err = capsys.readouterr().err
    assert "gaitkeeper fetch g1_rl_lab" in err and "Traceback" not in err


def test_demo_runs_the_issue_145_task(monkeypatch, capsys):
    """The demo's task arguments point at the fetched files and the bundled tour
    (the task itself is covered by test_cli)."""
    from assets import UMJ_G1, URL_G1, need

    need(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx", UMJ_G1)
    import gaitkeeper.cli as cli

    seen = []
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 5)
    assert (
        cli.cmd_demo(cli.argparse.Namespace(seeds=3, workers=None, json=None, no_fetch=True)) == 0
    )
    (argv,) = seen
    assert argv[0] == "task"
    for flag in ("--deploy", "--onnx", "--mjcf", "--schedule"):
        v = argv[argv.index(flag) + 1]
        assert Path(v).exists(), (flag, v)
    assert len(argv[argv.index("--hold") + 1].split(",")) == 14
    out = capsys.readouterr().out
    assert "exits 5" in out and "L1 finding" in out


def test_cli_errors_are_one_line(tmp_path, capsys):
    assert main(["inspect", "--deploy", str(tmp_path / "nope.yaml")]) == 2
    err = capsys.readouterr().err
    assert "--deploy" in err and "no such file" in err
    from assets import URL_G1, need

    need(URL_G1 / "deploy.yaml")
    assert main(["inspect", "--deploy", str(URL_G1 / "deploy.yaml"), "--preset", "nosuch"]) == 2
    err = capsys.readouterr().err
    assert "unknown preset 'nosuch'" in err and "Traceback" not in err


def test_model_without_a_floor_is_flagged(tmp_path):
    from gaitkeeper.cli import _floor_warning

    robot = tmp_path / "robot.xml"
    robot.write_text(
        '<mujoco><worldbody><body name="b"><freejoint/><geom size="0.1"/></body>'
        "</worldbody></mujoco>"
    )
    assert "no floor" in _floor_warning(str(robot))
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/><body name="b"><freejoint/>'
        '<geom size="0.1"/></body></worldbody></mujoco>'
    )
    assert _floor_warning(str(scene)) is None


def test_groups_stay_out_of_all_and_are_fetched_by_name(monkeypatch):
    def fs(name, group=None, host="github"):
        return fixtures.FixtureSet(name, "o/r", "a" * 40, "", "", {}, None, host, group)

    sets = {
        "a": fs("a"),
        "b": fs("b"),
        "g1": fs("g1", "golden", "huggingface"),
        "g2": fs("g2", "golden", "huggingface"),
    }
    monkeypatch.setattr(fixtures, "manifest", lambda: sets)
    assert fixtures.expand(["all"]) == ["a", "b"]
    assert fixtures.expand(["golden"]) == ["g1", "g2"]
    assert fixtures.expand(["a", "golden", "a"]) == ["a", "g1", "g2"]
    with pytest.raises(fixtures.FixtureError, match="groups: golden"):
        fixtures.expand(["nope"])


def test_huggingface_files_resolve_at_the_pinned_revision():
    url = fixtures.HOSTS["huggingface"].format(
        repo="me/traces", commit="b" * 40, path="g1/golden.npz"
    )
    assert url == "https://huggingface.co/datasets/me/traces/resolve/" + "b" * 40 + "/g1/golden.npz"
