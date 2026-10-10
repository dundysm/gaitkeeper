import json
import math
import textwrap

from gaitkeeper.bench import (
    FINDING_WIDTH,
    BenchReport,
    attribute,
    doctor_markdown,
    step_losses,
)


def _st(surv, complete=0, n=2):
    return {
        "runs": [{}] * n,
        "seconds": 20.0,
        "mean_survival_s": surv,
        "complete": complete,
        "pos_err_cm": math.nan,
        "yaw_err_deg": 4.0,
    }


def test_attribution_names_the_costly_steps_largest_first():
    st = {"own": _st(20, 2), "arms_hold": _st(19), "arms_walk": _st(6), "punches": _st(4)}
    f = attribute(st, 20.0)
    assert len(f) == 1 and f[0].startswith("the random arm walk costs 13.0 s")
    st["punches"] = _st(0.5)
    f = attribute(st, 20.0)
    assert [x.split(" costs")[0] for x in f] == ["the random arm walk", "punching"]


def test_attribution_says_when_the_policy_falls_on_its_own_and_when_the_port_helps():
    f = attribute({"own": _st(8), "punches": _st(7), "port": _st(16)}, 20.0)
    assert f[0].startswith("falls in its own setup")
    assert any("the port survives 9.0 s longer" in x for x in f)


def test_report_gate_and_formats():
    rep = BenchReport("p", 20.0, {"own": _st(20, 2), "punches": _st(5, 0)})
    rep.findings = attribute(rep.stages, 20.0)
    assert not rep.gate()
    text = "\n".join(rep.lines())
    assert "plus punches (full benchmark)" in text and "Finding   punching costs 15.0 s" in text
    md = rep.markdown()
    assert "| own setup | 20.0 | 2/2 | – | 4 |" in md
    json.dumps(rep.to_json())
    rep.stages["port"] = _st(20, 2)
    assert rep.gate()


def test_doctor_markdown_has_each_section_and_the_verdict_last():
    rows = [
        (
            "arms its own",
            {
                "mean_survival_s": 20.0,
                "seconds": 20.0,
                "complete": 2,
                "runs": [1, 2],
                "pos_err_cm": 12.0,
            },
        ),
        (
            "with punches",
            {
                "mean_survival_s": 5.0,
                "seconds": 20.0,
                "complete": 0,
                "runs": [1, 2],
                "pos_err_cm": math.nan,
            },
        ),
    ]
    md = doctor_markdown(
        "p",
        20.0,
        ["policy_io.obs.history"],
        ["vx +0.40 m/s"],
        ["yaw stops responding"],
        rows,
        "It falls on the tour with nothing added.",
    )
    assert md.startswith("# gaitkeeper doctor: p") and "Evidence L1" in md
    assert "- `policy_io.obs.history`" in md
    assert "DEAD ZONE  vx +0.40 m/s" in md and "Finding    yaw stops responding" in md
    assert "| Setting | Mean survival (s) | Complete | Pos err (cm) |" in md
    assert "| with punches | 5.0 | 0/2 | – |" in md  # nan reads as a dash, as in bench
    assert "## Summary" in md and "It falls on the tour with nothing added." in md
    order = [
        md.index(s)
        for s in ["## Contract", "## Command response", "## Waypoint tour", "## Summary"]
    ]
    assert order == sorted(order)
    # The summary says the answer rests on defaults, so it is not read on its own.
    assert "1 field(s) no file states, filled from defaults for this run" in md
    # And it repeats the tour steps that cost survival.
    assert "With punches costs it 15 s (20.0 to 5.0 s)." in md


def test_doctor_markdown_names_the_policy_file_and_wraps_the_terminal():
    rows = [
        (
            "arms its own",
            {
                "mean_survival_s": 20.0,
                "seconds": 20.0,
                "complete": 2,
                "runs": [1, 2],
                "pos_err_cm": 12.0,
            },
        ),
        (
            "with punches",
            {
                "mean_survival_s": 5.0,
                "seconds": 20.0,
                "complete": 0,
                "runs": [1, 2],
                "pos_err_cm": math.nan,
            },
        ),
    ]
    md = doctor_markdown(
        "my_policy",
        20.0,
        [],
        [],
        [],
        rows,
        "It falls on the tour with nothing added.",
        source="policies/my_policy/policy.yaml",
    )
    # A report is attached to a PR, so the heading and the policy line name the file.
    assert md.startswith("# gaitkeeper doctor: my_policy")
    assert "Policy: `policies/my_policy/policy.yaml`" in md

    sentence = step_losses(rows)[0][1]
    assert sentence == "With punches costs it 15 s (20.0 to 5.0 s)."
    # Wrapped for a terminal, every line fits the width the project reports to.
    assert all(len(line) <= FINDING_WIDTH for line in textwrap.wrap(sentence, FINDING_WIDTH))


def test_step_losses_ignores_a_stage_under_the_notable_threshold():
    rows = [
        ("a", {"mean_survival_s": 20.0, "complete": 1, "runs": [1], "pos_err_cm": 0.0}),
        ("b", {"mean_survival_s": 16.0, "complete": 1, "runs": [1], "pos_err_cm": 0.0}),
    ]
    # 4 s is under NOTABLE_S (5 s), so nothing is called out.
    assert step_losses(rows) == []


def test_doctor_markdown_says_so_when_there_is_nothing_to_report():
    rows = [
        (
            "arms its own",
            {
                "mean_survival_s": 20.0,
                "seconds": 20.0,
                "complete": 3,
                "runs": [1, 2, 3],
                "pos_err_cm": 9.0,
            },
        )
    ]
    md = doctor_markdown("p", 20.0, [], [], [], rows, "It survives the tour.")
    assert "Every field the runner needs comes from a file." in md
    assert "Tracks the commands it was swept with." in md


def test_bench_command_runs_the_ladder_on_the_g1(tmp_path, capsys):
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.cli import main

    wp = tmp_path / "wp.yaml"
    wp.write_text("waypoints:\n  - [0.3, 0.0, 0.0]\n")
    out, md = tmp_path / "b.json", tmp_path / "b.md"
    argv = ["bench", "--deploy", str(URL_G1 / "deploy.yaml"), "--onnx", str(URL_G1 / "policy.onnx")]
    argv += ["--mjcf", str(UMJ_G1), "--waypoints", str(wp), "--point-s", "2", "--seeds", "1"]
    argv += ["--stages", "own,arms_walk", "--json", str(out), "--md", str(md)]
    code = main(argv)
    text = capsys.readouterr().out
    assert "own setup" in text and "harness walks the arms" in text
    d = json.loads(out.read_text())
    assert d["command"] == "bench" and list(d["stages"]) == ["own", "arms_walk"]
    assert md.read_text().startswith("# gaitkeeper bench")
    assert code in (0, 5)


class _C:
    def __init__(self, unlisted):
        self.u = unlisted

    def get(self, path, default=None):
        assert path == "control.unlisted"
        return self.u if self.u is not None else default


def test_unlisted_diff_reports_how_the_port_holds_the_arms():
    from gaitkeeper.bench import unlisted_diff

    def c(kp):
        return _C({"pose": {"a": 0.2, "b": 0.0}, "kp": {"a": kp, "b": 40.0}})

    lines = unlisted_diff(c(1.3), c(40.0))
    assert lines[0] == "unlisted pose: same on 2 joint(s)"
    assert lines[1] == "unlisted kp: 1 of 2 joint(s) differ; largest a port 1.3 against 40"
    assert unlisted_diff(_C(None), c(1.0))[0].endswith("own contract holds them (control.unlisted)")


def test_attribution_starts_from_the_upstream_config():
    st = {"upstream": _st(20, 2), "own": _st(3), "punches": _st(2.5)}
    f = attribute(st, 20.0)
    assert f[0].startswith("switching from the upstream config to the policy's contract")
    assert "17.0 s" in f[0]


def test_bench_runs_an_upstream_stage_with_its_controller_filled(tmp_path, capsys):
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.cli import main

    wp = tmp_path / "wp.yaml"
    wp.write_text("waypoints:\n  - [0.3, 0.0, 0.0]\n")
    out = tmp_path / "b.json"
    dep, onnx = str(URL_G1 / "deploy.yaml"), str(URL_G1 / "policy.onnx")
    argv = ["bench", "--deploy", dep, "--onnx", onnx, "--upstream-deploy", dep]
    argv += ["--mjcf", str(UMJ_G1), "--waypoints", str(wp), "--point-s", "2", "--seeds", "1"]
    argv += ["--stages", "own", "--json", str(out)]
    main(argv)
    text = capsys.readouterr().out
    assert "upstream config (as trained or deployed)" in text
    assert "The policy's contract against the upstream config: no differences" in text
    d = json.loads(out.read_text())
    assert list(d["stages"]) == ["upstream", "own"]


def test_doctor_reads_a_config_by_content_and_runs_the_checks(tmp_path, capsys):
    from assets import UMJ_G1, URL_G1, need

    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.cli import main

    out = tmp_path / "d.json"
    code = main(
        [
            "doctor",
            "--config",
            str(URL_G1 / "deploy.yaml"),
            "--onnx",
            str(URL_G1 / "policy.onnx"),
            "--mjcf",
            str(UMJ_G1),
            "--seeds",
            "1",
            "--quick",
            "--json",
            str(out),
        ]
    )
    text = capsys.readouterr().out
    assert "1. Contract" in text and "DEAD ZONE" in text and "arms its own" in text
    d = json.loads(out.read_text())
    assert d["command"] == "doctor" and d["dead_zones"]
    assert code == 5  # dead zones: not a clean bill
