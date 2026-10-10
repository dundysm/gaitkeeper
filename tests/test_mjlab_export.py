"""Integration with the shipped unitree_rl_mjlab export and a recorded golden trace.

Skipped unless those files are present (`gaitkeeper fetch g1_rl_mjlab`; the
golden trace from the mjlab recorder under runs/ or GAITKEEPER_GOLDEN).
"""

from pathlib import Path

import pytest
from assets import RUNS, UMJLAB_G1

from gaitkeeper.env import env

GOLDEN = Path(env("GOLDEN") or RUNS / "g1_golden_a")
ONNX, YAML = UMJLAB_G1 / "policy.onnx", UMJLAB_G1 / "deploy.yaml"

need_export = pytest.mark.skipif(not ONNX.exists(), reason="mjlab export not present")
need_golden = pytest.mark.skipif(
    not (ONNX.exists() and (GOLDEN / "golden.npz").exists()), reason="golden trace not present"
)


@need_export
def test_reader_findings_on_the_shipped_export():
    from gaitkeeper.readers.mjlab_export import read_mjlab_export

    c, findings = read_mjlab_export(ONNX, YAML)
    kinds = {(f.path, f.kind) for f in findings}
    assert ("control.actions.joint_pos.scale", "rounding") in kinds  # wrist 0.07 against 0.075
    assert c.prov("control.actions.joint_pos.scale").resolution == 0.0005
    assert c.prov("control.actuators.kind").source == "unknown"
    assert len(c.get("policy_io.joints.names")) == 29


@need_golden
def test_golden_trace_passes_against_the_files_contract():
    from gaitkeeper.compare import verify
    from gaitkeeper.policy import OnnxPolicy
    from gaitkeeper.readers.mjlab_export import read_mjlab_export
    from gaitkeeper.trace import Trace

    c, _ = read_mjlab_export(ONNX, YAML)
    rep = verify(Trace.load(GOLDEN), c, OnnxPolicy(ONNX))
    assert rep.verdict == "PASS", rep.summary()


def test_deploy_yaml_clip_in_either_shape_reaches_the_runner():
    from gaitkeeper.contract import Contract, action_clip_pairs
    from gaitkeeper.readers.mjlab_export import _clip_list

    for raw in ([-3, 3], [[-3, 3], [-2, 2]]):
        c = Contract({})
        c.set("control.actions.joint_pos.clip", _clip_list(raw), "file", "t")
        c.set("control.actions.joint_pos.clip_stage", "processed", "file", "t")
        pairs = action_clip_pairs(c, 2)
        assert pairs[0] == (-3.0, 3.0) and len(pairs) == 2
