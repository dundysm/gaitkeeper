"""Integration with the shipped unitree_rl_mjlab export and a recorded golden trace.

Skipped unless those files are present (paths from SIM2SIM_MJLAB_EXPORT and
SIM2SIM_GOLDEN, or the defaults used during development).
"""

import os
from pathlib import Path

import pytest

EXPORT = Path(
    os.environ.get(
        "SIM2SIM_MJLAB_EXPORT",
        "unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0",
    )
)
GOLDEN = Path(os.environ.get("SIM2SIM_GOLDEN", Path(__file__).parents[1] / "runs" / "g1_golden_a"))
ONNX, YAML = EXPORT / "exported" / "policy.onnx", EXPORT / "params" / "deploy.yaml"

need_export = pytest.mark.skipif(not ONNX.exists(), reason="mjlab export not present")
need_golden = pytest.mark.skipif(
    not (ONNX.exists() and (GOLDEN / "golden.npz").exists()), reason="golden trace not present"
)


@need_export
def test_reader_findings_on_the_shipped_export():
    from sim2sim.readers.mjlab_export import read_mjlab_export

    c, findings = read_mjlab_export(ONNX, YAML)
    kinds = {(f.path, f.kind) for f in findings}
    assert ("control.actions.joint_pos.scale", "rounding") in kinds  # wrist 0.07 against 0.075
    assert c.prov("control.actions.joint_pos.scale").resolution == 0.0005
    assert c.prov("control.actuators.kind").source == "unknown"
    assert len(c.get("policy_io.joints.names")) == 29


@need_golden
def test_golden_trace_passes_against_the_files_contract():
    from sim2sim.compare import verify
    from sim2sim.policy import OnnxPolicy
    from sim2sim.readers.mjlab_export import read_mjlab_export
    from sim2sim.trace import Trace

    c, _ = read_mjlab_export(ONNX, YAML)
    rep = verify(Trace.load(GOLDEN), c, OnnxPolicy(ONNX))
    assert rep.verdict == "PASS", rep.summary()
