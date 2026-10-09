import numpy as np
import pytest
from assets import UMJLAB_G1, need

from gaitkeeper.deviation import compare
from gaitkeeper.readers.mjlab_export import read_mjlab_export
from gaitkeeper.readers.unitree_deploy import read_unitree_deploy


def test_mjlab_deploy_against_onnx_metadata():
    need(UMJLAB_G1 / "deploy.yaml")
    dep, _ = read_unitree_deploy(UMJLAB_G1 / "deploy.yaml", UMJLAB_G1 / "policy.onnx")
    ref, _ = read_mjlab_export(UMJLAB_G1 / "policy.onnx", None)
    names = dep.get("policy_io.joints.names")
    rng = np.random.default_rng(0)
    actions = rng.normal(0, 1, (200, 29))
    dev = compare(dep, ref, actions, names, "ONNX")
    i = names.index("left_wrist_pitch_joint")
    assert dev.fields["scale"]["deploy"][i] == 0.07 and dev.fields["scale"]["reference"][i] == 0.075
    assert dev.rel("scale")[i] == pytest.approx(-0.0667, abs=1e-4)
    assert dev.rel("kd")[i] == pytest.approx(0.03, abs=1e-3)
    assert dev.target["max"][i] == pytest.approx(np.abs(actions[:, i]).max() * 0.005)
    assert dev.phase["lead_steps"] == 2 and dev.phase["lead_deg"] == pytest.approx(24.0)
    text = "\n".join(dev.lines())
    assert "offset: no joint differs" in text
