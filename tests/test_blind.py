"""The blind protocol end to end on synthetic logs: seal, run, score, and the hash checks."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import synth

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
onnx = pytest.importorskip("onnx")

import blind  # noqa: E402

from gaitkeeper.inject import Harness, defects  # noqa: E402


def _onnx_of(pol: synth.LinearPolicy, path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array((pol.w * 2.0).astype(np.float32), "w")
    k = numpy_helper.from_array(np.array(1.5, np.float32), "k")
    g = helper.make_graph(
        [
            helper.make_node("MatMul", ["obs", "w"], ["z"]),
            helper.make_node("Tanh", ["z"], ["t"]),
            helper.make_node("Mul", ["t", "k"], ["actions"]),
        ],
        "linear",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, pol.n_in])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, pol.w.shape[1]])],
        [w, k],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8  # readable by every onnxruntime the package supports
    onnx.save(model, str(path))


def _submission(tmp_path, truth, files, policy):
    g = synth.make_golden()
    first = Harness(g, truth, policy, files).standard()
    g.arrays["obs"], g.arrays["action"] = first["obs"], first["action"]
    h = Harness(g, truth, policy, files)
    picked = {d.name: d for d in defects()}
    sub = tmp_path / "submission"
    (sub / "logs").mkdir(parents=True)
    files.save(sub / "contract.yaml")
    _onnx_of(policy, sub / "policy.onnx")
    cases = {
        "log_a": (picked["clean harness log (control)"], {"kind": "none", "boundary": None}),
        "log_b": (
            picked["gyro in the world frame"],
            {"kind": "contract", "boundary": "A", "tokens": ["base_ang_vel"]},
        ),
        "log_c": (picked["kp scaled 0.7"], {"kind": "contract", "boundary": "C", "tokens": ["kp"]}),
    }
    for name, (d, _) in cases.items():
        d.build(h).save(sub / "logs" / f"{name}.npz")
    labels = tmp_path / "labels.json"
    labels.write_text(
        json.dumps(
            {"salt": "a long random salt 0123", "cases": {k: v for k, (_, v) in cases.items()}}
        )
    )
    return sub, labels


def test_blind_protocol_scores_only_the_committed_files(tmp_path, truth, files, policy, capsys):
    sub, labels = _submission(tmp_path, truth, files, policy)
    blind.main(["seal", str(labels)])
    lab_hash = hashlib.sha256(labels.read_bytes()).hexdigest()
    assert lab_hash in capsys.readouterr().out

    blind.main(["run", str(sub)])
    out = json.loads((sub / "outputs.json").read_text())["cases"]
    assert out["log_a"]["verdict"] != "CONTRACT"
    assert out["log_b"]["verdict"] == "CONTRACT" and "A" in out["log_b"]["failing"]
    assert out["log_c"]["verdict"] == "CONTRACT" and "C" in out["log_c"]["failing"]
    out_hash = hashlib.sha256((sub / "outputs.json").read_bytes()).hexdigest()

    blind.main(
        ["score", str(sub), str(labels), "--labels-sha256", lab_hash, "--outputs-sha256", out_hash]
    )
    res = json.loads((sub / "blind_results.json").read_text())
    assert res["cases"] == 3 and res["false_confident"] == 0
    assert res["detection"]["A"]["detected"] == 1 and res["detection"]["C"]["detected"] == 1

    # A label changed after the run, or an output edited after the commitment, is refused.
    labels.write_text(labels.read_text().replace("base_ang_vel", "joint_vel_rel"))
    with pytest.raises(SystemExit, match="not the committed"):
        blind.main(
            [
                "score",
                str(sub),
                str(labels),
                "--labels-sha256",
                lab_hash,
                "--outputs-sha256",
                out_hash,
            ]
        )


def test_labels_need_a_salt(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"salt": "short", "cases": {}}))
    with pytest.raises(SystemExit, match="salt"):
        blind.main(["seal", str(p)])
