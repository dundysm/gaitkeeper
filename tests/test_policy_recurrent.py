import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from sim2sim.policy import OnnxPolicy  # noqa: E402


def _model(path):
    # action = obs + h ; h_next = h + 1
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 2])
    h = helper.make_tensor_value_info("h_in", TensorProto.FLOAT, [1, 2])
    act = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])
    hn = helper.make_tensor_value_info("h_out", TensorProto.FLOAT, [1, 2])
    one = helper.make_tensor("one", TensorProto.FLOAT, [1, 2], [1.0, 1.0])
    g = helper.make_graph(
        [
            helper.make_node("Add", ["obs", "h_in"], ["actions"]),
            helper.make_node("Add", ["h_in", "one"], ["h_out"]),
        ],
        "rec",
        [obs, h],
        [act, hn],
        [one],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def test_state_carried_and_zeroed_at_reset(tmp_path):
    p = tmp_path / "rec.onnx"
    _model(str(p))
    pol = OnnxPolicy(p)
    assert pol.is_recurrent and pol.recurrent == [{"in": "h_in", "out": "h_out"}]
    obs = np.zeros((5, 2), np.float32)
    out = pol(obs, reset=np.array([1, 0, 0, 1, 0], bool))
    assert out[:, 0].tolist() == [0, 1, 2, 0, 1]
    pol.reset()
    assert pol.step(np.zeros(2))[0] == 0 and pol.step(np.zeros(2))[0] == 1
