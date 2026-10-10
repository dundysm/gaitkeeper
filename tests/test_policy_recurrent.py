import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from gaitkeeper.policy import OnnxPolicy  # noqa: E402


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


def _gain_model(path, k):
    obs = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 2])
    act = helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])
    kk = helper.make_tensor("k", TensorProto.FLOAT, [1, 2], [k, k])
    g = helper.make_graph(
        [helper.make_node("Mul", ["obs", "k"], ["actions"])], "g", [obs], [act], [kk]
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def test_a_policy_switched_by_the_command(tmp_path):
    from gaitkeeper.contract import Contract
    from gaitkeeper.policy import SwitchedPolicy, load_policy

    _gain_model(str(tmp_path / "walk.onnx"), 1.0)
    _gain_model(str(tmp_path / "stand.onnx"), 2.0)
    c = Contract({})
    c.set(
        "policy_io.graph.switch",
        {"by": "command_norm", "threshold": 0.05, "above": "walk.onnx", "below": "stand.onnx"},
        "file",
        "t",
    )
    pol = load_policy(c, tmp_path / "walk.onnx")
    assert isinstance(pol, SwitchedPolicy) and not pol.is_recurrent
    obs = np.ones(2)
    pol.command = np.array([0.0, 0.0, 0.04])
    assert pol.step(obs).tolist() == [2.0, 2.0]
    pol.command = np.array([0.04, 0.0, 0.04])
    assert pol.step(obs).tolist() == [1.0, 1.0]
    out = pol(np.ones((2, 2)), command=np.array([[0.0, 0, 0], [0.3, 0, 0]]))
    assert out[:, 0].tolist() == [2.0, 1.0]
    (tmp_path / "stand.onnx").unlink()
    with pytest.raises(FileNotFoundError, match="stand.onnx"):
        load_policy(c, tmp_path / "walk.onnx")
