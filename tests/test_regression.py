"""Appendix A facts measured during the plan reviews, as bounds.

UMJ G1 scene (unitree_mujoco @ 1eb6642), unitree_rl_lab G1 policy and
deploy.yaml (@ 4960b84), unitree_rl_mjlab G1 policy (@ 1425b15).
"""

import pytest
from assets import UMJ_G1, UMJLAB_G1, URL_G1, need

from gaitkeeper.checks import s17a_margin, s17b_modes
from gaitkeeper.readers.unitree_deploy import read_unitree_deploy
from gaitkeeper.runner import RunConfig, Runner


@pytest.fixture(scope="module")
def url():
    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.policy import OnnxPolicy

    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    return Runner(c, UMJ_G1, OnnxPolicy(URL_G1 / "policy.onnx"))


def test_r1_correct_contract_walks_4_69_m(url):
    res = url.run(RunConfig(backend="python_pd", seconds=10.0, command=(0.5, 0.0, 0.0)))
    assert res.survived and res.dist == pytest.approx(4.69, abs=0.1)


@pytest.mark.parametrize(
    "cmd,axis,walks",
    [
        ((0.20, 0, 0), 0, False),
        ((0.22, 0, 0), 0, True),
        ((0, 0.25, 0), 1, False),
        ((0, 0.28, 0), 1, True),
    ],
)
def test_r2_dead_zone_edges(url, cmd, axis, walks):
    out = []
    for backend in ("python_pd", "native_implicit"):
        res = url.run(RunConfig(backend=backend, seconds=15.0, command=cmd))
        v = (res.vx_tail, res.vy_tail)[axis]
        assert res.survived
        assert (v > 0.2 * cmd[axis]) == walks, (backend, v)
        out.append(v)
    assert abs(out[0] - out[1]) < 0.02  # native agrees with python PD


def test_r2_no_in_place_yaw_at_the_yaml_limit(url):
    res = url.run(RunConfig(seconds=15.0, command=(0.0, 0.0, 0.2)))
    assert res.survived and abs(res.wz_tail) < 0.2 * 0.2


@pytest.mark.parametrize("backend", ["python_pd", "native_implicit"])
def test_r4_zero_action_falls_in_1_to_2_s(url, backend):
    res = url.run(RunConfig(backend=backend, policy_mode="zero", seconds=5.0))
    assert res.fell_at is not None and 1.0 <= res.fell_at <= 2.0


def test_r6_armature_zero_falls_under_python_pd_and_walks_native(url):
    py = url.run(RunConfig(backend="python_pd", armature=0.0, seconds=10.0))
    assert py.fell_at is not None and py.fell_at < 1.0
    nat = url.run(RunConfig(backend="native_implicit", armature=0.0, seconds=10.0))
    assert nat.survived and nat.dist > 4.0


def test_s17a_wrist_roll_margin(url):
    r = s17a_margin(url, "python_pd")
    m = r.data["margins"]["python_pd"]
    i = url.names.index("left_wrist_roll_joint")
    assert m[i] == pytest.approx(3.88, abs=0.01) and m.max() < 4.0
    assert r.status == "PASS"


def test_s17b_native_only_tipping_and_python_pd_armature_zero(url):
    nat = s17b_modes(url, "native_implicit")
    assert nat.status == "PASS" and nat.data["outside"] == 1
    assert 0.3 < nat.data["slow_tc"][0] < 0.5
    py = s17b_modes(url, "python_pd", RunConfig(armature=0.0))
    assert py.status == "FAIL" and py.data["outside"] == 11
    assert py.data["most_negative"] == pytest.approx(-49, abs=1.0)


def test_mjlab_policy_on_umj_walks_4_82_m():
    need(UMJ_G1, UMJLAB_G1 / "deploy.yaml")
    from gaitkeeper.policy import OnnxPolicy

    c, _ = read_unitree_deploy(UMJLAB_G1 / "deploy.yaml", UMJLAB_G1 / "policy.onnx")
    r = Runner(c, UMJ_G1, OnnxPolicy(UMJLAB_G1 / "policy.onnx"))
    res = r.run(RunConfig(seconds=10.0, command=(0.5, 0.0, 0.0)))
    assert res.survived and res.dist == pytest.approx(4.82, abs=0.1)
    py = r.run(RunConfig(backend="python_pd", armature=0.0, seconds=5.0))
    assert py.fell_at is not None and py.fell_at < 1.0
