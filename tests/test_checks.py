import math

import mujoco
import numpy as np
import pytest
import yaml
from assets import UMJ_G1, URL_G1, need

from sim2sim.checks import classify_modes, joint_inertia, pd_margin, s16_symmetry, s18_commands
from sim2sim.readers.unitree_deploy import read_unitree_deploy
from sim2sim.tables import G1_29_SDK

HINGE = """
<mujoco><option timestep="{h}" integrator="Euler" gravity="0 0 0"/>
<worldbody><body><joint name="j" type="hinge" axis="0 0 1" armature="{arm}"/>
<geom type="sphere" size="0.01" mass="1e-6"/></body></worldbody>
<actuator><motor joint="j"/></actuator></mujoco>
"""


def _simulate(kp, kd, h=0.002, arm=0.01, steps=4000):
    m = mujoco.MjModel.from_xml_string(HINGE.format(h=h, arm=arm))
    d = mujoco.MjData(m)
    d.qpos[0] = 0.1
    for _ in range(steps):
        d.ctrl[0] = -kp * d.qpos[0] - kd * d.qvel[0]  # PD from the state at the step start
        mujoco.mj_step(m, d)
    return m, d


@pytest.mark.parametrize("margin,stable", [(3.9, True), (4.1, False)])
def test_s17a_margin_bound_matches_simulation(margin, stable):
    h, arm = 0.002, 0.01
    m, d = _simulate(0, 0, h, arm, steps=0)
    inertia = joint_inertia(m, d, np.array([0]))[0]
    assert inertia == pytest.approx(arm, rel=1e-3)
    b = 0.5  # 2 kd h / I = 2b ... split the margin between damping and stiffness
    kd = b * inertia / h
    kp = (margin - 2 * b) * inertia / h**2
    assert pd_margin(np.array([kp]), np.array([kd]), np.array([inertia]), h)[0] == pytest.approx(
        margin
    )
    _, d = _simulate(kp, kd, h, arm, steps=60)  # short: MuJoCo resets a diverged state
    assert (abs(d.qpos[0]) < 0.01) if stable else (abs(d.qpos[0]) > 1.0)


def test_s17b_mode_classes():
    h = 0.002
    tipping = math.exp(h / 0.4)
    eig = np.array([0.9, 0.5 + 0.2j, 0.5 - 0.2j, tipping, 1.000001])
    c = classify_modes(eig, h)
    assert c["outside"] == 1 and not c["bad"] and c["slow"][0][1] == pytest.approx(0.4)
    c = classify_modes(np.array([tipping, -1.5, math.exp(h / 0.05), 1.02 * np.exp(0.3j)]), h)
    assert len(c["slow"]) == 1 and len(c["bad"]) == 3
    c = classify_modes(np.array([1.00002, 0.99]), h)  # neutral drift within finite-difference noise
    assert c["outside"] == 0 and not c["bad"]


def test_s17b_slow_complex_pair_is_drift_fast_turning_is_not():
    h = 0.005
    pair = [1.000114 + 0.000117j, 1.000114 - 0.000117j]  # growth 44 s, turning 0.023 rad/s
    c = classify_modes(np.array(pair), h)
    assert c["outside"] == 0 and not c["bad"] and len(c["drift"]) == 2
    turning = math.exp(h / 10.0) * np.exp(1j * 2 * math.pi * 1.0 * h)  # slow growth, 1 Hz
    assert len(classify_modes(np.array([turning, turning.conjugate()]), h)["bad"]) == 2
    fast = math.exp(h / 0.5) * np.exp(0.0002j)  # turns slowly but grows in 0.5 s
    assert len(classify_modes(np.array([fast, fast.conjugate()]), h)["bad"]) == 2


def test_s17b_open_case_rl_lab_umj_5ms():
    """The pair once reported at 5 ms (modulus 1.0001, growth about 44 s): the same
    for every finite-difference step and scheme, and present at 2 ms with the same
    rate per second, so neither numerical nor a 5 ms effect. It is slow drift."""
    need(URL_G1 / "deploy.yaml", UMJ_G1)
    from sim2sim.behavior import VARIANTS
    from sim2sim.checks import s17b_modes, settle
    from sim2sim.policy import OnnxPolicy
    from sim2sim.presets import apply_preset
    from sim2sim.runner import RunConfig, Runner

    c = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")[0]
    apply_preset(c, "unitree_rl_lab_g1_29dof_velocity@4960b84")
    r = Runner(c, str(UMJ_G1), OnnxPolicy(URL_G1 / "policy.onnx"))
    rates = {}
    for h, edit in ((0.002, None), (0.005, VARIANTS["step 5 ms"][0])):
        cfg = RunConfig(model_edit=edit)
        qpos, qvel = settle(r, cfg=cfg)
        m, d, b = r.build(cfg, "native_implicit")
        m.dof_frictionloss[:] = 0.0
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, d)
        d.ctrl[b.aid] = r.default
        found = set()
        for eps, centered in ((1e-4, True), (1e-6, True), (1e-8, True), (1e-6, False)):
            a = np.zeros((2 * m.nv, 2 * m.nv))
            mujoco.mjd_transitionFD(m, d, eps, centered, a, None, None, None)
            w = np.linalg.eigvals(a)
            z = max((x for x in w if x.imag > 1e-6), key=abs)
            found.add((round(math.log(abs(z)) / h, 3), round(np.angle(z) / h, 3)))
        assert len(found) == 1, found  # independent of the difference step and scheme
        rates[h] = found.pop()
    (g2, w2), (g5, w5) = rates[0.002], rates[0.005]
    assert (
        0.015 < g5 < 0.035 and g2 == pytest.approx(g5, rel=0.2) and w2 == pytest.approx(w5, rel=0.2)
    )
    res = s17b_modes(r, "native_implicit", cfg=RunConfig(model_edit=VARIANTS["step 5 ms"][0]))
    assert res.status == "PASS" and res.data["bad"] == 0, res.lines
    assert res.data["slow_tc"][0] == pytest.approx(0.36, abs=0.02)
    assert any("drift, not counted" in x and "growth time" in x for x in res.lines)


def _url(tmp_path, mutate=None):
    y = yaml.safe_load((URL_G1 / "deploy.yaml").read_text())
    if mutate:
        mutate(y)
    p = tmp_path / "deploy.yaml"
    p.write_text(yaml.safe_dump(y))
    return read_unitree_deploy(p)[0]


def test_s16_correct_and_misordered_reads(tmp_path):
    need(URL_G1 / "deploy.yaml")
    r = s16_symmetry(_url(tmp_path))
    assert r.status == "PASS" and r.data["mismatched"] == {"default": 0, "kp": 0, "kd": 0}

    c = _url(tmp_path)
    y = yaml.safe_load((URL_G1 / "deploy.yaml").read_text())
    # default pose list (policy order) read as if it were in SDK order
    c.set(
        "control.default_joint_pos",
        {n: float(v) for n, v in zip(G1_29_SDK, y["default_joint_pos"])},
        "file",
    )
    assert s16_symmetry(c).data["mismatched"]["default"] == 9

    c = _url(tmp_path)
    names = c.get("policy_io.joints.names")
    # gain lists (SDK order) read as if they were in policy order
    c.set("control.actuators.kp", {n: float(v) for n, v in zip(names, y["stiffness"])}, "file")
    c.set("control.actuators.kd", {n: float(v) for n, v in zip(names, y["damping"])}, "file")
    r = s16_symmetry(c)
    assert (
        r.status == "FAIL" and r.data["mismatched"]["kp"] == 3 and r.data["mismatched"]["kd"] == 3
    )


def test_s18_trained_limit_and_dead_zone(tmp_path):
    need(URL_G1 / "deploy.yaml")
    c = _url(tmp_path)
    r = s18_commands(c, [(0.5, 0.0, 0.0)])
    assert r.status == "PASS" and "no trained range known" in r.lines[0]
    c.set(
        "policy_io.commands.base_velocity.trained",
        {"vx": [-0.5, 1.0], "vy": [-0.3, 0.3], "wz": [-0.1, 0.1]},
        "preset",
    )
    r = s18_commands(c, [(0.0, 0.0, 0.2)])
    assert r.status == "FAIL" and "outside trained" in r.lines[-1]
    r = s18_commands(c, [(0.2, 0.0, 0.0)], {"vx": (-0.15, 0.2)})
    assert r.status == "FAIL" and "dead zone" in r.lines[-1]
    r = s18_commands(c, [(0.5, 0.0, 0.0)])
    assert r.status == "WARN"  # the wz limit 0.2 was never trained
