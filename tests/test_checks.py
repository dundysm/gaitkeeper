import math

import mujoco
import numpy as np
import pytest
import yaml
from assets import URL_G1, need

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
