"""Boundary D: floor, injected armature (AT12), friction search, confounding (AT11),
and the stand-in drive's Appendix B terms."""

import json

import numpy as np
import pytest
from assets import GOLDEN_A, GOLDEN_B, GOLDEN_C, MJLAB_ONNX, UMJ_G1, need

from sim2sim import residual as R
from sim2sim.contract import Contract
from sim2sim.inject import physics_edit
from sim2sim.models import load_model
from sim2sim.trace import Trace

ENGINE = "mjlab 1.2.0 / mujoco_warp 3.5.0"


def golden(p):
    need(p / "golden.npz", p / "model_patch.json")
    return Trace.load(p), Contract.load(p / "contract.live.yaml")


def edited(path, edit):
    m = load_model(path).model
    edit(m)
    return m


def test_floor_table_is_calibrated_for_mjlab():
    fl = json.loads(R.FLOORS_PATH.read_text())["engines"][ENGINE]
    assert len(fl["joint_rms"]) == 29
    assert fl["calibrated_on"] == ["g1_golden_a", "g1_golden_b", "g1_golden_c"]
    assert max(fl["joint_rms"].values()) < 0.01  # N m: float32 source, float64 analysis
    assert fl["root_force_rms"] < 1.0 and fl["root_torque_rms"] < 0.5


@pytest.mark.parametrize("path", [GOLDEN_A, GOLDEN_B, GOLDEN_C], ids="abc")
def test_recorded_model_is_at_floor(path):
    tr, c = golden(path)
    d = R.dynamics_residual(tr, c, path)
    assert d.status == "at_floor", d.lines()
    assert d.engine == ENGINE
    assert all(v["ratio"] <= 1.0 + 1e-9 for v in d.chains.values())  # calibrated on these traces
    # legs are judged on swing steps only, and there are enough of them
    legs = [j for j in d.joints if j.chain.endswith("leg")]
    assert all(500 < j.n_clean < 0.5 * tr["p/qpos"].shape[0] for j in legs)


def test_uncalibrated_engine_blocks():
    tr, c = golden(GOLDEN_A)
    d = R.dynamics_residual(tr, c, GOLDEN_A, floors={})
    assert d.status == "uncalibrated" and "blocked" in d.reason


def test_no_physics_rows_is_not_measured():
    tr, c = golden(GOLDEN_A)
    t = Trace(
        {k: v for k, v in tr.arrays.items() if not k.startswith("p/")}, dict(tr.meta), "golden"
    )
    assert R.dynamics_residual(t, c, GOLDEN_A).status == "not_measured"


@pytest.mark.parametrize("delta", [0.01, -0.002])
def test_injected_ankle_armature_sign_and_magnitude(delta):
    """AT12 inside one engine: the change is named on the legs, not the arms or waist, and the
    research fit recovers it on every ankle joint with held-out agreement."""
    tr, c = golden(GOLDEN_A)
    m = edited(GOLDEN_A, physics_edit("armature", value=delta, joints=("ankle",), add=True))
    d = R.dynamics_residual(tr, c, m, fits=True)
    assert d.status == "above_floor"
    # Both legs; the root residual may also move, since the discrete inverse maps
    # acceleration through the whole mass matrix before inferring contact forces.
    assert (
        {"left leg", "right leg"}
        <= set(d.above_chains)
        <= {"left leg", "right leg", "root/contact"}
    )
    for f in d.fits:
        if "ankle" in f["joint"]:
            assert f["status"] == "fit", f
            assert f["dI"] == pytest.approx(delta, rel=0.02, abs=2e-4)
            assert f["r2_held_out"] > 0.9
        elif f["status"] == "fit":
            assert abs(f["dI"]) < 2e-4


def _runner_trace(model, edit, backend="native_implicit", seconds=5.0):
    from sim2sim.policy import OnnxPolicy
    from sim2sim.runner import RunConfig, Runner

    c = Contract.load(GOLDEN_A / "contract.live.yaml")
    sched = [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.5, 0.0, 0.0)), (3.0, (0.3, 0.0, 0.5))]
    r = Runner(c, model, OnnxPolicy(MJLAB_ONNX))
    res = r.run(
        RunConfig(
            backend=backend,
            seconds=seconds,
            schedule=sched,
            record=True,
            physics=True,
            model_edit=edit,
        )
    )
    assert res.survived
    return r.to_trace(res, kind="golden"), c


def test_friction_on_both_sides_uses_the_armature_search(monkeypatch):
    """Joint friction in source and analysis: the linear fit is not used; a search
    over armature finds the injected wrist change with a sharp minimum."""
    need(UMJ_G1, MJLAB_ONNX, GOLDEN_A / "contract.live.yaml")
    tr, c = _runner_trace(UMJ_G1, None)  # UMJ keeps its joint friction and damping
    m = edited(UMJ_G1, physics_edit("armature", value=0.004, joints=("wrist",), add=True))
    monkeypatch.setattr(R, "ARMATURE_GRID", np.round(np.arange(-0.01, 0.01001, 0.001), 4))
    d = R.dynamics_residual(tr, c, m, fits=True)
    assert all(m.dof_frictionloss[6:] > 0)
    wrists = [f for f in d.fits if "wrist" in f["joint"]]
    others = [f for f in d.fits if "wrist" not in f["joint"]]
    assert len(wrists) == 6
    for f in wrists:
        assert f["method"].startswith("armature search")
        assert f["dI"] == pytest.approx(0.004, abs=5e-4), f
        assert f["status"] == "fit" and f["sharpness"] > 1.5  # the search acceptance rule
    for f in others:
        assert f["method"].startswith("armature search")
        assert abs(f["dI"]) <= 5e-4 or f["status"] == "abstain", f


def test_confounded_change_is_named_as_a_chain():
    """AT11: armature, damping and effort limit changed together on the left leg.
    D names the left leg (and possibly the root), not the other chains; no
    parameter appears in the finding."""
    tr, c = golden(GOLDEN_A)
    edit = physics_edit(
        "chain_change",
        joints=("left_hip", "left_knee", "left_ankle"),
        armature=0.01,
        damping=0.5,
        limit=0.5,
    )
    d = R.dynamics_residual(tr, c, edited(GOLDEN_A, edit))
    assert "left leg" in d.above_chains, d.lines()
    assert set(d.above_chains) <= {"left leg", "root/contact"}
    head = d.lines()[0]
    assert "armature" not in head and "damping" not in head


def test_standin_drive_terms_and_physics_recording():
    """Appendix B: a position-implicit drive analysed with implicitfast shows
    armature -h^2 kp and damping -h kp, exactly; a native trace sits at zero."""
    need(UMJ_G1, MJLAB_ONNX, GOLDEN_A / "contract.live.yaml")
    edit = physics_edit("frictionless", timestep=0.005)
    target = edited(UMJ_G1, edit)
    out = {}
    for be in ("native_implicit", "standin_implicit"):
        out[be], c = _runner_trace(UMJ_G1, edit, backend=be)
        assert out[be].meta["engine"].endswith(f"({be})")
    nat = R.dynamics_residual(out["native_implicit"], c, target)
    assert nat.status == "uncalibrated"  # no floor for this runner engine in the table
    assert max(j.rms for j in nat.joints) < 1e-9
    fl = R.calibrate([out["native_implicit"]], lambda t: c, lambda t: target)
    key = out["standin_implicit"].meta["engine"]
    d = R.dynamics_residual(out["standin_implicit"], c, target, floors={key: fl}, fits=True)
    assert d.status == "above_floor"
    kp = c.get("control.actuators.kp")
    h = 0.005
    for f in d.fits:
        assert f["status"] == "fit", f
        assert f["dI"] == pytest.approx(-h * h * kp[f["joint"]], rel=1e-3)
        assert f["dB"] == pytest.approx(-h * kp[f["joint"]], rel=1e-3)
