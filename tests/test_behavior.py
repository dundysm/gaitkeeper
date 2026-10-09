"""Behavior probes: standstill, kicks, pushes and the physics fragility sweep.

Unit tests use small hand-built inputs. The integration tests use the
unitree_mujoco G1 scene and the unitree_rl_lab G1 policy and check the plan's
section 1 finding and the armature 0 demo at L1."""

import math

import mujoco
import numpy as np
import pytest
from assets import UMJ_G1, URL_G1, need

from gaitkeeper.behavior import (
    VARIANTS,
    _dead_commands,
    contract_header,
    fall_comparison,
    fragility_table,
    is_still,
    stillness,
    yaw_walk,
)
from gaitkeeper.contract import Contract
from gaitkeeper.envelope import Envelope
from gaitkeeper.runner import Push, PushGenerator, RunConfig, Runner, ground_contacts

BOX_ON_FLOOR = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1"/></body>
    <body name="ball" pos="0 0 2"><freejoint/><geom type="sphere" size="0.1"/></body>
  </worldbody>
</mujoco>
"""


def test_ground_contacts_flags_bodies_touching_the_world():
    m = mujoco.MjModel.from_xml_string(BOX_ON_FLOOR)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    flags = ground_contacts(m, d)
    assert flags[m.body("box").id] and not flags[m.body("ball").id] and not flags[0]


def test_still_needs_all_three_measures():
    base = {"fell_at": None, "action_std": 0.01, "joint_speed": 0.001, "contact_switches": 0}
    assert is_still(base)
    assert not is_still({**base, "action_std": 0.02})
    assert not is_still({**base, "joint_speed": 0.002})
    assert not is_still({**base, "contact_switches": 1})
    assert not is_still({**base, "fell_at": 3.0})
    assert not is_still({**base, "contact_switches": None})  # unmeasured is not still


def test_header_counts_default_and_unknown_fields():
    c = Contract()
    c.set("source", {"format": "unitree_deploy", "files": [{"path": "deploy.yaml"}]}, "file")
    c.set("timing.policy_dt", 0.02, "file")
    c.set("timing.sim_dt", None, "unknown")
    c.set("timing.decimation", 4, "default")
    h = contract_header(c)
    assert "unitree_deploy (deploy.yaml) | level L1 | 2 fields default or unknown" in h
    assert "(1 unknown, 1 default)" in h


def _row(cmd, ach, fell=None):
    return {"cmd": cmd, "achieved": ach, "fell_at": fell}


def test_yaw_while_walking_ratio_stays_inside_the_limit_range():
    env = Envelope(
        {
            "wz_walk": [
                _row((0.5, 0, -0.5), (0.5, 0, -0.33)),
                _row((0.5, 0, -0.2), (0.5, 0, -0.15)),
                _row((0.5, 0, 0.2), (0.5, 0, 0.13)),
                _row((0.5, 0, 0.5), (0.5, 0, 0.32)),
            ]
        },
        {},
        "trained",
        {},
        {"wz": [-0.2, 0.2]},
        None,
    )
    y = yaw_walk(env)
    assert y["commands"] == [-0.2, 0.2]
    assert [round(r, 2) for r in y["ratios"]] == [0.75, 0.65]


def test_dead_zone_probe_commands_use_the_largest_ignored_command_inside_the_limit():
    rows = [_row((0, 0, c), (0, 0, 0.0)) for c in (-0.5, -0.2, 0.1, 0.2, 0.5)]
    env = Envelope({"wz": rows}, {}, "trained", {}, {"wz": [-0.2, 0.2]}, None)
    env.dead = {"wz": {"pos_edge": 0.5, "neg_edge": -0.5}}
    assert _dead_commands(env) == [(0.0, 0.0, 0.2), (0.0, 0.0, -0.2)]


def _chk(s17a="PASS", s17b="PASS", over=0, tc=0.36):
    return {"s17a": s17a, "s17a_over": over, "s17b": s17b, "tipping_tc": tc}


def test_fall_time_comparison_is_suppressed_when_s17a_warns_or_a_joint_is_at_its_limit():
    runs = [{"scenario": "fwd", "fell_at": 0.4, "at_limit": []}]
    zero = {"fell_at": 0.62}
    out = fall_comparison(runs, zero, _chk("WARN", over=12))
    assert out["suppressed"] and "before_baseline" not in out
    runs_lim = [{"scenario": "fwd", "fell_at": 0.4, "at_limit": ["left_ankle_pitch_joint"]}]
    assert fall_comparison(runs_lim, zero, _chk())["suppressed"]
    out = fall_comparison(runs, zero, _chk())
    assert not out["suppressed"] and out["before_baseline"] == ["fwd"]
    assert out["tipping_tc"] == 0.36 and out["baseline"] == 0.62


def _frag(v, bk, scen, fell=None, ach=(0.5, 0, 0), chk=None, zero=1.34):
    cmd = {"fwd": (0.5, 0, 0)}[scen]
    return [
        {
            "tag": "frag_run",
            "variant": v,
            "backend": bk,
            "scenario": scen,
            "cmd": cmd,
            "fell_at": fell,
            "achieved": ach,
            "sat_max": 0.0,
            "at_limit": [],
        },
        {"tag": "frag_zero", "variant": v, "backend": bk, "fell_at": zero},
        {"tag": "frag_check", "variant": v, "backend": bk, **(chk or _chk())},
    ]


def test_fragility_names_numerical_instability_only_when_a_check_fires_beyond_nominal():
    res = (
        _frag("nominal", "python_pd", "fwd")
        + _frag("armature 0", "python_pd", "fwd", fell=0.4, chk=_chk("WARN", "FAIL", 12))
        + _frag("friction 2.0", "python_pd", "fwd", fell=2.3)
    )
    f = fragility_table(
        res, ["nominal", "armature 0", "friction 2.0"], ["python_pd"], [{"name": "fwd"}]
    )
    loud = {x["variant"]: x for x in f["loud"]}
    assert loud["armature 0"]["numerical"] and not loud["friction 2.0"]["numerical"]
    assert loud["friction 2.0"]["lost"] == ["fwd"]


def test_variants_cover_the_plan_sweep():
    names = " ".join(VARIANTS)
    for word in ("friction", "contacts", "armature", "kp", "step"):
        assert word in names


# -- integration: unitree_rl_lab G1 in the unitree_mujoco scene ------------------------


@pytest.fixture(scope="module")
def url():
    need(UMJ_G1, URL_G1 / "deploy.yaml")
    from gaitkeeper.policy import OnnxPolicy
    from gaitkeeper.presets import apply_preset
    from gaitkeeper.readers.unitree_deploy import read_unitree_deploy

    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    apply_preset(c, "unitree_rl_lab_g1_29dof_velocity@4960b84")
    return Runner(c, UMJ_G1, OnnxPolicy(URL_G1 / "policy.onnx"))


def _still(r, cmd, backend, seconds=15.0, pushes=()):
    res = r.run(
        RunConfig(
            backend=backend,
            command=cmd,
            seconds=seconds,
            record=True,
            contacts=True,
            pushes=list(pushes),
        )
    )
    return stillness(res, r.policy_dt)


@pytest.mark.parametrize("backend", ["native_implicit", "python_pd"])
def test_dead_zone_is_a_true_standstill(url, backend):
    for cmd in ((0.20, 0, 0), (0, 0.25, 0), (0, 0, 0.2)):
        s = _still(url, cmd, backend)
        assert s["still"], (cmd, s)
    assert _still(url, (0, 0, 0.2), backend)["action_std"] == pytest.approx(0.0137, abs=0.002)
    s = _still(url, (0.22, 0, 0), backend)
    assert not s["still"] and s["contact_switches"] > 0


def test_standstill_returns_after_a_kick_except_at_the_lateral_edge(url):
    kick = [Push(5.0, "velocity", (0.5, 0.5, 0.0))]
    assert _still(url, (0.20, 0, 0), "native_implicit", 20.0, kick)["still"]
    assert _still(url, (0, 0, 0.2), "native_implicit", 20.0, kick)["still"]
    # at the lateral edge the kick starts a sideways walk that does not stop
    s = _still(url, (0, 0.25, 0), "native_implicit", 20.0, kick)
    assert not s["still"] and s["speed"] > 0.1


def test_survives_trained_kicks_and_falls_to_punches(url):
    kicks = PushGenerator(every_s=5.0, velocity=0.5)
    punches = PushGenerator(every_s=5.0, force=600.0, duration=0.08)
    for seed in (1, 2, 3):
        base = dict(command=(0.5, 0, 0), seconds=30.0, seed=seed)
        assert url.run(RunConfig(push_generator=kicks, **base)).survived
        assert not url.run(RunConfig(push_generator=punches, **base)).survived


def test_armature_zero_demo(url):
    from gaitkeeper.checks import s17a_margin, s17b_modes

    cfg = dict(model_edit=VARIANTS["armature 0"][0])
    pol = url.run(RunConfig(backend="python_pd", command=(0.5, 0, 0), **cfg))
    zero = url.run(RunConfig(backend="python_pd", policy_mode="zero", command=(0, 0, 0), **cfg))
    assert pol.fell_at is not None and pol.fell_at < 1.0
    assert zero.fell_at == pytest.approx(0.62, abs=0.1)  # baseline on the same model and backend
    a = s17a_margin(url, "python_pd", cfg=RunConfig(**cfg))
    b = s17b_modes(url, "python_pd", cfg=RunConfig(**cfg))
    assert a.status == "WARN" and b.status == "FAIL"
    tc = b.data["slow_tc"][0]
    assert 0.2 < tc < 0.6
    chk = {"s17a": a.status, "s17a_over": 1, "s17b": b.status, "tipping_tc": tc}
    runs = [
        {
            "scenario": "fwd",
            "fell_at": pol.fell_at,
            "at_limit": [n for n, f in zip(pol.names, pol.sat_frac) if f > 0.01],
        }
    ]
    assert fall_comparison(runs, {"fell_at": zero.fell_at}, chk)["suppressed"]
    nat = url.run(RunConfig(backend="native_implicit", command=(0.5, 0, 0), **cfg))
    assert nat.survived and not math.isnan(nat.vx_tail) and nat.vx_tail > 0.4
    assert np.isfinite(pol.dist)
