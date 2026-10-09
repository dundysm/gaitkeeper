"""Build results.json: the false confident attribution rate, the abstention rate
and the detection rate per boundary over every case with a known cause
(plan section 10).

Cases:

* the synthetic injection corpus (``sim2sim.inject.defects``) on the test
  harness, checked by ``verify``;
* the development set of harness logs from the unitree_rl_lab G1 deploy files
  (29 logs, labels below), checked by ``verify``;
* physics and task cases on the mjlab golden trace: two positive controls
  (floor friction 0.1, torso +15 kg), AT1, AT9 and AT15, through
  ``diagnose_trace``.

Targets whose truth is unknown (unitree_mujoco and Menagerie G1 against the
mjlab trace) are reported from ``runs/step5`` but left out of the metric.

    python tools/results.py --out results.json
"""

import argparse
import json
import os
import sys
import tempfile
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from assets import DATA, GOLDEN_A, MJLAB_ONNX, RUNS, UMJ_G1, URL_G1  # noqa: E402

from sim2sim.compare import verify  # noqa: E402
from sim2sim.contract import Contract  # noqa: E402
from sim2sim.metrics import Case, score  # noqa: E402

PRESET = "unitree_rl_lab_g1_29dof_velocity@4960b84"
# The 29 development harness logs are not public; point --devset or
# SIM2SIM_DEVSET at them. Without them the metric leaves the development set out.
DEVSET = Path(os.environ.get("SIM2SIM_DEVSET", RUNS / "devset"))

# case: (boundary that must fail, substrings the named patterns must contain)
DEVSET_LABELS = {
    "00_correct": (None, []),
    "01_hist_init_zeros": ("A", ["history init is zeros"]),
    "02_hist_newest_first": ("A", ["history order is newest_first"]),
    "03_hist_frame_major": ("A", ["history layout is time_major"]),
    "04_kp_x0.7": ("C", ["kp scaled x0.7"]),
    "05_kp_x1.3": ("C", ["kp scaled x1.3"]),
    "06_kd_x0.5": ("C", ["kd scaled x0.5"]),
    "07_kd_x2_native": ("C", ["kd scaled x2"]),
    "08_gains_asset": ("C", ["kp and kd differ on", "waist_roll_joint"]),
    "09_gains_asset_waist": ("C", ["kp differ on ['waist_roll_joint', 'waist_pitch_joint']"]),
    "10_gains_asset_arms": ("C", ["kd differ on", "left_shoulder_pitch_joint"]),
    "11_angvel_scale_0.25": ("A", ["base_ang_vel: constant scale x1.25"]),
    "12_angvel_scale_1.0": ("A", ["base_ang_vel: constant scale x5"]),
    "13_angvel_zeroed": ("A", ["base_ang_vel: zero (term not filled)"]),
    "14_angvel_world_yaw90": ("A", ["base_ang_vel: angular velocity in the world frame"]),
    "15_jointvel_scale_0.1": ("A", ["joint_vel_rel: constant scale x2"]),
    "16_jointvel_zeroed": ("A", ["joint_vel_rel: zero (term not filled)"]),
    "17_jointpos_scale_1.1": ("A", ["joint_pos_rel: constant scale x1.1"]),
    "18_gravity_sign": ("A", ["projected_gravity: sign flip"]),
    "19_last_action_zeroed": ("A", ["last_action: zero (term not filled)"]),
    "20_action_delay_1": ("C", ["1 policy step late"]),
    "21_action_delay_2": ("C", ["2 policy step late"]),
    "22_action_offset_dropped": ("C", ["default offset not added"]),
    "23_action_scale_0.5": ("C", ["action scale x2 on every joint"]),
    "24_action_scale_0.2": ("C", ["action scale x0.8 on every joint"]),
    "25_cmd_obs_x2": ("A", ["velocity_commands: constant scale x2"]),
    "26_cmd_obs_x0.5": ("A", ["velocity_commands: constant scale x0.5"]),
    "27_quat_xyzw": ("A", ["(w, x, y, z) read as (x, y, z, w)"]),
    "28_joint_remap_skipped": ("A+C", ["remap skipped"]),
}


def _from_report(name, truth, rep) -> Case:
    failing = sorted(k for k, b in rep.boundaries.items() if b.status == "fail")
    text = " | ".join(p for b in rep.boundaries.values() for p in b.patterns)
    return Case(name, truth, rep.verdict, text or None, [], failing, list(rep.findings))


def injection_cases() -> list[Case]:
    import synth

    from sim2sim.inject import Harness, defects, history5

    truth_c, files = synth.make_contract(rounded=False), synth.make_contract(rounded=True)
    pol = synth.LinearPolicy()
    g = synth.make_golden()
    first = Harness(g, truth_c, pol, files).standard()
    g.arrays["obs"], g.arrays["action"] = first["obs"], first["action"]
    h = Harness(g, truth_c, pol, files)
    out = []
    for d in defects():
        contract = history5(files) if d.contract_variant == "history5" else files
        rep = verify(d.build(h), contract, None if d.contract_variant == "history5" else pol)
        if d.expect_status == "fail":
            truth = {
                "kind": "contract",
                "boundary": d.expect_boundary,
                "tokens": [x for x in (d.expect_pattern or "").split("|") if x],
            }
        else:
            truth = {"kind": "none", "boundary": None}
        out.append(_from_report(f"inject: {d.name}", truth, rep))
    return out


def devset_cases(logs: Path) -> list[Case]:
    from sim2sim.policy import OnnxPolicy
    from sim2sim.presets import apply_preset
    from sim2sim.readers.unitree_deploy import read_unitree_deploy
    from sim2sim.trace import Trace

    c, _ = read_unitree_deploy(URL_G1 / "deploy.yaml", URL_G1 / "policy.onnx")
    apply_preset(c, PRESET)
    pol = OnnxPolicy(URL_G1 / "policy.onnx")
    out = []
    for case, (want, subs) in DEVSET_LABELS.items():
        p = logs / f"{case}.npz"
        if not p.exists():
            continue
        rep = verify(Trace.load(p), c, pol)
        truth = (
            {"kind": "none", "boundary": None}
            if want is None
            else {"kind": "contract", "boundary": want, "tokens": subs}
        )
        out.append(_from_report(f"devset: {case}", truth, rep))
    return out


def _from_diagnosis(name, truth, dg) -> Case:
    dec = dg.decision
    groups = list(dg.cf.localized) if dg.cf and dec.verdict == "PHYSICS" else []
    failing = [
        k for k, b in (dg.report.boundaries if dg.report else {}).items() if b.status == "fail"
    ]
    return Case(name, truth, dec.verdict, dec.cause, groups, failing, list(dec.findings))


def physics_cases(workers: int | None) -> list[Case]:
    from sources import concat, external, runner_trace, schedule_of

    from sim2sim import diagnose as D
    from sim2sim import residual as R
    from sim2sim.inject import compose, physics_edit
    from sim2sim.models import load_model
    from sim2sim.policy import OnnxPolicy
    from sim2sim.task import TaskSpec
    from sim2sim.trace import Trace

    tr = Trace.load(GOLDEN_A)
    c = Contract.load(GOLDEN_A / "contract.live.yaml")
    pol, onnx = OnnxPolicy(MJLAB_ONNX), str(MJLAB_ONNX)
    seeds = tuple(range(1, 13))
    dead = TaskSpec(
        "dead zone",
        7.0,
        [(0.0, (0.0, 0.0, 0.0)), (1.0, (0.05, 0.0, 0.0)), (4.0, (0.0, 0.0, 0.08))],
        [],
    )
    out = []

    def golden(name, truth, **kw):
        dg = D.diagnose_trace(tr, c, pol, onnx, str(GOLDEN_A), seeds, workers=workers, **kw)
        out.append(_from_diagnosis(name, truth, dg))

    golden(
        "positive: floor friction 0.1 on the target",
        {"kind": "physics", "boundary": "D", "groups": ["contact parameters"]},
        target_edit=physics_edit("floor_friction", mu=0.1),
    )
    golden(
        "positive: torso +15 kg on the target",
        {"kind": "physics", "boundary": "D", "groups": ["mass and inertia"]},
        target_edit=physics_edit("body_mass", body="torso_link", delta=15.0),
    )
    at1 = compose(physics_edit("armature", value=0.0), physics_edit("floor_friction", mu=2.0))
    golden(
        "AT1: armature 0 and floor friction 2.0 on the target",
        {"kind": "physics", "boundary": "D", "groups": ["armature", "contact parameters"]},
        target_edit=at1,
    )
    golden("AT9: matched model, dead zone task", {"kind": "task", "boundary": None}, task=dead)

    with tempfile.TemporaryDirectory() as tmp:
        s1 = [
            (0.0, (0.0, 0.0, 0.0)),
            (1.0, (0.5, 0.0, 0.0)),
            (4.0, (0.0, 0.4, 0.0)),
            (7.0, (0.0, 0.0, 0.5)),
        ]
        s2 = [(0.0, (0.0, 0.0, 0.0)), (1.0, (-0.4, 0.0, 0.0)), (4.0, (0.6, -0.2, 0.4))]
        eps = [
            runner_trace(c, UMJ_G1, MJLAB_ONNX, s1, 9.0, at1),
            runner_trace(c, UMJ_G1, MJLAB_ONNX, s2, 7.0, at1, seed=1),
        ]
        src = external(
            concat(*eps), Path(tmp) / "at1", UMJ_G1, at1, schedule_of((s1, 9.0), (s2, 7.0))
        )
        out.append(
            _from_report(
                "AT1: trace recorded on armature 0 and floor friction 2.0",
                {"kind": "physics", "boundary": None},
                verify(src, c, pol),
            )
        )

        edit = physics_edit("frictionless", timestep=0.005)
        s1 = [
            (0.0, (0.0, 0.0, 0.0)),
            (1.0, (0.5, 0.0, 0.0)),
            (3.0, (0.0, 0.4, 0.0)),
            (5.0, (0.0, 0.0, 0.5)),
            (7.0, (0.4, 0.0, 0.5)),
            (9.0, (0.0, 0.0, 0.0)),
        ]
        s2 = [
            (0.0, (0.0, 0.0, 0.0)),
            (1.0, (0.8, -0.3, 0.3)),
            (3.0, (-0.4, 0.0, 0.0)),
            (5.0, (0.0, -0.4, 0.0)),
            (7.0, (0.0, 0.0, 0.0)),
        ]

        def episodes(backend):
            return concat(
                runner_trace(c, UMJ_G1, MJLAB_ONNX, s1, 11.0, edit, backend),
                runner_trace(c, UMJ_G1, MJLAB_ONNX, s2, 9.0, edit, backend, seed=1),
            )

        native = episodes("native_implicit")
        src = external(
            episodes("standin_implicit"),
            Path(tmp) / "at15",
            UMJ_G1,
            edit,
            schedule_of((s1, 11.0), (s2, 9.0)),
        )
        target = load_model(UMJ_G1).model
        edit(target)
        floors = {src.meta["engine"]: R.calibrate([native], lambda t: c, lambda t: target)}
        saved = D.dynamics_residual
        D.dynamics_residual = partial(saved, floors=floors)
        try:
            dg = D.diagnose_trace(
                src, c, pol, onnx, str(UMJ_G1), seeds, task=dead, workers=workers, target_edit=edit
            )
        finally:
            D.dynamics_residual = saved
        out.append(
            _from_diagnosis(
                "AT15: stand-in drive source, dead zone task",
                {"kind": "task", "boundary": None},
                dg,
            )
        )
    return out


def reported_only() -> list[dict]:
    out = []
    for name in ("cf_c_umj", "cf_c_menagerie"):
        p = RUNS / "step5" / f"{name}.json"
        if p.exists():
            j = json.loads(p.read_text())
            out.append({"name": name, "truth": "unknown", "counterfactual": j})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--devset", default=str(DEVSET))
    ap.add_argument("--skip-physics", action="store_true")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    dev = devset_cases(Path(args.devset))
    if not dev:
        print(f"no development logs in {args.devset}: the metric leaves them out", file=sys.stderr)
    cases = injection_cases() + dev
    if not args.skip_physics:
        cases += physics_cases(args.workers)
    res = {
        "metric": score(cases),
        "cases": [
            {
                "name": c.name,
                "truth": c.truth,
                "verdict": c.verdict,
                "cause": c.cause,
                "groups": c.groups,
                "failing": c.boundaries,
            }
            for c in cases
        ],
        "reported_not_scored": reported_only(),
        "data": str(DATA),
    }
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    m = res["metric"]
    print(
        f"{m['cases']} cases, {m['confident']} confident, false confident {m['false_confident']} "
        f"(rate {m['false_confident_attribution_rate']:.3f}), abstention {m['abstention_rate']:.3f}"
    )
    for b, v in m["detection"].items():
        print(f"  boundary {b}: detected {v['detected']} of {v['cases']}")
    for name, why in m["false_confident_cases"]:
        print(f"  WRONG {name}: {why}")
    return 1 if m["false_confident"] else 0


if __name__ == "__main__":
    sys.exit(main())
