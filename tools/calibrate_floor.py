"""Calibrate the boundary D floor for a source engine (plan section 7.5).

Each trace is analysed against the model it was simulated on (the recorder's
compiled model, rebuilt from the XML and patch when the MuJoCo versions
differ), with the contract the recorder wrote next to it. The floor is the
largest clean-step RMS per joint and the largest root residual RMS over the
traces. Writes or updates ``src/gaitkeeper/data/floors.json``.

    python tools/calibrate_floor.py runs/g1_golden_a runs/g1_golden_b runs/g1_golden_c

A trace from an engine other than MuJoCo carries no MuJoCo model. Give one with ``--mjcf``;
for Isaac Lab traces the joint armature the recorder read from PhysX
(``isaac_model.json``) is written into it by joint name. The floor then also holds the
difference between that MJCF and the simulated asset, so it is an upper bound, and the
floor table names the MJCF it was calibrated against.

    python tools/calibrate_floor.py runs/isaac_golden_a runs/isaac_golden_b \\
        --mjcf <unitree_mujoco scene_29dof.xml> --frictionless
"""

import argparse
import json
from pathlib import Path

from gaitkeeper.contract import Contract
from gaitkeeper.residual import FLOORS_PATH, calibrate, engine_key
from gaitkeeper.trace import Trace


def source_model(mjcf: str, trace: Trace, frictionless: bool):
    """The given MJCF with the source's own joint armature, when the trace recorded it."""
    import mujoco

    from gaitkeeper.inject import physics_edit
    from gaitkeeper.models import load_model

    m = load_model(mjcf).model
    if frictionless:
        physics_edit("frictionless")(m)
    live = Path(trace.path) / "isaac_model.json" if trace.path else None
    if live and live.exists():
        d = json.loads(live.read_text())
        for name, arm in zip(d["joint_names"], d["armature"]):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j < 0:
                raise SystemExit(f"{mjcf}: no joint {name} (from {live})")
            m.dof_armature[m.jnt_dofadr[j]] = float(arm)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=str(FLOORS_PATH))
    ap.add_argument("--mjcf", help="analysis model for traces without their own MuJoCo model")
    ap.add_argument("--frictionless", action="store_true", help="with --mjcf: zero joint friction")
    args = ap.parse_args()
    traces = [Trace.load(p) for p in args.traces]
    keys = {engine_key(t.meta) for t in traces}
    if len(keys) != 1:
        raise SystemExit(f"traces come from more than one engine: {sorted(keys)}")
    key = keys.pop()
    floor = calibrate(
        traces,
        contract_of=lambda t: Contract.load(Path(t.path) / "contract.live.yaml"),
        model_of=(
            (lambda t: source_model(args.mjcf, t, args.frictionless))
            if args.mjcf
            else (lambda t: t.path)
        ),
    )
    if args.mjcf:
        floor["analysis_model"] = Path(args.mjcf).name + (
            " (frictionless)" if args.frictionless else ""
        )
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {"engines": {}}
    old = data["engines"].get(key)
    if old:
        for k in ("joint_rms", "joint_max"):
            for n, v in old[k].items():
                floor[k][n] = max(floor[k].get(n, 0.0), v)
        floor["root_force_rms"] = max(floor["root_force_rms"], old["root_force_rms"])
        floor["root_torque_rms"] = max(floor["root_torque_rms"], old["root_torque_rms"])
        floor["analysis_mujoco"] = sorted(
            set(old["analysis_mujoco"]) | set(floor["analysis_mujoco"])
        )
    floor["calibrated_on"] = [Path(p).name for p in floor["calibrated_on"]]
    data["engines"][key] = floor
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    worst = max(floor["joint_rms"].items(), key=lambda kv: kv[1])
    print(
        f"{key}: {len(floor['joint_rms'])} joints, worst {worst[0]} {worst[1]:.4g} N m, "
        f"root {floor['root_force_rms']:.4g} N / {floor['root_torque_rms']:.4g} N m, "
        f"analysis MuJoCo {floor['analysis_mujoco']}"
    )


if __name__ == "__main__":
    main()
