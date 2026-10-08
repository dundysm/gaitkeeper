"""Calibrate the boundary D floor for a source engine (plan section 7.5).

Each trace is analysed against the model it was simulated on (the recorder's
compiled model, rebuilt from the XML and patch when the MuJoCo versions
differ), with the contract the recorder wrote next to it. The floor is the
largest clean-step RMS per joint and the largest root residual RMS over the
traces. Writes or updates ``src/sim2sim/data/floors.json``.

    python tools/calibrate_floor.py runs/g1_golden_a runs/g1_golden_b runs/g1_golden_c
"""

import argparse
import json
from pathlib import Path

from sim2sim.contract import Contract
from sim2sim.residual import FLOORS_PATH, calibrate, engine_key
from sim2sim.trace import Trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out", default=str(FLOORS_PATH))
    args = ap.parse_args()
    traces = [Trace.load(p) for p in args.traces]
    keys = {engine_key(t.meta) for t in traces}
    if len(keys) != 1:
        raise SystemExit(f"traces come from more than one engine: {sorted(keys)}")
    key = keys.pop()
    floor = calibrate(
        traces,
        contract_of=lambda t: Contract.load(Path(t.path) / "contract.live.yaml"),
        model_of=lambda t: t.path,
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
