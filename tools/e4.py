"""Analyse the E4 traces (plan Appendix B) on the CPU.

    python tools/e4.py --contract contract.yaml --mjcf <scene.xml> --frictionless \\
        runs/e4_stock_5ms runs/e4_stock_2p5ms --ankle runs/e4_ankle_5ms --ankle-change 0.01

The analysis model is the target MJCF at each trace's step, with the
contract's drive. Joint friction in the analysis model leaves a joint out of
the linear fits, so pass --frictionless or accept fewer joints.
"""

import argparse
import json
import sys
from pathlib import Path

from gaitkeeper.contract import Contract
from gaitkeeper.e4 import e4
from gaitkeeper.trace import Trace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stock", nargs="+", help="stock traces at two or more step sizes")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--ankle", help="trace with the ankle armature changed in the source")
    ap.add_argument("--ankle-change", type=float, default=0.01)
    ap.add_argument("--json")
    ap.add_argument(
        "--frictionless",
        action="store_true",
        help="zero the target's joint friction and passive damping, so every joint can be fit",
    )
    args = ap.parse_args()
    target = args.mjcf
    if args.frictionless:
        from gaitkeeper.inject import physics_edit
        from gaitkeeper.models import load_model

        target = load_model(args.mjcf).model
        physics_edit("frictionless")(target)
    r = e4(
        [Trace.load(p) for p in args.stock],
        Contract.load(args.contract),
        target,
        Trace.load(args.ankle) if args.ankle else None,
        args.ankle_change if args.ankle else None,
    )
    print("\n".join(r.lines()))
    if args.json:
        Path(args.json).write_text(json.dumps(r.to_json(), indent=1))
    return 0 if r.explained and r.ankle_ok is not False else 5


if __name__ == "__main__":
    sys.exit(main())
