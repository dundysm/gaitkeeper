"""Command line: read a contract from exported files, verify a trace against it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compare import verify
from .contract import Contract
from .trace import Trace


def _contract(args: argparse.Namespace) -> Contract:
    if args.contract:
        return Contract.load(args.contract)
    if not args.onnx:
        sys.exit(
            "give --contract, or --onnx (and optionally --yaml) to read one from exported files"
        )
    from .readers.mjlab_export import read_mjlab_export

    c, findings = read_mjlab_export(args.onnx, args.yaml)
    for f in findings:
        print(f"reader {f.kind}: {f.path}: {f.message}", file=sys.stderr)
    return c


def cmd_inspect(args: argparse.Namespace) -> int:
    c = _contract(args)
    if args.out:
        c.save(args.out)
        print(f"wrote {args.out}")
    else:
        print(c.dump())
    unknown = c.unknown_fields()
    if unknown:
        print(f"unknown fields: {unknown}", file=sys.stderr)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    trace = Trace.load(args.trace)
    contract = _contract(args)
    policy = None
    pol_path = args.policy or args.onnx
    if pol_path:
        from .policy import OnnxPolicy

        policy = OnnxPolicy(pol_path)
    rep = verify(trace, contract, policy)
    print(rep.summary())
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "verdict": rep.verdict,
                    "evidence": rep.evidence,
                    "label": rep.label,
                    "findings": rep.findings,
                    "excitation": rep.excitation,
                    "boundaries": {
                        k: {
                            "status": b.status,
                            "patterns": b.patterns,
                            "notes": b.notes,
                            "terms": [t.__dict__ for t in b.terms],
                        }
                        for k, b in rep.boundaries.items()
                    },
                },
                indent=1,
                default=str,
            )
        )
    return {"PASS": 0, "UNDETERMINED": 3, "CONTRACT": 2}.get(rep.verdict, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sim2sim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def contract_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--contract", help="contract.yaml")
        p.add_argument("--onnx", help="exported policy; its metadata is read as the contract")
        p.add_argument("--yaml", help="deploy.yaml exported next to the policy")

    p = sub.add_parser("inspect", help="read a contract from exported files")
    contract_args(p)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("verify", help="check boundaries B, A, C of a trace against a contract")
    p.add_argument("trace", help="golden trace directory or harness log .npz")
    contract_args(p)
    p.add_argument("--policy", help="policy file for boundary B (defaults to --onnx)")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
