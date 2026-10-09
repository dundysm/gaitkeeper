"""Blind test of `verify` on harness logs whose labels gaitkeeper's author has not seen.

The protocol is in docs/BLIND_TEST.md. In short: the submitter seals the labels with a
hash before the logs are run, gaitkeeper's outputs are committed with a hash before the
labels are revealed, and the score is computed only from those two committed files.

    # submitter, before sending anything
    python tools/blind.py seal labels.json            # prints the commitment to publish

    # maintainer, on the submitted folder (logs/*.npz plus contract and policy)
    python tools/blind.py run submission/              # writes submission/outputs.json, prints its hash

    # after both hashes are public, the submitter reveals labels.json
    python tools/blind.py score submission/ labels.json --labels-sha256 <hash> --outputs-sha256 <hash>

labels.json: {"salt": "<random text>", "cases": {"<log file stem>": {"kind": "contract" | "none",
"boundary": "A" | "B" | "C" | "A+C" | null, "tokens": ["<term id or joint name>", ...]}, ...}}

A case counts as a false confident attribution when `verify` says CONTRACT and the label is
"none", or names a cause missing any of the label's tokens. Tokens are names the submitter
knows without reading gaitkeeper: observation term ids from the contract (base_ang_vel,
joint_vel_rel, ...) and joint names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

from gaitkeeper.cli import _contract, _policy
from gaitkeeper.compare import verify
from gaitkeeper.metrics import Case, score
from gaitkeeper.trace import Trace


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_labels(d: dict) -> None:
    if len(str(d.get("salt", ""))) < 16:
        raise SystemExit(
            "labels.json needs a random salt of 16 or more characters, so the hash cannot be guessed"
        )
    for name, t in d.get("cases", {}).items():
        if t.get("kind") not in ("contract", "none"):
            raise SystemExit(f"{name}: kind must be 'contract' or 'none' for a harness log")
        if t["kind"] == "contract" and not t.get("boundary"):
            raise SystemExit(f"{name}: a contract label needs its boundary (A, B, C or A+C)")


def cmd_seal(args: Namespace) -> None:
    p = Path(args.labels)
    check_labels(json.loads(p.read_text()))
    print(f"labels sha256 {digest(p)}")
    print(
        "publish this hash (an issue comment works) before sending the logs; keep labels.json private"
    )


def submission_inputs(folder: Path) -> Namespace:
    a = Namespace(
        contract=None,
        deploy=None,
        onnx=None,
        yaml=None,
        policy=None,
        robot="unitree_g1_29dof",
        preset=[],
    )
    if (folder / "contract.yaml").exists():
        a.contract = str(folder / "contract.yaml")
    elif (folder / "deploy.yaml").exists():
        a.deploy = str(folder / "deploy.yaml")
    elif (folder / "export.yaml").exists():
        a.yaml = str(folder / "export.yaml")
    onnx = sorted(folder.glob("*.onnx"))
    if not onnx:
        raise SystemExit(f"{folder}: no policy .onnx")
    a.onnx = a.policy = str(onnx[0])
    meta = folder / "submission.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        a.robot = m.get("robot", a.robot)
        a.preset = m.get("presets", [])
    return a


def cmd_run(args: Namespace) -> None:
    folder = Path(args.folder)
    inputs = submission_inputs(folder)
    contract = _contract(inputs)
    policy = _policy(inputs, contract)
    logs = sorted((folder / "logs").glob("*.npz"))
    if not logs:
        raise SystemExit(f"{folder}/logs: no .npz harness logs")
    out = {"gaitkeeper_commit": args.commit, "cases": {}}
    for p in logs:
        rep = verify(Trace.load(p), contract, policy)
        out["cases"][p.stem] = {
            "verdict": rep.verdict,
            "evidence": rep.evidence,
            "failing": sorted(k for k, b in rep.boundaries.items() if b.status == "fail"),
            "cause": " | ".join(x for b in rep.boundaries.values() for x in b.patterns) or None,
            "findings": list(rep.findings),
        }
        print(f"{p.stem}: {rep.verdict}")
    dest = folder / "outputs.json"
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"outputs sha256 {digest(dest)}  ({dest})")
    print("publish this hash before the labels are revealed")


def cmd_score(args: Namespace) -> None:
    folder = Path(args.folder)
    outputs, labels = folder / "outputs.json", Path(args.labels)
    for p, want in ((labels, args.labels_sha256), (outputs, args.outputs_sha256)):
        if digest(p) != want:
            raise SystemExit(f"{p}: sha256 {digest(p)} is not the committed {want}; not scoring")
    lab = json.loads(labels.read_text())
    check_labels(lab)
    out = json.loads(outputs.read_text())["cases"]
    missing = sorted(set(lab["cases"]) ^ set(out))
    if missing:
        raise SystemExit(f"labels and outputs cover different logs: {missing}")
    cases = [
        Case(name, lab["cases"][name], o["verdict"], o["cause"], [], o["failing"], o["findings"])
        for name, o in sorted(out.items())
    ]
    res = score(cases)
    res["note"] = "blind: labels sealed before the run, outputs sealed before the reveal"
    dest = folder / "blind_results.json"
    dest.write_text(json.dumps(res, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: res[k]
                for k in ("cases", "confident", "false_confident", "abstention_rate", "detection")
            },
            indent=1,
        )
    )
    print(f"written {dest}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("labels")
    s.set_defaults(fn=cmd_seal)
    r = sub.add_parser("run")
    r.add_argument("folder")
    r.add_argument("--commit", default="unknown", help="gaitkeeper commit the run used")
    r.set_defaults(fn=cmd_run)
    c = sub.add_parser("score")
    c.add_argument("folder")
    c.add_argument("labels")
    c.add_argument("--labels-sha256", required=True)
    c.add_argument("--outputs-sha256", required=True)
    c.set_defaults(fn=cmd_score)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
