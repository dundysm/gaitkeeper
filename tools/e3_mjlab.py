"""Experiment E3 on the mjlab stack: verify a golden trace against the contract
read from the exported files, compare exported files with the live env, and run
the injected-defect suite. Writes runs/e3/report.json and runs/e3/report.md.

Usage:
    python tools/e3_mjlab.py --golden runs/g1_golden_a --golden-b runs/g1_golden_b \\
        --onnx .../policy.onnx --yaml .../deploy.yaml --out runs/e3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from sim2sim.compare import TOLERANCE_CLASSES, Report, verify
from sim2sim.contract import Contract, compare_contracts
from sim2sim.inject import Harness, defects, history5, judge
from sim2sim.policy import OnnxPolicy
from sim2sim.readers.mjlab_export import read_mjlab_export
from sim2sim.trace import Trace

DIFF_PATHS = [
    "timing.policy_dt",
    "timing.sim_dt",
    "timing.decimation",
    "policy_io.joints.names",
    "policy_io.observation_groups.policy.history",
    "control.default_joint_pos",
    "control.actions.joint_pos.scale",
    "control.actions.joint_pos.offset",
    "control.actuators.kp",
    "control.actuators.kd",
    "control.actuators.kind",
    "control.actuators.pd_period",
    "control.actuators.integrator",
    "model.effort_limit",
    "model.armature",
    "model.joint_friction",
    "model.joint_damping",
    "policy_io.commands.base_velocity.limit",
    "policy_io.commands.base_velocity.trained",
]


def term_rows(rep: Report) -> list[dict[str, Any]]:
    rows = []
    for b in ("B", "A", "C"):
        br = rep.boundaries.get(b)
        if br is None:
            continue
        for t in br.terms:
            rows.append(
                {
                    "boundary": b,
                    "term": t.term,
                    "status": t.status,
                    "max_abs": t.max_abs,
                    "rms": t.rms,
                    "held_out_max_abs": t.held_out_max_abs,
                    "max_tol": t.max_tol,
                    "worst_ratio": t.worst_ratio,
                    "tol_class": t.tol_class,
                    "rows": t.n_rows,
                    "pattern": t.pattern,
                    "not_separated": t.not_separated,
                    "detail": _small(t.detail),
                }
            )
    return rows


def _small(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k not in ("per_joint_max_abs",)}


def determinism(a: Trace, b: Trace) -> dict[str, Any]:
    out = {}
    for k in sorted(a.arrays):
        if k not in b.arrays or a.arrays[k].dtype.kind not in "fiub":
            continue
        x, y = a.arrays[k], b.arrays[k]
        if x.shape != y.shape:
            out[k] = "shape differs"
            continue
        out[k] = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()) if x.size else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--golden-b")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--out", default="runs/e3")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    golden = Trace.load(args.golden)
    files, reader_findings = read_mjlab_export(args.onnx, args.yaml)
    files.save(out / "contract.files.yaml")
    live = Contract.load(Path(args.golden) / "contract.live.yaml")
    policy = OnnxPolicy(args.onnx)

    rep_files = verify(golden, files, policy)
    rep_live = verify(golden, live, policy)
    diffs = [d.__dict__ for d in compare_contracts(files, live, DIFF_PATHS)]
    for d in diffs:
        for k in ("left", "right"):
            if isinstance(d[k], (dict, list)) and len(json.dumps(d[k], default=str)) > 300:
                d[k] = "(long)"

    h = Harness(golden, live, policy, files)
    results = []
    for d in defects():
        log = d.build(h)
        contract = history5(files) if d.contract_variant == "history5" else files
        pol = None if d.contract_variant == "history5" else policy
        rep = verify(log, contract, pol)
        results.append(judge(d, rep))

    det = determinism(golden, Trace.load(args.golden_b)) if args.golden_b else None
    report = {
        "golden": {
            "path": args.golden,
            "steps": golden.n_steps,
            "physics_steps": int(len(golden.physics("step"))),
            "manifest": {
                k: golden.meta.get(k)
                for k in (
                    "state_check",
                    "obs_state_gap",
                    "excitation",
                    "under_excited",
                    "wall_time_s",
                    "versions",
                )
            },
        },
        "tolerance_classes": TOLERANCE_CLASSES,
        "verify_files_contract": {
            "verdict": rep_files.verdict,
            "evidence": rep_files.evidence,
            "label": rep_files.label,
            "findings": rep_files.findings,
            "boundaries": {
                k: {"status": b.status, "patterns": b.patterns, "notes": b.notes}
                for k, b in rep_files.boundaries.items()
            },
            "terms": term_rows(rep_files),
        },
        "verify_live_contract": {
            "verdict": rep_live.verdict,
            "evidence": rep_live.evidence,
            "boundaries": {k: b.status for k, b in rep_live.boundaries.items()},
            "terms": term_rows(rep_live),
        },
        "reader_findings": [f.__dict__ for f in reader_findings],
        "files_vs_live": diffs,
        "injections": results,
        "determinism_max_abs": det,
        "wall_s": time.time() - t0,
    }
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    (out / "report.md").write_text(markdown(report))
    print(markdown(report))


def markdown(r: dict[str, Any]) -> str:
    L = ["# E3 on the mjlab stack", ""]
    v = r["verify_files_contract"]
    L += [
        f"Golden trace {r['golden']['path']}: {r['golden']['steps']} policy steps, "
        f"{r['golden']['physics_steps']} physics steps.",
        "",
        f"Against the contract read from the exported files: verdict {v['verdict']}, evidence {v['evidence']}"
        + (f", label {v['label']}" if v["label"] else "")
        + ".",
        "",
    ]
    for k, b in v["boundaries"].items():
        L.append(
            f"* {k}: {b['status']}"
            + (f" ({'; '.join(b['patterns'])})" if b["patterns"] else "")
            + (f" notes: {'; '.join(b['notes'])}" if b["notes"] else "")
        )
    L += [
        "",
        "| boundary | term | status | max abs | rms | held-out max | max tol | worst err/tol | class |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for t in v["terms"]:
        L.append(
            f"| {t['boundary']} | {t['term']} | {t['status']} | {t['max_abs']:.3g} | {t['rms']:.3g} | "
            f"{t['held_out_max_abs']:.3g} | {t['max_tol']:.3g} | {t['worst_ratio']:.3g} | {t['tol_class']} |"
        )
    lv = r["verify_live_contract"]
    L += [
        "",
        f"Against the live contract: verdict {lv['verdict']}, boundaries {lv['boundaries']}.",
        "",
        "| boundary | term | status | max abs | max tol | class |",
        "|---|---|---|---|---|---|",
    ]
    for t in lv["terms"]:
        L.append(
            f"| {t['boundary']} | {t['term']} | {t['status']} | {t['max_abs']:.3g} | {t['max_tol']:.3g} | "
            f"{t['tol_class']} |"
        )
    L += [
        "",
        "## Injected defects",
        "",
        "| defect | expected | verdict | named | correct |",
        "|---|---|---|---|---|",
    ]
    for x in r["injections"]:
        named = "; ".join(f"{k}: {' / '.join(p)}" for k, p in x["named"].items()) or "; ".join(
            f"{k}: {' / '.join(n)}" for k, n in x["notes"].items()
        )
        L.append(
            f"| {x['defect']} | {x['expected']} | {x['verdict']} | {named} | {'yes' if x['correct'] else 'NO'} |"
        )
    L += ["", "## Exported files against the live env", ""]
    for d in r["files_vs_live"]:
        if d["kind"] != "agree":
            L.append(
                f"* {d['path']}: {d['kind']} (max abs {d['max_abs']:.3g}, max rel {d['max_rel']:.3g})"
                + (f" keys {d['keys'][:6]}" if d["keys"] else "")
            )
    for f in r["reader_findings"]:
        L.append(f"* reader: {f['path']}: {f['kind']}: {f['message']}")
    if r["determinism_max_abs"]:
        nz = {k: v for k, v in r["determinism_max_abs"].items() if v != 0.0}
        L += [
            "",
            "## Determinism (two recordings, same seed)",
            "",
            f"{len(r['determinism_max_abs'])} arrays compared; nonzero differences: {nz or 'none'}",
        ]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
