"""Publish recorded golden traces as a Hugging Face dataset and pin them for `gaitkeeper fetch golden`.

Dry run (the default): lists each file with its sha256 and writes the dataset card to
``--card-out`` so it can be read before anything leaves the machine.

    python tools/publish_golden.py runs/g1_golden_a runs/g1_golden_b runs/g1_golden_c \\
        --repo <hf-user>/gaitkeeper-golden

Upload (needs ``pip install huggingface_hub`` and ``huggingface-cli login``):

    python tools/publish_golden.py runs/g1_golden_* --repo <hf-user>/gaitkeeper-golden \\
        --license <spdx-id> --upload

The upload creates the dataset if needed, uploads every file of each trace directory under
the directory's name, reads the resulting revision, and adds one ``golden`` set per trace to
``src/gaitkeeper/data/fixtures.json`` with that revision and every file's sha256. Commit that
file; `gaitkeeper fetch golden` then downloads exactly those bytes.

A license has to be chosen explicitly. The traces contain the outputs of the policy that was
recorded, so its license (or the lack of one) matters as much as yours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIXTURES = Path(__file__).parents[1] / "src" / "gaitkeeper" / "data" / "fixtures.json"
SKIP = {".DS_Store"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_of(run: Path) -> dict[str, Path]:
    out = {}
    for p in sorted(run.rglob("*")):
        if (
            p.is_file()
            and p.name not in SKIP
            and "__pycache__" not in p.parts
            and not p.name.endswith(".part")
        ):
            out[p.relative_to(run).as_posix()] = p
    return out


def describe(run: Path) -> dict:
    m = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    fw = m.get("framework", {})
    return {
        "name": run.name,
        "framework": " ".join(f"{k} {v}" for k, v in fw.items() if isinstance(v, str)) or "unknown",
        "task": m.get("task", "unknown"),
        "seed": m.get("seed", "unknown"),
        "sim_dt": m.get("sim_dt"),
        "decimation": m.get("decimation"),
        "policy_sha256": (m.get("policy") or {}).get("sha256", "unknown"),
        "under_excited": m.get("under_excited", []),
    }


def card(repo: str, license_id: str | None, runs: list[dict], sizes: dict[str, int]) -> str:
    rows = "\n".join(
        f"| `{r['name']}` | {r['framework']} | `{r['task']}` | {r['seed']} | "
        f"{r['sim_dt']} s x {r['decimation']} | {sizes[r['name']] / 1e6:.0f} MB |"
        for r in runs
    )
    lic = f"license: {license_id}\n" if license_id else ""
    return f"""---
{lic}pretty_name: gaitkeeper golden traces
tags:
- robotics
- humanoid
- locomotion
- mujoco
- sim-to-sim
---

# gaitkeeper golden traces

Ground truth recorded in a policy's training simulator, for
[gaitkeeper](https://github.com/dundysm/gaitkeeper): every policy input and output at the
control rate, the full simulator state at every physics step, the live contract
(`contract.live.yaml`), and the compiled model the trace was simulated on. gaitkeeper checks
a deployment against these, boundary by boundary.

| Trace | Source engine | Task | Seed | Step | Size |
|---|---|---|---|---|---|
{rows}

## Use

```bash
pip install gaitkeeper
gaitkeeper fetch golden        # downloads at a pinned revision and checks every sha256
gaitkeeper verify ~/.cache/gaitkeeper/g1_golden_a --onnx <policy.onnx> --yaml <deploy.yaml>
```

## Contents of each trace

| File | What |
|---|---|
| `golden.npz` | Control-rate arrays (`obs`, `action`, `command`, `qpos`, `qvel`, ...) and physics-rate `p/` arrays |
| `manifest.json` | Framework versions, seed, schedule, step sizes, state layout, excitation checks |
| `contract.live.yaml` | The contract read from the live training objects, with provenance |
| `model.mjb`, `model_patch.json`, `model_xml/` | The model as simulated, including startup randomization (`dr.json`) |

Recorded with `tools/record_mjlab.py`. Observation noise, pushes and standing envs are off;
the schedule changes forward, lateral and yaw commands and includes small commands.

## Provenance and license

The traces contain the outputs of the policy that was recorded (`policy_sha256` in each
manifest) and a model compiled from its robot description. Check those sources' licenses
before reusing the traces. Dataset: `{repo}`.
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--repo", required=True, help="Hugging Face dataset id, <user>/<name>")
    ap.add_argument("--license", help="SPDX id for the dataset card; required with --upload")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument(
        "--private",
        action="store_true",
        help="create the dataset private (fetch then needs a token)",
    )
    ap.add_argument("--card-out", type=Path, default=Path("runs/golden_dataset_card.md"))
    ap.add_argument("--fixtures", type=Path, default=FIXTURES)
    args = ap.parse_args()

    runs, listing, sizes = [], {}, {}
    for run in args.runs:
        if not (run / "golden.npz").exists():
            raise SystemExit(f"{run}: no golden.npz; not a recorded trace directory")
        files = files_of(run)
        listing[run.name] = {rel: (p, sha256(p)) for rel, p in files.items()}
        sizes[run.name] = sum(p.stat().st_size for p in files.values())
        runs.append(describe(run))

    for r in runs:
        print(
            f"{r['name']}: {r['framework']}, task {r['task']}, seed {r['seed']}, {sizes[r['name']] / 1e6:.0f} MB"
        )
        if r["under_excited"]:
            print(f"  warning: under-excited items {r['under_excited']}")
        for rel, (_, digest) in listing[r["name"]].items():
            print(f"  {rel:40s} {digest}")

    text = card(args.repo, args.license, runs, sizes)
    args.card_out.parent.mkdir(parents=True, exist_ok=True)
    args.card_out.write_text(text)
    print(f"dataset card: {args.card_out}")

    if not args.upload:
        print("dry run: nothing uploaded. Read the card, then rerun with --license <id> --upload")
        return
    if not args.license:
        raise SystemExit("--upload needs --license: choose one for the dataset card first")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=text.encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    for run in args.runs:
        api.upload_folder(
            folder_path=str(run),
            path_in_repo=run.name,
            repo_id=args.repo,
            repo_type="dataset",
            commit_message=f"Add {run.name}",
            ignore_patterns=["*.part", "__pycache__/*", ".DS_Store"],
        )
    revision = api.dataset_info(args.repo).sha
    print(f"uploaded to https://huggingface.co/datasets/{args.repo} at revision {revision}")

    data = json.loads(args.fixtures.read_text())
    for r in runs:
        data["sets"][r["name"]] = {
            "repo": args.repo,
            "host": "huggingface",
            "group": "golden",
            "about": f"golden trace, {r['framework']}, task {r['task']}, seed {r['seed']}",
            "license": f"{args.license} (dataset card); the recorded policy's own license applies to its outputs",
            "commit": revision,
            "files": {
                rel: {"path": f"{r['name']}/{rel}", "sha256": digest}
                for rel, (_, digest) in listing[r["name"]].items()
            },
        }
    args.fixtures.write_text(json.dumps(data, indent=1) + "\n")
    print(
        f"pinned {len(runs)} golden set(s) in {args.fixtures}; commit it, then `gaitkeeper fetch golden`"
    )


if __name__ == "__main__":
    main()
