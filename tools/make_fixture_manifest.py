"""Write src/gaitkeeper/data/fixtures.json from local clones at the pinned commits.

Each fixture set lists files by their path in the upstream repository, with
sha256, so `gaitkeeper fetch` can download them from the commit and check them.
Meshes are the ones the MJCF references, not the whole directory.

    python tools/make_fixture_manifest.py --unitree-rl-lab <clone> --unitree-rl-mjlab <clone> \\
        --unitree-mujoco <clone> --menagerie <clone>
"""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

OUT = Path(__file__).parents[1] / "src" / "gaitkeeper" / "data" / "fixtures.json"

SETS = {
    "g1_rl_lab": {
        "repo": "unitreerobotics/unitree_rl_lab",
        "arg": "unitree_rl_lab",
        "about": "unitree_rl_lab G1 29 dof velocity policy as shipped (deploy.yaml, policy.onnx)",
        "license": "no license file in the repository at this commit",
        "files": {
            "deploy.yaml": "deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml",
            "policy.onnx": "deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx",
        },
    },
    "g1_rl_mjlab": {
        "repo": "unitreerobotics/unitree_rl_mjlab",
        "arg": "unitree_rl_mjlab",
        "about": "unitree_rl_mjlab G1 velocity policy as shipped (deploy.yaml, policy.onnx)",
        "license": "no license file in the repository at this commit",
        "files": {
            "deploy.yaml": "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml",
            "policy.onnx": "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx",
        },
    },
    "g1_unitree_mujoco": {
        "repo": "unitreerobotics/unitree_mujoco",
        "arg": "unitree_mujoco",
        "about": "unitree_mujoco G1 29 dof scene (the Unitree simulator's model)",
        "license": "BSD-3-Clause (LICENSE)",
        "dir": "unitree_robots/g1",
        "scene": "scene_29dof.xml",
        "extra": ["LICENSE"],
    },
    "g1_menagerie": {
        "repo": "google-deepmind/mujoco_menagerie",
        "arg": "menagerie",
        "about": "MuJoCo Menagerie Unitree G1 scene",
        "license": "BSD-3-Clause (unitree_g1/LICENSE)",
        "dir": "unitree_g1",
        "scene": "scene.xml",
        "extra": ["unitree_g1/LICENSE"],
    },
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(repo: Path, *a: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *a], text=True).strip()


def referenced(scene: Path) -> list[Path]:
    """The scene, its includes, and every file they reference, relative to the scene's dir."""
    out, todo = [], [scene.name]
    base = scene.parent
    while todo:
        name = todo.pop()
        if name in out:
            continue
        out.append(name)
        text = (base / name).read_text()
        meshdir = re.search(r'meshdir="([^"]+)"', text)
        for m in re.finditer(r"<include\s+file=\"([^\"]+)\"", text):
            todo.append(m.group(1))
        for m in re.finditer(r"<(mesh|texture|hfield)\b[^>]*\bfile=\"([^\"]+)\"", text):
            d = meshdir.group(1) + "/" if meshdir else ""
            out.append(d + m.group(2))
    return [Path(n) for n in dict.fromkeys(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    for s in SETS.values():
        ap.add_argument("--" + s["arg"].replace("_", "-"), required=True)
    args = ap.parse_args()
    data = {"version": 1, "sets": {}}
    for name, s in SETS.items():
        repo = Path(getattr(args, s["arg"]))
        commit = git(repo, "rev-parse", "HEAD")
        files = {}
        if "files" in s:
            pairs = list(s["files"].items())
        else:
            rels = referenced(repo / s["dir"] / s["scene"])
            pairs = [(str(r), f"{s['dir']}/{r}") for r in rels]
            pairs += [(Path(e).name, e) for e in s["extra"]]
        for local, upstream in pairs:
            if git(repo, "status", "--porcelain", "--", upstream):
                raise SystemExit(f"{repo}: {upstream} differs from the commit")
            files[local] = {"path": upstream, "sha256": sha(repo / upstream)}
        entry = {k: s[k] for k in ("repo", "about", "license")}
        entry["commit"] = commit
        if "scene" in s:
            entry["scene"] = s["scene"]
        entry["files"] = files
        data["sets"][name] = entry
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(
        f"wrote {OUT}: "
        + ", ".join(f"{k} {len(v['files'])} files" for k, v in data["sets"].items())
    )


if __name__ == "__main__":
    main()
