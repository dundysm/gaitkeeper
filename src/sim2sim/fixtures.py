"""Third-party models and policies used by the demo and the integration tests.

Nothing here is vendored. ``data/fixtures.json`` pins each set to an upstream
commit and lists its files with sha256; ``fetch`` downloads them from that
commit into a local directory and refuses any file whose hash differs.

The directory is ``$SIM2SIM_DATA`` when set, else ``$XDG_CACHE_HOME/sim2sim``
(``~/.cache/sim2sim``).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


class FixtureError(RuntimeError):
    pass


@dataclass
class FixtureSet:
    name: str
    repo: str
    commit: str
    about: str
    license: str
    files: dict[str, dict[str, str]]
    scene: str | None = None

    def dir(self, root: Path | None = None) -> Path:
        return (root or data_dir()) / self.name

    def path(self, local: str, root: Path | None = None) -> Path:
        return self.dir(root) / local

    def present(self, root: Path | None = None) -> bool:
        return all(self.path(f, root).exists() for f in self.files)


def data_dir() -> Path:
    env = os.environ.get("SIM2SIM_DATA")
    if env:
        return Path(env).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache) / "sim2sim"


def manifest() -> dict[str, FixtureSet]:
    text = resources.files("sim2sim").joinpath("data/fixtures.json").read_text()
    out = {}
    for name, s in json.loads(text)["sets"].items():
        out[name] = FixtureSet(
            name, s["repo"], s["commit"], s["about"], s["license"], s["files"], s.get("scene")
        )
    return out


def get(name: str) -> FixtureSet:
    sets = manifest()
    if name not in sets:
        raise FixtureError(f"no fixture set {name!r}; known: {', '.join(sorted(sets))}")
    return sets[name]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    name: str,
    root: Path | None = None,
    base_url: str = RAW,
    log=print,
    timeout: float = 60.0,
) -> FixtureSet:
    """Download a set (skipping files already present with the right hash)."""
    s = get(name)
    d = s.dir(root)
    todo = [
        (local, f)
        for local, f in s.files.items()
        if not (s.path(local, root).exists() and _sha(s.path(local, root)) == f["sha256"])
    ]
    if not todo:
        return s
    log(f"fetching {name}: {s.about}")
    log(f"  from {s.repo} at {s.commit[:7]} ({len(todo)} files) into {d}")
    log(f"  license: {s.license}")
    for local, f in todo:
        url = base_url.format(repo=s.repo, commit=s.commit, path=f["path"])
        dest = s.path(local, root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
        except (urllib.error.URLError, OSError) as e:
            tmp.unlink(missing_ok=True)
            raise FixtureError(
                f"could not download {url}: {e}. Check the network, or copy the file from a "
                f"clone of {s.repo} at {s.commit} to {dest}"
            ) from e
        got = _sha(tmp)
        if got != f["sha256"]:
            tmp.unlink()
            raise FixtureError(
                f"{url}: sha256 {got} does not match the pinned {f['sha256']}; not using it"
            )
        tmp.replace(dest)
    return s


def require(name: str, root: Path | None = None) -> FixtureSet:
    """The set if present, else a FixtureError that says how to get it."""
    s = get(name)
    if not s.present(root):
        raise FixtureError(
            f"fixture set {name!r} is not in {s.dir(root)}; run `sim2sim fetch {name}` "
            f"(or set SIM2SIM_DATA to a directory that has it)"
        )
    return s
