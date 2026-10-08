"""Load a target model, or the model a trace was simulated on (plan sections 7.1, 7.5).

A target is an MJCF (``.xml``) or a compiled model (``.mjb``). A trace
directory written by a recorder holds the model the source simulated: the
compiled model (``model.mjb``) when the installed MuJoCo can read it, else the
exported scene XML with the recorder's patch (``model_patch.json``: solver
options and every numeric field that differs from the XML, randomization
included). Startup randomization (``dr.json``) is applied in both cases.

Names: recorders may prefix names (mjlab writes ``robot/left_knee_joint``).
Lookups accept the bare name when exactly one element ends with ``/name``.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

OPT_FIELDS = (
    "timestep",
    "integrator",
    "iterations",
    "ls_iterations",
    "solver",
    "cone",
    "impratio",
    "noslip_iterations",
    "tolerance",
    "ls_tolerance",
    "disableflags",
    "enableflags",
    "jacobian",
    "o_margin",
)
# Fields never patched: large derived tables the compiler rebuilds from geometry.
_PATCH_SKIP = ("bvh_", "mesh_", "tex_", "cam_", "light_", "flex", "skin", "hfield", "oct_", "mat_")


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class LoadedModel:
    model: mujoco.MjModel
    source: str
    exact: bool  # True when this is the compiled model itself
    notes: list[str] = field(default_factory=list)


def _set_field(m: mujoco.MjModel, name: str, value: Any) -> None:
    dst = getattr(m, name)
    dst[...] = np.asarray(value, dtype=dst.dtype).reshape(dst.shape)


def apply_randomization(m: mujoco.MjModel, dr: dict[str, Any]) -> list[str]:
    """Write recorded startup randomization (first environment) into the model."""
    done = []
    for k, v in dr.items():
        if v is None:
            continue
        arr = np.asarray(v, dtype=float)
        dst = getattr(m, k)
        if arr.size != dst.size:
            arr = arr.reshape((-1,) + dst.shape)[0]
        dst[...] = arr.reshape(dst.shape)
        done.append(k)
    return done


def model_patch(live: mujoco.MjModel, xml: mujoco.MjModel) -> dict[str, Any]:
    """Options and numeric fields of ``live`` that differ from ``xml`` (same MuJoCo)."""
    patch: dict[str, Any] = {"mujoco": mujoco.__version__, "opt": {}, "fields": {}}
    for o in OPT_FIELDS:
        if not hasattr(live.opt, o):
            continue
        x, y = getattr(live.opt, o), getattr(xml.opt, o)
        if np.any(np.asarray(x) != np.asarray(y)):
            patch["opt"][o] = np.asarray(x).tolist()
    for name in dir(live):
        if name.startswith("_") or name.startswith(_PATCH_SKIP):
            continue
        try:
            x, y = getattr(live, name), getattr(xml, name)
        except Exception:  # pragma: no cover - attributes that raise on access
            continue
        if not (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            continue
        if x.shape != y.shape:
            raise ModelUnavailable(f"model and XML differ in structure: {name} {x.shape} {y.shape}")
        if x.dtype.kind == "f" and x.size and not np.array_equal(x, y):
            patch["fields"][name] = x.tolist()
    return patch


def apply_patch(m: mujoco.MjModel, patch: dict[str, Any]) -> None:
    for o, v in patch.get("opt", {}).items():
        cur = getattr(m.opt, o)
        if isinstance(cur, np.ndarray):
            cur[...] = v
        else:
            setattr(m.opt, o, type(cur)(v))
    for k, v in patch.get("fields", {}).items():
        _set_field(m, k, v)


def trace_model(path: str | Path) -> LoadedModel:
    """The model a recorded trace was simulated on."""
    path = Path(path)
    meta = json.loads((path / "manifest.json").read_text())
    want = (meta.get("framework") or {}).get("mujoco")
    dr_path = path / "dr.json"
    dr = json.loads(dr_path.read_text()) if dr_path.exists() else {}
    notes = []
    mjb, xml, patch_path = (
        path / "model.mjb",
        path / "model_xml" / "scene.xml",
        path / "model_patch.json",
    )
    if mjb.exists() and want == mujoco.__version__:
        m = mujoco.MjModel.from_binary_path(str(mjb))
        exact = True
        src = f"{mjb} (MuJoCo {want})"
    elif xml.exists() and patch_path.exists():
        m = mujoco.MjModel.from_xml_path(str(xml))
        patch = json.loads(patch_path.read_text())
        apply_patch(m, patch)
        exact = False
        src = f"{xml} with {patch_path.name}"
        notes.append(
            f"rebuilt from the exported XML and the recorder's patch in MuJoCo {mujoco.__version__}; "
            f"the source ran MuJoCo {want}"
        )
    elif mjb.exists():
        raise ModelUnavailable(
            f"{mjb} needs MuJoCo {want} (installed {mujoco.__version__}) and there is no "
            f"model_patch.json; write one in the source environment with "
            f"`python tools/model_patch.py {path}`"
        )
    else:
        raise ModelUnavailable(f"{path}: no compiled model recorded")
    applied = apply_randomization(m, dr)
    if applied:
        notes.append(f"startup randomization applied: {', '.join(applied)}")
    return LoadedModel(m, src, exact, notes)


_CACHE: dict[tuple[str, float], LoadedModel] = {}


def load_model(src: str | Path, cache: bool = True) -> LoadedModel:
    """A fresh copy of the model at ``src`` (compiled once per process when cached)."""
    p = Path(src)
    key = (str(p.resolve()), p.stat().st_mtime)
    hit = _CACHE.get(key) if cache else None
    if hit is None:
        if p.is_dir():
            hit = trace_model(p)
        elif p.suffix == ".mjb":
            hit = LoadedModel(mujoco.MjModel.from_binary_path(str(p)), str(p), True)
        else:
            hit = LoadedModel(mujoco.MjModel.from_xml_path(str(p)), str(p), True)
        if cache:
            _CACHE.clear()  # one model per process keeps memory bounded
            _CACHE[key] = hit
    return LoadedModel(copy.copy(hit.model), hit.source, hit.exact, list(hit.notes))


def find_id(m: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    """Element id by exact name, else by a unique ``/name`` suffix; -1 when absent."""
    i = mujoco.mj_name2id(m, obj, name)
    if i >= 0:
        return i
    n = {
        mujoco.mjtObj.mjOBJ_JOINT: m.njnt,
        mujoco.mjtObj.mjOBJ_BODY: m.nbody,
        mujoco.mjtObj.mjOBJ_GEOM: m.ngeom,
        mujoco.mjtObj.mjOBJ_ACTUATOR: m.nu,
    }[obj]
    hits = [k for k in range(n) if (mujoco.mj_id2name(m, obj, k) or "").endswith("/" + name)]
    return hits[0] if len(hits) == 1 else -1


def bare(name: str) -> str:
    return name.rsplit("/", 1)[-1]
