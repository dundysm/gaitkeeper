"""Golden traces and harness logs (plan Appendix C).

A golden trace is a directory with ``golden.npz`` (arrays) and
``manifest.json`` (meta). Control-rate arrays have leading axis T (one row per
policy step); physics-rate arrays live under ``p/`` with leading axis S.

Control-rate alignment: row k holds the observation the policy received at
step k, the raw state it was built from (``qpos``, ``qvel``), the command in
effect when it was built, the policy's output, and the target that output
produced during the physics steps that followed.

A harness log is a single ``.npz`` with the same control-rate keys plus a JSON
``meta`` string. Derived values a harness logs for itself (``gyro``, ``quat``)
are optional and treated as claims, never as references.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TRACE_SCHEMA = "gaitkeeper/trace/v1"

CONTROL_KEYS_REQUIRED = ("obs", "action", "command", "qpos", "qvel", "reset", "episode_step")
CONTROL_KEYS_OPTIONAL = ("action_applied", "target", "effort", "obs_terms")


@dataclass
class Trace:
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]
    kind: str = "golden"  # "golden" or "harness"
    path: Path | None = None

    # -- convenience --------------------------------------------------------
    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.arrays.get(key, default)

    @property
    def n_steps(self) -> int:
        return int(self.arrays["obs"].shape[0])

    @property
    def self_consistent_only(self) -> bool:
        """True when the trace was written by gaitkeeper's own runner."""
        return bool(self.meta.get("written_by_gaitkeeper_runner", False))

    def obs_term(self, name: str) -> np.ndarray | None:
        return self.arrays.get(f"obs_terms/{name}")

    def physics(self, key: str) -> np.ndarray | None:
        return self.arrays.get(f"p/{key}")

    def slice_steps(self, start: int, stop: int) -> Trace:
        """Control-rate rows [start, stop); physics rows that belong to them."""
        arrays = {}
        for k, v in self.arrays.items():
            if k.startswith("p/"):
                continue
            arrays[k] = v[start:stop]
        if "p/step" in self.arrays:
            sel = (self.arrays["p/step"] >= start) & (self.arrays["p/step"] < stop)
            for k, v in self.arrays.items():
                if k.startswith("p/"):
                    arrays[k] = v[sel]
            arrays["p/step"] = arrays["p/step"] - start
        if "reset" in arrays and len(arrays["reset"]):
            arrays["reset"] = arrays["reset"].copy()
        meta = dict(self.meta)
        meta["sliced_from"] = [int(start), int(stop)]
        return Trace(arrays, meta, self.kind)

    def validate(self) -> list[str]:
        problems = []
        for k in CONTROL_KEYS_REQUIRED:
            if k == "episode_step" and self.kind == "harness":
                continue  # derived from resets when a harness does not log it
            if k not in self.arrays:
                problems.append(f"missing key {k}")
        if problems:
            return problems
        t = self.n_steps
        for k, v in self.arrays.items():
            if not k.startswith("p/") and v.shape[0] != t:
                problems.append(f"{k}: {v.shape[0]} rows, expected {t}")
        if "state_layout" not in self.meta:
            problems.append("meta.state_layout missing")
        return problems

    # -- io -------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        meta = {"schema": TRACE_SCHEMA, "kind": self.kind, **self.meta}
        if self.kind == "golden":
            path.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path / "golden.npz", **self.arrays)
            (path / "manifest.json").write_text(json.dumps(_jsonable(meta), indent=1))
        else:
            np.savez_compressed(path, meta=np.array(json.dumps(_jsonable(meta))), **self.arrays)

    @classmethod
    def load(cls, path: str | Path) -> Trace:
        path = Path(path)
        if path.is_dir():
            with np.load(path / "golden.npz") as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads((path / "manifest.json").read_text())
        else:
            with np.load(path) as z:
                arrays = {k: z[k] for k in z.files if k != "meta"}
                meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        if meta.get("schema") != TRACE_SCHEMA:
            raise ValueError(
                f"{path}: trace schema {meta.get('schema')!r}, expected {TRACE_SCHEMA!r}"
            )
        return cls(arrays, meta, meta.get("kind", "golden"), path)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


# -- integrity checks run by the recorder before writing and by verify ------------


@dataclass
class StateCheck:
    """``q[s+1] - q[s]`` against ``h * v[s+1]`` per hinge joint (semi-implicit Euler).

    Shows which velocity the engine reports and whether every physics step
    was recorded: a skipped step doubles the left side.
    """

    h: float
    max_abs: float
    max_rel_to_step: float
    n_pairs: int
    convention: str  # "v_next" if q[s+1]-q[s] matches h*v[s+1], else "v_prev" or "neither"
    worst_joint: int = -1
    details: dict[str, float] = field(default_factory=dict)


def check_state_increments(
    qpos: np.ndarray, qvel: np.ndarray, h: float, contiguous: np.ndarray
) -> StateCheck:
    """``contiguous[s]`` is True when rows s and s+1 are consecutive physics steps."""
    q = np.asarray(qpos, dtype=np.float64)[:, 7:]
    v = np.asarray(qvel, dtype=np.float64)[:, 6:]
    sel = np.asarray(contiguous, dtype=bool)[: len(q) - 1]
    dq = (q[1:] - q[:-1])[sel]
    e_next = np.abs(dq - h * v[1:][sel])
    e_prev = np.abs(dq - h * v[:-1][sel])
    m_next, m_prev = float(e_next.max(initial=0)), float(e_prev.max(initial=0))
    scale = float(np.abs(dq).max(initial=0)) or 1.0
    conv = "v_next" if m_next < 1e-3 * scale else ("v_prev" if m_prev < 1e-3 * scale else "neither")
    worst = int(np.unravel_index(np.argmax(e_next), e_next.shape)[1]) if e_next.size else -1
    return StateCheck(
        h,
        m_next,
        m_next / scale,
        int(sel.sum()),
        conv,
        worst,
        {"max_abs_v_prev": m_prev, "max_step_increment": scale},
    )
