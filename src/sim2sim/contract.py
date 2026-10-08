"""The typed contract: what a policy expects from the simulator around it.

A contract has three typed sections (``policy_io`` for boundaries A and B,
``control`` for boundary C, ``model`` for boundary D) plus ``timing`` and
``source``. Every non-obvious field carries provenance: where the value came
from, and for numbers read from text, the resolution it was printed with.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA = "sim2sim/contract/v3"

SOURCES = (
    "live",  # read from the running training env
    "file",  # read from an exported file
    "file+table",  # file plus a versioned table (e.g. SDK joint order)
    "default",  # a framework default at a named version
    "preset",
    "inferred",  # fitted from a trace, with its residual
    "user",
    "unknown",
    "conflicting",  # two sources disagree beyond rounding
)

SECTIONS = ("schema", "source", "timing", "policy_io", "control", "model", "evidence")

_MISSING = object()


@dataclass
class Provenance:
    source: str
    detail: str = ""
    # Half a unit in the last printed digit, for numbers read from text.
    # None means exact (or not a number).
    resolution: float | None = None
    # For "conflicting": the value each source gave.
    alternatives: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown provenance source {self.source!r}")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"from": self.source}
        if self.detail:
            d["detail"] = self.detail
        if self.resolution is not None:
            d["resolution"] = float(self.resolution)
        if self.alternatives:
            d["alternatives"] = _plain(self.alternatives)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Provenance:
        return cls(
            source=d["from"],
            detail=d.get("detail", ""),
            resolution=d.get("resolution"),
            alternatives=d.get("alternatives"),
        )


def _plain(x: Any) -> Any:
    """Convert numpy scalars and arrays to plain Python for YAML."""
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, float):
        return float(x)
    return x


class Contract:
    """Nested mapping with dotted-path access and per-path provenance."""

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        provenance: dict[str, Provenance] | None = None,
    ) -> None:
        self.data: dict[str, Any] = data if data is not None else {"schema": SCHEMA}
        self.provenance: dict[str, Provenance] = provenance or {}

    # -- access -------------------------------------------------------------
    def get(self, path: str, default: Any = _MISSING) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise KeyError(path)
                return default
        return node

    def has(self, path: str) -> bool:
        return self.get(path, None) is not None

    def set(
        self,
        path: str,
        value: Any,
        source: str | None = None,
        detail: str = "",
        resolution: float | None = None,
        alternatives: dict[str, Any] | None = None,
    ) -> None:
        parts = path.split(".")
        if parts[0] not in SECTIONS:
            raise KeyError(f"{path}: top level must be one of {SECTIONS}")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _plain(value)
        if source is not None:
            self.provenance[path] = Provenance(source, detail, resolution, alternatives)

    def prov(self, path: str) -> Provenance:
        """Provenance of a path, from the longest recorded prefix."""
        parts = path.split(".")
        for i in range(len(parts), 0, -1):
            p = ".".join(parts[:i])
            if p in self.provenance:
                return self.provenance[p]
        return Provenance("unknown")

    def resolution(self, path: str) -> float:
        r = self.prov(path).resolution
        return 0.0 if r is None else float(r)

    def unknown_fields(self) -> list[str]:
        return sorted(p for p, v in self.provenance.items() if v.source in ("unknown", "default"))

    def copy(self) -> Contract:
        return Contract(copy.deepcopy(self.data), copy.deepcopy(self.provenance))

    # -- io -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out = {k: self.data[k] for k in SECTIONS if k in self.data}
        out["provenance"] = {k: v.to_dict() for k, v in sorted(self.provenance.items())}
        return out

    def dump(self) -> str:
        return yaml.dump(self.to_dict(), Dumper=_Dumper, sort_keys=False, width=120)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.dump())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Contract:
        d = dict(d)
        prov = {k: Provenance.from_dict(v) for k, v in (d.pop("provenance", None) or {}).items()}
        if d.get("schema") != SCHEMA:
            raise ValueError(f"schema {d.get('schema')!r}, expected {SCHEMA!r}")
        unknown = set(d) - set(SECTIONS)
        if unknown:
            raise ValueError(f"unknown contract sections {sorted(unknown)}")
        return cls(d, prov)

    @classmethod
    def load(cls, path: str | Path) -> Contract:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))


class _Dumper(yaml.SafeDumper):
    """Safe dumper that writes short lists of scalars on one line."""


def _repr_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    flow = all(not isinstance(v, (dict, list)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_Dumper.add_representer(list, _repr_list)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- comparing two contracts ------------------------------------------------


@dataclass
class FieldDiff:
    path: str
    kind: str  # "agree", "rounding", "differs", "missing_left", "missing_right"
    left: Any = None
    right: Any = None
    max_abs: float = 0.0
    max_rel: float = 0.0
    detail: str = ""
    keys: list[str] = field(default_factory=list)


def _as_map(v: Any) -> dict[str, Any] | None:
    return v if isinstance(v, dict) else None


def compare_values(
    path: str, left: Any, right: Any, res_left: float, res_right: float
) -> FieldDiff:
    """Compare two field values, allowing for the resolution each was printed with."""
    if left is None and right is None:
        return FieldDiff(path, "agree")
    if left is None:
        return FieldDiff(path, "missing_left", left, right)
    if right is None:
        return FieldDiff(path, "missing_right", left, right)
    lm, rm = _as_map(left), _as_map(right)
    if lm is not None and rm is not None:
        keys = sorted(set(lm) | set(rm))
        worst = FieldDiff(path, "agree", left, right)
        bad: list[str] = []
        for k in keys:
            d = compare_values(f"{path}.{k}", lm.get(k), rm.get(k), res_left, res_right)
            if d.kind != "agree":
                bad.append(k)
            order = ["agree", "rounding", "missing_left", "missing_right", "differs"]
            if order.index(d.kind) > order.index(worst.kind):
                worst.kind = d.kind
            worst.max_abs = max(worst.max_abs, d.max_abs)
            worst.max_rel = max(worst.max_rel, d.max_rel)
        worst.keys = bad
        return worst
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        a, b = float(left), float(right)
        diff = abs(a - b)
        rel = diff / max(abs(a), abs(b), 1e-12)
        if diff == 0.0:
            return FieldDiff(path, "agree", a, b)
        if diff <= res_left + res_right + 1e-12:
            return FieldDiff(path, "rounding", a, b, diff, rel)
        return FieldDiff(path, "differs", a, b, diff, rel)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return FieldDiff(path, "differs", left, right, detail="length")
        worst = FieldDiff(path, "agree", left, right)
        for i, (x, y) in enumerate(zip(left, right)):
            d = compare_values(f"{path}[{i}]", x, y, res_left, res_right)
            if d.kind == "differs":
                worst.kind = "differs"
            elif d.kind == "rounding" and worst.kind == "agree":
                worst.kind = "rounding"
            worst.max_abs = max(worst.max_abs, d.max_abs)
            worst.max_rel = max(worst.max_rel, d.max_rel)
        return worst
    return FieldDiff(path, "agree" if left == right else "differs", left, right)


def compare_contracts(left: Contract, right: Contract, paths: list[str]) -> list[FieldDiff]:
    out = []
    for p in paths:
        lv, rv = left.get(p, None), right.get(p, None)
        out.append(compare_values(p, lv, rv, left.resolution(p), right.resolution(p)))
    return out
