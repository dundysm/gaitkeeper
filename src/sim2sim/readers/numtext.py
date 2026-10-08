"""Numbers that remember how they were printed.

Exporters round. A value read from text is only known to half a unit in its
last printed digit, and the comparator's tolerances are built from that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class TextFloat(float):
    """A float carrying the text it was parsed from."""

    text: str

    def __new__(cls, value: float, text: str) -> TextFloat:
        obj = super().__new__(cls, value)
        obj.text = text
        return obj

    @property
    def resolution(self) -> float:
        return resolution_of(self.text)


def resolution_of(text: str) -> float:
    """Half a unit in the last printed digit of a decimal literal."""
    t = text.strip().lower().lstrip("+-")
    exp = 0
    if "e" in t:
        t, e = t.split("e", 1)
        exp = int(e)
    decimals = len(t.split(".", 1)[1]) if "." in t else 0
    return 0.5 * 10.0 ** (-(decimals - exp))


def parse_float(text: str) -> TextFloat:
    return TextFloat(float(text), text.strip())


def parse_csv_floats(text: str) -> list[TextFloat]:
    return [parse_float(t) for t in text.split(",") if t.strip()]


def parse_csv_names(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


class _Loader(yaml.SafeLoader):
    pass


def _construct_float(loader: yaml.SafeLoader, node: yaml.Node) -> TextFloat:
    return TextFloat(yaml.SafeLoader.construct_yaml_float(loader, node), node.value)


def _construct_int(loader: yaml.SafeLoader, node: yaml.Node) -> TextFloat | int:
    return TextFloat(float(yaml.SafeLoader.construct_yaml_int(loader, node)), node.value)


_Loader.add_constructor("tag:yaml.org,2002:float", _construct_float)
_Loader.add_constructor("tag:yaml.org,2002:int", _construct_int)


def load_yaml_with_text(path: str | Path) -> Any:
    """Load YAML; every number becomes a TextFloat with its printed resolution."""
    return yaml.load(Path(path).read_text(), Loader=_Loader)  # noqa: S506 (safe subclass)


def res(x: Any) -> float:
    return x.resolution if isinstance(x, TextFloat) else 0.0
