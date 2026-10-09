"""Environment variables. Each is read as GAITKEEPER_<NAME>, then as SIM2SIM_<NAME>
(the project's name before it was renamed), so existing setups keep working."""

from __future__ import annotations

import os


def env(name: str, default: str | None = None) -> str | None:
    for prefix in ("GAITKEEPER_", "SIM2SIM_"):
        v = os.environ.get(prefix + name)
        if v:
            return v
    return default
