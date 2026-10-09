"""Third-party files the integration tests use, from the fixture directory
(``gaitkeeper fetch all``; ``$GAITKEEPER_DATA`` or ``~/.cache/gaitkeeper``). Each test
skips when its files are absent, so the unit suite runs anywhere.

Recorded golden traces are not downloadable: they come from the recorder and
live under ``runs/`` (``$GAITKEEPER_RUNS``)."""

from pathlib import Path

import pytest

from gaitkeeper.env import env
from gaitkeeper.fixtures import data_dir

DATA = data_dir()
UMJ_G1 = DATA / "g1_unitree_mujoco" / "scene_29dof.xml"  # unitree_mujoco @ 1eb6642
URL_G1 = DATA / "g1_rl_lab"  # unitree_rl_lab @ 4960b84 deploy.yaml and policy.onnx
UMJLAB_G1 = DATA / "g1_rl_mjlab"  # unitree_rl_mjlab @ 1425b15 deploy.yaml and policy.onnx
MENAGERIE_G1 = DATA / "g1_menagerie" / "scene.xml"  # mujoco_menagerie @ 0059d43
MJLAB_ONNX = UMJLAB_G1 / "policy.onnx"  # the export the mjlab golden traces were recorded with


def need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing {missing}; `gaitkeeper fetch all` downloads the fixtures")


RUNS = Path(env("RUNS") or Path(__file__).parents[1] / "runs")
GOLDEN_A, GOLDEN_B, GOLDEN_C = (RUNS / f"g1_golden_{x}" for x in "abc")
