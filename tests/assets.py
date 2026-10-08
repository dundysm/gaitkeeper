"""Locations of third-party files the integration tests use. Each test skips
when its files are absent, so the unit suite runs anywhere."""

import os
from pathlib import Path

import pytest

DATA = Path(os.environ.get("SIM2SIM_REVIEW_DATA", "data"))
UMJ_G1 = DATA / "umj_g1" / "scene_29dof.xml"  # unitree_mujoco @ 1eb6642
URL_G1 = DATA / "url_g1"  # unitree_rl_lab @ 4960b84 deploy.yaml and policy.onnx
UMJLAB_G1 = DATA / "umjlab_g1"  # unitree_rl_mjlab @ 1425b15 deploy.yaml and policy.onnx
MENAGERIE_G1 = Path(
    os.environ.get(
        "SIM2SIM_MENAGERIE_G1", "mujoco_menagerie/unitree_g1/scene.xml"
    )
)


def need(*paths: Path) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"missing {missing}")


RUNS = Path(os.environ.get("SIM2SIM_RUNS", Path(__file__).parents[1] / "runs"))
GOLDEN_A, GOLDEN_B, GOLDEN_C = (RUNS / f"g1_golden_{x}" for x in "abc")
MJLAB_ONNX = (
    Path(
        os.environ.get(
            "SIM2SIM_MJLAB_EXPORT",
            "unitree_rl_mjlab/deploy/robots/g1/config/policy/velocity/v0",
        )
    )
    / "exported"
    / "policy.onnx"
)
