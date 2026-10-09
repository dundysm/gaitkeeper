# Changelog

## Unreleased

* The Isaac Lab recorder (`gaitkeeper.recorders.isaaclab`, `tools/record_isaaclab.py`):
  `doctor`, `check` and `record` for unitree_rl_lab's G1 velocity policy. Untested on a
  GPU so far.
* `tools/gpu/setup_runpod.sh` and docs/GPU_SESSION.md for the session that runs it.
* `tools/e4.py --frictionless`, and `tools/calibrate_floor.py --mjcf` for traces from an
  engine with no MuJoCo model.

## 0.1.1

* Documentation only: the README on PyPI matches the repository.

## 0.1.0

First release on PyPI.

* Contract readers for mjlab ONNX exports and Unitree `deploy.yaml` (G1 29 dof, H1), with
  provenance for every field.
* `verify`: boundaries B, A and C against a golden trace or a harness log, a pattern
  classifier for the first failing term, and boundary D with the model counterfactual
  when the source model was recorded.
* The mjlab recorder (CPU), a closed-loop MuJoCo runner with four controller backends,
  `check`, `envelope`, `task`, `infer`, `residual`, `deviation`, and `demo`.
* `fetch` for pinned upstream fixtures, and for golden traces from a Hugging Face dataset.
* `tools/blind.py` and docs/BLIND_TEST.md for scoring harness logs with sealed labels.

Calibrated source engines: mjlab 1.2.0 on mujoco_warp 3.5.0 only. Nothing has been recorded
in Isaac Lab yet. See STATUS.md.
