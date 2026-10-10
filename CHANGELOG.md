# Changelog

## 0.3.0

* `gaitkeeper bench --upstream contract.yaml` (or `--upstream-deploy deploy.yaml`): runs the
  policy as its authors trained or deployed it first, compares the bench contract's values
  with it, and names the step from upstream to port when it costs survival. An upstream
  config that does not state the drive runs with the bench contract's, like for like.
* The Unitree deploy.yaml reader takes `scale_first` (wty-yy/unitree_cpp_deploy's fork:
  scale before clip).
* `gaitkeeper adapter policies/<name>/policy.cpp --mjcf ...`: reads a
  teleop-walking-benchmark adapter without parsing it. The CUDA adapter is compiled for the
  CPU with a shim (kernels run as loops, the TensorRT engine is replaced by a recorder), then
  probed: finite differences around the benchmark's stance give the action map, target
  bounds, every observation element's source and gain, the history and a gait clock's
  period, phase and gate. Writes `<name>.trained.yaml` and `<name>.port.yaml`, and verifies
  the port contract by building the observation both ways on random inputs. 16 of the
  benchmark's 34 adapters read and verify (holosoma reads but its stand-reset clock does not
  verify); the rest are reported with the reason.
  `gaitkeeper bench --adapter` goes from the adapter to the report in one command.
* Observation terms take `params.index` (a permutation or subset of the term's elements,
  as a port that feeds the command as [wz, vx, vy]) and a `constant` term (a fixed slot, as
  a height command). Terms are keyed by `source_name`, so one id can appear twice.
  `gait_phase_legs` takes `clock_offset_steps`.
* `gaitkeeper bench`: the tour at a ladder of stages (own setup, harness holds the arms,
  harness walks them, punches), plus a port contract with arms still and under the full
  stack. Names the stages that cost survival, compares the port's values and its holding
  of unlisted joints with the policy's own, writes JSON and markdown.
* `RunConfig.unlisted_trajectory`: joints in `control.unlisted` can follow a trajectory, so
  a legs-only policy gets the benchmark's arm walk too. `tour --arms-obs contract` keeps the
  observation mode a contract's externals give each arm; a contract's non-arm externals
  stay in place when the tour takes the arms.

## 0.2.0

* `gaitkeeper tour`: a closed-loop waypoint tour (RunConfig.command_source,
  gaitkeeper.tour). By default the tour of rhoyn/teleop-walking-benchmark, with its
  random arm walk (`--arms walk`) and punches (`--punches benchmark`) as options; over 11
  of its ported policies, survival matches the benchmark's (Pearson 0.94).
* `base_lin_vel` observation term; `--set PATH=VALUE` for contract fields no file states.
* `control.unlisted`: hold actuated joints a policy does not list (a legs-only policy's
  arms and waist) at a given pose and gains. `gait_phase_legs`: ClOBOT's two-leg clock.
* The Isaac Lab recorder (`gaitkeeper.recorders.isaaclab`, `tools/record_isaaclab.py`):
  `doctor`, `check`, `contract` and `record` for unitree_rl_lab's G1 velocity policy. Run
  on a GPU once (STATUS.md). Fixes from that run: the app launcher is kept alive, output is
  flushed before Kit ends the process, errors exit non-zero, a reset starts from a zero
  command, observation scales are read as arrays, held joints are kept out of `action`,
  and the golden pushes are 150 N for 0.1 s.
* `tools/gpu/setup_runpod.sh`: apt packages for Vulkan and X, flatdict built without
  isolation, install checks by package name.
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
