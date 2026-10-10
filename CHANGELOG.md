# Changelog

## Unreleased

Fixes from an external audit of 0.5.0.

* Boundary C applies the contract's processed-target clip, as the runner does. A correct
  trace from a deploy that clips its targets (Unitree deploy.yaml, Isaac Lab, legged_gym,
  adapters) no longer fails C with a false CONTRACT verdict.
* legged_gym reader: when several stiffness keys match a joint, the last one applies and
  damping is read with it, as legged_robot.py does (it took the first). Constant
  arithmetic such as `1/4` is read; any other expression is refused with the field named
  (it became None).
* Isaac Lab reader: a group `history_length` overrides each term's, as the
  ObservationManager does (the term's won).
* mjlab export reader: a per-joint action clip in deploy.yaml is read (it raised), and a
  single `[low, high]` clip no longer crashes the runner.
* Adapter builds compile to a temporary file and move into place, so parallel runs never
  load a partly written library.
* Input files are checked before any work: an ONNX file that does not load, an MJCF MuJoCo
  cannot parse, or a binary file given as `--config` exits 2 with the flag named. Every
  command-line mistake now exits 2 (some exited 1, the CONTRACT code).
* `--config` passes `--onnx` to the deploy.yaml and unitree_rl_gym readers, and a new
  `--joint-order deploy.yaml` gives an Isaac Lab env.yaml its joint order.
* `doctor` names the presets that fill its unknown fields and says how long the sweep takes;
  every option has help text; `adapter` reports MATCHES ADAPTER / DIFFERS FROM ADAPTER
  rather than "verified", which the README reserves for L2.
* `fetch` no longer suggests `fetch golden`, which does not exist yet.
* Docs: corrected the arm-walk survival (about 11 s, not 15), the number of refused
  adapters (17), test timings, and run times; the results page computes its correlations
  from the published values and states how much three long-surviving ports carry.

## 0.5.0

* `gaitkeeper doctor --onnx policy.onnx --config <config> --mjcf scene.xml`: one command for
  your own policy. Reports the contract fields no file states, the dead zones of the
  command response, and survival on a waypoint tour with the arms its own, moved at
  random, and punched.
* `--config` on every command that reads a contract: an Isaac Lab env.yaml, a Unitree
  deploy.yaml, a unitree_rl_gym config, a legged_gym config or a contract, by content.

## 0.4.0

* A reader for the env.yaml Isaac Lab writes with a training run (used by `bench
  --upstream`; from 0.5.0 also `--config env.yaml`, and from 0.5.1 `--joint-order
  deploy.yaml`): regex patterns over joint names resolved
  as Isaac Lab does (default pose, actuator groups with gains, armature and limits, action
  scale), observation terms with their scales, clips and history, training noise noted,
  the command start and limit ranges and which axes a curriculum widens. The joint order,
  which the file does not hold, comes from a deploy.yaml, from the bench contract under
  `bench --upstream`, or is assumed (unitree_rl_lab's G1 order, flagged).
* `bench --upstream` runs the upstream stage with the bench contract's drive when the two
  differ (an implicit PhysX drive against an explicit PD), so stages differ only in values,
  and reports when the tour commands more than the policy was trained on.
* A reader for legged_gym training configs (`--legged-gym g1_config.py --legged-gym-base
  legged_robot_config.py [--urdf ...] [--env-py ...]`): the config classes are evaluated
  without importing legged_gym or Isaac Gym (literal values, base classes merged), gains
  matched to joints by substring as legged_robot.py does, the DOF order read depth first
  from the URDF, the observation layout picked by `num_observations` (legged_gym's base or
  unitree_rl_gym's G1/H1 env), observations scaled then clipped, actions clipped, the
  trained command ranges and heading mode recorded.
* Fixed: an observation group's `clip_then_scale: false` was ignored, so a term with both
  a clip and a scale was always clipped first. It now scales first when the group says so
  (legged_gym, and wty-yy's `scale_first`).
* `bench` also compares the policy step, the physics step, the trained command ranges,
  the history length and the observation clips with the upstream config.
* A reader for unitree_rl_gym's MuJoCo deploy configs (`--rl-gym configs/g1.yaml`), the
  template many community G1 policies ship with: gains, default angles, scales and sizes
  from the YAML, the observation and phase clock as `deploy_mujoco.py` builds them
  (checked against a transcription of the script). A fork whose `num_obs` does not fit
  that layout is refused. Joints the deploy scene welds are held by stiff servos at zero.
* `bench --upstream` reads a contract, a Unitree deploy.yaml or a unitree_rl_gym config by
  content, and fills the command limits an upstream config lacks from the bench contract.

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
* `fetch` for pinned upstream fixtures (golden traces are not published yet; see STATUS).
* `tools/blind.py` and docs/BLIND_TEST.md for scoring harness logs with sealed labels.

Calibrated source engines: mjlab 1.2.0 on mujoco_warp 3.5.0 only. Nothing has been recorded
in Isaac Lab yet. See STATUS.md.
