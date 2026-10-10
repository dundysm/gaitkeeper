# Status

October 10, 2026. gaitkeeper 0.5.1 on PyPI (`pip install "gaitkeeper[sim]"`).

What works, what has been measured, what does not work yet, and what is left. The
commands, the evidence levels and the method are in the [README](README.md); the
benchmark results are in the [G1 Port Audit](https://dundysm.github.io/gaitkeeper/results/).

## What works

* **Reading a contract** from an mjlab ONNX export, a Unitree `deploy.yaml` (G1 29 dof,
  H1), a unitree_rl_gym deploy config, a legged_gym training config, an Isaac Lab
  `env.yaml`, or a teleop-walking-benchmark adapter (`policy.cpp`, compiled for the CPU
  and probed, then checked against the adapter on random inputs).
* **Verifying a trace** at boundaries B, A, C (L2 on a golden trace), and D with the model
  counterfactual when the source model was recorded.
* **Running a policy** in a target MJCF: the command response map with its dead zones,
  static and linearized checks, tasks with held joints and punches, the benchmark tour,
  and `bench`, which runs the tour one change at a time from the authors' config to the
  full benchmark.
* **Recorders** for mjlab velocity envs (CPU) and unitree_rl_lab's G1 velocity task in
  Isaac Lab (GPU, run once).

Tests: without the downloaded fixtures (as CI runs) the suite takes about 15 s; with them,
2 to 5 minutes depending on cores. Each skip names the files it is missing; the golden-trace
tests need traces recorded with `tools/record_mjlab.py`.

## Measured

* **Mapping (L2).** Three mjlab golden traces pass B, A and C against the contract read
  from the export. On a synthetic corpus, every injected defect is named as the boundary,
  the term and the defect, and nothing else.
* **Attribution.** 62 cases with a known cause: 55 confident outputs, 0 false confident
  attributions, abstention 6.5%. 29 of the cases are development logs that were used to
  fix the comparator, so this is not a rate on unseen harnesses.
* **Physics (L2, mjlab sources).** Menagerie's G1 against the mjlab trace with two 300 N
  pushes: `PHYSICS`, localized to the source's randomized torso center of mass (source
  falls 0 of 12, target 8 of 12). unitree_mujoco's G1: `UNDETERMINED`. Two positive
  controls (floor friction, torso mass) are localized correctly.
* **Issue 145 (L1, then Isaac Lab).** The shipped unitree_rl_lab G1 policy stands still
  below about 0.2 m/s and does not turn in place, in MuJoCo and in Isaac Lab alike, and
  falls at every 600 N punch. Its deploy.yaml mapping passes against an Isaac Lab golden
  trace (L2). The score is the policy's under that task, not the harness's.
* **Isaac Lab session (2026-10-09, one A40).** deploy.yaml's gains reproduce the MuJoCo
  behavior; the training config's arm damping at 4960b84 (1.0 against 10) makes 14 of 16
  envs fall, so the policy was trained with gains like deploy.yaml's. E4 recovers an
  ankle armature change on PhysX (detection only).
* **teleop-walking-benchmark (L1).** 16 of 34 ports read, with the observation gaitkeeper
  rebuilds matching the adapter's on random inputs (a check of the reading, not of the
  policy). 16 are benched: 15 of those (sunny has no published weights) and holosoma, which
  reads but whose stand-reset clock does not match. Survival on the full benchmark tracks
  the benchmark's own MuJoCo numbers (Pearson 0.94, Spearman 0.89, n = 16) with the port
  contracts read from the adapters, none written by hand. Three long-surviving ports carry
  much of the Pearson: over the 13 that last under 15 s it is 0.67; against the benchmark's
  PhysX numbers it is 0.81. Per-policy verdicts, with the authors'
  config as a first stage for eight of them, are in the
  [G1 Port Audit](https://dundysm.github.io/gaitkeeper/results/).

## Limitations

* The residual floor is calibrated for one engine (mjlab on mujoco_warp). D on Isaac
  traces stays uncalibrated until there is an MJCF made from the same USD; D on any
  uncalibrated engine blocks `PHYSICS` instead of guessing.
* Nothing here is a statement about a real robot.
* Runner results assume the controller the contract describes; a field taken from no
  source is printed as `CONTROLLER_ASSUMED`.
* The adapter reader refuses what it cannot rebuild exactly: reference-motion and latent
  policies, observations of joints the policy does not drive, and clocks that reset or
  warp. The audit page lists the 17 adapters it refuses and why.
* The two Unitree policy repositories have no LICENSE file at the pinned commits.
  `gaitkeeper fetch` says so and downloads them for local use; they are not
  redistributed. unitree_mujoco and MuJoCo Menagerie are BSD-3-Clause.
* One heavy mjlab job at a time on a 15 GB machine.

## What is left

* **Golden traces.** Publishing the mjlab and Isaac Lab traces (about 126 MB each) as a
  Hugging Face dataset for `gaitkeeper fetch golden`. Waits on a license: they contain
  the outputs of a policy whose repository has no license file (unitree_rl_lab issue 149).
* **Unseen harnesses.** Harness logs that were not used to build the comparator, with
  labels fixed before looking at the output. The protocol and tooling exist
  ([docs/BLIND_TEST.md](docs/BLIND_TEST.md), `tools/blind.py`); no submissions yet.
* **More engines.** A residual floor per engine, from traces recorded in that engine
  against its own model. mjlab is the only one calibrated.
* **Isaac Gym.** The legged_gym readers are checked against transcriptions of the code
  that builds the observation, not yet against a golden trace from Isaac Gym.
* **More readers.** mjlab's training config (josabb).
