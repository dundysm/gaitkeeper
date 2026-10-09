# Status

October 8, 2026.

A humanoid locomotion policy that walks in the simulator it was trained in
often fails in a second simulator, or scores nothing in someone else's
harness. Three different things can be wrong, and they ask for different
fixes: the harness does not implement the training contract (observation
layout, action scale, gains, joint order), the two simulators disagree on the
robot (mass, friction, armature, how the drive is integrated), or the policy
itself does not do the task it is being asked to do.

gaitkeeper separates those. It reads a contract from the exported files, with
the source of every field. Against a golden trace recorded in the training
simulator it checks one boundary at a time: the policy file on recorded
inputs (B), each observation term rebuilt from raw state (A), the action
turned into a joint target and an effort (C), then the inverse dynamics
residual of the trace under the target model (D). A `PHYSICS` verdict needs
all of the following: a golden trace that this tool did not write, mapping
that passes, the nominal closed loop failing, a residual above a floor
calibrated for that engine, and a counterfactual in which copying one
parameter group from the source model changes the outcome. Without a golden
trace it still runs the policy in a target MJCF and reports what the robot
does. That is a behavioral finding under stated assumptions, not a cause.

## Evidence levels

| Level | Name | Backed by |
|---|---|---|
| L0 | parsed | A file or a preset. Static checks only. |
| L1 | behaves | This runner only. Behavioral findings under stated controller assumptions. |
| L2 | conformant | Boundaries A to C match an independently recorded source trace on the covered states and channels. |
| L3 | matched | L2, and the residual is at the calibrated floor on the measured channels. |

"Verified" is only used at L2 or above. A trace written by gaitkeeper's own
runner never raises the level. L3 is agreement on what was measured, not a
claim that two models are the same.

Exit codes: 0 PASS, 1 CONTRACT, 2 INVALID_INPUT, 3 PHYSICS, 4
POLICY_UNDER_TASK, 5 UNDETERMINED or an L1 finding, 6 UNSUPPORTED.

## What works today

| Command | What it does | Level it can reach |
|---|---|---|
| `demo` | The unitree_rl_lab issue 145 setup, end to end. Downloads its files on first use. | L1 |
| `fetch` | Policies and MJCF scenes from pinned upstream commits, each file sha256 checked. Nothing third party is vendored. | none |
| `inspect` | Contract from an mjlab ONNX export or a Unitree deploy.yaml (G1 29 dof and H1 SDK tables). | none, a reading |
| `verify <trace>` | Boundaries B, A, C. | L2 on a golden trace, L1 on a harness log or a self trace |
| `verify <trace> --mjcf` | Plus D, the nominal closed loop, and the model counterfactual when the source model was recorded. | L2 for PHYSICS and POLICY_UNDER_TASK, L3 for PASS with D at the floor |
| `residual` | D alone, plus parameter fits that stay out of the verdict. | detection only |
| `task` | A command schedule, held joints and punches, with no reference. | L1 |
| `run`, `check`, `envelope`, `infer`, `deviation` | Closed loop, static and linearized checks, the command response map, observation layout from a trace, deploy values against training values. | L1 |
| `tools/e4.py` | Discretization terms of the training drive, from traces at two or more step sizes. | needs the traces |
| `tools/results.py` | False confident attributions, abstention, detection per boundary. | |

Tests: 182 pass in about 2.5 minutes on 8 cores when the fixtures and the
mjlab golden traces are present. Without them, 129 pass and 53 skip, and
each skip names the missing files. Every row of the verdict table is a test.

The recorder that exists is for mjlab manager based velocity envs, on CPU.
It writes the live contract, every physics step, and the compiled model with
its startup randomization.

## Measured results

The source engine for the numbers below is mjlab 1.2.0 on mujoco_warp 3.5.0
(float32), from three golden traces recorded on CPU. There is no Isaac Lab
or PhysX trace, and no floor for any engine but this one.

**Boundaries B, A, C (L2).** The three mjlab golden traces pass against the
contract read from the export. On a synthetic corpus, each injected defect
is named as the boundary, the term and the defect, and nothing else.

**Development harness logs, with a caveat.** 29 logs from the unitree_rl_lab
G1 deploy files are named as labeled. They were used to fix the comparator,
so this is not evidence on unseen harnesses. `tools/results.py` leaves them
out unless `--devset` points at them. They are not in this repository.

**Metric.** 62 cases with a known cause (the injection corpus, those 29
logs, and the physics and task cases below): 55 confident outputs, 0 false
confident attributions, abstention 6.5%. Detection: A 27 of 27, C 25 of 25,
D 3 of 3. Targets whose truth is unknown (unitree_mujoco and Menagerie
against the mjlab trace) are reported and not scored.

**Residual floor, one engine.** The traces against their own recorded model,
startup randomization applied: legs 8.8e-7 to 3.7e-5 N m clean RMS, waist
6.5e-4 to 1.8e-3, arms 7.6e-5 to 5.1e-4, root 0.51 N and 0.11 N m. The same
numbers come from MuJoCo 3.5 reading the compiled model and from MuJoCo 3.15
reading the exported XML plus the recorded patch. "Above the floor" means
more than 3 times the floor.

**D on real targets (detection only, trace a).** unitree_mujoco G1: legs
about 0.21 to 0.23 N m, waist 1.97, arms 0.21, root 273 N, all far above
the floor. Menagerie G1: legs about 0.30, waist 2.14, root 595 N. Where the
target has joint friction the source lacks, the armature search fits only
some leg joints and abstains on the rest. Those fits are not part of any
verdict.

**Counterfactual (L2, mjlab sources only).** Trace c, the golden schedule
with two 300 N pushes, 12 seeds:

* Menagerie G1: the source falls on 0 of 12, the target on 8 of 12,
  p = 0.0013, verdict `PHYSICS` (exit 3). Mass and inertia restores the
  outcome, and within that group the centers of mass. The difference is the
  source's randomized torso center of mass. Menagerie matches mjlab's
  nominal model, and the verdict says the source is one sample of the
  training distribution, not that nominal model.
* unitree_mujoco G1: 0 of 12 against 1 of 12, p = 1, no change.
  `UNDETERMINED` (exit 5), with the residual finding and no cause.
* Two positive controls on the mjlab model itself: floor friction 0.1 gives
  `PHYSICS` localized to contact parameters, and adding 15 kg to the torso
  gives `PHYSICS` localized to mass and inertia, then to body masses.

**Issue 145, from `gaitkeeper demo` (L1).** The shipped unitree_rl_lab G1
velocity policy on unitree_mujoco's G1 scene, arms held at the default pose,
a tour of small commands and in-place turns, then 600 N punches. In this
runner the policy stands still up to vx 0.20 forward and 0.15 backward, and
up to vy 0.25, and it does not turn in place. Yaw while walking at 0.5 m/s
tracks 67% to 73% of the commanded rate inside the trained limit. On the
tour, 3 of 3 seeds miss the bar: every small segment is a dead zone, the
0.3 m/s leg is walked (about 0.26 m/s), and every seed falls about 0.4 s
after the first punch. The finding is `TASK_FAILURE_OBSERVED /
BEHAVIORAL_LIMITATION`, evidence L1, and `gaitkeeper task` exits 5. The caveats
printed with it: a silent contract error is not excluded, and nothing is
attributed. It does not say the benchmark's harness is right or wrong. That
needs a golden trace from the training simulator under the same commands.

**S17b at a 5 ms step.** The complex pair at modulus 1.0001 is a real slow
mode of the linearized standing system, not a numerical artifact. It is
unchanged for finite difference steps from 1e-4 to 1e-9, centered or
forward. It is present at 2 ms with the same rate per second (0.025/s
against 0.023/s at 5 ms), in base y, yaw and ankle roll, next to the exact
translation and yaw symmetries. With the policy in the loop a yaw offset
holds at 0.0100 rad for 45 s, where the mode would predict 0.028. S17b
reports such a pair as drift, the way it already reported slow real modes.
The per step tolerance is unchanged. On the rerun of the command response
map, S17b passes at 5 ms under the implicit backend and fails under the
explicit Python PD backend, which is numerically unstable there.

**E4, on a MuJoCo stand-in only.** With unitree_rl_lab's asset gains, two
step sizes recover the drive (beta -1.000, alpha 0, gamma -1.000), a
by motor type armature difference exactly, and an injected ankle change of
-0.0100. With mjlab's gains, where every joint has the same kd/kp ratio, two
step sizes cannot separate the terms and the tool says so. Three step sizes
recover them. This shows the estimator works on a known drive. It does not
show that PhysX is one.

## Limitations

* No Isaac Lab or PhysX trace has been recorded, so nothing here is a
  statement about Isaac Lab, PhysX, a real robot, or a real harness.
* The residual floor is calibrated for one engine. D on any other engine
  blocks `PHYSICS` instead of guessing.
* The 0 false confident attributions are on 62 cases whose cause was known
  in advance, including logs that were used to fix the comparator. It is not
  a detection rate on unseen harnesses, and the injection count is not "N of
  M bugs caught".
* Runner results assume the controller the contract describes. A field taken
  from no source is printed as `CONTROLLER_ASSUMED`.
* The two Unitree policy repositories have no LICENSE file at the pinned
  commits (unitree_rl_lab's README shows an Apache-2.0 badge, unitree_rl_mjlab
  states nothing). `gaitkeeper fetch` says so and downloads them for local use. They
  are not redistributed from this repository. unitree_mujoco and MuJoCo
  Menagerie are BSD-3-Clause.
* The golden traces (about 126 MB each) are not published yet.
  `tools/publish_golden.py` puts them in a Hugging Face dataset and pins them
  for `gaitkeeper fetch golden`; it waits on a license choice. Tests that need
  them skip.
* Two counterfactual jobs at 4 workers each were killed on a machine with
  15 GB of RAM. One heavy mjlab job at a time is the working limit.

## What is left

**Isaac Lab recorder, and the GPU session it needs.** Not written. One GPU
with 16 GB or more (L4, A10G, RTX 4090 class), on the stack unitree_rl_lab
at 4960b84, which pins Isaac Sim 5.1.0 and Isaac Lab 2.3.0. Record those
versions in the trace. The policy folder ships `policy.onnx` and no
checkpoint, so the play env has to be driven from ONNX. Analysis after the
session is on CPU. Roughly 4 to 6 GPU hours including setup, about 1 hour of
that simulation, and another session if the Isaac Lab API has moved.

1. Small command check, about 0.25 h. Play env, shipped ONNX, 16 envs: a
   fixed 0.15 m/s forward, then 0.2 rad/s yaw alone, once with the asset
   config's gains and once with the deploy.yaml gains. Four runs of 20 s.
   This is what says which gains were trained, and whether the dead zone
   exists on the source physics.
2. Recorder, about 2 to 3 h, mostly debugging. Same trace schema as the
   mjlab recorder: control rate arrays, one physics row per substep (root
   and joint state, joint position targets, applied torque, step and
   substep), the articulation's armature, friction, damping, limits, masses,
   centers of mass, inertias and contact material, the startup randomization,
   and the live contract. Record the golden schedule (36 s) and the issue
   145 tour (`src/gaitkeeper/data/issue145_tour.yaml`, 3 seeds, with and without
   punches). Then `gaitkeeper verify` on CPU. Agreement there is the only path
   to `POLICY_UNDER_TASK` for issue 145.
3. E4 on PhysX, about 0.5 h. Same schedule and seed: stock at 5 ms
   (decimation 4), ankle armature set 0.01 above stock at 5 ms, stock at
   2.5 ms (decimation 8) and at 2 ms (decimation 10), policy step held at
   0.02 s. On CPU, `python tools/e4.py` against a frictionless unitree_mujoco
   G1. The bar already fixed, before any PhysX data: ankles within 20% of
   -0.01 and the fit explained. Then `tools/calibrate_floor.py` on the Isaac
   traces. Until that floor exists, D on an Isaac trace cannot give
   `PHYSICS`.

**More source engines.** A floor per engine, from traces recorded in that
engine against its own model. mjlab is the only one calibrated.

**legged_gym.** A preset for its explicit PD, and a reader for its configs,
including lifting LSTM state into the policy. Not started. Listed here
because the contract sources were designed to include it, not because it
works.

**Unseen harnesses.** The development logs cannot answer this. It needs
harness logs that were not used to build the comparator, with labels that
were fixed before looking at the output. The protocol and the tooling exist
(docs/BLIND_TEST.md, `tools/blind.py`, a blind test issue template); no
submissions yet.

**Drafts, not posted.** Notes on the `efferent` log format, a reply to
unitree_rl_lab issue 145, and a list of exporter keys. None of them are in
this repository, and none of them have been sent.

**Release.** 0.1.0 is ready for PyPI: `.github/workflows/release.yml`
publishes on a `v*` tag through trusted publishing, after a one-time pending
publisher is added on pypi.org.

**Open decisions.** The license for the published golden traces, which
contain the outputs of a policy whose repository has no license file. An
issue asking Unitree to add one is drafted, not posted.
