<p align="center">
  <img src="https://raw.githubusercontent.com/dundysm/gaitkeeper/main/docs/assets/banner.svg" alt="gaitkeeper: why does a walking policy that works in training fail in a second simulator?" width="900">
</p>

<p align="center">
  <a href="https://github.com/dundysm/gaitkeeper/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dundysm/gaitkeeper/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776ab">
  <img alt="MuJoCo on CPU" src="https://img.shields.io/badge/MuJoCo-CPU-1f6feb">
  <a href="https://github.com/dundysm/gaitkeeper/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-2ea043"></a>
  <a href="https://github.com/dundysm/gaitkeeper/blob/main/STATUS.md"><img alt="Status: early" src="https://img.shields.io/badge/status-early-d29922"></a>
</p>

A humanoid locomotion policy that walks in the simulator it was trained in often
fails in a second one, or scores nothing in someone else's harness. Three
different things can be wrong, and they need different fixes:

* **the contract**: the harness does not build the observation or apply the
  action the way training did (layout, scale, gains, joint order),
* **the physics**: the two simulators disagree on the robot (mass, friction,
  armature, how the drive is integrated),
* **the policy under the task**: it was never able to do what it is being asked.

gaitkeeper tells them apart, and says how much evidence backs the answer.

<p align="center">
  <img src="https://raw.githubusercontent.com/dundysm/gaitkeeper/main/docs/assets/dead-zone.gif" alt="Two G1 robots in MuJoCo. Left: commanded 0.15 m/s forward, it stands still. Right: commanded 0.50 m/s, it walks at 0.48 m/s." width="808">
  <br>
  <sub>The official unitree_rl_lab G1 policy in unitree_mujoco, rendered from gaitkeeper's runner
  (<code>tools/render_readme_media.py</code>). Below about 0.2 m/s it stands still. A waypoint tour is
  mostly small commands, which is one reason it can score 0% with a harness that follows the contract.</sub>
</p>

## How it decides

<p align="center">
  <img src="https://raw.githubusercontent.com/dundysm/gaitkeeper/main/docs/assets/boundaries.svg" alt="Boundaries A to D, from a golden trace recorded in the training simulator: A signals to observation, B observation to action, C action to effort, D effort to next state." width="900">
</p>

It reads the policy's contract from the exported files, with the source of
every field (`live`, `file`, `file+table`, `default`, `preset`, `inferred`,
`user`, `unknown` or `conflicting`). Against a golden trace recorded in the
training simulator it checks one boundary at a time and stops at the first that
disagrees:

| | Boundary | Question | A mismatch means |
|---|---|---|---|
| **B** | observation → action | Does the policy file give the recorded actions on the recorded inputs? | `CONTRACT` (wrong file, normalizer, state) |
| **A** | signals → observation | Is each observation term rebuilt from raw state the way training built it? | `CONTRACT`, with the term and the pattern (scale, permutation, frame, layout, time shift) |
| **C** | action → effort | Same scale, offset, clip, delay, gains by name? | `CONTRACT`; effort limits are reported, never a mismatch |
| **D** | effort → next state | Does the target model's inverse dynamics explain the recorded motion? | evidence for `PHYSICS`, never a cause by itself |

`PHYSICS` needs a golden trace this tool did not write, A to C passing, the
nominal closed loop failing, a residual above a floor calibrated for that
source engine, **and** a counterfactual: copying one parameter group from the
source model into the target changes the outcome. Without a golden trace,
gaitkeeper still runs the policy in a target MJCF and reports what the robot
does, labeled as a behavioral finding under stated assumptions, not a cause.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install "gaitkeeper[sim]"
gaitkeeper demo
```

`demo` runs the [unitree_rl_lab issue 145](https://github.com/unitreerobotics/unitree_rl_lab/issues/145)
setup end to end on the CPU: unitree_rl_lab's G1 velocity policy, as shipped, on
unitree_mujoco's G1 scene, driven through a tour of small commands and in-place
turns with the arms held, then 600 N punches. On first use it downloads the
policy and the scene (about 20 MB, pinned commits, sha256 checked) into
`~/.cache/gaitkeeper`. It takes under a minute and prints the contract it read,
the command response map with its dead zones, the task segment by segment, and
the finding:

```text
DEAD ZONE  vx ignored for |cmd| <= 0.20 on the + side (first tracked +0.22); ...
Task 'issue 145 tour' (3 seeds) on .../scene_29dof.xml: misses the bar on 3/3 seeds
    2.0 to   6.0 s (+0.15, +0.00, +0.00)  achieved (+0.00, -0.00, +0.00)  misses the bar on 3/3 seeds: dead zone
    ...
Finding  TASK_FAILURE_OBSERVED / BEHAVIORAL_LIMITATION  (evidence L1, exit 5)
  caveat: a silent contract error is not excluded
  caveat: no attribution: neither PHYSICS nor CONTRACT can be concluded without a reference
```

That is a finding about the policy in this runner, not a verdict on anyone's
harness. See [Evidence levels](#evidence-levels).

## Evidence levels

| Level | Name | Backed by |
|---|---|---|
| L0 | parsed | A file or preset. Static checks only. |
| L1 | behaves | This runner only: behavioral findings under stated controller assumptions. |
| L2 | conformant | Boundaries A to C match an independently recorded source trace on the covered states and channels. |
| L3 | matched | L2, and the dynamics residual is at the calibrated floor on the measured channels. |

"Verified" appears only at L2 or above. A trace written by gaitkeeper's own
runner never raises the level.

**Exit codes** (`verify`, `task`): `0` PASS (or L1 findings with nothing
failing) · `1` CONTRACT · `2` INVALID_INPUT · `3` PHYSICS · `4`
POLICY_UNDER_TASK · `5` UNDETERMINED or an L1 finding such as
TASK_FAILURE_OBSERVED · `6` UNSUPPORTED. Every command exits 2 with a one line
message when an input is missing or unreadable (`GAITKEEPER_DEBUG=1` shows the
traceback), and warns when the MJCF has no floor. `demo` exits 0 when it ran.

## Commands

| Command | What it does | Level it can reach |
|---|---|---|
| `demo` | The issue 145 setup, end to end | L1 |
| `fetch` | Pinned policies and MJCF scenes, sha256 checked | |
| `inspect` | Contract from an mjlab ONNX export or a Unitree `deploy.yaml` (G1 29 dof, H1) | a reading |
| `verify <trace>` | Boundaries B, A, C | L2 on a golden trace, L1 on a harness log or a self trace |
| `verify <trace> --mjcf` | Plus D, the nominal closed loop and the model counterfactual | L2 for PHYSICS and POLICY_UNDER_TASK, L3 for PASS |
| `residual` | D alone, plus parameter fits that stay out of the verdict | detection only |
| `task` | A command schedule, held joints and punches, with no reference | L1 |
| `tour` | A closed-loop waypoint tour; by default the teleop-walking-benchmark's, with its arm random walk and punches as options | L1 |
| `bench` | The tour at each step from a policy's own setup to the full benchmark (arms held, arms walked, punches), with a port contract run and compared alongside; names the step that costs survival | L1 |
| `run`, `check`, `envelope` | Closed loop, static and linearized checks, the command response map | L1 |
| `infer` | Observation layout from a trace, abstaining when ambiguous | |
| `deviation` | Deploy values against training values, per joint | |

<details>
<summary><b>Usage examples, runner backends and controller assumptions</b></summary>

    # golden trace in the training env (runs in the mjlab environment)
    python tools/record_mjlab.py --task-path <unitree_rl_mjlab> --onnx <policy.onnx> --out runs/g1_golden

    # contract from the exported files
    gaitkeeper inspect --onnx <policy.onnx> --yaml <deploy.yaml> --out contract.yaml

    # verify a golden trace or a harness log against a contract
    gaitkeeper verify runs/g1_golden --onnx <policy.onnx> --yaml <deploy.yaml>
    gaitkeeper verify harness_log.npz --contract contract.yaml --policy <policy.onnx>

    # with a target: boundary D, the nominal closed loop and the model counterfactual
    gaitkeeper verify runs/g1_golden --contract runs/g1_golden/contract.live.yaml \
        --onnx <policy.onnx> --mjcf <scene.xml> --seeds 12
    gaitkeeper residual runs/g1_golden --contract runs/g1_golden/contract.live.yaml --mjcf <scene.xml>

    # a task without a reference (L1 at most): schedule, held joints, punches
    gaitkeeper task --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> \
        --schedule tour.yaml --hold left_elbow_joint,right_elbow_joint \
        --push-every 3 --push-first 20 --push-force 600

    # Unitree deploy.yaml, read as what the robot runs (SDK tables: G1 29-DoF, H1)
    gaitkeeper inspect --deploy <deploy.yaml> --onnx <policy.onnx>

    # closed loop in a target MJCF; every result prints the controller assumptions
    gaitkeeper run --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> \
        --command 0.5,0,0 --kick 3,0.5,0 --push 6,0,150,0,torso_link,0.1

    # checks S16, S17a, S17b, S18, S19, S21 and the command response map
    gaitkeeper check --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> --scenario 0,0,0.2
    gaitkeeper envelope --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml>

    # behavior probes: standstill in the dead zone (two backends), after a kick,
    # yaw while walking, trained kicks and force punches over seeds, fragility sweep
    gaitkeeper envelope --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> \
        --push-seeds 10 --physics --json envelope.json

    # observation layout from a trace (exit 0 inferred, 3 partial or abstained)
    gaitkeeper infer runs/g1_golden --json layout.json
    gaitkeeper infer harness_log.npz --no-raw          # (obs, action) only

    # deploy values against training values, per joint
    gaitkeeper deviation --deploy <deploy.yaml> --reference runs/g1_golden/contract.live.yaml --trace runs/g1_golden

Runner backends: `native_implicit` (default; PD inside MuJoCo every step),
`explicit_zoh` (torque held over the training `sim_dt`; the MuJoCo step is
changed to divide it and the change is printed), `python_pd` (debugging),
`standin_implicit` (a position-implicit drive in damping and Euler, the way
some engines integrate PD; used to test that boundary D does not turn a drive
difference into a cause).
Torque limits come from the MJCF unless `--limits contract`. A field the
contract takes from no source prints `CONTROLLER_ASSUMED`. `--preset` fills
named training facts (for example
`unitree_rl_lab_g1_29dof_velocity@4960b84`) with provenance `preset`. A
controller field from a preset still prints `CONTROLLER_ASSUMED`: the preset
reads the training config at a commit, not the run that produced the policy.

</details>

## Install

```bash
pip install "gaitkeeper[sim]"            # runner, check, envelope, task, demo (MuJoCo)
pip install gaitkeeper                   # contracts and trace comparison only

git clone https://github.com/dundysm/gaitkeeper && cd gaitkeeper
pip install -e ".[sim,dev]"              # from source, with tests and lint
pip install -e ".[record-mjlab]"         # recorder (mjlab 1.2.0, MuJoCo 3.5.0, CPU is enough)
```

Python 3.10 or newer. Reading contracts and comparing traces needs only the
base install (numpy, pyyaml, onnxruntime).

### Third-party files

Nothing third party is vendored. `gaitkeeper fetch` downloads pinned files from
their upstream repositories at fixed commits and checks each sha256
(`src/gaitkeeper/data/fixtures.json`):

| Set | Source | What |
|---|---|---|
| `g1_rl_lab` | unitree_rl_lab @ 4960b84 | G1 29 dof velocity policy: deploy.yaml, policy.onnx |
| `g1_rl_mjlab` | unitree_rl_mjlab @ 1425b15 | G1 velocity policy: deploy.yaml, policy.onnx |
| `g1_unitree_mujoco` | unitree_mujoco @ 1eb6642 | G1 29 dof scene and meshes (BSD-3-Clause) |
| `g1_menagerie` | mujoco_menagerie @ 0059d43 | Unitree G1 scene and meshes (BSD-3-Clause) |

```bash
gaitkeeper fetch          # list the sets and whether they are present
gaitkeeper fetch all      # download every set
```

The two policy repositories have no LICENSE file at those commits
(unitree_rl_lab's README shows an Apache-2.0 badge; unitree_rl_mjlab states
no license); the files are fetched for local use, not redistributed. The directory is
`$GAITKEEPER_DATA` when set, else `~/.cache/gaitkeeper`.

## What has been measured

The source engine for every attribution so far is mjlab 1.2.0 on mujoco_warp
3.5.0. Nothing has been recorded in Isaac Lab or PhysX yet, so no Isaac source
is calibrated. A first Isaac Lab session (2026-10-09) recorded golden traces of the
unitree_rl_lab G1 policy in its training simulator: deploy.yaml's mapping passes against
them, and the policy fails the issue 145 tour there too (no motion for small commands, a
fall at every punch). See
[STATUS.md](https://github.com/dundysm/gaitkeeper/blob/main/STATUS.md) for what works, what was measured and what is left.

<details>
<summary><b>What the tests and the measurements show</b></summary>

`pytest` runs the unit suite on synthetic traces with a stand-in linear
policy anywhere. Integration tests run on the fixtures (`gaitkeeper fetch all`)
and on golden traces recorded with the mjlab recorder under `runs/`
(`$GAITKEEPER_RUNS`); each skips, naming what is missing, when its files are
absent. They show:

* A harness log built correctly from the contract passes B, A and C, where
  contract numbers printed to 3 decimals are admitted only through the
  `export_rounding` tolerance class; with that class removed, C fails.
* A wrong policy file fails B.
* For each injected defect (history layout, order and reset fill; both
  quaternion misreads, (w, x, y, z) read as (x, y, z, w) and the reverse; gyro
  in the world frame; joint remap skipped on input or on output; joint
  velocity scale; a term never filled; action scale taken from the rounded
  yaml; one action scale factor on every joint; action offset dropped; one
  and two step action delays, with a repeated or an empty (zero) delay line;
  raw action clipped; kp scaled, kd scaled, both scaled; gains bound by index
  in the wrong joint order; three defects at once; simulator state read one
  step late), the comparator names the boundary, the term and the defect, and
  nothing else, without using falls.
* Effort limits are reported, never compared. A harness that clips effort
  below the contract's limit passes C with a `LIMIT_DIFFERENCE_ACTIVE`
  finding; the clip level is inferred from the effort plateau only when the
  joint follows the PD law on every other row, so a gain error is not
  explained away as a clip.
* On segments that cannot tell the contract from a known alternative
  (standing at zero command; one constant walking command), it returns
  UNDETERMINED rather than PASS, including when such an invisible defect is
  present.
* When two equally simple explanations fit, it names neither.
* A trace written by gaitkeeper's own runner is labeled SELF_CONSISTENT and capped
  at evidence L1.
* The reader finds the rounded wrist action scale in the shipped deploy.yaml,
  and the recorded mjlab G1 golden trace passes against the contract read from
  the exported files (integration tests).

* The Unitree deploy reader binds gains by SDK index and pose, scale and
  history by policy index, skips H1's empty SDK slot, and reads the mjlab G1
  wrist action scale 0.07 as exact; against the ONNX metadata that is 6.7% low
  and the wrist kd 3.0% high, reported as deviations of the deployment.
* The streaming observation builder used by the runner gives the same vectors
  as the batch builder the comparator uses. Recurrent state is carried and
  zeroed at reset.
* S17a's bound of 4 matches a simulated one-joint PD (3.9 decays, 4.1
  diverges). S17b separates a slow tipping mode from negative, complex and
  fast modes. S16 finds 0 mismatched pairs in the shipped unitree_rl_lab G1
  yaml, 9 when the default pose is read in SDK order, and 3 kp and 3 kd when
  the gains are read in policy order.
* On the unitree_mujoco G1 scene (integration tests, Appendix A bounds): the
  unitree_rl_lab policy walks 4.69 m in 10 s at 0.5 m/s under python_pd;
  ignores vx 0.20 and walks at 0.22, ignores vy 0.25 and walks at 0.28, under
  both python_pd and native_implicit; does not turn in place at wz 0.2; falls
  in 1 to 2 s under zero action; falls under 1 s with armature 0 under
  python_pd and walks under native_implicit. The wrist roll margin is 3.88 at
  2 ms; native_implicit has only the tipping mode outside the unit circle and
  python_pd with armature 0 has 11, the most negative near -49. The mjlab
  policy walks 4.82 m.

* Behavior probes (unit tests on hand-built inputs; integration tests on the
  unitree_mujoco G1 scene with the unitree_rl_lab policy): stillness needs all
  three measures (action std, joint speed, no contact change); the fall time
  comparison is suppressed when S17a warns or a joint sits at its torque
  limit; a fragility row names numerical instability only when S17a or S17b
  fires beyond the nominal model. The dead zone is a true standstill under
  both backends; after a (0.5, 0.5) m/s kick the robot returns to standstill
  at the vx and wz edges but starts walking sideways at the vy edge (+-0.25)
  and does not stop; over 3 seeds it survives the trained kicks and falls to
  300 to 600 N punches; with armature 0 under python_pd it falls in under 1 s
  (no-policy baseline 0.62 s on the same model and backend), S17a warns, S17b
  fails with a tipping time constant between 0.2 and 0.6 s, and the fall time
  comparison is suppressed; under native_implicit it walks.
* Boundary D on the three mjlab golden traces against their own recorded
  model: at the calibrated floor (legs 1e-6 to 4e-5 N m clean RMS, arms
  8e-5 to 5e-4, waist up to 1.8e-3, root 0.51 N and 0.11 N m; float32 source,
  float64 analysis). Legs are judged on swing steps only. An injected ankle
  armature change (+0.01, -0.002) is named on both legs, and the research fit
  recovers it with the right sign and size; armature, damping and effort
  limit changed together on one leg are named as that chain, never as one
  parameter (AT11); with joint friction on both sides an armature search
  recovers a wrist change; the stand-in drive shows armature -h^2 kp and
  damping -h kp exactly (Appendix B).
* The counterfactual on the mjlab golden trace with the target changed: floor
  friction 0.1 gives `PHYSICS` with contact parameters as the group whose swap
  restores the outcome; torso +15 kg gives mass and inertia, then body masses.
  The source's own model with a dead-zone task gives `POLICY_UNDER_TASK`
  (AT9). Armature 0 and floor friction 2.0 never give `CONTRACT` (AT1). A
  source with the stand-in drive is far above floor on every chain, its
  behavior unchanged, and a dead-zone task failure gives `POLICY_UNDER_TASK`,
  never `PHYSICS` (AT15).
* Every row of the plan's decision table is a test, with the invariants
  (no `PHYSICS` without a nominal failure, D above a calibrated floor and a
  counterfactual change; no `PHYSICS` or `CONTRACT` without a reference).
* The issue 145 setup (unitree_rl_lab policy on unitree_mujoco's G1, arms
  held, small commands, in-place turns, 600 N punches) gives
  `TASK_FAILURE_OBSERVED / BEHAVIORAL_LIMITATION` at L1, exit 5, with the
  caveats that a silent contract error is not excluded and nothing is
  attributed.
* `infer` recovers history length, layout, order and init on all four layouts
  from `(obs, action)` alone; labels every term with its scale and joint order
  from raw state (including a world-frame gyro); abstains with no boundaries
  when a column is constant or copies more than one column (a constant
  command, AT4), and claims no history length when noise removes the exact
  copies. On a 30 s closed-loop rl_lab trace with a command schedule it
  returns history 5, term major, oldest first, repeat-first init, gyro scale
  0.2, joint velocity scale 0.05 and the Isaac joint order.

Measured outside the test suite (scripts and outputs under `runs/`, not
committed):

* The 29 development harness logs (G1, unitree_rl_lab policy, unitree_mujoco
  scene, regenerated with `qpos` and `qvel`): `verify` names 29 of 29 as
  labeled (16 observation, 12 gain or action, 1 clean that returns
  UNDETERMINED because its command never changes). E0's baseline named 11 of
  16 and 0 of 12. This is a development set: the comparator was fixed on it
  (effort clip levels, kp and kd named apart, empty delay lines, the second
  quaternion misread, zero scale as an unfilled term), so it is not evidence
  of accuracy on unseen harnesses. On the falling logs the harness clips at
  the MJCF limits (hip roll 88, waist roll and pitch 50) where training uses
  139 and 25, reported as `LIMIT_DIFFERENCE_ACTIVE`.
* `infer` abstains on all 29 development logs (constant command), returns the
  full layout on the three recorded mjlab golden traces (no history, the phase
  pair labeled from observations, SDK joint order), keeps the structure but
  loses most labels under per-frame noise stored in history, and abstains
  under independent per-entry noise.
* Envelope, unitree_rl_lab G1 policy on unitree_mujoco (`runs/step3`): dead
  zone vx 0.20 (+) and 0.15 (-), vy 0.25, no in-place turning; yaw while
  walking tracks 67% to 73% inside the limit range; the vy kick hysteresis
  above; trained kicks survived on 10 of 10 seeds, punches on 0 of 10. The
  lateral and in-place yaw scenarios are NONE: no tracked command sits 1.25
  times past the dead zone edge inside the limit range (vy edge 0.25, limit
  0.30). Fragility: friction 2.0 (stand drift, fwd + yaw), softer contacts
  (stand drift) and kp x0.8 (fwd + yaw) miss the bar; armature 0 and a 5 ms
  step under python_pd are numerical (S17a and S17b fire); S17b passes at 5 ms
  under native_implicit. The mjlab G1 policy has only its by-design 0.05 dead
  zone, tracks yaw while walking at 74% to 99% and survives 2 of 10 punch
  runs; its rerun under the new rules is unchanged.
* S17b at a 5 ms step (unitree_rl_lab policy, unitree_mujoco G1): the complex
  pair at modulus 1.0001 is a real slow mode of the linearized standing
  system, not numerical. It is the same for finite difference steps 1e-4 to
  1e-9, centered or forward; it is present at 2 ms with the same rate per
  second (0.025/s against 0.023/s, turning 0.025 rad/s), where the per-step
  tolerance hides it; it lives in base y, yaw and ankle roll, next to the
  exact translation and yaw symmetries, and comes and goes with the settle
  time. With the policy in the loop a yaw offset stays at 0.0100 rad for 45 s
  (the mode would predict 0.028). S17b now reports such a pair as drift, as it
  already did slow real modes; the per-step tolerance is unchanged.
* `results.json` (`tools/results.py`): 62 cases with a known cause (the
  injection corpus, the 29 development logs, the physics and task cases
  above), 55 confident outputs, 0 false confident attributions, abstention
  6.5%; A detected 27 of 27, C 25 of 25, D 3 of 3. The development logs were
  used to fix the comparator, so this is not a measure on unseen harnesses.

They do not show that any of this holds for other frameworks, other robots,
real harness logs, or the robot itself. Runner results are labeled with the
controller assumptions they rest on and are evidence L1 at most.

</details>

## Help test it

Every number above comes from cases built alongside the comparator. If you have a harness
of your own, a few of its logs, with labels you seal before sending, are the most useful
contribution: [docs/BLIND_TEST.md](https://github.com/dundysm/gaitkeeper/blob/main/docs/BLIND_TEST.md) describes the format and the
protocol, and `tools/blind.py` scores only against the committed hashes. Start with a
[blind test issue](https://github.com/dundysm/gaitkeeper/issues/new?template=blind-test.md).
Results are published whatever they are.

## License

Apache-2.0
