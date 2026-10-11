# Walkthrough: `doctor` on a public G1 policy

From `pip install` to a report, in about three minutes on a laptop CPU. No GPU and no
simulator install: MuJoCo comes with the package. The same steps run in a browser in the
[Colab notebook](https://colab.research.google.com/github/dundysm/gaitkeeper/blob/main/notebooks/quickstart.ipynb).

The policy is Unitree's own: the G1 29 dof velocity policy that ships in
[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), run on the G1 scene from
[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco). Nothing here is
specific to it. With your own files the command is the same.

## 1. Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install "gaitkeeper[sim]"
```

Python 3.10 or newer. `[sim]` adds MuJoCo.

## 2. Get the policy and the scene

```bash
gaitkeeper fetch g1_rl_lab g1_unitree_mujoco
```

This downloads the policy (`policy.onnx`, `deploy.yaml`) and the scene, about 20 MB, from
pinned upstream commits into `~/.cache/gaitkeeper`, and checks each file's sha256.
`gaitkeeper fetch --list` shows what is there.

## 3. Run it

```bash
D=~/.cache/gaitkeeper
gaitkeeper doctor --onnx $D/g1_rl_lab/policy.onnx --config $D/g1_rl_lab/deploy.yaml \
    --mjcf $D/g1_unitree_mujoco/scene_29dof.xml
```

About a minute on two cores. It exits 5: there are findings. A clean report exits 0.

## 4. Read the report

The report has three parts and a summary. The lines below are from the run above.

### Contract

```text
config: deploy.yaml read as unitree_deploy
Contract  unitree_deploy (deploy.yaml, policy.onnx) | level L1 | 19 fields default or unknown (12 unknown, 7 default)

1. Contract
  12 field(s) no file states; results below assume defaults for them:
    control.actuators.integrator
    control.actuators.kind
    ...
    timing.decimation
    timing.sim_dt
  a preset fills them for a known policy: --preset unitree_rl_lab_g1_29dof_velocity@4960b84
```

The contract is everything a harness has to get right to run the policy as it was
trained: the observation layout and scales, the joint order, the gains, the timing, how
the actuator is modelled. `doctor` reads what your files state and lists what they do not.

Here `deploy.yaml` states the observation and the gains, and leaves 12 fields out, among
them the simulation step and the decimation. A harness has to guess those, and so does
this runner: it uses defaults and says so. Every result after this line holds under those
defaults. For a policy gaitkeeper knows, `--preset` fills the fields from the training
repository at a pinned commit.

A long list here is the first thing to fix in a port. A policy that falls at once in a new
harness is more often a wrong contract than wrong physics.

### Command response

```text
2. Command response (does it do what it is told?)
  DEAD ZONE  vx ignored for |cmd| <= 0.20 on the + side (first tracked +0.22); vx ignored for |cmd| <= 0.15 on the - side (first tracked -0.20)
  DEAD ZONE  vy ignored for |cmd| <= 0.25 on the + side (first tracked +0.28); vy ignored for |cmd| <= 0.25 on the - side (first tracked -0.28)
  DEAD ZONE  no in-place turning + up to +0.50 (reference range ends at +0.20); no in-place turning - up to -0.75 (reference range ends at -0.20)
  Finding    BEHAVIORAL_LIMITATION (L1, this runner, these assumptions; not an attribution)
```

`doctor` commands a grid of forward, sideways and turning velocities and measures what the
robot does. A dead zone is a command the policy answers with less than 20% of what was
asked.

This policy stands still when asked to walk forward at 0.20 m/s or less, or sideways at
0.25 m/s or less, and it does not turn in place. Commanded 0.50 m/s forward it walks at
0.48. A harness that steers with small commands, as a waypoint follower does near its
target, will see a robot that does not move, and may score it as a failure of the harness.

`gaitkeeper envelope` prints the whole map, command by command.

### Waypoint tour

```text
3. Waypoint tour (closed loop, the teleop-walking-benchmark's draws)
  arms its own           mean survival  90.0 of 90 s, complete 3/3, waypoint error 9 cm
  arms moved at random   mean survival   6.2 of 90 s, complete 0/3, waypoint error 33 cm
  with punches           mean survival   9.7 of 90 s, complete 0/3, waypoint error 61 cm
```

The tour is 90 s of random waypoints, three seeds, run three ways. Each row adds one thing
to the row above:

* **arms its own**: the policy drives every joint it was trained to drive.
* **arms moved at random**: the harness moves the arms, as a teleoperation harness does.
* **with punches**: pushes on the torso as well.

This policy completes the tour on its own, and lasts about 6 s once something else moves
its arms. That locates the failure: it is the arm motion, which the policy was not trained
under, and not the walking. `--quick` runs only the first row.

### Summary

```text
Summary
  It survives but ignores some commands (dead zones above): a harness that sends small
  commands will see it stand still. That is the policy, if the contract is right.
  Arms moved at random costs it 84 s (90.0 to 6.2 s).
Evidence L1: this runner, this model, these assumptions. For a cause, record a golden
trace in the training simulator and run gaitkeeper verify (README: How it decides).
```

The last two lines matter as much as the rest. `doctor` reports what happens in this
runner under the stated assumptions (evidence level L1). It does not say why. "That is the
policy, if the contract is right" is as far as it goes without a reference.

## 5. Next steps

* **A machine-readable report**: add `--json report.json`.
* **Your own policy**: pass your `--onnx`, the config you trained or deployed with as
  `--config` (an Isaac Lab `env.yaml`, a Unitree `deploy.yaml`, a unitree_rl_gym or
  legged_gym config) and a MuJoCo scene of the robot as `--mjcf`. See
  [Run it on your policy](../README.md#run-it-on-your-policy).
* **A cause, not only a finding**: record a golden trace in the training simulator and run
  `gaitkeeper verify`. It separates a contract error from a physics difference from a
  limit of the policy. See [How it decides](../README.md#how-it-decides).
* **A port that scores badly in a benchmark**: `gaitkeeper bench --upstream <the authors'
  config>` runs the tour at each step from the authors' setup to the full benchmark.

If a step here was unclear or broke, that is a bug:
[open an issue](https://github.com/dundysm/gaitkeeper/issues/new/choose).
