# GPU session: golden traces from Isaac Lab

Everything gaitkeeper says about unitree_rl_lab's G1 policy so far is L1: measured in
MuJoCo, with no trace from the simulator it was trained in. This session records those
traces. It is the only way to get `POLICY_UNDER_TASK` (or `PHYSICS`) for unitree_rl_lab
issue 145, and to calibrate a boundary D floor for PhysX.

The Isaac code (`gaitkeeper.recorders.isaaclab`) has not run on a GPU yet. Expect the
first hour to go to fixing it against the live API; `doctor` is there to find that early.

## Machine

* One NVIDIA GPU with 16 GB or more and RT cores (L4, L40S, A10G, RTX 4090 class). Isaac
  Sim does not support A100 or H100, which have none.
* Ubuntu 22.04 or newer (GLIBC 2.35+), a driver for CUDA 12.8, 60 GB free disk.
* On RunPod: a PyTorch CUDA 12.8 Ubuntu 22.04 template with the volume at `/workspace`.

## 1. Set up (about 30 to 45 min, mostly downloads)

```bash
cd /workspace
git clone https://github.com/dundysm/gaitkeeper.git
bash gaitkeeper/tools/gpu/setup_runpod.sh
source /workspace/gaitkeeper_gpu.env
```

The script installs Isaac Sim 5.1.0, Isaac Lab 2.3.0, unitree_rl_lab at 4960b84, the G1
29 dof USD from the `unitreerobotics/unitree_model` dataset (its revision is written to
`/workspace/unitree_model/REVISION`), and gaitkeeper from the checkout. It ends with:

```bash
python $GK/tools/record_isaaclab.py doctor --usd $USD --onnx $ONNX
```

which builds the task with two envs, steps it, and checks 29 joints, a 480-wide
observation that matches the ONNX input, and implicit actuators only. It prints
`doctor: ok` or what is wrong. Isaac Sim asks to accept its EULA on the first start.

## 2. Small-command check (about 15 min)

```bash
mkdir -p runs
python $GK/tools/record_isaaclab.py check --onnx $ONNX --deploy $DEPLOY --usd $USD \
    --out runs/isaac_check.json
```

Sixteen envs, 20 s per command, once with the asset config's gains and once with the
deploy.yaml gains. The last lines say, per gain set, whether the policy walks at 0.5 m/s,
whether it stands still at 0.15 m/s forward (as it does in MuJoCo), and how fast it turns
for 0.2 rad/s in place (MuJoCo: 0.02). That settles two things: which gains the policy
was trained with, and whether the dead zone is the policy's or MuJoCo's.

Use the gain set that walks for everything below (`--gains asset` is the training
config and the default; add `--gains deploy --deploy $DEPLOY` if only deploy.yaml walks).

## 3. Golden traces (about 10 min)

```bash
for s in a:0 b:1 c:2; do
  python $GK/tools/record_isaaclab.py record --onnx $ONNX --deploy $DEPLOY --usd $USD \
      --seed ${s#*:} --out runs/isaac_golden_${s%%:*}
done
```

36 s each, the rl_lab schedule (every command inside deploy.yaml's limits, small
commands, in-place yaw, 14 s of walking while turning, one reset at 24 s) and two 300 N
pushes. Each prints a state check and the excitation items it lacks; an item missing on
all three seeds means the schedule needs to change, not the seed.

## 4. The issue 145 tour (about 10 min)

The task from issue 145 as `gaitkeeper demo` runs it in MuJoCo: the waypoint tour,
arms held at the default pose, then 600 N punches on the torso every 3 s from 27 s.

```bash
for s in 0 1 2; do
  python $GK/tools/record_isaaclab.py record --onnx $ONNX --deploy $DEPLOY --usd $USD \
      --seed $s --schedule $GK/src/gaitkeeper/data/issue145_tour.yaml --seconds 34 \
      --hold arms --punch-force 600 --punch-first 27 --punch-every 3 --punch-dur 0.1 \
      --out runs/isaac_tour_s$s
  python $GK/tools/record_isaaclab.py record --onnx $ONNX --deploy $DEPLOY --usd $USD \
      --seed $s --schedule $GK/src/gaitkeeper/data/issue145_tour.yaml --seconds 34 \
      --hold arms --no-pushes --out runs/isaac_tour_nopunch_s$s
done
```

## 5. E4 on PhysX (about 15 min)

Same schedule and seed; the policy step stays 0.02 s.

```bash
R="python $GK/tools/record_isaaclab.py record --onnx $ONNX --deploy $DEPLOY --usd $USD --seed 0"
$R --sim-dt 0.005  --out runs/e4_stock_5ms
$R --sim-dt 0.005  --ankle-armature-delta 0.01 --out runs/e4_ankle_5ms
$R --sim-dt 0.0025 --out runs/e4_stock_2p5ms
$R --sim-dt 0.002  --out runs/e4_stock_2ms
```

## 6. Bring the traces back

```bash
tar czf /workspace/isaac_runs.tgz runs/
```

Download `isaac_runs.tgz` (RunPod's file browser, `runpodctl send`, or `scp`), then stop
the pod. Everything below runs on a CPU.

## 7. Analysis (CPU)

```bash
# in a gaitkeeper checkout, with the traces unpacked under runs/
pip install -e ".[sim]" && gaitkeeper fetch all
D=~/.cache/gaitkeeper; ONNX=$D/g1_rl_lab/policy.onnx; DEPLOY=$D/g1_rl_lab/deploy.yaml
SCENE=$D/g1_unitree_mujoco/scene_29dof.xml

# The live contract against deploy.yaml: does the deployed mapping match training?
gaitkeeper verify runs/isaac_golden_a --deploy $DEPLOY --onnx $ONNX

# Boundary D, the closed loop and the counterfactual against unitree_mujoco
gaitkeeper verify runs/isaac_golden_a --contract runs/isaac_golden_a/contract.live.yaml \
    --onnx $ONNX --mjcf $SCENE --seeds 12

# The tour: does the policy fail it in its own simulator too?
gaitkeeper verify runs/isaac_tour_s0 --contract runs/isaac_tour_s0/contract.live.yaml \
    --onnx $ONNX --mjcf $SCENE

# E4 (the bar fixed before any PhysX data: ankles within 20% of -0.01, fit explained)
python tools/e4.py runs/e4_stock_5ms runs/e4_stock_2p5ms runs/e4_stock_2ms \
    --ankle runs/e4_ankle_5ms --ankle-change 0.01 \
    --contract runs/e4_stock_5ms/contract.live.yaml --mjcf $SCENE --frictionless --json runs/e4.json

# The PhysX floor for boundary D (an upper bound: it includes USD vs MJCF differences)
python tools/calibrate_floor.py runs/isaac_golden_a runs/isaac_golden_b runs/isaac_golden_c \
    --mjcf $SCENE --frictionless
```

Then `python tools/publish_golden.py runs/isaac_golden_* --repo <hf-user>/gaitkeeper-golden`
(a dry run) once the license question in unitree_rl_lab issue 149 has an answer.

## What each trace holds

| File | What |
|---|---|
| `golden.npz` | Control rate: `obs`, `action`, `action_applied`, `target`, `command`, `qpos`, `qvel`, `reset`, `episode_step`. Physics rate (`p/`): state before each step, drive targets, Isaac Lab's torque estimate, pushes |
| `manifest.json` | Versions, seed, schedule, pushes, step sizes, state layout, the recording changes, the state check, excitation items, the USD path and sha256 |
| `contract.live.yaml` | The contract read from the live environment: observation terms, action term, gains, armature, limits |
| `isaac_model.json` | What PhysX simulated: per-joint armature, friction, gains and limits; per-body masses, centers of mass, inertias, contact material; solver settings |

`p/effort` is Isaac Lab's estimate for implicit actuators (PhysX does not report the
drive torque), so it is recorded and never used as a reference. Per-term observations
(`obs_terms`) are not recorded: Isaac Lab's term values include the five-frame history.
