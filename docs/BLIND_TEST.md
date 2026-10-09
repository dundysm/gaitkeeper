# Blind test: harness logs gaitkeeper has never seen

Every number in the README comes from cases built by the same person who built the
comparator, including the 29 development logs that were used to fix it. That is the
weakest part of the evidence. This protocol is how logs from someone else's harness can
be scored without anyone tuning gaitkeeper to them.

## What to send

A folder with:

| File | What |
|---|---|
| `logs/*.npz` | Harness logs in gaitkeeper's harness format (below). Any number; a few clean ones help. |
| `policy.onnx` | The policy the harness ran |
| `contract.yaml`, `deploy.yaml` or `export.yaml` | What the policy expects: a gaitkeeper contract, a Unitree `deploy.yaml`, or the yaml exported next to an mjlab ONNX |
| `submission.json` (optional) | `{"robot": "unitree_g1_29dof", "presets": [...]}` when the contract needs them |

Keep `labels.json` to yourself until step 4. It says, for each log, what you know is wrong
with it:

```json
{
  "salt": "any random text, 16 characters or more",
  "cases": {
    "run_03": {"kind": "none", "boundary": null},
    "run_07": {"kind": "contract", "boundary": "A", "tokens": ["base_ang_vel"]},
    "run_11": {"kind": "contract", "boundary": "C", "tokens": ["left_knee_joint", "kp"]}
  }
}
```

`boundary` is `A` (observation built wrong), `B` (wrong policy file or state), `C` (action to
target or effort wrong) or `A+C`. `tokens` are words the named cause must contain: an
observation term id from the contract (`base_ang_vel`, `projected_gravity`, `joint_vel_rel`,
...), a joint name, or `kp` / `kd`. Use names you know from your own harness, not from
gaitkeeper's output.

## Harness log format

One row per policy step, written with `gaitkeeper.trace.Trace` so the schema id and `meta`
are stored with the arrays:

| Key | Required | Content |
|---|---|---|
| `qpos`, `qvel` | yes | MuJoCo's own state at the instant each observation was built (not your harness's derived values) |
| `obs`, `action`, `command` | yes | What the harness fed the policy and what it returned |
| `reset` | yes | True on the first step after each reset |
| `target` | for boundary C | The joint position targets the harness sent, in the contract's joint order |
| `effort` | for boundary C on torque harnesses | The torques the harness applied |
| `meta["state_layout"]` | yes | Hinge joint names in `qpos` order, and `free_joint`, `quat_order`, `ang_vel_frame`, `lin_vel_frame` |

```python
import numpy as np
from gaitkeeper.trace import Trace

arrays = {"obs": obs, "action": action, "command": command, "qpos": qpos, "qvel": qvel,
          "reset": reset, "target": target}          # each [T, ...], one row per policy step
meta = {"state_layout": {"joint_names": hinge_names, "free_joint": True, "quat_order": "wxyz",
                         "ang_vel_frame": "body", "lin_vel_frame": "world"}}
Trace(arrays, meta, kind="harness").save("logs/run_07.npz")
```

`gaitkeeper run ... --record example.npz` writes one from gaitkeeper's own runner to compare
against. Logs must change the command on every axis and turn past 90 degrees of yaw at some
point; otherwise `verify` abstains on what the log cannot separate.

## Steps

1. **Seal.** `python tools/blind.py seal labels.json` prints a sha256. Post it publicly
   (a comment on the blind-test issue) before sending anything.
2. **Send** the folder without `labels.json`.
3. **Run.** The maintainer runs `python tools/blind.py run <folder> --commit <gaitkeeper
   commit>` once, with no changes to gaitkeeper in between, and posts the sha256 of
   `outputs.json` together with the file.
4. **Reveal.** You post `labels.json`.
5. **Score.** `python tools/blind.py score <folder> labels.json --labels-sha256 <1>
   --outputs-sha256 <3>` refuses to score unless both files match their posted hashes, then
   writes `blind_results.json`: false confident attributions, abstentions, detection per
   boundary.

Anyone can repeat step 5 from the posted files.

## What counts

A confident output is `CONTRACT` (on harness logs there is no `PHYSICS` without a golden
trace). It is a **false confident attribution** when the label says `none`, or when the
named cause is missing one of the label's tokens. `UNDETERMINED` is an abstention, never an
error, and the abstention rate is reported next to it.

Results are published as they come, including bad ones. A change made to gaitkeeper because
of a blind set makes that set a development set from then on; its results stay published
under the commit that produced them.
