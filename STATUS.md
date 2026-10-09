# Status

Thursday, October 8, 2026 (ET). Plan: `PLAN-v3.3.md`, steps 1 to 5 of section
9 done, plus the CPU half of step 7 (the E4 analysis, tested on the MuJoCo
stand-in). Everything runs on a CPU. Nothing is public: no remote, no package,
no posts. The upstream drafts in `../upstream/` are held.

## What works

| Command | What it does | Evidence it can reach |
|---|---|---|
| `inspect` | Contract from the exported files (mjlab ONNX metadata, Unitree deploy.yaml with SDK tables), with the source of every field | none (a reading) |
| `verify <trace>` | Boundaries B, A, C against a golden trace or a harness log | L2 on a golden trace, L1 on a harness log or a self trace |
| `verify <trace> --mjcf` | Plus boundary D, the nominal closed loop on the target and, when the source's model is recorded, the model counterfactual with group swapping; the verdict per section 4 | L2 for `PHYSICS` and `POLICY_UNDER_TASK`, L3 for PASS with D at floor |
| `residual` | Boundary D alone, with research parameter fits | detection only |
| `task` | A schedule, held joints and punches in the runner, no reference | L1 findings only |
| `check`, `envelope`, `run`, `infer`, `deviation` | Static and linearized checks, the command response map and behavior probes, the closed loop, layout inference, deploy against training values | L1 |
| `tools/results.py` | `results.json`: false confident attribution rate, abstention, detection per boundary | |
| `tools/e4.py` | E4 analysis of PhysX traces (needs the traces) | |

Exit codes: 0 PASS, 1 CONTRACT, 2 INVALID_INPUT, 3 PHYSICS, 4 POLICY_UNDER_TASK,
5 UNDETERMINED or L1 findings, 6 UNSUPPORTED.

Tests: 174 pass (`pytest`, about 6 minutes on 8 cores; integration tests skip when the third-party files are absent). Every row of the section 4 decision table is a test, and
AT1, AT9, AT11, AT12 and AT15 are integration tests (AT2 and AT4 are covered by unit tests).

## What is shown, and at which level

Source engine for everything below: mjlab 1.2.0 on mujoco_warp 3.5.0
(float32), three golden traces recorded on CPU. No Isaac Lab or PhysX trace
exists yet.

* **Boundaries B, A, C (L2).** The three mjlab golden traces pass against the
  contract read from the export. The synthetic injection corpus is named
  exactly (boundary, term, defect). The 29 development harness logs are named
  as labeled, but they were used to fix the comparator: not evidence on unseen
  harnesses.
* **Boundary D floor (calibrated for one engine).** The traces against their
  own recorded model (startup randomization applied): legs 8.8e-7 to 3.7e-5
  N m clean RMS, waist 6.5e-4 to 1.8e-3, arms 7.6e-5 to 5.1e-4, root 0.51 N
  and 0.11 N m. Same numbers from MuJoCo 3.15 (XML plus patch) and 3.5 (the
  compiled model). Margin for "above": 3x the floor.
* **D against real targets (detection, trace a).** unitree_mujoco G1: legs
  0.21 to 0.23 N m, waist 1.97, arms 0.21, root 273 N, all far above floor.
  Menagerie G1: legs 0.30, waist 2.14, root 595 N. With joint friction in
  both targets the armature search fits only some leg joints (unitree_mujoco
  hip roll and knee about -0.011 to -0.014 against a known -0.015) and
  abstains on the rest: research, never in a verdict.
* **Counterfactual (L2, mjlab sources only).** Trace c (golden schedule with
  300 N pushes), 12 seeds:
  * Menagerie G1: source 0/12 against target 8/12 falls, p = 0.0013,
    `PHYSICS`, exit 3. Mass and inertia restores the outcome, within it the
    centers of mass, and that difference is the source's randomized torso
    center of mass (Menagerie matches mjlab's nominal); the verdict carries
    that caveat.
  * unitree_mujoco G1: 0/12 against 1/12, p = 1, no change: `UNDETERMINED`
    with the D finding, exit 5.
  * Positive controls on the mjlab model: floor friction 0.1 gives `PHYSICS`
    localized to contact parameters; torso +15 kg to mass and inertia, then
    body masses.
* **Acceptance tests.** AT1: altered physics never gives `CONTRACT`. AT9: the
  source's own model with a dead-zone task gives `POLICY_UNDER_TASK`. AT15: a
  stand-in drive source is far above floor on every chain with unchanged
  behavior, and a dead-zone failure gives `POLICY_UNDER_TASK`, never
  `PHYSICS`. AT11 and AT12 inside one engine (chain named, sign and size
  right, confounded change not reduced to one parameter).
* **Metric.** 62 cases with a known cause, 55 confident outputs, 0 false
  confident attributions, abstention 6.5%, detection A 27/27, C 25/25, D 3/3.
  The targets with unknown truth (unitree_mujoco, Menagerie) are reported and
  not scored.
* **Issue 145 (L1).** unitree_rl_lab policy on unitree_mujoco's G1, arms held,
  a tour of small commands and in-place turns, then 600 N punches
  (`runs/step5/issue145.*`): `TASK_FAILURE_OBSERVED / BEHAVIORAL_LIMITATION`,
  exit 5. Dead zone on every small segment on 3/3 seeds, the 0.3 m/s leg
  walks (0.26), every seed falls 0.4 s after the first punch. Caveats: L1,
  a silent contract error is not excluded, no attribution.
* **S17b at 5 ms (resolved).** The complex pair at modulus 1.0001 is a real
  slow mode of the linearized standing system, not numerical: unchanged for
  finite difference steps 1e-4 to 1e-9, centered or forward; present at 2 ms
  with the same rate per second (0.025/s against 0.023/s), where the per-step
  tolerance hides it; in base y, yaw and ankle roll next to the exact
  translation and yaw symmetries; with the policy, a yaw offset holds at 0.0100
  rad for 45 s where the mode predicts 0.028. S17b now counts such a pair as
  drift, like slow real modes; the tolerance is unchanged.
* **E4 analysis (MuJoCo stand-in only).** With unitree_rl_lab's asset gains,
  two step sizes recover the drive (beta -1.000, alpha 0, gamma -1.000), a
  by-motor-type armature difference exactly and the ankle check (-0.0100). With
  mjlab's gains (one kd / kp ratio for every joint) two steps cannot separate
  the terms; the tool says so, and three steps recover them. This shows the
  estimator works on a known drive, not that PhysX is one.

## What needs a GPU

All three items need one Isaac Lab session on the stack `unitree_rl_lab`
at 4960b84 pins (record the Isaac Sim and Isaac Lab versions). One GPU with
16 GB or more (L4, A10G, RTX 4090 class). Analysis afterwards runs on the box.

Before renting (CPU, can be written now but not run): `recorders/isaaclab.py`
modeled on `recorders/mjlab.py` (Appendix C keys; physics-rate rows per
substep: root and joint state, joint position targets, applied torque, step
and substep; a dump of the articulation's armature, friction, damping, limits,
masses, centers of mass, inertias and contact material, and of the startup
randomization; the live contract), and a policy wrapper that drives the play
env from `policy.onnx` (the folder ships no checkpoint).

1. **Isaac small-command check (plan 9, step 6).** Play env, shipped ONNX,
   16 envs: fixed 0.15 m/s forward, then 0.2 rad/s yaw alone, once with the
   asset config's gains and once with the yaml's (4 runs of 20 s). Report
   achieved velocity per run. Settles R7 (which gains were trained) and whether
   the dead zone exists on the source physics. About 0.25 h.
2. **Isaac recorder, E3 on the source stack.** Record the schedule of
   `g1_golden_a` (36 s) and the issue 145 tour
   (`runs/step5/issue145_schedule.yaml`, 3 seeds, with and without punches).
   Check on the box: `sim2sim verify <trace>` passes A to C. Then
   `sim2sim task` results can be compared with the source's: the only path to
   `POLICY_UNDER_TASK` for issue 145. About 2 to 3 h, mostly recorder
   debugging.
3. **E4 on PhysX.** Same schedule and seed: stock at 5 ms (decimation 4),
   ankle armature set 0.01 above stock at 5 ms, stock at 2.5 ms (decimation
   8, same policy step). Add stock at 2 ms (decimation 10): it costs minutes
   and removes the identifiability risk. On the box: `python tools/e4.py
   --contract <live contract> --mjcf <frictionless unitree_mujoco G1>
   <stock 5 ms> <stock 2.5 ms> <stock 2 ms> --ankle <ankle trace>
   --ankle-change 0.01`. Pass: ankles within 20% of -0.01 and the fit
   explained; then calibrate the PhysX floor with `tools/calibrate_floor.py`
   and Isaac traces can reach `PHYSICS`. Until then D on Isaac traces is
   uncalibrated and blocks it. About 0.5 h.

Rough total: 4 to 6 GPU hours with setup (about 1 h of it simulation). If
the recorder fails on Isaac Lab API changes, budget one more session.

## Decide before going public

* **Name (D5).** `sim2sim` is free on PyPI but generic and will not be
  found. Free alternatives at the time of the plan: `simverdict`,
  `sim2verdict`, `gapcheck` (check again before choosing).
* **License (D6).** `LICENSE` and `pyproject.toml` say Apache-2.0; confirm.
  Fixtures from unitree_mujoco (BSD-3) and Menagerie are referenced by path
  and commit, not vendored; keep it that way.
* **Repo location.** Personal account or an organization; where the golden
  trace corpus lives (D4: a Hugging Face dataset or a small repo). The tests
  read third-party files from the fixture directory paths (overridable by environment
  variables) and skip without them; public CI needs a fixture download step.
  `runs/` is not committed.
* **README claims (D8).** Only what passes in CI on the box. Safe today: B, A,
  C on the synthetic corpus and on the mjlab golden traces; D and the
  counterfactual for mjlab sources; the issue 145 finding as L1 in this
  runner. Not safe: anything about Isaac Lab or PhysX, real robots or real
  harnesses; any "catches N of M bugs" from our own injections; the
  development set score as accuracy; a statement that the issue 145 harness
  is correct or wrong.
* **Held drafts.** The `efferent` fixes (D2), the issue 145 reply (D3) and the
  exporter keys (D7) are drafts only and need your go and exact text review.

## Open, CPU only

* The armature search abstains on most joints when the analysis model has
  joint friction the source lacks; a model with friction on both sides is
  needed for those fits. Research only.
* Memory: each mjlab model takes about 450 MB per worker process; two
  counterfactual jobs at 4 workers each at once were killed for memory on the
  box (15 GB). Run one at a time.
