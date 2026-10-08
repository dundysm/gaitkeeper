# sim2sim

Checks a locomotion policy's deployment against the training environment it
came from, one boundary at a time, using a golden trace recorded in that
environment:

* B: observation to action (the policy file on recorded inputs)
* A: simulator state to observation (each term rebuilt from raw state)
* C: action to joint target and effort (scale, offset, gains, ordering)

The contract (what the policy expects) is read from the exported files and
written to `contract.yaml`, with the source of every field: `live`, `file`,
`file+table`, `default`, `preset`, `inferred`, `user`, `unknown` or
`conflicting`.

Without a golden trace, a closed-loop runner drives the policy in a target
MJCF on the CPU, `check` runs static and linearized checks of the contract
against that MJCF, and `envelope` maps how the robot responds to commands.

`infer` recovers the observation layout (history length, layout, order,
init, term boundaries and labels) from a trace, and abstains while any column
is ambiguous.

Status: early. The mjlab recorder, the B, A, C comparator, the Unitree
deploy.yaml reader, the runner, the checks, the envelope with its behavior
probes and `infer` exist. Physics attribution (boundary D) is not
implemented.

## Install

    pip install -e .[dev]                # comparator, tests
    pip install -e .[sim]                # runner, check, envelope (MuJoCo)
    pip install -e .[record-mjlab]       # recorder (mjlab 1.2.0, MuJoCo 3.5.0, CPU is enough)

Python 3.10 or newer.

## Use

    # golden trace in the training env (runs in the mjlab environment)
    python tools/record_mjlab.py --task-path <unitree_rl_mjlab> --onnx <policy.onnx> --out runs/g1_golden

    # contract from the exported files
    sim2sim inspect --onnx <policy.onnx> --yaml <deploy.yaml> --out contract.yaml

    # verify a golden trace or a harness log against a contract
    sim2sim verify runs/g1_golden --onnx <policy.onnx> --yaml <deploy.yaml>
    sim2sim verify harness_log.npz --contract contract.yaml --policy <policy.onnx>

`verify` exits 0 for PASS, 2 for CONTRACT, 3 for UNDETERMINED.

    # Unitree deploy.yaml, read as what the robot runs (SDK tables: G1 29-DoF, H1)
    sim2sim inspect --deploy <deploy.yaml> --onnx <policy.onnx>

    # closed loop in a target MJCF; every result prints the controller assumptions
    sim2sim run --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> \
        --command 0.5,0,0 --kick 3,0.5,0 --push 6,0,150,0,torso_link,0.1

    # checks S16, S17a, S17b, S18, S19, S21 and the command response map
    sim2sim check --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> --scenario 0,0,0.2
    sim2sim envelope --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml>

    # behavior probes: standstill in the dead zone (two backends), after a kick,
    # yaw while walking, trained kicks and force punches over seeds, fragility sweep
    sim2sim envelope --deploy <deploy.yaml> --onnx <policy.onnx> --mjcf <scene.xml> \
        --push-seeds 10 --physics --json envelope.json

    # observation layout from a trace (exit 0 inferred, 3 partial or abstained)
    sim2sim infer runs/g1_golden --json layout.json
    sim2sim infer harness_log.npz --no-raw          # (obs, action) only

    # deploy values against training values, per joint
    sim2sim deviation --deploy <deploy.yaml> --reference runs/g1_golden/contract.live.yaml --trace runs/g1_golden

Runner backends: `native_implicit` (default; PD inside MuJoCo every step),
`explicit_zoh` (torque held over the training `sim_dt`; the MuJoCo step is
changed to divide it and the change is printed), `python_pd` (debugging).
Torque limits come from the MJCF unless `--limits contract`. A field the
contract takes from no source prints `CONTROLLER_ASSUMED`. `--preset` fills
named training facts (for example
`unitree_rl_lab_g1_29dof_velocity@4960b84`) with provenance `preset`.

## What the tests show

`pytest` runs on synthetic traces with a stand-in linear policy, plus two
integration tests that run only when the unitree_rl_mjlab export and a
recorded golden trace are present. They show:

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
* A trace written by sim2sim's own runner is labeled SELF_CONSISTENT and capped
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
  above; trained kicks survived on 10 of 10 seeds, punches on 0 of 10. Fragility: friction 2.0, softer contacts (stand drift) and kp x0.8
  (fwd + yaw) miss the bar; armature 0 and a 5 ms step under python_pd are
  numerical. The mjlab G1 policy has only its by-design 0.05 dead zone, tracks
  yaw while walking at 74% to 99% and survives 2 of 10 punch runs.
* Open: S17b on native_implicit at a 5 ms step reports one complex pair just
  outside the unit circle (modulus about 1.0001, growth time about 44 s) while
  every scenario passes. The tolerance was left as specified.

They do not show that any of this holds for other frameworks, other robots,
real harness logs, or the robot itself. Runner results are labeled with the
controller assumptions they rest on and are evidence L1 at most.

## License

Apache-2.0
