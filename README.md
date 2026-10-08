# sim2sim

Checks a locomotion policy's deployment against the training environment it
came from, one boundary at a time, using a golden trace recorded in that
environment:

* B: observation to action (the policy file on recorded inputs)
* A: simulator state to observation (each term rebuilt from raw state)
* C: action to joint target and effort (scale, offset, gains, ordering)

The contract (what the policy expects) is read from the exported files and
written to `contract.yaml`, with the source of every field: `live`, `file`,
`file+table`, `default`, `inferred`, `user`, `unknown` or `conflicting`.

Status: early. Only the mjlab recorder and the B, A, C comparator exist.
Physics (boundary D) and target simulators are not implemented.

## Install

    pip install -e .[dev]                # comparator, tests
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

## What the tests show

`pytest` runs on synthetic traces with a stand-in linear policy, plus two
integration tests that run only when the unitree_rl_mjlab export and a
recorded golden trace are present. They show:

* A harness log built correctly from the contract passes B, A and C, where
  contract numbers printed to 3 decimals are admitted only through the
  `export_rounding` tolerance class; with that class removed, C fails.
* A wrong policy file fails B.
* For each injected defect (history layout, order and reset fill; quaternion
  read as xyzw; gyro in the world frame; joint remap skipped on input or on
  output; joint velocity scale; action scale taken from the rounded yaml;
  action offset dropped; one-step action delay; raw action clipped; gains
  bound by index in the wrong joint order; three defects at once; simulator
  state read one step late), the comparator names the boundary, the term and
  the defect, and nothing else, without using falls.
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

They do not show that any of this holds for other frameworks, other robots,
recurrent policies, or real harness logs.

## License

Apache-2.0
