# Contributing to gaitkeeper

Thanks for looking. gaitkeeper checks whether a humanoid locomotion policy is being run the
way it was trained, and it gets better mostly from contact with real policies, configs and
harnesses it has not seen. You do not need to write code to help.

## Ways to help

* **Run it on your policy.** `gaitkeeper doctor --onnx policy.onnx --config <config> --mjcf
  scene.xml`. If a field comes out `unknown`, a reader refuses your config, or a verdict looks
  wrong, open an issue with the command and the output. That is the most useful report there is.
* **Ask for a reader.** If your training stack writes a config gaitkeeper cannot read yet,
  open a [reader request](https://github.com/dundysm/gaitkeeper/issues/new?template=reader_request.yml)
  with a link to an example file.
* **Send labeled harness logs** for a blind test ([docs/BLIND_TEST.md](docs/BLIND_TEST.md)).
  Every accuracy number in the README was measured on cases built alongside the
  comparator; logs from a harness it has never seen are what would make those numbers mean
  something.
* **Pick up an issue** labeled
  [`good first issue`](https://github.com/dundysm/gaitkeeper/labels/good%20first%20issue) or
  [`help wanted`](https://github.com/dundysm/gaitkeeper/labels/help%20wanted). Comment on it
  first so two people do not do the same work.
* **Questions and ideas** go in [Discussions](https://github.com/dundysm/gaitkeeper/discussions).

## Development setup

```bash
git clone https://github.com/dundysm/gaitkeeper && cd gaitkeeper
python -m venv .venv && source .venv/bin/activate
pip install -e ".[sim,dev]"
python -m pytest -q -rs          # about 2 minutes; tests that need downloads skip and say why
ruff check . && ruff format --check .
```

Python 3.10 or newer. Everything runs on the CPU; no GPU is needed for the tests or the
runner. `gaitkeeper fetch` downloads the third-party models and policies some tests use;
without them those tests skip, which is fine for most changes. CI runs the same suite
without fixtures on Python 3.10 and 3.13.

## Where things are

| Path | What it holds |
|---|---|
| `src/gaitkeeper/contract.py` | the contract schema; every field carries where it came from |
| `src/gaitkeeper/readers/` | one module per config format, each returning a contract |
| `src/gaitkeeper/terms.py` | observation terms and how they are built from simulator state |
| `src/gaitkeeper/compare.py`, `verdict.py` | trace comparison at boundaries B, A, C and D |
| `src/gaitkeeper/runner.py`, `tour.py`, `bench.py` | closed-loop runs in MuJoCo, the tour, the bench ladder |
| `src/gaitkeeper/cli.py` | every command |
| `tests/data/` | small fixtures the tests read (configs, a toy adapter, a 12-dof URDF) |
| `tools/` | scripts for recording traces, calibration, the results page |

## Adding a reader

A reader turns a config file into a contract. The existing ones are the best templates:
`readers/rl_gym_deploy.py` is the shortest, `readers/isaaclab_env.py` the most complete.

1. Put a small, real example of the format in `tests/data/<format>/` (trim it, keep the
   license compatible, and say where it came from in a comment or a README next to it).
2. Every value the reader sets records its source (`file`, `default`, `preset`). A value the
   file does not hold is `unknown` or an explicit default, never a silent guess.
3. Refuse formats you cannot read exactly, with a message that says what is missing.
4. Add tests that check the joint order, gains, default pose, action scale and the
   observation layout against the source file.
5. Wire it into `_read_any` in `cli.py` so `--config` detects it by content.

## Pull requests

* Keep a pull request to one change, and say in the description what it changes and how
  you checked it.
* Add or update tests for behavior changes. `pytest` and `ruff` must pass.
* Add a line under `## Unreleased` in `CHANGELOG.md` for anything a user would notice.
* Do not commit downloaded policies, models or traces; `gaitkeeper fetch` pins them instead.
* By submitting a pull request you agree that your contribution is licensed under the
  project's [Apache-2.0 license](LICENSE).

## Writing style

Reports, docs and messages say what was measured and how, plainly. Claims about a policy
name the evidence level they rest on (L0 to L3). Nothing here is a statement about a real
robot, so please do not write findings as if it were.

## Conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
