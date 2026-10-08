"""Traces from this runner, for tests that need a source with known physics.

``external`` turns a runner trace into a golden trace from an independent
source with its compiled model recorded, the way a recorder writes one. Only
tests do this: it stands for a source engine whose physics we chose.
"""

from pathlib import Path

import mujoco
import numpy as np

from sim2sim.contract import Contract
from sim2sim.models import load_model
from sim2sim.policy import OnnxPolicy
from sim2sim.runner import RunConfig, Runner
from sim2sim.trace import Trace

Schedule = list[tuple[float, tuple[float, float, float]]]


def runner_trace(
    contract: Contract,
    model: Path,
    onnx: Path,
    schedule: Schedule,
    seconds: float,
    edit=None,
    backend: str = "native_implicit",
    seed: int = 0,
) -> Trace:
    r = Runner(contract, str(model), OnnxPolicy(onnx))
    cfg = RunConfig(
        backend=backend,
        seconds=seconds,
        schedule=schedule,
        seed=seed,
        record=True,
        physics=True,
        model_edit=edit,
    )
    res = r.run(cfg)
    assert res.survived, f"{backend} run fell at {res.fell_at}"
    return r.to_trace(res, kind="golden")


def concat(*traces: Trace) -> Trace:
    """Episodes one after another; each starts with a reset."""
    out: dict[str, list[np.ndarray]] = {k: [] for k in traces[0].arrays}
    n = 0
    for t in traces:
        for k, v in t.arrays.items():
            out[k].append(v + n if k == "p/step" else v)
        n += t.n_steps
    return Trace({k: np.concatenate(v) for k, v in out.items()}, dict(traces[0].meta), "golden")


def schedule_of(*episodes: tuple[Schedule, float]) -> list[list[float]]:
    rows, t0 = [], 0.0
    for sched, seconds in episodes:
        rows += [[t0 + t, *v] for t, v in sched]
        t0 += seconds
    return rows


def external(trace: Trace, path: Path, model: Path, edit, schedule: list[list[float]]) -> Trace:
    """Save as a golden trace from an independent source, with its compiled model."""
    trace.meta.update(
        written_by_sim2sim_runner=False,
        schedule=schedule,
        framework={"name": "test source", "mujoco": mujoco.__version__},
    )
    trace.save(path)
    m = load_model(model).model
    if edit is not None:
        edit(m)
    mujoco.mj_saveModel(m, str(path / "model.mjb"), None)
    return Trace.load(path)
