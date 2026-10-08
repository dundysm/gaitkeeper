"""A requested task: a command schedule with optional pushes and joint
ownership, run over seeded starts and judged per segment (plan sections 4, 7.2).

A segment is judged on its second half, after the policy has had time to
respond. Commanded axes need the right sign and at least half the commanded
speed (the L1 bar); a standing segment needs planar drift under 0.05 m/s. A run
also fails when the robot falls or a joint sits at its torque limit for more
than 20% of steps. Every failure is classified so the report can say what kind
of limit it is: ``dead zone`` (under 20% of the command, the envelope's
definition), ``partial`` (20% to 50%), ``wrong sign``, ``drift``, ``fall`` (with
the push that preceded it, if any) and ``torque limit``.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import Contract
from .envelope import DEAD_RATIO, L1_DRIFT, L1_SAT, L1_SPEED_RATIO
from .runner import External, Push, PushGenerator, RunConfig, Runner

AXES = ("vx", "vy", "wz")
MIN_SEGMENT_S = 1.0
PUSH_WINDOW_S = (
    2.0  # settle time after a push: not judged for tracking; a fall within it names the push
)
MIN_WINDOW_S = 0.5


@dataclass
class Segment:
    t0: float
    t1: float
    cmd: tuple[float, float, float]
    judged: tuple[bool, bool, bool] = (True, True, True)  # axes held to the bar
    judge_stand: bool = True

    def label(self) -> str:
        return f"{self.t0:5.1f} to {self.t1:5.1f} s ({', '.join(f'{c:+.2f}' for c in self.cmd)})"


@dataclass
class TaskSpec:
    name: str
    seconds: float
    schedule: list[tuple[float, tuple[float, float, float]]]
    pushes: list[Push] = field(default_factory=list)
    push_generator: PushGenerator | None = None
    hold: list[str] | None = None  # joints the policy does not own, held at the default
    unowned_obs: str = "real"
    segments: list[Segment] | None = None  # None: one per schedule row

    def segment_list(self) -> list[Segment]:
        if self.segments is not None:
            return self.segments
        out = []
        rows = sorted(self.schedule)
        for i, (t, c) in enumerate(rows):
            t1 = rows[i + 1][0] if i + 1 < len(rows) else self.seconds
            if t1 - t >= MIN_SEGMENT_S:
                out.append(Segment(float(t), float(t1), tuple(float(x) for x in c)))
        return out


@dataclass
class SegmentResult:
    segment: Segment
    achieved: tuple[float, float, float] | None
    problems: list[tuple[str, str]]  # (kind, text)


@dataclass
class SeedOutcome:
    seed: int
    fell_at: float | None
    fall_after_push: dict[str, Any] | None
    sat_max: float
    segments: list[SegmentResult]
    problems: list[tuple[str, str]]

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class TaskOutcome:
    spec: TaskSpec
    model: str
    seeds: list[SeedOutcome]
    controller: dict[str, Any]

    @property
    def n_failed(self) -> int:
        return sum(not s.passed for s in self.seeds)

    @property
    def passed(self) -> bool:
        return self.n_failed == 0

    def kinds(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.seeds:
            for k in {k for k, _ in s.problems}:
                out[k] = out.get(k, 0) + 1
        return out

    def segment_table(self) -> list[str]:
        """Per segment: mean achieved over surviving seeds and how many seeds missed the bar."""
        segs = self.spec.segment_list()
        out = []
        for i, sg in enumerate(segs):
            ach = [s.segments[i].achieved for s in self.seeds if s.segments[i].achieved is not None]
            miss = sum(bool(s.segments[i].problems) for s in self.seeds)
            kinds = sorted({k for s in self.seeds for k, _ in s.segments[i].problems})
            a = np.mean(ach, axis=0) if ach else [math.nan] * 3
            out.append(
                f"  {sg.label()}  achieved ({a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f})  "
                + (
                    f"misses the bar on {miss}/{len(self.seeds)} seeds: {', '.join(kinds)}"
                    if miss
                    else "ok"
                )
            )
        return out


def segment_problems(sg: Segment, a: tuple[float, float, float] | None) -> list[tuple[str, str]]:
    """The L1 bar on one segment's achieved velocity (None: not judged on this seed)."""
    if a is None:
        return []
    p: list[tuple[str, str]] = []
    if all(c == 0 for c in sg.cmd):
        drift = math.hypot(a[0], a[1])
        if sg.judge_stand and drift >= L1_DRIFT:
            p.append(("drift", f"{sg.label()}: drifts {drift:.3f} m/s while standing"))
    for i, c in enumerate(sg.cmd):
        if c == 0 or not sg.judged[i]:
            continue
        if a[i] * c <= 0 and abs(a[i]) >= DEAD_RATIO * abs(c):
            p.append(("wrong sign", f"{sg.label()}: {AXES[i]} {a[i]:+.2f} for {c:+.2f}"))
        elif abs(a[i]) < DEAD_RATIO * abs(c) or a[i] * c <= 0:
            p.append(("dead zone", f"{sg.label()}: {AXES[i]} {a[i]:+.2f} for {c:+.2f} (under 20%)"))
        elif abs(a[i]) < L1_SPEED_RATIO * abs(c):
            p.append(("partial", f"{sg.label()}: {AXES[i]} {a[i]:+.2f} for {c:+.2f} (under 50%)"))
    return p


def _window(sg: Segment, pushes: list[dict[str, Any]], policy_dt: float) -> tuple[int, int] | None:
    """Second half of the segment, cut before any push inside it or the settle time after one."""
    t0, t1 = sg.t0 + (sg.t1 - sg.t0) / 2, sg.t1
    for x in pushes:
        if x["kind"] != "force":
            continue
        if t0 - PUSH_WINDOW_S < x["t"] < t1:
            if x["t"] - t0 >= MIN_WINDOW_S:
                t1 = x["t"]
            else:
                t0 = max(t0, x["t"] + PUSH_WINDOW_S)
    if t1 - t0 < MIN_WINDOW_S:
        return None
    return int(round(t0 / policy_dt)), int(round(t1 / policy_dt))


def judge(res: Any, spec: TaskSpec, policy_dt: float, seed: int) -> SeedOutcome:
    vel = res.vel if res.vel is not None else np.zeros((0, 3))
    segs = []
    for sg in spec.segment_list():
        if res.fell_at is not None and res.fell_at < sg.t1 - 1e-9:
            segs.append(SegmentResult(sg, None, []))
            continue
        w = _window(sg, res.pushes, policy_dt)
        a = tuple(float(x) for x in vel[w[0] : w[1]].mean(0)) if w else None
        segs.append(SegmentResult(sg, a, segment_problems(sg, a)))
    return _seed_outcome(res, seed, segs)


def _seed_outcome(res: Any, seed: int, segs: list[SegmentResult]) -> SeedOutcome:
    problems: list[tuple[str, str]] = [p for s in segs for p in s.problems]
    after = None
    if res.fell_at is not None:
        prior = [
            x
            for x in res.pushes
            if x["kind"] == "force" and 0 <= res.fell_at - x["t"] <= PUSH_WINDOW_S
        ]
        after = prior[-1] if prior else None
        what = f"fell at {res.fell_at:.2f} s"
        if after:
            what += (
                f", {res.fell_at - after['t']:.2f} s after a {np.linalg.norm(after['vector']):.0f} N "
                f"push on {after['body']}"
            )
        problems.insert(0, ("fall", what))
    sat = float(res.sat_frac.max())
    if sat > L1_SAT:
        worst = res.names[int(res.sat_frac.argmax())]
        problems.append(("torque limit", f"{worst} at its torque limit {sat:.0%} of steps"))
    return SeedOutcome(seed, res.fell_at, after, sat, segs, problems)


def rejudge(o: TaskOutcome, segments: list[Segment]) -> TaskOutcome:
    """The same runs held to a different segment mask (achieved values are kept)."""
    seeds = []
    for s in o.seeds:
        segs = [
            SegmentResult(sg, r.achieved, segment_problems(sg, r.achieved))
            for sg, r in zip(segments, s.segments)
        ]
        keep = [p for p in s.problems if p[0] in ("fall", "torque limit")]
        seeds.append(
            SeedOutcome(
                s.seed,
                s.fell_at,
                s.fall_after_push,
                s.sat_max,
                segs,
                keep[:1] + [p for r in segs for p in r.problems] + keep[1:],
            )
        )
    spec = TaskSpec(**{**o.spec.__dict__, "segments": segments})
    return TaskOutcome(spec, o.model, seeds, o.controller)


def seed_start(seed: int) -> dict[str, float]:
    """Seeded start offset, as the envelope's scenarios use."""
    return {
        "yaw0": float(np.random.default_rng(seed).uniform(-math.pi, math.pi)),
        "init_noise": 0.02,
    }


# -- parallel runs --------------------------------------------------------------------------------

_W: dict[str, Any] = {}


def _winit(contract: dict[str, Any], model: str, onnx: str) -> None:
    from .policy import OnnxPolicy

    c = Contract.from_dict(contract)
    rec = c.get("policy_io.graph.recurrent", None)
    _W["runner"] = Runner(c, model, OnnxPolicy(onnx, rec or None))


def _wrun(job: tuple[TaskSpec, int, str | None, Any]) -> tuple[SeedOutcome, dict[str, Any]]:
    spec, seed, backend, edit = job
    r: Runner = _W["runner"]
    ext = (
        [External(joints=list(spec.hold), drive="hold", obs=spec.unowned_obs)]
        if spec.hold
        else None
    )
    cfg = RunConfig(
        backend=backend,
        seconds=spec.seconds,
        schedule=spec.schedule,
        pushes=list(spec.pushes),
        push_generator=spec.push_generator,
        external=ext,
        seed=seed,
        model_edit=edit,
        **seed_start(seed),
    )
    res = r.run(cfg)
    return judge(res, spec, r.policy_dt, seed), res.controller


def run_task(
    contract: Contract,
    model: str,
    onnx: str,
    spec: TaskSpec,
    seeds: list[int] | tuple[int, ...],
    backend: str | None = None,
    model_edit: Any = None,
    workers: int | None = None,
) -> TaskOutcome:
    workers = workers or min(4, os.cpu_count() or 1)
    jobs = [(spec, s, backend, model_edit) for s in seeds]
    with ProcessPoolExecutor(
        workers, initializer=_winit, initargs=(contract.to_dict(), str(model), str(onnx))
    ) as ex:
        out = list(ex.map(_wrun, jobs))
    return TaskOutcome(spec, str(model), [o for o, _ in out], out[0][1] if out else {})
