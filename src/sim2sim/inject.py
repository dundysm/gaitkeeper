"""Harness logs derived from a golden trace, with known defects injected.

Each defect rebuilds what a buggy harness would have logged from the trace's
raw state, so the comparator is tested on the defect alone. Observations are
fed through the policy again so boundary B stays consistent; targets and
efforts follow the harness's own (possibly wrong) action path. Nothing here
uses falls or closed-loop behavior.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .compare import Report, raw_state
from .contract import Contract
from .tables import TABLES
from .terms import (
    RawState,
    TermContext,
    apply_clip_scale,
    assemble,
    context_from_contract,
    term_slices,
    term_values,
)
from .trace import Trace

ISAAC = "unitree_g1_29dof_isaac_bfs@unitree_rl_lab:4960b84"


@dataclass
class Defect:
    name: str
    expect_boundary: str  # boundary the comparator should name, or "none" for controls
    expect_term: str | None
    expect_pattern: str | None  # substring the named pattern must contain
    build: Callable[[Harness], Trace]
    expect_status: str = "fail"  # fail | pass | not_covered
    contract_variant: str = "files"  # "files" or "history5"


class Harness:
    """Builds harness-style logs from a golden trace and a reference contract.

    ``truth`` holds what the training env really did (the live contract); the
    files contract is what the comparator checks against.
    """

    def __init__(
        self, golden: Trace, truth: Contract, policy: Any, files: Contract | None = None
    ) -> None:
        self.g = golden
        self.files = files
        self.truth = truth
        self.policy = policy
        self.state = raw_state(golden)
        self.ctx = context_from_contract(truth)
        self.names = list(truth.get("policy_io.joints.names"))
        g = truth.get("policy_io.observation_groups.policy")
        self.terms, self.history = g["terms"], g["history"]
        self.scale = np.array([truth.get("control.actions.joint_pos.scale")[n] for n in self.names])
        self.offset = np.array(
            [truth.get("control.actions.joint_pos.offset")[n] for n in self.names]
        )
        self.kp = np.array([truth.get("control.actuators.kp")[n] for n in self.names])
        self.kd = np.array([truth.get("control.actuators.kd")[n] for n in self.names])
        lay = golden.meta["state_layout"]["joint_names"]
        self.qi = [lay.index(n) for n in self.names]

    # -- pieces ---------------------------------------------------------------
    def obs(
        self,
        state: RawState | None = None,
        ctx: TermContext | None = None,
        terms: list[dict[str, Any]] | None = None,
        history: dict[str, Any] | None = None,
        edit: Callable[[dict[str, np.ndarray]], None] | None = None,
    ) -> np.ndarray:
        s = state or self.state
        tt = terms or self.terms
        vals = term_values(s, tt, ctx or self.ctx)
        if edit:
            edit(vals)
        o, _ = assemble(vals, tt, s.reset, history or self.history)
        return o.astype(np.float32)

    def closed(self, obs_fn: Callable[[np.ndarray], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Obs and actions consistent with each other, as a harness running the policy would log them.

        ``obs_fn(prev_actions)`` builds the observation for all rows given the
        previous-action channel. Only the last-action columns depend on it, so
        rows are filled in order: obs[k] uses action[k-1], action[k] = policy(obs[k]).
        """
        act = self.g["action"].astype(np.float64)
        if self.policy is None:
            prev = np.zeros_like(act)
            prev[1:] = act[:-1]
            prev[self.state.reset] = 0.0
            return obs_fn(prev), act
        o = obs_fn(np.zeros_like(act))
        cols = term_slices(self.terms, self.history)["last_action"]
        term = _term(self, "last_action")
        act = np.zeros_like(act)
        for k in range(len(o)):
            prev = np.zeros(act.shape[1]) if (k == 0 or self.state.reset[k]) else act[k - 1]
            o[k, cols] = apply_clip_scale(prev[None], term)[0].astype(np.float32)
            act[k] = self.policy(o[k : k + 1])[0]
        return o, act

    def target(
        self, act: np.ndarray, scale: np.ndarray | None = None, offset: np.ndarray | None = None
    ) -> np.ndarray:
        sc = self.scale if scale is None else scale
        of = self.offset if offset is None else offset
        return act * sc[None] + of[None]

    def effort(
        self, tgt: np.ndarray, kp: np.ndarray | None = None, kd: np.ndarray | None = None
    ) -> np.ndarray:
        q = self.g["qpos"].astype(np.float64)[:, 7:][:, self.qi]
        qd = self.g["qvel"].astype(np.float64)[:, 6:][:, self.qi]
        kp = self.kp if kp is None else kp
        kd = self.kd if kd is None else kd
        return kp[None] * (tgt - q) - kd[None] * qd

    def log(
        self, obs: np.ndarray, act: np.ndarray, tgt: np.ndarray | None, eff: np.ndarray | None
    ) -> Trace:
        g = self.g
        arrays = {
            "obs": obs,
            "action": act.astype(np.float32),
            "command": g["command"],
            "qpos": g["qpos"],
            "qvel": g["qvel"],
            "reset": g["reset"],
            "episode_step": g["episode_step"],
        }
        if tgt is not None:
            arrays["target"] = tgt.astype(np.float32)
        if eff is not None:
            arrays["effort"] = eff.astype(np.float32)
        meta = {
            "state_layout": g.meta["state_layout"],
            "target_joint_names": self.names,
            "effort_joint_names": self.names,
            "derived_from": "golden trace",
            "harness": "synthetic",
        }
        return Trace(arrays, meta, "harness")

    def with_prev(self, prev: np.ndarray) -> RawState:
        d = dict(self.state.__dict__)
        d["prev_action"] = prev
        return RawState(**d)

    def standard(
        self, state_fn: Callable[[np.ndarray], RawState] | None = None, **obs_kw: Any
    ) -> Trace:
        sf = state_fn or self.with_prev
        o, a = self.closed(lambda prev: self.obs(state=sf(prev), **obs_kw))
        t = self.target(a)
        return self.log(o, a, t, None)


def history5(contract: Contract) -> Contract:
    """The same contract with a five-step history (term-major, oldest first, repeat-first init).

    The shipped mjlab policy uses no history, so history defects are tested on
    the comparator's A path with this variant; boundary B is not applicable.
    """
    c = contract.copy()
    h = dict(c.get("policy_io.observation_groups.policy.history"))
    h.update({"length": 5, "layout": "term_major", "order": "oldest_first", "init": "repeat_first"})
    c.set(
        "policy_io.observation_groups.policy.history", h, "user", "test variant: five-step history"
    )
    return c


def defects() -> list[Defect]:
    def clean(h: Harness) -> Trace:
        return h.standard()

    def hist(h: Harness, **over: Any) -> Trace:
        hh = dict(h.history)
        hh.update(
            {"length": 5, "layout": "term_major", "order": "oldest_first", "init": "repeat_first"}
        )
        hh.update(over)
        act = h.g["action"].astype(np.float64)
        o = h.obs(history=hh)
        return h.log(o, act, h.target(act), None)

    def xyzw(h: Harness) -> Trace:
        q = h.state.root_quat[:, [1, 2, 3, 0]]

        def edit(v: dict[str, np.ndarray]) -> None:
            r = _state(h.state, root_quat=q)
            v["projected_gravity"] = term_values(r, [_term(h, "projected_gravity")], h.ctx)[
                "projected_gravity"
            ]

        return h.standard(edit=edit)

    def world_gyro(h: Harness) -> Trace:
        return h.standard(ctx=_ctx(h.ctx, imu_frame="world"))

    def obs_remap(h: Harness) -> Trace:
        idx = [h.names.index(n) for n in TABLES[ISAAC]]

        def edit(v: dict[str, np.ndarray]) -> None:
            v["joint_pos_rel"] = (v["joint_pos_rel"] + h.ctx.default_joint_pos[None])[
                :, idx
            ] - h.ctx.default_joint_pos[None]
            v["joint_vel_rel"] = v["joint_vel_rel"][:, idx]

        return h.standard(edit=edit)

    def action_remap(h: Harness) -> Trace:
        tr = h.standard()
        idx = [h.names.index(n) for n in TABLES[ISAAC]]
        t = tr["target"].astype(np.float64)
        # Motor slot j receives the value computed for policy slot j, but the
        # harness's motor order is the Isaac table: joint TABLE[j] gets slot j.
        t2 = np.empty_like(t)
        t2[:, idx] = t
        tr.arrays["target"] = t2.astype(np.float32)
        return tr

    def vel_scale(h: Harness) -> Trace:
        def edit(v: dict[str, np.ndarray]) -> None:
            v["joint_vel_rel"] = v["joint_vel_rel"] * 0.05

        return h.standard(edit=edit)

    def yaml_scale(h: Harness) -> Trace:
        tr = h.standard()
        # The harness takes action scales from deploy.yaml (two decimals).
        src = h.files if h.files is not None else h.truth
        alt = np.array(
            (src.prov("control.actions.joint_pos.scale").alternatives or {})["yaml"],
            dtype=np.float64,
        )
        tr.arrays["target"] = h.target(tr["action"].astype(np.float64), scale=alt).astype(
            np.float32
        )
        return tr

    def no_offset(h: Harness) -> Trace:
        tr = h.standard()
        tr.arrays["target"] = h.target(
            tr["action"].astype(np.float64), offset=np.zeros(len(h.names))
        ).astype(np.float32)
        return tr

    def delay(h: Harness) -> Trace:
        tr = h.standard()
        a = tr["action"].astype(np.float64)
        a1 = np.empty_like(a)
        a1[1:] = a[:-1]
        a1[0] = a[0]
        tr.arrays["target"] = h.target(a1).astype(np.float32)
        return tr

    def clip(h: Harness) -> Trace:
        tr = h.standard()
        a = tr["action"].astype(np.float64)
        q = float(np.quantile(np.abs(a), 0.95))
        c = max(
            x for x in (0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0) if x < q
        )  # a round level, as harnesses use
        tr.arrays["target"] = h.target(np.clip(a, -c, c)).astype(np.float32)
        tr.meta["injected_clip"] = c
        return tr

    def gains_order(h: Harness) -> Trace:
        tr = h.standard()
        order = [h.names.index(n) for n in TABLES[ISAAC]]
        kp, kd = h.kp[order], h.kd[order]  # gains listed in Isaac order, applied by index
        t = tr["target"].astype(np.float64)
        tr.arrays["effort"] = h.effort(t, kp, kd).astype(np.float32)
        del tr.arrays["target"]
        return tr

    def multi(h: Harness) -> Trace:
        tr = xyzw(h)
        a = tr["action"].astype(np.float64)
        t = h.target(a, offset=np.zeros(len(h.names)))
        order = [h.names.index(n) for n in TABLES[ISAAC]]
        tr.arrays["target"] = t.astype(np.float32)
        tr.arrays["effort"] = h.effort(t, h.kp[order], h.kd[order]).astype(np.float32)
        return tr

    def timing(h: Harness) -> Trace:
        # Simulator state read one policy step late (before the latest physics
        # steps); command, phase clock and last action are current.
        st = h.state
        lag = {
            k: _lag(getattr(st, k)) for k in ("root_quat", "ang_vel_body", "joint_pos", "joint_vel")
        }
        return h.standard(state_fn=lambda prev: _state(st, prev_action=prev, **lag))

    def under_stand(h: Harness) -> Trace:
        # Command scaled x2 while the command is zero: invisible on this segment.
        n = _const_rows(h, zero=True)
        sub = Harness(h.g.slice_steps(*n), h.truth, h.policy, h.files)
        return sub.standard(
            edit=lambda v: v.__setitem__("velocity_commands", v["velocity_commands"] * 2.0)
        )

    def under_walk(h: Harness) -> Trace:
        # No stand rule on the phase while a constant walking command is held.
        n = _const_rows(h, zero=False)
        sub = Harness(h.g.slice_steps(*n), h.truth, h.policy, h.files)
        terms = [dict(t) for t in sub.terms]
        for t in terms:
            if t["id"] == "gait_phase":
                t["params"] = {**t.get("params", {}), "stand_threshold": None}
        return sub.standard(terms=terms)

    def under_clean(h: Harness) -> Trace:
        n = _const_rows(h, zero=True)
        return Harness(h.g.slice_steps(*n), h.truth, h.policy, h.files).standard()

    return [
        Defect("clean harness log (control)", "none", None, None, clean, expect_status="pass"),
        Defect(
            "history frame-major (time-major layout)",
            "A",
            None,
            "history layout",
            lambda h: hist(h, layout="time_major"),
            contract_variant="history5",
        ),
        Defect(
            "history newest-first",
            "A",
            None,
            "history order",
            lambda h: hist(h, order="newest_first"),
            contract_variant="history5",
        ),
        Defect(
            "history zero-init at reset",
            "A",
            None,
            "history init",
            lambda h: hist(h, init="zeros"),
            contract_variant="history5",
        ),
        Defect("quaternion read as xyzw", "A", "projected_gravity", "xyzw", xyzw),
        Defect("gyro in the world frame", "A", "base_ang_vel", "world frame", world_gyro),
        Defect(
            "observation remap skipped (Isaac order)",
            "A",
            "joint_pos_rel",
            "remap skipped",
            obs_remap,
        ),
        Defect("action remap skipped (Isaac order)", "C", "target", "remap skipped", action_remap),
        Defect(
            "joint velocity scaled 0.05", "A", "joint_vel_rel", "constant scale x0.05", vel_scale
        ),
        Defect(
            "action scale from deploy.yaml", "C", "target", "action scale from yaml", yaml_scale
        ),
        Defect("action offset dropped", "C", "target", "default offset not added", no_offset),
        Defect("one-step action delay", "C", "target", "1 policy step late", delay),
        Defect("raw action clipped before scaling", "C", "target", "clipped at", clip),
        Defect(
            "gains bound by index in Isaac order",
            "C",
            "effort",
            "gains bound by index",
            gains_order,
        ),
        Defect(
            "multiple: xyzw + offset dropped + gains order",
            "A+C",
            None,
            "xyzw|default offset not added|gains bound by index",
            multi,
        ),
        Defect("simulator state read one step late", "A", None, "1 step late", timing),
        Defect(
            "under-excited, standing: command scaled x2 (invisible at zero command)",
            "A",
            None,
            None,
            under_stand,
            expect_status="not_covered",
        ),
        Defect(
            "under-excited, constant walk command: phase stand rule missing",
            "A",
            None,
            None,
            under_walk,
            expect_status="not_covered",
        ),
        Defect(
            "under-excited, standing: clean (control)",
            "A",
            None,
            None,
            under_clean,
            expect_status="not_covered",
        ),
        Defect(
            "history5 contract, clean (control)",
            "none",
            None,
            None,
            lambda h: hist(h),
            expect_status="pass",
            contract_variant="history5",
        ),
    ]


def _const_rows(h: Harness, zero: bool) -> tuple[int, int]:
    """Longest run of rows with one constant command inside one episode (zero or nonzero)."""
    c = h.g["command"]
    reset = h.g["reset"]
    best, start = (0, 0), 0
    for k in range(1, len(c) + 1):
        if k == len(c) or reset[k] or not np.array_equal(c[k], c[start]):
            is_zero = not np.any(c[start])
            if is_zero == zero and k - start > best[1] - best[0]:
                best = (start, k)
            start = k
    return best


def _lag(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[1:] = x[:-1]
    out[0] = x[0]
    return out


def _term(h: Harness, tid: str) -> dict[str, Any]:
    return next(t for t in h.terms if t["id"] == tid)


def _ctx(ctx: TermContext, **kw: Any) -> TermContext:
    d = dict(ctx.__dict__)
    d.update(kw)
    return TermContext(**d)


def _state(s: RawState, **kw: Any) -> RawState:
    d = dict(s.__dict__)
    d.update(kw)
    return RawState(**d)


def judge(d: Defect, rep: Report) -> dict[str, Any]:
    """Did the comparator name the injected defect (and nothing else), or abstain where it should?"""
    failing = {k: b for k, b in rep.boundaries.items() if b.status == "fail"}
    named = {k: list(b.patterns) for k, b in failing.items()}
    terms = {k: [t.term for t in b.terms if t.status == "fail"] for k, b in failing.items()}
    text = " | ".join(p for ps in named.values() for p in ps)
    ambiguous = "ambiguous" in text or "none fits" in text
    if d.expect_status == "pass":
        ok = rep.verdict == "PASS"
    elif d.expect_status == "not_covered":
        ok = rep.verdict == "UNDETERMINED" and not failing
    else:
        want_b = set(d.expect_boundary.split("+"))
        ok = rep.verdict == "CONTRACT" and set(failing) == want_b and not ambiguous
        if d.expect_pattern:
            ok = ok and all(p in text for p in d.expect_pattern.split("|"))
        if d.expect_term:
            ok = ok and any(d.expect_term in ts for ts in terms.values())
    return {
        "defect": d.name,
        "expected": f"{d.expect_status} {d.expect_boundary} {d.expect_term or ''} "
        f"{d.expect_pattern or ''}".strip(),
        "verdict": rep.verdict,
        "named": named,
        "failing_terms": terms,
        "notes": {k: b.notes for k, b in rep.boundaries.items() if b.notes},
        "correct": bool(ok),
    }
