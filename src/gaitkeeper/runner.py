"""Closed-loop runner: a policy, its contract, and a target MJCF (plan section 7.1).

Backends (``RunConfig.backend``):

* ``native_implicit`` (default for implicit PD or an unknown actuator class):
  the PD runs inside MuJoCo every physics step as an affine actuator with the
  contract gains, integrator implicitfast, torque limit as actuator force range.
* ``explicit_zoh`` (contract says explicit PD): torque is computed once per
  training ``sim_dt`` from the state at its start and held across MuJoCo
  substeps; implicitfast with the actuator as a plain motor; the limit clips the
  PD output. When the model timestep does not divide ``sim_dt`` the timestep is
  changed to the largest one that does and is no larger, and the run says so.
* ``python_pd`` (debugging): PD every MuJoCo step in Python, integrator as the
  model says.
* ``standin_implicit`` (test fixture): a drive implicit in position and
  velocity, built from Python PD with kd 0, joint damping kd + h kp and the
  Euler integrator, so that (M + h kd + h^2 kp) a = kp (q* - q) - (kd + h kp) v
  - bias. It stands in for a PhysX style drive in the residual tests (plan
  Appendix B); never a controller claim.

The model is an MJCF, a compiled ``.mjb``, or a recorded trace directory, in
which case the model the source simulated is loaded (``models.load_model``).

Torque limits come from the target MJCF unless ``limit_source="contract"``.
Every result carries the controller line with provenance; any field taken
from a default, preset or unknown source raises CONTROLLER_ASSUMED.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .contract import Contract
from .models import find_id, load_model
from .terms import ObservationBuilder, quat_to_mat

BACKENDS = ("native_implicit", "explicit_zoh", "python_pd", "standin_implicit")
FALL_HEIGHT_FRACTION = 0.6
FALL_TILT_RAD = 1.0


# -- configuration ------------------------------------------------------------------


@dataclass
class Push:
    """One disturbance. ``kind`` "force": ``vector`` in N on ``body`` for
    ``duration`` s. ``kind`` "velocity": ``vector`` m/s added to the base
    linear velocity (x, y, z) once."""

    t: float
    kind: str = "force"
    vector: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body: str | None = None
    duration: float = 0.1


@dataclass
class PushGenerator:
    """Periodic random pushes, seeded by the run seed."""

    every_s: float = 5.0
    first_s: float = 0.1
    force: float = 0.0  # N, magnitude uniform in [0.5, 1] * force unless fixed
    fixed: bool = False
    duration: float = 0.08
    body: str = "random"  # "random" or a body name
    direction: str = "sphere"  # "sphere" or "horizontal"
    velocity: float = 0.0  # m/s per axis, uniform in [-v, v] on x and y


@dataclass
class External:
    """How joints the policy does not own are driven and observed."""

    joints: list[str]
    drive: str = "hold"  # "hold" or "trajectory"
    pose: Any = "default"  # "default", "zero", {name: value}, or {name: [[t, value], ...]}
    kp: dict[str, float] | None = None
    kd: dict[str, float] | None = None
    obs: str = "real"  # "real", "echo_action", "default"


@dataclass
class RunConfig:
    backend: str | None = None  # None: chosen from the contract's actuator kind
    seconds: float = 10.0
    command: tuple[float, float, float] = (0.5, 0.0, 0.0)
    schedule: list[tuple[float, tuple[float, float, float]]] | None = None
    seed: int = 0
    yaw0: float = 0.0
    init_noise: float = 0.0
    policy_mode: str = "policy"  # "policy", "zero", "random"
    pushes: list[Push] = field(default_factory=list)
    push_generator: PushGenerator | None = None
    external: list[External] | None = None  # None: the contract's ownership
    limit_source: str = "model"  # "model" or "contract"
    timestep: float | None = None  # python_pd and standin_implicit: override the MuJoCo step
    integrator: str | None = None  # python_pd only: "euler", "implicitfast", "implicit"
    armature: float | dict[str, float] | None = None  # override on the bound joints
    model_edit: Callable[[mujoco.MjModel], None] | None = None
    tail_s: float = 5.0
    record: bool = False
    # joints in control.unlisted that follow {name: [[t, value], ...]} instead of their pose
    unlisted_trajectory: dict[str, Any] | None = None
    contacts: bool = False  # log which bodies touch the world at every physics step
    physics: bool = False  # with record: log the state before every physics step (p/ keys)
    # Closed-loop command: called each policy step with (time, free-joint qpos) and returns
    # (vx, vy, wz); overrides command and schedule. If it has report(fell_at), the result
    # carries that as ``task`` (see gaitkeeper.tour).
    command_source: Any = None


def pool_context():
    """Worker processes start from a fresh server, not a fork of a process that
    already runs inference threads (forking those can deadlock)."""
    import multiprocessing

    return multiprocessing.get_context("forkserver")


def load_schedule(path: str | Path) -> list[tuple[float, tuple[float, float, float]]]:
    """Command schedule from YAML (``schedule: [[t, vx, vy, wz], ...]``) or CSV rows t,vx,vy,wz."""
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        import yaml

        rows = yaml.safe_load(path.read_text())
        rows = rows["schedule"] if isinstance(rows, dict) else rows
    else:
        rows = [
            [float(x) for x in line.split(",")]
            for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "t"))
        ]
    return [(float(r[0]), (float(r[1]), float(r[2]), float(r[3]))) for r in rows]


# -- binding ------------------------------------------------------------------------


@dataclass
class Binding:
    names: list[str]
    jid: np.ndarray
    qadr: np.ndarray
    dadr: np.ndarray
    aid: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    limit: np.ndarray
    limit_from: list[str]
    model_limit: np.ndarray
    model_limit_from: list[str]
    base_qadr: int
    base_dadr: int
    base_body: int
    timestep: float
    timestep_note: str
    substeps: int  # MuJoCo steps per policy step
    pd_every: int  # MuJoCo steps per torque update
    integrator: str
    model_notes: list[str] = field(default_factory=list)
    # joints the policy does not list, held by the harness: (actuator, qpos address, pose)
    unlisted: list = field(default_factory=list)
    unlisted_names: list[str] = field(default_factory=list)


_INTEGRATORS = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
    "rk4": mujoco.mjtIntegrator.mjINT_RK4,
}
_INTEGRATOR_NAMES = {int(v): k for k, v in _INTEGRATORS.items()}


def ground_contacts(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """Bodies in contact with a geom of the world body, as a flag per body."""
    out = np.zeros(m.nbody, dtype=bool)
    if d.ncon:
        b1 = m.geom_bodyid[d.contact.geom1[: d.ncon]]
        b2 = m.geom_bodyid[d.contact.geom2[: d.ncon]]
        out[b2[b1 == 0]] = True
        out[b1[b2 == 0]] = True
    out[0] = False
    return out


def model_torque_limits(
    m: mujoco.MjModel, jid: np.ndarray, aid: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Torque limit per bound joint as the MJCF states it, with where it came from."""
    out = np.full(len(jid), np.inf)
    src = ["none"] * len(jid)
    for i, (j, a) in enumerate(zip(jid, aid)):
        cands = []
        if a >= 0:
            gear = abs(float(m.actuator_gear[a, 0])) or 1.0
            if m.actuator_forcelimited[a]:
                cands.append(
                    (float(np.min(np.abs(m.actuator_forcerange[a]))), "actuator forcerange")
                )
            motor = (
                m.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                and m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_NONE
            )
            if motor and m.actuator_ctrllimited[a]:
                lim = (
                    float(np.min(np.abs(m.actuator_ctrlrange[a])))
                    * gear
                    * abs(float(m.actuator_gainprm[a, 0]))
                )
                cands.append((lim, "motor ctrlrange x gear"))
        if m.jnt_actfrclimited[j]:
            cands.append((float(np.min(np.abs(m.jnt_actfrcrange[j]))), "joint actuatorfrcrange"))
        if cands:
            v, s = min(cands)
            out[i], src[i] = v, s
    return out, src


def _divisor_step(period: float, max_step: float) -> tuple[float, int]:
    """Largest step period/n that is not above max_step (with a 1e-12 margin)."""
    n = max(1, math.ceil(period / max_step - 1e-9))
    return period / n, n


# -- runner -------------------------------------------------------------------------


@dataclass
class RunResult:
    fell_at: float | None
    dist: float
    dx: float
    dy: float
    vx_tail: float
    vy_tail: float
    wz_tail: float
    names: list[str]
    tau_rms: np.ndarray
    tau_peak: np.ndarray
    qd_peak: np.ndarray
    sat_frac: np.ndarray
    limit: np.ndarray
    limit_from: list[str]
    model_limit: np.ndarray
    controller: dict[str, Any]
    findings: list[str]
    pushes: list[dict[str, Any]]
    timestep: float
    seconds: float
    log: dict[str, np.ndarray] | None = None
    dadr: np.ndarray | None = None  # qvel addresses of the policy joints
    contacts: np.ndarray | None = None  # (physics steps, bodies) touching a world geom
    body_names: list[str] | None = None
    vel: np.ndarray | None = None  # (policy steps, 3) body vx, vy and yaw rate after each step
    task: dict[str, Any] | None = None  # the command source's report, when it has one

    @property
    def survived(self) -> bool:
        return self.fell_at is None

    def summary(self) -> str:
        f = "never" if self.fell_at is None else f"{self.fell_at:.2f} s"
        return (
            f"fell {f}, distance {self.dist:.2f} m, tail vx {self.vx_tail:+.3f} "
            f"vy {self.vy_tail:+.3f} wz {self.wz_tail:+.3f}"
        )


class Runner:
    def __init__(self, contract: Contract, mjcf: str | Path, policy: Any = None):
        self.contract = contract
        self.mjcf = str(mjcf)
        self.policy = policy
        self.names = list(contract.get("policy_io.joints.names"))
        pdt = contract.get("timing.policy_dt", None)
        if pdt is None:
            raise ValueError(
                "the contract does not state timing.policy_dt; give a deploy/export yaml, "
                "a preset, or --set timing.policy_dt=<seconds>"
            )
        self.policy_dt = float(pdt)
        sd = contract.get("timing.sim_dt", None)
        self.sim_dt = float(sd) if sd is not None else None
        d = contract.get("control.default_joint_pos")
        self.default = np.array([d[n] for n in self.names])
        sc = contract.get("control.actions.joint_pos.scale")
        of = contract.get("control.actions.joint_pos.offset")
        self.scale = np.array([sc[n] for n in self.names])
        self.offset = np.array([of[n] for n in self.names])
        clip = contract.get("control.actions.joint_pos.clip", None)
        self.clip = np.asarray(clip, dtype=float) if clip else None
        kp = contract.get("control.actuators.kp")
        kd = contract.get("control.actuators.kd")
        self.kp = np.array([kp[n] for n in self.names], dtype=float)
        self.kd = np.array([kd[n] for n in self.names], dtype=float)

    # -- backend and controller description --
    def choose_backend(self, requested: str | None) -> str:
        if requested:
            if requested not in BACKENDS:
                raise ValueError(f"unknown backend {requested!r}; known: {BACKENDS}")
            return requested
        kind = self.contract.get("control.actuators.kind", None)
        return "explicit_zoh" if kind == "explicit_pd" else "native_implicit"

    def controller(self, backend: str, b: Binding | None = None) -> dict[str, Any]:
        c = self.contract
        lines = {"backend": backend}
        assumed = []
        for k in ("kind", "pd_period", "integrator", "torque_limit_at"):
            path = f"control.actuators.{k}"
            v = c.get(path, None)
            p = c.prov(path)
            src = p.source if v is not None else "unknown"
            lines[k] = {"value": v, "from": src, "detail": p.detail}
            # A preset is read from the training config at a commit, not from the run that
            # produced this policy file, so it does not confirm the controller either.
            if src in ("unknown", "default", "preset"):
                assumed.append(k)
        if b is not None:
            lines["simulated"] = {
                "timestep": b.timestep,
                "timestep_note": b.timestep_note,
                "pd_every_mujoco_steps": b.pd_every,
                "integrator": b.integrator,
                "limit_from": sorted(set(b.limit_from)),
            }
        kind = lines["kind"]["value"]
        mismatch = (backend == "native_implicit" and kind == "explicit_pd") or (
            backend == "explicit_zoh" and kind == "implicit_pd"
        )
        lines["assumed"] = assumed
        lines["mismatch"] = mismatch
        lines["CONTROLLER_ASSUMED"] = (
            bool(assumed) or mismatch or backend in ("python_pd", "standin_implicit")
        )
        return lines

    @staticmethod
    def controller_text(ctl: dict[str, Any]) -> list[str]:
        out = []
        head = "CONTROLLER_ASSUMED" if ctl["CONTROLLER_ASSUMED"] else "Controller"
        parts = []
        for k in ("kind", "pd_period", "integrator", "torque_limit_at"):
            e = ctl[k]
            parts.append(f"{k} {e['value'] if e['value'] is not None else 'unknown'} ({e['from']})")
        out.append(f"{head}: backend {ctl['backend']}; " + "; ".join(parts))
        sim = ctl.get("simulated")
        if sim:
            out.append(
                f"  simulated: MuJoCo step {sim['timestep'] * 1e3:.4g} ms, torque updated every "
                f"{sim['pd_every_mujoco_steps']} step(s), integrator {sim['integrator']}, "
                f"limits from {', '.join(sim['limit_from'])}"
            )
            if sim["timestep_note"]:
                out.append(f"  {sim['timestep_note']}")
        if ctl["assumed"]:
            out.append(
                f"  not from a source: {', '.join(ctl['assumed'])}; boundary C capped at assumed"
            )
        if ctl["mismatch"]:
            out.append("  backend differs from the contract's actuator kind")
        if ctl["backend"] == "python_pd":
            out.append("  python_pd is a debugging backend; its results are not a controller claim")
        return out

    def _unlisted(self, m: mujoco.MjModel, jid: np.ndarray, act_of: dict[int, int]) -> list:
        """Actuated hinge joints the contract does not list. With ``control.unlisted`` they are
        held as position servos at its pose (the harness's job for a legs-only policy);
        without it they get no torque, as before."""
        spec = self.contract.get("control.unlisted", None)
        if not spec:
            return []
        listed = {int(j) for j in jid}
        pose, kp, kd = spec.get("pose", {}) or {}, spec.get("kp", 0.0), spec.get("kd", 0.0)
        out = []
        for j in range(m.njnt):
            if j in listed or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or j not in act_of:
                continue
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            k = float(kp.get(nm, 0.0) if isinstance(kp, dict) else kp)
            c = float(kd.get(nm, 0.0) if isinstance(kd, dict) else kd)
            out.append((act_of[j], int(m.jnt_qposadr[j]), float(pose.get(nm, 0.0)), k, c, nm))
        return out

    # -- model --
    def build(self, cfg: RunConfig, backend: str) -> tuple[mujoco.MjModel, mujoco.MjData, Binding]:
        lm = load_model(self.mjcf)
        m = lm.model
        if cfg.model_edit is not None:
            cfg.model_edit(m)
        jid = np.array([find_id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names])
        missing = [n for n, j in zip(self.names, jid) if j < 0]
        if missing:
            raise KeyError(f"joints not in {self.mjcf}: {missing}")
        act_of = {}
        for a in range(m.nu):
            if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
                act_of.setdefault(int(m.actuator_trnid[a, 0]), a)
        aid = np.array([act_of.get(int(j), -1) for j in jid])
        if (aid < 0).any():
            raise KeyError(f"no actuator on joints {[n for n, a in zip(self.names, aid) if a < 0]}")
        if cfg.armature is not None:
            for i, n in enumerate(self.names):
                v = cfg.armature if not isinstance(cfg.armature, dict) else cfg.armature.get(n)
                if v is not None:
                    m.dof_armature[m.jnt_dofadr[jid[i]]] = float(v)
        model_lim, model_from = model_torque_limits(m, jid, aid)
        if cfg.limit_source == "contract":
            el = self.contract.get("model.effort_limit", None)
            if el is None:
                raise ValueError("limit_source contract: the contract has no model.effort_limit")
            lim = np.array([float(el[n]) for n in self.names])
            lim_from = ["contract"] * len(self.names)
        else:
            lim, lim_from = model_lim.copy(), list(model_from)

        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if not free:
            raise ValueError("fixed-base models are not supported")
        fj = free[0]

        h0 = float(m.opt.timestep)
        note = ""
        if backend == "explicit_zoh":
            period = self.sim_dt if self.sim_dt is not None else h0
            if self.sim_dt is None:
                note = "training sim_dt unknown: torque held over one MuJoCo step"
            h, n = _divisor_step(period, h0)
            if abs(h - h0) > 1e-12:
                note = (
                    f"timestep changed from {h0 * 1e3:.4g} ms to {h * 1e3:.4g} ms so that "
                    f"{n} MuJoCo steps make one training sim_dt of {period * 1e3:.4g} ms"
                )
            pd_every = n
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        elif backend == "native_implicit":
            h = h0
            pd_every = 1
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        elif backend == "standin_implicit":
            h = float(cfg.timestep) if cfg.timestep else h0
            pd_every = 1
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        else:
            h = float(cfg.timestep) if cfg.timestep else h0
            pd_every = 1
            if cfg.integrator:
                m.opt.integrator = _INTEGRATORS[cfg.integrator]
        if backend not in ("python_pd", "standin_implicit") and cfg.timestep:
            raise ValueError("timestep override is for python_pd and standin_implicit only")
        sub = self.policy_dt / h
        if abs(sub - round(sub)) > 1e-6:
            h2, _ = _divisor_step(self.policy_dt, h)
            note = (note + "; " if note else "") + (
                f"timestep {h * 1e3:.4g} ms does not divide the policy step; using {h2 * 1e3:.4g} ms"
            )
            h = h2
            sub = self.policy_dt / h
            if backend == "explicit_zoh" and self.sim_dt is not None:
                pd_every = max(1, round(self.sim_dt / h))
        m.opt.timestep = h

        for i in range(len(self.names)):
            a, j = int(aid[i]), int(jid[i])
            m.jnt_actfrclimited[j] = 0  # one place enforces the limit: below
            m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_gear[a, :] = 0.0
            m.actuator_gear[a, 0] = 1.0
            m.actuator_ctrllimited[a] = 0
            m.actuator_dyntype[a] = mujoco.mjtDyn.mjDYN_NONE
            if backend == "native_implicit":
                m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
                m.actuator_gainprm[a, :] = 0.0
                m.actuator_biasprm[a, :] = 0.0
                m.actuator_gainprm[a, 0] = self.kp[i]
                m.actuator_biasprm[a, :3] = [0.0, -self.kp[i], -self.kd[i]]
            else:
                m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
                m.actuator_gainprm[a, :] = 0.0
                m.actuator_biasprm[a, :] = 0.0
                m.actuator_gainprm[a, 0] = 1.0
            if np.isfinite(lim[i]):
                m.actuator_forcelimited[a] = 1
                m.actuator_forcerange[a] = [-lim[i], lim[i]]
            else:
                m.actuator_forcelimited[a] = 0
        unlisted = self._unlisted(m, jid, act_of)
        for a, _, _, kp_u, kd_u, _ in unlisted:
            m.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            m.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
            m.actuator_dyntype[a] = mujoco.mjtDyn.mjDYN_NONE
            m.actuator_gear[a, :] = 0.0
            m.actuator_gear[a, 0] = 1.0
            m.actuator_ctrllimited[a] = 0
            m.actuator_gainprm[a, :] = 0.0
            m.actuator_biasprm[a, :] = 0.0
            m.actuator_gainprm[a, 0] = kp_u
            m.actuator_biasprm[a, :3] = [0.0, -kp_u, -kd_u]
        d = mujoco.MjData(m)
        b = Binding(
            names=self.names,
            jid=jid,
            qadr=m.jnt_qposadr[jid].copy(),
            dadr=m.jnt_dofadr[jid].copy(),
            aid=aid,
            kp=self.kp.copy(),
            kd=self.kd.copy(),
            limit=lim,
            limit_from=lim_from,
            model_limit=model_lim,
            model_limit_from=model_from,
            base_qadr=int(m.jnt_qposadr[fj]),
            base_dadr=int(m.jnt_dofadr[fj]),
            base_body=int(m.jnt_bodyid[fj]),
            timestep=h,
            timestep_note=note,
            substeps=int(round(sub)),
            pd_every=pd_every,
            integrator=_INTEGRATOR_NAMES.get(int(m.opt.integrator), str(m.opt.integrator)),
            model_notes=list(lm.notes),
            unlisted=[(a, q, pose) for a, q, pose, _, _, _ in unlisted],
            unlisted_names=[nm for *_, nm in unlisted],
        )
        return m, d, b

    def externals(self, cfg: RunConfig) -> list[External]:
        if cfg.external is not None:
            return cfg.external
        own = self.contract.get("control.ownership", None) or {}
        out = []
        for e in own.get("external", []) or []:
            out.append(External(**e) if isinstance(e, dict) else e)
        return out

    def reset_state(
        self, m: mujoco.MjModel, d: mujoco.MjData, b: Binding, cfg: RunConfig, pose: np.ndarray, rng
    ):
        mujoco.mj_resetData(m, d)
        q0 = b.base_qadr
        d.qpos[q0 + 3 : q0 + 7] = [math.cos(cfg.yaw0 / 2), 0.0, 0.0, math.sin(cfg.yaw0 / 2)]
        d.qpos[b.qadr] = pose
        for a, q, p_ in b.unlisted:
            d.qpos[q] = p_
            d.ctrl[a] = p_
        if cfg.init_noise > 0:
            d.qpos[b.qadr] += rng.normal(0.0, cfg.init_noise, len(b.qadr))
        mujoco.mj_forward(m, d)

    def run(self, cfg: RunConfig | None = None) -> RunResult:
        cfg = cfg or RunConfig()
        backend = self.choose_backend(cfg.backend)
        m, d, b = self.build(cfg, backend)
        ctl = self.controller(backend, b)
        rng = np.random.default_rng(cfg.seed)
        n = len(self.names)
        idx = {nm: i for i, nm in enumerate(self.names)}

        # ownership
        owned = np.ones(n, dtype=bool)
        hold_pose = self.default.copy()
        ext_kp, ext_kd = self.kp.copy(), self.kd.copy()
        ext_obs = np.array(["real"] * n, dtype=object)
        trajectories: dict[int, np.ndarray] = {}
        for e in self.externals(cfg):
            for nm in e.joints:
                i = idx[nm]
                owned[i] = False
                ext_obs[i] = e.obs
                if e.kp:
                    ext_kp[i] = e.kp.get(nm, ext_kp[i])
                if e.kd:
                    ext_kd[i] = e.kd.get(nm, ext_kd[i])
                if e.pose == "zero":
                    hold_pose[i] = 0.0
                elif isinstance(e.pose, dict) and nm in e.pose:
                    v = e.pose[nm]
                    if e.drive == "trajectory":
                        trajectories[i] = np.asarray(v, dtype=float)
                        hold_pose[i] = trajectories[i][0, 1]
                    else:
                        hold_pose[i] = float(v)
        kp = np.where(owned, self.kp, ext_kp)
        kd = np.where(owned, self.kd, ext_kd)
        if backend == "standin_implicit":
            # Velocity and position implicit: Euler integrates joint damping implicitly.
            m.dof_damping[b.dadr] += kd + b.timestep * kp
            kd = np.zeros(n)
        if backend == "native_implicit":
            for i in np.flatnonzero(~owned):
                a = b.aid[i]
                m.actuator_gainprm[a, 0] = kp[i]
                m.actuator_biasprm[a, :3] = [0.0, -kp[i], -kd[i]]

        un_traj: list[tuple[int, np.ndarray]] = []
        for k, nm in enumerate(b.unlisted_names):
            if cfg.unlisted_trajectory and nm in cfg.unlisted_trajectory:
                tr = np.asarray(cfg.unlisted_trajectory[nm], dtype=float)
                a_, q_, _ = b.unlisted[k]
                b.unlisted[k] = (a_, q_, float(tr[0, 1]))
                un_traj.append((a_, tr))
        if cfg.unlisted_trajectory:
            stray = set(cfg.unlisted_trajectory) - set(b.unlisted_names)
            if stray:
                raise ValueError(f"unlisted_trajectory: not unlisted joints: {sorted(stray)}")
        start_pose = np.where(owned, self.default, hold_pose)
        self.reset_state(m, d, b, cfg, start_pose, rng)
        q0, v0 = b.base_qadr, b.base_dadr
        z0 = float(d.qpos[q0 + 2])
        p0 = d.qpos[q0 : q0 + 2].copy()
        builder = ObservationBuilder(self.contract)
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        if cfg.policy_mode == "policy" and self.policy is None:
            raise ValueError("no policy given")

        steps = int(round(cfg.seconds / self.policy_dt))
        cmd = np.array(cfg.command, dtype=float)
        prev_action = np.zeros(n)
        target = start_pose.copy()
        pushes = sorted(cfg.pushes, key=lambda p: p.t)
        gen = cfg.push_generator
        next_gen = gen.first_s if gen and (gen.force > 0 or gen.velocity > 0) else math.inf
        active: list[tuple[int, np.ndarray, float]] = []
        applied: list[dict[str, Any]] = []
        tau = np.zeros(n)
        tau_sq = np.zeros(n)
        tau_peak = np.zeros(n)
        qd_peak = np.zeros(n)
        sat = np.zeros(n)
        nphys = 0
        vx_h, vy_h, wz_h = [], [], []
        fell_at = None
        log: dict[str, list] | None = (
            {
                k: []
                for k in (
                    "obs",
                    "action",
                    "command",
                    "qpos",
                    "qvel",
                    "target",
                    "effort",
                    "episode_step",
                )
            }
            if cfg.record
            else None
        )
        contacts: list[np.ndarray] | None = [] if cfg.contacts else None
        plog: dict[str, list] | None = (
            {
                k: []
                for k in ("qpos", "qvel", "ctrl", "step", "substep", "time", "xfrc", "xfrc_body")
            }
            if cfg.record and cfg.physics
            else None
        )
        hinge_types = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        all_hinge = [j for j in range(m.njnt) if int(m.jnt_type[j]) in hinge_types]

        for t in range(steps):
            time = t * self.policy_dt
            if cfg.schedule:
                for ts, c_ in cfg.schedule:
                    if time + 1e-9 >= ts:
                        cmd = np.array(c_, dtype=float)
            if cfg.command_source is not None:
                cmd = np.asarray(cfg.command_source(time, d.qpos[q0 : q0 + 7].copy()), float)
            quat = d.qpos[q0 + 3 : q0 + 7].copy()
            w_b = d.qvel[v0 + 3 : v0 + 6].copy()
            qj = d.qpos[b.qadr].copy()
            vj = d.qvel[b.dadr].copy()
            for i in np.flatnonzero(~owned):
                if ext_obs[i] == "echo_action":
                    qj[i] = self.offset[i] + self.scale[i] * prev_action[i]
                    vj[i] = 0.0
                elif ext_obs[i] == "default":
                    qj[i] = self.default[i]
                    vj[i] = 0.0
            obs = builder.step(
                quat, w_b, qj, vj, self.names, cmd, t, prev_action, d.qvel[v0 : v0 + 3].copy()
            )
            if cfg.policy_mode == "policy":
                a = self.policy.step(obs)
            elif cfg.policy_mode == "zero":
                a = np.zeros(n)
            else:
                a = rng.normal(0.0, 1.0, n)
            tgt = self.offset + self.scale * a
            if self.clip is not None:
                tgt = np.clip(tgt, self.clip[:, 0], self.clip[:, 1])
            target = np.where(owned, tgt, hold_pose)
            for i, tr in trajectories.items():
                target[i] = np.interp(time, tr[:, 0], tr[:, 1])
            for a_, tr in un_traj:
                d.ctrl[a_] = np.interp(time, tr[:, 0], tr[:, 1])
            if log is not None:
                log["obs"].append(obs.astype(np.float32))
                log["action"].append(a.astype(np.float32))
                log["command"].append(cmd.copy())
                log["qpos"].append(d.qpos.copy())
                log["qvel"].append(d.qvel.copy())
                log["target"].append(target.copy())
                log["episode_step"].append(t)
            prev_action = a

            # disturbances
            while pushes and time + 1e-9 >= pushes[0].t:
                p = pushes.pop(0)
                applied.append(self._apply_push(m, d, b, p, time, active))
            if time + 1e-9 >= next_gen:
                next_gen += gen.every_s
                if gen.velocity > 0:
                    v = rng.uniform(-gen.velocity, gen.velocity, 2)
                    applied.append(
                        self._apply_push(
                            m, d, b, Push(time, "velocity", (v[0], v[1], 0.0)), time, active
                        )
                    )
                if gen.force > 0:
                    body = (
                        m.body(int(rng.integers(1, m.nbody))).name
                        if gen.body == "random"
                        else gen.body
                    )
                    u = rng.normal(size=3)
                    if gen.direction == "horizontal":
                        u[2] = 0.0
                    u /= np.linalg.norm(u)
                    mag = gen.force if gen.fixed else gen.force * rng.uniform(0.5, 1.0)
                    applied.append(
                        self._apply_push(
                            m,
                            d,
                            b,
                            Push(time, "force", tuple(u * mag), body, gen.duration),
                            time,
                            active,
                        )
                    )

            # physics
            eff_first = None
            for s in range(b.substeps):
                tnow = time + s * b.timestep
                d.xfrc_applied[:] = 0.0
                active = [(bid, f, until) for bid, f, until in active if tnow < until - 1e-12]
                for bid, f, _ in active:
                    d.xfrc_applied[bid, :3] += f
                if plog is not None:
                    plog["qpos"].append(d.qpos.copy())
                    plog["qvel"].append(d.qvel.copy())
                    plog["ctrl"].append(target.copy())
                    plog["step"].append(t)
                    plog["substep"].append(s)
                    plog["time"].append(tnow)
                    plog["xfrc"].append(
                        np.sum([f for _, f, _ in active], axis=0) if active else np.zeros(3)
                    )
                    bodies = {bid for bid, _, _ in active}
                    if len(bodies) > 1:
                        raise ValueError("physics log: concurrent pushes on two bodies")
                    plog["xfrc_body"].append(bodies.pop() if bodies else -1)
                if backend == "native_implicit":
                    d.ctrl[b.aid] = target
                elif s % b.pd_every == 0:
                    tau = kp * (target - d.qpos[b.qadr]) - kd * d.qvel[b.dadr]
                    tau = np.clip(tau, -b.limit, b.limit)
                    d.ctrl[b.aid] = tau
                mujoco.mj_step(m, d)
                if backend == "native_implicit":
                    tau = d.actuator_force[b.aid].copy()
                tau_sq += tau * tau
                tau_peak = np.maximum(tau_peak, np.abs(tau))
                sat += np.abs(tau) >= b.limit - 1e-6
                if eff_first is None:
                    eff_first = tau.copy()  # from the state the observation saw
                qd_peak = np.maximum(qd_peak, np.abs(d.qvel[b.dadr]))
                nphys += 1
                if contacts is not None:
                    contacts.append(ground_contacts(m, d))
            if log is not None:
                log["effort"].append(eff_first)

            r = quat_to_mat(d.qpos[q0 + 3 : q0 + 7][None])[0]
            vb = r.T @ d.qvel[v0 : v0 + 3]
            vx_h.append(vb[0])
            vy_h.append(vb[1])
            wz_h.append(d.qvel[v0 + 5])
            g = r.T @ np.array([0.0, 0.0, -1.0])
            roll = math.atan2(-g[1], -g[2])
            pitch = math.atan2(g[0], math.hypot(g[1], g[2]))
            if (
                d.qpos[q0 + 2] < FALL_HEIGHT_FRACTION * z0
                or abs(roll) > FALL_TILT_RAD
                or abs(pitch) > FALL_TILT_RAD
                or not np.isfinite(d.qpos).all()
            ):
                fell_at = (t + 1) * self.policy_dt
                break

        tail = max(1, int(round(cfg.tail_s / self.policy_dt)))
        findings = []
        if ctl["CONTROLLER_ASSUMED"]:
            findings.append("CONTROLLER_ASSUMED")
        res = RunResult(
            fell_at=fell_at,
            dist=float(np.linalg.norm(d.qpos[q0 : q0 + 2] - p0)),
            dx=float(d.qpos[q0] - p0[0]),
            dy=float(d.qpos[q0 + 1] - p0[1]),
            vx_tail=float(np.mean(vx_h[-tail:])) if vx_h else math.nan,
            vy_tail=float(np.mean(vy_h[-tail:])) if vy_h else math.nan,
            wz_tail=float(np.mean(wz_h[-tail:])) if wz_h else math.nan,
            names=self.names,
            tau_rms=np.sqrt(tau_sq / max(nphys, 1)),
            tau_peak=tau_peak,
            qd_peak=qd_peak,
            sat_frac=sat / max(nphys, 1),
            limit=b.limit,
            limit_from=b.limit_from,
            model_limit=b.model_limit,
            controller=ctl,
            findings=findings,
            pushes=applied,
            timestep=b.timestep,
            seconds=(fell_at if fell_at is not None else steps * self.policy_dt),
            dadr=b.dadr.copy(),
            vel=np.c_[vx_h, vy_h, wz_h] if vx_h else np.zeros((0, 3)),
        )
        if cfg.command_source is not None and hasattr(cfg.command_source, "report"):
            res.task = cfg.command_source.report(fell_at)
        if contacts is not None:
            res.contacts = np.array(contacts, dtype=bool).reshape(len(contacts), m.nbody)
            res.body_names = [m.body(i).name for i in range(m.nbody)]
        if log is not None:
            res.log = {k: np.asarray(v) for k, v in log.items()}
            res.log["reset"] = np.arange(len(log["obs"])) == 0
            res.log["_hinge_names"] = np.array([m.joint(j).name for j in all_hinge])
            if plog is not None:
                for k, v in plog.items():
                    res.log[f"p/{k}"] = np.asarray(v)
                res.log["_body_names"] = np.array([m.body(i).name for i in range(m.nbody)])
                res.log["_sim_dt"] = np.array(b.timestep)
        return res

    @staticmethod
    def _apply_push(m, d, b: Binding, p: Push, time: float, active: list) -> dict[str, Any]:
        if p.kind == "velocity":
            d.qvel[b.base_dadr : b.base_dadr + 3] += np.asarray(p.vector, dtype=float)
            mujoco.mj_forward(m, d)
            return {"t": time, "kind": "velocity", "vector": [float(x) for x in p.vector]}
        name = p.body
        bid = b.base_body if name in (None, "base") else find_id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"push body {name!r} not in the model")
        active.append((bid, np.asarray(p.vector, dtype=float), time + p.duration))
        return {
            "t": time,
            "kind": "force",
            "body": m.body(bid).name,
            "vector": [float(x) for x in p.vector],
            "duration": p.duration,
        }

    def to_trace(self, res: RunResult, meta: dict[str, Any] | None = None, kind: str = "harness"):
        """Trace of a recorded run, harness format by default. Marked as written by
        this runner, so verifying it can only show self-consistency. With
        ``physics`` logging the ``p/`` keys are included (golden format)."""
        from .trace import Trace

        if res.log is None:
            raise ValueError("run with record=True")
        lg = dict(res.log)
        hinge = [str(x) for x in lg.pop("_hinge_names")]
        bodies = lg.pop("_body_names", None)
        sim_dt = lg.pop("_sim_dt", None)
        arrays = {k: v for k, v in lg.items()}
        arrays["action_applied"] = arrays["action"]
        extra: dict[str, Any] = {}
        if sim_dt is not None:
            extra = {
                "sim_dt": float(sim_dt),
                "decimation": int(round(self.policy_dt / float(sim_dt))),
                "policy_dt": self.policy_dt,
                "target_joint_names": list(self.names),
                "body_names": [str(x) for x in bodies],
                "physics_rows": "state before each mj_step",
                "framework": {"name": "gaitkeeper runner", "mujoco": mujoco.__version__},
                "engine": f"mujoco {mujoco.__version__} ({res.controller['backend']})",
            }
        meta = {
            "written_by_gaitkeeper_runner": True,
            **extra,
            "controller": res.controller,
            "state_layout": {
                "free_joint": True,
                "quat_order": "wxyz",
                "ang_vel_frame": "body",
                "lin_vel_frame": "world",
                "joint_names": hinge,
            },
            **(meta or {}),
        }
        return Trace(arrays, meta, kind=kind)
