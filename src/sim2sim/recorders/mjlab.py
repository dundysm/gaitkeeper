"""Golden trace recorder for mjlab manager-based velocity envs.

Runs inside the mjlab venv on CPU. Steps the training env with the shipped
ONNX policy under an excitation schedule, writes the contract from live env
objects, every physics step's state and effort, and the compiled model.

    python -m sim2sim.recorders.mjlab --task Unitree-G1-Flat \
        --task-path /path/to/unitree_rl_mjlab --task-module src.tasks \
        --onnx policy.onnx --out runs/g1_golden
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import SCHEMA, Contract, sha256_file
from ..terms import StateLayout
from ..trace import Trace, check_state_increments
from .schedule import DEFAULT_PUSHES, DEFAULT_SCHEDULE, command_at, excitation_checklist, push_at

INTEGRATORS = {0: "mujoco_euler", 1: "mujoco_rk4", 2: "mujoco_implicit", 3: "mujoco_implicitfast"}


def _np(x: Any) -> np.ndarray:
    return x[0].detach().cpu().numpy().copy()


def _strip(name: str) -> str:
    return name.split("/", 1)[1] if "/" in name else name


def _ranges(r: Any) -> dict[str, list[float] | None]:
    def f(v: Any) -> list[float] | None:
        return None if v is None else [float(v[0]), float(v[1])]

    return {
        "vx": f(r.lin_vel_x),
        "vy": f(r.lin_vel_y),
        "wz": f(r.ang_vel_z),
        "heading": f(getattr(r, "heading", None)),
    }


def _tensor_list(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.reshape(-1).tolist() if x.ndim > 1 else x.tolist()
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    return float(x)


def _noise_desc(n: Any) -> Any:
    if n is None:
        return None
    d = {"type": type(n).__name__}
    for k in ("n_min", "n_max", "mean", "std", "operation"):
        if hasattr(n, k):
            v = getattr(n, k)
            d[k] = v if isinstance(v, str) else _tensor_list(v)
    return d


TERM_IDS = {
    "projected_gravity": "projected_gravity",
    "generated_commands": "velocity_commands",
    "phase": "gait_phase",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_rel": "joint_vel_rel",
    "last_action": "last_action",
    "base_ang_vel": "base_ang_vel",
}


def live_contract(
    env: Any, train_cfg: Any, play_cmd_ranges: dict[str, Any], onnx_path: Path
) -> Contract:
    import mujoco

    m = env.sim.mj_model
    robot = env.scene["robot"]
    act = env.action_manager.get_term("joint_pos")
    names = [_strip(n) for n in act.target_names]
    c = Contract({"schema": SCHEMA})
    live = "live"
    c.set(
        "source",
        {
            "format": "mjlab_live",
            "framework": {"name": "mjlab", "version": _version("mjlab")},
            "files": [{"path": onnx_path.name, "sha256": sha256_file(onnx_path)}],
        },
        live,
    )
    c.set("timing.policy_dt", float(env.step_dt), live, "env.step_dt")
    c.set("timing.sim_dt", float(m.opt.timestep), live, "mj_model.opt.timestep")
    c.set("timing.decimation", int(env.cfg.decimation), live, "env.cfg.decimation")
    c.set("timing.order", ["obs", "infer", "target", "pd", "step"], live, "ManagerBasedRlEnv.step")
    c.set("timing.target_hold", "zoh", live, "action applied unchanged on every substep")

    c.set("policy_io.joints.names", names, live, "action term target_names")
    # Observation terms of the actor group, from the live manager.
    om = env.observation_manager
    group = "actor" if "actor" in om._group_obs_term_names else "policy"
    terms = []
    hist = set()
    for tname, tcfg in zip(om._group_obs_term_names[group], om._group_obs_term_cfgs[group]):
        fname = tcfg.func.__name__
        params = {
            k: (v if isinstance(v, (int, float, str, bool)) else str(v))
            for k, v in tcfg.params.items()
        }
        tid = TERM_IDS.get(fname)
        if fname == "builtin_sensor" and "ang_vel" in str(tcfg.params.get("sensor_name", "")):
            tid = "base_ang_vel"
        if tid == "gait_phase":
            src = inspect.getsource(tcfg.func)
            params["stand_threshold"] = 0.1 if "< 0.1" in src else None
            params["clock"] = "episode_steps_since_reset"
            import torch

            params["arithmetic"] = str(torch.get_default_dtype()).removeprefix("torch.")
        if tid == "last_action":
            params["reset"] = "zeros"
        dim = int(om._group_obs_term_dim[group][len(terms)][-1])
        scale = _tensor_list(tcfg.scale)
        terms.append(
            {
                "id": tid,
                "source_name": tname,
                "function": f"{tcfg.func.__module__}.{fname}",
                "dim": dim,
                "scale": scale if isinstance(scale, list) else [scale or 1.0] * dim,
                "clip": _tensor_list(tcfg.clip),
                "params": params,
                "noise": _noise_desc(tcfg.noise),
                "delay_max_lag": int(getattr(tcfg, "delay_max_lag", 0)),
            }
        )
        hist.add(int(tcfg.history_length or 1))
    c.set(
        "policy_io.observation_groups.policy.terms",
        terms,
        live,
        f"observation_manager group '{group}'",
    )
    c.set(
        "policy_io.observation_groups.policy.history",
        {
            "length": max(hist),
            "layout": "term_major",
            "order": "oldest_first",
            "init": "repeat_first",
        },
        live,
        "per-term CircularBuffer, flattened per term",
    )
    c.set(
        "policy_io.observation_groups.policy.clip_then_scale",
        True,
        live,
        "ObservationManager.compute_group",
    )

    # IMU: the gyro site and its rotation relative to the root body.
    sid = None
    for tcfg in om._group_obs_term_cfgs[group]:
        if tcfg.func.__name__ == "builtin_sensor" and "ang_vel" in str(
            tcfg.params.get("sensor_name", "")
        ):
            sen = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, tcfg.params["sensor_name"])
            sid = int(m.sensor_objid[sen])
    root_body = int(m.jnt_bodyid[0 if m.jnt_type[0] == 0 else _free_joint(m)])
    if sid is not None:
        if int(m.site_bodyid[sid]) != root_body:
            raise RuntimeError("gyro site is not on the root body; extend the IMU model")
        c.set(
            "policy_io.imu",
            {
                "body": _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root_body)),
                "site": _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid)),
                "frame": "body",
                "rotation_in_root": m.site_quat[sid].tolist(),
            },
            live,
            "gyro sensor site",
        )

    # Commands: limit is the play config's range; trained comes from the training config.
    tc = train_cfg.commands["twist"]
    cur = train_cfg.curriculum.get("command_vel")
    stages = cur.params.get("velocity_stages") if cur is not None else None
    trained = _ranges(tc.ranges)
    if stages:
        last = dict(_ranges(tc.ranges))
        for st in stages:
            for ax, key in (("vx", "lin_vel_x"), ("vy", "lin_vel_y"), ("wz", "ang_vel_z")):
                if key in st:
                    last[ax] = [float(st[key][0]), float(st[key][1])]
        trained = last
    c.set(
        "policy_io.commands.base_velocity",
        {
            "name": "twist",
            "limit": {k: v for k, v in play_cmd_ranges.items() if k != "heading"},
            "trained": {k: trained[k] for k in ("vx", "vy", "wz")},
            "heading": "on" if tc.heading_command else "off",
        },
    )
    c.provenance["policy_io.commands.base_velocity.limit"] = _prov(
        live, "play config command ranges"
    )
    detail = (
        f"training config: final curriculum stage {stages[-1] if stages else None} if training ran past "
        f"its step; resampled norms <= 0.1 are zeroed; heading_command={tc.heading_command}, "
        f"rel_heading_envs={getattr(tc, 'rel_heading_envs', None)} (wz from the heading controller, "
        f"stiffness {tc.heading_control_stiffness}); rel_standing_envs={tc.rel_standing_envs}"
    )
    c.provenance["policy_io.commands.base_velocity.trained"] = _prov(live, detail)

    # Control.
    scale = _np(act.scale) if hasattr(act.scale, "shape") else np.full(len(names), float(act.scale))
    offset = (
        _np(act.offset) if hasattr(act.offset, "shape") else np.full(len(names), float(act.offset))
    )
    c.set("control.actions.joint_pos.kind", "joint_position", live, type(act).__name__)
    c.set("control.actions.joint_pos.joints", names, live, "")
    c.set(
        "control.actions.joint_pos.scale",
        dict(zip(names, scale.tolist())),
        live,
        "action term scale",
    )
    c.set(
        "control.actions.joint_pos.offset",
        dict(zip(names, offset.tolist())),
        live,
        "action term offset",
    )
    c.set("control.actions.joint_pos.clip", None, live, "no clip in the action term")
    c.set("control.actions.joint_pos.clip_stage", "none", live, "")
    c.set("control.actions.joint_pos.delay_steps", 0, live, "")
    c.set("control.actions.joint_pos.filter", "none", live, "")
    c.set("control.ownership", {"owned": "all", "external": []}, live, "")
    jn = [_strip(n) for n in robot.joint_names]
    c.set(
        "control.default_joint_pos",
        dict(zip(jn, _np(robot.data.default_joint_pos).tolist())),
        live,
        "entity default_joint_pos",
    )
    kp, kd, lim, kinds = {}, {}, {}, set()
    for a in range(m.nu):
        j = _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])))
        kinds.add((int(m.actuator_gaintype[a]), int(m.actuator_biastype[a])))
        kp[j] = float(m.actuator_gainprm[a, 0])
        kd[j] = float(-m.actuator_biasprm[a, 2])
        lim[j] = float(m.actuator_forcerange[a, 1]) if m.actuator_forcelimited[a] else None
    if kinds != {(0, 1)}:
        raise RuntimeError(f"unexpected actuator types {kinds}")
    c.set(
        "control.actuators.kind",
        "implicit_pd",
        live,
        "MuJoCo position actuators: force = kp (ctrl - q) - kv qd, damping integrated by the solver",
    )
    c.set("control.actuators.pd_period", "solver", live, "evaluated inside every physics step")
    c.set(
        "control.actuators.integrator",
        INTEGRATORS.get(int(m.opt.integrator), str(m.opt.integrator)),
        live,
        "mj_model.opt.integrator",
    )
    c.set("control.actuators.torque_limit_at", "actuator_force", live, "actuator forcerange")
    c.set("control.actuators.kp", {n: kp[n] for n in names}, live, "actuator gainprm[0]")
    c.set("control.actuators.kd", {n: kd[n] for n in names}, live, "-actuator biasprm[2]")
    c.set(
        "control.actuators.encoder_bias",
        dict(zip(names, _np(robot.data.encoder_bias).tolist())),
        live,
        "subtracted from the target in JointPositionAction.apply_actions; zero when the event is off",
    )

    # Model, read live.
    dof = {
        (_strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))): int(m.jnt_dofadr[j])
        for j in range(m.njnt)
        if m.jnt_type[j] != 0
    }
    c.set(
        "model.root",
        {"body": _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root_body))},
        live,
        "",
    )
    c.set(
        "model.armature",
        {n: float(m.dof_armature[dof[n]]) for n in names},
        live,
        "mj_model.dof_armature",
    )
    c.set(
        "model.joint_friction",
        {n: float(m.dof_frictionloss[dof[n]]) for n in names},
        live,
        "mj_model.dof_frictionloss",
    )
    c.set(
        "model.joint_damping",
        {n: float(m.dof_damping[dof[n]]) for n in names},
        live,
        "mj_model.dof_damping",
    )
    c.set("model.effort_limit", {n: lim[n] for n in names}, live, "actuator forcerange")
    c.set("model.velocity_limit", None, "unknown", "not enforced by MuJoCo position actuators")
    c.set(
        "model.training_envelope",
        _training_envelope(train_cfg),
        live,
        "training config events and noise",
    )
    c.set("evidence", {"level": "L0", "golden": None})
    return c


def _training_envelope(train_cfg: Any) -> dict[str, Any]:
    ev = train_cfg.events
    out: dict[str, Any] = {}
    for k in ("push_robot", "foot_friction", "encoder_bias", "base_com", "reset_base"):
        if k in ev:
            p = dict(ev[k].params)
            p.pop("asset_cfg", None)
            out[k] = json.loads(json.dumps(p, default=str))
            if getattr(ev[k], "interval_range_s", None):
                out[k]["interval_range_s"] = list(ev[k].interval_range_s)
    obs = train_cfg.observations["actor"]
    out["obs_noise"] = (
        {k: _noise_desc(t.noise) for k, t in obs.terms.items()} if obs.enable_corruption else None
    )
    out["rel_standing_envs"] = float(train_cfg.commands["twist"].rel_standing_envs)
    return out


def _free_joint(m: Any) -> int:
    for j in range(m.njnt):
        if m.jnt_type[j] == 0:
            return j
    raise RuntimeError("no free joint")


def _prov(source: str, detail: str):
    from ..contract import Provenance

    return Provenance(source, detail)


def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def record(args: argparse.Namespace) -> Path:
    if args.task_path:
        sys.path.insert(0, args.task_path)
    importlib.import_module(args.task_module)
    import mujoco
    import onnxruntime as ort
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    onnx_path = Path(args.onnx).resolve()
    train_cfg = load_env_cfg(args.task, play=False)
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = args.episode_s
    changes = {}
    # Excitation rules (plan 7.6): one env, fixed seed, noise and pushes off,
    # commands from the schedule. Encoder bias shifts the applied target, so it
    # is switched off and recorded as such.
    for name in ("push_robot", "encoder_bias"):
        if name in cfg.events:
            cfg.events.pop(name)
            changes[f"events.{name}"] = "removed"
    cfg.observations["actor"].enable_corruption = False
    tw = cfg.commands["twist"]
    play_ranges = _ranges(tw.ranges)
    tw.heading_command = False
    tw.ranges.heading = None
    tw.rel_standing_envs = 0.0
    tw.resampling_time_range = (1.0e9, 1.0e9)
    tw.ranges.lin_vel_x = (0.0, 0.0)
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.debug_vis = False
    changes["commands.twist"] = (
        "set from the schedule every step; heading control, standing envs and resampling off"
    )
    changes["episode_length_s"] = args.episode_s

    t0 = time.time()
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    m = env.sim.mj_model
    d = env.sim.data
    act_term = env.action_manager.get_term("joint_pos")
    policy_names = [_strip(n) for n in act_term.target_names]
    jnt_names = [
        _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
        for j in range(m.njnt)
        if m.jnt_type[j] != 0
    ]
    if [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] != 0] != list(range(7, m.nq)):
        raise RuntimeError("expected one free joint followed by hinge joints")
    # Actuator columns to joint (qpos) order.
    act_joint = [
        _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[a, 0])))
        for a in range(m.nu)
    ]
    a_of_joint = np.array([act_joint.index(n) for n in jnt_names])
    contact = (
        env.scene["feet_ground_contact"] if "feet_ground_contact" in env.scene.sensors else None
    )

    P: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("qpos", "qvel", "ctrl", "effort", "qacc", "step", "substep", "contact", "time")
    }
    state = {"k": 0, "s": 0}
    orig_step = env.sim.step
    pushes = [] if args.no_pushes else DEFAULT_PUSHES
    push_body = {
        b: next(
            i
            for i in range(m.nbody)
            if _strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "") == b
        )
        for b in {p[2] for p in pushes}
    }
    P["xfrc"] = []

    def step_hook() -> None:
        # Scheduled pushes: a world-frame force on a body, set before each physics step.
        t_now = state["k"] * env.step_dt + state["s"] * float(m.opt.timestep)
        push = push_at(pushes, t_now)
        d.xfrc_applied[0].zero_()
        if push is not None:
            d.xfrc_applied[0, push_body[push[0]], :3] = torch.as_tensor(
                push[1], dtype=d.xfrc_applied.dtype
            )
        P["xfrc"].append(np.array(push[1] if push is not None else np.zeros(3)))
        P["qpos"].append(_np(d.qpos))
        P["qvel"].append(_np(d.qvel))
        P["ctrl"].append(_np(d.ctrl)[a_of_joint])
        P["time"].append(_np(d.time).reshape(-1)[:1] if d.time.ndim else np.array([float(d.time)]))
        orig_step()
        P["effort"].append(_np(d.actuator_force)[a_of_joint])
        P["qacc"].append(_np(d.qacc))
        P["step"].append(np.array(state["k"]))
        P["substep"].append(np.array(state["s"]))
        if contact is not None:
            found = contact.data.found
            P["contact"].append(_np(found).reshape(-1) if found is not None else np.zeros(2))
        state["s"] += 1

    env.sim.step = step_hook

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    obs, _ = env.reset(seed=args.seed)
    n = int(round(args.seconds / env.step_dt))
    C: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "obs",
            "action",
            "action_applied",
            "target",
            "command",
            "qpos",
            "qvel",
            "reset",
            "episode_step",
            "gyro_sensor",
        )
    }
    terms_log: dict[str, list[np.ndarray]] = {}
    reset_flag = True
    t_sched = 0.0
    tstep = []
    for k in range(n):
        o = obs["actor"].detach().cpu().numpy().astype(np.float32)
        C["obs"].append(o[0])
        C["qpos"].append(_np(d.qpos))
        C["qvel"].append(_np(d.qvel))
        C["command"].append(_np(env.command_manager.get_command("twist")))
        C["episode_step"].append(np.array(int(env.episode_length_buf[0])))
        C["reset"].append(np.array(reset_flag))
        C["gyro_sensor"].append(_np(env.scene["robot/imu_ang_vel"].data))
        for name, vals in om_terms(env):
            terms_log.setdefault(name, []).append(np.asarray(vals, dtype=np.float32))
        a = sess.run(None, {in_name: o})[0]
        C["action"].append(a[0])
        # The command for the next observation, from the schedule.
        t_sched = (k + 1) * env.step_dt
        cmd = torch.tensor(command_at(DEFAULT_SCHEDULE, t_sched), dtype=torch.float32)
        env.command_manager.get_term("twist").vel_command_b[0] = cmd
        state["k"], state["s"] = k, 0
        s0 = time.time()
        obs, _, term, trunc, _ = env.step(torch.from_numpy(a))
        tstep.append(time.time() - s0)
        C["action_applied"].append(_np(env.action_manager.action))
        C["target"].append(_np(act_term._processed_actions))
        reset_flag = bool(term[0]) or bool(trunc[0])
    wall = time.time() - t0

    arrays: dict[str, np.ndarray] = {k: np.stack(v) for k, v in C.items()}
    arrays["reset"] = arrays["reset"].astype(bool)
    arrays["obs"] = arrays["obs"].astype(np.float32)
    for name, v in terms_log.items():
        arrays[f"obs_terms/{name}"] = np.stack(v)
    for k, v in P.items():
        if v:
            arrays[f"p/{k}"] = np.stack(v)
    # Physics-rate state continuity: rows s, s+1 are consecutive unless a reset
    # happened between their control steps.
    pstep = arrays["p/step"]
    reset_next = np.zeros(len(pstep), bool)
    resets = np.where(arrays["reset"])[0]
    for r in resets:
        if r > 0:
            reset_next |= pstep == (r - 1)
    contiguous = ~(reset_next[:-1] & (pstep[1:] != pstep[:-1]))
    sc = check_state_increments(
        arrays["p/qpos"], arrays["p/qvel"], float(m.opt.timestep), contiguous
    )
    # Every physics step recorded: the observation state equals the state after the last substep.
    gap = 0.0
    for k in range(1, n):
        if not arrays["reset"][k]:
            nxt = np.where(pstep == k)[0][0]
            gap = max(gap, float(np.abs(arrays["p/qpos"][nxt] - arrays["qpos"][k]).max()))

    contract = live_contract(env, train_cfg, play_ranges, onnx_path)
    lim = np.array([contract.get("model.effort_limit")[n] or np.inf for n in jnt_names])
    at_lim = float(np.mean(np.abs(arrays["p/effort"]) >= lim[None] - 1e-4))
    layout = StateLayout(joint_names=jnt_names)
    exc = excitation_checklist(
        arrays["command"], arrays["qpos"][:, 3:7], arrays["qvel"][:, 3:6], arrays["reset"], at_lim
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dr = {}
    for fld in ("geom_friction", "body_ipos", "body_mass"):
        try:
            v = getattr(env.sim.model, fld)
            dr[fld] = _np(v).tolist() if v.ndim == m.__getattribute__(fld).ndim + 1 else None
        except Exception:
            dr[fld] = None
    # The compiled model is saved as simulated: startup randomization written in.
    from ..models import apply_randomization, model_patch

    apply_randomization(m, dr)
    mujoco.mj_saveModel(m, str(out / "model.mjb"))
    try:
        env.scene.write(out / "model_xml")
        xml_note = "model_xml/scene.xml written from the scene spec (pre-randomization)"
        xml_model = mujoco.MjModel.from_xml_path(str(out / "model_xml" / "scene.xml"))
        (out / "model_patch.json").write_text(json.dumps(model_patch(m, xml_model)))
    except Exception as e:  # pragma: no cover - depends on the mjlab version
        xml_note = f"XML export or patch failed: {e}"
    meta = {
        "framework": {
            "name": "mjlab",
            "version": _version("mjlab"),
            "mujoco": _version("mujoco"),
            "mujoco_warp": _version("mujoco-warp"),
            "warp": _version("warp-lang"),
            "torch": _version("torch"),
            "onnxruntime": ort.__version__,
        },
        "python": platform.python_version(),
        "task": args.task,
        "device": "cpu",
        "seed": args.seed,
        "policy": {"path": onnx_path.name, "sha256": sha256_file(onnx_path)},
        "model": {
            "mjb": "model.mjb",
            "mjb_sha256": sha256_file(out / "model.mjb"),
            "xml": xml_note,
            "randomized_fields": "dr.json",
        },
        "sim_dt": float(m.opt.timestep),
        "decimation": int(env.cfg.decimation),
        "policy_dt": float(env.step_dt),
        "quat_order": "wxyz",
        "state_layout": layout.to_meta(),
        "obs_group": "actor",
        "policy_joint_names": policy_names,
        "target_joint_names": jnt_names,
        "effort_joint_names": jnt_names,
        "target_meaning": "position target written to the actuators (processed action minus encoder bias)",
        "effort_meaning": "actuator_force after each mj_step: kp (ctrl - q) - kv qd at the pre-step state, clamped",
        "physics_rows": "state before each mj_step; effort and qacc from that step",
        "recording_changes": changes,
        "schedule": DEFAULT_SCHEDULE,
        "pushes": [list(p[:3]) + [list(p[3])] for p in pushes],
        "state_check": sc.__dict__,
        "obs_state_matches_last_substep_max_abs": gap,
        "excitation": [e.__dict__ for e in exc],
        "under_excited": [e.name for e in exc if not e.ok],
        "written_by_sim2sim_runner": False,
        "wall_time_s": wall,
        "mean_env_step_s": float(np.mean(tstep)),
    }
    trace = Trace(arrays, meta, "golden")
    trace.save(out)
    contract.save(out / "contract.live.yaml")
    (out / "dr.json").write_text(json.dumps(dr))
    print(
        json.dumps(
            {
                "out": str(out),
                "steps": n,
                "wall_s": round(wall, 1),
                "mean_env_step_s": round(float(np.mean(tstep)), 4),
                "state_check": sc.__dict__,
                "obs_state_gap": gap,
                "under_excited": meta["under_excited"],
                "resets": int(arrays["reset"].sum()),
            },
            indent=1,
        )
    )
    return out


def om_terms(env: Any) -> list[tuple[str, Any]]:
    out = []
    for name, vals in env.observation_manager.get_active_iterable_terms(0):
        if name.startswith("actor-"):
            out.append((name.split("-", 1)[1], vals))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="Unitree-G1-Flat")
    ap.add_argument(
        "--task-path", default=None, help="directory to put on sys.path for task registration"
    )
    ap.add_argument("--task-module", default="src.tasks")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=36.0)
    ap.add_argument("--episode-s", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pushes", action="store_true", help="leave out the scheduled pushes")
    record(ap.parse_args(argv))


if __name__ == "__main__":
    main()
