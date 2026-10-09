"""Golden traces and the small-command check in Isaac Lab, for unitree_rl_lab's G1 velocity policy.

Runs inside an Isaac Sim 5.1 / Isaac Lab 2.3 Python on a GPU (docs/GPU_SESSION.md). It drives
the training environment with the shipped ``policy.onnx``: the policy folder has no
checkpoint, so ``play.py`` cannot be used.

    # which gains were trained, and whether the dead zone exists on PhysX
    python -m gaitkeeper.recorders.isaaclab check --onnx policy.onnx --deploy deploy.yaml \\
        --usd <unitree_model>/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd --out runs/isaac_check.json

    # a golden trace in the same schema as the mjlab recorder
    python -m gaitkeeper.recorders.isaaclab record --onnx policy.onnx --deploy deploy.yaml \\
        --usd <...>.usd --out runs/isaac_golden_a

Every physics step is recorded by wrapping the environment's ``sim.step``: the state before
the step, the joint position targets written to the drives, and Isaac Lab's torque estimate
for the implicit actuators (an estimate, never a reference; PhysX does not report the drive
torque). Observation noise, the interval pushes and the startup randomization are switched
off and listed in the trace's ``recording_changes``.

The helpers above ``launch`` import no Isaac module, so they are tested on the CPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import SCHEMA, Contract, Provenance, sha256_file
from ..terms import StateLayout
from ..trace import Trace, check_state_increments
from .schedule import (
    RL_LAB_EPISODE_S,
    RL_LAB_PUSHES,
    RL_LAB_SCHEDULE,
    command_at,
    excitation_checklist,
    push_at,
)

TASK = "Unitree-G1-29dof-Velocity"
ACTION_TERM = "JointPositionAction"
COMMAND_TERM = "base_velocity"
OBS_GROUP = "policy"
TERM_IDS = {
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "generated_commands": "velocity_commands",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_rel": "joint_vel_rel",
    "last_action": "last_action",
}
# The 14 arm joints (shoulders, elbows, wrists): `--hold arms` holds them at the default pose,
# as the MuJoCo harness in unitree_rl_lab issue 145 does.
ARM_JOINTS = [
    f"{side}_{j}_joint"
    for side in ("left", "right")
    for j in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    )
]
# The check: two questions, each with a control on either side.
CHECK_COMMANDS: list[tuple[str, tuple[float, float, float]]] = [
    ("stand", (0.0, 0.0, 0.0)),
    ("fwd 0.15 (MuJoCo: stands)", (0.15, 0.0, 0.0)),
    ("fwd 0.25 (MuJoCo: walks 0.22)", (0.25, 0.0, 0.0)),
    ("fwd 0.50 (MuJoCo: walks 0.48)", (0.5, 0.0, 0.0)),
    ("yaw 0.2 in place (MuJoCo: 0.02)", (0.0, 0.0, 0.2)),
    ("fwd 0.5 + yaw 0.2 (MuJoCo: 0.13)", (0.5, 0.0, 0.2)),
]


# -- helpers that need no Isaac module ------------------------------------------------------------


def assemble_state(
    root_pos_w: np.ndarray,
    root_quat_wxyz: np.ndarray,
    root_lin_vel_w: np.ndarray,
    root_ang_vel_b: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo free-joint convention: qpos = (position, quaternion wxyz, joints); qvel = (linear
    velocity of the root link origin in the world frame, angular velocity in the body frame, joints)."""
    q = np.asarray(root_quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    qpos = np.concatenate(
        [np.asarray(root_pos_w, np.float64), q, np.asarray(joint_pos, np.float64)]
    )
    qvel = np.concatenate(
        [
            np.asarray(root_lin_vel_w, np.float64),
            np.asarray(root_ang_vel_b, np.float64),
            np.asarray(joint_vel, np.float64),
        ]
    )
    return qpos, qvel


def deploy_gains(deploy_yaml: str | Path, onnx: str | Path | None = None) -> tuple[dict, dict]:
    """kp and kd by joint name from a Unitree deploy.yaml (listed there in SDK motor order)."""
    from ..readers.unitree_deploy import read_unitree_deploy

    c, _ = read_unitree_deploy(deploy_yaml, onnx)
    return dict(c.get("control.actuators.kp")), dict(c.get("control.actuators.kd"))


def punches(
    seed: int, first: float, every: float, until: float, force: float, dur: float, body: str
) -> list[tuple[float, float, str, tuple[float, float, float]]]:
    """Horizontal pushes of fixed size in a seeded random direction, in DEFAULT_PUSHES format."""
    rng = np.random.default_rng(seed)
    out = []
    t = first
    while t < until - 1e-9:
        a = rng.uniform(0.0, 2.0 * np.pi)
        out.append((t, dur, body, (force * np.cos(a), force * np.sin(a), 0.0)))
        t += every
    return out


def summarize(
    vel_b: np.ndarray, actions: np.ndarray, joint_vel: np.ndarray, fell: np.ndarray, tail: int
) -> dict[str, Any]:
    """Per-scenario numbers over the last ``tail`` policy steps, for envs that did not fall.

    vel_b: (T, E, 3) body vx, vy, yaw rate; actions: (T, E, A); joint_vel: (T, E, J); fell: (E,).
    """
    ok = ~fell
    out: dict[str, Any] = {"envs": int(len(fell)), "fell": int(fell.sum())}
    if not ok.any():
        return out
    v = vel_b[-tail:][:, ok]
    out["vx"] = float(v[..., 0].mean())
    out["vy"] = float(v[..., 1].mean())
    out["wz"] = float(v[..., 2].mean())
    out["vx_spread"] = float(v[..., 0].mean(0).std())
    out["action_std"] = float(actions[-tail:][:, ok].std(0).mean())
    out["mean_abs_joint_vel"] = float(np.abs(joint_vel[-tail:][:, ok]).mean())
    out["standstill"] = bool(out["action_std"] < 0.02 and out["mean_abs_joint_vel"] < 0.01)
    return out


def verdict_lines(results: dict[str, Any]) -> list[str]:
    """What the check settles, in words, from the summary of both gain sets."""
    lines = []
    for gains, rows in results.items():
        r = {name: s for name, s in rows}
        f15 = r.get("fwd 0.15 (MuJoCo: stands)", {})
        yaw = r.get("yaw 0.2 in place (MuJoCo: 0.02)", {})
        walk = r.get("fwd 0.50 (MuJoCo: walks 0.48)", {})
        if walk.get("fell", 0) or "vx" not in walk:
            lines.append(
                f"{gains} gains: does not walk at 0.5 m/s ({walk.get('fell')} of {walk.get('envs')} fell)"
            )
            continue
        lines.append(f"{gains} gains: walks at 0.5 m/s ({walk['vx']:.2f} m/s)")
        if "vx" in f15:
            stands = f15["standstill"] and abs(f15["vx"]) < 0.03
            what = (
                "stands still, as in MuJoCo"
                if stands
                else f"moves at {f15['vx']:.2f} m/s, unlike MuJoCo"
            )
            lines.append(f"  0.15 m/s forward: {what}")
        if "wz" in yaw:
            lines.append(f"  0.2 rad/s yaw in place: {yaw['wz']:.3f} rad/s (MuJoCo 0.02)")
    return lines


# -- Isaac Lab -------------------------------------------------------------------------------------


def launch(headless: bool = True) -> Any:
    """Start Isaac Sim. Must run before any isaaclab import."""
    from isaaclab.app import AppLauncher

    return AppLauncher(headless=headless).app


def _np(x: Any) -> np.ndarray:
    return x.detach().cpu().numpy().copy()


def _version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return None


def make_env(args: argparse.Namespace, num_envs: int) -> tuple[Any, dict[str, Any], Any]:
    """The play environment with recording changes applied. Returns (env, changes, train_cfg)."""
    import unitree_rl_lab.tasks  # noqa: F401  (registers the gym ids)
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    train_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg = load_cfg_from_registry(args.task, "play_env_cfg_entry_point")
    changes: dict[str, Any] = {}
    cfg.scene.num_envs = num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    if args.usd:
        if not hasattr(cfg.scene.robot.spawn, "usd_path"):
            raise SystemExit(
                f"the robot spawns from {type(cfg.scene.robot.spawn).__name__}, not a USD; "
                "--usd cannot replace it"
            )
        cfg.scene.robot.spawn.usd_path = str(Path(args.usd).resolve())
        changes["scene.robot.spawn.usd_path"] = cfg.scene.robot.spawn.usd_path
    if args.sim_dt:
        dec = int(round(0.02 / args.sim_dt))
        if abs(dec * args.sim_dt - 0.02) > 1e-9:
            raise SystemExit(f"--sim-dt {args.sim_dt} does not divide the 0.02 s policy step")
        cfg.sim.dt = args.sim_dt
        cfg.decimation = dec
        cfg.sim.render_interval = dec
        changes["sim.dt"], changes["decimation"] = args.sim_dt, dec
    for name in ("push_robot", "physics_material", "add_base_mass", "base_external_force_torque"):
        if getattr(cfg.events, name, None) is not None:
            setattr(cfg.events, name, None)
            changes[f"events.{name}"] = "removed"
    cfg.observations.policy.enable_corruption = False
    changes["observations.policy.enable_corruption"] = False
    for name in ("terrain_levels", "lin_vel_cmd_levels"):
        if getattr(cfg.curriculum, name, None) is not None:
            setattr(cfg.curriculum, name, None)
            changes[f"curriculum.{name}"] = "removed"
    bv = getattr(cfg.commands, COMMAND_TERM)
    bv.resampling_time_range = (1.0e9, 1.0e9)
    bv.rel_standing_envs = 0.0
    bv.heading_command = False
    bv.debug_vis = False
    changes[f"commands.{COMMAND_TERM}"] = (
        "set by the recorder every step; resampling and standing envs off"
    )
    cfg.episode_length_s = args.episode_s
    changes["episode_length_s"] = args.episode_s
    env = ManagerBasedRLEnv(cfg=cfg)
    return env, changes, train_cfg


def set_gains(env: Any, kp: dict[str, float], kd: dict[str, float]) -> None:
    """Write per-joint drive gains to PhysX and to the implicit actuators' own copies (which
    Isaac Lab uses for its torque estimate)."""
    import torch

    robot = env.scene["robot"]
    names = list(robot.joint_names)
    missing = [n for n in names if n not in kp or n not in kd]
    if missing:
        raise SystemExit(f"no gains for {missing}")
    stiff = torch.tensor([[kp[n] for n in names]], device=env.device).repeat(env.num_envs, 1)
    damp = torch.tensor([[kd[n] for n in names]], device=env.device).repeat(env.num_envs, 1)
    robot.write_joint_stiffness_to_sim(stiff)
    robot.write_joint_damping_to_sim(damp)
    for act in robot.actuators.values():
        idx = act.joint_indices
        idx = list(range(len(names))) if isinstance(idx, slice) else list(idx)
        act.stiffness[:] = stiff[:, idx]
        act.damping[:] = damp[:, idx]


def live_gains(env: Any) -> tuple[dict[str, float], dict[str, float]]:
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    return (
        dict(zip(names, _np(robot.data.joint_stiffness)[0].tolist())),
        dict(zip(names, _np(robot.data.joint_damping)[0].tolist())),
    )


def _policy(onnx: Path) -> Any:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def run(obs: np.ndarray) -> np.ndarray:
        # The shipped export has a fixed batch of 1.
        return np.concatenate([sess.run(None, {name: o[None].astype(np.float32)})[0] for o in obs])

    return run


def _set_command(env: Any, cmd: tuple[float, float, float]) -> None:
    import torch

    term = env.command_manager.get_term(COMMAND_TERM)
    term.vel_command_b[:] = torch.tensor(cmd, device=env.device, dtype=term.vel_command_b.dtype)


def doctor(args: argparse.Namespace) -> dict[str, Any]:
    """Smoke test of the stack: the app starts, the task builds with the given USD, ten steps
    run, and the numbers the recorder relies on are what it expects."""
    app = launch(not args.gui)
    import torch

    env, changes, _ = make_env(args, 2)
    robot = env.scene["robot"]
    act = env.action_manager.get_term(ACTION_TERM)
    obs, _ = env.reset(seed=args.seed)
    for _ in range(10):
        obs, *_ = env.step(torch.zeros(env.num_envs, act.action_dim, device=env.device))
    kinds = sorted({type(a).__name__ for a in robot.actuators.values()})
    om = env.observation_manager
    rep = {
        "framework": {k: _version(k) for k in ("isaaclab", "isaacsim", "torch", "onnxruntime")},
        "task": args.task,
        "usd": changes.get("scene.robot.spawn.usd_path"),
        "joints": len(robot.joint_names),
        "action_dim": int(act.action_dim),
        "obs_dim": int(obs[OBS_GROUP].shape[-1]),
        "obs_terms": list(om._group_obs_term_names[OBS_GROUP]),
        "sim_dt": float(env.physics_dt),
        "decimation": int(env.cfg.decimation),
        "actuators": kinds,
        "root_height": float(_np(robot.data.root_link_pos_w)[0, 2]),
    }
    want = {"joints": 29, "action_dim": 29, "obs_dim": 480}
    problems = [f"{k} {rep[k]}, expected {v}" for k, v in want.items() if rep[k] != v]
    if kinds != ["ImplicitActuator"]:
        problems.append(f"actuators {kinds}: the recorder expects implicit PD only")
    if args.onnx:
        import onnxruntime as ort

        n_in = ort.InferenceSession(args.onnx).get_inputs()[0].shape[-1]
        if n_in != rep["obs_dim"]:
            problems.append(f"policy.onnx takes {n_in} inputs, the env builds {rep['obs_dim']}")
    rep["problems"] = problems
    print(json.dumps(rep, indent=1))
    print("doctor: ok" if not problems else "doctor: PROBLEMS\n  " + "\n  ".join(problems))
    env.close()
    app.close()
    if problems:
        raise SystemExit(1)
    return rep


def check(args: argparse.Namespace) -> dict[str, Any]:
    app = launch(not args.gui)
    import torch

    run_policy = _policy(Path(args.onnx))
    gain_sets = {"asset config": None}
    if args.deploy:
        gain_sets["deploy.yaml"] = deploy_gains(args.deploy, args.onnx)
    env, changes, _ = make_env(args, args.envs)
    robot = env.scene["robot"]
    asset_gains = live_gains(env)
    n = int(round(args.seconds / env.step_dt))
    tail = int(round(5.0 / env.step_dt))
    results: dict[str, Any] = {}
    for gname, gains in gain_sets.items():
        set_gains(env, *(gains or asset_gains))
        rows = []
        for label, cmd in CHECK_COMMANDS:
            obs, _ = env.reset(seed=args.seed)
            vel, acts, qd = [], [], []
            fell = np.zeros(env.num_envs, bool)
            for _ in range(n):
                _set_command(env, cmd)
                a = run_policy(_np(obs[OBS_GROUP]))
                acts.append(a)
                obs, _, term, trunc, _ = env.step(torch.from_numpy(a).to(env.device))
                fell |= _np(term).astype(bool)
                v = np.concatenate(
                    [
                        _np(robot.data.root_link_lin_vel_b)[:, :2],
                        _np(robot.data.root_link_ang_vel_b)[:, 2:3],
                    ],
                    1,
                )
                vel.append(v)
                qd.append(_np(robot.data.joint_vel))
            s = summarize(np.stack(vel), np.stack(acts), np.stack(qd), fell, tail)
            s["command"] = list(cmd)
            rows.append((label, s))
            print(
                f"[{gname}] {label:34s} "
                + json.dumps(
                    {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
                ),
                flush=True,
            )
        results[gname] = rows
    out = {
        "task": args.task,
        "framework": {
            "isaaclab": _version("isaaclab"),
            "isaacsim": _version("isaacsim"),
            "torch": _version("torch"),
        },
        "policy_sha256": sha256_file(Path(args.onnx)),
        "envs": env.num_envs,
        "seconds": args.seconds,
        "sim_dt": float(env.physics_dt),
        "decimation": int(env.cfg.decimation),
        "changes": changes,
        "asset_gains": {"kp": asset_gains[0], "kd": asset_gains[1]},
        "results": {g: [{"scenario": lab, **s} for lab, s in rows] for g, rows in results.items()},
        "summary": verdict_lines(results),
    }
    print("\n".join(out["summary"]))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
        print(f"written {args.out}")
    env.close()
    app.close()
    return out


def live_contract(env: Any, train_cfg: Any, onnx: Path, gains_source: str) -> Contract:
    robot = env.scene["robot"]
    act = env.action_manager.get_term(ACTION_TERM)
    names = list(act._joint_names)
    jn = list(robot.joint_names)
    live = "live"
    c = Contract({"schema": SCHEMA})
    c.set(
        "source",
        {
            "format": "isaaclab_live",
            "framework": {"name": "isaaclab", "version": _version("isaaclab")},
            "files": [{"path": onnx.name, "sha256": sha256_file(onnx)}],
        },
        live,
    )
    c.set("timing.policy_dt", float(env.step_dt), live, "env.step_dt")
    c.set("timing.sim_dt", float(env.physics_dt), live, "env.physics_dt")
    c.set("timing.decimation", int(env.cfg.decimation), live, "env.cfg.decimation")
    c.set("timing.order", ["obs", "infer", "target", "pd", "step"], live, "ManagerBasedRLEnv.step")
    c.set(
        "timing.target_hold",
        "zoh",
        live,
        "apply_action on every substep with the same processed action",
    )
    c.set("policy_io.joints.names", names, live, f"action term {ACTION_TERM} joint names")

    om = env.observation_manager
    terms, hist = [], set()
    for k, (tname, tcfg) in enumerate(
        zip(om._group_obs_term_names[OBS_GROUP], om._group_obs_term_cfgs[OBS_GROUP])
    ):
        fname = tcfg.func.__name__
        h = int(tcfg.history_length or 0)
        dim_total = int(np.prod(om._group_obs_term_dim[OBS_GROUP][k]))
        dim = dim_total // max(h, 1)
        scale = tcfg.scale
        scale = (
            scale.tolist()
            if hasattr(scale, "tolist")
            else ([float(scale)] * dim if scale is not None else [1.0] * dim)
        )
        terms.append(
            {
                "id": TERM_IDS.get(fname, fname),
                "source_name": tname,
                "function": f"{tcfg.func.__module__}.{fname}",
                "dim": dim,
                "scale": scale if len(scale) == dim else [scale[0]] * dim,
                "clip": list(tcfg.clip) if tcfg.clip else None,
                "params": {
                    kk: (v if isinstance(v, (int, float, str, bool)) else str(v))
                    for kk, v in tcfg.params.items()
                },
                "noise": None,
            }
        )
        hist.add(h)
    c.set(
        "policy_io.observation_groups.policy.terms",
        terms,
        live,
        f"observation_manager group '{OBS_GROUP}'",
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
        "per-term CircularBuffer, flattened per term; filled with the first frame after a reset",
    )
    c.set(
        "policy_io.observation_groups.policy.clip_then_scale",
        True,
        live,
        "ObservationManager.compute_group",
    )
    c.set(
        "policy_io.imu",
        {"body": robot.body_names[0], "frame": "body"},
        live,
        "base_ang_vel uses root_ang_vel_b",
    )

    tc = getattr(train_cfg.commands, COMMAND_TERM)
    lim = tc.limit_ranges
    start = tc.ranges
    cur = getattr(train_cfg.curriculum, "lin_vel_cmd_levels", None)
    widen_x = widen_y = cur is not None
    ang_cur = getattr(train_cfg.curriculum, "ang_vel_cmd_levels", None)
    trained = {
        "vx": list(lim.lin_vel_x if widen_x else start.lin_vel_x),
        "vy": list(lim.lin_vel_y if widen_y else start.lin_vel_y),
        "wz": list(lim.ang_vel_z if ang_cur is not None else start.ang_vel_z),
    }
    c.set(
        "policy_io.commands.base_velocity",
        {
            "name": COMMAND_TERM,
            "limit": {
                "vx": list(lim.lin_vel_x),
                "vy": list(lim.lin_vel_y),
                "wz": list(lim.ang_vel_z),
            },
            "trained": trained,
            "heading": "on" if tc.heading_command else "off",
        },
    )
    c.provenance["policy_io.commands.base_velocity.limit"] = Provenance(
        live, "training config limit_ranges"
    )
    c.provenance["policy_io.commands.base_velocity.trained"] = Provenance(
        live,
        "start ranges, widened to the limits only on axes a curriculum term covers "
        f"(lin_vel_cmd_levels: {cur is not None}, ang_vel_cmd_levels: {ang_cur is not None}); "
        "reached only if training ran long enough",
    )

    scale = (
        _np(act._scale)[0]
        if hasattr(act._scale, "shape")
        else np.full(len(names), float(act._scale))
    )
    offset = (
        _np(act._offset)[0]
        if hasattr(act._offset, "shape")
        else np.full(len(names), float(act._offset))
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
    c.set("control.actions.joint_pos.clip", None, live, "no clip in the action term config")
    c.set("control.actions.joint_pos.clip_stage", "none", live, "")
    c.set("control.actions.joint_pos.delay_steps", 0, live, "")
    c.set("control.actions.joint_pos.filter", "none", live, "")
    c.set("control.ownership", {"owned": "all", "external": []}, live, "")
    c.set(
        "control.default_joint_pos",
        dict(zip(jn, _np(robot.data.default_joint_pos)[0].tolist())),
        live,
        "articulation default_joint_pos",
    )
    kinds = sorted({type(a).__name__ for a in robot.actuators.values()})
    if kinds != ["ImplicitActuator"]:
        raise RuntimeError(f"unexpected actuator classes {kinds}")
    c.set("control.actuators.kind", "implicit_pd", live, "ImplicitActuator: PD solved inside PhysX")
    c.set("control.actuators.pd_period", "solver", live, "inside every physics step")
    c.set(
        "control.actuators.integrator", "physx_tgs", live, "PhysX articulation drive (TGS solver)"
    )
    c.set(
        "control.actuators.torque_limit_at", "actuator_force", live, "joint effort limit in PhysX"
    )
    kp, kd = live_gains(env)
    c.set(
        "control.actuators.kp",
        {n: kp[n] for n in names},
        live,
        f"robot.data.joint_stiffness ({gains_source})",
    )
    c.set(
        "control.actuators.kd",
        {n: kd[n] for n in names},
        live,
        f"robot.data.joint_damping ({gains_source})",
    )

    d = robot.data

    def by_name(t: Any) -> dict[str, float]:
        return dict(zip(jn, _np(t)[0].tolist()))

    c.set("model.root", {"body": robot.body_names[0]}, live, "articulation root")
    c.set(
        "model.armature",
        {n: by_name(d.joint_armature)[n] for n in names},
        live,
        "robot.data.joint_armature",
    )
    c.set(
        "model.joint_friction",
        {n: by_name(d.joint_friction_coeff)[n] for n in names},
        live,
        "robot.data.joint_friction_coeff (PhysX joint friction coefficient)",
    )
    c.set(
        "model.joint_damping",
        {n: 0.0 for n in names},
        live,
        "no passive joint damping besides the drive",
    )
    c.set(
        "model.effort_limit",
        {n: by_name(d.joint_effort_limits)[n] for n in names},
        live,
        "robot.data.joint_effort_limits",
    )
    c.set(
        "model.velocity_limit",
        {n: by_name(d.joint_vel_limits)[n] for n in names},
        live,
        "robot.data.joint_vel_limits",
    )
    c.set("evidence", {"level": "L0", "golden": None})
    return c


def record(args: argparse.Namespace) -> Path:
    app = launch(not args.gui)
    import torch

    onnx = Path(args.onnx).resolve()
    run_policy = _policy(onnx)
    env, changes, train_cfg = make_env(args, 1)
    robot = env.scene["robot"]
    gains_source = "asset config"
    if args.gains == "deploy":
        set_gains(env, *deploy_gains(args.deploy, onnx))
        gains_source = "deploy.yaml"
        changes["gains"] = "deploy.yaml stiffness and damping written to the drives"
    if args.ankle_armature_delta:
        jn = list(robot.joint_names)
        ids = [i for i, nm in enumerate(jn) if "ankle" in nm]
        arm = robot.data.joint_armature.clone()
        arm[:, ids] += args.ankle_armature_delta
        robot.write_joint_armature_to_sim(arm)
        changes["ankle_armature_delta"] = args.ankle_armature_delta

    names = list(robot.joint_names)
    act = env.action_manager.get_term(ACTION_TERM)
    pnames = list(act._joint_names)
    hold = [n for n in (args.hold.split(",") if args.hold else []) if n]
    unknown = [n for n in hold if n not in pnames]
    if unknown:
        raise SystemExit(f"--hold: not policy joints: {unknown}")
    hold_idx = [pnames.index(n) for n in hold]
    if args.schedule:
        from ..runner import load_schedule

        sched = [(t, *c) for t, c in load_schedule(args.schedule)]
    else:
        sched = RL_LAB_SCHEDULE
    if args.punch_force:
        pushes = punches(
            args.seed,
            args.punch_first,
            args.punch_every,
            args.seconds,
            args.punch_force,
            args.punch_dur,
            args.punch_body,
        )
    elif args.no_pushes:
        pushes = []
    else:
        pushes = RL_LAB_PUSHES
    body_names = list(robot.body_names)
    push_body = {b: body_names.index(b) for b in {p[2] for p in pushes}}

    P: dict[str, list[np.ndarray]] = {
        k: []
        for k in ("qpos", "qvel", "ctrl", "effort", "step", "substep", "time", "xfrc", "xfrc_body")
    }
    state = {"k": 0, "s": 0}
    h = float(env.physics_dt)
    origin = _np(env.scene.env_origins)[0]
    orig_write = env.scene.write_data_to_sim
    orig_step = env.sim.step
    zeros = torch.zeros(1, 1, 3, device=env.device)

    def write_hook() -> None:
        t_now = state["k"] * env.step_dt + state["s"] * h
        push = push_at(pushes, t_now)
        if push is not None:
            f = torch.tensor(push[1], device=env.device, dtype=torch.float32).view(1, 1, 3)
            robot.set_external_force_and_torque(
                f, zeros, body_ids=[push_body[push[0]]], env_ids=[0], is_global=True
            )
        else:
            robot.set_external_force_and_torque(
                zeros, zeros, body_ids=[0], env_ids=[0], is_global=True
            )
        state["push"] = push
        orig_write()

    def step_hook(*a: Any, **kw: Any) -> Any:
        d = robot.data
        qpos, qvel = assemble_state(
            _np(d.root_link_pos_w)[0] - np.array([origin[0], origin[1], 0.0]),
            _np(d.root_link_quat_w)[0],
            _np(d.root_link_lin_vel_w)[0],
            _np(d.root_link_ang_vel_b)[0],
            _np(d.joint_pos)[0],
            _np(d.joint_vel)[0],
        )
        P["qpos"].append(qpos)
        P["qvel"].append(qvel)
        P["ctrl"].append(_np(d.joint_pos_target)[0])
        P["effort"].append(_np(d.applied_torque)[0])
        push = state.get("push")
        P["xfrc"].append(np.asarray(push[1]) if push is not None else np.zeros(3))
        P["xfrc_body"].append(np.array(push_body[push[0]] if push is not None else -1))
        P["step"].append(np.array(state["k"]))
        P["substep"].append(np.array(state["s"]))
        P["time"].append(np.array([state["k"] * env.step_dt + state["s"] * h]))
        out = orig_step(*a, **kw)
        state["s"] += 1
        return out

    env.scene.write_data_to_sim = write_hook
    env.sim.step = step_hook

    t0 = time.time()
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
        )
    }
    reset_flag = True
    _set_command(env, command_at(sched, 0.0))
    for k in range(n):
        d = robot.data
        qpos, qvel = assemble_state(
            _np(d.root_link_pos_w)[0] - np.array([origin[0], origin[1], 0.0]),
            _np(d.root_link_quat_w)[0],
            _np(d.root_link_lin_vel_w)[0],
            _np(d.root_link_ang_vel_b)[0],
            _np(d.joint_pos)[0],
            _np(d.joint_vel)[0],
        )
        o = _np(obs[OBS_GROUP])[0].astype(np.float32)
        C["obs"].append(o)
        C["qpos"].append(qpos)
        C["qvel"].append(qvel)
        C["command"].append(_np(env.command_manager.get_command(COMMAND_TERM))[0])
        C["episode_step"].append(np.array(int(env.episode_length_buf[0])))
        C["reset"].append(np.array(reset_flag))
        a = run_policy(o[None])
        if hold_idx:
            a[:, hold_idx] = (
                0.0  # target = default pose: held by the harness, with the policy's gains
            )
        C["action"].append(a[0])
        state["k"], state["s"] = k, 0
        _set_command(env, command_at(sched, (k + 1) * env.step_dt))
        obs, _, term, trunc, _ = env.step(torch.from_numpy(a).to(env.device))
        C["action_applied"].append(_np(env.action_manager.action)[0])
        C["target"].append(_np(act.processed_actions)[0])
        reset_flag = bool(term[0]) or bool(trunc[0])
    wall = time.time() - t0

    arrays: dict[str, np.ndarray] = {k: np.stack(v) for k, v in C.items()}
    arrays["reset"] = arrays["reset"].astype(bool)
    for k, v in P.items():
        arrays[f"p/{k}"] = np.stack(v)
    pstep = arrays["p/step"]
    contiguous = np.ones(len(pstep) - 1, bool)
    for r in np.where(arrays["reset"])[0]:
        if r > 0:
            contiguous &= ~((pstep[:-1] == r - 1) & (pstep[1:] == r))
    sc = check_state_increments(arrays["p/qpos"], arrays["p/qvel"], h, contiguous)
    gap = 0.0
    for k in range(1, n):
        if not arrays["reset"][k]:
            nxt = np.where(pstep == k)[0][0]
            gap = max(gap, float(np.abs(arrays["p/qpos"][nxt] - arrays["qpos"][k]).max()))

    contract = live_contract(env, train_cfg, onnx, gains_source)
    lim = np.array([contract.get("model.effort_limit").get(nm, np.inf) for nm in names])
    at_lim = float(np.mean(np.abs(arrays["p/effort"]) >= lim[None] - 1e-4))
    exc = excitation_checklist(
        arrays["command"], arrays["qpos"][:, 3:7], arrays["qvel"][:, 3:6], arrays["reset"], at_lim
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model_live = {
        "joint_names": names,
        "body_names": body_names,
        "armature": _np(robot.data.joint_armature)[0].tolist(),
        "joint_friction_coeff": _np(robot.data.joint_friction_coeff)[0].tolist(),
        "stiffness": _np(robot.data.joint_stiffness)[0].tolist(),
        "damping": _np(robot.data.joint_damping)[0].tolist(),
        "effort_limits": _np(robot.data.joint_effort_limits)[0].tolist(),
        "vel_limits": _np(robot.data.joint_vel_limits)[0].tolist(),
        "pos_limits": _np(robot.data.joint_pos_limits)[0].tolist(),
        "masses": _np(robot.root_physx_view.get_masses())[0].tolist(),
        "coms": _np(robot.root_physx_view.get_coms())[0].tolist(),
        "inertias": _np(robot.root_physx_view.get_inertias())[0].tolist(),
        "material_properties": _np(robot.root_physx_view.get_material_properties())[0].tolist(),
        "physx": {
            k: getattr(env.cfg.sim.physx, k)
            for k in (
                "solver_type",
                "min_position_iteration_count",
                "max_position_iteration_count",
                "min_velocity_iteration_count",
                "max_velocity_iteration_count",
                "enable_stabilization",
                "bounce_threshold_velocity",
            )
            if hasattr(env.cfg.sim.physx, k)
        },
        "articulation_solver_iterations": [
            getattr(env.cfg.scene.robot.spawn.articulation_props, k, None)
            for k in ("solver_position_iteration_count", "solver_velocity_iteration_count")
        ],
    }
    (out / "isaac_model.json").write_text(json.dumps(model_live, default=str))
    meta = {
        "framework": {
            "name": "isaaclab",
            "version": _version("isaaclab"),
            "isaacsim": _version("isaacsim"),
            "torch": _version("torch"),
            "onnxruntime": _version("onnxruntime"),
        },
        "engine": f"isaaclab {_version('isaaclab')} / physx (isaacsim {_version('isaacsim')})",
        "python": platform.python_version(),
        "task": args.task,
        "device": args.device,
        "seed": args.seed,
        "policy": {"path": onnx.name, "sha256": sha256_file(onnx)},
        "model": {
            "source": "usd",
            "usd": str(args.usd) if args.usd else None,
            "usd_sha256": sha256_file(Path(args.usd)) if args.usd else None,
            "live": "isaac_model.json",
            "note": "no MuJoCo model: analyse against a target MJCF",
        },
        "sim_dt": h,
        "decimation": int(env.cfg.decimation),
        "policy_dt": float(env.step_dt),
        "quat_order": "wxyz",
        "state_layout": StateLayout(joint_names=names).to_meta(),
        "obs_group": OBS_GROUP,
        "policy_joint_names": pnames,
        "target_joint_names": names,
        "effort_joint_names": names,
        "body_names": body_names,
        "target_meaning": "robot.data.joint_pos_target written to the PhysX drives before each physics step",
        "effort_meaning": "robot.data.applied_torque: Isaac Lab's estimate for implicit actuators, never a reference",
        "physics_rows": "state before each physics step (root link pose, root link velocity: linear world, angular body)",
        "recording_changes": changes,
        "gains": gains_source,
        "hold": hold,
        "schedule": [list(r) for r in sched],
        "pushes": [list(p[:3]) + [list(p[3])] for p in pushes],
        "state_check": sc.__dict__,
        "obs_state_matches_last_substep_max_abs": gap,
        "excitation": [e.__dict__ for e in exc],
        "under_excited": [e.name for e in exc if not e.ok],
        "written_by_gaitkeeper_runner": False,
        "wall_time_s": wall,
    }
    Trace(arrays, meta, "golden").save(out)
    contract.save(out / "contract.live.yaml")
    print(
        json.dumps(
            {
                "out": str(out),
                "steps": n,
                "wall_s": round(wall, 1),
                "state_check": sc.__dict__,
                "obs_state_gap": gap,
                "under_excited": meta["under_excited"],
                "resets": int(arrays["reset"].sum()),
            },
            indent=1,
        )
    )
    env.close()
    app.close()
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task", default=TASK)
        p.add_argument("--onnx", required=p.prog.split()[-1] != "doctor")
        p.add_argument("--deploy", help="Unitree deploy.yaml next to the policy (its gains)")
        p.add_argument(
            "--usd", help="G1 29 dof USD from unitree_model (replaces the asset's placeholder path)"
        )
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument(
            "--sim-dt",
            type=float,
            default=None,
            help="physics step; decimation follows from 0.02 s",
        )
        p.add_argument(
            "--episode-s",
            type=float,
            default=None,
            help="episode length (default: no timeout for check; 24 s for the golden schedule; "
            "longer than the run for --schedule)",
        )
        p.add_argument("--gui", action="store_true")

    d = sub.add_parser("doctor", help="smoke test: build the task, step it, check sizes")
    common(d)
    d.set_defaults(fn=doctor)

    c = sub.add_parser("check", help="small commands under both gain sets")
    common(c)
    c.add_argument("--envs", type=int, default=16)
    c.add_argument("--seconds", type=float, default=20.0)
    c.add_argument("--out", default=None)
    c.set_defaults(fn=check)

    r = sub.add_parser("record", help="a golden trace")
    common(r)
    r.add_argument("--out", required=True)
    r.add_argument("--seconds", type=float, default=36.0)
    r.add_argument("--gains", choices=["asset", "deploy"], default="asset")
    r.add_argument("--schedule", help="command schedule YAML (default: the rl_lab golden schedule)")
    r.add_argument(
        "--hold", help="joints held at the default pose: comma-separated names, or 'arms'"
    )
    r.add_argument("--no-pushes", action="store_true")
    r.add_argument("--punch-force", type=float, default=0.0)
    r.add_argument("--punch-first", type=float, default=27.0)
    r.add_argument("--punch-every", type=float, default=3.0)
    r.add_argument("--punch-dur", type=float, default=0.1)
    r.add_argument("--punch-body", default="torso_link")
    r.add_argument(
        "--ankle-armature-delta",
        type=float,
        default=0.0,
        help="E4: ankle armature change in the source",
    )
    r.set_defaults(fn=record)
    args = ap.parse_args(argv)
    if args.cmd == "record" and args.gains == "deploy" and not args.deploy:
        ap.error("--gains deploy needs --deploy")
    if args.episode_s is None:
        if args.cmd in ("check", "doctor"):
            args.episode_s = 1.0e6
        else:
            args.episode_s = args.seconds + 1.0 if args.schedule else RL_LAB_EPISODE_S
    if args.cmd == "record" and args.hold:
        args.hold = ",".join(ARM_JOINTS) if args.hold == "arms" else args.hold
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
