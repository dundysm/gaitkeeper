"""Contract from the env.yaml Isaac Lab writes next to a training run (params/env.yaml).

Isaac Lab dumps the whole ManagerBasedRLEnv config when a run starts. The policy's
training facts are all in it, as patterns over joint names rather than per joint:

* ``sim.dt`` and ``decimation``;
* ``scene.robot.init_state.joint_pos`` (the default pose, regex keys) and
  ``scene.robot.actuators`` (groups by ``joint_names_expr``, each with stiffness, damping,
  armature and limits, a number or regex keys);
* ``observations.policy``: the terms in order, each a function, scale, clip and history,
  the group's history length; Isaac Lab clips, then scales, and keeps a per-term buffer
  filled with the first frame after a reset;
* ``actions.JointPositionAction``: scale, offset or the default pose, clip;
* ``commands.base_velocity``: the start ranges, the limit ranges, and whether a curriculum
  term widens the first toward the second.

What the file does not hold is the articulation's joint order, which Isaac Lab takes from
the USD (breadth first). The reader takes it from a ``joint_ids_map`` (a deploy.yaml) or a
list, and for the 29 dof G1 assumes unitree_rl_lab's order when given neither, flagged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from ..contract import SCHEMA, Contract, sha256_file
from ..tables import G1_29_ISAAC, SDK_TABLES

TERM_IDS = {
    "base_ang_vel": "base_ang_vel",
    "base_lin_vel": "base_lin_vel",
    "projected_gravity": "projected_gravity",
    "generated_commands": "velocity_commands",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_rel": "joint_vel_rel",
    "last_action": "last_action",
    "gait_phase": "gait_phase",
    "gait_phase_legs": "gait_phase_legs",
}
GROUP_KEYS = {
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
}
MANAGER = "isaaclab ObservationManager.compute_group (noise, clip, scale per term)"


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda ld, n: ld.construct_sequence(n))
_Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda ld, suffix, n: None)
_Loader.add_multi_constructor("!", lambda ld, suffix, n: None)


def load_env_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.load(Path(path).read_text(), Loader=_Loader)


def resolve(spec: Any, names: list[str], what: str, default: Any = None) -> dict[str, Any]:
    """A number, or {regex: value} matched in full against joint names (one key per joint,
    as Isaac Lab's resolve_matching_names_values requires)."""
    if not isinstance(spec, dict):
        return {n: spec if spec is not None else default for n in names}
    out: dict[str, Any] = {}
    for n in names:
        hits = [v for k, v in spec.items() if re.fullmatch(k, n)]
        if len(hits) > 1:
            raise ValueError(f"{what}: {n} matches {len(hits)} keys")
        out[n] = hits[0] if hits else default
    return out


def read_isaaclab_env(
    env_yaml: str | Path,
    joint_order: list[str] | str | Path | None = None,
    robot: str = "unitree_g1_29dof",
    onnx_path: str | Path | None = None,
) -> tuple[Contract, list[str]]:
    """Returns (contract, findings). ``joint_order``: joint names, or a deploy.yaml whose
    joint_ids_map gives them over the robot's SDK table."""
    env_yaml = Path(env_yaml)
    y = load_env_yaml(env_yaml)
    findings: list[str] = []
    table_name, table = SDK_TABLES[robot]
    if "robot" not in (y.get("scene") or {}) or "policy" not in (y.get("observations") or {}):
        hint = (
            " (its scene has entities: an mjlab config)"
            if "entities" in (y.get("scene") or {})
            else ""
        )
        raise ValueError(f"{env_yaml}: not an Isaac Lab manager-based env.yaml{hint}")
    robot_cfg = y["scene"]["robot"]

    if isinstance(joint_order, (str, Path)):
        d = yaml.safe_load(Path(joint_order).read_text())
        names = [table[i] for i in d["joint_ids_map"]]
        order_from = f"joint_ids_map of {Path(joint_order).name}"
    elif joint_order:
        names = list(joint_order)
        order_from = "given"
    else:
        names = list(G1_29_ISAAC)
        order_from = "assumed: unitree_rl_lab's G1 29 dof articulation order (breadth first)"
        findings.append("joint order assumed (Isaac Lab's G1 order); give a deploy.yaml to read it")

    # actuators
    kp: dict[str, float] = {}
    kd: dict[str, float] = {}
    arm: dict[str, float] = {}
    eff: dict[str, float] = {}
    vel: dict[str, float] = {}
    kinds = set()
    for gname, g in (robot_cfg.get("actuators") or {}).items():
        exprs = g.get("joint_names_expr") or []
        members = [n for n in names if any(re.fullmatch(e, n) for e in exprs)]
        if not members:
            continue
        kinds.add(str(g.get("class_type", "")).rsplit(":", 1)[-1])
        for field, out in (
            ("stiffness", kp),
            ("damping", kd),
            ("armature", arm),
            ("effort_limit_sim", eff),
            ("velocity_limit_sim", vel),
        ):
            for n, v in resolve(g.get(field), members, f"{gname}.{field}").items():
                if v is not None:
                    out[n] = float(v)
    missing = [n for n in names if n not in kp]
    if missing:
        raise ValueError(f"no actuator group gives stiffness for {missing}")
    implicit = kinds == {"ImplicitActuator"}
    default = {
        n: float(v or 0.0)
        for n, v in resolve(
            robot_cfg["init_state"].get("joint_pos"), names, "init_state.joint_pos", 0.0
        ).items()
    }

    sim_dt = float(y["sim"]["dt"])
    dec = int(y["decimation"])
    c = Contract({"schema": SCHEMA})
    files = [{"path": env_yaml.name, "sha256": sha256_file(env_yaml)}]
    if onnx_path:
        files.append({"path": Path(onnx_path).name, "sha256": sha256_file(onnx_path)})
    c.set(
        "source",
        {
            "format": "isaaclab_env_yaml",
            "framework": {"name": "isaaclab", "version": None},
            "files": files,
            "robot": robot,
        },
        "file",
    )
    c.set("timing.policy_dt", sim_dt * dec, "file", "sim.dt * decimation")
    c.set("timing.sim_dt", sim_dt, "file", "sim.dt")
    c.set("timing.decimation", dec, "file", "decimation")
    c.set(
        "timing.order",
        ["obs", "infer", "target", "pd", "step"],
        "default",
        "ManagerBasedRLEnv.step",
    )
    c.set("timing.target_hold", "zoh", "default", "the processed action applied on every substep")
    c.set(
        "policy_io.joints",
        {"names": names, "table": table_name, "joint_ids_map": [table.index(n) for n in names]},
        "file+table",
        order_from,
    )
    c.set(
        "policy_io.imu",
        {"body": "pelvis", "frame": "body", "rotation_in_root": [1.0, 0.0, 0.0, 0.0]},
        "default",
        "root_ang_vel_b",
    )

    # observations
    grp = y["observations"]["policy"]
    gh = int(grp.get("history_length") or 0)
    terms, hist = [], set()
    n = len(names)
    for key, t in grp.items():
        if key in GROUP_KEYS or not isinstance(t, dict):
            continue
        fname = str(t.get("func", "")).rsplit(":", 1)[-1]
        tid = TERM_IDS.get(fname)
        if tid is None:
            raise ValueError(f"observation {key}: function {t.get('func')} has no term here")
        dim = {"gait_phase": 2, "gait_phase_legs": 4}.get(
            tid, n if tid.startswith(("joint", "last")) else 3
        )
        sc = t.get("scale")
        scale = (
            [float(x) for x in sc]
            if isinstance(sc, list)
            else [float(sc if sc is not None else 1.0)] * dim
        )
        params: dict[str, Any] = {}
        tp = t.get("params") or {}
        if tid == "velocity_commands":
            params = {"command_name": tp.get("command_name", "base_velocity")}
        elif tid == "last_action":
            params = {"reset": "zeros"}
        elif tid == "gait_phase":
            params = {
                "period": float(tp["period"]),
                "arithmetic": "float32",
                "clock_offset_steps": 0,
            }
        elif tid == "gait_phase_legs":
            params = {"period": float(tp["period"])}
        if t.get("noise"):
            findings.append(
                f"{key}: noise in training ({str(t['noise'].get('func', '')).rsplit(':', 1)[-1]})"
            )
        terms.append(
            {
                "id": tid,
                "source_name": key,
                "dim": dim,
                "scale": scale,
                "clip": list(t["clip"]) if t.get("clip") else None,
                "params": params,
            }
        )
        hist.add(int(t.get("history_length") or 0) or gh)
    if len(hist) > 1:
        raise ValueError(f"per-term history lengths differ: {sorted(hist)}")
    H = max(hist.pop() if hist else 1, 1)
    c.set(
        "policy_io.observation_groups.policy",
        {
            "terms": terms,
            "history": {
                "length": H,
                "layout": "term_major",
                "order": "oldest_first",
                "init": "repeat_first",
            },
            "clip_then_scale": True,
        },
        "file",
        f"observations.policy of {env_yaml.name}; {MANAGER}",
    )

    # commands
    cmd = (y.get("commands") or {}).get("base_velocity") or {}
    start, lim = cmd.get("ranges") or {}, cmd.get("limit_ranges") or {}
    cur = y.get("curriculum") or {}
    widen_lin = "lin_vel_cmd_levels" in cur
    widen_ang = "ang_vel_cmd_levels" in cur

    def rng(d: dict, k: str) -> list[float] | None:
        return [float(v) for v in d[k]] if d.get(k) is not None else None

    limit = {"vx": rng(lim, "lin_vel_x"), "vy": rng(lim, "lin_vel_y"), "wz": rng(lim, "ang_vel_z")}
    trained = {
        "vx": rng(lim if widen_lin and lim else start, "lin_vel_x"),
        "vy": rng(lim if widen_lin and lim else start, "lin_vel_y"),
        "wz": rng(lim if widen_ang and lim else start, "ang_vel_z"),
    }
    c.set(
        "policy_io.commands.base_velocity",
        {
            "limit": limit if lim else None,
            "trained": trained,
            "heading": "on" if cmd.get("heading_command") else "off",
        },
        "file",
        "commands.base_velocity: ranges, limit_ranges; trained ranges widened to the limits "
        f"only on axes a curriculum covers (lin_vel_cmd_levels {widen_lin}, ang_vel_cmd_levels "
        f"{widen_ang}), reached only if training ran long enough",
    )
    c.set(
        "policy_io.graph",
        {
            "inputs": [{"name": "obs", "shape": [1, H * sum(t["dim"] for t in terms)]}],
            "outputs": [{"name": "actions", "shape": [1, n]}],
            "recurrent": [],
        },
        "file",
        "term sizes",
    )

    # actions
    act = (y.get("actions") or {}).get("JointPositionAction")
    if act is None:
        raise ValueError("no actions.JointPositionAction")
    if act.get("preserve_order"):
        findings.append("preserve_order: the action follows joint_names, not the articulation")
    scale = resolve(act.get("scale"), names, "actions.scale", 1.0)
    if act.get("use_default_offset", True):
        offset = dict(default)
    else:
        offset = {
            k: float(v or 0.0)
            for k, v in resolve(act.get("offset"), names, "actions.offset", 0.0).items()
        }
    clip = act.get("clip")
    clip_l = None
    if clip:
        cr = resolve(clip, names, "actions.clip")
        clip_l = [[float(cr[k][0]), float(cr[k][1])] if cr[k] else [-1e3, 1e3] for k in names]
        findings.append("actions.clip: Isaac Lab clips the processed target")
    c.set("control.default_joint_pos", default, "file", "scene.robot.init_state.joint_pos")
    c.set(
        "control.actions.joint_pos",
        {
            "scale": {k: float(v) for k, v in scale.items()},
            "offset": offset,
            "clip": clip_l,
            "clip_stage": "processed" if clip_l else "none",
        },
        "file",
        "actions.JointPositionAction",
    )
    c.set(
        "control.actuators",
        {
            "kp": kp,
            "kd": kd,
            "kind": "implicit_pd" if implicit else "explicit_pd",
            "pd_period": "solver" if implicit else "sim_step",
            "integrator": "physx",
            "torque_limit_at": "actuator_force",
        },
        "file",
        f"scene.robot.actuators ({', '.join(sorted(kinds))})",
    )
    c.set("control.ownership", {"owned": "all", "external": []}, "file", "")
    c.set("model.armature", arm or None, "file" if arm else "unknown", "actuators.armature")
    c.set(
        "model.effort_limit",
        eff or None,
        "file" if eff else "unknown",
        "actuators.effort_limit_sim",
    )
    c.set(
        "model.velocity_limit",
        vel or None,
        "file" if vel else "unknown",
        "actuators.velocity_limit_sim",
    )
    for k in ("joint_friction", "joint_damping"):
        c.set(f"model.{k}", None, "unknown", "from the USD, not read")
    c.set("evidence", {"level": "L0", "golden": None}, "default", "")
    return c, findings
