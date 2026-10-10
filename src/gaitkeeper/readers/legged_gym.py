"""Contract from a legged_gym training config: what the policy was trained with.

legged_gym (and unitree_rl_gym, which builds its G1 and H1 tasks on it) keeps a task's
training facts in Python classes: ``class G1RoughCfg(LeggedRobotCfg)`` with nested
``env``, ``control``, ``init_state``, ``commands``, ``normalization`` and ``sim``, each
inheriting the base class's defaults. The reader evaluates those classes without
importing them (``ast``, literal values only), with the base config given as a second
file, so neither legged_gym nor Isaac Gym is needed.

From the classes:

* joints and their order: the URDF's revolute joints in depth-first order, which is the
  order Isaac Gym gives the DOFs (when no URDF is given, the Unitree SDK prefix, flagged);
* gains: ``control.stiffness``/``damping`` matched to joint names by substring, the rule in
  ``legged_robot.py _init_buffers``; ``action_scale``; ``decimation`` and ``sim.dt``;
* actions clipped to +-``normalization.clip_actions`` before use and before they are
  observed as the last action;
* observation: ``num_observations`` picks the layout. 12 + 3n is legged_gym's base
  (base_lin_vel, base_ang_vel, gravity, commands, dof_pos, dof_vel, actions); 9 + 3n + 2 is
  unitree_rl_gym's G1/H1 env (no lin_vel, a [sin, cos] gait phase at the end, period from
  the env file when given, else 0.8 s). Scales from ``normalization.obs_scales``, the
  command scaled by [lin_vel, lin_vel, ang_vel]; the whole vector clipped to
  +-``clip_observations`` after scaling.
* the trained command ranges, and whether yaw comes from a heading command.
"""

from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..contract import SCHEMA, Contract, sha256_file
from ..tables import SDK_TABLES

ENV = "legged_gym/envs/base/legged_robot.py (unitreerobotics/unitree_rl_gym@276801e)"


class _Cls:
    def __init__(self, name: str):
        self.name = name
        self.attrs: dict[str, Any] = {}

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if isinstance(node, _Cls) and part in node.attrs:
                node = node.attrs[part]
            else:
                return default
        return node


def _classes(src: str, known: dict[str, _Cls]) -> dict[str, _Cls]:
    """Top-level classes of a module, nested classes merged over the bases they name."""
    tree = ast.parse(src)

    def resolve(expr: ast.expr) -> _Cls | None:
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name):
            return None
        parts.append(expr.id)
        parts.reverse()
        node = known.get(parts[0])
        for p in parts[1:]:
            node = node.attrs.get(p) if isinstance(node, _Cls) else None
        return node if isinstance(node, _Cls) else None

    def build(cd: ast.ClassDef) -> _Cls:
        c = _Cls(cd.name)
        for b in cd.bases:
            base = resolve(b)
            if base is not None:
                c.attrs.update(base.attrs)
        for stmt in cd.body:
            if isinstance(stmt, ast.ClassDef):
                # a nested class replaces the inherited one; it inherits only what it names
                c.attrs[stmt.name] = build(stmt)
            elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                t = stmt.targets[0]
                if isinstance(t, ast.Name):
                    try:
                        c.attrs[t.id] = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError, TypeError):
                        c.attrs[t.id] = None
        return c

    out = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out[node.name] = known[node.name] = build(node)
    return out


def urdf_dof_order(urdf: str | Path) -> list[str]:
    """Revolute and prismatic joints, depth first from the root link, children in file
    order: the order Isaac Gym gives an asset's DOFs."""
    root = ET.parse(urdf).getroot()
    joints = root.findall("joint")
    children: dict[str, list[ET.Element]] = {}
    child_links = set()
    for j in joints:
        p = j.find("parent").get("link")
        children.setdefault(p, []).append(j)
        child_links.add(j.find("child").get("link"))
    roots = [ln.get("name") for ln in root.findall("link") if ln.get("name") not in child_links]
    out: list[str] = []

    def walk(link: str) -> None:
        for j in children.get(link, []):
            if j.get("type") in ("revolute", "continuous", "prismatic"):
                out.append(j.get("name"))
            walk(j.find("child").get("link"))

    for r in roots:
        walk(r)
    return out


def read_legged_gym(
    cfg_py: str | Path,
    base_py: str | Path,
    cls: str | None = None,
    urdf: str | Path | None = None,
    env_py: str | Path | None = None,
    robot: str = "unitree_g1_29dof",
    welded_kp: float = 500.0,
    welded_kd: float = 10.0,
) -> tuple[Contract, list[str]]:
    """Returns (contract, findings)."""
    findings: list[str] = []
    known: dict[str, _Cls] = {}
    _classes(Path(base_py).read_text(), known)
    mine = _classes(Path(cfg_py).read_text(), known)
    if cls is None:
        cands = [k for k, v in mine.items() if v.get("control") is not None and "PPO" not in k]
        if len(cands) != 1:
            raise ValueError(f"{cfg_py}: name the config class (--cls), one of {sorted(mine)}")
        cls = cands[0]
    cfg = mine.get(cls) or known.get(cls)
    if cfg is None:
        raise ValueError(f"{cfg_py}: no class {cls}")
    n = int(cfg.get("env.num_actions"))
    nobs = int(cfg.get("env.num_observations"))
    table_name, table = SDK_TABLES[robot]
    if urdf:
        names = urdf_dof_order(urdf)
        joints_from = f"{Path(urdf).name}, depth first (Isaac Gym's DOF order)"
        if len(names) != n:
            raise ValueError(f"{urdf}: {len(names)} DOFs, the config has {n} actions")
    else:
        names = [table[i] for i in range(n)]
        joints_from = f"assumed: the {table_name} prefix (give the URDF to read the order)"
        findings.append("joint order assumed from the SDK prefix; give --urdf to read it")
    unknown = [nm for nm in names if nm not in table]
    if unknown:
        raise ValueError(f"joints not in {table_name}: {unknown}")

    def by_substring(key: str) -> dict[str, float]:
        d = cfg.get(f"control.{key}") or {}
        out = {}
        for nm in names:
            hit = next((float(v) for k, v in d.items() if k in nm), None)
            if hit is None:
                raise ValueError(f"control.{key}: no entry matches {nm}")
            out[nm] = hit
        return out

    defaults = cfg.get("init_state.default_joint_angles") or {}
    missing = [nm for nm in names if nm not in defaults]
    if missing:
        raise ValueError(f"init_state.default_joint_angles lacks {missing}")
    default = {nm: float(defaults[nm]) for nm in names}
    sim_dt = float(cfg.get("sim.dt"))
    dec = int(cfg.get("control.decimation"))
    sc = cfg.get("normalization.obs_scales")
    clip_obs = float(cfg.get("normalization.clip_observations"))
    clip_act = float(cfg.get("normalization.clip_actions"))
    action_scale = float(cfg.get("control.action_scale"))
    detail = f"{Path(cfg_py).name} {cls} over {Path(base_py).name}"

    c = Contract({"schema": SCHEMA})
    files = [
        {"path": Path(p).name, "sha256": sha256_file(p)}
        for p in (cfg_py, base_py, urdf, env_py)
        if p
    ]
    c.set(
        "source",
        {
            "format": "legged_gym_cfg",
            "framework": {"name": "legged_gym", "version": None},
            "files": files,
            "robot": robot,
        },
        "file",
    )
    c.set("timing.policy_dt", sim_dt * dec, "file", "sim.dt * control.decimation")
    c.set("timing.sim_dt", sim_dt, "file", "sim.dt")
    c.set("timing.decimation", dec, "file", "control.decimation")
    c.set("timing.order", ["obs", "infer", "target", "pd", "step"], "default", ENV)
    c.set("timing.target_hold", "zoh", "file", ENV)
    c.set("policy_io.joints", {"names": names, "table": table_name}, "file", joints_from)
    c.set(
        "policy_io.imu",
        {"body": "base", "frame": "body", "rotation_in_root": [1.0, 0.0, 0.0, 0.0]},
        "file",
        "root state rotated into the base frame (quat_rotate_inverse)",
    )

    def term(tid: str, dim: int, scale: Any, params: dict | None = None) -> dict[str, Any]:
        s = [float(x) for x in scale] if isinstance(scale, list) else [float(scale)] * dim
        return {
            "id": tid,
            "source_name": tid,
            "dim": dim,
            "scale": s,
            "clip": [-clip_obs, clip_obs],
            "params": params or {},
        }

    cmd_scale = [sc.get("lin_vel"), sc.get("lin_vel"), sc.get("ang_vel")]
    joint_terms = [
        term("joint_pos_rel", n, sc.get("dof_pos")),
        term("joint_vel_rel", n, sc.get("dof_vel")),
        term("last_action", n, 1.0, {"reset": "zeros"}),
    ]
    head = [
        term("base_ang_vel", 3, sc.get("ang_vel")),
        term("projected_gravity", 3, 1.0),
        term("velocity_commands", 3, cmd_scale, {"command_name": "base_velocity"}),
    ]
    if nobs == 12 + 3 * n:
        terms = [term("base_lin_vel", 3, sc.get("lin_vel"))] + head + joint_terms
        layout = "legged_gym base: lin_vel, ang_vel, gravity, commands, dof_pos, dof_vel, actions"
    elif nobs == 9 + 3 * n + 2:
        period = 0.8
        if env_py:
            m = re.search(r"period\s*=\s*([0-9.]+)", Path(env_py).read_text())
            if m:
                period = float(m.group(1))
        else:
            findings.append("gait period assumed 0.8 s (give the env file to read it)")
        terms = (
            head
            + joint_terms
            + [
                term(
                    "gait_phase",
                    2,
                    1.0,
                    {"period": period, "arithmetic": "float64", "clock_offset_steps": 1},
                )
            ]
        )
        layout = "unitree_rl_gym G1/H1 env: ang_vel, gravity, commands, dof_pos, dof_vel, actions, sin, cos"
    else:
        raise ValueError(
            f"num_observations {nobs} fits neither legged_gym's base layout ({12 + 3 * n}) "
            f"nor unitree_rl_gym's G1/H1 env ({9 + 3 * n + 2})"
        )
    c.set(
        "policy_io.observation_groups.policy",
        {
            "terms": terms,
            "history": {
                "length": 1,
                "layout": "term_major",
                "order": "oldest_first",
                "init": "repeat_first",
            },
            "clip_then_scale": False,
        },
        "file",
        f"{detail}; layout {layout}; scaled, then clipped to +-{clip_obs:g}",
    )
    rng = cfg.get("commands.ranges")
    heading = bool(cfg.get("commands.heading_command"))
    trained = {
        "vx": [float(x) for x in rng.get("lin_vel_x")],
        "vy": [float(x) for x in rng.get("lin_vel_y")],
        "wz": [float(x) for x in rng.get("ang_vel_yaw")] if not heading else [-1.0, 1.0],
    }
    c.set(
        "policy_io.commands.base_velocity",
        {"limit": None, "trained": trained, "heading": "on" if heading else "off"},
        "file",
        "commands.ranges"
        + ("; heading_command: yaw rate from the heading error, clipped to +-1" if heading else ""),
    )
    c.set(
        "policy_io.graph",
        {
            "inputs": [{"name": "obs", "shape": [1, nobs]}],
            "outputs": [{"name": "actions", "shape": [1, n]}],
            "recurrent": [],
        },
        "file",
        "env.num_observations, env.num_actions",
    )
    c.set("control.default_joint_pos", default, "file", "init_state.default_joint_angles")
    c.set(
        "control.actions.joint_pos",
        {
            "scale": {k: action_scale for k in names},
            "offset": dict(default),
            "clip": [
                [default[k] - action_scale * clip_act, default[k] + action_scale * clip_act]
                for k in names
            ],
            "clip_stage": "processed",
        },
        "file",
        f"control.action_scale; actions clipped to +-{clip_act:g} (normalization.clip_actions)",
    )
    c.set(
        "control.actuators",
        {
            "kp": by_substring("stiffness"),
            "kd": by_substring("damping"),
            "kind": "explicit_pd",
            "pd_period": "sim_step",
            "integrator": None,
            "torque_limit_at": "actuator_force",
        },
        "file",
        f"control.stiffness/damping by joint-name substring ({ENV} _init_buffers); "
        "torques in Python every sim step",
    )
    c.set("control.actuators.integrator", None, "unknown", "Isaac Gym (PhysX), not MuJoCo")
    if cfg.get("control.control_type", "P") != "P":
        findings.append(f"control_type {cfg.get('control.control_type')!r}: not position control")
    c.set("control.ownership", {"owned": "all", "external": []}, "file", "")
    rest = [table[i] for i in range(len(table)) if table[i] and table[i] not in names]
    if rest:
        why = (
            "joints the training asset does not have (welded): approximated by stiff servos at zero"
        )
        c.set(
            "control.unlisted",
            {
                "pose": {k: 0.0 for k in rest},
                "kp": {k: welded_kp for k in rest},
                "kd": {k: welded_kd for k in rest},
                "note": why,
            },
            "default",
            why,
        )
    for k in ("effort_limit", "velocity_limit", "armature", "joint_friction", "joint_damping"):
        c.set(f"model.{k}", None, "unknown", "from the URDF and asset options, not read")
    c.set("evidence", {"level": "L0", "golden": None}, "default", "")
    return c, findings
