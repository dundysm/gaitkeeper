"""Contract from a unitree_rl_gym MuJoCo deploy config (deploy/deploy_mujoco/configs/*.yaml).

unitree_rl_gym's ``deploy_mujoco.py`` is the template many community G1 policies are
deployed with: a YAML of gains, default angles, scales and sizes, and an observation the
script builds by hand. The script is read as unitree_rl_gym has it (pinned below):

* observation, single frame: ang_vel * ang_vel_scale, projected gravity, cmd * cmd_scale,
  (q - default) * dof_pos_scale, dq * dof_vel_scale, the last raw action, then
  [sin, cos] of a gait phase with period 0.8 s (``gait_period`` when the config has one,
  as forks add). The phase time is the physics step counter times simulation_dt, read
  after the counter is advanced, so the first policy input sees one policy step of phase.
* action: target = action * action_scale + default_angles, no clip; torque
  kp (target - q) - kd dq every physics step.
* joints: the script reads ``qpos[7:]`` of its own scene, whose joints are the first
  ``num_actions`` of the Unitree SDK order for the 12 and 29 dof G1 scenes.

A fork that changes the script (another observation, history, a recurrent policy) is not
this format; the reader checks ``num_obs`` against the layout above and refuses otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..contract import SCHEMA, Contract, sha256_file
from ..tables import SDK_TABLES

SCRIPT = "unitree_rl_gym deploy/deploy_mujoco/deploy_mujoco.py (unitreerobotics/unitree_rl_gym@main, 2025)"
WELDED = (
    "the unitree_rl_gym G1 scene for this policy has no joints past the ones it lists "
    "(waist and arms welded): approximated by stiff servos at zero"
)


def read_rl_gym_deploy(
    yaml_path: str | Path,
    onnx_path: str | Path | None = None,
    robot: str = "unitree_g1_29dof",
    welded_kp: float = 500.0,
    welded_kd: float = 10.0,
) -> tuple[Contract, list[str]]:
    """Returns (contract, findings)."""
    yaml_path = Path(yaml_path)
    y = yaml.safe_load(yaml_path.read_text())
    findings: list[str] = []
    table_name, table = SDK_TABLES[robot]
    n = int(y["num_actions"])
    expect = 9 + 3 * n + 2
    if int(y["num_obs"]) != expect:
        raise ValueError(
            f"{yaml_path.name}: num_obs {y['num_obs']}, the deploy_mujoco layout for {n} "
            f"actions has {expect}; this fork builds another observation"
        )
    names = [table[i] for i in range(n)]
    if "" in names:
        raise ValueError(f"{n} actions do not map onto the {table_name} prefix")

    def per_joint(key: str) -> dict[str, float]:
        v = [float(x) for x in y[key]]
        if len(v) < n:
            raise ValueError(f"{key}: {len(v)} entries for {n} actions")
        if len(v) > n:
            findings.append(f"{key}: {len(v)} entries, the first {n} used")
        return dict(zip(names, v[:n]))

    sim_dt = float(y["simulation_dt"])
    dec = int(y["control_decimation"])
    policy_dt = sim_dt * dec
    period = float(y.get("gait_period", 0.8))
    default = per_joint("default_angles")
    c = Contract({"schema": SCHEMA})
    files = [{"path": yaml_path.name, "sha256": sha256_file(yaml_path)}]
    if onnx_path:
        files.append({"path": Path(onnx_path).name, "sha256": sha256_file(onnx_path)})
    c.set(
        "source",
        {
            "format": "unitree_rl_gym_deploy",
            "framework": {"name": "legged_gym", "version": None},
            "files": files,
            "robot": robot,
        },
        "file",
    )
    c.set("timing.policy_dt", policy_dt, "file", "simulation_dt * control_decimation")
    c.set("timing.sim_dt", sim_dt, "file", "simulation_dt")
    c.set("timing.decimation", dec, "file", "control_decimation")
    c.set("timing.order", ["obs", "infer", "target", "pd", "step"], "default", SCRIPT)
    c.set("timing.target_hold", "zoh", "file", SCRIPT)
    c.set(
        "policy_io.joints",
        {"names": names, "table": table_name, "joint_ids_map": list(range(n))},
        "file+table",
        f"qpos[7:] of the deploy scene, the {table_name} prefix",
    )
    c.set(
        "policy_io.imu",
        {"body": "pelvis", "frame": "body", "rotation_in_root": [1.0, 0.0, 0.0, 0.0]},
        "file",
        "qvel[3:6] and qpos[3:7] of the free joint",
    )

    def term(tid: str, dim: int, scale: Any, params: dict | None = None) -> dict[str, Any]:
        s = [float(x) for x in scale] if isinstance(scale, list) else [float(scale)] * dim
        return {
            "id": tid,
            "source_name": tid,
            "dim": dim,
            "scale": s,
            "clip": None,
            "params": params or {},
        }

    terms = [
        term("base_ang_vel", 3, y["ang_vel_scale"]),
        term("projected_gravity", 3, 1.0),
        term("velocity_commands", 3, list(y["cmd_scale"]), {"command_name": "base_velocity"}),
        term("joint_pos_rel", n, y["dof_pos_scale"]),
        term("joint_vel_rel", n, y["dof_vel_scale"]),
        term("last_action", n, 1.0, {"reset": "zeros"}),
        term(
            "gait_phase",
            2,
            1.0,
            {"period": period, "arithmetic": "float64", "clock_offset_steps": 1},
        ),
    ]
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
            "clip_then_scale": True,
        },
        "file",
        f"{yaml_path.name} scales and sizes; layout and phase from {SCRIPT}",
    )
    c.set(
        "policy_io.commands.base_velocity",
        {"limit": None, "trained": None, "heading": "off"},
        "unknown",
        "the deploy config holds one command (cmd_init), not a range",
    )
    c.set(
        "policy_io.graph",
        {
            "inputs": [{"name": "obs", "shape": [1, expect]}],
            "outputs": [{"name": "actions", "shape": [1, n]}],
            "recurrent": [],
        },
        "file",
        "num_obs, num_actions",
    )
    c.set("control.default_joint_pos", default, "file", "default_angles")
    scale = float(y["action_scale"])
    c.set(
        "control.actions.joint_pos",
        {
            "scale": {k: scale for k in names},
            "offset": dict(default),
            "clip": None,
            "clip_stage": "none",
        },
        "file",
        f"action_scale, default_angles; no clip in {SCRIPT}",
    )
    c.set(
        "control.actuators",
        {
            "kp": per_joint("kps"),
            "kd": per_joint("kds"),
            "kind": "explicit_pd",
            "pd_period": "sim_step",
            "integrator": None,
            "torque_limit_at": "actuator_force",
        },
        "file",
        f"kps, kds; pd_control every physics step in {SCRIPT}",
    )
    c.set("control.actuators.integrator", None, "unknown", "from the deploy scene, not read")
    c.set("control.ownership", {"owned": "all", "external": []}, "file", "")
    rest = [table[i] for i in range(len(table)) if table[i] and i >= n]
    if rest:
        c.set(
            "control.unlisted",
            {
                "pose": {k: 0.0 for k in rest},
                "kp": {k: welded_kp for k in rest},
                "kd": {k: welded_kd for k in rest},
                "note": WELDED,
            },
            "default",
            WELDED,
        )
        findings.append(f"{len(rest)} joints past the policy's are welded upstream: " + WELDED)
    for k in ("effort_limit", "velocity_limit", "armature", "joint_friction", "joint_damping"):
        c.set(f"model.{k}", None, "unknown", "from the deploy scene, not read")
    c.set("evidence", {"level": "L0", "golden": None}, "default", "")
    return c, findings
