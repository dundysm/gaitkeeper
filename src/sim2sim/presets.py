"""Training-side facts no exported file states, for named source versions.

A preset is used only when the user names it. Every field it fills carries
provenance ``preset`` with the file and commit it was read from, so a report
always shows that the value is not from the policy's own files.
"""

from __future__ import annotations

from typing import Any

from .contract import Contract
from .tables import G1_29_SDK


def _by_group(groups: list[tuple[tuple[str, ...], float]]) -> dict[str, float]:
    out = {}
    for n in G1_29_SDK:
        for keys, v in groups:
            if any(k in n for k in keys):
                out[n] = v
                break
    return out


_SRC = "unitree_rl_lab@4960b84 source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py UNITREE_G1_29DOF_CFG"
_ENV = "unitree_rl_lab@4960b84 tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py"

# Actuator groups of UNITREE_G1_29DOF_CFG (all ImplicitActuatorCfg).
_EFFORT = _by_group(
    [
        (("hip_pitch", "hip_yaw", "waist_yaw"), 88.0),
        (("hip_roll", "knee"), 139.0),
        (("wrist_pitch", "wrist_yaw"), 5.0),
        (("shoulder", "elbow", "wrist_roll", "ankle", "waist_roll", "waist_pitch"), 25.0),
    ]
)
_VEL = _by_group(
    [
        (("hip_pitch", "hip_yaw", "waist_yaw"), 32.0),
        (("hip_roll", "knee"), 20.0),
        (("wrist_pitch", "wrist_yaw"), 22.0),
        (("shoulder", "elbow", "wrist_roll", "ankle", "waist_roll", "waist_pitch"), 37.0),
    ]
)
_KP = _by_group(
    [
        (("waist_yaw",), 200.0),
        (("knee",), 150.0),
        (("hip",), 100.0),
        (("shoulder", "elbow", "wrist", "ankle", "waist_roll", "waist_pitch"), 40.0),
    ]
)
_KD = _by_group(
    [
        (("waist",), 5.0),
        (("knee",), 4.0),
        (("hip", "ankle"), 2.0),
        (("shoulder", "elbow", "wrist"), 1.0),
    ]
)

PRESETS: dict[str, dict[str, Any]] = {
    "unitree_rl_lab_g1_29dof_velocity@4960b84": {
        "timing.sim_dt": (0.005, _ENV + " __post_init__ sim.dt"),
        "timing.decimation": (4, _ENV + " __post_init__ decimation"),
        "control.actuators.kind": ("implicit_pd", _SRC + ": every group is ImplicitActuatorCfg"),
        "control.actuators.pd_period": (
            "solver",
            "Isaac Lab implicit actuators: PD inside the PhysX solver",
        ),
        "control.actuators.integrator": (
            "physx",
            "Isaac Lab PhysX articulation (solver type not pinned)",
        ),
        "control.actuators.torque_limit_at": ("actuator_force", _SRC + " effort_limit_sim"),
        "model.effort_limit": (_EFFORT, _SRC + " effort_limit_sim"),
        "model.velocity_limit": (_VEL, _SRC + " velocity_limit_sim"),
        "model.armature": ({n: 0.01 for n in G1_29_SDK}, _SRC + " armature=0.01 in every group"),
        "model.joint_friction": (
            {n: 0.0 for n in G1_29_SDK},
            _SRC + " sets no friction (Isaac Lab default 0)",
        ),
        "model.joint_damping": (
            {n: 0.0 for n in G1_29_SDK},
            "implicit drive damping is the kd gain; no passive damping set",
        ),
        "policy_io.commands.base_velocity.trained": (
            {"vx": [-0.5, 1.0], "vy": [-0.3, 0.3], "wz": [-0.1, 0.1]},
            _ENV
            + " CommandsCfg ranges start at 0.1 on every axis; lin_vel_cmd_levels widens x and y "
            "by 0.1 up to limit_ranges when tracking exceeds 80%; yaw is never widened. x and y reach "
            "the limit only if the curriculum got there.",
        ),
        "policy_io.commands.base_velocity.heading": ("off", _ENV + " heading_command=False"),
        "model.training_envelope": (
            {
                "pushes": {
                    "kind": "velocity_kick",
                    "every_s": 5.0,
                    "x": [-0.5, 0.5],
                    "y": [-0.5, 0.5],
                },
                "floor_friction": [0.3, 1.0],
                "obs_noise": "uniform, base_ang_vel +-0.2 and others",
            },
            _ENV + " EventCfg push_robot, physics_material",
        ),
        "_asset_gains": ({"kp": _KP, "kd": _KD}, _SRC + " stiffness and damping"),
    },
}


def apply_preset(c: Contract, name: str, overwrite: bool = False) -> list[str]:
    """Fill unknown fields from a named preset. Returns the paths it filled."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {sorted(PRESETS)}")
    filled = []
    for path, (value, detail) in PRESETS[name].items():
        if path.startswith("_"):
            continue
        cur = c.get(path, None)
        if cur is None or overwrite or c.prov(path).source in ("unknown", "default"):
            c.set(path, value, "preset", f"{name}: {detail}")
            filled.append(path)
    gains = PRESETS[name].get("_asset_gains")
    if gains is not None:
        # The training asset's gains, kept as an alternative to the deployed ones (S21).
        for key in ("kp", "kd"):
            path = f"control.actuators.{key}"
            if c.get(path, None) is not None:
                p = c.prov(path)
                alts = dict(p.alternatives or {})
                names = list(c.get("policy_io.joints.names"))
                alts["asset config"] = [gains[0][key][n] for n in names]
                p.alternatives = alts
    c.set(
        "source.presets",
        sorted({*(c.get("source.presets", None) or []), name}),
        "user",
        "named by the user",
    )
    return filled


def fill_from(c: Contract, ref: Contract, paths: list[str]) -> list[str]:
    """Fill fields the contract does not know from a reference contract (for example a live one)."""
    filled = []
    for path in paths:
        if c.get(path, None) is None or c.prov(path).source in ("unknown",):
            v = ref.get(path, None)
            if v is not None:
                rp = ref.prov(path)
                c.set(
                    path, v, rp.source, f"from the reference contract: {rp.detail}", rp.resolution
                )
                filled.append(path)
    return filled
