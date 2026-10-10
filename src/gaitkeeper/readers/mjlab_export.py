"""Read an mjlab policy export (ONNX metadata plus Unitree-style deploy.yaml).

Trust order (plan section 6): ONNX metadata, then deploy.yaml. Both are read;
a value given by both is compared at the resolution each was printed with.
Agreement within rounding keeps the more precise value and notes the other's
error; disagreement beyond rounding marks the field ``conflicting``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort

from ..contract import SCHEMA, Contract, sha256_file
from .numtext import TextFloat, load_yaml_with_text, parse_csv_floats, parse_csv_names, res

# mjlab observation term names (ONNX metadata) and deploy.yaml keys, mapped to
# the term library's ids.
ONNX_TERM_IDS = {
    "base_lin_vel": "base_lin_vel",
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "command": "velocity_commands",
    "phase": "gait_phase",
    "joint_pos": "joint_pos_rel",
    "joint_vel": "joint_vel_rel",
    "actions": "last_action",
}
YAML_TERM_IDS = {
    "base_lin_vel": "base_lin_vel",
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "velocity_commands": "velocity_commands",
    "gait_phase": "gait_phase",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_rel": "joint_vel_rel",
    "last_action": "last_action",
}
TERM_DIMS = {
    "base_lin_vel": 3,
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "gait_phase": 2,
}
JOINT_TERMS = ("joint_pos_rel", "joint_vel_rel", "last_action")

# Facts about term implementations that no exported file states. Each is a
# framework default at a named source version and is labeled as such.
PHASE_DEFAULT = {
    "stand_threshold": 0.1,
    "clock": "episode_steps_since_reset",
    "arithmetic": "float32",
    "detail": "unitree_rl_mjlab src/tasks/velocity/mdp/observations.py@1425b15: "
    "phase is zero when the command norm is below 0.1; clock is episode_length_buf * step_dt",
}


@dataclass
class ReaderFinding:
    path: str
    kind: str  # "rounding", "conflicting", "missing", "naming"
    message: str
    max_rel: float = 0.0


def _merge_list(
    c: Contract,
    path: str,
    names: list[str],
    onnx_vals: list[TextFloat] | None,
    yaml_vals: list[Any] | None,
    findings: list[ReaderFinding],
    onnx_key: str,
    yaml_key: str,
) -> None:
    """Merge a per-joint numeric list from the two files into a name-keyed map."""
    if onnx_vals is None and yaml_vals is None:
        c.set(path, None, "unknown", "in neither exported file")
        findings.append(ReaderFinding(path, "missing", "not in ONNX metadata or deploy.yaml"))
        return
    primary = onnx_vals if onnx_vals is not None else yaml_vals
    psrc = f"ONNX metadata {onnx_key}" if onnx_vals is not None else f"deploy.yaml {yaml_key}"
    assert primary is not None
    if len(primary) != len(names):
        raise ValueError(f"{path}: {len(primary)} values for {len(names)} joints")
    value = {n: float(v) for n, v in zip(names, primary)}
    pres = max(res(v) for v in primary)
    if onnx_vals is None or yaml_vals is None:
        c.set(path, value, "file", psrc, resolution=pres)
        return
    worst_rel, worst_abs, worst_joint, conflicts = 0.0, 0.0, "", []
    for n, a, b in zip(names, onnx_vals, yaml_vals):
        d = abs(float(a) - float(b))
        if d > res(a) + res(b) + 1e-12:
            conflicts.append(n)
        rel = d / max(abs(float(a)), 1e-12)
        if rel > worst_rel:
            worst_rel, worst_abs, worst_joint = rel, d, n
    if conflicts:
        c.set(
            path,
            value,
            "conflicting",
            f"{psrc} used; deploy.yaml {yaml_key} differs beyond rounding on {conflicts}",
            resolution=pres,
            alternatives={
                "onnx": [float(v) for v in onnx_vals],
                "yaml": [float(v) for v in yaml_vals],
            },
        )
        findings.append(
            ReaderFinding(
                path, "conflicting", f"yaml differs beyond rounding on {conflicts}", worst_rel
            )
        )
    else:
        detail = f"{psrc}; deploy.yaml {yaml_key} agrees within its rounding"
        if worst_rel > 0:
            detail += f" (largest error {worst_rel:.1%} on {worst_joint}, {worst_abs:.4g} absolute)"
            findings.append(
                ReaderFinding(
                    path,
                    "rounding",
                    f"deploy.yaml {yaml_key} is rounded; {worst_rel:.1%} off on {worst_joint} "
                    f"({float(yaml_vals[names.index(worst_joint)])} against "
                    f"{float(onnx_vals[names.index(worst_joint)])})",
                    worst_rel,
                )
            )
        c.set(
            path,
            value,
            "file",
            detail,
            resolution=pres,
            alternatives={
                "onnx": [float(v) for v in onnx_vals],
                "yaml": [float(v) for v in yaml_vals],
            },
        )


def _clip_list(clip: Any) -> list:
    """One [low, high] for every joint, or one per joint, as deploy.yaml writes it."""
    if all(isinstance(v, (int, float)) for v in clip):
        return [float(v) for v in clip]
    return [[float(a), float(b)] for a, b in clip]


def read_mjlab_export(
    onnx_path: str | Path, yaml_path: str | Path | None
) -> tuple[Contract, list[ReaderFinding]]:
    onnx_path = Path(onnx_path)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    y: dict[str, Any] = load_yaml_with_text(yaml_path) if yaml_path else {}
    findings: list[ReaderFinding] = []
    c = Contract({"schema": SCHEMA})

    files = [{"path": onnx_path.name, "sha256": sha256_file(onnx_path)}]
    if yaml_path:
        files.append({"path": Path(yaml_path).name, "sha256": sha256_file(yaml_path)})
    c.set(
        "source",
        {
            "format": "mjlab_onnx",
            "framework": {"name": "mjlab", "version": None},
            "run_path": meta.get("run_path"),
            "files": files,
        },
        "file",
    )

    # -- timing
    step_dt = y.get("step_dt")
    c.set(
        "timing.policy_dt",
        float(step_dt) if step_dt is not None else None,
        "file" if step_dt is not None else "unknown",
        "deploy.yaml step_dt",
    )
    c.set("timing.sim_dt", None, "unknown", "not exported")
    c.set("timing.decimation", None, "unknown", "not exported")
    c.set(
        "timing.order",
        ["obs", "infer", "target", "pd", "step"],
        "default",
        "manager-based env step",
    )
    c.set("timing.target_hold", "zoh", "default", "target held between policy updates")

    # -- graph
    ins = [{"name": i.name, "shape": list(i.shape), "dtype": i.type} for i in sess.get_inputs()]
    outs = [{"name": o.name, "shape": list(o.shape)} for o in sess.get_outputs()]
    c.set("policy_io.graph.inputs", ins, "file", "ONNX graph")
    c.set("policy_io.graph.outputs", outs, "file", "ONNX graph")
    c.set(
        "policy_io.graph.recurrent",
        [] if len(ins) == 1 else None,
        "file" if len(ins) == 1 else "unknown",
        "single input, no state tensors",
    )

    # -- joints
    names = parse_csv_names(meta["joint_names"]) if "joint_names" in meta else None
    if names is None:
        raise ValueError("ONNX metadata has no joint_names; a joint table is required")
    c.set("policy_io.joints.names", names, "file", "ONNX metadata joint_names")
    jmap = y.get("joint_ids_map")
    if jmap is not None:
        jmap = [int(v) for v in jmap]
        c.set("policy_io.joints.joint_ids_map", jmap, "file", "deploy.yaml joint_ids_map")
        if jmap != list(range(len(names))):
            findings.append(
                ReaderFinding(
                    "policy_io.joints.joint_ids_map",
                    "naming",
                    "yaml lists per-joint values in SDK order; reordered by joint_ids_map",
                )
            )
    order = jmap if jmap is not None else list(range(len(names)))

    def yaml_list(v: Any) -> list[Any] | None:
        if v is None:
            return None
        v = list(v)
        out: list[Any] = [None] * len(v)
        # deploy.yaml lists gains and default pose in SDK order; joint_ids_map[i]
        # is the SDK index of policy joint i.
        for i, sdk in enumerate(order):
            out[i] = v[sdk]
        return out

    # -- imu: not stated by any exported file
    c.set(
        "policy_io.imu",
        {"body": None, "frame": "body", "rotation_in_root": [1.0, 0.0, 0.0, 0.0]},
        "default",
        "mjlab and Isaac Lab report base angular velocity in the root body frame",
    )

    # -- observation terms
    onnx_terms = parse_csv_names(meta.get("observation_names", ""))
    yaml_obs = y.get("observations") or {}
    yaml_terms = list(yaml_obs.keys())
    ids_onnx = [ONNX_TERM_IDS.get(t) for t in onnx_terms]
    ids_yaml = [YAML_TERM_IDS.get(t) for t in yaml_terms]
    if onnx_terms and yaml_terms and ids_onnx != ids_yaml:
        findings.append(
            ReaderFinding(
                "policy_io.observation_groups.policy.terms",
                "conflicting",
                f"term order differs: onnx {onnx_terms} yaml {yaml_terms}",
            )
        )
    ids = ids_onnx if onnx_terms else ids_yaml
    if None in ids:
        raise ValueError(f"unmapped observation terms: {onnx_terms or yaml_terms}")
    terms: list[dict[str, Any]] = []
    hist = set()
    for k, tid in enumerate(ids):
        ykey = yaml_terms[k] if k < len(yaml_terms) else None
        yt = yaml_obs.get(ykey, {}) if ykey else {}
        dim = TERM_DIMS.get(tid, len(names) if tid in JOINT_TERMS else None)
        scale = yt.get("scale")
        scale = [float(s) for s in scale] if scale is not None else [1.0] * dim
        if len(scale) != dim:
            raise ValueError(f"term {tid}: scale has {len(scale)} entries, dim {dim}")
        term: dict[str, Any] = {
            "id": tid,
            "source_name": onnx_terms[k] if onnx_terms else ykey,
            "dim": dim,
            "scale": scale,
            "clip": yt.get("clip"),
            "params": {
                kk: (float(vv) if isinstance(vv, TextFloat) else vv)
                for kk, vv in (yt.get("params") or {}).items()
            },
        }
        if tid == "gait_phase":
            term["params"].setdefault("stand_threshold", PHASE_DEFAULT["stand_threshold"])
            term["params"].setdefault("clock", PHASE_DEFAULT["clock"])
            term["params"].setdefault("arithmetic", PHASE_DEFAULT["arithmetic"])
        if tid == "last_action":
            term["params"]["reset"] = "zeros"
        terms.append(term)
        hist.add(int(yt.get("history_length", 1) or 1))
        base = f"policy_io.observation_groups.policy.terms.{tid}"
        c.provenance[base] = c.provenance.get(base) or _prov(
            "file", f"ONNX observation_names[{k}], deploy.yaml {ykey}"
        )
        if tid == "gait_phase":
            c.provenance[base + ".params.stand_threshold"] = _prov(
                "default", PHASE_DEFAULT["detail"]
            )
            c.provenance[base + ".params.arithmetic"] = _prov(
                "default",
                "torch default dtype float32; the argument's rounding reaches 2e-5 by 20 s",
            )
            findings.append(
                ReaderFinding(
                    base + ".params.stand_threshold",
                    "missing",
                    "the phase term's stand rule (zero below command norm 0.1) "
                    "is in no exported file; taken from the source at 1425b15",
                )
            )
        if tid == "last_action":
            c.provenance[base + ".params.reset"] = _prov(
                "default", "action manager zeros raw actions on reset"
            )
    c.set("policy_io.observation_groups.policy.terms", terms)
    if len(hist) > 1:
        raise ValueError(f"per-term history lengths differ: {hist}")
    hl = hist.pop() if hist else 1
    c.set(
        "policy_io.observation_groups.policy.history",
        {"length": hl, "layout": "term_major", "order": "oldest_first", "init": "repeat_first"},
        "default",
        "history_length from deploy.yaml; layout, order, init are framework defaults",
    )
    c.provenance["policy_io.observation_groups.policy.history.length"] = _prov(
        "file", "deploy.yaml history_length"
    )
    c.set(
        "policy_io.observation_groups.policy.clip_then_scale",
        True,
        "default",
        "manager-based observation order",
    )

    # -- commands
    cmd_names = parse_csv_names(meta.get("command_names", ""))
    ycmd = y.get("commands") or {}
    ckey = next(iter(ycmd), None)
    rng = (ycmd.get(ckey) or {}).get("ranges", {}) if ckey else {}
    limit = None
    if rng:
        limit = {
            ax: [float(v) for v in rng[k]]
            for ax, k in (("vx", "lin_vel_x"), ("vy", "lin_vel_y"), ("wz", "ang_vel_z"))
            if rng.get(k) is not None
        }
    c.set(
        "policy_io.commands.base_velocity",
        {
            "name": cmd_names[0] if cmd_names else ckey,
            "limit": limit,
            "trained": None,
            "heading": "off" if rng.get("heading") is None else "on",
        },
    )
    c.provenance["policy_io.commands.base_velocity.limit"] = _prov(
        "file", f"deploy.yaml commands.{ckey}.ranges (play-time limits, not training ranges)"
    )
    c.provenance["policy_io.commands.base_velocity.trained"] = _prov(
        "unknown", "training ranges and curriculum are not exported"
    )
    if cmd_names and ckey and cmd_names[0] != ckey:
        findings.append(
            ReaderFinding(
                "policy_io.commands.base_velocity.name",
                "naming",
                f"command is '{cmd_names[0]}' in ONNX metadata, '{ckey}' in deploy.yaml",
            )
        )

    # -- control
    ya = (y.get("actions") or {}).get("JointPositionAction") or {}
    onnx_scale = parse_csv_floats(meta["action_scale"]) if "action_scale" in meta else None
    _merge_list(
        c,
        "control.actions.joint_pos.scale",
        names,
        onnx_scale,
        yaml_list(ya.get("scale")),
        findings,
        "action_scale",
        "actions.JointPositionAction.scale",
    )
    onnx_def = parse_csv_floats(meta["default_joint_pos"]) if "default_joint_pos" in meta else None
    _merge_list(
        c,
        "control.default_joint_pos",
        names,
        onnx_def,
        yaml_list(y.get("default_joint_pos")),
        findings,
        "default_joint_pos",
        "default_joint_pos",
    )
    _merge_list(
        c,
        "control.actions.joint_pos.offset",
        names,
        onnx_def,
        yaml_list(ya.get("offset")),
        findings,
        "default_joint_pos",
        "actions.JointPositionAction.offset",
    )
    c.set(
        "control.actions.joint_pos.kind",
        "joint_position",
        "file",
        "deploy.yaml JointPositionAction",
    )
    c.set("control.actions.joint_pos.joints", names, "file", "ONNX joint_names")
    clip = ya.get("clip")
    c.set(
        "control.actions.joint_pos.clip",
        None if clip is None else _clip_list(clip),
        "file",
        "deploy.yaml clip",
    )
    c.set(
        "control.actions.joint_pos.clip_stage", "none" if clip is None else "processed", "file", ""
    )
    c.set(
        "control.actions.joint_pos.delay_steps", 0, "default", "no delay in the manager-based env"
    )
    c.set("control.actions.joint_pos.filter", "none", "default", "")
    c.set("control.ownership", {"owned": "all", "external": []}, "file", "all joints in the action")

    c.set("control.actuators.kind", None, "unknown", "actuator class is not exported")
    c.set("control.actuators.pd_period", None, "unknown", "")
    c.set("control.actuators.integrator", None, "unknown", "")
    c.set("control.actuators.torque_limit_at", None, "unknown", "")
    onnx_kp = parse_csv_floats(meta["joint_stiffness"]) if "joint_stiffness" in meta else None
    onnx_kd = parse_csv_floats(meta["joint_damping"]) if "joint_damping" in meta else None
    _merge_list(
        c,
        "control.actuators.kp",
        names,
        onnx_kp,
        yaml_list(y.get("stiffness")),
        findings,
        "joint_stiffness",
        "stiffness",
    )
    _merge_list(
        c,
        "control.actuators.kd",
        names,
        onnx_kd,
        yaml_list(y.get("damping")),
        findings,
        "joint_damping",
        "damping",
    )

    # -- model: none of it is exported
    for f in ("armature", "joint_friction", "joint_damping", "effort_limit", "velocity_limit"):
        c.set(f"model.{f}", None, "unknown", "not exported")
    c.set(
        "model.training_envelope",
        {"pushes": None, "floor_friction": None, "obs_noise": None},
        "unknown",
        "not exported",
    )
    c.set("evidence", {"level": "L0", "golden": None})
    return c, findings


def _prov(source: str, detail: str):
    from ..contract import Provenance

    return Provenance(source, detail)
