"""Read a Unitree deploy.yaml (unitree_rl_lab and unitree_rl_mjlab) as the contract
the robot actually runs.

The deploy runtime (``deploy/include/isaaclab`` in both repositories) builds its
whole observation and action path from this file, so its numbers are taken as
exact: the robot runs 0.07 when the file says 0.07, however the trained value
was rounded. Differences against training-side values (ONNX metadata, a live
contract) are boundary C deviations of the deployment, reported by
``gaitkeeper.deviation``, not file conflicts.

What the runtime reads, at unitree_rl_lab@4960b84 and unitree_rl_mjlab@1425b15
(same lines in both, ``deploy/include``):

* ``step_dt``, ``joint_ids_map``, ``default_joint_pos``, ``stiffness``, ``damping``:
  isaaclab/envs/manager_based_rl_env.h lines 29 to 40. Stiffness and damping are
  in SDK order (isaaclab/assets/articulation/articulation.h lines 19 and 20) and
  are written to every motor as kp and kd on entering the policy state
  (FSM/State_RLBase.h lines 18 to 21).
* ``actions.JointPositionAction`` ``scale``, ``offset``, ``clip``:
  isaaclab/envs/mdp/actions/joint_actions.h lines 28 to 35; processed as
  raw * scale + offset, then clip (lines 46 to 57); the processed target of
  policy joint i goes to motor joint_ids_map[i] (robots/<robot>/src/State_RLBase.cpp).
* ``observations``: per-term ``scale``, ``clip``, ``history_length``
  (isaaclab/manager/observation_manager.h lines 131 to 142). History is a deque,
  oldest first, filled with copies of the first value at reset
  (manager/manager_term_cfg.h lines 27 to 58), concatenated term by term unless
  ``use_gym_history`` is set (observation_manager.h line 72).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import onnxruntime as ort

from ..contract import SCHEMA, Contract, sha256_file
from ..tables import SDK_TABLES
from .mjlab_export import ReaderFinding
from .numtext import TextFloat, load_yaml_with_text

TERM_IDS = {
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "velocity_commands": "velocity_commands",
    "keyboard_velocity_commands": "velocity_commands",
    "gait_phase": "gait_phase",
    "joint_pos_rel": "joint_pos_rel",
    "joint_vel_rel": "joint_vel_rel",
    "last_action": "last_action",
}
DIMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3, "gait_phase": 2}
JOINT_TERMS = ("joint_pos_rel", "joint_vel_rel", "last_action")

CPP = "unitree deploy C++ (deploy/include, same in unitree_rl_lab@4960b84 and unitree_rl_mjlab@1425b15)"
PHASE_CPP = (
    CPP + ": isaaclab/envs/mdp/observations/observations.h lines 125 to 151 (gait_phase adds "
    "step_dt/period to a float32 accumulator before each value and returns zeros when the command "
    "norm is below 0.1); envs/manager_based_rl_env.h lines 50 to 57 (reset zeroes the accumulator, "
    "then the observation reset evaluates every term once). The first policy input therefore sees "
    "phase 2 steps after reset, where training sees 0."
)


def _f(v: Any) -> float:
    return float(v)


def read_unitree_deploy(
    yaml_path: str | Path,
    onnx_path: str | Path | None = None,
    robot: str = "unitree_g1_29dof",
) -> tuple[Contract, list[ReaderFinding]]:
    yaml_path = Path(yaml_path)
    y = load_yaml_with_text(yaml_path)
    findings: list[ReaderFinding] = []
    if robot not in SDK_TABLES:
        raise KeyError(f"no SDK table for {robot!r}; known: {sorted(SDK_TABLES)}")
    table_name, table = SDK_TABLES[robot]
    c = Contract({"schema": SCHEMA})
    files = [{"path": yaml_path.name, "sha256": sha256_file(yaml_path)}]
    if onnx_path:
        files.append({"path": Path(onnx_path).name, "sha256": sha256_file(onnx_path)})
    c.set(
        "source",
        {
            "format": "unitree_deploy",
            "framework": {"name": None, "version": None},
            "files": files,
            "robot": robot,
        },
        "file",
    )

    # -- joints: joint_ids_map[i] is the SDK motor of policy joint i
    jmap = [int(v) for v in y["joint_ids_map"]]
    names = [table[m] for m in jmap]
    if "" in names or len(set(names)) != len(names):
        raise ValueError(f"joint_ids_map points at unused or repeated SDK slots: {jmap}")
    c.set(
        "policy_io.joints",
        {"names": names, "table": table_name, "joint_ids_map": jmap},
        "file+table",
        f"deploy.yaml joint_ids_map over {table_name}",
    )

    def sdk_list(key: str) -> dict[str, float]:
        v = list(y[key])
        if len(v) < max(jmap) + 1:
            raise ValueError(f"{key}: {len(v)} entries, joint_ids_map needs {max(jmap) + 1}")
        return {table[m]: _f(v[m]) for m in jmap}

    def policy_list(v: Any, key: str) -> dict[str, float]:
        v = list(v)
        if len(v) != len(names):
            raise ValueError(f"{key}: {len(v)} entries for {len(names)} policy joints")
        return {n: _f(x) for n, x in zip(names, v)}

    exact = "exact: the deploy runtime runs this number as printed"
    c.set(
        "timing.policy_dt",
        _f(y["step_dt"]),
        "file",
        f"deploy.yaml step_dt ({CPP}, manager_based_rl_env.h:29)",
    )
    for k in ("sim_dt", "decimation"):
        c.set(f"timing.{k}", None, "unknown", "not in deploy.yaml")
    c.set(
        "timing.order", ["obs", "infer", "target", "pd", "step"], "default", "deploy runtime step()"
    )
    c.set("timing.target_hold", "zoh", "default", "motor targets held between policy updates")

    c.set(
        "control.default_joint_pos",
        policy_list(y["default_joint_pos"], "default_joint_pos"),
        "file",
        f"deploy.yaml default_joint_pos, policy order; {exact}",
    )
    act = (y.get("actions") or {}).get("JointPositionAction") or {}
    if not act:
        raise ValueError("deploy.yaml has no actions.JointPositionAction")
    sc = act.get("scale")
    of = act.get("offset")
    c.set(
        "control.actions.joint_pos.scale",
        policy_list(sc, "scale") if sc is not None else {n: 1.0 for n in names},
        "file",
        f"deploy.yaml actions.JointPositionAction.scale (joint_actions.h:28); {exact}",
    )
    c.set(
        "control.actions.joint_pos.offset",
        policy_list(of, "offset") if of is not None else {n: 0.0 for n in names},
        "file",
        f"deploy.yaml actions.JointPositionAction.offset (joint_actions.h:31); {exact}",
    )
    clip = act.get("clip")
    c.set(
        "control.actions.joint_pos.clip",
        [[_f(a), _f(b)] for a, b in clip] if clip else None,
        "file",
        "deploy.yaml actions.JointPositionAction.clip, applied to processed targets (joint_actions.h:57)",
    )
    c.set(
        "control.actions.joint_pos.clip_stage",
        "processed" if clip else "none",
        "file",
        "joint_actions.h:54",
    )
    c.set(
        "control.actuators.kp",
        sdk_list("stiffness"),
        "file+table",
        f"deploy.yaml stiffness in SDK order, sent as motor kp (State_RLBase.h:20); {exact}",
    )
    c.set(
        "control.actuators.kd",
        sdk_list("damping"),
        "file+table",
        f"deploy.yaml damping in SDK order, sent as motor kd (State_RLBase.h:21); {exact}",
    )
    for k in ("kind", "pd_period", "integrator", "torque_limit_at"):
        c.set(
            f"control.actuators.{k}",
            None,
            "unknown",
            "the training actuator class is in no deploy file",
        )
    c.set(
        "control.ownership",
        {"owned": "all", "external": []},
        "default",
        "policy commands every joint",
    )
    for k in ("effort_limit", "velocity_limit", "armature", "joint_friction", "joint_damping"):
        c.set(f"model.{k}", None, "unknown", "not in deploy.yaml")
    c.set(
        "policy_io.imu",
        {"body": "pelvis", "frame": "body", "rotation_in_root": [1.0, 0.0, 0.0, 0.0]},
        "default",
        "deploy runtime reads the IMU gyro of the pelvis in the body frame",
    )

    # -- observations
    obs = dict(y.get("observations") or {})
    gym_history = bool(obs.pop("use_gym_history", False))
    terms: list[dict[str, Any]] = []
    hist = set()
    for key, t in obs.items():
        tid = TERM_IDS.get(key)
        if tid is None:
            raise ValueError(f"no term library entry for observation {key!r}")
        t = t or {}
        dim = DIMS.get(tid, len(names) if tid in JOINT_TERMS else None)
        scale = [_f(s) for s in t["scale"]] if t.get("scale") is not None else [1.0] * dim
        if len(scale) != dim:
            raise ValueError(f"{key}: scale has {len(scale)} entries, term has {dim}")
        params = {
            k: (_f(v) if isinstance(v, TextFloat) else v)
            for k, v in (t.get("params") or {}).items()
        }
        base = f"policy_io.observation_groups.policy.terms.{tid}"
        if tid == "gait_phase":
            params.setdefault("stand_threshold", 0.1)
            params["clock"] = "deploy_accumulator"
            params["clock_offset_steps"] = 2
            params["arithmetic"] = "float32_accumulate"
            c.provenance[base + ".params"] = _prov("default", PHASE_CPP)
        if tid == "last_action":
            params["reset"] = "zeros"
        terms.append(
            {
                "id": tid,
                "source_name": key,
                "dim": dim,
                "scale": scale,
                "clip": t.get("clip"),
                "params": params,
            }
        )
        hist.add(int(t.get("history_length", 1) or 1))
        c.provenance.setdefault(base, _prov("file", f"deploy.yaml observations.{key}"))
    if len(hist) > 1:
        raise ValueError(f"per-term history lengths differ: {sorted(hist)} (not supported)")
    c.set(
        "policy_io.observation_groups.policy",
        {
            "terms": terms,
            "history": {
                "length": hist.pop() if hist else 1,
                "layout": "time_major" if gym_history else "term_major",
                "order": "oldest_first",
                "init": "repeat_first",
            },
            "clip_then_scale": True,
        },
        "file",
        "deploy.yaml observations, in file order",
    )
    c.set(
        "policy_io.observation_groups.policy.history.layout",
        "time_major" if gym_history else "term_major",
        "default",
        f"{CPP}: observation_manager.h:72, term by term unless use_gym_history",
    )
    c.set(
        "policy_io.observation_groups.policy.history.order",
        "oldest_first",
        "default",
        f"{CPP}: manager_term_cfg.h:48 push_back, :55 read front to back",
    )
    c.set(
        "policy_io.observation_groups.policy.history.init",
        "repeat_first",
        "default",
        f"{CPP}: manager_term_cfg.h:27 reset adds the first value history_length times",
    )

    # -- commands
    ycmd = (y.get("commands") or {}).get("base_velocity") or {}
    rng = ycmd.get("ranges") or {}
    lim = {
        ax: [_f(v) for v in rng[k]]
        for ax, k in (("vx", "lin_vel_x"), ("vy", "lin_vel_y"), ("wz", "ang_vel_z"))
        if rng.get(k) is not None
    }
    c.set(
        "policy_io.commands.base_velocity.limit",
        lim or None,
        "file" if lim else "unknown",
        "deploy.yaml commands.base_velocity.ranges (the exporter writes limit_ranges under this key)",
    )
    c.set(
        "policy_io.commands.base_velocity.trained",
        None,
        "unknown",
        "training ranges are in no deploy file",
    )
    c.set(
        "policy_io.commands.base_velocity.heading",
        "off" if rng.get("heading") is None else "on",
        "file",
        "deploy.yaml ranges.heading",
    )

    # -- graph
    if onnx_path:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ins = [{"name": i.name, "shape": list(i.shape)} for i in sess.get_inputs()]
        outs = [{"name": o.name, "shape": list(o.shape)} for o in sess.get_outputs()]
        c.set("policy_io.graph.inputs", ins, "file", "ONNX graph")
        c.set("policy_io.graph.outputs", outs, "file", "ONNX graph")
        rec = [
            {"in": i["name"], "out": o["name"], "init": "zeros", "reset": "on_episode"}
            for i, o in zip(ins[1:], outs[1:])
        ]
        c.set(
            "policy_io.graph.recurrent",
            rec,
            "inferred" if rec else "file",
            "extra inputs paired with extra outputs in order" if rec else "single input",
        )
        n_in = ins[0]["shape"][-1]
        n_obs = sum(t["dim"] for t in terms) * (
            c.get("policy_io.observation_groups.policy.history.length")
        )
        if isinstance(n_in, int) and n_in != n_obs:
            findings.append(
                ReaderFinding(
                    "policy_io.graph.inputs",
                    "conflicting",
                    f"ONNX takes {n_in} inputs, deploy.yaml builds {n_obs}",
                )
            )
        meta = dict(sess.get_modelmeta().custom_metadata_map)
        if meta:
            c.set("source.onnx_metadata_keys", sorted(meta), "file", "ONNX custom metadata")
    return c, findings


def _prov(source: str, detail: str):
    from ..contract import Provenance

    return Provenance(source, detail)
