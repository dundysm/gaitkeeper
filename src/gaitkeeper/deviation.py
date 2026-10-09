"""What the robot runs against what the policy was trained with (boundary C, and A
for the phase clock).

The deploy contract (from deploy.yaml, read as exact) is compared with a
reference contract: a live one recorded from the training env, or the ONNX
metadata. Differences are deviations of the deployment, reported per joint,
and turned into target differences over a trace's actions when one is given.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import Contract

FIELDS = (
    ("scale", "control.actions.joint_pos.scale"),
    ("offset", "control.actions.joint_pos.offset"),
    ("default", "control.default_joint_pos"),
    ("kp", "control.actuators.kp"),
    ("kd", "control.actuators.kd"),
)


@dataclass
class Deviation:
    names: list[str]
    fields: dict[str, dict[str, np.ndarray]]  # field -> {"deploy", "reference"}
    target: dict[str, np.ndarray] = field(default_factory=dict)  # "max", "rms" per joint (rad)
    phase: dict[str, Any] | None = None
    reference_name: str = "reference"
    reference_resolution: dict[str, float] = field(default_factory=dict)

    def rel(self, f: str) -> np.ndarray:
        v = self.fields[f]
        ref = v["reference"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                ref != 0,
                (v["deploy"] - ref) / np.abs(ref),
                np.where(v["deploy"] == ref, 0.0, np.inf),
            )

    def lines(self, top: int = 40) -> list[str]:
        out = [
            f"deploy.yaml against {self.reference_name} (boundary C deviation: the robot runs the deploy values)"
        ]
        for f, _ in FIELDS:
            v = self.fields.get(f)
            if v is None:
                continue
            d = v["deploy"] - v["reference"]
            res = self.reference_resolution.get(f, 0.0)
            # float32 storage of a short decimal (0.3 as 0.30000001) is not a deviation
            tol = res + 1e-6 * np.maximum(np.abs(v["reference"]), 1e-3)
            idx = np.flatnonzero(np.abs(d) > tol)
            if not len(idx):
                why = f"resolution {res:g}" if res else "float32 rounding"
                out.append(f"{f}: no joint differs beyond the reference's {why}")
                continue
            r = self.rel(f)
            worst = idx[np.argsort(-np.abs(r[idx]))]
            out.append(
                f"{f}: {len(idx)} of {len(self.names)} joints differ; largest relative "
                f"{100 * r[worst[0]]:+.2f}% ({self.names[worst[0]]})"
            )
            seen = set()
            for i in worst[:top]:
                key = (f, round(float(v["deploy"][i]), 9), round(float(v["reference"][i]), 9))
                group = [
                    self.names[j]
                    for j in worst
                    if (f, round(float(v["deploy"][j]), 9), round(float(v["reference"][j]), 9))
                    == key
                ]
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    f"  {_short(group)}: deploy {v['deploy'][i]:.6g}, {self.reference_name} "
                    f"{v['reference'][i]:.10g}, diff {d[i]:+.4g} ({100 * r[i]:+.2f}%)"
                )
        if self.target:
            mx, rms = self.target["max"], self.target["rms"]
            order = np.argsort(-mx)
            out.append(
                "target difference over the trace's actions, a * (scale_deploy - scale_ref) + offset diff:"
            )
            for i in order[:6]:
                if mx[i] <= 0:
                    break
                out.append(
                    f"  {self.names[i]}: max {mx[i]:.4g} rad ({np.degrees(mx[i]):.3f} deg), rms {rms[i]:.3g} rad"
                )
        if self.phase:
            p = self.phase
            out.append(
                f"phase clock: deploy runs {p['lead_steps']:+d} policy steps ({p['lead_s'] * 1e3:.0f} ms, "
                f"{p['lead_deg']:.0f} deg of a {p['period']:g} s gait) ahead of training (boundary A deviation)"
            )
        return out


def _short(names: list[str]) -> str:
    sides = {n.replace("left_", "").replace("right_", "") for n in names}
    if len(names) == 2 * len(sides) and len(sides) == 1:
        return f"{next(iter(sides))} (both sides)"
    return ", ".join(names) if len(names) <= 3 else f"{names[0]} and {len(names) - 1} more"


def _vals(c: Contract, path: str, names: list[str]) -> np.ndarray | None:
    v = c.get(path, None)
    if v is None:
        return None
    return np.array([float(v[n]) for n in names])


def compare(
    deploy: Contract,
    reference: Contract,
    actions: np.ndarray | None = None,
    action_names: list[str] | None = None,
    reference_name: str = "reference",
) -> Deviation:
    names = list(deploy.get("policy_io.joints.names"))
    ref_names = set(reference.get("policy_io.joints.names"))
    missing = [n for n in names if n not in ref_names]
    if missing:
        raise KeyError(f"reference lacks joints {missing}")
    fields: dict[str, dict[str, np.ndarray]] = {}
    res: dict[str, float] = {}
    for f, path in FIELDS:
        a, b = _vals(deploy, path, names), _vals(reference, path, names)
        if a is None or b is None:
            continue
        fields[f] = {"deploy": a, "reference": b}
        res[f] = float(reference.prov(path).resolution or 0.0)
    dev = Deviation(names, fields, reference_name=reference_name, reference_resolution=res)
    if actions is not None and "scale" in fields:
        an = action_names or names
        idx = [an.index(n) for n in names]
        a = np.asarray(actions, dtype=float)[:, idx]
        ds = fields["scale"]["deploy"] - fields["scale"]["reference"]
        do = (
            (fields["offset"]["deploy"] - fields["offset"]["reference"])
            if "offset" in fields
            else 0.0
        )
        dt = a * ds[None] + do
        dev.target = {"max": np.abs(dt).max(axis=0), "rms": np.sqrt((dt**2).mean(axis=0))}
    dev.phase = phase_lead(deploy, reference)
    return dev


def _phase_params(c: Contract) -> dict[str, Any] | None:
    for t in c.get("policy_io.observation_groups.policy.terms", []) or []:
        if t["id"] == "gait_phase":
            return t.get("params", {})
    return None


def phase_lead(deploy: Contract, reference: Contract) -> dict[str, Any] | None:
    a, b = _phase_params(deploy), _phase_params(reference)
    if a is None or b is None:
        return None
    lead = int(a.get("clock_offset_steps", 0)) - int(b.get("clock_offset_steps", 0))
    if lead == 0:
        return None
    dt = float(deploy.get("timing.policy_dt"))
    period = float(a["period"])
    return {
        "lead_steps": lead,
        "lead_s": lead * dt,
        "lead_deg": 360.0 * lead * dt / period,
        "period": period,
        "detail": deploy.prov("policy_io.observation_groups.policy.terms.gait_phase.params").detail,
    }
