"""E4: the training drive's discretization terms from traces at two step sizes
(plan Appendix B).

Each stock trace gives per joint ``b0`` (armature term) and ``dB0`` (damping
term) from the residual's linear fit. One step size cannot tell a real
armature difference from the drive's discretization, so the stock traces at
two step sizes are fitted together:

    b0_j(h)  = c_j + alpha * h * kd_j + beta * h^2 * kp_j
    dB0_j(h) = d_j + gamma * h * kp_j

with one ``c_j`` and ``d_j`` per joint (the real difference) and ``alpha``,
``beta``, ``gamma`` shared. An injected ankle change gives ``b1 - b0`` on the
ankles. The fit's residual decides what may ship: explained (subtract and
print the correction), or not explained (detection only, no localization).
Joints with joint friction are left out: their armature comes from a search,
not from the linear fit.

With two step sizes ``alpha`` and ``beta`` are separable only when ``kd / kp``
differs between joints; with one ratio for every joint (gains set from one
natural frequency, as mjlab does) they are not, and the result says so: a
third step size separates them for any gains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contract import Contract
from .residual import dynamics_residual
from .trace import Trace

E4_FIT_RESID = 2e-4  # armature units; 2% of the 0.01 ankle check. Fixed before any PhysX data.
ANKLE_TOL = 0.2  # b1 - b0 within 20% of the injected change


@dataclass
class E4Result:
    steps: list[float]
    joints: list[str]
    b0: dict[float, dict[str, float]]
    dB0: dict[float, dict[str, float]]
    c: dict[str, float] = field(default_factory=dict)
    d: dict[str, float] = field(default_factory=dict)
    alpha: float = float("nan")
    beta: float = float("nan")
    gamma: float = float("nan")
    resid_b: float = float("nan")
    resid_dB: float = float("nan")
    ankle: dict[str, float] = field(default_factory=dict)
    ankle_expected: float | None = None
    left_out: list[str] = field(default_factory=list)
    identifiable: bool = True

    @property
    def explained(self) -> bool:
        return self.identifiable and self.resid_b <= E4_FIT_RESID

    @property
    def ankle_ok(self) -> bool | None:
        if not self.ankle or self.ankle_expected is None:
            return None
        e = self.ankle_expected
        return all(abs(v - e) <= ANKLE_TOL * abs(e) for v in self.ankle.values())

    def lines(self) -> list[str]:
        if not self.identifiable:
            return [
                f"E4 over steps {', '.join(f'{h * 1e3:g} ms' for h in self.steps)}: the armature "
                "terms h kd and h^2 kp cannot be told apart (kd / kp is the same on every joint); "
                "record a third step size. Detection only, no localization."
            ]
        out = [
            f"E4 over steps {', '.join(f'{h * 1e3:g} ms' for h in self.steps)}, "
            f"{len(self.joints)} joints"
            + (
                f" ({len(self.left_out)} left out: joint friction or no fit)"
                if self.left_out
                else ""
            ),
            f"  shared terms: alpha {self.alpha:+.3f} (h kd), beta {self.beta:+.3f} (h^2 kp), "
            f"gamma {self.gamma:+.3f} (h kp, damping)",
            f"  fit residual: armature {self.resid_b:.2e}, damping {self.resid_dB:.2e} "
            + (
                "(explained: subtract and print the correction)"
                if self.explained
                else "(not explained: detection only, no localization)"
            ),
        ]
        big = sorted(self.c.items(), key=lambda x: -abs(x[1]))[:6]
        out.append(
            "  real armature difference c_j, largest: " + ", ".join(f"{k} {v:+.4f}" for k, v in big)
        )
        if self.ankle:
            ok = self.ankle_ok
            out.append(
                "  ankles b1 - b0: "
                + ", ".join(f"{k} {v:+.4f}" for k, v in self.ankle.items())
                + (
                    f" (expected {self.ankle_expected:+.4f}: {'pass' if ok else 'FAIL'})"
                    if ok is not None
                    else ""
                )
            )
        return out

    def to_json(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "resid_armature": self.resid_b,
            "resid_damping": self.resid_dB,
            "explained": self.explained,
            "c": self.c,
            "d": self.d,
            "b0": {str(h): v for h, v in self.b0.items()},
            "dB0": {str(h): v for h, v in self.dB0.items()},
            "ankle_b1_minus_b0": self.ankle,
            "ankle_expected": self.ankle_expected,
            "ankle_ok": self.ankle_ok,
            "left_out": self.left_out,
            "identifiable": self.identifiable,
        }


def _fits(trace: Trace, contract: Contract, target: Any) -> dict[str, dict[str, Any]]:
    d = dynamics_residual(trace, contract, target, floors={}, fits=True)
    return {f["joint"]: f for f in d.fits or [] if f.get("method") == "linear fit"}


def e4(
    stock: list[Trace],
    contract: Contract,
    target: Any,
    ankle: Trace | None = None,
    ankle_change: float | None = None,
) -> E4Result:
    """``stock``: traces at two or more step sizes, same schedule and seed.
    ``ankle``: a trace at the first stock step with the ankle armature changed
    by ``ankle_change`` in the source."""
    steps = [float(t.meta["sim_dt"]) for t in stock]
    if len(set(steps)) < 2:
        raise ValueError("E4 needs stock traces at two step sizes")
    fits = {h: _fits(t, contract, target) for h, t in zip(steps, stock)}
    names = list(next(iter(fits.values())))
    joints = [n for n in names if all(fits[h].get(n, {}).get("status") == "fit" for h in steps)]
    left = [n for n in names if n not in joints]
    kp, kd = contract.get("control.actuators.kp"), contract.get("control.actuators.kd")
    J, H = len(joints), len(steps)
    A = np.zeros((J * H, J + 2))
    yb = np.zeros(J * H)
    Ad = np.zeros((J * H, J + 1))
    yd = np.zeros(J * H)
    for k, h in enumerate(steps):
        for i, n in enumerate(joints):
            row = k * J + i
            A[row, i] = 1.0
            A[row, J] = h * kd[n]
            A[row, J + 1] = h * h * kp[n]
            yb[row] = fits[h][n]["dI"]
            Ad[row, i] = 1.0
            Ad[row, J] = h * kp[n]
            yd[row] = fits[h][n]["dB"]
    sv = np.linalg.svd(A / np.linalg.norm(A, axis=0), compute_uv=False)
    identifiable = bool(sv[-1] > 1e-6 * sv[0])
    xb, *_ = np.linalg.lstsq(A, yb, rcond=None)
    xd, *_ = np.linalg.lstsq(Ad, yd, rcond=None)
    res = E4Result(
        steps,
        joints,
        {h: {n: fits[h][n]["dI"] for n in joints} for h in steps},
        {h: {n: fits[h][n]["dB"] for n in joints} for h in steps},
        c={n: float(xb[i]) for i, n in enumerate(joints)},
        d={n: float(xd[i]) for i, n in enumerate(joints)},
        alpha=float(xb[J]),
        beta=float(xb[J + 1]),
        gamma=float(xd[J]),
        resid_b=float(np.sqrt(np.mean((A @ xb - yb) ** 2))),
        resid_dB=float(np.sqrt(np.mean((Ad @ xd - yd) ** 2))),
        left_out=left,
        identifiable=identifiable,
    )
    if ankle is not None:
        h0 = float(ankle.meta["sim_dt"])
        if h0 not in fits:
            raise ValueError("the ankle trace must use one of the stock step sizes")
        f1 = _fits(ankle, contract, target)
        res.ankle = {
            n: float(f1[n]["dI"] - fits[h0][n]["dI"])
            for n in joints
            if "ankle" in n and f1.get(n, {}).get("status") == "fit"
        }
        res.ankle_expected = -ankle_change if ankle_change is not None else None
    return res
