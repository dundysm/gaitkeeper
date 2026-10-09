"""Command schedules for golden traces, and the excitation checklist (plan 7.6)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (start_s, vx, vy, wz). Each row holds until the next start.
DEFAULT_SCHEDULE: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),  # stand, zero command
    (1.5, 0.05, 0.0, 0.0),  # small commands, below the 0.1 stand threshold
    (3.0, 0.0, 0.06, 0.0),
    (4.5, 0.0, 0.0, 0.08),
    (6.0, 0.5, 0.0, 0.0),
    (9.0, 1.5, 0.0, 0.0),  # fast, to load the legs
    (11.0, 0.0, 0.4, 0.0),
    (13.0, 0.0, -0.4, 0.0),
    (15.0, 0.0, 0.0, 0.5),  # in-place yaw
    (17.5, 0.4, 0.0, 0.5),  # turn while walking, same direction
    (20.0, 0.0, 0.0, 0.0),  # the episode resets near here (episode length 20 s)
    (22.0, 0.15, 0.0, 0.0),  # small, above the stand threshold
    (24.0, 0.6, 0.2, 0.4),
    (27.0, 0.0, 0.0, 0.5),
    (30.0, -0.4, 0.0, 0.0),
    (32.0, 0.8, -0.3, 0.3),
    (34.0, 0.0, 0.0, 0.0),
]


# For unitree_rl_lab's G1 velocity policy: every command inside its deploy.yaml limits
# (vx -0.5 to 1.0, vy +-0.3, wz +-0.2). Small commands below and above the 0.1 stand
# threshold, in-place yaw at the limit, and fourteen seconds of walking while turning,
# which is what moves the heading past 90 degrees: in MuJoCo this policy turns at about
# 0.13 rad/s while walking and barely in place. Recorded with a 24 s episode, so the
# reset falls between the two halves.
RL_LAB_SCHEDULE: list[tuple[float, float, float, float]] = [
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.05, 0.0, 0.0),
    (2.5, 0.0, 0.06, 0.0),
    (4.0, 0.0, 0.0, 0.08),
    (5.0, 0.15, 0.0, 0.0),  # between the stand threshold and the measured dead zone edge
    (6.0, 0.0, 0.0, 0.2),  # in-place yaw at the limit
    (7.5, 0.6, 0.0, 0.2),  # walk and turn: the heading sweep
    (21.5, 0.0, 0.0, 0.0),  # the episode resets at 24 s
    (25.0, 1.0, 0.0, 0.0),
    (26.5, 0.0, 0.3, 0.0),
    (28.0, 0.0, -0.3, 0.0),
    (29.5, 0.0, 0.0, -0.2),
    (31.0, -0.5, 0.0, 0.0),
    (32.5, 0.8, -0.2, -0.2),
    (35.0, 0.0, 0.0, 0.0),
]
RL_LAB_EPISODE_S = 24.0
RL_LAB_PUSHES: list[tuple[float, float, str, tuple[float, float, float]]] = [
    (12.0, 0.15, "torso_link", (0.0, 300.0, 0.0)),  # sideways while walking and turning
    (33.5, 0.15, "torso_link", (-300.0, 0.0, 0.0)),  # backward while walking forward and turning
]


# (start_s, duration_s, body, (fx, fy, fz) in N, world frame). Pushes load the
# legs past what walking needs so that a torque limit is reached (plan 7.6).
DEFAULT_PUSHES: list[tuple[float, float, str, tuple[float, float, float]]] = [
    (7.5, 0.15, "torso_link", (0.0, 300.0, 0.0)),  # sideways while walking at 0.5 m/s
    (25.5, 0.15, "torso_link", (-300.0, 0.0, 0.0)),  # backward while walking forward and turning
]


def push_at(
    pushes: list[tuple[float, float, str, tuple[float, float, float]]], t: float
) -> tuple[str, np.ndarray] | None:
    for start, dur, body, f in pushes:
        if start - 1e-9 <= t < start + dur - 1e-9:
            return body, np.asarray(f, dtype=np.float64)
    return None


def command_at(schedule: list[tuple[float, float, float, float]], t: float) -> np.ndarray:
    row = schedule[0]
    for r in schedule:
        if r[0] <= t + 1e-9:
            row = r
    return np.array(row[1:], dtype=np.float64)


@dataclass
class ExcitationItem:
    name: str
    ok: bool
    value: str


def excitation_checklist(
    command: np.ndarray,
    root_quat: np.ndarray,
    ang_vel_body: np.ndarray,
    reset: np.ndarray,
    effort_at_limit_share: float | None,
    stand_threshold: float = 0.1,
    gyro_noise: float = 0.05,
) -> list[ExcitationItem]:
    """The items plan 7.6 requires before a trace is written."""
    from ..terms import yaw_of

    items = []
    for i, ax in enumerate(("vx", "vy", "wz")):
        vals = np.unique(np.round(command[:, i], 6))
        items.append(
            ExcitationItem(f"command {ax} changes", len(vals) >= 3, f"{len(vals)} distinct values")
        )
    norm = np.linalg.norm(command, axis=1)
    small = (norm > 0) & (norm < stand_threshold)
    items.append(
        ExcitationItem(
            "small commands inside the stand threshold",
            bool(small.any()),
            f"{int(small.sum())} steps",
        )
    )
    in_place = (np.abs(command[:, 2]) > stand_threshold) & (
        np.linalg.norm(command[:, :2], axis=1) == 0
    )
    items.append(
        ExcitationItem("in-place yaw command", bool(in_place.any()), f"{int(in_place.sum())} steps")
    )
    yaw = np.unwrap(yaw_of(root_quat))
    seg_span = 0.0
    start = 0
    for k in range(1, len(yaw) + 1):
        if k == len(yaw) or reset[k]:
            seg = yaw[start:k]
            seg_span = max(seg_span, float(seg.max() - seg.min()) if len(seg) else 0.0)
            start = k
    items.append(
        ExcitationItem(
            "heading spans more than 90 degrees",
            seg_span > np.pi / 2,
            f"{np.degrees(seg_span):.0f} degrees within one episode",
        )
    )
    rp = np.sqrt(np.mean(ang_vel_body[:, :2] ** 2, axis=0))
    items.append(
        ExcitationItem(
            "roll and pitch rates above noise",
            bool((rp > gyro_noise).all()),
            f"rms roll {rp[0]:.3f}, pitch {rp[1]:.3f} rad/s",
        )
    )
    n_reset = int(np.asarray(reset[1:], dtype=bool).sum())
    items.append(
        ExcitationItem("at least one reset inside the trace", n_reset >= 1, f"{n_reset} resets")
    )
    if effort_at_limit_share is None:
        items.append(ExcitationItem("load touches a torque limit", False, "effort limits unknown"))
    else:
        items.append(
            ExcitationItem(
                "load touches a torque limit",
                effort_at_limit_share > 0,
                f"{effort_at_limit_share:.2%} of joint-steps at the limit",
            )
        )
    return items
