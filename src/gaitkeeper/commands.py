"""Command shaping: when a port turns the harness's task into its own velocity command.

Most ports pass the harness's velocity command straight to the policy. Some steer from the
task instead (the distance and bearing to the current waypoint), and the policy then sees
the port's command, not the harness's. The contract records that as
``policy_io.commands.base_velocity.shaping``; the runner applies it before building the
observation, so the policy sees what the port would have given it.

The task is the harness's four numbers (teleop-walking-benchmark main.cpp): distance to the
waypoint (m), yaw error (rad), and the waypoint in the body frame (x, y). It is all zeros
when there is no waypoint (before the tour starts and after it ends).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

ZERO_TASK = (0.0, 0.0, 0.0, 0.0)


def _wrap(a: float) -> float:
    return a - 2.0 * math.pi * round(a / (2.0 * math.pi))


def waypoint_follow(cmd: np.ndarray, task: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Walk toward the waypoint at a capped speed, facing it when it is far and the target
    heading when it is near (a port's own steering; zealot in teleop-walking-benchmark).

    With a waypoint (distance > 0): unless the harness's planar command is zero (position
    reached), vx = max(cos bearing, 0) * min(walk_p * distance, walk_speed) and vy the same
    speed times sin bearing, clamped to vy_abs. The yaw rate is yaw_p times the heading to aim
    at: the bearing, blended in by distance between face_near_m and face_far_m, else the yaw
    error (zero once the harness's yaw command is zero), clamped to yaw_rate_abs."""
    c = np.asarray(cmd, dtype=np.float64)[:3]
    out = c.copy()
    dist, yaw_err, bx, by = (float(x) for x in np.asarray(task, dtype=np.float64)[:4])
    if not dist > 0.0:
        return out
    pos_reached = math.hypot(c[0], c[1]) < 1e-9
    yaw_reached = abs(c[2]) < 1e-9
    bearing, face_w = 0.0, 0.0
    if not pos_reached:
        bearing = math.atan2(by, bx)
        near, far = float(p["face_near_m"]), float(p["face_far_m"])
        face_w = min(max((dist - near) / (far - near), 0.0), 1.0)
        gate = max(math.cos(bearing), 0.0)
        speed = gate * min(float(p["walk_p"]) * dist, float(p["walk_speed"]))
        vy = float(p["vy_abs"])
        out[0] = speed
        out[1] = min(max(speed * math.sin(bearing), -vy), vy)
    if face_w > 0.0:
        aim = _wrap(yaw_err + face_w * _wrap(bearing - yaw_err))
    else:
        aim = 0.0 if yaw_reached else yaw_err
    wz = float(p["yaw_rate_abs"])
    out[2] = min(max(float(p["yaw_p"]) * aim, -wz), wz)
    return out


SHAPERS = {"waypoint_follow": waypoint_follow}


def shape(cmd: np.ndarray, task: Any, shaping: dict[str, Any] | None) -> np.ndarray:
    """The command the policy sees: ``cmd`` itself, or the port's shaping of it."""
    if not shaping:
        return np.asarray(cmd, dtype=np.float64)
    kind = shaping.get("kind")
    if kind not in SHAPERS:
        raise ValueError(f"policy_io.commands.base_velocity.shaping: unknown kind {kind!r}")
    t = np.asarray(ZERO_TASK if task is None else task, dtype=np.float64)
    return SHAPERS[kind](cmd, t, shaping.get("params", {}))
