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


def speed_to_distance(cmd: np.ndarray, task: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Keep the harness's direction, set the speed from the distance (a port's own approach;
    falcon and homie in teleop-walking-benchmark).

    While the harness's planar command is nonzero, its direction is kept at
    min(pos_p * distance, speed_cap) m/s, vx clamped to ``vx`` and vy to +-vy_abs; a zero
    planar command stays zero. The yaw rate is the harness's (``yaw: "pass"``), or the port's:
    yaw_p times the heading to aim at, the direction of travel blended in by distance between
    face_near_m and face_far_m, else the yaw error while the harness still turns (zero once
    its yaw command is zero), clamped to yaw_rate_abs."""
    c = np.asarray(cmd, dtype=np.float64)[:3]
    out = c.copy()
    dist, yaw_err = (float(x) for x in np.asarray(task, dtype=np.float64)[:2])
    moving = c[0] != 0.0 or c[1] != 0.0
    bearing = 0.0
    if moving:
        bearing = math.atan2(c[1], c[0])
        speed = min(float(p["pos_p"]) * dist, float(p["speed_cap"]))
        lo, hi = (float(x) for x in p["vx"])
        vy = float(p["vy_abs"])
        out[0] = min(max(speed * math.cos(bearing), lo), hi)
        out[1] = min(max(speed * math.sin(bearing), -vy), vy)
    yaw = p.get("yaw", "pass")
    if yaw == "pass":
        return out
    face_w = 0.0
    if moving:
        near, far = float(yaw["face_near_m"]), float(yaw["face_far_m"])
        face_w = min(max((dist - near) / (far - near), 0.0), 1.0)
    wz = float(yaw["yaw_rate_abs"])
    if face_w > 0.0:
        aim = _wrap(yaw_err + face_w * _wrap(bearing - yaw_err))
        out[2] = min(max(float(yaw["yaw_p"]) * aim, -wz), wz)
    elif c[2] != 0.0:
        out[2] = min(max(float(yaw["yaw_p"]) * yaw_err, -wz), wz)
    return out


SHAPERS = {"waypoint_follow": waypoint_follow, "speed_to_distance": speed_to_distance}


class CommandGate:
    """When a port passes its command to the policy at all (``base_velocity.gate``); the
    policy sees the command times the gate, and the gate itself as a fourth element (the
    ``command_gate`` term reads it).

    ``on: command_nonzero``: open while any component of the harness's command is nonzero
    (falcon's stand flag). ``on: task_latch``: opens when the waypoint is farther than
    ``enter.dist`` or the yaw error exceeds ``enter.yaw``, and closes once both are within
    ``exit`` (asap's walk latch). Either stays shut for ``warmup_s`` after an episode starts."""

    def __init__(self, spec: dict[str, Any], policy_dt: float) -> None:
        self.spec = spec
        self.dt = float(policy_dt)
        self.on = spec.get("on", "command_nonzero")
        if self.on not in ("command_nonzero", "task_latch"):
            raise ValueError(f"policy_io.commands.base_velocity.gate: unknown on {self.on!r}")
        self.warmup = float(spec.get("warmup_s", 0.0))
        self.latched = False

    def reset(self) -> None:
        self.latched = False

    def __call__(self, cmd: np.ndarray, task: Any, episode_step: int) -> float:
        if episode_step == 0:
            self.reset()
        if self.on == "command_nonzero":
            c = np.asarray(cmd, dtype=np.float64)[:3]
            open_ = bool(np.any(c != 0.0))
        else:
            t = np.asarray(ZERO_TASK if task is None else task, dtype=np.float64)
            dist, yaw = float(t[0]), abs(float(t[1]))
            ent, ex = self.spec["enter"], self.spec["exit"]
            if self.latched:
                if dist < float(ex["dist"]) and yaw < float(ex["yaw"]):
                    self.latched = False
            elif dist > float(ent["dist"]) or yaw > float(ent["yaw"]):
                self.latched = True
            open_ = self.latched
        warm = episode_step * self.dt >= self.warmup - 1e-9
        return 1.0 if (open_ and warm) else 0.0


def policy_command(
    cmd: np.ndarray,
    task: Any,
    shaping: dict[str, Any] | None,
    gate: CommandGate | None = None,
    episode_step: int = 0,
) -> np.ndarray:
    """What the policy is given as its command: shaped, then gated (with the gate as a
    fourth element) when the contract has a gate."""
    out = shape(cmd, task, shaping)
    if gate is None:
        return out
    g = gate(cmd, task, episode_step)
    return np.r_[out[:3] * g, g]


def shape(cmd: np.ndarray, task: Any, shaping: dict[str, Any] | None) -> np.ndarray:
    """The command the policy sees: ``cmd`` itself, or the port's shaping of it."""
    if not shaping:
        return np.asarray(cmd, dtype=np.float64)
    kind = shaping.get("kind")
    if kind not in SHAPERS:
        raise ValueError(f"policy_io.commands.base_velocity.shaping: unknown kind {kind!r}")
    t = np.asarray(ZERO_TASK if task is None else task, dtype=np.float64)
    return SHAPERS[kind](cmd, t, shaping.get("params", {}))
