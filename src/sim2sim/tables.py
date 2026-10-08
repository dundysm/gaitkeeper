"""Versioned joint tables. A permutation the comparator finds is named against these."""

from __future__ import annotations

# Unitree G1 29-DoF SDK motor order (unitree_sdk2), which is also the MJCF order
# of unitree_mujoco and unitree_rl_mjlab.
G1_29_SDK = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# unitree_rl_lab deploy.yaml joint_ids_map at 4960b84: Isaac Lab's breadth-first
# order, as SDK indices.
G1_29_ISAAC_IDS = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
                   17, 24, 18, 25, 19, 26, 20, 27, 21, 28]  # fmt: skip
G1_29_ISAAC = [G1_29_SDK[i] for i in G1_29_ISAAC_IDS]

TABLES: dict[str, list[str]] = {
    "unitree_g1_29dof_sdk": G1_29_SDK,
    "unitree_g1_29dof_isaac_bfs@unitree_rl_lab:4960b84": G1_29_ISAAC,
}


def name_permutation(perm_names: list[str], policy_names: list[str]) -> str | None:
    """Name the order a harness used, if it matches a known table.

    ``perm_names[i]`` is the joint whose value appeared in policy slot i.
    """
    for tname, table in TABLES.items():
        if set(table) != set(policy_names):
            continue
        # Harness wrote values in table order into slots meant for policy order.
        if [table[i] for i in range(len(table))] == perm_names:
            return f"values in {tname} order written to policy-order slots"
        # Harness read policy-order values as if they were in table order.
        pos = {n: i for i, n in enumerate(table)}
        inv = [policy_names[pos[n]] for n in policy_names]
        if inv == perm_names:
            return f"policy-order values reindexed as {tname} order"
    return None
