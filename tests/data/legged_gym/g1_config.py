# Shaped like unitree_rl_gym's g1_config.py (values written for the tests).
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class G1RoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        default_joint_angles = {
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0,
            "left_hip_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0,
            "right_hip_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_actions = 12

    class control(LeggedRobotCfg.control):
        stiffness = {"hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40}
        damping = {"hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2}
        action_scale = 0.25


class G1RoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        rnn_type = "lstm"
