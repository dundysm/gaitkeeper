# Shaped like legged_gym's legged_robot_config.py (values written for the tests).
from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 48
        num_actions = 12

    class commands:
        heading_command = True

        class ranges:
            lin_vel_x = [-1.0, 1.0]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1, 1]
            heading = [-3.14, 3.14]

    class init_state:
        default_joint_angles = {"joint_a": 0.0}

    class control:
        control_type = "P"
        stiffness = {"joint_a": 10.0}
        damping = {"joint_a": 1.0}
        action_scale = 0.5
        decimation = 4

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class sim:
        dt = 0.005


class LeggedRobotCfgPPO:
    class policy:
        init_noise_std = 1.0
