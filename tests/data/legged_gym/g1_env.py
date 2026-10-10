# Shaped like unitree_rl_gym's g1_env.py: only the gait period line matters to the reader.
class G1Robot:
    def _post_physics_step_callback(self):
        period = 0.8  # noqa: F841
        offset = 0.5  # noqa: F841
