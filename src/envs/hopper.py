"""Hopper (Gymnasium's Hopper-v5 model) with the Walker2d task's cost structure.

Same state layout as Walker2d (time, rootx, rootz, rooty, joints..., velocities), so the Walker2d
cost, reset, step and rollout scoring apply unchanged; only the model and the health limits
differ (Gymnasium's: torso height above 0.7 m, torso angle within +-0.2 rad, every other state
entry within +-100).
"""

from pathlib import Path

import numpy as np

from src.envs.walker2d import Walker2d

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "hopper.xml")
_HEALTHY_Z_MIN = 0.7
_HEALTHY_ANGLE = 0.2

# the Walker2d task's cost weights (diagnostic_policies.NATIVE_TASK) at a hopping speed
HOPPER_TASK = {
    "frame_skip": 4,
    "cost_style": "target_velocity",
    "target_velocity": 1.0,
    "vel_cost_weight": 2.0,
    "vertical_velocity_cost_weight": 0.5,
    "angle_cost_weight": 0.1,
    "unhealthy_cost_weight": 5.0,
    "ctrl_cost_weight": 0.001,
    "terminal_unhealthy_weight": 20.0,
    "apply_terminal_cost_on_done": True,
}


class Hopper(Walker2d):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_path", _XML)
        super().__init__(**kwargs)

    @staticmethod
    def _is_healthy(z, angle):
        return (np.asarray(z) > _HEALTHY_Z_MIN) & (np.abs(np.asarray(angle)) < _HEALTHY_ANGLE)
