"""Dense, native Acrobot task for the feedback-controller loop.

The preserved dm_control model extends both links along +z: q=(0, 0)
is upright and q=(pi, 0) is hanging. Full states are
``[time, shoulder, elbow, shoulder_velocity, elbow_velocity]``.
This task deliberately differs from the historical, sensor-based Acrobot
cost: squared tip distance plus .01 velocity and control penalties, with
no terminal shaping. Costs are evaluated at POST-action boundaries.
"""

import math
from pathlib import Path

import mujoco
import numpy as np

from src.envs.mujoco_env import MuJoCoEnv

MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml"
TARGET = np.array([0.0, 0.0, 4.0])
RESET_STD = 0.05
BALANCE_SECONDS = 2.0
BALANCE_HEIGHT = 3.8


def tip_position(states: np.ndarray) -> np.ndarray:
    """Exact tip geometry from full states, including arbitrary batch axes."""
    states = np.asarray(states)
    if states.ndim < 1 or states.shape[-1] != 5:
        raise ValueError("Acrobot full states must have final dimension 5.")
    shoulder = states[..., 1]
    absolute_elbow = shoulder + states[..., 2]
    x = np.sin(shoulder) + np.sin(absolute_elbow)
    z = 2.0 + np.cos(shoulder) + np.cos(absolute_elbow)
    return np.stack([x, np.zeros_like(x), z], axis=-1)


def dense_cost(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Squared tip error plus .01||qdot||² + .01||u||², per control."""
    states = np.asarray(states)
    actions = np.asarray(actions)
    if actions.ndim < 1 or actions.shape[-1] != 1:
        raise ValueError("Acrobot actions must have final dimension 1.")
    position_cost = np.sum((tip_position(states) - TARGET) ** 2, axis=-1)
    velocity_cost = 0.01 * np.sum(states[..., 3:5] ** 2, axis=-1)
    action_cost = 0.01 * np.sum(actions**2, axis=-1)
    return position_cost + velocity_cost + action_cost


def balanced(
    states: np.ndarray,
    *,
    dt: float,
    hold_seconds: float = BALANCE_SECONDS,
    minimum_height: float = BALANCE_HEIGHT,
) -> np.ndarray:
    """Height-only success over the final complete hold window.

    ``states`` contains POST-action observations, shaped (..., T, 5).
    The number of controls is ceil(hold_seconds / dt), so changing the
    control period cannot silently shorten the two-second success window.
    This criterion is intentionally not a low-speed capture criterion;
    callers report velocities separately. Short or non-finite suffixes fail.
    """
    states = np.asarray(states)
    if states.ndim < 2 or states.shape[-1] != 5:
        raise ValueError("A trace must have shape (..., T, 5).")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive.")
    if not np.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("hold_seconds must be finite and positive.")
    hold_controls = math.ceil(hold_seconds / dt)
    if states.shape[-2] < hold_controls:
        return np.zeros(states.shape[:-2], dtype=bool)
    suffix = states[..., -hold_controls:, :]
    return np.isfinite(suffix).all(axis=(-1, -2)) & (
        tip_position(suffix)[..., 2] >= minimum_height
    ).all(axis=-1)


class Acrobot(MuJoCoEnv):
    """Elbow-only actuation, native bounded torques, and private reset RNG."""

    _w_unhealthy = 0.0
    _apply_terminal_cost_on_done = False

    def __init__(
        self,
        frame_skip: int = 2,
        nthread: int | None = None,
        seed: int | None = None,
        reset_std: float = RESET_STD,
    ) -> None:
        if frame_skip not in (1, 2):
            raise ValueError("frame_skip must be 1 or 2 (100 or 50 Hz controls).")
        if not np.isfinite(reset_std) or reset_std < 0:
            raise ValueError("reset_std must be finite and nonnegative.")
        self._reset_std = reset_std
        self._reset_rng = np.random.default_rng(seed)
        super().__init__(str(MODEL_PATH), frame_skip=frame_skip, nthread=nthread)
        if (self.model.nq, self.model.nv, self.model.nu, self._nstate) != (2, 2, 1, 5):
            raise ValueError("The frozen Acrobot model must have dimensions (2, 2, 1, 5).")

    def reset(
        self,
        state: np.ndarray | None = None,
        *,
        seed: int | None = None,
    ) -> np.ndarray:
        if state is not None:
            if seed is not None:
                raise ValueError("Pass an exact state or a reset seed, not both.")
            return super().reset(state=state)
        if seed is not None:
            self._reset_rng = np.random.default_rng(seed)
        super().reset()
        self.data.qpos[:] = np.array([np.pi, 0.0]) + self._reset_rng.normal(0.0, self._reset_std, 2)
        self.data.qvel[:] = self._reset_rng.normal(0.0, self._reset_std, 2)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    @staticmethod
    def _is_healthy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.ones(np.broadcast_shapes(np.shape(a), np.shape(b)), dtype=bool)

    def running_cost(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        return dense_cost(states, actions)

    def terminal_cost(
        self,
        states: np.ndarray,
        sensordata: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.zeros(np.asarray(states).shape[:-1])

    def _get_obs(self) -> np.ndarray:
        return self.get_state()[1:5]

    @property
    def state_dim(self) -> int:
        return 5
