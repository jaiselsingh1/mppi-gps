"""Hopper with Walker2d's target-velocity cost and absorbing fall semantics.

The model is Gymnasium's hopper.xml (copied to assets/). Healthy follows
Gymnasium Hopper-v5: torso height above .7, |pitch| below .2 and every other
coordinate bounded by 100. The cost mirrors the repository's Walker2d
target-velocity task, including scoring a planned rollout through its first
fall and then charging an absorbing unhealthy cost per remaining step, so the
same MPPI configuration semantics carry over. A fall ends the episode.
"""
from pathlib import Path

import mujoco
import numpy as np

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "hopper.xml")

_HEALTHY_Z = 0.7
_HEALTHY_ANGLE = 0.2
_HEALTHY_STATE = 100.0
_RESET_NOISE = 5e-3


class Hopper(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 4,
        target_velocity: float = 1.0,
        vel_cost_weight: float = 2.0,
        vertical_velocity_cost_weight: float = 0.5,
        angle_cost_weight: float = 0.1,
        unhealthy_cost_weight: float = 5.0,
        ctrl_cost_weight: float = 1e-3,
        terminal_unhealthy_weight: float = 20.0,
        **kwargs,
    ):
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq  # 6
        self._nv = self.model.nv  # 6
        self._v_target = target_velocity
        self._w_vel = vel_cost_weight
        self._w_vertical_velocity = vertical_velocity_cost_weight
        self._w_angle = angle_cost_weight
        self._w_unhealthy = unhealthy_cost_weight
        self._w_ctrl = ctrl_cost_weight
        self._w_term = terminal_unhealthy_weight
        self._init_qpos = self.data.qpos.copy()

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)
        super().reset()
        self.data.qpos[:] = self._init_qpos + np.random.uniform(-_RESET_NOISE, _RESET_NOISE, size=self._nq)
        self.data.qvel[:] = np.random.uniform(-_RESET_NOISE, _RESET_NOISE, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """A Gymnasium-style perturbed standing start, drawn from `rng`."""
        super().reset()
        self.data.qpos[:] = self._init_qpos + rng.uniform(-_RESET_NOISE, _RESET_NOISE, size=self._nq)
        self.data.qvel[:] = rng.uniform(-_RESET_NOISE, _RESET_NOISE, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, cost, _, info = super().step(action)
        state = self.get_state()
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Hopper simulator produced a non-finite state.")
        healthy = bool(self._healthy(state[None])[0])
        if not healthy:
            cost += float(self._w_term)
        info.update({"x_velocity": float(self.data.qvel[0]), "healthy": healthy})
        return obs, cost, not healthy, info

    def _healthy(self, states: np.ndarray) -> np.ndarray:
        qpos = states[..., 1 : 1 + self._nq]
        qvel = states[..., 1 + self._nq : 1 + self._nq + self._nv]
        rest = np.concatenate([qpos[..., 2:], qvel], axis=-1)
        return (
            (qpos[..., 1] > _HEALTHY_Z)
            & (np.abs(qpos[..., 2]) < _HEALTHY_ANGLE)
            & np.all(np.abs(rest) < _HEALTHY_STATE, axis=-1)
        )

    def running_cost(self, states, actions, sensordata=None):
        vx = states[..., 1 + self._nq]
        vz = states[..., 2 + self._nq]
        angle = states[..., 3]
        healthy = self._healthy(states)
        return (
            self._w_vel * np.abs(self._v_target - vx)
            + self._w_vertical_velocity * vz**2
            + self._w_angle * angle**2
            + self._w_unhealthy * (~healthy)
            + self._w_ctrl * np.sum(actions**2, axis=-1)
        )

    def terminal_cost(self, states, sensordata=None):
        return self._w_term * (~self._healthy(states)).astype(float)

    def rollout_cost(self, states, actions, sensordata):
        """Score through the first fall, then an absorbing unhealthy cost per remaining step."""
        healthy = self._healthy(states)
        active = np.ones_like(healthy, dtype=bool)
        if states.shape[1] > 1:
            active[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
        running = np.where(active, self.running_cost(states, actions, sensordata), 0.0).sum(axis=1)
        absorbing = self._w_unhealthy * np.sum(~active, axis=1)
        fell = np.any(~healthy, axis=1)
        terminal = np.where(fell, self._w_term, 0.0)
        costs = running + absorbing + terminal
        valid = np.all(np.isfinite(states), axis=-1) & np.all(np.isfinite(actions), axis=-1)
        return np.where(np.all(valid | ~active, axis=1), costs, np.inf)

    def healthy_mask(self, states: np.ndarray) -> np.ndarray:
        return self._healthy(states)

    def observation(self, states: np.ndarray) -> np.ndarray:
        """Policy/value features: everything but the absolute x position."""
        return np.concatenate([states[..., 2 : 1 + self._nq], states[..., 1 + self._nq : 1 + self._nq + self._nv]], axis=-1)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[1:], self.data.qvel])
