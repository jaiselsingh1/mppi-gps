"""acrobot swing up env"""

import mujoco
import numpy as np
from typing import NamedTuple
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml")

_TARGET = np.array([0.0, 0.0, 4.0])
_TARGET_RADIUS = 0.2
_TOLERANCE_MARGIN = 2.0
_TOLERANCE_VALUE_AT_MARGIN = 0.1
_CONROL_COST_WEIGHT = 0.0
_QVEL_COST_WEIGHT = 0.15
_QVEL_EXCESS_COST_WEIGHT = 0.05
_QVEL_TARGET_GATE_POWER = 2.0
_QVEL_SCALE = np.array([2.5, 5.0])
_QVEL_EXCESS_THRESHOLD = 8.0
_HEIGHT_COST_WEIGHT = 1.0
_CENTER_COST_WEIGHT = 0.05
_TERMINAL_TARGET_COST_WEIGHT = 100.0
_TERMINAL_QVEL_COST_WEIGHT = 25.0

# dm_control suite acrobot SwingUp reward:
# rewards.tolerance(tip_to_target, bounds=(0, target_radius), margin=1).

def _dm_control_smooth_tolerance(
    x: Float[Array, "..."],
    upper: float,
    margin: float,
    value_at_margin: float,
) -> Float[Array, "..."]:
    """dm_control rewards.tolerance for non-negative distances, gaussian sigmoid."""
    in_bounds = x <= upper
    if margin == 0:
        return np.where(in_bounds, 1.0, 0.0)

    distance_from_bound = np.maximum(x - upper, 0.0) / margin
    reward = np.exp(np.log(value_at_margin) * distance_from_bound**2)
    return np.where(in_bounds, 1.0, reward)


def _dense_swingup_target_cost(
    tip_pos: Float[Array, "... 3"],
) -> Float[Array, "..."]:
    height_cost = 1.0 - np.clip(tip_pos[..., 2] / _TARGET[2], 0.0, 1.0)
    center_cost = np.sum((tip_pos[..., :2] - _TARGET[:2]) ** 2, axis=-1)
    return _HEIGHT_COST_WEIGHT * height_cost + _CENTER_COST_WEIGHT * center_cost


class CostComponents(NamedTuple):
    tip_dist: Float[Array, "..."]
    target_reward: Float[Array, "..."]
    target_cost: Float[Array, "..."]
    qvel_norm: Float[Array, "..."]
    qvel_cost: Float[Array, "..."]
    qvel_excess_cost: Float[Array, "..."]
    control_cost: Float[Array, "..."]


class WeightedCostComponents(NamedTuple):
    target_cost: Float[Array, "..."]
    qvel_cost: Float[Array, "..."]
    qvel_excess_cost: Float[Array, "..."]
    control_cost: Float[Array, "..."]
    total: Float[Array, "..."]


class TerminalCostComponents(NamedTuple):
    total: Float[Array, "..."]


class WeightedTerminalCostComponents(NamedTuple):
    total: Float[Array, "..."]


class Acrobot(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 1,
        qvel_cost_weight: float = _QVEL_COST_WEIGHT,
        qvel_excess_cost_weight: float = _QVEL_EXCESS_COST_WEIGHT,
        qvel_target_gate_power: float = _QVEL_TARGET_GATE_POWER,
        qvel_scale: tuple[float, float] = tuple(_QVEL_SCALE),
        qvel_excess_threshold: float = _QVEL_EXCESS_THRESHOLD,
        terminal_qvel_cost_weight: float = _TERMINAL_QVEL_COST_WEIGHT,
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq  # 2
        self._nv = self.model.nv  # 2
        self._qvel_cost_weight = qvel_cost_weight
        self._qvel_excess_cost_weight = qvel_excess_cost_weight
        self._qvel_target_gate_power = qvel_target_gate_power
        self._qvel_scale = np.asarray(qvel_scale, dtype=float)
        self._qvel_excess_threshold = qvel_excess_threshold
        self._terminal_qvel_cost_weight = terminal_qvel_cost_weight
        if self._qvel_scale.shape != (self._nv,):
            raise ValueError(
                f"qvel_scale must have shape {(self._nv,)}, got {self._qvel_scale.shape}."
            )
        if np.any(self._qvel_scale <= 0.0):
            raise ValueError("qvel_scale entries must be positive.")
        if self._qvel_excess_threshold <= 0.0:
            raise ValueError("qvel_excess_threshold must be positive.")

    def tip_pos(self) -> np.ndarray:
        return self.data.site("tip").xpos.copy()

    def task_metrics(self) -> dict:
        tip = self.tip_pos()
        dist = np.linalg.norm(tip - _TARGET)
        qvel = self.data.qvel.copy()
        qvel_norm = np.linalg.norm(qvel)
        reward = _dm_control_smooth_tolerance(
            dist,
            upper=_TARGET_RADIUS,
            margin=_TOLERANCE_MARGIN,
            value_at_margin=_TOLERANCE_VALUE_AT_MARGIN,
        )
        target_cost = _dense_swingup_target_cost(tip)
        return {
            "tip_dist": float(dist),
            "tip_z": float(tip[2]),
            "target_reward": float(reward),
            "target_cost": float(target_cost),
            "qvel_norm": float(qvel_norm),
            "success": bool(dist <= _TARGET_RADIUS),
        }

    def _state_qvel(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... nv"]:
        if states.shape[-1] == self._nq + self._nv:
            return states[..., self._nq : self._nq + self._nv]
        return self.state_qvel(states)

    def reset(
            self,
            state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-np.pi, np.pi, size=self._nq)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def running_cost_components(
        self,
        states: Float[Array, "... nstate"],
        actions: Float[Array, "... nu"],
        sensordata: Float[Array, "... nsensor"] | None = None,
    ) -> CostComponents:
        if sensordata is None:
            raise ValueError("Acrobot.running_cost requires tip-position sensordata.")

        tip_pos = sensordata[..., :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        reward = _dm_control_smooth_tolerance(
            dist,
            upper=_TARGET_RADIUS,
            margin=_TOLERANCE_MARGIN,
            value_at_margin=_TOLERANCE_VALUE_AT_MARGIN,
        )
        # Previous sparse-ish dm_control-style target cost:
        # target_cost = 1.0 - reward
        target_cost = _dense_swingup_target_cost(tip_pos)
        qvel = self._state_qvel(states)
        qvel_norm = np.linalg.norm(qvel, axis=-1)
        qvel_sq = np.sum((qvel / self._qvel_scale) ** 2, axis=-1)
        qvel_gate = reward ** self._qvel_target_gate_power
        qvel_cost = qvel_gate * qvel_sq
        qvel_excess = np.maximum(np.abs(qvel) - self._qvel_excess_threshold, 0.0)
        qvel_excess_cost = np.sum(
            (qvel_excess / self._qvel_excess_threshold) ** 2,
            axis=-1,
        )

        # du = np.diff(actions, axis=-2)
        # action_rate_cost = np.zeros(actions.shape[:-1])
        # action_rate_cost[..., 1:] = np.sum(du**2, axis=-1)
        
        control_cost = np.linalg.norm(actions, axis=-1) ** 2

        return CostComponents(
            tip_dist=dist,
            target_reward=reward,
            target_cost=target_cost,
            qvel_norm=qvel_norm,
            qvel_cost=qvel_cost,
            qvel_excess_cost=qvel_excess_cost,
            control_cost=control_cost
        )

    def weighted_cost_components(self, c: CostComponents) -> WeightedCostComponents:
        control_cost = _CONROL_COST_WEIGHT * c.control_cost
        qvel_cost = self._qvel_cost_weight * c.qvel_cost
        qvel_excess_cost = self._qvel_excess_cost_weight * c.qvel_excess_cost
        total = c.target_cost + qvel_cost + qvel_excess_cost + control_cost
        return WeightedCostComponents(
            target_cost=c.target_cost,
            qvel_cost=qvel_cost,
            qvel_excess_cost=qvel_excess_cost,
            control_cost=control_cost,
            total=total,
        )

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        c = self.running_cost_components(states, actions, sensordata)
        return self.weighted_cost_components(c).total

    def terminal_cost_components(
        self,
        states: Float[Array, "... nstate"],
        sensordata: Float[Array, "... nsensor"] | None = None,
    ) -> TerminalCostComponents:
        if sensordata is None:
            raise ValueError("Acrobot.terminal_cost requires tip-position sensordata.")

        tip_pos = sensordata[..., :3]
        # Previous no-terminal-shaping behavior:
        # terminal_cost = np.zeros(states.shape[:-1])
        target_cost = _TERMINAL_TARGET_COST_WEIGHT * _dense_swingup_target_cost(tip_pos)

        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        reward = _dm_control_smooth_tolerance(
            dist,
            upper=_TARGET_RADIUS,
            margin=_TOLERANCE_MARGIN,
            value_at_margin=_TOLERANCE_VALUE_AT_MARGIN,
        )
        qvel = self._state_qvel(states)
        qvel_sq = np.sum((qvel / self._qvel_scale) ** 2, axis=-1)
        qvel_gate = reward ** self._qvel_target_gate_power
        qvel_cost = self._terminal_qvel_cost_weight * qvel_gate * qvel_sq

        terminal_cost = target_cost + qvel_cost
        return TerminalCostComponents(total=terminal_cost)

    def weighted_terminal_cost_components(
        self,
        c: TerminalCostComponents,
    ) -> WeightedTerminalCostComponents:
        return WeightedTerminalCostComponents(total=c.total)

    def terminal_cost(
            self,
            states: Float[Array, "K nstate"],
            sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        c = self.terminal_cost_components(states, sensordata)
        return self.weighted_terminal_cost_components(c).total

    def _get_obs(self) -> Float[ndarray, "4"]:
        return np.concatenate([self.data.qpos, self.data.qvel])
