"""MuJoCo humanoid locomotion environment for MPPI experiments."""

from pathlib import Path

import mujoco
import numpy as np
from jaxtyping import Array, Float

from src.envs.mujoco_env import MuJoCoEnv
import typing

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "humanoid.xml")

_HEAD_STAND_HEIGHT = 1.4
_HEAD_OFFSET = 0.19
_MIN_HEALTHY_HEIGHT = 0.8

_STAND_HEIGHT_MARGIN = _HEAD_STAND_HEIGHT / 4.0
_UPRIGHT_THRESHOLD = 0.9
_UPRIGHT_MARGIN = 1.9
_DONT_MOVE_MARGIN = 2.0

_W_LATERAL = 0.0
_W_LATERAL_VEL = 0.0
_W_ROOT_ANGVEL = 0.0
_W_POSTURE = 0.0
_W_QVEL = 0.0
_W_CTRL = 0.0
_W_TERMINAL_STAND = 0.0

class CostComponents(typing.NamedTuple):
    standing_reward: Float[Array, "K H"]
    task_reward: Float[Array, "K H"]
    small_control: Float[Array, "K H"]
    reward_cost: Float[Array, "K H"]
    task_cost: Float[Array, "K H"]
    root_y: Float[Array, "K H"]
    vy:  Float[Array, "K H"]
    root_angvel_sq: Float[Array, "K H"] 
    posture_sq: Float[Array, "K H"]
    qvel_sq: Float[Array, "K H"]
    ctrl_sq: Float[Array, "K H"]


class WeightedCostComponents(typing.NamedTuple):
    reward_cost: Float[Array, "K H"]
    stand_cost: Float[Array, "K H"]
    task_cost: Float[Array, "K H"]
    lateral_cost: Float[Array, "K H"]
    lateral_vel_cost: Float[Array, "K H"]
    root_angvel_cost: Float[Array, "K H"]
    posture_cost: Float[Array, "K H"]
    qvel_cost: Float[Array, "K H"]
    ctrl_cost: Float[Array, "K H"]
    total: Float[Array, "K H"]


def _quat_up_z(quat: np.ndarray) -> np.ndarray:
    """World z component of the torso local z axis for wxyz quaternions."""
    qx = quat[..., 1]
    qy = quat[..., 2]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def _sigmoid(value: np.ndarray, *, value_at_1: float, sigmoid: str) -> np.ndarray:
    value = np.asarray(value)
    if sigmoid == "gaussian":
        scale = np.sqrt(-2.0 * np.log(value_at_1))
        return np.exp(-0.5 * (value * scale) ** 2)
    if sigmoid == "linear":
        scaled = value * (1.0 - value_at_1)
        return np.where(np.abs(scaled) < 1.0, 1.0 - scaled, 0.0)
    if sigmoid == "quadratic":
        scaled = value * np.sqrt(1.0 - value_at_1)
        return np.where(np.abs(scaled) < 1.0, 1.0 - scaled**2, 0.0)
    raise ValueError(f"unsupported sigmoid: {sigmoid}")


def _tolerance(
    x: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.0, 0.0),
    margin: float = 0.0,
    sigmoid: str = "gaussian",
    value_at_margin: float = 0.1,
) -> np.ndarray:
    x = np.asarray(x)
    lower, upper = bounds
    in_bounds = np.logical_and(lower <= x, x <= upper)
    if margin <= 0.0:
        return in_bounds.astype(float)
    distance = np.where(x < lower, lower - x, x - upper) / margin
    return np.where(
        in_bounds,
        1.0,
        _sigmoid(distance, value_at_1=value_at_margin, sigmoid=sigmoid),
    )


def _standing_reward(root_z: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    upright = _quat_up_z(root_quat)
    head_height = root_z + _HEAD_OFFSET * upright
    standing = _tolerance(
        head_height,
        bounds=(_HEAD_STAND_HEIGHT, float("inf")),
        margin=_STAND_HEIGHT_MARGIN,
    )
    upright_reward = _tolerance(
        upright,
        bounds=(_UPRIGHT_THRESHOLD, float("inf")),
        margin=_UPRIGHT_MARGIN,
        sigmoid="linear",
        value_at_margin=0.0,
    )
    return standing * upright_reward


class Humanoid(MuJoCoEnv):
    def __init__(self, frame_skip: int = 5, target_speed: float = 0.0, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq
        self._nv = self.model.nv
        self._target_speed = target_speed
        self._act_joint_ids = self.model.actuator_trnid[:, 0].copy()
        self._act_qpos_adr = self.model.jnt_qposadr[self._act_joint_ids].copy()
        self._act_dof_adr = self.model.jnt_dofadr[self._act_joint_ids].copy()
        self._act_gear = self.model.actuator_gear[:, 0].copy()
        self._qpos0_tail = self.model.qpos0[7:].copy()
        self._torso_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "torso",
        )
        self._head_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "head",
        )

    def reset(
        self,
        state: Float[np.ndarray, "state_dim"] | None = None,
    ) -> Float[np.ndarray, "obs_dim"]:
        if state is not None:
            return super().reset(state=state)

        obs = super().reset()
        self.data.qpos[:] = self.model.qpos0
        self.data.qpos[7:] += np.random.uniform(-0.02, 0.02, size=self._nq - 7)
        self.data.qvel[:] = np.random.normal(0.0, 0.02, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def task_metrics(self) -> dict:
        torso_pos = self.data.xpos[self._torso_id].copy()
        torso_quat = self.data.xquat[self._torso_id].copy()
        upright = float(_quat_up_z(torso_quat))
        height = float(torso_pos[2])
        com_velocity = self.data.sensordata[:3].copy()
        return {
            "x_pos": float(torso_pos[0]),
            "y_pos": float(torso_pos[1]),
            "head_height": float(self.data.xpos[self._head_id, 2]),
            "torso_height": height,
            "forward_vx": float(com_velocity[0]),
            "com_vy": float(com_velocity[1]),
            "com_speed": float(np.linalg.norm(com_velocity[:2])),
            "upright": upright,
            "healthy": bool(height > _MIN_HEALTHY_HEIGHT and upright > 0.5),
        }

    def pose_pd_action(self, kp: float = 120.0, kd: float = 12.0) -> np.ndarray:
        """Joint-space stabilizing action around the model's standing qpos0."""
        q_err = self.model.qpos0[self._act_qpos_adr] - self.data.qpos[self._act_qpos_adr]
        qvel = self.data.qvel[self._act_dof_adr]
        joint_torque = kp * q_err - kd * qvel
        action = joint_torque / np.maximum(np.abs(self._act_gear), 1e-6)
        low, high = self.action_bounds
        return np.clip(action, low, high)

    def running_cost_components(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> CostComponents:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)

        root_y = qpos[..., 1]
        root_z = qpos[..., 2]
        root_quat = qpos[..., 3:7]
        joint_qpos = qpos[..., 7:]

        if sensordata is not None and sensordata.shape[-1] >= 3:
            vx = sensordata[..., 0]
            vy = sensordata[..., 1]
        else:
            vx = qvel[..., 0]
            vy = qvel[..., 1]
        root_angvel_sq = np.sum(qvel[..., 3:6] ** 2, axis=-1)
        qvel_sq = np.sum(qvel[..., 6:] ** 2, axis=-1)
        posture_sq = np.sum((joint_qpos - self._qpos0_tail) ** 2, axis=-1)
        ctrl_sq = np.sum(actions**2, axis=-1)

        stand_reward = _standing_reward(root_z, root_quat)
        small_control = _tolerance(
            actions,
            margin=1.0,
            sigmoid="quadratic",
            value_at_margin=0.0,
        ).mean(axis=-1)
        small_control = (4.0 + small_control) / 5.0

        if self._target_speed > 0.0:
            horizontal_speed = np.sqrt(vx**2 + vy**2)
            move_reward = _tolerance(
                horizontal_speed,
                bounds=(self._target_speed, float("inf")),
                margin=self._target_speed,
                sigmoid="linear",
                value_at_margin=0.0,
            )
            task_reward = (5.0 * move_reward + 1.0) / 6.0
        else:
            horizontal_velocity = np.stack([vx, vy], axis=-1)
            task_reward = _tolerance(
                horizontal_velocity,
                margin=_DONT_MOVE_MARGIN,
            ).mean(axis=-1)

        reward_cost = 1.0 - small_control * stand_reward * task_reward

        return CostComponents(
            standing_reward=stand_reward,
            task_reward=task_reward,
            small_control=small_control,
            reward_cost=reward_cost,
            task_cost=1.0 - task_reward,
            root_y=root_y,
            vy=vy,
            root_angvel_sq=root_angvel_sq,
            posture_sq=posture_sq, 
            qvel_sq=qvel_sq, 
            ctrl_sq=ctrl_sq,
        )
    
    def running_cost(self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,    
    ) -> Float[Array, "K H"]:
        c = self.running_cost_components(states, actions, sensordata)
        return self.weighted_cost_components(c).total

    def weighted_cost_components(
        self,
        c: CostComponents,
    ) -> WeightedCostComponents:
        reward_cost = c.reward_cost
        stand_cost = 1.0 - c.standing_reward
        task_cost = c.task_cost
        lateral_cost = _W_LATERAL * (c.root_y ** 2)
        lateral_vel_cost = _W_LATERAL_VEL * (c.vy ** 2)
        root_angvel_cost = _W_ROOT_ANGVEL * c.root_angvel_sq
        posture_cost = _W_POSTURE * c.posture_sq
        qvel_cost = _W_QVEL * c.qvel_sq
        ctrl_cost = _W_CTRL * c.ctrl_sq
        total = (
            reward_cost
            + lateral_cost
            + lateral_vel_cost
            + root_angvel_cost
            + posture_cost
            + qvel_cost
            + ctrl_cost
        )
        return WeightedCostComponents(
            reward_cost=reward_cost,
            stand_cost=stand_cost,
            task_cost=task_cost,
            lateral_cost=lateral_cost,
            lateral_vel_cost=lateral_vel_cost,
            root_angvel_cost=root_angvel_cost,
            posture_cost=posture_cost,
            qvel_cost=qvel_cost,
            ctrl_cost=ctrl_cost,
            total=total,
        )

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        qpos = self.state_qpos(states)
        stand_reward = _standing_reward(qpos[..., 2], qpos[..., 3:7])
        return _W_TERMINAL_STAND * (1.0 - stand_reward)

    def _get_obs(self) -> Float[np.ndarray, "obs_dim"]:
        return np.concatenate([
            self.data.qpos[2:],
            self.data.qvel,
        ])
