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

_W_STAND = 80.0
_W_MOVE = 2.0
_W_STILL = 10.0
_W_ROOT_X = 2.0
_W_LATERAL = 0.1
_W_LATERAL_VEL = 0.2
_W_ROOT_ANGVEL = 0.2
_W_POSTURE = 0.2
_W_QVEL = 0.002
_W_CTRL = 0.005
_W_TERMINAL_STAND = 1000.0

class CostComponents(typing.NamedTuple):
        standing_reward: Float[Array, "K H"]
        task_cost: Float[Array, "K H"]
        root_y: Float[Array, "K H"]
        vy:  Float[Array, "K H"]
        root_angvel_sq: Float[Array, "K H"] 
        posture_sq: Float[Array, "K H"]
        qvel_sq: Float[Array, "K H"]
        ctrl_sq: Float[Array, "K H"]


def _quat_up_z(quat: np.ndarray) -> np.ndarray:
    """World z component of the torso local z axis for wxyz quaternions."""
    qx = quat[..., 1]
    qy = quat[..., 2]
    return 1.0 - 2.0 * (qx * qx + qy * qy)


def _linear_tolerance_lower(x: np.ndarray, lower: float, margin: float) -> np.ndarray:
    """Linear tolerance that is 1 above lower and 0 at lower - margin."""
    if margin <= 0.0:
        return (x >= lower).astype(float)
    return np.clip((x - (lower - margin)) / margin, 0.0, 1.0)


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
        return {
            "x_pos": float(torso_pos[0]),
            "y_pos": float(torso_pos[1]),
            "head_height": float(self.data.xpos[self._head_id, 2]),
            "torso_height": height,
            "forward_vx": float(self.data.qvel[0]),
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

        root_x = qpos[..., 0]
        root_y = qpos[..., 1]
        root_z = qpos[..., 2]
        root_quat = qpos[..., 3:7]
        joint_qpos = qpos[..., 7:]

        vx = qvel[..., 0]
        vy = qvel[..., 1]
        root_angvel_sq = np.sum(qvel[..., 3:6] ** 2, axis=-1)
        qvel_sq = np.sum(qvel[..., 6:] ** 2, axis=-1)
        posture_sq = np.sum((joint_qpos - self._qpos0_tail) ** 2, axis=-1)
        ctrl_sq = np.sum(actions**2, axis=-1)
        upright = _quat_up_z(root_quat)
        head_height = root_z + _HEAD_OFFSET * upright

        standing = _linear_tolerance_lower(
            head_height,
            lower=_HEAD_STAND_HEIGHT,
            margin=_HEAD_STAND_HEIGHT / 4.0,
        )
        upright_reward = _linear_tolerance_lower(
            upright,
            lower=0.9,
            margin=1.9,
        )
        stand_reward = standing * upright_reward

        if self._target_speed > 0.0:
            forward_reward = _linear_tolerance_lower(
                vx,
                lower=self._target_speed,
                margin=self._target_speed,
            )
            task_cost = _W_MOVE * stand_reward * (1.0 - forward_reward)
        else:
            task_cost = _W_STILL * (vx**2 + vy**2) + _W_ROOT_X * root_x**2

        return CostComponents(
            standing_reward = stand_reward,
            task_cost = task_cost,
            root_y = root_y,
            vy = vy,
            root_angvel_sq = root_angvel_sq,
            posture_sq = posture_sq, 
            qvel_sq = qvel_sq, 
            ctrl_sq = ctrl_sq,)
    
    def running_cost(self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,    
    ) -> Float[Array, "K H"]:
        c = self.running_cost_components(states, actions, sensordata)

        return (
            _W_STAND * (1.0 - c.standing_reward)
            + c.task_cost
            + _W_LATERAL * (c.root_y ** 2)
            + _W_LATERAL_VEL * (c.vy ** 2)
            + _W_ROOT_ANGVEL * c.root_angvel_sq
            + _W_POSTURE * c.posture_sq
            + _W_QVEL * c.qvel_sq
            + _W_CTRL * c.ctrl_sq
        )

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        qpos = self.state_qpos(states)
        root_z = qpos[..., 2]
        root_quat = qpos[..., 3:7]
        upright = _quat_up_z(root_quat)
        head_height = root_z + _HEAD_OFFSET * upright
        standing = _linear_tolerance_lower(
            head_height,
            lower=_HEAD_STAND_HEIGHT,
            margin=_HEAD_STAND_HEIGHT / 4.0,
        )
        upright_reward = _linear_tolerance_lower(
            upright,
            lower=0.9,
            margin=0.4,
        )
        stand_reward = standing * upright_reward
        return _W_TERMINAL_STAND * (1.0 - stand_reward)

    def _get_obs(self) -> Float[np.ndarray, "obs_dim"]:
        return np.concatenate([
            self.data.qpos[2:],
            self.data.qvel,
        ])
