import mujoco 
import numpy as np 
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "walker2d.xml")

_HEALTHY_Z = (0.8, 2.0)
_HEALTHY_ANGLE = (-1.0, 1.0)
_RESET_NOISE = 5e-3
_QVEL_OBS_CLIP = 10.0

# model dt = 0.002 

class Walker2d(MuJoCoEnv):
    def __init__(
            self, 
            frame_skip: int = 4, # how many times/ frequently do you want to update your model
            cost_style: str = "gymnasium",
            target_velocity: float = 1.5,
            forward_reward_weight: float = 1.0,
            healthy_reward: float = 1.0,
            vel_cost_weight: float = 1.0,
            angle_cost_weight: float = 0.1,
            unhealthy_cost_weight: float = 5.0,
            ctrl_cost_weight: float = 1e-3,
            terminal_unhealthy_weight: float = 20.0,
            apply_terminal_cost_on_done: bool = True,
            **kwargs, 
    ):
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq  # 9
        self._nv = self.model.nv  # 9
        self._cost_style = cost_style
        self._v_target = target_velocity
        self._w_forward = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._w_vel = vel_cost_weight
        self._w_angle = angle_cost_weight
        self._w_unhealthy = unhealthy_cost_weight
        self._w_ctrl = ctrl_cost_weight
        self._w_term = terminal_unhealthy_weight
        self._apply_terminal_cost_on_done = apply_terminal_cost_on_done
        self._init_qpos = self.data.qpos.copy()  # rootz ref = 1.25
        if self._cost_style not in {"gymnasium", "target_velocity"}:
            raise ValueError(
                "cost_style must be 'gymnasium' or 'target_velocity', "
                f"got {self._cost_style!r}."
            )

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        if state is not None:
            return super().reset(state=state)
        super().reset()
        self.data.qpos[:] = self._init_qpos + np.random.uniform(
            -_RESET_NOISE, _RESET_NOISE, size=self._nq
        )
        self.data.qvel[:] = np.random.uniform(-_RESET_NOISE, _RESET_NOISE, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        obs, cost, _, info = super().step(action)
        healthy = bool(self._is_healthy(self.data.qpos[1], self.data.qpos[2]))
        done = not healthy
        fall_cost = 0.0
        if done and self._apply_terminal_cost_on_done:
            fall_cost = float(self.terminal_cost(self.get_state().reshape(1, -1)).item())
            cost += fall_cost
        vx = float(self.data.qvel[0])
        ctrl_cost = float(self._w_ctrl * np.sum(np.square(action)))
        info.update({
            "x_position": float(self.data.qpos[0]),
            "x_velocity": vx,
            "z": float(self.data.qpos[1]),
            "angle": float(self.data.qpos[2]),
            "healthy": healthy,
            "cost": float(cost),
            "fall_cost": fall_cost,
            "terminated": done,
            "cost_style": self._cost_style,
            "reward": float(-cost),
        })
        if self._cost_style == "gymnasium":
            reward_forward = self._w_forward * vx
            reward_survive = self._healthy_reward if healthy else 0.0
            info.update({
                "reward_forward": float(reward_forward),
                "reward_ctrl": float(-ctrl_cost),
                "reward_survive": float(reward_survive),
            })
        else:
            info.update({
                "velocity_error": float(abs(self._v_target - vx)),
                "ctrl_cost": ctrl_cost,
            })
        return obs, cost, done, info
    
    @staticmethod
    def _is_healthy(z, angle):
        return (
            (z > _HEALTHY_Z[0]) & (z < _HEALTHY_Z[1])
            & (angle > _HEALTHY_ANGLE[0]) & (angle < _HEALTHY_ANGLE[1])
        )
    
    def task_metrics(self) -> dict:
        z, angle = float(self.data.qpos[1]), float(self.data.qpos[2])
        vx = float(self.data.qvel[0])
        healthy = bool(self._is_healthy(z, angle))
        return {
            "vx": vx,
            "z": z,
            "angle": angle,
            "healthy": healthy,
            # walking at >= 50% of target speed while upright; sustained via
            # the hold counter in the GPS loop
            "success": healthy and vx >= 0.5 * self._v_target,
            # reuse the acrobot-named fields the GPS loop logs
            "tip_dist": max(self._v_target - vx, 0.0),
            "qvel_norm": float(np.linalg.norm(self.data.qvel)),
        }
    
    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        z = states[..., 2]
        angle = states[..., 3]
        vx = states[..., 1 + self._nq]
        healthy = self._is_healthy(z, angle)
        ctrl_cost = self._w_ctrl * np.sum(actions**2, axis=-1)
        if self._cost_style == "gymnasium":
            forward_cost = -self._w_forward * vx
            healthy_cost = -self._healthy_reward * healthy.astype(float)
            return forward_cost + healthy_cost + ctrl_cost

        vel_cost = self._w_vel * np.abs(self._v_target - vx)
        angle_cost = self._w_angle * angle**2
        unhealthy_cost = self._w_unhealthy * (~healthy)
        return vel_cost + angle_cost + unhealthy_cost + ctrl_cost
    
    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        healthy = self._is_healthy(states[..., 2], states[..., 3])
        return self._w_term * (~healthy).astype(float)

    def rollout_states_to_obs(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... 17"]:
        qpos_rest = states[..., 2 : 1 + self._nq]
        qvel = np.clip(states[..., 1 + self._nq : 1 + self._nq + self._nv],
                       -_QVEL_OBS_CLIP, _QVEL_OBS_CLIP)
        return np.concatenate([qpos_rest, qvel], axis=-1)

    def _get_obs(self) -> Float[ndarray, "17"]:
        return np.concatenate([
            self.data.qpos[1:],
            np.clip(self.data.qvel, -_QVEL_OBS_CLIP, _QVEL_OBS_CLIP),
        ])
    
    

    
