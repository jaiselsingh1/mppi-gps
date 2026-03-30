
"""acrobot swing up env"""

import numpy as np 
from pathlib import Path 
from jaxtyping import Array, Float
from numpy import ndarray 

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml")

class Acrobot(MuJoCoEnv):
    def __init__(self, frame_skip: int = 1, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq #2 
        self._nv = self.model.nv #2 

        # this is upright, zero velocity and then the angles for the two joints 
        self._x_goal: Float[ndarray, "4"] = np.array([np.pi, 0.0, 0.0, 0.0])

        # cost weights 
        self._Q: Float[ndarray, "4"] = np.array([1.0, 1.0, 1.0, 1.0])
        self._R = 0
        self._P_scale = 1.0 # terminal cost multiplier 

    def reset(
            self, 
            state: Float[ndarray, "state_dim"] | None = None, 
    ) -> Float[ndarray, "4"]:
        # should be able to establish a fresh simulator state or restore a full state that's been saved
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qvel[:] = np.random.normal(0.0, 1e-4, size = self._nv)
        
        return self._get_obs()

    def running_cost(
            self,
          states: Float[Array, "K H nstate"],
          actions: Float[Array, "K H nu"],
      ) -> Float[Array, "K H"]:
        qpos: Float[Array, "K H nq"] = self.state_qpos(states)
        qvel: Float[Array, "K H nv"] = self.state_qvel(states)

        # angle error (wrap the shoulder between negative pi and pi)
        angle_err_shoulder = _angle_diff(qpos[:, :, 0], self._x_goal[0])
        angle_err_elbow = qpos[:, :, 1] - self._x_goal[1]
        vel_err_shoulder = qvel[:, :, 0] - self._x_goal[2]
        vel_err_elbow = qvel[:, :, 1] - self._x_goal[3]

        # weighted quadratic cost
        cost = (self._Q[0] * angle_err_shoulder**2
            + self._Q[1] * angle_err_elbow**2
            + self._Q[2] * vel_err_shoulder**2
            + self._Q[3] * vel_err_elbow**2
            + self._R * np.sum(np.square(actions), axis=-1))
        return cost  # (K, H)
    
    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
    ) -> Float[Array, "K"]:
        qpos: Float[Array, "K nq"] = self.state_qpos(states)
        qvel: Float[Array, "K nv"] = self.state_qvel(states)

        angle_err_shoulder: Float[Array, "K"] = _angle_diff(qpos[:, 0], self._x_goal[0])
        angle_err_elbow: Float[Array, "K"] = qpos[:, 1] - self._x_goal[1]
        vel_err_shoulder: Float[Array, "K"] = qvel[:, 0] - self._x_goal[2]
        vel_err_elbow: Float[Array, "K"] = qvel[:, 1] - self._x_goal[3]

        return self._P_scale * (
            self._Q[0] * angle_err_shoulder**2
            + self._Q[1] * angle_err_elbow**2
            + self._Q[2] * vel_err_shoulder**2
            + self._Q[3] * vel_err_elbow**2
        )
    
    def _get_obs(self) -> Float[ndarray, "4"]:
        return np.concatenate([self.data.qpos, self.data.qvel])
    
def _angle_diff(
        a: Float[ndarray, "..."], 
        b: Float[ndarray, "..."],
) -> Float[ndarray, "..."]:
    diff = b - a 
    return ((diff + np.pi) % (2.0 * np.pi)) - np.pi


        