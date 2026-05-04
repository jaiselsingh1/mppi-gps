"""acrobot swing up env"""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "acrobot.xml")

_TARGET = np.array([0.0, 0.0, 4.0])
_SUCCESS_RADIUS = 0.25
_SUCCESS_QVEL_NORM = 1.5

# Cost weights. The running cost encourages swing-up; the terminal cost is the
# capture term that prevents high-speed pass-through from looking solved.
_W_HEIGHT = 1.0
_W_DIST = 2.0
_W_RUNNING_QVEL = 0.02
_W_CTRL = 0.001
_W_TERMINAL_DIST = 50.0
_W_TERMINAL_QVEL = 5.0


class Acrobot(MuJoCoEnv):
    def __init__(self, frame_skip: int = 1, **kwargs) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq  # 2
        self._nv = self.model.nv  # 2

        self._w_terminal = 1.0

    def tip_pos(self) -> np.ndarray:
        return self.data.site("tip").xpos.copy()

    def task_metrics(self) -> dict:
        tip = self.tip_pos()
        dist = np.linalg.norm(tip - _TARGET)
        qvel = self.data.qvel.copy()
        qvel_norm = np.linalg.norm(qvel)
        return {
            "tip_dist": float(dist),
            "tip_z": float(tip[2]),
            "qvel_norm": float(qvel_norm),
            "success": bool(dist < _SUCCESS_RADIUS and qvel_norm < _SUCCESS_QVEL_NORM),
        }

    def reset(
            self,
            state: Float[ndarray, "state_dim"] | None = None,
    ) -> Float[ndarray, "4"]:
        if state is not None:
            return super().reset(state = state)

        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-np.pi, np.pi, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 2.0, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        if sensordata is None:
            raise ValueError("Acrobot.running_cost requires tip-position sensordata.")

        tip_pos = sensordata[:, :, :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)
        tip_z = tip_pos[:, :, 2]

        # FULLPHYSICS layout: [time, qpos0, qpos1, qvel0, qvel1]
        qvel = self.state_qvel(states)
        qvel_norm_sq = np.sum(qvel**2, axis=-1)
        ctrl_sq = np.sum(actions**2, axis=-1)

        return (
            _W_HEIGHT * (_TARGET[2] - tip_z)
            + _W_DIST * dist
            + _W_RUNNING_QVEL * qvel_norm_sq
            + _W_CTRL * ctrl_sq
        )

    def terminal_cost(
            self,
            states: Float[Array, "K nstate"],
            sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        if sensordata is None:
            raise ValueError("Acrobot.terminal_cost requires tip-position sensordata.")

        tip_pos = sensordata[:, :3]
        dist = np.linalg.norm(tip_pos - _TARGET, axis=-1)

        qvel = self.state_qvel(states)
        qvel_norm_sq = np.sum(qvel**2, axis=-1)

        return self._w_terminal * (
            _W_TERMINAL_DIST * dist
            + _W_TERMINAL_QVEL * qvel_norm_sq
        )

    def _get_obs(self) -> Float[ndarray, "4"]:
        return np.concatenate([self.data.qpos, self.data.qvel])
