"""2D point-mass goal-reaching task for MPPI/GPS sanity checks."""

import mujoco
import numpy as np
from pathlib import Path
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "point_mass.xml")
_GOAL = np.array([0.0, 0.0], dtype=float)
_GOAL_RADIUS = 0.025
_SUCCESS_VEL = 0.05
_GOAL_WORKSPACE_LOW = np.array([-0.29, -0.29], dtype=float)
_GOAL_WORKSPACE_HIGH = np.array([0.29, 0.29], dtype=float)
_POS_COST_WEIGHT = 1.0
_VEL_COST_WEIGHT = 0.05
_TERMINAL_POS_COST_WEIGHT = 20.0
_TERMINAL_VEL_COST_WEIGHT = 1.0


class PointMass(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 1,
        ctrl_cost_weight: float = 0.01,
        random_goal: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._ctrl_w = float(ctrl_cost_weight)
        self._nq = self.model.nq
        self._nv = self.model.nv
        self._random_goal = random_goal
        self.goal = _GOAL.copy()
        self._set_goal_visual()

    def set_goal(self, goal: Float[ndarray, "2"]) -> None:
        self.goal = np.asarray(goal, dtype=float).copy()
        self._set_goal_visual()

    def _sample_goal(self) -> np.ndarray:
        return np.random.uniform(_GOAL_WORKSPACE_LOW, _GOAL_WORKSPACE_HIGH)

    def _set_goal_visual(self) -> None:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target")
        if geom_id >= 0:
            self.model.geom_pos[geom_id, :2] = self.goal

    def reset(
        self,
        state: Float[ndarray, "state_dim"] | None = None,
        goal: Float[ndarray, "2"] | None = None,
    ) -> Float[ndarray, "6"]:
        if goal is not None:
            self.set_goal(goal)
        elif self._random_goal:
            self.set_goal(self._sample_goal())

        if state is not None:
            return super().reset(state=state)

        obs = super().reset()
        self.data.qpos[:] = np.random.uniform(-0.25, 0.25, size=self._nq)
        self.data.qvel[:] = np.random.normal(0.0, 0.1, size=self._nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def task_metrics(self) -> dict:
        pos = self.data.qpos.copy()
        vel = self.data.qvel.copy()
        dist = float(np.linalg.norm(pos - self.goal))
        qvel_norm = float(np.linalg.norm(vel))
        return {
            "tip_dist": dist,
            "qvel_norm": qvel_norm,
            "success": bool(dist <= _GOAL_RADIUS and qvel_norm <= _SUCCESS_VEL),
            "x_pos": float(pos[0]),
            "y_pos": float(pos[1]),
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "target_cost": float(np.sum((pos - self.goal) ** 2)),
        }

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        pos_err = qpos - self.goal[None, None, :]
        pos_cost = _POS_COST_WEIGHT * np.sum(pos_err**2, axis=-1)
        vel_cost = _VEL_COST_WEIGHT * np.sum(qvel**2, axis=-1)
        ctrl_cost = self._ctrl_w * np.sum(actions**2, axis=-1)
        return pos_cost + vel_cost + ctrl_cost

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        qpos = self.state_qpos(states)
        qvel = self.state_qvel(states)
        pos_err = qpos - self.goal[None, :]
        pos_cost = _TERMINAL_POS_COST_WEIGHT * np.sum(pos_err**2, axis=-1)
        vel_cost = _TERMINAL_VEL_COST_WEIGHT * np.sum(qvel**2, axis=-1)
        return pos_cost + vel_cost

    def _get_obs(self) -> Float[ndarray, "6"]:
        return np.concatenate([self.data.qpos, self.data.qvel, self.goal])
