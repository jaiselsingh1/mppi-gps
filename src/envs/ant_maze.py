"""Ant U-maze goal-reaching task.

The robot model is Gymnasium's Ant-v5 model with Farama's U-maze wall layout
added in `assets/ant_maze.xml`. The environment keeps the project-local
BaseEnv/MuJoCoEnv contract so it can be used by the existing MPPI/GPS loops.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from jaxtyping import Array, Float
from numpy import ndarray

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "ant_maze.xml")

_INIT_QPOS = np.array(
    [
        0.0,
        0.0,
        0.55,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        -1.0,
        0.0,
        1.0,
    ],
    dtype=float,
)
_RESET_POS = np.array([-4.0, -4.0], dtype=float)
_GOAL = np.array([4.0, 4.0], dtype=float)
_U_MAZE_GOAL_CELLS = np.array(
    [
        [-4.0, 4.0],
        [0.0, 4.0],
        [4.0, 4.0],
        [4.0, 0.0],
        [-4.0, -4.0],
        [0.0, -4.0],
        [4.0, -4.0],
    ],
    dtype=float,
)
_ROUTE_WAYPOINTS = np.array(
    [
        [-4.0, -4.0],
        [4.0, -4.0],
        [4.0, 4.0],
    ],
    dtype=float,
)

_XY_RESET_NOISE = 0.25
_JOINT_RESET_NOISE = 0.05
_QVEL_RESET_NOISE = 0.05
_GOAL_RADIUS = 0.5
_HEALTHY_Z_RANGE = (0.2, 1.0)

_DIST_COST_WEIGHT = 1.0
_TERMINAL_DIST_COST_WEIGHT = 25.0
_PATH_PROGRESS_COST_WEIGHT = 1.0
_PATH_LATERAL_COST_WEIGHT = 2.0
_TERMINAL_PATH_PROGRESS_COST_WEIGHT = 25.0
_TERMINAL_PATH_LATERAL_COST_WEIGHT = 10.0
_CTRL_COST_WEIGHT = 0.01
_QVEL_COST_WEIGHT = 0.005
_UNHEALTHY_COST_WEIGHT = 10.0


def _route_progress_cost(
    xy: Float[Array, "... 2"],
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
    p0, p1, p2 = _ROUTE_WAYPOINTS
    seg0 = p1 - p0
    seg1 = p2 - p1
    len0 = float(np.linalg.norm(seg0))
    len1 = float(np.linalg.norm(seg1))

    t0 = np.clip(np.sum((xy - p0) * seg0, axis=-1) / np.dot(seg0, seg0), 0.0, 1.0)
    proj0 = p0 + t0[..., None] * seg0
    d0_sq = np.sum((xy - proj0) ** 2, axis=-1)

    t1 = np.clip(np.sum((xy - p1) * seg1, axis=-1) / np.dot(seg1, seg1), 0.0, 1.0)
    proj1 = p1 + t1[..., None] * seg1
    d1_sq = np.sum((xy - proj1) ** 2, axis=-1)

    use_seg0 = d0_sq <= d1_sq
    progress = np.where(use_seg0, t0 * len0, len0 + t1 * len1)
    lateral_sq = np.where(use_seg0, d0_sq, d1_sq)
    remaining = (len0 + len1) - progress
    return remaining, lateral_sq, progress


class AntMaze(MuJoCoEnv):
    def __init__(
        self,
        frame_skip: int = 5,
        random_goal: bool = False,
        random_reset: bool = False,
        cost_mode: Literal["route", "euclidean"] = "route",
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=frame_skip, **kwargs)
        self._nq = self.model.nq
        self._nv = self.model.nv
        self._random_goal = random_goal
        self._random_reset = random_reset
        self._cost_mode = cost_mode
        self.goal = _GOAL.copy()
        self.reset_pos = _RESET_POS.copy()
        self._target_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "target",
        )
        self._set_goal_visual()

    def set_goal(self, goal: Float[ndarray, "2"]) -> None:
        self.goal = np.asarray(goal, dtype=float).copy()
        self._set_goal_visual()

    def set_reset_pos(self, reset_pos: Float[ndarray, "2"]) -> None:
        self.reset_pos = np.asarray(reset_pos, dtype=float).copy()

    def _sample_goal(self) -> np.ndarray:
        return _U_MAZE_GOAL_CELLS[np.random.randint(len(_U_MAZE_GOAL_CELLS))].copy()

    def _sample_reset_pos(self) -> np.ndarray:
        candidates = _U_MAZE_GOAL_CELLS[
            np.linalg.norm(_U_MAZE_GOAL_CELLS - self.goal[None, :], axis=1) > 1.0
        ]
        return candidates[np.random.randint(len(candidates))].copy()

    def _set_goal_visual(self) -> None:
        if self._target_site_id >= 0:
            self.model.site_pos[self._target_site_id, :2] = self.goal

    def reset(
        self,
        state: Float[ndarray, "state_dim"] | None = None,
        goal: Float[ndarray, "2"] | None = None,
        reset_pos: Float[ndarray, "2"] | None = None,
    ) -> Float[ndarray, "31"]:
        if goal is not None:
            self.set_goal(goal)
        elif self._random_goal:
            self.set_goal(self._sample_goal())

        if reset_pos is not None:
            self.set_reset_pos(reset_pos)
        elif self._random_reset:
            self.set_reset_pos(self._sample_reset_pos())

        if state is not None:
            return super().reset(state=state)

        super().reset()
        qpos = _INIT_QPOS.copy()
        qpos[:2] = self.reset_pos + np.random.uniform(
            -_XY_RESET_NOISE,
            _XY_RESET_NOISE,
            size=2,
        )
        qpos[7:] += np.random.uniform(-_JOINT_RESET_NOISE, _JOINT_RESET_NOISE, size=self._nq - 7)
        qvel = np.random.uniform(-_QVEL_RESET_NOISE, _QVEL_RESET_NOISE, size=self._nv)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _split_states(
        self,
        states: Float[Array, "... nstate"],
    ) -> tuple[Float[Array, "... nq"], Float[Array, "... nv"]]:
        if states.shape[-1] == self._nq + self._nv:
            return states[..., : self._nq], states[..., self._nq : self._nq + self._nv]
        return self.state_qpos(states), self.state_qvel(states)

    def rollout_states_to_obs(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... 31"]:
        qpos, qvel = self._split_states(states)
        goal = np.broadcast_to(self.goal, (*qpos.shape[:-1], 2))
        return np.concatenate([qpos, qvel, goal], axis=-1)

    def task_metrics(self) -> dict:
        xy = self.data.qpos[:2].copy()
        qvel = self.data.qvel.copy()
        dist = float(np.linalg.norm(xy - self.goal))
        route_remaining, route_lateral_sq, route_progress = _route_progress_cost(xy)
        healthy = _HEALTHY_Z_RANGE[0] <= float(self.data.qpos[2]) <= _HEALTHY_Z_RANGE[1]
        return {
            "tip_dist": dist,
            "xy_dist": dist,
            "qvel_norm": float(np.linalg.norm(qvel)),
            "x_pos": float(xy[0]),
            "y_pos": float(xy[1]),
            "z_pos": float(self.data.qpos[2]),
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "healthy": bool(healthy),
            "success": bool(dist <= _GOAL_RADIUS and healthy),
            "target_cost": dist * dist,
            "route_remaining": float(route_remaining),
            "route_progress": float(route_progress),
            "route_lateral_dist": float(np.sqrt(route_lateral_sq)),
        }

    def running_cost(
        self,
        states: Float[Array, "K H nstate"],
        actions: Float[Array, "K H nu"],
        sensordata: Float[Array, "K H nsensor"] | None = None,
    ) -> Float[Array, "K H"]:
        del sensordata
        qpos, qvel = self._split_states(states)
        xy = qpos[..., :2]
        z = qpos[..., 2]
        if self._cost_mode == "route":
            remaining, lateral_sq, _ = _route_progress_cost(xy)
            task_cost = (
                _PATH_PROGRESS_COST_WEIGHT * remaining * remaining
                + _PATH_LATERAL_COST_WEIGHT * lateral_sq
            )
        else:
            task_cost = _DIST_COST_WEIGHT * np.sum((xy - self.goal) ** 2, axis=-1)
        qvel_cost = np.sum(qvel * qvel, axis=-1)
        ctrl_cost = np.sum(actions * actions, axis=-1)
        unhealthy = (z < _HEALTHY_Z_RANGE[0]) | (z > _HEALTHY_Z_RANGE[1])
        return (
            task_cost
            + _QVEL_COST_WEIGHT * qvel_cost
            + _CTRL_COST_WEIGHT * ctrl_cost
            + _UNHEALTHY_COST_WEIGHT * unhealthy.astype(float)
        )

    def terminal_cost(
        self,
        states: Float[Array, "K nstate"],
        sensordata: Float[Array, "K nsensor"] | None = None,
    ) -> Float[Array, "K"]:
        del sensordata
        qpos, qvel = self._split_states(states)
        xy = qpos[..., :2]
        z = qpos[..., 2]
        if self._cost_mode == "route":
            remaining, lateral_sq, _ = _route_progress_cost(xy)
            task_cost = (
                _TERMINAL_PATH_PROGRESS_COST_WEIGHT * remaining * remaining
                + _TERMINAL_PATH_LATERAL_COST_WEIGHT * lateral_sq
            )
        else:
            task_cost = _TERMINAL_DIST_COST_WEIGHT * np.sum((xy - self.goal) ** 2, axis=-1)
        qvel_cost = np.sum(qvel * qvel, axis=-1)
        unhealthy = (z < _HEALTHY_Z_RANGE[0]) | (z > _HEALTHY_Z_RANGE[1])
        return (
            task_cost
            + _QVEL_COST_WEIGHT * qvel_cost
            + _UNHEALTHY_COST_WEIGHT * unhealthy.astype(float)
        )

    def _get_obs(self) -> Float[ndarray, "31"]:
        return np.concatenate([self.data.qpos, self.data.qvel, self.goal])
