"""Point mass as an LQR oracle: linear MuJoCo dynamics and quadratic cost.

Joint limits are removed (contacts are already disabled), so one control step
is exactly x' = A x + B u with x = [qpos, qvel]. The repo's rollouts score
(x_{t+1}, u_t) pairs plus a terminal cost on x_H, which equals the standard
stage cost x_t'Q x_t + u_t'R u_t (minus the constant x_0'Q x_0) with terminal
weight Q + P_term. Choosing P_term = P - Q, with P the DARE solution, makes the
optimal first action of every H-step problem exactly -K x_0.
"""
from pathlib import Path

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from src.envs.mujoco_env import MuJoCoEnv

_XML = str(Path(__file__).resolve().parents[2] / "assets" / "point_mass.xml")


class LQRPointMass(MuJoCoEnv):
    def __init__(
        self,
        pos_weight: float = 100.0,
        vel_weight: float = 1.0,
        ctrl_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(model_path=_XML, frame_skip=1, **kwargs)
        self.model.jnt_limited[:] = 0
        self.A, self.B = self._linearize()
        self.Q = np.diag([pos_weight] * 2 + [vel_weight] * 2)
        self.R = ctrl_weight * np.eye(self.model.nu)
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.K_lqr = np.linalg.solve(self.R + self.B.T @ self.P @ self.B, self.B.T @ self.P @ self.A)
        self._P_term = self.P - self.Q

    def _linearize(self) -> tuple[np.ndarray, np.ndarray]:
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        n = 2 * self.model.nv
        A = np.zeros((n, n))
        B = np.zeros((n, self.model.nu))
        mujoco.mjd_transitionFD(self.model, data, 1e-6, True, A, B, None, None)
        return A, B

    def lqr_state(self, states: np.ndarray) -> np.ndarray:
        """[qpos, qvel] from full-physics states (time comes first)."""
        return states[..., 1 : 1 + 2 * self.model.nv]

    def full_state(self, x: np.ndarray) -> np.ndarray:
        data = mujoco.MjData(self.model)
        data.qpos[:] = x[: self.model.nq]
        data.qvel[:] = x[self.model.nq :]
        state = np.empty(self._nstate)
        mujoco.mj_getState(self.model, data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return state

    def running_cost(self, states, actions, sensordata=None):
        x = self.lqr_state(states)
        return np.einsum("...i,ij,...j->...", x, self.Q, x) + np.einsum(
            "...i,ij,...j->...", actions, self.R, actions
        )

    def terminal_cost(self, states, sensordata=None):
        x = self.lqr_state(states)
        return np.einsum("...i,ij,...j->...", x, self._P_term, x)
