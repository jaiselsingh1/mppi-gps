"""mujoco env with batched rollouts using mujoco.rollout."""

import os
import numpy as np 
import mujoco 
from jaxtyping import Array, Float
from mujoco import rollout 

from src.envs.base import BaseEnv
import warp as wp 
import mujoco_warp as mjw 


class MuJoCoEnv(BaseEnv):
    def __init__(self, model_path: str, nthread: int | None = None, frame_skip: int = 1, use_warp: bool = False, nworld: int = 8):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # adding frame skip 
        self._frame_skip = frame_skip
        self._dt = self.model.opt.timestep * frame_skip

        # state size for full physics state (qpos, qvel, actions, etc)
        self._nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )

        # thread pool for the batched rollouts 
        self._nthread = nthread or os.cpu_count()
        self._data_pool = [
            mujoco.MjData(self.model) for _ in range(self._nthread)
        ]

        self._rollout_ctx = rollout.Rollout(nthread = self._nthread)

        self._use_warp = use_warp
        if use_warp:
            assert self.model.na == 0, "warp path assumes na == 0"
            assert self.model.nmocap == 0, "warp path assumes no mocap bodies"
            self._wm = mjw.put_model(self.model)
            self._wd = mjw.make_data(self.model, nworld=nworld)
            self._warp_nworld = nworld
            self._warp_H = None
            self._qpos_buf = None
            self._qvel_buf = None
            self._sensor_buf = None
            self._actions_wp = None
            self._rollout_graph = None

    def reset(self, state: np.ndarray | None = None) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        if state is not None:
            self.set_state(state)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        self.data.ctrl[:] = action
        for _ in range(self._frame_skip):
            mujoco.mj_step(self.model, self.data)
            obs = self._get_obs()
            state = self.get_state()
            sensor = self.data.sensordata.copy().reshape(1, 1, -1)
            c = self.running_cost(
                state.reshape(1, 1, -1), action.reshape(1, 1, -1), sensor
            ).item()
        return obs, c, False, {}
    
    def get_state(self) -> np.ndarray:
        state = np.empty(self._nstate)
        mujoco.mj_getState(
            self.model, self.data, state, 
            mujoco.mjtState.mjSTATE_FULLPHYSICS, 
        )
        return state

    def set_state(self, state: np.ndarray) -> None:
        mujoco.mj_setState(
            self.model, self.data, state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )

    def state_qpos(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... nq"]:
        return states[..., 1 : 1 + self.model.nq]

    def state_qvel(
        self,
        states: Float[Array, "... nstate"],
    ) -> Float[Array, "... nv"]:
        start = 1 + self.model.nq
        end = start + self.model.nv
        return states[..., start:end]

    def batch_rollout(
            self, 
            initial_state: np.ndarray, 
            action_sequences: np.ndarray, 
    ) -> tuple[np.ndarray, np.ndarray]:
        
        if self._use_warp:
            return self._batch_rollout_warp(initial_state, action_sequences)

        K, H, _ = action_sequences.shape 
        
        # repeat each action for frame_skip physics steps 
        actions_expanded = np.repeat(action_sequences, self._frame_skip, axis = 1) 
        states_full, sensordata_full = self._rollout_ctx.rollout(
            self.model,
            self._data_pool,
            initial_state,
            actions_expanded,
        )

        # downsample states: take every frame_skip-th frame
        states = states_full[:, ::self._frame_skip, :]
        sensordata = sensordata_full[:, ::self._frame_skip, :]

        c = self.running_cost(states, action_sequences, sensordata) # (K, H)
        tc = self.terminal_cost(states[:, -1, :], sensordata[:, -1, :]) # (K, )
        costs = c.sum(axis = 1) + tc
        return states, costs, sensordata
    
    # you need to ensure that you have warp buffers for the h timsteps to live on 
    def _ensure_warp_buffers(self, K: int, H: int) -> None:
        if K != self._warp_nworld:
            raise RuntimeError(
                    f"nworld fixed at construction ({self._warp_nworld}); got K={K}. "
                    "Re-instantiate env with nworld=K.")
        # initial call this is set to None 
        if self._warp_H == H:
            return 
        # (H, K, *) so buf[h] is a (K, *) leading-axis subview wp.copy can target
        self._qpos_buf   = wp.zeros((H, K, self.model.nq),          dtype=wp.float32)
        self._qvel_buf   = wp.zeros((H, K, self.model.nv),          dtype=wp.float32)
        self._sensor_buf = wp.zeros((H, K, self.model.nsensordata), dtype=wp.float32)
        self._actions_wp = wp.zeros((H, K, self.model.nu),          dtype=wp.float32)
        self._rollout_graph = None    # buffers changed → prior graph is invalid
        self._warp_H = H

    def _run_rollout(self, H: int) -> None:
        """H outer steps, frame_skip inner mjw.step each, snapshot per outer step.

        Assumes self._wd.qpos/qvel hold the initial state and self._actions_wp
        is populated. Safe to capture into a CUDA graph.
        """
        for h in range(H):
            wp.copy(self._wd.ctrl, self._actions_wp[h])      # (K, nu) → d.ctrl
            for _ in range(self._frame_skip):
                mjw.step(self._wm, self._wd)
            wp.copy(self._qpos_buf[h],   self._wd.qpos)       # (K, nq)
            wp.copy(self._qvel_buf[h],   self._wd.qvel)       # (K, nv)
            wp.copy(self._sensor_buf[h], self._wd.sensordata) # (K, ns)

    def _batch_rollout_warp(self, initial_state, action_sequences):
        K, H, nu = action_sequences.shape
        self._ensure_warp_buffers(K, H)

        mujoco.mj_setState(
            self.model, self.data, initial_state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        nq, nv = self.model.nq, self.model.nv
        qpos0 = np.broadcast_to(self.data.qpos.astype(np.float32), (K, nq)).copy()
        qvel0 = np.broadcast_to(self.data.qvel.astype(np.float32), (K, nv)).copy()
        self._wd.qpos.assign(qpos0)
        self._wd.qvel.assign(qvel0)
        self._actions_wp.assign(
            np.ascontiguousarray(action_sequences.transpose(1, 0, 2).astype(np.float32))
        )

        if self._rollout_graph is None:
            with wp.ScopedCapture() as capture:
                self._run_rollout(H)
            self._rollout_graph = capture.graph
        wp.capture_launch(self._rollout_graph)

        qpos       = self._qpos_buf.numpy().transpose(1, 0, 2)
        qvel       = self._qvel_buf.numpy().transpose(1, 0, 2)
        sensordata = self._sensor_buf.numpy().transpose(1, 0, 2)

        time_col = np.zeros((K, H, 1), dtype=np.float32)
        states = np.concatenate([time_col, qpos, qvel], axis=-1)
        c  = self.running_cost(states, action_sequences, sensordata)
        tc = self.terminal_cost(states[:, -1, :], sensordata[:, -1, :])
        costs = c.sum(axis=1) + tc
        return states, costs, sensordata

    def _get_obs(self) -> np.ndarray:
        """ can be overrided in the sub class if you need to change the obs space"""
        return self.get_state()
    
    @property 
    def state_dim(self) -> int:
        return self._nstate
    
    @property 
    def action_dim(self) -> int:
        return self.model.nu 
    
    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.model.actuator_ctrlrange[:, 0].copy(),
            self.model.actuator_ctrlrange[:, 1].copy(),
        )

    def close(self):
        self._rollout_ctx.__exit__(None, None, None)

