"""Check the GPU cost orchestration against CPU tails using exact CPU physics.

The CPU-backed worlds remove float32 and CUDA nondeterminism, isolating whether
the two paths pass identical solver state at every tail control boundary.
"""

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from src.envs.walker2d import Walker2d
from src.gps.gpu_estep import WalkerGPUEStep, _Worlds
from src.gps.policy_tail import WalkerPolicyTail


def test_gpu_cost_uses_the_cpu_tail_solver_reset_semantics(monkeypatch):
    env = Walker2d(nthread=1)
    env.model.opt.iterations = 1  # Make solver initialization observably relevant.
    env.model.opt.tolerance = 0.0
    np.random.seed(721)
    env.reset()
    for _ in range(10):
        env.step(np.full(6, 0.15))
    initial, warmstart = env.get_state(), env.get_warmstart()
    actions = np.full((1, 3, 6), 0.15)
    const_policy = lambda obs: np.full((len(obs), 6), 0.15)
    states, costs, _ = env.batch_rollout(initial, actions, initial_warmstart=warmstart)
    expected = costs + WalkerPolicyTail(env, const_policy, 15)(states, actions)

    class CPUWorlds(_Worlds):
        def __init__(self):
            self.data = mujoco.MjData(env.model)
            mujoco.mj_setState(env.model, self.data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            self.data.qacc_warmstart[:] = warmstart
            self.qpos = torch.from_numpy(self.data.qpos[None])
            self.qvel = torch.from_numpy(self.data.qvel[None])
            self.ctrl = torch.from_numpy(self.data.ctrl[None])
            self.warmstart = torch.from_numpy(self.data.qacc_warmstart[None])
            self.graph = self.data

    def launch_cpu_control_step(data):
        for _ in range(env._frame_skip):
            mujoco.mj_step(env.model, data)

    original_as_tensor = torch.as_tensor
    monkeypatch.setattr(torch, "as_tensor", lambda a, **kw: original_as_tensor(a, dtype=torch.float64))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr("src.gps.gpu_estep.wp.capture_launch", launch_cpu_control_step)
    monkeypatch.setattr("src.gps.gpu_estep.wp.synchronize", lambda: None)
    gpu_cost = WalkerGPUEStep.__new__(WalkerGPUEStep)
    gpu_cost.env, gpu_cost.cfg, gpu_cost.tail_steps = env, SimpleNamespace(H=3), 15
    gpu_cost.policy = lambda obs: torch.full((len(obs), 6), 0.15, dtype=torch.float64)
    actual = gpu_cost._cost(CPUWorlds(), 1, actions)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    env.close()
