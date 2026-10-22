"""Batched GPU E-step vs the CPU E-step (needs a CUDA device; skipped elsewhere)."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
if wp.get_cuda_device_count() == 0:
    pytest.skip("no CUDA device", allow_module_level=True)

import torch  # noqa: E402

from src.envs.walker2d import Walker2d  # noqa: E402
from src.gps.gpu_estep import WalkerGPUEStep  # noqa: E402
from src.gps.mdgps import closed_loop_nominal, e_step  # noqa: E402
from src.gps.policy_tail import WalkerPolicyTail  # noqa: E402
from src.mppi.mppi import MPPI  # noqa: E402
from src.rl.sac import SacActor  # noqa: E402
from src.utils.config import MPPIConfig  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def test_gpu_estep_matches_cpu_labels_with_identical_noise():
    env = Walker2d(**json.loads((_ROOT / "configs/walker2d_task.json").read_text()), nthread=1)
    torch.manual_seed(0)
    low, high = env.action_bounds
    actor = SacActor(17, low, high).eval()
    actor_gpu = SacActor(17, low, high).cuda().eval()
    actor_gpu.load_state_dict(actor.state_dict())

    @torch.no_grad()
    def act_batch(obs):
        mean, _ = actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
        out = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
        return out if np.ndim(obs) > 1 else out[0]

    base = MPPIConfig(**json.loads((_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=64, H=8, initial_iterations=1, lam_target_neff_fraction=0.2)
    np.random.seed(3)
    env.reset()
    states, warmstarts = [], []
    for _ in range(3):
        for _ in range(5):
            env.step(np.random.uniform(-0.3, 0.3, env.action_dim))
        states.append(env.get_state())
        warmstarts.append(env.get_warmstart())
    seeds = [101, 202, 303]

    gpu = WalkerGPUEStep(env, actor_gpu, cfg, tail_steps=10, states_per_batch=2)
    result = gpu(np.asarray(states), np.asarray(warmstarts), seeds)

    tail = WalkerPolicyTail(env, act_batch, 10)
    for i, (state, warmstart) in enumerate(zip(states, warmstarts)):
        nominal, _ = closed_loop_nominal(env, state, lambda s: act_batch(env.rollout_states_to_obs(s)), cfg.H, warmstart)
        cpu = e_step(MPPI(env, cfg, seed=seeds[i]), env, state, nominal, warmstart, terminal=tail)
        np.testing.assert_allclose(result.labels[i], cpu.label, atol=1e-2)
        np.testing.assert_allclose(result.lam[i], cpu.lam, rtol=1e-2)
        np.testing.assert_allclose(result.nominal_cost[i], cpu.nominal_cost, rtol=1e-3)
        np.testing.assert_allclose(result.expected_cost[i], cpu.expected_cost, rtol=1e-3)
