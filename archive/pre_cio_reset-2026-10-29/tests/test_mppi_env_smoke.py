"""Verify retained task interfaces work with the standard CPU planner."""

import numpy as np
import pytest

from src.envs.acrobot import Acrobot
from src.envs.ant_maze import AntMaze
from src.envs.half_cheetah import HalfCheetah
from src.envs.humanoid import Humanoid
from src.envs.point_mass import PointMass
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


@pytest.mark.parametrize(
    "env_type", [Acrobot, AntMaze, HalfCheetah, Humanoid, PointMass, Walker2d],
    ids=lambda cls: cls.__name__,
)
def test_standard_mppi_plans_without_mutating_live_simulator(env_type):
    np.random.seed(91)
    env = env_type(nthread=1, use_warp=False)
    try:
        obs = env.reset()
        assert np.all(np.isfinite(obs))
        controller = MPPI(env, MPPIConfig(K=4, H=3, noise_sigma=0.1), seed=92)
        for _ in range(2):
            state = env.get_state()
            warmstart = env.get_warmstart()
            action, info = controller.plan_step(state, initial_warmstart=warmstart)

            np.testing.assert_array_equal(env.get_state(), state)
            np.testing.assert_array_equal(env.get_warmstart(), warmstart)
            assert action.shape == (env.action_dim,)
            assert np.all(np.isfinite(action))
            low, high = env.action_bounds
            assert np.all((action >= low) & (action <= high))
            assert np.isfinite(info["cost_min"])
            assert info["coupling_active"] == 0.0
            assert info["cost_track_mean"] == 0.0

            obs, cost, done, _ = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.all(np.isfinite(env.get_state()))
            assert np.isfinite(cost)
            if done:
                break
    finally:
        env.close()
