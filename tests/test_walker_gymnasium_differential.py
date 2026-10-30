"""Characterize the frozen task against the official Gymnasium environment.

The stock-v5 mismatch is deliberate evidence, not permission to silently change
the research task underlying recorded runs.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from src.envs.walker2d import _XML, Walker2d

XML = Path(_XML)


def align_initial_state(local, reference, seed):
    reference.reset(seed=seed)
    local.reset()
    local.data.qpos[:] = reference.data.qpos
    local.data.qvel[:] = reference.data.qvel
    local.data.time = reference.data.time
    mujoco.mj_forward(local.model, local.data)
    local.data.qacc_warmstart[:] = reference.data.qacc_warmstart


@pytest.mark.parametrize("seed", [914, 915, 916])
def test_same_xml_physics_observation_termination_and_independent_task_cost(seed):
    local = Walker2d(
        cost_style="target_velocity",
        vel_cost_weight=2.0,
        vertical_velocity_cost_weight=0.5,
        nthread=1,
    )
    reference = gym.make("Walker2d-v5", xml_file=str(XML)).unwrapped
    try:
        align_initial_state(local, reference, seed)
        for action in np.random.default_rng(seed + 1).uniform(-1, 1, (300, 6)):
            obs, cost, done, _ = local.step(action)
            ref_obs, _, ref_done, _, _ = reference.step(action)
            np.testing.assert_allclose(obs, ref_obs, rtol=0, atol=1e-12)
            np.testing.assert_allclose(local.data.qpos, reference.data.qpos, rtol=0, atol=1e-12)
            np.testing.assert_allclose(local.data.qvel, reference.data.qvel, rtol=0, atol=1e-12)
            assert done == ref_done
            expected = (
                2 * abs(1.5 - reference.data.qvel[0])
                + 0.5 * reference.data.qvel[1] ** 2
                + 0.1 * reference.data.qpos[2] ** 2
                + 0.001 * np.sum(action**2)
                + (25.0 if ref_done else 0.0)
            )
            assert cost == pytest.approx(expected, rel=0, abs=1e-12)
            if done:
                break
    finally:
        local.close()
        reference.close()


@pytest.mark.xfail(
    strict=True, reason="Frozen XML right-foot friction 0.9 differs from stock v5 1.9"
)
def test_frozen_walker_is_stock_v5_dynamics():
    local = Walker2d(nthread=1)
    reference = gym.make("Walker2d-v5").unwrapped
    try:
        align_initial_state(local, reference, 914)
        for action in np.random.default_rng(9).uniform(-1, 1, (100, 6)):
            obs, _, done, _ = local.step(action)
            ref_obs, _, _, _, _ = reference.step(action)
            np.testing.assert_allclose(obs, ref_obs, rtol=0, atol=1e-12)
            if done:
                break
    finally:
        local.close()
        reference.close()


def test_same_seed_is_not_same_reset_state_but_noise_support_matches():
    local = Walker2d(nthread=1)
    reference = gym.make("Walker2d-v5").unwrapped
    try:
        np.random.seed(914)
        local_obs = local.reset()
        reference_obs, _ = reference.reset(seed=914)
        assert not np.array_equal(local_obs, reference_obs)
        for env in (local, reference):
            np.testing.assert_array_less(np.abs(env.data.qvel), 0.005)
            np.testing.assert_array_less(np.abs(env.data.qpos - env.model.qpos0), 0.005)
    finally:
        local.close()
        reference.close()


def test_gymnasium_style_is_endpoint_velocity_surrogate_not_stock_reward():
    local = Walker2d(nthread=1, apply_terminal_cost_on_done=False)
    reference = gym.make("Walker2d-v5", xml_file=str(XML)).unwrapped
    try:
        align_initial_state(local, reference, 914)
        action = np.random.default_rng(9).uniform(-1, 1, 6)
        _, cost, _, _ = local.step(action)
        _, reward, _, _, info = reference.step(action)
        expected_difference = info["x_velocity"] - reference.data.qvel[0]
        assert abs(expected_difference) > 0.01
        assert cost + reward == pytest.approx(expected_difference, abs=1e-12)
    finally:
        local.close()
        reference.close()
