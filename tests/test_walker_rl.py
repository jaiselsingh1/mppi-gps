"""The RL adapter changes failure accounting, not Walker physics or task cost."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.envs.walker2d import Walker2d
from src.rl.walker import WalkerRL, absorbing_failure_tail


_TASK_PATH = Path(__file__).resolve().parents[1] / "configs/walker2d_task.json"
_TASK = json.loads(_TASK_PATH.read_text())


@pytest.fixture
def walkers():
    adapter = WalkerRL()
    local = Walker2d(**_TASK, nthread=1)
    try:
        yield adapter, local
    finally:
        adapter.close()
        local.close()


def _assert_rng_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def _near_fall_state(walker):
    """Start healthy, but cross the torso-angle boundary on the next step."""
    state = walker.get_state()
    state[3] = 0.99999
    state[1 + walker.model.nq:] = 0.0
    state[1 + walker.model.nq + 2] = 5.0
    assert walker._is_healthy(state[2], state[3])
    return state


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.995, 0.9999])
@pytest.mark.parametrize("cost", [0.0, 5.0])
def test_analytic_tail_matches_explicit_absorbing_rewards(gamma, cost):
    # The first absorbing reward occurs ONE transition after the fall reward.
    count = 10_000
    explicit = -cost * np.sum(gamma ** np.arange(1, count + 1))
    remainder = -cost * gamma ** (count + 1) / (1.0 - gamma)
    assert absorbing_failure_tail(gamma, cost) == pytest.approx(
        explicit + remainder, rel=1e-12, abs=1e-12
    )


def test_frozen_failure_tail_is_minus_995():
    assert absorbing_failure_tail(0.995, _TASK["unhealthy_cost_weight"]) == pytest.approx(-995.0)


def test_seeded_reset_matches_historical_walker_and_continues_its_sequence(walkers):
    adapter, local = walkers
    original_global_rng = np.random.get_state()
    try:
        np.random.seed(31)
        first = local.reset()
        first_state = local.get_state()
        second = local.reset()
        second_state = local.get_state()
    finally:
        np.random.set_state(original_global_rng)

    actual_first, _ = adapter.reset(seed=31)
    np.testing.assert_array_equal(actual_first, first)
    np.testing.assert_array_equal(adapter.walker.get_state(), first_state)
    actual_second, _ = adapter.reset()
    np.testing.assert_array_equal(actual_second, second)
    np.testing.assert_array_equal(adapter.walker.get_state(), second_state)
    repeated, _ = adapter.reset(seed=31)
    np.testing.assert_array_equal(repeated, first)


def test_constructor_and_resets_do_not_modify_global_numpy_rng():
    original = np.random.get_state()
    adapter = WalkerRL()
    try:
        _assert_rng_equal(np.random.get_state(), original)
        adapter.reset(seed=7)
        _assert_rng_equal(np.random.get_state(), original)
        adapter.reset()
        _assert_rng_equal(np.random.get_state(), original)
    finally:
        adapter.close()


def test_local_physics_spaces_and_control_timing_are_unchanged(walkers):
    adapter, local = walkers
    obs, _ = adapter.reset(seed=7)
    assert adapter.walker.model.nq == local.model.nq == 9
    assert adapter.walker.model.nv == local.model.nv == 9
    assert adapter.walker.model.nu == local.model.nu == 6
    assert adapter.walker._nthread == 1
    assert adapter.walker.model.opt.timestep == local.model.opt.timestep == 0.002
    assert adapter.walker._frame_skip == local._frame_skip == 4
    assert adapter.walker._dt == local._dt == 0.008
    assert obs.shape == adapter.observation_space.shape == (17,)
    assert obs.dtype == adapter.observation_space.dtype == np.float64
    assert adapter.observation_space.contains(obs)
    assert adapter.action_space.shape == (6,)
    low, high = local.action_bounds
    np.testing.assert_array_equal(adapter.action_space.low, low)
    np.testing.assert_array_equal(adapter.action_space.high, high)
    initial_time = adapter.walker.data.time
    adapter.step(np.zeros(6))
    assert adapter.walker.data.time - initial_time == pytest.approx(0.008)


def test_snapshot_restoration_and_live_steps_are_bit_identical(walkers):
    adapter, local = walkers
    adapter.reset(seed=30)
    local.reset(state=adapter.walker.get_state())
    local.set_warmstart(adapter.walker.get_warmstart())
    for phase in np.linspace(0.0, 1.0, 12):
        _, _, done, _ = local.step(0.2 * np.sin(phase + np.arange(6)))
        assert not done

    state, warmstart = local.get_state(), local.get_warmstart()
    # Reconstruct both from a full physics snapshot and the separate solver state.
    expected_obs = local.reset(state=state)
    local.set_warmstart(warmstart)
    actual_obs, _ = adapter.reset(options={"state": state, "warmstart": warmstart})
    np.testing.assert_array_equal(actual_obs, expected_obs)
    np.testing.assert_array_equal(adapter.walker.get_state(), state)
    np.testing.assert_array_equal(adapter.walker.get_warmstart(), warmstart)

    for step, phase in enumerate(np.linspace(0.0, 0.5, 5), start=1):
        # Non-float32-representable actions detect an accidental cast in the adapter.
        action = 0.234567891234 * np.cos(phase + np.arange(6))
        obs, cost, done, local_info = local.step(action)
        actual, reward, terminated, truncated, info = adapter.step(action)
        assert not done and not terminated and not truncated
        np.testing.assert_array_equal(actual, obs)
        np.testing.assert_array_equal(adapter.walker.get_state(), local.get_state())
        np.testing.assert_array_equal(adapter.walker.get_warmstart(), local.get_warmstart())
        np.testing.assert_array_equal(adapter.walker.data.ctrl, action)
        assert reward == -cost == info["raw_reward"] == info["training_reward"]
        assert info["task_cost"] == local_info["cost"]
        assert info["failure_tail"] == 0.0
        assert info["elapsed_steps"] == step


def test_observations_keep_local_position_exclusion_and_velocity_clipping(walkers):
    adapter, local = walkers
    adapter.reset(seed=7)
    state = adapter.walker.get_state()
    state[1] = 123.0  # Global x is not part of the 17-dimensional policy observation.
    state[1 + local.model.nq:] = np.linspace(-20.0, 20.0, local.model.nv)
    expected = local.reset(state=state)
    obs, _ = adapter.reset(options={"state": state})
    np.testing.assert_array_equal(obs, expected)
    np.testing.assert_array_equal(obs[:8], state[2:10])
    np.testing.assert_array_equal(obs[8:], np.clip(state[10:], -10.0, 10.0))


def test_fall_adds_future_tail_once_and_preserves_current_task_penalties(walkers):
    adapter, local = walkers
    adapter.reset(seed=7)
    state = _near_fall_state(adapter.walker)
    adapter.reset(options={"state": state})
    local.reset(state=state)
    action = np.linspace(-0.1, 0.1, 6)
    expected_obs, cost, done, local_info = local.step(action)
    obs, reward, terminated, truncated, info = adapter.step(action)
    assert done and terminated and not truncated
    np.testing.assert_array_equal(obs, expected_obs)
    np.testing.assert_array_equal(adapter.walker.get_state(), local.get_state())
    assert local_info["fall_cost"] == _TASK["terminal_unhealthy_weight"] == 20.0
    running = local.running_cost(local.get_state()[None, None], action[None, None]).item()
    assert cost == running + 20.0
    assert info["task_cost"] == cost
    assert info["raw_reward"] == -cost
    assert info["failure_tail"] == pytest.approx(-995.0)
    assert reward == info["training_reward"] == -cost + info["failure_tail"]
    # There is no second terminal penalty, and no absorbing reward at exponent zero.
    assert reward + cost == pytest.approx(-995.0)
    with pytest.raises(RuntimeError):
        adapter.step(action)


def test_time_limit_returns_final_observation_without_failure_tail_or_autoreset():
    adapter = WalkerRL(max_episode_steps=1)
    try:
        start, _ = adapter.reset(seed=7)
        obs, reward, terminated, truncated, info = adapter.step(np.zeros(6))
        assert not terminated and truncated
        assert info["failure_tail"] == 0.0
        assert reward == info["raw_reward"] == -info["task_cost"]
        assert info["elapsed_steps"] == 1
        np.testing.assert_array_equal(obs, adapter.walker._get_obs())
        assert not np.array_equal(obs, start)
        assert adapter.walker.data.time == pytest.approx(0.008)
        with pytest.raises(RuntimeError):
            adapter.step(np.zeros(6))
        reset_obs, _ = adapter.reset(seed=7)
        np.testing.assert_array_equal(reset_obs, start)
    finally:
        adapter.close()


def test_fall_on_last_allowed_step_is_terminal_not_truncated():
    adapter = WalkerRL(max_episode_steps=1)
    try:
        adapter.reset(seed=7)
        adapter.reset(options={"state": _near_fall_state(adapter.walker)})
        _, _, terminated, truncated, info = adapter.step(np.zeros(6))
        assert terminated and not truncated
        assert info["elapsed_steps"] == 1
        assert info["failure_tail"] == pytest.approx(-995.0)
    finally:
        adapter.close()


def test_step_requires_reset():
    adapter = WalkerRL()
    try:
        with pytest.raises(RuntimeError):
            adapter.step(np.zeros(6))
        adapter.reset(seed=7)
        adapter.step(np.zeros(6))
    finally:
        adapter.close()


@pytest.mark.parametrize("action", [
    np.zeros(5), np.zeros((1, 6)), np.full(6, np.nan), np.full(6, np.inf),
    np.full(6, 1.01), np.full(6, -1.01),
])
def test_invalid_actions_are_rejected_before_physics_changes(walkers, action):
    adapter, _ = walkers
    adapter.reset(seed=7)
    state, warmstart = adapter.walker.get_state(), adapter.walker.get_warmstart()
    with pytest.raises(ValueError):
        adapter.step(action)
    np.testing.assert_array_equal(adapter.walker.get_state(), state)
    np.testing.assert_array_equal(adapter.walker.get_warmstart(), warmstart)
    _, _, _, _, info = adapter.step(np.zeros(6))
    assert info["elapsed_steps"] == 1
