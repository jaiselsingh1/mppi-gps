import numpy as np
import pytest

from src.envs.walker2d import Walker2d
from src.utils.rollout_data import (
    episode_action_pairs,
    pre_action_states,
    pre_fall_action_mask,
)


@pytest.fixture
def walker():
    env = Walker2d(cost_style="target_velocity", target_velocity=1.5)
    try:
        yield env
    finally:
        env.close()


def test_pre_action_observations_match_live_observations_before_each_action(walker):
    np.random.seed(4)
    walker.reset()
    initial_state = walker.get_state()
    initial_warmstart = walker.get_warmstart()
    actions = np.random.default_rng(1).uniform(-0.3, 0.3, (2, 6, walker.action_dim))
    rollout_states, _, _ = walker.batch_rollout(
        initial_state, actions, initial_warmstart=initial_warmstart
    )
    observations = walker.rollout_states_to_obs(pre_action_states(initial_state, rollout_states))

    for sample in range(2):
        walker.set_state(initial_state)
        walker.set_warmstart(initial_warmstart)
        for t, action in enumerate(actions[sample]):
            np.testing.assert_allclose(
                observations[sample, t], walker._get_obs(), rtol=0, atol=1e-12
            )
            walker.step(action)


def test_same_index_pairing_would_score_the_post_action_state(walker):
    np.random.seed(4)
    walker.reset()
    initial_state = walker.get_state()
    actions = np.full((1, 3, walker.action_dim), 0.2)
    rollout_states, _, _ = walker.batch_rollout(
        initial_state, actions, initial_warmstart=walker.get_warmstart()
    )
    aligned = pre_action_states(initial_state, rollout_states)
    np.testing.assert_array_equal(aligned[:, 0], initial_state[None])
    np.testing.assert_array_equal(aligned[:, 1:], rollout_states[:, :-1])
    assert not np.array_equal(aligned[:, 0], rollout_states[:, 0])


def test_pre_fall_mask_keeps_the_falling_action_and_drops_later_ones():
    healthy = np.array(
        [
            [True, True, False, True, False],
            [True, True, True, True, True],
            [False, True, True, True, True],
        ]
    )
    np.testing.assert_array_equal(
        pre_fall_action_mask(healthy),
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False],
        ],
    )


def test_pre_fall_mask_matches_walker_absorbing_cost_prefix(walker):
    np.random.seed(123)
    walker.reset()
    actions = np.random.default_rng(0).uniform(-1.0, 1.0, (1, 80, walker.action_dim))
    states, costs, sensors = walker.batch_rollout(
        walker.get_state(), actions, initial_warmstart=walker.get_warmstart()
    )
    healthy = walker._is_healthy(states[..., 2], states[..., 3])
    mask = pre_fall_action_mask(healthy)
    assert not mask.all()

    running = walker.running_cost(states, actions, sensors)
    first_fall = int(np.argmax(~healthy[0]))
    expected = (
        np.where(mask, running, 0.0).sum()
        + walker._w_unhealthy * np.sum(~mask)
        + walker.terminal_cost(states[:, first_fall])[0]
    )
    np.testing.assert_allclose(costs[0], expected, rtol=0, atol=1e-10)


def test_episode_pairs_drop_only_the_final_post_action_state():
    states = np.arange(4)[:, None] * np.ones((4, 2))
    actions = np.arange(3)[:, None] * np.ones((3, 1))
    paired_states, paired_actions = episode_action_pairs(states, actions)
    np.testing.assert_array_equal(paired_states, states[:3])
    np.testing.assert_array_equal(paired_actions, actions)
    with pytest.raises(ValueError, match="initial state"):
        episode_action_pairs(states[:3], actions)


def test_shape_validation():
    with pytest.raises(ValueError, match="rollout_states"):
        pre_action_states(np.zeros(3), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="initial_state"):
        pre_action_states(np.zeros(2), np.zeros((2, 4, 3)))
    with pytest.raises(ValueError, match="post_action_healthy"):
        pre_fall_action_mask(np.ones(3, dtype=bool))
