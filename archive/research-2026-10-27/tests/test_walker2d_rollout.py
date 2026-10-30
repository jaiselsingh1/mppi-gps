import numpy as np
import pytest

from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


@pytest.fixture
def walker():
    env = Walker2d(cost_style="target_velocity", target_velocity=1.5)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.parametrize("vertical_weight", [0.0, 0.5])
def test_batch_rollout_matches_live_step_with_solver_warmstart(walker, vertical_weight):
    walker._w_vertical_velocity = vertical_weight
    np.random.seed(7)
    walker.reset()
    for phase in np.linspace(0.0, 2.0, 20):
        action = 0.25 * np.sin(phase + np.arange(walker.action_dim))
        walker.step(action)

    state = walker.get_state()
    warmstart = walker.get_warmstart()
    action = np.linspace(-0.4, 0.4, walker.action_dim)
    states, costs, _ = walker.batch_rollout(
        state,
        action.reshape(1, 1, -1),
        initial_warmstart=warmstart,
    )

    _, live_cost, _, _ = walker.step(action)

    np.testing.assert_allclose(states[0, 0], walker.get_state(), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(costs[0], live_cost, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("vertical_weight", [0.0, 0.5])
def test_fallen_rollout_uses_absorbing_failure_cost(walker, vertical_weight):
    walker._w_vertical_velocity = vertical_weight
    np.random.seed(123)
    walker.reset()
    actions = np.random.default_rng(0).uniform(
        -1.0, 1.0, (1, 80, walker.action_dim)
    )
    states, costs, _ = walker.batch_rollout(
        walker.get_state(),
        actions,
        initial_warmstart=walker.get_warmstart(),
    )

    live_costs = []
    live_states = []
    done = False
    for action in actions[0]:
        _, cost, done, _ = walker.step(action)
        live_costs.append(cost)
        live_states.append(walker.get_state())
        if done:
            break

    assert done
    np.testing.assert_allclose(
        states[0, :len(live_states)],
        np.asarray(live_states),
        rtol=0.0,
        atol=1e-12,
    )
    remaining_steps = actions.shape[1] - len(live_costs)
    expected = sum(live_costs) + walker._w_unhealthy * remaining_steps
    np.testing.assert_allclose(costs[0], expected, rtol=0.0, atol=1e-10)


def test_seeded_mppi_noise_can_be_replayed(walker):
    controller = MPPI(walker, MPPIConfig(H=2, K=4), seed=17)
    first = controller._sample_noise()
    controller.reset(seed=17)
    replay = controller._sample_noise()
    controller.reset(seed=18)
    different = controller._sample_noise()

    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, different)


def test_lowpass_noise_is_temporally_correlated_without_changing_sigma(walker):
    controller = MPPI(
        walker,
        MPPIConfig(
            K=4_000,
            H=48,
            noise_sigma=0.3,
            noise_lowpass_cutoff_hz=23.4375,
            noise_lowpass_sample_rate_hz=125.0,
            noise_lowpass_order=2,
        ),
        seed=23,
    )

    noise = controller._sample_noise()
    for step in (0, controller.H // 2, controller.H - 1):
        np.testing.assert_allclose(
            np.std(noise[:, step, :], axis=0),
            np.full(controller.nu, 0.3),
            rtol=0.08,
            atol=0.015,
        )
    lag_one_correlation = np.mean(noise[:, :-1] * noise[:, 1:]) / np.mean(
        noise[:, :-1] ** 2
    )
    assert lag_one_correlation > 0.65


def test_mppi_rejects_one_step_horizon(walker):
    with pytest.raises(ValueError, match="at least 2"):
        MPPI(walker, MPPIConfig(H=1))


def test_corrupt_unscored_joint_rejects_rollout(walker):
    walker.reset()
    states = np.tile(walker.get_state(), (1, 2, 1))
    states[0, 0, -1] = np.nan
    actions = np.zeros((1, 2, walker.action_dim))
    costs = walker.rollout_cost(states, actions, np.zeros((1, 2, 0)))
    assert np.isinf(costs[0])


def test_vertical_speed_cost_is_symmetric_and_does_not_penalize_horizontal_motion(walker):
    states = np.tile(walker.get_state(), (1, 3, 1))
    states[..., 10] = 1.5
    states[0, :, 11] = [-2.0, 0.0, 2.0]
    actions = np.zeros((1, 3, walker.action_dim))
    original = walker.running_cost(states, actions)
    walker._w_vertical_velocity = 0.5
    penalized = walker.running_cost(states, actions)
    np.testing.assert_allclose(penalized - original, [[2.0, 0.0, 2.0]])


def test_invalid_sample_batch_cannot_produce_a_teacher_action(walker):
    controller = MPPI(walker, MPPIConfig(H=2, K=2))
    with pytest.raises(FloatingPointError, match="no finite trajectory scores"):
        controller._softmin_weights(np.array([np.inf, np.nan]), 1.0)
