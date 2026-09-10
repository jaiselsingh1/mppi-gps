import numpy as np
import pytest

from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


class RecordingEnv:
    """Small deterministic rollout model for checking optimizer sequencing."""

    action_dim = 1
    action_bounds = (np.array([-1.0]), np.array([1.0]))

    def __init__(self):
        self.calls = []

    def batch_rollout(self, state, actions, initial_warmstart=None):
        self.calls.append((state.copy(), actions.copy(), initial_warmstart))
        states = state + np.cumsum(actions, axis=1)
        costs = np.sum((states - 0.5) ** 2 + 0.01 * actions ** 2, axis=(1, 2))
        sensors = np.full((*actions.shape[:2], 1), len(self.calls), dtype=float)
        return states, costs, sensors


def test_default_matches_single_pass_update_exactly_across_seeded_calls():
    env = RecordingEnv()
    cfg = MPPIConfig(K=4, H=3, lam=0.7, noise_sigma=0.3)
    controller = MPPI(env, cfg, seed=17)
    rng = np.random.default_rng(17)
    reference_nominal = np.zeros((cfg.H, env.action_dim))
    state = np.array([0.2])

    for _ in range(3):
        standard = rng.standard_normal((cfg.K, cfg.H, env.action_dim))
        noise = np.einsum('khi,ji->khj', standard, np.array([[cfg.noise_sigma]]))
        sampled = np.clip(reference_nominal[None] + noise, *env.action_bounds)
        eps = sampled - reference_nominal[None]
        states = state + np.cumsum(sampled, axis=1)
        costs = np.sum((states - 0.5) ** 2 + 0.01 * sampled ** 2, axis=(1, 2))
        weights = np.exp(-(costs - costs.min()) / cfg.lam)
        weights /= weights.sum()
        reference_nominal = np.clip(
            reference_nominal + np.einsum('k, kha -> ha', weights, eps),
            *env.action_bounds,
        )
        expected_action = reference_nominal[0].copy()
        reference_nominal[:-1] = reference_nominal[1:]
        reference_nominal[-1] = reference_nominal[-2].copy()

        action, info = controller.plan_step(state)

        np.testing.assert_array_equal(action, expected_action)
        np.testing.assert_array_equal(controller.U, reference_nominal)
        np.testing.assert_array_equal(controller._last_actions, sampled)
        np.testing.assert_array_equal(controller._last_states, states)
        np.testing.assert_array_equal(controller._last_costs, costs)
        np.testing.assert_array_equal(controller._last_weights, weights)
        assert info['optimization_iterations'] == 1
        assert info['cost_min'] == float(costs.min())
    assert len(env.calls) == 3


def test_initial_refinement_uses_same_state_and_shifts_only_after_final_pass(monkeypatch):
    env = RecordingEnv()
    controller = MPPI(env, MPPIConfig(K=1, H=3, initial_iterations=2))
    noise = np.array([[[0.1], [0.2], [0.3]]])
    monkeypatch.setattr(controller, '_sample_noise', lambda: noise.copy())
    state = np.array([0.0])
    warmstart = np.array([0.25])

    action, info = controller.plan_step(state, initial_warmstart=warmstart)

    np.testing.assert_allclose(action, [0.2], rtol=0, atol=1e-15)
    np.testing.assert_allclose(controller.U[:, 0], [0.4, 0.6, 0.6], rtol=0, atol=1e-15)
    assert info['optimization_iterations'] == 2
    assert len(env.calls) == 2
    for recorded_state, _, recorded_warmstart in env.calls:
        np.testing.assert_array_equal(recorded_state, state)
        assert recorded_warmstart is warmstart
    np.testing.assert_array_equal(state, [0.0])
    np.testing.assert_array_equal(warmstart, [0.25])
    np.testing.assert_array_equal(controller._last_actions, env.calls[-1][1])
    np.testing.assert_array_equal(controller._last_sensordata, np.full((1, 3, 1), 2.0))
    assert info['cost_min'] == float(controller._last_costs[0])

    _, second_info = controller.plan_step(state, initial_warmstart=warmstart)
    assert len(env.calls) == 3
    assert second_info['optimization_iterations'] == 1
    controller.reset(seed=17)
    _, reset_info = controller.plan_step(state, initial_warmstart=warmstart)
    assert len(env.calls) == 5
    assert reset_info['optimization_iterations'] == 2


@pytest.mark.parametrize('full_nominal', [True, False])
def test_initial_refinement_applies_supplied_nominal_only_once(monkeypatch, full_nominal):
    env = RecordingEnv()
    controller = MPPI(env, MPPIConfig(K=1, H=3, initial_iterations=2))
    monkeypatch.setattr(controller, '_sample_noise', lambda: np.full((1, 3, 1), 0.1))
    nominal = np.full((3, 1), 0.2)
    kwargs = {'nominal': nominal} if full_nominal else {'nominal_first': nominal[0]}

    action, _ = controller.plan_step(np.array([0.0]), **kwargs)

    np.testing.assert_allclose(action, [0.4], rtol=0, atol=1e-15)
    expected = [0.4, 0.4, 0.4] if full_nominal else [0.4, 0.2, 0.2]
    np.testing.assert_allclose(controller._last_actions[0, :, 0], expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(nominal, np.full((3, 1), 0.2))


def test_initial_refinement_calls_gps_callbacks_per_pass_and_keeps_final_scores():
    env = RecordingEnv()
    controller = MPPI(env, MPPIConfig(K=2, H=3, lam=0.7, initial_iterations=2), seed=13)
    callback_order, priors, coupled = [], [], []

    def prior(states, actions):
        callback_order.append('prior')
        tracking = np.sum((actions - 0.1 * states) ** 2, axis=(1, 2))
        priors.append(tracking.copy())
        return tracking

    def coupling(*, states, actions, costs, base_score, lam):
        callback_order.append('coupling')
        np.testing.assert_array_equal(base_score, costs + priors[-1])
        assert lam == controller.lam
        score = base_score + np.array([0.4, -0.2]) * len(priors)
        coupled.append({'states': states.copy(), 'actions': actions.copy(),
                        'costs': costs.copy(), 'score': score.copy()})
        return {'score': score, 'info': {'active': 1.0, 'score_mean': float(score.mean())}}

    action, info = controller.plan_step(np.array([0.2]), prior_cost=prior, coupling=coupling)

    assert callback_order == ['prior', 'coupling', 'prior', 'coupling']
    final = coupled[-1]
    weights = np.exp(-(final['score'] - final['score'].min()) / controller.lam)
    weights /= weights.sum()
    np.testing.assert_array_equal(controller._last_states, final['states'])
    np.testing.assert_array_equal(controller._last_actions, final['actions'])
    np.testing.assert_array_equal(controller._last_costs, final['costs'])
    np.testing.assert_array_equal(controller._last_weights, weights)
    np.testing.assert_array_equal(controller._last_sensordata, np.full((2, 3, 1), 2.0))
    np.testing.assert_allclose(action, np.einsum('k,ka->a', weights, final['actions'][:, 0]))
    assert info['cost_track_mean'] == float(priors[-1].mean())
    assert info['cost_s_mean'] == float(final['score'].mean())
    assert info['coupling_score_mean'] == float(final['score'].mean())
    assert info['coupling_active'] == 1.0
    assert info['optimization_iterations'] == 2


@pytest.mark.parametrize('iterations', [0, -1, 1.5, True, False, None])
def test_initial_iterations_requires_a_positive_integer(iterations):
    with pytest.raises(ValueError, match='initial_iterations must be a positive integer'):
        MPPI(RecordingEnv(), MPPIConfig(K=1, H=3, initial_iterations=iterations))
