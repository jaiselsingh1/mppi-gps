"""Mechanism checks for the isolated fixed-bank paired-endpoint planner."""
import copy

import numpy as np
import pytest

from src.gps.coupled_planner import MemoryArchive, local_discounted_score
from src.gps.paired_planner import PairedEndpointPlanner, finite_local_score
from src.utils.config import MPPIConfig


class ToyWalker:
    action_dim = 1
    action_bounds = (np.array([-1.]), np.array([1.]))
    _w_unhealthy = 5.
    _apply_terminal_cost_on_done = True

    def __init__(self, mean_falls=False):
        self.calls = []
        self.mean_falls = mean_falls

    @staticmethod
    def _is_healthy(z, angle):
        return (z > .8) & (z < 2.) & (np.abs(angle) < 1.)

    def running_cost(self, states, actions):
        return states[..., 4] ** 2 + .01 * np.sum(actions ** 2, axis=-1)

    def terminal_cost(self, states):
        return 20. * ~self._is_healthy(states[..., 2], states[..., 3])

    def batch_rollout(self, initial_state, actions, initial_warmstart=None):
        self.calls.append((initial_state.copy(), actions.copy(),
                           None if initial_warmstart is None else initial_warmstart.copy()))
        paths = np.broadcast_to(initial_state, (*actions.shape[:2], len(initial_state))).copy()
        paths[..., 4] = initial_state[4] + np.cumsum(actions[..., 0], axis=1)
        if self.mean_falls:
            paths[np.abs(actions[:, 0, 0]) < .2, :, 2] = .5
        costs, _ = local_discounted_score(paths, actions, self, 1., actions.shape[1])
        return paths, costs, np.empty((*actions.shape[:2], 0))


def state(y=0.):
    result = np.zeros(19)
    result[2], result[4] = 1.25, y
    return result


def config(**kwargs):
    return MPPIConfig(K=4, H=3, lam=.7, noise_sigma=.3, **kwargs)


def symmetric_noise(planner):
    noise = np.repeat(np.array([-.6, -.4, .4, .6])[:, None, None], planner.cfg.H, axis=1)
    planner.raw._sample_noise = lambda: noise.copy()


class ForbiddenValue:
    @property
    def metrics(self):
        raise AssertionError("Baseline accessed value metadata")

    def predict(self, *args):
        raise AssertionError("Baseline accessed value predictions")


class Value:
    def __init__(self, scale=-5., constant=20., support=True):
        self.scale, self.constant, self.support = scale, constant, support
        self.calls = []

    def predict(self, features, remaining):
        self.calls.append((features.copy(), remaining.copy()))
        support = self.support(features) if callable(self.support) else np.full(len(features), self.support)
        return (self.constant + self.scale * features[:, 2], support,
                np.where(support, 0., 10.))


def test_alpha_zero_does_not_access_value_or_metadata_and_preserves_exact_baseline():
    outcomes = []
    for value in (None, ForbiddenValue()):
        planner = PairedEndpointPlanner(ToyWalker(), config(), value=value, alpha=0., seed=8)
        action, info = planner.plan_step(state(), initial_warmstart=np.arange(9.))
        assert info["fallback_reason"] == "alpha_zero"
        assert not info["guidance_used"]
        assert not info["action_changed"]
        for key in ("predicted_values", "supported", "distances"):
            assert planner.last_decision[key] is None
        assert info["candidate_controls"] == 12
        assert info["verification_controls"] == 3
        assert len(planner.last_plans) == 5
        outcomes.append((action, planner.last_plan))
    for before, after in zip(*outcomes):
        np.testing.assert_array_equal(before, after)


def test_guidance_does_not_change_private_raw_nominal_proposals_or_rng():
    baseline = PairedEndpointPlanner(ToyWalker(), config(initial_iterations=2), seed=27)
    guided = PairedEndpointPlanner(ToyWalker(), config(initial_iterations=2), value=Value(), alpha=.5, seed=27)
    for physical_state in (state(), state(.1), state(-.4)):
        _, base_info = baseline.plan_step(physical_state)
        _, info = guided.plan_step(physical_state)
        assert info["guidance_used"]
        assert info["verification_controls"] == base_info["verification_controls"] == 3
        np.testing.assert_array_equal(guided.raw._last_actions, baseline.raw._last_actions)
        np.testing.assert_array_equal(guided.raw._last_plan, baseline.raw._last_plan)
        np.testing.assert_array_equal(guided.raw.U, baseline.raw.U)
        assert guided.raw._rng.bit_generator.state == baseline.raw._rng.bit_generator.state


def test_constant_value_with_partial_support_cannot_change_the_local_decision():
    baseline = PairedEndpointPlanner(ToyWalker(), config(), seed=1)
    value = Value(scale=0., support=lambda features: features[:, 2] >= -.1)
    guided = PairedEndpointPlanner(ToyWalker(), config(), value=value, alpha=.5, seed=1)
    for planner in (baseline, guided):
        symmetric_noise(planner)
    expected, _ = baseline.plan_step(state())
    actual, info = guided.plan_step(state())
    assert info["guidance_used"]
    assert info["baseline_endpoint_supported"]
    assert not guided.last_decision["supported"].all()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(guided.last_decision["scores"], guided.last_decision["local_costs"])


@pytest.mark.parametrize("alpha", [1e-12, .1, .5])
def test_unsupported_exact_baseline_forces_whole_decision_fallback(alpha):
    baseline = PairedEndpointPlanner(ToyWalker(), config(), seed=1)
    value = Value(scale=-1000., support=lambda features: np.abs(features[:, 2]) > .1)
    guided = PairedEndpointPlanner(ToyWalker(), config(), value=value, alpha=alpha, seed=1)
    for planner in (baseline, guided):
        symmetric_noise(planner)
    expected, _ = baseline.plan_step(state())
    actual, info = guided.plan_step(state())
    assert info["selected_source"] == "raw_mean"
    assert info["fallback_reason"] == "unsupported_baseline"
    assert not info["guidance_used"]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(guided.last_decision["scores"], guided.last_decision["local_costs"])
    assert np.isfinite(guided.last_decision["scores"]).all()


def test_relative_tail_shift_is_bounded_in_temperature_units():
    planner = PairedEndpointPlanner(ToyWalker(), config(), value=Value(scale=-1e9),
                                    alpha=.3, max_tail_shift_temperatures=2., seed=1)
    symmetric_noise(planner)
    planner.plan_step(state())
    bank = planner.last_decision
    assert np.max(np.abs(bank["corrections"])) == pytest.approx(.3 * 2 * .7)
    assert bank["corrections"][bank["baseline_index"]] == 0.
    np.testing.assert_allclose(bank["scores"] - bank["local_costs"], bank["corrections"], atol=1e-14)


def test_raw_samples_memory_and_one_raw_mean_have_no_learned_mean_work():
    archive = MemoryArchive(np.array([state()]), np.zeros((1, 9)),
                            np.full((1, 3, 1), .4), np.array([1000]))
    env = ToyWalker()
    planner = PairedEndpointPlanner(env, config(initial_iterations=3), archive=archive,
                                    value=Value(), alpha=.5, seed=5)
    warm = np.arange(9.)
    _, info = planner.plan_step(state(.7), initial_warmstart=warm)
    assert len(env.calls) == 5  # three raw passes, memory, one raw mean
    assert info["candidate_controls"] == 36
    assert info["memory_controls"] == 3
    assert info["verification_controls"] == 3
    assert len(planner.last_plans) == 6
    assert "guided_mean" not in planner.last_decision["sources"]
    np.testing.assert_array_equal(env.calls[-2][0], state(.7))
    np.testing.assert_array_equal(env.calls[-2][2], warm)
    np.testing.assert_array_equal(env.calls[-2][1], archive.plans)
    np.testing.assert_array_equal(env.calls[-1][1], planner.raw._last_plan[None])


def test_first_fall_has_one_terminal_charge_and_full_finite_absorbing_remainder():
    env = ToyWalker()
    plans = np.zeros((1, 3, 1))
    paths = np.array([[state(), state(), state()]])
    paths[0, 1, 2] = .5
    costs, healthy = finite_local_score(paths, plans, env, .5, 7)
    # A later healthy state does not undo the first fall at post-action index 1.
    expected = .5 * 20 + sum(.5 ** t * 5 for t in range(2, 7))
    assert costs[0] == pytest.approx(expected)
    assert not healthy[0]
    healthy_cost, _ = finite_local_score(np.array([[state(), state(), state()]]), plans, env, .5, 7)
    assert healthy_cost[0] == 0.  # no invented healthy continuation


def test_known_fallen_cost_is_identical_with_guidance_and_fallen_mean_rejected():
    banks = []
    for alpha in (0., .5):
        env = ToyWalker(mean_falls=True)
        planner = PairedEndpointPlanner(env, config(), value=Value(), alpha=alpha, seed=1)
        symmetric_noise(planner)
        action, _ = planner.plan_step(state())
        bank = planner.last_decision
        assert abs(action[0]) > .2
        fallen = ~bank["healthy"]
        assert fallen.any()
        np.testing.assert_array_equal(bank["scores"][fallen], bank["local_costs"][fallen])
        banks.append(bank["local_costs"])
    np.testing.assert_array_equal(*banks)


def test_fallen_baseline_abstains_and_horizon_boundary_never_reads_value():
    class EveryoneFalls(ToyWalker):
        def batch_rollout(self, initial_state, actions, initial_warmstart=None):
            paths, costs, sensors = super().batch_rollout(initial_state, actions, initial_warmstart)
            paths[..., 2] = .5
            return paths, costs, sensors
    planner = PairedEndpointPlanner(EveryoneFalls(), config(), value=Value(), alpha=.5)
    _, info = planner.plan_step(state())
    assert info["fallback_reason"] == "baseline_fallen"
    assert not info["guidance_used"]
    assert planner.acquisition_pair(np.random.default_rng(0), shadow_value=ForbiddenValue()) is None
    short = PairedEndpointPlanner(ToyWalker(), config(), remaining=2,
                                  value=ForbiddenValue(), alpha=.5)
    _, info = short.plan_step(state())
    assert info["fallback_reason"] == "horizon_exhausted"
    assert short.acquisition_pair(np.random.default_rng(0), shadow_value=ForbiddenValue()) is None


def test_shadow_acquisition_works_while_deployed_value_forbidden_without_simulation_or_rng_effects():
    env = ToyWalker()
    planner = PairedEndpointPlanner(env, config(), value=ForbiddenValue(), alpha=0., seed=8)
    symmetric_noise(planner)
    original_noise = planner.raw._sample_noise
    planner.raw._sample_noise = lambda: .1 * original_noise()
    planner.plan_step(state())
    before_calls = len(env.calls)
    before_rng = copy.deepcopy(planner.raw._rng.bit_generator.state)
    before_nominal = planner.raw.U.copy()
    shadow = Value(scale=-1e5, support=False)
    pair = planner.acquisition_pair(np.random.default_rng(1), shadow_value=shadow, probe_alpha=.5)
    assert len(shadow.calls) == 1
    assert not pair["candidate"]["supported"]
    assert pair["candidate_index"] != pair["baseline_index"]
    assert not pair["guidance_used"]
    assert pair["acquisition_mode"] == "guided"
    assert pair["remaining_tail"] == 997
    assert len(env.calls) == before_calls
    assert planner.raw._rng.bit_generator.state == before_rng
    np.testing.assert_array_equal(planner.raw.U, before_nominal)
    assert not np.shares_memory(pair["endpoints"], planner.last_decision["endpoints"])


def test_equal_candidate_baseline_is_preserved_and_cold_start_uses_supplied_rng():
    planner = PairedEndpointPlanner(ToyWalker(), config(), seed=8)
    symmetric_noise(planner)
    planner.plan_step(state())
    pair = planner.acquisition_pair(np.random.default_rng(0), shadow_value=Value(scale=0.))
    assert pair["candidate_equals_baseline"]
    np.testing.assert_array_equal(pair["candidate_endpoint"], pair["baseline_endpoint"])
    first = planner.acquisition_pair(np.random.default_rng(31))
    second = planner.acquisition_pair(np.random.default_rng(31), mode="uniform", shadow_value=ForbiddenValue())
    assert first["candidate_index"] == second["candidate_index"]
    assert first["acquisition_mode"] == "uniform_no_value"
    assert second["predicted_values"] is None
    assert first["acquisition_scores"] is None
    assert second["acquisition_scores"] is None


def test_supported_training_probe_matches_same_alpha_deployment_including_tail_clipping():
    value = Value(scale=-1e9)
    planner = PairedEndpointPlanner(ToyWalker(), config(), value=value, alpha=.5, seed=8)
    symmetric_noise(planner)
    planner.plan_step(state())
    bank = planner.last_decision
    unbounded = bank["local_costs"] + .5 * planner.gamma ** config().H * bank["predicted_values"]
    assert int(np.argmin(unbounded)) != bank["selected_index"]
    pair = planner.acquisition_pair(np.random.default_rng(0), shadow_value=value, probe_alpha=.5)
    assert pair["candidate_index"] == bank["selected_index"]
    assert pair["candidate_equals_baseline"]
    np.testing.assert_array_equal(pair["acquisition_scores"], bank["scores"])
    assert int(np.argmin(pair["acquisition_scores"])) == pair["candidate_index"]


def test_nonfinite_baseline_shadow_prediction_uses_uniform_even_with_finite_alternatives():
    class MissingBaseline(Value):
        def predict(self, features, remaining):
            costs, supported, distance = super().predict(features, remaining)
            costs[np.abs(features[:, 2]) < .1] = np.nan
            return costs, supported, distance
    planner = PairedEndpointPlanner(ToyWalker(), config(), seed=8)
    symmetric_noise(planner)
    planner.plan_step(state())
    actual = planner.acquisition_pair(np.random.default_rng(31), shadow_value=MissingBaseline())
    expected = planner.acquisition_pair(np.random.default_rng(31), mode="uniform")
    assert actual["acquisition_mode"] == "uniform_no_finite_baseline_prediction"
    assert actual["candidate_index"] == expected["candidate_index"]
    assert actual["acquisition_scores"] is None


def test_predictor_shape_errors_and_invalid_acquisition_calls_are_explicit():
    planner = PairedEndpointPlanner(ToyWalker(), config())
    with pytest.raises(RuntimeError, match="Plan a decision"):
        planner.acquisition_pair(np.random.default_rng(0))
    planner.plan_step(state())
    with pytest.raises(ValueError, match="mode"):
        planner.acquisition_pair(np.random.default_rng(0), mode="other")
    invalid = Value()
    invalid.predict = lambda features, remaining: (np.zeros((len(features), 1)), np.ones(len(features)), np.zeros(len(features)))
    with pytest.raises(ValueError, match="batch shapes"):
        planner.acquisition_pair(np.random.default_rng(0), shadow_value=invalid)


def test_reset_clears_acquisition_bank_and_restores_remaining_horizon():
    planner = PairedEndpointPlanner(ToyWalker(), config(), total_steps=20, remaining=10)
    planner.plan_step(state())
    assert planner.remaining == 9
    planner.reset(seed=12)
    assert planner.remaining == 20
    assert planner.last_decision is None
    assert planner.last_plan is None
    with pytest.raises(ValueError):
        planner.reset(remaining=0)


@pytest.mark.parametrize("remaining", [1, 2])
def test_short_remaining_horizon_scores_only_executable_prefix_but_charges_full_bank_work(remaining):
    env = ToyWalker()
    planner = PairedEndpointPlanner(env, config(initial_iterations=2), remaining=remaining,
                                    value=ForbiddenValue(), alpha=.5, seed=17)
    action, info = planner.plan_step(state(.2))
    bank = planner.last_decision
    mean_paths = np.broadcast_to(state(.2), (1, config().H, 19)).copy()
    simulated_paths = np.concatenate([planner.raw._last_states, mean_paths])
    # Reconstruct the verified mean's deterministic physical rollout directly,
    # without asking the environment to perform uncharged verification again.
    simulated_paths[-1, :, 4] = .2 + np.cumsum(planner.raw._last_plan[:, 0])
    expected, _ = local_discounted_score(simulated_paths, bank["plans"], env, .999, remaining)
    np.testing.assert_allclose(bank["local_costs"], expected)
    np.testing.assert_array_equal(bank["endpoints"], simulated_paths[:, remaining - 1])
    np.testing.assert_array_equal(action, bank["plans"][int(np.argmin(expected)), 0])
    assert info["fallback_reason"] == "horizon_exhausted"
    assert bank["H"] == remaining
    assert info["candidate_controls"] == 2 * config().K * config().H
    assert info["verification_controls"] == config().H
    assert len(env.calls) == 3
    assert planner.remaining == remaining - 1


@pytest.mark.parametrize("direction", [-1., 1.])
def test_saturated_native_bounds_hold_for_raw_mean_memory_and_selected_plan(direction):
    class StrictWalker(ToyWalker):
        action_dim = 6
        action_bounds = (np.full(6, -1.), np.ones(6))

        def batch_rollout(self, initial_state, actions, initial_warmstart=None):
            low, high = self.action_bounds
            assert np.all(actions >= low) and np.all(actions <= high)
            return super().batch_rollout(initial_state, actions, initial_warmstart)

    env = StrictWalker()
    cfg = MPPIConfig(K=32, H=48, lam=3., noise_sigma=.3)
    archive = MemoryArchive(np.array([state()] * 8), np.zeros((8, 9)),
        np.full((8, cfg.H, env.action_dim), direction), np.full(8, 1000))
    planner = PairedEndpointPlanner(env, cfg, archive=archive, value=Value(),
                                    alpha=.5, memory_candidates=8, seed=29)
    planner.raw._sample_noise = lambda: np.full((cfg.K, cfg.H, env.action_dim), direction * 10.)
    action, info = planner.plan_step(state())
    low, high = env.action_bounds
    for array in (action, planner.raw._last_plan, planner.last_plans, planner.last_plan):
        assert np.isfinite(array).all()
        assert np.all(array >= low) and np.all(array <= high)
    assert len(env.calls) == 3  # raw bank, replayed memory, verified raw mean
    assert info["verification_controls"] == cfg.H
    assert info["memory_controls"] == 8 * cfg.H
