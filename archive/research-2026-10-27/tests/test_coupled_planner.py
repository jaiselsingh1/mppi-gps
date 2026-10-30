import numpy as np
import pytest

from src.gps.coupled_planner import CoupledPlanner, MemoryArchive, local_discounted_score
from src.utils.config import MPPIConfig


class ToyWalker:
    """Post-transition Walker-shaped state, with one controllable coordinate."""
    action_dim = 1
    action_bounds = (np.array([-1.]), np.array([1.]))
    _w_unhealthy = 5.
    _apply_terminal_cost_on_done = True

    def __init__(self, mean_falls=False):
        self.calls = []
        self.mean_falls = mean_falls

    @staticmethod
    def _is_healthy(z, angle):
        return (z > .8) & (z < 2) & (np.abs(angle) < 1)

    def running_cost(self, states, actions):
        return np.square(states[..., 4]) + .01 * np.sum(actions ** 2, axis=-1)

    def terminal_cost(self, states):
        return 20. * ~self._is_healthy(states[..., 2], states[..., 3])

    def batch_rollout(self, state, actions, initial_warmstart=None):
        self.calls.append((state.copy(), actions.copy(), None if initial_warmstart is None else initial_warmstart.copy()))
        paths = np.broadcast_to(state, (*actions.shape[:2], len(state))).copy()
        paths[..., 4] = state[4] + np.cumsum(actions[..., 0], axis=1)
        if self.mean_falls:
            # Two safe modes have an unsafe average. This models the important
            # nonconvexity without relying on MuJoCo randomness in a unit test.
            paths[np.abs(actions[:, 0, 0]) < .2, :, 2] = .5
        costs, _ = local_discounted_score(paths, actions, self, 1., actions.shape[1])
        return paths, costs, np.empty((*actions.shape[:2], 0))


def state(coordinate=0.):
    result = np.zeros(19)
    result[2], result[4] = 1.25, coordinate
    return result


def config(**kwargs):
    return MPPIConfig(K=4, H=3, lam=.7, noise_sigma=.3, **kwargs)


class EndpointValue:
    metrics = {"calibration_eligible": True}

    def __init__(self, *, supported=True, distance=0.):
        self.calls = []
        self.supported, self.distance = supported, distance

    def predict(self, features, remaining):
        self.calls.append((features.copy(), remaining.copy()))
        return (20. - 5. * features[:, 2],
                np.full(len(features), self.supported),
                np.full(len(features), self.distance))


class ForbiddenValue:
    @property
    def metrics(self):
        raise AssertionError("Baseline must not inspect learned value metadata")

    def predict(self, *args):
        raise AssertionError("Baseline must not read learned endpoint values")


def test_zero_alpha_is_exact_baseline_with_absent_or_forbidden_values():
    plans = []
    for value in (None, ForbiddenValue()):
        planner = CoupledPlanner(ToyWalker(), config(), value=value, alpha=0., seed=19)
        action, info = planner.plan_step(state(), initial_warmstart=np.zeros(9))
        plans.append(planner.last_plan)
        assert not info["guidance_used"]
        assert info["fallback_reason"] == "alpha_zero"
        assert info["candidate_controls"] == 12
        assert info["verification_controls"] == 3
        assert info["memory_controls"] == 0
        np.testing.assert_array_equal(action, planner.last_plan[0])
    np.testing.assert_array_equal(*plans)


def test_value_reranking_never_changes_raw_proposals_nominal_or_rng():
    baseline = CoupledPlanner(ToyWalker(), config(), seed=31)
    guided = CoupledPlanner(ToyWalker(), config(), value=EndpointValue(), alpha=.5, seed=31)
    for physical_state in (state(), state(.2), state(-.3)):
        _, info = guided.plan_step(physical_state)
        _, baseline_info = baseline.plan_step(physical_state)
        assert info["guidance_used"]
        assert info["verification_controls"] == 6
        assert baseline_info["verification_controls"] == 3
        np.testing.assert_array_equal(guided.raw._last_actions, baseline.raw._last_actions)
        np.testing.assert_array_equal(guided.raw.U, baseline.raw.U)
        np.testing.assert_array_equal(guided.raw._last_plan, baseline.raw._last_plan)


def test_archive_proposals_are_replayed_from_current_state_and_warmstart():
    archive = MemoryArchive(np.array([state()]), np.ones((1, 9)),
                            np.full((1, 3, 1), .4), np.array([1000]))
    env = ToyWalker()
    planner = CoupledPlanner(env, config(), archive=archive, seed=4)
    live_state, warm = state(.7), np.arange(9.)
    _, info = planner.plan_step(live_state, initial_warmstart=warm)
    assert info["memory_controls"] == 3
    replay_state, replay_actions, replay_warm = env.calls[1]
    np.testing.assert_array_equal(replay_state, live_state)
    np.testing.assert_array_equal(replay_warm, warm)
    np.testing.assert_array_equal(replay_actions, archive.plans)
    # Archive physical snapshots never replace the actual rollout initial state.
    assert replay_state[4] != archive.states[0, 4]


def test_archive_roundtrip_padding_and_translation_invariance():
    states = np.array([state(float(i)) for i in range(5)])
    episode = dict(states=states, warmstarts=np.zeros((5, 9)),
                   actions=np.arange(4.).reshape(4, 1), remaining_start=10)
    archive = MemoryArchive.from_episodes([episode], H=3, stride=2)
    assert len(archive.states) == 2
    np.testing.assert_array_equal(archive.plans[1, :, 0], [2, 3, 3])
    np.testing.assert_array_equal(archive.remaining, [10, 8])
    restored = MemoryArchive.from_state_dict(archive.state_dict())
    shifted = states[2].copy()
    shifted[:2] += 10000  # elapsed simulation time and absolute horizontal x
    plans, distances = restored.retrieve(shifted, 8, 1)
    np.testing.assert_array_equal(plans[0], archive.plans[1])
    np.testing.assert_array_equal(distances, [0.])


def test_memory_retains_good_measured_suffix_despite_tiny_reset_differences():
    states = np.array([state(.001 * i) for i in range(8)])
    archive = MemoryArchive(states, np.zeros((8, 9)),
        np.broadcast_to(np.arange(8.).reshape(8, 1, 1), (8, 3, 1)),
        np.full(8, 1000), suffix_costs=np.array([100.] * 7 + [1.]))
    plans, _ = archive.retrieve(state(), 1000, k=4)
    identifiers = plans[:, 0, 0]
    assert 0 in identifiers and 1 in identifiers  # nearest half
    assert 7 in identifiers  # best measured return gets a replay opportunity
    assert len(set(identifiers)) == 4
    assert len(archive.retrieve(state(), 1000, k=1)[0]) == 1


def test_memory_quality_includes_absorbing_failure_cost_and_no_unknown_free_tail():
    episode = dict(states=np.array([state(), state(), state()]),
        warmstarts=np.zeros((3, 9)), actions=np.zeros((2, 1)),
        remaining_start=4, gamma=.5, costs=np.array([1., 22.]), terminated=True)
    archive = MemoryArchive.from_episodes([episode], H=3, stride=1)
    assert archive.suffix_costs[0] == pytest.approx(1 + .5 * 22 + .25 * 5 + .125 * 5)
    assert archive.suffix_discounts[0] == pytest.approx(1 + .5 + .25 + .125)
    episode["terminated"] = False
    unknown = MemoryArchive.from_episodes([episode], H=3, stride=1)
    assert np.isnan(unknown.suffix_costs).all()


def test_fallen_weighted_mean_is_replayed_and_rejected():
    env = ToyWalker(mean_falls=True)
    cfg = MPPIConfig(K=2, H=2, lam=1., noise_sigma=.3)
    value = EndpointValue()
    value.predict = lambda features, remaining: (np.full(len(features), 20.),
        np.ones(len(features), dtype=bool), np.zeros(len(features)))
    planner = CoupledPlanner(env, cfg, value=value, alpha=.5, seed=0)
    planner.raw._sample_noise = lambda: np.array([[[-.8], [-.8]], [[.8], [.8]]])
    # Equal local costs produce a raw mean of zero, which is unhealthy here.
    action, info = planner.plan_step(state())
    assert info["baseline_local_cost"] >= 20
    assert info["selected_source"] == "sample"
    assert abs(action[0]) == pytest.approx(.8)
    assert not bool(env._is_healthy(planner.last_candidate_endpoints[-1, 2],
                                    planner.last_candidate_endpoints[-1, 3]))


@pytest.mark.parametrize("direction", [-1., 1.])
def test_guided_mean_roundoff_is_bounded_before_verification_and_execution(monkeypatch, direction):
    class StrictWalker(ToyWalker):
        action_dim = 6
        action_bounds = (np.full(6, -1.), np.ones(6))

        def batch_rollout(self, initial_state, actions, initial_warmstart=None):
            low, high = self.action_bounds
            if np.any(actions < low) or np.any(actions > high):
                raise ValueError("Verification received an out-of-bounds action")
            return super().batch_rollout(initial_state, actions, initial_warmstart)

    env = StrictWalker()
    cfg = MPPIConfig(K=32, H=48, lam=3., noise_sigma=.3)
    proposals = np.full((40, cfg.H, env.action_dim), direction)
    archive = MemoryArchive(np.array([state()] * 8), np.zeros((8, 9)),
                            proposals[32:], np.full(8, 1000))
    planner = CoupledPlanner(env, cfg, archive=archive, value=EndpointValue(),
                             alpha=.5, memory_candidates=8)
    scores = np.random.default_rng(0).normal(size=40) * cfg.lam
    weights = np.exp(-(scores - scores.min()) / cfg.lam)
    weights /= weights.sum()
    unprojected = np.einsum("k,kha->ha", weights, proposals)
    # Normalized nonnegative weights still permit a one-ULP boundary violation.
    assert weights.sum() == 1.
    assert np.max(direction * unprojected) > 1.

    def raw_plan_step(initial_state, initial_warmstart=None):
        planner.raw._last_actions = proposals[:32].copy()
        planner.raw._last_states = env.batch_rollout(initial_state, proposals[:32])[0]
        planner.raw._last_plan = np.zeros((cfg.H, env.action_dim))
        return np.zeros(env.action_dim), {"optimization_iterations": 1}

    def controlled_scores(paths, plans, remaining, *, use_value):
        # Isolate the mean's finite-precision geometry from cost-model details,
        # then select it so verification, storage and returned action are checked.
        assert use_value
        bank_scores = np.r_[scores, 1e6] if len(plans) == 41 else np.array([-1e6])
        return np.zeros(len(plans)), bank_scores, np.ones(len(plans), dtype=bool), np.ones(len(plans), dtype=bool)

    monkeypatch.setattr(planner.raw, "plan_step", raw_plan_step)
    monkeypatch.setattr(planner, "_scores", controlled_scores)
    action, info = planner.plan_step(state())
    expected = np.clip(unprojected, *env.action_bounds)
    assert info["selected_source"] == "guided_mean"
    np.testing.assert_array_equal(env.calls[-1][1], expected[None])
    np.testing.assert_array_equal(planner.last_plans[-1], expected)
    np.testing.assert_array_equal(planner.last_plan, expected)
    np.testing.assert_array_equal(action, expected[0])
    assert np.isfinite(action).all() and action.shape == (6,)
    assert np.all(action >= env.action_bounds[0]) and np.all(action <= env.action_bounds[1])


@pytest.mark.parametrize("unsupported", ["flag", "distance", "nan"])
def test_missing_empirical_support_falls_back_to_the_identical_baseline(unsupported):
    value = EndpointValue(supported=unsupported != "flag", distance=10. if unsupported == "distance" else 0.)
    if unsupported == "nan":
        value.predict = lambda features, remaining: (np.full(len(features), np.nan),
            np.ones(len(features), dtype=bool), np.zeros(len(features)))
    guided = CoupledPlanner(ToyWalker(), config(), value=value, alpha=.5, seed=2)
    baseline = CoupledPlanner(ToyWalker(), config(), seed=2)
    action, info = guided.plan_step(state())
    expected, _ = baseline.plan_step(state())
    np.testing.assert_array_equal(action, expected)
    assert not info["guidance_used"]
    assert info["fallback_reason"] == "insufficient_support"


@pytest.mark.parametrize("alpha", [1e-12, 1e-6, .1, .5])
def test_partial_support_cannot_discard_the_exact_local_baseline(alpha):
    class ValueWithUnsupportedBaseline:
        metrics = {"calibration_eligible": True}

        def predict(self, features, remaining):
            supported = np.abs(features[:, 2]) > .1
            return np.full(len(features), 10.), supported, np.where(supported, 0., 2.)

    cfg = MPPIConfig(K=4, H=3, lam=.7, noise_sigma=.3, initial_iterations=1)
    baseline = CoupledPlanner(ToyWalker(), cfg, seed=7)
    guided = CoupledPlanner(ToyWalker(), cfg, value=ValueWithUnsupportedBaseline(),
                             alpha=alpha, seed=7)
    # Symmetry gives the actual raw MPPI weighted mean a near-zero endpoint;
    # all four nonzero sample endpoints are supported. The mean is the local
    # winner, and lies in the deliberately unsupported feature-space hole.
    noise = np.repeat(np.array([-.6, -.4, .4, .6])[:, None, None], cfg.H, axis=1)
    for planner in (baseline, guided):
        planner.raw._sample_noise = lambda: noise.copy()
    expected, base_info = baseline.plan_step(state())
    actual, info = guided.plan_step(state())
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(guided.last_plan, baseline.last_plan)
    np.testing.assert_array_equal(guided.raw._last_actions, baseline.raw._last_actions)
    assert abs(actual[0]) < 1e-12
    assert base_info["selected_source"] == "raw_mean"
    assert not info["baseline_endpoint_supported"]
    assert info["support_fraction"] == pytest.approx(.8)
    assert info["fallback_reason"] == "unsupported_baseline"
    assert not info["guidance_used"]
    assert info["verification_controls"] == cfg.H  # no extra guided mean


def test_calibration_is_required_and_horizon_boundary_does_not_read_a_value():
    value = EndpointValue()
    value.metrics = {"calibration_eligible": False}
    planner = CoupledPlanner(ToyWalker(), config(), value=value, alpha=.5)
    _, info = planner.plan_step(state())
    assert not value.calls
    assert info["fallback_reason"] == "not_calibrated"
    planner.value = ForbiddenValue()
    _, info = planner.plan_step(state(), remaining=2)
    assert info["fallback_reason"] == "horizon_exhausted"
    assert info["candidate_controls"] == 12  # all simulated H slots count
    assert not info["guidance_used"]


def test_supported_long_term_choice_can_have_worse_local_cost_without_a_veto():
    env = ToyWalker()
    cfg = MPPIConfig(K=4, H=2, lam=.5, noise_sigma=.3)
    value = EndpointValue()
    def predict(features, remaining):
        endpoint = features[:, 2]
        return np.where(endpoint > .1, 0., 100.), endpoint >= 0., np.zeros(len(endpoint))
    value.predict = predict
    planner = CoupledPlanner(env, cfg, value=value, alpha=.5, seed=0)
    planner.raw._sample_noise = lambda: np.array([[[-.8], [-.8]], [[-.5], [-.5]], [[.5], [.5]], [[.8], [.8]]])
    action, info = planner.plan_step(state())
    assert info["guidance_used"]
    assert .5 <= info["support_fraction"] < 1.
    assert action[0] > 0.
    assert info["selected_local_cost"] > info["baseline_local_cost"]
    assert planner.last_plan[:, 0].min() > 0.  # unsupported negative mode receives zero weight


def test_first_fall_charge_discount_and_finite_boundary_are_exact():
    env = ToyWalker()
    paths = np.broadcast_to(state(), (1, 4, 19)).copy()
    paths[0, :, 4] = [1, 2, 1000, 1000]
    paths[0, 1:, 2] = .5
    actions = np.zeros((1, 4, 1))
    cost, healthy = local_discounted_score(paths, actions, env, .5, 4)
    # First two measured running costs, terminal once, then absorbing failure.
    assert cost[0] == pytest.approx(1 + .5 * (4 + 20) + .25 * 5 + .125 * 5)
    assert not healthy[0]
    cutoff, _ = local_discounted_score(paths, actions, env, .5, 1)
    assert cutoff[0] == 1  # future fall is beyond this declared episode
    planner = CoupledPlanner(env, MPPIConfig(K=1, H=4), value=EndpointValue(),
                             alpha=.5, gamma=.5)
    local, guided, _, _ = planner._scores(paths, actions, 6, use_value=True)
    assert guided[0] - local[0] == pytest.approx(.5 * .5 ** 4 * (5 + .5 * 5))
    assert not planner.value.calls  # an absorbing state needs no learned tail


def test_initial_refinement_and_all_verification_queries_are_counted():
    env = ToyWalker()
    planner = CoupledPlanner(env, config(initial_iterations=3), seed=8)
    _, first = planner.plan_step(state())
    assert first["candidate_controls"] == 4 * 3 * 3
    assert first["verification_controls"] == 3
    assert len(env.calls) == 4
    _, second = planner.plan_step(state())
    assert second["candidate_controls"] == 4 * 3
    assert planner.remaining == 998
    planner.reset(remaining=1)
    planner.plan_step(state())
    with pytest.raises(ValueError, match="finite episode horizon"):
        planner.plan_step(state())
