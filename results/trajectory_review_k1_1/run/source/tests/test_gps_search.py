"""Pure contract tests: batch doubles only; no simulator or physics queries."""

import copy
import json

import numpy as np
import pytest

from src.gps.mppi_gps.search import CoupledSelector, IndependentMPPI
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


class BatchDouble:
    """Canned state markers and an algebraic score, not a physical task."""

    action_dim = 6
    action_bounds = (-np.ones(6), np.ones(6))
    _dt = 0.008

    def __init__(self, reject_means=False):
        self.calls = []
        self.reject_means = reject_means

    def batch_rollout(self, state, actions, initial_warmstart=None):
        self.calls.append((np.array(state), np.array(actions), copy.deepcopy(initial_warmstart)))
        k, h, _ = actions.shape
        posts = np.broadcast_to(state, (k, h, len(state))).copy()
        posts[..., 0] += np.arange(1, h + 1)[None]
        posts[..., 1:7] = actions
        costs = np.sum((actions - 0.2) ** 2, axis=(1, 2))
        if self.reject_means and k == 1:
            costs[:] = 1e8
        return posts, costs, np.zeros((k, h, 0))

    def rollout_states_to_obs(self, states):
        return np.asarray(states)[..., :17]


class MeterDouble:
    def __init__(self):
        self.events = []
        self.arrays = []

    def record(self, event, **fields):
        json.dumps(fields, allow_nan=False)
        self.events.append((event, fields))
        return len(self.events)

    def save_arrays(self, tag, **arrays):
        self.arrays.append((tag, {name: np.array(value) for name, value in arrays.items()}))
        return {"file": "pure-test", "group": tag}


class ActorDouble:
    policy_hash = "deliberately-poor-actor"

    def __init__(self, action=-1.0):
        self.action = action
        self.calls = []

    def predict(self, observations, *, meter):
        self.calls.append(np.array(observations))
        return np.full((len(observations), 6), self.action)


def setup_search(*, reject_means=False, iterations=2):
    env = BatchDouble(reject_means)
    cfg = MPPIConfig(
        K=5,
        H=4,
        noise_sigma=0.5,
        initial_iterations=iterations,
        noise_lowpass_cutoff_hz=10.0,
        noise_lowpass_sample_rate_hz=125.0,
    )
    state, warm = np.arange(19, dtype=float), np.linspace(-1, 1, 9)
    return env, cfg, IndependentMPPI(env, cfg, seed=17), state, warm


def assert_tree_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    else:
        assert left == right


def test_native_noise_passes_nominal_and_caches_are_reused_exactly():
    env, cfg, independent, state, warm = setup_search()
    native = MPPI(BatchDouble(), cfg, seed=17)
    reference = native.U.copy()
    native.plan_step(state, initial_warmstart=warm)
    bank = independent.propose(state, warm)
    np.testing.assert_array_equal(bank.proposal_actions, native.get_rollout_data()["actions"])
    np.testing.assert_array_equal(bank.post_states[:-1], native.get_rollout_data()["states"])
    np.testing.assert_array_equal(bank.raw_costs[:-1], native.get_rollout_data()["costs"])
    np.testing.assert_array_equal(bank.reference_u, reference)
    np.testing.assert_array_equal(bank.actions[-1], native.last_plan)
    assert_tree_equal(independent.snapshot()["native"], native.snapshot())
    assert [call[1].shape[0] for call in env.calls] == [5, 5, 1]
    independent.propose(state, warm)
    assert [call[1].shape[0] for call in env.calls] == [5, 5, 1, 5, 1]


def test_pre_action_alignment_and_bank_immutability():
    _, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)
    np.testing.assert_array_equal(bank.observations[:, 0], np.broadcast_to(state[:17], (6, 17)))
    np.testing.assert_array_equal(bank.observations[:, 1:], bank.post_states[:, :-1, :17])
    np.testing.assert_array_equal(bank.pre_action_states[:, 1:], bank.post_states[:, :-1])
    with pytest.raises(ValueError):
        bank.actions[0, 0, 0] = 3
    with pytest.raises(ValueError):
        bank.actions.setflags(write=True)
    with pytest.raises(TypeError):
        bank.diagnostics["lam"] = 2


def test_bad_policy_never_changes_private_nominal_rng_or_next_bank():
    _, _, independent, state, warm = setup_search()
    initial = independent.snapshot()
    bank = independent.propose(state, warm)
    private_before = independent.snapshot()
    selector = CoupledSelector(alpha=100)
    selector.select(bank, ActorDouble(-1), MeterDouble(), enabled=True)
    assert_tree_equal(independent.snapshot(), private_before)
    next_guided = independent.propose(state, warm)
    independent.restore(initial)
    replay = independent.propose(state, warm)
    selector.select(replay, None, MeterDouble(), enabled=False)
    next_raw = independent.propose(state, warm)
    assert bank.bank_digest == replay.bank_digest
    assert next_guided.bank_digest == next_raw.bank_digest
    np.testing.assert_array_equal(next_guided.actions, next_raw.actions)


def test_disabled_zero_alpha_and_unsupported_are_exact_baseline_without_actor_calls():
    env, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)
    actor, meter = ActorDouble(), MeterDouble()
    before = len(env.calls)
    off, off_info = CoupledSelector().select(bank, actor, meter, enabled=False)
    zero, zero_info = CoupledSelector(alpha=0).select(bank, actor, meter, enabled=True)
    unsupported, unsupported_info = CoupledSelector().select(
        bank, actor, meter, enabled=True, supported=False
    )
    np.testing.assert_array_equal(off, zero)
    np.testing.assert_array_equal(off, unsupported)
    assert not actor.calls
    assert len(env.calls) == before + 3  # a paid baseline verification in each call
    for info in (off_info, zero_info, unsupported_info):
        assert not info["influence_used"]
        assert not info["changed_action"]
        assert not info["numerical_revocation"]


def test_raw_and_selector_means_are_verified_not_assumed_better():
    env, _, independent, state, warm = setup_search(reject_means=True)
    bank = independent.propose(state, warm)
    best = int(np.argmin(bank.raw_costs[:-1]))
    np.testing.assert_array_equal(bank.raw_plan, bank.actions[best])
    assert bank.raw_costs[-1] == 1e8
    plan, info = CoupledSelector().select(bank, None, MeterDouble())
    assert info["baseline_origin"].startswith("bank:")
    np.testing.assert_array_equal(plan, bank.actions[int(info["baseline_origin"].split(":")[1])])
    assert env.calls[-1][1].shape == (1, 4, 6)


def test_numerical_actor_failure_revokes_and_preserves_same_bank_baseline():
    _, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)
    baseline, _ = CoupledSelector().select(bank, None, MeterDouble())
    meter = MeterDouble()
    plan, info = CoupledSelector().select(bank, ActorDouble(np.nan), meter, enabled=True)
    np.testing.assert_array_equal(plan, baseline)
    assert info["numerical_revocation"]
    assert not info["influence_used"]
    assert not info["changed_action"]
    json.dumps(info, allow_nan=False)


def test_guidance_scores_pre_action_rows_and_retains_evidence():
    _, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)
    actor, meter = ActorDouble(0.8), MeterDouble()
    _, info = CoupledSelector(alpha=20).select(bank, actor, meter, enabled=True)
    np.testing.assert_array_equal(actor.calls[0], bank.observations.reshape(-1, 17))
    expected = np.sum((bank.actions - 0.8) ** 2, axis=(1, 2))
    stored = meter.arrays[0][1]
    np.testing.assert_allclose(stored["agreement_squared"], expected)
    np.testing.assert_allclose(stored["guided_scores"], stored["baseline_scores"] + 10 * expected)
    assert info["influence_used"]
    assert info["actor_hash"] == actor.policy_hash
    assert "selected_pre_action_observations" in stored
    assert "guided_mean_post_states" in stored


def test_partial_support_only_queries_supported_pre_action_rows():
    _, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)
    actor = ActorDouble()

    def support(obs):
        mask = np.zeros(obs.shape[:2], dtype=bool)
        mask[:, 0] = True
        return mask

    _, info = CoupledSelector().select(bank, actor, MeterDouble(), enabled=True, supported=support)
    assert actor.calls[0].shape == (6, 17)
    np.testing.assert_array_equal(actor.calls[0], bank.observations[:, 0])
    assert info["supported_rows"] == 6


def test_snapshot_caches_restore_without_aliases():
    _, _, independent, state, warm = setup_search()
    independent.propose(state, warm)
    snapshot = independent.snapshot()
    expected = copy.deepcopy(snapshot)
    independent.propose(state, warm)
    independent.restore(snapshot)
    assert_tree_equal(independent.snapshot(), expected)
    snapshot["native"]["U"][:] = 123
    assert_tree_equal(independent.snapshot(), expected)


def test_private_state_is_small_complete_for_next_proposals_and_not_an_alias():
    _, _, independent, state, warm = setup_search()
    independent.propose(state, warm)
    private = independent.private_state()
    assert set(private) == {"U", "rng_state", "initial_plan"}
    assert_tree_equal(private, independent.snapshot()["native"])
    private["U"][:] = 123
    assert not np.any(independent.private_state()["U"] == 123)


def test_budget_or_evidence_runtime_errors_stop_instead_of_becoming_fallbacks():
    env, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)

    def exhausted(*args, **kwargs):
        raise RuntimeError("query budget exhausted")

    env.batch_rollout = exhausted
    with pytest.raises(RuntimeError, match="budget exhausted"):
        CoupledSelector().select(bank, None, MeterDouble())


def test_paired_selector_instances_get_distinct_durable_evidence_tags():
    _, _, independent, state, warm = setup_search()
    bank = independent.propose(state, warm)
    meter = MeterDouble()
    CoupledSelector().select(bank, None, meter)
    CoupledSelector().select(bank, None, meter)
    assert meter.arrays[0][0] != meter.arrays[1][0]


@pytest.mark.parametrize("kwargs", [{"alpha": -1}, {"eta_u": np.nan}])
def test_invalid_cost_weights_rejected(kwargs):
    with pytest.raises(ValueError):
        CoupledSelector(**kwargs)
