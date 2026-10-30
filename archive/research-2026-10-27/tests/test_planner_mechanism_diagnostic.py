"""Independent integrity checks for the frozen-planner mechanism diagnostic."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from src.gps import planner_mechanism_diagnostic as diagnostic
from src.gps.expert_continuation import Snapshot, _restore, finite_absorbing_cost
from src.gps.paired_planner import PairedEndpointPlanner
from src.rl.walker import WalkerRL
from src.utils.config import MPPIConfig
from scripts.diagnostics.run_planner_mechanism import seeds_for, select_oracle


def _bank():
    plans = np.asarray([
        np.zeros((3, 1)), np.zeros((3, 1)), np.ones((3, 1)),
        -np.ones((3, 1)), np.full((3, 1), 2.),
    ])
    return dict(plans=plans, local_costs=np.asarray([0., .1, 1., 2., 3.]),
        predicted_values=np.asarray([100., 100., 99., 50., 1.]),
        scores=np.asarray([0., .1, .4, 1.4, 2.4]),
        corrections=np.asarray([0., 0., -.6, -.6, -.6]),
        baseline_index=0, selected_index=0, healthy=np.ones(5, dtype=bool),
        supported=np.ones(5, dtype=bool), support=np.ones(5, dtype=bool),
        actions=plans[:, 0], remaining=20, H=3,
        sources=["sample", "sample", "sample", "memory", "raw_mean"])


def test_candidate_roles_are_frozen_distinct_plans_and_do_not_relabel_model_choice():
    bank = _bank()
    before = copy.deepcopy(bank)
    selected = diagnostic.choose_candidate_indices(bank, seed=31, gamma_h=.9)
    indices = [row["index"] for row in selected["roles"]]
    assert len(indices) == 3
    assert indices[0] == 0
    assert 4 in indices  # The unbounded frozen-model alternative.
    assert len({bank["plans"][i].tobytes() for i in indices}) == len(indices)
    assert selected["actual_bounded_selected_index"] == 0
    assert selected["unbounded_best_index"] == 4
    # Counterfactual bank coverage must not be reported as actual deployment.
    assert selected["actual_bounded_selected_index"] != selected["unbounded_best_index"]
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(bank[key], value)
        else:
            assert bank[key] == value
    repeated = diagnostic.choose_candidate_indices(bank, seed=31, gamma_h=.9)
    assert selected == repeated


def test_candidate_sampling_never_fabricates_distinct_plans():
    bank = _bank()
    bank["plans"][:] = bank["plans"][0]
    selected = diagnostic.choose_candidate_indices(bank, seed=9)
    assert selected["requested"] == 3
    assert len(selected["roles"]) == 1
    assert selected["selected_count"] == 1
    assert selected["insufficient_unique_plans"]


def test_plan_ranking_uses_prefix_plus_discounted_tail_rather_than_tail_alone():
    bank = _bank()
    bank["local_costs"][4] = 1000.
    selected = diagnostic.choose_candidate_indices(bank, seed=9, gamma_h=.9)
    assert selected["unbounded_best_index"] == 3


def test_actual_bounded_nonbaseline_choice_has_priority_over_exploratory_alternative():
    bank = _bank()
    bank["selected_index"] = 2
    bank["scores"][2] = -.2
    selected = diagnostic.choose_candidate_indices(bank, seed=9, gamma_h=.9)
    model_role = next(row for row in selected["roles"] if row["role"] == "model_alternative")
    assert model_role["index"] == 2
    assert selected["actual_bounded_selected_index"] == 2
    assert selected["unbounded_best_index"] == 4


def test_changing_diagnostic_sampling_rng_does_not_change_actual_controller_choice():
    bank = _bank()
    for seed in range(8):
        chosen = diagnostic.choose_candidate_indices(bank, seed=seed, gamma_h=.9)
        assert chosen["actual_bounded_selected_index"] == 0
        assert chosen["unbounded_best_index"] == 4


def test_raw_clone_preserves_nominal_and_initial_flag_and_reseeds_only_future_noise():
    raw = dict(U=np.asarray([[.1], [.2], [.3]]),
        rng_state=copy.deepcopy(np.random.default_rng(8).bit_generator.state),
        initial_plan=False)
    original = copy.deepcopy(raw)
    branch = diagnostic.clone_raw_snapshot(raw, future_seed=91)
    np.testing.assert_array_equal(branch["U"], original["U"])
    assert branch["initial_plan"] is False
    assert branch["rng_state"] == np.random.default_rng(91).bit_generator.state
    assert raw["rng_state"] == original["rng_state"]
    branch["U"][0] = .9
    np.testing.assert_array_equal(raw["U"], original["U"])


def test_selected_nominal_handoff_is_explicit_and_not_default_deployment():
    raw = dict(U=np.asarray([[.1], [.2], [.3]]),
        rng_state=copy.deepcopy(np.random.default_rng(8).bit_generator.state),
        initial_plan=False)
    plan = np.asarray([[-.1], [-.2], [-.3]])
    actual = diagnostic.clone_raw_snapshot(raw)
    repaired = diagnostic.clone_raw_snapshot(raw, selected_plan=plan)
    np.testing.assert_array_equal(actual["U"], raw["U"])
    np.testing.assert_array_equal(repaired["U"], np.asarray([[-.2], [-.3], [-.3]]))
    np.testing.assert_array_equal(plan, np.asarray([[-.1], [-.2], [-.3]]))
    assert actual["rng_state"] == repaired["rng_state"]


@pytest.fixture(scope="module")
def short_physical_capture():
    cfg = diagnostic.normalize_config(dict(total_steps=32, H=8, K=4,
        memory_candidates=1, workers=1))
    capture = diagnostic.capture(91_020_000, cfg, None, None, None, seed=183,
                                  capture_step=3)
    assert capture["status"] == "captured"
    return cfg, capture


def _assert_trace_equal(left, right):
    for key in ("states", "warmstarts", "actions", "costs", "velocity_errors"):
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    for key in ("cost", "steps", "terminated", "survived", "ledger"):
        assert left[key] == right[key], key


def test_physical_first_action_clone_is_exact_and_charges_only_executed_futures(short_physical_capture):
    cfg, captured = short_physical_capture
    index = captured["bank"]["baseline_index"]
    original_raw = copy.deepcopy(captured["raw_post"])
    original_state = captured["snapshot"]["state"].copy()
    first = diagnostic.first_action_branch(captured, index, cfg, None, seeds=[781, 782])
    repeated = diagnostic.first_action_branch(captured, index, cfg, None, seeds=[781, 782])
    assert first["full_costs"] == repeated["full_costs"]
    for left, right in zip(first["traces"], repeated["traces"]):
        _assert_trace_equal(left, right)
        assert left["ledger"]["executed_controls"] == left["steps"]
        assert left["ledger"]["planner_calls"] == left["steps"] - 1
        assert left["ledger"]["candidate_controls"] == cfg.K * cfg.H * (left["steps"] - 1)
        assert left["ledger"]["verification_controls"] == cfg.H * (left["steps"] - 1)
        assert left["ledger"]["memory_controls"] == 0
        assert left["planning_info"][0]["forced_bank_first_action"]
        np.testing.assert_array_equal(left["initial_raw_snapshot"]["U"], original_raw["U"])
        assert not left["initial_raw_snapshot"]["initial_plan"]
    assert first["ledger"]["executed_controls"] == sum(t["steps"] for t in first["traces"])
    np.testing.assert_array_equal(captured["raw_post"]["U"], original_raw["U"])
    assert captured["raw_post"]["rng_state"] == original_raw["rng_state"]
    np.testing.assert_array_equal(captured["snapshot"]["state"], original_state)


def test_physical_first_action_matches_independently_executed_original_nominal_rule(short_physical_capture):
    cfg, captured = short_physical_capture
    index, seed = captured["bank"]["baseline_index"], 783
    actual = diagnostic.first_action_branch(captured, index, cfg, None, seeds=[seed])["traces"][0]
    n = captured["remaining"]
    env = WalkerRL(gamma=cfg.gamma, max_episode_steps=n)
    try:
        _restore(env, Snapshot(**captured["snapshot"]))
        mppi_cfg = MPPIConfig.load("walker2d")
        mppi_cfg.K, mppi_cfg.H, mppi_cfg.initial_iterations = cfg.K, cfg.H, 2
        mppi_cfg.lam_target_neff_fraction = None
        planner = PairedEndpointPlanner(env.walker, mppi_cfg, archive=None, seed=seed,
            remaining=n-1, total_steps=cfg.total_steps, gamma=cfg.gamma,
            memory_candidates=cfg.memory_candidates)
        raw = copy.deepcopy(captured["raw_post"])
        raw["rng_state"] = np.random.default_rng(seed).bit_generator.state
        planner.raw.restore(raw)
        states, warms, actions, costs = [env.walker.get_state()], [env.walker.get_warmstart()], [], []
        terminated = False
        for step in range(n):
            action = (captured["bank"]["plans"][index, 0].copy() if step == 0 else
                      planner.plan_step(env.walker.get_state(), initial_warmstart=env.walker.get_warmstart())[0])
            _, _, terminated, truncated, info = env.step(action)
            actions.append(action.copy()); costs.append(info["task_cost"])
            states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart())
            if terminated or truncated:
                break
        for key, expected in (("states", states), ("warmstarts", warms), ("actions", actions), ("costs", costs)):
            np.testing.assert_array_equal(actual[key], np.asarray(expected), err_msg=key)
        expected_cost = float(np.dot(cfg.gamma ** np.arange(len(costs)), costs))
        if terminated:
            expected_cost += cfg.gamma ** len(costs) * finite_absorbing_cost(cfg.gamma, n-len(costs))
        assert actual["cost"] == expected_cost
    finally:
        env.close()


def test_equal_first_actions_with_different_plan_tails_have_identical_production_credit(short_physical_capture):
    cfg, original = short_physical_capture
    captured = copy.deepcopy(original)
    baseline = captured["bank"]["baseline_index"]
    base_plan = captured["bank"]["plans"][baseline].copy()
    alternative = base_plan.copy()
    # Deliberately different bounded tails; the executed first action is exact.
    alternative[1:] = np.where(base_plan[1:] >= 0., -.25, .25)
    alternative_index = len(captured["bank"]["plans"])
    captured["bank"]["plans"] = np.concatenate([
        captured["bank"]["plans"], alternative[None]], axis=0)
    np.testing.assert_array_equal(base_plan[0], alternative[0])
    assert not np.array_equal(base_plan[1:], alternative[1:])
    seeds = [1_111, 1_112]
    baseline_result = diagnostic.first_action_branch(captured, baseline, cfg, None, seeds)
    alternative_result = diagnostic.first_action_branch(captured, alternative_index, cfg, None, seeds)
    assert baseline_result["full_costs"] == alternative_result["full_costs"]
    for base_trace, alternative_trace in zip(baseline_result["traces"], alternative_result["traces"]):
        _assert_trace_equal(base_trace, alternative_trace)
        np.testing.assert_array_equal(base_trace["initial_raw_snapshot"]["U"],
                                      captured["raw_post"]["U"])
        np.testing.assert_array_equal(alternative_trace["initial_raw_snapshot"]["U"],
                                      captured["raw_post"]["U"])
    # Selected-plan retention is a different, explicit internal-state rule.
    handoff_baseline = diagnostic.first_action_branch(captured, baseline, cfg, None,
        seeds=[1_111], handoff_selected_nominal=True)
    handoff_alternative = diagnostic.first_action_branch(captured, alternative_index, cfg, None,
        seeds=[1_111], handoff_selected_nominal=True)
    base_nominal = handoff_baseline["traces"][0]["initial_raw_snapshot"]["U"]
    alternative_nominal = handoff_alternative["traces"][0]["initial_raw_snapshot"]["U"]
    np.testing.assert_array_equal(base_nominal, np.concatenate([base_plan[1:], base_plan[-1:]]))
    np.testing.assert_array_equal(alternative_nominal, np.concatenate([alternative[1:], alternative[-1:]]))
    assert not np.array_equal(base_nominal, alternative_nominal)


def test_endpoint_replay_matches_batch_and_counts_prefix_once(short_physical_capture):
    cfg, captured = short_physical_capture
    index = captured["bank"]["baseline_index"]
    reference = dict(archive=None, value=None, alpha=0.)
    result = diagnostic.endpoint_branch(captured, index, cfg, reference, seeds=[901, 902])
    assert not result["fell_in_prefix"]
    assert result["endpoint_max_abs_discrepancy"] < 1e-8
    assert result["prefix_trace"]["steps"] == cfg.H
    assert result["remaining_tail"] == captured["remaining"] - cfg.H
    executed = cfg.H + sum(t["steps"] for t in result["suffix_traces"])
    assert result["ledger"]["executed_controls"] == executed
    for cost, suffix in zip(result["full_costs"], result["suffix_traces"]):
        assert cost == result["prefix_cost"] + cfg.gamma ** cfg.H * suffix["cost"]


@pytest.mark.parametrize("seeds", [[], [901, 901]])
def test_branch_seed_lists_require_distinct_replicates(short_physical_capture, seeds):
    cfg, captured = short_physical_capture
    index = captured["bank"]["baseline_index"]
    with pytest.raises(ValueError, match="distinct"):
        diagnostic.endpoint_branch(captured, index, cfg, dict(archive=None, value=None, alpha=0.), seeds)
    with pytest.raises(ValueError, match="distinct"):
        diagnostic.first_action_branch(captured, index, cfg, None, seeds)


def test_discovery_endpoint_confirmation_and_replanning_streams_are_disjoint_by_root():
    used = set()
    roots = [42_000_000+i for i in range(4)] + [42_000_100+i for i in range(4)]
    for root in roots:
        discovery, endpoint_confirmation, replanning = seeds_for(root)
        assert list(map(len, (discovery, endpoint_confirmation, replanning))) == [2, 4, 4]
        values = discovery + endpoint_confirmation + replanning
        assert len(set(values)) == 10
        assert not used.intersection(values)
        used.update(values)
    assert len(used) == len(roots) * 10


@pytest.mark.parametrize("discovery,confirmation", [(0, 4), (2, 0), (9, 4), (2, 9)])
def test_seed_budget_cannot_silently_overlap_roles(discovery, confirmation):
    with pytest.raises(ValueError, match="Bounded"):
        seeds_for(42_000_000, discovery, confirmation)


def test_oracle_selection_ignores_confirmation_outcomes_even_if_they_reverse_ranking():
    indices = [0, 4, 2]
    discovery = {0: [8., 8.], 4: [3., 5.], 2: [6., 6.]}
    positive_confirmation = {
        index: measured + [(-1e12 if index == 0 else 1e12)] * 4
        for index, measured in discovery.items()
    }
    reversed_confirmation = {
        index: measured + [(1e12 if index == 0 else -1e12)] * 4
        for index, measured in discovery.items()
    }
    assert select_oracle(indices, positive_confirmation, 2) == 4
    assert select_oracle(indices, reversed_confirmation, 2) == 4
    assert select_oracle(indices, discovery, 2) == 4


def test_confirmation_nan_does_not_enter_discovery_selector():
    costs = {0: [8., 8., np.nan, np.nan], 2: [4., 4., np.nan, np.nan]}
    assert select_oracle([0, 2], costs, 2) == 2


def test_oracle_ties_use_candidate_index_not_return_dictionary_order():
    costs = {9: [4., 4.], 2: [4., 4.], 0: [8., 8.]}
    assert select_oracle([9, 2, 0], costs, 2) == 2
    assert select_oracle([0, 2, 9], costs, 2) == 2


@pytest.mark.parametrize("indices,costs,discovery", [([], {}, 2), ([0], {0: [1.]}, 0), ([0], {0: [np.inf]}, 1)])
def test_oracle_selector_rejects_empty_or_nonfinite_discovery(indices, costs, discovery):
    with pytest.raises(ValueError):
        select_oracle(indices, costs, discovery)
