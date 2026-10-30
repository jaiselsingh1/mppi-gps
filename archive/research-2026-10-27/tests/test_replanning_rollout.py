"""Actual frozen-controller MC ownership, noise and accounting tests."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.expert_continuation import finite_absorbing_cost
from src.gps.replanning_rollout import measured_suffix_returns, paired_first_action, rollout_episode


def cfg():
    return SimpleNamespace(total_steps=24, H=8, K=4, gamma=.999, memory_candidates=0,
        query_stride=8, angle_noise=.03, velocity_noise=.10)


@pytest.mark.parametrize("gamma", [0., .7, .999, 1.])
def test_dense_mc_suffixes_include_exact_finite_absorption_without_double_fall(gamma):
    costs = np.array([2., 3., 25.])
    values = measured_suffix_returns(costs, gamma=gamma, remaining=8, terminated=True, unhealthy_cost=5.)
    for i in range(len(costs)):
        expected = float(np.dot(gamma**np.arange(len(costs)-i), costs[i:]))
        expected += gamma**(len(costs)-i)*finite_absorbing_cost(gamma, 5, 5.)
        assert values[i] == pytest.approx(expected)
    assert values[-1] == pytest.approx(25+gamma*finite_absorbing_cost(gamma, 5, 5.))


def test_healthy_cutoff_has_zero_tail_and_incomplete_healthy_trace_is_not_a_label():
    np.testing.assert_allclose(measured_suffix_returns([1., 2.], gamma=.9, remaining=2, terminated=False), [2.8, 2.])
    with pytest.raises(ValueError, match="incomplete"):
        measured_suffix_returns([1., 2.], gamma=.9, remaining=3, terminated=False)


@pytest.fixture(scope="module")
def trace():
    return rollout_episode(cfg(), dict(archive=None, value=None, alpha=0., version="C0"),
        reset_id=82_020_002, planner_seed=75, collect_points=True, dense_stride=8)


def test_episode_dense_targets_are_actual_own_complete_suffixes_and_grouped(trace):
    assert trace["records"]
    for row in trace["records"]:
        i = row["step"]
        assert row["cost"] == trace["mc_returns"][i]
        assert row["remaining"] == trace["remaining_start"]-i
        assert row["controller_version"] == "C0"
        assert row["group"] == 82_020_002
        assert row["source"] == "train"
        np.testing.assert_array_equal(row["context"]["state"], trace["states"][i])
        np.testing.assert_array_equal(row["context"]["warmstart"], trace["warmstarts"][i])
        np.testing.assert_array_equal(row["context"]["nominal"], trace["raw_snapshots"][i]["U"])
        assert row["context"]["initial_plan"] == trace["raw_snapshots"][i]["initial_plan"]
    assert trace["ledger"]["executed_controls"] == trace["steps"]
    assert trace["ledger"]["planner_calls"] == trace["steps"]
    assert trace["ledger"]["candidate_controls"] == (trace["steps"]+1)*4*8
    assert trace["ledger"]["verification_controls"] == trace["steps"]*8
    assert trace["ledger"]["one_step_verification_controls"] > 0


def test_forced_first_action_replanning_keeps_raw_post_context_and_fresh_rng(trace):
    point = trace["points"][0]
    bank = point["bank"]
    b = bank["baseline_index"]
    result = paired_first_action(cfg(), dict(archive=None, value=None, alpha=0., version="C0"),
        point, candidate_index=(b+1)%len(bank["plans"]), seeds=[199, 200])
    for side in ("candidate", "baseline"):
        for episode in result[f"{side}_traces"]:
            np.testing.assert_array_equal(episode["initial_raw_snapshot"]["U"], point["raw_post"]["U"])
            assert episode["initial_raw_snapshot"]["initial_plan"] is False
            assert episode["initial_raw_snapshot"]["rng_state"] != point["raw_post"]["rng_state"]
            assert episode["ledger"]["planner_calls"] == episode["steps"]-1
            for row in episode["records"]:
                assert row["step"] >= 1
                assert row["remaining"] <= point["remaining"]-1
    for a, b in zip(result["candidate_traces"], result["baseline_traces"]):
        assert a["initial_raw_snapshot"]["rng_state"] == b["initial_raw_snapshot"]["rng_state"]
    assert result["ledger"]["executed_controls"] == sum(e["steps"] for side in
        ("candidate", "baseline") for e in result[f"{side}_traces"])
    for row in result["records"]:
        if row["step"] == 1:
            assert row["pair_id"] == result["pair_id"]
            assert row["side"] in ("candidate", "baseline")


def test_same_first_action_different_tail_has_bitwise_equal_closed_loop_credit(trace):
    point = copy.deepcopy(trace["points"][0])
    bank = point["bank"]
    b = bank["baseline_index"]
    j = (b+1)%len(bank["plans"])
    bank["plans"][j, 0] = bank["plans"][b, 0]
    bank["plans"][j, 1:] = -bank["plans"][b, 1:]
    bank["actions"][j] = bank["actions"][b]
    result = paired_first_action(cfg(), dict(archive=None, value=None, alpha=0., version="C0"),
        point, candidate_index=j, seeds=[321, 322])
    assert result["equal_first_actions"]
    assert result["differences"] == [0., 0.]
    for left, right in zip(result["candidate_traces"], result["baseline_traces"]):
        for key in ("states", "warmstarts", "actions", "costs"):
            np.testing.assert_array_equal(left[key], right[key])
    assert result["ledger"]["executed_controls"] == sum(e["steps"] for e in result["baseline_traces"])
    assert all(row.get("side") != "candidate" for row in result["records"])


def test_candidate_origin_can_differ_from_actual_target_controller_without_relabeling(trace):
    point = copy.deepcopy(trace["points"][0])
    point["controller_version"] = "candidate-C1"
    result = paired_first_action(cfg(), dict(archive=None, value=None, alpha=0., version="C0"),
        point, candidate_index=point["bank"]["baseline_index"], seeds=[400])
    assert result["originating_controller_version"] == "candidate-C1"
    assert result["controller_version"] == "C0"
    assert point["controller_version"] == "candidate-C1"
    assert all(row["controller_version"] == "C0" for row in result["records"])


@pytest.mark.parametrize("seeds", [[], [1, 1]])
def test_pair_queries_require_disjoint_future_replicates(trace, seeds):
    with pytest.raises(ValueError, match="distinct"):
        paired_first_action(cfg(), dict(archive=None, value=None, alpha=0., version="C0"),
            trace["points"][0], candidate_index=0, seeds=seeds)
