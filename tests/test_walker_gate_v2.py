"""V2 changes only the matched continuation score, not MPPI or execution."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from src.envs.walker2d import Walker2d
from src.mppi.gate import InvalidBaselineError
from src.mppi.gate_v2 import ContinuationGate
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


_TASK = json.loads((Path(__file__).resolve().parents[1] / "configs/walker2d_task.json").read_text())
_CFG = MPPIConfig(K=8, H=6, lam=3.0, noise_sigma=0.3, initial_iterations=2)


@pytest.fixture
def setup():
    env = Walker2d(**_TASK, nthread=1)
    env.reset(state=env.get_state())
    planner = MPPI(env, _CFG, seed=37)
    with ContinuationGate(_TASK, _CFG) as gate:
        try:
            state, warmstart = env.get_state(), env.get_warmstart()
            action, _ = planner.plan_step(state, initial_warmstart=warmstart)
            yield env, planner, gate, state, warmstart, action, planner.snapshot()
        finally:
            env.close()


def _snapshot_equal(left, right):
    np.testing.assert_array_equal(left["U"], right["U"])
    assert left["rng_state"] == right["rng_state"]
    assert left["initial_plan"] == right["initial_plan"]


def test_identical_actions_have_exactly_identical_scores_and_accept(setup):
    _, _, gate, state, warmstart, action, snapshot = setup
    result = gate.decide(state, warmstart, snapshot, action, action)
    assert result.accepted and result.reason == "accepted"
    assert result.baseline_cost == result.policy_cost
    left, right = result.baseline_score, result.candidate_score
    for field in ("action", "successor_state", "successor_warmstart", "optimized_plan",
                  "scored_plan", "continuation_states"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    _snapshot_equal(left.planner_snapshot_after, right.planner_snapshot_after)


def test_baseline_successor_plan_matches_actual_next_mppi_call_exactly(setup):
    env, planner, gate, state, warmstart, action, snapshot = setup
    # The test repeats across real reached states, not just the reset state.
    for _ in range(8):
        result = gate.decide(state, warmstart, snapshot, action, np.zeros(6))
        score = result.baseline_score
        _, immediate, done, _ = env.step(action)
        assert not done
        np.testing.assert_array_equal(score.successor_state, env.get_state())
        np.testing.assert_array_equal(score.successor_warmstart, env.get_warmstart())
        assert score.immediate_cost == immediate
        state, warmstart = env.get_state(), env.get_warmstart()
        action, _ = planner.plan_step(state, initial_warmstart=warmstart)
        np.testing.assert_array_equal(score.optimized_plan, planner.last_plan)
        _snapshot_equal(score.planner_snapshot_after, planner.snapshot())
        assert score.planner_snapshot_after["initial_plan"] is False
        snapshot = planner.snapshot()


def test_horizon_is_one_plus_h_minus_one_and_optimized_plan_is_explicitly_scored(setup, monkeypatch):
    _, _, gate, state, warmstart, action, snapshot = setup
    original = gate._env.batch_rollout
    calls = []

    def record(initial, plans, initial_warmstart=None):
        output = original(initial, plans, initial_warmstart=initial_warmstart)
        calls.append((initial.copy(), plans.copy(), initial_warmstart.copy(), output[1].copy()))
        return output

    monkeypatch.setattr(gate._env, "batch_rollout", record)
    result = gate.decide(state, warmstart, snapshot, action, np.full(6, 0.1))
    # initial_iterations=2 must NOT repeat here: this is post-planning memory.
    assert [call[1].shape for call in calls] == [(8, 6, 6), (1, 5, 6)] * 2
    for index, score in enumerate((result.baseline_score, result.candidate_score)):
        sampled, explicit = calls[2 * index:2 * index + 2]
        np.testing.assert_array_equal(explicit[0], score.successor_state)
        np.testing.assert_array_equal(explicit[2], score.successor_warmstart)
        np.testing.assert_array_equal(explicit[1][0], score.optimized_plan[:-1])
        assert score.continuation_cost == explicit[3][0]
        assert score.cost == score.immediate_cost + explicit[3][0]
        assert score.continuation_cost != np.min(sampled[3])
        assert len(score.scored_plan) + 1 == _CFG.H
    assert result.baseline_score.planner_snapshot_after["rng_state"] == result.candidate_score.planner_snapshot_after["rng_state"]


def test_inputs_live_state_planner_and_global_rng_are_untouched(setup):
    env, planner, gate, state, warmstart, action, snapshot = setup
    before = copy.deepcopy((state, warmstart, action, snapshot))
    live_state, live_warmstart, live_snapshot = env.get_state(), env.get_warmstart(), planner.snapshot()
    numpy_before, torch_before = np.random.get_state(), torch.get_rng_state().clone()
    result = gate.decide(state, warmstart, snapshot, action, np.full(6, -0.1))
    for value, expected in zip((state, warmstart, action), before[:3]):
        np.testing.assert_array_equal(value, expected)
    _snapshot_equal(snapshot, before[3])
    np.testing.assert_array_equal(env.get_state(), live_state)
    np.testing.assert_array_equal(env.get_warmstart(), live_warmstart)
    _snapshot_equal(planner.snapshot(), live_snapshot)
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    result.baseline_score.action[0] = 0.9
    result.baseline_score.planner_snapshot_after["U"][0, 0] = 0.8
    np.testing.assert_array_equal(action, before[2])
    _snapshot_equal(snapshot, before[3])


def test_decisions_are_repeatable_after_other_candidate_scoring(setup):
    _, _, gate, state, warmstart, action, snapshot = setup
    first = gate.decide(state, warmstart, snapshot, action, np.zeros(6))
    gate.decide(state, warmstart, snapshot, action, np.ones(6))
    repeated = gate.decide(state, warmstart, snapshot, action, np.zeros(6))
    assert first.baseline_cost == repeated.baseline_cost
    assert first.policy_cost == repeated.policy_cost
    np.testing.assert_array_equal(first.candidate_score.optimized_plan, repeated.candidate_score.optimized_plan)


@pytest.mark.parametrize("action", [np.zeros(5), np.zeros((1, 6)), np.full(6, np.nan),
                                    np.full(6, np.inf), np.full(6, 1.01), np.full(6, -1.01), "bad"])
def test_invalid_candidate_rejects_but_baseline_remains_scored(setup, action):
    _, _, gate, state, warmstart, baseline, snapshot = setup
    result = gate.decide(state, warmstart, snapshot, baseline, action)
    assert not result.accepted and result.reason == "invalid_policy_action"
    assert result.candidate_score is None and result.policy_cost is None and result.policy_fell is None
    assert np.isfinite(result.baseline_cost)


@pytest.mark.parametrize("which, kind", [("state", "shape"), ("state", "nan"),
                                        ("warmstart", "shape"), ("warmstart", "nan"),
                                        ("action", "shape"), ("action", "nan"),
                                        ("action", "bounds")])
def test_invalid_baseline_inputs_stop_before_simulation(setup, monkeypatch, which, kind):
    _, _, gate, state, warmstart, action, snapshot = setup
    inputs = {"state": state, "warmstart": warmstart, "action": action}
    value = inputs[which].copy()
    if kind == "shape":
        value = value[:-1]
    else:
        value[0] = np.nan if kind == "nan" else 1.01
    inputs[which] = value
    monkeypatch.setattr(gate, "_score_action", lambda *args: pytest.fail("Must validate before simulation"))
    with pytest.raises(InvalidBaselineError):
        gate.decide(inputs["state"], inputs["warmstart"], snapshot, inputs["action"], action)


@pytest.mark.parametrize("mutation", ["pre_call", "numeric_initial", "missing_rng", "bad_rng", "bad_U", "unbounded_U", "not_mapping"])
def test_requires_valid_post_planning_snapshot_even_if_action_would_fall(setup, mutation):
    _, _, gate, state, warmstart, action, snapshot = setup
    state[3], state[12] = 0.999, 2.0
    snapshot = copy.deepcopy(snapshot)
    if mutation == "pre_call":
        snapshot["initial_plan"] = True
    elif mutation == "numeric_initial":
        snapshot["initial_plan"] = 0
    elif mutation == "missing_rng":
        del snapshot["rng_state"]
    elif mutation == "bad_rng":
        snapshot["rng_state"] = {"bit_generator": "bad"}
    elif mutation == "bad_U":
        snapshot["U"][0, 0] = np.nan
    elif mutation == "unbounded_U":
        snapshot["U"][0, 0] = 1.01
    else:
        snapshot = None
    with pytest.raises(InvalidBaselineError):
        gate.decide(state, warmstart, snapshot, action, action)


def test_already_fallen_state_is_invalid_baseline(setup):
    _, _, gate, state, warmstart, action, snapshot = setup
    state[3] = 1.01
    with pytest.raises(InvalidBaselineError, match="after a fall"):
        gate.decide(state, warmstart, snapshot, action, action)


def test_immediate_fall_charges_one_fall_plus_absorbing_tail_and_never_plans(setup, monkeypatch):
    env, _, gate, state, _, action, snapshot = setup
    state[3], state[12] = 0.999, 2.0
    env.reset(state=state)
    warmstart = env.get_warmstart()
    monkeypatch.setattr(gate._planner, "plan_step", lambda *args, **kwargs: pytest.fail("No planning after fall"))
    result = gate.decide(state, warmstart, snapshot, action, action)
    _, immediate, done, info = env.step(action)
    assert done and info["fall_cost"] == 20.0
    expected = immediate + (_CFG.H - 1) * 5.0
    assert result.baseline_cost == result.policy_cost == expected
    assert result.baseline_fell and result.policy_fell and result.accepted
    for score in (result.baseline_score, result.candidate_score):
        assert score.immediate_fell
        assert score.optimized_plan is None and score.planner_snapshot_after is None
        assert len(score.scored_plan) == len(score.continuation_states) == 0


@pytest.mark.parametrize("fall_index", [0, 4])
def test_continuation_first_or_last_fall_uses_frozen_absorbing_cost(setup, monkeypatch, fall_index):
    _, _, gate, state, warmstart, action, snapshot = setup
    original = gate._env.batch_rollout

    def synthetic_explicit(initial, plans, initial_warmstart=None):
        if plans.shape[:2] != (1, _CFG.H - 1):
            return original(initial, plans, initial_warmstart=initial_warmstart)
        states = np.broadcast_to(initial, (1, _CFG.H - 1, len(initial))).copy()
        states[0, fall_index, 3] = 1.01
        # Non-finite unused suffix must not invalidate a valid falling prefix.
        states[0, fall_index + 1:] = np.nan
        sensors = np.zeros((1, _CFG.H - 1, gate._env.model.nsensordata))
        costs = gate._env.rollout_cost(states, plans, sensors)
        return states, costs, sensors

    monkeypatch.setattr(gate._env, "batch_rollout", synthetic_explicit)
    result = gate.decide(state, warmstart, snapshot, action, action)
    score = result.baseline_score
    assert score.fell and not score.immediate_fell
    states, actions = score.continuation_states, score.scored_plan
    running = gate._env.running_cost(states[None], actions[None], None)[0]
    expected = float(running[:fall_index + 1].sum() + 20.0 + (4 - fall_index) * 5.0)
    assert score.continuation_cost == expected
    assert score.cost == score.immediate_cost + expected
    assert result.accepted


@pytest.mark.parametrize("location", ["step", "plan", "explicit_score"])
def test_nonfinite_candidate_rejects_without_masking_baseline_failure(setup, monkeypatch, location):
    _, _, gate, state, warmstart, action, snapshot = setup
    original = gate._score_action
    count = 0

    def score(*args):
        nonlocal count
        count += 1
        if count == 2:
            method = {"step": "step", "plan": "plan_step", "explicit_score": "batch_rollout"}[location]
            owner = gate._planner if location == "plan" else gate._env
            original_method = getattr(owner, method)

            def fail(*method_args, **kwargs):
                if location != "explicit_score" or method_args[1].shape[0] == 1:
                    raise FloatingPointError("numerical failure")
                return original_method(*method_args, **kwargs)

            monkeypatch.setattr(owner, method, fail)
        return original(*args)

    monkeypatch.setattr(gate, "_score_action", score)
    result = gate.decide(state, warmstart, snapshot, action, np.ones(6))
    assert not result.accepted and result.reason == "nonfinite_candidate"
    assert result.candidate_score is None


def test_nonfinite_baseline_is_a_hard_failure(setup, monkeypatch):
    _, _, gate, state, warmstart, action, snapshot = setup

    def fail(*args):
        raise FloatingPointError("no finite trajectory")

    monkeypatch.setattr(gate, "_score_action", fail)
    with pytest.raises(InvalidBaselineError, match="cannot be scored"):
        gate.decide(state, warmstart, snapshot, action, action)


def test_programming_error_is_not_silently_rejected(setup, monkeypatch):
    _, _, gate, state, warmstart, action, snapshot = setup

    def fail(*args, **kwargs):
        raise RuntimeError("programming bug")

    monkeypatch.setattr(gate._planner, "plan_step", fail)
    with pytest.raises(RuntimeError, match="programming bug"):
        gate.decide(state, warmstart, snapshot, action, action)


@pytest.mark.parametrize("baseline_fell,candidate_fell,difference", [
    (False, False, -1), (False, False, 0), (False, False, 1),
    (False, True, -1), (True, False, -1), (True, True, -1), (True, False, 1),
])
def test_original_zero_slack_and_no_added_fall_rule_is_unchanged(
    setup, monkeypatch, baseline_fell, candidate_fell, difference
):
    _, _, gate, state, warmstart, action, snapshot = setup
    template = gate.decide(state, warmstart, snapshot, action, action).baseline_score
    scores = iter([replace(template, cost=10.0, fell=baseline_fell),
                   replace(template, cost=10.0 + difference, fell=candidate_fell)])
    monkeypatch.setattr(gate, "_score_action", lambda *args: next(scores))
    result = gate.decide(state, warmstart, snapshot, action, action)
    assert result.accepted == (difference <= 0 and (baseline_fell or not candidate_fell))


@pytest.mark.parametrize("horizon", [True, 0, 1, 2.5])
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError, match="horizon"):
        ContinuationGate(_TASK, replace(_CFG, H=horizon))


def test_closed_scorer_cannot_score_again(setup):
    _, _, gate, state, warmstart, action, snapshot = setup
    gate.close()
    gate.close()
    with pytest.raises(RuntimeError, match="closed"):
        gate.decide(state, warmstart, snapshot, action, action)
