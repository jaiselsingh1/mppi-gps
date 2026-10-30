"""S3 must score the two actual plans without perturbing raw MPPI execution."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.envs.walker2d import Walker2d
from src.mppi.gate import InvalidBaselineError, TaskCostGate, gate_accepts
from src.mppi.mppi import MPPI
from src.rl.sac import SacActor
from src.utils.config import MPPIConfig


_TASK = json.loads((Path(__file__).resolve().parents[1] / "configs/walker2d_task.json").read_text())
_H = 6


class RecordingPolicy:
    """A deterministic state-dependent policy, independent of checkpoint files."""

    def __init__(self):
        self.observations = []
        self.deterministic_flags = []

    def act(self, observation, deterministic=True):
        self.observations.append(observation.copy())
        self.deterministic_flags.append(deterministic)
        return 0.1 * np.tanh(observation[:6])


class FixedPolicy:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def act(self, observation, deterministic=True):
        self.calls += 1
        return self.action


@pytest.fixture
def walker():
    env = Walker2d(**_TASK, nthread=1)
    env.reset(state=env.get_state())
    try:
        yield env
    finally:
        env.close()


@pytest.fixture
def gate():
    with TaskCostGate(_TASK, _H, RecordingPolicy()) as instance:
        yield instance


@pytest.mark.parametrize("difference", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("baseline_fell, policy_fell", [
    (False, False), (False, True), (True, False), (True, True),
])
def test_zero_slack_rule_uses_both_actual_cost_and_extra_fall_check(
    difference, baseline_fell, policy_fell
):
    expected = difference <= 0 and (baseline_fell or not policy_fell)
    assert gate_accepts(10.0, 10.0 + difference, baseline_fell, policy_fell) == expected


@pytest.mark.parametrize("invalid_cost", [np.inf, -np.inf, np.nan])
def test_nonfinite_candidate_score_is_rejected(invalid_cost):
    assert not gate_accepts(10.0, invalid_cost, False, False)


def test_missing_candidate_score_or_fall_flag_cannot_be_accepted():
    assert not gate_accepts(10.0, None, False, False)
    assert not gate_accepts(10.0, 9.0, False, None)


@pytest.mark.parametrize("invalid_cost", [np.inf, -np.inf, np.nan])
def test_nonfinite_baseline_score_stops_the_decision(invalid_cost):
    with pytest.raises(InvalidBaselineError):
        gate_accepts(invalid_cost, 10.0, False, False)


@pytest.mark.parametrize("action", [
    np.full(6, np.nan), np.full(6, np.inf), np.full(6, 1.01),
    np.full(6, -1.01), np.zeros(5), np.zeros((1, 6)),
])
def test_bad_policy_actions_are_rejected_without_clipping(walker, action):
    policy = FixedPolicy(action)
    with TaskCostGate(_TASK, _H, policy) as gate:
        decision = gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
    assert not decision.accepted
    assert decision.reason == "invalid_policy_action"
    assert np.isfinite(decision.baseline_cost)
    assert decision.policy_cost is None
    assert policy.calls == 1


@pytest.mark.parametrize("plan", [
    np.zeros((_H - 1, 6)), np.zeros((_H, 5)), np.full((_H, 6), np.nan),
    np.full((_H, 6), np.inf), np.full((_H, 6), 1.01), np.full((_H, 6), -1.01),
])
def test_invalid_baseline_plan_raises_before_calling_policy(walker, plan):
    policy = FixedPolicy(np.zeros(6))
    with TaskCostGate(_TASK, _H, policy) as gate:
        with pytest.raises(InvalidBaselineError):
            gate.decide(walker.get_state(), walker.get_warmstart(), plan)
    assert policy.calls == 0


@pytest.mark.parametrize("which, malformed", [
    ("state", "shape"), ("state", "nonfinite"),
    ("warmstart", "shape"), ("warmstart", "nonfinite"),
])
def test_invalid_start_cannot_be_used_as_a_fallback(walker, which, malformed):
    state, warmstart = walker.get_state(), walker.get_warmstart()
    value = state if which == "state" else warmstart
    if malformed == "shape":
        value = value[:-1]
    else:
        value[0] = np.nan
    if which == "state":
        state = value
    else:
        warmstart = value
    policy = FixedPolicy(np.zeros(6))
    with TaskCostGate(_TASK, _H, policy) as gate:
        with pytest.raises(InvalidBaselineError):
            gate.decide(state, warmstart, np.zeros((_H, 6)))
    assert policy.calls == 0


def test_already_fallen_initial_state_stops_before_policy_query(walker):
    state = walker.get_state()
    state[3] = 1.01
    policy = FixedPolicy(np.zeros(6))
    with TaskCostGate(_TASK, _H, policy) as gate:
        with pytest.raises(InvalidBaselineError):
            gate.decide(state, walker.get_warmstart(), np.zeros((_H, 6)))
    assert policy.calls == 0


@pytest.mark.parametrize("horizon", [True, False, 0, 1, 2.5])
def test_gate_rejects_invalid_horizons(horizon):
    with pytest.raises(ValueError, match="horizon"):
        TaskCostGate(_TASK, horizon, RecordingPolicy())


def test_closed_gate_cannot_run_another_decision(walker):
    gate = TaskCostGate(_TASK, _H, RecordingPolicy())
    gate.close()
    gate.close()  # Closing an already closed scratch simulator is harmless.
    with pytest.raises(RuntimeError, match="closed"):
        gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))


def test_policy_is_queried_at_each_pre_action_state(walker, gate):
    decision = gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
    assert decision.candidate_steps == _H
    expected = walker.rollout_states_to_obs(decision.candidate_states[:-1])
    np.testing.assert_array_equal(gate.policy.observations, expected)
    assert gate.policy.deterministic_flags == [True] * _H
    # This specifically distinguishes pre-action from the old next-state bug.
    assert not np.array_equal(expected, walker.rollout_states_to_obs(decision.candidate_states[1:]))
    np.testing.assert_array_equal(decision.candidate_plan, 0.1 * np.tanh(expected[:, :6]))


def test_candidate_prefix_and_task_cost_match_open_loop_replay(walker, gate):
    state, warmstart = walker.get_state(), walker.get_warmstart()
    decision = gate.decide(state, warmstart, np.zeros((_H, 6)))
    states, costs, _ = walker.batch_rollout(
        state, decision.candidate_plan[None], initial_warmstart=warmstart
    )
    np.testing.assert_array_equal(states[0, :decision.candidate_steps], decision.candidate_states[1:])
    assert decision.policy_cost == costs[0]


def test_identical_actual_plans_are_accepted_with_no_extra_slack(walker, gate):
    state, warmstart = walker.get_state(), walker.get_warmstart()
    first = gate.decide(state, warmstart, np.zeros((_H, 6)))
    decision = gate.decide(state, warmstart, first.candidate_plan)
    np.testing.assert_array_equal(decision.baseline_plan, decision.candidate_plan)
    assert decision.baseline_cost == decision.policy_cost
    assert decision.accepted
    assert decision.reason == "accepted"


def test_decision_does_not_modify_or_alias_input_arrays(walker, gate):
    state, warmstart = walker.get_state(), walker.get_warmstart()
    plan = np.zeros((_H, 6))
    copies = [value.copy() for value in (state, warmstart, plan)]
    decision = gate.decide(state, warmstart, plan)
    for original, before in zip((state, warmstart, plan), copies):
        np.testing.assert_array_equal(original, before)
    decision.baseline_plan[0, 0] = 0.9
    decision.candidate_states[0, 1] += 1.0
    np.testing.assert_array_equal(plan, copies[2])
    np.testing.assert_array_equal(state, copies[0])


def _about_to_fall(walker):
    state = walker.get_state()
    state[3] = 0.999
    state[12] = 2.0
    walker.reset(state=state)
    return walker.get_state(), walker.get_warmstart()


def test_policy_queries_stop_at_first_fall_and_cost_keeps_absorbing_tail(walker):
    state, warmstart = _about_to_fall(walker)
    policy = RecordingPolicy()
    with TaskCostGate(_TASK, _H, policy) as gate:
        decision = gate.decide(state, warmstart, np.zeros((_H, 6)))
    assert decision.policy_fell
    assert decision.candidate_steps == 1
    assert len(policy.observations) == 1
    np.testing.assert_array_equal(policy.observations[0], walker.rollout_states_to_obs(state))
    np.testing.assert_array_equal(decision.candidate_plan[1:], np.zeros((_H - 1, 6)))
    _, live_cost, terminated, _ = walker.step(decision.candidate_plan[0])
    assert terminated
    np.testing.assert_array_equal(decision.candidate_states[1], walker.get_state())
    assert decision.policy_cost == live_cost + (_H - 1) * _TASK["unhealthy_cost_weight"]


def test_scores_the_optimized_baseline_plan_in_joint_batch_not_best_sample(walker, monkeypatch):
    controller = MPPI(walker, MPPIConfig(K=8, H=_H, lam=3.0, initial_iterations=2), seed=37)
    state, warmstart = walker.get_state(), walker.get_warmstart()
    action, _ = controller.plan_step(state, initial_warmstart=warmstart)
    plan = controller.last_plan
    # This comparison cannot be fulfilled by merely reusing the minimum sample score.
    _, expected, _ = walker.batch_rollout(state, plan[None], initial_warmstart=warmstart)
    assert expected[0] != np.min(controller._last_costs)
    with TaskCostGate(_TASK, _H, RecordingPolicy()) as gate:
        original = gate._env.batch_rollout
        calls = []

        def record_batch(initial_state, action_sequences, initial_warmstart=None):
            calls.append((initial_state.copy(), action_sequences.copy(), initial_warmstart.copy()))
            return original(initial_state, action_sequences, initial_warmstart=initial_warmstart)

        monkeypatch.setattr(gate._env, "batch_rollout", record_batch)
        decision = gate.decide(state, warmstart, plan)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], state)
    np.testing.assert_array_equal(calls[0][1], np.stack([plan, decision.candidate_plan]))
    np.testing.assert_array_equal(calls[0][2], warmstart)
    np.testing.assert_array_equal(decision.baseline_plan[0], action)
    assert decision.baseline_cost == expected[0]


@pytest.mark.parametrize("row", [0, 1])
def test_nonfinite_executable_prefix_is_never_a_valid_plan(walker, gate, monkeypatch, row):
    original = gate._env.batch_rollout

    def corrupt_prefix(state, actions, initial_warmstart=None):
        states, _, sensors = original(state, actions, initial_warmstart=initial_warmstart)
        states[row, 0, -1] = np.nan  # A joint ignored by the scalar running cost.
        costs = gate._env.rollout_cost(states, actions, sensors)
        return states, costs, sensors

    monkeypatch.setattr(gate._env, "batch_rollout", corrupt_prefix)
    if row == 0:
        with pytest.raises(InvalidBaselineError):
            gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
    else:
        decision = gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
        assert not decision.accepted
        assert decision.reason == "nonfinite_candidate"


def test_nonfinite_unused_suffix_after_fall_does_not_invalidate_prefix(walker, gate, monkeypatch):
    state, warmstart = _about_to_fall(walker)
    original = gate._env.batch_rollout

    def corrupt_unused_suffix(state, actions, initial_warmstart=None):
        states, costs, sensors = original(state, actions, initial_warmstart=initial_warmstart)
        states[:, 1:, -1] = np.nan
        rescored = gate._env.rollout_cost(states, actions, sensors)
        np.testing.assert_array_equal(rescored, costs)
        return states, rescored, sensors

    monkeypatch.setattr(gate._env, "batch_rollout", corrupt_unused_suffix)
    decision = gate.decide(state, warmstart, np.zeros((_H, 6)))
    assert np.isfinite(decision.baseline_cost)
    assert np.isfinite(decision.policy_cost)
    assert decision.candidate_steps == 1
    assert decision.reason not in {"nonfinite_candidate", "candidate_replay_mismatch"}


def test_closed_loop_replay_mismatch_rejects_the_candidate(walker, gate, monkeypatch):
    original = gate._env.batch_rollout

    def mismatched_replay(state, actions, initial_warmstart=None):
        states, costs, sensors = original(state, actions, initial_warmstart=initial_warmstart)
        states[1, 0, -1] += 1e-10
        return states, costs, sensors

    monkeypatch.setattr(gate._env, "batch_rollout", mismatched_replay)
    decision = gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
    assert not decision.accepted
    assert decision.reason == "candidate_replay_mismatch"


def test_invalid_candidate_does_not_hide_a_numerically_invalid_baseline(walker, monkeypatch):
    with TaskCostGate(_TASK, _H, FixedPolicy(np.full(6, np.nan))) as gate:
        original = gate._env.batch_rollout

        def invalid_baseline(state, actions, initial_warmstart=None):
            states, costs, sensors = original(state, actions, initial_warmstart=initial_warmstart)
            costs[0] = np.inf
            return states, costs, sensors

        monkeypatch.setattr(gate._env, "batch_rollout", invalid_baseline)
        with pytest.raises(InvalidBaselineError):
            gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))


@pytest.mark.parametrize("error", [FloatingPointError, RuntimeError])
def test_known_numerical_policy_error_rejects_but_programming_error_is_not_swallowed(walker, error):
    class BrokenPolicy:
        def act(self, obs, deterministic=True):
            raise error("test error")

    with TaskCostGate(_TASK, _H, BrokenPolicy()) as gate:
        if error is RuntimeError:
            with pytest.raises(RuntimeError, match="test error"):
                gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
        else:
            decision = gate.decide(walker.get_state(), walker.get_warmstart(), np.zeros((_H, 6)))
            assert not decision.accepted
            assert decision.reason == "nonfinite_candidate"


def test_scratch_simulator_history_does_not_change_repeated_decision(walker, gate):
    state, warmstart = walker.get_state(), walker.get_warmstart()
    plan = np.zeros((_H, 6))
    before = gate.decide(state, warmstart, plan)
    for _ in range(8):
        walker.step(np.linspace(-0.1, 0.2, 6))
    gate.decide(walker.get_state(), walker.get_warmstart(), plan)
    after = gate.decide(state, warmstart, plan)
    np.testing.assert_array_equal(before.candidate_states, after.candidate_states)
    np.testing.assert_array_equal(before.candidate_plan, after.candidate_plan)
    assert before.baseline_cost == after.baseline_cost
    assert before.policy_cost == after.policy_cost
    assert before.accepted == after.accepted


def _assert_numpy_rng_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def _assert_snapshots_equal(left, right):
    np.testing.assert_array_equal(left["U"], right["U"])
    assert left["initial_plan"] == right["initial_plan"]
    assert left["rng_state"] == right["rng_state"]


def test_twelve_shadow_decisions_leave_raw_mppi_and_every_rng_byte_identical(walker):
    """The shadow pass cannot leak into later samples, actions or physics."""
    shadow_env = Walker2d(**_TASK, nthread=1)
    shadow_env.reset(state=walker.get_state())
    shadow_env.set_warmstart(walker.get_warmstart())
    cfg = MPPIConfig(K=8, H=_H, lam=3.0, noise_sigma=0.3,
                     noise_lowpass_cutoff_hz=23.4375,
                     noise_lowpass_sample_rate_hz=125.0, initial_iterations=2)
    raw, shadow = MPPI(walker, cfg, seed=29), MPPI(shadow_env, cfg, seed=29)
    with torch.random.fork_rng():
        torch.manual_seed(23)
        policy = SacActor(17, *walker.action_bounds, hidden_dim=16).eval().requires_grad_(False)
    actor_before = copy.deepcopy(policy.state_dict())
    try:
        with TaskCostGate(_TASK, _H, policy) as gate:
            for _ in range(12):
                _assert_snapshots_equal(raw.snapshot(), shadow.snapshot())
                raw_action, raw_info = raw.plan_step(walker.get_state(), initial_warmstart=walker.get_warmstart())
                shadow_action, shadow_info = shadow.plan_step(
                    shadow_env.get_state(), initial_warmstart=shadow_env.get_warmstart()
                )
                np.testing.assert_array_equal(raw_action, shadow_action)
                np.testing.assert_array_equal(raw.last_plan, shadow.last_plan)
                np.testing.assert_array_equal(raw._last_states, shadow._last_states)
                np.testing.assert_array_equal(raw._last_costs, shadow._last_costs)
                assert raw_info == shadow_info
                post_plan = shadow.snapshot()
                state, warmstart = shadow_env.get_state(), shadow_env.get_warmstart()
                numpy_rng, torch_rng = np.random.get_state(), torch.random.get_rng_state()
                decision = gate.decide(state, warmstart, shadow.last_plan)
                _assert_snapshots_equal(post_plan, shadow.snapshot())
                _assert_numpy_rng_equal(numpy_rng, np.random.get_state())
                assert torch.equal(torch_rng, torch.random.get_rng_state())
                np.testing.assert_array_equal(state, shadow_env.get_state())
                np.testing.assert_array_equal(warmstart, shadow_env.get_warmstart())
                np.testing.assert_array_equal(decision.baseline_plan[0], shadow_action)
                # Shadow always executes MPPI, independent of this recommendation.
                raw_obs, raw_cost, raw_done, _ = walker.step(raw_action)
                shadow_obs, shadow_cost, shadow_done, _ = shadow_env.step(shadow_action)
                np.testing.assert_array_equal(raw_obs, shadow_obs)
                np.testing.assert_array_equal(walker.get_state(), shadow_env.get_state())
                np.testing.assert_array_equal(walker.get_warmstart(), shadow_env.get_warmstart())
                assert raw_cost == shadow_cost
                assert raw_done == shadow_done == False
                _assert_snapshots_equal(raw.snapshot(), shadow.snapshot())
    finally:
        shadow_env.close()
    for key, value in actor_before.items():
        assert torch.equal(value, policy.state_dict()[key])
