"""Actual-return ownership, precise takeover restoration and independent planning."""
import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.gps.expert_continuation import (
    Snapshot, discounted_raw_cost, finite_absorbing_cost, independent_teacher_suffix,
    raw_teacher_config, rollout_policy_prefix, takeover_objective,
)
from src.gps.policy import TanhMLPPolicy
from src.rl.walker import WalkerRL


class Actor:
    def __init__(self, action=2.):
        self.action = action
        self.calls = 0

    def act(self, obs, deterministic=True):
        assert deterministic
        self.calls += 1
        return np.array([self.action])


class ToyWalker:
    action_dim, _w_unhealthy = 1, 5.

    def get_state(self):
        return self.state.copy()

    def get_warmstart(self):
        return self.warmstart.copy()

    @staticmethod
    def _is_healthy(z, angle):
        return .8 < z < 2. and -1. < angle < 1.


class ToyEnv:
    fall_at = None
    resets, closed = [], 0

    def __init__(self, *, task_config, gamma, max_episode_steps):
        self.walker = ToyWalker()
        self.limit = max_episode_steps
        self.local_t = 0

    def obs(self):
        result = np.zeros(17)
        result[0] = self.walker.state[2]
        result[1] = self.walker.state[0]
        return result

    def reset(self, *, options):
        self.walker.state = options["state"].copy()
        self.walker.warmstart = options["warmstart"].copy()
        type(self).resets.append((self.walker.state.copy(), self.walker.warmstart.copy()))
        return self.obs(), {}

    def step(self, action):
        self.local_t += 1
        self.walker.state[0] += 1
        self.walker.warmstart += 1 + float(action[0])
        fell = self.fall_at is not None and self.walker.state[0] >= self.fall_at
        if fell:
            self.walker.state[2] = .5
        raw = 1. + float(action[0])**2 + (25. if fell else 0.)  #5 unhealthy+20terminal ONCE.
        truncated = self.local_t == self.limit and not fell
        # Deliberately wildly different adjusted reward: the oracle must ignore it.
        return self.obs(), -raw - (4995. if fell else 0.), fell, truncated, {"task_cost": raw}

    def close(self):
        type(self).closed += 1


class MemoryPlanner:
    instances = []

    def __init__(self, env, cfg, seed):
        self.env, self.cfg, self.seed, self.calls = env, cfg, seed, []
        type(self).instances.append(self)

    def plan_step(self, state, *, initial_warmstart):
        self.calls.append((state.copy(), initial_warmstart.copy()))
        # Increasing actions expose whether one planner's own history persists.
        action = np.array([len(self.calls)], dtype=float)
        passes = self.cfg.initial_iterations if len(self.calls) == 1 else 1
        return action, {"optimization_iterations": passes}


@pytest.fixture
def start():
    ToyEnv.resets, ToyEnv.closed, MemoryPlanner.instances = [], 0, []
    state = np.zeros(19); state[2] = 1.25
    return Snapshot(state, np.array([7.]))


def forbidden_planner(*args, **kwargs):
    raise AssertionError("An expert should not have been constructed")


def test_policy_prefix_preserves_full_endpoint_and_warmstart_for_fresh_expert(start):
    actor = Actor(2.)
    cfg = replace(raw_teacher_config(), K=4, H=3)
    result = takeover_objective(start, actor, prefix_steps=2, total_steps=5, planner_seed=123,
                                 cfg=cfg, gamma=.5, env_factory=ToyEnv, planner_factory=MemoryPlanner)
    assert actor.calls == result.prefix.steps == 2
    assert result.prefix.cost == 5. + .5 * 5.
    assert result.prefix.healthy and result.prefix.truncated
    np.testing.assert_array_equal(result.prefix.endpoint_warmstart, [13.])
    np.testing.assert_array_equal(ToyEnv.resets[1][0], result.prefix.endpoint_state)
    np.testing.assert_array_equal(ToyEnv.resets[1][1], result.prefix.endpoint_warmstart)
    suffix = result.suffix
    np.testing.assert_array_equal(suffix.actions, [[1.], [2.], [3.]])
    assert suffix.cost == 2. + .5 * 5. + .25 * 10.
    assert result.cost == result.prefix.cost + .5**2 * suffix.cost
    assert suffix.healthy and suffix.truncated and suffix.absorbing_cost == 0.
    assert result.ledger == {"policy_controls": 2, "teacher_controls": 3, "teacher_candidate_controls": 48,
                             "teacher_planner_calls": 3, "teacher_optimization_iterations": 4}
    assert len(MemoryPlanner.instances) == 1 and MemoryPlanner.instances[0].seed == 123
    np.testing.assert_array_equal(MemoryPlanner.instances[0].calls[0][1], [13.])
    assert ToyEnv.closed == 2


def test_prefix_fall_absorbs_whole_remaining_horizon_without_expert_or_extra_terminal(start):
    class FallAtTwo(ToyEnv):
        fall_at = 2

    actor = Actor(2.)
    result = takeover_objective(start, actor, prefix_steps=3, total_steps=5, planner_seed=4,
                                 gamma=.5, env_factory=FallAtTwo, planner_factory=forbidden_planner)
    # Raw falling cost30 already includes terminal20. No4995 reward tail.
    assert result.prefix.steps == actor.calls == 2
    assert result.prefix.cost == 5. + .5 * 30.
    assert result.cost == 5. + .5 * 30. + .5**2 * 5. * (1. + .5 + .25)
    assert result.suffix is None and not result.prefix.healthy
    assert result.ledger["teacher_candidate_controls"] == result.ledger["teacher_controls"] == 0


def test_expert_fall_charges_local_finite_absorption_then_outer_prefix_discount(start):
    class FallAtThree(ToyEnv):
        fall_at = 3

    cfg = replace(raw_teacher_config(), K=4, H=3)
    result = takeover_objective(start, Actor(2.), prefix_steps=2, total_steps=5, planner_seed=4, cfg=cfg,
                                 gamma=.5, env_factory=FallAtThree, planner_factory=MemoryPlanner)
    assert result.suffix.steps == 1 and result.suffix.terminated
    assert result.suffix.executed_cost == 27.  #1+1action²+5unhealthy+20terminal.
    assert result.suffix.absorbing_cost == .5 * 5. * (1. + .5)
    assert result.cost == 7.5 + .5**2 * (27. + .5 * 5. * (1. + .5))
    assert result.ledger["teacher_candidate_controls"] == 4 * 3 * 2


def test_full_takeover_at_L_equals_N_is_standalone_and_never_constructs_mppi(start):
    result = takeover_objective(start, Actor(2.), prefix_steps=3, total_steps=3, planner_seed=6,
                                 gamma=1., env_factory=ToyEnv, planner_factory=forbidden_planner)
    assert result.cost == 15.
    assert result.prefix.healthy and result.prefix.truncated and result.suffix is None
    assert result.ledger["policy_controls"] == 3
    assert result.ledger["teacher_controls"] == result.ledger["teacher_candidate_controls"] == 0


def test_zero_actor_prefix_uses_expert_only_and_teacher_api_has_no_actor_or_critic(start):
    class ForbiddenActor:
        def act(self, *args, **kwargs):
            raise AssertionError("Zero prefix called the actor")

    result = takeover_objective(start, ForbiddenActor(), prefix_steps=0, total_steps=3, planner_seed=7,
                                 gamma=1., env_factory=ToyEnv, planner_factory=MemoryPlanner)
    assert result.prefix.steps == result.prefix.cost == 0
    assert result.cost == 2. + 5. + 10.
    assert result.ledger["policy_controls"] == 0
    signature = inspect.signature(independent_teacher_suffix)
    assert "actor" not in signature.parameters and "critic" not in signature.parameters
    assert "prior_cost" not in signature.parameters and "nominal" not in signature.parameters


def test_already_fallen_snapshot_has_only_remaining_absorption_no_new_terminal(start):
    state = start.state.copy(); state[2] = .5
    result = takeover_objective(Snapshot(state, start.warmstart), Actor(), prefix_steps=2, total_steps=4,
                                 planner_seed=6, gamma=1., env_factory=ToyEnv, planner_factory=forbidden_planner)
    assert result.cost == 4 * 5.
    assert result.prefix.steps == 0 and result.prefix.terminated
    assert result.suffix is None


@pytest.mark.parametrize("gamma", [0., .5, .999, 1. - 1e-12, 1.])
def test_finite_absorbing_utility_matches_direct_sum_including_undiscounted_limit(gamma):
    assert finite_absorbing_cost(gamma, 0) == 0.
    expected = sum(5. * gamma**t for t in range(37))
    assert finite_absorbing_cost(gamma, 37) == pytest.approx(expected, rel=1e-13)
    assert discounted_raw_cost([2., 3.], gamma) == 2. + gamma * 3.


def test_actual_walker_raw_expert_query_is_identical_after_actor_and_global_rng_changes():
    env = WalkerRL(max_episode_steps=10)
    try:
        env.reset(seed=9_400_001)
        snapshot = Snapshot(env.walker.get_state(), env.walker.get_warmstart())
        low, high = env.walker.action_bounds
    finally:
        env.close()
    cfg = replace(raw_teacher_config(), K=4, H=4)
    first = independent_teacher_suffix(snapshot, steps=3, planner_seed=9_410_001, cfg=cfg)
    torch.manual_seed(934)
    actor = TanhMLPPolicy(17, low, high, hidden_dim=8)
    rollout_policy_prefix(snapshot, actor, steps=2)
    np.random.seed(123); np.random.normal(size=100)
    torch.rand(100)
    second = independent_teacher_suffix(snapshot, steps=3, planner_seed=9_410_001, cfg=cfg)
    np.testing.assert_array_equal(first.actions, second.actions)
    np.testing.assert_array_equal(first.raw_costs, second.raw_costs)
    np.testing.assert_array_equal(first.endpoint_state, second.endpoint_state)
    np.testing.assert_array_equal(first.endpoint_warmstart, second.endpoint_warmstart)
    assert first.cost == second.cost
    assert first.ledger["teacher_candidate_controls"] == cfg.K * cfg.H * (2 + 1 + 1)


def test_actual_walker_finite_undiscounted_prefix_does_not_use_infinite_reward_tail():
    env = WalkerRL(max_episode_steps=3)
    try:
        obs, _ = env.reset(seed=9_400_002)
        snapshot = Snapshot(env.walker.get_state(), env.walker.get_warmstart())
        low, high = env.walker.action_bounds
    finally:
        env.close()
    actor = TanhMLPPolicy(17, low, high, hidden_dim=8)
    result = takeover_objective(snapshot, actor, prefix_steps=2, total_steps=2, planner_seed=9_410_002,
                                 gamma=1., planner_factory=forbidden_planner)
    assert result.prefix.steps == 2 and result.prefix.healthy
    assert result.cost == sum(result.prefix.raw_costs)


def test_nonfinite_raw_cost_cannot_become_an_oracle_target(start):
    class BrokenEnv(ToyEnv):
        def step(self, action):
            obs, reward, done, cutoff, info = super().step(action)
            info["task_cost"] = float("nan")
            return obs, reward, done, cutoff, info
    with pytest.raises(ValueError, match="finite"):
        rollout_policy_prefix(start, Actor(), steps=1, env_factory=BrokenEnv)


def test_reset_must_reproduce_solver_warmstart_exactly(start):
    class BrokenReset(ToyEnv):
        def reset(self, *, options):
            result = super().reset(options=options)
            self.walker.warmstart[:] = 0.
            return result
    with pytest.raises(RuntimeError, match="warmstart"):
        rollout_policy_prefix(start, Actor(), steps=1, env_factory=BrokenReset)
