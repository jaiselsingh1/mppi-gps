"""A coordinated expert can help even when its first action has bad Q^pi.

This two-step convex toy exercises the real takeover/prefix-objective/search
interfaces. It is a regression for objective ownership and complete expert
withdrawal, not a learning-performance claim about Walker or neural critics.
"""
from dataclasses import replace

import numpy as np
import pytest
import torch

from scripts.training.train_expert_withdrawal import prefix_objective
from src.gps.expert_continuation import Snapshot, raw_teacher_config, rollout_policy_prefix, takeover_objective
from src.gps.prefix_search import antithetic_candidates, propose_es_update, verify_parameter_proposal


class TwoStepActor(torch.nn.Module):
    """Two trainable feedback actions, selected by the observed phase."""

    def __init__(self, first=0., second=0.):
        super().__init__()
        self.parameters_by_phase = torch.nn.Parameter(torch.tensor([first, second], dtype=torch.float64))

    def act(self, obs, deterministic=True):
        assert deterministic
        phase = int(obs[0])
        return np.array([self.parameters_by_phase[phase].detach().item()])


class TwoStepWalker:
    action_dim, _w_unhealthy = 1, 5.

    @staticmethod
    def _is_healthy(z, angle):
        return .8 < z < 2. and -1. < angle < 1.

    def get_state(self):
        return self.state.copy()

    def get_warmstart(self):
        return self.warmstart.copy()


class TwoStepEnv:
    """c0=.25*a0²; c1=(a0-1)²+(a1-a0)²; no fall, finite horizon2."""

    def __init__(self, *, task_config, gamma, max_episode_steps):
        self.walker, self.limit, self.local_steps = TwoStepWalker(), max_episode_steps, 0

    def observation(self):
        obs = np.zeros(17)
        obs[0], obs[1] = self.walker.state[0], self.walker.state[4]
        return obs

    def reset(self, *, options):
        self.walker.state = options["state"].copy()
        self.walker.warmstart = options["warmstart"].copy()
        return self.observation(), {}

    def step(self, action):
        phase, action = int(self.walker.state[0]), float(action[0])
        if phase == 0:
            cost = .25 * action**2
            self.walker.state[4] = action
        elif phase == 1:
            first = self.walker.state[4]
            cost = (first - 1.)**2 + (action - first)**2
        else:
            raise AssertionError("The finite two-step task was stepped beyond its horizon")
        self.walker.state[0] += 1.
        self.walker.warmstart += action + 1.
        self.local_steps += 1
        return self.observation(), -cost, False, self.local_steps == self.limit, {"task_cost": cost}

    def close(self):
        pass


class CoordinatedExpert:
    """An independent(1,1) expert, adapting its second action to actual a0."""

    def __init__(self, env, cfg, seed):
        self.calls = 0

    def plan_step(self, state, *, initial_warmstart):
        action = 1. if int(state[0]) == 0 else state[4]
        self.calls += 1
        return np.array([action]), {"optimization_iterations": 1}


class ExactExpertValue:
    def __init__(self):
        self.calls = 0

    def predict(self, observations, remaining):
        assert remaining == 1
        self.calls += 1
        # At x1 the expert executes a1=a0, so its remaining raw cost is exact.
        return (observations[:, 1] - 1.)**2


class ForbiddenContinuation:
    def predict(self, *args, **kwargs):
        raise AssertionError("Full takeover L=N queried an expert value")


def _start():
    state = np.zeros(19); state[2] = 1.25
    return Snapshot(state, np.array([3.]))


def _actual(actor, length, *, planner_factory=CoordinatedExpert):
    return takeover_objective(_start(), actor, prefix_steps=length, total_steps=2, planner_seed=0,
                               cfg=replace(raw_teacher_config(), K=1, H=2, initial_iterations=1),
                               gamma=1., env_factory=TwoStepEnv, planner_factory=planner_factory)


def _estimated(actor, length, value):
    prefix = rollout_policy_prefix(_start(), actor, steps=length, gamma=1., env_factory=TwoStepEnv)
    return prefix_objective(prefix, total_steps=2, value=value)


def test_coordinated_expert_first_action_has_negative_advantage_under_weak_learner_continuation():
    weak = _actual(TwoStepActor(0., 0.), 2)
    expert = _actual(TwoStepActor(0., 0.), 0)
    first_expert_then_weak = _actual(TwoStepActor(1., 0.), 2)
    assert weak.cost == 1.
    assert expert.cost == .25
    assert first_expert_then_weak.cost == 1.25
    assert expert.cost < weak.cost < first_expert_then_weak.cost
    # Reward advantages have the opposite sign from these cost differences.
    assert -first_expert_then_weak.cost - (-weak.cost) == -.25
    assert expert.ledger["policy_controls"] == 0
    assert first_expert_then_weak.ledger["teacher_controls"] == 0


def test_antithetic_return_search_learns_complete_policy_after_expert_withdrawal():
    actor, value = TwoStepActor(), ExactExpertValue()
    accepted, stage_objectives = [], []
    for length, iterations, pairs in ((1, 28, 16), (2, 64, 24)):
        stage_value = value if length == 1 else ForbiddenContinuation()
        previous = _actual(actor, length).cost
        stage_costs = [previous]
        for iteration in range(iterations):
            candidates = antithetic_candidates(actor, pairs=pairs, sigma=.05,
                                                seed=8_000 + 100 * length + iteration)
            candidate_costs = np.array([_estimated(candidates.actor_copy(actor, i), length, stage_value)
                                        for i in range(2 * pairs)])
            update = propose_es_update(candidates, candidate_costs, learning_rate=.15, max_step_norm=.2)
            proposal = candidates.space.actor_copy(actor, update.proposed_parameters)
            # Actual expert continuation verifies stage1. Stage2 is entirely
            # the candidate policy; any expert construction would be a failure.
            def forbidden_planner(*args, **kwargs):
                raise AssertionError("Full takeover constructed an expert planner")
            actual = _actual(proposal, length,
                             planner_factory=CoordinatedExpert if length == 1 else forbidden_planner)
            assert _estimated(proposal, length, stage_value) == pytest.approx(actual.cost, abs=1e-14)
            verified = verify_parameter_proposal(update, base_cost=previous, proposed_cost=actual.cost)
            accepted.append(verified.accepted)
            actor = candidates.space.actor_copy(actor, verified.parameters)
            if verified.accepted:
                previous = actual.cost
            stage_costs.append(previous)
            if length == 2:
                assert actual.suffix is None
                assert actual.ledger["teacher_controls"] == actual.ledger["teacher_candidate_controls"] == 0
        assert np.all(np.diff(stage_costs) <= 0.)
        stage_objectives.append(stage_costs)
    assert sum(accepted) > 40 and value.calls > 0
    learned = actor.parameters_by_phase.detach().numpy()
    np.testing.assert_allclose(learned, [.8, .8], atol=2e-3)
    final = _actual(actor, 2)
    assert final.cost == pytest.approx(.2, abs=1e-5)
    assert final.cost < .25  # Better than the(1,1) teacher, without action targets.
    assert not np.allclose(learned, [1., 1.])
    assert final.ledger["policy_controls"] == 2
    assert final.ledger["teacher_controls"] == 0
    assert stage_objectives[1][-1] < stage_objectives[1][0]
