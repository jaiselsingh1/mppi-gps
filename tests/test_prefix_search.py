"""Parameter-only ES mechanics; no planner action targets or learned critic."""

import copy

import numpy as np
import pytest
import torch
from torch import nn

from src.gps.prefix_search import (
    PolicyParameterSpace, antithetic_candidates, propose_es_update, verify_parameter_proposal,
)


class SmallActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([.2, .2], dtype=torch.float64))
        self.log_std = nn.Parameter(torch.tensor([-.5, -.5], dtype=torch.float64))
        self.fixed = nn.Parameter(torch.tensor(7.), requires_grad=False)
        self.register_buffer("offset", torch.tensor([1., 2.]))

    def forward(self, phase):
        return self.mean[phase]


def test_search_space_excludes_logstd_frozen_parameters_and_buffers():
    actor = SmallActor()
    space = PolicyParameterSpace.from_actor(actor)
    assert space.names == ("mean",) and space.size == 2
    np.testing.assert_array_equal(space.flatten(actor), [.2, .2])
    clone = space.actor_copy(actor, np.array([.6, -.8]))
    np.testing.assert_array_equal(space.flatten(clone), [.6, -.8])
    np.testing.assert_array_equal(space.flatten(actor), [.2, .2])
    for key in ("log_std", "fixed", "offset"):
        torch.testing.assert_close(clone.state_dict()[key], actor.state_dict()[key], rtol=0, atol=0)
    assert not clone.training
    clone.mean.data.fill_(12)
    np.testing.assert_array_equal(space.flatten(actor), [.2, .2])


def test_private_rng_reproducible_antithetic_pairs_leave_global_state_unchanged():
    actor = SmallActor()
    numpy_before = copy.deepcopy(np.random.get_state())
    torch_before = torch.random.get_rng_state().clone()
    first = antithetic_candidates(actor, pairs=4, sigma=.03, seed=8)
    second = antithetic_candidates(actor, pairs=4, sigma=.03, seed=8)
    other = antithetic_candidates(actor, pairs=4, sigma=.03, seed=9)
    first.actor_copy(actor, 0)
    np.testing.assert_array_equal(first.parameters, second.parameters)
    assert not np.array_equal(first.parameters, other.parameters)
    np.testing.assert_allclose(first.parameters[0::2] + first.parameters[1::2],
                               np.broadcast_to(2 * first.base_parameters, (4, 2)), rtol=0, atol=1e-16)
    assert all(np.array_equal(a, b) for a, b in zip(numpy_before, np.random.get_state()))
    torch.testing.assert_close(torch_before, torch.random.get_rng_state(), rtol=0, atol=0)
    assert not first.parameters.flags.writeable and not first.perturbations.flags.writeable
    with pytest.raises(ValueError):
        first.parameters[0, 0] = 1


@pytest.mark.parametrize("scaled", [False, True])
def test_paired_cost_difference_estimator_uses_effective_sigma(scaled):
    actor = SmallActor()
    batch = antithetic_candidates(actor, pairs=5, sigma=.1, seed=2, scale_by_parameter_count=scaled)
    assert batch.effective_sigma == pytest.approx(.1 / np.sqrt(2) if scaled else .1)
    direction = np.array([2., -3.])
    costs = batch.parameters @ direction + 13.
    expected = np.mean((batch.perturbations @ direction)[:, None] * batch.perturbations, axis=0)
    result = propose_es_update(batch, costs, learning_rate=.01, max_step_norm=100.)
    np.testing.assert_allclose(result.gradient, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.step, -.01 * expected)
    assert result.cost_scale == 1. and not result.costs_normalized
    np.testing.assert_array_equal(batch.space.flatten(actor), [.2, .2])


def test_normalization_is_explicit_and_complete_step_is_norm_clamped():
    batch = antithetic_candidates(SmallActor(), pairs=4, sigma=.02, seed=4)
    costs = np.array([1., 3., 2., 8., 4., 2., 9., 5.])
    raw = propose_es_update(batch, costs, learning_rate=.1, max_step_norm=1e9)
    normalized = propose_es_update(batch, costs, learning_rate=.1, max_step_norm=.03,
                                    normalize_costs=True)
    np.testing.assert_allclose(normalized.gradient, raw.gradient / np.std(costs))
    assert normalized.costs_normalized and normalized.cost_scale == pytest.approx(np.std(costs))
    assert normalized.raw_step_norm > .03 and normalized.step_norm == pytest.approx(.03)
    np.testing.assert_allclose(normalized.proposed_parameters, batch.base_parameters + normalized.step)


def test_constant_cost_has_zero_update_even_with_normalization():
    batch = antithetic_candidates(SmallActor(), pairs=3, sigma=.01, seed=4)
    result = propose_es_update(batch, np.ones(6) * 19, learning_rate=1, max_step_norm=1,
                               normalize_costs=True)
    np.testing.assert_array_equal(result.proposed_parameters, result.base_parameters)
    assert result.gradient_norm == result.raw_step_norm == result.step_norm == 0
    assert result.cost_scale == 1e-12


@pytest.mark.parametrize("costs", [[1, 2], [1, 2, 3, np.nan], [1, 2, np.inf, 4], [[1, 2], [3, 4]]])
def test_malformed_or_nonfinite_candidate_costs_are_rejected(costs):
    batch = antithetic_candidates(SmallActor(), pairs=2, sigma=.01, seed=0)
    with pytest.raises(ValueError):
        propose_es_update(batch, costs, learning_rate=1, max_step_norm=1)


@pytest.mark.parametrize("changes", [{"pairs": 0}, {"pairs": True}, {"sigma": 0},
                                     {"sigma": np.inf}, {"seed": -1}, {"seed": True},
                                     {"scale_by_parameter_count": "yes"}])
def test_invalid_search_configuration_rejected(changes):
    args = dict(pairs=2, sigma=.01, seed=0)
    args.update(changes)
    with pytest.raises(ValueError):
        antithetic_candidates(SmallActor(), **args)


def test_actor_layout_mismatch_and_dtype_overflow_are_rejected():
    actor = SmallActor()
    space = PolicyParameterSpace.from_actor(actor)
    actor.mean.requires_grad_(False)
    with pytest.raises(ValueError):
        space.flatten(actor)
    actor.mean.requires_grad_(True)
    with pytest.raises(ValueError):
        space.actor_copy(actor, np.zeros(3))
    with pytest.raises(FloatingPointError):
        space.actor_copy(actor.float(), np.array([1e100, 2.]))


def test_proposed_update_requires_actual_cost_verification_and_rejects_overshoot():
    actor = SmallActor()
    # This scalar trajectory outcome rewards a coordinated pair of actions.
    outcome_cost = lambda parameters: float((parameters[0] * parameters[1] - 1.) ** 2)
    batch = antithetic_candidates(actor, pairs=32, sigma=.01, seed=3)
    costs = np.array([outcome_cost(row) for row in batch.parameters])
    result = propose_es_update(batch, costs, learning_rate=100., max_step_norm=10.)
    baseline, proposed = outcome_cost(batch.base_parameters), outcome_cost(result.proposed_parameters)
    assert proposed > baseline
    verified = verify_parameter_proposal(result, base_cost=baseline, proposed_cost=proposed)
    assert not verified.accepted
    np.testing.assert_array_equal(verified.parameters, batch.base_parameters)
    assert not verify_parameter_proposal(result, base_cost=1., proposed_cost=1.).accepted
    accepted = verify_parameter_proposal(result, base_cost=1., proposed_cost=.5)
    assert accepted.accepted
    np.testing.assert_array_equal(accepted.parameters, result.proposed_parameters)
    with pytest.raises(ValueError):
        verify_parameter_proposal(result, base_cost=1., proposed_cost=np.nan)


def test_two_control_coordinated_outcome_improves_without_action_targets():
    actor = SmallActor()

    def cost(policy):
        # Evaluate the actor's OWN two actions. No teacher action appears here.
        first, second = float(policy(0).detach()), float(policy(1).detach())
        return (first * second - 1.) ** 2 + .02 * (first * first + second * second)

    initial = cost(actor)
    accepted = 0
    for iteration in range(30):
        batch = antithetic_candidates(actor, pairs=32, sigma=.03, seed=iteration)
        costs = np.array([cost(batch.actor_copy(actor, i)) for i in range(2 * batch.pairs)])
        proposal = propose_es_update(batch, costs, learning_rate=.15, max_step_norm=.1)
        candidate_actor = batch.space.actor_copy(actor, proposal.proposed_parameters)
        verified = verify_parameter_proposal(proposal, base_cost=cost(actor), proposed_cost=cost(candidate_actor))
        if verified.accepted:
            accepted += 1
            actor = batch.space.actor_copy(actor, verified.parameters)
    assert accepted > 15 and cost(actor) < .05 and cost(actor) < initial / 10
