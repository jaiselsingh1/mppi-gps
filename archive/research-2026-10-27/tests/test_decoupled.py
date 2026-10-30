"""Proposal accounting, strict verification gates and discrete-mode provenance."""
import math

import pytest
import torch

from src.gps.decoupled import make_proposals, synthesize_verified
from src.gps.torch_mppi import softmin_weights, synthesize_plan, temperature_for_neff_fraction


def _bounds(actions=1, limit=2.):
    return torch.full((actions,), -limit, dtype=torch.float64), torch.full((actions,), limit, dtype=torch.float64)


def _saddle(plans, rows):
    u, v = plans[:, 0, 0], plans[:, 0, 1]
    return (1. + u.square()) * (v.square() - 1.).square()


def test_seeded_cloud_exactly_preserves_existing_bounded_noise_arithmetic():
    nominal = torch.tensor([[[.8], [-.8]]], dtype=torch.float64)
    noise = torch.tensor([[[[.4], [-.4]], [[-.4], [.4]]]], dtype=torch.float64)
    low, high = _bounds(limit=1.)
    actual = make_proposals(nominal, noise, low, high, policy_sigma=.17)
    torch.testing.assert_close(actual, torch.maximum(torch.minimum(nominal[:, None] + noise, high), low), rtol=0, atol=0)
    assert actual.dtype == torch.float64


def test_split_is_128_plus_128_with_fixed_independent_noise_and_no_policy_center_leakage():
    nominal = torch.full((2, 3, 1), .4, dtype=torch.float64)
    standardized = torch.linspace(-.5, .5, 256, dtype=torch.float64).reshape(1, 256, 1, 1).expand(2, -1, 3, -1)
    low, high = _bounds()
    cloud = make_proposals(nominal, .15 * standardized, low, high,
                           proposal_mode="split_proposal", policy_sigma=.15)
    torch.testing.assert_close(cloud[:, :128], nominal[:, None] + .15 * standardized[:, :128], rtol=0, atol=0)
    torch.testing.assert_close(cloud[:, 128:], .3 * standardized[:, 128:], rtol=0, atol=0)
    changed = make_proposals(-nominal, .075 * standardized, low, high,
                             proposal_mode="split_proposal", policy_sigma=.075)
    torch.testing.assert_close(changed[:, 128:], cloud[:, 128:], rtol=0, atol=0)
    assert cloud.shape == (2, 256, 3, 1)


def test_held_independent_center_requires_explicit_action_and_never_uses_policy_nominal():
    nominal = torch.full((1, 3, 1), .8, dtype=torch.float64)
    noise = torch.zeros((1, 4, 3, 1), dtype=torch.float64)
    low, high = _bounds()
    held = make_proposals(nominal, noise, low, high, proposal_mode="split_proposal",
                          independent_center="hold", previous_action=torch.tensor([[-.25]]))
    torch.testing.assert_close(held[:, 2:], torch.full((1, 2, 3, 1), -.25, dtype=torch.float64))
    with pytest.raises(ValueError, match="explicit"):
        make_proposals(nominal, noise, low, high, proposal_mode="split_proposal", independent_center="hold")


def test_seeded_mean_matches_existing_ess_and_effective_clipped_noise_synthesis():
    rng = torch.Generator().manual_seed(84)
    nominal = torch.rand((3, 4, 2), generator=rng, dtype=torch.float64) * .8
    low, high = _bounds(actions=2, limit=1.)
    sampled = (nominal[:, None] + .3 * torch.randn((3, 8, 4, 2), generator=rng, dtype=torch.float64)).clamp(-1., 1.)
    scores = sampled.square().sum(dim=(2, 3))
    baseline = torch.full((3,), 100., dtype=torch.float64)
    result = synthesize_verified(sampled, scores, nominal, baseline, low, high,
                                  lambda plans, rows: plans.square().sum(dim=(1, 2)))
    lam = temperature_for_neff_fraction(scores, .2)
    weights = softmin_weights(scores, lam)
    expected_mean = synthesize_plan(nominal, sampled, weights, low, high)
    torch.testing.assert_close(result.temperatures, lam, rtol=0, atol=0)
    torch.testing.assert_close(result.weights, weights, rtol=0, atol=0)
    torch.testing.assert_close(result.plans, expected_mean, rtol=0, atol=0)
    assert result.accepted.all()
    assert (result.source_codes == 1).all()


def test_equal_cost_mean_is_rejected_but_not_counted_as_strict_collapse_and_can_be_rescued():
    nominal = torch.zeros((2, 1, 1), dtype=torch.float64)
    sampled = torch.tensor([[[[.2]], [[.4]]], [[[.5]], [[.7]]]], dtype=torch.float64)
    scores = torch.tensor([[.2, .4], [1., 2.]], dtype=torch.float64)
    baseline = torch.ones(2, dtype=torch.float64)
    result = synthesize_verified(sampled, scores, nominal, baseline, *_bounds(),
                                  lambda plans, rows: torch.ones(len(rows), dtype=torch.float64),
                                  synthesis_mode="medoid_fallback")
    assert result.mean_tied.all() and result.mean_rejected.all()
    assert not result.bimodal_collapse.any()
    torch.testing.assert_close(result.accepted, torch.tensor([True, False]))
    assert result.source_codes.tolist() == [2, 0]
    assert result.anchor_indices.tolist() == [0, -1]
    assert result.diagnostics()["medoid_rescue_rate"] is None
    assert result.diagnostics()["eligible_rescue_rate"] == 1.
    assert result.diagnostics()["all_rejected_rescue_rate"] == .5


def test_anchor_beta_is_explicit_and_only_first_action_distance_affects_tie_objective():
    nominal = torch.zeros((1, 2, 1), dtype=torch.float64)
    sampled = torch.tensor([[[[.2], [1.8]], [[.8], [0.]]]], dtype=torch.float64)
    scores, baseline = torch.tensor([[.7, .4]], dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    verify = lambda plans, rows: torch.full((len(rows),), 2., dtype=torch.float64)
    cost_only = synthesize_verified(sampled, scores, nominal, baseline, *_bounds(), verify,
                                    synthesis_mode="medoid_fallback")
    proximity = synthesize_verified(sampled, scores, nominal, baseline, *_bounds(), verify,
                                    synthesis_mode="medoid_fallback", beta=1.)
    assert cost_only.anchor_indices.item() == 1
    assert proximity.anchor_indices.item() == 0
    assert proximity.costs.item() == .7


def test_optional_local_ball_still_verifies_and_falls_back_to_single_anchor_when_it_crosses_saddle():
    nominal = torch.zeros((1, 1, 2), dtype=torch.float64)
    sampled = torch.tensor([[[[1., 1.]], [[1., -1.]]]], dtype=torch.float64)
    scores, baseline = torch.zeros((1, 2), dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    result = synthesize_verified(sampled, scores, nominal, baseline, *_bounds(2), _saddle,
                                  synthesis_mode="medoid_fallback", cluster_radius=2.1)
    assert result.cluster_candidate_count.item() == 2
    assert result.source_codes.item() == 2
    assert result.costs.item() == 0.
    assert result.verification_callback_calls == result.verification_rollouts == 2
    # At radius exactly2 the second mode lies on the boundary and is excluded.
    strict = synthesize_verified(sampled, scores, nominal, baseline, *_bounds(2), _saddle,
                                  synthesis_mode="medoid_fallback", cluster_radius=2.)
    assert strict.cluster_candidate_count.item() == 1
    assert strict.verification_rollouts == 1


def test_verified_local_cluster_can_be_accepted_and_preserves_anchor_provenance():
    nominal = torch.zeros((1, 1, 2), dtype=torch.float64)
    sampled = torch.tensor([[[[1., 1.]], [[1., 1.1]], [[1., -1.]]]], dtype=torch.float64)
    scores = _saddle(sampled.flatten(0, 1), None).reshape(1, 3)
    result = synthesize_verified(sampled, scores, nominal, torch.ones(1, dtype=torch.float64), *_bounds(2), _saddle,
                                  synthesis_mode="medoid_fallback", neff_fraction=.8, cluster_radius=.3)
    assert result.mean_rejected.item()
    assert result.source_codes.item() == 3
    assert result.anchor_indices.item() == 0
    assert result.cluster_candidate_count.item() == 2
    assert result.accepted.item()
    assert result.costs.item() == _saddle(result.plans, None).item() < 1.


def test_nonfinite_candidate_scores_are_excluded_without_contaminating_weighted_expectation():
    nominal = torch.zeros((1, 1, 1), dtype=torch.float64)
    sampled = torch.tensor([[[[.2]], [[.3]], [[.4]]]], dtype=torch.float64)
    scores = torch.tensor([[.2, float("nan"), float("inf")]], dtype=torch.float64)
    result = synthesize_verified(sampled, scores, nominal, torch.ones(1, dtype=torch.float64), *_bounds(),
                                  lambda plans, rows: plans[:, 0, 0])
    assert result.weights.tolist() == [[1., 0., 0.]]
    assert result.expected_candidate_cost.item() == .2
    assert result.costs.item() == .2
    assert result.improving_candidate_count.item() == 1


@pytest.mark.parametrize("which", ["nominal", "sampled", "baseline", "all_scores", "verification"])
def test_nonfinite_critical_inputs_or_verification_fail_closed(which):
    nominal = torch.zeros((1, 1, 1), dtype=torch.float64)
    sampled = torch.zeros((1, 2, 1, 1), dtype=torch.float64)
    scores, baseline = torch.zeros((1, 2), dtype=torch.float64), torch.ones(1, dtype=torch.float64)
    if which == "nominal": nominal[:] = float("nan")
    if which == "sampled": sampled[:] = float("inf")
    if which == "baseline": baseline[:] = float("nan")
    if which == "all_scores": scores[:] = float("inf")
    with pytest.raises(FloatingPointError):
        synthesize_verified(sampled, scores, nominal, baseline, *_bounds(),
                             lambda plans, rows: torch.full((len(rows),), float("nan") if which == "verification" else 0., dtype=torch.float64))


def test_malformed_callback_cost_shape_cannot_be_broadcast_into_a_gate():
    with pytest.raises(ValueError, match="scalar cost"):
        synthesize_verified(torch.zeros((1, 2, 1, 1)), torch.zeros((1, 2)), torch.zeros((1, 1, 1)),
                             torch.ones(1), *_bounds(), lambda plans, rows: torch.zeros((1, 1)))


@pytest.mark.parametrize("settings", [{"policy_sigma": 0.}, {"policy_sigma": math.nan},
                                      {"proposal_mode": "unknown"}, {"independent_sigma": math.inf}])
def test_bad_proposal_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        make_proposals(torch.zeros((1, 1, 1)), torch.zeros((1, 256, 1, 1)), *_bounds(), **settings)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_verified_saddle_and_discrete_provenance_remain_on_cuda_in_float64():
    device = torch.device("cuda")
    nominal = torch.zeros((1, 1, 2), device=device, dtype=torch.float64)
    sampled = torch.tensor([[[[1., 1.]], [[1., -1.]]]], device=device, dtype=torch.float64)
    low, high = (bound.to(device) for bound in _bounds(2))
    result = synthesize_verified(sampled, torch.zeros((1, 2), device=device, dtype=torch.float64), nominal,
                                  torch.ones(1, device=device, dtype=torch.float64), low, high, _saddle,
                                  synthesis_mode="medoid_fallback")
    assert result.plans.device.type == result.weights.device.type == "cuda"
    assert result.plans.dtype == result.weights.dtype == torch.float64
    assert result.source_codes.item() == 2 and result.costs.item() == 0.
