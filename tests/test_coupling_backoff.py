from copy import deepcopy

import pytest

from src.gps.mppi_gps.coupling_backoff import propose_weight


def score(successes=0, survived=0):
    return dict(episodes=16, successes=successes, survived=survived)


def call(guided, independent, **changes):
    args = dict(guided_parent_hash="a"*64, independent_parent_hash="a"*64,
                updated_guided_hash="b"*64, guided_seeds=list(range(16)),
                independent_seeds=list(range(16)))
    args.update(changes)
    return propose_weight(.1199341983507337, guided, independent, **args)


def test_observed_v31_survival_loss_predicts_half_weight_for_updated_actor():
    result = call(score(), score(survived=3))
    assert result["proposed_rho"] == .05996709917536685
    assert result["reasons"] == ["lost_full_survival"]
    assert result["next_acquisition_actor_hash"] == "b"*64
    assert not result["empirical_benefit_established"]


def test_success_loss_triggers_even_without_survival_loss():
    assert call(score(1, 5), score(2, 5))["trigger"]


def test_conservative_or_rule_is_explicit_when_success_and_survival_trade_off():
    result = call(score(2, 2), score(1, 3))
    assert result["trigger"] and result["reasons"] == ["lost_full_survival"]


def test_equal_outcomes_hold_despite_worse_cost_agreement_or_mse():
    guided = dict(score(), mean_cost=9000, agreement=1000, mse=1000)
    original = deepcopy(guided)
    result = call(guided, dict(score(), mean_cost=100, agreement=0, mse=0))
    assert result["proposed_rho"] == .1199341983507337
    assert not result["automatic_increase"] and guided == original


@pytest.mark.parametrize("changes", [
    dict(independent_parent_hash="c"*64),
    dict(independent_seeds=list(range(1, 17))),
    dict(guided_seeds=[0]*16, independent_seeds=[0]*16),
])
def test_unmatched_responses_are_rejected(changes):
    with pytest.raises(ValueError):
        call(score(), score(), **changes)


def test_success_without_survival_is_invalid():
    with pytest.raises(ValueError):
        call(score(1, 0), score())
