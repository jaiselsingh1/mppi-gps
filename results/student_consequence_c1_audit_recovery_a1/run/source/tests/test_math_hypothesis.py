"""Property-based checks; install the optional audit tools to execute these."""

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings
from hypothesis import strategies as st

from src.utils.math import (
    effective_sample_size,
    plug_in_kl,
    softmin_weights,
    temperature_for_budget,
    temperature_for_neff_fraction,
)

scores_strategy = st.lists(
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    min_size=2,
    max_size=30,
).map(np.asarray)


@settings(max_examples=150, derandomize=True)
@given(scores_strategy, st.floats(min_value=0.05, max_value=20))
def test_gibbs_normalization_order_improvement_and_renyi_bound(scores, lam):
    weights = softmin_weights(scores, lam)
    assert np.isfinite(weights).all() and (weights >= 0).all()
    assert abs(weights.sum() - 1.0) < 1e-12
    order = np.argsort(scores)
    assert (np.diff(weights[order]) <= 1e-14).all()
    assert np.dot(weights, scores) <= scores.mean() + 1e-12
    ess = effective_sample_size(weights)
    assert 1.0 - 1e-12 <= ess <= len(weights) + 1e-12
    assert -1e-12 <= plug_in_kl(weights) <= np.log(len(weights) / ess) + 1e-12


@settings(max_examples=150, derandomize=True)
@given(scores_strategy, st.floats(min_value=0.1, max_value=10), st.floats(-100, 100))
def test_gibbs_shift_scale_equivariance_and_temperature_order(scores, scale, offset):
    np.testing.assert_allclose(
        softmin_weights(scores, 2.0),
        softmin_weights(scale * scores + offset, 2.0 * scale),
        atol=1e-12,
    )
    assert effective_sample_size(softmin_weights(scores, 1.0)) <= (
        effective_sample_size(softmin_weights(scores, 10.0)) + 1e-12
    )


@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-1000, 1000), min_size=3, max_size=30, unique=True),
    st.floats(min_value=0.4, max_value=0.95),
)
def test_temperature_root_hits_feasible_ess_target(values, rho):
    scores = np.asarray(values, dtype=float)
    weights = softmin_weights(scores, temperature_for_neff_fraction(scores, rho))
    np.testing.assert_allclose(effective_sample_size(weights), rho * len(scores), rtol=1e-9)
    assert plug_in_kl(weights) <= np.log(1.0 / rho) + 1e-12


@settings(max_examples=100, derandomize=True)
@given(
    st.lists(st.integers(-1000, 1000), min_size=3, max_size=30, unique=True),
    st.floats(min_value=0.05, max_value=0.95),
)
def test_budget_root_hits_feasible_target(values, fraction):
    scores = np.asarray(values, dtype=float)
    budget = scores.min() + fraction * (scores.mean() - scores.min())
    weights = softmin_weights(scores, temperature_for_budget(scores, budget))
    np.testing.assert_allclose(np.dot(weights, scores), budget, atol=1e-9)


def test_unreachable_ess_targets_report_saturation_not_a_kl_guarantee():
    tied = np.array([0.0, 0.0, 4.0, np.inf])
    low = softmin_weights(tied, temperature_for_neff_fraction(tied, 0.1))
    high = softmin_weights(tied, temperature_for_neff_fraction(tied, 0.9))
    np.testing.assert_allclose(effective_sample_size(low), 2.0)
    np.testing.assert_allclose(effective_sample_size(high), 3.0)
    assert plug_in_kl(high) > np.log(1.0 / 0.9)
