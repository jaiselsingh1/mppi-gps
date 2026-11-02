"""Independent numerical properties and boundary regressions for Gibbs weights."""

import numpy as np
import pytest

from src.utils.math import softmin_weights, temperature_for_budget


def test_budget_zero_temperature_has_a_defined_minimum_limit():
    scores = np.array([3.0, 3.0, np.inf])
    lam = temperature_for_budget(scores, 2.0)
    assert lam == 0.0
    np.testing.assert_array_equal(softmin_weights(scores, lam), [0.5, 0.5, 0.0])


def test_infinite_temperature_ignores_invalid_scores():
    scores = np.array([1.0, 3.0, np.inf, np.nan])
    lam = temperature_for_budget(scores, 2.0)
    assert lam == np.inf
    np.testing.assert_array_equal(softmin_weights(scores, lam), [0.5, 0.5, 0.0, 0.0])


def test_softmin_rejects_an_unscored_batch():
    with pytest.raises(FloatingPointError, match="finite"):
        softmin_weights(np.array([np.inf, np.nan]), 1.0)


@pytest.mark.parametrize("lam", [-1.0, -np.inf, np.nan])
def test_softmin_rejects_invalid_temperature(lam):
    with pytest.raises(ValueError, match="temperature"):
        softmin_weights(np.array([1.0, 2.0]), lam)
