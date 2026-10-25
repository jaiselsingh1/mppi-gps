"""Device-batched MPPI weighting agrees with independent and scalar solvers."""

import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from src.gps.torch_mppi import (
    softmin_weights,
    synthesize_plan,
    temperature_for_neff_fraction,
    temperature_for_neff_fraction_with_details,
)
from src.utils.math import softmin_weights as numpy_weights
from src.utils.math import temperature_for_neff_fraction as numpy_temperature


def test_batched_temperatures_match_scalar_and_independent_roots() -> None:
    scores = np.random.default_rng(17).normal(size=(9, 64)) * 12.0 + 1e6
    tensor = torch.from_numpy(scores)
    temperatures = temperature_for_neff_fraction(tensor, 0.2)
    weights = softmin_weights(tensor, temperatures)
    np.testing.assert_allclose(weights.square().sum(1).reciprocal(), 12.8, rtol=1e-9)
    for row, actual in zip(scores, temperatures.numpy()):
        shifted = row - row.min()

        def independent_ess(log_temperature: float) -> float:
            raw = np.exp(-shifted / np.exp(log_temperature))
            normalized = raw / raw.sum()
            return float(1.0 / np.dot(normalized, normalized))

        independent = np.exp(brentq(lambda value: independent_ess(value) - 12.8, -20.0, 20.0))
        np.testing.assert_allclose(actual, independent, rtol=1e-9)
        np.testing.assert_allclose(actual, numpy_temperature(row, 0.2), rtol=1e-9)


def test_weights_are_stable_under_large_score_shift_and_rescaling() -> None:
    scores = torch.tensor([[0.0, 1.0, 5.0, 12.0], [3.0, -1.0, 0.0, 7.0]], dtype=torch.float64)
    original = softmin_weights(scores, temperature_for_neff_fraction(scores, 0.6))
    transformed = scores * 1000.0 + 1e12
    actual = softmin_weights(transformed, temperature_for_neff_fraction(transformed, 0.6))
    torch.testing.assert_close(actual, original, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(actual.sum(1), torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("fraction, expected_ess", [(0.1, 2.0), (0.9, 3.0)])
def test_unreachable_target_uses_attainable_tied_or_invalid_sample_limit(
    fraction: float, expected_ess: float,
) -> None:
    scores = torch.tensor([[0.0, 0.0, 4.0, torch.inf]], dtype=torch.float64)
    temperatures = temperature_for_neff_fraction(scores, fraction)
    weights = softmin_weights(scores, temperatures)
    np.testing.assert_allclose(weights.square().sum(1).reciprocal(), expected_ess)
    assert weights[0, -1] == 0
    assert torch.isfinite(weights).all()


def test_constant_rows_and_explicit_temperature_limits() -> None:
    scores = torch.tensor([[3.0, 3.0, torch.inf, torch.nan], [0.0, 2.0, 4.0, torch.inf]], dtype=torch.float64)
    weights = softmin_weights(scores, torch.tensor([0.0, torch.inf], dtype=torch.float64))
    np.testing.assert_array_equal(weights[0], [0.5, 0.5, 0.0, 0.0])
    np.testing.assert_allclose(weights[1], [1 / 3, 1 / 3, 1 / 3, 0.0])
    constant = scores[:1]
    np.testing.assert_array_equal(temperature_for_neff_fraction(constant, 0.2), [1.0])


def test_bounded_synthesis_matches_the_numpy_effective_noise_update() -> None:
    rng = np.random.default_rng(18)
    nominal = rng.uniform(-1, 1, (5, 4, 3))
    samples = np.clip(nominal[:, None] + rng.normal(size=(5, 32, 4, 3)), -1, 1)
    scores = rng.normal(size=(5, 32))
    temperatures = temperature_for_neff_fraction(torch.from_numpy(scores), 0.2)
    weights = softmin_weights(torch.from_numpy(scores), temperatures)
    actual = synthesize_plan(torch.from_numpy(nominal), torch.from_numpy(samples), weights,
                             -torch.ones(3), torch.ones(3))
    expected = np.stack([
        np.clip(center + np.einsum("k,kha->ha", numpy_weights(cost, lam), drawn - center), -1, 1)
        for center, drawn, cost, lam in zip(nominal, samples, scores, temperatures.numpy())
    ])
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("scores", [torch.empty(2, 0), torch.ones(3), torch.ones(2, 3, dtype=torch.int64)])
def test_bad_score_shapes_and_types_are_rejected(scores: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="floating tensor"):
        temperature_for_neff_fraction(scores, 0.2)


def test_invalid_rows_and_temperature_arguments_are_rejected() -> None:
    with pytest.raises(FloatingPointError, match="finite trajectory"):
        temperature_for_neff_fraction(torch.tensor([[torch.inf, torch.nan]]), 0.2)
    for fraction in (0.0, 1.0, float("nan")):
        with pytest.raises(ValueError, match="target_fraction"):
            temperature_for_neff_fraction(torch.ones(1, 4), fraction)
    for temperature in (-1.0, float("nan")):
        with pytest.raises(ValueError, match="temperatures"):
            softmin_weights(torch.ones(1, 4), torch.tensor([temperature]))
    with pytest.raises(ValueError, match="shape"):
        softmin_weights(torch.ones(2, 4), torch.ones(1))
    with pytest.raises(ValueError, match="iterations"):
        temperature_for_neff_fraction(torch.ones(1, 4), 0.2, iterations=0)


def test_empty_state_batch_has_empty_temperatures_and_weights() -> None:
    scores = torch.empty(0, 4, dtype=torch.float64)
    temperatures = temperature_for_neff_fraction(scores, 0.2)
    assert temperatures.shape == (0,)
    assert softmin_weights(scores, temperatures).shape == (0, 4)


def test_solver_details_detect_feasible_root_outside_initial_bracket() -> None:
    # The ESS=1.2 root exists between the two nearly tied minima, below the
    # solver's spread-scaled lower bracket of 1.0.
    scores = torch.tensor([[0.0, 1e-12, 1e8]], dtype=torch.float64)
    temperatures, saturated = temperature_for_neff_fraction_with_details(scores, 0.4)
    assert saturated.tolist() == [True]
    np.testing.assert_allclose(temperatures, [1.0], rtol=1e-9)
    np.testing.assert_allclose(softmin_weights(scores, temperatures).square().sum(1).reciprocal(), [2.0])


def test_solver_details_distinguish_regular_tied_flat_and_invalid_limits() -> None:
    scores = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 4.0],
        [2.0, 2.0, 2.0, 2.0],
        [0.0, 1.0, torch.inf, torch.inf],
    ], dtype=torch.float64)
    temperatures, saturated = temperature_for_neff_fraction_with_details(scores, 0.5)
    assert saturated.tolist() == [False, True, True, True]
    torch.testing.assert_close(temperatures, temperature_for_neff_fraction(scores, 0.5))
    assert temperatures[1] == 0 and temperatures[2] == 1 and torch.isinf(temperatures[3])
