import numpy as np
import pytest

from src.utils.stats import (
    non_inferior,
    paired_bootstrap_mean_difference,
    paired_proportion_difference,
    wilson_interval,
)


def _paired(both, candidate_only, reference_only, neither):
    candidate = np.array([1] * both + [1] * candidate_only + [0] * reference_only + [0] * neither)
    reference = np.array([1] * both + [0] * candidate_only + [1] * reference_only + [0] * neither)
    return candidate, reference


@pytest.mark.parametrize(
    "successes, n, expected",
    [
        (4, 5, (0.3755, 0.9638)),
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
    ],
)
def test_wilson_interval_known_values(successes, n, expected):
    np.testing.assert_allclose(wilson_interval(successes, n), expected, atol=1e-4)


# Newcombe, Statist. Med. 17:2635-2650 (1998), Table III, method 10, as
# (e, f, g, h): (both, candidate only, reference only, neither). The paper
# prints 0.8736 for (1, 97, 1, 1); its exact value 0.87367 matches method 8,
# which coincides with method 10 whenever eh <= fg.
@pytest.mark.parametrize(
    "cells, expected",
    [
        ((36, 12, 2, 0), (0.0569, 0.3404)),
        ((20, 12, 2, 16), (0.0562, 0.3292)),
        ((18, 12, 2, 18), (0.0562, 0.3290)),
        ((36, 14, 0, 0), (0.1528, 0.4167)),
        ((35, 14, 0, 1), (0.1461, 0.4175)),
        ((18, 14, 0, 18), (0.1441, 0.3963)),
        ((2, 97, 1, 0), (0.8721, 0.9854)),
        ((1, 97, 1, 1), (0.8736, 0.9850)),
        ((0, 29, 1, 0), (0.6666, 0.9882)),
        ((2, 98, 0, 0), (0.9178, 0.9945)),
        ((1, 98, 0, 1), (0.9171, 0.9916)),
        ((0, 30, 0, 0), (0.8395, 1.0)),
        ((54, 0, 0, 0), (-0.0664, 0.0664)),
    ],
)
def test_paired_difference_matches_newcombe_method_10(cells, expected):
    result = paired_proportion_difference(*_paired(*cells))
    np.testing.assert_allclose((result["lower"], result["upper"]), expected, atol=1.5e-4)


@pytest.mark.parametrize("cells", [(30, 0, 0, 0), (0, 0, 0, 30), (12, 0, 0, 18)])
def test_zero_discordant_pairs_do_not_imply_certainty(cells):
    result = paired_proportion_difference(*_paired(*cells))
    assert result["difference"] == 0.0
    assert result["lower"] < -0.05 and result["upper"] > 0.05


def test_paired_difference_has_near_nominal_coverage():
    rng = np.random.default_rng(0)
    cell_probabilities = np.array([0.80, 0.07, 0.03, 0.10])
    true_difference = cell_probabilities[1] - cell_probabilities[2]
    covered = 0
    repetitions = 2000
    for _ in range(repetitions):
        cells = rng.choice(4, size=30, p=cell_probabilities)
        candidate = np.isin(cells, [0, 1]).astype(int)
        reference = np.isin(cells, [0, 2]).astype(int)
        result = paired_proportion_difference(candidate, reference)
        covered += result["lower"] <= true_difference <= result["upper"]
    assert 0.93 <= covered / repetitions <= 0.995


def test_paired_difference_rejects_non_binary_or_unpaired_input():
    with pytest.raises(ValueError, match="binary"):
        paired_proportion_difference(np.array([0, 2]), np.array([0, 1]))
    with pytest.raises(ValueError, match="equal-length"):
        paired_proportion_difference(np.array([0, 1]), np.array([1]))


def test_bootstrap_is_seeded_and_refuses_binary_outcomes():
    rng = np.random.default_rng(3)
    candidate = rng.normal(0.15, 0.05, 40)
    reference = candidate + rng.normal(0.02, 0.01, 40)
    first = paired_bootstrap_mean_difference(candidate, reference, seed=11)
    second = paired_bootstrap_mean_difference(candidate, reference, seed=11)
    assert first == second
    assert first["lower"] < first["difference"] < first["upper"] < 0.0
    with pytest.raises(ValueError, match="paired_proportion_difference"):
        paired_bootstrap_mean_difference(np.ones(5), np.ones(5))


def test_non_inferiority_direction():
    completion = {"lower": -0.04, "upper": 0.02}
    assert non_inferior(completion, margin=0.05, higher_is_better=True)
    assert not non_inferior(completion, margin=0.03, higher_is_better=True)
    speed_error = {"lower": -0.01, "upper": 0.015}
    assert non_inferior(speed_error, margin=0.02, higher_is_better=False)
    assert not non_inferior(speed_error, margin=0.01, higher_is_better=False)
    with pytest.raises(ValueError, match="margin"):
        non_inferior(completion, margin=-0.1, higher_is_better=True)
