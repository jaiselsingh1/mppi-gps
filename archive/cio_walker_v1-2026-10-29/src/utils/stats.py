"""Interval estimates for paired controller comparisons.

Completion outcomes are binary and paired by reset/planner seed. Their
difference uses Newcombe's (1998) method 10 score interval, which stays
non-degenerate when no pair disagrees; a paired bootstrap collapses to [0, 0]
in that case. Continuous metrics use a paired percentile bootstrap.
"""

import math

import numpy as np
from scipy.stats import norm


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError("Require n > 0 and 0 <= successes <= n.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1).")
    z = norm.ppf(0.5 + 0.5 * confidence)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half_width = z / denominator * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    # Clamp floating-point overshoot at the boundary proportions.
    return max(0.0, center - half_width), min(1.0, center + half_width)


def paired_proportion_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    confidence: float = 0.95,
) -> dict:
    """Newcombe method-10 interval for p_candidate - p_reference on paired outcomes.

    candidate[i] and reference[i] are binary outcomes for the same seed pair.
    Method 10 (Statist. Med. 17:2635-2650, 1998) combines Wilson intervals
    for the two marginal proportions with a continuity-corrected phi.
    """
    candidate = np.asarray(candidate)
    reference = np.asarray(reference)
    if candidate.shape != reference.shape or candidate.ndim != 1 or len(candidate) == 0:
        raise ValueError("candidate and reference must be nonempty, equal-length 1-D arrays.")
    for values in (candidate, reference):
        if not np.all(np.isin(values, (0, 1))):
            raise ValueError("Paired proportions require binary outcomes.")
    candidate = candidate.astype(bool)
    reference = reference.astype(bool)
    both = int(np.sum(candidate & reference))
    candidate_only = int(np.sum(candidate & ~reference))
    reference_only = int(np.sum(~candidate & reference))
    neither = int(np.sum(~candidate & ~reference))
    n = len(candidate)

    p_candidate = (both + candidate_only) / n
    p_reference = (both + reference_only) / n
    l1, u1 = wilson_interval(both + candidate_only, n, confidence)
    l2, u2 = wilson_interval(both + reference_only, n, confidence)
    margins = (
        (both + candidate_only) * (reference_only + neither)
        * (both + reference_only) * (candidate_only + neither)
    )
    # Method 10 continuity-corrects only a positive phi numerator.
    numerator = both * neither - candidate_only * reference_only
    if numerator > 0:
        numerator = max(numerator - n / 2.0, 0.0)
    phi = 0.0 if margins == 0 else numerator / math.sqrt(margins)
    difference = p_candidate - p_reference
    lower_term = (p_candidate - l1) ** 2 - 2.0 * phi * (p_candidate - l1) * (u2 - p_reference) + (u2 - p_reference) ** 2
    upper_term = (u1 - p_candidate) ** 2 - 2.0 * phi * (u1 - p_candidate) * (p_reference - l2) + (p_reference - l2) ** 2
    return {
        "difference": difference,
        "lower": max(-1.0, difference - math.sqrt(max(lower_term, 0.0))),
        "upper": min(1.0, difference + math.sqrt(max(upper_term, 0.0))),
        "n": n,
        "both": both,
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "neither": neither,
        "confidence": confidence,
    }


def paired_bootstrap_mean_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap for mean(candidate - reference) over paired episodes.

    Use paired_proportion_difference for binary outcomes instead.
    """
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1 or len(candidate) < 2:
        raise ValueError("candidate and reference must be equal-length 1-D arrays with n >= 2.")
    if not (np.all(np.isfinite(candidate)) and np.all(np.isfinite(reference))):
        raise ValueError("Bootstrap inputs must be finite.")
    if np.all(np.isin(candidate, (0.0, 1.0))) and np.all(np.isin(reference, (0.0, 1.0))):
        raise ValueError("Binary outcomes need paired_proportion_difference.")
    if not 0.0 < confidence < 1.0 or n_resamples < 1:
        raise ValueError("Require confidence in (0, 1) and n_resamples >= 1.")
    differences = candidate - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    means = differences[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {
        "difference": float(differences.mean()),
        "lower": float(np.quantile(means, 0.5 * alpha)),
        "upper": float(np.quantile(means, 1.0 - 0.5 * alpha)),
        "n": len(differences),
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
    }


def non_inferior(
    interval: dict,
    margin: float,
    higher_is_better: bool,
) -> bool:
    """Whether a candidate - reference interval rules out losses beyond margin.

    For a higher-is-better metric the lower bound must exceed -margin; for a
    lower-is-better metric the upper bound must stay below +margin.
    """
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and nonnegative.")
    if higher_is_better:
        return interval["lower"] > -margin
    return interval["upper"] < margin
