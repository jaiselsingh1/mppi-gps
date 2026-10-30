"""Numerical utilities."""

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = 1 / sum(w_k^2). Range [1, K]."""
    return 1.0 / np.sum(weights**2)


def softmin_weights(scores: np.ndarray, lam: float) -> np.ndarray:
    """Normalized Gibbs weights; nonfinite scores get zero weight.

    Zero and infinite temperatures use their limits (uniform over finite
    minima and all finite samples, respectively), including the boundary
    temperatures returned by :func:`temperature_for_budget`.
    """
    if np.isnan(lam) or lam < 0.0:
        raise ValueError("softmin temperature must be nonnegative and not NaN.")
    finite = np.isfinite(scores)
    if not finite.any():
        raise FloatingPointError("No finite scores to weight.")
    unnorm = np.zeros_like(scores, dtype=float)
    if np.isinf(lam):
        unnorm[finite] = 1.0
    elif lam == 0.0:
        unnorm[finite] = scores[finite] == np.min(scores[finite])
    else:
        unnorm[finite] = np.exp(-(scores[finite] - np.min(scores[finite])) / lam)
    return unnorm / np.sum(unnorm)


def plug_in_kl(weights: np.ndarray) -> float:
    """KL(sum_k w_k delta_k || uniform over K samples) = log K - H(w)."""
    positive = weights[weights > 0.0]
    return float(np.log(len(weights)) + np.sum(positive * np.log(positive)))


def temperature_for_neff_fraction(
    scores: np.ndarray,
    target_fraction: float,
    iterations: int = 60,
) -> float:
    """Temperature whose softmin weights have n_eff close to target_fraction * K.

    n_eff is non-decreasing in the temperature, so this bisects in log space.
    Fixing n_eff/K = rho bounds the plug-in KL of the step by log(1/rho)
    (docs/objective.md, fact 3). If the target is unreachable, the nearest
    bracket end is returned.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must lie strictly between 0 and 1.")
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        raise FloatingPointError("No finite scores to set a temperature from.")
    spread = float(np.max(finite) - np.min(finite))
    if spread == 0.0:
        return 1.0
    target = target_fraction * len(scores)
    log_lo, log_hi = np.log(spread * 1e-8), np.log(spread * 1e8)
    for _ in range(iterations):
        log_mid = 0.5 * (log_lo + log_hi)
        if effective_sample_size(softmin_weights(scores, np.exp(log_mid))) < target:
            log_lo = log_mid
        else:
            log_hi = log_mid
    return float(np.exp(0.5 * (log_lo + log_hi)))


def temperature_for_budget(
    scores: np.ndarray,
    budget: float,
    iterations: int = 60,
) -> float:
    """Temperature whose softmin-weighted mean score equals `budget`.

    The weighted mean is non-decreasing in the temperature, so this bisects in
    log space. Returns inf when the plain mean already meets the budget (no
    step is needed: full trust, docs/theory.md Prop. 2), and the smallest
    bracketed temperature when the budget lies below every sample.
    """
    finite = np.isfinite(scores)
    if not finite.any():
        raise FloatingPointError("No finite scores to set a temperature from.")
    if np.mean(scores[finite]) <= budget:
        return float("inf")
    spread = float(np.max(scores[finite]) - np.min(scores[finite]))
    if spread == 0.0:
        return 0.0
    log_lo, log_hi = np.log(spread * 1e-8), np.log(spread * 1e8)
    for _ in range(iterations):
        log_mid = 0.5 * (log_lo + log_hi)
        weights = softmin_weights(scores, np.exp(log_mid))
        if np.sum(weights[finite] * scores[finite]) > budget:
            log_hi = log_mid
        else:
            log_lo = log_mid
    return float(np.exp(0.5 * (log_lo + log_hi)))
