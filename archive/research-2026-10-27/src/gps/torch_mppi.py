"""Batched Gibbs weighting and bounded MPPI synthesis on a torch device."""

import math

import torch
from jaxtyping import Bool, Float


def _check_scores(scores: torch.Tensor) -> torch.Tensor:
    """Validate scores (B, K), returning their finite-entry mask (B, K)."""
    if scores.ndim != 2 or scores.shape[1] < 1 or not scores.is_floating_point():
        raise ValueError("scores must be a floating tensor of shape (batch, samples).")
    finite = torch.isfinite(scores)
    if not torch.all(finite.any(dim=1)).item():
        raise FloatingPointError("Each state needs at least one finite trajectory score.")
    return finite


def _weights(
    shifted: torch.Tensor,
    finite: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """Normalized Gibbs weights, including zero/infinite temperature limits."""
    regular = (temperatures > 0) & torch.isfinite(temperatures)
    divisor = torch.where(regular, temperatures, torch.ones_like(temperatures))
    raw = torch.exp(-shifted / divisor[:, None])
    raw = torch.where(torch.isinf(temperatures[:, None]), finite.to(raw.dtype), raw)
    raw = torch.where(temperatures[:, None] == 0, (shifted == 0).to(raw.dtype), raw)
    return raw / raw.sum(dim=1, keepdim=True)


def softmin_weights(
    scores: Float[torch.Tensor, "batch samples"],
    temperatures: Float[torch.Tensor, "batch"],
) -> Float[torch.Tensor, "batch samples"]:
    """Shift each score row by its finite minimum and normalize its weights.

    Nonfinite scores receive zero weight. Temperature zero selects finite
    minima uniformly; infinity selects all finite samples uniformly.
    """
    finite = _check_scores(scores)
    if temperatures.shape != scores.shape[:1] or temperatures.device != scores.device:
        raise ValueError("temperatures must have shape (batch,) on the scores' device.")
    if torch.any(torch.isnan(temperatures) | (temperatures < 0)).item():
        raise ValueError("temperatures must be nonnegative and not NaN.")
    valid_scores = torch.where(finite, scores, torch.inf)
    shifted = valid_scores - valid_scores.amin(dim=1, keepdim=True)
    return _weights(shifted, finite, temperatures)


def temperature_for_neff_fraction(
    scores: Float[torch.Tensor, "batch samples"],
    target_fraction: float,
    iterations: int = 40,
) -> Float[torch.Tensor, "batch"]:
    """Solve all rows' ESS targets together by normalized log-space bisection.

    The target is ``target_fraction * K``, including invalid samples in K.
    Tied minima or invalid scores can make it unreachable: zero/infinity
    returns the nearest attainable limit. Constant finite rows use 1.0.
    Report achieved ESS rather than interpreting an unreachable target as a
    guaranteed trust bound. The returned tensor stays on the input device.
    """
    temperatures, _ = temperature_for_neff_fraction_with_details(scores, target_fraction, iterations)
    return temperatures


def temperature_for_neff_fraction_with_details(
    scores: Float[torch.Tensor, "batch samples"],
    target_fraction: float,
    iterations: int = 40,
) -> tuple[Float[torch.Tensor, "batch"], Bool[torch.Tensor, "batch"]]:
    """Return temperatures and whether the target hits a solve boundary.

    Saturation includes both initial finite bracket endpoints and degenerate
    tied/constant/invalid-sample limits. It is measured from endpoint weights,
    rather than inferred only from the final achieved ESS. No rollout occurs.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must lie strictly between 0 and 1.")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    finite = _check_scores(scores)
    valid_scores = torch.where(finite, scores, torch.inf)
    minimum = valid_scores.amin(dim=1, keepdim=True)
    shifted = valid_scores - minimum
    spread = torch.where(finite, shifted, 0.0).amax(dim=1)
    log_spread = torch.log(torch.where(spread > 0, spread, torch.ones_like(spread)))
    log_lo, log_hi = log_spread - math.log(1e8), log_spread + math.log(1e8)
    target = target_fraction * scores.shape[1]
    low_ess = _weights(shifted, finite, log_lo.exp()).square().sum(dim=1).reciprocal()
    high_ess = _weights(shifted, finite, log_hi.exp()).square().sum(dim=1).reciprocal()
    saturated = (low_ess >= target) | (high_ess <= target) | (spread == 0)
    for _ in range(iterations):
        log_mid = (log_lo + log_hi) * 0.5
        weights = _weights(shifted, finite, log_mid.exp())
        ess = weights.square().sum(dim=1).reciprocal()
        below = ess < target
        log_lo = torch.where(below, log_mid, log_lo)
        log_hi = torch.where(below, log_hi, log_mid)
    temperatures = ((log_lo + log_hi) * 0.5).exp()
    minima_count = (shifted == 0).sum(dim=1)
    temperatures = torch.where(target <= minima_count, 0.0, temperatures)
    temperatures = torch.where(target >= finite.sum(dim=1), torch.inf, temperatures)
    return torch.where(spread == 0, 1.0, temperatures), saturated


def synthesize_plan(
    nominal: Float[torch.Tensor, "batch horizon action"],
    sampled: Float[torch.Tensor, "batch samples horizon action"],
    weights: Float[torch.Tensor, "batch samples"],
    low: Float[torch.Tensor, "action"],
    high: Float[torch.Tensor, "action"],
) -> Float[torch.Tensor, "batch horizon action"]:
    """Synthesize all bounded mean plans using the effective clipped noise."""
    if nominal.ndim != 3 or sampled.ndim != 4 or sampled.shape[0] != nominal.shape[0]:
        raise ValueError("nominal and sampled must have shapes (B,H,A) and (B,K,H,A).")
    if sampled.shape[2:] != nominal.shape[1:] or weights.shape != sampled.shape[:2]:
        raise ValueError("sampled trajectories and weights must match the nominal batch.")
    if low.shape != nominal.shape[-1:] or high.shape != low.shape:
        raise ValueError("action bounds must have shape (action,).")
    perturbation = sampled - nominal[:, None]
    plan = nominal + torch.einsum("bk,bkha->bha", weights, perturbation)
    return torch.maximum(torch.minimum(plan, high), low)
