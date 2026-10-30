"""Post-planning policy selection and frozen, weighted supervised datasets.

This module never changes a planner's objective or its proposed actions. A
selection is a snapshot of the policy and noise scale that generated its labels;
replay concatenates those snapshots without recomputing their weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch

SelectionMode = Literal["anchor", "selective"]
FloatArray = NDArray[np.float64]


def _snapshot(value: np.ndarray) -> FloatArray:
    """Copy an array as immutable float64 storage, preserving its shape."""
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SelectionDataset:
    """Frozen labels and weights at ``M`` states, for state/action sizes ``O/A``.

    ``observations`` has shape [M, O]; ``actions``, ``proposed_actions`` and
    ``old_actions`` have shape [M, A]. ``weights``, ``improvement`` and
    ``action_delta`` have shape [M]. ``action_delta`` measures the *raw*
    proposal's Euclidean distance from the policy before any anchoring.
    """

    observations: FloatArray
    actions: FloatArray
    weights: FloatArray
    proposed_actions: FloatArray
    old_actions: FloatArray
    improvement: FloatArray
    action_delta: FloatArray


def build_selection(
    observations: np.ndarray,
    proposals: np.ndarray,
    old_actions: np.ndarray,
    nominal_cost: np.ndarray,
    plan_cost: np.ndarray,
    sigma: float,
    mode: SelectionMode = "anchor",
    tau_s: float = 1.0,
) -> SelectionDataset:
    """Build labels after MPPI, using the policy and sigma of this iteration.

    Arguments have shapes ``observations[M, O]``, ``proposals[M, A]``,
    ``old_actions[M, A]`` and ``nominal_cost[M]``/``plan_cost[M]``. Both costs
    must score the same task-plus-tail objective. All arrays must be finite;
    corrupt E-step results raise instead of silently becoming training labels.

    ``anchor`` replaces strictly worse proposals with the old policy action,
    retaining unit weights (equal-cost proposals are retained). ``selective``
    assigns zero weight unless the plan strictly improves, then uses
    tanh(improvement / tau_s) * exp(-||proposal-old_action||² / (2 sigma²)).
    Inputs are copied; the returned snapshot cannot be modified in place.
    """
    if mode not in {"anchor", "selective"}:
        raise ValueError("mode must be 'anchor' or 'selective'.")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive.")
    if not np.isfinite(tau_s) or tau_s <= 0:
        raise ValueError("tau_s must be finite and positive.")

    obs = np.asarray(observations, dtype=np.float64)
    proposed = np.asarray(proposals, dtype=np.float64)
    old = np.asarray(old_actions, dtype=np.float64)
    nominal = np.asarray(nominal_cost, dtype=np.float64)
    planned = np.asarray(plan_cost, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] == 0:
        raise ValueError("observations must have shape [M, O] with O > 0.")
    if proposed.ndim != 2 or proposed.shape[0] != len(obs) or proposed.shape[1] == 0:
        raise ValueError("proposals must have shape [M, A] with A > 0.")
    if old.shape != proposed.shape:
        raise ValueError("old_actions must have the same [M, A] shape as proposals.")
    if nominal.shape != (len(obs),) or planned.shape != (len(obs),):
        raise ValueError("nominal_cost and plan_cost must have shape [M].")
    for name, value in (("observations", obs), ("proposals", proposed),
                        ("old_actions", old), ("nominal_cost", nominal), ("plan_cost", planned)):
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values.")

    with np.errstate(over="ignore", invalid="ignore"):
        improvement = nominal - planned
        difference = proposed - old
        action_delta = np.linalg.norm(difference, axis=1)
    if not np.isfinite(improvement).all() or not np.isfinite(action_delta).all():
        raise ValueError("cost improvement and action distance must remain finite.")
    actions = proposed.copy()
    if mode == "anchor":
        actions[planned > nominal] = old[planned > nominal]
        weights = np.ones(len(obs), dtype=np.float64)
    else:
        weights = np.zeros(len(obs), dtype=np.float64)
        improving = improvement > 0
        # Scaling before squaring handles tiny sigma without dividing by an
        # underflowed sigma². Huge distances and advantages saturate safely.
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            scaled_distance = action_delta[improving] / sigma
            proximity = np.exp(-0.5 * np.square(scaled_distance))
            weights[improving] = np.tanh(improvement[improving] / tau_s) * proximity
    return SelectionDataset(*(_snapshot(value) for value in
                              (obs, actions, weights, proposed, old, improvement, action_delta)))


def recent_selection(
    iterations: Sequence[SelectionDataset], window: int = 2,
) -> SelectionDataset:
    """Concatenate the last ``window`` snapshots, preserving their weights.

    State and action dimensions must match across snapshots. The result has
    shapes [sum(M), O], [sum(M), A] and [sum(M)] as in ``SelectionDataset``.
    """
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer.")
    if not iterations:
        raise ValueError("replay needs at least one selection dataset.")
    recent = iterations[-window:]
    dimensions = {(item.observations.shape[1], item.actions.shape[1]) for item in recent}
    if len(dimensions) != 1:
        raise ValueError("replay state and action dimensions must match.")
    fields = ("observations", "actions", "weights", "proposed_actions", "old_actions",
              "improvement", "action_delta")
    return SelectionDataset(*(_snapshot(np.concatenate([getattr(item, field) for item in recent]))
                              for field in fields))


def weighted_mse(
    prediction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    """Return sum_i w_i ||prediction_i-target_i||² / sum_i w_i.

    ``prediction`` and ``target`` are floating tensors [B, A]; ``weights``
    is a finite nonnegative floating tensor [B] on the same device. Empty or
    zero-weight batches return differentiable, finite zero. This uses the
    squared action norm, not a mean over action components.
    """
    if prediction.ndim != 2 or prediction.shape[1] == 0 or target.shape != prediction.shape:
        raise ValueError("prediction and target must share shape [B, A] with A > 0.")
    if weights.shape != (len(prediction),):
        raise ValueError("weights must have shape [B].")
    if any(not value.is_floating_point() for value in (prediction, target, weights)):
        raise ValueError("prediction, target and weights must be floating tensors.")
    if target.device != prediction.device or weights.device != prediction.device:
        raise ValueError("prediction, target and weights must share a device.")
    if not all(bool(torch.isfinite(value).all()) for value in (prediction, target, weights)):
        raise ValueError("prediction, target and weights must be finite.")
    if bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative.")
    positive = weights > 0
    if not bool(positive.any()):
        return prediction[positive].sum()
    # Remove zero-weight rows before computing errors; they cannot affect the
    # objective even when their squared errors would overflow the tensor type.
    error = (prediction[positive] - target[positive]).square().sum(dim=1)
    selected_weights = weights[positive]
    return torch.sum(selected_weights * error) / selected_weights.sum()
