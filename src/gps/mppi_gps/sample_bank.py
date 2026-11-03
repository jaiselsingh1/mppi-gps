"""Saved candidate-bank utilities, extracted from the SAC branch prototypes.

This is finite-bank candidate reweighting, not a teachability estimator. It uses
the policy's present actions; no learning update or new MPPI search occurs.
The original arithmetic and dtype conversions are retained. In particular the
stored clipped prefixes are float16, and the Gaussian correction uses pre-clip
first actions. That correction does not prove exact importance sampling for the
clipped, temporally correlated, truncated bank.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


def policy_observations(states: NDArray[Any]) -> NDArray[Any]:
    """Historical 19-value physics-state to 17-value clipped policy interface."""
    import numpy as np

    return np.concatenate([states[:, 2:10], np.clip(states[:, 10:19], -10.0, 10.0)], axis=1)


def candidate_labels(
    episode: dict[str, Any],
    policy_actions: NDArray[Any],
    bandwidth: float,
    temperature: float,
    sampling_sd: float = 0.3,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Reproduce policy_picks' labels and effective sample sizes for one episode.

    Callers supply already-recorded policy actions. This function never loads a
    checkpoint, predicts actions, fits a policy or calls a simulator. Empty or
    malformed data fail rather than silently becoming useful teaching evidence.
    """
    import numpy as np

    if not all(
        math.isfinite(value) and value > 0 for value in (bandwidth, temperature, sampling_sd)
    ):
        raise ValueError("Kernel scales must be finite and positive")
    prefix = episode["prefix"]
    if prefix.ndim != 4 or any(size == 0 for size in prefix.shape):
        raise ValueError("Candidate prefixes must have nonempty row, sample, time and action axes")
    rows, samples, _, dimensions = prefix.shape
    expected_shapes = {
        "raw_first": (rows, samples, dimensions),
        "nominal_first": (rows, dimensions),
        "costs": (rows, samples),
    }
    if any(episode[name].shape != shape for name, shape in expected_shapes.items()):
        raise ValueError("Candidate costs and sampling context must match prefix rows")
    if policy_actions.shape != (rows, dimensions):
        raise ValueError("Policy actions must match candidate rows and action dimensions")
    if not all(
        np.isfinite(values).all()
        for values in (prefix, policy_actions, *(episode[name] for name in expected_shapes))
    ):
        raise ValueError("Candidate arrays and policy actions must be finite")
    actions = prefix[:, :, 0].astype(np.float64)
    drift = episode["raw_first"] - episode["nominal_first"][:, None]
    cost = episode["costs"] - episode["costs"].min(axis=1, keepdims=True)
    log_weights = (
        -cost / temperature
        - ((actions - policy_actions[:, None]) ** 2).sum(-1) / (2 * bandwidth**2)
        + (drift**2).sum(-1) / (2 * sampling_sd**2)
    )
    weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("tk,tka->ta", weights, actions), 1.0 / (weights**2).sum(axis=1)
