"""Pure saved-array bank metrics preserved from the exploratory action spectrum.

These functions retain the old estimator, operation order and RNG consumption.
No simulator, policy, fitting, filtering or scientific command is introduced.
New uses must declare their rows/normalization/geometry and interpretation;
these scores alone do not establish learning or walking.
"""

from __future__ import annotations

import numpy as np

FIRST = 100
SEPARATION = 100
K = 10


def neighbour_index(states: list[np.ndarray]) -> np.ndarray:
    """Legacy KNN row lookup with unchanged pooling, exclusion and tie handling.

    The caller supplies per-trajectory feature matrices in the recipe's chosen
    normalization and time range. FIRST is a caller-side crop convention; this
    function does not crop again. Rows from the same trajectory fewer than
    SEPARATION controls away are excluded. The original 1,000-row blocking and
    argpartition/argsort operation order are retained exactly.

    Require enough eligible rows for K=10 neighbours; this legacy helper adds
    no new validation, fallback, normalization or outcome interpretation.
    """
    pooled = np.concatenate(states)
    walk = np.concatenate([np.full(len(s), k) for k, s in enumerate(states)])
    step = np.concatenate([np.arange(len(s)) for s in states])
    square = (pooled**2).sum(1)
    index = np.empty((len(pooled), K), dtype=int)
    for start in range(0, len(pooled), 1000):
        rows = slice(start, min(start + 1000, len(pooled)))
        d = square[rows, None] + square[None, :] - 2.0 * pooled[rows] @ pooled.T
        near = (walk[rows, None] == walk[None, :]) & (
            np.abs(step[rows, None] - step[None, :]) < SEPARATION
        )
        d[near] = np.inf
        part = np.argpartition(d, K, axis=1)[:, :K]
        order = np.take_along_axis(d, part, 1).argsort(1)
        index[rows] = np.take_along_axis(part, order, 1)
    return index


def state_dependence(
    y: np.ndarray, index: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Legacy neighbour-mean estimator and random-pair comparison, unchanged.

    `y` and `index` must refer to the same pooled row order. The caller owns the
    private generator. The single (2, 200_000) integer draw and all reductions
    stay in their original order, including division behavior at zero variance.
    NaN or infinity on degenerate input is not replaced by a favorable score.

    The KNN R² depends on sampling density, feature geometry and temporal
    exclusions. It is not an intrinsic-noise estimate, a teachability test, or
    evidence that the unexplained actions are unimportant for control.
    """
    predicted = y[index].mean(axis=1)
    residual = ((y - predicted) ** 2).sum()
    total = ((y - y.mean(0)) ** 2).sum()
    a, b = rng.integers(0, len(y), (2, 200_000))
    nearest = index[:, 0]
    return dict(
        r2=float(1.0 - residual / total),
        variance=float(y.var(axis=0).sum()),
        mean_abs=float(np.abs(y).mean()),
        neighbour_over_random=float(np.mean((y - y[nearest]) ** 2) / np.mean((y[a] - y[b]) ** 2)),
    )
