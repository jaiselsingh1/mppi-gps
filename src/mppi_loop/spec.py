"""What the policy observes and what counts as doing the task (Walker2d; Hopper builds its own)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

WALK_MAE = 0.3  # walking: mean |forward speed - 1.5| after the first 100 controls at most this


@dataclass(frozen=True)
class Task:
    observation: Callable[[np.ndarray], np.ndarray]
    observed: np.ndarray  # state indices the observation reads, in observation order
    clipped: np.ndarray  # which observation entries are clipped at +-clip
    clip: float
    admissible: Callable[[np.ndarray], np.ndarray]  # states (..., T, nstate) -> does the task
    free: tuple = (0,)  # scaled coordinates with no cost or perturbation (Walker: forward position)

    def slope_factor(self, xbar: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """d observation / d scaled coordinate along x_bar: (T, nobs), zero where clipped."""
        factor = np.broadcast_to(
            scale[..., self.observed - 1], xbar.shape[:-1] + (len(self.observed),)
        )
        held = self.clipped & (np.abs(xbar[..., self.observed]) > self.clip)
        return np.where(held, 0.0, factor)


def walker_observation(states: np.ndarray) -> np.ndarray:
    """Policy observation from the 19-value physics state: qpos[1:] and clipped qvel."""
    return np.concatenate([states[:, 2:10], np.clip(states[:, 10:19], -10.0, 10.0)], axis=1)


def walks(states) -> np.ndarray:
    healthy = (states[..., 2] > 0.8) & (states[..., 2] < 2.0) & (np.abs(states[..., 3]) < 1.0)
    return healthy.all(-1) & (np.mean(np.abs(states[..., 10] - 1.5), -1) <= WALK_MAE)


WALKER = Task(
    observation=walker_observation,
    observed=np.arange(2, 19),
    clipped=np.arange(17) >= 8,
    clip=10.0,
    admissible=walks,
)
