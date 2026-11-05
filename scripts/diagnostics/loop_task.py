"""What the similarity loop needs to know about a task, so its search and fits run on more than Walker2d.

A state is MuJoCo's full-physics vector (time first); scaled coordinates are
the state without time, divided by a per-coordinate sd. The policy sees
observation(states), which reads the state entries listed in observed (in
order), optionally clipping some of them.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.diagnostics.coupled_mppi import WALK_MAE
from scripts.diagnostics.noise_cutoff import observation as walker_observation


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
