"""Trust field for trust-priced consensus (docs/trust_priced_consensus.md).

Trust is measured, not predicted. The deterministic policy runs alone from the
exact simulator state and solver warm start for `window` controls. The state is
trusted if the policy survives and its mean speed error is at most
`max_speed_error`; pass `None` to require survival only.

TrustField turns those labels into tau_hat(s): the Wilson lower bound of the
trusted fraction among the k nearest probed states, in observation space
standardized by the probes. A state farther from every probe than the probes'
`familiar_quantile` nearest-neighbour distance gets 0, because the policy has
not shown it can handle it.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from src.envs.walker2d import Walker2d


@dataclass(frozen=True)
class Probe:
    survived: bool
    steps: int
    speed_error: float


def probe(env: Walker2d, act: Callable[[np.ndarray], np.ndarray], state: np.ndarray,
          warmstart: np.ndarray, window: int) -> Probe:
    """The policy alone from (state, warmstart) for `window` controls.

    speed_error is the mean |v_target - vx| over the controls taken.
    """
    if window < 1:
        raise ValueError("window must be positive.")
    obs = env.reset(state=state)
    env.set_warmstart(warmstart)
    low, high = env.action_bounds
    errors = []
    for _ in range(window):
        obs, _, done, info = env.step(np.clip(act(obs), low, high))
        errors.append(info["velocity_error"])
        if done:
            return Probe(False, len(errors), float(np.mean(errors)))
    return Probe(True, window, float(np.mean(errors)))


def is_trusted(result: Probe, max_speed_error: float | None) -> bool:
    return result.survived and (max_speed_error is None or result.speed_error <= max_speed_error)


def wilson_lower(successes: np.ndarray, n: int, z: float = 1.0) -> np.ndarray:
    """Wilson score lower bound of a binomial proportion."""
    p = np.asarray(successes, dtype=float) / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return np.clip((centre - spread) / (1 + z * z / n), 0.0, 1.0)


class TrustField:
    def __init__(self, obs: np.ndarray, labels: np.ndarray, k: int = 20, z: float = 1.0,
                 familiar_quantile: float = 0.95) -> None:
        obs = np.asarray(obs, dtype=float)
        labels = np.asarray(labels, dtype=bool)
        if obs.ndim != 2 or len(obs) != len(labels) or len(obs) <= k or not np.isfinite(obs).all():
            raise ValueError("TrustField needs finite (n, obs_dim) probes with n > k and one label each.")
        self.mean = obs.mean(axis=0)
        self.std = np.maximum(obs.std(axis=0), 1e-8)
        self.labels = labels
        self.k, self.z = k, z
        self.tree = cKDTree((obs - self.mean) / self.std)
        nearest, _ = self.tree.query(self.tree.data, k=2)
        self.max_distance = float(np.quantile(nearest[:, 1], familiar_quantile))

    def __call__(self, obs: np.ndarray) -> np.ndarray | float:
        """tau_hat for one observation (17,) or a batch (n, 17)."""
        batch = (np.atleast_2d(obs) - self.mean) / self.std
        distance, index = self.tree.query(batch, k=self.k)
        tau = wilson_lower(self.labels[index].sum(axis=1), self.k, self.z)
        tau = np.where(distance[:, 0] > self.max_distance, 0.0, tau)
        return tau if np.ndim(obs) > 1 else float(tau[0])
