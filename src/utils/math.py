"""Numerical utilities."""
import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = 1 / sum(w_k^2). Range [1, K]."""
    return 1.0 / np.sum(weights ** 2)
