"""numerical utils """
import numpy as np 

def log_sum_exp(x: np.ndarray) -> float:
    """this is for the numerical stability associated with log(sum(exp(x)))"""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))

def compute_weights(
    costs: np.ndarray,
    lam: float,
    log_prior: np.ndarray | None = None,
    log_proposal: np.ndarray | None = None,
) -> np.ndarray:
    """Information-theoretic importance weights (Williams et al. 2018).

    log_w_k = -S_k/λ + log p(U_k) - log q(U_k)

    When log_prior and log_proposal are None, p/q cancels and
    we recover vanilla MPPI: log_w_k = -S_k/λ."""
    log_w = -costs / lam
    if log_prior is not None:
        log_w += log_prior
    if log_proposal is not None:
        log_w -= log_proposal
    log_w -= log_sum_exp(log_w)
    return np.exp(log_w)

def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = 1 / sum(w_k^2). Range [1, K]"""
    return 1.0 / np.sum(weights ** 2)

def weighted_mean_cov(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Weighted mean and covariance. samples: (K, D), weights: (K,) → mean: (D,), cov: (D, D)"""
    mu = np.einsum('k,kd->d', weights, samples)
    diff = samples - mu
    cov = np.einsum('k,ki,kj->ij', weights, diff, diff)
    return mu, cov

def gaussian_log_prob(
      x: np.ndarray, mu: np.ndarray, sigma: float
  ) -> np.ndarray:
    """Log probability under diagonal Gaussian N(mu, sigma^2 I). x: (K, H, D), mu: (H, D) → returns (K,) summed over H and D."""
    d = x.shape[-1] * x.shape[-2]  # total dimensions (H * action_dim)
    diff = x - mu[None, :, :]
    return -0.5 * (d * np.log(2 * np.pi * sigma**2)
                    + np.sum(diff**2, axis=(1, 2)) / sigma**2)







