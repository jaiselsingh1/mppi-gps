"""Credit estimators: what each learns from complete outcomes, and how it ranks MPPI's options.

All share one linear tile-coded representation and an unpenalized intercept,
so unvisited states fall back to the data mean rather than to an optimistic
zero cost. They differ only in supervision:

- MC: every-visit regression of realized cost-to-go on the state.
- TD(lambda): LSTD(lambda), the least-squares fixed point of lambda-returns
  built from per-step costs and its own bootstrapped values (GAE's critic
  target); lambda = 1 is MC.
- Paired option scorer: only differences between complete outcomes of two of
  MPPI's options from the same recorded state under shared randomness (the
  executed continuation and a branch). It learns relative values of the state
  reached at the option's horizon end; common noise cancels.
- Return decomposition (RRD-style least squares): only each complete
  episode's total. Per-step credits r(s) are fitted so they sum to the total;
  options are scored by their in-horizon credit, with no terminal value, so it
  tests whether redistributed credit can carry the ending inside the horizon.

Value estimators score an option as its model cost plus the value at its
horizon end (zero at episode end); they can also serve as MPPI's terminal value.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import scipy.sparse as sp
from scipy.signal import lfilter

from src.credit.control import Options


class TileCoder:
    """Binary tile coding over a box, with asymmetric tiling offsets."""

    def __init__(self, lows, highs, bins, tilings: int):
        self.lows = np.asarray(lows, dtype=float)
        self.highs = np.asarray(highs, dtype=float)
        self.bins = np.asarray(bins, dtype=int)
        self.tilings = int(tilings)
        self.dim = len(self.lows)
        self.width = (self.highs - self.lows) / self.bins
        shape = self.bins + 1
        self.per_tiling = int(np.prod(shape))
        self.n_features = self.tilings * self.per_tiling
        self._strides = np.cumprod(np.concatenate([[1], shape[::-1][:-1]]))[::-1]
        odd = 2 * np.arange(self.dim) + 1
        self._offsets = (np.arange(self.tilings)[:, None] * odd[None, :] % self.tilings) / self.tilings

    def active(self, x: np.ndarray) -> np.ndarray:
        """Indices of active features, shape (..., tilings)."""
        x = np.asarray(x, dtype=float)[..., : self.dim]
        u = np.clip((x - self.lows) / self.width, 0.0, self.bins - 1e-9)
        cells = np.floor(u[..., None, :] + self._offsets).astype(int)
        flat = (cells * self._strides).sum(axis=-1)
        return flat + np.arange(self.tilings) * self.per_tiling

    def matrix(self, x: np.ndarray) -> sp.csr_matrix:
        idx = self.active(np.asarray(x).reshape(-1, np.asarray(x).shape[-1]))
        n = idx.shape[0]
        rows = np.repeat(np.arange(n), self.tilings)
        data = np.ones(n * self.tilings)
        return sp.csr_matrix((data, (rows, idx.reshape(-1))), shape=(n, self.n_features))


@dataclass
class TaskView:
    """What estimators need to know about a task's states."""

    coder: TileCoder
    terminal: Callable[[np.ndarray], np.ndarray]  # bool mask: episode over at this state
    active_step: Callable[[np.ndarray], np.ndarray]  # bool mask: transition into this state was charged


@dataclass
class TrajectoryData:
    """One complete continuation: states s_0..s_n, realized costs c_0..c_{n-1}."""

    states: np.ndarray
    costs: np.ndarray
    truncated: bool  # ended by a time cap rather than by the task


class LinearValue:
    """V(x) = b + w . phi(x), zero at terminal states."""

    def __init__(self, view: TaskView, w: np.ndarray, b: float):
        self.view, self.w, self.b = view, w, float(b)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        idx = self.view.coder.active(states)
        value = self.b + self.w[idx].sum(axis=-1)
        return np.where(self.view.terminal(states), 0.0, value)


def _ridge(phi: sp.csr_matrix, y: np.ndarray, ridge: float, intercept: bool, weights=None) -> tuple[np.ndarray, float]:
    """Minimize sum wt (y - b - phi w)^2 + ridge |w|^2 (b unpenalized)."""
    n, f = phi.shape
    wt = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if intercept:
        mean_y = np.sum(wt * y) / np.sum(wt)
        mean_phi = np.asarray(phi.T @ wt).ravel() / np.sum(wt)
    else:
        mean_y, mean_phi = 0.0, np.zeros(f)
    pw = phi.multiply(wt[:, None]).tocsr()
    gram = np.asarray((phi.T @ pw).todense())
    rhs = np.asarray(phi.T @ (wt * y)).ravel()
    total = np.sum(wt)
    gram -= total * np.outer(mean_phi, mean_phi)
    rhs -= total * mean_phi * mean_y
    gram[np.diag_indices(f)] += ridge
    w = np.linalg.solve(gram, rhs)
    return w, float(mean_y - mean_phi @ w)


def fit_mc(view: TaskView, data: list[TrajectoryData], ridge: float = 1.0) -> LinearValue:
    states = np.concatenate([d.states[:-1] for d in data])
    targets = np.concatenate([np.cumsum(d.costs[::-1])[::-1] for d in data])
    w, b = _ridge(view.coder.matrix(states), targets, ridge, intercept=True)
    return LinearValue(view, w, b)


def fit_lstd(view: TaskView, data: list[TrajectoryData], lam: float, ridge: float = 1.0) -> LinearValue:
    """LSTD(lambda), undiscounted, with an intercept feature.

    Forward view: A = sum_s phi_s y_s^T with y_s = sum_{t>=s} lam^{t-s}(phi_t - phi_{t+1}),
    b = sum_s phi_s sum_{t>=s} lam^{t-s} c_t; phi after a true episode end is zero,
    after a time cap it is the state's own features (bootstrap).
    """
    f = view.coder.n_features + 1
    A = np.zeros((f, f))
    rhs = np.zeros(f)
    for d in data:
        n = len(d.costs)
        phi = sp.hstack([sp.csr_matrix(np.ones((n + 1, 1))), view.coder.matrix(d.states)]).tocsr()
        if not d.truncated:
            phi = sp.vstack([phi[:n], sp.csr_matrix((1, f))]).tocsr()
        diff = (phi[:n] - phi[1:]).toarray()
        # Backward lambda-discounted sums: y_s = diff_s + lam * y_{s+1}.
        y = lfilter([1.0], [1.0, -lam], diff[::-1], axis=0)[::-1]
        g = lfilter([1.0], [1.0, -lam], d.costs[::-1])[::-1]
        A += np.asarray(phi[:n].T @ y)
        rhs += np.asarray(phi[:n].T @ g).ravel()
    reg = np.full(f, ridge)
    reg[0] = 1e-9
    A[np.diag_indices(f)] += reg
    sol = np.linalg.solve(A, rhs)
    return LinearValue(view, sol[1:], sol[0])


@dataclass
class Pair:
    """Two of MPPI's options from one recorded state under shared randomness."""

    end_a: np.ndarray  # state after H steps (or at episode end)
    end_b: np.ndarray
    rest_a: float  # realized outcome from the decision minus the model's in-horizon cost
    rest_b: float


def fit_paired(view: TaskView, pairs: list[Pair], ridge: float = 1.0) -> LinearValue:
    """Relative value at horizon ends from outcome differences (no intercept)."""
    usable = [p for p in pairs if not (view.terminal(p.end_a[None])[0] and view.terminal(p.end_b[None])[0])]
    if not usable:
        return LinearValue(view, np.zeros(view.coder.n_features), 0.0)
    ends_a = np.stack([p.end_a for p in usable])
    ends_b = np.stack([p.end_b for p in usable])
    mask_a = ~view.terminal(ends_a)
    mask_b = ~view.terminal(ends_b)
    phi = view.coder.matrix(ends_a).multiply(mask_a[:, None]) - view.coder.matrix(ends_b).multiply(mask_b[:, None])
    y = np.array([p.rest_a - p.rest_b for p in usable])
    w, _ = _ridge(sp.csr_matrix(phi), y, ridge, intercept=False)
    return LinearValue(view, w, 0.0)


class StepCredit:
    """Per-step credit r(s) = b + w . psi(s) for charged transitions into s."""

    def __init__(self, view: TaskView, w: np.ndarray, b: float):
        self.view, self.w, self.b = view, w, float(b)

    def __call__(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        idx = self.view.coder.active(states)
        return np.where(self.view.active_step(states), self.b + self.w[idx].sum(axis=-1), 0.0)


def fit_return_decomposition(view: TaskView, episodes: list[TrajectoryData], ridge: float = 1.0) -> StepCredit:
    """Fit sum_t r(s_{t+1}) to each episode's total cost, from totals alone.

    total = b * (charged steps) + w . (summed features); b is unpenalized and
    profiled out, w is ridge-penalized (minimum-norm when underdetermined).
    """
    rows, counts, totals = [], [], []
    for d in episodes:
        post = d.states[1:]
        live = view.active_step(post)
        rows.append(np.asarray(view.coder.matrix(post[live]).sum(axis=0)).ravel())
        counts.append(live.sum())
        totals.append(d.costs.sum())
    psi = np.stack(rows)
    c = np.asarray(counts, dtype=float)
    y = np.asarray(totals)
    psi_c = psi - np.outer(c, c @ psi) / (c @ c)
    y_c = y - c * (c @ y) / (c @ c)
    gram = psi_c.T @ psi_c
    gram[np.diag_indices(gram.shape[0])] += ridge
    w = np.linalg.solve(gram, psi_c.T @ y_c)
    b = float(c @ (y - psi @ w) / (c @ c))
    return StepCredit(view, w, b)


def value_scorer(value: Callable[[np.ndarray], np.ndarray]):
    """Score options by model cost plus value at the horizon end."""

    def score(state: np.ndarray, options: Options) -> np.ndarray:
        return options.model_cost + value(options.endpoints)

    return score


def credit_scorer(credit: StepCredit):
    """Score options by summed in-horizon credit only (return decomposition)."""

    def score(state: np.ndarray, options: Options) -> np.ndarray:
        return credit(options.states).sum(axis=1)

    return score


def greedy_scorer():
    def score(state: np.ndarray, options: Options) -> np.ndarray:
        return options.model_cost

    return score


def random_scorer(seed: int):
    rng = np.random.default_rng(seed)

    def score(state: np.ndarray, options: Options) -> np.ndarray:
        return rng.random(len(options.model_cost))

    return score
