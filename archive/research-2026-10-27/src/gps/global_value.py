"""Grounded Monte Carlo continuation values for one frozen controller version.

Every target is a measured complete finite return.  There are no Bellman
backups, teacher-action targets or target networks here.  The caller must keep
different continuation-controller versions in separate fits.  The numerical
normalizer and data-derived output cap are NOT physical return bounds, and the
nearest-neighbour support rule is a heuristic, not an accuracy certificate.
"""
from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from torch import nn

from src.gps.expert_continuation import finite_absorbing_cost


def _count(value: Any, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _features(value: Any, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (result.ndim != 2 or not result.shape[1]
            or (dimension is not None and result.shape[1] != dimension)
            or not np.isfinite(result).all()):
        raise ValueError("features must be a finite (M,D) matrix of the fitted dimension")
    return result.copy()


def _remaining(value: Any, count: int, total_steps: int, *, allow_zero: bool) -> np.ndarray:
    result = np.asarray(value)
    if (result.shape != (count,) or result.dtype.kind not in "iu"
            or np.any(result < (0 if allow_zero else 1)) or np.any(result > total_steps)):
        raise ValueError("remaining must be aligned integer counts within the configured horizon")
    return result.astype(np.int64, copy=True)


def _normalizers(gamma: float, remaining: np.ndarray) -> np.ndarray:
    return np.asarray([finite_absorbing_cost(gamma, int(n), 5.) for n in remaining])


def _rms(values: np.ndarray) -> float:
    """Avoid squaring raw-unit returns before scaling."""
    largest = float(np.max(np.abs(values)))
    return largest * float(np.sqrt(np.mean((values / largest) ** 2))) if largest else 0.


def _mean_abs(values: np.ndarray) -> float:
    largest = float(np.max(np.abs(values)))
    return largest * float(np.mean(np.abs(values / largest))) if largest else 0.


class _ValueNetwork(nn.Module):
    def __init__(self, dimension: int, hidden: int, cap: float):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cap = float(cap)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.cap * torch.sigmoid(self.layers(features).squeeze(-1))


class MonteCarloValue:
    """Direct-return regressor, held-out group audit and local support heuristic."""

    @classmethod
    def fit(cls, features: np.ndarray, remaining: np.ndarray, costs: np.ndarray,
            groups: np.ndarray, *, total_steps: int, gamma: float, seed: int,
            updates: int = 600, hidden: int = 128, holdout_fraction: float = .25,
            support_threshold: float = 1.5,
            remaining_tolerance: int | None = None) -> "MonteCarloValue":
        total_steps = _count(total_steps, "total_steps")
        updates, hidden = _count(updates, "updates"), _count(hidden, "hidden")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite in [0,1]")
        if not np.isfinite(holdout_fraction) or not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must lie in (0,1)")
        if not np.isfinite(support_threshold) or support_threshold <= 0:
            raise ValueError("support_threshold must be positive and finite")
        if remaining_tolerance is None:
            remaining_tolerance = max(2, int(math.ceil(total_steps * .05)))
        remaining_tolerance = _count(remaining_tolerance, "remaining_tolerance", 0)
        x = _features(features)
        n = _remaining(remaining, len(x), total_steps, allow_zero=False)
        y = np.asarray(costs, dtype=np.float64)
        g = np.asarray(groups)
        if y.shape != (len(x),) or not np.isfinite(y).all() or np.any(y < 0):
            raise ValueError("costs must be finite nonnegative measured raw returns")
        if g.shape != (len(x),) or g.dtype.kind not in "iuUS":
            raise ValueError("groups must be aligned integer or string whole-query identifiers")
        unique = np.unique(g)
        if len(unique) < 4:
            raise ValueError("Require at least four whole-query groups for a disjoint group split")
        # A fixed newest-group holdout, never a split of correlated query rows.
        holdout_count = min(len(unique) - 2, max(2, int(math.ceil(len(unique) * holdout_fraction))))
        held_groups = unique[-holdout_count:]
        held = np.isin(g, held_groups)
        train = ~held
        if not train.any() or not held.any():
            raise ValueError("Require nonempty training and holdout groups")
        normalizer = _normalizers(float(gamma), n)
        targets = y / normalizer
        if not np.isfinite(targets).all():
            raise ValueError("Normalized measured costs must remain finite")
        cap = max(1., float(np.max(targets[train])) * 1.5)
        if not np.isfinite(cap) or cap > np.finfo(np.float32).max / 4:
            raise ValueError("Measured training costs exceed representable normalized range")
        raw_inputs = np.column_stack((x, n / total_steps))
        mean = raw_inputs[train].mean(axis=0)
        scale = raw_inputs[train].std(axis=0)
        scale = np.where(scale >= 1e-4, scale, 1.)
        standardized = (raw_inputs - mean) / scale
        if not np.isfinite(standardized).all():
            raise ValueError("Feature standardization exceeded finite numerical range")
        # Local RNGs plus fork_rng leave the caller's NumPy/Torch streams intact.
        rng = np.random.default_rng(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            network = _ValueNetwork(x.shape[1] + 1, hidden, cap)
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
            xt = torch.as_tensor(np.clip(standardized[train], -1e6, 1e6), dtype=torch.float32)
            yt = torch.as_tensor(targets[train], dtype=torch.float32)
            network.train()
            for _ in range(updates):
                indices = rng.choice(len(xt), size=min(256, len(xt)), replace=False)
                prediction = network(xt[indices])
                # Normalize by the train-only cap to retain finite gradients
                # when a large measured outlier determines the output range.
                loss = ((prediction / cap - yt[indices] / cap) ** 2).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite direct Monte Carlo regression loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), 10.)
                optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        network.eval().requires_grad_(False)
        result = cls()
        result.network = network
        result.feature_dimension, result.hidden = x.shape[1], hidden
        result.total_steps, result.gamma = total_steps, float(gamma)
        result.mean, result.scale = mean.copy(), scale.copy()
        result.support_features = standardized[train].copy()
        result.support_remaining = n[train].copy()
        result.output_cap = cap
        result.support_threshold = float(support_threshold)
        result.remaining_tolerance = remaining_tolerance
        predicted, supported, distances = result.predict(x[held], n[held])
        normalized_errors = predicted / normalizer[held] - targets[held]
        raw_errors = predicted - y[held]
        with torch.no_grad():
            fitted_targets = network(torch.as_tensor(
                np.clip(standardized[train], -1e6, 1e6), dtype=torch.float32)).numpy().astype(np.float64)
        # Fixed train-only baseline exposes an uninformative or overfit fit.
        # Normalize before averaging so differing remaining horizons do not
        # turn this into an accidental horizon-length predictor.
        training_mean = float(np.mean(targets[train]))
        mean_baseline_errors = training_mean - targets[held]
        finite_distances = distances[np.isfinite(distances)]
        finite = bool(np.isfinite(predicted).all() and np.isfinite(normalized_errors).all())
        normalized_rmse = _rms(normalized_errors)
        supported_fraction = float(np.mean(supported))
        result.metrics = dict(training_count=int(train.sum()), holdout_count=int(held.sum()),
            training_groups=unique[:-holdout_count].tolist(), holdout_groups=held_groups.tolist(),
            training_group_count=int(len(unique) - holdout_count), holdout_group_count=int(holdout_count),
            holdout_raw_rmse=_rms(raw_errors),
            holdout_normalized_rmse=normalized_rmse,
            holdout_normalized_mae=_mean_abs(normalized_errors),
            training_normalized_rmse=_rms(fitted_targets - targets[train]),
            holdout_normalized_bias=float(np.mean(normalized_errors)),
            normalized_training_mean=training_mean,
            holdout_training_mean_normalized_rmse=_rms(mean_baseline_errors),
            holdout_beats_training_mean=bool(normalized_rmse < _rms(mean_baseline_errors)),
            holdout_supported_fraction=supported_fraction,
            holdout_unsupported_count=int((~supported).sum()),
            holdout_distance_mean=(float(np.mean(finite_distances)) if len(finite_distances) else None),
            calibration_eligible=bool(train.sum() >= 8 and held.sum() >= 3
                                      and normalized_rmse <= .35 and supported_fraction >= .5 and finite),
            all_finite=finite, normalized_output_cap=cap,
            normalizer="finite_absorbing_cost(gamma,remaining,5); scale only, not a return bound",
            target_source="measured complete finite continuation returns; no bootstrap",
            support_rule="RMS standardized nearest training query with remaining-horizon tolerance; heuristic")
        return result

    def predict(self, features: np.ndarray, remaining: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = _features(features, self.feature_dimension)
        n = _remaining(remaining, len(x), self.total_steps, allow_zero=True)
        if not len(x):
            return np.empty(0), np.empty(0, dtype=bool), np.empty(0)
        standardized = (np.column_stack((x, n / self.total_steps)) - self.mean) / self.scale
        with torch.no_grad():
            normalized = self.network(torch.as_tensor(np.clip(standardized, -1e6, 1e6),
                                                       dtype=torch.float32)).numpy().astype(np.float64)
        costs = normalized * _normalizers(self.gamma, n)
        distances = np.full(len(x), np.inf)
        # Bound working memory when planning batches contain hundreds of plans.
        for start in range(0, len(x), 64):
            stop = min(start + 64, len(x))
            diff = standardized[start:stop, None, :] - self.support_features[None, :, :]
            squared = np.mean(diff ** 2, axis=-1)
            close_time = np.abs(n[start:stop, None] - self.support_remaining[None, :]) <= self.remaining_tolerance
            squared = np.where(close_time, squared, np.inf)
            distances[start:stop] = np.sqrt(np.min(squared, axis=1))
        supported = np.isfinite(costs) & (distances <= self.support_threshold)
        costs[n == 0], supported[n == 0], distances[n == 0] = 0., True, 0.
        return costs, supported, distances

    def packet(self) -> dict[str, Any]:
        """Independent NumPy weights avoid Torch multiprocessing shared-FD IPC."""
        return dict(format_version=1, feature_dimension=self.feature_dimension, hidden=self.hidden,
            total_steps=self.total_steps, gamma=self.gamma, mean=self.mean.copy(), scale=self.scale.copy(),
            support_features=self.support_features.copy(), support_remaining=self.support_remaining.copy(),
            output_cap=self.output_cap, support_threshold=self.support_threshold,
            remaining_tolerance=self.remaining_tolerance, metrics=copy.deepcopy(self.metrics),
            network={key: value.detach().cpu().numpy().copy() for key, value in self.network.state_dict().items()})

    to_packet = packet
    state_dict = packet

    @classmethod
    def from_packet(cls, packet: dict[str, Any]) -> "MonteCarloValue":
        if packet.get("format_version") != 1:
            raise ValueError("Unsupported MonteCarloValue packet version")
        result = cls()
        for name in ("feature_dimension", "hidden", "total_steps", "gamma", "output_cap",
                     "support_threshold", "remaining_tolerance"):
            setattr(result, name, packet[name])
        for name in ("mean", "scale", "support_features", "support_remaining"):
            setattr(result, name, np.array(packet[name], copy=True))
        result.metrics = copy.deepcopy(packet["metrics"])
        with torch.random.fork_rng(devices=[]):
            network = _ValueNetwork(result.feature_dimension + 1, result.hidden, result.output_cap)
        network.load_state_dict({key: torch.as_tensor(np.array(value, copy=True))
                                 for key, value in packet["network"].items()})
        result.network = network.eval().requires_grad_(False)
        return result
