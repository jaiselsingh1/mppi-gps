"""Supervised finite-horizon cost of an independent MPPI continuation oracle.

This is V_MPPI(obs, remaining), not V_pi or a Bellman/hybrid target. Labels
must already contain the oracle's complete finite remaining-horizon cost,
including its single fall charge and absorbing suffix. This module performs
no simulation and never updates an actor. Holdouts are separated by reset ID.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F

OBS_DIM = 17
EXPERT_SOURCE = "independent_mppi_continuation"
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    """Validate a Python integer configuration without accepting booleans."""
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _vector(value: np.ndarray, count: int, name: str, *, integer: bool = False) -> np.ndarray:
    """Copy a finite aligned (N,) vector into immutable numeric storage."""
    array = np.asarray(value)
    if (array.shape != (count,) or not np.isfinite(array).all()
            or (integer and not np.issubdtype(array.dtype, np.integer)) or np.any(array < 0)):
        raise ValueError(f"{name} must be an aligned finite nonnegative {'integer ' if integer else ''}(N,) vector")
    array = np.array(array, dtype=np.int64 if integer else np.float64, copy=True)
    if np.any(array < 0):
        raise ValueError(f"{name} exceeds the supported integer range")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ExpertValueData:
    """Realized oracle labels: observations (N,17), costs/remaining/reset_ids (N,).

    ``source`` is required deliberately: policy/accepted-plan/bootstrapped
    labels cannot be silently treated as independent-teacher outcomes.
    ``remaining`` counts controls, not a normalized time or an infinite tail.
    Optional provenance may include ``expert_id`` to bind fits to one oracle.
    """

    observations: FloatArray
    cost_targets: FloatArray
    remaining: IntArray
    reset_ids: IntArray
    source: str
    provenance: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate source, complete finite labels, shapes, and zero-horizon costs."""
        if self.source != EXPERT_SOURCE:
            raise ValueError("Expert value requires actual independent_mppi_continuation labels, not policy/TD/hybrid data")
        obs = np.array(self.observations, dtype=np.float64, copy=True)
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM or not len(obs) or not np.isfinite(obs).all():
            raise ValueError("observations must be finite nonempty (N,17)")
        obs.setflags(write=False)
        values = {"cost_targets": _vector(self.cost_targets, len(obs), "cost_targets"),
                  "remaining": _vector(self.remaining, len(obs), "remaining", integer=True),
                  "reset_ids": _vector(self.reset_ids, len(obs), "reset_ids", integer=True)}
        if np.any(values["cost_targets"][values["remaining"] == 0] != 0):
            raise ValueError("Zero remaining horizon must have exact zero cost")
        metadata = {} if self.provenance is None else dict(self.provenance)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
            raise ValueError("Provenance keys and values must be strings")
        object.__setattr__(self, "observations", obs)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "provenance", MappingProxyType(metadata))


class ExpertCostValue(nn.Module):
    """18 -> 256 -> 256 -> softplus(1), multiplied by observed cost scale."""

    def __init__(self, *, horizon: int = 1000, hidden_dim: int = 256, seed: int = 0) -> None:
        """Use 17 raw observation features plus remaining/horizon as feature 18."""
        super().__init__()
        self.horizon = _integer(horizon, "horizon")
        self.hidden_dim = _integer(hidden_dim, "hidden_dim")
        _integer(seed, "seed", minimum=0)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.fc1 = nn.Linear(OBS_DIM + 1, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, 1)
        self.register_buffer("cost_scale", torch.tensor(1., dtype=torch.float32))

    def forward(self, observations: torch.Tensor, remaining: torch.Tensor | int) -> torch.Tensor:
        """Map observations (N,17) and integer remaining (N,) to costs (N,)."""
        if observations.ndim != 2 or observations.shape[1] != OBS_DIM or not torch.isfinite(observations).all():
            raise ValueError("Value observations must be finite (N,17)")
        left = torch.as_tensor(remaining, device=observations.device)
        if left.ndim == 0:
            left = left.expand(len(observations))
        if (left.shape != observations.shape[:1] or left.dtype == torch.bool or not torch.isfinite(left).all()
                or ((left < 0) | (left > self.horizon) | (left != left.round())).any()):
            raise ValueError("remaining must be integer controls in [0,horizon], shaped (N,) or scalar")
        obs = observations.to(dtype=self.fc1.weight.dtype)
        inputs = torch.cat([obs, left.to(dtype=obs.dtype)[:, None] / self.horizon], dim=1)
        hidden = torch.tanh(self.fc2(torch.tanh(self.fc1(inputs))))
        costs = self.cost_scale * F.softplus(self.head(hidden)).squeeze(-1)
        return torch.where(left == 0, torch.zeros_like(costs), costs)

    @torch.no_grad()
    def initialize_cost_scale(self, mean_realized_cost: float) -> None:
        """Ground the scale and constant initial prediction in measured cost units."""
        if not np.isfinite(mean_realized_cost) or mean_realized_cost < 0:
            raise ValueError("Require a finite nonnegative mean realized expert cost")
        scale = max(float(mean_realized_cost), 1e-6)
        self.cost_scale.fill_(scale)
        self.head.weight.zero_()
        normalized_mean = max(float(mean_realized_cost) / scale, 1e-6)
        self.head.bias.fill_(float(np.log(np.expm1(normalized_mean))))


def _predict(model: ExpertCostValue, observations: np.ndarray, remaining: np.ndarray | int,
             *, grounded: bool) -> FloatArray:
    """Checked NumPy scoring; a zero-horizon batch never needs a grounded model."""
    obs = np.array(observations, dtype=np.float64, copy=True)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM or not np.isfinite(obs).all():
        raise ValueError("Predict observations must be finite (N,17)")
    left = np.asarray(remaining)
    if left.ndim == 0:
        left = np.full(len(obs), left.item())
    left = _vector(left, len(obs), "remaining", integer=True)
    if np.any(left > model.horizon):
        raise ValueError("remaining exceeds configured finite horizon")
    if not np.any(left):
        return np.zeros(len(obs), dtype=np.float64)
    if not grounded:
        raise RuntimeError("Expert value has insufficient grounded independent-teacher data")
    device = model.cost_scale.device
    with torch.no_grad():
        values = model(torch.tensor(obs, dtype=torch.float32, device=device),
                       torch.tensor(left, dtype=torch.int64, device=device)).cpu().numpy().astype(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise FloatingPointError("Expert value produced invalid costs")
    return values


@dataclass(frozen=True)
class ExpertValueSnapshot:
    """Independent parameter copy for fixed scoring during parameter search."""

    model: ExpertCostValue
    grounded: bool
    expert_id: str | None

    def predict(self, observations: np.ndarray, remaining: np.ndarray | int) -> FloatArray:
        """Return finite costs (N,) without gradients or simulation."""
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError("Expert scoring snapshot must remain frozen")
        return _predict(self.model, observations, remaining, grounded=self.grounded)


class ExpertValueTrainer:
    """Seeded supervised fitting to complete, realized finite MPPI continuations."""

    def __init__(self, *, horizon: int = 1000, hidden_dim: int = 256, lr: float = 3e-4,
                 seed: int = 0, device: str | torch.device = "cpu") -> None:
        """Initialize under fork_rng while leaving caller CPU/CUDA RNG unchanged."""
        _integer(seed, "seed", minimum=0)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive and finite")
        self.device, self.lr, self.seed = torch.device(device), float(lr), seed
        self.model = ExpertCostValue(horizon=horizon, hidden_dim=hidden_dim, seed=seed).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.grounded, self.total_updates, self.expert_id = False, 0, None
        self.provenance: dict[str, dict[str, str]] = {}

    def snapshot(self) -> ExpertValueSnapshot:
        """Freeze an independent model; later fits/checkpoint edits cannot change it."""
        model = copy.deepcopy(self.model).eval().requires_grad_(False)
        return ExpertValueSnapshot(model, self.grounded, self.expert_id)

    def predict(self, observations: np.ndarray, remaining: np.ndarray | int) -> FloatArray:
        """Score with current parameters, requiring grounding for positive horizons."""
        return _predict(self.model, observations, remaining, grounded=self.grounded)

    def _metrics(self, data: ExpertValueData, prefix: str) -> dict[str, float]:
        """Report positive-horizon raw-cost errors; boundary zeros do not dilute them."""
        mask = data.remaining > 0
        error = _predict(self.model, data.observations[mask], data.remaining[mask], grounded=True) - data.cost_targets[mask]
        scale = float(self.model.cost_scale)
        return {f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))), f"{prefix}_mae": float(np.mean(np.abs(error))),
                f"{prefix}_bias": float(np.mean(error)), f"{prefix}_max_abs_error": float(np.max(np.abs(error))),
                f"{prefix}_normalized_mse": float(np.mean((error / scale)**2))}

    def fit(self, training: ExpertValueData, holdout: ExpertValueData, *, updates: int = 1000,
            batch_size: int = 128, seed: int = 0) -> dict[str, Any]:
        """Fit normalized MSE using local minibatch RNG and reset-disjoint holdouts."""
        _integer(updates, "updates"); _integer(batch_size, "batch_size"); _integer(seed, "seed", minimum=0)
        for data in (training, holdout):
            if data.source != EXPERT_SOURCE or np.any(data.remaining > self.model.horizon):
                raise ValueError("Require independent MPPI labels within the configured finite horizon")
        train_mask, holdout_mask = training.remaining > 0, holdout.remaining > 0
        if train_mask.sum() < 2 or holdout_mask.sum() < 1:
            raise RuntimeError("Insufficient grounded expert data: need >=2 positive-horizon training and >=1 holdout rows")
        if np.intersect1d(training.reset_ids, holdout.reset_ids).size:
            raise ValueError("Training and holdout reset IDs must be disjoint")
        identity = training.provenance.get("expert_id")
        if identity != holdout.provenance.get("expert_id") or (self.expert_id is not None and identity != self.expert_id):
            raise ValueError("Expert identity differs across fit or holdout data")
        mean_target = float(training.cost_targets[train_mask].mean())
        if not self.grounded:
            self.model.initialize_cost_scale(mean_target)
        before = {**self._metrics(training, "train"), **self._metrics(holdout, "holdout")}
        x = torch.tensor(training.observations[train_mask], dtype=torch.float32, device=self.device)
        remaining = torch.tensor(training.remaining[train_mask], dtype=torch.int64, device=self.device)
        targets = torch.tensor(training.cost_targets[train_mask], dtype=torch.float32, device=self.device)
        rng, losses = np.random.default_rng(seed), []
        self.model.train()
        for _ in range(updates):
            indices = torch.tensor(rng.integers(0, len(x), size=batch_size), dtype=torch.int64, device=self.device)
            loss = torch.mean(((self.model(x[indices], remaining[indices]) - targets[indices]) / self.model.cost_scale)**2)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite normalized expert-value fit loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach()))
        self.model.eval()
        after = {**self._metrics(training, "train"), **self._metrics(holdout, "holdout")}
        self.grounded, self.expert_id = True, identity
        self.total_updates += updates
        self.provenance = {"training": dict(training.provenance), "holdout": dict(holdout.provenance)}
        return {**after, "train_rmse_before": before["train_rmse"], "holdout_rmse_before": before["holdout_rmse"],
                "loss_normalized": float(np.mean(losses[-100:])), "cost_scale": float(self.model.cost_scale),
                "training_target_mean": mean_target, "training_rows": len(training.remaining),
                "training_positive_rows": int(train_mask.sum()), "holdout_rows": len(holdout.remaining),
                "holdout_positive_rows": int(holdout_mask.sum()), "training_reset_count": len(np.unique(training.reset_ids)),
                "holdout_reset_count": len(np.unique(holdout.reset_ids)), "updates": updates,
                "total_updates": self.total_updates, "horizon": self.model.horizon, "source": EXPERT_SOURCE,
                "target_kind": "realized_finite_horizon_expert_cost", "expert_id": identity,
                "interpretation": "Supervised oracle-cost regression; held-out error is not a performance certificate."}

    def state_dict(self) -> dict[str, Any]:
        """Copy model/Adam/provenance into a torch-save-compatible checkpoint."""
        return copy.deepcopy({"format_version": 1, "kind": "finite_horizon_independent_mppi_value",
                              "horizon": self.model.horizon, "hidden_dim": self.model.hidden_dim,
                              "learning_rate": self.lr, "initial_seed": self.seed, "grounded": self.grounded, "expert_id": self.expert_id,
                              "total_updates": self.total_updates, "provenance": self.provenance,
                              "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a matching finite-horizon architecture and independent optimizer state."""
        if (state.get("format_version") != 1 or state.get("kind") != "finite_horizon_independent_mppi_value"
                or state.get("horizon") != self.model.horizon or state.get("hidden_dim") != self.model.hidden_dim):
            raise ValueError("Expert-value checkpoint kind, horizon or architecture differs")
        if (type(state.get("grounded")) is not bool or not np.isfinite(state["learning_rate"])
                or state["learning_rate"] <= 0 or state["grounded"] and state["total_updates"] < 1):
            raise ValueError("Invalid expert-value checkpoint fit metadata")
        candidate = copy.deepcopy(self.model)
        candidate.load_state_dict(state["model"])
        if any(not torch.isfinite(value).all() for value in candidate.state_dict().values()) or float(candidate.cost_scale) <= 0:
            raise ValueError("Expert-value checkpoint contains invalid parameters or cost scale")
        self.model.load_state_dict(copy.deepcopy(state["model"]))
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.lr = float(state["learning_rate"])
        self.seed = _integer(state["initial_seed"], "initial_seed", minimum=0)
        self.grounded = bool(state["grounded"])
        self.total_updates = _integer(state["total_updates"], "total_updates", minimum=0)
        self.expert_id = state["expert_id"]
        self.provenance = copy.deepcopy(state["provenance"])
