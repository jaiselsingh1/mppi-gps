"""Policy-evaluation cost values and explicitly predictive endpoint credit.

Only actual deterministic learner trajectories fit this critic. Accepted MPPI
plans have a different continuation law and are rejected as policy-evaluation
data. A lower fitted value is an estimated change, not evidence that the
policy's realized return improved.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OBS_DIM = 17
POLICY_DATA = "policy_evaluation"


def _array(value, *, ndim: int, name: str, finite: bool = True) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != ndim or (finite and not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array.")
    result.setflags(write=False)
    return result


def _discount(gamma: float) -> float:
    if not np.isfinite(gamma) or not 0 <= gamma < 1:
        raise ValueError("gamma must be finite and in [0,1).")
    return float(gamma)


def _policy_source(source: str) -> None:
    if source != POLICY_DATA:
        raise ValueError("V^pi requires actual policy_evaluation rollouts; accepted-plan/hybrid data are not V^pi targets.")


@dataclass(frozen=True)
class PolicyEpisode:
    """One actual policy episode, ending at its first fall or a healthy cutoff.

    Costs are RAW nonnegative task costs, including the configured terminal
    fall charge once. They exclude analytic absorbing tails and SAC rewards.
    """

    observations: np.ndarray
    costs: np.ndarray
    terminated: bool
    policy_id: str
    source: str = POLICY_DATA

    def __post_init__(self):
        obs = _array(self.observations, ndim=2, name="observations")
        costs = _array(self.costs, ndim=1, name="costs")
        if not len(costs) or obs.shape != (len(costs) + 1, OBS_DIM) or np.any(costs < 0):
            raise ValueError("Require observations[L+1,17] and nonnegative raw costs[L].")
        if not isinstance(self.terminated, (bool, np.bool_)) or not self.policy_id:
            raise ValueError("Require an explicit termination flag and policy_id.")
        object.__setattr__(self, "observations", obs)
        object.__setattr__(self, "costs", costs)


@dataclass(frozen=True)
class ValueTransitionBatch:
    """Discounted executed prefixes; only healthy endpoints get bootstrapped."""

    observations: np.ndarray
    next_observations: np.ndarray
    discounted_costs: np.ndarray
    controls: np.ndarray
    terminated: np.ndarray
    gamma: float
    policy_id: str
    source: str = POLICY_DATA

    def __post_init__(self):
        obs = _array(self.observations, ndim=2, name="observations")
        nxt = _array(self.next_observations, ndim=2, name="next_observations", finite=False)
        costs = _array(self.discounted_costs, ndim=1, name="discounted_costs")
        count = len(obs)
        controls = np.array(self.controls, copy=True)
        terminated = np.array(self.terminated, copy=True)
        if not count or obs.shape != (count, OBS_DIM) or nxt.shape != obs.shape or costs.shape != (count,):
            raise ValueError("Value transitions must have aligned [N,17] observations and [N] costs.")
        if controls.shape != (count,) or not np.issubdtype(controls.dtype, np.integer) or np.any(controls < 1):
            raise ValueError("controls must be positive integer counts.")
        if terminated.shape != (count,) or not np.isin(terminated, (0, 1)).all():
            raise ValueError("terminated must be an aligned boolean mask.")
        terminated = terminated.astype(bool)
        if not np.isfinite(nxt[~terminated]).all() or np.any(costs < 0) or not self.policy_id:
            raise ValueError("Healthy endpoints and nonnegative costs must be finite; policy_id is required.")
        controls.setflags(write=False)
        terminated.setflags(write=False)
        for name, value in (("observations", obs), ("next_observations", nxt), ("discounted_costs", costs),
                            ("controls", controls), ("terminated", terminated), ("gamma", _discount(self.gamma))):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class PolicyReturnAnchors:
    """Realized long policy returns, with estimated healthy suffixes explicit.

    ``exact`` means a real fall plus its analytic absorbing future was observed.
    Healthy cutoffs can carry a frozen prior-critic estimate, but are not Monte
    Carlo truth. Unbootstrapped healthy prefixes are lower bounds and excluded.
    """

    observations: np.ndarray
    cost_targets: np.ndarray
    exact: np.ndarray
    gamma: float
    policy_id: str
    target_kind: str = "realized_absorbing"
    source: str = POLICY_DATA

    def __post_init__(self):
        obs = _array(self.observations, ndim=2, name="anchor observations")
        targets = _array(self.cost_targets, ndim=1, name="anchor costs")
        exact = np.array(self.exact, copy=True)
        if not len(obs) or obs.shape != (len(targets), OBS_DIM) or exact.shape != targets.shape:
            raise ValueError("Anchor observations, costs and exact mask must align.")
        if not np.isin(exact, (0, 1)).all() or np.any(targets < 0) or not self.policy_id:
            raise ValueError("Anchors require nonnegative cost targets, exact flags and policy_id.")
        if self.target_kind not in {"realized_absorbing", "frozen_bootstrap", "healthy_prefix_lower_bound"}:
            raise ValueError("Unknown anchor target_kind.")
        if self.target_kind == "realized_absorbing" and not exact.all():
            raise ValueError("Realized absorbing anchors must all be exact.")
        exact = exact.astype(bool)
        exact.setflags(write=False)
        for name, value in (("observations", obs), ("cost_targets", targets), ("exact", exact),
                            ("gamma", _discount(self.gamma))):
            object.__setattr__(self, name, value)


class PolicyCostValue(nn.Module):
    """17 -> 256 -> 256 -> 1 cost value, with an explicit physical cost scale."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer.")
        self.hidden_dim = hidden_dim
        self.fc1, self.fc2, self.head = nn.Linear(OBS_DIM, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, 1)
        self.register_buffer("cost_scale", torch.tensor(1.0))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != OBS_DIM:
            raise ValueError("Value observations must have 17 features.")
        hidden = torch.tanh(self.fc1(observations))
        hidden = torch.tanh(self.fc2(hidden))
        return self.cost_scale * F.softplus(self.head(hidden)).squeeze(-1)

    @torch.no_grad()
    def initialize_cost_scale(self, realized_mean: float) -> None:
        if not np.isfinite(realized_mean) or realized_mean < 0:
            raise ValueError("A realized nonnegative mean cost is required.")
        scale = max(float(realized_mean), 1.0)
        self.cost_scale.fill_(scale)
        # Constant positive starting prediction in grounded cost units. Hidden
        # features remain trainable; a zero head is released by its first update.
        self.head.weight.zero_()
        desired = max(float(realized_mean) / scale, 1e-4)
        self.head.bias.fill_(float(np.log(np.expm1(desired))))


def build_nstep_targets(discounted_costs: torch.Tensor, controls: torch.Tensor,
                        terminated: torch.Tensor, bootstrap_values: torch.Tensor, *,
                        gamma: float, absorbing_cost: float = 5.0) -> torch.Tensor:
    """Detached cost targets; a fall's analytic future is added exactly once.

    Prefixes contain raw task costs only. A fall after h executed controls has
    future cost gamma**h * absorbing_cost/(1-gamma). A healthy h-control cutoff
    has gamma**h * V_target(x_h), regardless of administrative truncation.
    """
    gamma = _discount(gamma)
    if not np.isfinite(absorbing_cost) or absorbing_cost < 0:
        raise ValueError("absorbing_cost must be finite and nonnegative.")
    tensors = (discounted_costs, controls, terminated, bootstrap_values)
    if discounted_costs.ndim != 1 or any(t.shape != discounted_costs.shape or t.device != discounted_costs.device for t in tensors):
        raise ValueError("Target tensors must be aligned vectors on one device.")
    if not discounted_costs.is_floating_point() or not torch.isfinite(discounted_costs).all() or (discounted_costs < 0).any():
        raise ValueError("Prefixes must be finite nonnegative discounted costs, not SAC rewards.")
    if controls.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or (controls < 1).any():
        raise ValueError("controls must be positive integers.")
    if not torch.all((terminated == 0) | (terminated == 1)):
        raise ValueError("terminated must be boolean.")
    with torch.no_grad():
        ended = terminated.bool()
        if not torch.isfinite(bootstrap_values[~ended]).all() or (bootstrap_values[~ended] < 0).any():
            raise ValueError("Healthy endpoint values must be finite nonnegative costs.")
        discount = torch.pow(torch.as_tensor(gamma, dtype=discounted_costs.dtype, device=discounted_costs.device), controls)
        targets = discounted_costs.detach().clone()
        targets[ended] += discount[ended] * absorbing_cost / (1.0 - gamma)
        targets[~ended] += discount[~ended] * bootstrap_values.detach()[~ended]
    return targets


@dataclass(frozen=True)
class FrozenPolicyValue:
    model: PolicyCostValue
    gamma: float
    policy_id: str | None
    grounded: bool

    @torch.no_grad()
    def predict(self, observations: np.ndarray, *, require_grounding: bool = True) -> np.ndarray:
        if require_grounding and not self.grounded:
            raise RuntimeError("Value critic has not been grounded in realized policy costs.")
        obs = np.asarray(observations)
        if obs.ndim != 2 or obs.shape[1] != OBS_DIM or not np.isfinite(obs).all():
            raise ValueError("Require finite observations[N,17].")
        device = next(self.model.parameters()).device
        values = self.model(torch.tensor(obs, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite value predictions.")
        return values


def value_data_from_episodes(episodes: Sequence[PolicyEpisode], *, gamma: float = .995,
                             horizon: int = 32, stride: int | None = None,
                             absorbing_cost: float = 5.0,
                             previous: FrozenPolicyValue | None = None) -> tuple[ValueTransitionBatch, PolicyReturnAnchors]:
    """Make TD(H) prefixes and measured remaining-return anchors without simulation."""
    gamma = _discount(gamma)
    if not episodes or type(horizon) is not int or horizon < 1:
        raise ValueError("Need policy episodes and a positive horizon.")
    stride = horizon if stride is None else stride
    if type(stride) is not int or stride < 1 or not np.isfinite(absorbing_cost) or absorbing_cost < 0:
        raise ValueError("Invalid stride or absorbing cost.")
    policy_id = episodes[0].policy_id
    observations, endpoints, prefix_costs, lengths, ended, returns, exact = [], [], [], [], [], [], []
    for episode in episodes:
        _policy_source(episode.source)
        if episode.policy_id != policy_id:
            raise ValueError("Current-only policy evaluation cannot mix policy versions.")
        count = len(episode.costs)
        tail = 0.0
        if not episode.terminated and previous is not None and previous.grounded:
            if previous.gamma != gamma:
                raise ValueError("Previous critic uses a different discount.")
            tail = float(previous.predict(episode.observations[-1:])[0])
        for start in range(0, count, stride):
            stop = min(start + horizon, count)
            falls = episode.terminated and stop == count
            observations.append(episode.observations[start])
            endpoints.append(episode.observations[stop])
            prefix_costs.append(float(np.dot(gamma ** np.arange(stop - start), episode.costs[start:stop])))
            lengths.append(stop - start)
            ended.append(falls)
            remaining = count - start
            observed = float(np.dot(gamma ** np.arange(remaining), episode.costs[start:]))
            continuation = absorbing_cost / (1 - gamma) if episode.terminated else tail
            returns.append(observed + gamma ** remaining * continuation)
            exact.append(episode.terminated)
    source = POLICY_DATA
    batch = ValueTransitionBatch(np.asarray(observations), np.asarray(endpoints), np.asarray(prefix_costs),
                                 np.asarray(lengths), np.asarray(ended), gamma, policy_id, source)
    kind = ("realized_absorbing" if all(exact) else "frozen_bootstrap" if previous is not None and previous.grounded
            else "healthy_prefix_lower_bound")
    anchors = PolicyReturnAnchors(np.asarray(observations), np.asarray(returns), np.asarray(exact), gamma, policy_id, kind, source)
    return batch, anchors


class ValueCreditTrainer:
    """Current-policy TD(H) plus realized-return anchors, with frozen target fits."""

    def __init__(self, *, gamma: float = .995, absorbing_cost: float = 5.0,
                 hidden_dim: int = 256, lr: float = 3e-4, device: str | torch.device = "cpu"):
        self.gamma = _discount(gamma)
        if not np.isfinite(absorbing_cost) or absorbing_cost < 0 or not np.isfinite(lr) or lr <= 0:
            raise ValueError("Invalid absorbing cost or learning rate.")
        self.absorbing_cost, self.device = float(absorbing_cost), torch.device(device)
        self.value = PolicyCostValue(hidden_dim).to(self.device)
        self.target = copy.deepcopy(self.value).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.grounded, self.policy_id, self.total_updates = False, None, 0

    def snapshot(self) -> FrozenPolicyValue:
        return FrozenPolicyValue(copy.deepcopy(self.value).eval().requires_grad_(False), self.gamma, self.policy_id, self.grounded)

    @torch.no_grad()
    def predict(self, observations: np.ndarray) -> np.ndarray:
        # The temporary wrapper does not copy parameters; snapshots do.
        return FrozenPolicyValue(self.value, self.gamma, self.policy_id, self.grounded).predict(observations)

    def hard_sync_target(self) -> None:
        self.target.load_state_dict(self.value.state_dict())
        self.target.eval().requires_grad_(False)

    def state_dict(self) -> dict:
        """Independent checkpoint of the scalar learner, its target and optimizer."""
        return copy.deepcopy({"format_version": 1, "gamma": self.gamma,
                              "absorbing_cost": self.absorbing_cost, "hidden_dim": self.value.hidden_dim,
                              "value": self.value.state_dict(), "target": self.target.state_dict(),
                              "optimizer": self.optimizer.state_dict(), "grounded": self.grounded,
                              "policy_id": self.policy_id, "total_updates": self.total_updates})

    def load_state_dict(self, state: dict) -> None:
        if (state.get("format_version") != 1 or state.get("gamma") != self.gamma
                or state.get("absorbing_cost") != self.absorbing_cost
                or state.get("hidden_dim") != self.value.hidden_dim):
            raise ValueError("Value checkpoint configuration differs.")
        self.value.load_state_dict(state["value"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.grounded, self.policy_id = bool(state["grounded"]), state["policy_id"]
        self.total_updates = int(state["total_updates"])
        self.value.eval()
        self.target.eval().requires_grad_(False)

    def fit(self, batch: ValueTransitionBatch, anchors: PolicyReturnAnchors, *,
            updates: int = 2000, batch_size: int = 256, seed: int = 0,
            anchor_weight: float = 1.0, allow_estimated_anchors: bool = True) -> dict:
        """Fit once to a fixed target copy; hard-sync only after all updates.

        Initial scale must have exact absorbing policy-return anchors. Later
        healthy anchors may include an explicitly frozen prior continuation.
        Unbootstrapped healthy lower bounds are never fitted as exact values.
        """
        for data in (batch, anchors):
            _policy_source(data.source)
            if data.gamma != self.gamma:
                raise ValueError("Training data discount differs from the critic.")
        if batch.policy_id != anchors.policy_id:
            raise ValueError("TD and anchor data must evaluate the same current policy.")
        if type(updates) is not int or updates < 1 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("updates and batch_size must be positive integers.")
        if not np.isfinite(anchor_weight) or anchor_weight <= 0:
            raise ValueError("A positive realized-return anchor weight is required.")
        fit_anchor = anchors.exact.copy()
        if allow_estimated_anchors and anchors.target_kind == "frozen_bootstrap" and self.grounded:
            fit_anchor[:] = True
        cold_start = not self.grounded
        if cold_start:
            if not anchors.exact.any():
                raise RuntimeError("Cold-start value critic needs realized absorbing policy returns, not zero bootstrap or healthy lower bounds.")
            self.value.initialize_cost_scale(float(anchors.cost_targets[anchors.exact].mean()))
            self.hard_sync_target()
        if not fit_anchor.any():
            raise RuntimeError("No grounded anchors available; provide exact returns or explicit frozen-bootstrap anchors.")
        previous_policy_id = self.policy_id
        tensor = lambda array: torch.as_tensor(np.array(array, copy=True), dtype=torch.float32, device=self.device)
        obs, next_obs = tensor(batch.observations), tensor(batch.next_observations)
        prefix = tensor(batch.discounted_costs)
        controls = torch.as_tensor(np.array(batch.controls, copy=True), dtype=torch.int64, device=self.device)
        ended = torch.as_tensor(np.array(batch.terminated, copy=True), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            bootstrap = torch.zeros_like(prefix)
            if (~ended).any():
                bootstrap[~ended] = self.target(next_obs[~ended])
            targets = build_nstep_targets(prefix, controls, ended, bootstrap, gamma=self.gamma, absorbing_cost=self.absorbing_cost)
        anchor_obs, anchor_targets = tensor(anchors.observations[fit_anchor]), tensor(anchors.cost_targets[fit_anchor])
        before_anchor = self.value(anchor_obs).detach()
        scale = float(self.value.cost_scale)
        generator = np.random.default_rng(seed)
        losses = []
        self.value.train()
        for _ in range(updates):
            indices = torch.as_tensor(generator.integers(0, len(obs), size=min(batch_size, len(obs))), device=self.device)
            anchor_indices = torch.as_tensor(generator.integers(0, len(anchor_obs), size=min(batch_size, len(anchor_obs))), device=self.device)
            td_loss = F.mse_loss(self.value(obs[indices]) / scale, targets[indices] / scale)
            anchor_loss = F.mse_loss(self.value(anchor_obs[anchor_indices]) / scale, anchor_targets[anchor_indices] / scale)
            loss = td_loss + anchor_weight * anchor_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite policy-value loss.")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.value.parameters()):
                raise FloatingPointError("Non-finite policy-value gradient.")
            self.optimizer.step()
            losses.append(float(loss.detach()))
        self.value.eval()
        with torch.no_grad():
            after_anchor = self.value(anchor_obs)
            if not torch.isfinite(after_anchor).all():
                raise FloatingPointError("Non-finite fitted policy values.")
            log = {"loss_normalized": float(np.mean(losses[-100:])), "cost_scale": scale,
                   "anchor_rmse_before": float((before_anchor - anchor_targets).square().mean().sqrt()),
                   "anchor_rmse_after": float((after_anchor - anchor_targets).square().mean().sqrt()),
                   "td_target_mean": float(targets.mean()), "td_target_min": float(targets.min()), "td_target_max": float(targets.max()),
                   "anchor_target_mean": float(anchor_targets.mean()), "exact_anchor_count": int(anchors.exact.sum()),
                   "fitted_anchor_count": int(fit_anchor.sum()), "healthy_bootstrap_count": int((~ended).sum()),
                   "fall_count": int(ended.sum()), "updates": updates, "gamma": self.gamma,
                   "policy_id": batch.policy_id, "previous_policy_id": previous_policy_id,
                   "bootstrap_initialization": "constant_from_realized_absorbing_mean" if cold_start else "previous_fitted_target",
                   "anchor_target_kind": anchors.target_kind, "target_frozen_during_fit": True,
                   "cost_units": "discounted raw task cost plus one absorbing future; no SAC reward or entropy"}
        self.grounded, self.policy_id = True, batch.policy_id
        self.total_updates += updates
        self.hard_sync_target()
        return log


def endpoint_value_credit(previous: FrozenPolicyValue, current: FrozenPolicyValue,
                           endpoints: np.ndarray, *, tau: float = 1.0) -> dict:
    """Predict decreases at the SAME fixed endpoints; this is not a return test."""
    if previous.gamma != current.gamma or not np.isfinite(tau) or tau <= 0:
        raise ValueError("Endpoint critics need matching discounts and a positive credit scale.")
    if any(p.requires_grad for model in (previous.model, current.model) for p in model.parameters()):
        raise ValueError("Endpoint credit requires frozen critic snapshots.")
    obs = np.asarray(endpoints)
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM or not np.isfinite(obs).all():
        raise ValueError("Require finite fixed endpoints[N,17].")
    if not previous.grounded or not current.grounded:
        return {"ready": False, "predicted_decrease": np.zeros(len(obs)), "weights": np.zeros(len(obs)),
                "interpretation": "Uninitialized value estimates supply no credit; bootstrap handling belongs to the caller."}
    old, new = previous.predict(obs), current.predict(obs)
    delta = old - new
    return {"ready": True, "previous_values": old, "current_values": new,
            "predicted_decrease": delta, "weights": np.tanh(np.maximum(delta, 0.0) / tau),
            "interpretation": "Positive fitted cost decrease at fixed endpoints; may reflect estimation error, not realized policy improvement."}
