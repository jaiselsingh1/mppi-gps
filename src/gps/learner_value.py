"""Finite learner-policy Q evaluation from mixed, actually executed transitions.

A teacher transition supervises its RECORDED action and immediate raw cost.
Every healthy bootstrap instead follows a frozen current learner actor. No
complete teacher return, recorded next teacher action, entropy term or cloning
loss enters this module. Twin agreement and held-out Bellman error are not
independent Monte Carlo calibration or an accuracy certificate.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


OBS_DIM, ACTION_DIM = 17, 6
SOURCES = ("teacher", "policy")


def _int(value: int, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _gamma(value: float) -> float:
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Finite-horizon gamma must lie in [0,1]")
    return float(value)


def _float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _integers(value: Any, shape: tuple[int, ...], name: str, minimum: int = 0) -> np.ndarray:
    raw = np.asarray(value)
    if (raw.shape != shape or not np.issubdtype(raw.dtype, np.integer)
            or not np.isfinite(raw).all() or np.any(raw < minimum)):
        raise ValueError(f"{name} must be integer counts >= {minimum} with shape {shape}")
    array = np.array(raw, dtype=np.int64, copy=True)
    if np.any(array < minimum):
        raise ValueError(f"{name} overflows signed integer storage")
    array.setflags(write=False)
    return array


def _remaining(value: Any, count: int, horizon: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim == 0:
        raw = np.full(count, raw.item())
    array = _integers(raw, (count,), "remaining")
    if np.any(array > horizon):
        raise ValueError("Remaining horizon exceeds the configured horizon")
    return array


def _finite_tail(gamma: float, remaining: int, cost: float = 5.) -> float:
    if remaining == 0:
        return 0.
    if gamma == 1:
        return cost * remaining
    if gamma == 0:
        return cost
    return float(cost * math.expm1(remaining * math.log(gamma)) / math.expm1(math.log(gamma)))


@dataclass(frozen=True)
class TransitionBatch:
    """One-step raw observations/actions; remaining includes THIS transition.

    ``costs`` include a real fall's terminal charge once, without any adapter
    failure tail. ``terminated`` is physical failure, not an acquisition cutoff.
    ``reset_ids`` separate training and held-out reset episodes. Sources label
    actual behavior, not whose continuation is used by the Bellman target.
    """

    observations: np.ndarray
    actions: np.ndarray
    costs: np.ndarray
    next_observations: np.ndarray
    remaining: np.ndarray
    terminated: np.ndarray
    source: str | np.ndarray
    reset_ids: np.ndarray

    def __post_init__(self) -> None:
        raw = np.asarray(self.observations)
        if raw.ndim != 2 or raw.shape[1:] != (OBS_DIM,) or not len(raw):
            raise ValueError("Require nonempty observations[N,17]")
        count = len(raw)
        for name, shape in (("observations", (count, OBS_DIM)), ("actions", (count, ACTION_DIM)),
                            ("costs", (count,)), ("next_observations", (count, OBS_DIM))):
            object.__setattr__(self, name, _float_array(getattr(self, name), shape, name))
        if np.any(self.costs < 0):
            raise ValueError("Costs must be nonnegative RAW task costs")
        object.__setattr__(self, "remaining", _integers(self.remaining, (count,), "remaining", 1))
        object.__setattr__(self, "reset_ids", _integers(self.reset_ids, (count,), "reset_ids"))
        ended = np.asarray(self.terminated)
        if ended.shape != (count,) or ended.dtype != np.bool_:
            raise ValueError("terminated must be a boolean (N,) vector of physical failures")
        ended = np.array(ended, copy=True); ended.setflags(write=False)
        object.__setattr__(self, "terminated", ended)
        source = np.asarray(self.source)
        if source.ndim == 0:
            source = np.full(count, source.item())
        if source.shape != (count,) or not np.isin(source, SOURCES).all():
            raise ValueError("Source must be 'teacher'/'policy' or an aligned vector of those names")
        source = np.array(source, dtype="U7", copy=True); source.setflags(write=False)
        object.__setattr__(self, "source", source)


class LearnerCostQ(nn.Module):
    """17 observations + 6 actions + remaining/N -> positive raw-cost value."""

    def __init__(self, *, horizon: int, hidden_dim: int, cost_scale: float):
        super().__init__()
        self.horizon, self.hidden_dim = _int(horizon, "horizon"), _int(hidden_dim, "hidden_dim")
        if not np.isfinite(cost_scale) or cost_scale <= 0:
            raise ValueError("Cost scale must be finite and positive")
        self.fc1, self.fc2 = nn.Linear(24, hidden_dim), nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.register_buffer("cost_scale", torch.tensor(float(cost_scale), dtype=torch.float32))
        # A declared physical-unit prior, not a measured learner return or bound.
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, math.log(math.expm1(1.)))

    def forward(self, observations: torch.Tensor, actions: torch.Tensor,
                remaining: torch.Tensor) -> torch.Tensor:
        if (observations.ndim != 2 or observations.shape[1] != OBS_DIM
                or actions.shape != (len(observations), ACTION_DIM) or remaining.shape != (len(observations),)):
            raise ValueError("Require observations[N,17], actions[N,6], remaining[N]")
        dtype = self.fc1.weight.dtype
        inputs = torch.cat((observations.to(dtype), actions.to(dtype),
                            remaining.to(dtype)[:, None] / self.horizon), dim=-1)
        hidden = torch.tanh(self.fc2(torch.tanh(self.fc1(inputs))))
        value = self.cost_scale * F.softplus(self.head(hidden)).squeeze(-1)
        return torch.where(remaining == 0, torch.zeros_like(value), value)


def learner_cost_targets(costs: torch.Tensor, remaining: torch.Tensor, terminated: torch.Tensor,
                         next_q1: torch.Tensor, next_q2: torch.Tensor, *, gamma: float,
                         absorbing_cost: float = 5.) -> torch.Tensor:
    """Detached finite Bellman targets, using pessimistic max-cost twin tails.

    next_q values must already query the frozen CURRENT learner action. A fall
    has c_raw + gamma*5*sum_{j=0}^{n-2} gamma**j; healthy n=1 has only c_raw.
    Terminal/zero-horizon rows never use next_q, even if its entries are NaN.
    """
    gamma = _gamma(gamma)
    if not np.isfinite(absorbing_cost) or absorbing_cost < 0:
        raise ValueError("Absorbing future cost must be finite and nonnegative")
    tensors = (costs, remaining, terminated, next_q1, next_q2)
    if costs.ndim != 1 or any(t.shape != costs.shape or t.device != costs.device for t in tensors):
        raise ValueError("Bellman target tensors must be aligned vectors on the same device")
    if not costs.is_floating_point() or not torch.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Require nonnegative finite raw costs")
    if remaining.dtype not in (torch.int32, torch.int64) or (remaining < 1).any() or terminated.dtype != torch.bool:
        raise ValueError("Require positive integer remaining counts and boolean failure flags")
    with torch.no_grad():
        targets = costs.detach().clone()
        future = remaining - 1
        ended = terminated & (future > 0)
        if ended.any() and gamma > 0:
            if gamma == 1:
                tail = future[ended].to(costs.dtype) * absorbing_cost
            else:
                tail = absorbing_cost * torch.expm1(future[ended].to(costs.dtype) * math.log(gamma)) / math.expm1(math.log(gamma))
            targets[ended] += gamma * tail
        bootstrap = ~terminated & (future > 0) & (gamma > 0)
        if bootstrap.any():
            q1, q2 = next_q1[bootstrap], next_q2[bootstrap]
            if (not torch.isfinite(q1).all() or not torch.isfinite(q2).all()
                    or (q1 < 0).any() or (q2 < 0).any()):
                raise ValueError("Learner bootstrap costs must be finite and nonnegative")
            targets[bootstrap] += gamma * torch.maximum(q1.detach(), q2.detach())
        if not torch.isfinite(targets).all():
            raise FloatingPointError("Finite learner Bellman targets overflowed")
    return targets


@dataclass(frozen=True)
class LearnerValueSnapshot:
    q1: LearnerCostQ
    q2: LearnerCostQ
    grounded: bool
    policy_id: str | None

    @torch.no_grad()
    def predict(self, observations: np.ndarray, actions: np.ndarray,
                remaining: np.ndarray | int) -> np.ndarray:
        """Return max twin cost; boundary zero is exact even before first fit."""
        obs = np.asarray(observations)
        if obs.ndim != 2 or obs.shape[1:] != (OBS_DIM,):
            raise ValueError("Require observations[N,17]")
        obs = _float_array(obs, (len(obs), OBS_DIM), "observations")
        actions = _float_array(actions, (len(obs), ACTION_DIM), "actions")
        left = _remaining(remaining, len(obs), self.q1.horizon)
        values = np.zeros(len(obs), dtype=np.float64)
        positive = left > 0
        if not positive.any():
            return values
        if not self.grounded:
            raise RuntimeError("Learner critic has not been fitted to actual transition data")
        if any(p.requires_grad for model in (self.q1, self.q2) for p in model.parameters()):
            raise ValueError("Scoring snapshot must remain frozen")
        device = self.q1.cost_scale.device
        o = torch.tensor(obs[positive], dtype=torch.float32, device=device)
        a = torch.tensor(actions[positive], dtype=torch.float32, device=device)
        n = torch.tensor(left[positive], dtype=torch.int64, device=device)
        values[positive] = torch.maximum(self.q1(o, a, n), self.q2(o, a, n)).cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise FloatingPointError("Invalid predicted learner cost")
        return values


class LearnerValueTrainer:
    """Fitted finite Q^pi from mixed data, with a frozen actor and Polyak targets.

    Both twins use max-cost target aggregation, the cost analogue of clipped
    minimum reward Q. This bias heuristic is not a conservative-value guarantee.
    The scale 5*sum_{j=0}^{N-1}gamma**j is a reference unit, NOT a cost bound.
    """

    def __init__(self, *, horizon: int, gamma: float = .999, hidden_dim: int = 128,
                 lr: float = 3e-4, seed: int = 0, device: str | torch.device = "cpu"):
        self.horizon, self.gamma = _int(horizon, "horizon"), _gamma(gamma)
        self.hidden_dim, seed = _int(hidden_dim, "hidden_dim"), _int(seed, "seed", 0)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        self.device, self.lr, self.tau = torch.device(device), float(lr), .005
        self.cost_scale = _finite_tail(self.gamma, self.horizon)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            self.q1 = LearnerCostQ(horizon=horizon, hidden_dim=hidden_dim, cost_scale=self.cost_scale).to(self.device)
            self.q2 = LearnerCostQ(horizon=horizon, hidden_dim=hidden_dim, cost_scale=self.cost_scale).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.grounded, self.policy_id, self.total_updates = False, None, 0

    def snapshot(self) -> LearnerValueSnapshot:
        return LearnerValueSnapshot(copy.deepcopy(self.q1).eval().requires_grad_(False),
                                    copy.deepcopy(self.q2).eval().requires_grad_(False),
                                    self.grounded, self.policy_id)

    @torch.no_grad()
    def predict(self, observations: np.ndarray, actions: np.ndarray, remaining: np.ndarray | int) -> np.ndarray:
        return self.snapshot().predict(observations, actions, remaining)

    @staticmethod
    def _actor_actions(actor: nn.Module, observations: torch.Tensor) -> torch.Tensor:
        mean, _ = actor(observations)
        actions = torch.tanh(mean) * actor.action_scale + actor.action_bias
        if actions.shape != (len(observations), ACTION_DIM) or not torch.isfinite(actions).all():
            raise FloatingPointError("Frozen learner produced invalid bootstrap actions")
        return actions

    @torch.no_grad()
    def _targets(self, data: TransitionBatch, indices: np.ndarray, actor: nn.Module) -> torch.Tensor:
        costs = torch.tensor(data.costs[indices], dtype=torch.float32, device=self.device)
        remaining = torch.tensor(data.remaining[indices], dtype=torch.int64, device=self.device)
        ended = torch.tensor(data.terminated[indices], dtype=torch.bool, device=self.device)
        q1, q2 = torch.zeros_like(costs), torch.zeros_like(costs)
        mask = ~ended & (remaining > 1) & (self.gamma > 0)
        if mask.any():
            obs = torch.tensor(data.next_observations[indices][mask.cpu().numpy()], dtype=torch.float32, device=self.device)
            actions = self._actor_actions(actor, obs)
            q1[mask] = self.q1_target(obs, actions, remaining[mask] - 1)
            q2[mask] = self.q2_target(obs, actions, remaining[mask] - 1)
        return learner_cost_targets(costs, remaining, ended, q1, q2, gamma=self.gamma)

    def _metrics(self, data: TransitionBatch, actor: nn.Module, prefix: str) -> dict[str, float]:
        with torch.no_grad():
            indices = np.arange(len(data.costs))
            target = self._targets(data, indices, actor)
            obs = torch.tensor(np.array(data.observations), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(data.actions), dtype=torch.float32, device=self.device)
            remaining = torch.tensor(np.array(data.remaining), dtype=torch.int64, device=self.device)
            first, second = self.q1(obs, actions, remaining), self.q2(obs, actions, remaining)
            error = torch.maximum(first, second) - target
            return {f"{prefix}_bellman_rmse": float(torch.sqrt(error.square().mean())),
                    f"{prefix}_bellman_bias": float(error.mean()),
                    f"{prefix}_twin_disagreement": float((first - second).abs().mean()),
                    f"{prefix}_target_mean": float(target.mean())}

    def fit(self, training: TransitionBatch, holdout: TransitionBatch, actor: nn.Module, *,
            updates: int = 1000, batch_size: int = 256, seed: int = 0) -> dict[str, Any]:
        updates, batch_size, seed = _int(updates, "updates"), _int(batch_size, "batch_size"), _int(seed, "seed", 0)
        for data in (training, holdout):
            if np.any(data.remaining > self.horizon):
                raise ValueError("Transition remaining horizon exceeds this critic's horizon")
        if np.intersect1d(training.reset_ids, holdout.reset_ids).size:
            raise ValueError("Training and holdout resets must be disjoint")
        frozen_actor = copy.deepcopy(actor).to(self.device).eval().requires_grad_(False)
        digest = hashlib.sha256()
        for name, value in frozen_actor.state_dict().items():
            digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        policy_id = digest.hexdigest()
        before = self._metrics(holdout, frozen_actor, "holdout")
        rng = np.random.default_rng(seed)
        by_source = {name: np.flatnonzero(training.source == name) for name in SOURCES}
        balanced = all(len(rows) for rows in by_source.values())
        sampled = {name: 0 for name in SOURCES}
        obs = torch.tensor(np.array(training.observations), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(training.actions), dtype=torch.float32, device=self.device)
        remaining = torch.tensor(np.array(training.remaining), dtype=torch.int64, device=self.device)
        losses = []
        self.q1.train(); self.q2.train()
        for _ in range(updates):
            if balanced:
                teacher_count = batch_size // 2 + (int(rng.integers(2)) if batch_size % 2 else 0)
                indices = np.concatenate((rng.choice(by_source["teacher"], teacher_count, replace=True),
                    rng.choice(by_source["policy"], batch_size - teacher_count, replace=True)))
                rng.shuffle(indices)
            else:
                indices = rng.integers(len(training.costs), size=batch_size)
            for name in SOURCES:
                sampled[name] += int(np.count_nonzero(training.source[indices] == name))
            target = self._targets(training, indices, frozen_actor)
            index = torch.tensor(indices, dtype=torch.int64, device=self.device)
            first, second = self.q1(obs[index], actions[index], remaining[index]), self.q2(obs[index], actions[index], remaining[index])
            loss = (((first - target) / self.cost_scale).square().mean()
                    + ((second - target) / self.cost_scale).square().mean())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite learner-critic normalized loss")
            self.optimizer.zero_grad(set_to_none=True); loss.backward(); self.optimizer.step()
            with torch.no_grad():
                for source, target_net in ((self.q1, self.q1_target), (self.q2, self.q2_target)):
                    for parameter, target_parameter in zip(source.parameters(), target_net.parameters()):
                        target_parameter.lerp_(parameter, self.tau)
            losses.append(float(loss.detach()))
        self.q1.eval(); self.q2.eval()
        self.grounded, self.policy_id = True, policy_id
        self.total_updates += updates
        metrics = {**self._metrics(training, frozen_actor, "train"), **self._metrics(holdout, frozen_actor, "holdout")}
        return {**metrics, "holdout_bellman_rmse_before": before["holdout_bellman_rmse"],
                "loss_normalized": float(np.mean(losses[-100:])), "cost_scale": self.cost_scale,
                "training_rows": len(training.costs), "holdout_rows": len(holdout.costs),
                "training_teacher_rows": int(len(by_source["teacher"])), "training_policy_rows": int(len(by_source["policy"])),
                "balanced_source_sampling": bool(balanced), "sampled_sources": sampled,
                "updates": updates, "total_updates": self.total_updates, "policy_id": policy_id,
                "gamma": self.gamma, "horizon": self.horizon, "target_polyak": self.tau,
                "target_continuation": "frozen current learner action; never recorded next teacher action",
                "target_kind": "finite Bellman policy evaluation; raw costs and one finite absorbing future",
                "aggregation": "maximum cost across twins; heuristic, not a bound",
                "interpretation": "Held-out Bellman residual and twin agreement are not independent return calibration."}

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"format_version": 1, "kind": "finite_mixed_transition_learner_q",
            "horizon": self.horizon, "gamma": self.gamma, "hidden_dim": self.hidden_dim,
            "lr": self.lr, "tau": self.tau, "cost_scale": self.cost_scale,
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "grounded": self.grounded,
            "policy_id": self.policy_id, "total_updates": self.total_updates})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (state.get("format_version") != 1 or state.get("kind") != "finite_mixed_transition_learner_q"
                or state.get("horizon") != self.horizon or state.get("gamma") != self.gamma
                or state.get("hidden_dim") != self.hidden_dim):
            raise ValueError("Learner-Q checkpoint architecture, horizon or gamma differs")
        copies = []
        for name in ("q1", "q2", "q1_target", "q2_target"):
            model = copy.deepcopy(getattr(self, name)); model.load_state_dict(state[name])
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()) or float(model.cost_scale) <= 0:
                raise ValueError("Checkpoint contains nonfinite weights or invalid scale")
            copies.append(model)
        scale = float(state["cost_scale"])
        if not np.isfinite(scale) or scale <= 0 or any(not np.isclose(float(m.cost_scale), scale) for m in copies):
            raise ValueError("Checkpoint physical cost scale is inconsistent")
        for name, model in zip(("q1", "q2", "q1_target", "q2_target"), copies):
            getattr(self, name).load_state_dict(model.state_dict())
        self.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        self.lr, self.tau, self.cost_scale = float(state["lr"]), float(state["tau"]), scale
        if not np.isfinite(self.lr) or self.lr <= 0 or not 0 < self.tau <= 1:
            raise ValueError("Invalid checkpoint optimization settings")
        self.grounded, self.policy_id = bool(state["grounded"]), state["policy_id"]
        self.total_updates = _int(state["total_updates"], "total_updates", 0)
        self.q1.eval(); self.q2.eval()
        self.q1_target.eval().requires_grad_(False); self.q2_target.eval().requires_grad_(False)
