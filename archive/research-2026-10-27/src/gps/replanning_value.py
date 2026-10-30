"""Complete measured returns of one frozen, retained-state replanning controller.

The target is the controller that actually continues after the recorded context,
not a canonical restart. Every nominal, solver warmstart and remaining horizon is
part of the context. PRNG states are logged by the caller for replay, but the
value averages over declared future planner randomness. There are no Bellman,
action-imitation or actor-policy targets. Ensemble spread, held-root residuals
and nearest-training-context support are diagnostics, never certificates.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral
from typing import Any

import numpy as np
import torch
from torch import nn


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _version(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("controller_version must be a nonempty string")
    return value


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return x


def replanning_features(snapshot: dict[str, Any], raw: dict[str, Any], remaining: int,
                        total_steps: int) -> np.ndarray:
    """Unclipped physical + solver + retained-plan context, before next planning.

    Walker state is (time, global x, 17 physical coordinates). Global translation
    and absolute simulator time are excluded; remaining/N carries finite time.
    The solver warmstart has nine coordinates. The nominal is the raw planner's
    already retained next-call U; callers must not shift or reset it here.
    """
    total_steps = _integer(total_steps, "total_steps", 1)
    remaining = _integer(remaining, "remaining")
    if remaining > total_steps:
        raise ValueError("remaining exceeds total_steps")
    if not isinstance(snapshot, dict) or not isinstance(raw, dict):
        raise ValueError("snapshot and raw must be context dictionaries")
    state = _finite_array(snapshot.get("state"), (19,), "state")
    warm = _finite_array(snapshot.get("warmstart"), (9,), "warmstart")
    nominal = np.asarray(raw.get("U"), dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 6 or nominal.shape[0] < 1 or not np.isfinite(nominal).all():
        raise ValueError("U must be a finite (H,6) raw nominal")
    initial = raw.get("initial_plan")
    if not isinstance(initial, (bool, np.bool_)):
        raise ValueError("initial_plan must be boolean")
    return np.r_[state[2:], warm, nominal.ravel(), float(initial), remaining / total_steps]


def contexts_features(contexts: list[dict[str, Any]], total_steps: int) -> np.ndarray:
    """Convenience adapter for stored full contexts using nominal or U keys."""
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("contexts must be a nonempty list")
    rows = [replanning_features(dict(state=c.get("state"), warmstart=c.get("warmstart")),
                               dict(U=c.get("nominal", c.get("U")), initial_plan=c.get("initial_plan")),
                               c.get("remaining"), total_steps) for c in contexts]
    if len({len(row) for row in rows}) != 1:
        raise ValueError("all contexts must use the same nominal horizon")
    return np.asarray(rows)


def _normalizers(gamma: float, remaining: np.ndarray) -> np.ndarray:
    if gamma == 1.:
        return 5. * remaining.astype(np.float64)
    if gamma == 0.:
        return 5. * (remaining > 0).astype(np.float64)
    return 5. * np.expm1(remaining * math.log(gamma)) / math.expm1(math.log(gamma))


def _rms(x: np.ndarray, weights: np.ndarray | None = None) -> float:
    if not len(x):
        return 0.
    largest = float(np.max(np.abs(x)))
    return largest * float(np.sqrt(np.average((x / largest) ** 2, weights=weights))) if largest else 0.


def _features(x: Any, remaining: Any, total_steps: int,
              dimension: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = np.asarray(remaining)
    if (x.ndim != 2 or x.shape[1] < 34 or (x.shape[1] - 28) % 6
            or (dimension is not None and x.shape[1] != dimension) or not np.isfinite(x).all()):
        raise ValueError("features must be finite full replanning context (M,28+6H)")
    if (n.shape != (len(x),) or n.dtype.kind not in "iu"
            or np.any(n < 0) or np.any(n > total_steps)):
        raise ValueError("remaining must be aligned integer counts within [0,total_steps]")
    if (not np.all((x[:, -2] == 0) | (x[:, -2] == 1))
            or not np.allclose(x[:, -1], n / total_steps, rtol=0, atol=1e-12)):
        raise ValueError("context initial flag or remaining/N coordinate is inconsistent")
    return x.copy(), n.astype(np.int64, copy=True)


def _weights(rows: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    """Equal root mass, then equal episode mass within each root."""
    roots: dict[int, dict[str, list[int]]] = {}
    for local, index in enumerate(indices):
        row = rows[int(index)]
        roots.setdefault(row["group"], {}).setdefault(row["episode_id"], []).append(local)
    result = np.zeros(len(indices))
    for episodes in roots.values():
        for members in episodes.values():
            result[members] = 1. / (len(roots) * len(episodes) * len(members))
    return result


def _pairs(rows: list[dict[str, Any]], indices: np.ndarray) -> list[tuple[int, int]]:
    found: dict[tuple[Any, ...], dict[str, int]] = {}
    for i in indices:
        row = rows[int(i)]
        if row.get("pair_id") is None:
            continue
        key = (row["group"], row["pair_id"], row["replicate"], row["remaining"])
        sides = found.setdefault(key, {})
        if row["side"] in sides:
            raise ValueError("a matched pair side has ambiguous multiple records")
        sides[row["side"]] = int(i)
    return [(sides["candidate"], sides["baseline"]) for sides in found.values()
            if "candidate" in sides and "baseline" in sides]


def _pair_weights(rows: list[dict[str, Any]], pairs: list[tuple[int, int]]) -> np.ndarray:
    groups = np.asarray([rows[a]["group"] for a, _ in pairs])
    unique = np.unique(groups)
    return np.asarray([1. / (len(unique) * np.count_nonzero(groups == g)) for g in groups])


def _distances(left: np.ndarray, right: np.ndarray, dimension_weights: np.ndarray) -> np.ndarray:
    # All five context blocks receive equal mass. Long nominal arrays cannot
    # numerically drown the physical state or the solver warmstart.
    with np.errstate(over="ignore", invalid="ignore"):
        delta = left[:, None, :] - right[None, :, :]
        magnitude = np.max(np.abs(delta), axis=2)
        scaled = np.divide(delta, magnitude[:, :, None], out=np.zeros_like(delta),
                           where=magnitude[:, :, None] != 0)
        d = magnitude * np.sqrt(np.sum(scaled ** 2 * dimension_weights, axis=2))
    return np.where(np.isfinite(d), d, np.inf)


def _nearest_distances(left: np.ndarray, query_remaining: np.ndarray, right: np.ndarray,
                       support_remaining: np.ndarray, dimension_weights: np.ndarray,
                       remaining_tolerance: int, embedding: np.ndarray | None = None,
                       squared_norm: np.ndarray | None = None) -> np.ndarray:
    """Exact weighted nearest-context distances with bounded 2-D work arrays.

    The ordinary path uses weighted dot products. Cancellation-scale pairs are
    recomputed by direct differences, preserving near-identical context accuracy.
    The stable direct implementation remains a fallback for extreme magnitudes.
    No (queries, training rows, feature dimension) array is materialized.
    """
    if not len(left):
        return np.empty(0)
    if embedding is None:
        embedding = right * np.sqrt(dimension_weights)
    if squared_norm is None:
        squared_norm = np.einsum("ij,ij->i", embedding, embedding)
    query = left * np.sqrt(dimension_weights)
    query_norm = np.einsum("ij,ij->i", query, query)
    result = np.full(len(left), np.inf)
    eps = np.finfo(np.float64).eps
    for start in range(0, len(right), 1024):
        stop = min(start + 1024, len(right))
        with np.errstate(over="ignore", invalid="ignore"):
            magnitude = query_norm[:, None] + squared_norm[None, start:stop]
            squared = magnitude - 2. * (query @ embedding[start:stop].T)
        needs_direct = (~np.isfinite(squared)) | (squared <= 64. * eps * magnitude)
        for i in np.flatnonzero(np.any(needs_direct, axis=1)):
            indices = np.flatnonzero(needs_direct[i])
            # One query at a time bounds direct fallback memory even when an
            # entire static nominal block makes many contexts nearly identical.
            squared[i, indices] = _distances(left[i:i+1], right[start+indices], dimension_weights)[0] ** 2
        distances = np.sqrt(np.maximum(squared, 0.))
        close_time = np.abs(query_remaining[:, None] - support_remaining[None, start:stop]) <= remaining_tolerance
        result = np.minimum(result, np.min(np.where(close_time, distances, np.inf), axis=1))
    return result


class _Network(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nonnegative without a data-derived upper cap on long-term costs.
        return torch.nn.functional.softplus(self.layers(x).squeeze(-1))


class ReplanningValue:
    """Small deterministic ensemble of measured retained-controller MC returns."""

    @classmethod
    def fit(cls, records: list[dict[str, Any]], *, total_steps: int, gamma: float,
            controller_version: str, seed: int, updates: int = 300, hidden: int = 64,
            ensemble_size: int = 3, pair_weight: float = 1., support_threshold: float = 1.5,
            remaining_tolerance: int | None = None) -> "ReplanningValue":
        total_steps = _integer(total_steps, "total_steps", 1)
        seed, updates = _integer(seed, "seed"), _integer(updates, "updates", 1)
        hidden, ensemble_size = _integer(hidden, "hidden", 1), _integer(ensemble_size, "ensemble_size", 1)
        version = _version(controller_version)
        if not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite in [0,1]")
        if not math.isfinite(pair_weight) or pair_weight < 0:
            raise ValueError("pair_weight must be finite and nonnegative")
        if not math.isfinite(support_threshold) or support_threshold <= 0:
            raise ValueError("support_threshold must be finite and positive")
        if remaining_tolerance is None:
            remaining_tolerance = max(2, int(math.ceil(total_steps * .05)))
        remaining_tolerance = _integer(remaining_tolerance, "remaining_tolerance")
        if not isinstance(records, list) or not records:
            raise ValueError("records must be a nonempty list")
        rows, seen, episode_ancestry = [], set(), {}
        for row in records:
            if not isinstance(row, dict) or row.get("source") != "train":
                raise ValueError("only source='train' records may enter a value fit")
            if row.get("controller_version") != version:
                raise ValueError("mixed or incorrect controller_version in training records")
            n = _integer(row.get("remaining"), "remaining", 1)
            group = _integer(row.get("group"), "group")
            episode = row.get("episode_id")
            if not isinstance(episode, str) or not episode:
                raise ValueError("episode_id must be a nonempty string")
            if episode in episode_ancestry and episode_ancestry[episode] != group:
                raise ValueError("one episode_id cannot cross root ancestry groups")
            episode_ancestry[episode] = group
            row_features = np.asarray(row.get("features"))
            if row_features.ndim != 1:
                raise ValueError("each record requires one full replanning feature vector")
            feature, _ = _features(row_features[None, :], np.array([n]), total_steps)
            try:
                cost = float(row["cost"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("cost must be finite nonnegative measured return") from error
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("cost must be finite nonnegative measured return")
            pair_id, side, replicate = row.get("pair_id"), row.get("side"), row.get("replicate")
            if pair_id is not None:
                if not isinstance(pair_id, str) or not pair_id or side not in ("candidate", "baseline"):
                    raise ValueError("pair records need pair_id and candidate/baseline side")
                replicate = _integer(replicate, "replicate")
            elif side is not None or replicate is not None:
                raise ValueError("side and replicate require pair_id")
            key = (group, episode, n, tuple(feature[0]), cost, pair_id, side, replicate)
            if key not in seen:
                seen.add(key)
                rows.append(dict(features=feature[0], remaining=n, cost=cost, group=group,
                                 episode_id=episode, pair_id=pair_id, side=side, replicate=replicate))
        if len({len(row["features"]) for row in rows}) != 1:
            raise ValueError("all contexts must have the same nominal horizon")
        x = np.asarray([row["features"] for row in rows])
        n = np.asarray([row["remaining"] for row in rows], dtype=np.int64)
        costs = np.asarray([row["cost"] for row in rows])
        groups = np.asarray([row["group"] for row in rows], dtype=np.int64)
        unique = np.unique(groups)
        if len(unique) < 4:
            raise ValueError("require at least four whole root groups for disjoint holdout")
        held_count = min(len(unique) - 2, max(2, int(math.ceil(.25 * len(unique)))))
        held = np.flatnonzero(np.isin(groups, unique[-held_count:]))
        train = np.flatnonzero(~np.isin(groups, unique[-held_count:]))
        train_weights, held_weights = _weights(rows, train), _weights(rows, held)
        normalizers = _normalizers(float(gamma), n)
        targets = costs / normalizers
        if not np.isfinite(targets).all() or np.max(targets[train]) > 1e12:
            raise ValueError("normalized costs exceed finite training numerical range")
        mean = np.sum(x[train] * train_weights[:, None], axis=0)
        deviation = x[train] - mean
        # Scale before squaring to protect finite but very large physical states.
        largest = np.max(np.abs(deviation), axis=0)
        scaled = np.divide(deviation, largest, out=np.zeros_like(deviation), where=largest != 0)
        scale = largest * np.sqrt(np.sum(scaled ** 2 * train_weights[:, None], axis=0))
        scale = np.where(scale >= 1e-4, scale, 1.)
        with np.errstate(over="ignore", invalid="ignore"):
            z = (x - mean) / scale
        if not np.isfinite(z).all() or np.max(np.abs(z)) > 1e12:
            raise ValueError("context standardization exceeds finite numerical range")
        train_pairs, held_pairs = _pairs(rows, train), _pairs(rows, held)
        dimension_weights = np.r_[np.full(17, .2 / 17), np.full(9, .2 / 9),
                                  np.full(x.shape[1] - 28, .2 / (x.shape[1] - 28)), .2, .2]
        result = cls()
        result.feature_dimension, result.hidden = x.shape[1], hidden
        result.ensemble_size, result.total_steps = ensemble_size, total_steps
        result.gamma, result.controller_version = float(gamma), version
        result.mean, result.scale = mean.copy(), scale.copy()
        result.support_features, result.support_remaining = z[train].copy(), n[train].copy()
        result.dimension_weights = dimension_weights
        result._cache_support()
        result.support_threshold, result.remaining_tolerance = float(support_threshold), remaining_tolerance
        result.networks = []
        # Train-only scale conditions optimization; it never caps predictions.
        loss_scale = max(1., float(np.max(targets[train])))
        xt = torch.as_tensor(z, dtype=torch.float32)
        yt = torch.as_tensor(targets / loss_scale, dtype=torch.float32)
        train_tensor = torch.as_tensor(train, dtype=torch.long)
        pair_indices = np.asarray(train_pairs, dtype=np.int64).reshape(-1, 2)
        pair_weights = _pair_weights(rows, train_pairs) if train_pairs else np.empty(0)
        # Independent initializations and root-level Bayesian bootstrap. Dense
        # labels share episode/root weights rather than acting as independent roots.
        with torch.random.fork_rng(devices=[]):
            for member in range(ensemble_size):
                torch.manual_seed(seed + 1009 * member)
                rng = np.random.default_rng(seed + 1009 * member)
                network = _Network(x.shape[1], hidden)
                optimizer = torch.optim.Adam(network.parameters(), lr=2e-3)
                root_bootstrap = dict(zip(unique[:-held_count], rng.exponential(size=len(unique) - held_count)))
                bootstrap = np.asarray([root_bootstrap[groups[i]] for i in train]) * train_weights
                bootstrap /= bootstrap.sum()
                weights_t = torch.as_tensor(bootstrap, dtype=torch.float32)
                if train_pairs:
                    a, b = torch.as_tensor(pair_indices.T, dtype=torch.long)
                    pair_t = torch.as_tensor(pair_weights, dtype=torch.float32)
                    delta_t = yt[a] - yt[b]
                for _ in range(updates):
                    # At this small fixed dataset budget, full-batch gradients
                    # make deterministic root/episode balancing explicit.
                    prediction = network(xt)
                    loss = torch.sum(weights_t * (prediction[train_tensor] / loss_scale - yt[train_tensor]) ** 2)
                    if train_pairs and pair_weight:
                        delta = (prediction[a] - prediction[b]) / loss_scale
                        loss = loss + pair_weight * torch.sum(pair_t * (delta - delta_t) ** 2)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite measured return regression loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), 10.)
                    optimizer.step()
                result.networks.append(network.eval().requires_grad_(False))
        result.uncertainty_floor = 0.
        details = result.predict_details(x, n, controller_version=version)
        predicted = details["cost"]
        errors = predicted / normalizers - targets
        result.uncertainty_floor = _rms(errors[held], held_weights)
        train_mean = float(np.sum(train_weights * targets[train]))
        metrics: dict[str, Any] = dict(model_kind="replanning_mc_ensemble",
            controller_version=version, ensemble_size=ensemble_size, hidden=hidden,
            fit_parameter_count=sum(p.numel() for net in result.networks for p in net.parameters()),
            input_record_count=len(records), deduplicated_record_count=len(rows),
            exact_duplicate_count=len(records) - len(rows), training_count=len(train), holdout_count=len(held),
            training_groups=unique[:-held_count].tolist(), holdout_groups=unique[-held_count:].tolist(),
            training_group_count=len(unique) - held_count, holdout_group_count=held_count,
            training_episode_count=len({rows[i]["episode_id"] for i in train}),
            holdout_episode_count=len({rows[i]["episode_id"] for i in held}),
            training_pair_count=len(train_pairs), holdout_pair_count=len(held_pairs),
            pair_difference_weight=float(pair_weight), training_normalized_rmse=_rms(errors[train], train_weights),
            holdout_normalized_rmse=result.uncertainty_floor,
            holdout_raw_rmse=_rms((predicted - costs)[held], held_weights),
            holdout_normalized_mae=float(np.sum(held_weights * np.abs(errors[held]))),
            holdout_normalized_bias=float(np.sum(held_weights * errors[held])),
            normalized_training_mean=train_mean,
            holdout_training_mean_normalized_rmse=_rms(train_mean - targets[held], held_weights),
            holdout_supported_fraction=float(np.sum(held_weights * details["supported"][held])),
            holdout_ensemble_std_normalized_rms=_rms(details["ensemble_std"][held] / normalizers[held], held_weights),
            uncertainty_normalized_residual_floor=result.uncertainty_floor,
            uncertainty_rule="sqrt(ensemble variance + held-root normalized residual RMSE squared); diagnostic only",
            split_rule="newest 25% whole root ancestry, minimum two held roots",
            fit_weighting="equal roots, equal episodes within root, equal unique dense labels within episode; root-bootstrap ensemble",
            support_rule="nearest train-only standardized context; equal mass physical/warmstart/nominal/flag/time; heuristic",
            target_source="complete measured returns of frozen retained-state replanning controller; source=train only; no bootstrap",
            normalizer="5*sum(gamma**j,j=0..remaining-1); numerical scale, not cost bound",
            output_cap=None, all_finite=bool(np.isfinite(predicted).all()))
        metrics["holdout_beats_training_mean"] = bool(metrics["holdout_normalized_rmse"] < metrics["holdout_training_mean_normalized_rmse"])
        metrics.update(holdout_pair_normalized_difference_rmse=None, holdout_pair_raw_difference_rmse=None,
                       holdout_pair_rank_count=0, holdout_pair_rank_accuracy=None)
        if held_pairs:
            a, b = np.asarray(held_pairs).T
            observed = targets[a] - targets[b]
            estimated = predicted[a] / normalizers[a] - predicted[b] / normalizers[b]
            informative = np.abs(observed) > 1e-12
            pw = _pair_weights(rows, held_pairs)
            metrics.update(holdout_pair_normalized_difference_rmse=_rms(estimated - observed, pw),
                holdout_pair_raw_difference_rmse=_rms((predicted[a] - predicted[b]) - (costs[a] - costs[b]), pw),
                holdout_pair_rank_count=int(informative.sum()),
                holdout_pair_rank_accuracy=(float(np.average((estimated[informative] * observed[informative]) > 0,
                                                            weights=pw[informative])) if informative.any() else None))
        result.metrics = metrics
        return result

    def _cache_support(self) -> None:
        self._support_embedding = np.ascontiguousarray(self.support_features * np.sqrt(self.dimension_weights))
        self._support_squared_norm = np.einsum("ij,ij->i", self._support_embedding, self._support_embedding)

    def predict_details(self, features: Any, remaining: Any, *, controller_version: str | None = None) -> dict[str, np.ndarray]:
        if controller_version is not None and controller_version != self.controller_version:
            raise ValueError("prediction controller_version differs from fitted continuation controller")
        x, n = _features(features, remaining, self.total_steps, self.feature_dimension)
        if not len(x):
            return dict(cost=np.empty(0), supported=np.empty(0, dtype=bool), distance=np.empty(0),
                        ensemble_std=np.empty(0), uncertainty=np.empty(0))
        with np.errstate(over="ignore", invalid="ignore"):
            z = (x - self.mean) / self.scale
        if not np.isfinite(z).all() or np.max(np.abs(z)) > 1e12:
            raise ValueError("query standardization exceeds finite numerical range")
        with torch.no_grad():
            xt = torch.as_tensor(z, dtype=torch.float32)
            members = np.asarray([network(xt).numpy().astype(np.float64) for network in self.networks])
        normalized = members.mean(axis=0)
        normalized_std = members.std(axis=0)
        normalizer = _normalizers(self.gamma, n)
        costs, spread = normalized * normalizer, normalized_std * normalizer
        if not np.isfinite(costs).all() or not np.isfinite(spread).all():
            raise FloatingPointError("nonfinite replanning value prediction")
        distances = np.full(len(x), np.inf)
        for start in range(0, len(x), 32):
            stop = min(start + 32, len(x))
            distances[start:stop] = _nearest_distances(z[start:stop], n[start:stop],
                self.support_features, self.support_remaining, self.dimension_weights, self.remaining_tolerance,
                self._support_embedding, self._support_squared_norm)
        supported = distances <= self.support_threshold
        uncertainty = np.hypot(normalized_std, self.uncertainty_floor) * normalizer
        zero = n == 0
        costs[zero], spread[zero], uncertainty[zero], distances[zero], supported[zero] = 0., 0., 0., 0., True
        return dict(cost=costs, supported=supported, distance=distances,
                    ensemble_std=spread, uncertainty=uncertainty)

    def predict(self, features: Any, remaining: Any, *, controller_version: str | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        details = self.predict_details(features, remaining, controller_version=controller_version)
        return details["cost"], details["supported"], details["distance"]

    def packet(self) -> dict[str, Any]:
        return dict(format_version=1, model_kind="replanning_mc_ensemble", feature_dimension=self.feature_dimension,
                    hidden=self.hidden, ensemble_size=self.ensemble_size, total_steps=self.total_steps, gamma=self.gamma,
                    controller_version=self.controller_version, mean=self.mean.copy(), scale=self.scale.copy(),
                    support_features=self.support_features.copy(), support_remaining=self.support_remaining.copy(),
                    dimension_weights=self.dimension_weights.copy(), support_threshold=self.support_threshold,
                    remaining_tolerance=self.remaining_tolerance, uncertainty_floor=self.uncertainty_floor,
                    metrics=copy.deepcopy(self.metrics), networks=[{name: tensor.detach().cpu().numpy().copy()
                    for name, tensor in network.state_dict().items()} for network in self.networks])

    to_packet = packet
    state_dict = packet

    @classmethod
    def from_packet(cls, packet: dict[str, Any]) -> "ReplanningValue":
        if not isinstance(packet, dict) or packet.get("format_version") != 1 or packet.get("model_kind") != "replanning_mc_ensemble":
            raise ValueError("unsupported ReplanningValue packet")
        result = cls()
        for name in ("feature_dimension", "hidden", "ensemble_size", "total_steps", "gamma", "controller_version",
                     "support_threshold", "remaining_tolerance", "uncertainty_floor"):
            setattr(result, name, copy.deepcopy(packet[name]))
        _integer(result.total_steps, "total_steps", 1)
        _integer(result.feature_dimension, "feature_dimension", 34)
        _integer(result.hidden, "hidden", 1)
        _integer(result.ensemble_size, "ensemble_size", 1)
        _integer(result.remaining_tolerance, "remaining_tolerance")
        _version(result.controller_version)
        if (not math.isfinite(result.gamma) or not 0 <= result.gamma <= 1
                or not math.isfinite(result.support_threshold) or result.support_threshold <= 0
                or not math.isfinite(result.uncertainty_floor) or result.uncertainty_floor < 0
                or result.feature_dimension < 34 or (result.feature_dimension - 28) % 6):
            raise ValueError("invalid ReplanningValue packet configuration")
        for name in ("mean", "scale", "support_features", "support_remaining", "dimension_weights"):
            setattr(result, name, np.array(packet[name], copy=True))
        d = result.feature_dimension
        if (result.mean.shape != (d,) or result.scale.shape != (d,) or result.dimension_weights.shape != (d,)
                or result.support_features.ndim != 2 or result.support_features.shape[1] != d
                or not len(result.support_features)
                or result.support_remaining.shape != (len(result.support_features),)
                or result.support_remaining.dtype.kind not in "iu" or np.any(result.support_remaining < 1)
                or np.any(result.support_remaining > result.total_steps)
                or not all(np.isfinite(getattr(result, name)).all() for name in ("mean", "scale", "support_features", "dimension_weights"))
                or np.max(np.abs(result.support_features)) > 1e12
                or np.any(result.scale <= 0) or np.any(result.dimension_weights <= 0)
                or not np.isclose(result.dimension_weights.sum(), 1.)):
            raise ValueError("invalid ReplanningValue packet arrays")
        result._cache_support()
        result.networks = []
        if len(packet["networks"]) != result.ensemble_size:
            raise ValueError("invalid ReplanningValue ensemble size")
        with torch.random.fork_rng(devices=[]):
            for weights in packet["networks"]:
                network = _Network(d, result.hidden)
                if not all(np.isfinite(np.asarray(value)).all() for value in weights.values()):
                    raise ValueError("nonfinite ReplanningValue network weights")
                network.load_state_dict({key: torch.as_tensor(np.array(value, copy=True)) for key, value in weights.items()})
                result.networks.append(network.eval().requires_grad_(False))
        result.metrics = copy.deepcopy(packet["metrics"])
        return result
