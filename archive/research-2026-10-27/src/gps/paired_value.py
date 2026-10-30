"""Small return model fitted only to measured training continuations.

The predictor estimates a fixed reference controller's complete finite suffix
cost. Paired endpoints add difference observations to an ordinary ridge fit;
they are neither action labels nor Bellman targets. Whole collection roots,
including all their paired endpoints and replicas, stay on one side of a fixed
newest-root holdout. Support and eligibility are empirical diagnostics, not
confidence or safety certificates.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral
from typing import Any

import numpy as np


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _normalizers(gamma: float, remaining: np.ndarray) -> np.ndarray:
    if gamma == 1.:
        return 5. * remaining.astype(np.float64)
    if gamma == 0.:
        return 5. * (remaining > 0).astype(np.float64)
    return 5. * np.expm1(remaining * math.log(gamma)) / math.expm1(math.log(gamma))


def _rms(values: np.ndarray) -> float:
    if not len(values):
        return 0.
    largest = float(np.max(np.abs(values)))
    return largest * float(np.sqrt(np.mean((values / largest) ** 2))) if largest else 0.


def _mean(values: np.ndarray) -> float:
    largest = float(np.max(np.abs(values))) if len(values) else 0.
    return largest * float(np.mean(values / largest)) if largest else 0.


def _distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """RMS distance, scaling first to avoid squaring large query features."""
    with np.errstate(over="ignore", invalid="ignore"):
        delta = left[:, None, :] - right[None, :, :]
        maximum = np.max(np.abs(delta), axis=2)
        scaled = np.divide(delta, maximum[:, :, None],
                           out=np.zeros_like(delta), where=maximum[:, :, None] != 0)
        result = maximum * np.sqrt(np.mean(scaled ** 2, axis=2))
    return np.where(np.isfinite(result), result, np.inf)


def _validated_records(records: list[dict[str, Any]], total_steps: int
                       ) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list of measured training records")
    parsed, seen = [], set()
    for record in records:
        if not isinstance(record, dict) or record.get("source") != "train":
            raise ValueError("Only source='train' records may enter the value fit")
        features = np.asarray(record.get("features"), dtype=np.float64)
        if features.shape != (17,) or not np.isfinite(features).all():
            raise ValueError("Every record needs 17 finite physical features")
        remaining = _integer(record.get("remaining"), "remaining", 1)
        if remaining > total_steps:
            raise ValueError("remaining exceeds total_steps")
        group = _integer(record.get("group"), "group")
        replicate = _integer(record.get("replicate"), "replicate")
        pair_id, side = record.get("pair_id"), record.get("side")
        if not isinstance(pair_id, str) or not pair_id or side not in ("candidate", "baseline"):
            raise ValueError("Every record needs a nonempty pair_id and candidate/baseline side")
        try:
            cost = float(record["cost"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("cost must be a finite nonnegative measured suffix return") from error
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("cost must be a finite nonnegative measured suffix return")
        key = (group, pair_id, side, replicate, tuple(features), remaining, cost)
        if key not in seen:
            seen.add(key)
            parsed.append(dict(features=features.copy(), remaining=remaining, cost=cost,
                               group=group, pair_id=pair_id, side=side, replicate=replicate))
    return parsed, len(records) - len(parsed)


def _pairs(records: list[dict[str, Any]], selected: np.ndarray) -> list[tuple[int, int]]:
    groups: dict[tuple[Any, ...], dict[str, int]] = {}
    for index in np.flatnonzero(selected):
        row = records[int(index)]
        key = (row["group"], row["pair_id"], row["replicate"], row["remaining"])
        sides = groups.setdefault(key, {})
        if row["side"] in sides:
            raise ValueError("Ambiguous endpoint measurement: a pair side has multiple records for one replica")
        sides[row["side"]] = int(index)
    return [(sides["candidate"], sides["baseline"]) for sides in groups.values()
            if "candidate" in sides and "baseline" in sides]


class PairedRidgeValue:
    """Regularized linear return estimate plus measured pair-difference rows."""

    @classmethod
    def fit(cls, records: list[dict[str, Any]], *, total_steps: int, gamma: float,
            ridge: float = 10., support_threshold: float = 1.5,
            remaining_tolerance: int | None = None) -> "PairedRidgeValue":
        total_steps = _integer(total_steps, "total_steps", 1)
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite in [0,1]")
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        if not np.isfinite(support_threshold) or support_threshold <= 0:
            raise ValueError("support_threshold must be positive and finite")
        if remaining_tolerance is None:
            remaining_tolerance = max(2, int(math.ceil(total_steps * .05)))
        remaining_tolerance = _integer(remaining_tolerance, "remaining_tolerance")
        rows, duplicates = _validated_records(records, total_steps)
        features = np.asarray([row["features"] for row in rows])
        remaining = np.asarray([row["remaining"] for row in rows], dtype=np.int64)
        costs = np.asarray([row["cost"] for row in rows])
        groups = np.asarray([row["group"] for row in rows], dtype=np.int64)
        unique = np.unique(groups)
        if len(unique) < 4:
            raise ValueError("Require at least four whole collection-root groups for a disjoint split")
        held_count = min(len(unique) - 2, max(2, int(math.ceil(len(unique) * .25))))
        held = np.isin(groups, unique[-held_count:])
        train = ~held
        normalizers = _normalizers(float(gamma), remaining)
        targets = costs / normalizers
        if not np.isfinite(targets).all():
            raise ValueError("Normalized measured costs must remain finite")
        cap = max(1., 1.5 * float(np.max(targets[train])))
        if not math.isfinite(cap):
            raise ValueError("Training costs exceed the representable normalized range")
        inputs = np.column_stack((features, remaining / total_steps))
        mean = inputs[train].mean(axis=0)
        scale = inputs[train].std(axis=0)
        scale = np.where(scale >= 1e-4, scale, 1.)
        with np.errstate(over="ignore", invalid="ignore"):
            standardized = (inputs - mean) / scale
        if not np.isfinite(standardized).all():
            raise ValueError("Feature standardization exceeded the finite numerical range")
        train_pairs, held_pairs = _pairs(rows, train), _pairs(rows, held)
        design = np.column_stack((standardized[train], np.ones(int(train.sum()))))
        responses = targets[train].copy()
        difference_rows, differences = [], []
        for candidate, baseline in train_pairs:
            delta = standardized[candidate] - standardized[baseline]
            # An identical endpoint adds no identifiable difference information.
            if np.any(delta != 0):
                difference_rows.append(np.r_[delta, 0.])
                differences.append(targets[candidate] - targets[baseline])
        if difference_rows:
            design = np.vstack((design, difference_rows))
            responses = np.r_[responses, differences]
        # Unit weight for each measured return and each distinct pair contrast.
        # Scale targets before solving to avoid overflowing normal equations.
        penalty = np.column_stack((math.sqrt(float(ridge)) * np.eye(18), np.zeros(18)))
        weights = np.linalg.lstsq(np.vstack((design, penalty)),
                                 np.r_[responses / cap, np.zeros(18)], rcond=None)[0] * cap
        if not np.isfinite(weights).all():
            raise FloatingPointError("Nonfinite paired ridge coefficients")
        result = cls()
        result.feature_dimension = 17
        result.total_steps, result.gamma = total_steps, float(gamma)
        result.ridge, result.output_cap = float(ridge), cap
        result.mean, result.scale, result.weights = mean.copy(), scale.copy(), weights.copy()
        result.support_features, result.support_remaining = standardized[train].copy(), remaining[train].copy()
        result.support_threshold = float(support_threshold)
        result.remaining_tolerance = remaining_tolerance
        prediction, support, distances = result.predict(features, remaining)
        predicted_targets = prediction / normalizers
        errors = predicted_targets - targets
        training_mean = _mean(targets[train])
        mean_errors = training_mean - targets[held]
        # A small fixed KNN audit uses the exact same train-only coordinates.
        knn_distances = _distances(standardized[held], standardized[train])
        neighbors = np.argsort(knn_distances, axis=1, kind="stable")[:, :min(5, int(train.sum()))]
        knn_targets = []
        for row_index, indices in enumerate(neighbors):
            d = knn_distances[row_index, indices]
            zero = d <= 1e-12
            weights_knn = zero.astype(float) if zero.any() else 1. / np.maximum(d, 1e-12)
            if not np.isfinite(weights_knn).all() or weights_knn.sum() == 0:
                knn_targets.append(training_mean)
            else:
                weights_knn /= weights_knn.sum()
                knn_targets.append(float(np.dot(weights_knn, targets[train][indices])))
        knn_errors = np.asarray(knn_targets) - targets[held]
        held_distances = distances[held & np.isfinite(distances)]
        holdout_rmse = _rms(errors[held])
        finite = bool(np.isfinite(prediction).all() and np.isfinite(errors).all())
        support_fraction = float(np.mean(support[held]))
        pair_metrics: dict[str, Any] = dict(holdout_pair_count=len(held_pairs),
            holdout_pair_normalized_difference_rmse=None, holdout_pair_raw_difference_rmse=None,
            holdout_pair_rank_count=0, holdout_pair_rank_correct_count=0,
            holdout_pair_rank_accuracy=None, holdout_pair_supported_fraction=None)
        if held_pairs:
            candidate, baseline = np.asarray(held_pairs, dtype=np.int64).T
            observed_delta = targets[candidate] - targets[baseline]
            predicted_delta = predicted_targets[candidate] - predicted_targets[baseline]
            informative = np.abs(observed_delta) > 1e-12
            correct = informative & (predicted_delta * np.sign(observed_delta) > 0)
            pair_metrics.update(
                holdout_pair_normalized_difference_rmse=_rms(predicted_delta - observed_delta),
                holdout_pair_raw_difference_rmse=_rms(
                    (prediction[candidate] - prediction[baseline]) - (costs[candidate] - costs[baseline])),
                holdout_pair_rank_count=int(informative.sum()),
                holdout_pair_rank_correct_count=int(correct.sum()),
                holdout_pair_rank_accuracy=(float(correct.sum() / informative.sum()) if informative.any() else None),
                holdout_pair_supported_fraction=float(np.mean(support[candidate] & support[baseline])))
        result.metrics = dict(model_kind="paired_ridge", training_count=int(train.sum()),
            holdout_count=int(held.sum()), input_record_count=len(records), deduplicated_record_count=len(rows),
            exact_duplicate_count=duplicates, training_groups=unique[:-held_count].tolist(),
            holdout_groups=unique[-held_count:].tolist(), training_group_count=len(unique) - held_count,
            holdout_group_count=held_count, training_pair_count=len(train_pairs),
            training_pair_difference_rows=len(difference_rows), pair_difference_weight=1.,
            fit_row_count=len(design), fit_parameter_count=19, ridge=float(ridge),
            training_normalized_rmse=_rms(errors[train]), holdout_normalized_rmse=holdout_rmse,
            holdout_raw_rmse=_rms(prediction[held] - costs[held]),
            holdout_normalized_mae=_mean(np.abs(errors[held])),
            holdout_normalized_bias=_mean(errors[held]), normalized_training_mean=training_mean,
            holdout_training_mean_normalized_rmse=_rms(mean_errors),
            holdout_beats_training_mean=bool(holdout_rmse < _rms(mean_errors)),
            holdout_knn_normalized_rmse=_rms(knn_errors),
            holdout_beats_knn=bool(holdout_rmse < _rms(knn_errors)),
            holdout_supported_fraction=support_fraction,
            holdout_unsupported_count=int((~support[held]).sum()),
            holdout_distance_mean=(_mean(held_distances) if len(held_distances) else None),
            calibration_eligible=bool(train.sum() >= 8 and held.sum() >= 3 and finite
                                      and support_fraction >= .5 and holdout_rmse <= .35),
            all_finite=finite, normalized_output_cap=cap, output_cap_source="training labels only",
            normalizer="5*sum(gamma**j,j=0..remaining-1); scale only, not a return bound",
            target_source="measured complete fixed-reference suffix returns; no bootstrap; source=train only",
            split_rule="newest 25% whole collection-root groups, at least two held roots",
            support_rule="RMS standardized nearest training query with remaining tolerance; heuristic",
            **pair_metrics)
        return result

    def predict(self, features: np.ndarray, remaining: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.asarray(features, dtype=np.float64)
        remaining = np.asarray(remaining)
        if features.ndim != 2 or features.shape[1] != 17 or not np.isfinite(features).all():
            raise ValueError("features must be a finite (M,17) matrix")
        if (remaining.shape != (len(features),) or remaining.dtype.kind not in "iu"
                or np.any(remaining < 0) or np.any(remaining > self.total_steps)):
            raise ValueError("remaining must be aligned integers within [0,total_steps]")
        if not len(features):
            return np.empty(0), np.empty(0, dtype=bool), np.empty(0)
        standardized = (np.column_stack((features, remaining / self.total_steps)) - self.mean) / self.scale
        if not np.isfinite(standardized).all():
            raise ValueError("Query standardization exceeded the finite numerical range")
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = np.clip(standardized @ self.weights[:-1] + self.weights[-1], 0., self.output_cap)
        if not np.isfinite(normalized).all():
            raise FloatingPointError("Value prediction exceeded the finite numerical range")
        costs = normalized * _normalizers(self.gamma, remaining)
        distances = np.full(len(features), np.inf)
        for start in range(0, len(features), 64):
            stop = min(start + 64, len(features))
            matrix = _distances(standardized[start:stop], self.support_features)
            close = np.abs(remaining[start:stop, None] - self.support_remaining[None, :]) <= self.remaining_tolerance
            distances[start:stop] = np.min(np.where(close, matrix, np.inf), axis=1)
        supported = np.isfinite(costs) & (distances <= self.support_threshold)
        costs[remaining == 0], supported[remaining == 0], distances[remaining == 0] = 0., True, 0.
        return costs, supported, distances

    def packet(self) -> dict[str, Any]:
        return dict(format_version=1, model_kind="paired_ridge", feature_dimension=self.feature_dimension,
            total_steps=self.total_steps, gamma=self.gamma, ridge=self.ridge,
            mean=self.mean.copy(), scale=self.scale.copy(), weights=self.weights.copy(),
            support_features=self.support_features.copy(), support_remaining=self.support_remaining.copy(),
            output_cap=self.output_cap, support_threshold=self.support_threshold,
            remaining_tolerance=self.remaining_tolerance, metrics=copy.deepcopy(self.metrics))

    to_packet = packet
    state_dict = packet

    @classmethod
    def from_packet(cls, packet: dict[str, Any]) -> "PairedRidgeValue":
        if packet.get("format_version") != 1 or packet.get("model_kind") != "paired_ridge":
            raise ValueError("Unsupported PairedRidgeValue packet")
        result = cls()
        for name in ("feature_dimension", "total_steps", "gamma", "ridge", "output_cap",
                     "support_threshold", "remaining_tolerance"):
            setattr(result, name, packet[name])
        for name in ("mean", "scale", "weights", "support_features", "support_remaining"):
            setattr(result, name, np.array(packet[name], copy=True))
        result.metrics = copy.deepcopy(packet["metrics"])
        if (result.feature_dimension != 17 or result.mean.shape != (18,)
                or result.scale.shape != (18,) or result.weights.shape != (19,)
                or result.support_features.ndim != 2 or result.support_features.shape[1] != 18
                or not len(result.support_features)
                or result.support_remaining.shape != (len(result.support_features),)
                or result.support_remaining.dtype.kind not in "iu"
                or not all(np.isfinite(getattr(result, key)).all()
                           for key in ("mean", "scale", "weights", "support_features"))
                or np.any(result.scale <= 0)):
            raise ValueError("Invalid paired ridge packet arrays")
        _integer(result.total_steps, "total_steps", 1)
        _integer(result.remaining_tolerance, "remaining_tolerance")
        if (not 0 <= result.gamma <= 1 or not math.isfinite(result.gamma)
                or not math.isfinite(result.ridge) or result.ridge <= 0
                or not math.isfinite(result.output_cap) or result.output_cap < 1
                or not math.isfinite(result.support_threshold) or result.support_threshold <= 0
                or np.any(result.support_remaining < 1)
                or np.any(result.support_remaining > result.total_steps)):
            raise ValueError("Invalid paired ridge packet configuration")
        return result
