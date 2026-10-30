"""Small, evidence-targeted value models for the independent-search pilot.

The model is a global RBF regressor, not an actor. MC labels are complete
returns of a named frozen controller. The POLO-style control instead fits
optimized model-rollout backups; these have deliberately different semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import numpy as np
from scipy.spatial import cKDTree


def features(states, remaining, total_steps):
    states = np.asarray(states, dtype=float)
    remaining = np.broadcast_to(np.asarray(remaining, dtype=float), states.shape[:-1])
    return np.concatenate((states / np.array([1.8, .07]),
                           (remaining / total_steps)[..., None]), axis=-1)


def measured_returns(costs, remaining, complete=True):
    costs = np.asarray(costs, dtype=float)
    if not complete or len(costs) != remaining or not np.isfinite(costs).all():
        raise ValueError("Incomplete traces are not complete-return evidence")
    return np.cumsum(costs[::-1])[::-1]


@dataclass
class Value:
    centers: np.ndarray
    weights: np.ndarray
    train_features: np.ndarray
    total_steps: int
    width: float
    support_radius: float
    target_version: str
    kind: str

    @cached_property
    def support_tree(self):
        return cKDTree(self.train_features)

    def basis(self, x):
        x = np.atleast_2d(x)
        sq = np.sum((x[:, None] - self.centers[None]) ** 2, axis=-1)
        return np.column_stack((np.ones(len(x)), x,
                                np.exp(-sq / (2 * self.width**2))))

    def predict(self, states, remaining):
        states = np.atleast_2d(np.asarray(states, dtype=float))
        remaining = np.broadcast_to(np.asarray(remaining, dtype=float), (len(states),))
        x = features(states, remaining, self.total_steps)
        # Physical range is an exact finite-cost bound, not a success assumption.
        values = np.clip(self.basis(x) @ self.weights, 0., 1.02) * remaining
        distances = self.support_tree.query(x)[0]
        supported = (distances <= self.support_radius) | (remaining == 0)
        goal = states[:, 0] >= .45
        supported |= goal
        values[goal | (remaining == 0)] = 0.
        return dict(values=values, supported=supported, distances=distances)

    @property
    def digest(self):
        h = hashlib.sha256((self.target_version + self.kind).encode())
        for array in (self.centers, self.weights, self.train_features):
            h.update(np.ascontiguousarray(array).tobytes())
        return h.hexdigest()

    def packet(self):
        return dict(centers=self.centers, weights=self.weights,
                    train_features=self.train_features, total_steps=self.total_steps,
                    width=self.width, support_radius=self.support_radius,
                    target_version=self.target_version, kind=self.kind)


def fit_value(states, remaining, targets, groups, *, total_steps, version, kind,
              centers=64, width=.25, ridge=.01, support_radius=.25):
    x = features(states, remaining, total_steps)
    targets = np.asarray(targets, dtype=float)
    remaining = np.asarray(remaining, dtype=float)
    groups = np.asarray(groups)
    if not len(x) or np.any(remaining <= 0) or not np.isfinite(targets).all():
        raise ValueError("No valid complete targets")
    # Root-balanced farthest-point centers. No validation states fit scalers/support.
    unique = np.unique(groups)
    chosen = [int(np.flatnonzero(groups == g)[0]) for g in unique]
    chosen = chosen[:centers]
    nearest = np.min(np.sum((x[:, None] - x[chosen][None])**2, axis=-1), axis=1)
    while len(chosen) < min(centers, len(x)):
        index = int(np.argmax(nearest))
        if nearest[index] < 1e-14:
            break
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((x - x[index])**2, axis=-1))
    result = Value(x[chosen].copy(), np.empty(4+len(chosen)), x.copy(), total_steps,
                   width, support_radius, version, kind)
    phi = result.basis(x)
    counts = {g: np.count_nonzero(groups == g) for g in unique}
    weight = np.array([1. / counts[g] for g in groups])
    weight *= len(x) / weight.sum()
    gram = phi.T @ (phi * weight[:, None])
    regularizer = ridge * np.eye(phi.shape[1])
    regularizer[0, 0] = ridge * .01
    result.weights = np.linalg.solve(gram + regularizer,
                                     phi.T @ (weight * targets / remaining))
    return result


def score_guidance(task, decision, value, kind):
    """Rank a fixed independent bank; return separately charged scratch work.

    MC mode evaluates ONE first action then the actual target controller.
    Terminal mode evaluates H-step commitment plus an approximate optimal value.
    Repeated deployment of either scorer is tested as a whole controller.
    """
    bank = decision["bank"]
    baseline = int(decision["baseline_index"])
    remaining = int(bank["remaining"])
    if kind == "mc":
        actions, inverse = np.unique(bank["actions"], axis=0, return_inverse=True)
        state = np.broadcast_to(bank["state"], (len(actions), 2))
        try:
            next_states, costs, _ = task.transition(state, actions)
            pred = value.predict(next_states, remaining-1)
        except Exception as error:
            error.guidance_work = dict(scratch_controls=len(actions), prediction_rows=len(actions),
                                       failed_queries=1)
            raise
        q = (costs + pred["values"])[inverse]
        support = pred["supported"][inverse]
        distances = pred["distances"][inverse]
        scratch = len(actions)
    elif kind == "polo":
        try:
            pred = value.predict(bank["endpoints"], bank["endpoint_remaining"])
        except Exception as error:
            error.guidance_work = dict(scratch_controls=0, prediction_rows=len(bank["endpoints"]),
                                       failed_queries=1)
            raise
        q = bank["stage_costs"] + pred["values"]
        support, distances = pred["supported"], pred["distances"]
        scratch = 0
    else:
        raise ValueError(kind)
    # Unknown outcomes never receive an optimistic learned preference.
    supported_baseline = bool(support[baseline])
    scores = np.where(support, q, np.inf)
    selected = int(np.argmin(scores)) if supported_baseline else baseline
    # Preserve the baseline on exact ties, including identical first actions.
    if scores[baseline] <= scores[selected] + 1e-12:
        selected = baseline
    if not np.isfinite(q).all():
        selected = baseline
    return dict(selected=selected, scores=q, supported=support,
                support_distances=distances, supported_baseline=supported_baseline,
                scratch_controls=int(scratch), prediction_rows=len(pred["values"]),
                changed=not np.array_equal(bank["actions"][selected],
                                           bank["actions"][baseline]))


def paired_summary(candidate, baseline):
    diff = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    # Exact paired-root bootstrap; descriptive when only four qualification roots.
    rng = np.random.default_rng(90210)
    means = diff[rng.integers(0, len(diff), (4000, len(diff)))].mean(axis=1)
    return dict(mean_difference=float(diff.mean()), wins=int(np.sum(diff < 0)),
                interval80=np.quantile(means, [.1, .9]).tolist(), differences=diff.tolist())


def promotion(candidate, baseline, candidate_success, baseline_success, *, margin=2.):
    info = paired_summary(candidate, baseline)
    info["accepted"] = bool(info["mean_difference"] <= -margin
                            and info["wins"] >= 3
                            and info["interval80"][1] < 0
                            and sum(candidate_success) >= sum(baseline_success))
    return info
