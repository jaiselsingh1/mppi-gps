"""Critic-free and measured-continuation learning primitives.

This module has no simulator or controller dependency and performs zero
physics queries. Values are Monte Carlo regressions of complete, externally
measured frozen-actor returns. A value's target actor is part of its identity.
No MPPI actions, TD targets, or optimal-value claims enter these routines.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json

import numpy as np
from scipy.spatial import cKDTree


TARGET_SEMANTICS = "complete Monte Carlo cost-to-go of the named frozen base actor; no bootstrap"
COST_UPPER_BOUND = 25.0


def _finite_array(value, name, *, ndim=None):
    array = np.asarray(value, dtype=np.float64)
    if (ndim is not None and array.ndim != ndim) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array" + (f" with {ndim} dimensions" if ndim is not None else ""))
    return array


def generate_candidates(parameters, seed, pairs=8, sigma=.3):
    """Common private-RNG Gaussian directions; candidates interleave + and -."""
    base = _finite_array(parameters, "parameters", ndim=1)
    if base.shape != (3,):
        raise ValueError("The ending actor has exactly three parameters")
    if not isinstance(pairs, (int, np.integer)) or isinstance(pairs, (bool, np.bool_)) or pairs < 1:
        raise ValueError("pairs must be a positive integer")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    eps = np.random.default_rng(seed).standard_normal((pairs, len(base)))
    candidates = np.empty((2*pairs, len(base)), dtype=np.float64)
    candidates[0::2] = base + sigma*eps
    candidates[1::2] = base - sigma*eps
    return dict(candidates=candidates, eps=eps, base=base.copy(), sigma=float(sigma), seed=int(seed))


def _candidate_means(costs, pairs):
    costs = _finite_array(costs, "costs")
    if costs.ndim < 1 or costs.shape[0] != 2*pairs or costs.size == 0:
        raise ValueError("costs must have one interleaved +/- row per candidate")
    return costs.reshape(2*pairs, -1).mean(axis=1)


def es_gradient(eps, costs, sigma, *, cost_scale=20.):
    """Symmetric Gaussian ES cost gradient with a fixed physical cost scale.

    Additional cost axes are paired-root samples and are averaged, rather than
    treated as independent parameter directions. No rank transform or sample
    standardization changes this estimator.
    """
    eps = _finite_array(eps, "eps", ndim=2)
    if eps.shape[1:] != (3,) or not len(eps):
        raise ValueError("eps must have shape (positive pairs, 3)")
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("sigma and cost_scale must be positive and finite")
    means = _candidate_means(costs, len(eps))
    difference = means[0::2] - means[1::2]
    return np.sum(difference[:, None]*eps, axis=0)/(2*len(eps)*sigma*cost_scale)


def propose_update(base, eps, costs, sigma, lr=.4, maxnorm=.35, cost_scale=20.):
    """Return a bounded cost-descending ES proposal; acceptance is external."""
    base = _finite_array(base, "base", ndim=1)
    if base.shape != (3,):
        raise ValueError("The ending actor has exactly three parameters")
    if not np.isfinite(lr) or lr <= 0 or not np.isfinite(maxnorm) or maxnorm <= 0:
        raise ValueError("lr and maxnorm must be positive and finite")
    gradient = es_gradient(eps, costs, sigma, cost_scale=cost_scale)
    raw = -lr*gradient
    norm = float(np.linalg.norm(raw))
    clipped = norm > maxnorm
    step = raw*(maxnorm/norm) if clipped else raw.copy()
    return dict(parameters=base+step, gradient=gradient, step=step,
                unclipped_step=raw, unclipped_norm=norm,
                step_norm=float(np.linalg.norm(step)), clipped=bool(clipped),
                sigma=float(sigma), lr=float(lr), cost_scale=float(cost_scale))


def complete_returns(costs, *, remaining=None, complete=True):
    """Measured undiscounted suffix sums, with no invented terminal value.

    The caller supplies a complete finite-horizon trajectory. ``remaining``
    optionally checks its declared initial remaining controls; ``complete``
    rejects an explicitly truncated record. Length/completion cannot be
    inferred from costs alone. The terminal boundary return is implicitly 0
    and is not appended as a fictitious transition.
    """
    costs = _finite_array(costs, "costs", ndim=1)
    if not complete:
        raise ValueError("Truncated trajectories are not complete-return targets")
    if remaining is not None:
        if (not isinstance(remaining, (int, np.integer)) or isinstance(remaining, (bool, np.bool_))
                or remaining < 0 or remaining != len(costs)):
            raise ValueError("The complete trace must exhaust its declared remaining controls")
    return np.cumsum(costs[::-1])[::-1]


def _states(states, total_steps):
    states = _finite_array(states, "states")
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != 2:
        raise ValueError("states must have shape (n,2) for [time,y]")
    if np.any(states[:, 0] < 0) or np.any(states[:, 0] > total_steps) or np.any(np.abs(states[:, 1]) > 1.+1e-12):
        raise ValueError("states are outside the ending task's time/y domain")
    return states


def features(states, total_steps):
    """Fixed physical scaling; no heldout data determines the representation."""
    states = _states(states, total_steps)
    return states/np.array([float(total_steps), 2.])


@dataclass
class MCValue:
    centers: np.ndarray
    weights: np.ndarray
    train_features: np.ndarray
    train_groups: np.ndarray
    total_steps: int
    gate: int
    ending_y: float
    target_actor_hash: str
    width: float = .25
    ridge: float = .01
    support_radius: float = .35

    @cached_property
    def support_tree(self):
        return cKDTree(self.train_features)

    @property
    def target_semantics(self):
        return TARGET_SEMANTICS

    @property
    def fit_updates(self):
        return 1

    @property
    def training_rows(self):
        return len(self.train_features)

    def basis(self, states):
        states = _states(states, self.total_steps)
        x = features(states, self.total_steps)
        sq = np.sum((x[:, None]-self.centers[None])**2, axis=-1)
        post_gate = (states[:, 0] >= self.gate).astype(float)
        bad_lane = (states[:, 1] > self.ending_y).astype(float)
        # Known physical/event indicators, with coefficients fitted only from
        # measured labels. No known penalty or desired return is inserted.
        return np.column_stack((np.ones(len(x)), x, post_gate, bad_lane,
                                post_gate*bad_lane,
                                np.exp(-sq/(2*self.width**2))))

    def predict(self, states):
        states = _states(states, self.total_steps)
        x = features(states, self.total_steps)
        values = np.clip(self.basis(states) @ self.weights, 0., COST_UPPER_BOUND)
        distances = self.support_tree.query(x)[0]
        terminal = states[:, 0] == self.total_steps
        values[terminal] = 0.
        supported = (distances <= self.support_radius) | terminal
        return dict(values=values, supported=supported, distances=distances,
                    prediction_rows=len(states), simulated_controls=0,
                    target_actor_hash=self.target_actor_hash,
                    target_semantics=self.target_semantics)

    def score_predict(self, states):
        return self.predict(states)

    @property
    def digest(self):
        meta = dict(total_steps=self.total_steps, gate=self.gate, ending_y=self.ending_y,
                    target_actor_hash=self.target_actor_hash, width=self.width,
                    ridge=self.ridge, support_radius=self.support_radius,
                    target_semantics=self.target_semantics,
                    feature_schema="[t/T,y/2]; intercept,linear,postgate,badlane,postgate*badlane,RBF")
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode())
        for array in (self.centers, self.weights, self.train_features):
            array = np.ascontiguousarray(array, dtype="<f8")
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        digest.update(json.dumps(self.train_groups.tolist(), ensure_ascii=True).encode())
        return digest.hexdigest()

    def packet(self):
        """All identity/representation arrays are safe for allow_pickle=False."""
        return dict(centers=self.centers.copy(), weights=self.weights.copy(),
                    train_features=self.train_features.copy(), train_groups=self.train_groups.copy(),
                    total_steps=self.total_steps, gate=self.gate, ending_y=self.ending_y,
                    target_actor_hash=self.target_actor_hash, width=self.width, ridge=self.ridge,
                    support_radius=self.support_radius, target_semantics=self.target_semantics,
                    cost_upper_bound=COST_UPPER_BOUND, fit_updates=1,
                    training_rows=self.training_rows, simulated_controls=0, digest=self.digest)

    @classmethod
    def from_packet(cls, packet):
        if str(np.asarray(packet["target_semantics"]).item()) != TARGET_SEMANTICS:
            raise ValueError("Unsupported target semantics")
        result = cls(centers=np.array(packet["centers"], dtype=float, copy=True),
                     weights=np.array(packet["weights"], dtype=float, copy=True),
                     train_features=np.array(packet["train_features"], dtype=float, copy=True),
                     train_groups=np.array(packet["train_groups"], dtype=str, copy=True),
                     total_steps=int(packet["total_steps"]), gate=int(packet["gate"]),
                     ending_y=float(packet["ending_y"]),
                     target_actor_hash=str(np.asarray(packet["target_actor_hash"]).item()),
                     width=float(packet["width"]), ridge=float(packet["ridge"]),
                     support_radius=float(packet["support_radius"]))
        if "digest" in packet and result.digest != str(np.asarray(packet["digest"]).item()):
            raise ValueError("Saved value digest does not match its contents")
        return result


def fit_value(states, targets, groups, *, total_steps, gate, ending_y,
              target_actor_hash, centers=48, width=.25, ridge=.01, support_radius=.35):
    """Root-balanced regression of externally measured frozen-actor labels.

    Each root has total regression weight one. Duplicating all suffix rows of
    a root therefore does not make that root more influential or weaken the
    fixed regularizer. Targets' completion and actor provenance must be
    established by the caller's raw trajectory records.
    """
    if (not isinstance(total_steps, (int, np.integer)) or isinstance(total_steps, (bool, np.bool_)) or total_steps < 1
            or not isinstance(gate, (int, np.integer)) or isinstance(gate, (bool, np.bool_)) or not 0 <= gate <= total_steps):
        raise ValueError("total_steps and gate must be valid integer task times")
    if not np.isfinite(ending_y) or not -1 <= ending_y <= 1:
        raise ValueError("ending_y must be in [-1,1]")
    if not isinstance(target_actor_hash, str) or not target_actor_hash:
        raise ValueError("A nonempty frozen target actor hash is required")
    if not isinstance(centers, (int, np.integer)) or isinstance(centers, (bool, np.bool_)) or centers < 1:
        raise ValueError("centers must be a positive integer")
    if (not np.isfinite(width) or width <= 0 or not np.isfinite(ridge) or ridge <= 0
            or not np.isfinite(support_radius) or support_radius <= 0):
        raise ValueError("width, ridge and support_radius must be positive and finite")
    states = _states(states, total_steps)
    targets = _finite_array(targets, "targets", ndim=1)
    groups = np.asarray(groups)
    if groups.ndim != 1 or len(targets) != len(states) or len(groups) != len(states) or not len(states):
        raise ValueError("Each training state requires one target and one root group")
    if np.any(targets < 0) or np.any(targets > COST_UPPER_BOUND):
        raise ValueError("Measured targets violate the task's exact finite-cost bound")
    terminal = states[:, 0] == total_steps
    if np.any(targets[terminal] != 0):
        raise ValueError("A completed terminal state's remaining return is exactly zero")
    live = ~terminal
    if not live.any():
        raise ValueError("At least one nonterminal measured target is required")
    states, targets, groups = states[live], targets[live], groups[live].astype(str)
    x = features(states, total_steps)
    roots = np.unique(groups)
    chosen = [int(np.flatnonzero(groups == group)[0]) for group in roots][:centers]
    nearest = np.min(np.sum((x[:, None]-x[chosen][None])**2, axis=-1), axis=1)
    while len(chosen) < min(centers, len(x)):
        index = int(np.argmax(nearest))
        if nearest[index] < 1e-14:
            break
        chosen.append(index)
        nearest = np.minimum(nearest, np.sum((x-x[index])**2, axis=-1))
    result = MCValue(x[chosen].copy(), np.empty(6+len(chosen)), x.copy(), groups.copy(),
                     int(total_steps), int(gate), float(ending_y), target_actor_hash,
                     float(width), float(ridge), float(support_radius))
    phi = result.basis(states)
    counts = {group: np.count_nonzero(groups == group) for group in roots}
    weights = np.array([1./counts[group] for group in groups])
    regularizer = ridge*np.eye(phi.shape[1])
    regularizer[0, 0] = ridge*.01
    result.weights = np.linalg.solve(phi.T @ (weights[:, None]*phi) + regularizer,
                                     phi.T @ (weights*targets))
    return result


def gradient_alignment(first, second):
    first = _finite_array(first, "first gradient", ndim=1)
    second = _finite_array(second, "second gradient", ndim=1)
    if first.shape != second.shape:
        raise ValueError("Gradient shapes must match")
    product = float(np.linalg.norm(first)*np.linalg.norm(second))
    return dict(cosine=(float(np.clip(np.dot(first, second)/product, -1., 1.)) if product else None),
                first_norm=float(np.linalg.norm(first)), second_norm=float(np.linalg.norm(second)),
                difference_norm=float(np.linalg.norm(first-second)))


def compare_scores(full_candidate, exact_hybrid, approximate_hybrid, *, eps=None, sigma=.3, cost_scale=20.):
    """Separate continuation mismatch (B-A) and value error (C-B).

    Inputs have interleaved +/- candidates on axis zero; optional further
    axes are paired heldout reset roots. Pair differences, not only absolute
    values, expose errors that can reverse a parameter-search direction.
    """
    a = _finite_array(full_candidate, "full_candidate")
    b = _finite_array(exact_hybrid, "exact_hybrid")
    c = _finite_array(approximate_hybrid, "approximate_hybrid")
    if a.shape != b.shape or a.shape != c.shape or a.ndim < 1 or not a.size or a.shape[0] % 2:
        raise ValueError("A, B, C need matching nonempty interleaved +/- candidate shapes")
    pair_a, pair_b, pair_c = (value[0::2]-value[1::2] for value in (a, b, c))

    def discrepancy(predicted, target):
        errors = predicted-target
        nonzero = target != 0
        return dict(errors=errors, rmse=float(np.sqrt(np.mean(errors**2))),
                    sign_accuracy=float(np.mean(np.sign(predicted) == np.sign(target))),
                    nonzero_target_sign_accuracy=(float(np.mean(np.sign(predicted[nonzero]) == np.sign(target[nonzero])))
                                                  if nonzero.any() else None),
                    nonzero_target_pairs=int(nonzero.sum()))
    result = dict(full_candidate=a, exact_hybrid=b, approximate_hybrid=c,
                  continuation_error=b-a, critic_error=c-b, combined_error=c-a,
                  continuation_rmse=float(np.sqrt(np.mean((b-a)**2))),
                  critic_rmse=float(np.sqrt(np.mean((c-b)**2))),
                  combined_rmse=float(np.sqrt(np.mean((c-a)**2))),
                  full_pair_differences=pair_a, exact_hybrid_pair_differences=pair_b,
                  approximate_hybrid_pair_differences=pair_c,
                  continuation_pair_error=discrepancy(pair_b, pair_a),
                  critic_pair_error=discrepancy(pair_c, pair_b),
                  combined_pair_error=discrepancy(pair_c, pair_a),
                  simulated_controls=0)
    if eps is not None:
        ga = es_gradient(eps, a, sigma, cost_scale=cost_scale)
        gb = es_gradient(eps, b, sigma, cost_scale=cost_scale)
        gc = es_gradient(eps, c, sigma, cost_scale=cost_scale)
        result["gradients"] = dict(full=ga, exact_hybrid=gb, approximate_hybrid=gc)
        result["gradient_alignments"] = dict(continuation=gradient_alignment(gb, ga),
                                             critic=gradient_alignment(gc, gb),
                                             combined=gradient_alignment(gc, ga))
    return result
