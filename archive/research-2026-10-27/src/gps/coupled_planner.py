"""Independent MPPI proposals, replayed trajectory memory and grounded value scores.

The value never changes the raw MPPI nominal, perturbations or private RNG. It
only re-ranks physically simulated action sequences. Memory is an action
proposal archive: retrieved fragments are re-simulated from the *current* state
and solver warmstart, never treated as a transition to an archived state.

Endpoint support is an empirical eligibility rule, not a performance guarantee.
The value describes a freshly restarted baseline controller, not the history-
conditioned raw planner whose shifted nominal is retained during an episode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.gps.expert_continuation import finite_absorbing_cost
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def _features(states: np.ndarray) -> np.ndarray:
    """Flat-Walker translation-invariant, unclipped physical features."""
    return np.asarray(states, dtype=np.float64)[..., 2:]


@dataclass
class MemoryArchive:
    states: np.ndarray
    warmstarts: np.ndarray
    plans: np.ndarray
    remaining: np.ndarray
    feature_scale: np.ndarray | None = None
    suffix_costs: np.ndarray | None = None
    suffix_discounts: np.ndarray | None = None

    def __post_init__(self):
        self.states = np.array(self.states, dtype=np.float64, copy=True)
        self.warmstarts = np.array(self.warmstarts, dtype=np.float64, copy=True)
        self.plans = np.array(self.plans, dtype=np.float64, copy=True)
        self.remaining = np.array(self.remaining, dtype=np.int64, copy=True)
        count = len(self.states)
        if (self.states.ndim != 2 or self.states.shape[1] < 3
                or self.warmstarts.ndim != 2 or self.plans.ndim != 3
                or len(self.warmstarts) != count or len(self.plans) != count
                or self.remaining.shape != (count,) or self.plans.shape[1] < 2
                or np.any(self.remaining < 1)):
            raise ValueError("Memory requires aligned states, warmstarts, plans and positive remaining controls")
        if not all(np.isfinite(a).all() for a in (self.states, self.warmstarts, self.plans)):
            raise ValueError("Memory arrays must be finite")
        if self.feature_scale is None:
            features = _features(self.states)
            # A floor prevents tiny reset noise from defining enormous distances.
            self.feature_scale = np.maximum(np.std(features, axis=0), 0.1) if count else np.full(self.states.shape[1] - 2, 0.1)
        else:
            self.feature_scale = np.array(self.feature_scale, dtype=np.float64, copy=True)
        if (self.feature_scale.shape != (self.states.shape[1] - 2,)
                or not np.isfinite(self.feature_scale).all() or np.any(self.feature_scale <= 0)):
            raise ValueError("Memory feature_scale must be finite and positive")
        self.suffix_costs = (np.full(count, np.nan) if self.suffix_costs is None else
                             np.array(self.suffix_costs, dtype=np.float64, copy=True))
        self.suffix_discounts = (np.ones(count) if self.suffix_discounts is None else
                                 np.array(self.suffix_discounts, dtype=np.float64, copy=True))
        if (self.suffix_costs.shape != (count,) or self.suffix_discounts.shape != (count,)
                or np.any(np.isinf(self.suffix_costs)) or not np.isfinite(self.suffix_discounts).all()
                or np.any(self.suffix_discounts <= 0)):
            raise ValueError("Invalid archived measured-return metadata")

    @classmethod
    def from_episodes(cls, episodes: Iterable[dict], H: int, stride: int = 16,
                      max_entries: int = 2048) -> "MemoryArchive":
        """Archive measured action suffixes; short fragments repeat their last action.

        Repeating an action is only proposal padding. The resulting plan still
        undergoes a fresh rollout and first-fall scoring before selection.
        Uniform chronological thinning bounds storage without selecting rewards.
        """
        if type(H) is not int or H < 2 or stride < 1 or max_entries < 1:
            raise ValueError("Require H>=2 and positive stride/max_entries")
        states, warms, plans, remaining, suffix_costs, suffix_discounts = [], [], [], [], [], []
        state_dim, warm_dim, action_dim = 19, 9, 6
        for episode in episodes:
            st = np.asarray(episode["states"], dtype=np.float64)
            ws = np.asarray(episode["warmstarts"], dtype=np.float64)
            ac = np.asarray(episode["actions"], dtype=np.float64)
            if st.ndim != 2 or ws.ndim != 2 or ac.ndim != 2 or len(st) != len(ac) + 1 or len(ws) != len(st):
                raise ValueError("Episodes require T+1 states/warmstarts and T actions")
            state_dim, warm_dim, action_dim = st.shape[1], ws.shape[1], ac.shape[1]
            start = int(episode.get("remaining_start", len(ac)))
            gamma = float(episode.get("gamma", .999))
            if not 0 <= gamma <= 1:
                raise ValueError("Invalid episode discount")
            raw_costs = episode.get("costs")
            if raw_costs is not None:
                raw_costs = np.asarray(raw_costs, dtype=np.float64)
                if raw_costs.shape != (len(ac),) or not np.isfinite(raw_costs).all():
                    raise ValueError("Archived measured costs must align with executed actions")
            terminated = bool(episode.get("terminated", False))
            absorbing = float(episode.get("unhealthy_cost", 5.))
            for index in range(0, min(len(ac), start), stride):
                fragment = ac[index:index + H]
                if len(fragment) < H:
                    fragment = np.concatenate([fragment, np.repeat(fragment[-1:], H - len(fragment), axis=0)])
                states.append(st[index]); warms.append(ws[index]); plans.append(fragment)
                remaining.append(start - index)
                suffix_length = min(len(ac), start) - index
                suffix_discount = finite_absorbing_cost(gamma, start - index, 1.)
                suffix_discounts.append(suffix_discount)
                # Rank only completed finite returns. A short healthy fragment
                # is not secretly granted a free unmeasured continuation.
                if raw_costs is not None and (terminated or len(ac) >= start):
                    result = float(np.dot(gamma ** np.arange(suffix_length), raw_costs[index:index + suffix_length]))
                    if terminated:
                        result += gamma ** suffix_length * finite_absorbing_cost(gamma,
                            start - index - suffix_length, absorbing)
                    suffix_costs.append(result)
                else:
                    suffix_costs.append(np.nan)
        if not plans:
            return cls(np.empty((0, state_dim)), np.empty((0, warm_dim)),
                       np.empty((0, H, action_dim)), np.empty(0, dtype=np.int64))
        keep = np.linspace(0, len(plans) - 1, min(max_entries, len(plans)), dtype=int)
        return cls(np.asarray(states)[keep], np.asarray(warms)[keep],
                   np.asarray(plans)[keep], np.asarray(remaining)[keep],
                   suffix_costs=np.asarray(suffix_costs)[keep],
                   suffix_discounts=np.asarray(suffix_discounts)[keep])

    def retrieve(self, state: np.ndarray, remaining: int, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
        """Half nearest, half best measured suffixes in the nearest 64 states."""
        if type(remaining) is not int or remaining < 1 or k < 0:
            raise ValueError("Require positive remaining controls and nonnegative k")
        if not len(self.states) or not k:
            return self.plans[:0].copy(), np.empty(0)
        features = _features(np.asarray(state))
        if features.shape != self.feature_scale.shape or not np.isfinite(features).all():
            raise ValueError("Query state has invalid memory features")
        distances = np.sqrt(np.mean(((_features(self.states) - features) / self.feature_scale) ** 2, axis=1))
        # Remaining time breaks physical-state ties, without teleporting the
        # episode clock or excluding a transferable measured action fragment.
        ordered = np.lexsort((np.arange(len(distances)), np.abs(self.remaining - remaining), distances))
        count = min(k, len(ordered))
        nearest_count = (count + 1) // 2
        indices = list(ordered[:nearest_count])
        # Give known successful/cheap measured continuations a fair proposal
        # slot, rather than losing all of them to tiny reset-state differences.
        # These return labels never substitute for replay from the current state.
        neighborhood = ordered[:64]
        known = neighborhood[np.isfinite(self.suffix_costs[neighborhood])]
        if len(known):
            quality = self.suffix_costs[known] / self.suffix_discounts[known]
            quality_order = known[np.lexsort((known, distances[known], quality))]
            for index in quality_order:
                if len(indices) >= count:
                    break
                if int(index) not in indices:
                    indices.append(int(index))
        for index in ordered:
            if len(indices) == count:
                break
            if int(index) not in indices:
                indices.append(int(index))
        indices = np.asarray(indices, dtype=np.int64)
        return self.plans[indices].copy(), distances[indices].copy()

    def state_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name).copy() for name in
                ("states", "warmstarts", "plans", "remaining", "feature_scale", "suffix_costs", "suffix_discounts")}

    @classmethod
    def from_state_dict(cls, state: dict) -> "MemoryArchive":
        return cls(**state)


def local_discounted_score(paths: np.ndarray, actions: np.ndarray, env: Any,
                           gamma: float, remaining: int) -> tuple[np.ndarray, np.ndarray]:
    """Discount a local prefix, including one fall charge and local absorption.

    ``paths`` contains post-action states, exactly as ``batch_rollout`` returns.
    Only min(H, remaining) controls enter the declared objective even though all
    physically simulated slots must still be charged to the query ledger.
    """
    if not np.isfinite(gamma) or not 0 <= gamma <= 1 or type(remaining) is not int or remaining < 1:
        raise ValueError("Invalid finite-horizon discount or remaining controls")
    paths, actions = np.asarray(paths), np.asarray(actions)
    if paths.ndim != 3 or actions.ndim != 3 or paths.shape[:2] != actions.shape[:2] or paths.shape[1] < 1:
        raise ValueError("Expected aligned candidate path/action banks")
    horizon = min(paths.shape[1], remaining)
    states, controls = paths[:, :horizon], actions[:, :horizon]
    healthy = np.asarray(env._is_healthy(states[..., 2], states[..., 3]), dtype=bool)
    active = np.ones_like(healthy)
    if horizon > 1:
        active[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
    running = np.asarray(env.running_cost(states, controls), dtype=np.float64)
    discounts = gamma ** np.arange(horizon)
    costs = np.sum(np.where(active, running, float(env._w_unhealthy)) * discounts, axis=1)
    fell = np.any(~healthy, axis=1)
    if bool(getattr(env, "_apply_terminal_cost_on_done", True)) and np.any(fell):
        rows = np.flatnonzero(fell)
        first = np.argmax(~healthy[rows], axis=1)
        costs[rows] += discounts[first] * np.asarray(env.terminal_cost(states[rows, first]))
    finite = np.all(np.isfinite(states), axis=-1) & np.all(np.isfinite(controls), axis=-1)
    valid = np.all(finite | ~active, axis=1)
    return np.where(valid & np.isfinite(costs), costs, np.inf), (~fell & valid)


class CoupledPlanner:
    """Verified selection from raw MPPI, memory, and a value-weighted mean.

    ``value.predict(features17, remaining)`` returns cost, support and normalized
    distance arrays. Learned scoring is enabled only after external calibration
    sets ``value.metrics['calibration_eligible']`` and the caller supplies alpha.
    No such attribute or prediction is accessed when alpha is zero.
    """

    def __init__(self, env: Any, cfg: MPPIConfig, archive: MemoryArchive | None = None,
                 value: Any = None, alpha: float = 0., seed: int = 0,
                 remaining: int | None = None, total_steps: int = 1000,
                 gamma: float = .999, memory_candidates: int = 4,
                 support_radius: float = 1.5, min_support_fraction: float = .5):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0,1]")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0,1]")
        if memory_candidates < 0 or support_radius <= 0 or not 0 <= min_support_fraction <= 1:
            raise ValueError("Invalid memory or empirical support configuration")
        self.env, self.cfg, self.archive, self.value = env, cfg, archive, value
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.memory_candidates, self.support_radius = int(memory_candidates), float(support_radius)
        self.min_support_fraction = float(min_support_fraction)
        self.total_steps = int(total_steps)
        self.remaining = int(total_steps if remaining is None else remaining)
        if self.total_steps < 1 or self.remaining < 1:
            raise ValueError("Require positive episode horizon")
        self.raw = MPPI(env, cfg, seed=seed)
        self.last_plan: np.ndarray | None = None
        self.last_plans: np.ndarray | None = None
        self.last_candidate_endpoints: np.ndarray | None = None

    def reset(self, seed: int | None = None, remaining: int | None = None):
        self.raw.reset(seed=seed)
        self.remaining = self.total_steps if remaining is None else int(remaining)
        if self.remaining < 1:
            raise ValueError("Require positive remaining controls")
        self.last_plan = self.last_plans = self.last_candidate_endpoints = None

    def _rollout(self, state, plans, initial_warmstart):
        kwargs = {} if initial_warmstart is None else {"initial_warmstart": initial_warmstart}
        return self.env.batch_rollout(state, plans, **kwargs)[0]

    def _scores(self, paths, plans, remaining, *, use_value):
        local, healthy = local_discounted_score(paths, plans, self.env, self.gamma, remaining)
        supported = np.ones(len(plans), dtype=bool)
        if not use_value or remaining <= self.cfg.H:
            return local, local.copy(), supported, healthy
        h = min(self.cfg.H, remaining)
        tail = np.full(len(plans), finite_absorbing_cost(self.gamma, remaining - h,
                                                        float(self.env._w_unhealthy)))
        supported[:] = ~healthy
        if np.any(healthy):
            rows = np.flatnonzero(healthy)
            cost, support, distance = self.value.predict(_features(paths[rows, h - 1]),
                np.full(len(rows), remaining - h, dtype=np.int64))
            cost, support, distance = np.asarray(cost), np.asarray(support, dtype=bool), np.asarray(distance)
            if cost.shape != (len(rows),) or support.shape != cost.shape or distance.shape != cost.shape:
                raise ValueError("Value predictor returned invalid batch shapes")
            supported[rows] = support & np.isfinite(cost) & np.isfinite(distance) & (distance <= self.support_radius)
            tail[rows] = np.where(supported[rows], cost, 0.)
        score = local + self.alpha * self.gamma ** h * tail
        return local, np.where(supported, score, np.inf), supported, healthy

    def plan_step(self, state: np.ndarray, initial_warmstart: np.ndarray | None = None,
                  remaining: int | None = None) -> tuple[np.ndarray, dict]:
        n = self.remaining if remaining is None else remaining
        if type(n) is not int or n < 1:
            raise ValueError("Cannot plan after the finite episode horizon")
        _, raw_info = self.raw.plan_step(state, initial_warmstart=initial_warmstart)
        raw_plans = self.raw._last_actions.copy()
        raw_paths = self.raw._last_states.copy()
        raw_mean = self.raw._last_plan.copy()
        if self.archive is not None:
            memory, distances = self.archive.retrieve(state, n, self.memory_candidates)
            if memory.shape[1:] != raw_mean.shape:
                raise ValueError("Memory action dimensions/horizon do not match the planner")
        else:
            memory, distances = raw_plans[:0], np.empty(0)
        memory_paths = self._rollout(state, memory, initial_warmstart) if len(memory) else raw_paths[:0]
        # First mandatory verification: raw weighted mean is not presumed to
        # inherit any sampled trajectory's health or value.
        raw_mean_paths = self._rollout(state, raw_mean[None], initial_warmstart)
        plans = np.concatenate([raw_plans, memory, raw_mean[None]])
        paths = np.concatenate([raw_paths, memory_paths, raw_mean_paths])
        sources = ["sample"] * len(raw_plans) + ["memory"] * len(memory) + ["raw_mean"]
        use_value = self.alpha > 0 and self.value is not None and n > self.cfg.H
        reason = ("alpha_zero" if self.alpha == 0 else
                  "horizon_exhausted" if n <= self.cfg.H else "no_value")
        if use_value:
            use_value = bool(getattr(self.value, "metrics", {}).get("calibration_eligible", False))
            reason = "not_calibrated" if not use_value else ""
        local, scores, supported, healthy = self._scores(paths, plans, n, use_value=use_value)
        baseline_index = int(np.argmin(local))
        fraction = float(np.mean(supported[healthy])) if np.any(healthy) else 0.
        if use_value and n > self.cfg.H and (not np.any(healthy & supported) or fraction < self.min_support_fraction):
            use_value, reason, scores = False, "insufficient_support", local.copy()
        elif use_value and not supported[baseline_index]:
            # A population support fraction cannot justify removing the exact
            # alpha-zero winner. Abstain for the whole decision so even tiny
            # positive influence cannot discard an unevaluated baseline.
            # Do not give that endpoint an invented zero continuation value.
            use_value, reason, scores = False, "unsupported_baseline", local.copy()
        if not np.any(np.isfinite(scores)):
            raise FloatingPointError("No finite physically verified candidate score")
        # Mean formation uses sample+memory proposals, not the separately
        # verified raw mean. Unsupported endpoints receive exactly zero weight.
        proposal_scores = scores[:-1]
        finite = np.isfinite(proposal_scores)
        formed_guided_mean = bool(use_value and np.any(finite))
        if formed_guided_mean:
            weights = np.zeros(len(proposal_scores))
            weights[finite] = np.exp(-(proposal_scores[finite] - np.min(proposal_scores[finite])) / self.cfg.lam)
            weights /= weights.sum()
            second_mean = np.einsum("k,kha->ha", weights, plans[:-1])
            # Floating-point accumulation can put a convex mean one ULP beyond
            # a saturated actuator bound. Verify and execute the bounded plan.
            second_mean = np.clip(second_mean, *self.env.action_bounds)
            # This is additional useful work by the guided controller. The
            # baseline does not replay a duplicate to manufacture equal costs.
            second_paths = self._rollout(state, second_mean[None], initial_warmstart)
            second_local, second_score, _, _ = self._scores(second_paths, second_mean[None], n, use_value=True)
            all_scores = np.concatenate([scores, second_score])
            all_local = np.concatenate([local, second_local])
            all_plans = np.concatenate([plans, second_mean[None]])
            all_paths = np.concatenate([paths, second_paths])
            sources.append("guided_mean")
        else:
            all_scores, all_local, all_plans, all_paths = scores, local, plans, paths
        selected = int(np.argmin(all_scores))
        self.last_plan = all_plans[selected].copy()
        self.last_plans = all_plans.copy()
        self.last_candidate_endpoints = all_paths[:, min(self.cfg.H, n) - 1].copy()
        self.remaining = n - 1
        passes = int(raw_info["optimization_iterations"])
        info = dict(raw_info)
        info.update(candidate_controls=self.cfg.K * self.cfg.H * passes,
                    memory_controls=len(memory) * self.cfg.H,
                    verification_controls=(1 + int(formed_guided_mean)) * self.cfg.H,
                    raw_optimization_passes=passes,
                    support_fraction=fraction if use_value or reason in {
                        "insufficient_support", "unsupported_baseline"} else 0.,
                    guidance_used=bool(use_value and n > self.cfg.H),
                    fallback_reason=reason, selected_source=sources[selected],
                    selected_score=float(all_scores[selected]),
                    selected_local_cost=float(all_local[selected]),
                    baseline_local_cost=float(local[-1]),
                    baseline_candidate_index=baseline_index,
                    baseline_candidate_source=sources[baseline_index],
                    baseline_candidate_local_cost=float(local[baseline_index]),
                    baseline_endpoint_supported=bool(supported[baseline_index]),
                    memory_candidates=len(memory),
                    memory_nearest_distance=float(distances[0]) if len(distances) else None,
                    remaining=n, alpha=self.alpha)
        return self.last_plan[0].copy(), info
