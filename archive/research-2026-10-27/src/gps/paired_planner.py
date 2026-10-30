"""Fixed-bank MPPI reranking and training-only paired endpoint acquisition.

Learned values never change the raw MPPI proposal process. Every decision uses
the same raw samples, re-simulated archive fragments and physically verified raw
weighted mean at alpha zero and positive alpha. No learned weighted mean is
formed. The cached bank can also supply *training* endpoint pairs when deployed
guidance abstains, keeping lack of support from blocking data acquisition.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.gps.coupled_planner import MemoryArchive, local_discounted_score
from src.gps.expert_continuation import finite_absorbing_cost
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def finite_local_score(paths: np.ndarray, plans: np.ndarray, env: Any,
                       gamma: float, remaining: int) -> tuple[np.ndarray, np.ndarray]:
    """Local prefix cost, plus the entire known absorbing remainder after a fall.

Healthy paths have no invented tail. Fallen paths already include one terminal
charge and absorption through the simulated prefix; only the *remaining* finite
absorption is added. This cost is the same with and without learned guidance.
    """
    costs, healthy = local_discounted_score(paths, plans, env, gamma, remaining)
    h = min(plans.shape[1], remaining)
    if remaining > h:
        costs = costs + np.where(healthy, 0., gamma ** h * finite_absorbing_cost(
            gamma, remaining - h, float(env._w_unhealthy)))
    return costs, healthy


class PairedEndpointPlanner:
    """Independent MPPI and memory with bounded relative endpoint corrections.

The generic value interface is ``predict(features17, remaining)`` returning
three aligned arrays: finite cost estimates, empirical support flags, and
normalized support distances. No value metadata is needed or inspected.
    """

    def __init__(self, env: Any, cfg: MPPIConfig,
                 archive: MemoryArchive | None = None, value: Any = None,
                 alpha: float = 0., seed: int = 0, remaining: int = 1000,
                 total_steps: int = 1000, gamma: float = .999,
                 memory_candidates: int = 8, support_radius: float = 1.5,
                 max_tail_shift_temperatures: float = 2.):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0,1]")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must lie in [0,1]")
        if type(remaining) is not int or remaining < 1 or type(total_steps) is not int or total_steps < 1:
            raise ValueError("Require positive integer episode horizons")
        if type(memory_candidates) is not int or memory_candidates < 0:
            raise ValueError("memory_candidates must be a nonnegative integer")
        if not np.isfinite(support_radius) or support_radius <= 0:
            raise ValueError("support_radius must be positive and finite")
        if not np.isfinite(max_tail_shift_temperatures) or max_tail_shift_temperatures < 0:
            raise ValueError("Tail correction bound must be nonnegative and finite")
        self.env, self.cfg, self.archive, self.value = env, cfg, archive, value
        self.alpha, self.gamma = float(alpha), float(gamma)
        self.remaining, self.total_steps = remaining, total_steps
        self.memory_candidates = memory_candidates
        self.support_radius = float(support_radius)
        self.max_tail_shift_temperatures = float(max_tail_shift_temperatures)
        self.raw = MPPI(env, cfg, seed=seed)
        self.last_plan: np.ndarray | None = None
        self.last_plans: np.ndarray | None = None
        self.last_candidate_endpoints: np.ndarray | None = None
        self.last_decision: dict | None = None

    def reset(self, seed: int | None = None, remaining: int | None = None):
        n = self.total_steps if remaining is None else remaining
        if type(n) is not int or n < 1:
            raise ValueError("Require positive remaining controls")
        self.raw.reset(seed=seed)
        self.remaining = n
        self.last_plan = self.last_plans = self.last_candidate_endpoints = None
        self.last_decision = None

    def _rollout(self, state, plans, initial_warmstart):
        kwargs = {} if initial_warmstart is None else {"initial_warmstart": initial_warmstart}
        return self.env.batch_rollout(state, plans, **kwargs)[0]

    def _predict(self, value, endpoints, healthy, remaining):
        """Read healthy endpoints only; preserve unsupported finite predictions."""
        predicted = np.full(len(endpoints), np.nan)
        supported = np.zeros(len(endpoints), dtype=bool)
        distances = np.full(len(endpoints), np.inf)
        rows = np.flatnonzero(healthy)
        if len(rows):
            costs, support, distance = value.predict(endpoints[rows, 2:],
                np.full(len(rows), remaining, dtype=np.int64))
            costs = np.asarray(costs, dtype=np.float64)
            support = np.asarray(support, dtype=bool)
            distance = np.asarray(distance, dtype=np.float64)
            if costs.shape != (len(rows),) or support.shape != costs.shape or distance.shape != costs.shape:
                raise ValueError("Value predictor returned invalid batch shapes")
            predicted[rows], distances[rows] = costs, distance
            supported[rows] = (support & np.isfinite(costs) & np.isfinite(distance)
                               & (distance <= self.support_radius))
        return predicted, supported, distances

    def plan_step(self, state: np.ndarray,
                  initial_warmstart: np.ndarray | None = None,
                  remaining: int | None = None) -> tuple[np.ndarray, dict]:
        n = self.remaining if remaining is None else remaining
        if type(n) is not int or n < 1:
            raise ValueError("Cannot plan after the finite episode horizon")
        _, raw_info = self.raw.plan_step(state, initial_warmstart=initial_warmstart)
        raw_plans, raw_paths = self.raw._last_actions.copy(), self.raw._last_states.copy()
        raw_mean = self.raw._last_plan.copy()
        if self.archive is None:
            memory, memory_distances = raw_plans[:0], np.empty(0)
        else:
            memory, memory_distances = self.archive.retrieve(state, n, self.memory_candidates)
            if memory.ndim != 3 or memory.shape[1:] != raw_mean.shape:
                raise ValueError("Memory action dimensions/horizon do not match the planner")
        memory_paths = self._rollout(state, memory, initial_warmstart) if len(memory) else raw_paths[:0]
        # This mandatory check is identical at alpha zero and positive alpha.
        mean_paths = self._rollout(state, raw_mean[None], initial_warmstart)
        plans = np.concatenate([raw_plans, memory, raw_mean[None]])
        paths = np.concatenate([raw_paths, memory_paths, mean_paths])
        sources = ["sample"] * len(raw_plans) + ["memory"] * len(memory) + ["raw_mean"]
        local, healthy = finite_local_score(paths, plans, self.env, self.gamma, n)
        if not np.isfinite(local).any():
            raise FloatingPointError("No finite physically verified candidate score")
        h = min(self.cfg.H, n)
        endpoints = paths[:, h - 1].copy()
        baseline = int(np.argmin(local))
        scores, correction = local.copy(), np.zeros(len(plans))
        predicted = supported = distances = None
        use_value = False
        reason = ("alpha_zero" if self.alpha == 0 else "horizon_exhausted" if n <= h
                  else "no_value" if self.value is None else "")
        if not reason:
            predicted, supported, distances = self._predict(self.value, endpoints, healthy, n - h)
            if not healthy[baseline]:
                reason = "baseline_fallen"
            elif not supported[baseline]:
                reason = "unsupported_baseline"
            else:
                # Unsupported plans retain their local score. They are never
                # excluded merely because the predictor lacks evidence.
                eligible = healthy & supported
                bound = self.cfg.lam * self.max_tail_shift_temperatures
                correction[eligible] = self.alpha * np.clip(
                    self.gamma ** h * (predicted[eligible] - predicted[baseline]), -bound, bound)
                scores += correction
                use_value = True
        selected = int(np.argmin(scores))
        self.last_plan = plans[selected].copy()
        self.last_plans = plans.copy()
        self.last_candidate_endpoints = endpoints.copy()
        self.remaining = n - 1
        action_changed = not np.array_equal(plans[selected, 0], plans[baseline, 0])
        self.last_decision = dict(plans=plans.copy(), endpoints=endpoints.copy(),
            local_costs=local.copy(), scores=scores.copy(), corrections=correction.copy(),
            healthy=healthy.copy(), baseline_index=baseline, selected_index=selected,
            remaining=n, H=h, sources=list(sources), predicted_values=predicted,
            supported=supported, support=supported, distances=distances,
            actions=plans[:, 0].copy(), baseline_action=plans[baseline, 0].copy(),
            selected_action=plans[selected, 0].copy(), guidance_used=use_value,
            action_changed=action_changed, fallback_reason=reason)
        passes = int(raw_info["optimization_iterations"])
        fraction = float(np.mean(supported[healthy])) if supported is not None and healthy.any() else 0.
        info = dict(raw_info)
        info.update(candidate_controls=self.cfg.K * self.cfg.H * passes,
            memory_controls=len(memory) * self.cfg.H, verification_controls=self.cfg.H,
            raw_optimization_passes=passes, guidance_used=use_value,
            action_changed=action_changed, fallback_reason=reason,
            selected_source=sources[selected], selected_score=float(scores[selected]),
            selected_local_cost=float(local[selected]), baseline_local_cost=float(local[baseline]),
            baseline_candidate_index=baseline, baseline_candidate_source=sources[baseline],
            baseline_candidate_local_cost=float(local[baseline]),
            baseline_endpoint_supported=(bool(supported[baseline]) if supported is not None else None),
            selected_endpoint_supported=(bool(supported[selected]) if supported is not None else None),
            support_fraction=fraction, memory_candidates=len(memory),
            memory_nearest_distance=float(memory_distances[0]) if len(memory_distances) else None,
            selected_score_correction=float(correction[selected]), remaining=n, alpha=self.alpha)
        return self.last_plan[0].copy(), info

    def acquisition_pair(self, rng: np.random.Generator, shadow_value: Any = None,
                         probe_alpha: float = .1, mode: str = "guided") -> dict | None:
        """Choose a training pair from the cached bank without any new physics.

        The candidate may be unsupported: measuring that endpoint is precisely how
training can repair an abstention. The bounded relative correction otherwise
matches deployment, so a training proposal can earn influence within that same
budget. No endpoint prediction is used as a label.
Both measured continuations must later use the caller's matched reference RNG.
If candidate equals baseline it stays equal, retaining an unbiased zero pair.
No pair is possible when the local winner has fallen or has no remaining tail.
        """
        if self.last_decision is None:
            raise RuntimeError("Plan a decision before requesting endpoint acquisition")
        if mode not in {"guided", "uniform"}:
            raise ValueError("Acquisition mode must be guided or uniform")
        if not np.isfinite(probe_alpha) or not 0 <= probe_alpha <= 1:
            raise ValueError("probe_alpha must lie in [0,1]")
        bank = self.last_decision
        baseline = bank["baseline_index"]
        n, h = bank["remaining"], bank["H"]
        if n <= h or not bank["healthy"][baseline]:
            return None
        rows = np.flatnonzero(bank["healthy"] & np.isfinite(bank["local_costs"]))
        if not len(rows):
            return None
        predicted = supported = distances = None
        acquisition_scores = None
        method = mode
        if mode == "guided" and shadow_value is not None:
            predicted, supported, distances = self._predict(shadow_value,
                bank["endpoints"], bank["healthy"], n - h)
            finite = rows[np.isfinite(predicted[rows])]
            if len(finite) and np.isfinite(predicted[baseline]):
                bound = self.cfg.lam * self.max_tail_shift_temperatures
                acquisition_scores = np.full(len(bank["plans"]), np.inf)
                acquisition_scores[finite] = bank["local_costs"][finite] + probe_alpha * np.clip(
                    self.gamma ** h * (predicted[finite] - predicted[baseline]), -bound, bound)
                candidate = int(np.argmin(acquisition_scores))
            else:
                candidate = int(rng.choice(rows))
                method = ("uniform_no_finite_baseline_prediction" if not np.isfinite(predicted[baseline])
                          else "uniform_no_finite_prediction")
        else:
            candidate = int(rng.choice(rows))
            if mode == "guided":
                method = "uniform_no_value"
        def item(index):
            return dict(index=index, plan=bank["plans"][index].copy(),
                endpoint=bank["endpoints"][index].copy(),
                prefix_score=float(bank["local_costs"][index]), source=bank["sources"][index],
                predicted_tail=(float(predicted[index]) if predicted is not None and np.isfinite(predicted[index]) else None),
                supported=bool(supported[index]) if supported is not None else None)
        result = dict(candidate=item(candidate), baseline=item(baseline),
            candidate_index=candidate, baseline_index=baseline,
            candidate_plan=bank["plans"][candidate].copy(), baseline_plan=bank["plans"][baseline].copy(),
            candidate_endpoint=bank["endpoints"][candidate].copy(), baseline_endpoint=bank["endpoints"][baseline].copy(),
            endpoints=bank["endpoints"].copy(), plans=bank["plans"].copy(),
            local_costs=bank["local_costs"].copy(), healthy=bank["healthy"].copy(),
            sources=list(bank["sources"]), predicted_values=predicted,
            supported=supported, support=supported, distances=distances,
            acquisition_scores=acquisition_scores,
            selected_index=bank["selected_index"], guidance_used=bank["guidance_used"],
            action_changed=bank["action_changed"], remaining=n, H=h,
            remaining_tail=n - h, gamma_h=self.gamma ** h,
            candidate_equals_baseline=candidate == baseline, acquisition_mode=method,
            probe_alpha=float(probe_alpha))
        return result
