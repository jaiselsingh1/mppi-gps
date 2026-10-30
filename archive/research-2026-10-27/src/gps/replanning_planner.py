"""Independent MPPI banks scored by the outcomes of actual replanning.

The old endpoint controller supplies an unchanged alpha-zero proposal bank.
Only the candidate's FIRST action is physically forked.  Every successor owns
the same raw POST-bank nominal and first-pass flag.  Candidate tails never get
credit for actions the controller will not execute.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from src.gps.expert_continuation import finite_absorbing_cost
from src.gps.paired_planner import PairedEndpointPlanner
from src.gps.replanning_value import contexts_features


def augmented_context(state, warmstart, raw, remaining: int,
                      controller_version: str) -> dict:
    """Complete physical/solver/nominal decision boundary; RNG is replay only.

    The value averages fresh future planner randomness.  Its input therefore
    does not encode the opaque PRNG state.  The exact raw snapshot is retained
    separately for replay and counterfactual forks.
    """
    if type(remaining) is not int or remaining < 0 or not controller_version:
        raise ValueError("Require nonnegative remaining controls and controller version")
    return dict(state=np.array(state, dtype=np.float64, copy=True),
        warmstart=np.array(warmstart, dtype=np.float64, copy=True),
        nominal=np.array(raw["U"], dtype=np.float64, copy=True),
        initial_plan=bool(raw["initial_plan"]), remaining=remaining,
        controller_version=str(controller_version))


def raw_from_context(context: dict, seed: int) -> dict:
    """Retain exact nominal/flag, replace only future private PCG64 stream."""
    return dict(U=np.array(context["nominal"], dtype=np.float64, copy=True),
        initial_plan=bool(context["initial_plan"]),
        rng_state=copy.deepcopy(np.random.default_rng(int(seed)).bit_generator.state))


def score_decision(bank: dict, value: Any, alpha: float, *, total_steps: int,
                   gamma: float, value_target_version: str) -> dict:
    """Pure cached-bank reranking: no physics, random draws or label access.

    S_i = (1-alpha)(B_i-B_b) + alpha(Q_i-Q_b), Q_i=c_i+gamma V^C(z'_i).
    This is an explicit local/full-return score blend for alpha<1, not an exact
    policy value.  It has no temperature-based correction cap.  Unsupported
    alternatives cannot displace the baseline; unsupported baseline abstains
    for the entire decision.  Immediate falls and horizon exhaustion have
    exactly known finite returns and need no learned prediction.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    count, baseline = len(bank["plans"]), int(bank["baseline_index"])
    local = np.asarray(bank["local_costs"], dtype=np.float64)
    scores = local-local[baseline]
    if "one_step_costs" not in bank:
        if alpha > 0:
            raise ValueError("Positive reranking requires prepared one-step successors")
        return dict(scores=scores, predicted_values=np.full(count, np.nan),
            supported=np.zeros(count, dtype=bool), distances=np.full(count, np.inf),
            ensemble_std=np.full(count, np.nan), q_values=np.full(count, np.nan),
            selected_index=baseline, baseline_index=baseline, guidance_used=False,
            model_guidance_used=False, fallback_reason="alpha_zero", action_changed=False,
            value_prediction_rows=0, alpha=0., value_target_version=str(value_target_version))
    predictions = np.full(count, np.nan)
    support = np.asarray(bank["one_step_terminated"], dtype=bool).copy()
    if bank["remaining"] == 1:
        support[:] = True
    distance, std = np.full(count, np.inf), np.full(count, np.nan)
    q = np.asarray(bank["one_step_costs"], dtype=np.float64).copy()
    known = support.copy()
    q[known & bank["one_step_terminated"]] += gamma * finite_absorbing_cost(
        gamma, int(bank["remaining"])-1, float(bank["unhealthy_cost"]))
    q[~known] = np.nan
    predictions[known], distance[known], std[known] = 0., 0., 0.
    reason = "alpha_zero" if alpha == 0 else "no_value" if value is None and not known.all() else ""
    value_rows = np.flatnonzero(~known)
    model_rows = 0
    if not reason and len(value_rows):
        contexts = [dict(bank["next_contexts"][i], controller_version=value_target_version)
                    for i in value_rows]
        x = contexts_features(contexts, total_steps)
        remaining = np.full(len(value_rows), int(bank["remaining"])-1, dtype=np.int64)
        if hasattr(value, "predict_details"):
            result = value.predict_details(x, remaining, controller_version=value_target_version)
            costs, supported, distances = result["cost"], result["supported"], result["distance"]
            deviations = result["ensemble_std"]
        else:
            costs, supported, distances = value.predict(x, remaining,
                controller_version=value_target_version)
            deviations = np.zeros(len(value_rows))
        costs, supported, distances, deviations = [np.asarray(v) for v in
                                                   (costs, supported, distances, deviations)]
        if any(a.shape != (len(value_rows),) for a in (costs, supported, distances, deviations)):
            raise ValueError("Invalid replanning value prediction shapes")
        predictions[value_rows], distance[value_rows], std[value_rows] = costs, distances, deviations
        support[value_rows] = supported & np.isfinite(costs) & np.isfinite(distances)
        q[value_rows] = np.asarray(bank["one_step_costs"])[value_rows] + gamma * costs
        model_rows = len(value_rows)
    if not reason and not support[baseline]:
        reason = "unsupported_baseline"
    used = bool(not reason)
    if used:
        # Unsupported rows retain their nonnegative local gap. They cannot
        # strictly improve on the exact baseline's zero score without evidence.
        scores[support] = ((1-alpha)*(local[support]-local[baseline])
                          + alpha*(q[support]-q[baseline]))
        scores[baseline] = 0.
    selected = int(np.argmin(scores))
    return dict(scores=scores, predicted_values=predictions, supported=support,
        distances=distance, ensemble_std=std, q_values=q, selected_index=selected,
        baseline_index=baseline, guidance_used=used, fallback_reason=reason,
        model_guidance_used=bool(used and model_rows > 0),
        action_changed=not np.array_equal(bank["actions"][selected], bank["actions"][baseline]),
        value_prediction_rows=model_rows, alpha=float(alpha),
        value_target_version=str(value_target_version))


class ReplanningPlanner:
    """MPPI plus trajectory memory with controller-consistent one-step values."""

    def __init__(self, env, cfg, archive=None, value=None, alpha: float = 0.,
                 seed: int = 0, remaining: int = 1000, total_steps: int = 1000,
                 gamma: float = .999, memory_candidates: int = 8,
                 controller_version: str = "memory", value_target_version: str | None = None):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0,1]")
        self.env, self.cfg = env, cfg
        self.value, self.alpha = value, float(alpha)
        self.gamma, self.total_steps = float(gamma), int(total_steps)
        self.controller_version = str(controller_version)
        self.value_target_version = str(value_target_version or getattr(value, "controller_version", controller_version))
        self.base = PairedEndpointPlanner(env, cfg, archive=archive, value=None, alpha=0.,
            seed=seed, remaining=remaining, total_steps=total_steps, gamma=gamma,
            memory_candidates=memory_candidates)
        self.raw = self.base.raw
        self.last_decision = self.last_plan = self.last_plans = None

    @property
    def remaining(self):
        return self.base.remaining

    @remaining.setter
    def remaining(self, value):
        self.base.remaining = value

    def reset(self, seed=None, remaining=None):
        self.base.reset(seed, remaining)
        self.last_decision = self.last_plan = self.last_plans = None

    def _one_step_bank(self, state, warmstart, bank, raw_post):
        """Scratch-step unique first controls and restore the live simulator."""
        original_state, original_warm = self.env.get_state(), self.env.get_warmstart()
        actions = np.asarray(bank["plans"])[:, 0]
        states, warms, costs, terminated, contexts = [], [], [], [], []
        cache = {}
        try:
            for action in actions:
                key = np.ascontiguousarray(action).tobytes()
                if key not in cache:
                    self.env.reset(state=np.asarray(state))
                    self.env.set_warmstart(warmstart)
                    _, cost, fell, _ = self.env.step(action)
                    cache[key] = (self.env.get_state(), self.env.get_warmstart(), float(cost), bool(fell))
                next_state, next_warm, cost, fell = cache[key]
                states.append(next_state.copy()); warms.append(next_warm.copy())
                costs.append(cost); terminated.append(fell)
                contexts.append(augmented_context(next_state, next_warm, raw_post,
                    int(bank["remaining"])-1, self.value_target_version))
        finally:
            self.env.reset(state=original_state)
            self.env.set_warmstart(original_warm)
        return dict(one_step_states=np.asarray(states), one_step_warmstarts=np.asarray(warms),
            one_step_costs=np.asarray(costs), one_step_terminated=np.asarray(terminated),
            next_contexts=contexts, one_step_verification_controls=len(cache),
            unhealthy_cost=float(self.env._w_unhealthy))

    def plan_step(self, state, initial_warmstart=None, remaining=None, prepare_successors=False):
        state = np.array(state, dtype=np.float64, copy=True)
        warm = (self.env.get_warmstart() if initial_warmstart is None
                else np.array(initial_warmstart, dtype=np.float64, copy=True))
        raw_pre = self.raw.snapshot()
        _, info = self.base.plan_step(state, initial_warmstart=warm, remaining=remaining)
        bank = copy.deepcopy(self.base.last_decision)
        raw_post = self.raw.snapshot()
        if self.alpha > 0 or prepare_successors:
            bank.update(self._one_step_bank(state, warm, bank, raw_post))
        else:
            bank["one_step_verification_controls"] = 0
        bank.update(raw_pre=copy.deepcopy(raw_pre), raw_post=copy.deepcopy(raw_post),
            snapshot=dict(state=state.copy(), warmstart=warm.copy()),
            controller_version=self.controller_version)
        ranked = self.rerank_decision(bank=bank)
        bank.update(ranked)
        selected, baseline = bank["selected_index"], bank["baseline_index"]
        self.last_plan, self.last_plans = bank["plans"][selected].copy(), bank["plans"].copy()
        self.last_decision = bank
        info.update({key: ranked[key] for key in ("guidance_used", "model_guidance_used", "action_changed", "fallback_reason",
                                                "value_prediction_rows")})
        info.update(one_step_verification_controls=bank["one_step_verification_controls"],
            selected_source=bank["sources"][selected], selected_score=float(bank["scores"][selected]),
            selected_local_cost=float(bank["local_costs"][selected]),
            baseline_local_cost=float(bank["local_costs"][baseline]),
            selected_q=float(bank["q_values"][selected]) if np.isfinite(bank["q_values"][selected]) else None,
            baseline_q=float(bank["q_values"][baseline]) if np.isfinite(bank["q_values"][baseline]) else None,
            alpha=self.alpha, controller_version=self.controller_version,
            value_target_version=self.value_target_version)
        return self.last_plan[0].copy(), info

    def rerank_decision(self, value=None, alpha=None, value_target_version=None, bank=None):
        """Shadow-score cached bank without changing deployed action or raw state."""
        bank = self.last_decision if bank is None else bank
        if bank is None:
            raise RuntimeError("Plan a decision before reranking")
        return score_decision(bank, self.value if value is None else value,
            self.alpha if alpha is None else float(alpha), total_steps=self.total_steps,
            gamma=self.gamma, value_target_version=value_target_version or self.value_target_version)
