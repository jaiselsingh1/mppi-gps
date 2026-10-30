"""A protected native MPPI search and a separate policy-coupled selector.

The native planner never receives an actor, selected guided plan, or coupling
cost.  Guidance only reweights a copied final-pass bank.  Both selectors use
the same actor-independent proximity term and verify their synthesized mean
with the instrumented environment before executing it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np

from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import softmin_weights
from src.utils.rollout_data import pre_action_states, pre_fall_action_mask


def _readonly(value: np.ndarray) -> np.ndarray:
    """Own immutable bytes, rather than a view whose write flag can be reset."""
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _digest(*arrays: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        if value is None:
            digest.update(b"None")
            continue
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def _actor_hash(actor: Any) -> str | None:
    if actor is None:
        return None
    for name in ("policy_hash", "actor_hash", "hash", "digest"):
        value = getattr(actor, name, None)
        if value is not None:
            value = value() if callable(value) else value
            return str(value)
    return None


def _active_mask(env: Any, post_states: np.ndarray) -> np.ndarray:
    # Walker's raw batch continues simulating after a fall.  Those paid queries
    # remain archived, but cannot supply policy-agreement evidence.
    if hasattr(env, "_is_healthy"):
        healthy = env._is_healthy(post_states[..., 2], post_states[..., 3])
        return pre_fall_action_mask(healthy)
    return np.ones(post_states.shape[:2], dtype=bool)


@dataclass(frozen=True)
class SearchBank:
    """Final native samples plus its separately verified task-only mean.

    ``actions[-1]`` is the native mean; the preceding K entries are the final
    native sampling pass.  Earlier refinement passes remain in the runtime's
    query ledger.  All numerical arrays own immutable bytes.  The environment
    is the instrumented model interface, never the private planner.
    """

    actions: np.ndarray
    observations: np.ndarray
    post_states: np.ndarray
    raw_costs: np.ndarray
    reference_u: np.ndarray
    raw_plan: np.ndarray
    raw_score: float
    initial_state: np.ndarray
    warmstart: np.ndarray | None
    active_mask: np.ndarray
    temperature: float
    provenance: tuple[str, ...]
    diagnostics: Mapping[str, Any]
    bank_digest: str
    _env: Any = field(repr=False, compare=False)

    @property
    def pre_action_observations(self) -> np.ndarray:
        return self.observations

    @property
    def pre_action_states(self) -> np.ndarray:
        return _readonly(pre_action_states(self.initial_state, self.post_states))

    @property
    def reference_U(self) -> np.ndarray:
        return self.reference_u

    @property
    def proposal_actions(self) -> np.ndarray:
        return self.actions[:-1]

    @property
    def mean_index(self) -> int:
        return len(self.actions) - 1


class IndependentMPPI:
    """Own the native planner's task-only nominal, caches, and private RNG."""

    def __init__(self, env: Any, cfg: MPPIConfig, seed: int):
        if seed is None:
            raise ValueError("IndependentMPPI requires a private seeded RNG.")
        self.env = env
        self.cfg = copy.deepcopy(cfg)
        self._planner = MPPI(env, self.cfg, seed=seed)

    def snapshot(self) -> dict:
        return {
            "native": self._planner.snapshot(),
            "caches": {
                name: copy.deepcopy(getattr(self._planner, name))
                for name in ("_last_plan", "_last_states", "_last_actions",
                             "_last_weights", "_last_costs", "_last_sensordata")
            },
        }

    def private_state(self) -> dict:
        """Small copied nominal/RNG snapshot for durable decision evidence."""
        return self._planner.snapshot()

    def restore(self, snapshot: dict) -> None:
        self._planner.restore(copy.deepcopy(snapshot["native"]))
        for name, value in snapshot.get("caches", {}).items():
            if name not in ("_last_plan", "_last_states", "_last_actions",
                            "_last_weights", "_last_costs", "_last_sensordata"):
                raise ValueError(f"Unknown native cache: {name}")
            setattr(self._planner, name, copy.deepcopy(value))

    def propose(self, state: np.ndarray, warmstart: np.ndarray | None = None) -> SearchBank:
        state = np.array(state, dtype=float, copy=True)
        warmstart = None if warmstart is None else np.array(warmstart, dtype=float, copy=True)
        reference = np.clip(self._planner.U.copy(), *self.env.action_bounds)
        # There is deliberately no actor-facing argument on this call.
        _, diagnostics = self._planner.plan_step(state, initial_warmstart=warmstart)
        data = self._planner.get_rollout_data()
        actions = np.array(data["actions"], copy=True)
        posts = np.array(data["states"], copy=True)
        costs = np.array(data["costs"], dtype=float, copy=True)
        mean = self._planner.last_plan
        mean_error = None
        try:
            kwargs = {} if warmstart is None else {"initial_warmstart": warmstart}
            mean_posts, mean_cost, _ = self.env.batch_rollout(state, mean[None], **kwargs)
            mean_posts = np.asarray(mean_posts)
            mean_cost = np.asarray(mean_cost, dtype=float)
            if mean_posts.shape != (1, *posts.shape[1:]) or mean_cost.shape != (1,):
                raise ValueError("Native mean verification returned incompatible shapes.")
        except (FloatingPointError, ValueError) as error:
            mean_error = f"{type(error).__name__}: {error}"
            mean_posts = np.full((1, *posts.shape[1:]), np.nan)
            mean_cost = np.array([np.inf])
        actions = np.concatenate((actions, mean[None]), axis=0)
        posts = np.concatenate((posts, mean_posts), axis=0)
        costs = np.concatenate((costs, mean_cost), axis=0)
        if not np.any(np.isfinite(costs)):
            raise FloatingPointError("No verified finite independent plan.")
        best = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
        observations = self.env.rollout_states_to_obs(pre_action_states(state, posts))
        diagnostics = dict(diagnostics, raw_mean_verification_error=mean_error,
                           raw_selected_index=best)
        return SearchBank(
            actions=_readonly(actions), observations=_readonly(observations),
            post_states=_readonly(posts), raw_costs=_readonly(costs),
            reference_u=_readonly(reference), raw_plan=_readonly(actions[best]),
            raw_score=float(costs[best]), initial_state=_readonly(state),
            warmstart=None if warmstart is None else _readonly(warmstart),
            active_mask=_readonly(_active_mask(self.env, posts)),
            temperature=float(diagnostics["lam"]),
            provenance=("native_sample",) * (len(actions) - 1) + ("verified_raw_mean",),
            diagnostics=MappingProxyType(diagnostics),
            bank_digest=_digest(state, warmstart, actions, posts, costs, reference),
            _env=self.env,
        )


class CoupledSelector:
    """Reweight a shared bank, without changing its independent producer.

    ``supported`` is a bool or callable receiving (..., observation_dim) and
    returning a bool mask with shape (...,) (or one flag per candidate).  None
    permits every finite healthy-prefix observation.  Policy agreement is
    evaluated only on supported, executable-prefix rows.  No actor inference
    occurs when disabled, alpha=0, or the whole bank is unsupported.
    """

    def __init__(self, alpha: float = 1.0, eta_u: float = 0.01):
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(eta_u) or eta_u < 0:
            raise ValueError("alpha and eta_u must be finite and nonnegative.")
        self.alpha = float(alpha)
        self.eta_u = float(eta_u)
        self._calls = 0

    @staticmethod
    def _support(observations: np.ndarray, active: np.ndarray,
                 supported: bool | Callable | None) -> np.ndarray:
        mask = active & np.all(np.isfinite(observations), axis=-1)
        value = True if supported is None else (
            supported(observations) if callable(supported) else supported
        )
        support = np.asarray(value, dtype=bool)
        if support.shape == (len(observations),):
            support = support[:, None]
        try:
            return mask & np.broadcast_to(support, mask.shape)
        except ValueError as error:
            raise ValueError("supported must return a scalar, (K,), or (K,H) mask.") from error

    def _components(self, actions: np.ndarray, raw: np.ndarray,
                    reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prox = 0.5 * self.eta_u * np.sum((actions - reference[None]) ** 2, axis=(1, 2))
        return raw + prox, prox

    @staticmethod
    def _agreement(actor: Any, observations: np.ndarray, actions: np.ndarray,
                   support: np.ndarray, meter: Any) -> np.ndarray:
        values = np.zeros(actions.shape[:2], dtype=float)
        if np.any(support):
            predictions = np.asarray(actor.predict(observations[support], meter=meter))
            if predictions.shape != actions[support].shape or not np.all(np.isfinite(predictions)):
                raise FloatingPointError("Actor returned nonfinite or incorrectly shaped actions.")
            values[support] = np.sum((actions[support] - predictions) ** 2, axis=-1)
        return np.sum(values, axis=1)

    @staticmethod
    def _verify(bank: SearchBank, plan: np.ndarray) -> dict:
        kwargs = {} if bank.warmstart is None else {"initial_warmstart": bank.warmstart}
        posts, costs, _ = bank._env.batch_rollout(bank.initial_state, plan[None], **kwargs)
        posts, costs = np.asarray(posts), np.asarray(costs, dtype=float)
        if posts.shape != (1, *bank.post_states.shape[1:]) or costs.shape != (1,):
            raise ValueError("Selector mean verification returned incompatible shapes.")
        if not np.isfinite(costs[0]):
            raise FloatingPointError("Selector mean has no finite measured task score.")
        observations = bank._env.rollout_states_to_obs(pre_action_states(bank.initial_state, posts))
        return {"plan": plan, "posts": posts, "observations": observations,
                "raw": costs, "active": _active_mask(bank._env, posts)}

    def select(self, bank: SearchBank, actor: Any, meter: Any, enabled: bool = False,
               supported: bool | Callable | None = None) -> tuple[np.ndarray, dict]:
        self._calls += 1
        # A durable meter event distinguishes paired selectors that see the
        # exact same bank, including after checkpoint continuation.
        request_id = meter.record("selection_requested", bank_digest=bank.bank_digest,
                                  enabled=bool(enabled), alpha=self.alpha,
                                  eta_u=self.eta_u) if meter is not None else self._calls
        tag = f"selection_{bank.bank_digest[:16]}_{request_id}"
        raw_scores, prox = self._components(bank.actions, bank.raw_costs, bank.reference_u)
        baseline_index = int(np.argmin(np.where(np.isfinite(raw_scores), raw_scores, np.inf)))
        baseline_plan = bank.actions[baseline_index].copy()
        baseline_score = float(raw_scores[baseline_index])
        baseline_raw = float(bank.raw_costs[baseline_index])
        baseline_posts = bank.post_states[baseline_index:baseline_index + 1]
        baseline_obs = bank.observations[baseline_index:baseline_index + 1]
        baseline_origin = f"bank:{baseline_index}"
        baseline_error = None
        baseline_mean = None
        try:
            base_weights = softmin_weights(raw_scores, bank.temperature)
            plan = np.einsum("k,kha->ha", base_weights, bank.actions)
            baseline_mean = self._verify(bank, plan)
            score, _ = self._components(plan[None], baseline_mean["raw"], bank.reference_u)
            if score[0] < baseline_score:
                baseline_plan, baseline_score = plan, float(score[0])
                baseline_raw = float(baseline_mean["raw"][0])
                baseline_posts, baseline_obs = baseline_mean["posts"], baseline_mean["observations"]
                baseline_origin = "verified_baseline_mean"
        except (FloatingPointError, ValueError) as error:
            baseline_error = f"{type(error).__name__}: {error}"

        plan, selected_score, selected_raw = baseline_plan.copy(), baseline_score, baseline_raw
        selected_posts, selected_obs, origin = baseline_posts, baseline_obs, baseline_origin
        agreement = np.zeros(len(bank.actions))
        guided_scores = raw_scores.copy()
        support = np.zeros(bank.actions.shape[:2], dtype=bool)
        influence, numerical_revocation, guided_error = False, False, None
        guided_mean = None
        if enabled and self.alpha > 0:
            try:
                support = self._support(bank.observations, bank.active_mask, supported)
                support &= np.isfinite(bank.raw_costs)[:, None]
                if np.any(support):
                    if actor is None:
                        raise ValueError("Enabled supported guidance requires an actor.")
                    agreement = self._agreement(actor, bank.observations, bank.actions, support, meter)
                    guided_scores = raw_scores + 0.5 * self.alpha * agreement
                    index = int(np.argmin(np.where(np.isfinite(guided_scores), guided_scores, np.inf)))
                    candidate_plan = bank.actions[index].copy()
                    candidate_score, candidate_raw = float(guided_scores[index]), float(bank.raw_costs[index])
                    candidate_posts = bank.post_states[index:index + 1]
                    candidate_obs = bank.observations[index:index + 1]
                    candidate_origin = f"bank:{index}"
                    weights = softmin_weights(guided_scores, bank.temperature)
                    mean_plan = np.einsum("k,kha->ha", weights, bank.actions)
                    guided_mean = self._verify(bank, mean_plan)
                    mean_support = self._support(guided_mean["observations"], guided_mean["active"], supported)
                    mean_agreement = self._agreement(actor, guided_mean["observations"],
                                                      mean_plan[None], mean_support, meter)
                    mean_score, _ = self._components(mean_plan[None], guided_mean["raw"], bank.reference_u)
                    mean_score += 0.5 * self.alpha * mean_agreement
                    if mean_score[0] < candidate_score:
                        candidate_plan, candidate_score = mean_plan, float(mean_score[0])
                        candidate_raw = float(guided_mean["raw"][0])
                        candidate_posts, candidate_obs = guided_mean["posts"], guided_mean["observations"]
                        candidate_origin = "verified_guided_mean"
                    plan, selected_score, selected_raw = candidate_plan, candidate_score, candidate_raw
                    selected_posts, selected_obs, origin = candidate_posts, candidate_obs, candidate_origin
                    influence = True
            except (FloatingPointError, ValueError) as error:
                guided_error = f"{type(error).__name__}: {error}"
                numerical_revocation = True

        arrays = dict(bank_actions=bank.actions, bank_observations=bank.observations,
                      bank_post_states=bank.post_states, bank_raw_costs=bank.raw_costs,
                      reference_u=bank.reference_u, initial_state=bank.initial_state,
                      bank_active_mask=bank.active_mask, raw_plan=bank.raw_plan,
                      proximity_costs=prox, baseline_scores=raw_scores,
                      agreement_squared=agreement, guided_scores=guided_scores,
                      support=support, selected_plan=plan,
                      selected_post_states=selected_posts,
                      selected_pre_action_observations=selected_obs,
                      baseline_plan=baseline_plan)
        if bank.warmstart is not None:
            arrays["initial_warmstart"] = bank.warmstart
        for prefix, result in (("baseline_mean", baseline_mean), ("guided_mean", guided_mean)):
            if result is not None:
                arrays.update({f"{prefix}_plan": result["plan"],
                               f"{prefix}_post_states": result["posts"],
                               f"{prefix}_observations": result["observations"],
                               f"{prefix}_raw_costs": result["raw"]})
        evidence = meter.save_arrays(tag, **arrays) if meter is not None else None
        info = _json_safe(dict(
            bank_digest=bank.bank_digest, actor_hash=_actor_hash(actor),
            enabled=bool(enabled), alpha=self.alpha, eta_u=self.eta_u,
            influence_used=influence, numerical_revocation=numerical_revocation,
            changed_action=bool(not np.array_equal(plan[0], baseline_plan[0])),
            raw_first_action=bank.raw_plan[0], baseline_first_action=baseline_plan[0],
            guided_first_action=plan[0], raw_score=bank.raw_score,
            baseline_score=baseline_score, baseline_task_cost=baseline_raw,
            selected_score=selected_score, selected_task_cost=selected_raw,
            selected_origin=origin, baseline_origin=baseline_origin,
            supported_rows=int(np.sum(support)),
            baseline_verification_error=baseline_error, guided_error=guided_error,
            raw_diagnostics=bank.diagnostics, evidence=evidence,
        ))
        if meter is not None:
            meter.record("coupled_selection", **info)
        return np.array(plan, copy=True), info
