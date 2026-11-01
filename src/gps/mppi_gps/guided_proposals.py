"""H5.0: isolate proposal guidance from a protected task-only producer.

The existing torque agreement is measured in every arm. Only arm P puts it
inside the second producer's MPPI update. Observer computations in I and S
return the unmodified task score and cannot change either producer's state.
No simulator, neural library, policy checkpoint or experiment is loaded here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from src.utils.config import MPPIConfig
from src.utils.math import softmin_weights
from src.utils.rollout_data import pre_action_states

from .search import (
    CoupledSelector,
    IndependentMPPI,
    SearchBank,
    _active_mask,
    _actor_hash,
    _digest,
    _json_safe,
    _readonly,
)


class NativeGuardViolation(ValueError):
    """A finite verified blend exceeds the common native-cost bound."""


@dataclass(frozen=True)
class Agreement:
    """Existing D reduction and the exact predictions used to compute it."""

    squared: np.ndarray
    support: np.ndarray
    predictions: np.ndarray
    request_id: str | None = None


def agreement(
    actor: Any,
    observations: np.ndarray,
    actions: np.ndarray,
    active_mask: np.ndarray,
    raw_costs: np.ndarray,
    meter: Any = None,
    supported: bool | Callable | None = None,
    *,
    purpose: str = "agreement",
    scored_native_query_ids: list[str] | None = None,
) -> Agreement:
    """Sum six native bounded-torque errors, then time, with existing masks.

    The mask includes the fall-causing action and excludes later actions.
    There is no horizon/action averaging, motion distance, or error weighting.
    Predictions are in the flattened supported-row order used by the selector.
    """
    support = CoupledSelector._support(observations, active_mask, supported)
    support &= np.isfinite(raw_costs)[:, None]
    native_query_ids = [value["query_id"] for value in getattr(meter, "recent_model_queries", ())]
    if scored_native_query_ids is None:
        scored_native_query_ids = native_query_ids[-1:]
    request = (
        meter.record(
            "agreement_requested",
            purpose=purpose,
            actor_hash=_actor_hash(actor),
            input_digest=_digest(observations, actions, support, raw_costs),
            supported_rows=int(np.count_nonzero(support)),
            native_query_ids=native_query_ids,
            scored_native_query_ids=scored_native_query_ids,
            context=copy.deepcopy(getattr(meter, "context", {})),
        )
        if meter is not None
        else None
    )

    class Capture:
        predictions = np.empty((0, actions.shape[-1]), dtype=float)

        def predict(self, rows: np.ndarray, meter: Any = None) -> np.ndarray:
            if actor is None:
                raise ValueError("Agreement observations require a bound actor.")
            values = np.asarray(actor.predict(rows, meter=meter))
            self.predictions = values.copy()
            return values

    capture = Capture()
    try:
        squared = CoupledSelector._agreement(capture, observations, actions, support, meter)
    except BaseException as error:
        if meter is not None:
            meter.record(
                "agreement_error",
                request_id=request,
                error_type=type(error).__name__,
                error=str(error),
                context=copy.deepcopy(getattr(meter, "context", {})),
            )
        raise
    return Agreement(
        _readonly(squared), _readonly(support), _readonly(capture.predictions), request
    )


def _save(meter: Any, event: str, arrays: dict, **fields: Any) -> dict | None:
    if meter is None:
        return None
    fields.setdefault("context", copy.deepcopy(getattr(meter, "context", {})))
    request = meter.record(event, **_json_safe(fields))
    reference = meter.save_arrays(f"h5_{request}", **arrays)
    meter.record(event + "_evidence", request_id=request, evidence=reference)
    return reference


class GuidedProposalMPPI(IndependentMPPI):
    """Own a second nominal/RNG and optionally guide its optimization passes.

    ``enabled=False`` still measures agreement as explicitly paid observer
    work. It returns the exact incoming score object to the MPPI hook.
    Neither the guide nor a subsequently selected plan replaces this nominal.
    """

    def propose(
        self,
        state: np.ndarray,
        warmstart: np.ndarray | None = None,
        *,
        actor: Any,
        meter: Any = None,
        enabled: bool,
        alpha: float,
        supported: bool | Callable | None = None,
    ) -> SearchBank:
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and nonnegative.")
        initial = np.array(state, dtype=float, copy=True)
        warmstart = None if warmstart is None else np.array(warmstart, dtype=float, copy=True)
        reference = np.clip(self._planner.U.copy(), *self.env.action_bounds)
        passes = []

        def score_pass(*, states, actions, costs, base_score, lam):
            observations = self.env.rollout_states_to_obs(pre_action_states(initial, states))
            measured = agreement(
                actor,
                observations,
                actions,
                _active_mask(self.env, states),
                costs,
                meter,
                supported,
                purpose="proposal_pass",
            )
            policy_cost = 0.5 * alpha * measured.squared
            score = base_score + policy_cost if enabled else base_score
            evidence = _save(
                meter,
                "proposal_pass_scored",
                dict(
                    initial_state=initial,
                    nominal_before=self._planner.U.copy(),
                    support=measured.support,
                    predictions=measured.predictions,
                    agreement_squared=measured.squared,
                    score=score,
                ),
                pass_index=len(passes),
                actor_hash=_actor_hash(actor),
                guidance_enabled=bool(enabled),
                alpha=float(alpha),
                temperature=float(lam),
                rollout_digest=_digest(initial, warmstart, actions, states, costs),
                supported_rows=int(np.count_nonzero(measured.support)),
                agreement_request_id=measured.request_id,
            )
            passes.append(
                dict(
                    nominal_before=_readonly(self._planner.U),
                    score=_readonly(score),
                    agreement=measured,
                    evidence=evidence,
                )
            )
            return {
                "score": score,
                "fallback_score": base_score,
                "info": {
                    "active": float(enabled and np.any(measured.support)),
                    "policy_cost_mean": float(np.mean(policy_cost)) if enabled else 0.0,
                    "policy_cost_std": float(np.std(policy_cost)) if enabled else 0.0,
                    "score_mean": self._planner._finite_mean(score),
                },
            }

        _, diagnostics = self._planner.plan_step(
            initial, coupling=score_pass, initial_warmstart=warmstart
        )
        data = self._planner.get_rollout_data()
        actions = np.array(data["actions"], copy=True)
        posts = np.array(data["states"], copy=True)
        costs = np.array(data["costs"], dtype=float, copy=True)
        mean = self._planner.last_plan
        mean_error = None
        try:
            kwargs = {} if warmstart is None else {"initial_warmstart": warmstart}
            mean_posts, mean_cost, _ = self.env.batch_rollout(initial, mean[None], **kwargs)
            mean_posts, mean_cost = np.asarray(mean_posts), np.asarray(mean_cost, dtype=float)
            if mean_posts.shape != (1, *posts.shape[1:]) or mean_cost.shape != (1,):
                raise ValueError("Proposal mean verification returned incompatible shapes.")
        except (FloatingPointError, ValueError) as error:
            mean_error = f"{type(error).__name__}: {error}"
            mean_posts = np.full((1, *posts.shape[1:]), np.nan)
            mean_cost = np.array([np.inf])
        actions = np.concatenate((actions, mean[None]), axis=0)
        posts = np.concatenate((posts, mean_posts), axis=0)
        costs = np.concatenate((costs, mean_cost), axis=0)
        if not np.any(np.isfinite(costs)):
            raise FloatingPointError("No verified finite proposal plan.")
        best = int(np.argmin(np.where(np.isfinite(costs), costs, np.inf)))
        observations = self.env.rollout_states_to_obs(pre_action_states(initial, posts))
        self.last_passes = tuple(passes)
        return SearchBank(
            actions=_readonly(actions),
            observations=_readonly(observations),
            post_states=_readonly(posts),
            raw_costs=_readonly(costs),
            reference_u=_readonly(reference),
            raw_plan=_readonly(actions[best]),
            raw_score=float(costs[best]),
            initial_state=_readonly(initial),
            warmstart=None if warmstart is None else _readonly(warmstart),
            active_mask=_readonly(_active_mask(self.env, posts)),
            temperature=float(diagnostics["lam"]),
            provenance=("proposal_sample",) * (len(actions) - 1) + ("verified_proposal_mean",),
            diagnostics=MappingProxyType(
                dict(
                    diagnostics, raw_mean_verification_error=mean_error, measured_passes=len(passes)
                )
            ),
            bank_digest=_digest(initial, warmstart, actions, posts, costs, reference),
            _env=self.env,
        )


def merge_banks(protected: SearchBank, proposal: SearchBank) -> SearchBank:
    """Merge immutable final banks, keeping the protected proximity anchor."""
    if (
        protected._env is not proposal._env
        or not np.array_equal(protected.initial_state, proposal.initial_state)
        or protected.temperature != proposal.temperature
        or (protected.warmstart is None) != (proposal.warmstart is None)
        or (
            protected.warmstart is not None
            and not np.array_equal(protected.warmstart, proposal.warmstart)
        )
    ):
        raise ValueError("Both streams must share the same native query context and temperature.")
    arrays = {
        name: _readonly(np.concatenate((getattr(protected, name), getattr(proposal, name))))
        for name in ("actions", "observations", "post_states", "raw_costs", "active_mask")
    }
    return SearchBank(
        **arrays,
        reference_u=protected.reference_u,
        raw_plan=protected.raw_plan,
        raw_score=protected.raw_score,
        initial_state=protected.initial_state,
        warmstart=protected.warmstart,
        temperature=protected.temperature,
        provenance=tuple("protected:" + p for p in protected.provenance)
        + tuple("proposal:" + p for p in proposal.provenance),
        diagnostics=MappingProxyType(
            dict(protected_digest=protected.bank_digest, proposal_digest=proposal.bank_digest)
        ),
        bank_digest=_digest(
            protected.initial_state,
            protected.warmstart,
            arrays["actions"],
            arrays["post_states"],
            arrays["raw_costs"],
            protected.reference_u,
        ),
        _env=protected._env,
    )


class TwoStreamMPPI:
    """Common H5.0 controller; I/S/P vary only proposal/selector influence.

    Every complete decision attempts two producer mean verifications and two
    selector mean verifications. Every arm measures the second stream's pass
    agreement, merged-bank agreement and guided-mean agreement. I's guided
    selector is a discarded observer. Native cost feasibility is independent
    of agreement, with original measured costs retained unchanged.
    """

    def __init__(
        self,
        env: Any,
        cfg: MPPIConfig,
        protected_seed: int,
        proposal_seed: int,
        *,
        alpha: float = 0.1199341983507337,
        eta_u: float = 0.01,
        native_cost_ratio: float = 1.05,
    ):
        if not np.isfinite(native_cost_ratio) or native_cost_ratio < 1:
            raise ValueError("Native cost ratio must be finite and at least one.")
        self.selector = CoupledSelector(alpha=alpha, eta_u=eta_u)
        self.native_cost_ratio = float(native_cost_ratio)
        self.protected = IndependentMPPI(env, cfg, protected_seed)
        self.proposal = GuidedProposalMPPI(env, cfg, proposal_seed)
        self.last_bank: SearchBank | None = None

    def snapshot(self) -> dict:
        return dict(protected=self.protected.snapshot(), proposal=self.proposal.snapshot())

    def private_state(self) -> dict:
        return dict(
            protected=self.protected.private_state(), proposal=self.proposal.private_state()
        )

    def restore(self, snapshot: dict) -> None:
        self.protected.restore(snapshot["protected"])
        self.proposal.restore(snapshot["proposal"])
        self.last_bank = None

    def decide(
        self,
        state: np.ndarray,
        actor: Any,
        meter: Any = None,
        *,
        arm: str,
        warmstart: np.ndarray | None = None,
        supported: bool | Callable | None = None,
    ) -> tuple[np.ndarray, dict]:
        if arm not in ("I", "S", "P"):
            raise ValueError("H5.0 arm must be I, S or P.")
        before = self.private_state()
        incoming_context = _save(
            meter,
            "h5_decision_context",
            dict(protected_before=before["protected"], proposal_before=before["proposal"]),
            arm=arm,
            input_digest=_digest(np.asarray(state), warmstart),
        )
        if meter is not None:
            meter.record(
                "h5_decision_started",
                arm=arm,
                input_digest=_digest(np.asarray(state), warmstart),
                protected_nominal_digest=_digest(before["protected"]["U"]),
                proposal_nominal_digest=_digest(before["proposal"]["U"]),
                optimization_iterations=self.protected.cfg.initial_iterations
                if before["protected"]["initial_plan"]
                else 1,
                context=copy.deepcopy(getattr(meter, "context", {})),
            )
        protected = self.protected.propose(state, warmstart)
        proposal = self.proposal.propose(
            state,
            warmstart,
            actor=actor,
            meter=meter,
            enabled=arm == "P",
            alpha=self.selector.alpha,
            supported=supported,
        )
        bank = merge_banks(protected, proposal)
        if protected.raw_score < 0 or np.any(bank.raw_costs[np.isfinite(bank.raw_costs)] < 0):
            raise ValueError("The relative native guard requires this nonnegative Walker task.")
        bound = self.native_cost_ratio * protected.raw_score + 1e-12
        feasible = np.isfinite(bank.raw_costs) & (bank.raw_costs <= bound)
        raw_scores, proximity = self.selector._components(
            bank.actions, bank.raw_costs, bank.reference_u
        )
        query_ids = [value["query_id"] for value in getattr(meter, "recent_model_queries", ())]
        passes = proposal.diagnostics["optimization_iterations"]
        final_query_indices = (passes - 1, passes, 2 * passes, 2 * passes + 1)
        final_query_ids = [query_ids[index] for index in final_query_indices] if query_ids else []
        cache_queries = dict(
            zip(
                (
                    "protected_final_sample",
                    "protected_verified_mean",
                    "proposal_final_sample",
                    "proposal_verified_mean",
                ),
                final_query_ids,
                strict=bool(final_query_ids),
            )
        )
        measured = agreement(
            actor,
            bank.observations,
            bank.actions,
            bank.active_mask,
            bank.raw_costs,
            meter,
            supported,
            purpose="final_bank",
            scored_native_query_ids=final_query_ids,
        )
        guided_scores = raw_scores + 0.5 * self.selector.alpha * measured.squared
        alternatives = {}
        means = {}
        for name, scores in (("baseline", raw_scores), ("guided", guided_scores)):
            eligible_scores = np.where(feasible, scores, np.inf)
            index = int(np.argmin(eligible_scores))
            weights = softmin_weights(eligible_scores, bank.temperature)
            mean = np.einsum("k,kha->ha", weights, bank.actions)
            result = dict(
                plan=bank.actions[index].copy(),
                raw_cost=float(bank.raw_costs[index]),
                score=float(scores[index]),
                origin=f"bank:{index}:{bank.provenance[index]}",
                fallback=False,
                verification_error=None,
                verification_status="pending",
            )
            verified = None
            try:
                verified = self.selector._verify(bank, mean)
                means[name] = dict(
                    plan=mean,
                    raw_cost=float(verified["raw"][0]),
                    agreement=None,
                    rollout_digest=_digest(
                        bank.initial_state, bank.warmstart, mean, verified["posts"]
                    ),
                )
                if not np.all(np.isfinite(verified["posts"])):
                    raise FloatingPointError("Selector mean returned nonfinite physical states.")
                measured_mean = None
                if name == "guided":
                    try:
                        measured_mean = agreement(
                            actor,
                            verified["observations"],
                            mean[None],
                            verified["active"],
                            verified["raw"],
                            meter,
                            supported,
                            purpose="guided_selector_mean",
                        )
                    except (ValueError, FloatingPointError) as error:
                        raise RuntimeError(
                            "Agreement failed on a verified selector mean."
                        ) from error
                mean_cost = float(verified["raw"][0])
                means[name]["agreement"] = measured_mean
                if mean_cost < 0:
                    raise ValueError("A negative native task score cannot use the relative guard.")
                if mean_cost > bound:
                    raise NativeGuardViolation(
                        "Verified selector mean violates native feasibility guard."
                    )
                result["verification_status"] = "ok"
                mean_score = self.selector._components(
                    mean[None], verified["raw"], bank.reference_u
                )[0][0]
                if measured_mean is not None:
                    mean_score += 0.5 * self.selector.alpha * measured_mean.squared[0]
                if mean_score < result["score"]:
                    result.update(
                        plan=mean.copy(),
                        raw_cost=mean_cost,
                        score=float(mean_score),
                        origin=f"verified_{name}_mean",
                    )
            except (FloatingPointError, ValueError) as error:
                # Never execute an unverified blend or silently use a policy action.
                result.update(
                    plan=protected.raw_plan.copy(),
                    raw_cost=protected.raw_score,
                    score=None,
                    origin="protected_guard_fallback",
                    fallback=True,
                    verification_error=f"{type(error).__name__}: {error}",
                    verification_status="guard_fallback"
                    if isinstance(error, NativeGuardViolation)
                    else "native_error",
                )
                means.setdefault(name, dict(plan=mean, raw_cost=None, agreement=None))
            alternatives[name] = result
        chosen = alternatives["baseline" if arm == "I" else "guided"]
        producer_validation = {
            f"{name}_mean_verification_error": value.diagnostics.get("raw_mean_verification_error")
            for name, value in (("protected", protected), ("proposal", proposal))
        }
        for name, value in (("protected", protected), ("proposal", proposal)):
            producer_validation[f"{name}_mean_native_cost_finite"] = bool(
                np.isfinite(value.raw_costs[value.mean_index])
            )
            producer_validation[f"{name}_mean_states_finite"] = bool(
                np.all(np.isfinite(value.post_states[value.mean_index]))
            )
        self.last_bank = bank
        arrays = dict(
            raw_costs=bank.raw_costs,
            feasible=feasible,
            proximity=proximity,
            agreement_squared=measured.squared,
            support=measured.support,
            predictions=measured.predictions,
            baseline_scores=raw_scores,
            guided_scores=guided_scores,
            selected_plan=chosen["plan"],
            baseline_mean=means["baseline"]["plan"],
            guided_mean=means["guided"]["plan"],
            baseline_mean_raw_cost=np.array(
                [] if means["baseline"]["raw_cost"] is None else [means["baseline"]["raw_cost"]]
            ),
            guided_mean_raw_cost=np.array(
                [] if means["guided"]["raw_cost"] is None else [means["guided"]["raw_cost"]]
            ),
            protected_before=before["protected"],
            proposal_before=before["proposal"],
            protected_after=self.protected.private_state(),
            proposal_after=self.proposal.private_state(),
            protected_final_weights=self.protected._planner.get_rollout_data()["weights"],
            proposal_final_weights=self.proposal._planner.get_rollout_data()["weights"],
        )
        guided_mean = means["guided"]["agreement"]
        if guided_mean is not None:
            arrays.update(
                guided_mean_support=guided_mean.support,
                guided_mean_predictions=guided_mean.predictions,
                guided_mean_agreement_squared=guided_mean.squared,
            )
        evidence = _save(
            meter,
            "two_stream_selection",
            arrays,
            arm=arm,
            actor_hash=_actor_hash(actor),
            bank_digest=bank.bank_digest,
            protected_bank_digest=protected.bank_digest,
            proposal_bank_digest=proposal.bank_digest,
            producer_cache_query_ids=cache_queries,
            decision_context_group=None if incoming_context is None else incoming_context["group"],
            native_guard_bound=bound,
            selected_origin=chosen["origin"],
            selected_native_cost=chosen["raw_cost"],
            observer_guided_selection=arm == "I",
            bank_agreement_request_id=measured.request_id,
            guided_mean_agreement_request_id=None
            if guided_mean is None
            else guided_mean.request_id,
            baseline_verification_error=alternatives["baseline"]["verification_error"],
            guided_verification_error=alternatives["guided"]["verification_error"],
            baseline_verification_status=alternatives["baseline"]["verification_status"],
            guided_verification_status=alternatives["guided"]["verification_status"],
            **producer_validation,
        )
        info = dict(
            arm=arm,
            selected_origin=chosen["origin"],
            selected_raw_cost=chosen["raw_cost"],
            fallback=chosen["fallback"],
            native_guard_bound=bound,
            feasible_count=int(np.count_nonzero(feasible)),
            baseline_verification_error=alternatives["baseline"]["verification_error"],
            guided_verification_error=alternatives["guided"]["verification_error"],
            baseline_verification_status=alternatives["baseline"]["verification_status"],
            guided_verification_status=alternatives["guided"]["verification_status"],
            **producer_validation,
            protected_bank_digest=protected.bank_digest,
            proposal_bank_digest=proposal.bank_digest,
            bank_digest=bank.bank_digest,
            optimization_iterations=proposal.diagnostics["optimization_iterations"],
            # Episode infos must not duplicate every array binding 1,000 times
            # inside one Unicode JSON array. Full refs are already durable in
            # the selection ledger/index; this pointer identifies that group.
            evidence=None
            if evidence is None
            else {key: value for key, value in evidence.items() if key != "arrays"},
        )
        return chosen["plan"].copy(), _json_safe(info)
