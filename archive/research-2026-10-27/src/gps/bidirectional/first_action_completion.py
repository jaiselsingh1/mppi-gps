"""Classical one-action rollout with a frozen actor continuation, Ending only.

This selector neither proposes into nor updates IndependentSearch. Its complete
Q^pi queries evaluate one candidate action followed by deterministic pi, not the
subsequently executed replanning controller. That controller still needs fresh
complete qualification. These hypothetical queries are not actor-owned training
episodes. No novelty is claimed for the rollout construction.
"""
from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import hashlib
import json
import time

import numpy as np

from src.ending.core import Task as EndingTask


SEMANTICS = "one_candidate_action_then_deterministic_actor"


def policy_hash(actor):
    from .policy import policy_hash as implementation
    return implementation(actor)


def controller_hash(actor, family):
    from .policy import controller_hash as implementation
    return implementation(actor, deterministic=True, family=family)


def attach_meter(actor, archive):
    from .meter import attach_meter as implementation
    implementation(actor, archive, "first_action_actor_completion")


def immutable(values):
    value = np.ascontiguousarray(values, dtype=np.float64)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def completion_work_bound(remaining):
    """Maximum additional query controls and actor rows for one decision."""
    if type(remaining) is not int or remaining < 0:
        raise ValueError("Remaining controls must be a nonnegative integer")
    return dict(model_controls=5*remaining, model_queries=5*remaining,
                actor_rows=(1+5*(remaining-1) if remaining else 0), candidates=5)


class _NumericalActorError(ValueError):
    pass


@dataclass(frozen=True)
class CompletionSelection:
    plan: np.ndarray
    query_ids: tuple[str, ...]
    changed: bool
    guidance_weight: float
    influenced: bool
    raw_score: float
    augmented_score: float
    reason: str
    score_semantics: str
    selected_candidate_id: str | None = None


class FirstActionCompletionSelector:
    """Rerank at most five first actions using actual full actor-tail queries.

    Trials require exactly weight1 and current eligible support, normally from
    CandidateTrustManager(1). Ordinary selection requires promoted support.
    A standard manager's .1-first ladder is not silently overridden here.
    """
    def __init__(self, executor, actor, trust, family=0):
        if type(family) is not int or family < 0:
            raise ValueError("A nonnegative integer family is required")
        self.executor, self.actor, self.trust, self.family = executor, actor, trust, family
        task = executor.task
        if not isinstance(getattr(task, "dynamics", None), EndingTask):
            raise ValueError("Full-completion sums are restricted to the endpoint-cost-free Ending task")
        if task.config.T != task.dynamics.T:
            raise ValueError("Ending task and controller horizons differ")
        self._actor_hash = policy_hash(actor)
        self._actor_controller_hash = controller_hash(actor, family)
        self._task_hash, self._T = task.task_hash, int(task.config.T)
        self.protocol_specification = dict(version="ending-first-action-actor-completion-v1",
            semantics=SEMANTICS, actor_hash=self._actor_hash,
            actor_controller_hash=self._actor_controller_hash, task_hash=self._task_hash,
            family=family, total_steps=self._T, empirical_quantiles=[0., .5, 1.],
            quantile_rule="floor(q*(K-1)) of stable-sorted native bank first actions",
            order=["actor_first", "independent_champion", "bank_min", "bank_median", "bank_max"],
            deduplication="exact scalar equality; first role retains tie priority",
            cost="sum one-control Ending totals through true T; no endpoint heuristic",
            selection="stable minimum complete cost; actor-first ties",
            deployment="current promoted support; eligible weight1 trial only")
        self.protocol_identity = hashlib.sha256(json.dumps(self.protocol_specification,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _fallback(self, bank, reason, query_ids=()):
        return CompletionSelection(immutable(bank.independent_plan), tuple(query_ids), False,
            0., False, float(bank.independent_score), float(bank.independent_score), reason,
            "independent_short_horizon_fallback")

    def _version_check(self):
        try:
            current = policy_hash(self.actor)
        except (ValueError, FloatingPointError) as error:
            raise _NumericalActorError("Actor version is nonfinite") from error
        if current != self._actor_hash:
            raise _NumericalActorError("Frozen actor version changed")

    def _record_supports(self, record, observation, elapsed, remaining, state):
        if ((record.actor_hash, record.actor_controller_hash, record.task_hash, record.family_id)
                != (self._actor_hash, self._actor_controller_hash, self._task_hash, self.family)
                or record.state != state or record.deterministic is not True
                or record.total_steps != self._T):
            return False
        if not (record.time_interval[0] <= elapsed <= record.time_interval[1]
                and record.remaining_interval[0] <= remaining <= record.remaining_interval[1]):
            return False
        anchors, scale = np.asarray(record.support_anchors), np.asarray(record.observation_scale)
        return bool(np.isfinite(observation).all() and np.any(np.linalg.norm(
            (observation[None]-anchors)/scale[None], axis=1) <= record.support_radius))

    def _weight(self, observation, elapsed, remaining, trial_weight):
        if trial_weight is not None and (isinstance(trial_weight, bool)
                or not isinstance(trial_weight, (int, float)) or trial_weight != 1):
            raise ValueError("Completion qualification trials require exactly weight1")
        state = "promoted" if trial_weight is None else "eligible"
        if not any(self._record_supports(r, observation, elapsed, remaining, state)
                   for r in self.trust.records):
            return 0.
        weight = self.trust.weight_for(self._actor_hash, self._task_hash, self.family,
            elapsed, remaining, observation, trial_weight=trial_weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Invalid current-state trust weight")
        return float(weight)

    def _act(self, snapshot):
        # The real actor's forward hooks record requests before inference. Actor
        # execution exceptions (including work-cap rejection) propagate. Only a
        # returned invalid bounded action is a numerical-revocation fallback.
        value = np.asarray(self.actor.act(self.executor.task.observation(snapshot),
            family=self.family, deterministic=True), dtype=np.float64)
        if value.shape != (1,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
            raise _NumericalActorError("Actor returned an invalid bounded scalar action")
        return immutable(value)

    def _candidates(self, bank, actor_first):
        actions = np.asarray(bank.actions)
        independent = np.asarray(bank.independent_plan)
        if (actions.ndim != 3 or actions.shape[0] < 1 or actions.shape[1] < 1
                or actions.shape[2] != 1 or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1) or independent.ndim != 2
                or independent.shape[0] < 1 or independent.shape[1] != 1
                or not np.isfinite(independent).all() or np.any(np.abs(independent) > 1)):
            raise ValueError("Invalid independent scalar-action bank")
        order = np.argsort(actions[:, 0, 0], kind="stable")
        ranks = [0, (len(order)-1)//2, len(order)-1]
        proposals = [(float(actor_first[0]), "actor_first", None),
            (float(independent[0, 0]), "independent_champion", None)]
        proposals += [(float(actions[order[rank], 0, 0]), role, int(order[rank]))
            for rank, role in zip(ranks, ("bank_min", "bank_median", "bank_max"))]
        candidates = []
        for action, role, row in proposals:
            previous = next((candidate for candidate in candidates if candidate["action"] == action), None)
            alias = dict(role=role, bank_row=row)
            if previous is None:
                candidates.append(dict(candidate_id=f"candidate-{len(candidates):03d}",
                    action=action, roles=[alias]))
            else:
                previous["roles"].append(alias)
        return candidates

    def _save(self, request_id, bank, candidates, traces, costs, selected, status):
        archive = self.executor.archive
        raw = archive.save_npz(f"completion/{request_id}.npz",
            candidate_first_actions=np.array([c["action"] for c in candidates])[:, None],
            candidate_roles_json=np.array(json.dumps(candidates, sort_keys=True)),
            complete_candidate_costs=np.asarray(costs),
            completed_candidate_indices=np.arange(len(costs), dtype=np.int64),
            trace_candidate_indices=np.array([r["candidate_index"] for r in traces], dtype=np.int64),
            trace_steps=np.array([r["step"] for r in traces], dtype=np.int64),
            prestates=np.asarray([r["prestate"] for r in traces]).reshape(-1, 2),
            states=np.asarray([r["state"] for r in traces]).reshape(-1, 2),
            observations=np.asarray([r["observation"] for r in traces]).reshape(-1, 8),
            actions=np.asarray([r["action"] for r in traces]).reshape(-1, 1),
            raw_costs=np.array([r["cost"] for r in traces]),
            query_ids=np.asarray([r["query_id"] for r in traces], dtype=str),
            selected_index=np.array(-1 if selected is None else selected),
            actor_hash=np.array(self._actor_hash), actor_controller_hash=np.array(self._actor_controller_hash),
            context_id=np.array(bank.initial.context_id), initial_state=bank.initial.state,
            protocol_identity=np.array(self.protocol_identity), semantics=np.array(SEMANTICS),
            status=np.array(status))
        return raw

    def select(self, bank, *, trial_weight=None):
        started = time.perf_counter(); archive = self.executor.archive
        snapshot = bank.initial; state = np.asarray(snapshot.state)
        if (not is_dataclass(snapshot) or state.shape != (2,) or not np.isfinite(state).all()
                or type(snapshot.elapsed) is not int or state[0] != snapshot.elapsed
                or abs(state[1]) > 1.
                or not 0 <= snapshot.elapsed < self._T or self.executor.task.task_hash != self._task_hash):
            raise ValueError("Completion requires a finite current Ending snapshot")
        remaining = self._T-snapshot.elapsed
        candidates, traces, costs, query_ids = [], [], [], []
        request_id = None
        try:
            self._version_check()
            current = np.asarray(self.executor.task.observation(snapshot))
            if current.shape != (8,) or not np.isfinite(current).all():
                raise ValueError("Ending support requires one finite observation8")
            weight = self._weight(current, snapshot.elapsed, remaining, trial_weight)
            if weight == 0:
                return self._fallback(bank, "unsupported_current_state")
            request_id = archive.append("completion_requested", actor_hash=self._actor_hash,
                actor_controller_hash=self._actor_controller_hash, protocol_identity=self.protocol_identity,
                semantics=SEMANTICS, context_id=snapshot.context_id, elapsed=snapshot.elapsed,
                remaining=remaining, guidance_weight=weight, trial_weight=trial_weight,
                maximum_additional_work=completion_work_bound(remaining))
            attach_meter(self.actor, archive)
            actor_first = self._act(snapshot)
            candidates = self._candidates(bank, actor_first)
            archive.append("completion_candidates", request_id=request_id,
                actor_hash=self._actor_hash, candidates=candidates, tie_order="actor_first_then_recorded_order")
            for i, candidate in enumerate(candidates):
                local = replace(snapshot, state=immutable(snapshot.state))
                total = 0.
                for step in range(remaining):
                    action = immutable([candidate["action"]]) if step == 0 else self._act(local)
                    observation = np.asarray(self.executor.task.observation(local)).copy()
                    result = self.executor.model_rollout(local, action.reshape(1, 1, 1),
                        kind="actor_completion_step", metadata=dict(semantics=SEMANTICS,
                            completion_request_id=request_id, protocol_identity=self.protocol_identity,
                            actor_hash=self._actor_hash, actor_controller_hash=self._actor_controller_hash,
                            candidate_id=candidate["candidate_id"], candidate_index=i, step=step,
                            first_action_roles=candidate["roles"],
                            owner=dict(semantics=SEMANTICS, actor_hash=self._actor_hash,
                                actor_controller_hash=self._actor_controller_hash, family=self.family,
                                deterministic=True, candidate_id=candidate["candidate_id"])))
                    qid = self.executor.latest_query_id
                    if not isinstance(qid, str) or qid in query_ids:
                        raise RuntimeError("Completion model query reference missing or reused")
                    query_ids.append(qid)
                    next_states, totals = np.asarray(result.states), np.asarray(result.total_costs)
                    if (hasattr(result, "terminal_heuristic") or next_states.shape != (1, 1, 2)
                            or totals.shape != (1,) or not np.isfinite(next_states).all()
                            or not np.isfinite(totals).all() or np.any(totals < 0.)
                            or abs(next_states[0, 0, 1]) > 1.
                            or next_states[0, 0, 0] != local.elapsed+1):
                        raise RuntimeError("Completion received an invalid/heuristic-bearing Ending query result")
                    next_state = immutable(next_states[0, 0])
                    cost = float(totals[0]); total += cost
                    traces.append(dict(candidate_index=i, step=step, prestate=local.state.copy(),
                        state=next_state.copy(), action=action.copy(), observation=observation,
                        cost=cost, query_id=qid))
                    local = replace(local, state=next_state, elapsed=local.elapsed+1)
                if local.elapsed != self._T or not np.isfinite(total):
                    raise RuntimeError("Actor completion did not end with a finite true full cost")
                costs.append(total)
            self._version_check()
            selected = int(np.argmin(costs))
            raw = self._save(request_id, bank, candidates, traces, costs, selected, "complete")
            archive.append("completion_decision", request_id=request_id, raw=raw,
                actor_hash=self._actor_hash, protocol_identity=self.protocol_identity,
                semantics=SEMANTICS, candidate_costs=costs,
                candidate_ids=[c["candidate_id"] for c in candidates], selected_index=selected,
                selected_candidate_id=candidates[selected]["candidate_id"],
                selected_first_action=candidates[selected]["action"], query_ids=query_ids,
                guidance_weight=weight, completion_wall_seconds=time.perf_counter()-started)
            plan = immutable([[candidates[selected]["action"]]])
            changed = not np.array_equal(plan[0], bank.independent_plan[0])
            return CompletionSelection(plan, tuple(query_ids), changed, weight, True,
                costs[selected], costs[selected], "complete_actor_completion", SEMANTICS,
                candidates[selected]["candidate_id"])
        except _NumericalActorError as error:
            self.trust.revoke_actor(self._actor_hash, reason="actor_completion_numerical_revocation:"+str(error))
            raw = self._save(request_id, bank, candidates, traces, costs, None,
                "numerical_revocation") if request_id is not None else None
            archive.append("completion_revocation", request_id=request_id, raw=raw,
                actor_hash=self._actor_hash, reason=str(error), completed_query_ids=query_ids,
                completion_wall_seconds=time.perf_counter()-started)
            return self._fallback(bank, "numerical_revocation", query_ids)
        except BaseException as error:
            # No failed model call can silently become an independent action.
            # The executor's before-call journal remains authoritative for failed
            # or interrupted attempts; completed refs below are supplementary.
            try:
                raw = self._save(request_id, bank, candidates, traces, costs, None,
                    "execution_error") if request_id is not None else None
                archive.append("completion_error", request_id=request_id, raw=raw,
                    actor_hash=self._actor_hash, error_type=type(error).__name__, error=str(error),
                    completed_query_ids=query_ids, latest_executor_query_id=self.executor.latest_query_id,
                    latest_dynamics_query_id=getattr(self.executor.task.dynamics, "latest_query_id", None),
                    completion_wall_seconds=time.perf_counter()-started)
            except BaseException as evidence_error:
                error.add_note("Completion error evidence failed: "+type(evidence_error).__name__)
            raise
