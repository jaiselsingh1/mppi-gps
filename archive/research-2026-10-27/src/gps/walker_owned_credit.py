"""Complete candidate-owned credit plumbing for a new Walker diagnostic.

No existing study or legacy teacher helper is changed. Candidate ranking has
one source: complete measured returns of that very candidate. There is no
prefix, base-policy, critic, teacher-action, or capacity-witness fallback.
The rollout evaluator is imported only when an actual query is requested.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.utils.config import MPPIConfig


QUALIFIED_PLANNER_SIGNATURE = MappingProxyType({
    "K": 256, "H": 48, "lam": 3., "noise_sigma": .3,
    "use_is_correction": False, "noise_std": None, "noise_cov": None,
    "noise_lowpass_cutoff_hz": 23.4375, "noise_lowpass_sample_rate_hz": 125.,
    "noise_lowpass_order": 2, "initial_iterations": 2, "lam_target_neff_fraction": None,
})
TARGET_KIND = "complete_changed_actor_return"


def _immutable(value, *, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value, name, minimum=0):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _gamma(value):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0. <= value < 1.:
        raise ValueError("gamma must be finite in [0,1)")
    return float(value)


def _json_identity(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("ancestry must contain finite JSON metadata") from error


def finite_absorbing_cost(gamma, remaining, absorbing_cost=5.):
    """Pure finite geometric cost; no simulator or infinite reward adapter.

    This separately implemented arithmetic also lets the wrapper check the
    existing rollout helper's complete return against its raw transition trace.
    """
    gamma = _gamma(gamma)
    remaining = _integer(remaining, "remaining")
    if not np.isfinite(absorbing_cost) or absorbing_cost < 0.:
        raise ValueError("absorbing_cost must be nonnegative and finite")
    if not remaining:
        return 0.
    if gamma == 0.:
        return float(absorbing_cost)
    return float(absorbing_cost * math.expm1(remaining * math.log(gamma)) / math.expm1(math.log(gamma)))


def load_qualified_planner_config(path) -> MPPIConfig:
    """Load explicit tuned bytes, then fail closed on the entire signature.

    Optional absent JSON entries resolve through MPPIConfig defaults and must
    still match all declared settings. In particular no K32 legacy override,
    adaptive temperature, alternate covariance, or lowpass change is accepted.
    This verifies configuration identity only; it does not establish that the
    planner passed a fresh live qualification gate or any behavioral criterion.
    """
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Planner JSON must contain an object")
    try:
        config = MPPIConfig(**payload)
    except TypeError as error:
        raise ValueError("Planner JSON is not a complete recognized MPPIConfig signature") from error
    actual = asdict(config)
    if set(actual) != set(QUALIFIED_PLANNER_SIGNATURE):
        raise ValueError("MPPIConfig fields differ from the qualified signature")
    mismatches = []
    for name, expected in QUALIFIED_PLANNER_SIGNATURE.items():
        value = actual[name]
        if expected is None:
            matches = value is None
        elif isinstance(expected, bool):
            matches = type(value) is bool and value == expected
        elif isinstance(expected, int):
            matches = type(value) is int and value == expected
        else:
            matches = (type(value) in (int, float) and np.isfinite(value) and value == expected)
        if not matches:
            mismatches.append(name)
    if mismatches:
        raise ValueError("Planner differs from qualified tuned signature: " + ", ".join(mismatches))
    return config


def actor_owner_hash(actor):
    """Hash all actor parameters and buffers without action inference."""
    if not callable(getattr(actor, "state_dict", None)):
        raise ValueError("Actor ownership requires a complete state_dict")
    packet = actor.state_dict()
    if not isinstance(packet, Mapping) or not packet:
        raise ValueError("Actor state_dict must be a nonempty mapping")
    digest = hashlib.sha256()
    digest.update((type(actor).__module__ + "." + type(actor).__qualname__).encode())
    for name in sorted(packet):
        if not isinstance(name, str):
            raise ValueError("Actor state_dict keys must be strings")
        value = packet[name]
        if callable(getattr(value, "detach", None)):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError("Actor state_dict must contain finite numeric arrays")
        array = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def snapshot_digest(snapshot):
    """Identity includes both FULLPHYSICS state and excluded solver warmstart."""
    digest = hashlib.sha256()
    for name in ("state", "warmstart"):
        array = np.asarray(getattr(snapshot, name), dtype="<f8")
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("Snapshot state and warmstart must be finite vectors")
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _finite_return(raw_costs, *, remaining, executed_steps, terminated, gamma, absorbing_cost):
    remaining = _integer(remaining, "remaining", 1)
    executed_steps = _integer(executed_steps, "executed_steps")
    if executed_steps > remaining:
        raise ValueError("Executed steps exceed the complete remaining horizon")
    if type(terminated) not in (bool, np.bool_):
        raise ValueError("terminated must be an explicit boolean")
    if executed_steps < remaining and not terminated:
        raise ValueError("Incomplete healthy query cannot supply complete candidate credit")
    raw = np.asarray(raw_costs, dtype=np.float64)
    if raw.shape != (executed_steps,) or not np.isfinite(raw).all() or np.any(raw < 0.):
        raise ValueError("Raw costs must match every executed control")
    executed = float(np.dot(gamma ** np.arange(executed_steps), raw))
    absorbing_steps = remaining - executed_steps if terminated else 0
    absorption = gamma ** executed_steps * finite_absorbing_cost(gamma, absorbing_steps, absorbing_cost)
    return executed + absorption, executed, absorption, absorbing_steps


def _default_evaluator(snapshot, actor, **kwargs):
    from src.gps.mppi_experience import finite_policy_rollout
    return finite_policy_rollout(snapshot, actor, **kwargs)


def _validated_rollout(result, initial_state, initial_warmstart, *, remaining, gamma):
    prefix = result.prefix
    if (prefix.requested_steps != remaining or prefix.gamma != gamma
            or prefix.unhealthy_cost != 5. or bool(prefix.healthy) == bool(prefix.terminated)
            or (prefix.terminated and prefix.truncated)):
        raise ValueError("Rollout continuation/horizon metadata differs from the candidate query")
    total, executed, absorption, absorbing_steps = _finite_return(prefix.raw_costs,
        remaining=remaining, executed_steps=prefix.steps, terminated=prefix.terminated,
        gamma=gamma, absorbing_cost=5.)
    if (not np.isclose(prefix.cost, executed, rtol=1e-11, atol=1e-9)
            or not np.isclose(result.cost, total, rtol=1e-11, atol=1e-9)):
        raise ValueError("Complete return disagrees with raw costs and finite absorbing fall cost")
    steps = prefix.steps
    states, warms, observations = (np.asarray(getattr(result, name))
                                  for name in ("states", "warmstarts", "observations"))
    actions = np.asarray(prefix.actions)
    if (states.ndim != 2 or warms.ndim != 2 or states.shape[0] != steps + 1
            or warms.shape[0] != steps + 1 or observations.shape != (steps + 1, 17)
            or actions.shape != (steps, 6) or not np.isfinite(states).all()
            or not np.isfinite(warms).all() or not np.isfinite(observations).all()
            or not np.isfinite(actions).all()):
        raise ValueError("Complete raw state/warmstart/observation/action trace is missing")
    if (not np.array_equal(states[0], initial_state)
            or not np.array_equal(warms[0], initial_warmstart)
            or not np.array_equal(states[-1], prefix.endpoint_state)
            or not np.array_equal(warms[-1], prefix.endpoint_warmstart)
            or not np.array_equal(observations[-1], prefix.endpoint_obs)):
        raise ValueError("Raw trace does not preserve the exact queried snapshot and endpoint")
    if (result.ledger.get("policy_controls") != steps
            or prefix.ledger.get("policy_controls") != steps
            or any(result.ledger.get(key, 0) != 0 for key in
                   ("teacher_controls", "teacher_candidate_controls", "teacher_planner_calls",
                    "teacher_optimization_iterations"))):
        raise ValueError("Complete candidate query has inconsistent work or a foreign controller")
    return total, executed, absorption, absorbing_steps, steps


def candidate_owned_rollout(snapshot, actor, *, remaining, gamma, owner_hash,
                            context_id, context_ancestry, query_ref,
                            phase="candidate_complete", reset_id=0, evaluator=None):
    """Query and retain the full named candidate, then verify raw finite credit.

    The injected evaluator uses the finite_policy_rollout signature. Neither a
    base actor nor a rescue controller can be passed through this API. Actual
    work remains in the returned original ExperienceRollout and its ledger.
    Post-query validation errors retain ``rollout``, ``ledger`` and
    ``query_metadata`` attributes so a failed query's work remains chargeable.
    An evaluator that itself fails before returning must preserve its partial
    work externally. A future executor needs durable per-attempt/transition
    accounting; this wrapper alone cannot recover an interrupted physics call.
    """
    remaining, gamma = _integer(remaining, "remaining", 1), _gamma(gamma)
    owner_hash = _text(owner_hash, "owner_hash")
    context_id, query_ref = _text(context_id, "context_id"), _text(query_ref, "query_ref")
    phase = _text(phase, "phase")
    if not isinstance(context_ancestry, Mapping):
        raise ValueError("context_ancestry must be a mapping")
    _json_identity(context_ancestry)
    initial_digest = snapshot_digest(snapshot)
    initial_state = np.array(snapshot.state, copy=True)
    initial_warmstart = np.array(snapshot.warmstart, copy=True)
    if actor_owner_hash(actor) != owner_hash:
        raise ValueError("Candidate actor does not match declared owner hash")
    result = (evaluator or _default_evaluator)(snapshot, actor, remaining=remaining,
                                               gamma=gamma, reset_id=reset_id)
    try:
        if actor_owner_hash(actor) != owner_hash:
            raise ValueError("Candidate actor changed during its complete query")
        if snapshot_digest(snapshot) != initial_digest:
            raise ValueError("Evaluator changed the queried state or solver warmstart")
        total, executed, absorption, absorbing_steps, steps = _validated_rollout(
            result, initial_state, initial_warmstart, remaining=remaining, gamma=gamma)
    except Exception as error:
        error.rollout = result
        error.ledger = deepcopy(getattr(result, "ledger", {}))
        error.query_metadata = {"owner_hash": owner_hash, "context_id": context_id,
            "query_ref": query_ref, "phase": phase, "remaining": remaining, "gamma": gamma,
            "context_ancestry": deepcopy(dict(context_ancestry)), "snapshot_digest": initial_digest,
            "failed_query": True}
        raise
    prefix = result.prefix
    return {"target_kind": TARGET_KIND, "continuation_owner": "candidate", "owner_hash": owner_hash,
            "phase": phase, "query_ref": query_ref, "context_id": context_id,
            "context_ancestry": deepcopy(dict(context_ancestry)), "snapshot_digest": initial_digest,
            "remaining": remaining, "executed_steps": steps, "gamma": gamma,
            "complete": True, "cutoff": False, "terminated": bool(prefix.terminated),
            "truncated": bool(prefix.truncated), "unhealthy_cost": 5., "cost": float(total),
            "executed_cost": executed, "absorbing_cost": absorption, "absorbing_steps": absorbing_steps,
            "raw_costs": _immutable(prefix.raw_costs), "rollout": result,
            "ledger": deepcopy(result.ledger)}


def complete_candidate_scores(outcomes, expected_owner_hashes, contexts, *, gamma,
                              phase="candidate_complete", normalize=True):
    """Rank all candidate/context queries using only complete owned returns.

    Contexts contain id, remaining, role ('root' or 'guide'), ancestry, and
    snapshot_digest. Returns are normalized by their finite absorbing scale,
    averaged within each role, then averaged equally across present roles.
    Query references must be distinct across the entire candidate bank.
    """
    gamma, phase = _gamma(gamma), _text(phase, "phase")
    if type(normalize) is not bool:
        raise ValueError("normalize must be an explicit boolean")
    if not isinstance(outcomes, Sequence) or not outcomes or len(outcomes) != len(expected_owner_hashes):
        raise ValueError("Every candidate requires a full row of complete queries and an expected owner")
    if not contexts or not isinstance(contexts, Sequence):
        raise ValueError("At least one declared context is required")
    context_ids, roles, scales = [], [], []
    for context in contexts:
        context_ids.append(_text(context["id"], "context id"))
        if context["role"] not in ("root", "guide"):
            raise ValueError("Context role must be root or guide")
        roles.append(context["role"])
        remaining = _integer(context["remaining"], "context remaining", 1)
        scales.append(max(1., finite_absorbing_cost(gamma, remaining)) if normalize else 1.)
        _text(context["snapshot_digest"], "snapshot_digest")
        _json_identity(context["ancestry"])
    if len(set(context_ids)) != len(context_ids):
        raise ValueError("Context ids must be unique")
    returns = np.empty((len(outcomes), len(contexts)), dtype=np.float64)
    references = []
    seen = set()
    for candidate, (queries, expected_owner) in enumerate(zip(outcomes, expected_owner_hashes)):
        _text(expected_owner, "expected candidate owner hash")
        if len(queries) != len(contexts):
            raise ValueError("Every candidate must be scored on every declared context")
        row_refs = []
        for index, (query, context) in enumerate(zip(queries, contexts)):
            if (query.get("target_kind") != TARGET_KIND or query.get("continuation_owner") != "candidate"
                    or query.get("owner_hash") != expected_owner):
                raise ValueError("Return is not owned by the expected complete candidate actor")
            if (query.get("complete") is not True or query.get("cutoff") is not False
                    or query.get("phase") != phase):
                raise ValueError("Incomplete or wrong-phase query cannot provide candidate credit")
            if (query.get("context_id") != context["id"] or query.get("remaining") != context["remaining"]
                    or query.get("snapshot_digest") != context["snapshot_digest"]
                    or _json_identity(query.get("context_ancestry")) != _json_identity(context["ancestry"])
                    or query.get("gamma") != gamma or query.get("unhealthy_cost") != 5.):
                raise ValueError("Candidate return belongs to a different context or objective")
            reference = _text(query.get("query_ref"), "query_ref")
            if reference in seen:
                raise ValueError("Complete candidate queries cannot reuse a query reference")
            seen.add(reference)
            computed, _, _, _ = _finite_return(query["raw_costs"], remaining=context["remaining"],
                executed_steps=query["executed_steps"], terminated=query["terminated"],
                gamma=gamma, absorbing_cost=5.)
            if not np.isclose(query["cost"], computed, rtol=1e-11, atol=1e-9):
                raise ValueError("Candidate aggregate return disagrees with its complete raw trace")
            returns[candidate, index] = computed
            row_refs.append(reference)
        references.append(tuple(row_refs))
    normalized = returns / np.asarray(scales)[None]
    role_array = np.asarray(roles)
    role_scores = {role: normalized[:, role_array == role].mean(axis=1)
                   for role in sorted(set(roles))}
    scores = np.stack(list(role_scores.values())).mean(axis=0)
    return {"scores": _immutable(scores), "complete_returns": _immutable(returns),
            "normalized_returns": _immutable(normalized), "normalization_scales": _immutable(scales),
            "role_scores": {key: _immutable(value) for key, value in role_scores.items()},
            "query_refs": tuple(references), "owner_hashes": tuple(expected_owner_hashes),
            "context_ids": tuple(context_ids), "phase": phase,
            "target_kind": TARGET_KIND, "candidate_count": len(outcomes), "context_count": len(contexts)}


def choose_bounded_candidate(base_parameters, candidate_parameters, scores, *,
                             max_step_norm, proposal_kinds=None):
    """Choose the lowest complete score among all eligible bounded proposals.

    Samples and synthesized ES proposals pass the same L2 check. This function
    does not accept an actor update or certify heldout control performance.
    """
    base = np.asarray(base_parameters, dtype=np.float64)
    candidates = np.asarray(candidate_parameters, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if (base.ndim != 1 or candidates.ndim != 2 or candidates.shape[1:] != base.shape
            or scores.shape != (len(candidates),) or not len(candidates)
            or not np.isfinite(base).all() or not np.isfinite(candidates).all()
            or not np.isfinite(scores).all()):
        raise ValueError("Need aligned finite base, candidate parameters and complete scores")
    if not np.isfinite(max_step_norm) or max_step_norm <= 0.:
        raise ValueError("max_step_norm must be positive and finite")
    if proposal_kinds is None:
        proposal_kinds = tuple("candidate" for _ in candidates)
    if len(proposal_kinds) != len(candidates) or any(not isinstance(kind, str) or not kind for kind in proposal_kinds):
        raise ValueError("Every proposal requires a kind label")
    steps = candidates - base
    norms = np.linalg.norm(steps, axis=1)
    eligible = norms <= max_step_norm + 1e-12 * max(1., max_step_norm)
    selected = int(np.argmin(np.where(eligible, scores, np.inf))) if eligible.any() else None
    return {"selected_index": selected, "selected_kind": None if selected is None else proposal_kinds[selected],
            "parameters": None if selected is None else _immutable(candidates[selected]),
            "eligible": _immutable(eligible, dtype=bool), "step_norms": _immutable(norms),
            "max_step_norm": float(max_step_norm), "proposal_kinds": tuple(proposal_kinds),
            "score": None if selected is None else float(scores[selected]),
            "accepted": False, "requires_complete_heldout_acceptance": True}
