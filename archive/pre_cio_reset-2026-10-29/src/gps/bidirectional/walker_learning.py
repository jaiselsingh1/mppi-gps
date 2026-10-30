"""Critic-free Walker policy learning from complete actor-owned outcomes.

Antithetic parameter exploration estimates a Gaussian-smoothed return gradient.
This is a standard ES component, not a new estimator or a completed feedback
algorithm. The caller owns native MuJoCo execution, inference accounting,
qualification and search authority. No teacher action is a learning target.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .policy import FamilyGaussianActor, policy_hash

VERSION = "walker-complete-outcome-es-adam-v1"
LEARNING_DEFAULTS = dict(directions=4, sigma=.02, cost_scale=2800., beta=1.,
                         learning_rate=3e-4, gradient_clip=.5, seed=6100000,
                         frozen_parameter_names=["log_std"])


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name + " must be a finite scalar")
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0):
        raise ValueError(name + " must be finite" + (" and positive" if positive else ""))
    return value


def es_gradient(epsilon, plus_costs, minus_costs, *, sigma, cost_scale, beta=1., budgets=None):
    """Equal-context antithetic ES estimate; inputs are measured scalar returns.

    Ordinary contexts have budget=None. A supplied teacher budget adds only
    beta*(max(J-B,0)/cost_scale)^2. No state/action regression or critic enters.
    epsilon has shape(directions,parameters); costs have shape(directions,contexts).
    """
    e, plus, minus = [np.asarray(x, dtype=np.float64) for x in (epsilon, plus_costs, minus_costs)]
    sigma = _scalar(sigma, "sigma", positive=True)
    scale = _scalar(cost_scale, "cost_scale", positive=True)
    beta = _scalar(beta, "beta")
    if (beta < 0 or e.ndim != 2 or not all(e.shape) or plus.ndim != 2 or not all(plus.shape)
            or minus.shape != plus.shape or len(plus) != len(e)
            or not all(np.isfinite(x).all() for x in (e, plus, minus))):
        raise ValueError("Finite, aligned nonempty directions and complete-cost matrices required")
    budgets = [None] * plus.shape[1] if budgets is None else list(budgets)
    if len(budgets) != plus.shape[1]:
        raise ValueError("Teacher budgets must align with original contexts")
    objective_plus, objective_minus = plus / scale, minus / scale
    excess_plus, excess_minus = np.zeros_like(plus), np.zeros_like(minus)
    for i, budget in enumerate(budgets):
        if budget is not None:
            bound = _scalar(budget, "teacher budget")
            excess_plus[:, i] = np.maximum(plus[:, i] - bound, 0.)
            excess_minus[:, i] = np.maximum(minus[:, i] - bound, 0.)
            objective_plus[:, i] += beta * np.square(excess_plus[:, i] / scale)
            objective_minus[:, i] += beta * np.square(excess_minus[:, i] / scale)
    differences = objective_plus.mean(axis=1) - objective_minus.mean(axis=1)
    gradient = np.sum(differences[:, None] * e, axis=0) / (2 * len(e) * sigma)
    if not all(np.isfinite(x).all() for x in (objective_plus, objective_minus, gradient)):
        raise FloatingPointError("Nonfinite complete-return objective or ES gradient")
    return dict(gradient=gradient, plus_objectives=objective_plus, minus_objectives=objective_minus,
                differences=differences, plus_excess=excess_plus, minus_excess=excess_minus)


def snapshot_identity(snapshot):
    """Bind the full native snapshot, including task history and warmstart."""
    value = snapshot.to_dict() if hasattr(snapshot, "to_dict") else snapshot
    if not isinstance(value, dict):
        raise ValueError("An explicit full snapshot dictionary is required")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _contexts(contexts, budgets):
    contexts, budgets = list(contexts), list(budgets)
    if len(contexts) != 4 or len(budgets) != 4:
        raise ValueError("Exactly two ordinary and two actual MPPI-acquired contexts required")
    rows, ids = [], set()
    for context, budget in zip(contexts, budgets):
        if (not isinstance(context, dict) or "snapshot" not in context
                or not all(isinstance(context.get(k), str) and context[k] for k in ("context_id", "snapshot_id"))
                or context["context_id"] in ids):
            raise ValueError("Distinct context IDs and explicit native snapshot identities required")
        if snapshot_identity(context["snapshot"]) != context["snapshot_id"]:
            raise ValueError("Context identity differs from its actual full snapshot")
        ids.add(context["context_id"])
        if context.get("source") == "ordinary":
            if budget is not None:
                raise ValueError("Ordinary contexts must not have fabricated teacher budgets")
        elif context.get("source") == "mppi_acquired":
            if (not isinstance(budget, dict) or budget.get("snapshot_id") != context["snapshot_id"]
                    or not isinstance(budget.get("teacher_hash"), str) or not budget["teacher_hash"]
                    or not isinstance(budget.get("query_ids"), list) or not budget["query_ids"]
                    or not all(isinstance(q, str) and q for q in budget["query_ids"])
                    or len(set(budget["query_ids"])) != len(budget["query_ids"])):
                raise ValueError("An acquired budget must bind the same full snapshot and executed teacher")
            _scalar(budget.get("cost"), "measured teacher cost")
        else:
            raise ValueError("Only ordinary and actual MPPI-acquired source contexts are supported")
        rows.append(dict(context_id=context["context_id"], snapshot_id=context["snapshot_id"],
                         source=context["source"], teacher_budget=deepcopy(budget)))
    if sum(r["source"] == "ordinary" for r in rows) != 2:
        raise ValueError("Exactly two ordinary contexts required")
    return contexts, budgets, rows


def _owned_trace(trace, owner, context):
    if (not isinstance(trace, dict) or trace.get("complete") is not True
            or trace.get("actor_hash") != owner or trace.get("context_id") != context["context_id"]
            or trace.get("snapshot_id") != context["snapshot_id"]
            or not isinstance(trace.get("query_ids"), list) or not trace["query_ids"]
            or not all(isinstance(q, str) and q for q in trace["query_ids"])
            or len(set(trace["query_ids"])) != len(trace["query_ids"])):
        raise ValueError("Complete return must belong to this actor and exact native snapshot")
    return _scalar(trace.get("total_cost"), "complete actor-owned cost")


def _vector(actor):
    return torch.nn.utils.parameters_to_vector(actor.parameters()).detach().cpu().numpy().copy()


def _copy_actor(actor):
    return FamilyGaussianActor.from_checkpoint(actor.policy_state(), device=actor.action_scale.device)


def _checkpoint(archive, name, actor, state, **metadata):
    path = Path(archive.path) / ("walker_learning/" + name + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(dict(actor=actor.policy_state(), learner=deepcopy(state), **metadata), stream)
    return dict(file=str(path.relative_to(archive.path)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size)


def learn_update(executor, actor, optimizer_state, contexts, budgets, config, tag):
    """Pay eight perturbed actor versions, then propose one Adam-updated actor.

    executor.complete_rollout(actor,snapshot,tag) returns a dict containing
    total_cost,actor_hash,context_id,snapshot_id,complete=True,query_ids. Its
    raw_refs may bind additional traces. executor.archive supplies path,
    append and save_npz; the caller meters all native controls and inference.
    Returns(proposed_actor, full_private_learner_state, evidence). The proposal
    has no measured return or authority until the caller executes and qualifies it.
    """
    started = time.perf_counter()
    if config != LEARNING_DEFAULTS or not isinstance(tag, str) or not tag or Path(tag).name != tag:
        raise ValueError("Exact declared ES profile and a simple unique update tag required")
    if (not isinstance(actor, FamilyGaussianActor) or (actor.obs_dim, actor.hidden_dim, actor.num_families, actor.act_dim) != (23, 128, 1, 6)
            or not np.array_equal(actor.action_low.detach().cpu().numpy(), np.full(6, -1.))
            or not np.array_equal(actor.action_high.detach().cpu().numpy(), np.full(6, 1.))):
        raise ValueError("Walker requires the declared23/128/1/6 bounded action actor")
    contexts, budgets, rows = _contexts(contexts, budgets)
    archive, owner = executor.archive, policy_hash(actor)
    named = list(actor.named_parameters())
    names, shapes = [n for n, _ in named], [list(p.shape) for _, p in named]
    before = _vector(actor)
    mask = np.concatenate([np.full(p.numel(), n not in config["frozen_parameter_names"]) for n, p in named])
    if not all(n in names for n in config["frozen_parameter_names"]):
        raise ValueError("Frozen parameter name is absent")
    state = deepcopy(optimizer_state) if optimizer_state is not None else dict(
        schema=VERSION, actor_hash=owner, adam_state=None, rng_state=np.random.PCG64(config["seed"]).state,
        updates=0, parameter_names=names, parameter_shapes=shapes, config=deepcopy(config))
    if (state.get("schema") != VERSION or state.get("actor_hash") != owner or state.get("parameter_names") != names
            or state.get("parameter_shapes") != shapes or state.get("config") != config):
        raise ValueError("Optimizer/RNG checkpoint belongs to a different actor or learning profile")
    rng = np.random.Generator(np.random.PCG64(0))
    rng.bit_generator.state = deepcopy(state["rng_state"])
    rng_before = deepcopy(rng.bit_generator.state)
    epsilon = np.zeros((config["directions"], len(before)), dtype=np.float64)
    epsilon[:, mask] = rng.standard_normal((config["directions"], int(mask.sum())))
    rng_after = deepcopy(rng.bit_generator.state)
    packet = archive.save_npz("walker_learning/" + tag + "_directions.npz", before_parameters=before,
        epsilon=epsilon, trainable_mask=mask, contexts_json=np.array(json.dumps(rows, sort_keys=True)),
        rng_before_json=np.array(json.dumps(rng_before, sort_keys=True)), rng_after_json=np.array(json.dumps(rng_after, sort_keys=True)))
    base_cp = _checkpoint(archive, tag + "_before", actor, state, role="current_actor", no_action_labels=True)
    request = archive.append("walker_es_update_requested", tag=tag, owner=owner, raw=packet,
        checkpoint=base_cp, gaussian_variates=int(epsilon.size - len(epsilon) * (~mask).sum()),
        directions=config["directions"], contexts=4, teacher_action_labels=False, learned_critic=False)
    plus, minus, records, seen = [], [], [], set()
    completed_rollouts = 0
    try:
        for direction, noise in enumerate(epsilon):
            for sign, target in ((1, plus), (-1, minus)):
                label = f"{tag}_d{direction}_{'plus' if sign == 1 else 'minus'}"
                perturbed = _copy_actor(actor)
                assignment = archive.append("walker_parameter_assignment_requested", tag=label, parent=owner)
                nominal = before.astype(np.float64) + sign * config["sigma"] * noise
                after = nominal.astype(before.dtype)
                with torch.no_grad():
                    torch.nn.utils.vector_to_parameters(torch.as_tensor(after, device=actor.action_scale.device).clone(), perturbed.parameters())
                actual = _vector(perturbed)
                if not np.array_equal(actual, after) or not np.array_equal(actual[~mask], before[~mask]):
                    raise ValueError("Actual perturbation or frozen parameter coordinates differ")
                changed_owner = policy_hash(perturbed)
                displacement = archive.save_npz("walker_learning/" + label + "_parameters.npz",
                    before=before, after=actual, nominal_target=nominal, actual_delta=actual.astype(float) - before.astype(float))
                cp = _checkpoint(archive, label, perturbed, state, role="perturbed_actor_only", parent=owner)
                archive.append("walker_parameter_assignment_complete", request_id=assignment, tag=label,
                    owner=changed_owner, raw=displacement, checkpoint=cp)
                costs, traces = [], []
                for i, context in enumerate(contexts):
                    rollout_tag = label + "_c" + str(i)
                    rollout_request = archive.append("walker_owned_return_requested", tag=rollout_tag, owner=changed_owner,
                        context_id=context["context_id"], snapshot_id=context["snapshot_id"], checkpoint=cp)
                    trace = executor.complete_rollout(perturbed, context["snapshot"], rollout_tag)
                    cost = _owned_trace(trace, changed_owner, context)
                    if seen.intersection(trace["query_ids"]) or policy_hash(perturbed) != changed_owner:
                        raise ValueError("Reused return queries or changed generating actor")
                    seen.update(trace["query_ids"])
                    completed_rollouts += 1
                    archive.append("walker_owned_return_complete", request_id=rollout_request, tag=rollout_tag,
                        owner=changed_owner, context_id=context["context_id"], snapshot_id=context["snapshot_id"], trace=trace)
                    costs.append(cost); traces.append(trace)
                target.append(costs)
                records.append(dict(direction=direction, sign=sign, actor_hash=changed_owner, checkpoint=cp,
                                    displacement=displacement, traces=traces))
        values = es_gradient(epsilon, plus, minus, sigma=config["sigma"], cost_scale=config["cost_scale"],
            beta=config["beta"], budgets=[None if b is None else b["cost"] for b in budgets])
        if np.any(values["gradient"][~mask] != 0):
            raise ValueError("Unused deterministic variance acquired a gradient")
        proposed = _copy_actor(actor)
        optimizer = torch.optim.Adam(proposed.parameters(), lr=config["learning_rate"], betas=(.9, .999), eps=1e-8)
        if state["adam_state"] is not None:
            optimizer.load_state_dict(deepcopy(state["adam_state"]))
        offset = 0
        for _, parameter in proposed.named_parameters():
            part = values["gradient"][offset:offset + parameter.numel()]
            parameter.grad = torch.as_tensor(part, dtype=parameter.dtype, device=parameter.device).reshape(parameter.shape).clone()
            offset += parameter.numel()
        assigned = torch.nn.utils.parameters_to_vector([p.grad for p in proposed.parameters()]).detach().cpu().numpy().copy()
        norm = torch.nn.utils.clip_grad_norm_(proposed.parameters(), config["gradient_clip"], error_if_nonfinite=True)
        clipped = torch.nn.utils.parameters_to_vector([p.grad for p in proposed.parameters()]).detach().cpu().numpy().copy()
        estimate = archive.save_npz("walker_learning/" + tag + "_estimate.npz", **values,
            plus_costs=np.asarray(plus), minus_costs=np.asarray(minus), assigned_gradient=assigned, clipped_gradient=clipped)
        opt_request = archive.append("optimizer_requested", tag=tag, optimizer="Adam", owner=owner,
            raw=estimate, gradient_norm_before_clip=float(norm), backward_calls=0)
        opt_start = time.perf_counter()
        try:
            optimizer.step()
        except BaseException as error:
            archive.append("optimizer_error", request_id=opt_request, tag=tag, error=str(error),
                optimizer_wall_seconds=time.perf_counter() - opt_start, partial_work_unknown=True)
            raise
        archive.append("optimizer_complete", request_id=opt_request, tag=tag,
            optimizer_wall_seconds=time.perf_counter() - opt_start, backward_calls=0)
        changed = _vector(proposed)
        if not np.isfinite(changed).all() or not np.array_equal(changed[~mask], before[~mask]) or policy_hash(actor) != owner:
            raise ValueError("Nonfinite proposal, changed frozen variance or mutated current actor")
        new_state = dict(schema=VERSION, actor_hash=policy_hash(proposed), adam_state=deepcopy(optimizer.state_dict()),
            rng_state=rng_after, updates=state["updates"] + 1, parameter_names=names, parameter_shapes=shapes, config=deepcopy(config))
        delta = archive.save_npz("walker_learning/" + tag + "_updated_parameters.npz", before=before, after=changed,
            actual_delta=changed.astype(float) - before.astype(float))
        cp = _checkpoint(archive, tag + "_proposed", proposed, new_state, role="unqualified_proposal",
                         own_return_measured=False, authority=False)
        evidence = dict(schema=VERSION, tag=tag, owner_before=owner, owner_after=policy_hash(proposed),
            before_checkpoint=base_cp, proposed_checkpoint=cp, directions=packet, estimate=estimate,
            displacement=delta, perturbed=records, complete_owned_rollouts=32, optimizer_steps=1,
            backward_calls=0, teacher_action_labels=False, learned_critic=False, deployment_qualified=False,
            updated_actor_return_measured=False, zero_estimated_gradient=bool(np.all(values["gradient"] == 0)))
        archive.append("walker_es_update_complete", request_id=request, **evidence,
                       update_wall_seconds=time.perf_counter() - started)
        return proposed, new_state, evidence
    except BaseException as error:
        archive.append("walker_es_update_error", request_id=request, tag=tag, owner=owner,
            completed_owned_rollouts=completed_rollouts, error_type=type(error).__name__,
            error=str(error), update_wall_seconds=time.perf_counter() - started)
        raise
