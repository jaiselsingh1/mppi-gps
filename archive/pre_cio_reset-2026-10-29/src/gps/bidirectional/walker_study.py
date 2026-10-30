"""One outcome-trained MuJoCo Walker workflow; no imitation or learned critic.

The physical task, independent search and policy learning are separate owners.
Only actual complete executions can qualify learning or guidance. Public source
freezing and child dispatch belong to the existing training CLI.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .evidence import Archive, file_sha256, verify_experiment
from .search import IndependentSearch, SearchConfig, TrajectoryMemory, state_json
from .trust import TrustManager, CandidateTrustManager, Outcome, LADDER

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "bidirectional-walker-return-v1"
SCALE = [1.] * 8 + [5.] * 9 + [1.5, 1., 1., 5., 2., 1.]


def default_config():
    from .walker import WalkerConfig
    task = asdict(WalkerConfig(nthread=4))
    return dict(schema=SCHEMA, task=task,
        search=dict(K=256, H=48, temperature=3., sigma=.3, smooth_candidates=32,
                    memory_candidates=16, knot_spacing=10, initial_iterations=2),
        native_noise=dict(noise_lowpass_cutoff_hz=23.4375, noise_lowpass_sample_rate_hz=125., noise_lowpass_order=2),
        memory_scale=SCALE.copy(), actor_device="cpu", actor_seed=6000000,
        learning=dict(directions=4, sigma=.02, cost_scale=2800., beta=1., learning_rate=3e-4,
                      gradient_clip=.5, seed=6100000, frozen_parameter_names=["log_std"]),
        updates=6, feedback_rounds=2, acquisition_fractions=[.5, .8],
        roots=dict(discovery=720000000, planner=730000000, training=740000000,
                   qualification=750000000, held=760000000, trial=770000000, confirmation=780000000),
        gates=dict(training_gain=1e-8, child_gain=.001, controller_gain=.001,
                   capture_floor=.9, support_radius=1.),
        validation=dict(root=710000000, planner_seed=710100000, tape_seed=710200000,
                        parity_atol=1e-9, actor_probe=True),
        budgets=dict(requested_controls=350000000, neural_rows=250000000,
                     validation_controls=876, validation_neural_rows=600),
        qualification_episodes=8, held_episodes=16,
        paid_full_protocol_repeat=False, novelty_claim=False)


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(state_json(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _identity(value):
    return hashlib.sha256(json.dumps(state_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _prepare(out, config):
    out = Path(out).resolve()
    if config.get("schema") != SCHEMA or len(config["memory_scale"]) != 23:
        raise ValueError("Declared real-Walker workflow and23-dimensional support scales required")
    verify_experiment(ROOT, out)
    if json.loads((out / "config.json").read_text()) != state_json(config):
        raise ValueError("Caller differs from frozen Walker configuration")
    return out


def _checkpoint(archive, tag, actor, learner_state=None):
    import torch
    path = archive.path / ("checkpoints/" + tag + ".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(dict(actor=actor.policy_state(), learner=deepcopy(learner_state)), stream)
    return dict(file=str(path.relative_to(archive.path)), sha256=file_sha256(path), bytes=path.stat().st_size)


def _new_actor(config):
    from .policy import FamilyGaussianActor
    return FamilyGaussianActor(obs_dim=23, hidden_dim=128, num_families=1,
        action_low=(-1.,) * 6, action_high=(1.,) * 6, seed=config["actor_seed"]).to(config["actor_device"])


class WalkerIndependentSearch(IndependentSearch):
    """Native Butterworth Gaussian exploration plus actor-free smooth/memory channels."""
    def __init__(self, executor, config, native_noise, seed, memory):
        from src.mppi.mppi import MPPI
        from src.utils.config import MPPIConfig
        super().__init__(executor, SearchConfig(**config), seed, memory, action_dim=6)
        self.native = MPPI(executor.task.env, MPPIConfig(K=self.config.K, H=self.config.H,
            lam=self.config.temperature, noise_sigma=self.config.sigma,
            initial_iterations=self.config.initial_iterations, **native_noise), seed=seed)

    def _sample(self, snapshot, h):
        cfg = self.config
        self.native._rng = self.rng
        # Always draw the original configured H; do not shorten the RNG stream.
        noise = self.native._sample_noise()[:, :h]
        gaussian = np.clip(self.nominal[:h][None] + noise, -1., 1.)
        gaussian[0] = self.nominal[:h]  # Explicit independent nominal reserve.
        parts, names = [gaussian], ["native_butterworth_gaussian"] * cfg.K
        if cfg.smooth_candidates:
            knots = np.arange(0, h + cfg.knot_spacing, cfg.knot_spacing)
            values = self.rng.uniform(-1., 1., (cfg.smooth_candidates, len(knots), 6))
            smooth = np.array([np.column_stack([np.interp(np.arange(h), knots, row[:, j])
                                                for j in range(6)]) for row in values])
            parts.append(smooth); names += ["independent_knots"] * len(smooth)
        nearest = self.memory.nearest(self.executor.task.observation(snapshot),
            self.executor.task.config.T - snapshot.elapsed, cfg.memory_candidates)
        rows = []
        for i in range(cfg.memory_candidates):
            if nearest:
                fragment = nearest[i % len(nearest)]
                rows.append(np.pad(fragment.actions[:h], ((0, max(0, h-len(fragment.actions))), (0, 0)), mode="edge")[:h])
                names.append("memory:" + fragment.episode_id)
            else:
                rows.append(np.clip(self.nominal[:h] + self.rng.normal(0., cfg.sigma, (h, 6)), -1., 1.))
                names.append("independent_memory_fallback")
        if rows:
            parts.append(np.asarray(rows))
        return np.concatenate(parts), tuple(names)


def _reserve_neural(archive, rows, tag):
    used = getattr(archive, "_walker_neural_reserved", 0)
    if used + rows > archive.max_neural_rows:
        archive.append("neural_rejected", rows=rows, purpose=tag, before_forward=True)
        raise RuntimeError("Frozen Walker inference reservation exhausted")
    archive._walker_neural_reserved = used + rows
    return archive.append("neural_requested", rows=rows, purpose=tag, reservation=True)


def _mean(actor, observation):
    import torch
    from .policy import actor_forward
    with torch.no_grad():
        mean, _ = actor_forward(actor, np.asarray(observation)[None], 0,
                                inference_contract="native_batch1_rows")
        return actor._bounded(mean)[0].detach().cpu().numpy().copy()


def _actor_rollout(executor, actor, snapshot, tag):
    from .policy import policy_hash
    from .walker_learning import snapshot_identity
    archive, count = executor.archive, [0]
    owner, sid = policy_hash(actor), snapshot_identity(snapshot)
    references = getattr(archive, "_walker_actor_references", {})
    if owner not in references:
        references[owner] = _checkpoint(archive, "actor_" + owner, actor)
        archive._walker_actor_references = references
    request = _reserve_neural(archive, executor.task.config.T-snapshot.elapsed, tag)
    pending = [False]
    def action(obs):
        pending[0] = True
        result = _mean(actor, obs)
        pending[0] = False; count[0] += 1
        return result
    try:
        episode = executor.execute_policy(action, snapshot, kind=tag,
            metadata=dict(owner=dict(actor_hash=owner, deterministic=True, family=0,
                                     inference_contract="native_batch1_rows", checkpoint=references[owner]), snapshot_id=sid))
    finally:
        archive.append("neural_complete", request_id=request, rows=count[0], returned=not pending[0],
            known_completed_rows=count[0], unknown_rows=int(pending[0]),
            unused_reserved_rows=executor.task.config.T-snapshot.elapsed-count[0], purpose=tag)
    return dict(total_cost=episode.cost, actor_hash=owner, context_id=sid, snapshot_id=sid,
                complete=episode.complete, capture=episode.captured, query_ids=[episode.query_id])


def _reset(task, archive, seed, tag):
    archive.append("walker_reset_requested", seed=seed, context=tag, physics_frames=0)
    snapshot = task.reset(seed=seed, context_id="root:" + str(seed))
    raw = archive.save_npz("resets/" + tag + ".npz", snapshot_json=np.array(json.dumps(snapshot.to_dict(), sort_keys=True)))
    archive.append("walker_reset_complete", seed=seed, raw=raw, physics_frames=0)
    return snapshot


def _select(executor, planner, bank, actor=None, trust=None, trial_weight=None, tag="selection"):
    from .policy import policy_hash
    before, digest = _identity(planner.snapshot()), bank.digest
    selected, score, influenced = bank.independent_plan, bank.independent_score, False
    if actor is not None and trust is not None:
        owner = policy_hash(actor); task = executor.task
        current = task.observation(bank.initial); remaining = task.config.T-bank.initial.elapsed
        weight = trust.weight_for(owner, task.task_hash, 0, bank.initial.elapsed, remaining, current, trial_weight)
        if weight > 0:
            actions = np.concatenate((bank.actions, bank.independent_plan[None]))
            observations = np.concatenate((bank.observations, bank.independent_observations[None]))
            raw_scores = np.r_[bank.scores, bank.independent_score]
            weights = trust.weights_for_batch(owner, task.task_hash, 0, observations, remaining, trial_weight)
            active = weights > 0; penalties = np.zeros(weights.shape)
            request = _reserve_neural(executor.archive, int(active.sum()), tag)
            predictions = []; completed = 0
            try:
                for observation in observations[active]:
                    predictions.append(_mean(actor, observation)); completed += 1
            finally:
                executor.archive.append("neural_complete", request_id=request, rows=completed,
                    returned=completed == int(active.sum()), purpose=tag,
                    unknown_rows=int(completed != int(active.sum())))
            if predictions:
                predictions = np.asarray(predictions)
                penalties[active] = weights[active]*np.square(actions[active]-predictions).sum(axis=1)
            augmented = raw_scores+penalties.sum(axis=1)
            index = int(np.argmin(augmented)); selected, score, influenced = actions[index], raw_scores[index], True
            raw = executor.archive.save_npz("agreement/" + tag + ".npz", weights=weights,
                active_predictions=np.asarray(predictions), raw_scores=raw_scores, augmented_scores=augmented,
                selected_index=np.array(index), bank_query_ids=np.asarray(bank.query_ids))
            executor.archive.append("walker_agreement_selection", raw=raw, actor_hash=owner,
                trial_weight=trial_weight, bank_digest=digest, copied_bank_only=True)
    if before != _identity(planner.snapshot()) or digest != bank.digest:
        raise ValueError("Learned selection mutated the independent bank/private planner")
    return selected, float(score), influenced, not np.array_equal(selected[0], bank.independent_plan[0])


def validate(out, config):
    from .walker import WalkerTask, WalkerExecutor
    started = time.perf_counter(); out = _prepare(out, config)
    task = None; error = None; checks = {}; work = {}
    with Archive(out / "validation") as archive:
        archive.max_neural_rows = config["budgets"]["validation_neural_rows"]
        try:
            task = WalkerTask(config["task"])
            executor = WalkerExecutor(task, archive, max_requested_controls=config["budgets"]["validation_controls"])
            initial = _reset(task, archive, config["validation"]["root"], "validation_initial")
            rng = np.random.default_rng(config["validation"]["tape_seed"])
            actions = rng.uniform(-.1, .1, (4, 12, 6))
            model = executor.model_rollout(initial, actions, kind="validation_fixed_bank")
            task.restore(initial); states = []; costs = []
            for action in actions[0]:
                step = executor.step(action, metadata={"purpose": "validation_first_row_live"})
                states.append(step.snapshot.state); costs.append(step.total_cost)
            tolerance = config["validation"]["parity_atol"]
            checks["model_live_states"] = bool(np.allclose(model.states[0], states, rtol=0., atol=tolerance))
            checks["model_live_cost"] = bool(np.isclose(model.actual_partial_costs[0], sum(costs), rtol=0., atol=tolerance))
            physical = task.snapshot(); task.restore(physical)
            checks["full_snapshot_restoration"] = physical.to_dict() == task.snapshot().to_dict()
            search = dict(config["search"], K=4, H=12, smooth_candidates=2, memory_candidates=2, initial_iterations=1)
            memory = TrajectoryMemory(config["memory_scale"], task.config.T)
            planner = WalkerIndependentSearch(executor, search, config["native_noise"], config["validation"]["planner_seed"], memory)
            private = planner.snapshot(); first = planner.propose(physical)
            after = _identity(planner.snapshot())
            actor = _new_actor(config)
            selected = _select(executor, planner, first, actor, TrustManager(), tag="validation_absent_authority")
            checks["absent_authority_fallback"] = bool(np.array_equal(selected[0], first.independent_plan) and not selected[2])
            checks["absent_authority_zero_inference"] = getattr(archive, "_walker_neural_reserved", 0) == 0
            task.restore(physical); planner.restore(private); repeated = planner.propose(physical)
            checks["independent_bank_replay"] = first.digest == repeated.digest
            checks["private_planner_replay"] = after == _identity(planner.snapshot())
            archive.save_npz("validation_proof.npz", model_states=model.states[0], live_states=np.asarray(states),
                first_actions=first.actions, repeated_actions=repeated.actions,
                first_scores=first.scores, repeated_scores=repeated.scores,
                physical_snapshot_json=np.array(json.dumps(physical.to_dict(), sort_keys=True)),
                planner_before_json=np.array(json.dumps(state_json(private), sort_keys=True)))
            if not all(checks.values()):
                raise ValueError("Walker physical/independence validation failed")
            if config["validation"]["actor_probe"]:
                trace = _actor_rollout(executor, actor, initial, "validation_actor_complete")
                checks["actor_owned_complete"] = trace["complete"]
            work = executor.totals()
            checks["fixed_reservation_roster"] = work["requested_controls"] == (876 if config["validation"]["actor_probe"] else 276)
            checks["fixed_query_roster"] = work["queries"] == (18 if config["validation"]["actor_probe"] else 17)
            if not all(checks.values()):
                raise ValueError("Walker validation roster or owned episode failed")
            verify_experiment(ROOT, out)
        except BaseException as failure:
            error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
            if task is not None:
                work = executor.totals()
            archive.append("walker_validation_error", error=error)
        finally:
            if task is not None:
                task.close()
    result = dict(schema=SCHEMA, stage="validate", passed=error is None and all(checks.values()),
        checks=checks, work=work, error=error, actor_training=False, guidance_promoted=False,
        paid_full_protocol_repeat=False, wall_seconds=time.perf_counter()-started)
    _write(out / "result.json", result)
    return result


def _controller_id(config, memory, actor_hash=None, weight=None, trust=None):
    support = []
    if trust is not None:
        for record in trust.records:
            active_weight = (float(weight) if weight is not None and record.state == "eligible"
                             else record.lambda_weight if record.state == "promoted" else 0.)
            support.append(dict(actor_hash=record.actor_hash, task_hash=record.task_hash,
                family=record.family_id, time_interval=record.time_interval,
                remaining_interval=record.remaining_interval, anchors=record.support_anchors,
                scale=record.observation_scale, radius=record.support_radius, weight=active_weight))
    return _identity(dict(schema=SCHEMA, search=config["search"], native_noise=config["native_noise"],
        memory=memory.fragments, memory_scale=config["memory_scale"], actor_hash=actor_hash, weight=weight,
        support=support, inference_contract="native_batch1_rows", execution="complete_executed_replanning"))


def _mpc_episode(executor, config, memory, seed, tag, actor=None, trust=None, weight=None):
    from .policy import policy_hash
    task, archive = executor.task, executor.archive
    initial = _reset(task, archive, seed, tag + "_reset")
    planner = WalkerIndependentSearch(executor, config["search"], config["native_noise"],
        config["roots"]["planner"] + seed, deepcopy(memory))
    owner = None if actor is None else policy_hash(actor)
    controller = _controller_id(config, memory, owner, weight, trust)
    start_raw = archive.save_npz("planner/" + tag + "_start.npz",
        state_json=np.array(json.dumps(state_json(planner.snapshot()), sort_keys=True)))
    snapshots, actions, costs, live_queries, bank_queries = [initial], [], [], [], []
    changed = influenced = 0; first_digest = None
    while task.remaining:
        snapshot = task.snapshot(); bank = planner.propose(snapshot)
        if first_digest is None:
            first_digest = bank.digest
        decision = tag + "_t" + str(snapshot.elapsed)
        private_after = _identity(planner.snapshot())
        try:
            selected, score, active, intervention = _select(executor, planner, bank, actor, trust, weight, decision)
        except (ValueError, RuntimeError, FloatingPointError) as failure:
            if actor is None or trust is None:
                raise
            trust.revoke_actor(owner, "selection_error:" + type(failure).__name__)
            archive.append("walker_guidance_revoked", actor_hash=owner, reason=str(failure),
                immediate_independent_fallback=True)
            if private_after != _identity(planner.snapshot()):
                raise ValueError("Failed selection changed independent planner") from failure
            selected, score, active, intervention = bank.independent_plan, bank.independent_score, False, False
        raw = archive.save_npz("planner/" + decision + ".npz", independent_plan=bank.independent_plan,
            selected_plan=selected, independent_score=np.array(bank.independent_score), selected_raw_score=np.array(score),
            nominal_before=bank.proposal_state_before["nominal"], nominal_after=bank.proposal_state_after["nominal"],
            initial_plan_before=np.array(bank.proposal_state_before["initial"]),
            initial_plan_after=np.array(bank.proposal_state_after["initial"]),
            rng_before_json=np.array(json.dumps(state_json(bank.proposal_state_before["rng"]), sort_keys=True)),
            rng_after_json=np.array(json.dumps(state_json(bank.proposal_state_after["rng"]), sort_keys=True)),
            bank_query_ids=np.asarray(bank.query_ids), bank_digest=np.array(bank.digest),
            initial_snapshot_json=np.array(json.dumps(snapshot.to_dict(), sort_keys=True)))
        archive.append("walker_executed_decision", raw=raw, controller_hash=controller, actor_hash=owner,
            influenced=active, changed=intervention, planner_initial=start_raw,
            independent_private_state_unchanged_by_selection=True)
        step = executor.step(np.asarray(selected[0]), metadata=dict(owner=dict(controller_hash=controller,
            actor_hash=owner, semantics="executed_replanning"), decision=raw))
        live_queries.append(executor.latest_query_id); bank_queries.extend(bank.query_ids)
        snapshots.append(step.snapshot); actions.append(step.action); costs.append(step.total_cost)
        changed += int(intervention); influenced += int(active)
    raw = archive.save_npz("episodes/" + tag + ".npz", states=np.stack([s.state for s in snapshots]),
        snapshots_json=np.array(json.dumps([s.to_dict() for s in snapshots], sort_keys=True)),
        actions=np.asarray(actions), costs=np.asarray(costs), live_query_ids=np.asarray(live_queries),
        model_query_ids=np.asarray(bank_queries), controller_hash=np.array(controller),
        actor_hash=np.array(owner or ""), captured=np.array(step.captured))
    archive.append("walker_controller_episode", raw=raw, controller_hash=controller, actor_hash=owner,
        total_cost=float(sum(costs)), captured=bool(step.captured), complete=True,
        changed_decisions=changed, influenced_decisions=influenced, query_ids=live_queries)
    return dict(tag=tag, initial=initial, snapshots=snapshots, actions=np.asarray(actions), costs=np.asarray(costs),
        cost=float(sum(costs)), capture=bool(step.captured), complete=True, controller_hash=controller,
        actor_hash=owner, query_ids=live_queries, raw=raw, changed_decisions=changed,
        influenced_decisions=influenced, first_bank_digest=first_digest)


def _episode_summary(episode):
    return {k: v for k, v in episode.items() if k not in ("snapshots", "actions", "costs", "initial")}


def _ingest(memory, episodes, executor, tag):
    before = len(memory.fragments)
    for episode in episodes:
        if episode["actor_hash"] is not None:
            raise ValueError("Only actually executed independent RAW trajectories enter private memory")
        observations = np.stack([executor.task.observation(s) for s in episode["snapshots"][:-1]])
        memory.add(observations, episode["actions"], executor.task.config.T,
                   episode["tag"], family=0, stride=20)
    raw = executor.archive.save_npz("memory/" + tag + ".npz",
        memory_json=np.array(json.dumps(state_json(memory.fragments), sort_keys=True)))
    executor.archive.append("walker_raw_memory_ingestion", raw=raw, before=before,
        after=len(memory.fragments), episodes=[e["raw"] for e in episodes], guided_ingested=False)
    return dict(before=before, after=len(memory.fragments), raw=raw)


def _acquired(episode, fraction, task):
    from .walker_learning import snapshot_identity
    length = len(episode["actions"])
    if length < 1:
        raise ValueError("No actual executed source for an acquired context")
    index = min(length-1, int(np.floor(fraction*length)))
    snapshot = episode["snapshots"][index]
    if snapshot.terminated or not bool(task.env._is_healthy(snapshot.state[2], snapshot.state[3])):
        raise ValueError("The predeclared acquired point is not an actual healthy pre-action state")
    sid = snapshot_identity(snapshot)
    context = dict(snapshot=snapshot, context_id=sid, snapshot_id=sid, source="mppi_acquired")
    budget = dict(cost=float(episode["costs"][index:].sum()), snapshot_id=sid,
        teacher_hash=episode["controller_hash"], query_ids=episode["query_ids"][index:],
        source_episode=episode["raw"], elapsed=snapshot.elapsed, source_fraction=fraction)
    return context, budget


def _contexts_for_update(executor, config, episodes, stage, update, tag):
    from .walker_learning import snapshot_identity
    contexts, budgets = [], []
    for i in range(2):
        seed = config["roots"]["training"] + stage*10000 + update*10 + i
        snapshot = _reset(executor.task, executor.archive, seed, tag + "_ordinary" + str(i))
        sid = snapshot_identity(snapshot)
        contexts.append(dict(snapshot=snapshot, context_id=sid, snapshot_id=sid, source="ordinary")); budgets.append(None)
    fraction = config["acquisition_fractions"][update % 2]
    for episode in episodes:
        context, budget = _acquired(episode, fraction, executor.task)
        contexts.append(context); budgets.append(budget)
    return contexts, budgets


def _objective(traces, budgets, config):
    costs = np.array([r["total_cost"] for r in traces]); scale = config["learning"]["cost_scale"]
    values = costs/scale
    for i, budget in enumerate(budgets):
        if budget is not None:
            values[i] += config["learning"]["beta"]*(max(costs[i]-budget["cost"], 0.)/scale)**2
    return float(values.mean())


def _train_child(executor, config, incoming, incoming_state, episodes, stage, arm):
    import torch
    from types import SimpleNamespace
    from .policy import FamilyGaussianActor, policy_hash
    from .walker_learning import learn_update
    actor = FamilyGaussianActor.from_checkpoint(incoming.policy_state(), device=config["actor_device"])
    state = deepcopy(incoming_state); history = []
    wrapper = SimpleNamespace(archive=executor.archive,
        complete_rollout=lambda a, s, tag: _actor_rollout(executor, a, s, tag))
    for update in range(config["updates"]):
        tag = f"stage{stage}_{arm}_update{update}"
        contexts, budgets = _contexts_for_update(executor, config, episodes, stage, update, tag)
        proposed, proposed_state, evidence = learn_update(wrapper, actor, state, contexts, budgets, config["learning"], tag)
        base = [_actor_rollout(executor, actor, c["snapshot"], tag + "_base" + str(i)) for i, c in enumerate(contexts)]
        candidate = [_actor_rollout(executor, proposed, c["snapshot"], tag + "_candidate" + str(i)) for i, c in enumerate(contexts)]
        first, second = _objective(base, budgets, config), _objective(candidate, budgets, config)
        captures = [sum(r["capture"] for r in panel) for panel in (base, candidate)]
        accepted = second < first-config["gates"]["training_gain"] and captures[1] >= captures[0]
        before_owner, candidate_owner = policy_hash(actor), policy_hash(proposed)
        if accepted:
            actor, state = proposed, proposed_state
        else:
            # Restore Adam/actor exactly; consume the tested ES directions once.
            state = torch.load(executor.archive.path/evidence["before_checkpoint"]["file"],
                               map_location="cpu", weights_only=True)["learner"]
            state["rng_state"] = deepcopy(proposed_state["rng_state"])
        record = dict(stage=stage, arm=arm, update=update, accepted=bool(accepted),
            owner_before=before_owner, candidate_owner=candidate_owner, retained_owner=policy_hash(actor),
            before_objective=first, candidate_objective=second, captures_before=captures[0], captures_candidate=captures[1],
            before_cost=float(np.mean([r["total_cost"] for r in base])),
            candidate_cost=float(np.mean([r["total_cost"] for r in candidate])),
            before=base, candidate=candidate, evidence=evidence, changing_ordinary_roster=True,
            rejected_adam_rolled_back=not accepted, tested_es_rng_advanced=True)
        executor.archive.append("walker_learning_acceptance", **record); history.append(record)
    cp = _checkpoint(executor.archive, f"stage{stage}_{arm}_frontier", actor, state)
    return actor, state, dict(arm=arm, actor_hash=policy_hash(actor), checkpoint=cp,
                             accepted=sum(r["accepted"] for r in history), updates=history)


def _actor_panel(executor, config, actors, root, count, tag):
    panels = {}
    for role, actor in actors.items():
        rows = []
        for i in range(count):
            snapshot = _reset(executor.task, executor.archive, root+i, tag + "_" + role + "_" + str(i))
            rows.append(_actor_rollout(executor, actor, snapshot, tag + "_" + role + "_episode" + str(i)))
        panels[role] = dict(actor_hash=rows[0]["actor_hash"], mean_cost=float(np.mean([r["total_cost"] for r in rows])),
            captures=sum(r["capture"] for r in rows), episodes=count, rows=rows)
    return panels


def _stage(executor, config, incoming, incoming_state, data, stage):
    from .policy import policy_hash
    children, states, records = {}, {}, {}
    for arm, episodes in data.items():
        children[arm], states[arm], records[arm] = _train_child(executor, config, incoming, incoming_state, episodes, stage, arm)
    actors = dict(parent=incoming, **children)
    panels = _actor_panel(executor, config, actors, config["roots"]["qualification"]+stage*1000,
                          config["qualification_episodes"], f"stage{stage}_qualification")
    qualified = {}
    for arm in children:
        child, parent = panels[arm], panels["parent"]
        qualified[arm] = bool(child["actor_hash"] != parent["actor_hash"]
            and parent["mean_cost"]-child["mean_cost"] > config["gates"]["child_gain"]
            and child["captures"] >= parent["captures"]
            and child["captures"]/child["episodes"] >= config["gates"]["capture_floor"])
    choices = [arm for arm in children if qualified[arm]]
    selected = min(choices, key=lambda arm: panels[arm]["mean_cost"]) if choices else "parent"
    held = _actor_panel(executor, config, actors, config["roots"]["held"]+stage*1000,
                       config["held_episodes"], f"stage{stage}_held")
    guided_advantage = bool("guided" in panels and qualified.get("guided", False)
        and panels["raw"]["mean_cost"]-panels["guided"]["mean_cost"] > config["gates"]["child_gain"]
        and panels["guided"]["captures"] >= panels["raw"]["captures"])
    result = dict(stage=stage, incoming_actor=policy_hash(incoming), learning=records,
        qualification=panels, qualified=qualified, selected=selected, held=held,
        held_did_not_select=True, guided_next_policy_advantage=guided_advantage,
        retained_actor=panels[selected]["actor_hash"])
    executor.archive.append("walker_stage_result", result=result)
    return (incoming, incoming_state, result) if selected == "parent" else (children[selected], states[selected], result)


def _suffix_eligibility(executor, config, actor, episodes, tag):
    from .policy import policy_hash
    manager = TrustManager(); anchors = []
    for episode in episodes:
        for fraction in config["acquisition_fractions"]:
            anchors.append(_acquired(episode, fraction, executor.task)[0])
    owner = policy_hash(actor); task = executor.task
    actor_controller = _identity(dict(actor_hash=owner, task_hash=task.task_hash,
        deterministic=True, family=0, inference_contract="native_batch1_rows"))
    clocks = [c["snapshot"].elapsed for c in anchors]
    record = manager.register(actor_hash=owner, actor_controller_hash=actor_controller,
        task_hash=task.task_hash, family_id=0, total_steps=task.config.T,
        time_interval=(min(clocks), max(clocks)), remaining_interval=(task.config.T-max(clocks), task.config.T-min(clocks)),
        support_anchors=np.stack([task.observation(c["snapshot"]) for c in anchors]),
        observation_scale=config["memory_scale"], support_radius=config["gates"]["support_radius"],
        anchor_context_ids=[c["context_id"] for c in anchors], deterministic=True)
    outcomes = []
    for i, context in enumerate(anchors):
        trace = _actor_rollout(executor, actor, context["snapshot"], tag + "_suffix" + str(i))
        outcomes.append(Outcome(context["context_id"], trace["total_cost"], trace["capture"], trace["complete"],
            actor_controller, owner, 0, task.config.T-context["snapshot"].elapsed, task.task_hash,
            tuple(trace["query_ids"]), "actor_suffix"))
    record = manager.qualify(record.record_id, outcomes)
    result = dict(record=state_json(asdict(record)), empirical_unperturbed_support_only=True,
        physical_perturbation_coverage_tested=False, outcomes=state_json(outcomes))
    executor.archive.append("walker_suffix_eligibility", **result)
    return manager, result


def _pairs(executor, config, memory, actor, manager, weight, roots, tag):
    pairs = []
    for i, seed in enumerate(roots):
        raw = _mpc_episode(executor, config, memory, seed, tag + "_raw" + str(i))
        guided = _mpc_episode(executor, config, memory, seed, tag + "_guided" + str(i), actor, manager, weight)
        if raw["initial"].to_dict() != guided["initial"].to_dict() or raw["first_bank_digest"] != guided["first_bank_digest"]:
            raise ValueError("Paired controllers did not share the physical/private initial conditions")
        pairs.append(dict(raw=raw, guided=guided))
    return pairs


def _pair_gate(pairs, minimum_gain):
    gains = [p["raw"]["cost"]-p["guided"]["cost"] for p in pairs]
    captures = [sum(p[role]["capture"] for p in pairs) for role in ("raw", "guided")]
    changed = sum(p["guided"]["changed_decisions"] for p in pairs)
    return dict(passed=bool(float(np.mean(gains)) > minimum_gain and captures[1] >= captures[0] and changed > 0),
        mean_gain=float(np.mean(gains)), raw_captures=captures[0], guided_captures=captures[1],
        changed_decisions=changed, pair_count=len(pairs))


def _outcome(episode, task):
    return Outcome(episode["initial"].context_id, episode["cost"], episode["capture"], episode["complete"],
        episode["controller_hash"], episode["actor_hash"], 0, task.config.T, task.task_hash,
        tuple(episode["query_ids"]), "executed_replanning")


def _guidance(executor, config, memory, actor, sources, round_index, old=None):
    eligibility, suffix = _suffix_eligibility(executor, config, actor, sources, f"round{round_index}")
    result = dict(round=round_index, suffix=suffix, candidates=[], promoted=False,
                  old_policy_counterfactual=None)
    raw_experience = []
    if not any(record.state == "eligible" for record in eligibility.records):
        result["stop_reason"] = "actor_suffix_not_eligible"
        return None, None, result, raw_experience
    for index, weight in enumerate(LADDER):
        manager = CandidateTrustManager.from_eligible(eligibility, weight)
        roots = [config["roots"]["trial"]+round_index*10000+index*100+i for i in range(2)]
        trial = _pairs(executor, config, memory, actor, manager, weight, roots, f"round{round_index}_lambda{index}_trial")
        raw_experience += [p["raw"] for p in trial]
        gate = _pair_gate(trial, config["gates"]["controller_gain"])
        candidate = dict(weight=weight, trial_gate=gate,
            trial=[{role: _episode_summary(p[role]) for role in p} for p in trial])
        result["candidates"].append(candidate)
        if not gate["passed"]:
            continue
        roots = [config["roots"]["confirmation"]+round_index*1000+i for i in range(2)]
        confirmation = _pairs(executor, config, memory, actor, manager, weight, roots,
                              f"round{round_index}_lambda{index}_confirmation")
        raw_experience += [p["raw"] for p in confirmation]
        final = _pair_gate(confirmation, config["gates"]["controller_gain"])
        candidate.update(confirmation_gate=final,
            confirmation=[{role: _episode_summary(p[role]) for role in p} for p in confirmation])
        if not final["passed"]:
            result["stop_reason"] = "fresh_controller_confirmation_failed"
            return None, None, result, raw_experience
        if old is not None:
            old_actor, old_manager = old
            counter = [_mpc_episode(executor, config, memory, seed,
                f"round{round_index}_old_policy_{i}", old_actor, deepcopy(old_manager)) for i, seed in enumerate(roots)]
            old_gain = float(np.mean([a["cost"]-p["guided"]["cost"] for a, p in zip(counter, confirmation)]))
            result["old_policy_counterfactual"] = dict(episodes=[_episode_summary(e) for e in counter], mean_gain=old_gain,
                experimental_counterfactual_only=True, inherited_deployment_authority=False)
            if old_gain <= config["gates"]["controller_gain"] or sum(p["guided"]["capture"] for p in confirmation) < sum(e["capture"] for e in counter):
                candidate["old_policy_gate_passed"] = False
                result["stop_reason"] = "new_guidance_did_not_improve_old_policy_counterfactual"
                return None, None, result, raw_experience
        else:
            result["old_policy_counterfactual"] = dict(exercised=False, reason="incoming parent had no previously promoted guidance")
        outcomes = trial + confirmation
        record = manager.records[0]
        promoted = manager.promote(record.record_id, weight,
            [_outcome(p["raw"], executor.task) for p in outcomes],
            [_outcome(p["guided"], executor.task) for p in outcomes],
            baseline_controller_hash=outcomes[0]["raw"]["controller_hash"],
            guided_controller_hash=outcomes[0]["guided"]["controller_hash"])
        if promoted.state != "promoted":
            raise ValueError("Externally passed complete outcome panels failed trust promotion")
        result.update(promoted=True, selected_weight=weight, promoted_record=state_json(asdict(promoted)),
                      controller_confirmation_pass=True)
        executor.archive.append("walker_guidance_promoted", result=result)
        return manager, confirmation, result, raw_experience
    result["stop_reason"] = "no_guidance_setting_passed_trial_and_confirmation"
    return None, None, result, raw_experience


def _revoke_proof(manager, actor, task, archive, tag):
    from .policy import policy_hash
    owner = policy_hash(actor); record = manager.records[0]
    elapsed = record.time_interval[0]; remaining = task.config.T-elapsed; observation = record.support_anchors[0]
    before = manager.weight_for(owner, task.task_hash, 0, elapsed, remaining, observation)
    manager.set_version("invalidated:" + owner, task.task_hash)
    after = manager.weight_for(owner, task.task_hash, 0, elapsed, remaining, observation)
    if before <= 0 or after != 0:
        raise ValueError("Exact-version guidance revocation failed")
    result = dict(exercised=True, before=before, after=after, actor_hash=owner)
    archive.append("walker_forced_version_revocation", tag=tag, **result)
    return result


def _work(executor):
    rows = [json.loads(line) for line in executor.archive.ledger_path.read_text().splitlines()]
    result = executor.totals()
    result.update(neural_rows_reserved=sum(r.get("rows", 0) for r in rows if r["kind"] == "neural_requested"),
        neural_rows_completed=sum(r.get("known_completed_rows", r.get("rows", 0)) for r in rows if r["kind"] == "neural_complete"),
        optimizer_requests=sum(r["kind"] == "optimizer_requested" for r in rows),
        optimizer_steps=sum(r["kind"] == "optimizer_complete" for r in rows),
        optimizer_errors=sum(r["kind"] == "optimizer_error" for r in rows),
        backward_calls=sum(r["kind"] == "backward_complete" for r in rows),
        parameter_assignments=sum(r["kind"] == "walker_parameter_assignment_complete" for r in rows))
    return result


def train(out, config):
    from .walker import WalkerTask, WalkerExecutor
    from .policy import policy_hash
    started = time.perf_counter(); out = _prepare(out, config)
    study = dict(schema=SCHEMA, stages=[], rounds=[], beneficial_loop_pass=False,
                 changed_policy_feedback_rounds=0, revocations=[], stop_reason=None,
                 paid_full_protocol_repeat_performed=False, novelty_claim=False)
    task = executor = None; error = None; work = {}
    with Archive(out / "experiment") as archive:
        archive.max_neural_rows = config["budgets"]["neural_rows"]
        try:
            import torch
            torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
            task = WalkerTask(config["task"])
            executor = WalkerExecutor(task, archive, max_requested_controls=config["budgets"]["requested_controls"])
            actor = _new_actor(config); learner_state = None
            study["initial_actor"] = policy_hash(actor)
            study["initial_checkpoint"] = _checkpoint(archive, "initial_random_actor", actor)
            memory = TrajectoryMemory(config["memory_scale"], task.config.T)
            discovery = [_mpc_episode(executor, config, memory, config["roots"]["discovery"]+i,
                                      "discovery_raw" + str(i)) for i in range(2)]
            study["discovery"] = [_episode_summary(e) for e in discovery]
            study["initial_raw_memory"] = _ingest(memory, discovery, executor, "initial_discovery")
            actor, learner_state, stage = _stage(executor, config, actor, learner_state,
                                                 dict(bootstrap=discovery), 0)
            study["stages"].append(stage)
            if stage["selected"] == "parent":
                study["stop_reason"] = "bootstrap_actor_not_qualified"
            sources, old = discovery, None
            for round_index in range(config["feedback_rounds"]):
                if study["stop_reason"]:
                    break
                manager, confirmation, row, raw_experience = _guidance(executor, config, memory, actor, sources, round_index, old)
                study["rounds"].append(row)
                if manager is None:
                    study["stop_reason"] = row["stop_reason"]
                    row["raw_memory_ingestion"] = _ingest(memory, raw_experience, executor, f"round{round_index}_raw")
                    break
                old_actor, old_manager = actor, deepcopy(manager)
                actor, learner_state, stage = _stage(executor, config, actor, learner_state,
                    dict(raw=[p["raw"] for p in confirmation], guided=[p["guided"] for p in confirmation]), round_index+1)
                study["stages"].append(stage)
                row["next_stage"] = stage["stage"]
                study["changed_policy_feedback_rounds"] += 1
                study["revocations"].append(_revoke_proof(manager, old_actor, task, archive, "round" + str(round_index)))
                row["raw_memory_ingestion"] = _ingest(memory, raw_experience, executor, f"round{round_index}_raw")
                if not stage["guided_next_policy_advantage"]:
                    study["stop_reason"] = "guided_data_child_failed_next_policy_comparison"
                    break
                if stage["selected"] != "guided":
                    raise ValueError("Guided next-policy advantage did not retain that qualified actor")
                sources = [p["guided"] for p in confirmation]
                old = old_actor, old_manager
            study["final_actor"] = policy_hash(actor)
            study["final_checkpoint"] = _checkpoint(archive, "final_retained_actor", actor, learner_state)
            study["beneficial_loop_pass"] = bool(study["stop_reason"] is None
                and study["changed_policy_feedback_rounds"] == config["feedback_rounds"]
                and all(s["guided_next_policy_advantage"] for s in study["stages"][1:]))
            if study["stop_reason"] is None:
                study["stop_reason"] = "requested_feedback_rounds_completed"
            if not study["revocations"]:
                study["revocations"] = [dict(exercised=False, reason="no guidance was promoted")]
            verify_experiment(ROOT, out)
        except BaseException as failure:
            error = dict(type=type(failure).__name__, message=str(failure), traceback=traceback.format_exc())
            study["stop_reason"] = "software_or_evidence_failure"
            archive.append("walker_study_error", error=error)
        finally:
            if executor is not None:
                work = _work(executor)
            if task is not None:
                task.close()
    study.update(work=work, error=error, wall_seconds=time.perf_counter()-started)
    _write(out / "study.json", study)
    result = dict(schema=SCHEMA, stage="train", software_pass=error is None,
        beneficial_loop_pass=bool(error is None and study["beneficial_loop_pass"]),
        changed_policy_feedback_rounds=study["changed_policy_feedback_rounds"], stop_reason=study["stop_reason"],
        work=work, error=error, paid_full_protocol_repeat_performed=False, wall_seconds=study["wall_seconds"])
    _write(out / "result.json", result)
    return result


def report(run, out):
    """Saved-only learning, qualification and executed-controller figures."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    started = time.perf_counter(); run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = [run / name for name in ("study.json", "result.json", "config.json", "manifest.json")]
    bindings = [dict(file=str(p), sha256=file_sha256(p), bytes=p.stat().st_size) for p in paths]
    study, result = json.loads(paths[0].read_text()), json.loads(paths[1].read_text())
    rows = []
    for stage in study["stages"]:
        for arm, child in stage["learning"].items():
            for row in child["updates"]:
                rows.append({k: row[k] for k in ("stage", "arm", "update", "accepted", "before_objective",
                    "candidate_objective", "captures_before", "captures_candidate", "before_cost", "candidate_cost")})
    columns = ("stage", "arm", "update", "accepted", "before_objective", "candidate_objective",
               "captures_before", "captures_candidate", "before_cost", "candidate_cost")
    with (out / "learning.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for stage in study["stages"]:
        for arm, child in stage["learning"].items():
            records = child["updates"]; label = f"stage{stage['stage']} {arm}"
            axes[0].plot([r["update"] for r in records], [r["before_objective"]-r["candidate_objective"] for r in records], marker="o", label=label)
        panel = stage["qualification"]
        roles = list(panel)
        axes[1].plot(roles, [panel[r]["mean_cost"] for r in roles], marker="o", label=f"fresh8 stage{stage['stage']}")
        axes[2].plot(roles, [panel[r]["captures"]/panel[r]["episodes"] for r in roles], marker="o", label=f"stage{stage['stage']}")
    axes[0].axhline(0., color="black", linewidth=.7)
    axes[0].set(title="Paired objective gain within each update", xlabel="Update (ordinary roster changes)", ylabel="Before − candidate")
    axes[1].set(title="Fresh qualification; held panels do not select", ylabel="Complete original cost")
    axes[2].set(title="Fresh standalone capture", ylim=(-.02, 1.02), ylabel="Capture fraction")
    for axis in axes:
        if axis.lines:
            axis.legend(fontsize=7)
    fig.suptitle("MuJoCo Walker pilot: " + study["stop_reason"])
    fig.tight_layout(); fig.savefig(out / "walker_learning.png", dpi=160); plt.close(fig)
    _write(out / "controller_rounds.json", study["rounds"])
    _write(out / "receipt.json", dict(schema=SCHEMA, input_bindings=bindings,
        software_pass=result["software_pass"], beneficial_loop_pass=result["beneficial_loop_pass"],
        actual_feedback_rounds=result["changed_policy_feedback_rounds"], new_physics_controls=0,
        new_neural_rows=0, new_optimizer_steps=0, new_solver_calls=0,
        limitation="Saved scientific records summarized; full raw-query audit is separate, no full paid repeat claimed.",
        wall_seconds=time.perf_counter()-started))
    return json.loads((out / "receipt.json").read_text())
