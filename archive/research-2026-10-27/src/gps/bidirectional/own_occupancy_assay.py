"""Finite measured-action-gradient assay on actual current-actor occupancy.

No work occurs on import. Complete frozen-parent Q probes are auxiliary local
slopes, not updated-actor returns or teacher actions. Own complete stochastic
MC targets remain unchanged. No planning/data-feedback claim is made here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

from .interpolation_assay import CHECKPOINTS, _sha, _write, _comparison, _summary, interpolate_actor
from .qfilter_assay import UNIFORM_CHECKPOINT, _policy_hash, _controller_hash


SEMANTICS = "frozen_parent_own_occupancy_action_gradient"
DEFAULT = dict(schema="bidirectional-ending-own-occupancy-assay-v0",
    task=dict(gate=12, delay=12, move=.2, ending_y=-.1, effort=.02, greed=.2,
              detour=.2, detour_steps=5, penalty=20.),
    learning=dict(lr=.0003, max_grad_norm=.5, epochs=4, clip_ratio=.2, aux_weight=.1,
        batch_size=256, bootstrap_updates=2000, minimum_replicates=2, return_scale=25.),
    occupancy_roots=16, occupancy_seed=181000, probe_delta=.05, local_horizon=4,
    gradient_rows=200, gradient_sampling="linspace_all_400_rows_including_postgate_zero",
    gradient_loss="mean((bounded_mean-saved_actual_parent_mean)*detached_measured_slope/25)",
    arms=["full", "local", "null"], alphas=[.0625, .125, .25, .5, 1.],
    qualification_roots=16, qualification_seed=182000, qualification_noise_seed=184000,
    evaluation_roots=64, evaluation_seed=183000, evaluation_noise_seed=185000,
    minimum_complete_gain=.001, capture_min=.9,
    max_controls=39600, max_model_controls=10400, max_executed_controls=29200,
    max_actor_rows=80000, max_adam_steps=84, max_diagnostic_backward_passes=3)


def load_config(path=None):
    value = deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if value != DEFAULT:
        raise ValueError("Only the fixed own-occupancy gradient assay is accepted")
    return value


def work_bound(config=None):
    config = DEFAULT if config is None else config
    if config != DEFAULT:
        raise ValueError("Undeclared own-occupancy assay")
    T = 25; roots = 16; n = 64*T; steps = 4*math.ceil(n/256)
    model = 2*roots*sum(range(1, T+1)); tails = 2*roots*sum(range(T))
    qualification = (1+3*5)*16*T*2; final = 5*64*T*2
    fit = 3*(n+4*n+steps*200)
    return dict(model_controls=model, occupancy_controls=roots*T,
        qualification_controls=qualification, final_controls=final,
        executed_controls=roots*T+qualification+final,
        total_controls=model+roots*T+qualification+final,
        actor_rows=roots*T+tails+fit+qualification+final,
        occupancy_actor_rows=roots*T, model_tail_actor_rows=tails,
        training_actor_rows=fit, evaluation_actor_rows=qualification+final,
        actor_row_cap_headroom=config["max_actor_rows"]-(roots*T+tails+fit+qualification+final),
        adam_steps=3*steps, diagnostic_backward_passes=3,
        scope="All400actual states and both complete probes retained; null/local/full have200unit-weight rows and identical sampler schedules; no retry or delta tuning")


def input_provenance(precision_run, interpolation_run):
    from .qfilter_assay import input_provenance as ancestor
    value = ancestor(precision_run, interpolation_run)
    value["ancestor_provenance_schema"] = value["schema"]
    value["schema"] = "ending-own-occupancy-input-provenance-v1"
    value["legacy_teacher_inputs"] = "Hash-bound historical ancestry only; never loaded as training labels in this assay"
    return value


def snapshot_inputs(precision_run, interpolation_run, out):
    from .interpolation_assay import _safe_source
    value = input_provenance(precision_run, interpolation_run); target = Path(out)/"inputs"
    target.mkdir(exist_ok=False)
    for row in value["files"]:
        source = _safe_source(Path(value["source_roots"][row["source_scope"]]), row["file"])
        dest = target/row["input_file"]; dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, dest.open("xb") as dst:
            for block in iter(lambda: src.read(1024*1024), b""):
                dst.write(block)
            dst.flush(); os.fsync(dst.fileno())
        if _sha(dest) != row["sha256"] or dest.stat().st_size != row["bytes"]:
            raise ValueError("Ancestor bytes changed during setup")
    _write(Path(out)/"input_provenance.json", value)
    return value


def verify_inputs(out):
    from .qfilter_assay import verify_inputs as implementation
    value = implementation(out)
    if value["schema"] != "ending-own-occupancy-input-provenance-v1":
        raise ValueError("Wrong own-occupancy input provenance")
    return value


def load_owned_episodes(out, provenance, expected_owner):
    """Exact old MC only; no historical teacher action arrays are loaded."""
    import numpy as np
    from .learning import BehaviorEpisode
    from .feedback import _digest
    episodes = []
    for record in provenance["own_episodes"]:
        with np.load(Path(out)/"inputs"/record["input_file"], allow_pickle=False) as raw:
            arrays = {k: raw[k].copy() for k in ("states", "observations", "actions", "pre_tanh", "old_log_probs", "raw_costs")}
            owner = json.loads(str(raw["owner_json"])); context = str(raw["context_id"])
            complete = bool(raw["complete"])
        n = len(arrays["raw_costs"]); states = arrays["states"]
        if (not complete or owner != record["original_event"]["owner"] or owner["actor_hash"] != expected_owner
                or owner["semantics"] != "actor_suffix" or owner["deterministic"] is not False
                or states.shape != (n+1, 2) or n != 25-int(states[0, 0]) or states[-1, 0] != 25
                or arrays["observations"].shape != (n+1, 8) or arrays["actions"].shape != (n, 1)
                or arrays["pre_tanh"].shape != (n, 1) or arrays["old_log_probs"].shape != (n,)
                or not all(np.isfinite(a).all() for a in arrays.values())):
            raise ValueError("Original own MC is incomplete, misaligned, or not frozen-parent owned")
        ref = str(Path(provenance["original_precision_run"])/"evidence/assay"/record["original_event"]["raw"]["file"])
        episodes.append(BehaviorEpisode(arrays["observations"][:-1], arrays["actions"], arrays["pre_tanh"], arrays["old_log_probs"],
            arrays["raw_costs"], context, n, expected_owner, owner["controller_hash"], family=owner["family"], query_ref=ref))
    if len(episodes) != 64 or _digest(episodes) != provenance["behavior_sha256"]:
        raise ValueError("The exact old64 stochastic behavior episodes are required")
    return episodes


def difference_targets(first_actions, costs, center_full, center_local, local_horizon=4):
    """Pure finite-difference arithmetic; never invents smoothness or a value."""
    import numpy as np
    first = np.asarray(first_actions, dtype=np.float64); stage = np.asarray(costs, dtype=np.float64)
    if (first.shape != (2, 1) or stage.ndim != 2 or stage.shape[0] != 2 or stage.shape[1] < 1
            or not np.isfinite(first).all() or not np.isfinite(stage).all()
            or np.any(np.abs(first) > 1) or first[1, 0] <= first[0, 0] or local_horizon != 4
            or not math.isfinite(center_full) or not math.isfinite(center_local)):
        raise ValueError("Two ordered bounded full branches and actual center suffix costs required")
    full = stage.sum(axis=1); local = stage[:, :min(4, stage.shape[1])].sum(axis=1)
    width = float(first[1, 0]-first[0, 0])
    gfull = float((full[1]-full[0])/width); glocal = float((local[1]-local[0])/width)
    return dict(full_costs=full.tolist(), local_costs=local.tolist(), slope_full=gfull, slope_local=glocal,
        slope_null=0., width=width, curvature_full=float(full.sum()-2*center_full),
        curvature_local=float(local.sum()-2*center_local), slope_sign_disagrees=bool(np.sign(gfull) != np.sign(glocal)))


def measure_slope(executor, actor, snapshot, saved_mean, center_costs, center_query_ids, center_raw, source_episode, probe_id):
    """Two complete first-action probes, then frozen actor feedback through T."""
    import numpy as np
    from src.ending.core import Task
    task = executor.task; archive = executor.archive
    if (not isinstance(getattr(task, "dynamics", None), Task) or task.config.T != 25 or task.dynamics.T != 25
            or type(snapshot.elapsed) is not int or not 0 <= snapshot.elapsed < 25
            or np.shape(snapshot.state) != (2,) or snapshot.state[0] != snapshot.elapsed):
        raise ValueError("Exact true-T25 Ending snapshot required")
    mean = np.asarray(saved_mean, dtype=np.float64); costs = np.asarray(center_costs, dtype=np.float64)
    remaining = 25-snapshot.elapsed
    if (mean.shape != (1,) or not np.isfinite(mean).all() or np.any(np.abs(mean) > 1)
            or costs.shape != (remaining,) or not np.isfinite(costs).all()
            or len(center_query_ids) != remaining or len(set(center_query_ids)) != remaining):
        raise ValueError("Actual own mean action and complete saved center suffix required")
    owner = _policy_hash(actor); cid = _controller_hash(actor)
    first = np.clip(np.stack((mean-.05, mean+.05)), -1., 1.)
    center_full = float(np.sum(costs)); center_local = float(np.sum(costs[:min(4, remaining)]))
    request = archive.append("own_gradient_probe_requested", probe_id=probe_id, actor_hash=owner,
        actor_controller_hash=cid, task_hash=task.task_hash, family=0, semantics=SEMANTICS,
        source_episode=source_episode, elapsed=snapshot.elapsed, initial_state=snapshot.state,
        saved_mean=mean, first_actions=first, delta=.05, local_horizon=4,
        center_raw=center_raw, center_query_ids=list(center_query_ids),
        center_full_cost=center_full, center_local_cost=center_local,
        maximum_additional_work=dict(model_controls=2*remaining, actor_rows=2*(remaining-1)))
    traces = []; targets = None
    def save(status):
        return archive.save_npz(f"own_gradient/{probe_id}.npz", status=np.array(status),
            actor_hash=np.array(owner), actor_controller_hash=np.array(cid), task_hash=np.array(task.task_hash),
            semantics=np.array(SEMANTICS), initial_state=np.asarray(snapshot.state), saved_mean=mean,
            first_actions=first, delta=np.array(.05), local_horizon=np.array(4),
            branch_indices=np.array([r["branch"] for r in traces], dtype=np.int64),
            steps=np.array([r["step"] for r in traces], dtype=np.int64),
            prestates=np.asarray([r["prestate"] for r in traces]).reshape(-1, 2),
            states=np.asarray([r["state"] for r in traces]).reshape(-1, 2),
            observations=np.asarray([r["observation"] for r in traces]).reshape(-1, 8),
            actions=np.asarray([r["action"] for r in traces]).reshape(-1, 1),
            raw_costs=np.array([r["cost"] for r in traces]),
            query_ids=np.asarray([r["query_id"] for r in traces], dtype=str),
            center_query_ids=np.asarray(center_query_ids, dtype=str), center_raw_costs=costs,
            center_full_cost=np.array(center_full), center_local_cost=np.array(center_local),
            targets_json=np.array(json.dumps(targets, sort_keys=True)))
    try:
        for branch in (0, 1):
            local = replace(snapshot, state=np.asarray(snapshot.state).copy())
            for step in range(remaining):
                if _policy_hash(actor) != owner: raise ValueError("Frozen Q continuation owner changed")
                if step == 0: action = first[branch].copy()
                else:
                    action = np.asarray(actor.act(task.observation(local), family=0, deterministic=True), dtype=np.float64)
                if action.shape != (1,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError("Invalid bounded frozen actor action")
                observation = np.asarray(task.observation(local)).copy()
                result = executor.model_rollout(local, action.reshape(1, 1, 1), kind="own_gradient_completion_step",
                    metadata=dict(semantics=SEMANTICS, actor_hash=owner, actor_controller_hash=cid,
                        task_hash=task.task_hash, family=0, probe_id=probe_id, source_episode=source_episode,
                        branch=branch, branch_role="minus" if branch == 0 else "plus", step=step, remaining=remaining))
                state = np.asarray(result.states[0, -1]).copy(); cost = np.asarray(result.total_costs)
                if (state.shape != (2,) or state[0] != local.elapsed+1 or not np.isfinite(state).all()
                        or abs(state[1]) > 1 or cost.shape != (1,) or not np.isfinite(cost).all()):
                    raise ValueError("Invalid one-control Ending probe result")
                traces.append(dict(branch=branch, step=step, prestate=local.state.copy(), state=state,
                    observation=observation, action=action.copy(), cost=float(cost[0]), query_id=executor.latest_query_id))
                local = replace(local, state=state, elapsed=local.elapsed+1)
            if local.elapsed != 25: raise ValueError("Probe failed to reach the true ending")
        stages = np.asarray([r["cost"] for r in traces]).reshape(2, remaining)
        targets = difference_targets(first, stages, center_full, center_local)
        endpoints = [traces[(b+1)*remaining-1]["state"][1] <= task.dynamics.ending_y for b in (0, 1)]
        targets.update(endpoint_capture=[bool(v) for v in endpoints],
            probes_clipped=bool(np.any(mean-.05 < -1.) or np.any(mean+.05 > 1.)),
            postgate_invariant=bool(snapshot.elapsed >= 12 and np.allclose(stages[0], stages[1], rtol=0, atol=1e-12)))
        if _policy_hash(actor) != owner: raise ValueError("Parent version changed during probes")
        raw = save("complete"); ids = [r["query_id"] for r in traces]
        record = dict(probe_id=probe_id, actor_hash=owner, task_hash=task.task_hash, source_episode=source_episode,
            elapsed=snapshot.elapsed, initial_state=snapshot.state.tolist(), saved_mean=mean.tolist(),
            center_query_ids=list(center_query_ids), minus_query_ids=ids[:remaining], plus_query_ids=ids[remaining:],
            center_raw=center_raw, raw=raw, targets=targets)
        archive.append("own_gradient_probe_complete", request_id=request, **record)
        return record
    except BaseException as error:
        raw = save("error")
        archive.append("own_gradient_probe_error", request_id=request, probe_id=probe_id, raw=raw,
            error_type=type(error).__name__, error=str(error), completed_query_ids=[r["query_id"] for r in traces])
        raise


def assess_full(mean, stochastic, final=False, minimum=.001, capture_min=.9):
    refs = ("parent", "best_local", "best_null", "uniform_best_raw") if final else ("parent", "local", "null")
    if not {"full", *refs} <= set(mean).intersection(stochastic):
        raise ValueError("All declared actual full/local/null control panels required")
    comparisons = {}
    for data, mode, det in ((mean, "mean", True), (stochastic, "stochastic", False)):
        comparisons[mode] = {ref: _comparison(data[ref], data["full"], minimum, det) for ref in refs}
    capture = sum(r.capture for r in mean["full"])/len(mean["full"])
    return dict(passed=bool(capture >= capture_min and all(r["improved"] for m in comparisons.values() for r in m.values())),
        deterministic_capture_rate=capture, comparisons=comparisons)


def assess_control(mean, stochastic, arm, minimum=.001, capture_min=.9):
    a = _comparison(mean["parent"], mean[arm], minimum, True)
    b = _comparison(stochastic["parent"], stochastic[arm], minimum, False)
    capture = sum(r.capture for r in mean[arm])/len(mean[arm])
    return dict(passed=bool(a["improved"] and b["improved"] and capture >= capture_min),
        mean_vs_parent=a, stochastic_vs_parent=b, deterministic_capture_rate=capture)


def select_steps(rows):
    if [r["alpha"] for r in rows] != DEFAULT["alphas"]:
        raise ValueError("Every fixed alpha must be paid and retained")
    selected = {}
    for arm in ("full", "local", "null"):
        eligible = [r for r in rows if r["criteria"][arm]["passed"]]
        if any(not math.isfinite(r["mean_costs"][arm]) for r in rows):
            raise ValueError("Nonfinite alpha score")
        selected[arm] = None if not eligible else min(eligible, key=lambda r: (r["mean_costs"][arm], r["alpha"]))["alpha"]
    return selected


def run_assay(config, out):
    if config != DEFAULT: raise ValueError("Undeclared own-occupancy assay")
    bound = work_bound(config)
    for key, cap in (("total_controls", "max_controls"), ("model_controls", "max_model_controls"),
                     ("executed_controls", "max_executed_controls"), ("adam_steps", "max_adam_steps"),
                     ("diagnostic_backward_passes", "max_diagnostic_backward_passes")):
        if bound[key] != config[cap]: raise ValueError("Fixed schedule/cap differs: "+key)
    if bound["actor_rows"] > config["max_actor_rows"]:
        raise ValueError("Derived inference schedule exceeds the declared cap")
    out = Path(out).resolve(); manifest = json.loads((out/"manifest.json").read_text())
    if (not Path(__file__).resolve().is_relative_to(out/"source") or _sha(out/"config.json") != manifest["config_sha256"]
            or any(_sha(out/r["file"]) != r["sha256"] for r in manifest["files"])):
        raise ValueError("Use immutable frozen own-occupancy source/config")
    provenance = verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork, WorkPlugin
    from .evidence import Archive
    from .policy import FamilyGaussianActor, policy_hash, controller_hash
    from .learning import PolicyLearner, LearningConfig
    from .own_gradient import OwnGradientBatch, clone_gradient_learner
    from .feedback import _digest
    from .ending_pilot import CountedEnding, EndingContext, EndingExecutor, EndingSnapshot, _episode, _actor_panel
    from .meter import attach_meter
    from .workflow import checkpoint
    start = time.perf_counter(); archive = Archive(out/"evidence"/"assay")
    archive.work_scope = "setup"; archive.budget_failed = False; archive.query_failed = False
    work = DurableWork(out/"work_ledger.jsonl"); plugin = WorkPlugin(work)
    result = dict(schema=config["schema"], passed=False, stage="setup", negative_retained=True,
        qualification=[], selected_steps=None, no_teacher_action_targets=True, no_new_discovery=True,
        full_feedback_established=False, bidirectional_data_benefit_tested=False,
        efficiency_claim=False, no_deployment_promotion=True)
    actor_rows = adam_attempts = diagnostic_attempts = 0; installed = False; dynamics = None
    def phase(name):
        archive.work_scope = name; work.current_test = name; archive.append("assay_phase", phase=name)
    original_before = plugin._before
    def bounded_before(module, args):
        nonlocal actor_rows
        if isinstance(module, FamilyGaussianActor):
            shape = np.shape(args[0]); rows = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            if actor_rows+rows > config["max_actor_rows"]:
                archive.budget_failed = True
                work.append("neural_rejected", rows=rows, reason="fixed_actor_row_cap", before_forward=True)
                raise RuntimeError("Own-occupancy actor-row cap exhausted before call")
            actor_rows += rows
        original_before(module, args)
    plugin._before = bounded_before
    class BoundedEnding(CountedEnding):
        def __init__(self):
            super().__init__(archive, work, config["max_controls"], **config["task"])
            object.__setattr__(self, "model_reserved", 0); object.__setattr__(self, "live_reserved", 0)
        def _begin(self, state, actions, kind, metadata):
            model = np.ndim(actions) == 3; count = int(np.prod(np.shape(actions)[:-1]))
            name = "model_reserved" if model else "live_reserved"
            cap = config["max_model_controls" if model else "max_executed_controls"]
            if getattr(self, name)+count > cap:
                archive.budget_failed = True
                archive.append("query_rejected", query_kind=kind, requested_controls=count,
                    scope=archive.work_scope, reason="fixed_own_gradient_category_cap", before_physics=True)
                raise RuntimeError("Own-occupancy model/live cap exhausted before call")
            object.__setattr__(self, name, getattr(self, name)+count)
            return super()._begin(state, actions, kind, metadata)
    def evaluate(actor, seeds, tag, det, noise):
        attach_meter(actor, archive, tag)
        values = _actor_panel(executor, actor, seeds, tag, deterministic=det, stochastic_seed=noise)
        ah = policy_hash(actor); cid = controller_hash(actor, deterministic=det, family=0)
        if any(r.actor_hash != ah or r.controller_hash != cid or r.semantics != "actor_suffix"
               or not r.complete or r.remaining_steps != 25 or r.deterministic is not det for r in values):
            raise RuntimeError("Complete actual actor outcome owner differs")
        return values
    try:
        plugin.install(); installed = True
        original_adam = torch.optim.Adam.step
        def bounded_adam(optimizer, *args, **kwargs):
            nonlocal adam_attempts
            if adam_attempts >= config["max_adam_steps"]:
                archive.budget_failed = True
                work.append("optimizer_rejected", updates=1, reason="fixed_Adam_cap", before_optimizer=True)
                raise RuntimeError("Own-gradient Adam cap exhausted before update")
            adam_attempts += 1; return original_adam(optimizer, *args, **kwargs)
        torch.optim.Adam.step = bounded_adam
        original_grad = torch.autograd.grad
        def bounded_grad(*args, **kwargs):
            nonlocal diagnostic_attempts
            if diagnostic_attempts >= config["max_diagnostic_backward_passes"]:
                archive.budget_failed = True
                work.append("autograd_grad_rejected", calls=1, reason="fixed_diagnostic_cap", before_backward=True)
                raise RuntimeError("Own-gradient diagnostic backward cap exhausted")
            diagnostic_attempts += 1; return original_grad(*args, **kwargs)
        torch.autograd.grad = bounded_grad
        dynamics = BoundedEnding(); executor = EndingExecutor(EndingContext(dynamics), archive)
        saved = torch.load(out/"inputs/precision/checkpoints"/(CHECKPOINTS["parent"][0]+".pt"), map_location="cpu", weights_only=False)
        parent = PolicyLearner(FamilyGaussianActor.from_checkpoint(saved["learner"]["policy"]), LearningConfig(**config["learning"]))
        parent.load_state_dict(saved["learner"]); parent_digest = _digest(parent.state_dict()); ah = policy_hash(parent.actor)
        if parent_digest != provenance["parent_state_sha256"] or ah != provenance["parent_actor_hash"]:
            raise ValueError("Exact precision parent/optimizer/private RNG mismatch")
        uniform_saved = torch.load(out/"inputs/uniform/checkpoints"/(UNIFORM_CHECKPOINT[0]+".pt"), map_location="cpu", weights_only=False)
        uniform = FamilyGaussianActor.from_checkpoint(uniform_saved["actor"])
        if policy_hash(uniform) != provenance["uniform_actor_hash"]: raise ValueError("Uniform reference actor differs")
        checkpoint(out, "own_gradient_parent", dict(learner=parent.state_dict(), promotion_claim=False), archive)
        checkpoint(out, "own_gradient_uniform_best_raw", dict(actor=uniform.policy_state(),
            source_checkpoint_sha256=UNIFORM_CHECKPOINT[1], promotion_claim=False), archive)
        episodes = load_owned_episodes(out, provenance, ah); own_digest = _digest(episodes)
        archive.append("own_gradient_input_reuse", provenance_sha256=_sha(out/"input_provenance.json"),
            parent_state_sha256=parent_digest, behavior_sha256=own_digest, complete_episodes=len(episodes),
            controls=sum(len(e.raw_costs) for e in episodes), target_owner="original_complete_stochastic_parent",
            legacy_teacher_labels_used=False, new_stochastic_behavior_queries=0)
        phase("own_gradient_occupancy"); attach_meter(parent.actor, archive, "own_gradient_actual_mean_occupancy")
        occupancy = []
        for i in range(16):
            name = f"own_gradient_occupancy_parent_{i:03d}"
            initial = executor.task.reset(config["occupancy_seed"]+i)
            outcome, trace, _, _ = _episode(executor, initial, name, actor=parent.actor, deterministic=True)
            occupancy.append((name, outcome, trace))
        result["occupancy"] = dict(outcomes=_summary([r[1] for r in occupancy]),
            episodes=[dict(episode=name, raw=trace["raw"], query_ids=list(outcome.query_ids)) for name, outcome, trace in occupancy],
            states=400, semantics="Actual complete deterministic current-parent ordinary trajectories")
        archive.append("own_gradient_occupancy_complete", result=result["occupancy"])
        phase("own_gradient_measured_slopes"); attach_meter(parent.actor, archive, "own_gradient_frozen_parent_tails")
        measured = []; refs = []; states = []; obs = []; means = []
        for episode_name, outcome, trace in occupancy:
            for t in range(25):
                snapshot = EndingSnapshot(trace["states"][t].copy(), t, outcome.context_id)
                record = measure_slope(executor, parent.actor, snapshot, trace["actions"][t], trace["raw_costs"][t:],
                    outcome.query_ids[t:], trace["raw"], episode_name, f"probe-{len(measured):04d}")
                measured.append(record); states.append(snapshot.state); obs.append(trace["observations"][t]); means.append(trace["actions"][t])
                refs.append(dict(semantics=SEMANTICS, actor_hash=ah, task_hash=executor.task.task_hash,
                    elapsed=t, state=snapshot.state.tolist(), source_episode=episode_name,
                    center_query_ids=record["center_query_ids"], minus_query_ids=record["minus_query_ids"], plus_query_ids=record["plus_query_ids"],
                    center_raw=trace["raw"], slope_raw=record["raw"]))
        indices = np.linspace(0, 399, 200, dtype=np.int64)
        states = np.asarray(states); obs = np.asarray(obs); means = np.asarray(means)
        slopes = {arm: np.array([r["targets"]["slope_"+arm] for r in measured])[:, None] for arm in config["arms"]}
        batch_raw = archive.save_npz("own_gradient/gradient_batch.npz", all_states=states, all_observations=obs,
            all_saved_means=means, training_indices=indices, all_weights=np.ones(400),
            all_row_references_json=np.array(json.dumps(refs, sort_keys=True)),
            **{"all_slope_"+arm: values for arm, values in slopes.items()},
            selected_states=states[indices], selected_observations=obs[indices], selected_saved_means=means[indices],
            selected_families=np.zeros(200, dtype=np.int64), selected_weights=np.ones(200),
            actor_hash=np.array(ah), task_hash=np.array(executor.task.task_hash), semantics=np.array(SEMANTICS))
        batches = {arm: OwnGradientBatch(obs[indices], np.zeros(200, dtype=np.int64), states[indices], means[indices],
            slopes[arm][indices], ah, executor.task.task_hash, tuple(refs[i] for i in indices), arm) for arm in config["arms"]}
        result["gradients"] = dict(raw=batch_raw, probes=measured, full_rows=400, selected_rows=200,
            weights="all1_including_null", actor_hash=ah, task_hash=executor.task.task_hash,
            objective=SEMANTICS, fixed_delta=.05, local_horizon=4,
            targets_are_updated_actor_returns=False, teacher_actions_used=False)
        archive.append("own_gradient_credit_complete", result=result["gradients"])
        if _digest(parent.state_dict()) != parent_digest: raise RuntimeError("Measured probes changed the frozen parent")
        children = {arm: clone_gradient_learner(parent, executor.task.task_hash) for arm in config["arms"]}
        if any(_digest(c.state_dict()) != parent_digest for c in children.values()): raise RuntimeError("Unmatched child optimizer/RNG state")
        phase("own_gradient_child_fit"); learning = {}
        for arm, child in children.items():
            attach_meter(child.actor, archive, "own_gradient_child_"+arm)
            learning[arm] = child.update(episodes, ah, aux=batches[arm])
            learning[arm].update(auxiliary_loss_semantics=SEMANTICS, gradient_kind=arm,
                auxiliary_is_action_target_regression=False, gradient_batch_raw=batch_raw, unit_weight_rows=200,
                loss_linearization="mean((bounded_mean-saved_actual_mean)*detached_slope/25)")
            archive.append("fit_work", purpose="own_gradient_fit_"+arm, **learning[arm])
            if _digest(episodes) != own_digest: raise RuntimeError("Original MC data changed between arms")
            checkpoint(out, "own_gradient_child_"+arm, dict(learner=child.state_dict(), parent_actor_hash=ah,
                parent_state_sha256=parent_digest, behavior_sha256=own_digest, gradient_batch_raw=batch_raw,
                gradient_kind=arm, own_gradient_task_hash=executor.task.task_hash, learning=learning[arm], promotion_claim=False), archive)
        if len({v["gradient_steps"] for v in learning.values()}) != 1: raise RuntimeError("Unequal arm optimizer work")
        if len({_digest(c.rng.bit_generator.state) for c in children.values()}) != 1: raise RuntimeError("Own-gradient sampler schedules diverged")
        result["learning"] = learning; result["matched_realized_sampler_schedule"] = True
        candidates = {}
        for alpha in config["alphas"]:
            candidates[alpha] = {}
            for arm, child in children.items():
                actor = interpolate_actor(parent.actor, child.actor, alpha); candidates[alpha][arm] = actor
                checkpoint(out, "own_gradient_a"+str(alpha).replace(".", "p")+"_"+arm,
                    dict(actor=actor.policy_state(), alpha=alpha, parameter_names=[n for n, _ in actor.named_parameters()],
                        parent_actor_hash=ah, source_child_actor_hash=policy_hash(child.actor), promotion_claim=False), archive)
        phase("own_gradient_qualification_parent")
        seeds = range(config["qualification_seed"], config["qualification_seed"]+16); noise = config["qualification_noise_seed"]
        pmean = evaluate(parent.actor, seeds, "own_gradient_qual_parent_mean", True, noise)
        pstoch = evaluate(parent.actor, seeds, "own_gradient_qual_parent_stochastic", False, noise)
        result["qualification_parent"] = dict(mean=_summary(pmean), stochastic=_summary(pstoch))
        for alpha in config["alphas"]:
            phase("own_gradient_qualification_alpha_"+str(alpha)); mean = {"parent": pmean}; stochastic = {"parent": pstoch}
            for arm, actor in candidates[alpha].items():
                tag = "own_gradient_qual_a"+str(alpha).replace(".", "p")+"_"+arm
                mean[arm] = evaluate(actor, seeds, tag+"_mean", True, noise)
                stochastic[arm] = evaluate(actor, seeds, tag+"_stochastic", False, noise)
            criteria = dict(full=assess_full(mean, stochastic),
                local=assess_control(mean, stochastic, "local"), null=assess_control(mean, stochastic, "null"))
            row = dict(alpha=alpha, criteria=criteria,
                mean_costs={arm: math.fsum(o.cost for o in mean[arm])/16 for arm in config["arms"]},
                mean_panels={n: _summary(p) for n, p in mean.items() if n != "parent"},
                stochastic_panels={n: _summary(p) for n, p in stochastic.items() if n != "parent"},
                actor_hashes={n: policy_hash(a) for n, a in candidates[alpha].items()})
            result["qualification"].append(row); archive.append("own_gradient_qualification", result=row)
        selected = select_steps(result["qualification"]); result["selected_steps"] = selected
        archive.append("own_gradient_selection", selected_steps=selected,
            control_parent_fallback={arm: selected[arm] is None for arm in ("local", "null")},
            rule="Each arm independently eligible;minimum_mean_cost;smaller_alpha_tie;controls_parent_fallback", final_evidence_not_seen=True)
        if selected["full"] is None:
            result["stage"] = "failed_all_own_gradient_qualification"
        else:
            phase("own_gradient_final_fresh_evaluation")
            seeds = range(config["evaluation_seed"], config["evaluation_seed"]+64); noise = config["evaluation_noise_seed"]
            roster = dict(parent=parent.actor, full=candidates[selected["full"]]["full"],
                best_local=parent.actor if selected["local"] is None else candidates[selected["local"]]["local"],
                best_null=parent.actor if selected["null"] is None else candidates[selected["null"]]["null"], uniform_best_raw=uniform)
            result["final_actor_hashes"] = {n: policy_hash(a) for n, a in roster.items()}
            mean = {n: evaluate(a, seeds, "own_gradient_final_"+n+"_mean", True, noise) for n, a in roster.items()}
            stochastic = {n: evaluate(a, seeds, "own_gradient_final_"+n+"_stochastic", False, noise) for n, a in roster.items()}
            criteria = assess_full(mean, stochastic, final=True)
            result.update(passed=criteria["passed"], negative_retained=not criteria["passed"],
                stage="complete" if criteria["passed"] else "failed_fresh_own_gradient_improvement",
                final=dict(criteria=criteria, mean_panels={n: _summary(p) for n, p in mean.items()},
                    stochastic_panels={n: _summary(p) for n, p in stochastic.items()}))
        if _digest(parent.state_dict()) != parent_digest: raise RuntimeError("Frozen source parent changed")
    except BaseException as error:
        result.update(passed=False, stage="execution_error", negative_retained=True, error_type=type(error).__name__, error=str(error))
        archive.append("assay_error", error_type=type(error).__name__, error=str(error))
    finally:
        if installed: plugin.uninstall()
        work.close(); archive.close()
    events = [json.loads(line) for line in work.path.read_text().splitlines()]
    local = [json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
    result["accounting"] = dict(
        requested_scalar_controls=sum(e["controls"] for e in events if e["kind"] == "scalar_transition_requested"),
        completed_scalar_controls=sum(e["controls"] for e in events if e["kind"] == "scalar_transition_complete"),
        reserved_query_controls=0 if dynamics is None else dynamics.reserved,
        model_controls=0 if dynamics is None else dynamics.model_reserved,
        executed_controls=0 if dynamics is None else dynamics.live_reserved,
        actor_rows=sum(e["rows"] for e in events if e["kind"] == "neural_requested"),
        actor_calls=sum(e["kind"] == "neural_requested" for e in events),
        adam_attempts=sum(e["kind"] == "optimizer_update_requested" for e in events),
        adam_completed=sum(e["kind"] == "optimizer_update_complete" for e in events),
        diagnostic_backward_attempts=sum(e.get("passes", 0) for e in local if e["kind"] == "diagnostic_backward_requested"),
        gradient_backward_attempts=sum(e.get("passes", 0) for e in local if e["kind"] == "gradient_backward_requested"),
        global_autograd_grad_calls=sum(e["kind"] == "autograd_grad_requested" for e in events),
        global_autograd_backward_calls=sum(e["kind"] == "autograd_backward_requested" for e in events),
        new_mujoco_controls=0, new_warp_controls=0, query_totals=archive.totals(),
        overlap="Global hooks and attached local meters observe the same work; never sum them")
    old = provenance["precision"]["precision_paid_ancestry"]; v3 = provenance["precision"]["v3_paid_ancestry"]
    uniform_paid = provenance["uniform_paid_ancestry"]; fresh = result["accounting"]
    result["cost_crosswalk"] = dict(new_assay=fresh, precision_ancestor=old, v3_ancestor=v3, uniform_step_ancestor=uniform_paid,
        **{"unique_lineage_"+k: sum(r[k] for r in (fresh, old, v3, uniform_paid)) for k in ("requested_scalar_controls", "actor_rows", "adam_attempts")},
        scope="Shared data ancestors each once; other negative research trials remain separately paid in the global work index")
    unchanged = all(_sha(out/r["file"]) == r["sha256"] for r in manifest["files"])
    if not unchanged: result.update(passed=False, stage="source_changed", negative_retained=True)
    result.update(config_sha256=_sha(out/"config.json"), manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"), source_unchanged=unchanged,
        query_ledger_sha256=_sha(archive.ledger_path), work_ledger_sha256=_sha(work.path),
        assay_wall_seconds=time.perf_counter()-start, declared_work_bound=bound,
        reused_ancestry_wall_seconds=provenance["precision"]["precision_wall_seconds"]+provenance["precision"]["v3_wall_seconds"]+provenance["uniform_wall_seconds"],
        no_retuning=True, old_negative_unchanged=True,
        scope="One critic-free own-occupancy finite-action-gradient falsifier; no teacher-action imitation, no completed two-round data feedback, no exploration/efficiency/physical-task claim")
    _write(out/"assay_result.json", result)
    return result
