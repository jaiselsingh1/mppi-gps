"""Paid exact frozen-parent Q weights for actual teacher auxiliary actions.

No work runs on import. Counterfactual deterministic Q queries weight executed
teacher actions only; actual stochastic parent episodes retain MC ownership.
This is a bounded learner assay, not a full feedback or efficiency result.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .interpolation_assay import (
    CHECKPOINTS, _sha, _write, _safe_source, _file_ref, interpolate_actor,
    _comparison, _summary, assess_panels, assess_raw, choose_alpha, choose_raw_alpha)


UNIFORM_CHECKPOINT = ("interpolation_a0p5_raw", "a9ed48b00c9995d8fdae1efbb7d46ebdafcfa4f87ac2f19d86ce48f43b4f0422")
DEFAULT = dict(schema="bidirectional-ending-qfilter-assay-v0", source_arm="precision",
    task=dict(gate=12, delay=12, move=.2, ending_y=-.1, effort=.02, greed=.2,
              detour=.2, detour_steps=5, penalty=20.),
    learning=dict(lr=.0003, max_grad_norm=.5, epochs=4, clip_ratio=.2, aux_weight=.1,
        batch_size=256, bootstrap_updates=2000, minimum_replicates=2, return_scale=25.),
    auxiliary_rows=200, positive_resampling="linspace_indices_over_original_positive_rows_to_200",
    q_semantics="one_actual_teacher_action_then_complete_deterministic_frozen_parent",
    weight_rule="max(Q_parent_mean-Q_actual_teacher,0)",
    cache_rule="exact_float64_state_action_bytes+parent_actor_hash+task_hash+family",
    alphas=[.0625, .125, .25, .5, 1.],
    qualification_roots=16, qualification_seed=171000, qualification_noise_seed=173000,
    evaluation_roots=64, evaluation_seed=172000, evaluation_noise_seed=174000,
    minimum_complete_gain=.001, capture_min=.9,
    max_controls=34800, max_model_controls=10000, max_executed_controls=24800,
    max_actor_rows=61800, max_adam_steps=56, max_diagnostic_backward_passes=2)


def load_config(path=None):
    value = deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if value != DEFAULT:
        raise ValueError("Only the fixed Q-filter assay is accepted")
    return value


def work_bound(config=None):
    config = DEFAULT if config is None else config
    if config != DEFAULT:
        raise ValueError("Undeclared Q-filter assay")
    pairs = 200; T = 25; n = 64*T; steps = 4*math.ceil(n/256)
    model = pairs*2*T
    q_rows = pairs*(1+2*(T-1))
    fit_rows = 2*(n+4*n+steps*200)
    qualification = (1+2*len(config["alphas"]))*16*T*2
    final = 5*64*T*2
    return dict(model_controls=model, executed_controls=qualification+final,
        qualification_controls=qualification, final_controls=final,
        total_controls=model+qualification+final, q_actor_rows=q_rows,
        training_actor_rows=fit_rows, evaluation_actor_rows=qualification+final,
        actor_rows=q_rows+fit_rows+qualification+final,
        adam_steps=2*steps, diagnostic_backward_passes=2, maximum_unique_pairs=pairs,
        scope="All original auxiliary rows retained; exact duplicate pairs reused, both Q branches always queried; all fixed alpha outcomes charged; no retries")


def validate_arm_event(recorded, closed):
    """Allow only the documented post-closure cost crosswalk addition."""
    within_arm = {key: value for key, value in closed.items() if key != "costs"}
    if recorded != within_arm:
        raise ValueError("Closed precision arm differs from its original event")


def input_provenance(precision_run, interpolation_run):
    """Hash-bind closed evidence only, before any actor load or new query."""
    from .interpolation_assay import input_provenance as precision_provenance
    precision = Path(precision_run).resolve(); uniform = Path(interpolation_run).resolve()
    parent = precision_provenance(precision)
    result = json.loads((precision/"assay_result.json").read_text())
    arm = result["arms"]["precision"]
    manifest = json.loads((precision/"manifest.json").read_text())
    learning_source = "src/gps/bidirectional/learning.py"
    rows = [r for r in manifest["files"] if r["file"] == "source/"+learning_source]
    if (len(rows) != 1 or _sha(_safe_source(precision, "source/"+learning_source)) != rows[0]["sha256"]
            or _sha(Path(__file__).resolve().parents[3]/learning_source) != rows[0]["sha256"]):
        raise ValueError("The exact original learner implementation is required")
    expected = [f"precision_precision_own_c{i:02d}_r{j}" for i in range(16) for j in range(4)]
    outcomes = {}; arm_event = None
    with (precision/"evidence/assay/ledger.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "episode_outcome" and row.get("episode") in expected:
                if row["episode"] in outcomes:
                    raise ValueError("Duplicate original owned episode")
                outcomes[row["episode"]] = row
            if row.get("kind") == "precision_arm_result" and row.get("arm") == "precision":
                if arm_event is not None:
                    raise ValueError("Duplicate precision-arm result")
                arm_event = row
    # The closed producer appends its final accounting crosswalk after writing
    # precision_arm_result. The immutable within-arm record lacks only costs.
    if set(outcomes) != set(expected) or arm_event is None:
        raise ValueError("Missing or changed complete parent-owned MC corpus")
    validate_arm_event(arm_event["result"], arm)
    refs = []; seen = set()
    def add(scope, root, relative, expected_ref=None):
        ref = _file_ref(root, relative)
        if expected_ref is not None and (ref["sha256"] != expected_ref["sha256"] or ref["bytes"] != expected_ref["bytes"]):
            raise ValueError("Historical raw bytes changed: "+str(relative))
        dest = str(Path(scope)/relative)
        if dest not in seen:
            refs.append(dict(**ref, source_scope=scope, input_file=dest)); seen.add(dest)
        return ref
    for ref in parent["files"]:
        add("precision", precision, ref["file"], ref)
    own = []
    for name in expected:
        row = outcomes[name]; owner = row["owner"]
        if (owner["actor_hash"] != arm["parent_actor_hash"] or owner["deterministic"] is not False
                or owner["semantics"] != "actor_suffix" or not row["outcome"]["complete"]):
            raise ValueError("Only the exact complete stochastic parent may own reused MC targets")
        relative = "evidence/assay/"+row["raw"]["file"]
        add("precision", precision, relative, row["raw"])
        own.append(dict(episode=name, input_file="precision/"+relative, original_event=row))
    auxiliary = arm["auxiliary_raw"]
    add("precision", precision, "evidence/assay/"+auxiliary["file"], auxiliary)
    v3_inputs = json.loads((precision/"input_provenance.json").read_text())
    teacher_refs = []
    for teacher in v3_inputs["episodes"]:
        relative = "inputs/"+teacher["file"]
        add("precision", precision, relative, teacher)
        teacher_refs.append(dict(**teacher, input_file="precision/"+relative))
    # Uniform-data reference was already selected by a separate qualification
    # panel. Its old held outcomes are never training or selection inputs here.
    uniform_names = ["manifest.json", "config.json", "assay_result.json", "input_provenance.json",
        "evidence/assay/ledger.jsonl", "work_ledger.jsonl", "checkpoints/"+UNIFORM_CHECKPOINT[0]+".pt"]
    for name in uniform_names:
        add("uniform", uniform, name)
    uresult = json.loads((uniform/"assay_result.json").read_text())
    umanifest = json.loads((uniform/"manifest.json").read_text())
    uconfig = json.loads((uniform/"config.json").read_text())
    if (uresult.get("schema") != "bidirectional-ending-interpolation-assay-v0"
            or uresult.get("stage") != "failed_fresh_interpolation_improvement" or uresult.get("passed") is not False
            or uresult["raw_selected_alpha"] != .5 or uresult["best_raw_parent_fallback"] is not False
            or uconfig["task"] != DEFAULT["task"]):
        raise ValueError("Require the original closed uniform-data interpolation control")
    for name, field in (("manifest.json", "manifest_sha256"), ("input_provenance.json", "input_provenance_sha256"),
                        ("evidence/assay/ledger.jsonl", "query_ledger_sha256"), ("work_ledger.jsonl", "work_ledger_sha256")):
        if _sha(uniform/name) != uresult[field]:
            raise ValueError("Uniform reference closure changed")
    if _sha(uniform/"config.json") != umanifest["config_sha256"]:
        raise ValueError("Uniform reference configuration changed")
    uniform_checkpoint = None; last = None
    with (uniform/"evidence/assay/ledger.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line); last = row
            if row.get("kind") == "checkpoint" and row.get("label") == UNIFORM_CHECKPOINT[0]:
                if uniform_checkpoint is not None:
                    raise ValueError("Duplicate uniform reference checkpoint")
                uniform_checkpoint = row
    if (last is None or last.get("kind") != "archive_close" or uniform_checkpoint is None
            or uniform_checkpoint["sha256"] != UNIFORM_CHECKPOINT[1]
            or _sha(uniform/"checkpoints"/(UNIFORM_CHECKPOINT[0]+".pt")) != UNIFORM_CHECKPOINT[1]):
        raise ValueError("Wrong/unclosed uniform reference checkpoint")
    uinputs = json.loads((uniform/"input_provenance.json").read_text())
    for role, (_, digest) in CHECKPOINTS.items():
        if uinputs["checkpoints"][role]["sha256"] != digest:
            raise ValueError("Uniform control did not descend from the same precision actors")
    return dict(schema="ending-qfilter-input-provenance-v1", files=refs,
        source_roots=dict(precision=str(precision), uniform=str(uniform)),
        precision=parent, own_episodes=own,
        auxiliary_input_file="precision/evidence/assay/"+auxiliary["file"],
        behavior_sha256=arm["behavior_sha256"], parent_state_sha256=arm["parent_state_sha256"],
        teacher_episodes=teacher_refs, parent_actor_hash=arm["parent_actor_hash"],
        original_precision_run=str(Path(parent["checkpoints"]["parent"]["original_event"]["file"]).parents[1]),
        learning_source_sha256=rows[0]["sha256"], uniform_checkpoint_event=uniform_checkpoint,
        uniform_actor_hash=uresult["best_raw_actor_hash"], uniform_paid_ancestry=uresult["accounting"],
        uniform_wall_seconds=uresult["assay_wall_seconds"],
        ancestry_relationship="V3, precision, and uniform interpolation are one shared paid lineage; each stage's own work is charged once. Old held outcomes are not training labels.",
        no_actor_forward_or_query_in_setup=True, efficiency_claim=False)


def snapshot_inputs(precision_run, interpolation_run, out):
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
            raise ValueError("Ancestor changed during copying; preserve failed preparation")
    _write(Path(out)/"input_provenance.json", value)
    return value


def verify_inputs(out):
    out = Path(out); value = json.loads((out/"input_provenance.json").read_text())
    if (value["precision"]["checkpoints"]["parent"]["sha256"] != CHECKPOINTS["parent"][1]
            or value["uniform_checkpoint_event"]["sha256"] != UNIFORM_CHECKPOINT[1]):
        raise ValueError("Wrong exact frozen parent/uniform reference")
    for row in value["files"]:
        path = _safe_source(out/"inputs", row["input_file"])
        if _sha(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError("Frozen input bytes changed")
    return value


def _policy_hash(actor):
    from .policy import policy_hash
    return policy_hash(actor)


def _controller_hash(actor):
    from .policy import controller_hash
    return controller_hash(actor, deterministic=True, family=0)


def pair_key(state, action, actor_hash, task_hash, family=0):
    import numpy as np
    state = np.ascontiguousarray(state, dtype=np.float64)
    action = np.ascontiguousarray(action, dtype=np.float64)
    if (state.shape != (2,) or action.shape != (1,) or not np.isfinite(state).all()
            or not np.isfinite(action).all() or abs(float(action[0])) > 1
            or not 0 <= state[0] < 25 or state[0] != int(state[0]) or abs(state[1]) > 1):
        raise ValueError("Finite exact Ending state/scalar action required")
    if (type(family) is not int or family != 0 or not isinstance(actor_hash, str) or not actor_hash
            or not isinstance(task_hash, str) or not task_hash):
        raise ValueError("Explicit frozen family/owner/task required")
    digest = hashlib.sha256(json.dumps(dict(owner=actor_hash, task=task_hash, family=family), sort_keys=True).encode())
    digest.update(state.tobytes()); digest.update(action.tobytes())
    return digest.hexdigest()


def measure_pair(executor, actor, snapshot, teacher_action, pair_id):
    """Pay two complete deterministic parent continuations; never relabel MC."""
    import numpy as np
    from src.ending.core import Task
    task = executor.task; archive = executor.archive
    if (not isinstance(getattr(task, "dynamics", None), Task) or task.config.T != 25
            or task.dynamics.T != 25 or np.shape(snapshot.state) != (2,)
            or type(snapshot.elapsed) is not int or snapshot.state[0] != snapshot.elapsed
            or not 0 <= snapshot.elapsed < 25):
        raise ValueError("Exact Ending snapshot and true T25 required")
    owner = _policy_hash(actor); cid = _controller_hash(actor)
    teacher = np.asarray(teacher_action, dtype=np.float64)
    key = pair_key(snapshot.state, teacher, owner, task.task_hash)
    remaining = 25-snapshot.elapsed
    requested = archive.append("qfilter_pair_requested", pair_id=pair_id, pair_key=key,
        actor_hash=owner, actor_controller_hash=cid, task_hash=task.task_hash, family=0,
        initial_state=np.asarray(snapshot.state), context_id=snapshot.context_id,
        remaining=remaining, teacher_action=teacher,
        semantics=DEFAULT["q_semantics"],
        maximum_additional_work=dict(model_controls=2*remaining, actor_rows=1+2*(remaining-1)))
    traces = []; totals = []; first = []
    def act(local):
        value = np.asarray(actor.act(task.observation(local), family=0, deterministic=True), dtype=np.float64)
        if value.shape != (1,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
            raise ValueError("Frozen actor returned invalid bounded scalar action")
        return value.copy()
    def save(status):
        return archive.save_npz(f"qfilter/{pair_id}.npz", pair_key=np.array(key), status=np.array(status),
            initial_state=np.asarray(snapshot.state), remaining=np.array(remaining), actor_hash=np.array(owner),
            actor_controller_hash=np.array(cid), task_hash=np.array(task.task_hash),
            branch_indices=np.array([r["branch"] for r in traces], dtype=np.int64),
            steps=np.array([r["step"] for r in traces], dtype=np.int64),
            prestates=np.asarray([r["prestate"] for r in traces]).reshape(-1, 2),
            states=np.asarray([r["state"] for r in traces]).reshape(-1, 2),
            observations=np.asarray([r["observation"] for r in traces]).reshape(-1, 8),
            actions=np.asarray([r["action"] for r in traces]).reshape(-1, 1),
            raw_costs=np.array([r["cost"] for r in traces]),
            query_ids=np.asarray([r["query_id"] for r in traces], dtype=str),
            first_actions=np.asarray(first).reshape(-1, 1), complete_branch_costs=np.array(totals))
    try:
        first = [act(snapshot), teacher.copy()]
        for branch in range(2):
            local = replace(snapshot, state=np.asarray(snapshot.state).copy()); costs = []
            for step in range(remaining):
                if _policy_hash(actor) != owner:
                    raise ValueError("Frozen parent actor version changed")
                action = first[branch].copy() if step == 0 else act(local)
                observation = np.asarray(task.observation(local)).copy()
                result = executor.model_rollout(local, action.reshape(1, 1, 1), kind="qfilter_actor_completion_step",
                    metadata=dict(semantics=DEFAULT["q_semantics"], actor_hash=owner, actor_controller_hash=cid,
                        task_hash=task.task_hash, family=0, pair_id=pair_id, pair_key=key,
                        branch=branch, branch_role="actor_mean" if branch == 0 else "actual_teacher",
                        step=step, remaining=remaining))
                next_state = np.asarray(result.states[0, -1]).copy()
                cost = np.asarray(result.total_costs)
                if (next_state.shape != (2,) or next_state[0] != local.elapsed+1 or not np.isfinite(next_state).all()
                        or abs(next_state[1]) > 1 or cost.shape != (1,) or not np.isfinite(cost).all() or cost[0] < 0):
                    raise ValueError("Invalid complete one-control model return")
                traces.append(dict(branch=branch, step=step, prestate=local.state.copy(), state=next_state,
                    observation=observation, action=action, cost=float(cost[0]), query_id=executor.latest_query_id))
                costs.append(float(cost[0])); local = replace(local, state=next_state, elapsed=local.elapsed+1)
            if local.elapsed != 25:
                raise ValueError("Q continuation did not reach the true ending")
            totals.append(sum(costs))
        if _policy_hash(actor) != owner:
            raise ValueError("Parent actor version changed during Q qualification")
        weight = max(totals[0]-totals[1], 0.)
        raw = save("complete")
        record = dict(pair_id=pair_id, pair_key=key, actor_hash=owner, task_hash=task.task_hash,
            q_actor=totals[0], q_teacher=totals[1], weight=weight, raw=raw,
            query_ids=[r["query_id"] for r in traces], remaining=remaining)
        archive.append("qfilter_pair_complete", request_id=requested, **record)
        return record
    except BaseException as error:
        raw = save("error")
        archive.append("qfilter_pair_error", request_id=requested, pair_id=pair_id,
            error_type=type(error).__name__, error=str(error), raw=raw,
            completed_query_ids=[r["query_id"] for r in traces])
        raise


def resample_positive(observations, actions, families, weights, rows=200):
    """Pure fixed sampling; zero originals stay visible in the saved source table."""
    import numpy as np
    obs, actions, families, weights = map(np.asarray, (observations, actions, families, weights))
    if (type(rows) is not int or rows != 200 or obs.shape != (200, 8) or actions.shape != (200, 1)
            or families.shape != (200,) or weights.shape != (200,) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isfinite(obs).all() or not np.isfinite(actions).all()):
        raise ValueError("Fixed complete auxiliary rows and nonnegative Q weights required")
    positive = np.flatnonzero(weights > 0)
    indices = np.arange(rows, dtype=np.int64) if not len(positive) else positive[np.linspace(0, len(positive)-1, rows, dtype=np.int64)]
    return dict(observations=obs[indices].copy(), actions=actions[indices].copy(), families=families[indices].copy(),
        weights=weights[indices].copy(), original_indices=indices, positive_original_indices=positive,
        no_auxiliary_signal=not bool(len(positive)))


def assess_final(mean, stochastic, minimum=.001, capture_min=.9):
    if not {"parent", "raw", "guided", "best_raw", "uniform_best_raw"} <= set(mean).intersection(stochastic):
        raise ValueError("All five fresh complete control roles required in both modes")
    base = assess_panels(mean, stochastic, minimum, capture_min)
    if "uniform_best_raw" not in mean or "uniform_best_raw" not in stochastic:
        raise ValueError("Fresh uniform-best-raw control required in both modes")
    for data, field, deterministic in ((mean, "deterministic_guided_vs", True),
                                       (stochastic, "stochastic_guided_vs", False)):
        item = _comparison(data["uniform_best_raw"], data["guided"], minimum, deterministic)
        base[field]["uniform_best_raw"] = item
        base["passed"] = bool(base["passed"] and item["improved"])
    return base


def _load_reused(out, provenance, expected_owner):
    """Recover the exact complete episodes and pre-action auxiliary alignment."""
    import numpy as np
    from .learning import BehaviorEpisode
    from .feedback import _digest
    root = Path(out)/"inputs"; episodes = []
    for row in provenance["own_episodes"]:
        with np.load(root/row["input_file"], allow_pickle=False) as raw:
            arrays = {k: raw[k].copy() for k in ("states", "observations", "actions", "pre_tanh", "old_log_probs", "raw_costs")}
            owner = json.loads(str(raw["owner_json"])); context = str(raw["context_id"])
            complete = bool(raw["complete"])
        n = len(arrays["raw_costs"]); states = arrays["states"]
        if (not complete or owner != row["original_event"]["owner"] or owner["actor_hash"] != expected_owner
                or owner["deterministic"] is not False or owner["semantics"] != "actor_suffix"
                or states.shape != (n+1, 2) or n != 25-int(states[0, 0]) or states[-1, 0] != 25
                or arrays["observations"].shape != (n+1, 8) or arrays["actions"].shape != (n, 1)
                or arrays["pre_tanh"].shape != (n, 1) or arrays["old_log_probs"].shape != (n,)
                or not all(np.isfinite(a).all() for a in arrays.values())):
            raise ValueError("Reused actual MC episode has incomplete/mixed ownership or alignment")
        original = row["original_event"]["raw"]["file"]
        query_ref = str(Path(provenance["original_precision_run"])/"evidence/assay"/original)
        episodes.append(BehaviorEpisode(arrays["observations"][:-1], arrays["actions"], arrays["pre_tanh"],
            arrays["old_log_probs"], arrays["raw_costs"], context, n, expected_owner,
            owner["controller_hash"], family=owner["family"], query_ref=query_ref))
    if len(episodes) != 64 or _digest(episodes) != provenance["behavior_sha256"]:
        raise ValueError("Reused complete own-return dataset is not the exact original 64 episodes")
    groups = {k: [] for k in ("shared", "raw", "guided")}
    for ref in provenance["teacher_episodes"]:
        with np.load(root/ref["input_file"], allow_pickle=False) as raw:
            value = {k: raw[k].copy() for k in ("states", "observations", "actions")}
            captured = bool(raw["captured"])
            if not bool(raw["complete"]) or captured != ref["capture"]:
                raise ValueError("Only complete executed source teachers may supply auxiliary actions")
        if (value["states"].shape != (26, 2) or value["observations"].shape != (26, 8)
                or value["actions"].shape != (25, 1) or not all(np.isfinite(a).all() for a in value.values())):
            raise ValueError("Teacher pre-action arrays are not aligned through true T")
        groups[ref["role"]].append(dict(**value, successful=captured, source=ref))
    shared = [t for t in groups["shared"] if t["successful"]]
    if not shared:
        raise ValueError("No successful common teacher stratum")
    with np.load(root/provenance["auxiliary_input_file"], allow_pickle=False) as raw:
        auxiliary = {k: raw[k].copy() for k in raw.files}
    source_rows = {}; batches = {}
    for arm in ("raw", "guided"):
        specific = [t for t in groups[arm] if t["successful"]]
        if bool(auxiliary[arm+"_specific_is_shared_fallback"]) != (not bool(specific)):
            raise ValueError("Original teacher fallback metadata changed")
        selected = []
        for stratum, traces in (("common", shared), ("specific", specific or shared)):
            indices = auxiliary[arm+"_"+stratum]
            if indices.shape != (100,) or indices.dtype.kind not in "iu":
                raise ValueError("Fixed original auxiliary index strata required")
            for flat in indices:
                index = int(flat)
                if not 0 <= index < 25*len(traces):
                    raise ValueError("Auxiliary source index out of range")
                trace = traces[index//25]; t = index%25
                selected.append(dict(state=trace["states"][t].copy(), observation=trace["observations"][t].copy(),
                    action=trace["actions"][t].copy(), elapsed=t, stratum=stratum,
                    source_episode=trace["source"]["episode"], source_input_file=trace["source"]["input_file"],
                    source_sha256=trace["source"]["sha256"]))
        obs = auxiliary[arm+"_observations"]; actions = auxiliary[arm+"_actions"]; families = auxiliary[arm+"_families"]
        if (obs.shape != (200, 8) or actions.shape != (200, 1) or families.shape != (200,)
                or not np.array_equal(obs, np.stack([r["observation"] for r in selected]))
                or not np.array_equal(actions, np.stack([r["action"] for r in selected])) or np.any(families != 0)):
            raise ValueError("Original auxiliary labels do not match executed pre-action teacher rows")
        source_rows[arm] = selected
        batches[arm] = dict(observations=obs, actions=actions, families=families)
    return episodes, batches, source_rows


def run_assay(config, out):
    if config != DEFAULT:
        raise ValueError("Undeclared Q-filter assay")
    bound = work_bound(config)
    if any(bound[key] != config[target] for key, target in (
            ("total_controls", "max_controls"), ("model_controls", "max_model_controls"),
            ("executed_controls", "max_executed_controls"), ("actor_rows", "max_actor_rows"),
            ("adam_steps", "max_adam_steps"), ("diagnostic_backward_passes", "max_diagnostic_backward_passes"))):
        raise ValueError("Fixed Q-filter schedule and work caps differ")
    out = Path(out).resolve(); manifest = json.loads((out/"manifest.json").read_text())
    if (not Path(__file__).resolve().is_relative_to(out/"source")
            or _sha(out/"config.json") != manifest["config_sha256"]
            or any(_sha(out/r["file"]) != r["sha256"] for r in manifest["files"])):
        raise ValueError("Run only immutable frozen assay source/config")
    provenance = verify_inputs(out)
    import numpy as np
    import torch
    from scripts.diagnostics.run_bidirectional_tests import DurableWork, WorkPlugin
    from .evidence import Archive
    from .policy import FamilyGaussianActor, policy_hash, controller_hash
    from .learning import PolicyLearner, LearningConfig, AuxiliaryBatch
    from .feedback import _digest
    from .ending_pilot import CountedEnding, EndingContext, EndingExecutor, EndingSnapshot, _actor_panel
    from .meter import attach_meter
    from .workflow import checkpoint
    start = time.perf_counter(); archive = Archive(out/"evidence"/"assay")
    archive.work_scope = "setup"; archive.budget_failed = False; archive.query_failed = False
    work = DurableWork(out/"work_ledger.jsonl"); plugin = WorkPlugin(work)
    result = dict(schema=config["schema"], passed=False, stage="setup", negative_retained=True,
        qualification=[], selected_alpha=None, raw_selected_alpha=None,
        full_feedback_established=False, efficiency_claim=False, no_new_discovery=True,
        no_critic=True, q_queries_are_behavior_targets=False, no_deployment_promotion=True)
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
                raise RuntimeError("Fixed Q-filter actor-row cap exhausted before call")
            actor_rows += rows
        original_before(module, args)
    plugin._before = bounded_before
    class BoundedEnding(CountedEnding):
        def __init__(self):
            super().__init__(archive, work, config["max_controls"], **config["task"])
            object.__setattr__(self, "model_reserved", 0); object.__setattr__(self, "live_reserved", 0)
        def _begin(self, state, actions, kind, metadata):
            model = np.ndim(actions) == 3; count = int(np.prod(np.shape(actions)[:-1]))
            key = "model_reserved" if model else "live_reserved"
            cap = config["max_model_controls" if model else "max_executed_controls"]
            if getattr(self, key)+count > cap:
                archive.budget_failed = True
                archive.append("query_rejected", query_kind=kind, requested_controls=count,
                    scope=archive.work_scope, reason="fixed_Qfilter_category_cap", before_physics=True)
                raise RuntimeError("Fixed model/live control category cap exhausted before call")
            object.__setattr__(self, key, getattr(self, key)+count)
            return super()._begin(state, actions, kind, metadata)
    def evaluate(actor, seeds, tag, deterministic, noise):
        attach_meter(actor, archive, tag)
        rows = _actor_panel(executor, actor, seeds, tag, deterministic=deterministic, stochastic_seed=noise)
        ah = policy_hash(actor); cid = controller_hash(actor, deterministic=deterministic, family=0)
        if any(r.actor_hash != ah or r.controller_hash != cid or r.semantics != "actor_suffix"
               or not r.complete or r.remaining_steps != 25 or r.deterministic is not deterministic for r in rows):
            raise RuntimeError("Fresh complete actor outcomes have the wrong continuation owner")
        return rows
    try:
        plugin.install(); installed = True
        original_adam = torch.optim.Adam.step
        def bounded_adam(optimizer, *args, **kwargs):
            nonlocal adam_attempts
            if adam_attempts >= config["max_adam_steps"]:
                archive.budget_failed = True
                work.append("optimizer_rejected", updates=1, reason="fixed_Adam_cap", before_optimizer=True)
                raise RuntimeError("Fixed Adam cap exhausted before call")
            adam_attempts += 1
            return original_adam(optimizer, *args, **kwargs)
        torch.optim.Adam.step = bounded_adam
        original_grad = torch.autograd.grad
        def bounded_grad(*args, **kwargs):
            nonlocal diagnostic_attempts
            if diagnostic_attempts >= config["max_diagnostic_backward_passes"]:
                archive.budget_failed = True
                work.append("autograd_grad_rejected", calls=1, reason="fixed_diagnostic_cap", before_backward=True)
                raise RuntimeError("Fixed diagnostic backward cap exhausted")
            diagnostic_attempts += 1
            return original_grad(*args, **kwargs)
        torch.autograd.grad = bounded_grad
        dynamics = BoundedEnding(); executor = EndingExecutor(EndingContext(dynamics), archive)
        parent_path = out/"inputs/precision/checkpoints"/(CHECKPOINTS["parent"][0]+".pt")
        saved = torch.load(parent_path, map_location="cpu", weights_only=False)
        parent = PolicyLearner(FamilyGaussianActor.from_checkpoint(saved["learner"]["policy"]), LearningConfig(**config["learning"]))
        parent.load_state_dict(saved["learner"]); parent_state = _digest(parent.state_dict()); ah = policy_hash(parent.actor)
        if parent_state != provenance["parent_state_sha256"] or ah != provenance["parent_actor_hash"]:
            raise ValueError("Exact frozen precision parent/optimizer/private RNG was not recovered")
        uniform_path = out/"inputs/uniform/checkpoints"/(UNIFORM_CHECKPOINT[0]+".pt")
        uniform = FamilyGaussianActor.from_checkpoint(torch.load(uniform_path, map_location="cpu", weights_only=False)["actor"])
        if policy_hash(uniform) != provenance["uniform_actor_hash"]:
            raise ValueError("Uniform reference actor identity differs")
        checkpoint(out, "qfilter_parent", dict(learner=parent.state_dict(), promotion_claim=False), archive)
        checkpoint(out, "qfilter_uniform_best_raw", dict(actor=uniform.policy_state(),
            source_checkpoint_sha256=UNIFORM_CHECKPOINT[1], promotion_claim=False), archive)
        episodes, batches, source_rows = _load_reused(out, provenance, ah)
        own_digest = _digest(episodes)
        archive.append("qfilter_input_reuse", provenance_sha256=_sha(out/"input_provenance.json"),
            behavior_sha256=own_digest, parent_state_sha256=parent_state, complete_episodes=len(episodes),
            actual_old_controls=sum(len(e.raw_costs) for e in episodes), target_owner="original_complete_stochastic_parent",
            new_behavior_queries=0, old_held_outcomes_used_for_training=False)
        phase("qfilter_measured_credit"); attach_meter(parent.actor, archive, "qfilter_parent_completion")
        unique = {}; row_keys = {arm: [] for arm in batches}
        for arm, rows in source_rows.items():
            for index, row in enumerate(rows):
                key = pair_key(row["state"], row["action"], ah, executor.task.task_hash)
                row_keys[arm].append(key)
                if key not in unique:
                    unique[key] = dict(pair_id=f"pair-{len(unique):04d}", row=row, aliases=[])
                unique[key]["aliases"].append(dict(arm=arm, row=index, source_episode=row["source_episode"],
                    elapsed=row["elapsed"], source_sha256=row["source_sha256"], stratum=row["stratum"]))
        if len(unique) > bound["maximum_unique_pairs"]:
            raise ValueError("Exact original auxiliary corpus exceeds predeclared unique pair cap")
        archive.append("qfilter_pair_sources", parent_actor_hash=ah, task_hash=executor.task.task_hash,
            unique_pairs=len(unique), logical_rows=400,
            pairs=[dict(pair_key=k, pair_id=v["pair_id"], aliases=v["aliases"]) for k, v in unique.items()])
        measured = {}
        for key, entry in unique.items():
            row = entry["row"]
            snapshot = EndingSnapshot(row["state"].copy(), row["elapsed"], "qfilter:"+entry["pair_id"])
            measured[key] = measure_pair(executor, parent.actor, snapshot, row["action"], entry["pair_id"])
        filtered = {}; resampled = {}; arrays = {}
        for arm, batch in batches.items():
            weights = np.array([measured[k]["weight"] for k in row_keys[arm]])
            selected = resample_positive(**batch, weights=weights, rows=200); resampled[arm] = selected
            filtered[arm] = AuxiliaryBatch(selected["observations"], selected["actions"], selected["families"], selected["weights"])
            arrays.update({arm+"_original_"+k: v for k, v in batch.items()})
            arrays[arm+"_original_weights"] = weights
            arrays[arm+"_pair_ids"] = np.array([unique[k]["pair_id"] for k in row_keys[arm]])
            arrays[arm+"_source_rows_json"] = np.array(json.dumps([{k: v for k, v in r.items() if k not in ("state", "observation", "action")}
                for r in source_rows[arm]], sort_keys=True))
            arrays[arm+"_source_states"] = np.stack([r["state"] for r in source_rows[arm]])
            for name in ("observations", "actions", "families", "weights", "original_indices", "positive_original_indices"):
                arrays[arm+"_filtered_"+name] = selected[name]
        qraw = archive.save_npz("qfilter/auxiliary_credit.npz", **arrays)
        result["qfilter"] = dict(unique_pairs=len(unique), logical_rows=400,
            raw=qraw, pairs=list(measured.values()),
            arms={arm: dict(positive_original_rows=len(v["positive_original_indices"]),
                no_auxiliary_signal=v["no_auxiliary_signal"], original_rows=200, filtered_rows=200,
                weight_sum=float(np.sum(arrays[arm+"_original_weights"]))) for arm, v in resampled.items()},
            identical_initial_optimizer_and_rng=True,
            matched_realized_sampler_schedule=(resampled["raw"]["no_auxiliary_signal"] == resampled["guided"]["no_auxiliary_signal"]),
            sampler_scope="Existing learner skips auxiliary draws when all weights are zero; asymmetric empty support can change later PG permutations despite identical initial RNG and complete data.",
            target_semantics="Measured deterministic parent Q only weights actual teacher actions; unchanged actual stochastic parent G owns PPO targets")
        archive.append("qfilter_credit_complete", result=result["qfilter"])
        if _digest(parent.state_dict()) != parent_state:
            raise RuntimeError("Frozen Q queries changed parent parameters/optimizer/private RNG")
        children = {name: parent.clone_for_trial() for name in ("raw", "guided")}
        if any(_digest(child.state_dict()) != parent_state for child in children.values()):
            raise RuntimeError("Unmatched child initial state")
        phase("qfilter_child_fit"); learning = {}
        for name, child in children.items():
            attach_meter(child.actor, archive, "qfilter_child_"+name)
            learning[name] = child.update(episodes, ah, aux=filtered[name])
            archive.append("fit_work", purpose="qfilter_child_"+name, **learning[name])
            if _digest(episodes) != own_digest:
                raise RuntimeError("Original own-return dataset was mutated")
            checkpoint(out, "qfilter_child_"+name, dict(learner=child.state_dict(),
                parent_actor_hash=ah, parent_state_sha256=parent_state, behavior_sha256=own_digest,
                auxiliary_raw=qraw, learning=learning[name], promotion_claim=False), archive)
        if learning["raw"]["gradient_steps"] != learning["guided"]["gradient_steps"]:
            raise RuntimeError("Unequal child optimizer work")
        result["learning"] = learning
        candidates = {}
        for alpha in config["alphas"]:
            label = "qfilter_a"+str(alpha).replace(".", "p"); candidates[alpha] = {}
            for name, child in children.items():
                actor = interpolate_actor(parent.actor, child.actor, alpha); candidates[alpha][name] = actor
                checkpoint(out, label+"_"+name, dict(actor=actor.policy_state(), alpha=alpha,
                    parameter_names=[n for n, _ in actor.named_parameters()], parent_actor_hash=ah,
                    source_child_actor_hash=policy_hash(child.actor), promotion_claim=False), archive)
        phase("qfilter_qualification_parent")
        seeds = range(config["qualification_seed"], config["qualification_seed"]+16); noise = config["qualification_noise_seed"]
        pmean = evaluate(parent.actor, seeds, "qfilter_qual_parent_mean", True, noise)
        pstoch = evaluate(parent.actor, seeds, "qfilter_qual_parent_stochastic", False, noise)
        result["qualification_parent"] = dict(mean=_summary(pmean), stochastic=_summary(pstoch))
        for alpha in config["alphas"]:
            phase("qfilter_qualification_alpha_"+str(alpha)); panels = {"parent": pmean}; stochastic = {"parent": pstoch}
            for name, actor in candidates[alpha].items():
                tag = "qfilter_qual_a"+str(alpha).replace(".", "p")+"_"+name
                panels[name] = evaluate(actor, seeds, tag+"_mean", True, noise)
                stochastic[name] = evaluate(actor, seeds, tag+"_stochastic", False, noise)
            row = dict(alpha=alpha, criteria=assess_panels(panels, stochastic, config["minimum_complete_gain"], config["capture_min"]),
                raw_criteria=assess_raw(panels, stochastic, config["minimum_complete_gain"], config["capture_min"]),
                guided_mean_complete_cost=math.fsum(r.cost for r in panels["guided"])/16,
                raw_mean_complete_cost=math.fsum(r.cost for r in panels["raw"])/16,
                mean_panels={n: _summary(p) for n, p in panels.items() if n != "parent"},
                stochastic_panels={n: _summary(p) for n, p in stochastic.items() if n != "parent"},
                actor_hashes={n: policy_hash(a) for n, a in candidates[alpha].items()})
            result["qualification"].append(row); archive.append("qfilter_qualification", result=row)
        selected = choose_alpha(result["qualification"]); raw_selected = choose_raw_alpha(result["qualification"])
        result.update(selected_alpha=selected, raw_selected_alpha=raw_selected,
            best_raw_parent_fallback=raw_selected is None)
        archive.append("qfilter_selection", selected_alpha=selected, raw_selected_alpha=raw_selected,
            best_raw_parent_fallback=raw_selected is None, final_evidence_not_seen=True,
            guided_rule="qualified_guided_mean_cost_minimum;smaller_alpha_tie",
            raw_rule="independently_qualified_raw_mean_cost_minimum;smaller_alpha_tie;parent_fallback")
        if selected is None:
            result["stage"] = "failed_all_qfilter_qualification"
        else:
            phase("qfilter_final_fresh_evaluation")
            seeds = range(config["evaluation_seed"], config["evaluation_seed"]+64); noise = config["evaluation_noise_seed"]
            roster = {"parent": parent.actor, **candidates[selected],
                "best_raw": parent.actor if raw_selected is None else candidates[raw_selected]["raw"],
                "uniform_best_raw": uniform}
            result["final_actor_hashes"] = {n: policy_hash(a) for n, a in roster.items()}
            panels = {n: evaluate(a, seeds, "qfilter_final_"+n+"_mean", True, noise) for n, a in roster.items()}
            stochastic = {n: evaluate(a, seeds, "qfilter_final_"+n+"_stochastic", False, noise) for n, a in roster.items()}
            criteria = assess_final(panels, stochastic, config["minimum_complete_gain"], config["capture_min"])
            result.update(passed=criteria["passed"], negative_retained=not criteria["passed"],
                stage="complete" if criteria["passed"] else "failed_fresh_qfilter_improvement",
                final=dict(criteria=criteria, mean_panels={n: _summary(p) for n, p in panels.items()},
                    stochastic_panels={n: _summary(p) for n, p in stochastic.items()}))
        if _digest(parent.state_dict()) != parent_state:
            raise RuntimeError("The original frozen parent changed")
    except BaseException as error:
        result.update(passed=False, stage="execution_error", negative_retained=True,
            error_type=type(error).__name__, error=str(error))
        archive.append("assay_error", error_type=type(error).__name__, error=str(error))
    finally:
        if installed:
            plugin.uninstall()
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
        overlap="Global and local actor/backward/optimizer meters observe the same work; never add them")
    fresh = result["accounting"]; old = provenance["precision"]["precision_paid_ancestry"]
    v3 = provenance["precision"]["v3_paid_ancestry"]; prior_step = provenance["uniform_paid_ancestry"]
    result["cost_crosswalk"] = dict(new_assay=fresh, precision_ancestor=old, v3_ancestor=v3, uniform_step_ancestor=prior_step,
        **{"unique_lineage_"+k: sum(r[k] for r in (fresh, old, v3, prior_step))
           for k in ("requested_scalar_controls", "actor_rows", "adam_attempts")},
        scope="V3, precision, uniform-step assay and current Q-filter each charged once; shared arm attributions overlap")
    unchanged = all(_sha(out/r["file"]) == r["sha256"] for r in manifest["files"])
    if not unchanged:
        result.update(passed=False, stage="source_changed", negative_retained=True)
    result.update(config_sha256=_sha(out/"config.json"), manifest_sha256=_sha(out/"manifest.json"),
        input_provenance_sha256=_sha(out/"input_provenance.json"), source_unchanged=unchanged,
        query_ledger_sha256=_sha(archive.ledger_path), work_ledger_sha256=_sha(work.path),
        assay_wall_seconds=time.perf_counter()-start, declared_work_bound=bound,
        reused_ancestry_wall_seconds=provenance["precision"]["precision_wall_seconds"]+provenance["precision"]["v3_wall_seconds"]+provenance["uniform_wall_seconds"],
        no_retuning=True, old_negative_unchanged=True,
        scope="One paid exact-Q auxiliary learnability assay; not completed two-round bidirectionality, new exploration, efficiency, or physical-task evidence")
    _write(out/"assay_result.json", result)
    return result
