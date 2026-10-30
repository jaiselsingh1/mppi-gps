"""Fresh, non-resumable, bounded candidate-owned Walker signal assay.

No inference or engine is constructed until the explicit frozen manifest,
qualification certificate, corpus and qualified Linux runtime pass preflight.
The independent auditor, not this runner, decides whether the signal gate passes.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.gps.walker_signal_search import (
    PARAMETER_COUNT, PARAMETER_NAMES, PARAMETER_SHAPES, ReservationBudget,
    action_effects, balanced_bank, bounded_norm, canonical_contexts, declared_budget,
    es_combination, flatten_parameters, observations_from_states, select_feasible, validate_config,
)

SERIALIZATION_SECONDS = 0.


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path, value, *, replace=False):
    global SERIALIZATION_SECONDS
    started = time.perf_counter()
    path = Path(path)
    data = (json.dumps(plain(value), indent=2, allow_nan=False)+"\n").encode()
    if replace:
        temporary = path.with_suffix(path.suffix+".tmp")
        with temporary.open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(path)
    else:
        with path.open("xb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)
    SERIALIZATION_SECONDS += time.perf_counter()-started


def write_npz(path, arrays):
    global SERIALIZATION_SECONDS
    started = time.perf_counter()
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    fsync_directory(Path(path).parent)
    SERIALIZATION_SECONDS += time.perf_counter()-started


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    fsync_directory(path.parent)


def role_effects(base_actions, candidate_actions, contexts):
    """Diagnostic arithmetic on already charged calibration actions, no new inference."""
    roles = np.asarray([context["role"] for context in contexts])
    return {role:dict(action_rms=np.sqrt(np.mean((candidate_actions[:, roles == role]-base_actions[None, roles == role])**2,
        axis=(1, 2))), saturation_fractions=np.mean(abs(candidate_actions[:, roles == role]) >= .98, axis=(1, 2)),
        context_count=int(np.sum(roles == role))) for role in ("root", "guide")}


class EventLog:
    def __init__(self, path, clock):
        self.path, self.sequence, self.started = Path(path), 0, time.perf_counter()
        self.clock, self.write_seconds = clock, 0.
        self.file = self.path.open("x", encoding="utf-8")
        fsync_directory(self.path.parent)

    def emit(self, kind, **values):
        started = time.perf_counter()
        row = dict(event_seq=self.sequence, global_event_seq=self.clock[0], kind=kind, utc=datetime.now(timezone.utc).isoformat(),
                   wall_seconds=time.perf_counter()-self.started, **plain(values))
        self.file.write(json.dumps(row, separators=(",", ":"), allow_nan=False)+"\n")
        self.file.flush(); os.fsync(self.file.fileno())
        self.sequence += 1
        self.clock[0] += 1
        self.write_seconds += time.perf_counter()-started
        return row["event_seq"]

    def close(self):
        self.file.close()


def rooted(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Frozen input paths must be relative and contained")
    path = (root/path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Frozen path escapes source packet")
    return path


def preflight(root, config_path, manifest_path):
    """File/metadata checks only; no Torch, actor, checkpoint or engine loading."""
    root, config_path, manifest_path = root.resolve(), config_path.resolve(), manifest_path.resolve()
    config = validate_config(json.loads(config_path.read_text()))
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "walker-owned-signal-manifest-v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("Explicit frozen source/input manifest required")
    files = manifest["files"]
    required = {"scripts/diagnostics/run_walker_owned_signal.py", "src/gps/walker_signal_search.py",
        "src/gps/walker_owned_credit.py", "src/gps/walker_query_journal.py", "src/gps/mppi_experience.py",
        "src/gps/learner_value.py",
        "src/gps/expert_continuation.py", "src/gps/policy.py", "src/rl/walker.py",
        "src/envs/walker2d.py", "src/envs/mujoco_env.py", "src/envs/base.py",
        "src/mppi/mppi.py", "src/utils/config.py", "src/utils/math.py", "assets/walker2d.xml",
        "configs/walker2d_best.json", "inputs/runtime_manifest.json", config["catalog_path"],
        config["qualification_gate_path"], config["task_path"], str(config_path.relative_to(root))}
    if not required.issubset(files):
        raise ValueError("Frozen manifest missing critical inputs: "+str(sorted(required-set(files))))
    comparisons = {}
    for relative, digest in files.items():
        if not isinstance(digest, str) or len(digest) != 64 or sha(rooted(root, relative)) != digest:
            raise ValueError("Frozen file hash mismatch: "+relative)
        comparisons[relative] = digest
    gate = json.loads(rooted(root, config["qualification_gate_path"]).read_text())
    if gate.get("ready") is not True or gate.get("phase") != "PASS" or not all(
            gate.get("proof", {}).get(key) is True for key in ("source_identity_verified",
                "raw_stream_binding_verified", "runtime_metadata_verified", "fully_expanded_config_verified")):
        raise ValueError("Fresh independent qualification is not fully certified")
    runtime = json.loads((root/"inputs/runtime_manifest.json").read_text())
    for relative, digest in runtime["files"].items():
        if sha(rooted(root, relative)) != digest:
            raise ValueError("Actual packet differs from qualified runtime source: "+relative)
    for row in gate["source_checks"]:
        relative = row["path"]
        if (row.get("matched") is not True or sha(rooted(root, relative)) != row.get("runtime")
                or row.get("runtime") != runtime["files"].get(relative)):
            raise ValueError("Qualification certificate and actual source packet disagree: "+relative)
    expected = config["runtime"]
    if (expected["backend"], expected["frame_skip"], expected["control_dt"], expected["platform_system"]
            ) != ("mujoco_cpu", 4, .008, "Linux") or platform.system() != "Linux":
        raise ValueError("Execution requires the qualified Linux CPU runtime")
    for key in ("backend", "frame_skip", "control_dt"):
        if runtime.get(key) != expected[key]:
            raise ValueError("Qualified runtime metadata mismatch: "+key)
    versions = {name: importlib.metadata.version(name) for name in expected["packages"]}
    if versions != expected["packages"] or any(runtime["packages"].get(name) != version
            for name, version in versions.items() if name in runtime["packages"]):
        raise ValueError("Installed packages differ from the frozen qualified runtime")
    catalog = json.loads(rooted(root, config["catalog_path"]).read_text())
    if (catalog["metadata"].get("task_sha256") != sha(rooted(root, config["task_path"]))
            or catalog["metadata"].get("model_sha256") != sha(root/"assets/walker2d.xml")
            or catalog["metadata"].get("teacher_action_or_return_labels_exported") is not False):
        raise ValueError("Historical state corpus model/task identity or no-label contract differs")
    contexts = canonical_contexts(catalog)
    task = json.loads(rooted(root, config["task_path"]).read_text())
    if task != dict(frame_skip=4, cost_style="target_velocity", target_velocity=1.5,
        vel_cost_weight=2., vertical_velocity_cost_weight=.5, angle_cost_weight=.1,
        unhealthy_cost_weight=5., ctrl_cost_weight=.001, terminal_unhealthy_weight=20.,
        apply_terminal_cost_on_done=True):
        raise ValueError("Assay task differs from qualified raw-cost task")
    return config, contexts, task, dict(manifest_sha256=sha(manifest_path),
        manifest_path=str(manifest_path),
        config_sha256=sha(config_path), verified_files=comparisons, packages=versions,
        backend="mujoco_cpu", frame_skip=4, control_dt=.008, declared_budget=declared_budget(),
        qualification_gate_sha256=sha(rooted(root, config["qualification_gate_path"])))


class Assay:
    def __init__(self, out, root, config, contexts, task, proof):
        # Imports are intentionally delayed until the entire file/qualification preflight passed.
        import torch
        from src.gps.policy import TanhMLPPolicy
        from src.gps.walker_owned_credit import actor_owner_hash, candidate_owned_rollout, complete_candidate_scores
        self.torch, self.actor_type = torch, TanhMLPPolicy
        self.owner_hash, self.owned_rollout, self.score_owned = actor_owner_hash, candidate_owned_rollout, complete_candidate_scores
        torch.set_num_threads(1)
        self.out, self.root, self.cfg, self.contexts, self.task = out, root, config, contexts, task
        for name in ("actors", "calibration", "queries", "banks", "rounds", "chains", "checkpoints", "held"):
            (out/name).mkdir()
        fsync_directory(out)
        self.clock = [0]
        self.events, self.inferences = EventLog(out/"events.jsonl", self.clock), EventLog(out/"inferences.jsonl", self.clock)
        self.budget = ReservationBudget(controls=config["max_reserved_controls"],
            queries=config["max_queries"], inferences=config["max_inference_attempts"])
        self.actors, self.calibrations, self.panels, self.query_cache = {}, {}, {}, {}
        self.inference_completions = self.completed_controls = self.completed_queries = 0
        self.timing = dict(query_seconds=0., simulator_call_seconds=0., calibration_seconds=0.,
                           proposal_geometry_seconds=0., selection_score_seconds=0., initialization_seconds=0.)
        self.started, self.blind_open = time.perf_counter(), False
        write_json(out/"config.json", config); write_json(out/"preflight.json", proof)
        write_json(out/"contexts.json", contexts)
        for relative in (config["catalog_path"], config["qualification_gate_path"],
                         config["task_path"], "inputs/runtime_manifest.json"):
            write_bytes(out/relative, rooted(root, relative).read_bytes())
        write_bytes(out/"frozen_manifest.json", Path(proof["manifest_path"]).read_bytes())
        self.frozen_files = proof["verified_files"]
        self.events.emit("assay_begin", declared_budget=declared_budget(), preflight_sha256=sha(out/"preflight.json"))

    def relative(self, path):
        return str(Path(path).relative_to(self.out))

    def arrays(self, actor):
        return {name: tensor.detach().cpu().numpy().copy() for name, tensor in actor.state_dict().items()}

    def register(self, actor):
        owner = self.owner_hash(actor)
        if owner not in self.actors:
            arrays = self.arrays(actor)
            parameters = flatten_parameters(arrays)
            path = self.out/"actors"/(owner+".npz")
            write_npz(path, arrays)
            write_json(path.with_suffix(".json"), dict(owner_hash=owner,
                class_identity="src.gps.policy.TanhMLPPolicy", parameter_names=PARAMETER_NAMES,
                parameter_shapes=PARAMETER_SHAPES, parameter_count=PARAMETER_COUNT,
                raw_path=self.relative(path), raw_sha256=sha(path), parameter_sha256=array_sha(parameters),
                observation_transform="identity", hidden_dim=256, policy_type="tanh_mlp"))
            self.actors[owner] = dict(actor=actor, parameters=parameters, path=self.relative(path), sha256=sha(path))
        return owner

    def clone(self, owner, vector):
        actor = copy.deepcopy(self.actors[owner]["actor"])
        named, offset = dict(actor.named_parameters()), 0
        with self.torch.no_grad():
            for name, shape in zip(PARAMETER_NAMES, PARAMETER_SHAPES):
                size = int(np.prod(shape))
                named[name].copy_(self.torch.tensor(np.asarray(vector[offset:offset+size]).reshape(shape), dtype=self.torch.float32))
                offset += size
        actor.requires_grad_(False).eval()
        new_owner = self.register(actor)
        if not np.array_equal(self.actors[new_owner]["parameters"], vector):
            raise ValueError("Actual serialized candidate differs from recorded float32 proposal")
        bounded_norm(self.actors[owner]["parameters"], self.actors[new_owner]["parameters"], self.cfg["max_step_norm"])
        return new_owner

    def infer(self, owner, observation, *, phase, context_id, query_ref=None, step_index=None):
        if phase == "blind_verification" and not self.blind_open:
            raise ValueError("Held inference attempted before whole-chain freeze")
        observation = np.asarray(observation, dtype=np.float64)
        metadata = dict(owner_hash=owner, phase=phase, context_id=context_id, query_ref=query_ref,
                        step_index=step_index, input_sha256=array_sha(observation),
                        input_dtype=str(observation.dtype), input_shape=list(observation.shape))
        identifier = self.budget.reserve_inference(self.inferences.emit, **metadata)
        returned = False
        try:
            action = np.asarray(self.actors[owner]["actor"].act(observation, deterministic=True))
            returned = True
            if action.shape != (6,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
                raise ValueError("Actor returned invalid bounded action")
            self.inferences.emit("inference_complete", attempt_id=identifier, inference_id=identifier,
                output_sha256=array_sha(action), output_dtype=str(action.dtype), output_shape=list(action.shape), **metadata)
            self.inference_completions += 1
            return action.copy(), identifier
        except BaseException as error:
            self.inferences.emit("inference_error", attempt_id=identifier, inference_id=identifier,
                call_returned=returned, error_type=type(error).__name__, error=str(error), **metadata)
            raise

    def calibrate(self, owner):
        if owner in self.calibrations:
            self.events.emit("calibration_cache_hit", owner_hash=owner)
            return self.calibrations[owner]
        started = time.perf_counter()
        contexts = self.contexts["train"]
        observations = observations_from_states(np.stack([context["state"] for context in contexts]))
        pairs = [self.infer(owner, observation, phase="calibration", context_id=context["id"])
                 for context, observation in zip(contexts, observations)]
        actions, ids = np.stack([pair[0] for pair in pairs]), [pair[1] for pair in pairs]
        path = self.out/"calibration"/(owner+".npz")
        write_npz(path, dict(observations=observations, actions=actions))
        write_json(path.with_suffix(".json"), dict(owner_hash=owner, context_ids=[c["id"] for c in contexts],
            inference_ids=ids, raw_path=self.relative(path), raw_sha256=sha(path)))
        self.calibrations[owner] = actions
        self.timing["calibration_seconds"] += time.perf_counter()-started
        self.events.emit("calibration_complete", owner_hash=owner, rows=24,
                         raw_path=self.relative(path), raw_sha256=sha(path))
        return actions

    def query(self, owner, context, *, phase, seed, round_index):
        from src.gps.expert_continuation import Snapshot
        from src.gps.mppi_experience import finite_policy_rollout
        from src.gps.walker_query_journal import JournaledEnv, QueryJournal
        from src.rl.walker import WalkerRL
        if phase == "blind_verification" and not self.blind_open:
            raise ValueError("Held query attempted before whole-chain freeze")
        key = (owner, context["snapshot_digest"], context["remaining"], self.cfg["gamma"], phase)
        if key in self.query_cache:
            result = self.query_cache[key]
            self.events.emit("query_cache_hit", query_ref=result["query_ref"], owner_hash=owner,
                context_id=context["id"], phase=phase, seed=seed, round=round_index)
            return result
        reference = f"q{self.budget.queries:06d}"
        metadata = dict(owner_hash=owner, context_id=context["id"], phase=phase, seed=seed, round=round_index)
        self.budget.reserve_query(reference, context["remaining"],
            lambda kind, **values: self.events.emit(kind, **values, **metadata))
        reservation_seq = self.events.sequence-1
        reservation_global_seq = self.clock[0]-1
        directory = self.out/"queries"/reference; directory.mkdir()
        fsync_directory(directory.parent)
        journal = QueryJournal(directory/"journal.jsonl", query_id=reference, owner_hash=owner,
            context_id=context["id"], ancestry=context["ancestry"], budget_controls=context["remaining"],
            task_config=self.task, metadata=dict(**metadata, snapshot_digest=context["snapshot_digest"],
                initial_state=context["state"], initial_warmstart=context["warmstart"]))
        fsync_directory(directory)
        inference_ids = []
        assay = self

        class CountedActor:
            def act(self, observation, deterministic=True):
                action, identifier = assay.infer(owner, observation, phase=phase,
                    context_id=context["id"], query_ref=reference, step_index=len(inference_ids))
                inference_ids.append(identifier)
                return action

        class RecordedEnv(JournaledEnv):
            def reset(self, **kwargs):
                journal._emit("reset_attempt", options=kwargs)
                result = self._env.reset(**kwargs)
                journal._emit("reset_complete", state=self.walker.get_state(),
                              warmstart=self.walker.get_warmstart(), observation=result[0])
                return result

        def env_factory(**kwargs):
            self.events.emit("environment_attempt", query_ref=reference, **metadata)
            env = WalkerRL(task_config=rooted(self.root, self.cfg["task_path"]), **kwargs)
            try:
                if (env.walker._use_warp or env.walker._frame_skip != 4
                        or env.walker._dt != .008 or env.walker.model.na != 0
                        or env.walker.model.nmocap != 0 or env.walker._nthread != 1):
                    raise ValueError("Constructed engine differs from frozen CPU restoration recipe")
                self.events.emit("environment_created", query_ref=reference, **metadata)
                return RecordedEnv(env, journal)
            except BaseException:
                env.close()
                raise

        def evaluator(snapshot, actor, **kwargs):
            # Original actor ownership is checked outside; this proxy adds only durable inference accounting.
            if actor is not self.actors[owner]["actor"]:
                raise ValueError("Evaluator received a different candidate actor")
            return finite_policy_rollout(snapshot, CountedActor(), env_factory=env_factory, **kwargs)

        started = time.perf_counter()
        try:
            result = self.owned_rollout(Snapshot(context["state"], context["warmstart"]),
                self.actors[owner]["actor"], remaining=context["remaining"], gamma=self.cfg["gamma"],
                owner_hash=owner, context_id=context["id"], context_ancestry=context["ancestry"],
                query_ref=reference, phase=phase, reset_id=context["ancestry"]["reset_seed"], evaluator=evaluator)
            rollout = result.pop("rollout")
            controls = [json.loads(line) for line in journal.path.read_text().splitlines()
                        if json.loads(line)["kind"] == "control_complete"]
            parts = np.asarray([[row["cost_components"][name] for name in
                ("velocity", "vertical_velocity", "angle", "unhealthy", "control", "fall")]
                for row in controls], dtype=np.float64).reshape(-1, 6)
            path = directory/"raw.npz"
            write_npz(path, dict(states=rollout.states, warmstarts=rollout.warmstarts,
                observations=rollout.observations, actions=rollout.prefix.actions,
                raw_costs=rollout.prefix.raw_costs, cost_components=parts))
            journal.finish(metadata=dict(cost=result["cost"], executed_steps=result["executed_steps"],
                absorbing_steps=result["absorbing_steps"], absorbing_cost=result["absorbing_cost"]))
            journal.close()
            if len(inference_ids) != result["executed_steps"] or len(controls) != result["executed_steps"]:
                raise ValueError("Every executed transition must have one completed inference and control record")
            result.update(raw_path=self.relative(path), raw_sha256=sha(path),
                actor_path=self.actors[owner]["path"], actor_sha256=self.actors[owner]["sha256"],
                journal_path=self.relative(journal.path), journal_sha256=sha(journal.path),
                journal_stats=journal.stats, inference_ids=inference_ids,
                reservation_event_seq=reservation_seq, seed=seed, round=round_index,
                reservation_global_event_seq=reservation_global_seq,
                simulator_call_wall_seconds=sum(row["call_wall_seconds"] for row in controls),
                query_wall_seconds=time.perf_counter()-started)
            write_json(directory/"result.json", {k:v for k,v in result.items() if k != "raw_costs"})
            self.query_cache[key] = result
            self.completed_controls += result["executed_steps"]; self.completed_queries += 1
            self.timing["query_seconds"] += result["query_wall_seconds"]
            self.timing["simulator_call_seconds"] += result["simulator_call_wall_seconds"]
            self.events.emit("query_complete", query_ref=reference, result_path=self.relative(directory/"result.json"),
                result_sha256=sha(directory/"result.json"), executed_controls=result["executed_steps"], **metadata)
            return result
        except BaseException as error:
            partial = getattr(error, "rollout", None)
            partial_record = {}
            if partial is not None:
                partial_path = directory/"failed_returned_rollout.npz"
                write_npz(partial_path, dict(states=partial.states, warmstarts=partial.warmstarts,
                    observations=partial.observations, actions=partial.prefix.actions,
                    raw_costs=partial.prefix.raw_costs))
                partial_record = dict(returned_rollout_path=self.relative(partial_path), returned_rollout_sha256=sha(partial_path))
            if not journal._ended:
                journal.finish(status="error", metadata=dict(error_type=type(error).__name__, error=str(error)))
            journal.close()
            write_json(directory/"failure.json", dict(query_ref=reference, **metadata,
                error_type=type(error).__name__, error=str(error), journal_path=self.relative(journal.path),
                journal_sha256=sha(journal.path), journal_stats=journal.stats, inference_ids=inference_ids, **partial_record))
            self.events.emit("query_failure", query_ref=reference, error=str(error), **metadata)
            raise

    def panel(self, owner, contexts, *, phase, seed, round_index):
        key = (phase, owner)
        if key in self.panels:
            self.events.emit("panel_cache_hit", owner_hash=owner, phase=phase, seed=seed, round=round_index,
                             query_refs=self.panels[key]["query_refs"])
            return self.panels[key]
        queries = [self.query(owner, context, phase=phase, seed=seed, round_index=round_index) for context in contexts]
        started = time.perf_counter()
        scored = self.score_owned([queries], [owner], contexts, gamma=self.cfg["gamma"], phase=phase)
        result = dict(owner_hash=owner, score=float(scored["scores"][0]),
            role_scores={role:float(values[0]) for role, values in scored["role_scores"].items()},
            complete_returns=scored["complete_returns"][0], normalized_returns=scored["normalized_returns"][0],
            query_refs=[query["query_ref"] for query in queries], context_ids=[context["id"] for context in contexts],
            phase=phase)
        self.panels[key] = result
        self.timing["selection_score_seconds"] += time.perf_counter()-started
        self.events.emit("panel_complete", owner_hash=owner, phase=phase, seed=seed, round=round_index,
                         score=result["score"], query_refs=result["query_refs"])
        return result

    def checkpoint(self, seed, index, owner, **metadata):
        path = self.out/"checkpoints"/f"seed{seed}_checkpoint{index}.json"
        value = dict(index=index, owner_hash=owner, actor_path=self.actors[owner]["path"],
            actor_sha256=self.actors[owner]["sha256"], seed=seed,
            status="tentative_not_deployed", **metadata)
        write_json(path, value)
        return dict(index=index, owner_hash=owner, checkpoint_path=self.relative(path), checkpoint_sha256=sha(path))

    def train_seed(self, seed):
        started = time.perf_counter()
        with self.torch.random.fork_rng(devices=[]):
            self.torch.manual_seed(seed)
            actor = self.actor_type(17, -np.ones(6), np.ones(6), hidden_dim=256).cpu().requires_grad_(False).eval()
        base = self.register(actor)
        self.timing["initialization_seconds"] += time.perf_counter()-started
        chain = dict(seed=seed, distribution_failed=False, rounds=[], tentative=True,
                     checkpoints=[self.checkpoint(seed, 0, base, initialization_seed=seed)])
        for index in range(3):
            started = time.perf_counter()
            bank = balanced_bank(self.actors[base]["parameters"], seed=seed*100000+index)
            owners = [self.clone(base, vector) for vector in bank["sample_parameters"]]
            self.timing["proposal_geometry_seconds"] += time.perf_counter()-started
            base_actions = self.calibrate(base)
            actions = np.stack([self.calibrate(owner) for owner in owners])
            effects = action_effects(base_actions, actions)
            effects["by_role"] = role_effects(base_actions, actions, self.contexts["train"])
            prefix = self.out/"banks"/f"seed{seed}_round{index}"
            sample_path = Path(str(prefix)+"_samples.npz")
            write_npz(sample_path, {**{key:bank[key] for key in ("base_parameters", "directions", "intended_parameters",
                "sample_parameters", "step_norms")}, "base_actions":base_actions, "sample_actions":actions,
                "action_rms":effects["action_rms"], "saturation_fractions":effects["saturation_fractions"]})
            write_json(sample_path.with_suffix(".json"), dict(seed=seed, round=index, base_owner=base,
                owner_hashes=owners, context_ids=[c["id"] for c in self.contexts["train"]],
                rng_pre=bank["rng_pre"], rng_post=bank["rng_post"], proposal_seed=bank["proposal_seed"],
                radius=bank["radius"], distribution=effects, raw_path=self.relative(sample_path), raw_sha256=sha(sample_path)))
            if not effects["passes"]:
                chain["distribution_failed"] = True
                self.events.emit("distribution_failure", seed=seed, round=index, base_owner=base,
                    bank_path=self.relative(sample_path.with_suffix(".json")))
                failed = dict(seed=seed, round=index, phase="selection_complete", status="DISTRIBUTION_FAILURE",
                    base_owner=base, sample_bank_path=self.relative(sample_path.with_suffix(".json")),
                    roster=[], tentative_changed=False, checkpoint_owner=base)
                round_path = self.out/"rounds"/f"seed{seed}_round{index}.json"
                write_json(round_path, failed); chain["rounds"].append(self.relative(round_path))
                break
            base_panel = self.panel(base, self.contexts["train"], phase="selection_complete", seed=seed, round_index=index)
            sample_panels = [self.panel(owner, self.contexts["train"], phase="selection_complete", seed=seed,
                                        round_index=index) for owner in owners]
            sample_scores = np.asarray([panel["score"] for panel in sample_panels])
            started = time.perf_counter()
            es = es_combination(bank["base_parameters"], bank["directions"], sample_scores)
            es_owner = self.clone(base, es["parameters"])
            self.timing["proposal_geometry_seconds"] += time.perf_counter()-started
            es_actions = self.calibrate(es_owner)
            es_effects = action_effects(base_actions, es_actions[None], minimum_effective=1)
            es_effects["by_role"] = role_effects(base_actions, es_actions[None], self.contexts["train"])
            es_path = Path(str(prefix)+"_es.npz")
            write_npz(es_path, dict(base_parameters=bank["base_parameters"], es_parameters=es["parameters"],
                combination=es["combination"], paired_differences=es["paired_differences"],
                sample_scores=sample_scores, es_actions=es_actions))
            duplicate = next((label for label, owner in [("base",base)]+[(f"sample:{i}",o) for i,o in enumerate(owners)]
                              if owner == es_owner), None)
            write_json(es_path.with_suffix(".json"), dict(seed=seed, round=index, base_owner=base,
                owner_hash=es_owner, duplicate_of=duplicate, step_norm=es["step_norm"],
                zero_combination=es["zero_combination"], action_effects=es_effects,
                meaning=es["meaning"], raw_path=self.relative(es_path), raw_sha256=sha(es_path)))
            roster, by_owner = [], {}
            entries = [("base",-1,base,base_panel)] + [("sample",i,owner,panel)
                for i,(owner,panel) in enumerate(zip(owners,sample_panels))]
            es_panel = self.panel(es_owner, self.contexts["train"], phase="selection_complete", seed=seed, round_index=index)
            entries.append(("es",-1,es_owner,es_panel))
            for kind, ordinal, owner, panel in entries:
                if owner in by_owner:
                    roster[by_owner[owner]]["duplicate_aliases"].append(dict(kind=kind, index=ordinal))
                    continue
                by_owner[owner] = len(roster)
                roster.append(dict(kind=kind, index=ordinal, **panel, duplicate_aliases=[]))
            step_norms = [bounded_norm(self.actors[base]["parameters"], self.actors[row["owner_hash"]]["parameters"],
                self.cfg["max_step_norm"]) for row in roster]
            choice = select_feasible([row["score"] for row in roster],
                [row["role_scores"]["root"] for row in roster], step_norms,
                maximum=self.cfg["max_step_norm"], minimum_gain=self.cfg["selection_min_improvement"],
                root_tolerance=self.cfg["selection_root_regression_tolerance"])
            for row, eligible, norm in zip(roster, choice["eligible"], step_norms):
                row.update(eligible=bool(eligible), step_norm=norm)
            selected = choice["selected_index"]
            best = roster[selected]
            gain, root_regression, changed = choice["improvement"], choice["root_regression"], bool(choice["tentative_changed"])
            next_owner = best["owner_hash"] if changed else base
            checkpoint = self.checkpoint(seed, index+1, next_owner, round=index, tentative_changed=changed)
            record = dict(seed=seed, round=index, phase="selection_complete", status="TENTATIVE",
                base_owner=base, roster=roster, selected_index=selected, selected_owner=best["owner_hash"],
                feasible_flags=choice["eligible"], unconstrained_selected_index=choice["unconstrained_selected_index"],
                selected_kind=best["kind"], train_improvement=gain, root_regression=root_regression,
                tentative_changed=changed, checkpoint_owner=next_owner, checkpoint_path=checkpoint["checkpoint_path"],
                sample_bank_path=self.relative(sample_path.with_suffix(".json")), es_bank_path=self.relative(es_path.with_suffix(".json")))
            path = self.out/"rounds"/f"seed{seed}_round{index}.json"
            write_json(path, record); chain["rounds"].append(self.relative(path)); chain["checkpoints"].append(checkpoint)
            self.events.emit("tentative_step", seed=seed, round=index, owner_hash=next_owner,
                selected_owner=best["owner_hash"], tentative_changed=changed, record_path=self.relative(path), record_sha256=sha(path))
            base = next_owner
            self.status("SELECTION_TENTATIVE", seed=seed, round=index)
        path = self.out/"chains"/f"seed{seed}.json"
        write_json(path, chain)
        self.events.emit("chain_frozen", seed=seed, chain_path=self.relative(path), chain_sha256=sha(path))
        return chain

    def status(self, phase, **metadata):
        write_json(self.out/"status.json", dict(phase=phase, **metadata, budget=self.budget.summary(),
            completed_queries=self.completed_queries, completed_controls=self.completed_controls,
            inference_completions=self.inference_completions, wall_seconds=time.perf_counter()-self.started,
            timing={**self.timing, "serialization_seconds":SERIALIZATION_SECONDS,
                "event_log_write_seconds":self.events.write_seconds+self.inferences.write_seconds},
            timing_fields_overlap=True,
            trusted_deployment=False, verdict="NOT_ASSIGNED"), replace=True)

    def run(self):
        chains = [self.train_seed(seed) for seed in self.cfg["seeds"]]
        frozen = dict(event_seq=self.events.sequence, global_event_seq=self.clock[0], chains=[dict(seed=c["seed"],
            path=f"chains/seed{c['seed']}.json", sha256=sha(self.out/"chains"/f"seed{c['seed']}.json")) for c in chains])
        write_json(self.out/"chains_frozen.json", frozen)
        self.events.emit("all_chains_frozen", freeze_path="chains_frozen.json", freeze_sha256=sha(self.out/"chains_frozen.json"))
        self.blind_open = True
        self.events.emit("held_phase_begin")
        for chain in chains:
            if chain["distribution_failed"]:
                self.events.emit("held_seed_skipped_distribution_failure", seed=chain["seed"])
                continue
            for checkpoint in chain["checkpoints"]:
                owner = checkpoint["owner_hash"]
                cached = ("blind_verification", owner) in self.panels
                panel = self.panel(owner, self.contexts["heldout"], phase="blind_verification",
                                   seed=chain["seed"], round_index=checkpoint["index"]-1)
                path = self.out/"held"/f"seed{chain['seed']}_checkpoint{checkpoint['index']}.json"
                write_json(path, dict(**panel, seed=chain["seed"], checkpoint_index=checkpoint["index"], cache_alias=cached))
                self.events.emit("held_panel_complete", seed=chain["seed"], checkpoint_index=checkpoint["index"],
                    owner_hash=owner, record_path=self.relative(path), record_sha256=sha(path))
                self.status("BLIND_VERIFICATION", seed=chain["seed"], checkpoint_index=checkpoint["index"])
        for relative, digest in self.frozen_files.items():
            if sha(rooted(self.root, relative)) != digest:
                raise ValueError("Frozen source/input changed during the assay: "+relative)
        self.events.emit("frozen_files_rechecked", files=len(self.frozen_files))
        self.events.emit("assay_complete", budget=self.budget.summary(), trusted_deployment=False,
                         verdict="INDEPENDENT_AUDIT_REQUIRED")
        self.status("COMPLETED_TENTATIVE_BLIND_CHECK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Checks precede output construction and every actor/engine import.
    config, contexts, task, proof = preflight(ROOT, args.config, args.manifest)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fsync_directory(out.parent)
    assay = None
    try:
        assay = Assay(out, ROOT, config, contexts, task, proof)
        assay.run()
    except BaseException as error:
        if assay is not None:
            assay.events.emit("assay_failure", error_type=type(error).__name__, error=str(error),
                              budget=assay.budget.summary())
            assay.status("FAILED_PRESERVED_NO_RETRY", error_type=type(error).__name__, error=str(error))
        else:
            write_json(out/"initialization_failure.json", dict(error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        if assay is not None:
            assay.events.close(); assay.inferences.close()


if __name__ == "__main__":
    main()
