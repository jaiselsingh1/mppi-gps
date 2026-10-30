"""Create-only query evidence and explicit, durable control accounting.

Requested horizon slots are reservations, not silently called executed physics.
Every call is durably recorded before entering the task. Failed/interrupted
calls keep their possible physical work as unknown; no retries are performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .task import (AcrobotCaptureTask, ModelResult, Snapshot, StepResult,
                   TaskConfig, observation, trajectory_family)


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Metadata keys must be strings")
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and np.isfinite(value):
        return value
    raise ValueError("Metadata must contain finite JSON values")


def _packed(value):
    return np.array(json.dumps(_json(value), sort_keys=True, separators=(",", ":")))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_experiment(root,out,config):
    """Copy the small active checkout before execution, never historical runs."""
    import platform,shutil,sys,subprocess,importlib.metadata
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir();files=[]
    paths=[]
    for name in ('src','scripts','tests','configs','docs','assets'):
        paths.extend(p for p in (root/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.extend(root/name for name in ('README.md','pyproject.toml','uv.lock','.python-version','.gitignore') if (root/name).is_file())
    for path in sorted(paths):
        relative=path.relative_to(root);target=source/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        files.append(dict(file=str(relative),sha256=file_sha256(target)))
    with (out/'config.json').open('x') as f:json.dump(_json(config),f,indent=2);f.write('\n')
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=False).stdout.strip()
    manifest=dict(schema='coupled-source-snapshot-v1',files=files,config_sha256=file_sha256(out/'config.json'),
        python=sys.version,platform=platform.platform(),hardware=platform.machine(),git_head=head,
        source_is_worktree_snapshot=True,source_root=str(root),entry_wall_time=time.time(),
        dependency_versions={p:importlib.metadata.version(p) for p in ('numpy','torch')})
    with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')


def verify_experiment(root,out):
    root,out=Path(root).resolve(),Path(out).resolve()
    if root!=out/'source':raise ValueError('Execute from the frozen source tree')
    manifest=json.loads((out/'manifest.json').read_text())
    if file_sha256(out/'config.json')!=manifest['config_sha256'] or any(
            file_sha256(root/r['file'])!=r['sha256'] for r in manifest['files']):
        raise ValueError('Frozen source/configuration changed')


def _error_text(error):
    """Evidence failures must never replace the original validation exception."""
    try:return str(error)
    except Exception:return type(error).__name__+" (exception text unavailable)"


def _rejected_actions(actions,mode):
    """Preserve numeric bytes without ever serializing arbitrary Python objects."""
    descriptor={"input_type":type(actions).__module__+"."+type(actions).__qualname__,
                "numeric_actions_saved":False,"shape":None,"dtype":None}
    arrays={};shape=None
    try:
        value=np.asarray(actions)
        shape=tuple(int(size) for size in value.shape)
        descriptor.update(shape=list(shape),dtype=str(value.dtype))
        if not value.dtype.hasobject and value.dtype.kind in "biufc":
            arrays["actions"]=value.copy()
            descriptor.update(numeric_actions_saved=True,nonfinite_values=int((~np.isfinite(value)).sum()))
        else:descriptor["omission_reason"]="Non-numeric/object values are not pickled or stringified"
    except Exception as error:
        descriptor["conversion_error_type"]=type(error).__name__
    attempted=1 if mode=="live" else (
        int(shape[0]*shape[1]) if shape is not None and len(shape)>=2 else None)
    return arrays,descriptor,attempted


class Archive:
    """One append-only evidence directory, never resumed or overwritten."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/"queries").mkdir()
        self.ledger_path = self.path/"ledger.jsonl"
        self._ledger = self.ledger_path.open("x", encoding="utf-8")
        self._sequence = self._query_sequence = 0
        self._started = time.perf_counter()
        self.append("archive_begin", schema="bidirectional-evidence-v1")

    def append(self, kind, **payload):
        if self._ledger.closed:
            raise RuntimeError("Archive is closed")
        row = _json(dict(id=f"event-{self._sequence:09d}", kind=kind,
            utc=datetime.now(timezone.utc).isoformat(),
            wall_seconds=time.perf_counter()-self._started, **payload))
        self._ledger.write(json.dumps(row, sort_keys=True, separators=(",", ":"))+"\n")
        self._ledger.flush()
        os.fsync(self._ledger.fileno())
        self._sequence += 1
        return row["id"]

    def next_query_id(self):
        identifier = f"q{self._query_sequence:09d}"
        self._query_sequence += 1
        return identifier

    def save_npz(self, name, **arrays):
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".npz":
            raise ValueError("Archive arrays require a safe relative .npz path")
        target = self.path/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        converted = {key: np.asarray(value) for key, value in arrays.items()}
        if any(value.dtype.hasobject for value in converted.values()):
            raise ValueError("Object/pickle arrays are prohibited")
        with target.open("xb") as stream:
            np.savez_compressed(stream, **converted)
            stream.flush()
            os.fsync(stream.fileno())
        return dict(file=str(relative), sha256=file_sha256(target), bytes=target.stat().st_size)

    def totals(self):
        return ledger_totals(self.ledger_path)

    def close(self):
        if not self._ledger.closed:
            self.append("archive_close", accounting=self.totals())
            self._ledger.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def ledger_totals(path):
    """Inspect durable complete lines, conservatively retaining interrupted work."""
    rows, incomplete_tail = [], False
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeError):
            if index != len(lines)-1 or line.endswith(b"\n"):
                raise ValueError("Corrupt middle ledger record")
            incomplete_tail = True
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate evidence record IDs")
    begins = {r["query_id"]: r for r in rows if r["kind"] == "query_requested"}
    completed = {r["query_id"]: r for r in rows if r["kind"] == "query_complete"}
    errors = {r["query_id"]: r for r in rows if r["kind"] == "query_error"}
    if set(completed) & set(errors) or not (set(completed)|set(errors)) <= set(begins):
        raise ValueError("Inconsistent query completion/error references")
    if len(begins) != sum(r["kind"] == "query_requested" for r in rows):
        raise ValueError("Duplicate query requests")
    for kind, records in (("query_complete", completed), ("query_error", errors)):
        if len(records) != sum(r["kind"] == kind for r in rows):
            raise ValueError("Duplicate query resolutions")
    unknown = set(begins)-set(completed)-set(errors)
    requested = sum(r["requested_controls"] for r in begins.values())
    physical = sum(r["requested_physics_controls"] for r in begins.values())
    returned_error = {key: row for key, row in errors.items() if row["call_returned"]}
    known = {**completed, **returned_error}
    rejected=sum(r["kind"]=="query_rejected" for r in rows)
    return dict(requested_controls=requested, charged_requested_controls=requested,
        requested_physics_controls=physical,
        requested_physics_frames=sum(r["requested_physics_frames"] for r in begins.values()),
        completed_controls=sum(r["completed_controls"] for r in known.values()),
        completed_physics_frames=sum(r["completed_physics_frames"] for r in known.values()),
        cutoff_skipped_controls=requested-physical,
        error_requested_controls=sum(begins[q]["requested_controls"] for q in errors),
        unknown_error_controls=sum(begins[q]["requested_physics_controls"] for q in errors if q not in returned_error),
        inflight_or_interrupted_controls=sum(begins[q]["requested_physics_controls"] for q in unknown),
        unknown_controls=sum(begins[q]["requested_physics_controls"] for q in unknown | (set(errors)-set(returned_error))),
        queries=len(begins), query_attempts=len(begins)+rejected,
        completed_queries=len(completed), error_queries=len(errors),
        inflight_or_interrupted_queries=len(unknown),
        rejected_queries=rejected,
        incomplete_final_line=incomplete_tail)


class QueryExecutor:
    """Only this object's step/model methods should cross the physics boundary.

    Direct calls through ``task.env`` would bypass accounting; the runner must
    keep that object private and bind this executor in its frozen source.
    """
    def __init__(self, task: AcrobotCaptureTask, archive: Archive, *, scope="shared",
                 owner=None, max_requested_controls=None):
        self.task, self.archive = task, archive
        self.scope, self.owner = str(scope), _json({} if owner is None else owner)
        if max_requested_controls is not None and (
                type(max_requested_controls) is not int or max_requested_controls < 0):
            raise ValueError("Explicit optional budget must be a nonnegative integer")
        self.max_requested_controls = max_requested_controls
        self.latest_query_id = None

    @property
    def ledger_totals(self):
        return self.archive.totals()

    def _record_rejection(self,initial,actions,*,mode,kind,metadata,error):
        identifier=self.archive.next_query_id();self.latest_query_id=identifier
        arrays,descriptor,attempted=_rejected_actions(actions,mode)
        safe_metadata={"unavailable_input_type":type(metadata).__name__}
        try:safe_metadata=_json({} if metadata is None else metadata)
        except Exception:pass
        owner=self.owner
        if isinstance(safe_metadata,dict):
            try:owner=_json(safe_metadata.get("owner",self.owner))
            except Exception:pass
        snapshot={"available":False,"input_type":type(initial).__name__}
        if isinstance(initial,Snapshot):
            snapshot=initial.to_dict()
            arrays.update(initial_state=initial.state,initial_warmstart=initial.warmstart)
        arrays.update(actions_descriptor_json=_packed(descriptor),
            initial_snapshot_json=_packed(snapshot),metadata_json=_packed(safe_metadata),
            owner_json=_packed(owner))
        raw=None;evidence_error=None
        try:raw=self.archive.save_npz(f"queries/{identifier}_rejected_inputs.npz",**arrays)
        except Exception as failure:evidence_error=type(failure).__name__+": "+_error_text(failure)
        self.archive.append("query_rejected",query_id=identifier,scope=self.scope,owner=owner,
            query_kind=kind,mode=mode,metadata=safe_metadata,error_type=type(error).__name__,
            error=_error_text(error),attempted_controls_unvalidated=attempted,
            attempted_controls_scope="Shape-derived input attempt only; no physics requested",
            action_descriptor=descriptor,requested_controls=0,requested_physics_controls=0,
            requested_physics_frames=0,completed_controls=0,raw=raw,rejected_evidence_error=evidence_error)

    def _query(self, initial, actions, *, mode, kind, metadata):
        unvalidated_actions,unvalidated_metadata=actions,metadata
        try:
            metadata = _json({} if metadata is None else metadata)
            owner = _json(metadata.get("owner", self.owner))
            self.task.validate_snapshot(initial)
            actions = self.task.validate_actions(actions, batch=mode == "model")
            remaining = self.task.config.T-initial.elapsed
            if not remaining:
                raise RuntimeError("Finite episode is already terminated")
            K, H = actions.shape[:2] if mode == "model" else (1, 1)
            requested, physical = int(K*H), int(K*min(H, remaining))
            if self.max_requested_controls is not None and (
                    self.ledger_totals["requested_controls"]+requested > self.max_requested_controls):
                raise RuntimeError("Query budget exhausted before any physics call")
        except Exception as error:
            try:self._record_rejection(initial,unvalidated_actions,mode=mode,kind=kind,
                metadata=unvalidated_metadata,error=error)
            except Exception as evidence_error:
                error.add_note("Rejected-input evidence could not be fully persisted: "+
                               type(evidence_error).__name__+": "+_error_text(evidence_error))
            raise
        identifier = self.archive.next_query_id()
        self.latest_query_id = identifier
        inputs = self.archive.save_npz(f"queries/{identifier}_inputs.npz",
            actions=actions, initial_state=initial.state, initial_warmstart=initial.warmstart,
            initial_snapshot_json=_packed(initial.to_dict()), metadata_json=_packed(metadata),
            owner_json=_packed(owner), task_json=_packed(self.task.config.to_dict()))
        # This fsynced record precedes the model/live call. Unknown partial work
        # survives failures and process death; requested padding is kept separate.
        frames = getattr(self.task.env, "_frame_skip", 1)
        self.archive.append("query_requested", query_id=identifier, scope=self.scope,
            owner=owner, query_kind=kind, mode=mode, metadata=metadata,
            context_id=initial.context_id, task_hash=initial.task_hash,
            initial_elapsed=initial.elapsed, requested_controls=requested,
            requested_physics_controls=physical, requested_physics_frames=physical*frames,
            cutoff_skipped_controls=requested-physical, inputs=inputs)
        started, returned = time.perf_counter(), False
        try:
            result = self.task.model_rollout(initial, actions) if mode == "model" else self.task.step(actions)
            returned = True
            if mode == "model":
                arrays = dict(prestates=result.prestates, states=result.states,
                    observations=result.observations, actions=result.actions,
                    stage_costs=result.stage_costs, component_costs=result.component_costs,
                    terminal_heuristic=result.terminal_heuristic, terminal_costs=result.terminal_costs,
                    capture_failures=result.capture_failures, total_costs=result.total_costs,
                    actual_partial_costs=result.actual_partial_costs, hold_counts=result.hold_counts,
                    tip_distances=result.tip_distances, qvel_norms=result.qvel_norms,
                    legacy_sensors=result.legacy_sensors, boundary_sensors=result.boundary_sensors,
                    legacy_base_scores=result.legacy_base_scores,
                    legacy_stage_costs=result.legacy_stage_costs,
                    legacy_terminal_heuristic=result.legacy_terminal_heuristic,
                    cost_boundary_source=np.array(result.cost_boundary_source),
                    endpoint_warmstarts=np.zeros((len(result.actions), 2)),
                    endpoint_warmstart_exact=np.zeros(len(result.actions), dtype=bool),
                    endpoint_restart_semantics=np.array("explicit-zero-solver-warmstart-intervention"))
            else:
                arrays = dict(prestates=initial.state[None], states=result.snapshot.state[None],
                    pre_warmstarts=initial.warmstart[None], warmstarts=result.snapshot.warmstart[None],
                    observations=self.task.observation(initial)[None], actions=result.action[None],
                    stage_costs=np.array([result.stage_cost]), component_costs=result.components[None],
                    terminal_costs=np.array([result.terminal_cost]),
                    capture_failures=np.array([result.capture_failure]), total_costs=np.array([result.total_cost]),
                    elapsed=np.array([result.snapshot.elapsed]), hold_counts=np.array([result.snapshot.hold_count]),
                    tip_distances=np.array([result.tip_distance]), qvel_norms=np.array([result.qvel_norm]),
                    done=np.array([result.done]), captured=np.array([result.captured]))
                arrays.update(legacy_stage_costs=np.array([result.legacy_stage_cost]),
                    legacy_sensors=result.legacy_sensors[None],
                    boundary_sensors=result.boundary_sensors[None],
                    cost_boundary_source=np.array(result.cost_boundary_source))
            raw = self.archive.save_npz(f"queries/{identifier}_result.npz", **arrays,
                initial_state=initial.state, initial_warmstart=initial.warmstart,
                initial_snapshot_json=_packed(initial.to_dict()), metadata_json=_packed(metadata),
                owner_json=_packed(owner), task_hash=np.array(initial.task_hash),
                component_names=np.array(self.task.component_names),
                backend_json=_packed(self.task.last_model_backend if mode == "model" else
                                     {"backend": "mujoco_cpu", "frame_skip": frames}))
            self.archive.append("query_complete", query_id=identifier, scope=self.scope,
                completed_controls=physical, completed_physics_frames=physical*frames,
                owner=owner, backend=self.task.last_model_backend if mode == "model" else "mujoco_cpu",
                model_construction_events=self.task.model_construction_events,
                query_wall_seconds=time.perf_counter()-started, result=raw)
            return result
        except BaseException as error:
            self.archive.append("query_error", query_id=identifier, scope=self.scope,
                call_returned=returned, completed_controls=physical if returned else 0,
                completed_physics_frames=physical*frames if returned else 0,
                partial_physics_unknown=not returned, error_type=type(error).__name__,
                error=str(error), query_wall_seconds=time.perf_counter()-started)
            raise

    def model_rollout(self, initial, actions, *, kind="planner_bank", metadata=None):
        return self._query(initial, actions, mode="model", kind=kind, metadata=metadata)

    def step(self, action, *, metadata=None):
        return self._query(self.task.snapshot(), action, mode="live", kind="executed_control",
                           metadata=metadata)


@dataclass(frozen=True)
class EpisodeTrace:
    """Complete actual episode; targets must retain this controller ownership."""
    initial_snapshot: Snapshot
    states: np.ndarray
    warmstarts: np.ndarray
    elapsed: np.ndarray
    hold_counts: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    stage_costs: np.ndarray
    terminal_costs: np.ndarray
    capture_failures: np.ndarray
    component_costs: np.ndarray
    tip_distances: np.ndarray
    family: str
    actor_owner: Any
    execution_semantics: str
    source: Any
    query_ids: tuple[str, ...]
    complete: bool
    captured: bool
    T: int = 800

    @classmethod
    def from_steps(cls, initial, steps, *, config, actor_owner,
                   execution_semantics, source, query_ids=()):
        config = TaskConfig.from_dict(config) if isinstance(config, dict) else config
        steps = tuple(steps)
        if not steps:
            raise ValueError("An episode trace needs at least one actual transition")
        previous = initial
        for step in steps:
            before = step.initial_snapshot
            if (before.context_id != initial.context_id or before.task_hash != initial.task_hash
                    or before.elapsed != previous.elapsed
                    or before.hold_count != previous.hold_count
                    or not np.array_equal(before.state, previous.state)
                    or not np.array_equal(before.warmstart, previous.warmstart)):
                raise ValueError("Episode transitions have mismatched root ancestry or boundaries")
            if step.snapshot.elapsed != previous.elapsed+1 or previous.elapsed >= config.T:
                raise ValueError("Episode clock is not consecutive")
            previous = step.snapshot
        if initial.task_hash != config.task_hash:
            raise ValueError("Episode configuration does not match initial task")
        if query_ids and len(query_ids) != len(steps):
            raise ValueError("Every recorded transition needs its own query reference")
        boundaries = (initial,)+tuple(step.snapshot for step in steps)
        states = np.stack([s.state for s in boundaries])
        # Kinematics are read from saved state; no simulator or action inference.
        q = states[:, 1:3]
        x = np.sin(q[:, 0])+np.sin(q[:, 0]+q[:, 1])
        z = 2.+np.cos(q[:, 0])+np.cos(q[:, 0]+q[:, 1])
        distances = np.sqrt(x*x+(z-4.)**2)
        return cls(initial, states, np.stack([s.warmstart for s in boundaries]),
            np.array([s.elapsed for s in boundaries]), np.array([s.hold_count for s in boundaries]),
            np.stack([observation(s, config) for s in boundaries]),
            np.stack([s.action for s in steps]), np.array([s.stage_cost for s in steps]),
            np.array([s.terminal_cost for s in steps]), np.array([s.capture_failure for s in steps]),
            np.stack([s.components for s in steps]), distances,
            trajectory_family(states, distances, approach_distance=config.local_distance),
            _json(actor_owner), str(execution_semantics), _json(source), tuple(query_ids),
            initial.elapsed == 0 and previous.elapsed == config.T, bool(steps[-1].captured), config.T)

    @property
    def total_cost(self):
        return float(self.stage_costs.sum()+self.terminal_costs.sum()+self.capture_failures.sum())

    @property
    def successful(self):
        return self.complete and self.captured

    @property
    def continuation_complete(self):
        """The queried controller reached T, including a non-root handover."""
        return bool(self.elapsed[-1] == self.T)

    def arrays(self):
        result = {name: getattr(self, name) for name in (
            "states", "warmstarts", "elapsed", "hold_counts", "observations", "actions",
            "stage_costs", "terminal_costs", "capture_failures", "component_costs", "tip_distances")}
        result.update(initial_snapshot_json=_packed(self.initial_snapshot.to_dict()),
            family=np.array(self.family), actor_owner_json=_packed(self.actor_owner),
            execution_semantics=np.array(self.execution_semantics), source_json=_packed(self.source),
            query_ids=np.array(self.query_ids, dtype=str), complete=np.array(self.complete),
            captured=np.array(self.captured), total_cost=np.array(self.total_cost), T=np.array(self.T))
        return result

    def to_npz(self, path):
        target = Path(path)
        with target.open("xb") as stream:
            np.savez_compressed(stream, **self.arrays())
            stream.flush()
            os.fsync(stream.fileno())

    @classmethod
    def from_npz(cls, path):
        with np.load(path, allow_pickle=False) as raw:
            initial = Snapshot.from_dict(json.loads(str(raw["initial_snapshot_json"])))
            return cls(initial, *(raw[name].copy() for name in (
                "states", "warmstarts", "elapsed", "hold_counts", "observations", "actions",
                "stage_costs", "terminal_costs", "capture_failures", "component_costs", "tip_distances")),
                str(raw["family"]), json.loads(str(raw["actor_owner_json"])),
                str(raw["execution_semantics"]), json.loads(str(raw["source_json"])),
                tuple(str(x) for x in raw["query_ids"]), bool(raw["complete"]), bool(raw["captured"]),
                int(raw["T"]))

    def archive(self, archive, episode_id):
        # Failures are saved too. Only a genuinely complete capture enters a
        # successful family partition; no discarded experience is implied.
        partition = f"success/{self.family}" if self.successful else "failures"
        if not isinstance(episode_id, str) or not episode_id or "/" in episode_id or ".." in episode_id:
            raise ValueError("Episode IDs must be safe path components")
        saved = archive.save_npz(f"episodes/{partition}/{episode_id}.npz", **self.arrays())
        archive.append("episode_saved", episode_id=episode_id, successful=self.successful,
            complete=self.complete, family=self.family, context_id=self.initial_snapshot.context_id,
            task_hash=self.initial_snapshot.task_hash, actor_owner=self.actor_owner,
            execution_semantics=self.execution_semantics, total_cost=self.total_cost, raw=saved)
        return saved
