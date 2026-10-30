"""Append-only families of complete, actually executed Acrobot teachers.

This is a provenance/alignment gate, not an independent physics audit. Query
references and the original teacher ownership are retained; a downstream raw
evidence audit must establish that those references name real live controls.
Neither hypothetical planner banks nor analytic reference controllers qualify.
Failures stay in the index even though successful teachers alone supply the
optional auxiliary action loss. A family is fixed at ingestion, using the
recorded shoulder unwrap at the first measured approach to the target.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from .evidence import EpisodeTrace, file_sha256
from .task import Snapshot, TaskConfig, observation, post_action_tip, trajectory_family


FAMILY_IDS = {"positive": 0, "negative": 1}
SCHEMA = "bidirectional-family-dataset-v1"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reference_marker(value):
    if isinstance(value, dict):
        if value.get("excluded_from_teacher_aux") is True:
            return True
        if any(value.get(key) == "analytic_reference" for key in ("controller", "semantics", "source_type")):
            return True
        return any(_reference_marker(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_reference_marker(item) for item in value)
    return False


def teacher_eligibility(trace, *, independent_only=False):
    """Classify provenance only; malformed eligible arrays still raise later."""
    if not isinstance(trace, EpisodeTrace):
        return False, "requires an EpisodeTrace"
    if trace.execution_semantics != "executed_replanning":
        return False, "only actually executed replanning teachers are eligible"
    owner = trace.actor_owner
    if not isinstance(owner, dict) or owner.get("semantics") != "executed_replanning":
        return False, "teacher controller ownership is missing or hypothetical"
    if _reference_marker(owner) or _reference_marker(trace.source):
        return False, "analytic reference is excluded from teacher supervision"
    if not trace.complete or trace.initial_snapshot.elapsed != 0:
        return False, "teacher must be a complete ordinary-root episode"
    if independent_only and owner.get("actor_hash") is not None:
        return False, "teacher is guided rather than independent"
    return True, "eligible"


def _trace_digest(trace):
    digest = sha256()
    for name, value in sorted(trace.arrays().items()):
        array = np.ascontiguousarray(value)
        digest.update(_canonical([name, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_teacher(trace, task_config):
    """Validate boundary/pre-action alignment and recompute its fixed family."""
    eligible, reason = teacher_eligibility(trace)
    if not eligible:
        raise ValueError(reason)
    config = task_config
    count = len(trace.actions)
    if trace.T != config.T or trace.initial_snapshot.task_hash != config.task_hash or count != config.T:
        raise ValueError("Complete teacher must match the exact dataset task and T")
    shapes = {"states": (count+1, 5), "warmstarts": (count+1, 2),
        "elapsed": (count+1,), "hold_counts": (count+1,),
        "observations": (count+1, 8), "actions": (count, 1),
        "stage_costs": (count,), "terminal_costs": (count,),
        "capture_failures": (count,), "component_costs": (count, 4),
        "tip_distances": (count+1,)}
    for name, shape in shapes.items():
        array = np.asarray(getattr(trace, name))
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Teacher {name} must be finite and aligned as {shape}")
    if not np.array_equal(trace.elapsed, np.arange(count+1)):
        raise ValueError("Teacher elapsed boundaries must cover 0 through T consecutively")
    if not np.allclose(trace.states[:, 0], config.dt*trace.elapsed, atol=1e-10, rtol=0):
        raise ValueError("Teacher physical clock disagrees with elapsed controls")
    if not np.array_equal(trace.states[0], trace.initial_snapshot.state) or not np.array_equal(
            trace.warmstarts[0], trace.initial_snapshot.warmstart):
        raise ValueError("Teacher initial snapshot is not its first saved boundary")
    if np.max(abs(trace.actions)) > 1+1e-7:
        raise ValueError("Teacher controls must be actual bounded normalized actions")
    if len(trace.query_ids) != count or len(set(trace.query_ids)) != count or any(
            not isinstance(ref, str) or not ref for ref in trace.query_ids):
        raise ValueError("Every executed teacher control needs a distinct nonempty query reference")
    owner = trace.actor_owner
    if not isinstance(owner.get("controller_hash"), str) or not owner["controller_hash"]:
        raise ValueError("Teacher requires the executed controller identity")
    if owner.get("actor_hash") is not None and not isinstance(owner["actor_hash"], str):
        raise ValueError("Guided teacher actor identity must be explicit")
    if not np.allclose(trace.component_costs.sum(axis=1), trace.stage_costs, atol=1e-9, rtol=1e-9):
        raise ValueError("Teacher stage costs disagree with their raw components")
    if np.any(trace.terminal_costs[:-1]) or np.any(trace.capture_failures[:-1]):
        raise ValueError("True ending cost/failure may occur only on the last actual control")
    tip = post_action_tip(trace.states)
    distances = np.linalg.norm(tip-np.array([0., 0., 4.]), axis=1)
    if not np.allclose(distances, trace.tip_distances, atol=1e-9, rtol=1e-9):
        raise ValueError("Teacher approach distances disagree with saved physical states")
    speeds = np.linalg.norm(trace.states[:, 3:5], axis=1)
    holds = [0]
    for distance, speed in zip(distances[1:], speeds[1:]):
        holds.append(min(holds[-1]+1, config.capture_steps) if
            distance <= config.capture_distance and speed <= config.capture_qvel_norm else 0)
    if trace.initial_snapshot.hold_count != 0 or not np.array_equal(trace.hold_counts, holds):
        raise ValueError("Teacher hold counts disagree with actual post-action capture boundaries")
    captured = holds[-1] >= config.capture_steps
    if bool(trace.captured) != captured or not np.isclose(trace.capture_failures[-1],
            0. if captured else config.failure_penalty, atol=1e-9, rtol=0):
        raise ValueError("Teacher ending flag/failure charge disagrees with the complete measured ending")
    expected = np.stack([observation(Snapshot(trace.states[t], trace.warmstarts[t], t,
        int(holds[t]), config.task_hash, trace.initial_snapshot.context_id), config) for t in range(count+1)])
    if not np.allclose(expected, trace.observations, atol=1e-9, rtol=1e-9):
        raise ValueError("Teacher observations must align with saved pre-action boundaries")
    family = trajectory_family(trace.states, distances, approach_distance=config.local_distance)
    if trace.family != family:
        raise ValueError("Recorded family disagrees with shoulder unwrap at first approach")
    return family


def _immutable_trace(trace):
    arrays = {}
    for name in ("states", "warmstarts", "elapsed", "hold_counts", "observations", "actions",
            "stage_costs", "terminal_costs", "capture_failures", "component_costs", "tip_distances"):
        array = np.ascontiguousarray(getattr(trace, name))
        arrays[name] = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    # Ownership/source dictionaries are freshly loaded too; mutating them does
    # not change the stored raw episode or its fixed append-only index entry.
    return replace(trace, **arrays)


class FamilyDataset:
    """Single-file append records with immutable raw episodes and a hash chain.

    A POSIX lock serializes appenders. Failed writes leave an unindexed raw
    orphan, never an implicitly eligible episode; existing files are untouched.
    Reading an entry rechecks its raw hash and all alignment invariants.
    """
    def __init__(self, path, task_config=None, *, create=True):
        self.path = Path(path)
        supplied = task_config
        if isinstance(supplied, dict):
            supplied = TaskConfig.from_dict(supplied)
        if create:
            if not isinstance(supplied, TaskConfig):
                raise TypeError("Creating a family dataset requires its explicit TaskConfig")
            self.path.mkdir(parents=True, exist_ok=False)
            (self.path/"episodes").mkdir()
            packet = dict(schema=SCHEMA, task=supplied.to_dict(), task_hash=supplied.task_hash,
                family_rule="unwrap shoulder, displacement at first measured local-distance approach",
                family_ids=FAMILY_IDS, includes_complete_failures=True,
                hypothetical_plans_allowed=False, analytic_reference_allowed=False)
            with (self.path/"dataset.json").open("x") as stream:
                stream.write(_canonical(packet)+"\n");stream.flush();os.fsync(stream.fileno())
            (self.path/"index.jsonl").open("x").close()
        packet = json.loads((self.path/"dataset.json").read_text())
        if packet.get("schema") != SCHEMA or packet.get("family_ids") != FAMILY_IDS:
            raise ValueError("Unsupported or modified family dataset manifest")
        self.config = TaskConfig.from_dict(packet["task"])
        if packet["task_hash"] != self.config.task_hash or supplied is not None and supplied != self.config:
            raise ValueError("Family dataset task/model identity mismatch")
        self.manifest_sha256 = file_sha256(self.path/"dataset.json")
        self._entries = self._read_index()

    def _read_index(self):
        entries = [];previous = None;identifiers=set();digests=set()
        for line in (self.path/"index.jsonl").read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise ValueError("Incomplete append remains charged/provisional; dataset index requires recovery outside this API")
            row = json.loads(line);digest = row.pop("entry_sha256")
            if (digest != sha256(_canonical(row).encode()).hexdigest() or row["previous_entry_sha256"] != previous
                    or row["sequence"] != len(entries) or row["manifest_sha256"] != self.manifest_sha256):
                raise ValueError("Family dataset append chain or task manifest was modified")
            if row["episode_id"] in identifiers or row["trace_digest"] in digests:
                raise ValueError("Duplicate teacher episode/experience cannot count as independent acquisition")
            identifiers.add(row["episode_id"]);digests.add(row["trace_digest"])
            row["entry_sha256"] = digest;entries.append(row);previous=digest
        return entries

    @property
    def entries(self):
        return json.loads(_canonical(self._entries))

    def append(self, trace, episode_id, *, source_path=None):
        family = validate_teacher(trace, self.config)
        if not isinstance(episode_id, str) or not episode_id or any(c not in
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in episode_id):
            raise ValueError("Teacher episode IDs must be safe path components")
        origin = None
        if source_path is not None:
            source_path = Path(source_path)
            imported = EpisodeTrace.from_npz(source_path)
            if _trace_digest(imported) != _trace_digest(trace):
                raise ValueError("Imported teacher source differs from the appended actual trace")
            origin = dict(path=str(source_path.resolve()), sha256=file_sha256(source_path), bytes=source_path.stat().st_size)
        import fcntl
        with (self.path/"append.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._entries = self._read_index()
            if file_sha256(self.path/"dataset.json") != self.manifest_sha256:
                raise ValueError("Dataset manifest changed before append")
            semantic_digest = _trace_digest(trace)
            if any(row["episode_id"] == episode_id or row["trace_digest"] == semantic_digest for row in self._entries):
                raise ValueError("Teacher episode/experience was already appended")
            relative = f"episodes/{episode_id}.npz";target = self.path/relative
            trace.to_npz(target)
            row = dict(sequence=len(self._entries), episode_id=episode_id, file=relative,
                sha256=file_sha256(target), bytes=target.stat().st_size, trace_digest=semantic_digest,
                manifest_sha256=self.manifest_sha256,
                previous_entry_sha256=self._entries[-1]["entry_sha256"] if self._entries else None,
                family=family, family_id=FAMILY_IDS.get(family), successful=bool(trace.successful),
                captured=bool(trace.captured), complete=True, controls=len(trace.actions),
                total_cost=trace.total_cost, context_id=trace.initial_snapshot.context_id,
                task_hash=trace.initial_snapshot.task_hash, actor_owner=trace.actor_owner,
                teacher_kind="independent" if trace.actor_owner.get("actor_hash") is None else "guided_executed",
                execution_semantics=trace.execution_semantics, source=trace.source,
                query_ids=list(trace.query_ids), imported_from=origin)
            row["entry_sha256"] = sha256(_canonical(row).encode()).hexdigest()
            with (self.path/"index.jsonl").open("a") as stream:
                stream.write(_canonical(row)+"\n");stream.flush();os.fsync(stream.fileno())
            self._entries.append(row)
            return json.loads(_canonical(row))

    def traces(self, *, family=None, successful_only=False, independent_only=False):
        if isinstance(family, int):
            if family not in FAMILY_IDS.values():raise ValueError("Unknown fixed actor family")
            family = next(name for name, value in FAMILY_IDS.items() if value == family)
        for row in self._entries:
            if family is not None and row["family"] != family:continue
            if successful_only and not row["successful"]:continue
            if independent_only and row["teacher_kind"] != "independent":continue
            relative=Path(row["file"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "episodes":
                raise ValueError("Unsafe raw teacher path in dataset index")
            path=self.path/relative
            if path.stat().st_size != row["bytes"] or file_sha256(path) != row["sha256"]:
                raise ValueError("Immutable raw teacher bytes changed")
            trace=EpisodeTrace.from_npz(path)
            if validate_teacher(trace,self.config) != row["family"] or _trace_digest(trace) != row["trace_digest"]:
                raise ValueError("Indexed teacher family or contents changed")
            yield _immutable_trace(trace)

    def auxiliary(self, family=None, *, independent_only=False):
        """Fixed-family bounded means target ACTUAL saved pre-action controls."""
        from .learning import AuxiliaryBatch
        traces=[trace for trace in self.traces(family=family,successful_only=True,
            independent_only=independent_only) if trace.family in FAMILY_IDS]
        if not traces:return None
        return AuxiliaryBatch(np.concatenate([trace.observations[:-1] for trace in traces]),
            np.concatenate([trace.actions for trace in traces]),
            np.concatenate([np.full(len(trace.actions),FAMILY_IDS[trace.family],dtype=np.int64) for trace in traces]),
            source="executed_teacher")

    def ingest_tree(self, root, *, independent_only=False):
        """Import eligible executed episodes; disclose every excluded raw file."""
        imported=[];excluded=[]
        for path in sorted(Path(root).rglob("episodes/**/*.npz")):
            trace=EpisodeTrace.from_npz(path)
            eligible,reason=teacher_eligibility(trace,independent_only=independent_only)
            if not eligible:
                excluded.append(dict(path=str(path),sha256=file_sha256(path),reason=reason));continue
            identifier=f"import_{len(self._entries):06d}_{sha256(str(path.resolve()).encode()).hexdigest()[:12]}"
            imported.append(self.append(trace,identifier,source_path=path))
        return dict(imported=imported,excluded=excluded,new_simulator_controls=0,new_actor_inference_rows=0,new_fits=0)

