"""Metered heterogeneous native CPU H1 dynamics, with no engine imports.

The caller supplies a task whose frozen XML/native CPU runtime it has checked.
Every simulator call follows an Archive.save_npz and fsynced Archive.append.
This schema is dynamics-only: it does not supply scores, returns, or endpoint
solver states, and is separate from the homogeneous QueryExecutor raw schema.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np


SCHEMA = "bidirectional-heterogeneous-native-h1-v1"
_CLOCK_ATOL = 1e-12
_WARMSTART = "explicit-zero-solver-warmstart-intervention"


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


def _canonical(value):
    return json.dumps(_json(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _packed(value):
    return np.array(_canonical(value))


def _purpose(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A nonempty purpose is required")
    return value


@dataclass(frozen=True)
class ControlReservation:
    reservation_id: str
    requested_controls: int
    cumulative_requested_controls: int
    purpose: str


class SharedControlBudget:
    """One cap shared by heterogeneous calls and ordinary executor wrappers.

    reserve precedes any underlying executor call. The underlying Archive
    request remains the durable accounting authority; this object prevents
    overspend in the current process and never refunds failed reservations.
    A wrapper resolves each token exactly once with complete or error.
    """
    def __init__(self, max_requested_controls):
        if type(max_requested_controls) is not int or max_requested_controls < 0:
            raise ValueError("Budget cap must be a nonnegative integer")
        self.max_requested_controls = max_requested_controls
        self._requested = 0
        self._records = {}
        self._lock = threading.Lock()

    @property
    def requested_controls(self):
        with self._lock:
            return self._requested

    def reserve(self, controls, *, purpose):
        if type(controls) is not int or controls <= 0:
            raise ValueError("Reserved controls must be a positive integer")
        purpose = _purpose(purpose)
        with self._lock:
            if self._requested + controls > self.max_requested_controls:
                raise RuntimeError("Shared control budget exhausted before simulator call")
            self._requested += controls
            token = ControlReservation(f"b{len(self._records):09d}", controls,
                                       self._requested, purpose)
            self._records[token.reservation_id] = dict(token=token, status="pending")
            return token

    def _resolve(self, token, status, *, call_returned, attempted):
        if type(call_returned) is not bool or type(attempted) is not bool:
            raise ValueError("Explicit boolean call provenance is required")
        if call_returned and not attempted:
            raise ValueError("A returned call must have been attempted")
        with self._lock:
            record = self._records.get(getattr(token, "reservation_id", None))
            if record is None or record["token"] is not token:
                raise ValueError("Reservation belongs to a different shared budget")
            if record["status"] != "pending":
                raise ValueError("Reservation is already resolved")
            record.update(status=status, call_returned=call_returned, attempted=attempted)

    def complete(self, token):
        self._resolve(token, "complete", call_returned=True, attempted=True)

    def error(self, token, *, call_returned=False, attempted=True):
        self._resolve(token, "error", call_returned=call_returned, attempted=attempted)

    def totals(self):
        with self._lock:
            rows = list(self._records.values())
            count = lambda predicate: sum(r["token"].requested_controls for r in rows if predicate(r))
            return dict(max_requested_controls=self.max_requested_controls,
                requested_controls=self._requested, charged_requested_controls=self._requested,
                completed_controls=count(lambda r: r["status"] == "complete" or
                                         (r["status"] == "error" and r["call_returned"])),
                error_requested_controls=count(lambda r: r["status"] == "error"),
                unknown_error_controls=count(lambda r: r["status"] == "error" and
                                             r["attempted"] and not r["call_returned"]),
                unknown_controls=count(lambda r: r["status"] == "pending" or
                                       (r["status"] == "error" and r["attempted"] and not r["call_returned"])),
                not_attempted_error_controls=count(lambda r: r["status"] == "error" and not r["attempted"]),
                inflight_or_interrupted_controls=count(lambda r: r["status"] == "pending"),
                reservations=len(rows), completed_reservations=sum(r["status"] == "complete" for r in rows),
                error_reservations=sum(r["status"] == "error" for r in rows))


@dataclass(frozen=True)
class H1Evaluation:
    """Read-only next full states and durable references; no return semantics."""
    states: np.ndarray
    query_id: str
    inputs: dict
    result: dict


class HeterogeneousH1Executor:
    """Evaluate one native .01s CPU control from each independent full state.

    simulator, if supplied, has the env.batch_rollout signature and returns
    its (states[K,1,5], ignored_costs, ignored_sensors) tuple. Inputs always
    include explicit zero warmstarts[K,2]. The adapter never constructs an
    environment, changes an XML, computes scores, or recovers solver states.
    """
    def __init__(self, task, archive, *, budget, simulator=None,
                 scope="native_shooting", owner=None, source_fingerprints=None):
        if getattr(task, "backend", None) != "cpu":
            raise ValueError("Heterogeneous H1 requires the native CPU task")
        config = _json(task.config.to_dict())
        if (not isinstance(config, dict) or type(config.get("T")) is not int
                or config["T"] != 800 or type(config.get("dt")) not in (int, float)
                or config["dt"] != .01):
            raise ValueError("Heterogeneous H1 preserves T800 and dt .01")
        task_hash = getattr(task, "task_hash", None)
        if not isinstance(task_hash, str) or len(task_hash) != 64 or any(c not in "0123456789abcdef" for c in task_hash):
            raise ValueError("Frozen task SHA256 is required")
        if not isinstance(budget, SharedControlBudget):
            raise TypeError("A shared control budget is required")
        metadata_owner = _json({} if owner is None else owner)
        if not isinstance(metadata_owner, dict):
            raise ValueError("Owner must be a JSON object")
        fingerprints = _json({} if source_fingerprints is None else source_fingerprints)
        if not isinstance(fingerprints, dict) or any(not isinstance(v, str) or len(v) != 64
                or any(c not in "0123456789abcdef" for c in v) for v in fingerprints.values()):
            raise ValueError("Source fingerprints must map names to SHA256 strings")
        fingerprints = dict(fingerprints)
        mandatory = dict(task_hash=task_hash,
            task_config_sha256=hashlib.sha256(_canonical(config).encode()).hexdigest(),
            heterogeneous_executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if any(key in fingerprints and fingerprints[key] != value for key, value in mandatory.items()):
            raise ValueError("Conflicting task/executor source fingerprint")
        fingerprints.update(mandatory)
        if simulator is None:
            env = task.env
            if getattr(env, "_use_warp", False) or getattr(env, "_frame_skip", None) != 1:
                raise ValueError("Native CPU H1 requires frame_skip1 and no Warp")
            simulator = env.batch_rollout
        if not callable(simulator):
            raise TypeError("An injected simulator callable is required")
        self.task, self.archive, self.budget = task, archive, budget
        self._simulator = simulator
        self.scope, self.owner = _purpose(scope), metadata_owner
        self.config, self.task_hash, self.source_fingerprints = config, task_hash, fingerprints
        self.latest_query_id = None

    @property
    def ledger_totals(self):
        return self.archive.totals()

    def evaluate(self, states, actions, *, purpose, metadata=None):
        purpose = _purpose(purpose)
        metadata = _json({} if metadata is None else metadata)
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        state_values, action_values = np.asarray(states), np.asarray(actions)
        if state_values.dtype.kind not in "biuf" or action_values.dtype.kind not in "biuf":
            raise ValueError("Native states/actions must be real numeric arrays")
        states = np.array(state_values, dtype=np.float64, copy=True)
        actions = np.array(action_values, dtype=np.float64, copy=True)
        if (states.ndim != 2 or states.shape[1:] != (5,) or not 1 <= len(states) <= 8000
                or actions.shape != (len(states), 1) or not np.isfinite(states).all()
                or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1.)):
            raise ValueError("Finite states[K,5] with 1<=K<=8000 and bounded native actions[K,1] are required")
        clocks = states[:, 0]
        if np.any(clocks < -_CLOCK_ATOL) or np.any(clocks > 7.99+_CLOCK_ATOL):
            raise ValueError("Every H1 row must start before finite termination")
        elapsed = np.rint(clocks/.01).astype(np.int64)
        if np.any(elapsed < 0) or np.any(elapsed >= 800) or not np.allclose(clocks, elapsed*.01, rtol=0, atol=_CLOCK_ATOL):
            raise ValueError("Per-row native clock must match integer elapsed at .01")
        count = len(states)
        action_sequences = actions[:, None, :].copy()
        warmstarts = np.zeros((count, 2), dtype=np.float64)
        token = self.budget.reserve(count, purpose=purpose)
        requested = attempted = returned = False
        identifier = None
        started = time.perf_counter()
        try:
            identifier = self.archive.next_query_id()
            self.latest_query_id = identifier
            common = dict(schema=np.array(SCHEMA), task_hash=np.array(self.task_hash),
                task_json=_packed(self.config), source_fingerprints_json=_packed(self.source_fingerprints),
                owner_json=_packed(self.owner), metadata_json=_packed(metadata), purpose=np.array(purpose),
                dynamics_only=np.array(True), returns_semantics=np.array("none"),
                initial_states=states, initial_elapsed=elapsed, initial_warmstarts=warmstarts,
                initial_warmstart_exact=np.zeros(count, dtype=bool),
                initial_restart_semantics=np.array(_WARMSTART), actions=action_sequences)
            inputs = self.archive.save_npz(f"queries/{identifier}_heterogeneous_inputs.npz", **common)
            self.archive.append("query_requested", query_id=identifier, schema=SCHEMA,
                query_kind="heterogeneous_native_h1", mode="model_heterogeneous", scope=self.scope,
                owner=self.owner, metadata=metadata, purpose=purpose, task_hash=self.task_hash,
                source_fingerprints=self.source_fingerprints, rows=count, horizon=1,
                initial_elapsed=elapsed.tolist(), dynamics_only=True, returns_semantics="none",
                requested_controls=count, requested_physics_controls=count,
                requested_physics_frames=count, cutoff_skipped_controls=0,
                budget_reservation_id=token.reservation_id,
                budget_requested_cumulative=token.cumulative_requested_controls, inputs=inputs)
            requested = True
            supplied_states, supplied_actions, supplied_warmstarts = states.copy(), action_sequences.copy(), warmstarts.copy()
            attempted = True
            result = self._simulator(supplied_states, supplied_actions, initial_warmstart=supplied_warmstarts)
            returned = True
            if (not isinstance(result, (tuple, list)) or len(result) != 3
                    or not np.array_equal(supplied_states, states)
                    or not np.array_equal(supplied_actions, action_sequences)
                    or not np.array_equal(supplied_warmstarts, warmstarts)):
                raise ValueError("Simulator tuple/input immutability contract differs")
            output_values = np.asarray(result[0])
            if output_values.dtype.kind not in "biuf":
                raise ValueError("Simulator must return real numeric native states")
            next_states = np.array(output_values, dtype=np.float64, copy=True)
            if next_states.shape != (count, 1, 5) or not np.isfinite(next_states).all():
                raise ValueError("Simulator must return finite states[K,1,5]")
            if not np.allclose(next_states[:, 0, 0], clocks+.01, rtol=0, atol=_CLOCK_ATOL):
                raise ValueError("Simulator H1 output clock differs from each native input clock")
            raw = self.archive.save_npz(f"queries/{identifier}_heterogeneous_result.npz", **common,
                prestates=states[:, None, :], states=next_states, elapsed=elapsed+1,
                input_sha256=np.array(inputs["sha256"]),
                endpoint_warmstarts=np.zeros((count, 2)), endpoint_warmstart_exact=np.zeros(count, dtype=bool),
                endpoint_restart_semantics=np.array(_WARMSTART),
                backend_json=_packed(dict(backend="mujoco_cpu", frame_skip=1, horizon=1,
                                         timestep=.01, integrator="RK4", heterogeneous_initial_states=True)))
            self.archive.append("query_complete", query_id=identifier, schema=SCHEMA,
                scope=self.scope, owner=self.owner, purpose=purpose, dynamics_only=True,
                returns_semantics="none", backend="mujoco_cpu", completed_controls=count,
                completed_physics_frames=count, query_wall_seconds=time.perf_counter()-started,
                budget_reservation_id=token.reservation_id, result=raw)
        except BaseException as error:
            self.budget.error(token, call_returned=returned, attempted=attempted)
            if requested:
                try:
                    self.archive.append("query_error", query_id=identifier, schema=SCHEMA,
                        scope=self.scope, purpose=purpose, call_started=attempted, call_returned=returned,
                        completed_controls=count if returned else 0, completed_physics_frames=count if returned else 0,
                        partial_physics_unknown=attempted and not returned, unknown_partial=attempted and not returned,
                        error_type=type(error).__name__, error=str(error),
                        query_wall_seconds=time.perf_counter()-started, budget_reservation_id=token.reservation_id)
                except BaseException as evidence_error:
                    error.add_note("Could not durably append heterogeneous query error: "+type(evidence_error).__name__)
            raise
        self.budget.complete(token)
        output = next_states[:, 0].copy()
        output.setflags(write=False)
        return H1Evaluation(output, identifier, inputs, raw)
