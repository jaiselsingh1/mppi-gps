"""Independent bounded readers and arithmetic for the H5.0 evidence stream.

No worker, planner, policy, simulator or production evidence decoder is imported.
Torch is imported only by the explicit saved-prediction verifier. These helpers
do not launch fitting or native replay. A structural stream check alone is not
the complete scientific round audit or a successful-loop certificate.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

FORMAT = "walker-array-stream-v1"
POLICY_SCHEMA = "walker-tanh-policy-v1"


class AuditError(RuntimeError):
    """Saved evidence or its explicit audit reservation is insufficient."""


class AuditJournal:
    """Bounded durable paid-audit request/completion evidence."""

    def __init__(self, path: Path, *, max_bytes: int, max_line_bytes: int = 1024):
        self.stream = Path(path).open("xb", buffering=0)
        self.max_bytes, self.max_line_bytes = max_bytes, max_line_bytes
        self.bytes = self.events = 0

    def record(self, event: dict[str, Any]) -> None:
        payload = _canonical(dict(event, id=self.events)) + b"\n"
        _need(
            len(payload) <= self.max_line_bytes and self.bytes + len(payload) <= self.max_bytes,
            "Audit journal storage exhausted before paid boundary",
        )
        remaining = memoryview(payload)
        while remaining:
            count = self.stream.write(remaining)
            _need(bool(count), "Audit journal write failed; retain partial evidence")
            self.bytes += count
            remaining = remaining[count:]
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.events += 1

    def close(self) -> None:
        self.stream.close()


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _memory_boundary() -> int:
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024
    _need(peak <= 4 * 1024**3, "Audit peak-RSS reservation exhausted")
    return peak


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    _need(type(value) is int and value >= minimum, "Invalid integer: " + label)
    return value


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        _need(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(payload: str | bytes) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AuditError("Nonfinite JSON " + value)
            ),
        )
    except (ValueError, UnicodeDecodeError) as error:
        raise AuditError("Malformed saved JSON") from error


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _reference_hash(reference: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(reference)).hexdigest()


def _identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class StreamArchive:
    """Stream/hash the append-only index, with a bounded external lookup index.

    SQLite stores only group IDs and reference digests; original array metadata
    and payloads remain in the immutable source stream. The disposable database
    lives outside that source and has an explicit page-count ceiling. It is not
    a new scientific representation or an HDF5/compression allowance.
    """

    def __init__(
        self,
        root: Path,
        index_database: Path,
        *,
        max_read_bytes: int,
        max_database_bytes: int = 1024**3,
        max_array_bytes: int = 16 * 1024**2,
        max_metadata_bytes: int = 16384,
        max_arrays_per_group: int = 32,
        deadline: float | None = None,
    ):
        self.root, self.database_path = Path(root).resolve(), Path(index_database).resolve()
        _need(
            not self.database_path.is_relative_to(self.root), "Audit index must be outside evidence"
        )
        _need(not self.database_path.exists(), "Audit index is create-only")
        self.max_read_bytes = _integer(max_read_bytes, "max_read_bytes", 1)
        self.deadline = deadline
        self.max_array_bytes = _integer(max_array_bytes, "max_array_bytes", 1)
        self.max_metadata_bytes = _integer(max_metadata_bytes, "max_metadata_bytes", 1)
        self.max_arrays_per_group = _integer(max_arrays_per_group, "max_arrays_per_group", 1)
        database_bytes = _integer(max_database_bytes, "max_database_bytes", 4096)
        self.read_bytes = self.array_checks = self.group_checks = self.ledger_lines = 0
        self.file_hashes: dict[str, str] = {}
        self._paths = {
            name: self.root / name for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl")
        }
        for path in self._paths.values():
            _need(
                path.is_file() and path.resolve().parent == self.root, "Missing or escaped stream"
            )
        self._identities = {name: _identity(path) for name, path in self._paths.items()}
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.database_path)
        self._db.execute("PRAGMA page_size=4096")
        self._db.execute("PRAGMA journal_mode=OFF")
        self._db.execute("PRAGMA cache_size=-8192")
        self._db.execute("PRAGMA max_page_count=" + str(database_bytes // 4096))
        self._db.execute("CREATE TABLE groups (name TEXT PRIMARY KEY, digest TEXT NOT NULL)")
        self._verified = False

    def _read(self, stream: Any, length: int) -> bytes:
        _memory_boundary()
        _need(
            self.deadline is None or time.monotonic() <= self.deadline,
            "Audit wall-time boundary exhausted",
        )
        _integer(length, "read length")
        _need(self.read_bytes + length <= self.max_read_bytes, "Audit source-read budget exhausted")
        payload = stream.read(length)
        self.read_bytes += len(payload)
        _need(len(payload) == length, "Truncated array payload")
        return payload

    def _lines(self, path: Path) -> Iterator[dict[str, Any]]:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                _need(
                    self.deadline is None or time.monotonic() <= self.deadline,
                    "Audit wall-time boundary exhausted",
                )
                remaining = self.max_read_bytes - self.read_bytes
                _need(remaining > 0, "Audit source-read budget exhausted before metadata read")
                payload = stream.readline(min(self.max_metadata_bytes + 1, remaining))
                self.read_bytes += len(payload)
                if not payload:
                    self.file_hashes[path.name] = digest.hexdigest()
                    return
                digest.update(payload)
                _need(len(payload) <= self.max_metadata_bytes, "Metadata line exceeds frozen limit")
                _need(payload.endswith(b"\n"), "Truncated metadata line or source-read budget")
                value = _json(payload)
                _need(isinstance(value, dict), "Metadata line must be an object")
                yield value

    def _shape(self, value: Any) -> tuple[np.dtype, tuple[int, ...], int]:
        _need(isinstance(value, dict), "Array binding must be an object")
        _need(
            set(value) == {"sha256", "shape", "dtype", "encoding", "offset", "byte_length"},
            "Array binding roster differs",
        )
        offset, length = (
            _integer(value["offset"], "offset"),
            _integer(value["byte_length"], "length"),
        )
        _need(length <= self.max_array_bytes, "Array exceeds its frozen payload bound")
        _need(re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None, "Invalid C-byte hash")
        shape = value["shape"]
        _need(isinstance(shape, list) and len(shape) <= 8, "Invalid bounded array rank")
        for size in shape:
            _integer(size, "shape dimension")
        try:
            dtype = np.dtype(value["dtype"])
        except (TypeError, ValueError) as error:
            raise AuditError("Invalid explicit array dtype") from error
        _need(
            not dtype.hasobject
            and not dtype.fields
            and dtype.kind in "biufcSU"
            and dtype.itemsize > 0,
            "Object/structured/implicit array dtype is forbidden",
        )
        _need(math.prod(shape) * dtype.itemsize == length, "Array shape/dtype byte length differs")
        _need(value["encoding"] in ("array", "json"), "Unknown array encoding")
        _need(offset + length <= self._identities["arrays.bin"][2], "Array range escapes payload")
        return dtype, tuple(shape), length

    def verify_index(self) -> dict[str, int]:
        _need(not self._verified, "Index verification is single-use")
        lines = iter(self._lines(self._paths["arrays.index.jsonl"]))
        expected_offset = 0
        payload_digest = hashlib.sha256()
        with self._paths["arrays.bin"].open("rb") as payload:
            for intent in lines:
                _need(
                    intent.get("event") == "arrays_requested", "Array intent must precede payload"
                )
                reference = {key: value for key, value in intent.items() if key != "event"}
                _need(
                    set(reference) == {"format", "file", "index", "group", "tag", "arrays"},
                    "Array reference roster differs",
                )
                self._reference_header(reference)
                match = re.fullmatch(
                    r"arrays/a([0-9]{9})_[A-Za-z0-9_.-]{1,100}", reference["group"]
                )
                _need(
                    match is not None and int(match[1]) == self.group_checks,
                    "Array group order differs",
                )
                arrays = reference["arrays"]
                _need(
                    isinstance(arrays, dict) and len(arrays) <= self.max_arrays_per_group,
                    "Group array bound",
                )
                # Canonical JSON sorts object keys; payload order is explicitly
                # bound by byte offsets, including zero-byte arrays at a boundary.
                ordered = sorted(
                    arrays.items(),
                    key=lambda item: (
                        item[1].get("offset", -1),
                        bool(item[1].get("byte_length", 0)),
                    ),
                )
                for name, binding in ordered:
                    _need(
                        isinstance(name, str) and bool(name) and "/" not in name,
                        "Unsafe array name",
                    )
                    _, _, length = self._shape(binding)
                    _need(
                        binding["offset"] == expected_offset,
                        "Payload gaps, overlap or reordered arrays",
                    )
                    digest = hashlib.sha256()
                    remaining = length
                    while remaining:
                        chunk = self._read(payload, min(1024**2, remaining))
                        digest.update(chunk)
                        payload_digest.update(chunk)
                        remaining -= len(chunk)
                    _need(digest.hexdigest() == binding["sha256"], "Array C-byte hash differs")
                    expected_offset += length
                    self.array_checks += 1
                try:
                    complete = next(lines)
                except StopIteration as error:
                    raise AuditError("Array group has unclosed intent") from error
                _need(
                    complete
                    == dict(
                        event="arrays_complete",
                        format=FORMAT,
                        group=reference["group"],
                        end_offset=expected_offset,
                    ),
                    "Array completion differs from paid payload intent",
                )
                self._db.execute(
                    "INSERT INTO groups VALUES (?,?)",
                    (reference["group"], _reference_hash(reference)),
                )
                self.group_checks += 1
                if self.group_checks % 1000 == 0:
                    self._db.commit()
        self._db.commit()
        _need(
            expected_offset == self._identities["arrays.bin"][2],
            "Unindexed trailing or missing payload bytes",
        )
        self._verified = True
        self.file_hashes["arrays.bin"] = payload_digest.hexdigest()
        self.verify_unchanged()
        return dict(
            groups=self.group_checks, arrays=self.array_checks, source_read_bytes=self.read_bytes
        )

    def _reference_header(self, reference: Mapping[str, Any]) -> None:
        _need(
            reference.get("format") == FORMAT
            and reference.get("file") == "arrays.bin"
            and reference.get("index") == "arrays.index.jsonl",
            "Unsupported or escaped evidence reference",
        )

    def verify_reference(self, reference: Mapping[str, Any]) -> None:
        _need(self._verified, "Verify the source index before decoding referenced arrays")
        self._reference_header(reference)
        row = self._db.execute(
            "SELECT digest FROM groups WHERE name=?", (reference["group"],)
        ).fetchone()
        _need(
            row is not None and row[0] == _reference_hash(reference),
            "Ledger reference differs from committed index",
        )

    def array(self, reference: Mapping[str, Any], key: str) -> np.ndarray:
        self.verify_reference(reference)
        _need(key in reference["arrays"], "Missing referenced evidence array: " + key)
        binding = reference["arrays"][key]
        dtype, shape, length = self._shape(binding)
        with self._paths["arrays.bin"].open("rb") as stream:
            stream.seek(binding["offset"])
            payload = self._read(stream, length)
        _need(
            hashlib.sha256(payload).hexdigest() == binding["sha256"],
            "Array changed after index audit",
        )
        return np.frombuffer(payload, dtype=dtype).reshape(shape)

    def json_array(self, reference: Mapping[str, Any], key: str) -> Any:
        value = self.array(reference, key)
        _need(value.shape == () and value.dtype.kind in "SU", "Expected scalar JSON encoding")
        return _json(value.item())

    def ledger(self) -> Iterator[dict[str, Any]]:
        _need(self._verified, "Verify index before ledger ownership")
        for ordinal, event in enumerate(self._lines(self._paths["ledger.jsonl"])):
            _need(event.get("id") == f"event-{ordinal:09d}", "Ledger event sequence differs")
            _need(event.get("kind") == event.get("event"), "Ledger kind/event differs")
            self.ledger_lines += 1
            yield event

    def verify_unchanged(self) -> None:
        for name, expected in self._identities.items():
            _need(_identity(self._paths[name]) == expected, "Evidence stream changed during audit")

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> StreamArchive:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def native_observations(initial_state: np.ndarray, posts: np.ndarray) -> np.ndarray:
    """Independent reconstruction of native clipped17 pre-action observations."""
    initial, posts = np.asarray(initial_state), np.asarray(posts)
    _need(posts.ndim == 3 and posts.shape[-1] == 19 and posts.shape[1] > 0, "Native rollout shape")
    if initial.shape == (19,):
        initial = np.broadcast_to(initial, (len(posts), 19))
    _need(initial.shape == (len(posts), 19), "Native initial-state ownership")
    pres = np.concatenate((initial[:, None], posts[:, :-1]), axis=1)
    return np.concatenate((pres[..., 2:10], np.clip(pres[..., 10:19], -10, 10)), axis=-1)


def healthy_prefix(posts: np.ndarray) -> np.ndarray:
    _need(posts.ndim == 3 and posts.shape[-1] == 19, "Native states for prefix")
    healthy = (
        (posts[..., 2] > 0.8) & (posts[..., 2] < 2) & (posts[..., 3] > -1) & (posts[..., 3] < 1)
    )
    active = np.ones(healthy.shape, dtype=bool)
    if posts.shape[1] > 1:
        active[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
    return active


def native_cost(posts: np.ndarray, actions: np.ndarray, task: Mapping[str, Any]) -> np.ndarray:
    """Recompute stored native horizon cost; zero integration calls."""
    posts, actions = np.asarray(posts), np.asarray(actions)
    _need(
        posts.shape[:2] == actions.shape[:2] and actions.shape[-1] == 6,
        "Native controls/state join",
    )
    active = healthy_prefix(posts)
    z, pitch, vx = posts[..., 2], posts[..., 3], posts[..., 10]
    healthy = (z > 0.8) & (z < 2) & (pitch > -1) & (pitch < 1)
    running = (
        task["vel_cost_weight"] * np.abs(task["target_velocity"] - vx)
        + task["vertical_velocity_cost_weight"] * posts[..., 11] ** 2
        + task["angle_cost_weight"] * pitch**2
        + task["unhealthy_cost_weight"] * (~healthy)
        + task["ctrl_cost_weight"] * np.sum(actions**2, axis=-1)
    )
    costs = np.where(active, running, 0.0).sum(axis=1)
    costs += task["unhealthy_cost_weight"] * np.sum(~active, axis=1)
    fell = np.any(~healthy, axis=1)
    if task["apply_terminal_cost_on_done"]:
        costs += task["terminal_unhealthy_weight"] * fell.astype(float)
    valid = np.all(np.isfinite(posts), axis=-1) & np.all(np.isfinite(actions), axis=-1)
    return np.where(np.all(valid | ~active, axis=1), costs, np.inf)


def agreement_values(
    observations: np.ndarray,
    actions: np.ndarray,
    posts: np.ndarray,
    costs: np.ndarray,
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen null-radius D, including the fall-causing control and finite J."""
    support = healthy_prefix(posts) & np.all(np.isfinite(observations), axis=-1)
    support &= np.isfinite(costs)[:, None]
    _need(predictions.shape == actions[support].shape, "Prediction/support row binding differs")
    _need(np.isfinite(predictions).all(), "Nonfinite recorded policy predictions")
    values = np.zeros(actions.shape[:2], dtype=float)
    values[support] = np.sum((actions[support] - predictions) ** 2, axis=-1)
    return np.sum(values, axis=1), support


def plan_digest(*arrays: np.ndarray | None) -> str:
    """Independently preserve the existing search-array digest contract."""
    digest = hashlib.sha256()
    for value in arrays:
        if value is None:
            digest.update(b"None")
        else:
            value = np.ascontiguousarray(value)
            digest.update(str(value.dtype).encode())
            digest.update(json.dumps(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


class SavedActorVerifier:
    """Exact saved CPU float32 network replica, with independent row accounting."""

    def __init__(
        self,
        registry: Mapping[str, Mapping[str, Any]],
        run: Path,
        *,
        max_neural_rows: int,
        max_prediction_calls: int = 564270,
        max_scope_neural_rows: Mapping[str, int] | None = None,
        deadline: float | None = None,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.registry, self.run = dict(registry), Path(run).resolve()
        self.max_neural_rows = _integer(max_neural_rows, "audit neural rows")
        self.max_prediction_calls = _integer(max_prediction_calls, "audit prediction calls")
        self.max_scope_neural_rows = (
            None
            if max_scope_neural_rows is None
            else {
                name: _integer(limit, "audit scope neural rows")
                for name, limit in max_scope_neural_rows.items()
            }
        )
        self.scope: str | None = None
        self.scope_neural_rows: dict[str, int] = {}
        self.scope_completed_neural_rows: dict[str, int] = {}
        self.scope_unknown_partial_neural_rows: dict[str, int] = {}
        self.deadline, self.event_sink = deadline, event_sink
        self.completed_neural_rows = self.unknown_partial_neural_rows = 0
        self.neural_rows = self.prediction_calls = self.checkpoint_loads = (
            self.checkpoint_read_bytes
        ) = 0
        _need(len(self.registry) <= 15, "Actor registry exceeds the fixed round-one roster")
        self._actors: dict[str, Mapping[str, Any]] = {}
        self.payloads: dict[str, Mapping[str, Any]] = {}
        self.record_payloads: dict[str, Mapping[str, Any]] = {}
        self._record_bindings: dict[str, tuple[int, str]] = {}

    def _load(self, owner: str) -> Mapping[str, Any]:
        if owner in self._actors:
            return self._actors[owner]
        _need(owner in self.registry, "Prediction actor is absent from registry")
        binding = self.registry[owner]
        payload = self._decode(owner, binding)
        state = payload["state_dict"]
        self._actors[owner] = state
        self.payloads[owner] = payload
        return state

    def _decode(self, owner: str, binding: Mapping[str, Any]) -> Mapping[str, Any]:
        if binding["file"] in self.record_payloads:
            payload = self.record_payloads[binding["file"]]
            _need(payload["policy_hash"] == owner, "Checkpoint file has conflicting actor owners")
            _need(
                self._record_bindings[binding["file"]] == (binding["bytes"], binding["sha256"]),
                "Checkpoint has conflicting byte bindings",
            )
            return payload
        _need(self.checkpoint_loads < 15, "Audit checkpoint-load reservation exhausted")
        path = (self.run / binding["file"]).resolve()
        _need(path.is_relative_to(self.run) and path.is_file(), "Escaped or missing checkpoint")
        _need(
            path.stat().st_size == binding["bytes"] and path.stat().st_size <= 16 * 1024**2,
            "Checkpoint size binding",
        )
        data = path.read_bytes()
        self.checkpoint_read_bytes += len(data)
        _need(hashlib.sha256(data).hexdigest() == binding["sha256"], "Checkpoint byte binding")
        import torch

        torch.set_num_threads(1)
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        self.checkpoint_loads += 1
        _need(
            payload.get("schema") == POLICY_SCHEMA and payload.get("policy_hash") == owner,
            "Checkpoint actor ownership",
        )
        state = payload["state_dict"]
        expected = {
            "observation_mean": (17,),
            "observation_std": (17,),
            "network.0.weight": (128, 17),
            "network.0.bias": (128,),
            "network.2.weight": (128, 128),
            "network.2.bias": (128,),
            "network.4.weight": (6, 128),
            "network.4.bias": (6,),
        }
        _need(set(state) == set(expected), "Checkpoint tensor roster")
        digest = hashlib.sha256(POLICY_SCHEMA.encode())
        for name, tensor in sorted(state.items()):
            _need(
                tuple(tensor.shape) == expected[name]
                and tensor.dtype == torch.float32
                and bool(torch.isfinite(tensor).all()),
                "Checkpoint tensor interface",
            )
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(
                json.dumps(
                    [name, list(array.shape), str(array.dtype)], separators=(",", ":")
                ).encode()
            )
            digest.update(array.tobytes())
        _need(
            digest.hexdigest() == owner and bool((state["observation_std"] > 0).all()),
            "Checkpoint named-tensor hash/normalization",
        )
        self.record_payloads[binding["file"]] = payload
        self._record_bindings[binding["file"]] = binding["bytes"], binding["sha256"]
        return payload

    def load_all(self, records: list[dict[str, Any]] | None = None) -> None:
        if records is None:
            for owner in self.registry:
                self._load(owner)
        else:
            _need(len(records) <= 15, "Too many checkpoint endpoint records")
            for record in records:
                self._decode(record["actor_hash"], record)

    def set_scope(self, scope: str | None) -> None:
        """Bind the next forward to its work owner, never infer it from the actor."""
        if self.max_scope_neural_rows is not None:
            _need(scope in self.max_scope_neural_rows, "Unknown audit prediction budget scope")
        self.scope = scope

    def predict(self, owner: str, observations: np.ndarray) -> np.ndarray:
        _memory_boundary()
        _need(
            self.deadline is None or time.monotonic() <= self.deadline,
            "Audit wall-time boundary exhausted",
        )
        observations = np.asarray(observations)
        _need(
            observations.ndim in (1, 2)
            and observations.shape[-1] == 17
            and np.isfinite(observations).all(),
            "Bound finite prediction inputs",
        )
        rows = 1 if observations.ndim == 1 else len(observations)
        if self.max_scope_neural_rows is not None:
            _need(
                self.scope in self.max_scope_neural_rows,
                "Missing audit prediction budget scope before forward",
            )
            _need(
                self.scope_neural_rows.get(self.scope, 0) + rows
                <= self.max_scope_neural_rows[self.scope],
                "Audit scope neural reservation exhausted before forward",
            )
        _need(
            self.neural_rows + rows <= self.max_neural_rows,
            "Audit neural reservation exhausted before forward",
        )
        _need(
            self.prediction_calls < self.max_prediction_calls,
            "Audit prediction-call reservation exhausted",
        )
        if self.event_sink is not None:
            self.event_sink(
                dict(
                    event="prediction_requested",
                    actor_hash=owner,
                    scope=self.scope,
                    rows=rows,
                    call=self.prediction_calls,
                    requested_rows_after=self.neural_rows + rows,
                )
            )
        self.neural_rows += rows
        if self.scope is not None:
            self.scope_neural_rows[self.scope] = self.scope_neural_rows.get(self.scope, 0) + rows
        state = self._load(owner)
        import torch
        import torch.nn.functional as functional

        self.prediction_calls += 1
        try:
            with torch.no_grad():
                value = torch.as_tensor(
                    observations.astype(np.float32, copy=True), dtype=torch.float32
                )
                value = (value - state["observation_mean"]) / state["observation_std"]
                for layer in (0, 2, 4):
                    value = torch.tanh(
                        functional.linear(
                            value, state[f"network.{layer}.weight"], state[f"network.{layer}.bias"]
                        )
                    )
        except BaseException as error:
            self.unknown_partial_neural_rows += rows
            if self.scope is not None:
                self.scope_unknown_partial_neural_rows[self.scope] = (
                    self.scope_unknown_partial_neural_rows.get(self.scope, 0) + rows
                )
            if self.event_sink is not None:
                self.event_sink(
                    dict(
                        event="prediction_error",
                        actor_hash=owner,
                        scope=self.scope,
                        rows=rows,
                        error_type=type(error).__name__,
                    )
                )
            raise
        self.completed_neural_rows += rows
        if self.scope is not None:
            self.scope_completed_neural_rows[self.scope] = (
                self.scope_completed_neural_rows.get(self.scope, 0) + rows
            )
        if self.event_sink is not None:
            self.event_sink(
                dict(
                    event="prediction_complete",
                    actor_hash=owner,
                    scope=self.scope,
                    rows=rows,
                    completed_rows_after=self.completed_neural_rows,
                )
            )
        output = value.cpu().numpy().copy()
        _need(np.isfinite(output).all(), "Nonfinite independent network output")
        return output

    def verify(self, owner: str, observations: np.ndarray, saved: np.ndarray) -> None:
        output = self.predict(owner, observations)
        _need(
            output.dtype == saved.dtype
            and output.shape == saved.shape
            and output.tobytes() == saved.tobytes(),
            "Saved prediction bytes differ from independent actor replay",
        )


def audit_stream_structure(
    archive: StreamArchive, *, event_observer: Callable[[dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    """Reconcile every pre-call native/measurement request and completion.

    This structural proof deliberately does not claim D/selection/learning
    outcome or network verification. Scientific callers add those checks and
    compare all recorded end counters rather than treating this as a full audit.
    """
    counts: dict[str, int] = {}
    pending_queries: dict[str, dict[str, Any]] = {}
    pending_measures: dict[str, dict[str, Any]] = {}
    recent: deque[dict[str, Any]] = deque(maxlen=8)
    first = closed = None
    events = 0
    for event in archive.ledger():
        events += 1
        _need(closed is None, "Ledger continued after runtime close")
        if first is None:
            first = event
            _need(event["event"] == "runtime_open", "Missing runtime opening")
        kind = event["event"]
        if kind == "query_requested":
            query = event["query_id"]
            _need(
                query not in pending_queries and len(pending_queries) == 0,
                "Native calls must have unique synchronous ownership",
            )
            archive.verify_reference(event["inputs"])
            controls = _integer(event["requested_controls"], "requested controls", 1)
            substeps = _integer(event["requested_physics_substeps"], "physics substeps", 1)
            _need(substeps == 4 * controls and event["frame_skip"] == 4, "Native frame-skip census")
            mode = event["mode"]
            _need(mode in ("live", "model"), "Unknown native query mode")
            if mode == "model":
                shape = event["inputs"]["arrays"]["actions"]["shape"]
                _need(
                    len(shape) == 3 and shape[2] == 6 and math.prod(shape[:2]) == controls,
                    "Native batch action/control count",
                )
            else:
                _need(
                    controls == 1 and event["inputs"]["arrays"]["action"]["shape"] == [6],
                    "Live action/control count",
                )
            for name, amount in (
                ("requested_controls", controls),
                ("requested_" + mode + "_controls", controls),
                ("requested_physics_substeps", substeps),
                (mode + "_queries", 1),
            ):
                counts[name] = counts.get(name, 0) + amount
            pending_queries[query] = event
        elif kind in ("query_complete", "query_error"):
            _need(event.get("query_id") in pending_queries, "Orphan native completion/error")
            request = pending_queries.pop(event["query_id"])
            _need(
                event.get("request_id") == request["id"] and event.get("mode") == request["mode"],
                "Native request/completion ownership",
            )
            if kind == "query_error":
                raise AuditError(
                    "Failed native query retained; successful round proof is incomplete"
                )
            _need(
                event["completed_controls"] == request["requested_controls"]
                and event["completed_physics_substeps"] == request["requested_physics_substeps"],
                "Native paid/completed work differs",
            )
            archive.verify_reference(event["result"])
            for name, amount in (
                ("completed_controls", event["completed_controls"]),
                ("completed_" + event["mode"] + "_controls", event["completed_controls"]),
                ("completed_physics_substeps", event["completed_physics_substeps"]),
            ):
                counts[name] = counts.get(name, 0) + amount
            if event["mode"] == "model":
                recent.append(
                    dict(
                        query_id=event["query_id"], inputs=request["inputs"], result=event["result"]
                    )
                )
        elif kind == "measurement_requested":
            _need(not pending_measures, "Measurement calls must have synchronous ownership")
            amount = _integer(event["amount"], "measurement amount")
            key = "requested_" + event["measure_kind"]
            counts[key] = counts.get(key, 0) + amount
            counts["measurement_requests"] = counts.get("measurement_requests", 0) + 1
            pending_measures[event["id"]] = event
        elif kind in ("measurement_complete", "measurement_error"):
            _need(
                event.get("request_id") in pending_measures, "Orphan measurement completion/error"
            )
            request = pending_measures.pop(event["request_id"])
            _need(
                event["amount"] == request["amount"]
                and event["measure_kind"] == request["measure_kind"],
                "Measurement amount/kind differs",
            )
            if kind == "measurement_error":
                raise AuditError(
                    "Failed neural/optimizer measurement retained; complete-round proof is incomplete"
                )
            key = "completed_" + event["measure_kind"]
            counts[key] = counts.get(key, 0) + event["amount"]
        elif kind in ("query_rejected", "measurement_rejected"):
            raise AuditError("Rejected boundary retained; complete-round proof is incomplete")
        elif kind == "runtime_close":
            closed = event
            _need(not pending_queries and not pending_measures, "Unclosed paid requests")
            for key, amount in counts.items():
                _need(event["totals"].get(key) == amount, "Ledger closing counter differs: " + key)
        if event_observer is not None:
            event_observer(event)
    _need(first is not None and closed is not None, "Missing closed runtime ledger")
    archive.verify_unchanged()
    return dict(
        schema="walker-h5.0-stream-structure-audit-v1",
        structure_complete=True,
        scientific_round_audit_complete=False,
        prediction_replay_complete=False,
        source_read_bytes=archive.read_bytes,
        array_groups=archive.group_checks,
        array_checks=archive.array_checks,
        ledger_events=events,
        work=counts,
        new_physics_controls=0,
        new_training_updates=0,
        new_network_rows=0,
    )


def _same(actual: Any, expected: Any, label: str) -> None:
    a, b = np.asarray(actual), np.asarray(expected)
    _need(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label)


def _close(actual: Any, expected: Any, label: str) -> None:
    _need(np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True), label)


def _weights(scores: np.ndarray) -> np.ndarray:
    finite = np.isfinite(scores)
    _need(np.any(finite), "No finite independently weighted candidate")
    values = np.zeros(scores.shape, dtype=float)
    values[finite] = np.exp(-(scores[finite] - np.min(scores[finite])) / 3.0)
    return values / values.sum()


def _planner_weights(scores: np.ndarray) -> np.ndarray:
    """Preserve the planner's full-vector arithmetic, distinct from selector."""
    finite = np.isfinite(scores)
    _need(np.any(finite), "No finite planner scores")
    rho = np.min(scores[finite])
    shifted = np.where(finite, scores - rho, np.inf)
    unnormalized = np.exp(-shifted / 3.0)
    normalizer = np.sum(unnormalized)
    if not np.isfinite(normalizer) or normalizer <= 0:
        weights = np.zeros_like(scores, dtype=float)
        weights[np.argmin(shifted)] = 1
        return weights
    return unnormalized / normalizer


def _noise_cholesky(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Independently reconstruct the frozen stationary Butterworth covariance."""
    from scipy.linalg import solve_discrete_lyapunov, toeplitz
    from scipy.signal import butter, tf2ss

    _need(
        config["noise_sigma"] == 0.3
        and config["noise_lowpass_order"] == 2
        and config["noise_lowpass_sample_rate_hz"] == 125.0
        and config["noise_lowpass_cutoff_hz"] == 23.4375,
        "Changed perturbation distribution",
    )
    numerator, denominator = butter(2, 23.4375, btype="lowpass", fs=125.0, output="ba")
    a, b, c, d = tf2ss(numerator, denominator)
    covariance = solve_discrete_lyapunov(a, b @ b.T)
    autocovariance = np.empty(48, dtype=float)
    autocovariance[0] = (c @ covariance @ c.T + d @ d.T).item()
    previous = np.eye(a.shape[0], dtype=float)
    for lag in range(1, 48):
        current = a @ previous
        autocovariance[lag] = (c @ current @ covariance @ c.T + c @ previous @ b @ d.T).item()
        previous = current
    correlation = toeplitz(autocovariance / autocovariance[0])
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    return np.linalg.cholesky(correlation), np.linalg.cholesky((0.3**2) * np.eye(6))


class RoundChecks:
    """Independent streamed scientific checks over a complete H5.0 ledger.

    ``predictor`` must verify/predict the exact original input shape and dtype;
    ``SavedActorVerifier`` provides the real frozen-CPU implementation. Tests
    may supply a deterministic fake, which is not evidence of a real NN audit.
    This class does not claim to reproduce fitting or prove host time/storage.
    """

    def __init__(
        self,
        archive: StreamArchive,
        recipe: Mapping[str, Any],
        predictor: Any,
        *,
        run: Path,
    ):
        self.archive, self.recipe, self.predictor = archive, recipe, predictor
        self.run = Path(run).resolve()
        _need(
            recipe["guidance"]["support_radius"] is None,
            "Frozen audit only permits null-radius support",
        )
        _need(
            recipe["guidance"]["alpha"] == 0.1199341983507337
            and recipe["guidance"]["eta_u"] == 0.01,
            "Changed agreement coefficient requires another audit version",
        )
        self._pending_queries: dict[str, dict[str, Any]] = {}
        self._queries: deque[dict[str, Any]] = deque(maxlen=8)
        self._live_queries: list[dict[str, Any]] = []
        self._agreements: dict[str, dict[str, Any]] = {}
        self._scored: dict[str, dict[str, Any]] = {}
        self._selection: dict[str, Any] | None = None
        self._decision: dict[str, Any] | None = None
        self._incoming_context: dict[str, Any] | None = None
        self._incoming_reference: dict[str, Any] | None = None
        self._proposal_passes: list[dict[str, Any]] = []
        self._expected_live_action: np.ndarray | None = None
        self._context: dict[str, Any] = {}
        self._npz: dict[str, dict[str, np.ndarray]] = {}
        self.input_read_bytes = self.verified_prediction_rows = self.agreement_checks = 0
        self.native_cost_checks = self.selection_checks = self.episode_checks = 0
        self.direct_episode_checks = self.posthoc_checks = self.metric_checks = 0
        self.episodes: dict[str, dict[str, Any]] = {}
        self.metric_datasets: dict[str, dict[str, Any]] = {}
        self.checkpoints: list[dict[str, Any]] = []
        self.fit_updates: dict[str, list[dict[str, Any]]] = {}
        self.fit_measurements: dict[str, dict[str, int]] = {}
        self.fit_initial_states: dict[str, dict[str, Any]] = {}
        self.parent_actors: dict[int, str] = {}
        self.standard_normal_draws = 0
        self._noise_factors: tuple[np.ndarray, np.ndarray] | None = None
        self._decision_queries: list[str] = []
        self._planning_context: dict[str, np.ndarray] | None = None
        self._last_private: dict[str, Any] | None = None
        self._seen_teacher_tags: set[str] = set()
        self._selection_infos: list[dict[str, Any]] = []
        self.teacher_posthoc_tags: set[str] = set()
        self.teacher_controls = 0

    def _prediction_scope(self, scope: str | None) -> None:
        setter = getattr(self.predictor, "set_scope", None)
        if setter is not None:
            setter(scope)

    def _metric_scope(self, tag: str) -> None:
        if getattr(self.predictor, "set_scope", None) is None:
            return
        if re.fullmatch(r"s[0-2]_parent(?:_initial)?/(?:train|validation)", tag):
            self._prediction_scope("shared_parent")
            return
        child = re.fullmatch(
            r"s[0-2]_(independent|selector_only|proposal_guided)_child/"
            r"(?:incoming|endpoint|train|validation)",
            tag,
        )
        _need(child is not None, "Unknown full-data metric audit budget scope")
        self._prediction_scope(child[1])

    def _query(self, identifier: str) -> dict[str, Any]:
        matches = [query for query in self._queries if query["query_id"] == identifier]
        _need(len(matches) == 1, "Agreement lacks its bounded recent native query binding")
        return matches[0]

    def _model_arrays(self, identifiers: list[str]) -> dict[str, np.ndarray | None]:
        _need(
            1 <= len(identifiers) <= 4 and len(set(identifiers)) == len(identifiers),
            "Invalid scored native-query roster",
        )
        actions, posts, costs, observations = [], [], [], []
        initial = warm = None
        for identifier in identifiers:
            query = self._query(identifier)
            current_initial = self.archive.array(query["inputs"], "initial_state")
            current_warm = self.archive.array(query["inputs"], "initial_warmstart")
            if initial is None:
                initial, warm = current_initial, current_warm
            else:
                _same(current_initial, initial, "Scored candidates have different initial states")
                _same(current_warm, warm, "Scored candidates have different native warmstarts")
            action = self.archive.array(query["inputs"], "actions")
            post = self.archive.array(query["result"], "states")
            cost = self.archive.array(query["result"], "raw_scores")
            actions.append(action)
            posts.append(post)
            costs.append(cost)
            observations.append(native_observations(current_initial, post))
        return dict(
            initial=initial,
            warmstart=warm,
            actions=np.concatenate(actions),
            posts=np.concatenate(posts),
            costs=np.concatenate(costs),
            observations=np.concatenate(observations),
        )

    def _agreement(self, request_id: str, reference: dict, *, prefix: str = "") -> dict[str, Any]:
        _need(request_id in self._agreements, "Missing saved agreement request")
        request = self._agreements.pop(request_id)
        arrays = self._model_arrays(request["scored_native_query_ids"])
        predictions = self.archive.array(reference, prefix + "predictions")
        squared, support = agreement_values(
            arrays["observations"], arrays["actions"], arrays["posts"], arrays["costs"], predictions
        )
        _same(
            self.archive.array(reference, prefix + "support"),
            support,
            "Saved agreement support differs",
        )
        _same(
            self.archive.array(reference, prefix + "agreement_squared"),
            squared,
            "Saved D differs from frozen sum",
        )
        _need(
            request["supported_rows"] == int(np.count_nonzero(support)),
            "Agreement row count differs",
        )
        _need(
            request["input_digest"]
            == plan_digest(arrays["observations"], arrays["actions"], support, arrays["costs"]),
            "Agreement input digest differs",
        )
        if np.any(support):
            self._prediction_scope(request.get("context", {}).get("arm"))
            self.predictor.verify(
                request["actor_hash"], arrays["observations"][support], predictions
            )
        self.verified_prediction_rows += int(np.count_nonzero(support))
        self.agreement_checks += 1
        return dict(arrays, squared=squared, support=support, request=request)

    def _proposal(self, event: dict, reference: dict) -> None:
        _need(self._decision is not None, "Proposal pass outside a decision")
        measured = self._agreement(event["agreement_request_id"], reference)
        nominal = self.archive.array(reference, "nominal_before")
        _same(
            self.archive.array(reference, "initial_state"),
            measured["initial"],
            "Proposal pass initial state differs",
        )
        expected_score = measured["costs"]
        if event["guidance_enabled"]:
            expected_score = expected_score + 0.5 * event["alpha"] * measured["squared"]
        _need(
            event["guidance_enabled"] is (self._decision["arm"] == "P"),
            "Wrong arm changed inner optimization",
        )
        _same(self.archive.array(reference, "score"), expected_score, "Wrong proposal objective")
        _need(event["pass_index"] == len(self._proposal_passes), "Proposal pass order differs")
        if self._proposal_passes:
            _same(
                nominal,
                self._proposal_passes[-1]["updated"],
                "Initial refinement shifted or replaced the guided nominal",
            )
        updated = np.clip(
            nominal
            + np.einsum(
                "k,kha->ha", _planner_weights(expected_score), measured["actions"] - nominal[None]
            ),
            -1,
            1,
        )
        self._proposal_passes.append(
            dict(
                nominal=nominal,
                updated=updated,
                score=expected_score,
                query_id=measured["request"]["scored_native_query_ids"][0],
            )
        )

    def _sample(self, generator: np.random.Generator, nominal: np.ndarray) -> np.ndarray:
        if self._noise_factors is None:
            self._noise_factors = _noise_cholesky(self.recipe["mppi"])
        temporal, spatial = self._noise_factors
        standard = generator.standard_normal((256, 48, 6))
        self.standard_normal_draws += standard.size
        shaped = np.einsum("ht,kta->kha", temporal, standard, optimize=True)
        noise = np.einsum("khi,ji->khj", shaped, spatial)
        return np.clip(nominal[None] + noise, -1, 1)

    def _private_continuity(self, protected: dict, proposal: dict) -> None:
        context = self._decision["context"]
        episode, step = context.get("episode"), context.get("step")
        match = re.fullmatch(
            r"(gap_)?teacher_s([0-2])_(independent|selector_only|proposal_guided)_([0-9]{2})",
            str(episode),
        )
        if match is None:
            _need("stream" not in context, "Unrecognized real teacher context")
            return
        _need(
            type(step) is int and context["stream"] == int(match[2]) and context["arm"] == match[3],
            "Decision/episode/stream ownership differs",
        )
        _need(
            self._decision["arm"]
            == dict(independent="I", selector_only="S", proposal_guided="P")[match[3]],
            "Decision used the wrong contemporary control",
        )
        if step == 0:
            _need(
                episode not in self._seen_teacher_tags, "A teacher episode/controller was restarted"
            )
            self._seen_teacher_tags.add(episode)
            stream = self.recipe["streams"][int(match[2])]
            prefix = "gap_" if match[1] else ""
            index = int(match[4])
            _need(index < (2 if match[1] else 4), "Unrequested teacher slot")
            for name, before in (("protected", protected), ("second", proposal)):
                _same(
                    np.asarray(before["U"], dtype=float),
                    np.zeros((48, 6)),
                    "A fresh producer did not begin with zero nominal",
                )
                _need(before["initial_plan"] is True, "A fresh producer skipped initial refinement")
                seed_key = prefix + (
                    "protected_planner_seeds" if name == "protected" else "second_planner_seeds"
                )
                expected_rng = np.random.Generator(
                    np.random.PCG64(stream[seed_key][index])
                ).bit_generator.state
                _need(
                    before["rng_state"] == expected_rng,
                    "A fresh producer used the wrong private seed",
                )
        else:
            _need(
                self._last_private is not None
                and self._last_private["episode"] == episode
                and self._last_private["step"] == step - 1,
                "Producer continuation skipped or reused a control",
            )
            _need(
                protected == self._last_private["protected"]
                and proposal == self._last_private["proposal"],
                "Producer before state differs from previous exact after state",
            )
        _need(
            self._decision["optimization_iterations"] == (2 if step == 0 else 1),
            "Initial/subsequent refinement budget changed",
        )

    def _selection_check(self, event: dict, reference: dict) -> None:
        _need(self._decision is not None, "Selection outside a decision")
        bank = self._agreement(event["bank_agreement_request_id"], reference)
        before = self.archive.json_array(reference, "protected_before")
        proposal_before = self.archive.json_array(reference, "proposal_before")
        _need(
            self._incoming_reference is not None
            and event["decision_context_group"] == self._incoming_reference["group"],
            "Pre-query planner context binding missing",
        )
        _need(
            before == self.archive.json_array(self._incoming_reference, "protected_before")
            and proposal_before
            == self.archive.json_array(self._incoming_reference, "proposal_before"),
            "Post-hoc before snapshot differs from pre-query owner",
        )
        after = self.archive.json_array(reference, "protected_after")
        proposal_after = self.archive.json_array(reference, "proposal_after")
        nominal = np.asarray(before["U"], dtype=float)
        proposal_nominal = np.asarray(proposal_before["U"], dtype=float)
        _need(
            plan_digest(nominal) == self._decision["protected_nominal_digest"],
            "Protected before-state binding",
        )
        _need(
            plan_digest(proposal_nominal) == self._decision["proposal_nominal_digest"],
            "Proposal before-state binding",
        )
        passes = self._decision["optimization_iterations"]
        _need(len(self._proposal_passes) == passes, "Missing initial/continuing proposal passes")
        _same(
            self._proposal_passes[0]["nominal"],
            proposal_nominal,
            "Proposal nominal was externally replaced",
        )
        query_ids = bank["request"]["native_query_ids"]
        _need(
            len(query_ids) == 2 * passes + 2, "Unexpected proposal/producer verification call count"
        )
        expected_ids = [
            query_ids[index] for index in (passes - 1, passes, 2 * passes, 2 * passes + 1)
        ]
        _need(
            expected_ids == bank["request"]["scored_native_query_ids"],
            "Final bank omitted/replaced a producer candidate family",
        )
        _need(
            event["producer_cache_query_ids"]
            == dict(
                protected_final_sample=query_ids[passes - 1],
                protected_verified_mean=query_ids[passes],
                proposal_final_sample=query_ids[2 * passes],
                proposal_verified_mean=query_ids[2 * passes + 1],
            ),
            "Producer cached rollout lineage differs",
        )
        _need(
            self._decision_queries[: 2 * passes + 2] == query_ids
            and len(self._decision_queries) == 2 * passes + 4,
            "Current decision native query sequence differs",
        )
        self._private_continuity(before, proposal_before)
        protected_rng = np.random.Generator(np.random.PCG64())
        protected_rng.bit_generator.state = before["rng_state"]
        for index in range(passes):
            protected_pass = self._model_arrays([query_ids[index]])
            _same(
                protected_pass["actions"],
                self._sample(protected_rng, nominal),
                "Protected samples differ from its actor-free private RNG distribution",
            )
            protected_weights = _planner_weights(protected_pass["costs"])
            nominal = np.clip(
                nominal
                + np.einsum(
                    "k,kha->ha",
                    protected_weights,
                    protected_pass["actions"] - nominal[None],
                ),
                -1,
                1,
            )
        _same(
            self.archive.array(reference, "protected_final_weights"),
            protected_weights,
            "Protected cached weights contain guidance",
        )
        _same(
            self.archive.array(reference, "proposal_final_weights"),
            _planner_weights(self._proposal_passes[-1]["score"]),
            "Proposal cached weights differ from its scored update",
        )
        protected_mean = self._model_arrays([query_ids[passes]])["actions"][0]
        _same(protected_mean, nominal, "Protected mean contains actor/selector influence")
        guided_stream_mean = self._model_arrays([query_ids[2 * passes + 1]])["actions"][0]
        _same(
            guided_stream_mean,
            self._proposal_passes[-1]["updated"],
            "Guided proposal update did not define its verified mean",
        )
        for saved, last in ((after, nominal), (proposal_after, guided_stream_mean)):
            shifted = last.copy()
            shifted[:-1] = shifted[1:]
            shifted[-1] = shifted[-2].copy()
            _same(
                np.asarray(saved["U"], dtype=float),
                shifted,
                "Selected plan contaminated a producer's shifted nominal",
            )
            _need(saved["initial_plan"] is False, "Producer initial-plan flag not advanced")
        _need(
            protected_rng.bit_generator.state == after["rng_state"],
            "Protected RNG advancement differs",
        )
        proposal_rng = np.random.Generator(np.random.PCG64())
        proposal_rng.bit_generator.state = proposal_before["rng_state"]
        for scored in self._proposal_passes:
            query = self._query(scored["query_id"])
            _same(
                self.archive.array(query["inputs"], "actions"),
                self._sample(proposal_rng, scored["nominal"]),
                "Guided samples differ from its nominal/private RNG distribution",
            )
        _need(
            proposal_rng.bit_generator.state == proposal_after["rng_state"],
            "Proposal RNG advancement differs",
        )
        actions, costs = bank["actions"], bank["costs"]
        _same(
            self.archive.array(reference, "raw_costs"),
            costs,
            "Native costs were changed before selection",
        )
        protected_count = 257
        for digest_key, selection, center in (
            (
                "protected_bank_digest",
                slice(0, protected_count),
                np.asarray(before["U"], dtype=float),
            ),
            (
                "proposal_bank_digest",
                slice(protected_count, None),
                np.asarray(proposal_before["U"], dtype=float),
            ),
            ("bank_digest", slice(None), np.asarray(before["U"], dtype=float)),
        ):
            expected_digest = plan_digest(
                bank["initial"],
                bank["warmstart"],
                actions[selection],
                bank["posts"][selection],
                costs[selection],
                np.clip(center, -1, 1),
            )
            _need(
                event[digest_key] == expected_digest,
                "Bank provenance digest differs: " + digest_key,
            )
        raw_index = int(
            np.argmin(
                np.where(np.isfinite(costs[:protected_count]), costs[:protected_count], np.inf)
            )
        )
        raw_plan, raw_score = actions[raw_index], float(costs[raw_index])
        bound = 1.05 * raw_score + 1e-12
        _need(event["native_guard_bound"] == bound, "Native guard reference differs")
        feasible = np.isfinite(costs) & (costs <= bound)
        _same(
            self.archive.array(reference, "feasible"), feasible, "Native feasibility mask differs"
        )
        reference_u = np.clip(np.asarray(before["U"], dtype=float), -1, 1)
        proximity = 0.5 * 0.01 * np.sum((actions - reference_u[None]) ** 2, axis=(1, 2))
        baseline = costs + proximity
        guided = baseline + 0.5 * 0.1199341983507337 * bank["squared"]
        for key, expected in (
            ("proximity", proximity),
            ("baseline_scores", baseline),
            ("guided_scores", guided),
        ):
            _same(
                self.archive.array(reference, key),
                expected,
                "Final score component differs: " + key,
            )
        recent_ids = [query["query_id"] for query in self._queries]
        selector_ids = recent_ids[-2:]
        mean_agreement = None
        if event.get("guided_mean_agreement_request_id") is not None:
            mean_agreement = self._agreement(
                event["guided_mean_agreement_request_id"], reference, prefix="guided_mean_"
            )
            _need(
                mean_agreement["request"]["scored_native_query_ids"] == selector_ids[-1:],
                "Guided mean scored the wrong native continuation",
            )
        alternatives = {}
        for name, scores, identifier in (
            ("baseline", baseline, selector_ids[0]),
            ("guided", guided, selector_ids[1]),
        ):
            eligible = np.where(feasible, scores, np.inf)
            index = int(np.argmin(eligible))
            chosen = actions[index]
            mean = np.einsum("k,kha->ha", _weights(eligible), actions)
            _same(
                self.archive.array(reference, name + "_mean"),
                mean,
                "Selector mean arithmetic differs",
            )
            query = self._model_arrays([identifier])
            _same(query["actions"][0], mean, "Selector executed an unbound mean verification")
            mean_cost = float(query["costs"][0])
            if not np.isfinite(mean_cost) or mean_cost > bound or mean_cost < 0:
                chosen = raw_plan
            else:
                mean_score = mean_cost + 0.5 * 0.01 * float(np.sum((mean - reference_u) ** 2))
                if name == "guided":
                    _need(
                        mean_agreement is not None, "Finite guided mean lacks a prediction binding"
                    )
                    mean_score += 0.5 * 0.1199341983507337 * mean_agreement["squared"][0]
                if mean_score < scores[index]:
                    chosen = mean
            alternatives[name] = chosen
        selected = alternatives["baseline" if self._decision["arm"] == "I" else "guided"]
        _same(
            self.archive.array(reference, "selected_plan"),
            selected,
            "Selected plan/tie/guard rule differs",
        )
        self._expected_live_action = selected[0]
        self._selection_infos.append(
            dict(
                policy_hash=event["actor_hash"],
                selection={key: reference[key] for key in ("file", "group", "tag")},
            )
        )
        self._last_private = dict(
            episode=self._decision["context"].get("episode"),
            step=self._decision["context"].get("step"),
            protected=after,
            proposal=proposal_after,
        )
        self.selection_checks += 1
        self._decision, self._proposal_passes = None, []
        self._incoming_context = self._incoming_reference = None

    def _episode(self, event: dict) -> None:
        reference = event["raw"]
        states = self.archive.array(reference, "states")
        observations = self.archive.array(reference, "observations")
        actions = self.archive.array(reference, "actions")
        costs = self.archive.array(reference, "costs")
        velocities = self.archive.array(reference, "velocities")
        warmstarts = self.archive.array(reference, "warmstarts")
        integrations = self.archive.array(reference, "integrations")
        n = _integer(event["steps"], "episode controls", 1)
        _need(
            states.shape == (n + 1, 19)
            and actions.shape == (n, 6)
            and observations.shape == (n, 17),
            "Episode N/N+1 state/action ownership",
        )
        _need(
            warmstarts.shape == (n + 1, 9)
            and integrations.ndim == 2
            and integrations.shape[0] == n + 1
            and 1 <= integrations.shape[1] <= 128,
            "Episode N+1 solver context is missing or unbounded",
        )
        _need(
            len(self._live_queries) == n,
            "Episode controls lack complete native live-query ownership",
        )
        for index, query in enumerate(self._live_queries):
            inputs, result = query["inputs"], query["result"]
            _same(
                self.archive.array(inputs, "state"),
                states[index],
                "Episode initial live state differs",
            )
            _same(
                self.archive.array(result, "state"),
                states[index + 1],
                "Episode native post state differs",
            )
            for query_key, episode_values in (
                ("warmstart", warmstarts),
                ("integration", integrations),
            ):
                _same(
                    self.archive.array(inputs, query_key),
                    episode_values[index],
                    "Episode pre-action solver context differs",
                )
                _same(
                    self.archive.array(result, query_key),
                    episode_values[index + 1],
                    "Episode terminal/next solver context differs",
                )
            _same(
                self.archive.array(inputs, "action").astype(np.float64),
                actions[index].astype(np.float64),
                "Episode executed control differs from native query",
            )
            _close(
                self.archive.array(result, "cost"),
                costs[index],
                "Episode native per-control cost differs",
            )
        expected_obs = np.concatenate(
            (states[:-1, 2:10], np.clip(states[:-1, 10:19], -10, 10)), axis=-1
        )
        _same(observations, expected_obs, "Episode pre-action observations differ")
        _same(velocities, states[1:, 10], "Episode velocity join differs")
        healthy = healthy_prefix(states[None, 1:])[0]
        _need(np.all(healthy), "Episode continued after its first fall")
        last = states[-1]
        fell = not (0.8 < last[2] < 2 and -1 < last[3] < 1)
        survived = n == 1000 and not fell
        _need(fell or n == 1000, "Incomplete nonterminal episode")
        _close(
            np.sum(costs),
            native_cost(states[None, 1:], actions[None].astype(np.float64), self.recipe["task"])[0],
            "Episode native cost reconstruction",
        )
        mae = float(np.mean(np.abs(velocities - 1.5)))
        absorbing = 5.0 * (1000 - n) if fell else 0.0
        _need(
            event["survived"] is survived and event["success"] is (survived and mae <= 0.3),
            "Walking/full-survival outcomes differ",
        )
        _close(event["mae"], mae, "Episode velocity outcome differs")
        _close(
            event["cost"],
            np.sum(costs) + absorbing,
            "Episode absorbing/terminal accounting differs",
        )
        _close(event["absorbing_cost"], absorbing, "Absorbing tail differs")
        _close(event["terminal_adjustment"], 0.0, "Native terminal was charged twice")
        _close(
            self.archive.array(reference, "terminal_cost"),
            20.0 if fell else 0.0,
            "Saved real terminal cost differs",
        )
        _close(
            self.archive.array(reference, "terminal_adjustment"),
            0.0,
            "Saved terminal adjustment differs",
        )
        _close(
            self.archive.array(reference, "absorbing_cost"),
            absorbing,
            "Saved absorbing tail differs",
        )
        if self._context.get("role") == "actor_only":
            infos = self.archive.json_array(reference, "infos")["rows"]
            owner = self._context["actor_hash"]
            _need(
                len(infos) == n and all(row["controller"]["policy_hash"] == owner for row in infos),
                "Standalone actor ownership differs",
            )
            for observation, action in zip(observations, actions, strict=True):
                arm = self._context.get("arm")
                self._prediction_scope("shared_parent" if arm is None else arm)
                self.predictor.verify(owner, observation, action)
            self.verified_prediction_rows += n
            self.direct_episode_checks += 1
        elif self._context.get("role") == "teacher" and "stream" in self._context:
            infos = self.archive.json_array(reference, "infos")["rows"]
            _need(
                len(self._selection_infos) == n and len(infos) == n,
                "Teacher control lacks exactly one verified selection",
            )
            for recorded, expected in zip(infos, self._selection_infos, strict=True):
                _need(
                    recorded["controller"]["policy_hash"] == expected["policy_hash"]
                    and recorded["controller"]["selection"] == expected["selection"]
                    and recorded["controller"]["arm"] == self._context["arm"],
                    "Teacher controller pointer/parent ownership differs",
                )
            self.teacher_controls += n
        self.episode_checks += 1
        _need(event["tag"] not in self.episodes, "An episode tag was reused")
        self.episodes[event["tag"]] = dict(event, context=dict(self._context))
        self._live_queries = []
        self._selection_infos = []

    def _input(self, binding: dict) -> np.ndarray:
        if "reference" in binding:
            return self.archive.array(binding["reference"], binding["key"])
        filename = binding["file"]
        if filename not in self._npz:
            path = (self.run / filename).resolve()
            _need(path.is_relative_to(self.run) and path.is_file(), "Metric input file escapes run")
            candidates = [
                entry for entry in self.recipe["inputs"].values() if entry.get("file") == filename
            ]
            _need(len(candidates) == 1, "Metric input lacks a frozen file binding")
            expected = candidates[0]
            _need(
                path.stat().st_size == expected["bytes"] and path.stat().st_size <= 4 * 1024**2,
                "Metric input exceeds fixed corpus binding",
            )
            data = path.read_bytes()
            self.input_read_bytes += len(data)
            _need(
                hashlib.sha256(data).hexdigest() == expected["sha256"],
                "Metric corpus file hash differs",
            )
            with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                self._npz[filename] = {key: arrays[key] for key in arrays.files}
        _need(binding["key"] in self._npz[filename], "Missing corpus array")
        return self._npz[filename][binding["key"]]

    def event(self, event: dict) -> None:
        if isinstance(event.get("context"), dict):
            self._context = event["context"]
        kind = event["event"]
        if kind == "query_requested":
            self._pending_queries[event["query_id"]] = event
            if event["mode"] == "model":
                _need(
                    self._decision is not None,
                    "Native proposal query occurred outside a bound decision",
                )
                self._decision_queries.append(event["query_id"])
            if (
                event["mode"] == "live"
                and self._context.get("role") == "teacher"
                and "stream" in self._context
            ):
                _need(
                    self._expected_live_action is not None,
                    "Teacher live control has no verified decision",
                )
            if event["mode"] == "live" and self._expected_live_action is not None:
                _same(
                    self.archive.array(event["inputs"], "action"),
                    self._expected_live_action,
                    "Executed teacher action differs from verified selection",
                )
                _need(
                    self._planning_context is not None,
                    "Selected plan lacks a full physical planning context",
                )
                for key in ("state", "warmstart", "integration"):
                    _same(
                        self.archive.array(event["inputs"], key),
                        self._planning_context[key],
                        "Live control was chosen at a stale simulator/solver context",
                    )
                self._expected_live_action = None
                self._planning_context = None
        elif kind == "query_complete":
            request = self._pending_queries.pop(event["query_id"])
            if event["mode"] == "model":
                actions = self.archive.array(request["inputs"], "actions")
                posts = self.archive.array(event["result"], "states")
                saved_cost = self.archive.array(event["result"], "raw_scores")
                planning = dict(
                    state=self.archive.array(request["inputs"], "initial_state"),
                    warmstart=self.archive.array(request["inputs"], "initial_warmstart"),
                    integration=self.archive.array(request["inputs"], "live_integration_before"),
                )
                _need(
                    plan_digest(planning["state"], planning["warmstart"])
                    == self._decision["input_digest"],
                    "Decision digest does not bind its actual native planning input",
                )
                if self._planning_context is None:
                    self._planning_context = planning
                else:
                    for key in planning:
                        _same(
                            planning[key],
                            self._planning_context[key],
                            "Native queries used different full physical/solver contexts",
                        )
                _close(
                    saved_cost,
                    native_cost(posts, actions, self.recipe["task"]),
                    "Native horizon score differs",
                )
                self.native_cost_checks += len(actions)
                self._queries.append(
                    dict(
                        query_id=event["query_id"], inputs=request["inputs"], result=event["result"]
                    )
                )
            else:
                _need(len(self._live_queries) < 1000, "Live episode exceeds frozen control horizon")
                self._live_queries.append(dict(inputs=request["inputs"], result=event["result"]))
        elif kind == "h5_decision_context":
            _need(
                self._incoming_context is None
                and not self._pending_queries
                and self._decision is None
                and self._expected_live_action is None,
                "Planner before context was written after model work",
            )
            self._incoming_context = event
        elif kind == "h5_decision_context_evidence":
            _need(
                self._incoming_context is not None
                and event["request_id"] == self._incoming_context["id"],
                "Pre-query context evidence owner missing",
            )
            self.archive.verify_reference(event["evidence"])
            self._incoming_reference = event["evidence"]
        elif kind == "h5_decision_started":
            _need(
                self._decision is None and not self._agreements and not self._scored,
                "Unclosed previous decision",
            )
            _need(
                self._incoming_context is not None
                and self._incoming_reference is not None
                and self._incoming_context["input_digest"] == event["input_digest"]
                and self._incoming_context["arm"] == event["arm"],
                "Decision lacks a pre-query full planner context",
            )
            self._decision = event
            self._decision_queries = []
        elif kind == "agreement_requested":
            _need(len(self._agreements) < 4, "Unbounded/unclosed agreement requests")
            self._agreements[event["id"]] = event
            _need(
                event["native_query_ids"] == self._decision_queries,
                "Agreement used stale or unbound native query IDs",
            )
            context = event.get("context", {})
            if "stream" in context:
                _need(
                    context.get("role") == "teacher"
                    and context["stream"] in self.parent_actors
                    and event["actor_hash"] == self.parent_actors[context["stream"]],
                    "Acquisition guide is not the frozen common incoming parent",
                )
        elif kind == "proposal_pass_scored":
            self._scored[event["id"]] = event
        elif kind == "proposal_pass_scored_evidence":
            _need(event["request_id"] in self._scored, "Proposal evidence request missing")
            self._proposal(self._scored.pop(event["request_id"]), event["evidence"])
        elif kind == "two_stream_selection":
            _need(self._selection is None, "Unclosed selection evidence")
            self._selection = event
        elif kind == "two_stream_selection_evidence":
            _need(
                self._selection is not None and event["request_id"] == self._selection["id"],
                "Selection evidence ownership",
            )
            self._selection_check(self._selection, event["evidence"])
            self._selection = None
        elif kind == "episode_complete":
            self._episode(event)
        elif kind == "teacher_agreement_complete":
            episode = event["episode_raw"]
            owners = [row for row in self.episodes.values() if row["raw"] == episode]
            _need(
                len(owners) == 1 and owners[0]["context"].get("role") == "teacher",
                "Posthoc rows do not belong to one actual teacher episode",
            )
            tag = owners[0]["tag"]
            _need(tag not in self.teacher_posthoc_tags, "Teacher episode was scored twice posthoc")
            self.teacher_posthoc_tags.add(tag)
            _need(
                event["actor_hash"] == owners[0]["context"]["actor_hash"],
                "Teacher posthoc used a different guide actor",
            )
            observations = self.archive.array(episode, "observations")
            actions = self.archive.array(episode, "actions")
            predictions = self._input(event["predictions"])
            self._prediction_scope(owners[0]["context"].get("arm"))
            self.predictor.verify(event["actor_hash"], observations, predictions)
            squared = np.sum((actions - predictions) ** 2, axis=1)
            _need(event["rows"] == len(observations), "Posthoc row census differs")
            _need(
                event["action_agreement"] == float(np.mean(squared)),
                "Posthoc scalar differs from its complete executed rows",
            )
            squared_binding = event["squared_disagreement"]
            _same(
                self._input(squared_binding),
                squared,
                "Teacher posthoc disagreement differs",
            )
            self.verified_prediction_rows += len(observations)
            self.posthoc_checks += 1
        elif kind == "policy_metric_complete":
            _need(
                event["tag"] not in self.metric_datasets,
                "A full-data metric tag was reused or selected",
            )
            x = self._input(event["observations"]).astype(np.float32, copy=True)
            y = self._input(event["targets"]).astype(np.float32, copy=True)
            _need(
                x.shape == (event["rows"], 17)
                and y.shape == (event["rows"], 6)
                and event["batch_size"] == 1024,
                "Metric rows/batch/target contract",
            )
            error = 0.0
            self._metric_scope(event["tag"])
            for start in range(0, len(x), 1024):
                prediction = self.predictor.predict(event["actor_hash"], x[start : start + 1024])
                residual = prediction.astype(np.float64) - y[start : start + 1024].astype(
                    np.float64
                )
                error += float(np.sum(residual**2, dtype=np.float64))
            _need(event["action_mse"] == error / y.size, "Held/reconstruction MSE differs")
            self.verified_prediction_rows += len(x)
            self.metric_checks += 1
            self.metric_datasets[event["tag"]] = dict(
                observations=event["observations"],
                targets=event["targets"],
                actor_hash=event["actor_hash"],
            )
        elif kind == "checkpoint":
            _need(len(self.checkpoints) < 15, "Checkpoint endpoint roster exceeded")
            _need(
                not any(row["file"] == event["file"] for row in self.checkpoints),
                "Checkpoint endpoint path reused",
            )
            self.checkpoints.append(event)
            parent = re.fullmatch(r"checkpoints/s([0-2])_parent\.pt", event["file"])
            if parent:
                self.parent_actors[int(parent[1])] = event["actor_hash"]
        elif kind == "fit_initial_state":
            tag = event["tag"]
            _need(
                tag not in self.fit_initial_states
                and tag not in self.fit_updates
                and tag not in self.fit_measurements,
                "Fit initial state must be recorded once before all paid updates",
            )
            _need(len(self.fit_initial_states) < 12, "Unexpected extra fit block")
            self.archive.verify_reference(event["expected_tape"])
            self.fit_initial_states[tag] = event
        elif kind == "fit_update":
            tag = event["tag"]
            _need(
                tag in self.fit_initial_states,
                "Actual update lacks observed incoming policy/optimizer state",
            )
            _need(tag in self.fit_updates or len(self.fit_updates) < 12, "Too many fit blocks")
            rows = self.fit_updates.setdefault(tag, [])
            _need(
                len(rows) < 2000 and event["fit_update"] == len(rows) + 1,
                "Unbounded or reordered fit-update events",
            )
            indices = np.asarray(event["indices"])
            _need(
                indices.shape == (256,)
                and indices.dtype.kind in "iu"
                and np.all((indices >= 0) & (indices < 4000)),
                "Actual minibatch indices are invalid",
            )
            fields = {
                key: event[key]
                for key in (
                    "update",
                    "fit_update",
                    "action_mse",
                    "proximal_loss",
                    "total_loss",
                    "unclipped_gradient_norm",
                )
            }
            rows.append(dict(curve=fields, indices=indices.astype(np.int64, copy=True)))
        elif kind == "measurement_requested":
            purpose = event.get("purpose", "")
            if isinstance(purpose, str) and "/" in purpose:
                tag, suffix = purpose.rsplit("/", 1)
                if suffix in ("forward", "backward", "adam") and re.fullmatch(
                    r"s[0-2]_(parent|independent_child|selector_only_child|proposal_guided_child)",
                    tag,
                ):
                    expected_kind = dict(
                        forward="neural_rows", backward="backward_rows", adam="optimizer_steps"
                    )[suffix]
                    _need(
                        tag in self.fit_initial_states,
                        "A fitting operation began before incoming-state observation",
                    )
                    _need(
                        event["measure_kind"] == expected_kind,
                        "Fit request mislabeled its paid operation",
                    )
                    counts = self.fit_measurements.setdefault(tag, {})
                    counts[expected_kind] = counts.get(expected_kind, 0) + event["amount"]
        elif kind == "runtime_close":
            _need(
                not self._pending_queries
                and not self._agreements
                and not self._scored
                and self._selection is None
                and self._decision is None
                and self._expected_live_action is None,
                "Unclosed scientific decision/prediction evidence",
            )

    def summary(self) -> dict[str, Any]:
        return dict(
            native_trajectory_cost_checks=self.native_cost_checks,
            agreement_checks=self.agreement_checks,
            selection_checks=self.selection_checks,
            episodes=self.episode_checks,
            standalone_episode_checks=self.direct_episode_checks,
            teacher_posthoc_checks=self.posthoc_checks,
            full_metric_checks=self.metric_checks,
            independently_verified_prediction_rows=self.verified_prediction_rows,
            frozen_input_read_bytes=self.input_read_bytes,
            training_reproduced=False,
            native_replay_controls=0,
            independently_recreated_standard_normal_draws=self.standard_normal_draws,
            covered_teacher_controls=self.teacher_controls,
        )

    def finish(self, *, require_complete_episodes: bool = True) -> dict[str, Any]:
        _need(
            not self._pending_queries
            and not self._agreements
            and not self._scored
            and self._selection is None
            and self._decision is None
            and self._expected_live_action is None,
            "Scientific requests remain unclosed",
        )
        if require_complete_episodes:
            _need(not self._live_queries, "Executed live queries lack a completed episode record")
        _need(
            set(self.fit_initial_states) == set(self.fit_updates),
            "Observed fitting block is incomplete",
        )
        return self.summary()


def optimizer_fingerprint(value: Any) -> str:
    """Independent typed-tree hash, including tuple/list/scalar distinctions."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if hasattr(item, "detach"):
            visit(item.detach().cpu().numpy())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(json.dumps(["array", list(array.shape), str(array.dtype)]).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda value: (type(value).__name__, repr(value))):
                visit((type(key).__name__, key))
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(json.dumps([type(item).__name__, item], allow_nan=False).encode())

    visit(value)
    return digest.hexdigest()


def _named_tensor_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(POLICY_SCHEMA.encode())
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(
            json.dumps([name, list(array.shape), str(array.dtype)], separators=(",", ":")).encode()
        )
        digest.update(array.tobytes())
    return digest.hexdigest()


def _fit_data_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def verify_fitting_ownership(
    verifier: SavedActorVerifier, checker: RoundChecks, recipe: Mapping[str, Any]
) -> dict[str, int]:
    """Verify exact saved continuation/tapes; do not rerun any training step."""
    verifier.load_all(checker.checkpoints)
    import torch

    expected_global_rng = (
        torch.Generator(device="cpu")
        .manual_seed(recipe["learning"]["global_torch_seed"])
        .get_state()
        .numpy()
    )
    fit_blocks = tape_rows = private_rng_checks = 0
    global_rng = None
    expected_names = [
        f"network.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")
    ]
    for binding in checker.checkpoints:
        owner = binding["actor_hash"]
        payload = verifier.record_payloads[binding["file"]]
        state = payload["optimizer_state"]
        _need(
            optimizer_fingerprint(state) == binding["optimizer_sha256"],
            "Complete optimizer typed-tree ownership differs",
        )
        current_global_rng = checker._input(binding["global_rng"])
        _same(
            current_global_rng,
            expected_global_rng,
            "Global Torch RNG differs from its declared initial seed",
        )
        if global_rng is None:
            global_rng = current_global_rng
        else:
            _same(current_global_rng, global_rng, "Global Torch RNG changed across frozen actors")
        for tensor_name, input_key in (("observation_mean", "mean"), ("observation_std", "std")):
            expected_normal = checker._input(
                dict(file=recipe["inputs"]["normalization"]["file"], key=input_key)
            )
            _same(
                payload["state_dict"][tensor_name].numpy(),
                expected_normal,
                "Checkpoint does not own the frozen W2 normalization",
            )
        point = re.fullmatch(
            r"s([0-2])_(parent_initial|parent|independent_child|selector_only_child|proposal_guided_child)",
            payload["metadata"]["name"],
        )
        _need(
            point is not None
            and payload["seed"] == recipe["streams"][int(point[1])]["policy_seed"],
            "Checkpoint policy seed differs from its stream owner",
        )
        _need(
            binding["file"] == f"checkpoints/{payload['metadata']['name']}.pt",
            "Checkpoint filename/endpoint role differs",
        )
        _need(
            (state is None) is (point[2] == "parent_initial"),
            "Fitted endpoint was mislabeled as an initial actor",
        )
        if state is None:
            _need(
                binding["fitting_evidence"] is None and binding["fit_tape"] is None,
                "Initial actor has fabricated fit evidence",
            )
            continue
        _need(
            state.get("schema") == "walker-adam-proximal-fit-v1"
            and state["policy_hash"] == owner
            and state["parameter_names"] == expected_names,
            "Optimizer actor/parameter ownership",
        )
        parent_hash = state["parent_policy_hash"]
        name = payload["metadata"]["name"]
        match = re.fullmatch(
            r"s([0-2])_(parent|independent_child|selector_only_child|proposal_guided_child)", name
        )
        _need(match is not None, "Unexpected fitted endpoint name")
        expected_parent_name = f"s{match[1]}_parent" + ("_initial" if match[2] == "parent" else "")
        parents = [
            row
            for row in verifier.record_payloads.values()
            if row["policy_hash"] == parent_hash and row["metadata"]["name"] == expected_parent_name
        ]
        _need(len(parents) == 1, "Fit parent endpoint checkpoint missing or ambiguous")
        parent_payload = parents[0]
        incoming = parent_payload["optimizer_state"]
        _need(
            state["anchor_policy_hash"] == parent_hash == _named_tensor_hash(state["anchor"]),
            "New fit block has wrong proximal anchor owner",
        )
        for tensor_name, tensor in state["anchor"].items():
            _same(
                tensor.detach().cpu().numpy(),
                parent_payload["state_dict"][tensor_name].detach().cpu().numpy(),
                "Anchor differs from exact incoming policy",
            )
        for tensor_name in ("observation_mean", "observation_std"):
            _same(
                payload["state_dict"][tensor_name].detach().cpu().numpy(),
                parent_payload["state_dict"][tensor_name].detach().cpu().numpy(),
                "Training normalization was changed",
            )
        _same(
            payload["private_rng_state"].numpy(),
            parent_payload["private_rng_state"].numpy(),
            "Private Torch RNG did not continue exactly",
        )
        private_rng_checks += 1
        stream = recipe["streams"][int(match[1])]
        _need(payload["seed"] == stream["policy_seed"], "Fresh policy seed owner differs")
        cumulative = 2000 if incoming is None else incoming["updates"] + 2000
        _need(
            state["updates"] == cumulative and state["fit_updates"] == 2000,
            "Adam cumulative versus fit-block update count differs",
        )
        config = dict(
            batch_size=256,
            lr=0.0003,
            eta_theta=0.0,
            gradient_clip=0.5,
            prox_reduction="mean",
            anchor_refresh=0,
        )
        _need(state["fit_config"] == config, "Unexpected learner objective or optimizer settings")
        adam = state["adam"]
        _need(
            set(adam["state"]) == set(range(6)) and len(adam["param_groups"]) == 1,
            "Adam tensor roster differs",
        )
        _need(
            adam["param_groups"][0]["params"] == list(range(6))
            and adam["param_groups"][0]["lr"] == 0.0003,
            "Adam parameter order/lr differs",
        )
        for key, expected in recipe["learning"]["adam"].items():
            actual = adam["param_groups"][0].get(key)
            if isinstance(actual, tuple):
                actual = list(actual)
            _need(actual == expected, "Frozen Adam setting differs: " + key)
        for index, key in enumerate(expected_names):
            entry = adam["state"][index]
            _need(float(entry["step"]) == cumulative, "Adam step continuation differs")
            for moment in ("exp_avg", "exp_avg_sq"):
                array = entry[moment].detach().cpu().numpy()
                _need(
                    array.shape == tuple(payload["state_dict"][key].shape)
                    and np.isfinite(array).all(),
                    "Nonfinite/misbound Adam moments",
                )
                if moment == "exp_avg_sq":
                    _need(np.all(array >= 0), "Negative Adam second moments")
        evidence_path = (verifier.run / binding["fitting_evidence"]).resolve()
        _need(
            evidence_path.is_relative_to(verifier.run)
            and evidence_path.is_file()
            and evidence_path.stat().st_size <= 4 * 1024**2,
            "Missing or unbounded fit evidence",
        )
        evidence = _json(evidence_path.read_bytes())
        _need(name in checker.fit_initial_states, "Fit incoming-state observation is missing")
        observed_initial = checker.fit_initial_states[name]
        _need(
            observed_initial["parent_hash"] == parent_hash
            and observed_initial["policy_state_sha256"]
            == optimizer_fingerprint(parent_payload["state_dict"]),
            "Actual fitting began from a different policy tensor tree",
        )
        _need(
            evidence["initial_policy_state_sha256"] == observed_initial["policy_state_sha256"],
            "Initial tensor hash differs between fit observer and fit artifact",
        )
        expected_initial_rng = (
            np.random.Generator(np.random.PCG64(stream["fit_seed"])).bit_generator.state
            if incoming is None
            else incoming["rng"]
        )
        expected_initial_adam = (
            {"state": {}, "param_groups": adam["param_groups"]}
            if incoming is None
            else incoming["adam"]
        )
        _need(
            observed_initial["rng"] == expected_initial_rng
            and observed_initial["adam_sha256"] == optimizer_fingerprint(expected_initial_adam)
            and observed_initial["private_rng_sha256"]
            == optimizer_fingerprint(parent_payload["private_rng_state"]),
            "Actual fitting restarted or changed incoming Adam/private RNG",
        )
        _need(
            observed_initial["expected_tape"] == binding["fit_tape"] == evidence["sampled_indices"],
            "Incoming expected tape and endpoint tape have different raw owners",
        )
        _need(
            evidence["fit_config"] == config
            and evidence["fit_data"] == state["fit_data"]
            and evidence["child_policy_hash"] == owner,
            "Fit evidence/data/actor join differs",
        )
        if match[2] == "parent":
            expected_data = recipe["inputs"]["train"]["fit_data"]
        else:
            _need(
                name + "/incoming" in checker.metric_datasets,
                "Child dataset has no incoming-parent metric binding",
            )
            dataset = checker.metric_datasets[name + "/incoming"]
            _need(
                dataset["actor_hash"] == parent_hash,
                "Incoming child metric used a different parent",
            )
            observations, targets = (
                checker._input(dataset["observations"]),
                checker._input(dataset["targets"]),
            )
            _need(
                observations.shape == (4000, 17) and targets.shape == (4000, 6),
                "Changed child row budget",
            )
            expected_data = dict(
                observations_sha256=_fit_data_hash(observations),
                actions_sha256=_fit_data_hash(targets),
                rows=4000,
            )
            reference = dataset["observations"].get("reference")
            _need(
                reference is not None and reference == dataset["targets"].get("reference"),
                "Child observations and targets have different raw owners",
            )
            labels = checker.archive.array(reference, "row_episode")
            steps = checker.archive.array(reference, "row_step")
            arm = match[2].removesuffix("_child")
            expected_labels = [f"teacher_s{match[1]}_{arm}_{index:02d}" for index in range(4)]
            for index, label in enumerate(expected_labels):
                _need(
                    label in checker.episodes
                    and checker.episodes[label]["success"] is True
                    and checker.episodes[label]["steps"] == 1000,
                    "Child used an unqualified/replaced teacher",
                )
                start, stop = index * 1000, (index + 1) * 1000
                _need(np.all(labels[start:stop] == label), "Child teacher-source rows changed")
                _same(steps[start:stop], np.arange(1000), "Child control indices lost/reordered")
                source = checker.episodes[label]["raw"]
                _same(
                    observations[start:stop],
                    checker.archive.array(source, "observations"),
                    "Child observation rows differ from complete acquisition",
                )
                _same(
                    targets[start:stop],
                    checker.archive.array(source, "actions"),
                    "Child target rows differ from executed teacher controls",
                )
        _need(
            state["fit_data"] == expected_data,
            "Fit arrays differ from their fixed source/control targets",
        )
        _need(
            evidence["incoming_optimizer_sha256"] == optimizer_fingerprint(incoming),
            "Incoming Adam/RNG owner differs",
        )
        _need(
            evidence["initial_adam_sha256"]
            == optimizer_fingerprint({"state": {}, "param_groups": adam["param_groups"]})
            if incoming is None
            else evidence["initial_adam_sha256"] == optimizer_fingerprint(incoming["adam"]),
            "Initial Adam moments differ from parent continuation",
        )
        _need(
            evidence["initial_private_rng_sha256"]
            == optimizer_fingerprint(parent_payload["private_rng_state"]),
            "Initial private Torch RNG evidence differs",
        )
        _need(
            evidence["resume_fit"] is False
            and evidence["completed_updates"] == 2000
            and evidence["anchor_refreshes"] == [],
            "Changed data reused a block or changed update budget",
        )
        _need(evidence["expected_terminal_rng"] == state["rng"], "Fit terminal RNG owner differs")
        curve = evidence["learning_curve"]
        _need(
            len(curve) == 2000
            and all(
                row["update"] == cumulative - 2000 + index + 1 and row["fit_update"] == index + 1
                for index, row in enumerate(curve)
            ),
            "Fit curve lost or changed updates",
        )
        actual_updates = checker.fit_updates.get(name, [])
        _need(len(actual_updates) == 2000, "Actual Adam-update events are incomplete")
        for work_key, count in (
            ("neural_rows", 512000),
            ("backward_rows", 512000),
            ("optimizer_steps", 2000),
        ):
            _need(
                evidence["work"][work_key]
                == dict(requested=count, completed=count, unknown_partial=0),
                "Fit work census differs",
            )
            _need(
                checker.fit_measurements.get(name, {}).get(work_key) == count,
                "Fit measurements lack matching pre-call requests",
            )
        tape = checker.archive.array(binding["fit_tape"], "indices")
        _need(
            tape.shape == (2000, 256) and tape.dtype == np.int64, "Minibatch index tape interface"
        )
        generator = np.random.Generator(np.random.PCG64(stream["fit_seed"]))
        if incoming is not None:
            generator.bit_generator.state = incoming["rng"]
        for row in tape:
            _same(row, generator.integers(0, 4000, size=256), "Paired minibatch stream differs")
        for index, update in enumerate(actual_updates):
            _same(update["indices"], tape[index], "Actual fit minibatch differs from saved tape")
            _need(update["curve"] == curve[index], "Fit curve differs from actual update event")
        _need(
            generator.bit_generator.state == state["rng"],
            "Minibatch terminal PCG64 continuation differs",
        )
        tape_rows += tape.size
        fit_blocks += 1
    return dict(
        complete_fit_blocks=fit_blocks,
        replayed_minibatch_indices=tape_rows,
        private_rng_checks=private_rng_checks,
        training_forward_rows=0,
        adam_updates_reproduced=0,
    )


def audit_preparation(run: Path, out: Path, *, max_read_bytes: int = 1024**3) -> dict[str, Any]:
    """Independently hash a frozen preparation without decoding input arrays."""
    import time

    run, out = Path(run).resolve(), Path(out).resolve()
    _need(not out.is_relative_to(run), "Audit output must remain outside the immutable run")
    out.mkdir(parents=True, exist_ok=False)
    started, used, checks = time.perf_counter(), 0, 0
    bindings: dict[str, dict[str, Any]] = {}

    def bind(path: Path, expected: str | None = None, size: int | None = None) -> str:
        nonlocal used, checks
        path = path.resolve()
        _need(path.is_relative_to(run) and path.is_file(), "Missing/escaped preparation file")
        count = path.stat().st_size
        _need(used + count <= max_read_bytes, "Preparation audit read budget exhausted")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                used += len(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        _need(expected is None or actual == expected, "Preparation file hash differs: " + str(path))
        _need(size is None or count == size, "Preparation file size differs")
        checks += 1
        bindings[str(path.relative_to(run))] = dict(sha256=actual, bytes=count)
        return actual

    def load(path: Path) -> Any:
        nonlocal used
        bind(path)
        _need(
            used + path.stat().st_size <= max_read_bytes, "Preparation JSON read budget exhausted"
        )
        data = path.read_bytes()
        used += len(data)
        _need(
            hashlib.sha256(data).hexdigest()
            == bindings[str(path.resolve().relative_to(run))]["sha256"],
            "Preparation metadata changed during decode",
        )
        return _json(data)

    config = load(run / "config.json")
    manifest = load(run / "manifest.json")
    seal = load(run / "input_manifest.json")
    prepared = load(run / "result.json")
    _need(config["schema"] == "walker-teacher-proposal-h5.0-r1", "Wrong prepared study")
    _need(
        config["learning"]["global_torch_seed"] == 2791000000,
        "Missing or changed global Torch seed owner",
    )
    _need(
        manifest["config_sha256"] == bindings["config.json"]["sha256"],
        "Prepared config/source owner differs",
    )
    _need(
        seal["config_sha256"] == bindings["config.json"]["sha256"]
        and seal["source_manifest_sha256"] == bindings["manifest.json"]["sha256"],
        "Preparation input/config/source seal differs",
    )
    for row in manifest["files"]:
        bind(run / "source" / row["file"], row["sha256"])
    _need(
        {row["file"] for row in manifest["files"]}
        == {
            str(path.relative_to(run / "source"))
            for path in (run / "source").rglob("*")
            if path.is_file()
        },
        "Frozen source contains missing or unsealed extra files",
    )
    for name, binding in seal["files"].items():
        if isinstance(binding, str):
            bind(run / name, binding)
        else:
            bind(run / name, binding["sha256"], binding["bytes"])
    frozen_recipe = load(run / "source/configs/studies/teacher_proposal_h5_0.json")
    _need(
        config == frozen_recipe, "Prepared configuration differs from its frozen checked-in recipe"
    )
    for binding in config["inputs"].values():
        recorded = bindings.get(binding["file"])
        _need(
            recorded == dict(bytes=binding["bytes"], sha256=binding["sha256"]),
            "Original corpus/normalization/task binding differs",
        )
    census = config["call_census"]
    episode_model = (2 * (1000 - 1 + 2) * 256 + 4 * 1000) * 48
    episode_neural = ((1000 - 1 + 2) * 256 + 515 * 1000) * 48
    _need(
        config["mppi"]["K"] == 256
        and config["mppi"]["H"] == 48
        and config["mppi"]["initial_iterations"] == 2,
        "Changed search schedule",
    )
    _need(
        len(config["streams"]) == 3
        and config["arms"] == ["independent", "selector_only", "proposal_guided"],
        "Changed control/stream roster",
    )
    _need(
        census["per_episode"]["model_controls"] == episode_model
        and census["per_episode"]["guidance_forward_rows"] == episode_neural,
        "First/subsequent decision census differs",
    )
    expected = dict(
        model_controls=54 * episode_model,
        live_controls=456000,
        requested_controls=54 * episode_model + 456000,
        physics_substeps=4 * (54 * episode_model + 456000),
        neural_rows=54 * episode_neural + 6144000 + 162000 + 54000 + 402000,
        backward_rows=6144000,
        optimizer_steps=24000,
        audit_neural_rows=54 * episode_neural + 162000 + 54000 + 402000,
    )
    for key, value in expected.items():
        _need(config["budgets"][key] == value, "Independent census differs: " + key)
    _need(
        config["budgets"]["total_worker_and_audit_neural_rows"]
        == expected["neural_rows"] + expected["audit_neural_rows"],
        "Audit was omitted from total neural work",
    )
    _need(
        prepared["preparation_complete"] is True
        and prepared["execution_complete"] is False
        and prepared["work"]
        == dict(network_forwards=0, physics_controls=0, optimizer_steps=0, mppi_queries=0),
        "Preparation receipt claims paid research work",
    )
    _need(
        config["eligibility"]["useful_learning_prerequisite_satisfied"] is False,
        "Unmet useful-learning prerequisite was silently cleared",
    )
    for filename in ("launch_receipt.json", "call_census.json"):
        if filename not in bindings:
            bind(run / filename)
    receipt = dict(
        schema="walker-h5.0-preparation-audit-v1",
        preparation_audit_complete=True,
        scientific_round_audit_complete=False,
        checks=checks,
        source_read_bytes=used,
        array_decodes=0,
        new_network_rows=0,
        new_physics_controls=0,
        new_optimizer_steps=0,
        launch_authorized=config["launch_authorized"],
        eligibility_decision=config["eligibility"]["decision"],
        wall_seconds=time.perf_counter() - started,
        bindings=bindings,
    )
    with (out / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def _outcome(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    _need(bool(episodes), "Empty cohort has no independently measured outcome")
    return dict(
        episodes=len(episodes),
        successes=sum(row["success"] for row in episodes),
        survived=sum(row["survived"] for row in episodes),
        controls=sum(row["steps"] for row in episodes),
        mean_cost=sum(row["cost"] for row in episodes) / len(episodes),
        costs=[row["cost"] for row in episodes],
        complete=all(row["complete"] for row in episodes),
    )


def _metric_roster(checker: RoundChecks, *, complete_learning: bool) -> None:
    config = checker.recipe
    by_file = {row["file"]: row["actor_hash"] for row in checker.checkpoints}
    expected = {}
    for stream in config["streams"]:
        sid = stream["id"]
        for suffix in ("parent_initial", "parent"):
            point = f"s{sid}_{suffix}"
            for label in ("train", "validation"):
                filename = config["inputs"][label]["file"]
                expected[point + "/" + label] = dict(
                    actor_hash=by_file[f"checkpoints/{point}.pt"],
                    observations=dict(file=filename, key="observations"),
                    targets=dict(file=filename, key="actions"),
                )
        if complete_learning:
            for arm in ("independent", "selector_only", "proposal_guided"):
                point = f"s{sid}_{arm}_child"
                incoming_key = point + "/incoming"
                endpoint_key = point + "/endpoint"
                _need(
                    incoming_key in checker.metric_datasets
                    and endpoint_key in checker.metric_datasets,
                    "Missing fixed new-data metric endpoints",
                )
                incoming = checker.metric_datasets[incoming_key]
                endpoint = checker.metric_datasets[endpoint_key]
                _need(
                    incoming["observations"] == endpoint["observations"]
                    and incoming["targets"] == endpoint["targets"],
                    "Child reconstruction endpoints use different data",
                )
                _need(
                    "reference" in incoming["observations"]
                    and incoming["observations"]["key"] == "observations"
                    and incoming["targets"]["key"] == "actions"
                    and incoming["observations"]["reference"]
                    == incoming["targets"].get("reference"),
                    "New-data metric lost aligned source targets",
                )
                for ending, owner in (
                    ("incoming", by_file[f"checkpoints/s{sid}_parent.pt"]),
                    ("endpoint", by_file[f"checkpoints/{point}.pt"]),
                ):
                    expected[point + "/" + ending] = dict(
                        actor_hash=owner,
                        observations=incoming["observations"],
                        targets=incoming["targets"],
                    )
                for label in ("train", "validation"):
                    filename = config["inputs"][label]["file"]
                    expected[point + "/" + label] = dict(
                        actor_hash=by_file[f"checkpoints/{point}.pt"],
                        observations=dict(file=filename, key="observations"),
                        targets=dict(file=filename, key="actions"),
                    )
    _need(checker.metric_datasets == expected, "Full-data metric tag/source/actor roster differs")


def verify_outcome_roster(checker: RoundChecks, result: dict[str, Any]) -> dict[str, Any]:
    """Recompute full teacher/student panels and finite admission independently."""

    config = checker.recipe
    arms = ("independent", "selector_only", "proposal_guided")
    expected_tags: set[str] = set()

    def cohort(tags: list[str], seeds: list[int]) -> list[dict[str, Any]]:
        _need(len(tags) == len(seeds), "Fixed reset roster differs")
        rows = []
        for tag, seed in zip(tags, seeds, strict=True):
            _need(
                tag in checker.episodes and tag not in expected_tags,
                "Missing/reused episode slot: " + tag,
            )
            row = checker.episodes[tag]
            _need(
                row["seed"] == seed and row["complete"] is True,
                "Episode/reset slot ownership differs",
            )
            expected_tags.add(tag)
            rows.append(row)
        return rows

    def pair(first: dict, second: dict) -> None:
        _need(first["seed"] == second["seed"], "Unpaired reset seeds")
        for key in ("states", "warmstarts", "integrations", "observations"):
            _same(
                checker.archive.array(first["raw"], key)[0],
                checker.archive.array(second["raw"], key)[0],
                "Paired full native reset context differs",
            )

    all_walk, teacher_admission = True, {}
    for stream in config["streams"]:
        sid = str(stream["id"])
        rows = {
            arm: cohort(
                [f"teacher_s{sid}_{arm}_{index:02d}" for index in range(4)],
                stream["acquisition_seeds"],
            )
            for arm in arms
        }
        summaries = {arm: _outcome(rows[arm]) for arm in arms}
        _need(summaries == result["teacher_summaries"][sid], "Complete teacher outcomes differ")
        for arm in arms[1:]:
            for baseline, guided in zip(rows["independent"], rows[arm], strict=True):
                pair(baseline, guided)
        all_walk &= all(
            row["success"] and row["survived"] and row["steps"] == 1000
            for values in rows.values()
            for row in values
        )
        base = summaries["independent"]["mean_cost"]
        teacher_admission[sid] = {
            arm: dict(
                walking=all(row["success"] for row in rows[arm]),
                cost=summaries[arm]["mean_cost"] <= 1.05 * base,
            )
            for arm in arms
        }
    _need(
        teacher_admission == result["teacher_admission"],
        "Teacher competence/cost admission differs",
    )
    _metric_roster(checker, complete_learning=bool(all_walk))
    teacher_tags = {
        tag
        for tag in checker.episodes
        if tag.startswith("teacher_") or tag.startswith("gap_teacher_")
    }
    _need(
        checker.teacher_posthoc_tags == teacher_tags,
        "Some teacher episodes lack exactly one complete posthoc measurement",
    )
    actual_teacher_controls = sum(checker.episodes[tag]["steps"] for tag in teacher_tags)
    _need(
        checker.selection_checks == checker.teacher_controls == actual_teacher_controls,
        "Some teacher controls lack one verified current selection",
    )
    if not all_walk:
        _need(
            result["status"] == "teacher_competence_negative_no_child_fits"
            and result["comparison_complete"] is False
            and result["confirmed_benefit"] is False,
            "Failed teacher was promoted or partial learning contrast claimed",
        )
        _need(
            len(checker.checkpoints) == 6 and len(checker.fit_updates) == 3,
            "Unexpected child fit after teacher competence stop",
        )
        _need(
            set(checker.episodes) == expected_tags,
            "Unrequested replacement/evaluation/gap episode after teacher stop",
        )
        return dict(
            teacher_complete=True,
            learning_contrast_measured=False,
            standalone_panels_measured=False,
            confirmed_benefit=False,
            loop_closed=False,
        )
    panel_results = {}
    for panel, seeds in (
        ("primary", config["primary_seeds"]),
        ("confirmation", config["confirmation_seeds"]),
    ):
        panel_results[panel] = {}
        for stream in config["streams"]:
            sid = str(stream["id"])
            rows = {
                arm: cohort([f"{panel}_s{sid}_{arm}_{index:02d}" for index in range(16)], seeds)
                for arm in ("parent", *arms)
            }
            summary = {arm: _outcome(values) for arm, values in rows.items()}
            _need(summary == result["evaluations"][panel][sid], "Fresh standalone outcomes differ")
            for arm in arms:
                for parent, child in zip(rows["parent"], rows[arm], strict=True):
                    pair(parent, child)
            panel_results[panel][sid] = summary
    for stream in config["streams"]:
        sid = str(stream["id"])
        for arm in arms:
            teacher = cohort(
                [f"gap_teacher_s{sid}_{arm}_{index:02d}" for index in range(2)], stream["gap_seeds"]
            )
            student = cohort(
                [f"gap_s{sid}_{arm}_{index:02d}" for index in range(2)], stream["gap_seeds"]
            )
            for first, second in zip(teacher, student, strict=True):
                pair(first, second)
            expected = dict(
                teacher=_outcome(teacher),
                student=_outcome(student),
                child_minus_teacher_cost=[
                    second["cost"] - first["cost"]
                    for first, second in zip(teacher, student, strict=True)
                ],
                seeds=stream["gap_seeds"],
                scope="Two fresh exactly paired contexts; separate from primary panels",
            )
            _need(
                expected == result["paired_teacher_student_gap"][sid][arm],
                "Paired performance gap differs",
            )
    passes = {}
    for panel, streams in panel_results.items():
        positive = safe = 0
        for actors in streams.values():
            child = actors["proposal_guided"]
            positive += (
                child["successes"] - actors["parent"]["successes"] >= 2
                and child["successes"] - actors["selector_only"]["successes"] >= 2
            )
            safe += all(
                child[key] >= actors[arm][key]
                for arm in ("parent", "selector_only", "independent")
                for key in ("successes", "survived")
            )
        passes[panel] = positive >= 2 and safe == 3
        _need(
            result[panel + "_gate"]["passed"] is passes[panel], "Walking/nonregression gate differs"
        )
    acquisition_teacher_pass = all(
        all(criteria.values())
        for by_arm in teacher_admission.values()
        for criteria in by_arm.values()
    )
    gap_teacher_pass = all(
        value["teacher"]["successes"] == 2
        and value["teacher"]["survived"] == 2
        and value["teacher"]["mean_cost"]
        <= 1.05 * per_stream["independent"]["teacher"]["mean_cost"]
        for per_stream in result["paired_teacher_student_gap"].values()
        for value in per_stream.values()
    )
    teacher_pass = acquisition_teacher_pass and gap_teacher_pass
    confirmed = passes["primary"] and passes["confirmation"] and teacher_pass
    _need(
        result["confirmed_benefit"] is confirmed
        and result["complete_teacher_gate"] is teacher_pass,
        "Useful-learning admission differs",
    )
    _need(
        result["acquisition_teacher_gate"] is acquisition_teacher_pass
        and result["gap_teacher_gate"] is gap_teacher_pass,
        "Whole teacher admission omitted acquisition or gap outcomes",
    )
    competence = {
        sid: {
            arm: all(
                panel_results[panel][sid][arm]["successes"] >= 15
                for panel in ("primary", "confirmation")
            )
            for arm in ("parent", *arms)
        }
        for sid in panel_results["primary"]
    }
    _need(
        result["individual_competence"] == competence
        and result["second_round_review_may_be_prepared"] is confirmed,
        "Competence or future-review status differs",
    )
    _need(
        set(checker.episodes) == expected_tags and len(expected_tags) == 456,
        "Full roster contains missing/replaced/extra episodes",
    )
    _need(
        len(checker.checkpoints) == 15 and len(checker.fit_updates) == 12,
        "Fixed parent/child checkpoint/fit roster differs",
    )
    _need(
        result["next_round_eligible"] is False and result["loop_closed"] is False,
        "One contrast was promoted to a loop claim",
    )
    return dict(
        teacher_complete=True,
        learning_contrast_measured=True,
        standalone_panels_measured=True,
        confirmed_benefit=confirmed,
        loop_closed=False,
    )


def audit_round(run: Path, out: Path, *, max_read_bytes: int = 3 * 1024**4) -> dict[str, Any]:
    """Execute the future full saved-only/prediction audit, never native replay."""
    import time

    run, out = Path(run).resolve(), Path(out).resolve()
    _need(not out.is_relative_to(run), "Audit output must remain outside immutable evidence")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    started_monotonic = time.monotonic()
    verifier = archive = None
    journal = AuditJournal(out / "work.jsonl", max_bytes=3 * 1024**3 // 2)
    try:
        preparation = audit_preparation(run, out / "preparation")
        config = _json((run / "config.json").read_bytes())
        result = _json((run / "worker_result.json").read_bytes())
        _need(
            result.get("execution_complete") is True and not result.get("error"),
            "Only a completed bounded round can pass full scientific audit",
        )
        deadline = started_monotonic + config["budgets"]["audit_wall_seconds"]
        verifier = SavedActorVerifier(
            result["actor_registry"],
            run,
            max_neural_rows=config["budgets"]["audit_neural_rows"],
            max_scope_neural_rows={
                **{arm: 666527184 for arm in config["arms"]},
                "shared_parent": 132000,
            },
            deadline=deadline,
            event_sink=journal.record,
        )
        with StreamArchive(
            run / "evidence",
            out / "array-index.sqlite",
            max_read_bytes=max_read_bytes,
            max_database_bytes=1024**3,
            deadline=deadline,
        ) as archive:
            archive.verify_index()
            checker = RoundChecks(archive, config, verifier, run=run)
            structure = audit_stream_structure(archive, event_observer=checker.event)
            scientific = checker.finish()
            fitting = verify_fitting_ownership(verifier, checker, config)
            outcomes = verify_outcome_roster(checker, result)
            for key, count in structure["work"].items():
                _need(
                    result["work"].get(key) == count,
                    "Worker result disagrees with paid ledger work",
                )
            training_forwards = sum(
                value.get("neural_rows", 0) for value in checker.fit_measurements.values()
            )
            _need(
                scientific["independently_verified_prediction_rows"]
                == structure["work"].get("requested_neural_rows", 0) - training_forwards,
                "Some saved nontraining policy forwards lack independent replay",
            )
            _need(
                verifier.neural_rows == scientific["independently_verified_prediction_rows"],
                "Audit neural-row accounting differs",
            )
            _need(
                not result["reservation_census"]["pending"]
                and not result["reservation_census"]["stopped"]
                and not any(result["reservation_census"]["errors"].values()),
                "Worker reservation guard has unclosed/failed work",
            )
            archive.verify_unchanged()
            _need(time.monotonic() <= deadline, "Audit wall-time reservation exhausted")
            receipt = dict(
                schema="walker-h5.0-round-audit-v1",
                scientific_round_audit_complete=True,
                prediction_replay_complete=True,
                preparation=preparation,
                structure=structure,
                scientific=scientific,
                fitting=fitting,
                outcomes=outcomes,
                source_read_bytes=archive.read_bytes,
                checkpoint_read_bytes=verifier.checkpoint_read_bytes,
                frozen_input_read_bytes=checker.input_read_bytes,
                audit_neural_rows=verifier.neural_rows,
                audit_prediction_calls=verifier.prediction_calls,
                audit_checkpoint_loads=verifier.checkpoint_loads,
                audit_completed_neural_rows=verifier.completed_neural_rows,
                audit_unknown_partial_neural_rows=verifier.unknown_partial_neural_rows,
                audit_scope_requested_neural_rows=verifier.scope_neural_rows,
                audit_scope_completed_neural_rows=verifier.scope_completed_neural_rows,
                audit_scope_unknown_partial_neural_rows=verifier.scope_unknown_partial_neural_rows,
                journal_bytes=journal.bytes,
                journal_events=journal.events,
                peak_rss_bytes=_memory_boundary(),
                file_hashes=archive.file_hashes,
                new_native_controls=0,
                new_optimizer_steps=0,
                wall_seconds=time.perf_counter() - started,
            )
        with (out / "receipt.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return receipt
    except BaseException as error:
        failure = dict(
            schema="walker-h5.0-round-audit-failure-v1",
            scientific_round_audit_complete=False,
            error=dict(type=type(error).__name__, message=str(error)),
            source_read_bytes=0 if archive is None else archive.read_bytes,
            audit_requested_neural_rows=0 if verifier is None else verifier.neural_rows,
            audit_completed_neural_rows=0 if verifier is None else verifier.completed_neural_rows,
            audit_unknown_partial_neural_rows=0
            if verifier is None
            else verifier.unknown_partial_neural_rows,
            checkpoint_loads=0 if verifier is None else verifier.checkpoint_loads,
            audit_scope_requested_neural_rows={}
            if verifier is None
            else verifier.scope_neural_rows,
            audit_scope_completed_neural_rows={}
            if verifier is None
            else verifier.scope_completed_neural_rows,
            audit_scope_unknown_partial_neural_rows={}
            if verifier is None
            else verifier.scope_unknown_partial_neural_rows,
            journal_bytes=journal.bytes,
            wall_seconds=time.perf_counter() - started,
        )
        with (out / "failure.json").open("x") as stream:
            json.dump(failure, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    finally:
        journal.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("preparation", "round"), default="preparation")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return audit_worker(args.run, args.out)
    if args.mode == "round":
        return launch_round_audit(args.run, args.out)
    operation = audit_preparation if args.mode == "preparation" else audit_round
    result = operation(args.run, args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "bindings"}, indent=2))
    return 0


def _file_sha(path: Path) -> str:
    _need(
        path.is_file() and path.stat().st_size <= 16 * 1024**2,
        "Missing or unbounded audit ownership file",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_runtime(config: dict[str, Any]) -> None:
    import importlib.metadata
    import platform
    import socket

    profile = config["runtime"]
    _need(
        sys.platform == "linux" and socket.gethostname() == profile["host"],
        "Audit host differs from pinned native owner",
    )
    _need(
        sys.version == profile["python"] and platform.platform() == profile["platform"],
        "Audit Python/platform differs",
    )
    for name, version in profile["distributions"].items():
        _need(
            importlib.metadata.version(name) == version,
            "Audit distribution version differs: " + name,
        )
    _need(sorted(os.sched_getaffinity(0)) == profile["cpu_affinity"], "Audit CPU affinity differs")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        _need(os.environ.get(name) == profile[name], "Audit process environment differs: " + name)


def audit_worker(run: Path, out: Path) -> int:
    """Authenticated numerical child; marker verification precedes Torch imports."""
    run, out = Path(run).resolve(), Path(out).resolve()
    marker = _json((out / "ownership.json").read_bytes())
    _need(
        os.getppid() == marker["coordinator_pid"]
        and os.environ.get("H5_AUDIT_NONCE") == marker["nonce"],
        "Audit worker lacks a live authenticated coordinator",
    )
    for name, key in (
        ("config.json", "config_sha256"),
        ("manifest.json", "source_manifest_sha256"),
        ("input_manifest.json", "input_manifest_sha256"),
        ("worker_result.json", "worker_result_sha256"),
    ):
        _need(_file_sha(run / name) == marker[key], "Audit worker ownership changed: " + name)
    _need(
        _file_sha(Path(__file__).resolve()) == marker["auditor_sha256"],
        "Audit worker source differs from frozen auditor",
    )
    config = _json((run / "config.json").read_bytes())
    _audit_runtime(config)
    result = audit_round(run, out / "result")
    print(
        json.dumps(
            dict(
                scientific_round_audit_complete=result["scientific_round_audit_complete"],
                audit_neural_rows=result["audit_neural_rows"],
            )
        )
    )
    return 0


def launch_round_audit(run: Path, out: Path) -> int:
    """Bound the whole future audit process group; no retries or native replay."""
    import secrets

    from ..proposal_launch import bounded_process

    run, out = Path(run).resolve(), Path(out).resolve()
    _need(not out.is_relative_to(run), "Audit launch output must be outside the immutable run")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = _json((run / "config.json").read_bytes())
    _need(sys.platform == "linux", "Round audit requires the pinned Linux host")
    previous = os.sched_getaffinity(0)
    affinity = config["runtime"]["cpu_affinity"]
    _need(set(affinity).issubset(previous), "Audit CPUs unavailable")
    auditor = run / "source/src/gps/mppi_gps/audits/teacher_proposal.py"
    marker = dict(
        coordinator_pid=os.getpid(),
        nonce=secrets.token_hex(32),
        config_sha256=_file_sha(run / "config.json"),
        source_manifest_sha256=_file_sha(run / "manifest.json"),
        input_manifest_sha256=_file_sha(run / "input_manifest.json"),
        worker_result_sha256=_file_sha(run / "worker_result.json"),
        auditor_sha256=_file_sha(auditor),
        watchdog_wall_seconds=config["budgets"]["audit_wall_seconds"],
        peak_rss_bytes=config["budgets"]["audit_peak_rss_bytes"],
    )
    with (out / "ownership.json").open("x") as stream:
        json.dump(marker, stream, indent=2, allow_nan=False)
        stream.write("\n")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", H5_AUDIT_NONCE=marker["nonce"])
    environment.pop("PYTHONPATH", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        environment[name] = config["runtime"][name]
    command = [
        config["locations"]["remote_interpreter"],
        "-B",
        str(run / "source/scripts/audits/teacher_proposal.py"),
        "--mode",
        "round",
        "--worker",
        "--run",
        str(run),
        "--out",
        str(out),
    ]
    os.sched_setaffinity(0, affinity)
    try:
        receipt = bounded_process(
            command,
            cwd=run / "source",
            environment=environment,
            output=out / "stdout.txt",
            wall_seconds=max(
                0.001, config["budgets"]["audit_wall_seconds"] - (time.monotonic() - started)
            ),
            stdout_bytes=64 * 1024**2,
            peak_rss_bytes=config["budgets"]["audit_peak_rss_bytes"],
        )
    except BaseException as error:
        receipt = getattr(
            error,
            "process_receipt",
            dict(process_started=False, error=dict(type=type(error).__name__, message=str(error))),
        )
        with (out / "exit.json").open("x") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    finally:
        os.sched_setaffinity(0, previous)
    with (out / "exit.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt["exit_code"] if receipt["exit_code"] >= 0 else 1
