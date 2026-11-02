"""Bounded H5 evidence adapter; historical HDF5 Meter bytes stay unchanged.

``walker-array-stream-v1`` stores exact NumPy C bytes in one append-only file.
The index durably records intent before payload and completion after fsync.
Offsets, lengths, dtype, shape and C-byte SHA256 are explicit. Metadata has
finite line/array-count limits, so a byte census does not assume HDF5 overhead
or compression. This is a Meter-compatible storage boundary, not a new model.
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import BudgetExhausted, EvidenceError, Meter, _amount, _dumps

FORMAT = "walker-array-stream-v1"
COUNTERS = (
    "requested_controls completed_controls requested_model_controls completed_model_controls "
    "requested_live_controls completed_live_controls requested_physics_substeps "
    "completed_physics_substeps unknown_partial_controls unknown_partial_physics_substeps "
    "model_queries live_queries failed_queries rejected_queries measurement_requests "
    "failed_measurements rejected_measurements requested_neural_rows completed_neural_rows "
    "unknown_partial_neural_rows requested_backward_rows completed_backward_rows "
    "unknown_partial_backward_rows requested_optimizer_steps completed_optimizer_steps "
    "unknown_partial_optimizer_steps resets reset_errors restores snapshots episodes"
).split()


class BinaryEvidenceMeter(Meter):
    """Existing query/measurement accounting with explicit evidence byte limits.

    ``reserved_bytes`` protects a declared allowance for source/checkpoints or
    other files outside this meter's create-only directory. The owner must
    enforce that separate allowance; it is not a claim to inspect those files.
    Eight bounded emergency ledger lines are reserved by default for failure
    and closure records. No quota rejection writes a partial intended group.
    OS write failures retain partial intent/payload and poison further writes.
    """

    def __init__(
        self,
        out: Path,
        *,
        max_controls: int,
        max_neural_rows: int,
        max_bytes: int,
        reserved_bytes: int = 0,
        max_event_bytes: int = 16384,
        max_header_bytes: int = 16384,
        max_array_bytes: int = 16 * 1024**2,
        max_arrays_per_group: int = 32,
        emergency_events: int = 8,
    ):
        self.max_controls, self.max_neural_rows = _amount(max_controls), _amount(max_neural_rows)
        self.max_bytes, self.reserved_bytes = _amount(max_bytes), _amount(reserved_bytes)
        self.max_event_bytes, self.max_header_bytes = (
            _amount(max_event_bytes),
            _amount(max_header_bytes),
        )
        self.max_array_bytes = _amount(max_array_bytes)
        self.max_arrays_per_group = _amount(max_arrays_per_group)
        self.emergency_bytes = _amount(emergency_events) * self.max_event_bytes
        self.storage_limit = self.max_bytes - self.reserved_bytes
        if (
            min(self.max_event_bytes, self.max_header_bytes, self.max_array_bytes) < 1
            or self.max_event_bytes < 2048
            or self.max_arrays_per_group < 1
            or emergency_events < 2
            or self.storage_limit <= self.emergency_bytes + self.max_event_bytes
        ):
            raise ValueError("Positive bounded metadata/payload and closure space required.")
        self.out = self.path = Path(out)
        self.out.mkdir(parents=True, exist_ok=False)
        self.ledger_path = self.out / "ledger.jsonl"
        self.raw_path = self.out / "arrays.bin"
        self.index_path = self.out / "arrays.index.jsonl"
        self._raw = self.raw_path.open("xb", buffering=0)
        self._index = self.index_path.open("xb", buffering=0)
        self._ledger = self.ledger_path.open("xb", buffering=0)
        self._lock = threading.RLock()
        self._sequence = self._array_sequence = self._query_sequence = 0
        self._started = time.perf_counter()
        self.closed, self.context = False, {}
        self._counts = dict.fromkeys(COUNTERS, 0)
        self._storage_bytes = 0
        self._poisoned = False
        self._recent_context: str | None = None
        self._recent_model: list[dict] = []
        self._pending_model: dict[str, dict] = {}
        try:
            self.record(
                "runtime_open",
                schema="mppi-gps-runtime-v1",
                array_format=FORMAT,
                max_controls=max_controls,
                max_neural_rows=max_neural_rows,
                max_bytes=max_bytes,
                reserved_external_bytes=reserved_bytes,
                max_event_bytes=max_event_bytes,
                max_header_bytes=max_header_bytes,
                max_array_bytes=max_array_bytes,
                max_arrays_per_group=max_arrays_per_group,
                emergency_bytes=self.emergency_bytes,
                raw_file=self.raw_path.name,
                array_index=self.index_path.name,
            )
        except BaseException:
            self._raw.close()
            self._index.close()
            self._ledger.close()
            self.closed = True
            raise

    def _reserve(self, amount: int, *, emergency: bool = False) -> None:
        self._open()
        if self._poisoned:
            raise EvidenceError("Evidence stream is poisoned after a partial OS write.")
        limit = self.storage_limit - (0 if emergency else self.emergency_bytes)
        if self._storage_bytes + amount > limit:
            raise BudgetExhausted("Evidence storage reservation exhausted before write.")

    def _append(self, stream: Any, payload: bytes | memoryview, *, sync: bool = True) -> None:
        value = memoryview(payload).cast("B")
        try:
            while len(value):
                count = stream.write(value)
                if not count:
                    raise OSError("Evidence stream accepted no bytes.")
                self._storage_bytes += count
                value = value[count:]
            if sync:
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            self._poisoned = True
            raise EvidenceError("Partial evidence write retained; stop the worker.") from error

    def _sync_context(self) -> None:
        key = _dumps(self.context)
        if key != self._recent_context:
            self._recent_context = key
            self._recent_model = []
            self._pending_model = {}

    @property
    def recent_model_queries(self) -> list[dict]:
        """Completed model bindings in the current episode/step context only."""
        with self._lock:
            self._sync_context()
            return copy.deepcopy(self._recent_model)

    def record(self, event: str, **fields: Any) -> str:
        with self._lock:
            self._open()
            if (
                not isinstance(event, str)
                or not event
                or set(fields) & {"id", "event", "kind", "wall_seconds"}
            ):
                raise ValueError("Explicit event and nonreserved metadata keys required.")
            identifier = f"event-{self._sequence:09d}"
            row = dict(
                fields,
                id=identifier,
                event=event,
                kind=event,
                wall_seconds=time.perf_counter() - self._started,
            )
            payload = (_dumps(row) + "\n").encode("utf-8")
            if len(payload) > self.max_event_bytes:
                raise BudgetExhausted("Event metadata exceeds its frozen line limit.")
            emergency = event.endswith("_error") or event == "runtime_close"
            self._reserve(len(payload), emergency=emergency)
            self._append(self._ledger, payload)
            self._sequence += 1
            self._sync_context()
            if fields.get("mode") == "model":
                if event == "query_requested":
                    self._pending_model[fields["query_id"]] = copy.deepcopy(fields)
                elif event == "query_complete":
                    request = self._pending_model.pop(fields["query_id"], {})
                    self._recent_model.append(
                        dict(
                            query_id=fields["query_id"],
                            inputs=request.get("inputs"),
                            result=copy.deepcopy(fields.get("result")),
                            context=copy.deepcopy(self.context),
                        )
                    )
                elif event == "query_error":
                    self._pending_model.pop(fields["query_id"], None)
            return identifier

    def save_arrays(self, tag: str, **arrays: Any) -> dict:
        with self._lock:
            self._open()
            if not isinstance(tag, str) or not tag:
                raise ValueError("An explicit array-group tag is required.")
            if len(arrays) > self.max_arrays_per_group:
                raise BudgetExhausted("Array count exceeds the frozen group limit.")
            label = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:100]
            name = f"arrays/a{self._array_sequence:09d}_{label}"
            offset = self._raw.tell()
            refs, packed = {}, []
            for key, value in arrays.items():
                if not isinstance(key, str) or not key or "/" in key:
                    raise ValueError("Array names must be simple nonempty strings.")
                encoding = "array"
                if isinstance(value, dict):
                    value, encoding = np.array(_dumps(value)), "json"
                array = np.array(value, copy=True, order="C")
                if array.dtype.hasobject or array.dtype.fields or array.dtype.kind not in "biufcSU":
                    raise ValueError(
                        "Only explicit numeric/string NumPy C-byte arrays are allowed."
                    )
                if array.nbytes > self.max_array_bytes:
                    raise BudgetExhausted("Array exceeds the frozen payload limit.")
                payload = memoryview(b"") if array.nbytes == 0 else memoryview(array).cast("B")
                refs[key] = dict(
                    sha256=hashlib.sha256(payload).hexdigest(),
                    shape=list(array.shape),
                    dtype=array.dtype.str,
                    encoding=encoding,
                    offset=offset,
                    byte_length=array.nbytes,
                )
                packed.append(array)
                offset += array.nbytes
            reference = dict(
                format=FORMAT,
                file=self.raw_path.name,
                index=self.index_path.name,
                group=name,
                tag=tag,
                arrays=refs,
            )
            intent = (_dumps(dict(event="arrays_requested", **reference)) + "\n").encode()
            complete = (
                _dumps(dict(event="arrays_complete", format=FORMAT, group=name, end_offset=offset))
                + "\n"
            ).encode()
            if max(len(intent), len(complete)) > self.max_header_bytes:
                raise BudgetExhausted("Array metadata exceeds its frozen index-line limit.")
            self._reserve(sum(array.nbytes for array in packed) + len(intent) + len(complete))
            self._append(self._index, intent)
            self._array_sequence += 1
            for array in packed:
                payload = memoryview(b"") if array.nbytes == 0 else memoryview(array).cast("B")
                self._append(self._raw, payload, sync=False)
            self._append(self._raw, b"")
            self._append(self._index, complete)
            return reference

    def totals(self) -> dict:
        return super().totals() | dict(
            array_format=FORMAT,
            storage_bytes=self._storage_bytes,
            storage_limit_bytes=self.storage_limit,
            reserved_external_bytes=self.reserved_bytes,
            emergency_reserved_bytes=self.emergency_bytes,
            evidence_poisoned=self._poisoned,
        )

    def close(self) -> None:
        if not self.closed:
            try:
                if not self._poisoned:
                    self.record("runtime_close", totals=self.totals())
            finally:
                self._raw.close()
                self._index.close()
                self._ledger.close()
                self.closed = True


def read_array(root: Path, reference: dict, key: str, *, max_bytes: int) -> np.ndarray:
    """Bounded engineering reader; scientific audit must verify the index too."""
    if reference.get("format") != FORMAT or reference.get("file") != "arrays.bin":
        raise ValueError("Unsupported array stream format or file.")
    value = reference["arrays"][key]
    offset, length = value["offset"], value["byte_length"]
    if type(offset) is not int or type(length) is not int or min(offset, length) < 0:
        raise ValueError("Explicit nonnegative array byte range required.")
    if length > _amount(max_bytes):
        raise BudgetExhausted("Read budget exhausted before array read.")
    dtype = np.dtype(value["dtype"])
    shape = value["shape"]
    if (
        dtype.hasobject
        or dtype.fields
        or not isinstance(shape, list)
        or any(type(size) is not int or size < 0 for size in shape)
        or int(np.prod(shape, dtype=object)) * dtype.itemsize != length
    ):
        raise ValueError("Array shape/dtype does not match its explicit byte length.")
    path = (Path(root) / "arrays.bin").resolve()
    if not path.is_relative_to(Path(root).resolve()) or offset + length > path.stat().st_size:
        raise EvidenceError("Array range escapes the evidence file.")
    with path.open("rb") as stream:
        stream.seek(offset)
        payload = stream.read(length)
    if hashlib.sha256(payload).hexdigest() != value["sha256"]:
        raise EvidenceError("Array C-byte hash differs.")
    return np.frombuffer(payload, dtype=dtype).reshape(shape)
