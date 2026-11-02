"""C1's bounded durable native prefixes, using the existing array archive.

The temporary spool is one ordinary, non-pickled NumPy structured array. Its
unused capacity is never interpreted as observations. A row becomes committed
only after all returned values and their digest are fsynced, followed by its
separately fsynced validity ordinal. The archive still contains the existing
``walker-array-stream-v1`` groups of thirteen plain named arrays.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from .proposal_evidence import BinaryEvidenceMeter
from .runtime import BudgetExhausted, EvidenceError, _dumps

SPOOL_SCHEMA = "walker-c1-active-prefix-v1"
MAX_SPOOL_BYTES = 8 * 1024**2
MAX_HORIZON = 1000
MAX_INTEGRATION = 128
COMPACT_EVENT_BYTES = 512
UNUSED = np.iinfo(np.uint32).max
TRACE_ARRAY_NAMES = (
    "observations",
    "states",
    "warmstarts",
    "integrations",
    "actions",
    "native_costs",
    "x_velocities",
    "fall_costs",
    "done",
    "control_ordinals",
    "row_digests",
    "initial_digest",
    "metadata",
)


class CompactEvidenceMeter(BinaryEvidenceMeter):
    """Keep old writer contracts while bounding C1's frequent generated rows.

    The two rare lifecycle summaries use the existing 2 KiB minimum. All other
    lines, including generated IDs, event/kind and timing fields, must fit 512
    bytes before any write. Long diagnostics belong in bounded sidecars.
    """

    def __init__(self, out: Path, **kwargs: Any):
        settings = dict(max_event_bytes=2048, max_header_bytes=8192)
        settings.update(kwargs)
        super().__init__(out, **settings)

    def _append(self, stream: Any, payload: bytes | memoryview, *, sync: bool = True) -> None:
        if stream is self._ledger and len(payload) > COMPACT_EVENT_BYTES:
            event = json.loads(bytes(payload))["event"]
            if event not in {"runtime_open", "runtime_close"}:
                raise BudgetExhausted("C1 compact ledger line exceeds 512 bytes before write.")
        super()._append(stream, payload, sync=sync)


def _dtype(integration_size: int) -> np.dtype:
    return np.dtype(
        [
            ("observations", "<f8", (17,)),
            ("states", "<f8", (19,)),
            ("warmstarts", "<f8", (9,)),
            ("integrations", "<f8", (integration_size,)),
            ("actions", "<f8", (6,)),
            ("native_costs", "<f8"),
            ("x_velocities", "<f8"),
            ("fall_costs", "<f8"),
            ("done", "u1"),
            ("control_ordinals", "<u4"),
            ("row_digests", "u1", (32,)),
            ("valid", "<u4"),
            ("prediction_actor", "S80"),
            ("prediction_digest", "u1", (32,)),
            ("prediction_valid", "<u4"),
        ]
    )


def _finite(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype="<f8")
    if result.shape != shape or not np.isfinite(result).all():
        raise EvidenceError(f"Invalid finite C1 {name} shape/value.")
    return np.array(result, copy=True, order="C")


def _boundary(snapshot: dict[str, Any], observation: Any, size: int) -> dict[str, np.ndarray]:
    return dict(
        observations=_finite(observation, (17,), "observation"),
        states=_finite(snapshot["state"], (19,), "state"),
        warmstarts=_finite(snapshot["warmstart"], (9,), "warmstart"),
        integrations=_finite(snapshot["integration"], (size,), "integration"),
    )


def _digest(row: np.void, *, initial: bool, previous: bytes = b"") -> bytes:
    digest = hashlib.sha256(previous)
    keys = TRACE_ARRAY_NAMES[:4] if initial else TRACE_ARRAY_NAMES[:10]
    for key in keys:
        digest.update(np.asarray(row[key]).tobytes(order="C"))
    return digest.digest()


def _prediction_digest(row: np.void, previous: bytes, position: int) -> bytes:
    digest = hashlib.sha256(previous)
    digest.update(np.array(position, dtype="<u4").tobytes())
    digest.update(np.asarray(row["actions"]).tobytes())
    digest.update(np.array(row["prediction_actor"], dtype="S80").tobytes())
    return digest.digest()


def _validate_boundary(row: np.void) -> None:
    if any(not np.isfinite(row[key]).all() for key in TRACE_ARRAY_NAMES[:4]):
        raise EvidenceError("C1 boundary contains nonfinite values.")


def _validate_return(row: np.void, ordinal: int) -> None:
    _validate_boundary(row)
    if (
        any(not np.isfinite(row[key]).all() for key in TRACE_ARRAY_NAMES[4:8])
        or np.any(np.abs(row["actions"]) > 1)
        or int(row["done"]) not in (0, 1)
        or int(row["control_ordinals"]) != ordinal
    ):
        raise EvidenceError("C1 returned values/ordinal are invalid.")


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_active_prefix(directory: Path, *, max_bytes: int = MAX_SPOOL_BYTES) -> dict[str, Any]:
    """Read-only hard-kill inspection; never repair, delete or promote rows.

    A digest-valid row without its committed marker is reported separately.
    It is not added to the committed prefix or assumed to have a completion
    ledger event. The caller joins intents/completions from the paid journal.
    """
    directory = Path(directory)
    spool, manifest = directory / "active.npy", directory / "active.json"
    if not spool.is_file() or not manifest.is_file() or spool.is_symlink() or manifest.is_symlink():
        raise EvidenceError("Missing or escaped C1 active spool/manifest.")
    if spool.stat().st_size + manifest.stat().st_size > min(max_bytes, MAX_SPOOL_BYTES):
        raise BudgetExhausted("C1 active spool read exceeds its bounded allowance.")
    if manifest.stat().st_size > 4096:
        raise EvidenceError("Oversized C1 active manifest.")
    metadata = json.loads(manifest.read_text(encoding="ascii"))
    horizon, size = metadata.get("horizon"), metadata.get("integration_size")
    if (
        metadata.get("schema") != SPOOL_SCHEMA
        or type(horizon) is not int
        or not 1 <= horizon <= MAX_HORIZON
        or type(size) is not int
        or not 1 <= size <= MAX_INTEGRATION
    ):
        raise EvidenceError("Unknown or invalid C1 active schema/dimensions.")
    try:
        rows = np.load(spool, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EvidenceError("Unreadable/truncated C1 active NumPy spool.") from error
    try:
        if rows.shape != (horizon + 1,) or rows.dtype != _dtype(size):
            raise EvidenceError("C1 active dtype/shape differs from its checked schema.")
        if int(rows[0]["valid"]) != 0:
            raise EvidenceError("C1 initial boundary is not committed.")
        _validate_boundary(rows[0])
        previous = _digest(rows[0], initial=True)
        if rows[0]["row_digests"].tobytes() != previous:
            raise EvidenceError("C1 initial boundary digest differs.")
        count = 0
        uncommitted = None
        uncommitted_return = None
        for position in range(1, horizon + 1):
            row = rows[position]
            marker = int(row["valid"])
            if marker == UNUSED:
                candidate = _digest(row, initial=False, previous=previous)
                if row["row_digests"].tobytes() == candidate:
                    _validate_return(row, position - 1)
                    if count and bool(rows[position - 1]["done"]):
                        raise EvidenceError("C1 spool invents a returned row after done.")
                    uncommitted = position - 1
                    uncommitted_return = {
                        key: np.array(row[key], copy=True) for key in TRACE_ARRAY_NAMES[:11]
                    }
                if np.any(rows["valid"][position + 1 :] != UNUSED):
                    raise EvidenceError("C1 validity markers contain a nonprefix row.")
                break
            if marker != position or int(row["control_ordinals"]) != position - 1:
                raise EvidenceError("C1 committed row ordinal differs.")
            _validate_return(row, position - 1)
            expected = _digest(row, initial=False, previous=previous)
            if row["row_digests"].tobytes() != expected:
                raise EvidenceError("C1 returned-row digest differs.")
            if count and bool(rows[position - 1]["done"]):
                raise EvidenceError("C1 spool invents returned controls after done.")
            previous, count = expected, position
        pending_prediction = None
        if count < horizon:
            row = rows[count + 1]
            marker = int(row["prediction_valid"])
            if marker != UNUSED:
                expected = _prediction_digest(row, previous, count + 1)
                actor = row["prediction_actor"].tobytes().decode("ascii")
                if (
                    marker != count + 1
                    or not actor
                    or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", actor)
                    or not np.isfinite(row["actions"]).all()
                    or np.any(np.abs(row["actions"]) > 1)
                    or row["prediction_digest"].tobytes() != expected
                    or (count and bool(rows[count]["done"]))
                ):
                    raise EvidenceError("C1 pending prediction binding/digest differs.")
                pending_prediction = dict(
                    ordinal=count,
                    actor=actor,
                    action=np.array(row["actions"], copy=True),
                    sha256=expected.hex(),
                )
        arrays = {key: np.array(rows[key][: count + 1], copy=True) for key in TRACE_ARRAY_NAMES[:4]}
        arrays.update(
            {key: np.array(rows[key][1 : count + 1], copy=True) for key in TRACE_ARRAY_NAMES[4:11]}
        )
        arrays["initial_digest"] = np.array(rows[0]["row_digests"], copy=True)
        return dict(
            schema=SPOOL_SCHEMA,
            manifest=metadata,
            completed_controls=count,
            additional_digest_valid_row=uncommitted,
            additional_digest_valid_return=uncommitted_return,
            normally_closed=False,
            availability="active_prefix_only",
            pending_prediction=pending_prediction,
            arrays=arrays,
        )
    finally:
        rows._mmap.close()


class C1TraceRecorder:
    """One active native branch; native work counters stay with the caller.

    Supply the actual pre-context to every ``intent``. The recorder verifies
    it byte-for-byte against the last durable boundary before authorizing the
    next call. Neither ``fail`` nor the prefix reader retries native work.
    The owner reserves the 8 MiB spool in its global external-file allowance.
    """

    def __init__(self, meter: BinaryEvidenceMeter, spool_directory: Path):
        self.meter = meter
        self.directory = Path(spool_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise EvidenceError(
                "C1 active directory must be empty; retained evidence forbids restart."
            )
        self._rows: np.memmap | None = None
        self._stream: Any = None
        self._active = self._poisoned = self._failed = False
        self._pending: np.ndarray | None = None
        self._count = 0
        self._actor: str | None = None

    @property
    def completed_controls(self) -> int:
        return self._count

    @property
    def poisoned(self) -> bool:
        return self._poisoned or self.meter._poisoned

    def _open(self) -> None:
        if not self._active or self._poisoned or self.meter._poisoned:
            raise EvidenceError("C1 trace is inactive or poisoned; no further native work.")

    def _sync(self) -> None:
        self._rows.flush()
        os.fsync(self._stream.fileno())

    def _poison(self) -> None:
        self._poisoned = True
        self.meter._poisoned = True

    def _release_mapping(self) -> None:
        if self._rows is not None:
            self._rows._mmap.close()
            self._rows = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def begin(
        self,
        alias: str,
        horizon: int,
        initial_snapshot: dict[str, Any],
        observation: Any,
    ) -> None:
        if self._active or self._poisoned or self._failed or any(self.directory.iterdir()):
            raise EvidenceError("C1 trace cannot replace an active/retained prefix.")
        if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", alias):
            raise ValueError("C1 trace alias must be a short ASCII catalog key.")
        if type(horizon) is not int or not 1 <= horizon <= MAX_HORIZON:
            raise ValueError("C1 horizon must be 1..1000 controls.")
        size = np.asarray(initial_snapshot["integration"]).size
        if not 1 <= size <= MAX_INTEGRATION:
            raise EvidenceError("C1 native integration size exceeds its frozen bound.")
        boundary = _boundary(initial_snapshot, observation, size)
        dtype = _dtype(size)
        spool_bound = (horizon + 1) * dtype.itemsize + 8192
        if spool_bound > MAX_SPOOL_BYTES:
            raise BudgetExhausted("C1 spool reservation exceeds 8 MiB.")
        # Reserve the worst full branch archive and compact journal before any
        # paid call, rather than discovering exhaustion after simulation.
        # Four rows cover native intent/return and, for actor execution, the
        # owner's neural request/completion pair. This reserves, not charges.
        self._archive_reservation = 1493 * horizon + 5512 + 2 * 8192 + (4 * horizon + 3) * 512
        self.meter._reserve(self._archive_reservation)
        self.alias, self.horizon, self.integration_size = alias, horizon, size
        self._count, self._pending, self._failed = 0, None, False
        self._actor = None
        metadata = dict(schema=SPOOL_SCHEMA, alias=alias, horizon=horizon, integration_size=size)
        try:
            with (self.directory / "active.json").open("x", encoding="ascii") as stream:
                stream.write(_dumps(metadata) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._rows = np.lib.format.open_memmap(
                self.directory / "active.npy",
                mode="w+",
                dtype=dtype,
                shape=(horizon + 1,),
            )
            self._stream = (self.directory / "active.npy").open("r+b", buffering=0)
            self._rows["valid"] = UNUSED
            self._rows["prediction_valid"] = UNUSED
            self._rows["control_ordinals"] = UNUSED
            for key, value in boundary.items():
                self._rows[0][key] = value
            self._rows[0]["row_digests"] = np.frombuffer(_digest(self._rows[0], initial=True), "u1")
            self._sync()
            self._rows[0]["valid"] = 0
            self._sync()
            _sync_directory(self.directory)
            self.meter.record("trace_begin", alias=alias, horizon=horizon, integration_size=size)
            self._active = True
        except BaseException:
            self._poison()
            self._release_mapping()
            raise

    def check_boundary(self, snapshot: dict[str, Any], observation: Any) -> None:
        self._open()
        boundary = _boundary(snapshot, observation, self.integration_size)
        for key, value in boundary.items():
            if value.tobytes() != self._rows[self._count][key].tobytes():
                raise EvidenceError(f"C1 current pre-context differs in {key}.")

    def prediction(
        self,
        action: Any,
        *,
        actor_alias: str | None = None,
        actor_hash: str | None = None,
    ) -> str:
        """Fsync a returned actor action before the caller marks NN completion.

        No extra ledger event is emitted: the owning neural measurement joins
        this pending action to the already durable current observation. The
        subsequent returned-row marker logically consumes the prediction.
        """
        self._open()
        if (actor_alias is None) == (actor_hash is None):
            raise ValueError("Exactly one actor catalog alias or checkpoint hash is required.")
        actor = actor_alias if actor_hash is None else actor_hash
        if not isinstance(actor, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", actor):
            raise ValueError("Explicit bounded ASCII actor binding required.")
        if actor_hash is not None and not re.fullmatch(r"[a-f0-9]{64}", actor_hash):
            raise ValueError("Checkpoint hash must be lowercase SHA256.")
        if self._actor is not None and actor != self._actor:
            raise EvidenceError("C1 actor binding cannot change within one execution trace.")
        if self._failed or self._pending is not None or self._count >= self.horizon:
            raise EvidenceError("C1 prediction has failed/pending/exhausted branch.")
        if self._count and bool(self._rows[self._count]["done"]):
            raise EvidenceError("C1 prediction after done is forbidden.")
        position = self._count + 1
        row = self._rows[position]
        if int(row["prediction_valid"]) != UNUSED:
            raise EvidenceError("C1 pending prediction cannot be replaced or recomputed.")
        value = _finite(action, (6,), "predicted action")
        if np.any(np.abs(value) > 1):
            raise EvidenceError("C1 predicted action is outside [-1,1].")
        try:
            row["actions"] = value
            row["prediction_actor"] = actor
            digest = _prediction_digest(
                row, self._rows[self._count]["row_digests"].tobytes(), position
            )
            row["prediction_digest"] = np.frombuffer(digest, "u1")
            self._sync()
            row["prediction_valid"] = position
            self._sync()
            self._actor = actor
            return digest.hex()
        except BaseException:
            self._poison()
            raise

    def intent(self, action: Any, *, snapshot: dict[str, Any], observation: Any) -> str:
        self._open()
        if self._failed or self._pending is not None or self._count >= self.horizon:
            raise EvidenceError("C1 intent has failed/pending/exhausted branch.")
        if self._count and bool(self._rows[self._count]["done"]):
            raise EvidenceError("C1 native work after a fall/done is forbidden.")
        self.check_boundary(snapshot, observation)
        value = _finite(action, (6,), "action")
        if np.any(np.abs(value) > 1):
            raise EvidenceError("C1 control is outside [-1,1].")
        pending_row = self._rows[self._count + 1]
        if (
            int(pending_row["prediction_valid"]) != UNUSED
            and value.tobytes() != pending_row["actions"].tobytes()
        ):
            raise EvidenceError("C1 native intent differs from the durable prediction.")
        remaining = self.horizon - self._count
        self.meter._reserve(1493 * self.horizon + 5512 + 2 * 8192 + (4 * remaining + 2) * 512)
        identifier = self.meter.record(
            "trace_intent",
            alias=self.alias,
            ordinal=self._count,
            action=value,
            requested_controls=1,
            requested_physics_substeps=4,
        )
        self._pending = value
        return identifier

    def completed(
        self,
        snapshot: dict[str, Any],
        observation: Any,
        *,
        cost: float,
        velocity: float,
        fall: float,
        done: bool,
    ) -> str:
        self._open()
        if self._pending is None:
            raise EvidenceError("C1 completion has no prepaid intent.")
        boundary = _boundary(snapshot, observation, self.integration_size)
        scalars = _finite([cost, velocity, fall], (3,), "returned scalars")
        if type(done) not in (bool, np.bool_):
            raise EvidenceError("C1 returned done must be boolean.")
        position = self._count + 1
        try:
            row = self._rows[position]
            for key, value in boundary.items():
                row[key] = value
            row["actions"] = self._pending
            row["native_costs"], row["x_velocities"], row["fall_costs"] = scalars
            row["done"], row["control_ordinals"] = done, self._count
            digest = _digest(
                row, initial=False, previous=self._rows[position - 1]["row_digests"].tobytes()
            )
            row["row_digests"] = np.frombuffer(digest, "u1")
            self._sync()
            row["valid"] = position
            self._sync()
            # Count the durable row even if the completion ledger write fails.
            self._count = position
            identifier = self.meter.record(
                "trace_returned",
                alias=self.alias,
                ordinal=position - 1,
                digest=digest.hex(),
                completed_controls=1,
                completed_physics_substeps=4,
            )
            self._pending = None
            return identifier
        except BaseException:
            self._poison()
            raise

    def close(self, absorbing: float = 0.0, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        self._open()
        missing = self.horizon - self._count
        fallen = bool(self._count and self._rows[self._count]["done"])
        if not self._failed and (self._pending is not None or (missing and not fallen)):
            raise EvidenceError("C1 normal closure requires horizon or actual done.")
        expected_tail = 5.0 * missing if fallen and not self._failed else 0.0
        if not np.isfinite(absorbing) or float(absorbing) != expected_tail:
            raise EvidenceError("C1 absorbing tail differs from actual fall/missing prefix.")
        metadata = dict(meta or {})
        reserved = {
            "alias",
            "horizon",
            "completed_controls",
            "availability",
            "absorbing_cost",
            "unknown_partial_controls",
            "unknown_partial_physics_substeps",
            "pending_prediction",
            "prediction_actor",
        }
        if reserved & metadata.keys():
            raise ValueError("C1 caller metadata replaces reserved closure fields.")
        metadata.update(
            alias=self.alias,
            horizon=self.horizon,
            completed_controls=self._count,
            availability="failed_actual_prefix" if self._failed else "closed_actual_prefix",
            absorbing_cost=float(absorbing),
            unknown_partial_controls=int(self._pending is not None),
            unknown_partial_physics_substeps=4 * int(self._pending is not None),
        )
        if self._actor is not None:
            metadata["prediction_actor"] = self._actor
        if self._count < self.horizon:
            pending = self._rows[self._count + 1]
            if int(pending["prediction_valid"]) != UNUSED:
                metadata["pending_prediction"] = dict(
                    actor=pending["prediction_actor"].tobytes().decode("ascii"),
                    action=pending["actions"].tolist(),
                    sha256=pending["prediction_digest"].tobytes().hex(),
                    ordinal=self._count,
                )
        text = _dumps(metadata)
        if not text.isascii() or len(text) > 1024:
            raise BudgetExhausted("C1 trace metadata exceeds 1024 ASCII characters.")
        arrays = {
            key: np.array(self._rows[key][: self._count + 1], copy=True)
            for key in TRACE_ARRAY_NAMES[:4]
        }
        arrays.update(
            {
                key: np.array(self._rows[key][1 : self._count + 1], copy=True)
                for key in TRACE_ARRAY_NAMES[4:11]
            }
        )
        arrays["initial_digest"] = np.array(self._rows[0]["row_digests"], copy=True)
        arrays["metadata"] = metadata
        try:
            reference = self.meter.save_arrays(self.alias, **arrays)
            self.meter.record(
                "trace_complete",
                alias=self.alias,
                group=reference["group"],
                reference_sha256=hashlib.sha256(_dumps(reference).encode()).hexdigest(),
                completed_controls=self._count,
                availability=metadata["availability"],
                unknown_partial_controls=metadata["unknown_partial_controls"],
            )
            self._release_mapping()
            (self.directory / "active.npy").unlink()
            (self.directory / "active.json").unlink()
            _sync_directory(self.directory)
            self._active = False
            return reference
        except BaseException:
            self._poison()
            raise

    def fail(
        self, error: BaseException | str, *, meta: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Preserve a graceful actual prefix, or retain poisoned spool untouched."""
        self._failed = True
        if self._poisoned or self.meter._poisoned:
            self._release_mapping()
            return None
        self._open()
        self.meter.record("trace_error", alias=self.alias, error_type=type(error).__name__)
        return self.close(0.0, dict(meta or {}, failure_type=type(error).__name__))
