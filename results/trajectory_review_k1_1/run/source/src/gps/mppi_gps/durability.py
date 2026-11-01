"""Create-only durable episode segments and verified independent copies.

This module uses only the standard library. The caller closes the existing
Meter and precreates/fsyncs fixed ancestors before sealing. These helpers commit
the immediate file/directory parents; hashes bind bytes and the ledger binds
ownership/work. Scientific arrays are checked by the independent saved-only audit.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "walker-durable-episode-v1"
DATA_FILES = ("raw.h5", "ledger.jsonl")
OWNER_FIELDS = {
    "ordinal",
    "point",
    "panel",
    "seed",
    "tag",
    "boot_id",
    "boot_id_after",
    "owner_pid",
    "launch_receipt_sha256",
    "config_sha256",
    "input_manifest_sha256",
    "source_manifest_sha256",
    "checkpoint_sha256",
    "policy_hash",
    "optimizer_fingerprint",
    "initial_context_fingerprint",
    "cumulative_before",
}
HASH_FIELDS = {
    "launch_receipt_sha256",
    "config_sha256",
    "input_manifest_sha256",
    "source_manifest_sha256",
    "checkpoint_sha256",
    "policy_hash",
    "optimizer_fingerprint",
    "initial_context_fingerprint",
}


class DurabilityError(ValueError):
    """An artifact is unsafe, incomplete, changed or outside its declared budget."""


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DurabilityError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _load_json(data: str | bytes) -> Any:
    def invalid(value: str) -> None:
        raise DurabilityError("Nonfinite JSON value: " + value)

    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=invalid)


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise DurabilityError("Evidence must be a regular, non-symlink file: " + str(path))
    return info


def fsync_directory(path: str | Path) -> None:
    """Persist directory-entry changes; propagate unsupported/failed fsync."""
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise DurabilityError("A real directory is required for fsync")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_json(path: str | Path, value: Any) -> None:
    """Fsync a temporary JSON, atomically create the target, and fsync its directory.

    POSIX hard-link creation never replaces an existing target. On failure an
    owned temporary file is removed; prior targets are never overwritten.
    """
    path = Path(path)
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    owned_inode = os.fstat(descriptor)
    linked = False
    # Keep this descriptor open until the rollback ownership decision finishes.
    # Otherwise unlinking the last name allows Linux to reuse this inode for a
    # different writer's replacement before the device/inode comparison below.
    with os.fdopen(descriptor, "wb") as stream:
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary_path, path)
            linked = True
            temporary_path.unlink()
            fsync_directory(path.parent)
        except BaseException:
            # A failed commit is not a valid acknowledgement. Remove only a
            # target created by this call; another writer's target stays intact.
            if linked:
                try:
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == (owned_inode.st_dev, owned_inode.st_ino):
                        path.unlink()
                except OSError:
                    pass
            raise
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def file_binding(path: str | Path) -> dict[str, Any]:
    """Hash regular file bytes while detecting changes during the read."""
    path = Path(path)
    before = _regular(path)
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            digest.update(block)
    after = _regular(path)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, key) != getattr(after, key) for key in fields) or count != after.st_size:
        raise DurabilityError("Evidence changed while hashing: " + str(path))
    return {"sha256": digest.hexdigest(), "bytes": count}


def _fsync_file(path: Path) -> None:
    _regular(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _segment_roster(path: Path, *, sealed: bool) -> None:
    if path.is_symlink() or not path.is_dir():
        raise DurabilityError("A real episode segment directory is required")
    expected = set(DATA_FILES) | ({"receipt.json"} if sealed else set())
    if {entry.name for entry in path.iterdir()} != expected:
        raise DurabilityError("Unsafe extra or missing episode segment files")
    for name in expected:
        _regular(path / name)


def _hdf_boundary(path: Path) -> None:
    """Check the Meter's offset-zero HDF superblock for a truncated boundary.

    The HDF Group format specification defines the little-endian EOF fields:
    https://portal.hdfgroup.org/documentation/hdf5/latest/_f_m_t4.html#sec_fmt_super
    This is a boundary check, not a replacement for the independent HDF audit.
    """
    with path.open("rb") as stream:
        header = stream.read(128)
    if len(header) < 48 or header[:8] != b"\x89HDF\r\n\x1a\n":
        raise DurabilityError("Missing supported HDF signature/superblock")
    version = header[8]
    if version in (0, 1):
        width = header[13]
        base_offset = 24 if version == 0 else 28
    elif version in (2, 3):
        width, base_offset = header[9], 12
        if version == 3 and header[11] & 0x05:
            raise DurabilityError("HDF still carries an open-writer flag")
    else:
        raise DurabilityError("Unsupported HDF superblock version")
    if width not in (2, 4, 8, 16) or len(header) < base_offset + 3 * width:
        raise DurabilityError("Unsupported HDF address width")
    base = int.from_bytes(header[base_offset : base_offset + width], "little")
    end = int.from_bytes(header[base_offset + 2 * width : base_offset + 3 * width], "little")
    if base != 0 or end < len(header) or end > path.stat().st_size:
        raise DurabilityError("HDF segment is truncated or has an unsupported boundary")


def _counter_map(value: Any, name: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise DurabilityError(name + " must be a counter dictionary")
    for key, amount in value.items():
        if not isinstance(key, str) or not key:
            raise DurabilityError("Counter names must be nonempty strings")
        if key == "wall_seconds":
            if type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0:
                raise DurabilityError("Invalid nonnegative wall time")
        elif type(amount) is not int or amount < 0:
            raise DurabilityError("Work counters must be explicit nonnegative integers")
    return value


def _hash(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_receipt(receipt: Any) -> None:
    required = {"schema", "ownership", "episode", "work", "cumulative", "files"}
    if not isinstance(receipt, dict) or set(receipt) != required or receipt["schema"] != SCHEMA:
        raise DurabilityError("Unexpected durable episode receipt schema")
    owner, episode = receipt["ownership"], receipt["episode"]
    if not isinstance(owner, dict) or not OWNER_FIELDS.issubset(owner):
        raise DurabilityError("Episode ownership fields are incomplete")
    for key in HASH_FIELDS:
        if not _hash(owner[key]):
            raise DurabilityError("Invalid ownership hash: " + key)
    if "worker_started_sha256" in owner and not _hash(owner["worker_started_sha256"]):
        raise DurabilityError("Invalid worker-started ownership hash")
    for key in ("ordinal", "seed"):
        if type(owner[key]) is not int or owner[key] < 0:
            raise DurabilityError("Invalid episode ordinal/reset seed")
    if type(owner["owner_pid"]) is not int or owner["owner_pid"] <= 0:
        raise DurabilityError("Invalid owner process identity")
    if any(
        not isinstance(owner[key], str) or not owner[key]
        for key in ("point", "panel", "tag", "boot_id", "boot_id_after")
    ):
        raise DurabilityError("Missing episode/boot identity")
    if owner["boot_id"] != owner["boot_id_after"]:
        raise DurabilityError("Host boot identity changed during the episode")
    if not isinstance(episode, dict) or episode.get("complete") is not True:
        raise DurabilityError("Only complete actual episodes can be sealed")
    if episode.get("tag") != owner["tag"] or episode.get("seed") != owner["seed"]:
        raise DurabilityError("Episode/reset ownership differs")
    if type(episode.get("seed")) is not int or any(
        type(episode.get(key)) is not bool for key in ("success", "survived")
    ):
        raise DurabilityError("Invalid native outcome/reset types")
    if episode["success"] and not episode["survived"]:
        raise DurabilityError("Walking cannot be reported without survival")
    for key in ("cost", "mae"):
        value = episode.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise DurabilityError("Invalid finite native outcome: " + key)
    if not isinstance(episode.get("raw"), dict) or episode["raw"].get("file") != "raw.h5":
        raise DurabilityError("Episode raw evidence must belong to its segment")
    if type(episode.get("steps")) is not int or episode["steps"] < 1:
        raise DurabilityError("An actual episode requires executed controls")
    work = _counter_map(receipt["work"], "work")
    before = _counter_map(owner["cumulative_before"], "cumulative_before")
    after = _counter_map(receipt["cumulative"], "cumulative")
    expected_keys = (set(before) | set(work)) - {"wall_seconds"}
    if set(after) - {"wall_seconds"} != expected_keys:
        raise DurabilityError("Cumulative counter roster differs")
    for key in expected_keys:
        if after[key] != before.get(key, 0) + work.get(key, 0):
            raise DurabilityError("Cumulative work does not add across segments: " + key)
    if work.get("episodes") != 1 or work.get("completed_live_controls") != episode["steps"]:
        raise DurabilityError("Segment work does not describe exactly one actual episode")
    if any(amount for key, amount in work.items() if key.startswith("unknown_partial_")):
        raise DurabilityError("Complete segment has unresolved partial work")
    if any(
        work.get(key, 0)
        for key in (
            "failed_queries",
            "rejected_queries",
            "failed_measurements",
            "rejected_measurements",
            "reset_errors",
        )
    ):
        raise DurabilityError("Complete segment has failed or rejected boundaries")
    if not isinstance(receipt["files"], dict) or set(receipt["files"]) != set(DATA_FILES):
        raise DurabilityError("Unsafe receipt file roster")
    for binding in receipt["files"].values():
        if (
            not isinstance(binding, dict)
            or set(binding) != {"sha256", "bytes"}
            or not _hash(binding["sha256"])
            or type(binding["bytes"]) is not int
            or binding["bytes"] < 1
        ):
            raise DurabilityError("Invalid segment file binding")
    _json_bytes(receipt)


def _closed_ledger(path: Path, receipt: dict[str, Any]) -> None:
    episodes = []
    last = None
    first_event = None
    with path.open("rb") as stream:
        for index, line in enumerate(stream):
            if not line.endswith(b"\n"):
                raise DurabilityError("Ledger has an unclosed/truncated final record")
            row = _load_json(line)
            if not isinstance(row, dict) or row.get("id") != f"event-{index:09d}":
                raise DurabilityError("Ledger sequence differs")
            if first_event is None:
                first_event = row.get("event")
            if row.get("event") == "episode_complete":
                episodes.append(row)
            if last is not None and last.get("event") == "runtime_close":
                raise DurabilityError("Ledger continued after closure")
            last = row
    if (
        first_event != "runtime_open"
        or last is None
        or last.get("event") != "runtime_close"
        or len(episodes) != 1
    ):
        raise DurabilityError("A closed ledger must contain exactly one actual episode")
    expected = receipt["episode"]
    for key in ("tag", "seed", "steps", "complete", "cost", "success", "survived", "mae", "raw"):
        if key not in expected or episodes[0].get(key) != expected[key]:
            raise DurabilityError("Episode receipt differs from its closed ledger: " + key)
    totals = _counter_map(last.get("totals"), "ledger totals")
    if set(totals) != set(receipt["work"]):
        raise DurabilityError("Closed ledger/work counter roster differs")
    if any(
        receipt["work"][key] != amount for key, amount in totals.items() if key != "wall_seconds"
    ):
        raise DurabilityError("Closed ledger/work values differ")


def seal_segment(
    segment: Path, episode: dict, ownership: dict, work: dict, cumulative: dict
) -> dict:
    """Commit a closed one-episode Meter without changing its HDF/ledger bytes."""
    segment = Path(segment)
    _segment_roster(segment, sealed=False)
    _hdf_boundary(segment / "raw.h5")
    receipt = _load_json(
        _json_bytes(
            {
                "schema": SCHEMA,
                "ownership": ownership,
                "episode": episode,
                "work": work,
                "cumulative": cumulative,
                "files": {name: file_binding(segment / name) for name in DATA_FILES},
            }
        )
    )
    _validate_receipt(receipt)
    _closed_ledger(segment / "ledger.jsonl", receipt)
    for name in DATA_FILES:
        _fsync_file(segment / name)
    fsync_directory(segment)
    fsync_directory(segment.parent)
    durable_json(segment / "receipt.json", receipt)
    verified = verify_segment(segment, file_binding(segment / "receipt.json")["sha256"])
    fsync_directory(segment.parent)
    return verified


def verify_segment(path: Path, expected_receipt_sha: str | None = None) -> dict:
    """Verify exact segment files, ownership, closure and cumulative work."""
    path = Path(path)
    _segment_roster(path, sealed=True)
    binding = file_binding(path / "receipt.json")
    if expected_receipt_sha is not None and (
        not _hash(expected_receipt_sha) or binding["sha256"] != expected_receipt_sha
    ):
        raise DurabilityError("Receipt SHA-256 differs")
    receipt = _load_json((path / "receipt.json").read_bytes())
    _validate_receipt(receipt)
    for name in DATA_FILES:
        if file_binding(path / name) != receipt["files"][name]:
            raise DurabilityError("Segment file bytes differ: " + name)
    _hdf_boundary(path / "raw.h5")
    _closed_ledger(path / "ledger.jsonl", receipt)
    return receipt


def mirror_segment(source: Path, target: Path, expected_receipt_sha: str) -> dict:
    """Copy immutable closed files to a new directory, fsync, then verify."""
    source, target = Path(source), Path(target)
    verify_segment(source, expected_receipt_sha)
    if target.resolve().is_relative_to(source.resolve()):
        raise DurabilityError("A mirror cannot be created inside its immutable source")
    target.mkdir(parents=True, exist_ok=False)
    fsync_directory(target.parent)
    for name in (*DATA_FILES, "receipt.json"):
        with (source / name).open("rb") as incoming, (target / name).open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    fsync_directory(target)
    receipt = verify_segment(target, expected_receipt_sha)
    verify_segment(source, expected_receipt_sha)
    fsync_directory(target.parent)
    return receipt


def read_boot_id(
    path: str | Path = "/proc/sys/kernel/random/boot_id",
    *,
    reader: Callable[[Path], str] | None = None,
) -> str:
    """Read the canonical Linux boot UUID, with an injectable synthetic reader."""
    value = (reader(Path(path)) if reader is not None else Path(path).read_text()).strip()
    try:
        canonical = str(uuid.UUID(value))
    except (ValueError, AttributeError) as error:
        raise DurabilityError("Invalid host boot ID") from error
    if value.lower() != canonical:
        raise DurabilityError("Host boot ID is not a canonical UUID")
    return canonical


def require_boot_id(expected: str, *, reader: Callable[[], str] = read_boot_id) -> str:
    actual = reader()
    if actual != expected:
        raise DurabilityError("Host boot identity changed")
    return actual


class CumulativeBudget:
    """Check one global integer work budget without resetting between segments."""

    def __init__(self, limits: Mapping[str, int], initial: Mapping[str, int] | None = None):
        self._limits = dict(limits)
        if not self._limits or any(
            not isinstance(key, str) or not key or type(value) is not int or value < 0
            for key, value in self._limits.items()
        ):
            raise DurabilityError("Explicit nonnegative integer budget limits required")
        self._totals = dict.fromkeys(self._limits, 0)
        self.add(dict(initial or {}))

    def add(self, work: Mapping[str, int]) -> dict[str, int]:
        incoming = dict(work)
        if set(incoming) - set(self._limits) or any(
            type(value) is not int or value < 0 for value in incoming.values()
        ):
            raise DurabilityError("Invalid or unknown cumulative work counter")
        candidate = {key: self._totals[key] + incoming.get(key, 0) for key in self._limits}
        if any(candidate[key] > self._limits[key] for key in candidate):
            raise DurabilityError("Cumulative work budget exceeded across segments")
        self._totals = candidate
        return self.totals()

    def totals(self) -> dict[str, int]:
        return dict(self._totals)

    def remaining(self) -> dict[str, int]:
        return {key: value - self._totals[key] for key, value in self._limits.items()}
