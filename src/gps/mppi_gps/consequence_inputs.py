"""Bounded, saved-only C1 source preparation; no simulator or policy imports.

The closed H5 audit's authenticated SQLite cache binds canonical group
references. Binary search retrieves only selected index entries; selected
array C bytes are then checked individually. Neither large source stream is
scanned or rehashed. The old run and its cache are always read-only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import resource
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np

MIB = 1024**2
FORMAT = "walker-array-stream-v1"
GROUP = re.compile(r"arrays/a([0-9]{9})_[A-Za-z0-9_.-]{1,100}")
SHA256 = re.compile(r"[0-9a-f]{64}")


class ConsequenceInputError(RuntimeError):
    """Missing, changed or over-budget evidence stops C1 preparation."""


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ConsequenceInputError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(payload: bytes | bytearray | str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ConsequenceInputError("Nonfinite JSON")),
        )
    except (ValueError, UnicodeError) as error:
        raise ConsequenceInputError("Malformed source JSON") from error


def _identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _safe_file(root: Path, relative: str) -> Path:
    _need(isinstance(relative, str) and bool(relative), "Explicit relative source path required")
    _need(
        not Path(relative).is_absolute() and ".." not in Path(relative).parts, "Escaped source path"
    )
    path = (root / relative).resolve()
    _need(path.is_relative_to(root) and path.is_file(), "Missing or escaped source file")
    return path


class _Budget:
    def __init__(self, read: int, output: int, seconds: float, rss: int):
        _need(0 < read <= 512 * MIB, "C1 preparation read ceiling exceeds 512 MiB")
        _need(65536 < output <= 256 * MIB, "C1 preparation output ceiling exceeds 256 MiB")
        _need(0 < seconds <= 600, "C1 preparation deadline exceeds 600 seconds")
        _need(0 < rss <= 512 * MIB, "C1 preparation RSS ceiling exceeds 512 MiB")
        self.read_limit, self.output_limit, self.rss_limit = read, output, rss
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.read_bytes = self.output_reserved = 0
        self.read_by_kind: dict[str, int] = {}
        self.identities: dict[Path, tuple[int, int, int, int]] = {}
        self.output_files: dict[str, dict[str, Any]] = {}

    def check(self) -> None:
        _need(time.monotonic() <= self.deadline, "C1 preparation deadline exhausted")
        _need(_peak_rss_bytes() <= self.rss_limit, "C1 preparation peak RSS exhausted")

    def track(self, path: Path, expected_bytes: int | None = None) -> None:
        self.check()
        actual = _identity(path)
        _need(expected_bytes is None or actual[2] == expected_bytes, "Source file size changed")
        previous = self.identities.setdefault(path, actual)
        _need(previous == actual, "Source file changed during preparation")

    def read(self, stream: BinaryIO, amount: int, kind: str, *, line: bool = False) -> bytes:
        self.check()
        _need(
            amount >= 0 and self.read_bytes + amount <= self.read_limit,
            "Source read budget exhausted",
        )
        value = stream.readline(amount) if line else stream.read(amount)
        self.read_bytes += len(value)
        self.read_by_kind[kind] = self.read_by_kind.get(kind, 0) + len(value)
        return value

    def whole(self, path: Path, binding: Mapping[str, Any], kind: str) -> bytearray:
        _need(type(binding.get("bytes")) is int and binding["bytes"] >= 0, "Missing byte binding")
        _need(
            SHA256.fullmatch(str(binding.get("sha256", ""))) is not None, "Missing SHA256 binding"
        )
        self.track(path, binding["bytes"])
        _need(self.read_bytes + binding["bytes"] <= self.read_limit, "File cannot fit read budget")
        value = bytearray()
        with path.open("rb", buffering=0) as stream:
            remaining = binding["bytes"]
            while remaining:
                part = self.read(stream, min(MIB, remaining), kind)
                _need(bool(part), "Truncated source file")
                value.extend(part)
                remaining -= len(part)
        _need(hashlib.sha256(value).hexdigest() == binding["sha256"], "Source file SHA256 differs")
        return value

    def reserve_output(self, amount: int, *, emergency: bool = False) -> None:
        self.check()
        available = self.output_limit - (0 if emergency else 65536)
        _need(
            amount >= 0 and self.output_reserved + amount <= available,
            "Preparation output budget exhausted",
        )
        self.output_reserved += amount

    def unchanged(self) -> None:
        for path, original in self.identities.items():
            _need(_identity(path) == original, "Source changed after selected-byte verification")

    def summary(self) -> dict[str, Any]:
        return dict(
            source_read_bytes=self.read_bytes,
            source_read_bytes_by_kind=self.read_by_kind,
            output_reserved_bytes=self.output_reserved,
            peak_rss_bytes=_peak_rss_bytes(),
            wall_seconds=time.monotonic() - self.started,
            new_neural_rows=0,
            new_native_controls=0,
            new_optimizer_steps=0,
        )


class _GuardedOutput:
    def __init__(self, stream: BinaryIO, limit: int):
        self.stream, self.limit = stream, limit

    def write(self, value):
        remaining = memoryview(value).cast("B")
        length = len(remaining)
        _need(self.stream.tell() + length <= self.limit, "Prepared file exceeds reservation")
        while remaining:
            count = self.stream.write(remaining)
            _need(bool(count), "Prepared file accepted no bytes")
            remaining = remaining[count:]
        return length

    def seek(self, offset, whence=0):
        result = self.stream.seek(offset, whence)
        _need(0 <= result <= self.limit, "Prepared file seek exceeds reservation")
        return result

    def tell(self):
        return self.stream.tell()

    def flush(self):
        self.stream.flush()

    def read(self, size=-1):
        raise io.UnsupportedOperation("Prepared output is write-only")


def _descriptor(path: Path, root: Path, budget: _Budget) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        remaining = path.stat().st_size
        while remaining:
            part = budget.read(stream, min(MIB, remaining), "prepared_output_verification")
            _need(bool(part), "Truncated prepared output")
            digest.update(part)
            remaining -= len(part)
    return dict(
        file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=digest.hexdigest()
    )


def _write_json(root: Path, relative: str, value: Any, budget: _Budget, *, emergency=False):
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    budget.reserve_output(len(payload), emergency=emergency)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = dict(file=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    budget.output_files[relative] = descriptor
    return descriptor


def _write_arrays(root: Path, relative: str, arrays: Mapping[str, np.ndarray], budget: _Budget):
    for value in arrays.values():
        _need(
            not value.dtype.hasobject and value.dtype.fields is None,
            "Only ordinary typed arrays may be saved",
        )
    allowance = sum(value.nbytes for value in arrays.values()) + 4096 * len(arrays) + 65536
    budget.reserve_output(allowance)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb", buffering=0) as stream:
        guarded = _GuardedOutput(stream, allowance)
        if path.suffix == ".npy":
            _need(len(arrays) == 1, "One named array per NPY sidecar")
            np.lib.format.write_array(guarded, next(iter(arrays.values())), allow_pickle=False)
        else:
            np.savez(guarded, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = _descriptor(path, root, budget)
    budget.output_files[relative] = descriptor
    return descriptor | {
        "arrays": {
            key: dict(
                shape=list(value.shape),
                dtype=value.dtype.str,
                sha256=hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            )
            for key, value in arrays.items()
        }
    }


def _copy_file(source: Path, root: Path, relative: str, binding, budget: _Budget):
    budget.track(source, binding["bytes"])
    budget.reserve_output(binding["bytes"])
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb", buffering=0) as incoming, target.open("xb", buffering=0) as outgoing:
        remaining = binding["bytes"]
        while remaining:
            part = budget.read(incoming, min(MIB, remaining), "copied_source_files")
            _need(bool(part), "Truncated copied input")
            _GuardedOutput(outgoing, binding["bytes"]).write(part)
            digest.update(part)
            remaining -= len(part)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    _need(digest.hexdigest() == binding["sha256"], "Copied input SHA256 differs")
    descriptor = dict(file=relative, bytes=binding["bytes"], sha256=digest.hexdigest())
    budget.output_files[relative] = descriptor
    return descriptor


class _SelectedArchive:
    """Authenticated sparse reads from the existing append-only array format."""

    line_limit = 8192

    def __init__(
        self, run: Path, sources: Mapping[str, Any], cache: Path, cache_binding, budget: _Budget
    ):
        self.budget = budget
        self.raw = _safe_file(run, "evidence/arrays.bin")
        self.index = _safe_file(run, "evidence/arrays.index.jsonl")
        for name, path in [
            ("evidence/arrays.bin", self.raw),
            ("evidence/arrays.index.jsonl", self.index),
        ]:
            budget.track(path, sources["external_evidence_files"][name]["bytes"])
        self.size = self.index.stat().st_size
        _need(
            _peak_rss_bytes() + 2 * cache_binding["bytes"] + 16 * MIB <= budget.rss_limit,
            "Closed audit cache cannot fit the preparation memory envelope",
        )
        payload = budget.whole(cache, cache_binding, "closed_audit_SQLite")
        self.db = sqlite3.connect(":memory:")
        _need(
            hasattr(self.db, "deserialize"),
            "SQLite deserialize support is required within the read budget",
        )
        self.db.deserialize(payload)
        del payload
        self.db.execute("PRAGMA query_only=ON")
        columns = self.db.execute("PRAGMA table_info(groups)").fetchall()
        _need([v[1] for v in columns] == ["name", "digest"], "Closed audit cache schema differs")
        self.groups_verified = 0
        budget.check()

    def close(self):
        self.db.close()

    def verify(self, reference: Mapping[str, Any]) -> None:
        self.budget.check()
        _need(
            set(reference) == {"format", "file", "index", "group", "tag", "arrays"},
            "Array reference keys differ",
        )
        _need(
            reference["format"] == FORMAT
            and reference["file"] == "arrays.bin"
            and reference["index"] == "arrays.index.jsonl",
            "Unsupported source reference",
        )
        _need(GROUP.fullmatch(reference["group"]) is not None, "Unsafe source group")
        row = self.db.execute(
            "SELECT digest FROM groups WHERE name=?", (reference["group"],)
        ).fetchone()
        _need(
            row is not None and row[0] == hashlib.sha256(_canonical(reference)).hexdigest(),
            "Selected reference differs from closed audit cache",
        )
        self.groups_verified += 1

    def array(self, reference, key: str) -> np.ndarray:
        self.verify(reference)
        _need(key in reference["arrays"], "Missing selected array: " + key)
        item = reference["arrays"][key]
        _need(
            set(item) == {"sha256", "shape", "dtype", "encoding", "offset", "byte_length"},
            "Array descriptor keys differ",
        )
        dtype = np.dtype(item["dtype"])
        shape = item["shape"]
        _need(
            not dtype.hasobject
            and dtype.fields is None
            and dtype.kind in "biufcSU"
            and dtype.itemsize > 0,
            "Unsupported selected dtype",
        )
        _need(
            isinstance(shape, list)
            and len(shape) <= 8
            and all(type(n) is int and n >= 0 for n in shape),
            "Invalid selected shape",
        )
        offset, length = item["offset"], item["byte_length"]
        _need(
            type(offset) is int and type(length) is int and min(offset, length) >= 0,
            "Invalid selected byte range",
        )
        _need(
            math.prod(shape) * dtype.itemsize == length and length <= 16 * MIB,
            "Unbounded selected array",
        )
        _need(offset + length <= self.raw.stat().st_size, "Selected bytes escape raw stream")
        with self.raw.open("rb", buffering=0) as stream:
            stream.seek(offset)
            payload = self.budget.read(stream, length, "selected_array_C_bytes")
        _need(
            len(payload) == length and hashlib.sha256(payload).hexdigest() == item["sha256"],
            "Selected array C-byte SHA256 differs",
        )
        return np.frombuffer(payload, dtype=dtype).reshape(shape)

    def json_array(self, reference, key):
        value = self.array(reference, key)
        _need(value.shape == () and value.dtype.kind in "SU", "Expected scalar source JSON")
        return _json(value.item())

    def _line(self, stream, offset):
        stream.seek(offset)
        if offset:
            stream.seek(offset - 1)
            if self.budget.read(stream, 1, "selected_index") != b"\n":
                skipped = self.budget.read(stream, self.line_limit + 1, "selected_index", line=True)
                _need(
                    len(skipped) <= self.line_limit and skipped.endswith(b"\n"),
                    "Overlong/truncated source index line",
                )
        start = stream.tell()
        payload = self.budget.read(stream, self.line_limit + 1, "selected_index", line=True)
        if not payload:
            return None
        _need(
            len(payload) <= self.line_limit and payload.endswith(b"\n"),
            "Overlong/truncated source index line",
        )
        row = _json(payload)
        _need(
            isinstance(row, dict) and GROUP.fullmatch(str(row.get("group", ""))) is not None,
            "Malformed source index group",
        )
        return start, stream.tell(), row

    def find_group(self, ordinal: int) -> dict[str, Any]:
        _need(type(ordinal) is int and ordinal >= 0, "Invalid source group ordinal")
        low, high, hit = 0, self.size, None
        with self.index.open("rb", buffering=0) as stream:
            for _ in range(64):
                if high - low <= 4 * self.line_limit:
                    cursor = max(0, low - self.line_limit)
                    end = min(self.size, high + 2 * self.line_limit)
                    while cursor < end:
                        line = self._line(stream, cursor)
                        if line is None:
                            break
                        start, cursor, row = line
                        number = int(GROUP.fullmatch(row["group"])[1])
                        if number == ordinal:
                            hit = (start, cursor, row)
                            break
                        if number > ordinal:
                            break
                    break
                midpoint = (low + high) // 2
                line = self._line(stream, midpoint)
                if line is None:
                    high = midpoint
                    continue
                start, end, row = line
                number = int(GROUP.fullmatch(row["group"])[1])
                if number == ordinal:
                    hit = line
                    break
                if number < ordinal:
                    low = end
                else:
                    high = min(start, midpoint) if start >= high else start
            _need(hit is not None, "Selected source group not found within bounded index search")
            start, end, row = hit
            if row.get("event") == "arrays_complete":
                begin = max(0, start - self.line_limit)
                stream.seek(begin)
                prefix = self.budget.read(stream, start - begin, "selected_index")
                lines = prefix.splitlines(keepends=True)
                _need(bool(lines) and lines[-1].endswith(b"\n"), "Missing source array intent")
                row = _json(lines[-1])
                end = start
            _need(row.get("event") == "arrays_requested", "Source group lacks intent")
            reference = {k: v for k, v in row.items() if k != "event"}
            _need(
                int(GROUP.fullmatch(reference["group"])[1]) == ordinal,
                "Source index ordinal differs",
            )
            completion = self._line(stream, end)
            _need(completion is not None, "Source group lacks completion")
            maximum = max(
                (v["offset"] + v["byte_length"] for v in reference["arrays"].values()), default=0
            )
            _need(
                completion[2]
                == dict(
                    event="arrays_complete",
                    format=FORMAT,
                    group=reference["group"],
                    end_offset=maximum,
                ),
                "Source array completion differs",
            )
            self.verify(reference)
            return reference


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _typed(value: np.ndarray, shape: tuple[int, ...], dtype: str, name: str) -> np.ndarray:
    _need(
        value.shape == shape and value.dtype == np.dtype(dtype),
        "Source shape/dtype differs: " + name,
    )
    _need(
        value.dtype.kind not in "fc" or np.isfinite(value).all(), "Nonfinite source array: " + name
    )
    return value


def _episode_arrays(archive: _SelectedArchive, source: Mapping[str, Any]):
    event = source["exact_episode_complete_event"]
    agreement = source["exact_teacher_agreement_complete_event"]
    _need(
        event["steps"] == 1000 and event["complete"] and event["success"] and event["survived"],
        "Ineligible independent teacher source",
    )
    raw = event["raw"]
    _need(
        agreement["episode_raw"] == raw and agreement["rows"] == 1000,
        "Episode/prediction join differs",
    )
    shapes = dict(
        observations=(1000, 17),
        actions=(1000, 6),
        states=(1001, 19),
        warmstarts=(1001, 9),
        integrations=(1001, 91),
        costs=(1000,),
        velocities=(1000,),
    )
    arrays = {k: _typed(archive.array(raw, k), v, "<f8", k) for k, v in shapes.items()}
    p = agreement["predictions"]
    arrays["policy_actions"] = _typed(
        archive.array(p["reference"], p["key"]), (1000, 6), "<f4", "policy_actions"
    )
    p = agreement["squared_disagreement"]
    arrays["squared_disagreement"] = _typed(
        archive.array(p["reference"], p["key"]), (1000,), "<f8", "squared_disagreement"
    )
    _need(
        np.all(np.abs(arrays["actions"]) <= 1) and np.all(np.abs(arrays["policy_actions"]) <= 1),
        "Unbounded recorded action",
    )
    ordinal = int(GROUP.fullmatch(raw["group"])[1])
    before = archive.find_group(ordinal - 2)
    after = archive.find_group(ordinal - 1)
    _need(
        re.fullmatch(r"q[0-9]{9}_inputs", before["tag"]) is not None
        and after["tag"] == before["tag"][:-7] + "_result",
        "Episode predecessor is not its last live input/result pair",
    )
    _need(
        set(before["arrays"])
        == {"action", "state", "warmstart", "integration", "spec", "dimensions"},
        "Last query is not a full live context",
    )
    _need(
        set(after["arrays"])
        == {"observation", "cost", "done", "state", "warmstart", "integration", "info"},
        "Last query is not a native live return",
    )
    for name, key in [
        ("action", "actions"),
        ("state", "states"),
        ("warmstart", "warmstarts"),
        ("integration", "integrations"),
    ]:
        expected = arrays[key][-1 if name == "action" else -2]
        _need(
            _same_bytes(archive.array(before, name), expected),
            "Last live input differs from episode prefix: " + name,
        )
    for name, key in [
        ("state", "states"),
        ("warmstart", "warmstarts"),
        ("integration", "integrations"),
    ]:
        _need(
            _same_bytes(archive.array(after, name), arrays[key][-1]),
            "Terminal live context differs: " + name,
        )
    cost = _typed(archive.array(after, "cost"), (), "<f8", "last cost")
    _need(_same_bytes(cost, arrays["costs"][-1:][0]), "Last live cost differs")
    done = archive.array(after, "done")
    _need(
        done.shape == () and done.dtype.kind == "b" and not bool(done),
        "Qualified teacher has terminal done",
    )
    terminal = _typed(archive.array(after, "observation"), (17,), "<f8", "terminal observation")
    spec = archive.array(before, "spec")
    dims = archive.array(before, "dimensions")
    _need(
        spec.shape == () and spec.dtype.kind in "iu" and int(spec) > 0,
        "Missing integration specification",
    )
    _need(
        dims.shape == (3,) and dims.dtype.kind in "iu" and np.array_equal(dims, [9, 9, 6]),
        "Native dimensions differ",
    )
    infos = archive.json_array(raw, "infos")
    _need(
        isinstance(infos, dict) and set(infos) == {"rows"} and len(infos["rows"]) == 1000,
        "Native row metadata differs",
    )
    native = [v["native"] for v in infos["rows"]]
    _need(native[-1] == archive.json_array(after, "info"), "Last native info differs from episode")
    falls = np.asarray([v["fall_cost"] for v in native], dtype=np.float64)
    velocities = np.asarray([v["x_velocity"] for v in native], dtype=np.float64)
    _need(
        np.isfinite(falls).all()
        and np.all(falls == 0)
        and _same_bytes(velocities, arrays["velocities"]),
        "Source native fall/velocity metadata differs",
    )
    arrays["observations"] = np.concatenate((arrays["observations"], terminal[None]), axis=0)
    arrays.update(
        integration_spec=spec,
        dimensions=dims,
        seed=np.asarray(event["seed"], dtype=np.int64),
        fall_costs=falls,
    )
    return arrays, dict(
        last_live_inputs=before,
        last_live_result=after,
        terminal_observation_key="observation",
        integration_spec_key="spec",
        dimensions_key="dimensions",
    )


def prepare_inputs(
    source_run: Path,
    source_bindings: Path,
    audit_index: Path,
    audit_index_binding: Mapping[str, Any],
    out: Path,
    *,
    source_bindings_sha256: str,
    max_read_bytes: int = 512 * MIB,
    max_output_bytes: int = 256 * MIB,
    deadline_seconds: float = 600,
    max_rss_bytes: int = 512 * MIB,
) -> dict[str, Any]:
    """Create C1 inputs from authenticated selected H5 bytes, or stop closed.

    Output is create-only. Failures retain partial output and a small receipt;
    no retries, alternate sources, source mutations or paid science occur.
    All paths and the closed-audit cache binding must be frozen by the caller.
    """
    source_run, source_bindings, audit_index, out = map(
        Path, (source_run, source_bindings, audit_index, out)
    )
    source_run = source_run.resolve()
    _need(not out.exists(), "Prepared output already exists")
    out.mkdir(parents=True, exist_ok=False)
    budget = _Budget(max_read_bytes, max_output_bytes, deadline_seconds, max_rss_bytes)
    archive = None
    try:
        binding = dict(bytes=source_bindings.stat().st_size, sha256=source_bindings_sha256)
        sources = _json(budget.whole(source_bindings.resolve(), binding, "source_binding_JSON"))
        _need(
            sources["source_eligibility"]["completed_R1_scientific_audit_passed"] is True
            and sources["source_eligibility"]["all_12_independent_sources_full_walking"] is True,
            "Full audited independent-source eligibility is required",
        )
        for relative in (
            "config.json",
            "manifest.json",
            "input_manifest.json",
            "worker_result.json",
        ):
            expected = sources["metadata_files"]["run/" + relative]
            budget.whole(_safe_file(source_run, relative), expected, "bound_run_JSON")
        archive = _SelectedArchive(
            source_run, sources, audit_index.resolve(), audit_index_binding, budget
        )
        expected_tags = [f"teacher_s{s}_independent_{i:02d}" for s in range(3) for i in range(4)]
        episodes = sources["source_episodes_in_fixed_training_order"]
        _need(
            [v["tag"] for v in episodes] == expected_tags,
            "The frozen 12-source roster/order differs",
        )
        for number, source in enumerate(episodes):
            _need(
                source["stream"] == number // 4
                and source["case"] == number % 4
                and source["exact_episode_complete_event"]["tag"] == source["tag"]
                and source["exact_episode_complete_event"]["seed"] == source["source_reset_seed"],
                "Source episode/reset ownership differs",
            )
        catalog: dict[str, Any] = dict(
            schema="walker-consequence-inputs-v1",
            status="complete",
            source_run=str(source_run),
            episodes={},
            streams={},
            parents={},
            inputs={},
        )
        catalog["source_bindings"] = _copy_file(
            source_bindings.resolve(), out, "metadata/source_bindings.json", binding, budget
        )
        catalog["source_run_metadata"] = {}
        for relative in (
            "config.json",
            "manifest.json",
            "input_manifest.json",
            "worker_result.json",
        ):
            expected = sources["metadata_files"]["run/" + relative]
            catalog["source_run_metadata"][relative] = _copy_file(
                _safe_file(source_run, relative),
                out,
                "metadata/h5_source/" + relative,
                expected,
                budget,
            )
        catalog["authenticated_audit_cache"] = dict(
            dict(audit_index_binding),
            file=str(audit_index.resolve()),
            deserialized_into_memory=True,
            original_read_only=True,
        )
        stream_parts: dict[int, list[dict[str, np.ndarray]]] = {i: [] for i in range(3)}
        common_spec = None
        for source in episodes:
            stream = source["stream"]
            parent = sources["incoming_parents"][str(stream)]["checkpoint_record"]
            _need(
                source["exact_teacher_agreement_complete_event"]["actor_hash"]
                == parent["actor_hash"],
                "Source prediction actor differs from retained parent",
            )
            arrays, terminal_binding = _episode_arrays(archive, source)
            actual_spec = int(arrays["integration_spec"])
            common_spec = actual_spec if common_spec is None else common_spec
            _need(
                actual_spec == common_spec,
                "Source integration specification changed across episodes",
            )
            saved = _write_arrays(out, f"episodes/{source['tag']}.npz", arrays, budget)
            catalog["episodes"][source["tag"]] = saved | dict(
                stream=stream,
                case=source["case"],
                seed=source["source_reset_seed"],
                source_raw=source["exact_episode_complete_event"]["raw"],
                source_predictions=source["exact_teacher_agreement_complete_event"]["predictions"],
                source_squared_disagreement=source["exact_teacher_agreement_complete_event"][
                    "squared_disagreement"
                ],
                terminal_binding=terminal_binding,
            )
            stream_parts[stream].append(
                dict(
                    observations=arrays["observations"][:-1].astype(np.float32),
                    actions=arrays["actions"].astype(np.float32),
                    policy_actions=arrays["policy_actions"],
                    squared_disagreement=arrays["squared_disagreement"],
                    episode_index=np.full(1000, source["case"], dtype=np.int64),
                    control_index=np.arange(1000, dtype=np.int64),
                )
            )
        for stream, parts in stream_parts.items():
            arrays = {
                key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]
            }
            catalog["streams"][str(stream)] = _write_arrays(
                out, f"streams/s{stream}_corpus.npz", arrays, budget
            ) | dict(rows=4000, episode_tags=expected_tags[4 * stream : 4 * stream + 4])
        del stream_parts
        for stream in range(3):
            owner = sources["incoming_parents"][str(stream)]
            checkpoint = owner["checkpoint_record"]
            copied = _copy_file(
                _safe_file(source_run, checkpoint["file"]),
                out,
                f"parents/s{stream}_parent.pt",
                checkpoint,
                budget,
            )
            rng_ref = checkpoint["global_rng"]
            rng = archive.array(rng_ref["reference"], rng_ref["key"])
            _need(rng.dtype == np.dtype("u1") and rng.ndim == 1, "Invalid global Torch RNG bytes")
            sidecar = _write_arrays(
                out, f"parents/s{stream}_global_torch_rng.npy", {"state": rng}, budget
            )
            catalog["parents"][str(stream)] = copied | dict(
                actor_hash=checkpoint["actor_hash"],
                incoming_optimizer_sha256=checkpoint["optimizer_sha256"],
                global_torch_rng=sidecar,
                original_checkpoint_record=checkpoint,
            )
        for name, value in sources["normalization_and_historical_inputs"]["config_inputs"].items():
            catalog["inputs"][name] = _copy_file(
                _safe_file(source_run, value["file"]), out, value["file"], value, budget
            )
        catalog["normalization"] = sources["normalization_and_historical_inputs"]["config_inputs"][
            "normalization"
        ]
        catalog["integration_spec"] = common_spec
        catalog["dimensions"] = [9, 9, 6]
        catalog["selected_groups_verified"] = archive.groups_verified
        budget.unchanged()
        catalog["work"] = budget.summary()
        _write_json(out, "catalog.json", catalog, budget)
        files = sorted(budget.output_files.values(), key=lambda value: value["file"])
        _write_json(
            out,
            "prepared_input_manifest.json",
            dict(
                schema="walker-consequence-prepared-input-manifest-v1",
                files=files,
                manifest_excludes_itself_and_final_receipt=True,
            ),
            budget,
        )
        receipt = dict(
            schema="walker-consequence-input-preparation-receipt-v1",
            complete=True,
            work=budget.summary(),
            work_boundary="All reads and data/catalog/manifest writes complete; final receipt write itself excluded from output-reservation/wall counters.",
            source_unchanged=True,
            selected_source_references_authenticated=True,
            native_replay_gate_passed=False,
            C1_prediction_reduction_or_learning_executed=False,
        )
        _write_json(out, "preparation_receipt.json", receipt, budget)
        budget.unchanged()
        return catalog
    except BaseException as error:
        try:
            _write_json(
                out,
                "preparation_failure.json",
                dict(
                    schema="walker-consequence-input-preparation-failure-v1",
                    complete=False,
                    error_type=type(error).__name__,
                    error=str(error),
                    work=budget.summary(),
                    no_retry=True,
                ),
                budget,
                emergency=True,
            )
        except BaseException:
            pass
        raise
    finally:
        if archive is not None:
            archive.close()
