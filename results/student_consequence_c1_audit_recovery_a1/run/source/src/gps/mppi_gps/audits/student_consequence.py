"""Independent saved-only C1 trace, assay and accounting checks.

No worker, simulator, optimizer or production prefix decoder is imported.
General array-stream structural verification is reused from the independent
H5 auditor. A structural receipt explicitly does not certify the complete
scientific contrast; additional source, fitting and standalone joins are
required for that verdict.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import resource
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .teacher_proposal import (
    AuditError,
    AuditJournal,
    SavedActorVerifier,
    StreamArchive,
    optimizer_fingerprint,
)

TRACE_KEYS = (
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
BOUNDARY_KEYS = TRACE_KEYS[:4]
MAX_READ_BYTES = 64 * 1024**3
MAX_OUTPUT_BYTES = 4 * 1024**3
MAX_RSS_BYTES = 4 * 1024**3
MAX_WALL_SECONDS = 7200


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _json(value: bytes | str) -> Any:
    def pairs(items):
        result = {}
        for name, item in items:
            _need(name not in result, "Duplicate C1 JSON key")
            result[name] = item
        return result

    try:
        return json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(AuditError("Nonfinite C1 JSON")),
        )
    except (ValueError, UnicodeError) as error:
        raise AuditError("Malformed C1 JSON") from error


def _integer(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    _need(type(value) is int and value >= minimum, "Invalid C1 integer: " + label)
    _need(maximum is None or value <= maximum, "C1 integer exceeds bound: " + label)
    return value


def _array(value: Any, shape: tuple[int, ...], dtype: str, label: str) -> np.ndarray:
    value = np.asarray(value)
    _need(value.shape == shape and value.dtype == np.dtype(dtype), "C1 dtype/shape: " + label)
    _need(value.dtype.kind not in "fc" or np.isfinite(value).all(), "Nonfinite C1 " + label)
    return value


def _bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def verify_trace_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Independently verify the thirteen saved arrays and returned-row chain."""
    _need(set(arrays) == set(TRACE_KEYS), "C1 trace must contain exactly thirteen named arrays")
    encoded = np.asarray(arrays["metadata"])
    _need(encoded.shape == () and encoded.dtype.kind in "SU", "C1 scalar metadata encoding")
    text = encoded.item()
    if isinstance(text, bytes):
        text = text.decode("ascii")
    _need(isinstance(text, str) and text.isascii() and len(text) <= 1024, "C1 metadata bound")
    metadata = _json(text)
    _need(isinstance(metadata, dict), "C1 metadata is not a dictionary")
    horizon = _integer(metadata.get("horizon"), "horizon", 1, 1000)
    count = _integer(metadata.get("completed_controls"), "completed_controls", 0, horizon)
    integration = np.asarray(arrays["integrations"])
    _need(integration.ndim == 2 and 1 <= integration.shape[1] <= 128, "C1 integration width")
    for key, width in [
        ("observations", 17),
        ("states", 19),
        ("warmstarts", 9),
        ("integrations", integration.shape[1]),
    ]:
        _array(arrays[key], (count + 1, width), "<f8", key)
    _array(arrays["actions"], (count, 6), "<f8", "actions")
    for key in ("native_costs", "x_velocities", "fall_costs"):
        _array(arrays[key], (count,), "<f8", key)
    done = _array(arrays["done"], (count,), "u1", "done")
    _need(np.all(done <= 1) and not np.any(done[:-1]), "C1 has post-fall controls/nonbinary done")
    _need(np.all(np.abs(arrays["actions"]) <= 1), "C1 unbounded action")
    ordinals = _array(arrays["control_ordinals"], (count,), "<u4", "control_ordinals")
    _need(np.array_equal(ordinals, np.arange(count, dtype="<u4")), "C1 noncontiguous controls")
    digests = _array(arrays["row_digests"], (count, 32), "u1", "row_digests")
    initial = _array(arrays["initial_digest"], (32,), "u1", "initial_digest")
    digest = hashlib.sha256()
    for key in BOUNDARY_KEYS:
        digest.update(np.asarray(arrays[key][0]).tobytes())
    previous = digest.digest()
    _need(previous == initial.tobytes(), "C1 initial boundary digest differs")
    for index in range(count):
        digest = hashlib.sha256(previous)
        for key in BOUNDARY_KEYS:
            digest.update(np.asarray(arrays[key][index + 1]).tobytes())
        for key in TRACE_KEYS[4:10]:
            digest.update(np.asarray(arrays[key][index]).tobytes())
        previous = digest.digest()
        _need(previous == digests[index].tobytes(), "C1 returned-row digest differs")
    availability = metadata.get("availability")
    _need(
        availability in {"closed_actual_prefix", "failed_actual_prefix"}, "C1 unknown availability"
    )
    failed = availability == "failed_actual_prefix"
    fell = bool(count and done[-1])
    unknown = _integer(metadata.get("unknown_partial_controls"), "unknown partial controls", 0, 1)
    _need(
        metadata.get("unknown_partial_physics_substeps") == 4 * unknown, "C1 partial substep count"
    )
    _need(
        failed or (not unknown and (count == horizon or fell)), "C1 falsely normal partial closure"
    )
    absorbing = metadata.get("absorbing_cost")
    _need(
        type(absorbing) in (int, float) and math.isfinite(absorbing), "C1 nonfinite absorbing cost"
    )
    expected_absorbing = 5.0 * (horizon - count) if fell and not failed else 0.0
    _need(float(absorbing) == expected_absorbing, "C1 absorbing-tail cost differs")
    if "pending_prediction" in metadata:
        pending = metadata["pending_prediction"]
        _need(
            failed and count < horizon and not fell and pending.get("ordinal") == count,
            "C1 closed pending prediction is not an available next action",
        )
        actor = pending.get("actor")
        _need(
            isinstance(actor, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", actor) is not None
            and actor == metadata.get("prediction_actor"),
            "C1 closed prediction actor binding differs",
        )
        action = np.asarray(pending["action"], dtype="<f8")
        _need(
            action.shape == (6,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1),
            "C1 closed pending action differs",
        )
        digest = hashlib.sha256(previous)
        digest.update(np.array(count + 1, dtype="<u4").tobytes())
        digest.update(action.tobytes())
        digest.update(np.array(actor, dtype="S80").tobytes())
        _need(
            digest.hexdigest() == pending.get("sha256"),
            "C1 closed pending prediction digest differs",
        )
    if task is not None:
        verify_native_measurements(arrays, task)
    native_cost = float(np.sum(arrays["native_costs"], dtype=np.float64))
    native_terminal = float(np.sum(arrays["fall_costs"], dtype=np.float64))
    return dict(
        alias=metadata.get("alias"),
        metadata=metadata,
        horizon=horizon,
        controls=count,
        done=fell,
        normally_closed=not failed,
        raw_cost=native_cost - native_terminal,
        native_cost=native_cost,
        native_terminal=native_terminal,
        absorbing_cost=float(absorbing),
        cost=native_cost + float(absorbing),
        unknown_partial_controls=unknown,
        unknown_partial_physics_substeps=4 * unknown,
        initial_digest=initial.tobytes().hex(),
        final_digest=previous.hex(),
    )


def verify_native_measurements(arrays: Mapping[str, np.ndarray], task: Mapping[str, Any]) -> None:
    """Recompute observations, health, velocity and native task cost; no physics."""
    states, actions = arrays["states"], arrays["actions"]
    observations = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
    _need(
        np.array_equal(observations, arrays["observations"]),
        "C1 native clipped observation differs",
    )
    posts = states[1:]
    healthy = (posts[:, 2] > 0.8) & (posts[:, 2] < 2) & (posts[:, 3] > -1) & (posts[:, 3] < 1)
    _need(
        np.array_equal((~healthy).astype(np.uint8), arrays["done"]), "C1 native health/done differs"
    )
    _need(np.array_equal(posts[:, 10], arrays["x_velocities"]), "C1 native velocity differs")
    _need(task["cost_style"] == "target_velocity", "C1 task cost style changed")
    costs = (
        task["vel_cost_weight"] * np.abs(task["target_velocity"] - posts[:, 10])
        + task["vertical_velocity_cost_weight"] * posts[:, 11] ** 2
        + task["angle_cost_weight"] * posts[:, 3] ** 2
        + task["unhealthy_cost_weight"] * (~healthy)
        + task["ctrl_cost_weight"] * np.sum(actions**2, axis=1)
    )
    falls = task["terminal_unhealthy_weight"] * (~healthy).astype(float)
    _need(np.array_equal(falls, arrays["fall_costs"]), "C1 native terminal metadata differs")
    _need(
        np.allclose(costs + falls, arrays["native_costs"], rtol=0, atol=1e-10),
        "C1 native cost differs",
    )


def compare_baseline_arrays(
    actual: Mapping[str, np.ndarray], source: Mapping[str, np.ndarray]
) -> dict[str, float]:
    """Independent twelve-source replay gate, preserving its exact tolerances."""
    mapping = {
        "observations": "observations",
        "states": "states",
        "warmstarts": "warmstarts",
        "integrations": "integrations",
        "actions": "actions",
        "native_costs": "costs",
        "x_velocities": "velocities",
        "fall_costs": "fall_costs",
    }
    differences = {}
    for key, source_key in mapping.items():
        left, right = np.asarray(actual[key]), np.asarray(source[source_key])
        _need(
            left.shape == right.shape and left.dtype == right.dtype, "C1 replay shape/dtype: " + key
        )
        if key in BOUNDARY_KEYS:
            _need(_bytes_equal(left[0], right[0]), "C1 restore not byte-exact: " + key)
        _need(np.allclose(left, right, rtol=0, atol=1e-10), "C1 baseline replay differs: " + key)
        differences[key] = float(np.max(np.abs(left - right), initial=0))
    _need(not np.any(actual["done"]), "C1 qualified baseline teacher fell")
    return differences


def verify_assay_arrays(
    arrays: Mapping[str, np.ndarray],
    source: Mapping[str, np.ndarray],
    t: int,
    *,
    baseline_costs: np.ndarray,
) -> dict[str, Any]:
    summary = verify_trace_arrays(arrays)
    t = _integer(t, "assay anchor", 0, 999)
    horizon = min(100, 1000 - t)
    _need(
        summary["horizon"] == horizon and summary["normally_closed"],
        "C1 incomplete/wrong assay window",
    )
    count = summary["controls"]
    _need(count >= 1, "C1 assay has no returned first action")
    for key in BOUNDARY_KEYS:
        _need(_bytes_equal(arrays[key][0], source[key][t]), "C1 assay restore binding: " + key)
    expected_first = np.asarray(source["policy_actions"][t], dtype="<f8")
    _need(
        _bytes_equal(arrays["actions"][0], expected_first), "C1 saved parent substitution differs"
    )
    _need(
        _bytes_equal(arrays["actions"][1:], source["actions"][t + 1 : t + count]),
        "C1 fixed teacher suffix differs",
    )
    baseline_costs = _array(baseline_costs, (1000,), "<f8", "validated replay costs")
    baseline = float(np.sum(baseline_costs[t : t + horizon], dtype=np.float64))
    return summary | dict(t=t, h=horizon, baseline_cost=baseline, delta=summary["cost"] - baseline)


def recompute_weights(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C1's one predeclared rule; preserve float64 then actual float32 bytes."""
    values = np.asarray(deltas)
    _need(
        values.shape == (4000,) and values.dtype == np.float64 and np.isfinite(values).all(),
        "C1 delta roster",
    )
    raw = 1.0 + np.minimum(9.0, np.maximum(0.0, values) / 20.0)
    normalized = raw / np.mean(raw, dtype=np.float64)
    return normalized, normalized.astype(np.float32)


def recompute_shuffled_weights(
    weights: np.ndarray,
    squared_errors: np.ndarray,
    seed: int,
    *,
    details: bool = False,
) -> tuple:
    """Independent 624-group null construction across three separate calls."""
    weights, errors = np.asarray(weights), np.asarray(squared_errors)
    _need(
        weights.shape == errors.shape == (4000,)
        and np.isfinite(weights).all()
        and np.isfinite(errors).all(),
        "C1 shuffle rows",
    )
    _integer(seed, "shuffle seed")
    generator = np.random.Generator(np.random.PCG64(seed))
    initial = deepcopy(generator.bit_generator.state)
    groups = []
    indices = np.arange(4000, dtype=np.int64)
    boundaries = [(100 * i, 100 * (i + 1)) for i in range(9)] + [
        (990, 1000),
        (975, 990),
        (950, 975),
        (900, 950),
    ]
    for episode in range(4):
        offset = 1000 * episode
        for temporal, (begin, end) in enumerate(boundaries):
            members = np.arange(offset + begin, offset + end, dtype=np.int64)
            ranked = members[np.lexsort((members, errors[members]))]
            for quartile, group in enumerate(np.array_split(ranked, 4)):
                before = deepcopy(generator.bit_generator.state)
                permutation = generator.permutation(group)
                indices[group] = permutation
                groups.append(
                    dict(
                        group=len(groups),
                        episode=episode,
                        temporal=temporal,
                        quartile=quartile,
                        rows=len(group),
                        destination=group.tolist(),
                        source=permutation.tolist(),
                        rng_before=before,
                        rng_after=deepcopy(generator.bit_generator.state),
                    )
                )
    result = np.asarray(weights[indices], dtype=weights.dtype), indices
    return (
        (
            *result,
            dict(
                groups=groups,
                rng_initial=initial,
                rng_terminal=deepcopy(generator.bit_generator.state),
            ),
        )
        if details
        else result
    )


def _peak_rss() -> int:
    amount = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return amount if sys.platform == "darwin" else amount * 1024


def verify_active_spool(directory: Path) -> dict[str, Any]:
    """Independently inspect a stopped temporary prefix, without repair.

    Digest-valid uncommitted native rows and pending actor predictions remain
    separate from the marker-committed prefix. Journal ownership must still be
    joined before classifying any outstanding work as returned or unknown.
    """
    directory = Path(directory).resolve()
    paths = [directory / "active.json", directory / "active.npy"]
    _need(all(p.is_file() and not p.is_symlink() for p in paths), "Missing C1 active spool")
    identities = [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    _need(
        sum(size for size, _ in identities) <= 8 * 1024**2 and identities[0][0] <= 4096,
        "C1 active spool byte bound",
    )
    manifest = _json(paths[0].read_bytes())
    _need(manifest.get("schema") == "walker-c1-active-prefix-v1", "Unknown C1 active spool schema")
    horizon = _integer(manifest.get("horizon"), "active horizon", 1, 1000)
    width = _integer(manifest.get("integration_size"), "active integration size", 1, 128)
    # Independently declared temporary dtype; it is not the archive format.
    dtype = np.dtype(
        [
            ("observations", "<f8", (17,)),
            ("states", "<f8", (19,)),
            ("warmstarts", "<f8", (9,)),
            ("integrations", "<f8", (width,)),
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
    try:
        with paths[1].open("rb") as stream:
            payload = stream.read(identities[1][0] + 1)
        _need(len(payload) == identities[1][0], "C1 active spool size changed")
        rows = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError, EOFError) as error:
        raise AuditError("Truncated/unreadable C1 active NumPy spool") from error
    try:
        _need(rows.dtype == dtype and rows.shape == (horizon + 1,), "C1 active dtype/shape")
        _need(int(rows[0]["valid"]) == 0, "C1 active initial boundary not committed")
        sentinel = np.iinfo(np.uint32).max
        count = 0
        for position in range(1, horizon + 1):
            marker = int(rows[position]["valid"])
            if marker == sentinel:
                _need(
                    np.all(rows["valid"][position + 1 :] == sentinel), "Nonprefix C1 active marker"
                )
                break
            _need(marker == position, "C1 active marker differs from ordinal")
            count = position

        def prefix(length):
            arrays = {key: np.array(rows[key][: length + 1], copy=True) for key in TRACE_KEYS[:4]}
            arrays.update(
                {key: np.array(rows[key][1 : length + 1], copy=True) for key in TRACE_KEYS[4:11]}
            )
            arrays["initial_digest"] = np.array(rows[0]["row_digests"], copy=True)
            # Audit-derived validation metadata, never saved as original data.
            arrays["metadata"] = np.array(
                json.dumps(
                    dict(
                        alias=manifest["alias"],
                        horizon=horizon,
                        completed_controls=length,
                        availability="failed_actual_prefix",
                        absorbing_cost=0.0,
                        unknown_partial_controls=0,
                        unknown_partial_physics_substeps=0,
                    )
                )
            )
            return arrays

        arrays = prefix(count)
        summary = verify_trace_arrays(arrays)
        additional_return = None
        pending_prediction = None
        if count < horizon:
            row = rows[count + 1]
            previous = bytes.fromhex(summary["final_digest"])
            digest = hashlib.sha256(previous)
            for key in TRACE_KEYS[:10]:
                digest.update(np.asarray(row[key]).tobytes())
            if digest.digest() == row["row_digests"].tobytes():
                verify_trace_arrays(prefix(count + 1))
                additional_return = count
            marker = int(row["prediction_valid"])
            if marker != sentinel:
                actor = row["prediction_actor"].tobytes().decode("ascii")
                digest = hashlib.sha256(previous)
                digest.update(np.array(count + 1, dtype="<u4").tobytes())
                digest.update(row["actions"].tobytes())
                digest.update(np.array(row["prediction_actor"], dtype="S80").tobytes())
                _need(
                    marker == count + 1
                    and bool(actor)
                    and actor.isascii()
                    and np.isfinite(row["actions"]).all()
                    and np.all(np.abs(row["actions"]) <= 1)
                    and digest.digest() == row["prediction_digest"].tobytes(),
                    "C1 pending actor prediction differs",
                )
                pending_prediction = dict(
                    actor=actor,
                    ordinal=count,
                    action=row["actions"].tolist(),
                    sha256=digest.hexdigest(),
                )
        _need(
            identities == [(p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
            "C1 active spool changed during audit",
        )
        return dict(
            schema="walker-c1-active-spool-audit-v1",
            verified=True,
            normally_closed=False,
            committed_controls=count,
            additional_digest_valid_uncommitted_return=additional_return,
            pending_prediction=pending_prediction,
            alias=manifest["alias"],
            journal_ownership_verified=False,
            source_read_bytes=sum(size for size, _ in identities),
        )
    finally:
        payload = b""


def audit_saved_stream(
    evidence: Path,
    out: Path,
    *,
    task: Mapping[str, Any] | None = None,
    max_read_bytes: int = MAX_READ_BYTES,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    _trace_check: Callable[[dict, Mapping[str, np.ndarray], dict], None] | None = None,
    _event_check: Callable[[dict], None] | None = None,
    _archive_check: Callable[[StreamArchive, Mapping[str, dict]], None] | None = None,
    _archive_open: Callable[[StreamArchive], None] | None = None,
) -> dict[str, Any]:
    """Bounded structural/trace audit, including graceful incomplete prefixes.

    This entry point never labels its receipt a full scientific pass. It is
    useful when a prerequisite gate stops the worker before fitting begins.
    Disk-corrupted index/payloads fail explicitly; originals remain untouched.
    """
    evidence, out = Path(evidence).resolve(), Path(out).resolve()
    _need(not out.is_relative_to(evidence), "C1 audit output must be outside evidence")
    _need(0 < max_read_bytes <= MAX_READ_BYTES, "C1 read budget exceeds64GiB")
    _need(65536 <= max_output_bytes <= MAX_OUTPUT_BYTES, "C1 output budget exceeds4GiB")
    _need(0 < max_wall_seconds <= MAX_WALL_SECONDS, "C1 audit deadline exceeds2h")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + max_wall_seconds
    inventory: Counter[str] = Counter()
    references = {}
    pending = None
    traces = normal_traces = controls = unknown = 0
    requested = returned = 0
    output_bytes = 0
    receipt: dict[str, Any] = dict(
        schema="walker-c1-saved-stream-audit-v1",
        structural_passed=False,
        full_scientific_audit_passed=False,
        scope="stream_trace_and_journal_only",
        new_physics_controls=0,
        new_neural_rows=0,
        new_optimizer_steps=0,
    )
    archive = None
    try:
        with (
            StreamArchive(
                evidence,
                out / "stream_index.sqlite",
                max_read_bytes=max_read_bytes,
                max_database_bytes=min(512 * 1024**2, max_output_bytes // 2),
                max_metadata_bytes=8192,
                deadline=deadline,
            ) as archive,
            (out / "trace_summaries.jsonl").open("xb") as summaries,
        ):
            if _archive_open is not None:
                _archive_open(archive)
            archive.verify_index()
            for row in archive._lines(evidence / "arrays.index.jsonl"):
                if row["event"] == "arrays_requested":
                    reference = {key: value for key, value in row.items() if key != "event"}
                    _need(len(references) < 13468, "C1 array group census exceeded")
                    references[reference["group"]] = reference
            for row in archive.ledger():
                _need(
                    time.monotonic() <= deadline and _peak_rss() <= MAX_RSS_BYTES,
                    "C1 audit wall/RSS budget exhausted",
                )
                event = row["event"]
                if _event_check is not None:
                    _event_check(row)
                inventory[event] += 1
                _need(sum(inventory.values()) <= 5_000_000, "C1 ledger record census exceeded")
                if event not in {"runtime_open", "runtime_close"}:
                    _need(len(_canonical(row)) + 1 <= 512, "C1 generated compact event exceeds512B")
                if event == "trace_begin":
                    _need(pending is None, "Overlapping C1 native traces")
                    pending = dict(
                        alias=row["alias"],
                        horizon=row["horizon"],
                        actions=[],
                        digests=[],
                        intent=None,
                        failed=False,
                    )
                elif event == "trace_intent":
                    _need(
                        pending is not None
                        and row["alias"] == pending["alias"]
                        and pending["intent"] is None,
                        "Unowned C1 native intent",
                    )
                    _need(
                        row["ordinal"] == len(pending["actions"])
                        and row["requested_controls"] == 1
                        and row["requested_physics_substeps"] == 4,
                        "C1 native request census",
                    )
                    action = np.asarray(row["action"], dtype="<f8")
                    _need(
                        action.shape == (6,)
                        and np.isfinite(action).all()
                        and np.all(np.abs(action) <= 1),
                        "C1 intent action",
                    )
                    pending["intent"] = action
                    requested += 1
                elif event == "trace_returned":
                    _need(
                        pending is not None
                        and row["alias"] == pending["alias"]
                        and pending["intent"] is not None,
                        "Unowned C1 native completion",
                    )
                    _need(
                        row["ordinal"] == len(pending["actions"])
                        and row["completed_controls"] == 1
                        and row["completed_physics_substeps"] == 4,
                        "C1 native completion census",
                    )
                    pending["actions"].append(pending["intent"])
                    pending["digests"].append(row["digest"])
                    pending["intent"] = None
                    returned += 1
                elif event == "trace_error":
                    _need(
                        pending is not None and row["alias"] == pending["alias"],
                        "Unowned C1 trace failure",
                    )
                    pending["failed"] = True
                elif event == "trace_complete":
                    _need(
                        pending is not None and row["alias"] == pending["alias"],
                        "Unowned C1 closed trace",
                    )
                    reference = references.get(row["group"])
                    _need(
                        reference is not None
                        and hashlib.sha256(_canonical(reference)).hexdigest()
                        == row["reference_sha256"],
                        "C1 branch reference differs",
                    )
                    arrays = {key: archive.array(reference, key) for key in reference["arrays"]}
                    summary = verify_trace_arrays(arrays, task=task)
                    _need(
                        summary["alias"] == pending["alias"]
                        and summary["horizon"] == pending["horizon"],
                        "C1 trace metadata ownership",
                    )
                    _need(
                        summary["controls"] == len(pending["actions"]) == row["completed_controls"],
                        "C1 returned prefix/journal differs",
                    )
                    _need(
                        summary["normally_closed"] == (not pending["failed"]),
                        "C1 failure availability differs",
                    )
                    _need(
                        summary["unknown_partial_controls"] == int(pending["intent"] is not None),
                        "C1 unknown partial work differs",
                    )
                    if pending["actions"]:
                        _need(
                            _bytes_equal(arrays["actions"], np.asarray(pending["actions"])),
                            "C1 executed action/request differs",
                        )
                        _need(
                            [value.tobytes().hex() for value in arrays["row_digests"]]
                            == pending["digests"],
                            "C1 returned digest/completion differs",
                        )
                    if _trace_check is not None:
                        _trace_check(summary, arrays, reference)
                    compact = {key: value for key, value in summary.items() if key != "metadata"}
                    compact.update(
                        group=reference["group"],
                        close_event=row["id"],
                        kind=summary["metadata"].get("kind"),
                    )
                    payload = _canonical(compact) + b"\n"
                    output_bytes += len(payload)
                    _need(
                        output_bytes + (out / "stream_index.sqlite").stat().st_size + 65536
                        <= max_output_bytes,
                        "C1 audit output budget exhausted",
                    )
                    summaries.write(payload)
                    traces += 1
                    normal_traces += int(summary["normally_closed"])
                    controls += summary["controls"]
                    unknown += summary["unknown_partial_controls"]
                    pending = None
            archive.verify_unchanged()
            if _archive_check is not None:
                _archive_check(archive, references)
            receipt.update(
                structural_passed=True,
                complete_journal=pending is None,
                saved_closed_trace_groups=traces,
                normally_closed_traces=normal_traces,
                failed_prefix_trace_groups=traces - normal_traces,
                saved_controls=controls,
                requested_controls=requested,
                returned_controls=returned,
                unknown_partial_controls=unknown
                + int(pending is not None and pending["intent"] is not None),
                unknown_partial_physics_substeps=4
                * (unknown + int(pending is not None and pending["intent"] is not None)),
                event_inventory=dict(inventory),
                file_hashes=dict(archive.file_hashes),
                source_read_bytes=archive.read_bytes,
                array_groups=archive.group_checks,
                array_checks=archive.array_checks,
                pending_trace_alias=None if pending is None else pending["alias"],
            )
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        if archive is not None:
            receipt.update(source_read_bytes=archive.read_bytes, array_groups=archive.group_checks)
    receipt.update(wall_seconds=time.monotonic() - started, peak_rss_bytes=_peak_rss())
    payload = _canonical(receipt) + b"\n"
    _need(len(payload) <= 65536, "C1 audit receipt exceeds emergency reservation")
    with (out / "receipt.json").open("xb") as stream:
        stream.write(payload)
    return receipt


class _Inputs:
    """Bound small file reads and retain their exact immutable bindings."""

    def __init__(self, root: Path, deadline: float):
        self.root, self.deadline = root, deadline
        self.read_bytes = 0
        self.files: dict[str, dict[str, Any]] = {}
        self.identities: dict[Path, tuple[int, int]] = {}
        self.archive: StreamArchive | None = None
        self.checkpoint_read_bytes = 0

    def path(self, relative: str) -> Path:
        _need(
            isinstance(relative, str)
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "C1 audit source path escapes run",
        )
        path = (self.root / relative).resolve()
        _need(
            path.is_relative_to(self.root) and path.is_file(),
            "Missing/escaped C1 audit source: " + relative,
        )
        return path

    def read(
        self, relative: str, maximum: int = 16 * 1024**2, binding: Mapping[str, Any] | None = None
    ) -> bytes:
        _need(
            time.monotonic() <= self.deadline and _peak_rss() <= MAX_RSS_BYTES,
            "C1 audit wall/RSS exhausted",
        )
        path = self.path(relative)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns)
        _need(
            self.identities.setdefault(path, identity) == identity, "C1 source changed during audit"
        )
        archive_reads = 0 if self.archive is None else self.archive.read_bytes
        _need(
            stat.st_size <= maximum
            and self.read_bytes + archive_reads + self.checkpoint_read_bytes + stat.st_size + 1
            <= MAX_READ_BYTES,
            "C1 small-file read reservation exhausted",
        )
        with path.open("rb") as stream:
            payload = stream.read(stat.st_size + 1)
        self.read_bytes += len(payload)
        if self.archive is not None:
            self.archive.max_read_bytes = (
                MAX_READ_BYTES - self.read_bytes - self.checkpoint_read_bytes
            )
        _need(len(payload) == stat.st_size, "C1 source size changed during read")
        actual = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        if binding is not None:
            _need(
                binding.get("sha256") == actual["sha256"], "C1 small-file SHA differs: " + relative
            )
            if "bytes" in binding:
                _need(
                    binding["bytes"] == actual["bytes"], "C1 small-file size differs: " + relative
                )
        self.files[relative] = actual
        return payload

    def json(
        self, relative: str, maximum: int = 16 * 1024**2, binding: Mapping[str, Any] | None = None
    ) -> Any:
        return _json(self.read(relative, maximum, binding))

    def npz(self, relative: str, binding: Mapping[str, Any] | None = None) -> dict[str, np.ndarray]:
        with np.load(
            io.BytesIO(self.read(relative, 64 * 1024**2, binding)), allow_pickle=False
        ) as values:
            return {key: values[key].copy() for key in values.files}

    def npy(self, relative: str, binding: Mapping[str, Any] | None = None) -> np.ndarray:
        return np.load(io.BytesIO(self.read(relative, 16 * 1024**2, binding)), allow_pickle=False)

    def unchanged(self) -> None:
        for path, expected in self.identities.items():
            stat = path.stat()
            _need((stat.st_size, stat.st_mtime_ns) == expected, "C1 audited source changed")


def _fit_array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _source_bindings(inputs: _Inputs, config: dict) -> tuple[dict, dict, dict, dict]:
    """Authenticate the frozen recipe/export and all selected prepared bytes."""
    manifest = inputs.json("manifest.json")
    _need(
        inputs.files["config.json"]["sha256"] == manifest["config_sha256"],
        "C1 frozen config SHA differs",
    )
    for row in manifest["files"]:
        inputs.read("source/" + row["file"], binding=row)
    recipe = inputs.json(
        "source/configs/studies/student_consequence_c1_0.json",
        binding={"sha256": config["recipe_sha256"]},
    )
    active = dict(config)
    for key in ("prepared_manifest_sha256", "prerequisites_sha256", "recipe_sha256"):
        active.pop(key, None)
    _need(active == recipe, "C1 frozen config changes scientific recipe")
    prepared_manifest = inputs.json(
        "prepared/prepared_input_manifest.json",
        binding={"sha256": config["prepared_manifest_sha256"]},
    )
    for row in prepared_manifest["files"]:
        inputs.read("prepared/" + row["file"], 64 * 1024**2, row)
    prerequisites = inputs.json(
        "prerequisites.json", binding={"sha256": config["prerequisites_sha256"]}
    )
    _need(prerequisites.get("passed") is True, "C1 prelaunch prerequisites did not pass")
    for key in ("seed", "capacity", "arithmetic", "continuation"):
        _need(prerequisites[key].get("passed") is True, "Missing C1 prerequisite: " + key)
    catalog = inputs.json("prepared/catalog.json")
    _need(
        catalog.get("status") == "complete"
        and set(catalog["parents"]) == {"0", "1", "2"}
        and len(catalog["episodes"]) == 12,
        "C1 prepared roster differs",
    )
    sources = inputs.json(
        "prepared/metadata/source_bindings.json",
        binding={"sha256": config["source_bindings_sha256"]},
    )
    inherited_metadata = {}
    if "source_run_metadata" in catalog:
        _need(
            set(catalog["source_run_metadata"])
            == {"config.json", "manifest.json", "input_manifest.json", "worker_result.json"},
            "C1 inherited H5 metadata roster differs",
        )
        for relative, descriptor in catalog["source_run_metadata"].items():
            original = sources["metadata_files"]["run/" + relative]
            _need(
                descriptor["sha256"] == original["sha256"]
                and descriptor["bytes"] == original["bytes"],
                "C1 copied H5 metadata binding differs",
            )
            inherited_metadata[relative] = inputs.json(
                "prepared/" + descriptor["file"], binding=descriptor
            )
    for stream in range(3):
        descriptor = catalog["parents"][str(stream)]
        original_parent = sources["incoming_parents"][str(stream)]["checkpoint_record"]
        _need(
            descriptor["sha256"] == original_parent["sha256"]
            and descriptor["bytes"] == original_parent["bytes"]
            and descriptor["actor_hash"] == original_parent["actor_hash"]
            and descriptor["incoming_optimizer_sha256"] == original_parent["optimizer_sha256"]
            and descriptor["original_checkpoint_record"] == original_parent,
            "C1 parent/source checkpoint ownership differs",
        )
        inputs.read("prepared/" + descriptor["file"], binding=descriptor)
    original = {row["tag"]: row for row in sources["source_episodes_in_fixed_training_order"]}
    episodes = {}
    for tag, descriptor in catalog["episodes"].items():
        data = inputs.npz("prepared/" + descriptor["file"], descriptor)
        raw = original[tag]["exact_episode_complete_event"]["raw"]["arrays"]
        for key, shape in {
            "observations": (1001, 17),
            "states": (1001, 19),
            "warmstarts": (1001, 9),
            "integrations": (1001, 91),
            "actions": (1000, 6),
            "costs": (1000,),
            "velocities": (1000,),
        }.items():
            actual = _array(data[key], shape, "<f8", tag + "/" + key)
            bound = raw[key]
            hashed = actual[:-1] if key == "observations" else actual
            _need(
                hashlib.sha256(hashed.tobytes()).hexdigest() == bound["sha256"],
                "C1 prepared/source C-byte binding: " + tag + "/" + key,
            )
        terminal = descriptor["terminal_binding"]["last_live_result"]["arrays"]["observation"]
        _need(
            hashlib.sha256(data["observations"][-1].tobytes()).hexdigest() == terminal["sha256"],
            "C1 terminal observation binding differs",
        )
        before = descriptor["terminal_binding"]["last_live_inputs"]["arrays"]
        for key, original_key in (("integration_spec", "spec"), ("dimensions", "dimensions")):
            _need(
                hashlib.sha256(data[key].tobytes()).hexdigest() == before[original_key]["sha256"],
                "C1 integration semantic binding differs",
            )
        _need(
            int(data["integration_spec"]) == catalog["integration_spec"]
            and np.array_equal(data["dimensions"], [9, 9, 6])
            and np.all(data["fall_costs"] == 0),
            "C1 source native context semantics differ",
        )
        pred_binding = original[tag]["exact_teacher_agreement_complete_event"]["predictions"]
        pred_desc = pred_binding["reference"]["arrays"][pred_binding["key"]]
        prediction = _array(data["policy_actions"], (1000, 6), "<f4", "saved source predictions")
        _need(
            hashlib.sha256(prediction.tobytes()).hexdigest() == pred_desc["sha256"],
            "C1 reused prediction C bytes differ",
        )
        error = np.sum(
            (prediction.astype(np.float64) - data["actions"]) ** 2, axis=1, dtype=np.float64
        )
        _need(
            _bytes_equal(error, data["squared_disagreement"]), "C1 source sum-six agreement differs"
        )
        _need(
            original[tag]["exact_teacher_agreement_complete_event"]["actor_hash"]
            == catalog["parents"][str(descriptor["stream"])]["actor_hash"],
            "C1 source prediction actor differs",
        )
        episodes[tag] = data
    corpora = {}
    for stream in range(3):
        descriptor = catalog["streams"][str(stream)]
        data = inputs.npz("prepared/" + descriptor["file"], descriptor)
        tags = descriptor["episode_tags"]
        _need(
            len(tags) == 4 and [catalog["episodes"][tag]["case"] for tag in tags] == [0, 1, 2, 3],
            "C1 source corpus order differs",
        )
        for key in ("observations", "actions", "policy_actions", "squared_disagreement"):
            values = [
                episodes[tag][key][:-1] if key == "observations" else episodes[tag][key]
                for tag in tags
            ]
            expected = np.concatenate(values)
            if key in {"observations", "actions"}:
                expected = expected.astype(np.float32)
            _need(
                _bytes_equal(data[key], expected),
                "C1 stream corpus/source join: " + key,
            )
        corpora[stream] = data
    normalization = inputs.npz("prepared/inputs/normalization.npz")
    historical = {
        name: inputs.npz(f"prepared/inputs/{name}.npz") for name in ("train", "validation")
    }
    task = inputs.json("prepared/inputs/task.json")
    _need(
        task["task"] == config["task"], "C1 task configuration differs from inherited native task"
    )
    for descriptor in sources["normalization_and_historical_inputs"]["config_inputs"].values():
        actual = inputs.files.get("prepared/" + descriptor["file"])
        _need(
            actual is not None
            and actual["sha256"] == descriptor["sha256"]
            and actual["bytes"] == descriptor["bytes"],
            "C1 inherited historical input binding differs",
        )
    train = historical["train"]["observations"]
    _need(
        _bytes_equal(
            normalization["mean"], train.mean(axis=0, dtype=np.float64).astype(np.float32)
        ),
        "C1 W1 mean rule differs",
    )
    _need(
        _bytes_equal(
            normalization["std"],
            np.maximum(train.std(axis=0, dtype=np.float64), 0.05).astype(np.float32),
        ),
        "C1 W1 std rule differs",
    )
    return (
        catalog,
        episodes,
        corpora,
        dict(
            sources=sources,
            normalization=normalization,
            historical=historical,
            inherited_metadata=inherited_metadata,
        ),
    )


def _check_saved_summary(
    row: Mapping[str, Any], summary: Mapping[str, Any], reference: dict
) -> None:
    _need(
        row["trace"] == reference and row["tag"] == summary["alias"],
        "C1 result/trace ownership differs",
    )
    for key in (
        "controls",
        "done",
        "raw_cost",
        "native_cost",
        "native_terminal",
        "absorbing_cost",
        "cost",
    ):
        _need(row.get(key) == summary[key], "C1 saved outcome reduction differs: " + key)


def _independent_gates(panels: dict) -> dict:
    results = {}
    for name in ("primary", "confirmation"):
        qualified, regressions, uniform_qualified, uniform_regressions = [], [], [], []
        for stream in range(3):
            actors = panels[name][str(stream)]
            _need(
                set(actors) == {"parent", "uniform", "cost_effect", "shuffled"},
                "C1 outcome actor roster",
            )
            _need(all(v["episodes"] == 16 for v in actors.values()), "C1 incomplete paired panel")
            if all(
                actors["cost_effect"]["successes"] >= actors[key]["successes"] + 2
                for key in ("parent", "uniform", "shuffled")
            ):
                qualified.append(str(stream))
            if actors["uniform"]["successes"] >= actors["parent"]["successes"] + 2:
                uniform_qualified.append(str(stream))
            for key in ("parent", "uniform", "shuffled"):
                for outcome in ("successes", "survived"):
                    if actors["cost_effect"][outcome] < actors[key][outcome]:
                        regressions.append(
                            dict(stream=str(stream), comparator=key, outcome=outcome)
                        )
            for outcome in ("successes", "survived"):
                if actors["uniform"][outcome] < actors["parent"][outcome]:
                    uniform_regressions.append(dict(stream=str(stream), outcome=outcome))
        results[name] = dict(
            passed=len(qualified) >= 2 and not regressions,
            qualifying_streams=qualified,
            regressions=regressions,
        )
        results["uniform_parent_" + name] = dict(
            passed=len(uniform_qualified) >= 2 and not uniform_regressions,
            qualifying_streams=uniform_qualified,
            regressions=uniform_regressions,
        )
    shared = sorted(
        set(results["primary"]["qualifying_streams"])
        & set(results["confirmation"]["qualifying_streams"])
    )
    uniform_shared = sorted(
        set(results["uniform_parent_primary"]["qualifying_streams"])
        & set(results["uniform_parent_confirmation"]["qualifying_streams"])
    )
    results["confirmed"] = (
        results["primary"]["passed"] and results["confirmation"]["passed"] and len(shared) >= 2
    )
    results["uniform_parent_confirmed"] = (
        results["uniform_parent_primary"]["passed"]
        and results["uniform_parent_confirmation"]["passed"]
        and len(uniform_shared) >= 2
    )
    results["confirmed_streams"] = shared
    results["uniform_parent_confirmed_streams"] = uniform_shared
    results["individual_competence"] = {
        str(stream): {
            arm: all(
                panels[name][str(stream)][arm]["successes"] >= 15
                for name in ("primary", "confirmation")
            )
            for arm in ("parent", "uniform", "cost_effect", "shuffled")
        }
        for stream in range(3)
    }
    return results


class _Work:
    """Independent journal arithmetic; known returns and missing payload differ."""

    def __init__(self):
        self.requested: Counter[str] = Counter()
        self.completed: Counter[str] = Counter()
        self.unknown: Counter[str] = Counter()
        self.unpersisted: Counter[str] = Counter()
        self.pending: dict[int, tuple[str, int, str]] = {}
        self.native_pending: tuple[str, int] | None = None
        self.operations = 0
        self.environment_pending = False
        self.index_events: dict[str, list[dict]] = {}
        self.expected_events: dict[str, list[dict]] = {}
        self.shuffle_events: dict[int, list[dict]] = {}
        self.fit_measurements: dict[str, Counter[str]] = {}

    def event(self, row: dict) -> None:
        event = row["event"]
        if event == "work_requested":
            operation = _integer(row["operation"], "operation", 1)
            _need(
                operation == self.operations + 1 and operation not in self.pending,
                "C1 work operation order differs",
            )
            self.operations = operation
            category, amount = row["category"], _integer(row["amount"], "work amount")
            self.pending[operation] = category, amount, row.get("purpose", "")
            self.requested[category] += amount
            if category == "neural_rows":
                self.requested["prediction_calls"] += 1
        elif event in {"work_complete", "work_error"}:
            operation = row["operation"]
            _need(operation in self.pending, "C1 work completion has no request")
            category, amount, purpose = self.pending.pop(operation)
            _need(
                category == row["category"] and amount == row["amount"],
                "C1 measured work ownership differs",
            )
            counts = (
                self.completed
                if event == "work_complete" or row.get("returned") is True
                else self.unknown
            )
            counts[category] += amount
            if category == "neural_rows":
                counts["prediction_calls"] += 1
            if event == "work_error" and row.get("returned") is True:
                self.unpersisted[category] += amount
            match = re.fullmatch(
                r"(s[0-2]_(?:uniform|cost_effect|shuffled))/(forward|backward|adam)", purpose
            )
            if match is not None and counts is self.completed:
                expected = {
                    "forward": "neural_rows",
                    "backward": "backward_rows",
                    "adam": "optimizer_steps",
                }[match[2]]
                _need(category == expected, "C1 fit measured category/purpose differs")
                self.fit_measurements.setdefault(match[1], Counter())[category] += amount
        elif event == "trace_intent":
            _need(self.native_pending is None, "C1 overlapping native work")
            self.native_pending = row["alias"], row["ordinal"]
            self.requested["native_controls"] += 1
            self.requested["physics_substeps"] += 4
        elif event in {"trace_returned", "native_error"}:
            _need(
                self.native_pending == (row["alias"], row["ordinal"]),
                "C1 native return lacks request",
            )
            if event == "trace_returned" or row.get("returned") is True:
                self.completed["native_controls"] += 1
                self.completed["physics_substeps"] += 4
                if event == "native_error":
                    self.unpersisted["native_controls"] += 1
                    self.unpersisted["physics_substeps"] += 4
            else:
                self.unknown["native_controls"] += 1
                self.unknown["physics_substeps"] += 4
            self.native_pending = None
        elif event == "trace_complete" and self.native_pending is not None:
            # A graceful exception can retain intent without a native_error
            # record; it remains uncertain rather than silently free.
            self.unknown["native_controls"] += 1
            self.unknown["physics_substeps"] += 4
            self.native_pending = None
        elif event in {"index_before", "index_after"}:
            self.index_events.setdefault(row["tag"], []).append(row)
            self.requested["actual_index_draws"] += 256 if event == "index_before" else 0
            self.completed["actual_index_draws"] += 256 if event == "index_after" else 0
        elif event in {"expected_index_before", "expected_index_after"}:
            self.expected_events.setdefault(row["tag"], []).append(row)
            self.requested["expected_index_draws"] += 256 if event == "expected_index_before" else 0
            self.completed["expected_index_draws"] += 256 if event == "expected_index_after" else 0
        elif event in {"shuffle_requested", "shuffle_complete"}:
            self.shuffle_events.setdefault(row["stream"], []).append(row)
            counts = self.requested if event == "shuffle_requested" else self.completed
            counts["shuffle_calls"] += 1
            counts["shuffle_entries"] += row["rows"]
        elif event == "environment_requested":
            _need(
                not self.environment_pending and self.requested["environment_constructions"] == 0,
                "C1 duplicate environment construction",
            )
            self.environment_pending = True
            self.requested["environment_constructions"] += 1
        elif event == "environment_complete":
            _need(
                self.environment_pending
                and row["mjdata_objects"] == 17
                and row["integration_size"] == 91,
                "C1 native environment interface differs",
            )
            self.environment_pending = False
            self.completed["environment_constructions"] += 1
        elif event in {
            "fit_construction_requested",
            "fit_construction_complete",
            "endpoint_save_requested",
            "endpoint_save_complete",
        }:
            category = (
                "policy_constructions" if event.startswith("fit_construction") else "endpoint_saves"
            )
            counts = self.requested if event.endswith("requested") else self.completed
            counts[category] += 1

    def closed(self) -> None:
        _need(
            not self.pending and self.native_pending is None and not self.environment_pending,
            "C1 unresolved work intents remain",
        )


class _Scientific:
    def __init__(self, run: Path, out: Path, inputs: _Inputs, config: dict, result: dict):
        self.run, self.out, self.inputs, self.config, self.result = run, out, inputs, config, result
        self.catalog, self.episodes, self.corpora, self.inherited = _source_bindings(inputs, config)
        if self.inherited["inherited_metadata"]:
            inherited_result = self.inherited["inherited_metadata"]["worker_result.json"]
            original = self.inherited["sources"]
            expected = dict(
                H5_worker=inherited_result.get("work", {}),
                H5_worker_wall_seconds=inherited_result.get("wall_seconds"),
                H5_independent_audit=original["full_audit"]["inherited_audit_work"],
                scope="Previously paid H5 work, including parent preparation and all acquisition arms; not charged again to C1 worker limits",
                source_worker_result_sha256=original["metadata_files"]["run/worker_result.json"][
                    "sha256"
                ],
            )
            _need(
                result.get("inherited_work") == expected,
                "C1 inherited-work accounting differs from sealed H5 metadata",
            )
        self.baselines: dict[str, dict] = {}
        self.baseline_costs: dict[str, np.ndarray] = {}
        self.baseline_maes: dict[str, float] = {}
        self.deltas = {stream: [] for stream in range(3)}
        self.assay_rows: list[dict] = []
        self.evaluation_rows: list[dict] = []
        self.weights: dict[int, dict] = {}
        self.replay_violations: list[str] = []
        self.replay_mismatch_confirmed = False
        self.closed_prerequisite_negative = False
        self.work = _Work()
        self.verifier: SavedActorVerifier | None = None
        self.registry: dict[str, dict] = {}
        self.prediction_journal: AuditJournal | None = None
        self.trace_count = 0
        self.fit_checks: dict | None = None
        self.weights_gate: bool | None = None
        self.fresh_initial: dict[tuple[str, int], tuple[bytes, ...]] = {}
        self.baseline_gate = (
            inputs.json("worker/replay_gate.json")
            if (run / "worker/replay_gate.json").is_file()
            else None
        )
        self.assays = self._read_jsonl("worker/assays.jsonl")
        self.evaluations = self._read_jsonl("worker/evaluations.jsonl")
        self.assay_position = self.evaluation_position = 0

    def _read_jsonl(self, relative: str) -> list[dict]:
        if not (self.run / relative).is_file():
            return []
        payload = self.inputs.read(relative, 128 * 1024**2)
        return [_json(line) for line in payload.splitlines()]

    def _actors(self) -> None:
        if self.verifier is not None:
            return
        import torch

        _need(
            np.__version__ == self.config["runtime"]["distributions"]["numpy"]
            and torch.__version__ == self.config["runtime"]["distributions"]["torch"],
            "C1 audit prediction runtime differs from frozen CPU runtime",
        )
        self.registry = self.inputs.json("worker/actor_registry.json")
        expected = {
            f"s{s}_{arm}"
            for s in range(3)
            for arm in ("parent", "uniform", "cost_effect", "shuffled")
        }
        _need(set(self.registry) == expected, "C1 checkpoint actor roster differs")
        actual_registry = {}
        records = []
        for alias, binding in self.registry.items():
            _need(
                alias == f"s{binding['stream']}_{binding['arm']}"
                and binding["stream"] in range(3)
                and binding["arm"] in {"parent", "uniform", "cost_effect", "shuffled"},
                "C1 registry stream/arm alias differs",
            )
            path = self.inputs.path(binding["file"])
            actual = dict(binding, bytes=path.stat().st_size)
            _need(actual["bytes"] <= 16 * 1024**2, "C1 checkpoint size exceeds audit allowance")
            actual_registry.setdefault(binding["actor_hash"], actual)
            records.append(actual)
        _need(
            len({row["file"] for row in records}) == 12,
            "C1 checkpoint endpoint files are not distinct",
        )
        self.prediction_journal = AuditJournal(
            self.out / "predictions.jsonl", max_bytes=442460160, max_line_bytes=512
        )
        self.verifier = SavedActorVerifier(
            actual_registry,
            self.run,
            max_neural_rows=522000,
            max_prediction_calls=432090,
            deadline=self.inputs.deadline,
            event_sink=self.prediction_journal.record,
        )
        archive_reads = 0 if self.inputs.archive is None else self.inputs.archive.read_bytes
        _need(
            self.inputs.read_bytes + archive_reads + sum(row["bytes"] for row in records)
            <= MAX_READ_BYTES,
            "C1 checkpoint-read reservation exhausted",
        )
        self.verifier.load_all(records)
        self.inputs.checkpoint_read_bytes = self.verifier.checkpoint_read_bytes
        if self.inputs.archive is not None:
            self.inputs.archive.max_read_bytes = (
                MAX_READ_BYTES - self.inputs.read_bytes - self.inputs.checkpoint_read_bytes
            )
        _need(self.verifier.checkpoint_loads == 12, "C1 audit checkpoint load census differs")
        expected_payload = {
            "schema",
            "seed",
            "obs_dim",
            "policy_hash",
            "state_dict",
            "private_rng_state",
            "training",
            "optimizer_state",
            "metadata",
        }
        for payload in self.verifier.record_payloads.values():
            _need(
                set(payload) == expected_payload
                and payload["obs_dim"] == 17
                and type(payload["training"]) is bool
                and type(payload["seed"]) is int,
                "C1 checkpoint keys/interface changed",
            )
        _need(self.check_weights(), "C1 actor execution after degenerate contrast")
        self.fit_checks = self.check_fits()

    def trace(self, summary: dict, arrays: Mapping[str, np.ndarray], reference: dict) -> None:
        self.trace_count += 1
        metadata = summary["metadata"]
        kind, source_tag = metadata.get("kind"), metadata.get("source")
        if kind == "baseline":
            _need(
                not any(self.deltas.values())
                and not self.evaluation_rows
                and source_tag in self.episodes
                and source_tag not in self.baselines,
                "C1 baseline ordering/ownership differs",
            )
            source = self.episodes[source_tag]
            self.baselines[source_tag] = summary
            self.baseline_costs[source_tag] = arrays["native_costs"].copy()
            self.baseline_maes[source_tag] = float(np.mean(np.abs(arrays["x_velocities"] - 1.5)))
            count = summary["controls"]
            prefix = {
                key: value[: count + 1]
                if key in BOUNDARY_KEYS
                else value[:count]
                if key in {"actions", "costs", "velocities", "fall_costs"}
                else value
                for key, value in source.items()
            }
            try:
                compare_baseline_arrays(arrays, prefix)
            except AuditError as error:
                self.replay_mismatch_confirmed = True
                self.replay_violations.append(str(error))
            try:
                _need(
                    summary["normally_closed"] and count == 1000 and not summary["done"],
                    "C1 baseline did not complete",
                )
                _need(
                    float(np.mean(np.abs(arrays["x_velocities"] - 1.5))) <= 0.3,
                    "C1 replay teacher did not walk",
                )
            except AuditError as error:
                self.replay_violations.append(str(error))
            if self.baseline_gate is not None:
                rows = [
                    row for row in self.baseline_gate["episodes"] if row["source_tag"] == source_tag
                ]
                _need(len(rows) == 1, "C1 replay gate lost a source")
                _check_saved_summary(rows[0], summary, reference)
                _need(
                    _bytes_equal(
                        np.asarray(rows[0]["native_costs"], dtype=np.float64),
                        arrays["native_costs"],
                    ),
                    "C1 baseline cost tape differs",
                )
        elif kind == "assay":
            _need(
                len(self.baselines) == 12
                and not self.replay_violations
                and self.baseline_gate is not None
                and self.baseline_gate["passed"] is True,
                "C1 assay before complete replay gate",
            )
            if not summary["normally_closed"]:
                _need(
                    self.result["status"] != "COMPLETED_FIXED_C1_ROSTER",
                    "C1 complete roster has failed assay",
                )
                return
            _need(self.assay_position < len(self.assays), "C1 assay missing saved scalar row")
            row = self.assays[self.assay_position]
            self.assay_position += 1
            _check_saved_summary(row, summary, reference)
            _need(
                source_tag == row["source_tag"] and metadata["t"] == row["t"],
                "C1 assay anchor binding differs",
            )
            checked = verify_assay_arrays(
                arrays,
                self.episodes[source_tag],
                row["t"],
                baseline_costs=self.baseline_costs[source_tag],
            )
            _need(
                row["baseline_cost"] == checked["baseline_cost"]
                and row["delta"] == checked["delta"]
                and row["h"] == checked["h"],
                "C1 assay delta differs",
            )
            descriptor = self.catalog["episodes"][source_tag]
            _need(
                row["stream"] == descriptor["stream"] and row["case"] == descriptor["case"],
                "C1 assay stream/source differs",
            )
            stream = row["stream"]
            expected_row = len(self.deltas[stream])
            _need(
                (row["case"], row["t"]) == divmod(expected_row, 1000),
                "C1 assay rows reordered/repeated",
            )
            self.deltas[stream].append(row["delta"])
            self.assay_rows.append(row)
        elif kind == "evaluation":
            _need(
                all(len(v) == 4000 for v in self.deltas.values()),
                "C1 standalone before complete assay",
            )
            self._actors()
            _need(
                self.evaluation_position < len(self.evaluations), "C1 evaluation scalar row missing"
            )
            row = self.evaluations[self.evaluation_position]
            self.evaluation_position += 1
            _check_saved_summary(row, summary, reference)
            alias = metadata["actor"]
            _need(
                metadata.get("prediction_actor") == alias,
                "C1 deployed prediction persistence owner differs",
            )
            _need(
                alias == row["actor"]
                and row["stream"] == self.registry[alias]["stream"]
                and row["arm"] == self.registry[alias]["arm"],
                "C1 evaluation actor ownership differs",
            )
            _need(
                summary["horizon"] == 1000 and summary["normally_closed"],
                "C1 incomplete standalone episode",
            )
            if row["panel"] == "source":
                tags = self.catalog["streams"][str(row["stream"])]["episode_tags"]
                _need(
                    source_tag == row["source_tag"] == tags[row["case"]],
                    "C1 source-context evaluation pairing differs",
                )
                source = self.episodes[source_tag]
                _need(
                    row["seed"] == self.catalog["episodes"][source_tag]["seed"],
                    "C1 source reset seed differs",
                )
                for key in BOUNDARY_KEYS:
                    _need(
                        np.allclose(arrays[key][0], source[key][0], rtol=0, atol=1e-10),
                        "C1 source reset context differs",
                    )
            else:
                _need(
                    row["panel"] in {"primary", "confirmation"}
                    and row["seed"] == self.config["seeds"][row["panel"]][row["case"]]
                    and source_tag is None,
                    "C1 fresh paired reset differs",
                )
                key = row["panel"], row["seed"]
                initial = tuple(arrays[name][0].tobytes() for name in BOUNDARY_KEYS)
                _need(
                    self.fresh_initial.setdefault(key, initial) == initial,
                    "C1 fresh reset contexts are not exactly paired",
                )
            owner = self.registry[alias]["actor_hash"]
            for index in range(summary["controls"]):
                prediction = self.verifier.predict(owner, arrays["observations"][index])
                _need(
                    _bytes_equal(prediction.astype("<f8"), arrays["actions"][index]),
                    "C1 deployed single-row action differs",
                )
            mae = float(np.mean(np.abs(arrays["x_velocities"] - 1.5), dtype=np.float64))
            survived = summary["controls"] == 1000 and not summary["done"]
            _need(
                row["mae"] == mae
                and row["survived"] == survived
                and row["walking"] == (survived and mae <= 0.3),
                "C1 walking/survival reduction differs",
            )
            self.evaluation_rows.append(row)
        else:
            raise AuditError("Unknown C1 trace role")

    def event(self, row: dict) -> None:
        self.work.event(row)
        if row["event"] == "main_stage_admitted":
            _need(
                len(self.baselines) == 12 and not self.replay_violations,
                "C1 main stage admitted before replay eligibility",
            )
            _need(
                row["replay_gate_sha256"] == self.inputs.files["worker/replay_gate.json"]["sha256"],
                "C1 admitted replay gate SHA differs",
            )

    def check_weights(self) -> bool:
        if self.weights_gate is not None:
            return self.weights_gate
        gate = self.inputs.json("worker/contrast_gate.json")
        reasons = []
        for stream in range(3):
            _need(len(self.deltas[stream]) == 4000, "C1 weighting before all assays")
            record = gate["weights"][str(stream)]
            saved = self.inputs.npz(record["file"], record)
            delta = np.asarray(self.deltas[stream], dtype=np.float64)
            normalized, actual = recompute_weights(delta)
            raw = 1 + np.minimum(9.0, np.maximum(0.0, delta) / 20)
            shuffled, indices, details = recompute_shuffled_weights(
                actual,
                self.corpora[stream]["squared_disagreement"],
                self.config["seeds"]["shuffle"][stream],
                details=True,
            )
            for key, expected in dict(
                deltas=delta,
                raw=raw,
                normalized=normalized,
                cost_effect=actual,
                shuffled=shuffled,
                source_indices=indices,
            ).items():
                _need(
                    _bytes_equal(saved[key], expected),
                    "C1 weight/null numerical bytes differ: " + key,
                )
            _need(
                record["mean_raw"] == float(np.mean(raw, dtype=np.float64))
                and record["permutation_calls"] == 208
                and record["returned_entries"] == 4000,
                "C1 weight reduction/census differs",
            )
            groups = self.inputs.json(
                record["groups_file"], binding={"sha256": record["groups_sha256"]}
            )
            _need(
                groups == details["groups"]
                and record["rng_initial"] == details["rng_initial"]
                and record["rng_terminal"] == details["rng_terminal"],
                "C1 shuffled group/RNG transcript differs",
            )
            events = self.work.shuffle_events.get(stream, [])
            _need(len(events) == 416, "C1 shuffle paid journal incomplete")
            for group, before, after in zip(groups, events[::2], events[1::2], strict=True):
                _need(
                    before["event"] == "shuffle_requested"
                    and after["event"] == "shuffle_complete"
                    and before["group"] == after["group"] == group["group"]
                    and before["rows"] == after["rows"] == group["rows"],
                    "C1 shuffle operation join differs",
                )
                _need(
                    before["rng"] == optimizer_fingerprint(group["rng_before"])
                    and after["rng"] == optimizer_fingerprint(group["rng_after"]),
                    "C1 shuffled PCG fingerprints differ",
                )
            if np.all(actual == actual[0]):
                reasons.append(dict(stream=str(stream), reason="constant_cost_effect_weights"))
            if np.array_equal(actual, shuffled):
                reasons.append(dict(stream=str(stream), reason="shuffle_identical_to_cost_effect"))
            self.weights[stream] = dict(cost_effect=actual, shuffled=shuffled)
        expected_gate = dict(passed=not reasons, reasons=reasons, redraw_allowed=False)
        _need(
            {key: gate[key] for key in expected_gate} == expected_gate
            and self.result["contrast_gate"] == expected_gate
            and self.result["weights"] == gate["weights"],
            "C1 contrast gate changed",
        )
        self.weights_gate = not reasons
        return self.weights_gate

    def check_fits(self) -> dict:
        self._actors()
        records = self.inputs.json("worker/fit_records.json")
        _need(
            set(records)
            == {f"s{s}_{arm}" for s in range(3) for arm in ("uniform", "cost_effect", "shuffled")},
            "C1 fit roster differs",
        )
        _need(records == self.result["fit_records"], "C1 fit result/sidecar binding differs")
        import torch

        torch.set_num_threads(1)
        curve_rows = loss_checks = 0
        config = dict(
            batch_size=256,
            lr=0.0003,
            eta_theta=0.0,
            gradient_clip=0.5,
            prox_reduction="mean",
            anchor_refresh=0,
        )
        for stream in range(3):
            parent_alias = f"s{stream}_parent"
            parent_record = self.registry[parent_alias]
            parent = self.verifier.record_payloads[parent_record["file"]]
            incoming = parent["optimizer_state"]
            descriptor = self.catalog["parents"][str(stream)]
            _need(
                parent_record["actor_hash"] == descriptor["actor_hash"]
                and parent_record["sha256"] == descriptor["sha256"]
                and optimizer_fingerprint(incoming) == descriptor["incoming_optimizer_sha256"],
                "C1 incoming checkpoint continuation differs",
            )
            _need(
                incoming["updates"] == incoming["fit_updates"] == 2000
                and incoming["fit_config"] == config,
                "C1 incoming block settings differ",
            )
            for key, target in (("observation_mean", "mean"), ("observation_std", "std")):
                _need(
                    _bytes_equal(
                        parent["state_dict"][key].numpy(), self.inherited["normalization"][target]
                    ),
                    "C1 incoming normalization differs",
                )
            tape = self.inputs.npy(f"worker/learning/s{stream}_expected_indices.npy")
            _array(tape, (2000, 256), "<i8", "expected tape")
            generator = np.random.Generator(np.random.PCG64(0))
            generator.bit_generator.state = deepcopy(incoming["rng"])
            expected_events = self.work.expected_events.get(f"s{stream}", [])
            _need(len(expected_events) == 4000, "C1 expected tape journal incomplete")
            rng_states = []
            for index in range(2000):
                before = optimizer_fingerprint(generator.bit_generator.state)
                expected = generator.integers(0, 4000, size=256)
                after = optimizer_fingerprint(generator.bit_generator.state)
                _need(_bytes_equal(tape[index], expected), "C1 expected tape restarted/differed")
                first, last = expected_events[2 * index : 2 * index + 2]
                _need(
                    first["event"] == "expected_index_before"
                    and last["event"] == "expected_index_after"
                    and first["fit_update"] == last["fit_update"] == index + 1
                    and first["rng"] == before
                    and last["rng"] == after
                    and last["indices_sha256"] == hashlib.sha256(expected.tobytes()).hexdigest(),
                    "C1 expected RNG/tape join differs",
                )
                rng_states.append((before, after))
            terminal_rng = deepcopy(generator.bit_generator.state)
            data = self.corpora[stream]
            fit_data = dict(
                observations_sha256=_fit_array_hash(data["observations"]),
                actions_sha256=_fit_array_hash(data["actions"]),
                rows=4000,
            )
            global_rng = self.inputs.npy(
                "prepared/" + descriptor["global_torch_rng"]["file"], descriptor["global_torch_rng"]
            )
            for arm in ("uniform", "cost_effect", "shuffled"):
                alias = f"s{stream}_{arm}"
                record = records[alias]
                directory = "worker/learning/" + alias + "/"
                saved = self.inputs.json(directory + "fit.json")
                _need(record == saved and record["tag"] == alias, "C1 fit record ownership differs")
                endpoint = self.registry[alias]
                payload = self.verifier.record_payloads[endpoint["file"]]
                state = payload["optimizer_state"]
                _need(
                    endpoint["file"] == directory + "endpoint.pt"
                    and record["checkpoint"]["sha256"] == endpoint["sha256"]
                    and record["actor_hash"] == endpoint["actor_hash"]
                    and optimizer_fingerprint(state) == record["optimizer_sha256"],
                    "C1 fitted endpoint full ownership differs",
                )
                _need(
                    payload["seed"] == parent["seed"]
                    and payload["metadata"]["tag"] == alias
                    and state["policy_hash"] == endpoint["actor_hash"],
                    "C1 checkpoint seed/alias differs",
                )
                _need(
                    state["parent_policy_hash"]
                    == state["anchor_policy_hash"]
                    == parent_record["actor_hash"]
                    and optimizer_fingerprint(state["anchor"])
                    == optimizer_fingerprint(parent["state_dict"]),
                    "C1 new block anchor differs",
                )
                _need(
                    state["updates"] == 4000
                    and state["fit_updates"] == 2000
                    and state["fit_config"] == config
                    and state["fit_data"] == fit_data
                    and state["rng"] == terminal_rng,
                    "C1 Adam/data/PCG continuation differs",
                )
                _need(
                    optimizer_fingerprint(payload["private_rng_state"])
                    == optimizer_fingerprint(parent["private_rng_state"]),
                    "C1 private Torch continuation differs",
                )
                for key in ("observation_mean", "observation_std"):
                    _need(
                        torch.equal(payload["state_dict"][key], parent["state_dict"][key]),
                        "C1 normalizer changed",
                    )
                adam = state["adam"]
                names = [
                    f"network.{layer}.{kind}" for layer in (0, 2, 4) for kind in ("weight", "bias")
                ]
                _need(
                    state["schema"] == incoming["schema"] == "walker-adam-proximal-fit-v1"
                    and state["parameter_names"] == incoming["parameter_names"] == names,
                    "C1 optimizer schema/parameter ownership differs",
                )
                _need(
                    set(adam["state"]) == set(range(6)) and len(adam["param_groups"]) == 1,
                    "C1 Adam moments/parameter groups differ",
                )
                group = adam["param_groups"][0]
                _need(group["params"] == list(range(6)), "C1 Adam parameter order differs")
                for key, expected in self.config["learning"]["adam"].items():
                    actual = group.get(key)
                    _need(
                        (list(actual) if isinstance(actual, tuple) else actual) == expected,
                        "C1 Adam setting differs: " + key,
                    )
                for parameter, moments in adam["state"].items():
                    _need(
                        set(moments) == {"step", "exp_avg", "exp_avg_sq"}
                        and moments["step"].dtype == torch.float32
                        and moments["step"].shape == ()
                        and moments["exp_avg"].dtype == moments["exp_avg_sq"].dtype == torch.float32
                        and moments["exp_avg"].shape
                        == moments["exp_avg_sq"].shape
                        == payload["state_dict"][names[parameter]].shape
                        and float(moments["step"]) == 4000
                        and torch.isfinite(moments["exp_avg"]).all()
                        and torch.isfinite(moments["exp_avg_sq"]).all()
                        and torch.all(moments["exp_avg_sq"] >= 0),
                        "C1 Adam moment/step continuation differs",
                    )
                initial = self.inputs.json(
                    directory + record["initial"]["file"], binding=record["initial"]
                )
                weight = None if arm == "uniform" else self.weights[stream][arm]
                _need(
                    initial["actor_hash"] == parent_record["actor_hash"]
                    and initial["optimizer_sha256"] == optimizer_fingerprint(incoming)
                    and initial["adam_sha256"] == optimizer_fingerprint(incoming["adam"])
                    and initial["rng"] == incoming["rng"]
                    and initial["pcg_sha256"] == optimizer_fingerprint(incoming["rng"])
                    and initial["private_torch_sha256"]
                    == optimizer_fingerprint(parent["private_rng_state"])
                    and initial["global_torch_sha256"] == optimizer_fingerprint(global_rng)
                    and initial["anchor_policy_hash"] == parent_record["actor_hash"]
                    and initial["fit_config"] == config,
                    "C1 actual incoming fit state differs",
                )
                _need(
                    initial["expected_tape_sha256"]
                    == record["expected_tape_sha256"]
                    == optimizer_fingerprint(tape)
                    and initial["weight_sha256"]
                    == record["weight_sha256"]
                    == (None if weight is None else optimizer_fingerprint(weight)),
                    "C1 fit weight/index owner differs",
                )
                actual_tape = self.inputs.npy(
                    directory + record["actual_indices"]["file"], record["actual_indices"]
                )
                _need(_bytes_equal(actual_tape, tape), "C1 arm minibatches differ")
                index_events = self.work.index_events.get(alias, [])
                _need(len(index_events) == 4000, "C1 actual tape journal incomplete")
                for index, (before, after) in enumerate(rng_states):
                    first, last = index_events[2 * index : 2 * index + 2]
                    _need(
                        first["event"] == "index_before"
                        and last["event"] == "index_after"
                        and first["fit_update"] == last["fit_update"] == index + 1
                        and first["rng"] == before
                        and last["rng"] == after
                        and last["indices_sha256"]
                        == hashlib.sha256(tape[index].tobytes()).hexdigest(),
                        "C1 actual draw transcript differs",
                    )
                curve = self.inputs.npy(directory + record["curve"]["file"], record["curve"])
                predictions = self.inputs.npy(
                    directory + record["predictions"]["file"], record["predictions"]
                )
                valid = self.inputs.npy(
                    directory + record["prediction_valid"]["file"], record["prediction_valid"]
                )
                _array(predictions, (2000, 256, 6), "<f4", "training predictions")
                _need(
                    valid.shape == (2000,)
                    and valid.dtype == np.uint8
                    and np.all(valid == 1)
                    and curve.shape == (2000,)
                    and np.all(curve["valid"] == 1),
                    "C1 training prediction/curve prefix incomplete",
                )
                evidence = record["fitting_evidence"]
                _need(
                    dict(self.work.fit_measurements.get(alias, {}))
                    == dict(neural_rows=512000, backward_rows=512000, optimizer_steps=2000),
                    "C1 per-arm paid fitting work differs",
                )
                _need(
                    evidence["fit_config"] == config
                    and evidence["fit_data"] == fit_data
                    and evidence["child_policy_hash"] == endpoint["actor_hash"]
                    and evidence["resume_fit"] is False
                    and evidence["optimizer_continued"] is True
                    and evidence["anchor_refreshes"] == []
                    and evidence["completed_updates"]
                    == record["completed_updates"]
                    == record["sampled_updates"]
                    == 2000
                    and evidence["failed"] is False
                    and len(evidence["learning_curve"]) == 2000,
                    "C1 fit completion/objective differs",
                )
                for index in range(2000):
                    saved_row = evidence["learning_curve"][index]
                    _need(
                        curve[index]["update"] == saved_row["update"] == 2001 + index
                        and curve[index]["fit_update"] == saved_row["fit_update"] == index + 1,
                        "C1 learning curve update order differs",
                    )
                    residual = torch.from_numpy(predictions[index].copy()) - torch.from_numpy(
                        np.asarray(data["actions"][tape[index]], dtype=np.float32)
                    )
                    squared = residual**2
                    mse = float(torch.mean(squared))
                    weighted = (
                        mse
                        if weight is None
                        else float(
                            torch.mean(squared * torch.from_numpy(weight[tape[index], None].copy()))
                        )
                    )
                    expected = dict(
                        action_mse=mse,
                        weighted_action_mse=weighted,
                        proximal_loss=0.0,
                        total_loss=weighted,
                    )
                    for key, value in expected.items():
                        _need(
                            float(curve[index][key]) == value
                            and saved_row.get(key, saved_row["action_mse"]) == value,
                            "C1 saved training residual/loss differs: " + key,
                        )
                    norm = float(curve[index]["unclipped_gradient_norm"])
                    _need(
                        math.isfinite(norm)
                        and norm >= 0
                        and norm == saved_row["unclipped_gradient_norm"],
                        "C1 recorded gradient norm binding differs",
                    )
                    loss_checks += 1
                curve_rows += 2000
        return dict(
            fit_blocks=9,
            complete_curve_rows=curve_rows,
            recomputed_training_loss_rows=loss_checks,
            independent_optimizer_steps=0,
            independent_backward_rows=0,
            gradient_norm_scope="Worker-recorded finite values, hash/tape/update bound; gradients were not rerun.",
            audit_expected_tape_integer_calls=6000,
            audit_expected_tape_entries=1536000,
        )

    def check_metrics(self, archive: StreamArchive) -> dict:
        metrics = self.inputs.json("worker/metrics.json")
        _need(
            set(metrics) == set(self.registry) and metrics == self.result["metrics"],
            "C1 metric roster/result differs",
        )
        reused = {
            row["exact_saved_event"]["tag"]: row
            for row in self.inherited["sources"]["reused_incoming_historical_metric_events"]
        }
        rows = calls = 0
        for alias, values in metrics.items():
            stream, arm = self.registry[alias]["stream"], self.registry[alias]["arm"]
            for label, data in [
                ("own", self.corpora[stream]),
                ("train", self.inherited["historical"]["train"]),
                ("validation", self.inherited["historical"]["validation"]),
            ]:
                saved = values["own"] if label == "own" else values["historical"][label]
                if arm == "parent" and label != "own":
                    original = reused[alias + "/" + label]
                    event = original["exact_saved_event"]
                    _need(
                        hashlib.sha256(_canonical(event)).hexdigest()
                        == original["canonical_event_sha256"]
                        and event["actor_hash"] == self.registry[alias]["actor_hash"]
                        and saved
                        == dict(action_mse=event["action_mse"], reused_audited_event=event["id"]),
                        "C1 reused historical metric binding differs",
                    )
                    continue
                if arm == "parent":
                    prediction = data["policy_actions"]
                else:
                    prediction_parts = []
                    batch_size = 1000 if label == "own" else 1024
                    _need(
                        saved["actor_hash"] == self.registry[alias]["actor_hash"]
                        and saved["rows"] == len(data["actions"])
                        and saved["batch_size"] == batch_size,
                        "C1 metric actor/batch geometry differs",
                    )
                    expected_starts = list(range(0, len(data["actions"]), batch_size))
                    _need(
                        len(saved["batches"]) == len(expected_starts), "C1 metric batches missing"
                    )
                    for start, batch in zip(expected_starts, saved["batches"], strict=True):
                        stop = min(start + batch_size, len(data["actions"]))
                        _need(
                            batch["start"] == start and batch["stop"] == stop,
                            "C1 metric batch order differs",
                        )
                        reference = batch["reference"]
                        _need(
                            set(reference["arrays"]) == {"observations", "predictions", "targets"},
                            "C1 metric saved array roster",
                        )
                        observation, target, predicted = (
                            archive.array(reference, key)
                            for key in ("observations", "targets", "predictions")
                        )
                        _need(
                            _bytes_equal(
                                observation,
                                np.asarray(data["observations"][start:stop], dtype=np.float32),
                            )
                            and _bytes_equal(
                                target, np.asarray(data["actions"][start:stop], dtype=np.float32)
                            ),
                            "C1 metric input/target binding differs",
                        )
                        actual = self.verifier.predict(
                            self.registry[alias]["actor_hash"], observation
                        )
                        _need(_bytes_equal(predicted, actual), "C1 saved metric prediction differs")
                        prediction_parts.append(predicted)
                        rows += stop - start
                        calls += 1
                    prediction = np.concatenate(prediction_parts)
                residual = np.asarray(prediction, dtype=np.float32).astype(np.float64) - np.asarray(
                    data["actions"], dtype=np.float32
                ).astype(np.float64)
                if label == "own":
                    mse = float(np.mean(residual**2, dtype=np.float64))
                else:
                    total = 0.0
                    for begin in range(0, len(residual), 1024):
                        total += float(
                            np.sum(residual[begin : begin + 1024] ** 2, dtype=np.float64)
                        )
                    mse = total / residual.size
                _need(saved["action_mse"] == mse, "C1 fitting-target reconstruction differs")
                if label == "own":
                    weighted = float(
                        np.mean(
                            residual**2
                            * self.weights[stream]["cost_effect"].astype(np.float64)[:, None],
                            dtype=np.float64,
                        )
                    )
                    _need(
                        saved["weighted_action_mse"] == weighted,
                        "C1 weighted reconstruction differs",
                    )
        _need(rows == 90000 and calls == 90, "C1 new metric row/call census differs")
        return dict(
            new_metric_rows=rows,
            new_metric_calls=calls,
            reused_source_rows=12000,
            reused_historical_rows=18000,
        )

    def check_outcomes(self) -> dict:
        _need(len(self.evaluation_rows) == 432, "C1 standalone episode census differs")
        panels = {
            panel: {str(s): {} for s in range(3)} for panel in ("primary", "confirmation", "source")
        }
        gaps = []
        for panel in panels:
            for stream in range(3):
                for arm in ("parent", "uniform", "cost_effect", "shuffled"):
                    rows = [
                        row
                        for row in self.evaluation_rows
                        if (row["panel"], row["stream"], row["arm"]) == (panel, stream, arm)
                    ]
                    count = 4 if panel == "source" else 16
                    _need(
                        len(rows) == count
                        and sorted(row["case"] for row in rows) == list(range(count)),
                        "C1 paired episode roster/repetition differs",
                    )
                    panels[panel][str(stream)][arm] = dict(
                        episodes=len(rows),
                        successes=sum(int(row["walking"]) for row in rows),
                        survived=sum(int(row["survived"]) for row in rows),
                        controls=sum(row["controls"] for row in rows),
                        mean_cost=float(np.mean([row["cost"] for row in rows])),
                        mean_steps=float(np.mean([row["controls"] for row in rows])),
                        mean_velocity_mae=float(np.mean([row["mae"] for row in rows])),
                    )
        gates = _independent_gates(panels)
        for row in self.evaluation_rows:
            if row["panel"] == "source":
                teacher = self.baselines[row["source_tag"]]
                teacher_mae = self.baseline_maes[row["source_tag"]]
                gaps.append(
                    dict(
                        actor=row["actor"],
                        source_tag=row["source_tag"],
                        teacher_cost=teacher["cost"],
                        student_cost=row["cost"],
                        cost_gap=row["cost"] - teacher["cost"],
                        teacher_walking=teacher["controls"] == 1000
                        and not teacher["done"]
                        and teacher_mae <= 0.3,
                        student_walking=row["walking"],
                        teacher_survived=teacher["controls"] == 1000 and not teacher["done"],
                        student_survived=row["survived"],
                        teacher_controls=teacher["controls"],
                        student_controls=row["controls"],
                    )
                )
        saved = self.inputs.json("worker/outcomes.json")
        _need(
            saved["panels"] == panels
            and saved["source_context_gaps"] == gaps
            and saved["gates"] == gates
            and self.result["gates"] == gates
            and self.result["student_outcomes"] == panels
            and self.result["teacher_student_gap"] == gaps,
            "C1 standalone/gap/gate reduction differs",
        )
        return dict(panels=panels, gates=gates, source_context_gaps=gaps)

    def derivatives(self) -> dict:
        """Publish checked scalar rows; absent assays/weights stay absent."""
        columns = [
            "source_tag",
            "stream",
            "case",
            "t",
            "h",
            "controls",
            "done",
            "raw_cost",
            "native_cost",
            "native_terminal",
            "absorbing_cost",
            "cost",
            "baseline_cost",
            "delta",
            "parent_native_sum6_error",
            "cost_effect_weight",
            "shuffled_weight",
        ]
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in self.assay_rows:
            result = {key: row[key] for key in columns[:14]}
            stream, index = row["stream"], 1000 * row["case"] + row["t"]
            result["parent_native_sum6_error"] = float(
                self.corpora[stream]["squared_disagreement"][index]
            )
            for key, arm in (
                ("cost_effect_weight", "cost_effect"),
                ("shuffled_weight", "shuffled"),
            ):
                result[key] = (
                    float(self.weights[stream][arm][index]) if stream in self.weights else ""
                )
            writer.writerow(result)
        payload = output.getvalue().encode("ascii")
        _need(len(payload) <= 4 * 1024**2, "C1 assay scalar derivative exceeds4MiB")
        with (self.out / "assay_scalars.csv").open("xb") as stream:
            stream.write(payload)
        return dict(
            assay_scalars=dict(
                file="assay_scalars.csv",
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
                rows=len(self.assay_rows),
                complete_assay=len(self.assay_rows) == 12000,
                weights_available_streams=sorted(self.weights),
            )
        )

    def finish(self, archive: StreamArchive, references: Mapping[str, dict]) -> dict:
        _need(
            self.assay_position == len(self.assays)
            and self.evaluation_position == len(self.evaluations),
            "C1 saved scalar rows lack trace evidence",
        )
        _need(
            self.result["parent_refits"] == self.result["teacher_reacquisitions"] == 0
            and self.result["followup_launched"] is False,
            "C1 unauthorized additional research",
        )
        status = self.result["status"]
        if status == "STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE":
            if (
                self.replay_mismatch_confirmed
                and self.baseline_gate is None
                and not self.assays
                and not self.evaluations
            ):
                _need(
                    self.work.requested["neural_rows"]
                    == self.work.requested["optimizer_steps"]
                    == self.work.requested["backward_rows"]
                    == self.work.requested["checkpoint_loads"]
                    == self.work.requested["policy_constructions"]
                    == 0,
                    "C1 learning executed despite failed replay prerequisite",
                )
                self.work.closed()
                self.closed_prerequisite_negative = True
                return dict(
                    scientific_contrast_completed=False,
                    scientific_status="prerequisite_negative",
                    replay_violations=self.replay_violations,
                    complete_scientific_audit=True,
                    learning_measured=False,
                    standalone_measured=False,
                    weighting_confirmed=None,
                )
            return dict(
                scientific_contrast_completed=False,
                scientific_status="prerequisite_or_infrastructure_stop",
                replay_violations=self.replay_violations,
                complete_scientific_audit=False,
            )
        _need(
            len(self.baselines) == 12
            and not self.replay_violations
            and self.baseline_gate is not None,
            "C1 complete result lacks twelve qualified replays",
        )
        _need(
            self.baseline_gate["passed"] is True
            and self.baseline_gate["tolerance"] == dict(rtol=0, atol=1e-10)
            and self.baseline_gate["native_controls"] == 12000
            and self.baseline_gate["substitution_controls_before_gate"] == 0,
            "C1 replay gate changed",
        )
        _need(
            self.result["replay_gate"]
            == dict(
                passed=True,
                episodes=12,
                sha256=self.inputs.files["worker/replay_gate.json"]["sha256"],
            ),
            "C1 result replay gate binding differs",
        )
        _need(
            all(len(v) == 4000 for v in self.deltas.values()) and len(self.assays) == 12000,
            "C1 full assay census incomplete",
        )
        contrast = self.check_weights()
        if status == "STOPPED_DEGENERATE_PLANNED_CONTRAST":
            _need(
                not contrast
                and self.result["learning_measured"] is False
                and self.result["standalone_measured"] is False
                and not self.evaluations,
                "C1 degenerate contrast launched learning",
            )
            _need(
                not (self.run / "worker/actor_registry.json").exists()
                and not (self.run / "worker/fit_records.json").exists()
                and self.work.requested["neural_rows"]
                == self.work.requested["backward_rows"]
                == self.work.requested["optimizer_steps"]
                == 0,
                "C1 unadmitted fits/evaluations occurred",
            )
            return dict(
                scientific_contrast_completed=False,
                scientific_status="degenerate_planned_contrast",
                complete_scientific_audit=True,
                weighting_confirmed=False,
                assay_count=12000,
                audit_shuffle_calls=624,
                audit_shuffle_entries=12000,
            )
        _need(
            status == "COMPLETED_FIXED_C1_ROSTER" and contrast, "Unknown/unclosed C1 result status"
        )
        self._actors()
        metrics = self.check_metrics(archive)
        outcomes = self.check_outcomes()
        _need(
            self.result["learning_measured"] is True and self.result["standalone_measured"] is True,
            "C1 complete result hides missing learning/evaluations",
        )
        self.work.closed()
        return dict(
            scientific_contrast_completed=True,
            scientific_status="completed_fixed_roster",
            complete_scientific_audit=True,
            weighting_confirmed=outcomes["gates"]["confirmed"],
            assays=12000,
            teacher_replays=12,
            standalone_episodes=432,
            fitting=self.fit_checks,
            metrics=metrics,
            outcomes=outcomes,
            audit_shuffle_calls=624,
            audit_shuffle_entries=12000,
        )


def _reconcile_accounting(checks: _Scientific, stream: dict) -> dict:
    work, saved = checks.work, checks.result["costs"]
    # These native helper calls have no duplicate per-control journal lines.
    # Their counts follow from the fixed frozen execution boundary and saved
    # native/restore/reset intents, not from an additional physics replay.
    complete = checks.closed_prerequisite_negative or checks.result["status"] in {
        "COMPLETED_FIXED_C1_ROSTER",
        "STOPPED_DEGENERATE_PLANNED_CONTRAST",
    }
    if complete:
        for counts in (work.requested, work.completed):
            counts["external_mj_forward_calls"] = counts["restores"] + 2 * counts["resets"]
            counts["snapshot_reads"] = (
                2 * counts["native_controls"] + counts["restores"] + counts["resets"]
            )
        work.closed()
        native = (
            sum(row["controls"] for row in checks.baselines.values())
            + sum(row["controls"] for row in checks.assay_rows)
            + sum(row["controls"] for row in checks.evaluation_rows)
        )
        evaluation = sum(row["controls"] for row in checks.evaluation_rows)
        full = checks.result["status"] == "COMPLETED_FIXED_C1_ROSTER"
        restores = 12012 if not checks.closed_prerequisite_negative else len(checks.baselines)
        resets = 432 if full else 0
        geometry = dict(
            native_controls=native,
            physics_substeps=4 * native,
            restores=restores,
            resets=resets,
            environment_constructions=1,
            external_mj_forward_calls=restores + 2 * resets,
            snapshot_reads=2 * native + restores + resets,
        )
        if not checks.closed_prerequisite_negative:
            geometry.update(shuffle_calls=624, shuffle_entries=12000)
        if full:
            geometry.update(
                neural_rows=4608000 + 90000 + evaluation,
                prediction_calls=18000 + 90 + evaluation,
                backward_rows=4608000,
                optimizer_steps=18000,
                checkpoint_loads=12,
                policy_constructions=21,
                endpoint_saves=9,
                expected_index_draws=1536000,
                actual_index_draws=4608000,
            )
        _need(
            {key: value for key, value in work.completed.items() if value}
            == {key: value for key, value in geometry.items() if value},
            "C1 completed work differs from independently saved trace/fit geometry",
        )
        for name, counts in (
            ("requested", work.requested),
            ("completed", work.completed),
            ("unknown_partial", work.unknown),
            ("returned_unpersisted", work.unpersisted),
        ):
            actual = {key: value for key, value in saved.get(name, {}).items() if value}
            expected = {key: value for key, value in counts.items() if value}
            _need(actual == expected, "C1 complete worker work census differs: " + name)
    evidence = checks.run / "worker/evidence"
    sizes = {
        name: (evidence / name).stat().st_size
        for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl")
    }
    _need(
        saved["storage"]["storage_bytes"] == sum(sizes.values()),
        "C1 final evidence-byte accounting differs",
    )
    _need(
        sum(sizes.values()) <= 8 * 1024**3 and checks.result["worker_wall_seconds"] <= 21600,
        "C1 worker budget exceeded",
    )
    return dict(
        independent_requested=dict(work.requested),
        independent_completed=dict(work.completed),
        independent_unknown_partial=dict(work.unknown),
        independent_returned_unpersisted=dict(work.unpersisted),
        worker_reported=saved,
        evidence_file_bytes=sizes,
        source_read_bytes=stream["source_read_bytes"]
        + checks.inputs.read_bytes
        + checks.inputs.checkpoint_read_bytes,
        complete_reconciliation=complete,
        partial_scope="On early infrastructure stops, unmatched work and unavailable outputs remain explicit; exact native helper counts are not inferred beyond saved successful boundaries.",
    )


def audit_run(run: Path, out: Path) -> dict[str, Any]:
    """Audit one closed frozen C1 run, with no training/native replay or retry."""
    run, out = Path(run).resolve(), Path(out).resolve()
    _need(not out.is_relative_to(run), "C1 audit output must be outside the frozen run")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = _Inputs(run, started + MAX_WALL_SECONDS)
    receipt: dict[str, Any] = dict(
        schema="walker-student-consequence-c1.0-audit-v1",
        passed=False,
        scientific_contrast_completed=False,
        full_scientific_audit_passed=False,
        physics_controls=0,
        optimizer_steps=0,
        backward_rows=0,
    )
    checks = None
    stream = None
    try:
        config = inputs.json("config.json")
        _need(
            config["study"] == "C1.0"
            and config["user_authorized"] is True
            and config["automatic_followup"] is False,
            "C1 recipe identity/authorization differs",
        )
        result = inputs.json("worker_result.json", 64 * 1024**2)
        receipt.update(
            config_sha256=inputs.files["config.json"]["sha256"],
            worker_result_sha256=inputs.files["worker_result.json"]["sha256"],
        )
        _need(
            result["schema"] == "walker-student-consequence-c1.0-result-v1",
            "C1 result schema differs",
        )
        checks = _Scientific(run, out, inputs, config, result)
        scientific: dict = {}

        def finish(archive, references):
            scientific.update(checks.finish(archive, references))

        stream = audit_saved_stream(
            run / "worker/evidence",
            out / "stream",
            task=config["task"],
            max_read_bytes=MAX_READ_BYTES - inputs.read_bytes,
            max_wall_seconds=max(0.001, inputs.deadline - time.monotonic()),
            _trace_check=checks.trace,
            _event_check=checks.event,
            _archive_check=finish,
            _archive_open=lambda archive: setattr(inputs, "archive", archive),
        )
        _need(
            stream["structural_passed"] and stream["complete_journal"],
            "C1 saved stream audit failed: " + str(stream.get("error", "unclosed journal")),
        )
        accounting = _reconcile_accounting(checks, stream)
        _need(
            accounting["source_read_bytes"] <= MAX_READ_BYTES, "C1 total audit read budget exceeded"
        )
        inputs.unchanged()
        complete = scientific.get("complete_scientific_audit", False)
        receipt.update(
            passed=bool(complete),
            evidence_integrity_passed=True,
            full_scientific_audit_passed=bool(complete),
            **{
                key: value
                for key, value in scientific.items()
                if key != "complete_scientific_audit"
            },
            stream=stream,
            accounting=accounting,
            input_files=inputs.files,
        )
        if not complete:
            receipt["error"] = (
                "Recorded early stop preserved; complete scientific contrast/causal closure is unavailable."
            )
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error), input_files=inputs.files)
    finally:
        if stream is not None:
            receipt.setdefault("stream", stream)
        if checks is not None:
            receipt["derivatives"] = checks.derivatives()
            if checks.prediction_journal is not None:
                checks.prediction_journal.close()
            if checks.verifier is not None:
                verifier = checks.verifier
                receipt.update(
                    prediction_rows=verifier.neural_rows,
                    completed_prediction_rows=verifier.completed_neural_rows,
                    unknown_partial_prediction_rows=verifier.unknown_partial_neural_rows,
                    prediction_calls=verifier.prediction_calls,
                    checkpoint_loads=verifier.checkpoint_loads,
                    checkpoint_read_bytes=verifier.checkpoint_read_bytes,
                )
    receipt.setdefault("prediction_rows", 0)
    receipt.setdefault("prediction_calls", 0)
    receipt.setdefault("checkpoint_loads", 0)
    receipt["source_read_bytes_total"] = (
        inputs.read_bytes
        + (0 if stream is None else stream.get("source_read_bytes", 0))
        + (
            0
            if checks is None or checks.verifier is None
            else checks.verifier.checkpoint_read_bytes
        )
    )
    receipt.update(
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=_peak_rss(),
        input_read_bytes=inputs.read_bytes,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    receipt["process_lifetime_cpu"] = dict(
        user_seconds=usage.ru_utime,
        system_seconds=usage.ru_stime,
        peak_rss_bytes=_peak_rss(),
        reserved_cpu_cores=16,
        billing="No provider invoice/rate available; no dollar estimate inferred",
    )
    receipt["process_lifetime_io"] = (
        dict(
            counters={
                line.split(":")[0]: int(line.split(":")[1])
                for line in Path("/proc/self/io").read_text().splitlines()
            },
            scope="Linux process counters through this snapshot; final receipt write follows",
        )
        if Path("/proc/self/io").is_file()
        else None
    )
    existing_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    payload = _canonical(receipt) + b"\n"
    _need(
        existing_bytes + len(payload) <= MAX_OUTPUT_BYTES and _peak_rss() <= MAX_RSS_BYTES,
        "C1 audit final output/RSS budget exceeded",
    )
    with (out / "audit.json").open("xb") as stream:
        stream.write(payload)
    return receipt


run_c1_audit = audit_run
