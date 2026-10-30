"""Independent saved-evidence checks for heterogeneous native CPU H1 queries.

Only stdlib and NumPy are imported. No producer, task, engine, solver or policy
code is imported or executed. This checks arithmetic/identity/accounting and
does not independently integrate dynamics or establish reference feasibility.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


SCHEMA = "bidirectional-heterogeneous-native-h1-v1"
_RESTART = "explicit-zero-solver-warmstart-intervention"
_BASE_FILES = ("assets/acrobot.xml", "src/envs/acrobot.py", "src/envs/mujoco_env.py",
               "src/gps/bidirectional/task.py")
_EXECUTOR = "src/gps/bidirectional/hetero_executor.py"
_TASK = dict(T=800, dt=.01, capture_steps=100, capture_distance=.2, capture_qvel_norm=.5,
             local_distance=.5, local_by=600, failure_penalty=1000.,
             reset_angle_radius=.1, reset_velocity_radius=.05)
_COMMON = frozenset(("schema", "task_hash", "task_json", "source_fingerprints_json", "owner_json",
    "metadata_json", "purpose", "dynamics_only", "returns_semantics", "initial_states", "initial_elapsed",
    "initial_warmstarts", "initial_warmstart_exact", "initial_restart_semantics", "actions"))
_OUTPUT = _COMMON | frozenset(("prestates", "states", "elapsed", "input_sha256", "endpoint_warmstarts",
    "endpoint_warmstart_exact", "endpoint_restart_semantics", "backend_json"))
_EVENT = frozenset(("id", "kind", "utc", "wall_seconds"))
_REQUEST = _EVENT | frozenset(("query_id", "schema", "query_kind", "mode", "scope", "owner", "metadata",
    "purpose", "task_hash", "source_fingerprints", "rows", "horizon", "initial_elapsed", "dynamics_only",
    "returns_semantics", "requested_controls", "requested_physics_controls", "requested_physics_frames",
    "cutoff_skipped_controls", "budget_reservation_id", "budget_requested_cumulative", "inputs"))
_COMPLETE = _EVENT | frozenset(("query_id", "schema", "scope", "owner", "purpose", "dynamics_only",
    "returns_semantics", "backend", "completed_controls", "completed_physics_frames", "query_wall_seconds",
    "budget_reservation_id", "result"))
_ERROR = _EVENT | frozenset(("query_id", "schema", "scope", "purpose", "call_started", "call_returned",
    "completed_controls", "completed_physics_frames", "partial_physics_unknown", "unknown_partial",
    "error_type", "error", "query_wall_seconds", "budget_reservation_id"))
_CAUGHT = (ValueError, TypeError, KeyError, IndexError, OSError, OverflowError, EOFError,
           FloatingPointError, zipfile.BadZipFile, ET.ParseError)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _loads(data):
    def reject(value):
        raise ValueError("Nonfinite JSON constant: "+value)
    value = json.loads(data, parse_constant=reject)
    _canonical(value)
    return value


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "Invalid SHA256")
    return value


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label+" must be an integer >="+str(minimum))
    return value


def _text(value, label):
    _require(isinstance(value, str) and bool(value.strip()), "Missing "+label)
    return value


def _scalar(raw, key):
    value = raw[key]
    _require(value.shape == (), "Expected scalar "+key)
    return value.item()


def _packed(raw, key):
    value = _scalar(raw, key)
    _require(isinstance(value, str), "Packed JSON must be a string: "+key)
    return _loads(value)


def _numeric(value, shape, label):
    value = np.asarray(value)
    _require(value.shape == shape and value.dtype == np.dtype("float64") and np.isfinite(value).all(),
             "Invalid float64 "+label)
    return value


def _integers(value, shape, label):
    value = np.asarray(value)
    _require(value.shape == shape and value.dtype.kind in "iu", "Invalid integer "+label)
    return value


def _zero_flags(raw, key, count):
    value = raw[key]
    _require(value.shape == (count,) and value.dtype.kind == "b" and not value.any(), "Invalid "+key)


class _Reader:
    def __init__(self, root):
        self.root, self.hashes = root, {}

    def file(self, path, *, inside=None):
        path = Path(path)
        if not path.exists() and not path.is_symlink():
            raise FileNotFoundError("Missing artifact: "+str(path))
        _require(path.is_file() and not path.is_symlink(), "Missing/nonregular artifact: "+str(path))
        if inside is not None:
            _require(path.resolve().is_relative_to(Path(inside).resolve()), "Artifact escapes its declared root")
        data = path.read_bytes()
        self.hashes[str(path.resolve())] = _digest(data)
        return data

    def arrays(self, reference, expected_name):
        _require(isinstance(reference, dict) and set(reference) == {"file", "sha256", "bytes"}, "Invalid raw reference")
        name = PurePosixPath(reference["file"])
        _require(not name.is_absolute() and ".." not in name.parts and str(name) == expected_name,
                 "Raw artifact path/query binding differs")
        data = self.file(self.root/str(name), inside=self.root)
        _require(_digest(data) == _sha(reference["sha256"]), "Raw artifact SHA256 differs")
        _require(len(data) == _integer(reference["bytes"], "artifact bytes", 1), "Raw artifact byte count differs")
        with np.load(io.BytesIO(data), allow_pickle=False) as packet:
            _require(len(packet.files) == len(set(packet.files)), "Duplicate NPZ fields")
            return {key: packet[key].copy() for key in packet.files}


def _counts(rows, partial):
    requested = [r for r in rows if r.get("kind") == "query_requested"]
    done_rows = [r for r in rows if r.get("kind") == "query_complete"]
    failed_rows = [r for r in rows if r.get("kind") == "query_error"]
    valid_id = lambda r: isinstance(r.get("query_id"), str) and bool(r["query_id"])
    begins = {r["query_id"]: r for r in requested if valid_id(r)}
    done = {r["query_id"]: r for r in done_rows if valid_id(r)}
    failed = {r["query_id"]: r for r in failed_rows if valid_id(r)}
    returned = {q: r for q, r in failed.items() if r.get("call_returned") is True}
    unresolved = set(begins)-set(done)-set(failed)
    number = lambda r, key: r.get(key, 0) if type(r.get(key, 0)) is int and r.get(key, 0) >= 0 else 0
    physical = sum(number(r, "requested_physics_controls") for r in requested)
    requested_count = sum(number(r, "requested_controls") for r in requested)
    known = {**done, **returned}
    counts = dict(requested_controls=requested_count, charged_requested_controls=requested_count,
        requested_physics_controls=physical,
        requested_physics_frames=sum(number(r, "requested_physics_frames") for r in requested),
        completed_controls=sum(number(r, "completed_controls") for r in known.values()),
        completed_physics_frames=sum(number(r, "completed_physics_frames") for r in known.values()),
        cutoff_skipped_controls=requested_count-physical,
        error_requested_controls=sum(number(begins.get(q, {}), "requested_controls") for q in failed),
        unknown_error_controls=sum(number(begins.get(q, {}), "requested_physics_controls") for q in failed if q not in returned),
        inflight_or_interrupted_controls=sum(number(begins[q], "requested_physics_controls") for q in unresolved),
        unknown_controls=sum(number(begins.get(q, {}), "requested_physics_controls")
                             for q in unresolved | (set(failed)-set(returned))),
        queries=len(requested), query_attempts=len(requested), completed_queries=len(done_rows), error_queries=len(failed_rows),
        inflight_or_interrupted_queries=len(unresolved), rejected_queries=0, incomplete_final_line=partial)
    return counts, begins, done, failed


def _source_packet(root, source_root, reader, guard, pending):
    sources = {}
    source_root = Path(source_root) if source_root is not None else root.parents[1]/"source"
    for name in (*_BASE_FILES, _EXECUTOR):
        data = guard("frozen source "+name, lambda name=name: reader.file(source_root/name, inside=source_root), missing=True)
        if data is not None:
            sources[name] = _digest(data)
            if name == "assets/acrobot.xml":
                def xml_checks():
                    xml = ET.fromstring(data)
                    option = xml.find("option")
                    _require(option is not None and float(option.get("timestep", "nan")) == .01
                             and option.get("integrator") == "RK4", "Native XML clock/integrator differs")
                    flag = option.find("flag")
                    _require(flag is not None and flag.get("constraint") == "disable", "Native XML constraints enabled")
                    motors = xml.findall("actuator/motor")
                    _require(len(motors) == 1 and motors[0].get("joint") == "elbow"
                             and float(motors[0].get("gear", "nan")) == 2.
                             and motors[0].get("ctrllimited") == "true"
                             and motors[0].get("ctrlrange", "").split() == ["-1", "1"], "Native XML actuator differs")
                guard("frozen native XML", xml_checks)
    run_root = root.parents[1]
    manifest_blob = guard("run manifest", lambda: reader.file(run_root/"manifest.json", inside=run_root), missing=True)
    run_fingerprint = cap = None
    if manifest_blob is not None:
        def manifest_checks():
            manifest = _loads(manifest_blob)
            _require(isinstance(manifest, dict) and isinstance(manifest.get("files"), list), "Invalid source manifest")
            paths = []
            for item in manifest["files"]:
                _require(isinstance(item, dict), "Invalid source inventory row")
                name = PurePosixPath(item["file"])
                _require(not name.is_absolute() and ".." not in name.parts and name.parts[0] == "source", "Unsafe frozen source path")
                paths.append(str(name))
                blob = reader.file(run_root/str(name), inside=run_root)
                _require(_digest(blob) == _sha(item["sha256"]) and len(blob) == _integer(item["bytes"], "source bytes"),
                         "Frozen source inventory differs: "+str(name))
            _require(len(paths) == len(set(paths)), "Duplicate frozen source inventory")
            _require({"source/"+name for name in (*_BASE_FILES, _EXECUTOR)} <= set(paths), "Manifest lacks native source identities")
            for name, sha in sources.items():
                declared = next(item["sha256"] for item in manifest["files"] if item["file"] == "source/"+name)
                _require(sha == declared, "Explicit source root differs from frozen manifest")
            config_blob = reader.file(run_root/"config.json", inside=run_root)
            protocol_blob = reader.file(run_root/"protocol.json", inside=run_root)
            _require(_digest(config_blob) == _sha(manifest["config_sha256"])
                     and _digest(protocol_blob) == _sha(manifest["protocol_sha256"]), "Frozen config/protocol differs")
            configuration, protocol = _loads(config_blob), _loads(protocol_blob)
            _require(isinstance(configuration, dict) and isinstance(protocol, dict)
                     and isinstance(manifest["dependency_versions"], dict), "Invalid source identity metadata")
            fingerprint = _digest(_canonical(dict(files=manifest["files"], config_sha256=manifest["config_sha256"],
                protocol_sha256=manifest["protocol_sha256"], dependency_versions=manifest["dependency_versions"])).encode())
            _require(fingerprint == _sha(manifest["source_fingerprint"]), "Recomputed run source fingerprint differs")
            limit = _integer(configuration["max_controls"], "frozen shared control cap", 1)
            return fingerprint, limit
        verified = guard("run source manifest", manifest_checks, missing=True)
        if verified is not None:
            run_fingerprint, cap = verified
    return source_root, sources, run_fingerprint, cap


def audit_heterogeneous(path, *, source_root=None, max_requested_controls=None, require_closed=True):
    """Audit evidence/heterogeneous only; compose ordinary work separately.

    Shared-budget cumulative counts may jump because ordinary queries use a
    sibling archive. This report checks their monotonic lower bounds and cap;
    a parent audit must reconcile the complete shared-budget sequence.
    """
    started = time.perf_counter()
    root = Path(path).resolve(); reader = _Reader(root)
    errors, pending, rows, public = [], [], [], []
    partial = False
    def guard(label, callback, missing=False):
        try:
            return callback()
        except _CAUGHT as error:
            target = pending if missing and isinstance(error, FileNotFoundError) else errors
            target.append(label+": "+str(error))
            return None
    explicit_cap = None
    if max_requested_controls is not None:
        explicit_cap = guard("explicit shared cap", lambda: _integer(max_requested_controls, "shared control cap"))
    ledger_blob = guard("ledger", lambda: reader.file(root/"ledger.jsonl", inside=root), missing=True)
    if ledger_blob is not None:
        lines = ledger_blob.splitlines(keepends=True)
        for index, line in enumerate(lines):
            unfinished = index == len(lines)-1 and not line.endswith(b"\n")
            partial = partial or unfinished
            try:
                row = _loads(line)
                _require(isinstance(row, dict), "Ledger row must be an object")
                rows.append(row)
            except _CAUGHT as error:
                if not unfinished:
                    errors.append("ledger line "+str(index)+": "+str(error))
    counts, requests, completed, failed = _counts(rows, partial)
    archive_counts = dict(counts)
    closed = bool(rows and rows[-1].get("kind") == "archive_close" and not partial)
    if partial:
        pending.append("unfinished ledger tail")
    if require_closed and not closed:
        pending.append("heterogeneous archive is not closed")
    source_root, sources, run_fingerprint, frozen_cap = _source_packet(root, source_root, reader, guard, pending)
    limits = [limit for limit in (explicit_cap, frozen_cap) if limit is not None]
    cap = min(limits) if limits else None
    def ledger_checks():
        _require(rows and rows[0].get("kind") == "archive_begin" and rows[0].get("schema") == "bidirectional-evidence-v1", "Archive begin/schema missing")
        _require([r.get("id") for r in rows] == [f"event-{i:09d}" for i in range(len(rows))], "Ledger IDs repeat or reorder")
        _require(sum(r.get("kind") == "archive_begin" for r in rows) == 1, "Duplicate archive begin")
        _require(len(requests) == counts["queries"] and len(completed) == counts["completed_queries"]
                 and len(failed) == counts["error_queries"], "Duplicate/missing query IDs")
        _require(not set(completed) & set(failed) and (set(completed) | set(failed)) <= set(requests), "Query resolutions lack unique requests")
        seen, budget_ids, last_budget, last_index, last_wall = set(), set(), 0, -1, -1.
        for index, row in enumerate(rows):
            kind = row.get("kind")
            _require(kind in {"archive_begin", "archive_close", "query_requested", "query_complete", "query_error"}, "Unexpected heterogeneous ledger event")
            wall = row.get("wall_seconds")
            _require(type(wall) in (int, float) and math.isfinite(wall) and wall >= last_wall and wall >= 0, "Ledger wall clock regressed")
            last_wall = wall
            if kind == "archive_close":
                _require(index == len(rows)-1, "Events after archive close")
                for key, value in archive_counts.items():
                    if key == "incomplete_final_line" and partial:
                        continue  # A torn final newline is separately PENDING.
                    _require(_canonical(row["accounting"].get(key)) == _canonical(value), "Archive close accounting differs: "+key)
            if kind == "query_requested":
                _require(set(row) == _REQUEST, "Heterogeneous request fields differ")
                qid = row["query_id"]
                _require(isinstance(qid, str) and re.fullmatch(r"q[0-9]{9}", qid) is not None, "Invalid query ID")
                seen.add(qid)
                n = _integer(row["requested_controls"], "requested controls", 1)
                _require(row["requested_physics_controls"] == row["requested_physics_frames"] == n
                         and type(row["requested_physics_controls"]) is int and type(row["requested_physics_frames"]) is int
                         and type(row["cutoff_skipped_controls"]) is int and row["cutoff_skipped_controls"] == 0, "H1 requested counts differ")
                cumulative = _integer(row["budget_requested_cumulative"], "cumulative budget", 1)
                token = row["budget_reservation_id"]
                _require(isinstance(token, str) and re.fullmatch(r"b[0-9]{9}", token) is not None and token not in budget_ids, "Invalid/duplicate budget reservation")
                token_index = int(token[1:])
                _require(token_index > last_index and cumulative >= last_budget+n, "Shared budget regressed or erased heterogenous work")
                if cap is not None:
                    _require(cumulative <= cap, "Shared control cap exceeded")
                budget_ids.add(token); last_budget, last_index = cumulative, token_index
            elif kind in {"query_complete", "query_error"}:
                _require(row.get("query_id") in seen, "Query resolved before durable intent")
                duration = row.get("query_wall_seconds")
                _require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0, "Invalid query duration")
        counts["budget_requested_cumulative"] = last_budget
    guard("ledger invariants", ledger_checks)
    verified_rows = 0
    for qid, request in requests.items():
        def query_checks():
            _require(request["schema"] == SCHEMA and request["mode"] == "model_heterogeneous"
                     and request["query_kind"] == "heterogeneous_native_h1", "Unknown heterogeneous query schema/mode")
            _text(request["scope"], "scope"); _text(request["purpose"], "purpose")
            _require(request["dynamics_only"] is True and request["returns_semantics"] == "none", "H1 query claims return semantics")
            _require(isinstance(request["owner"], dict) and isinstance(request["metadata"], dict), "Owner/metadata must be objects")
            inputs = reader.arrays(request["inputs"], f"queries/{qid}_heterogeneous_inputs.npz")
            _require(set(inputs) == _COMMON, "H1 input fields/cost semantics differ")
            count = _integer(request["rows"], "rows", 1)
            _require(count <= 8000, "Heterogeneous native capacity exceeds8000 rows")
            _require(type(request["horizon"]) is int and request["horizon"] == 1
                     and count == request["requested_controls"], "H1 dimensions/counts differ")
            state = _numeric(inputs["initial_states"], (count, 5), "initial states")
            action = _numeric(inputs["actions"], (count, 1, 1), "native actions")
            _require(np.max(np.abs(action)) <= 1., "Native action bounds exceeded")
            elapsed = _integers(inputs["initial_elapsed"], (count,), "initial elapsed")
            _require(np.all((elapsed >= 0) & (elapsed < 800)) and
                     np.allclose(state[:, 0], elapsed*.01, atol=1e-12, rtol=0), "Native initial clock differs")
            _require(_canonical(request["initial_elapsed"]) == _canonical(elapsed.tolist()), "Per-row ledger elapsed differs")
            _require(not _numeric(inputs["initial_warmstarts"], (count, 2), "initial warmstarts").any(), "Initial warmstarts are not zero")
            _zero_flags(inputs, "initial_warmstart_exact", count)
            _require(_scalar(inputs, "initial_restart_semantics") == _RESTART, "Initial warmstart provenance differs")
            _require(_scalar(inputs, "schema") == SCHEMA and _scalar(inputs, "dynamics_only") is True
                     and _scalar(inputs, "returns_semantics") == "none", "Raw query schema/return semantics differ")
            _require(_scalar(inputs, "purpose") == request["purpose"] and
                     _canonical(_packed(inputs, "owner_json")) == _canonical(request["owner"]) and
                     _canonical(_packed(inputs, "metadata_json")) == _canonical(request["metadata"]), "Query intent/metadata binding differs")
            config = _packed(inputs, "task_json")
            _require(isinstance(config, dict) and set(config) == set(_TASK) and config == _TASK
                     and all(type(config[k]) is int for k in ("T", "capture_steps", "local_by"))
                     and all(type(v) in (int, float) for v in config.values()), "Frozen native task config differs")
            task_hash = _sha(request["task_hash"])
            _require(_scalar(inputs, "task_hash") == task_hash, "Raw/ledger task identity differs")
            if set(_BASE_FILES) <= set(sources):
                expected_task = _digest(_canonical(dict(schema="acrobot-capture-v2-post-action-fk", config=config,
                    base_cost="unchanged-src.envs.acrobot.Acrobot", cost_boundary_source="post-action-full-state-analytic-fk",
                    base_source_sha256={name: sources[name] for name in _BASE_FILES})).encode())
                _require(task_hash == expected_task, "Independently reconstructed task identity differs")
            fingerprints = _packed(inputs, "source_fingerprints_json")
            _require(isinstance(fingerprints, dict) and _canonical(fingerprints) == _canonical(request["source_fingerprints"]), "Source fingerprint binding differs")
            for value in fingerprints.values(): _sha(value)
            _require(fingerprints.get("task_hash") == task_hash and fingerprints.get("task_config_sha256") == _digest(_canonical(config).encode()), "Task/config fingerprint differs")
            if _EXECUTOR in sources:
                _require(fingerprints.get("heterogeneous_executor_sha256") == sources[_EXECUTOR], "Executor source fingerprint differs")
            for name, value in fingerprints.items():
                if name in {"task_hash", "task_config_sha256", "heterogeneous_executor_sha256"}: continue
                if name == "run_source_fingerprint":
                    if run_fingerprint is not None: _require(value == run_fingerprint, "Run source fingerprint differs")
                    continue
                rel = PurePosixPath(name)
                if not rel.is_absolute() and ".." not in rel.parts and name.startswith(("src/", "assets/", "tests/", "scripts/")):
                    blob = reader.file(source_root/name, inside=source_root)
                    _require(_digest(blob) == value, "Additional source fingerprint differs")
                else:
                    pending.append("unverified symbolic source fingerprint: "+name)
            _require("heterogeneous_executor_sha256" in fingerprints, "Missing executor source fingerprint")
            if qid not in completed:
                pending.append("failed/unresolved heterogeneous query lacks complete output: "+qid)
                return 0
            completion = completed[qid]
            _require(set(completion) == _COMPLETE, "H1 completion fields differ")
            for key in ("schema", "scope", "owner", "purpose", "budget_reservation_id", "dynamics_only", "returns_semantics"):
                _require(_canonical(completion[key]) == _canonical(request[key]), "Completion intent binding differs: "+key)
            _require(completion["backend"] == "mujoco_cpu" and type(completion["completed_controls"]) is int
                     and type(completion["completed_physics_frames"]) is int
                     and completion["completed_controls"] == completion["completed_physics_frames"] == count, "H1 completion controls/backend differ")
            raw = reader.arrays(completion["result"], f"queries/{qid}_heterogeneous_result.npz")
            _require(set(raw) == _OUTPUT, "H1 result fields/cost semantics differ")
            for key in _COMMON:
                _require(raw[key].dtype == inputs[key].dtype and np.array_equal(raw[key], inputs[key]), "Result/input binding differs: "+key)
            _require(_scalar(raw, "input_sha256") == request["inputs"]["sha256"], "Result input SHA binding differs")
            pre = _numeric(raw["prestates"], (count, 1, 5), "prestates")
            _require(np.array_equal(pre[:, 0], state), "H1 prestate differs")
            output = _numeric(raw["states"], (count, 1, 5), "output states")
            _require(np.allclose(output[:, 0, 0], state[:, 0]+.01, atol=1e-12, rtol=0), "H1 output clock differs")
            _require(np.array_equal(_integers(raw["elapsed"], (count,), "output elapsed"), elapsed+1), "H1 output elapsed differs")
            _require(not _numeric(raw["endpoint_warmstarts"], (count, 2), "endpoint warmstarts").any(), "Endpoint warmstarts are not zero")
            _zero_flags(raw, "endpoint_warmstart_exact", count)
            _require(_scalar(raw, "endpoint_restart_semantics") == _RESTART, "Endpoint warmstart provenance differs")
            _require(_packed(raw, "backend_json") == dict(backend="mujoco_cpu", frame_skip=1, horizon=1,
                timestep=.01, integrator="RK4", heterogeneous_initial_states=True), "Raw native backend differs")
            public.append(dict(query_id=qid, rows=count, initial_elapsed=elapsed.tolist(),
                purpose=request["purpose"], task_hash=task_hash, budget_requested_cumulative=request["budget_requested_cumulative"]))
            return count
        checked = guard("query "+qid, query_checks)
        if checked is not None: verified_rows += checked
    for qid, error in failed.items():
        def error_checks():
            request = requests[qid]
            _require(set(error) == _ERROR, "H1 error fields differ")
            for key in ("schema", "scope", "purpose", "budget_reservation_id"):
                _require(error[key] == request[key], "Error intent binding differs: "+key)
            returned, attempted = error["call_returned"], error["call_started"]
            _require(type(returned) is bool and type(attempted) is bool and not (returned and not attempted), "Invalid failed-call provenance")
            expected = request["requested_physics_controls"] if returned else 0
            _require(type(error["completed_controls"]) is int and type(error["completed_physics_frames"]) is int
                     and error["completed_controls"] == error["completed_physics_frames"] == expected
                     and error["partial_physics_unknown"] is (attempted and not returned)
                     and error["unknown_partial"] is (attempted and not returned), "Failed query erased/invented native work")
            _text(error["error_type"], "error type")
            _require(isinstance(error["error"], str), "Invalid error text")
        guard("query error "+qid, error_checks)
    counts["completed_rows_checked"] = verified_rows
    counts["queries_checked"] = len(public)
    return dict(schema="bidirectional-independent-heterogeneous-h1-audit-v1",
        status="FAIL" if errors else "PENDING" if pending else "PASS", arithmetic_valid=not errors,
        errors=errors, pending=list(dict.fromkeys(pending)), counts=counts, work=dict(counts),
        closed=closed, archive=str(root), queries=public, input_sha256=reader.hashes,
        source_sha256=_digest(Path(__file__).read_bytes()), wall_seconds=time.perf_counter()-started,
        additional_physics_controls=0, additional_actor_inference_rows=0, additional_fits=0,
        limitations=["Saved arithmetic, provenance and accounting only; no independent dynamics reintegration.",
            "Shared-budget jumps require parent reconciliation with the sibling native archive.",
            "No complete return, capture competence, learning or efficiency verdict is assigned."])
