"""Read saved run metadata without loading a policy or simulator.

Inspection is deliberately separate from execution and auditing. It reports what
is present, including missing bulk evidence; it does not establish scientific
validity. Hash verification reads files but never repairs or rewrites them.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        errors.append(f"{path}: {error}")
        return {}


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _contained_path(root: Path, name: str) -> Path:
    """Reject descriptors that escape the selected evidence directory."""
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"evidence path escapes run: {name}")
    return path


def _files(
    root: Path, bindings: dict[str, Any], verify: bool, errors: list[str]
) -> list[dict[str, Any]]:
    records = []
    for name, binding in bindings.items():
        expected = binding if isinstance(binding, str) else binding.get("sha256")
        size = binding.get("bytes") if isinstance(binding, dict) else None
        try:
            path = _contained_path(root, name)
            present = path.is_file()
            record = {
                "file": name,
                "available": present,
                "expected_sha256": expected,
                "expected_bytes": size,
                "bytes": path.stat().st_size if present else None,
            }
            if verify and present:
                record["sha256"] = _digest(path)
                record["verified"] = (expected is None or record["sha256"] == expected) and (
                    size is None or record["bytes"] == size
                )
            elif verify:
                record["verified"] = False
                errors.append(f"Missing bound evidence: {path}")
            records.append(record)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    return records


def _summaries(value: Any) -> Any:
    """Keep outcome scalars while omitting episode arrays and raw descriptors."""
    if not isinstance(value, dict):
        return value
    keys = ("episodes", "successes", "survived", "mean_cost", "mean_steps", "mean_velocity_mae")
    if "episodes" in value:
        return {key: value[key] for key in keys if key in value}
    if "cost" in value:
        return {
            key: value[key]
            for key in ("complete", "success", "survived", "steps", "cost", "mae")
            if key in value
        }
    return {key: _summaries(item) for key, item in value.items() if isinstance(item, dict)}


def _h5_compact_metadata(
    root: Path, name: str, bindings: dict[str, Any], *, verify: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one optional copied JSON, bounded independently of external evidence."""
    descriptor: dict[str, Any] = dict(file=name, available=False, binding_verified=None)
    try:
        path = _contained_path(root, name)
        if not path.is_file():
            return {}, descriptor
        before = path.stat()
        descriptor.update(available=True, bytes=before.st_size)
        limit = 4 * 1024**2
        if before.st_size > limit:
            raise ValueError("Optional compact audit JSON exceeds its 4 MiB read limit")
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
        descriptor["read_bytes"] = len(payload)
        after = path.stat()
        if len(payload) > limit or (before.st_size, before.st_mtime_ns) != (
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Optional compact audit JSON changed during bounded read")
        binding = bindings.get(name)
        digest = binding if isinstance(binding, str) else (binding or {}).get("sha256")
        size = binding.get("bytes") if isinstance(binding, dict) else None
        descriptor["bound_in_capsule"] = digest is not None
        if verify:
            descriptor["sha256"] = hashlib.sha256(payload).hexdigest()
            descriptor["binding_verified"] = digest is not None and (
                descriptor["sha256"] == digest and (size is None or len(payload) == size)
            )

        def finite_float(value: str) -> float:
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("Nonfinite compact audit JSON number")
            return result

        def reject(value: str) -> None:
            raise ValueError("Nonfinite compact audit JSON constant: " + value)

        value = json.loads(payload, parse_constant=reject, parse_float=finite_float)
        if not isinstance(value, dict):
            raise ValueError("Optional compact audit metadata must be a JSON object")
        return value, descriptor
    except (OSError, TypeError, ValueError) as error:
        descriptor["error"] = str(error)
        return {}, descriptor


def _h5_capsule_audit(root: Path, capsule: dict[str, Any], *, verify: bool) -> dict[str, Any]:
    """Expose recorded audit closure without claiming a repeated scientific audit."""
    bindings = capsule.get("files", {})
    receipt, receipt_file = _h5_compact_metadata(
        root, "audit/result/receipt.json", bindings, verify=verify
    )
    outer, outer_file = _h5_compact_metadata(root, "audit/exit.json", bindings, verify=verify)
    descriptors = [receipt_file, outer_file]
    readable = bool(receipt) and bool(outer) and not any(row.get("error") for row in descriptors)
    recorded_closed = readable and (
        receipt.get("schema") == "walker-h5.0-round-audit-v1"
        and receipt.get("scientific_round_audit_complete") is True
        and receipt.get("prediction_replay_complete") is True
        and type(outer.get("exit_code")) is int
        and outer.get("exit_code") == 0
        and "stopping_reason" in outer
        and outer.get("stopping_reason") is None
        and outer.get("unknown_partial_work_possible") is False
    )
    metadata_verified = all(row["binding_verified"] is True for row in descriptors)
    if any(row.get("error") for row in descriptors):
        status = "optional_compact_audit_missing_or_unreadable"
    elif not any(row["available"] for row in descriptors):
        status = "optional_compact_audit_unavailable"
    elif not readable:
        status = "optional_compact_audit_missing_or_unreadable"
    elif not recorded_closed:
        status = "reported_audit_failed_or_incomplete"
    elif verify and not metadata_verified:
        status = "reported_complete_audit_metadata_unverified"
    else:
        status = (
            "reported_complete_audit_metadata_verified" if verify else "reported_complete_audit"
        )
    work_keys = (
        "audit_neural_rows",
        "audit_completed_neural_rows",
        "audit_unknown_partial_neural_rows",
        "audit_prediction_calls",
        "audit_checkpoint_loads",
        "source_read_bytes",
        "checkpoint_read_bytes",
        "frozen_input_read_bytes",
        "peak_rss_bytes",
        "wall_seconds",
        "new_native_controls",
        "new_optimizer_steps",
    )
    reported_work = {key: receipt.get(key) for key in work_keys}
    preparation = receipt.get("preparation")
    reported_work["preparation_source_read_bytes"] = (
        preparation.get("source_read_bytes") if isinstance(preparation, dict) else None
    )
    return dict(
        status=status,
        capsule_reported_integrity_status=capsule.get("integrity_status"),
        capsule_reported_scientific_audit_complete=capsule.get("scientific_round_audit_complete"),
        reported_scientific_round_audit_complete=receipt.get("scientific_round_audit_complete"),
        reported_prediction_replay_complete=receipt.get("prediction_replay_complete"),
        recorded_outer_audit_closed=recorded_closed,
        copied_metadata_verified=metadata_verified if verify else None,
        metadata_files=descriptors,
        inspector_metadata_read_bytes=sum(row.get("read_bytes", 0) for row in descriptors),
        reported_work=reported_work,
        reported_outer_closure={
            key: outer.get(key)
            for key in (
                "exit_code",
                "stopping_reason",
                "unknown_partial_work_possible",
                "wall_seconds",
                "observed_peak_rss_bytes",
                "retries",
            )
        },
        reported_outcome_scope=receipt.get("outcomes", {}),
        verification_scope="Recorded compact audit metadata and, with --verify, copied JSON bytes against capsule hashes. No external raw evidence, checkpoints or predictions are reopened or recomputed; audit cost counters are saved declarations. Nested audit/outer wall times must not be added.",
    )


def _h5_measurement_scope(capsule: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Keep an unexecuted learning contrast distinct from zero policy successes."""
    negative = result.get("status") == "teacher_competence_negative_no_child_fits"
    keys = ("learning_contrast_measured", "standalone_panels_measured", "paired_gap_measured")
    flags = {}
    for key in keys:
        if negative:
            flags[key] = False
        else:
            declared = capsule.get(key)
            flags[key] = declared if type(declared) is bool else None
    return dict(
        **flags,
        outcome_availability={
            key: "not_measured"
            if value is False
            else "reported_measured"
            if value is True
            else "not_recorded"
            for key, value in flags.items()
        },
        capsule_reported_flags={key: capsule.get(key) for key in keys},
        metadata_conflicts=[key for key in keys if negative and capsule.get(key) is True],
        basis="Normal teacher-competence stop declared by the worker; no child, standalone or gap outcomes were measured."
        if negative
        else "Optional capsule declarations; missing flags remain unknown, not zero performance.",
    )


def _c1_capsule_audit(root: Path, capsule: dict[str, Any], *, verify: bool) -> dict[str, Any]:
    """Expose C1's failed outer closure separately from its saved scientific checks."""
    bindings = capsule.get("files", {})
    receipt_name = "audit/audit.json"
    receipt, receipt_file = _h5_compact_metadata(root, receipt_name, bindings, verify=verify)
    failure, failure_file = _h5_compact_metadata(
        root, "accounting/audit_launcher_failure.json", bindings, verify=verify
    )
    expected = dict(
        schema="walker-c1-audit-launcher-failure-v1",
        overall_audit_closure_passed=False,
        audit_coordinator_failed=True,
        original_audit_exit_exists=False,
        original_child_exit_code=None,
        coordinator_exit_code=1,
    )
    conflicts = [
        key
        for key, value in expected.items()
        if key not in failure or type(failure[key]) is not type(value) or failure[key] != value
    ]
    declared = failure.get("scientific_receipt", {})
    bound = bindings.get(receipt_name, {})
    recorded_digest = bound if isinstance(bound, str) else bound.get("sha256")
    observed_digest = receipt_file.get("sha256") if verify else recorded_digest
    associated = isinstance(declared, dict) and (
        type(declared.get("bytes")) is int
        and declared["bytes"] == receipt_file.get("bytes")
        and declared.get("sha256") is not None
        and declared["sha256"] == observed_digest
    )
    if not associated:
        conflicts.append("scientific_receipt_binding")
    if (root / "audit/exit.json").exists():
        conflicts.append("unexpected_copied_outer_exit_receipt")
    descriptors = [receipt_file, failure_file]
    verified = all(row.get("binding_verified") is True for row in descriptors)
    readable = bool(receipt) and bool(failure) and not any(row.get("error") for row in descriptors)
    return dict(
        status="AUDIT_COORDINATOR_FAILED"
        if readable and not conflicts
        else "C1_AUDIT_CLOSURE_METADATA_INCOMPLETE_OR_CONFLICTING",
        scientific_receipt_checks_passed=receipt.get("full_scientific_audit_passed"),
        overall_audit_closure_passed=failure.get("overall_audit_closure_passed"),
        audit_coordinator_failed=failure.get("audit_coordinator_failed"),
        original_audit_exit_exists=failure.get("original_audit_exit_exists"),
        original_outer_exit_receipt_missing=failure.get("original_audit_exit_exists") is False,
        original_child_exit_code=failure.get("original_child_exit_code"),
        original_child_exit_unknown="original_child_exit_code" in failure
        and failure["original_child_exit_code"] is None,
        coordinator_exit_code=failure.get("coordinator_exit_code"),
        scientific_receipt_binding_matches=associated,
        copied_metadata_verified=verified if verify else None,
        metadata_conflicts=conflicts,
        metadata_files=descriptors,
        inspector_metadata_read_bytes=sum(row.get("read_bytes", 0) for row in descriptors),
        reported_work={
            key: receipt.get(key)
            for key in (
                "prediction_rows",
                "completed_prediction_rows",
                "unknown_partial_prediction_rows",
                "prediction_calls",
                "checkpoint_loads",
                "checkpoint_read_bytes",
                "source_read_bytes_total",
                "wall_seconds",
                "peak_rss_bytes",
                "physics_controls",
                "optimizer_steps",
                "backward_rows",
            )
        },
        verification_scope="Recorded compact scientific receipt and failure companion only; --verify checks copied JSON bytes against capsule hashes. No external raw evidence, checkpoints, predictions, physics or learning are reopened or recomputed. Passing saved scientific checks does not complete the failed outer audit closure. Missing exit metadata is not reconstructed.",
    )


def _consequence_recovery_metadata(
    root: Path, run: Path, capsule: dict[str, Any], *, verify: bool
) -> dict[str, Any]:
    """Describe a new recovery admission without repairing the original audit."""
    prefix = "run/" if root != run else ""
    bindings = capsule.get("files", {})
    names = {
        "result": "result/recovery.json",
        "config": "config.json",
        "manifest": "manifest.json",
        "exit": "exit.json",
        "coordinator_exit": "coordinator_exit.json",
        "finalizer_exit": "finalizer_exit.json",
        "admission": "admission.json",
    }
    values, descriptors = {}, {}
    for key, name in names.items():
        # Association checks need the actual compact JSON hashes even when
        # capsule-wide --verify was not requested.
        value, descriptor = _h5_compact_metadata(root, prefix + name, bindings, verify=True)
        values[key], descriptors[key] = value, descriptor
    result, admission, outer = (values[key] for key in ("result", "admission", "exit"))
    conflicts = []

    def require(document: dict, expected: dict, label: str) -> None:
        for key, value in expected.items():
            if (
                key not in document
                or type(document[key]) is not type(value)
                or document[key] != value
            ):
                conflicts.append(label + "." + key)

    require(
        result,
        dict(
            schema="walker-c1-audit-recovery-result-v1",
            recovery_checks_passed=True,
            admitted_original_scientific_checks=True,
            retained_numerical_certification=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
            original_audit_coordinator_failed=True,
        ),
        "result",
    )
    require(
        admission,
        dict(
            schema="walker-c1-audit-recovery-admission-v1",
            audit_recovery_resolved=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
            retained_numerical_certification=True,
        ),
        "admission",
    )
    require(
        outer,
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
        ),
        "exit",
    )
    coordinator = values["coordinator_exit"]
    require(
        coordinator,
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
        ),
        "coordinator_exit",
    )
    require(
        values["finalizer_exit"],
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
        ),
        "finalizer_exit",
    )
    if (
        type(coordinator.get("pid")) is not int
        or coordinator["pid"] < 1
        or admission.get("coordinator_pid") != coordinator["pid"]
    ):
        conflicts.append("admission.coordinator_pid")
    science = result.get("new_scientific_work", {})
    science_keys = (
        "neural_rows",
        "prediction_calls",
        "checkpoint_tensor_loads",
        "native_controls",
        "physics_substeps",
        "backward_rows",
        "optimizer_steps",
        "resets",
        "restores",
        "mppi_queries",
    )
    require(science if isinstance(science, dict) else {}, dict.fromkeys(science_keys, 0), "work")
    if isinstance(science, dict) and set(science) != set(science_keys):
        conflicts.append("work.counter_schema")
    declared = admission.get("bindings", {})
    for key in ("result", "config", "manifest", "exit", "coordinator_exit"):
        expected = declared.get(key, {}) if isinstance(declared, dict) else {}
        actual = descriptors[key]
        if not (
            isinstance(expected, dict)
            and expected.get("file") == names[key]
            and type(expected.get("bytes")) is int
            and expected["bytes"] == actual.get("bytes")
            and isinstance(expected.get("sha256"), str)
            and expected["sha256"] == actual.get("sha256")
        ):
            conflicts.append("admission.binding." + key)
    if values["manifest"].get("config_sha256") != descriptors["config"].get("sha256"):
        conflicts.append("manifest.config_binding")
    readable = all(values.values()) and not any(row.get("error") for row in descriptors.values())
    copied_verified = bool(capsule) and all(
        row.get("binding_verified") is True for row in descriptors.values()
    )
    if verify and capsule and not copied_verified:
        conflicts.append("copied_metadata_bindings")
    accepted = bool(readable and not conflicts)
    return dict(
        status=(
            "REPORTED_RECOVERY_ACCEPTED_METADATA_VERIFIED"
            if accepted and verify
            else "REPORTED_RECOVERY_ACCEPTED"
            if accepted
            else "RECOVERY_METADATA_INCOMPLETE_OR_CONFLICTING"
        ),
        audit_recovery_resolved=accepted,
        recorded_recovery_resolved=admission.get("audit_recovery_resolved"),
        recovery_checks_passed=result.get("recovery_checks_passed"),
        admitted_original_scientific_checks=result.get("admitted_original_scientific_checks"),
        retained_numerical_certification=result.get("retained_numerical_certification"),
        original_overall_audit_closure_passed=result.get("original_overall_audit_closure_passed"),
        original_child_exit_code=result.get("original_child_exit_code"),
        original_child_exit_unknown="original_child_exit_code" in result
        and result["original_child_exit_code"] is None,
        original_audit_coordinator_failed=result.get("original_audit_coordinator_failed"),
        recorded_recovery_outer_exit_code=outer.get("exit_code"),
        recorded_recovery_coordinator_exit_code=coordinator.get("exit_code"),
        recorded_recovery_finalizer_exit_code=values["finalizer_exit"].get("exit_code"),
        admission_bindings_match=not any(
            item.startswith("admission.binding.") for item in conflicts
        ),
        copied_metadata_verified=copied_verified if verify and capsule else None,
        metadata_conflicts=conflicts,
        metadata_files=list(descriptors.values()),
        inspector_metadata_read_bytes=sum(row.get("read_bytes", 0) for row in descriptors.values()),
        reported_new_scientific_work=science,
        reported_accounting=result.get("accounting", {}),
        verified_file_count=result.get("verified_file_count"),
        verification_scope="Recorded recovery result, admission and clean outer exit associations only; --verify also checks copied capsule metadata or frozen source/input bytes. This inspector neither rehashes external scientific evidence nor reruns inference, physics, fitting or evaluations. Numerical certification is retained from the original completed scientific checks. The original audit closure remains failed and its child exit remains unknown.",
    )


def _certificate_inspection(
    root: Path, run: Path, *, verify: bool, detection_read_bytes: int = 0
) -> dict[str, Any]:
    """Inspect A2's tiny certificates, retaining both failed predecessor histories."""
    prefix = "run/" if root != run else ""
    capsule, capsule_descriptor = _h5_compact_metadata(root, "capsule.json", {}, verify=True)
    bindings = capsule.get("files", {})
    errors: list[str] = []
    conflicts: list[str] = []
    if prefix and (not isinstance(bindings, dict) or not capsule):
        errors.append("A2 capsule metadata is missing or invalid")
        bindings = {}
    if capsule_descriptor.get("error"):
        errors.append(capsule_descriptor["error"])
    names = dict(
        protocol="protocol.json",
        source_manifest="source_manifest.json",
        result="result/certificate_result.json",
        worker_exit="worker_exit.json",
        coordinator_exit="coordinator_exit.json",
        finalizer_exit="finalizer_exit.json",
        candidate="admission_candidate.json",
        admission="admission.json",
    )
    values, descriptors = {}, {}
    for key, name in names.items():
        value, descriptor = _h5_compact_metadata(root, prefix + name, bindings, verify=True)
        values[key], descriptors[key] = value, descriptor
    for row in descriptors.values():
        if row.get("error"):
            errors.append(row["file"] + ": " + row["error"])
    protocol, result, admission = (values[k] for k in ("protocol", "result", "admission"))
    source_manifest = values["source_manifest"]

    def require(document: dict, expected: dict, label: str) -> None:
        for key, value in expected.items():
            if (
                key not in document
                or type(document[key]) is not type(value)
                or document[key] != value
            ):
                conflicts.append(label + "." + key)

    retained = dict(
        certificate_checks_passed=True,
        retained_A1_integrity_certification=True,
        retained_original_numerical_certification=True,
        original_A1_admission_failed=True,
        original_C1_coordinator_failed=True,
        original_C1_child_exit_code=None,
        original_A1_failure_flags_preserved=True,
    )
    require(
        protocol,
        dict(
            schema="walker-c1-certificate-closure-protocol-v1",
            study="C1.0-A2",
            user_authorized=True,
            automatic_followup=False,
        ),
        "protocol",
    )
    require(result, dict(schema="walker-c1-certificate-closure-result-v1", **retained), "result")
    require(
        admission,
        dict(schema="walker-c1-certificate-admission-v1", audit_recovery_resolved=True, **retained),
        "admission",
    )
    require(
        values["candidate"],
        dict(
            schema="walker-c1-certificate-candidate-v1", audit_recovery_resolved=False, **retained
        ),
        "candidate",
    )
    science_keys = (
        "neural_rows",
        "prediction_calls",
        "checkpoint_tensor_loads",
        "native_controls",
        "physics_substeps",
        "backward_rows",
        "optimizer_steps",
        "resets",
        "restores",
        "mppi_queries",
    )
    for key in ("protocol", "result", "candidate", "admission"):
        science = values[key].get("new_scientific_work", {})
        if not isinstance(science, dict) or set(science) != set(science_keys):
            conflicts.append(key + ".new_scientific_work_schema")
        require(
            science if isinstance(science, dict) else {},
            dict.fromkeys(science_keys, 0),
            key + ".work",
        )
    limits = protocol.get("limits", {})
    for key, limit_key, cap in (
        ("worker_exit", "wall_seconds", 30),
        ("coordinator_exit", "coordinator_wall_seconds", 45),
        ("finalizer_exit", "finalization_wall_seconds", 15),
    ):
        outer = values[key]
        require(
            outer,
            dict(
                exit_code=0,
                leader_reaped=True,
                stopping_reason=None,
                process_group_kill_requested=False,
                unknown_partial_work_possible=False,
            ),
            key,
        )
        elapsed, maximum = outer.get("wall_seconds"), limits.get(limit_key)
        if not (
            type(maximum) is int
            and 0 < maximum <= cap
            and type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and 0 <= elapsed <= maximum
        ):
            conflicts.append(key + ".wall_seconds")
    coordinator = values["coordinator_exit"]
    if (
        type(coordinator.get("pid")) is not int
        or coordinator["pid"] < 1
        or type(admission.get("coordinator_pid")) is not int
        or admission.get("coordinator_pid") != coordinator["pid"]
    ):
        conflicts.append("admission.coordinator_pid")

    def associated(binding: Any, key: str) -> bool:
        observed = descriptors[key]
        return (
            isinstance(binding, dict)
            and binding.get("file") == names[key]
            and type(binding.get("bytes")) is int
            and binding["bytes"] == observed.get("bytes")
            and isinstance(binding.get("sha256"), str)
            and binding["sha256"] == observed.get("sha256")
        )

    if not isinstance(admission.get("bindings"), dict) or set(admission["bindings"]) != {
        "result",
        "protocol",
        "source_manifest",
        "worker_exit",
        "coordinator_exit",
    }:
        conflicts.append("admission.binding_schema")
    if not isinstance(values["candidate"].get("bindings"), dict) or set(
        values["candidate"]["bindings"]
    ) != {"result", "protocol", "source_manifest", "worker_exit"}:
        conflicts.append("candidate.binding_schema")
    for key in ("result", "protocol", "source_manifest", "worker_exit", "coordinator_exit"):
        declared = admission.get("bindings", {})
        if not associated(declared.get(key) if isinstance(declared, dict) else None, key):
            conflicts.append("admission.binding." + key)
    if not associated(admission.get("admission_candidate"), "candidate"):
        conflicts.append("admission.candidate_binding")
    for key in ("result", "protocol", "source_manifest", "worker_exit"):
        declared = values["candidate"].get("bindings", {})
        if not associated(declared.get(key) if isinstance(declared, dict) else None, key):
            conflicts.append("candidate.binding." + key)
    canonical_protocol = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if result.get("protocol_sha256") != canonical_protocol:
        conflicts.append("result.protocol_binding")
    if source_manifest.get("protocol_sha256") != descriptors["protocol"].get("sha256"):
        conflicts.append("source_manifest.protocol_binding")
    if source_manifest.get("schema") != "walker-c1-certificate-source-v1":
        conflicts.append("source_manifest.schema")
    source_commit = source_manifest.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(c not in "0123456789abcdef" for c in source_commit)
    ):
        conflicts.append("source_manifest.source_commit")

    def valid_binding(value: Any) -> bool:
        return (
            isinstance(value, dict)
            and type(value.get("bytes")) is int
            and value["bytes"] >= 0
            and isinstance(value.get("sha256"), str)
            and len(value["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in value["sha256"])
        )

    source_bindings = source_manifest.get("files", {})
    source_files = []
    expected_sources = {"observer.py", "closure_checks.py", "watchdog.py"}
    if (
        not isinstance(source_bindings, dict)
        or set(source_bindings) != expected_sources
        or any(not valid_binding(value) for value in source_bindings.values())
    ):
        conflicts.append("source_manifest.files")
    elif (run / "source").is_symlink():
        errors.append("A2 source directory cannot be a symlink")
    else:
        total = 0
        eligible = {}
        for name, binding in source_bindings.items():
            path = run / "source" / name
            if path.is_symlink() or (path.is_file() and path.stat().st_size > 4 * 1024**2):
                errors.append("Invalid or oversized A2 source: " + name)
                continue
            total += path.stat().st_size if path.is_file() else 0
            eligible[name] = binding
        if total > 4 * 1024**2:
            errors.append("A2 source verification exceeds4MiB")
        elif not errors:
            # Only these three tiny source files are hashed, never predecessor payloads.
            source_files = _files(run / "source", eligible, True, errors)
            if any(row.get("verified") is not True for row in source_files):
                conflicts.append("source_manifest.source_bytes")
    files = []
    if prefix and isinstance(bindings, dict):
        eligible, total = {}, 0
        if len(bindings) > 128:
            errors.append("A2 capsule has excessive compact files")
        else:
            for name, binding in bindings.items():
                try:
                    path = _contained_path(root, name)
                    size = path.stat().st_size if path.is_file() else 0
                    if (
                        not valid_binding(binding)
                        or path.is_symlink()
                        or path.suffix not in {".json", ".py", ".txt", ".md"}
                        or size > 4 * 1024**2
                    ):
                        raise ValueError("Invalid or oversized A2 compact file: " + name)
                    total += size
                    eligible[name] = binding
                except (OSError, ValueError) as error:
                    errors.append(str(error))
            if total > 32 * 1024**2:
                errors.append("A2 compact verification exceeds32MiB")
            elif not errors:
                files = _files(root, eligible, verify, errors)
    else:
        files = [
            dict(
                file=names[key],
                available=row["available"],
                bytes=row.get("bytes"),
                sha256=row.get("sha256"),
                verified=bool(values[key]) and not row.get("error"),
            )
            for key, row in descriptors.items()
        ]
    missing = [names[key] for key, row in descriptors.items() if not row["available"]]
    missing += ["source/" + row["file"] for row in source_files if not row["available"]]
    missing += [row["file"] for row in files if not row["available"]]
    mismatches = [
        row["file"]
        for row in files + source_files
        if row["available"] and row.get("verified") is False
    ]
    copied_verified = bool(prefix) and all(
        row.get("binding_verified") is True for row in descriptors.values()
    )
    if verify and prefix and not copied_verified:
        conflicts.append("copied_metadata_bindings")
    readable = all(values.values()) and not any(row.get("error") for row in descriptors.values())
    accepted = bool(readable and not conflicts and not errors and not missing and not mismatches)
    status = (
        "REPORTED_CERTIFICATE_RECOVERY_ACCEPTED_METADATA_VERIFIED"
        if accepted and verify
        else "REPORTED_CERTIFICATE_RECOVERY_ACCEPTED"
        if accepted
        else "CERTIFICATE_RECOVERY_METADATA_INCOMPLETE_OR_CONFLICTING"
    )
    recovery = dict(
        status=status,
        audit_recovery_resolved=accepted,
        recorded_recovery_resolved=admission.get("audit_recovery_resolved"),
        **{key: result.get(key) for key in retained},
        original_C1_child_exit_unknown="original_C1_child_exit_code" in result
        and result["original_C1_child_exit_code"] is None,
        admission_bindings_match=not any("binding" in item for item in conflicts),
        copied_metadata_verified=copied_verified if verify and prefix else None,
        metadata_conflicts=conflicts,
        metadata_files=list(descriptors.values()),
        inspector_metadata_read_bytes=detection_read_bytes
        + capsule_descriptor.get("read_bytes", 0)
        + sum(row.get("read_bytes", 0) for row in descriptors.values()),
        inspector_source_hash_read_bytes=sum(
            row.get("bytes", 0) for row in source_files if "sha256" in row
        ),
        inspector_capsule_hash_read_bytes=sum(
            row.get("bytes", 0) for row in files if row.get("available")
        )
        if verify and prefix
        else 0,
        reported_new_scientific_work=result.get("new_scientific_work", {}),
        reported_accounting=result.get("accounting", {}),
        recorded_recovery_exit_codes={
            key: values[key].get("exit_code")
            for key in ("worker_exit", "coordinator_exit", "finalizer_exit")
        },
        verification_scope="Bounded A2 certificate/admission/clean-exit associations and three tiny source files only; --verify also checks copied compact capsule bytes. A1 integrity/accounting certification is retained as of its recorded verification time, not a fresh hash pass over765 predecessor payloads. Original C1 and A1 launcher failures remain failed; no inference, physics, fitting or evaluations are executed.",
    )
    return dict(
        kind="result_capsule" if prefix else "frozen_run",
        directory=str(root),
        run_id=capsule.get("run_id", root.name),
        config=protocol,
        provenance=dict(
            manifest={key: value for key, value in source_manifest.items() if key != "files"},
            input_manifest=dict(
                scope="Retained point-in-time certificates; no predecessor payload scan"
            ),
            source_file_count=len(source_bindings) if isinstance(source_bindings, dict) else 0,
            source_present=(run / "source").is_dir(),
            launch={},
        ),
        outcomes={
            key: recovery[key]
            for key in (
                "status",
                "audit_recovery_resolved",
                "retained_A1_integrity_certification",
                "retained_original_numerical_certification",
                "original_A1_admission_failed",
                "original_C1_coordinator_failed",
                "original_C1_child_exit_code",
            )
        },
        teacher_outcomes={},
        student_outcomes={},
        teacher_student_gap={},
        gates={},
        costs=dict(
            new_scientific_work=result.get("new_scientific_work", {}),
            certificate_accounting=result.get("accounting", {}),
            limits=limits,
            scope="A2 certificate work only; original and A1 work remains retained lineage cost, not zero.",
        ),
        result_file=names["result"] if result else None,
        files=files,
        source_files=source_files,
        bulk_evidence={},
        missing_evidence=sorted(set(missing)),
        omitted_bulk_evidence=[],
        hash_mismatches=mismatches,
        errors=errors,
        verification_requested=verify,
        verification_scope=recovery["verification_scope"],
        audit_recovery=recovery,
    )


def _visitation_inspection(
    root: Path,
    run: Path,
    capsule: dict[str, Any],
    result: dict[str, Any],
    report: dict[str, Any],
    *,
    verify: bool,
) -> None:
    """Show V's saved phase evidence; worker claims never establish an audit."""
    metadata = {}
    descriptors = []
    prefix = "run/" if capsule else ""
    paths = {
        "preparation": "prepared/preparation_census.json",
        "static_inputs": "prepared/static_receipt.json",
        "static_contexts": "worker/static_gate.json",
        "parent_continuation": "worker/parent_actor_registry.json",
        "parent_replay": "worker/parent_replay_gate.json",
        "reference_positive_control": "worker/reference_positive_control_gate.json",
        "reference_usefulness": "worker/reference_usefulness_screen.json",
        "phase_status": "worker/phase_status.json",
    }
    for key, name in paths.items():
        value, descriptor = _h5_compact_metadata(
            root, prefix + name, capsule.get("files", {}), verify=verify
        )
        metadata[key] = value if descriptor["available"] and not descriptor.get("error") else None
        descriptors.append(descriptor)
        if descriptor.get("error"):
            report["errors"].append(name + ": " + descriptor["error"])
    preparation = metadata["preparation"]
    report["costs"] = dict(
        worker=result.get("costs"),
        worker_wall_seconds=result.get("worker_wall_seconds"),
        preparation=preparation.get("accounting") if preparation else None,
        limits=report["config"].get("limits"),
        scope="Saved preparation and V worker costs are separate declarations. Missing counters remain unavailable; inspection performs no policy inference or physics.",
    )
    report["gates"] = {
        key: metadata[key]
        for key in (
            "static_inputs",
            "static_contexts",
            "parent_continuation",
            "parent_replay",
            "reference_positive_control",
            "reference_usefulness",
        )
    }
    status = result.get("status")
    report["outcomes"] = dict(
        status=status if status is not None else "WORKER_RESULT_UNAVAILABLE",
        recorded_fixed_roster_complete=(status == "COMPLETED_V_FIXED_ROSTER") if status else None,
        learning_launched=result.get("learning_launched"),
        followup_launched=result.get("followup_launched"),
        error=result.get("error"),
        stopped_before=result.get("stopped_before"),
        stopped_during=result.get("stopped_during"),
    )
    report["diagnostics"] = dict(
        authorized_stage=report["config"].get("authorized_stage"),
        conditional_learning_authorized=report["config"].get("conditional_learning_authorized"),
        phase_status=result.get("phase_status") or metadata["phase_status"],
        reference_screen=result.get("reference_screen"),
        metadata_files=descriptors,
        scope="V only: restored closed-loop reference behavior at saved student states. A complete or negative worker screen is not an independently accepted result and does not authorize learning. Missing worker/preparation evidence does not imply zero work.",
    )
    if not capsule:
        prepared, prepared_file = _h5_compact_metadata(
            run, "prepared/prepared_input_manifest.json", {}, verify=False
        )
        report["prepared_inputs"] = _files(
            run / "prepared",
            {row["file"]: row for row in prepared.get("files", [])},
            verify,
            report["errors"],
        )
        if prepared_file.get("error"):
            report["errors"].append(prepared_file["error"])
        for name in (
            "prepared/prepared_input_manifest.json",
            "prepared/static_receipt.json",
            "prepared/preparation_census.json",
        ):
            if not (run / name).is_file():
                report["missing_evidence"].append(name)
        report["missing_evidence"] += [
            "prepared/" + row["file"] for row in report["prepared_inputs"] if not row["available"]
        ]
        report["hash_mismatches"] += [
            "prepared/" + row["file"]
            for row in report["prepared_inputs"]
            if row.get("verified") is False and row["available"]
        ]
    audit_name = "audit/audit.json"
    audit, audit_file = _h5_compact_metadata(
        root, audit_name, capsule.get("files", {}), verify=verify
    )
    recognized = audit.get("schema") == "walker-student-visitation-s1.2-v-audit-v1"
    report["visitation_saved_audit"] = dict(
        available=audit_file["available"],
        metadata_file=audit_file,
        schema_recognized=recognized if audit else None,
        reported_full_scientific_audit_passed=audit.get("full_scientific_audit_passed")
        if recognized
        else None,
        reported_passed=audit.get("passed") if recognized else None,
        closure_reason=audit.get("closure_reason") if recognized else None,
        reported_work=audit.get("work") if recognized else None,
        scope="Optional recorded independent audit receipt only. Capsule hashes bind copied bytes when verified; this inspector neither repeats the audit nor infers outer launcher/admission closure.",
    )
    if recognized:
        report["costs"]["independent_audit"] = audit.get("work")


def inspect_run(path: Path, *, verify: bool = False) -> dict[str, Any]:
    """Describe a full frozen run or a portable result capsule, read only.

    Capsule verification covers exact copied files. Full-run verification covers
    the source snapshot and sealed input files, not an expensive full raw audit.
    Raw costs/outcomes remain recorded claims until independently audited.
    """
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    # A2 deliberately has a tiny protocol/source bundle, not generic config/input files.
    detection_read_bytes = 0
    for candidate in (root / "run", root):
        for name, schema in (
            ("protocol.json", "walker-c1-certificate-closure-protocol-v1"),
            ("source_manifest.json", "walker-c1-certificate-source-v1"),
            ("result/certificate_result.json", "walker-c1-certificate-closure-result-v1"),
        ):
            if (candidate / name).is_file():
                value, descriptor = _h5_compact_metadata(candidate, name, {}, verify=False)
                detection_read_bytes += descriptor.get("read_bytes", 0)
                if value.get("schema") == schema or name == "result/certificate_result.json":
                    return _certificate_inspection(
                        root, candidate, verify=verify, detection_read_bytes=detection_read_bytes
                    )
    errors: list[str] = []
    capsule = _read_json(root / "capsule.json", errors)
    run = root / "run" if capsule else root
    config = _read_json(run / "config.json", errors)
    c1 = config.get("schema") == "walker-student-consequence-c1.0-executable-recipe-v1"
    visitation = config.get("schema") == "walker-student-visitation-v-execution-v1"
    c1_recovery = config.get("schema") == "walker-c1-audit-recovery-protocol-v1"
    result = _read_json(run / "result.json", errors)
    result_file = "result.json"
    if c1 or visitation:
        result = _read_json(run / "worker_result.json", errors)
        result_file = "worker_result.json"
    if c1_recovery:
        result_file = "result/recovery.json"
        result = _read_json(run / result_file, errors)
    h5_attempt_without_result = False
    if (
        config.get("schema") == "walker-teacher-proposal-h5.0-r1"
        and (run / "worker_result.json").is_file()
    ):
        result = _read_json(run / "worker_result.json", errors)
        result_file = "worker_result.json"
    elif config.get("schema") == "walker-teacher-proposal-h5.0-r1" and any(
        (run / name).exists() for name in ("launch_ownership.json", "worker_exit.json", "evidence")
    ):
        h5_attempt_without_result = True
        result = dict(
            result,
            status="launch_attempt_without_worker_result",
            execution_complete=False,
            work=dict(completion_unknown=True, authoritative_prefix="evidence/ledger.jsonl"),
        )
    if not result and config.get("schema") == "walker-student-source-s1-v1":
        for name in ("interruption_summary.json", "fit_result.json"):
            result = _read_json(run / name, errors)
            if result:
                result_file = name
                break
    manifest = _read_json(run / "manifest.json", errors)
    inputs = _read_json(run / "input_manifest.json", errors)
    launch = _read_json(run / "launch_receipt.json", errors)
    if not launch:
        launch = _read_json(run / "launch.json", errors)
    if capsule:
        files = _files(root, capsule.get("files", {}), verify, errors)
        source_files = []
    else:
        bindings = {name: value for name, value in inputs.get("files", {}).items()}
        bindings.setdefault("config.json", manifest.get("config_sha256"))
        if manifest.get("input_sha256"):
            bindings.setdefault("input.pt", manifest["input_sha256"])
        # Absent expected digests still expose presence, without fabricating a seal.
        bindings = {key: value for key, value in bindings.items() if value is not None}
        files = _files(run, bindings, verify, errors)
        source_bindings = {item["file"]: item["sha256"] for item in manifest.get("files", [])}
        source_files = _files(run / "source", source_bindings, verify, errors)
    required = (
        "config.json",
        result_file if c1 or c1_recovery or visitation else "result.json",
        "manifest.json",
        "input_manifest.json",
    )
    missing = [name for name in required if not (run / name).is_file()]
    if h5_attempt_without_result:
        missing.append("worker_result.json")
    missing += [item["file"] for item in files if not item["available"]]
    failed = [
        item["file"]
        for item in files + source_files
        if item["available"] and item.get("verified") is False
    ]
    omitted = capsule.get("bulk_evidence", {}).get("omitted", [])
    evidence_names = ("evidence/raw.h5", "evidence/ledger.jsonl")
    if c1_recovery:
        evidence_names = ()
        missing += [
            name
            for name in (
                "exit.json",
                "coordinator_exit.json",
                "finalizer_exit.json",
                "admission.json",
            )
            if not (run / name).is_file()
        ]
    if c1 or visitation:
        evidence_names = tuple(
            "worker/evidence/" + name
            for name in ("arrays.bin", "arrays.index.jsonl", "ledger.jsonl")
        )
    saved_metric_review = config.get("schema") == "walker-trajectory-review-config-v1"
    if saved_metric_review:
        evidence_names = ()
    motion_review = config.get("schema") == "walker-motion-diagnosis-config-v1"
    if motion_review:
        evidence_names = ("evidence/motion.npz", "work.jsonl")
    h5_preparation = config.get("schema") == "walker-teacher-proposal-h5.0-r1"
    if h5_preparation:
        evidence_names = (
            ("evidence/arrays.bin", "evidence/arrays.index.jsonl", "evidence/ledger.jsonl")
            if result_file == "worker_result.json" or h5_attempt_without_result
            else ()
        )
    if config.get("schema") == "walker-student-source-s1-v1":
        evidence_names = tuple(
            f"evidence/{phase}/{name}"
            for phase in ("fit", "eval")
            for name in ("raw.h5", "ledger.jsonl")
        )
    segmented = config.get("schema") == "walker-student-source-s1.1-v1"
    if segmented:
        evidence = []
        descriptors = result.get("segments", [])
        if result.get("setup"):
            descriptors = [result["setup"], *descriptors]
        for descriptor in descriptors:
            try:
                segment = descriptor["segment"]
                _contained_path(run, segment)
                evidence.extend(
                    f"{segment}/{name}" for name in ("raw.h5", "ledger.jsonl", "receipt.json")
                )
            except (KeyError, TypeError, ValueError) as error:
                errors.append("Invalid declared S1.1 segment: " + str(error))
        partial = result.get("uncommitted_partial", {}).get("directory")
        if partial:
            try:
                _contained_path(run, partial)
                evidence.extend(f"{partial}/{name}" for name in ("raw.h5", "ledger.jsonl"))
            except (TypeError, ValueError) as error:
                errors.append("Invalid partial S1.1 segment: " + str(error))
        evidence_names = tuple(dict.fromkeys(evidence))
    if not capsule:
        missing += [f"source/{item['file']}" for item in source_files if not item["available"]]
        missing += [name for name in evidence_names if not (run / name).is_file()]
    student_outcomes = result.get("cases", {})
    if c1:
        student_outcomes = result.get("student_outcomes", {})
    if "evaluations" in result:
        student_outcomes = {
            cohort: {name: record["summary"] for name, record in panel.items()}
            for cohort, panel in result["evaluations"].items()
        }
    summary_keys = (
        "schema",
        "execution_complete",
        "passed",
        "status",
        "error",
        "claim",
        "hypothesis",
        "mechanism_tested",
        "loop_closed",
        "scientific_loop_pass",
        "standalone_qualified",
        "guidance_promoted",
        "feedback_rounds_completed",
        "confirmed_benefit",
        "incremental_gain_confirmed",
        "feature_benefit_confirmed",
        "next_round_eligible",
        "comparison_complete",
        "audit_complete",
        "integrity_status",
        "preparation_complete",
        "launch_blockers",
        "useful_learning_prerequisite_satisfied",
    )
    report = {
        "directory": str(root),
        "kind": "result_capsule" if capsule else "frozen_run",
        "run_id": capsule.get("run_id", root.name),
        "config": config,
        "provenance": {
            "manifest": {key: value for key, value in manifest.items() if key != "files"},
            "input_manifest": {key: value for key, value in inputs.items() if key != "files"},
            "source_file_count": len(manifest.get("files", [])),
            "source_present": (run / "source").is_dir(),
            "launch": launch,
        },
        "outcomes": {key: result[key] for key in summary_keys if key in result},
        "teacher_outcomes": _summaries(
            result.get("teacher_summaries", result.get("teacher_outcomes", {}))
        ),
        "student_outcomes": _summaries(student_outcomes),
        "teacher_student_gap": result.get(
            "paired_teacher_student_gap", result.get("teacher_student_gap", {})
        ),
        "gates": result.get("gates")
        or {
            key: result[key]
            for key in ("primary_gate", "primary_feature_benefit", "teacher_admission")
            if key in result
        },
        "costs": {
            "work": result.get("work", {}),
            "phase_work": result.get("phase_work", {}),
            "inherited_work": result.get("inherited_work", {}),
            "wall_seconds": result.get("wall_seconds"),
            "accounting": result.get("accounting", {}),
        },
        "result_file": result_file if result else None,
        "files": files,
        "source_files": source_files,
        "policy_hashes": result.get("policy_hashes", {}),
        "bulk_evidence": capsule.get("bulk_evidence", {})
        if capsule
        else {
            name: {
                "available": (run / name).is_file(),
                "bytes": (run / name).stat().st_size if (run / name).is_file() else None,
            }
            for name in evidence_names
        },
        "missing_evidence": sorted(set(missing)),
        "omitted_bulk_evidence": omitted,
        "hash_mismatches": failed,
        "errors": errors,
        "verification_requested": verify,
        "verification_scope": "copied capsule bytes"
        if capsule
        else "source snapshot and sealed inputs",
    }
    if capsule and h5_preparation:
        report["h5_compact_audit"] = _h5_capsule_audit(root, capsule, verify=verify)
        report["h5_measurement_scope"] = _h5_measurement_scope(capsule, result)
        report["costs"]["independent_audit"] = {
            "work": report["h5_compact_audit"]["reported_work"],
            "outer_closure": report["h5_compact_audit"]["reported_outer_closure"],
            "scope": "Reported independent audit work, separate from worker work. Inspection verifies compact copied metadata only; missing counters remain unavailable.",
        }
    if c1:
        report["costs"] = dict(
            worker=result.get("costs", {}),
            worker_wall_seconds=result.get("worker_wall_seconds"),
            limits=config.get("limits", {}),
        )
        report["diagnostics"] = dict(
            replay_gate=result.get("replay_gate", {}),
            contrast_gate=result.get("contrast_gate", {}),
            learning_measured=result.get("learning_measured", False),
            standalone_measured=result.get("standalone_measured", False),
            scope="C1 fixed-suffix consequence-weighted fitting; walking and fitting are separate. Inspection does not independently audit science.",
        )
        if capsule and (
            "accounting/audit_launcher_failure.json" in capsule.get("files", {})
            or (root / "accounting/audit_launcher_failure.json").is_file()
        ):
            report["c1_compact_audit"] = _c1_capsule_audit(root, capsule, verify=verify)
            report["costs"]["independent_audit"] = dict(
                work=report["c1_compact_audit"]["reported_work"],
                overall_audit_closure_passed=report["c1_compact_audit"][
                    "overall_audit_closure_passed"
                ],
                scope="Saved audit counters, separate from worker work; inspector metadata reads are additional. The original outer audit closure failed; these counters are declarations, not a new audit.",
            )
        if not capsule:
            prepared = _read_json(run / "prepared/prepared_input_manifest.json", errors)
            bindings = {row["file"]: row for row in prepared.get("files", [])}
            report["prepared_inputs"] = _files(run / "prepared", bindings, verify, errors)
            report["missing_evidence"] += [
                "prepared/" + row["file"]
                for row in report["prepared_inputs"]
                if not row["available"]
            ]
            report["hash_mismatches"] += [
                "prepared/" + row["file"]
                for row in report["prepared_inputs"]
                if row.get("verified") is False and row["available"]
            ]
    if visitation:
        _visitation_inspection(root, run, capsule, result, report, verify=verify)
    if c1_recovery:
        recovery = _consequence_recovery_metadata(root, run, capsule, verify=verify)
        if verify and (report["hash_mismatches"] or report["errors"]):
            recovery["audit_recovery_resolved"] = False
            recovery["status"] = "RECOVERY_METADATA_INCOMPLETE_OR_CONFLICTING"
            recovery["metadata_conflicts"].append("source_or_capsule_verification")
        report["audit_recovery"] = recovery
        report["outcomes"] = {
            key: recovery[key]
            for key in (
                "status",
                "audit_recovery_resolved",
                "admitted_original_scientific_checks",
                "retained_numerical_certification",
                "original_overall_audit_closure_passed",
                "original_child_exit_code",
            )
        }
        report["costs"] = dict(
            new_scientific_work=result.get("new_scientific_work", {}),
            recovery_accounting=result.get("accounting", {}),
            limits=config.get("limits", {}),
            scope="Recovery work only. Previously paid scientific checks are retained, not repeated or charged as zero lineage cost.",
        )
    if result.get("schema") == "walker-transfer-diagnosis-result-v1":
        d1, d2 = result.get("d1", {}), result.get("d2", {})
        report["diagnostics"] = {
            "support_queries": d1.get("query_count"),
            "support_threshold_q95": d1.get("calibration", {}).get("threshold_q95"),
            "coverage_traces": len(d1.get("coverage_records", [])),
            "baseline_replays": len(d2.get("baselines", [])),
            "baseline_gate_passed": d2.get("baseline_gate_passed", False),
            "substituted_branches": len(d2.get("branches", [])),
            "scope": "Single-action substitution under a fixed open-loop teacher suffix; no learning or recovery claim.",
        }
    if saved_metric_review:
        report["diagnostics"] = {
            "reference": result.get("reference", {}),
            "analysis_counts": result.get("analysis_counts", {}),
            "groups": result.get("groups", []),
            "worker_exit_available": (run / "worker_exit.json").is_file(),
            "scope": "Saved fixed-reference action agreement; no new predictions, physics, fitting or optimization feedback. Ordered bins retain actual observed lengths. Inspection does not establish an independent audit pass.",
        }
    if motion_review:
        report["diagnostics"] = {
            "counts": result.get("counts", {}),
            "motion_archive": result.get("motion_archive", {}),
            "terminal_availability": {
                row["id"]: row.get("terminal_state_available")
                for row in result.get("trajectories", [])
            },
            "scope": "Saved physical state/action sequences and one rigid ordered-segment probe. No inference, simulation, learning or optimization feedback. Missing endpoints remain missing; this inspector does not establish an independent audit pass.",
        }
    if segmented:
        report["provenance"]["inherited_fit"] = result.get("inherited_fit", {})
        report["outcomes"]["scope"] = "fresh_evaluation_only_same_original_training_streams"
        report["segmented_evidence"] = {
            "planned_episodes": config.get("budgets", {}).get("episodes"),
            "declared_episode_segments": len(result.get("segments", [])),
            "declared_load_segment": bool(result.get("setup")),
            "declared_partial_segment": result.get("uncommitted_partial", {}).get("directory"),
            "worker_exit_available": (run / "worker_exit.json").is_file(),
            "owner_exit_available": (run / "owner_exit.json").is_file(),
            "scope": "Saved declarations and file availability only; closed bytes, reset pairing, cumulative work and distinct off-host ACKs require the independent audit. Original S1 native attempts are not pooled.",
        }
    if result_file == "fit_result.json":
        report["outcomes"]["scope"] = "fit_phase_only"
        report["outcomes"]["comparison_complete"] = False
    elif result_file == "interruption_summary.json":
        report["outcomes"]["scope"] = "interrupted_prefix_only"
        report["outcomes"]["comparison_complete"] = False
        audit = _read_json(run / "receipt.json", errors)
        report["outcomes"]["audit_passed"] = audit.get("passed")
        report["provenance"]["interruption_audit"] = audit
    return report


def format_inspection(report: dict[str, Any]) -> str:
    """Render compact metadata for a person; JSON retains all input details."""
    lines = [f"{report['kind']}: {report['run_id']}", report["directory"]]
    outcomes = report["outcomes"]
    lines.append(
        f"Outcome (recorded): {outcomes.get('status', 'unreported')}; error={outcomes.get('error')}"
    )
    lines.append(f"Config schema: {report['config'].get('schema', 'missing')}")
    if report.get("result_file") == "fit_result.json":
        lines.append("Result scope: fit phase only; standalone comparison is incomplete")
    elif report.get("result_file") == "interruption_summary.json":
        lines.append(
            "Result scope: interrupted prefix only; full comparison and cost census incomplete"
        )
    elif "segmented_evidence" in report:
        lines.append(
            "Result scope: fresh S1.1 evaluation only; original fitting and costs remain inherited"
        )
    provenance = report["provenance"]
    lines.append(
        f"Frozen source: {provenance['source_file_count']} recorded files; present={provenance['source_present']}"
    )
    for label in ("teacher_outcomes", "student_outcomes", "teacher_student_gap", "gates", "costs"):
        lines.append(f"{label}: {json.dumps(report[label], sort_keys=True)}")
    for label in (
        "h5_compact_audit",
        "h5_measurement_scope",
        "c1_compact_audit",
        "audit_recovery",
        "visitation_saved_audit",
    ):
        if label in report:
            lines.append(f"{label}: {json.dumps(report[label], sort_keys=True)}")
    if "diagnostics" in report:
        lines.append(f"diagnostics: {json.dumps(report['diagnostics'], sort_keys=True)}")
    if "segmented_evidence" in report:
        lines.append(
            f"segmented_evidence: {json.dumps(report['segmented_evidence'], sort_keys=True)}"
        )
        lines.append(f"inherited_fit: {json.dumps(provenance['inherited_fit'], sort_keys=True)}")
    lines.append(f"Inputs: {json.dumps(report['provenance']['input_manifest'], sort_keys=True)}")
    for item in report["files"]:
        state = "available" if item["available"] else "missing"
        if "verified" in item:
            state += "; hash OK" if item["verified"] else "; HASH MISMATCH"
        lines.append(f"  {item['file']}: {state}, bytes={item['bytes']}")
    for label in (
        "bulk_evidence",
        "missing_evidence",
        "omitted_bulk_evidence",
        "hash_mismatches",
        "errors",
    ):
        lines.append(f"{label}: {json.dumps(report[label], sort_keys=True)}")
    lines.append(
        f"Verification scope: {report['verification_scope']}; requested={report['verification_requested']}"
    )
    return "\n".join(lines)
