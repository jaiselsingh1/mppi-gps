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


def inspect_run(path: Path, *, verify: bool = False) -> dict[str, Any]:
    """Describe a full frozen run or a portable result capsule, read only.

    Capsule verification covers exact copied files. Full-run verification covers
    the source snapshot and sealed input files, not an expensive full raw audit.
    Raw costs/outcomes remain recorded claims until independently audited.
    """
    root = path.resolve()
    if not root.is_dir():
        raise ValueError(f"run directory does not exist: {root}")
    errors: list[str] = []
    capsule = _read_json(root / "capsule.json", errors)
    run = root / "run" if capsule else root
    config = _read_json(run / "config.json", errors)
    result = _read_json(run / "result.json", errors)
    result_file = "result.json"
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
    required = ("config.json", "result.json", "manifest.json", "input_manifest.json")
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
    for label in ("h5_compact_audit", "h5_measurement_scope"):
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
