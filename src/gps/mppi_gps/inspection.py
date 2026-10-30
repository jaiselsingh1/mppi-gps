"""Read saved run metadata without loading a policy or simulator.

Inspection is deliberately separate from execution and auditing. It reports what
is present, including missing bulk evidence; it does not establish scientific
validity. Hash verification reads files but never repairs or rewrites them.
"""

from __future__ import annotations

import hashlib
import json
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
    return {key: _summaries(item) for key, item in value.items() if isinstance(item, dict)}


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
    missing += [item["file"] for item in files if not item["available"]]
    failed = [item["file"] for item in files + source_files if item.get("verified") is False]
    omitted = capsule.get("bulk_evidence", {}).get("omitted", [])
    if not capsule:
        missing += [f"source/{item['file']}" for item in source_files if not item["available"]]
        missing += [
            name
            for name in ("evidence/raw.h5", "evidence/ledger.jsonl")
            if not (run / name).is_file()
        ]
    summary_keys = (
        "schema",
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
    )
    return {
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
        "teacher_outcomes": _summaries(result.get("teacher_summaries", {})),
        "student_outcomes": _summaries(result.get("cases", {})),
        "teacher_student_gap": result.get(
            "paired_teacher_student_gap", result.get("teacher_student_gap", {})
        ),
        "gates": {
            key: result[key]
            for key in ("primary_gate", "primary_feature_benefit", "teacher_admission")
            if key in result
        },
        "costs": {"work": result.get("work", {}), "wall_seconds": result.get("wall_seconds")},
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
            for name in ("evidence/raw.h5", "evidence/ledger.jsonl")
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


def format_inspection(report: dict[str, Any]) -> str:
    """Render compact metadata for a person; JSON retains all input details."""
    lines = [f"{report['kind']}: {report['run_id']}", report["directory"]]
    outcomes = report["outcomes"]
    lines.append(
        f"Outcome (recorded): {outcomes.get('status', 'unreported')}; error={outcomes.get('error')}"
    )
    lines.append(f"Config schema: {report['config'].get('schema', 'missing')}")
    provenance = report["provenance"]
    lines.append(
        f"Frozen source: {provenance['source_file_count']} recorded files; present={provenance['source_present']}"
    )
    for label in ("teacher_outcomes", "student_outcomes", "teacher_student_gap", "gates", "costs"):
        lines.append(f"{label}: {json.dumps(report[label], sort_keys=True)}")
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
