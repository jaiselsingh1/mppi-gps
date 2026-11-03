"""Validate a clean Git export using the caller's locked CPU environment.

The export must contain no .git, runs or artifacts directory. All rendering uses
saved capsule data, and all generated files go to a new temporary directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def verify_report_outputs(report: Path, bindings: dict[str, dict]) -> None:
    """Resolve host path aliases while rejecting files outside the report."""
    report = report.resolve()
    for name, binding in bindings.items():
        path = (report / name).resolve()
        if (
            not path.is_relative_to(report)
            or not path.is_file()
            or path.stat().st_size != binding["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
        ):
            raise ValueError(f"Rendered trajectory-review binding differs: {path}")


def verify_teacher_proposal_report(
    root: Path, capsule: Path, report: Path, *, diagnostics: bool
) -> None:
    """Verify saved H5 JSON, table and figure bindings without raw evidence."""
    receipt = json.loads((report / "receipt.json").read_text())
    if any(
        receipt[key] != 0
        for key in (
            "network_forwards",
            "physics_controls",
            "optimizer_steps",
            "raw_arrays_decoded",
            "checkpoints_loaded",
        )
    ):
        raise ValueError("Saved H5 report must perform zero scientific calls")
    run = capsule / "run"
    result_name = "worker_result.json" if (run / "worker_result.json").is_file() else "result.json"
    expected_inputs = {
        name: run / name for name in ("config.json", result_name, "call_census.json")
    }
    expected_inputs.update(
        {str(path.relative_to(run)): path for path in sorted((run / "learning").glob("*.json"))}
    )
    if diagnostics:
        expected_inputs["saved_diagnostics"] = capsule / "diagnostics.json"
    if set(receipt["inputs"]) != set(expected_inputs):
        raise ValueError("Rendered H5 input roster differs")
    for name, path in expected_inputs.items():
        binding = receipt["inputs"][name]
        if (
            path.stat().st_size != binding["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
        ):
            raise ValueError("Rendered H5 input binding differs: " + name)
    for name, key in (("config.json", "config_sha256"), (result_name, "source_result_sha256")):
        if receipt[key] != receipt["inputs"][name]["sha256"]:
            raise ValueError("Rendered H5 receipt input differs: " + name)
    source = root / "src/gps/mppi_gps/reporting/teacher_proposal.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != receipt["source_sha256"]:
        raise ValueError("Rendered H5 source differs")
    if receipt["source_read_bytes"] != sum(row["bytes"] for row in receipt["inputs"].values()):
        raise ValueError("Rendered H5 input-read census differs")
    required_tables = {
        "full_data_metrics.csv",
        "teacher_action_agreement.csv",
        "episode_outcomes.csv",
        "teacher_outcomes.csv",
        "student_outcomes.csv",
        "paired_gap.csv",
        "learning_curves.csv",
    }
    tables = set(receipt["table_rows"])
    if not required_tables <= tables or tables - required_tables - {"mechanism_summary.csv"}:
        raise ValueError("Rendered H5 table roster differs")
    if not (tables | {"report.md"}) <= set(receipt["outputs"]):
        raise ValueError("Rendered H5 tables or report lack output bindings")
    verify_report_outputs(report, receipt["outputs"])
    for name, expected_rows in receipt["table_rows"].items():
        with (report / name).open(newline="") as stream:
            actual_rows = sum(1 for _ in csv.DictReader(stream))
        if actual_rows != expected_rows:
            raise ValueError("Rendered H5 table row count differs: " + name)
    expected_curve_rows = sum(
        len(json.loads(path.read_text())["learning_curve"])
        for name, path in expected_inputs.items()
        if name.startswith("learning/")
    )
    if receipt["table_rows"]["learning_curves.csv"] != expected_curve_rows:
        raise ValueError("Rendered H5 learning-point roster differs")
    for suffix in ("png", "pdf"):
        name = "learning_curves." + suffix
        if expected_curve_rows and (
            name not in receipt["outputs"] or (report / name).stat().st_size == 0
        ):
            raise ValueError("Missing bound H5 learning figure: " + name)
    if diagnostics:
        saved = json.loads((capsule / "diagnostics.json").read_text())
        for name, key in (
            ("source_ledger", "source_ledger_binding"),
            ("audit_receipt", "audit_receipt_binding"),
        ):
            if receipt[key] != saved[name]:
                raise ValueError("Rendered H5 declared external binding differs: " + name)


def verify_student_consequence_report(root: Path, capsule: Path, report: Path) -> None:
    """Check the completed C1 capsule's saved-only report and all 18k points."""
    root, capsule, report = root.resolve(), capsule.resolve(), report.resolve()
    run = capsule / "run"
    metadata = json.loads((capsule / "capsule.json").read_text())
    result = json.loads((run / "worker_result.json").read_text())
    audit_path = capsule / "audit/audit.json"
    audit = json.loads(audit_path.read_text())
    receipt = json.loads((report / "receipt.json").read_text())
    if (
        result.get("status") != "COMPLETED_FIXED_C1_ROSTER"
        or result.get("learning_measured") is not True
        or result.get("standalone_measured") is not True
        or metadata.get("scientific_contrast_completed") is not True
        or metadata.get("full_scientific_audit_passed") is not True
        or audit.get("passed") is not True
        or audit.get("full_scientific_audit_passed") is not True
        or audit.get("scientific_contrast_completed") is not True
    ):
        raise ValueError("C1 clean-export validation requires the completed audited roster")
    if (
        receipt.get("C1_audit_bound") is not True
        or receipt.get("C1_audit_association") != "worker_result_sha256_verified"
        or receipt.get("C1_config_association") != "config_sha256_verified"
    ):
        raise ValueError("Rendered C1 audit association differs")
    if any(
        type(receipt.get(key)) is not int or receipt[key] != 0
        for key in ("new_neural_rows", "new_native_controls", "new_optimizer_steps")
    ):
        raise ValueError("Saved C1 report must perform zero scientific calls")
    if receipt.get("incomplete_JSONL_fragments") != []:
        raise ValueError("Rendered C1 report contains incomplete saved evidence")
    closure_path = capsule / "accounting/audit_launcher_failure.json"
    closure = None
    if receipt.get("overall_audit_closure_passed") is False:
        if (
            not closure_path.is_file()
            or receipt.get("scientific_receipt_checks_passed") is not True
        ):
            raise ValueError("Rendered C1 failed closure lacks its explicit accounting companion")
        closure = json.loads(closure_path.read_text())
        expected_closure = dict(
            schema="walker-c1-audit-launcher-failure-v1",
            overall_audit_closure_passed=False,
            audit_coordinator_failed=True,
            original_audit_exit_exists=False,
            original_child_exit_code=None,
            coordinator_exit_code=1,
        )
        if (
            any(
                key not in closure or type(closure[key]) is not type(value) or closure[key] != value
                for key, value in expected_closure.items()
            )
            or receipt.get("audit_closure_status") != "AUDIT_COORDINATOR_FAILED"
        ):
            raise ValueError("Rendered C1 launcher failure facts differ")
        original = closure.get("scientific_receipt", {})
        if (
            original.get("bytes") != audit_path.stat().st_size
            or original.get("sha256") != hashlib.sha256(audit_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Rendered C1 failure companion binds a different scientific receipt")
        commit = closure.get("frozen_source_commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
        ):
            raise ValueError("Rendered C1 failure companion lacks frozen-source provenance")
        for key in ("traceback", "gnu_time"):
            declared = closure.get(key, {})
            matches = [
                name
                for name, value in metadata.get("files", {}).items()
                if value.get("bytes") == declared.get("bytes")
                and value.get("sha256") == declared.get("sha256")
            ]
            if type(declared.get("bytes")) is not int or declared["bytes"] <= 0 or not matches:
                raise ValueError("Rendered C1 failure evidence is not tracked: " + key)
            for name in matches:
                path = (capsule / name).resolve()
                if (
                    not path.is_relative_to(capsule)
                    or not path.is_file()
                    or path.stat().st_size != declared["bytes"]
                    or hashlib.sha256(path.read_bytes()).hexdigest() != declared["sha256"]
                ):
                    raise ValueError("Rendered C1 tracked failure evidence differs: " + key)
    elif closure_path.is_file():
        raise ValueError("Rendered C1 report did not preserve the failed overall audit closure")

    names = (
        "config.json",
        "manifest.json",
        "worker_result.json",
        "worker_exit.json",
        "input_manifest.json",
        "prerequisites.json",
        "prepared/preparation_receipt.json",
        "worker/replay_gate.json",
        "worker/contrast_gate.json",
        "worker/metrics.json",
        "worker/actor_registry.json",
        "worker/fit_records.json",
        "worker/outcomes.json",
        "worker/evaluations.jsonl",
    )
    expected_inputs = {(run / name).resolve() for name in names if (run / name).is_file()}
    expected_inputs.update(path.resolve() for path in (run / "worker/learning").glob("*/*.json"))
    expected_inputs.add(audit_path.resolve())
    if closure is not None:
        expected_inputs.add(closure_path.resolve())
    actual_inputs = {}
    for name, binding in receipt["inputs"].items():
        path = Path(name)
        path = (path if path.is_absolute() else root / path).resolve()
        if path not in expected_inputs or path in actual_inputs:
            raise ValueError("Rendered C1 input roster differs")
        if (
            path.stat().st_size != binding["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != binding["sha256"]
        ):
            raise ValueError("Rendered C1 input binding differs: " + name)
        actual_inputs[path] = binding
    if set(actual_inputs) != expected_inputs:
        raise ValueError("Rendered C1 input roster differs")
    for name in ("config", "worker_result"):
        binding = actual_inputs[(run / (name + ".json")).resolve()]
        if audit.get(name + "_sha256") != binding["sha256"]:
            raise ValueError("Rendered C1 audit/result binding differs: " + name)
    if receipt.get("saved_JSON_read_bytes") != sum(row["bytes"] for row in actual_inputs.values()):
        raise ValueError("Rendered C1 input-read census differs")
    source = root / "src/gps/mppi_gps/reporting/student_consequence.py"
    if (
        receipt["source_code"]["bytes"] != source.stat().st_size
        or receipt["source_code"]["sha256"] != hashlib.sha256(source.read_bytes()).hexdigest()
    ):
        raise ValueError("Rendered C1 source binding differs")

    records = result["fit_records"]
    aliases = {f"s{s}_{arm}" for s in range(3) for arm in ("uniform", "cost_effect", "shuffled")}
    if set(records) != aliases or metadata.get("preserved_learning_points") != 18000:
        raise ValueError("Saved C1 learning roster differs")
    fields = (
        "update",
        "fit_update",
        "action_mse",
        "weighted_action_mse",
        "proximal_loss",
        "total_loss",
        "unclipped_gradient_norm",
    )
    expected_points = {}
    for alias, record in records.items():
        points = record["fitting_evidence"]["learning_curve"]
        if len(points) != 2000:
            raise ValueError("Saved C1 child curve lacks its fixed 2000 updates")
        for index, point in enumerate(points, 1):
            if point["fit_update"] != index or point["update"] != 2000 + index:
                raise ValueError("Saved C1 child update identity differs")
            expected_points[alias, index] = point

    tables = {}
    for name in ("curves", "outcomes", "metrics"):
        with (report / (name + ".csv")).open(newline="") as stream:
            tables[name] = list(csv.DictReader(stream))
    seen = set()
    for row in tables["curves"]:
        key = row["actor"], int(row["fit_update"])
        if key not in expected_points or key in seen:
            raise ValueError("Rendered C1 curve point roster differs")
        seen.add(key)
        for field in fields:
            saved = expected_points[key].get(field)
            if (saved is None and row[field] != "") or (
                saved is not None and float(row[field]) != saved
            ):
                raise ValueError("Rendered C1 saved curve value differs: " + field)
    if seen != set(expected_points):
        raise ValueError("Rendered C1 curve point roster differs")
    expected_counts = dict(curves=18000, outcomes=36 + 432 + 12, metrics=12 * 3)
    if (
        receipt.get("table_rows") != expected_counts
        or {name: len(rows) for name, rows in tables.items()} != expected_counts
    ):
        raise ValueError("Rendered C1 table row counts differ")
    required = {
        "curves.csv",
        "outcomes.csv",
        "metrics.csv",
        "report.md",
        "costs.json",
        "learning_curves.png",
        "learning_curves.svg",
        "standalone_outcomes.png",
        "standalone_outcomes.svg",
    }
    if not required <= set(receipt["outputs"]):
        raise ValueError("Rendered C1 table or figure lacks an output binding")
    verify_report_outputs(report, receipt["outputs"])
    if any((report / name).stat().st_size == 0 for name in required):
        raise ValueError("Rendered C1 table or figure is empty")
    if receipt.get("output_hash_read_bytes") != sum(
        row["bytes"] for row in receipt["outputs"].values()
    ):
        raise ValueError("Rendered C1 output-read census differs")
    costs = json.loads((report / "costs.json").read_text())
    if costs.get("worker") != result.get("costs") or costs.get("audit") != audit:
        raise ValueError("Rendered C1 cost/audit copy differs")
    if closure is not None and costs.get("audit_launcher_failure") != closure:
        raise ValueError("Rendered C1 closure accounting copy differs")
