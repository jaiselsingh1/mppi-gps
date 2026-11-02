"""Inspect saved C1 failure metadata without a new scientific audit."""

from __future__ import annotations

import builtins
import hashlib
import json

from src.gps.mppi_gps.inspection import format_inspection, inspect_run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def capsule(tmp_path, *, failure=True):
    root = tmp_path / "capsule"
    save(
        root / "run/config.json",
        dict(schema="walker-student-consequence-c1.0-executable-recipe-v1"),
    )
    save(root / "run/manifest.json", dict(files=[]))
    save(root / "run/input_manifest.json", dict(files={}))
    save(
        root / "run/worker_result.json",
        dict(status="COMPLETED_FIXED_C1_ROSTER", costs=dict(native_controls=17)),
    )
    receipt = root / "audit/audit.json"
    save(
        receipt,
        dict(
            schema="walker-student-consequence-c1.0-audit-v1",
            full_scientific_audit_passed=True,
            prediction_rows=1234,
            completed_prediction_rows=1234,
            unknown_partial_prediction_rows=0,
            prediction_calls=11,
            checkpoint_loads=12,
            source_read_bytes_total=50000,
            wall_seconds=2.0,
            physics_controls=0,
            optimizer_steps=0,
            backward_rows=0,
        ),
    )
    if failure:
        save(
            root / "accounting/audit_launcher_failure.json",
            dict(
                schema="walker-c1-audit-launcher-failure-v1",
                scientific_receipt=dict(
                    file="audit_001/result/audit.json",
                    bytes=receipt.stat().st_size,
                    sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
                ),
                overall_audit_closure_passed=False,
                audit_coordinator_failed=True,
                original_audit_exit_exists=False,
                original_child_exit_code=None,
                coordinator_exit_code=1,
            ),
        )
    bindings = {
        str(path.relative_to(root)): dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    save(root / "capsule.json", dict(files=bindings, run_id="synthetic-c1"))
    return root


def test_c1_inspection_separates_scientific_checks_and_failed_outer_closure(tmp_path, monkeypatch):
    root = capsule(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("numpy", "torch", "mujoco", "h5py"):
            raise AssertionError("Unexpected execution-stack import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    view = inspect_run(root, verify=True)
    audit = view["c1_compact_audit"]
    assert audit["status"] == "AUDIT_COORDINATOR_FAILED"
    assert audit["scientific_receipt_checks_passed"] is True
    assert audit["overall_audit_closure_passed"] is False
    assert audit["audit_coordinator_failed"] is True
    assert audit["original_outer_exit_receipt_missing"] is True
    assert audit["original_child_exit_code"] is None
    assert audit["original_child_exit_unknown"] is True
    assert audit["scientific_receipt_binding_matches"] is True
    assert audit["copied_metadata_verified"] is True
    assert audit["reported_work"]["prediction_rows"] == 1234
    assert audit["reported_work"]["physics_controls"] == 0
    assert view["costs"]["worker"]["native_controls"] == 17
    assert view["costs"]["independent_audit"]["work"]["prediction_rows"] == 1234
    assert "c1_compact_audit:" in format_inspection(view)
    assert "No external raw evidence" in audit["verification_scope"]
    assert view["hash_mismatches"] == view["errors"] == []
    assert not (root / "audit/exit.json").exists()


def test_c1_inspection_reports_metadata_only_without_hash_verification(tmp_path):
    audit = inspect_run(capsule(tmp_path))["c1_compact_audit"]
    assert audit["scientific_receipt_checks_passed"] is True
    assert audit["overall_audit_closure_passed"] is False
    assert audit["copied_metadata_verified"] is None


def test_changed_scientific_receipt_is_not_a_verified_failure_association(tmp_path):
    root = capsule(tmp_path)
    path = root / "audit/audit.json"
    path.write_text(path.read_text() + " ")
    view = inspect_run(root, verify=True)
    audit = view["c1_compact_audit"]
    assert audit["status"] == "C1_AUDIT_CLOSURE_METADATA_INCOMPLETE_OR_CONFLICTING"
    assert audit["copied_metadata_verified"] is False
    assert audit["scientific_receipt_binding_matches"] is False
    assert "audit/audit.json" in view["hash_mismatches"]


def test_missing_scientific_receipt_stays_unknown_and_cannot_close_audit(tmp_path):
    root = capsule(tmp_path)
    (root / "audit/audit.json").unlink()
    view = inspect_run(root, verify=True)
    audit = view["c1_compact_audit"]
    assert audit["scientific_receipt_checks_passed"] is None
    assert audit["overall_audit_closure_passed"] is False
    assert audit["status"] == "C1_AUDIT_CLOSURE_METADATA_INCOMPLETE_OR_CONFLICTING"
    assert "audit/audit.json" in view["missing_evidence"]


def test_c1_without_failure_companion_retains_existing_inspection(tmp_path):
    view = inspect_run(capsule(tmp_path, failure=False), verify=True)
    assert "c1_compact_audit" not in view
    assert "independent_audit" not in view["costs"]
