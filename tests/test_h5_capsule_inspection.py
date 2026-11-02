"""Inspect bounded copied H5 audit declarations without executing their source."""

from __future__ import annotations

import builtins
import hashlib
import json

from src.gps.mppi_gps.inspection import format_inspection, inspect_run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def capsule(tmp_path, *, with_audit=True, outer_failure=False, h5=True):
    root = tmp_path / "capsule"
    config = dict(schema="walker-teacher-proposal-h5.0-r1" if h5 else "synthetic-other-study")
    save(root / "run/config.json", config)
    save(root / "run/manifest.json", dict(files=[]))
    save(root / "run/input_manifest.json", dict(files={}))
    save(root / "run/result.json", dict(status="prepared_unlaunched"))
    save(
        root / "run/worker_result.json",
        dict(
            status="teacher_competence_negative_no_child_fits",
            execution_complete=True,
            comparison_complete=False,
            evaluations={},
            paired_teacher_student_gap={},
            work=dict(completed_neural_rows=999, completed_optimizer_steps=6),
        ),
    )
    if with_audit:
        save(
            root / "audit/result/receipt.json",
            dict(
                schema="walker-h5.0-round-audit-v1",
                scientific_round_audit_complete=True,
                prediction_replay_complete=True,
                audit_neural_rows=1234,
                audit_completed_neural_rows=1234,
                audit_unknown_partial_neural_rows=0,
                audit_prediction_calls=11,
                audit_checkpoint_loads=6,
                source_read_bytes=56789,
                checkpoint_read_bytes=100,
                frozen_input_read_bytes=200,
                peak_rss_bytes=1024,
                wall_seconds=2.0,
                new_native_controls=0,
                new_optimizer_steps=0,
                outcomes=dict(learning_contrast_measured=False, standalone_panels_measured=False),
            ),
        )
        save(
            root / "audit/exit.json",
            dict(
                exit_code=1 if outer_failure else 0,
                stopping_reason="wall_limit" if outer_failure else None,
                unknown_partial_work_possible=outer_failure,
                wall_seconds=2.1,
                observed_peak_rss_bytes=2048,
                retries=0,
            ),
        )
    bindings = {
        str(path.relative_to(root)): dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    save(
        root / "capsule.json",
        dict(
            schema="walker-result-capsule-v1",
            study="teacher_proposal_h5_0_round1",
            files=bindings,
            scientific_round_audit_complete=True,
            integrity_status="declared_by_capsule",
            learning_contrast_measured=False,
            standalone_panels_measured=False,
            bulk_evidence=dict(remote_directory="/never/open/external/raw"),
        ),
    )
    return root


def test_h5_inspection_separates_reported_audit_work_and_unmeasured_outcomes(tmp_path, monkeypatch):
    root = capsule(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("numpy", "torch", "mujoco", "h5py"):
            raise AssertionError("Unexpected execution-stack import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    view = inspect_run(root, verify=True)
    audit = view["h5_compact_audit"]
    assert audit["status"] == "reported_complete_audit_metadata_verified"
    assert audit["copied_metadata_verified"] is True
    assert audit["recorded_outer_audit_closed"] is True
    assert audit["reported_work"]["audit_neural_rows"] == 1234
    assert audit["reported_work"]["audit_prediction_calls"] == 11
    assert audit["reported_work"]["audit_checkpoint_loads"] == 6
    assert audit["reported_work"]["source_read_bytes"] == 56789
    assert audit["reported_work"]["new_native_controls"] == 0
    assert audit["reported_work"]["new_optimizer_steps"] == 0
    assert view["costs"]["work"]["completed_neural_rows"] == 999
    assert view["costs"]["independent_audit"]["work"]["audit_neural_rows"] == 1234
    assert view["student_outcomes"] == view["teacher_student_gap"] == {}
    scope = view["h5_measurement_scope"]
    assert scope["learning_contrast_measured"] is False
    assert scope["standalone_panels_measured"] is False
    assert scope["paired_gap_measured"] is False
    assert set(scope["outcome_availability"].values()) == {"not_measured"}
    assert "No external raw evidence" in audit["verification_scope"]
    assert "h5_compact_audit:" in format_inspection(view)
    assert view["errors"] == view["hash_mismatches"] == []


def test_audit_claim_without_successful_outer_closure_remains_incomplete(tmp_path):
    view = inspect_run(capsule(tmp_path, outer_failure=True), verify=True)
    audit = view["h5_compact_audit"]
    assert audit["status"] == "reported_audit_failed_or_incomplete"
    assert audit["reported_scientific_round_audit_complete"] is True
    assert audit["recorded_outer_audit_closed"] is False
    assert audit["reported_work"]["audit_neural_rows"] == 1234


def test_capsule_assertion_cannot_replace_missing_optional_audit_metadata(tmp_path):
    view = inspect_run(capsule(tmp_path, with_audit=False), verify=True)
    audit = view["h5_compact_audit"]
    assert audit["status"] == "optional_compact_audit_unavailable"
    assert audit["capsule_reported_scientific_audit_complete"] is True
    assert audit["reported_scientific_round_audit_complete"] is None
    assert audit["reported_work"]["audit_neural_rows"] is None
    assert audit["inspector_metadata_read_bytes"] == 0
    assert audit["copied_metadata_verified"] is False
    assert view["errors"] == view["missing_evidence"] == []


def test_changed_copied_audit_bytes_are_not_presented_as_verified(tmp_path):
    root = capsule(tmp_path)
    path = root / "audit/result/receipt.json"
    value = json.loads(path.read_text())
    value["audit_neural_rows"] = 4321
    save(path, value)
    view = inspect_run(root, verify=True)
    assert view["h5_compact_audit"]["status"] == "reported_complete_audit_metadata_unverified"
    assert view["h5_compact_audit"]["copied_metadata_verified"] is False
    assert "audit/result/receipt.json" in view["hash_mismatches"]


def test_without_hash_verification_only_recorded_closure_is_shown(tmp_path):
    view = inspect_run(capsule(tmp_path))
    assert view["h5_compact_audit"]["status"] == "reported_complete_audit"
    assert view["h5_compact_audit"]["copied_metadata_verified"] is None


def test_oversized_optional_audit_json_is_not_decoded(tmp_path):
    root = capsule(tmp_path, with_audit=False)
    path = root / "audit/result/receipt.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b" " * (4 * 1024**2 + 1))
    view = inspect_run(root)
    audit = view["h5_compact_audit"]
    assert audit["status"] == "optional_compact_audit_missing_or_unreadable"
    assert audit["inspector_metadata_read_bytes"] == 0
    assert "4 MiB" in audit["metadata_files"][0]["error"]


def test_other_capsules_keep_their_existing_inspection_fields(tmp_path):
    view = inspect_run(capsule(tmp_path, h5=False), verify=True)
    assert "h5_compact_audit" not in view
    assert "h5_measurement_scope" not in view
    assert "independent_audit" not in view["costs"]
