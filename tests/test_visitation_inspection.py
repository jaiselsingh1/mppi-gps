"""Read-only S1.2 V inspection never promotes worker claims to an audit."""

from __future__ import annotations

import builtins
import hashlib
import json

from src.gps.mppi_gps.inspection import format_inspection, inspect_run


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path, *, capsule=True, worker=True, prepared=True, audit=False):
    root = tmp_path / ("capsule" if capsule else "run")
    run = root / "run" if capsule else root
    save(
        run / "config.json",
        dict(
            schema="walker-student-visitation-v-execution-v1",
            authorized_stage="V",
            conditional_learning_authorized=False,
            limits=dict(native_controls=224000),
        ),
    )
    save(run / "manifest.json", dict(files=[]))
    save(run / "input_manifest.json", dict(files={}))
    if worker:
        save(
            run / "worker_result.json",
            dict(
                schema="walker-visitation-validity-result-v1",
                status="COMPLETED_V_FIXED_ROSTER",
                learning_launched=False,
                followup_launched=False,
                phase_status=dict(parent_replay="passed", takeovers="complete"),
                costs=dict(native_controls=11427),
            ),
        )
    if prepared:
        save(
            run / "prepared/preparation_census.json",
            dict(passed=True, accounting=dict(source_read_bytes=14000)),
        )
        save(run / "prepared/static_receipt.json", dict(passed=True, parent_episodes=96))
        save(run / "prepared/prepared_input_manifest.json", dict(files=[]))
        save(run / "worker/static_gate.json", dict(passed=True))
        save(
            run / "worker/parent_actor_registry.json",
            dict(s0_parent=dict(continuation_verified_before_constructor=True)),
        )
        save(run / "worker/parent_replay_gate.json", dict(passed=True, saved_falls_included=True))
        save(run / "worker/reference_positive_control_gate.json", dict(passed=True))
        save(run / "worker/reference_usefulness_screen.json", dict(passed=False))
    if audit:
        save(
            root / "audit/audit.json",
            dict(
                schema="walker-student-visitation-s1.2-v-audit-v1",
                passed=True,
                full_scientific_audit_passed=True,
                closure_reason="complete_reference_usefulness_screen",
                work=dict(neural_rows=100),
            ),
        )
    if capsule:
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
                run_id="synthetic-V",
                files=bindings,
                bulk_evidence=dict(omitted=["worker/evidence/arrays.bin"]),
            ),
        )
    return root, run


def test_worker_completion_does_not_establish_audit_or_learning(tmp_path, monkeypatch):
    root, _ = fixture(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"numpy", "torch", "mujoco", "h5py"}:
            raise AssertionError("Unexpected execution import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    view = inspect_run(root, verify=True)
    assert view["result_file"] == "worker_result.json"
    assert view["outcomes"]["recorded_fixed_roster_complete"] is True
    assert view["costs"]["worker"]["native_controls"] == 11427
    assert view["costs"]["preparation"]["source_read_bytes"] == 14000
    assert view["gates"]["reference_usefulness"]["passed"] is False
    assert view["diagnostics"]["conditional_learning_authorized"] is False
    assert view["visitation_saved_audit"]["reported_full_scientific_audit_passed"] is None
    assert "independent_audit" not in view["costs"]
    assert "visitation_saved_audit:" in format_inspection(view)
    assert view["hash_mismatches"] == view["errors"] == []
    assert "result.json" not in view["missing_evidence"]


def test_missing_worker_and_preparation_remain_unknown(tmp_path):
    root, _ = fixture(tmp_path, capsule=False, worker=False, prepared=False)
    view = inspect_run(root, verify=True)
    assert "worker_result.json" in view["missing_evidence"]
    assert "prepared/preparation_census.json" in view["missing_evidence"]
    assert view["costs"]["worker"] is None
    assert view["costs"]["preparation"] is None
    assert view["outcomes"]["recorded_fixed_roster_complete"] is None
    assert view["outcomes"]["learning_launched"] is None
    assert "worker/evidence/arrays.bin" in view["bulk_evidence"]
    assert view["bulk_evidence"]["worker/evidence/arrays.bin"]["available"] is False


def test_stopped_screen_is_reported_separately_from_complete_roster(tmp_path):
    root, run = fixture(tmp_path, capsule=False)
    save(
        run / "worker_result.json",
        dict(
            status="STOPPED_REFERENCE_POSITIVE_CONTROL_FAILED",
            stopped_before="reference_takeovers",
            learning_launched=False,
        ),
    )
    view = inspect_run(root)
    assert view["outcomes"]["recorded_fixed_roster_complete"] is False
    assert view["outcomes"]["stopped_before"] == "reference_takeovers"
    assert view["costs"]["worker"] is None


def test_optional_audit_receipt_is_only_recorded_copied_metadata(tmp_path):
    root, _ = fixture(tmp_path, audit=True)
    view = inspect_run(root, verify=True)
    audit = view["visitation_saved_audit"]
    assert audit["reported_full_scientific_audit_passed"] is True
    assert audit["metadata_file"]["binding_verified"] is True
    assert view["costs"]["independent_audit"] == dict(neural_rows=100)
    assert "outer launcher/admission" in audit["scope"]
    path = root / "audit/audit.json"
    path.write_text(path.read_text() + " ")
    changed = inspect_run(root, verify=True)
    assert changed["visitation_saved_audit"]["metadata_file"]["binding_verified"] is False
    assert "audit/audit.json" in changed["hash_mismatches"]


def test_full_run_prepared_hashes_are_checked_without_loading_checkpoint(tmp_path):
    root, run = fixture(tmp_path, capsule=False)
    path = run / "prepared/parents/s0_parent.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"opaque")
    save(
        run / "prepared/prepared_input_manifest.json",
        dict(
            files=[
                dict(
                    file="parents/s0_parent.pt",
                    bytes=6,
                    sha256=hashlib.sha256(b"opaque").hexdigest(),
                )
            ]
        ),
    )
    assert inspect_run(root, verify=True)["prepared_inputs"][0]["verified"] is True
    path.write_bytes(b"changed")
    assert "prepared/parents/s0_parent.pt" in inspect_run(root, verify=True)["hash_mismatches"]
