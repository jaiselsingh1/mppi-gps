import hashlib
import json
from pathlib import Path

import pytest

from src.gps.mppi_gps.inspection import format_inspection, inspect_run


@pytest.mark.parametrize("verify", [False, True])
def test_closed_similarity_metadata_exposes_outcomes_without_opening_cases(
    tmp_path, monkeypatch, verify
):
    (tmp_path / "worker").mkdir()
    for name, value in {
        "config.json": {
            "schema": "walker-similarity-j1-execution-v1",
            "limits": {"worker_stage_wall_seconds": 28800},
        },
        "manifest.json": {"files": []},
        "input_manifest.json": {"files": {}},
        "worker_result.json": {
            "status": "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
            "cases": 32,
            "eligible_cases": 32,
            "declared_source_groups": 16,
            "acquired_source_groups": 8,
            "objective_passed": False,
            "protected_test_revealed": False,
            "costs": {"completed": {"optimizer_steps": 64000}},
        },
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    # Opening partial/future case labels would be a bug: it is invalid JSON.
    (tmp_path / "worker/cases.json").write_text("DO NOT READ PROTECTED LABELS")
    forbid_payload_reads(monkeypatch)
    result = inspect_run(tmp_path, verify=verify)
    assert result["result_file"] == "worker_result.json"
    assert result["student_outcomes"]["acquired_source_groups"] == 8
    assert result["student_outcomes"]["declared_source_groups"] == 16
    assert result["gates"]["objective_passed"] is False
    assert result["gates"]["protected_test_revealed"] is False
    assert result["costs"]["worker"]["completed"]["optimizer_steps"] == 64000
    assert "fixed-update" in format_inspection(result)
    assert "result.json" not in result["missing_evidence"]


def collector_capsule(root):
    """Collector-shaped synthetic bytes, without any scientific trace or labels."""
    files = {}

    def save(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(value).encode()
        path.write_bytes(payload)
        files[name] = dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(), encoding="identity"
        )
        return files[name]["sha256"]

    config_hash = save(
        "run/config.json",
        dict(
            schema="walker-similarity-j1-execution-v1", limits=dict(worker_stage_wall_seconds=28800)
        ),
    )
    source_hash = save("run/manifest.json", dict(files=[], config_sha256=config_hash))
    input_hash = save("run/input_manifest.json", dict(files={}))
    worker_hash = save(
        "run/worker_result.json",
        dict(
            status="CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
            cases=32,
            eligible_cases=32,
            declared_source_groups=16,
            acquired_source_groups=8,
            objective_passed=False,
            protected_test_revealed=False,
            costs=dict(completed=dict(optimizer_steps=64000)),
        ),
    )
    save(
        "audit/result/audit.json",
        dict(
            schema="walker-similarity-j1-audit-v1",
            passed=True,
            full_scientific_audit_passed=True,
            objective_passed=False,
            config_sha256=config_hash,
            source_manifest_sha256=source_hash,
            input_manifest_sha256=input_hash,
            worker_result_sha256=worker_hash,
            work=dict(source_read_bytes=12345, neural_rows=456),
        ),
    )
    for name in (
        "run/worker/cases.json",
        "run/worker/cases.json.gz",
        "run/worker/fits/case/learning_curve.json.gz",
        "run/worker/predictive_comparison.json.gz",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"INVALID COMPRESSED SENTINEL: MUST NEVER BE OPENED")
        files[name] = dict(
            bytes=path.stat().st_size,
            sha256="0" * 64,
            encoding="gzip_exact_original_bytes",
            original_bytes=987654,
            original_sha256="1" * 64,
        )
    external = dict(
        schema="walker-j1-remote-evidence-availability-v1",
        host="original-host",
        source_location="/remote/run/source",
        source_identities="run/manifest.json",
        files=[
            dict(
                namespace="run",
                file="worker/evidence/arrays.bin",
                location="/remote/run/worker/evidence/arrays.bin",
                available=True,
                retained="remote_only",
                bytes=1000000,
                sha256="2" * 64,
                checksum_source="closed_audit_stream_checks",
                checksum_recomputed_by_collector=False,
                checksum_unavailable_reason=None,
            ),
            dict(
                namespace="run",
                file="worker/evidence/ledger.jsonl",
                location="/remote/run/worker/evidence/ledger.jsonl",
                available=True,
                retained="remote_only",
                bytes=500000,
                sha256=None,
                checksum_source=None,
                checksum_recomputed_by_collector=False,
                checksum_unavailable_reason="No closed whole-file hash binding",
            ),
            dict(
                namespace="artifacts",
                file="optional/uncreated",
                location="/remote/artifacts/optional/uncreated",
                available=False,
                retained="missing_or_uncreated",
                bytes=None,
                sha256=None,
            ),
        ],
    )
    external_path = root / "external_evidence.json"
    external_path.write_text(json.dumps(external))
    capsule = dict(schema="walker-j1-compact-collector-v1", passed=True, files=files)
    (root / "capsule.json").write_text(json.dumps(capsule))
    # This final collector manifest also seals external_evidence.json. Neither
    # case-data hash in the fixture is valid; this inspector intentionally does
    # not substitute for full capsule byte validation.
    manifest_files = dict(files)
    for name in ("external_evidence.json", "capsule.json"):
        payload = (root / name).read_bytes()
        manifest_files[name] = dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    (root / "files_manifest.json").write_text(
        json.dumps(dict(schema="walker-j1-compact-file-manifest-v1", files=manifest_files))
    )
    return files


def forbid_payload_reads(monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name.endswith(".gz") or path.name in {
            "cases.json",
            "arrays.bin",
            "ledger.jsonl",
            "learning_curve.json",
        }:
            pytest.fail("Inspector opened a scientific/label payload: " + str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


@pytest.mark.parametrize("verify", [False, True])
def test_collector_capsule_exposes_external_provenance_without_reading_cases(
    tmp_path, monkeypatch, verify
):
    collector_capsule(tmp_path)
    forbid_payload_reads(monkeypatch)
    result = inspect_run(tmp_path, verify=verify)
    assert result["errors"] == []
    assert result["hash_mismatches"] == []
    assert result["student_outcomes"]["measured_cases"] == 32
    assert result["diagnostics"]["predictive_comparison_available"] is True
    assert result["verification_details"]["mode"] == "metadata_only"
    assert result["verification_details"]["all_capsule_bytes_verified"] is False
    assert "run/worker/cases.json.gz" in result["verification_details"]["skipped_payloads"]
    assert result["similarity_saved_audit"]["associated_recorded_full_scientific_audit"] is True
    assert all(
        value["matches"] for value in result["similarity_saved_audit"]["associations"].values()
    )
    bulk = result["bulk_evidence"]["worker/evidence/arrays.bin"]
    assert bulk["recorded_available_at_source"] is True and bulk["locally_available"] is False
    assert bulk["recorded_sha256"] == "2" * 64 and bulk["verified_locally"] is False
    assert bulk["checksum_source"] == "closed_audit_stream_checks"
    ledger = result["bulk_evidence"]["worker/evidence/ledger.jsonl"]
    assert ledger["recorded_sha256"] is None
    assert ledger["checksum_unavailable_reason"] == "No closed whole-file hash binding"
    assert "worker/evidence/arrays.bin" not in result["missing_evidence"]
    assert "external:artifacts:optional/uncreated" in result["missing_evidence"]
    assert "metadata_only" in format_inspection(result)
    assert "similarity_saved_audit" in format_inspection(result)


def test_saved_audit_mismatched_actual_worker_hash_fails_closed_without_verify(
    tmp_path, monkeypatch
):
    collector_capsule(tmp_path)
    path = tmp_path / "run/worker_result.json"
    value = json.loads(path.read_text())
    value["cases"] = 99
    path.write_text(json.dumps(value))
    forbid_payload_reads(monkeypatch)
    result = inspect_run(tmp_path)
    saved = result["similarity_saved_audit"]
    assert saved["reported_passed"] is True
    assert saved["association_matches"] is False
    assert saved["associated_recorded_full_scientific_audit"] is False
    assert saved["status"] == "saved_audit_association_mismatch"
    assert saved["associations"]["worker_result_sha256"]["matches"] is False


def test_saved_audit_full_pass_flag_is_required_even_if_all_hashes_match(tmp_path, monkeypatch):
    collector_capsule(tmp_path)
    path = tmp_path / "audit/result/audit.json"
    value = json.loads(path.read_text())
    value["full_scientific_audit_passed"] = False
    path.write_text(json.dumps(value))
    forbid_payload_reads(monkeypatch)
    saved = inspect_run(tmp_path)["similarity_saved_audit"]
    assert saved["association_matches"] is True
    assert saved["reported_passed"] is True
    assert saved["associated_recorded_full_scientific_audit"] is False
    assert saved["status"] == "saved_audit_failed_or_incomplete"


def test_external_metadata_checksum_mismatch_is_visible_with_verify(tmp_path, monkeypatch):
    collector_capsule(tmp_path)
    path = tmp_path / "external_evidence.json"
    value = json.loads(path.read_text())
    value["host"] = "changed-host"
    path.write_text(json.dumps(value))
    forbid_payload_reads(monkeypatch)
    result = inspect_run(tmp_path, verify=True)
    assert "external_evidence.json" in result["hash_mismatches"]
    assert result["similarity_saved_audit"]["associated_recorded_full_scientific_audit"] is False


def test_optional_audit_and_external_inventory_are_missing_not_fabricated(tmp_path, monkeypatch):
    collector_capsule(tmp_path)
    (tmp_path / "audit/result/audit.json").unlink()
    (tmp_path / "external_evidence.json").unlink()
    forbid_payload_reads(monkeypatch)
    result = inspect_run(tmp_path)
    saved = result["similarity_saved_audit"]
    assert saved["reported_passed"] is None
    assert saved["associated_recorded_full_scientific_audit"] is False
    assert saved["status"] == "saved_audit_unavailable"
    assert result["external_evidence"]["schema_recognized"] is None
    assert result["external_evidence"]["files"] == []
