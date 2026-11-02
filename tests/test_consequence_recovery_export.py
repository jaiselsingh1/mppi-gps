"""Portable closure certificates are inspected without launching audit workers."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.validate_export import CLOSURE_SCIENCE_KEYS, verify_consequence_recovery_capsules

A1 = "student_consequence_c1_audit_recovery_a1"
A2 = "student_consequence_c1_certificate_closure_a2"


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def binding(path: Path, file: str) -> dict:
    payload = path.read_bytes()
    return dict(file=file, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def closure_pair(tmp_path):
    a1, a2 = (tmp_path / "results" / name for name in (A1, A2))
    for capsule in (a1, a2):
        save(capsule / "capsule.json", {})
    zero = dict.fromkeys(CLOSURE_SCIENCE_KEYS, 0)
    save(a1 / "run/result/verified_files.json", dict(files={str(i): {} for i in range(765)}))
    save(
        a1 / "run/result/recovery.json",
        dict(
            schema="walker-c1-audit-recovery-result-v1",
            recovery_checks_passed=True,
            admitted_original_scientific_checks=True,
            retained_numerical_certification=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
            original_audit_coordinator_failed=True,
            verified_file_count=765,
            files_catalog=binding(a1 / "run/result/verified_files.json", "verified_files.json"),
            new_scientific_work=zero,
        ),
    )
    save(
        a1 / "run/exit.json",
        dict(
            exit_code=0,
            stopping_reason="coordinator_exception",
            leader_reaped=True,
            process_group_kill_requested=True,
            unknown_partial_work_possible=True,
        ),
    )
    save(
        a1 / "operator/postmortem_001/external_failure.json",
        dict(
            schema="walker-c1-a1-external-failure-v1",
            external_tool_exit_code=1,
            original_coordinator_exit_available=False,
            original_candidate_exists=False,
            original_admission_exists=False,
            worker_exit_recorded_zero=True,
            worker_coordinator_exception_flags_preserved=True,
        ),
    )
    anchors = {}
    for name, relative in (
        ("a1_result", "run/result/recovery.json"),
        ("a1_worker_exit", "run/exit.json"),
        ("a1_verified_files", "run/result/verified_files.json"),
    ):
        original = a1 / relative
        anchors[name] = dict(root="a1_run", **binding(original, relative.removeprefix("run/")))
        target = a2 / "inputs" / (name + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
    protocol = dict(anchors=anchors, new_scientific_work=zero)
    save(a2 / "run/protocol.json", protocol)
    save(
        a2 / "run/source_manifest.json",
        dict(
            protocol_sha256=binding(a2 / "run/protocol.json", "protocol.json")["sha256"],
            source_commit="a" * 40,
        ),
    )
    retained = dict(
        certificate_checks_passed=True,
        retained_A1_integrity_certification=True,
        retained_original_numerical_certification=True,
        original_A1_admission_failed=True,
        original_C1_coordinator_failed=True,
        original_C1_child_exit_code=None,
        original_A1_failure_flags_preserved=True,
    )
    result = dict(
        schema="walker-c1-certificate-closure-result-v1",
        **retained,
        protocol_sha256=hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        new_scientific_work=zero,
        verified_certificates=anchors,
        work=dict(checked_certificates=len(anchors)),
    )
    save(a2 / "run/result/certificate_result.json", result)
    for name in ("worker_exit", "coordinator_exit", "finalizer_exit"):
        save(
            a2 / "run" / (name + ".json"),
            dict(
                exit_code=0,
                leader_reaped=True,
                stopping_reason=None,
                process_group_kill_requested=False,
                unknown_partial_work_possible=False,
                wall_seconds=0.1,
            ),
        )
    names = dict(
        result="result/certificate_result.json",
        protocol="protocol.json",
        source_manifest="source_manifest.json",
        worker_exit="worker_exit.json",
        coordinator_exit="coordinator_exit.json",
    )
    bindings = {key: binding(a2 / "run" / relative, relative) for key, relative in names.items()}
    candidate = dict(
        schema="walker-c1-certificate-candidate-v1",
        audit_recovery_resolved=False,
        **retained,
        new_scientific_work=zero,
        bindings={key: value for key, value in bindings.items() if key != "coordinator_exit"},
    )
    save(a2 / "run/admission_candidate.json", candidate)
    save(
        a2 / "run/admission.json",
        dict(
            schema="walker-c1-certificate-admission-v1",
            audit_recovery_resolved=True,
            **retained,
            new_scientific_work=zero,
            bindings=bindings,
            admission_candidate=binding(
                a2 / "run/admission_candidate.json", "admission_candidate.json"
            ),
        ),
    )
    save(
        a2 / "operator/launch_001/receipt.json",
        dict(
            audit_recovery_resolved=True,
            original_C1_failure_retained=True,
            original_A1_admission_remains_failed=True,
            admission_sha256=binding(a2 / "run/admission.json", "admission.json")["sha256"],
            source_commit="a" * 40,
            certificate_result=result,
            new_scientific_work=zero,
            stage_wall_seconds=0.4,
        ),
    )
    views = {
        A1: dict(
            audit_recovery_resolved=False,
            recovery_checks_passed=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
        ),
        A2: dict(
            status="REPORTED_CERTIFICATE_RECOVERY_ACCEPTED_METADATA_VERIFIED",
            audit_recovery_resolved=True,
            **retained,
        ),
    }
    inspections = {
        name: dict(verification_requested=True, errors=[], hash_mismatches=[], audit_recovery=view)
        for name, view in views.items()
    }
    return tmp_path, a1, a2, inspections


def test_absent_optional_closure_capsules_are_skipped(tmp_path):
    verify_consequence_recovery_capsules(tmp_path, {})


def test_separate_clean_closure_preserves_original_failure(closure_pair):
    root, a1, _, views = closure_pair
    before = (a1 / "run/exit.json").read_bytes()
    verify_consequence_recovery_capsules(root, views)
    assert (a1 / "run/exit.json").read_bytes() == before
    assert not (a1 / "run/admission.json").exists()


@pytest.mark.parametrize("field", ["admission_sha256", "source_commit", "certificate_result"])
def test_external_receipt_must_bind_exact_admission_and_result(closure_pair, field):
    root, _, a2, views = closure_pair
    path = a2 / "operator/launch_001/receipt.json"
    receipt = load(path)
    receipt[field] = {} if field == "certificate_result" else "b" * 64
    save(path, receipt)
    with pytest.raises(ValueError, match="external receipt/admission association"):
        verify_consequence_recovery_capsules(root, views)


def test_external_receipt_copied_result_preserves_boolean_types(closure_pair):
    root, _, a2, views = closure_pair
    path = a2 / "operator/launch_001/receipt.json"
    receipt = load(path)
    receipt["certificate_result"]["original_A1_admission_failed"] = 1
    save(path, receipt)
    with pytest.raises(ValueError, match="JSON value or type"):
        verify_consequence_recovery_capsules(root, views)


@pytest.mark.parametrize("field", ["stopping_reason", "unknown_partial_work_possible"])
def test_worker_exit_zero_cannot_erase_a1_closure_failure(closure_pair, field):
    root, a1, _, views = closure_pair
    path = a1 / "run/exit.json"
    receipt = load(path)
    receipt[field] = None if field == "stopping_reason" else False
    save(path, receipt)
    with pytest.raises(ValueError, match="preserved A1 failure"):
        verify_consequence_recovery_capsules(root, views)


def test_historical_a1_cannot_gain_reconstructed_admission(closure_pair):
    root, a1, _, views = closure_pair
    save(a1 / "run/admission.json", dict(audit_recovery_resolved=True))
    with pytest.raises(ValueError, match="Historical A1 missing closure was replaced"):
        verify_consequence_recovery_capsules(root, views)


@pytest.mark.parametrize("failure", ["missing", "nonzero", "unknown", "over_budget"])
def test_a2_requires_clean_observed_finalizer_exit(closure_pair, failure):
    root, _, a2, views = closure_pair
    path = a2 / "run/finalizer_exit.json"
    if failure == "missing":
        path.unlink()
    else:
        receipt = load(path)
        key, value = {
            "nonzero": ("exit_code", 1),
            "unknown": ("unknown_partial_work_possible", True),
            "over_budget": ("wall_seconds", 15.01),
        }[failure]
        receipt[key] = value
        save(path, receipt)
    with pytest.raises(ValueError):
        verify_consequence_recovery_capsules(root, views)


def test_copied_certificate_tampering_is_rejected_without_opening_original_payloads(closure_pair):
    root, _, a2, views = closure_pair
    (a2 / "inputs/a1_worker_exit.json").write_text("{}")
    with pytest.raises(ValueError, match="copied certificate differs"):
        verify_consequence_recovery_capsules(root, views)


def test_zero_science_counter_types_are_exact(closure_pair):
    root, _, a2, views = closure_pair
    path = a2 / "operator/launch_001/receipt.json"
    receipt = load(path)
    receipt["new_scientific_work"]["native_controls"] = False
    save(path, receipt)
    with pytest.raises(ValueError, match="new scientific work"):
        verify_consequence_recovery_capsules(root, views)


def test_export_gate_requires_actual_verified_inspector_output(closure_pair):
    root, _, _, views = closure_pair
    views = copy.deepcopy(views)
    views[A2]["verification_requested"] = False
    with pytest.raises(ValueError, match="actual verified inspector smoke"):
        verify_consequence_recovery_capsules(root, views)


def test_published_capsules_pass_public_read_only_cli_smoke(tmp_path):
    """This invokes inspection only; no worker, source replay or scientist is called."""
    root = Path(__file__).resolve().parents[1]
    if not all((root / "results" / name / "capsule.json").is_file() for name in (A1, A2)):
        pytest.skip("Optional historical closure capsules are absent from this export")
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    views = {}
    for name in (A1, A2):
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "scripts/train_mppi_gps.py",
                "inspect",
                "--run",
                "results/" + name,
                "--verify",
                "--json",
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        views[name] = json.loads(process.stdout)
    verify_consequence_recovery_capsules(root, views)
