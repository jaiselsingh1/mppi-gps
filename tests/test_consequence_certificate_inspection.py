"""Inspect A2 certificates without reopening predecessor scientific evidence."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from src.gps.mppi_gps.inspection import format_inspection, inspect_run

SCIENCE = (
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
FLAGS = dict(
    certificate_checks_passed=True,
    retained_A1_integrity_certification=True,
    retained_original_numerical_certification=True,
    original_A1_admission_failed=True,
    original_C1_coordinator_failed=True,
    original_C1_child_exit_code=None,
    original_A1_failure_flags_preserved=True,
)
NAMES = dict(
    result="result/certificate_result.json",
    protocol="protocol.json",
    source_manifest="source_manifest.json",
    worker_exit="worker_exit.json",
    coordinator_exit="coordinator_exit.json",
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def binding(path, relative):
    payload = path.read_bytes()
    return dict(file=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def seal_capsule(root, run):
    if root != run:
        save(
            root / "capsule.json",
            dict(
                run_id="synthetic-A2",
                files={
                    str(p.relative_to(root)): binding(p, str(p.relative_to(root)))
                    for p in root.rglob("*")
                    if p.is_file() and p.name != "capsule.json"
                },
            ),
        )


def rebind(root, run):
    candidate_path = run / "admission_candidate.json"
    candidate = json.loads(candidate_path.read_text())
    candidate["bindings"] = {
        k: binding(run / name, name) for k, name in NAMES.items() if k != "coordinator_exit"
    }
    save(candidate_path, candidate)
    admission_path = run / "admission.json"
    admission = json.loads(admission_path.read_text())
    admission["bindings"] = {k: binding(run / name, name) for k, name in NAMES.items()}
    admission["admission_candidate"] = binding(candidate_path, "admission_candidate.json")
    save(admission_path, admission)
    seal_capsule(root, run)


def make_run(tmp_path, *, capsule=True):
    root = tmp_path / ("capsule" if capsule else "full")
    run = root / "run" if capsule else root
    protocol = dict(
        schema="walker-c1-certificate-closure-protocol-v1",
        study="C1.0-A2",
        user_authorized=True,
        automatic_followup=False,
        new_scientific_work=dict.fromkeys(SCIENCE, 0),
        limits=dict(wall_seconds=30, coordinator_wall_seconds=45, finalization_wall_seconds=15),
    )
    save(run / "protocol.json", protocol)
    source_files = {}
    for name in ["observer.py", "closure_checks.py", "watchdog.py"]:
        path = run / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic closed source: " + name + "\n")
        source_files[name] = {k: v for k, v in binding(path, name).items() if k != "file"}
    save(
        run / "source_manifest.json",
        dict(
            schema="walker-c1-certificate-source-v1",
            protocol_sha256=binding(run / "protocol.json", "")["sha256"],
            source_commit="a" * 40,
            files=source_files,
        ),
    )
    save(
        run / NAMES["result"],
        dict(
            schema="walker-c1-certificate-closure-result-v1",
            **FLAGS,
            protocol_sha256=hashlib.sha256(
                json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            new_scientific_work=dict.fromkeys(SCIENCE, 0),
            accounting=dict(read_bytes=100, wall_seconds=0.1),
        ),
    )
    for name, pid in [
        ("worker_exit.json", 4241),
        ("coordinator_exit.json", 4242),
        ("finalizer_exit.json", 4243),
    ]:
        save(
            run / name,
            dict(
                exit_code=0,
                leader_reaped=True,
                stopping_reason=None,
                process_group_kill_requested=False,
                unknown_partial_work_possible=False,
                wall_seconds=0.1,
                pid=pid,
            ),
        )
    save(
        run / "admission_candidate.json",
        dict(
            schema="walker-c1-certificate-candidate-v1",
            audit_recovery_resolved=False,
            **FLAGS,
            new_scientific_work=dict.fromkeys(SCIENCE, 0),
        ),
    )
    save(
        run / "admission.json",
        dict(
            schema="walker-c1-certificate-admission-v1",
            audit_recovery_resolved=True,
            coordinator_pid=4242,
            **FLAGS,
            new_scientific_work=dict.fromkeys(SCIENCE, 0),
        ),
    )
    rebind(root, run)
    return root, run


def change(root, run, name, mutate, *, bind=True):
    path = run / name
    value = json.loads(path.read_text())
    mutate(value)
    save(path, value)
    if bind:
        rebind(root, run)
    else:
        seal_capsule(root, run)


@pytest.mark.parametrize("capsule", [False, True])
def test_a2_inspection_accepts_only_new_closed_certificates(tmp_path, capsule):
    root, _ = make_run(tmp_path, capsule=capsule)
    view = inspect_run(root, verify=True)
    recovery = view["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is True
    assert recovery["status"] == "REPORTED_CERTIFICATE_RECOVERY_ACCEPTED_METADATA_VERIFIED"
    assert recovery["original_A1_admission_failed"] is True
    assert recovery["original_C1_coordinator_failed"] is True
    assert recovery["original_C1_child_exit_code"] is None
    assert recovery["retained_A1_integrity_certification"] is True
    assert recovery["retained_original_numerical_certification"] is True
    assert "not a fresh hash pass" in recovery["verification_scope"]
    assert view["result_file"] == "result/certificate_result.json"
    assert view["missing_evidence"] == view["hash_mismatches"] == view["errors"] == []
    assert view["bulk_evidence"] == {}
    assert view["costs"]["new_scientific_work"] == dict.fromkeys(SCIENCE, 0)
    assert "audit_recovery:" in format_inspection(view)


def test_default_inspection_does_not_claim_capsule_wide_verification(tmp_path):
    root, _ = make_run(tmp_path)
    recovery = inspect_run(root)["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is True
    assert recovery["copied_metadata_verified"] is None
    assert recovery["status"] == "REPORTED_CERTIFICATE_RECOVERY_ACCEPTED"


@pytest.mark.parametrize(
    "name", ["worker_exit.json", "coordinator_exit.json", "finalizer_exit.json"]
)
@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 1),
        ("leader_reaped", False),
        ("stopping_reason", "rss_limit"),
        ("process_group_kill_requested", True),
        ("unknown_partial_work_possible", True),
        ("wall_seconds", -1),
        ("wall_seconds", 46),
        ("wall_seconds", True),
    ],
)
def test_each_clean_process_exit_is_required(tmp_path, name, field, value):
    root, run = make_run(tmp_path)
    change(root, run, name, lambda d: d.update({field: value}))
    assert inspect_run(root, verify=True)["audit_recovery"]["audit_recovery_resolved"] is False


@pytest.mark.parametrize(
    "name", [*NAMES.values(), "finalizer_exit.json", "admission_candidate.json", "admission.json"]
)
def test_missing_certificate_is_not_fabricated(tmp_path, name):
    root, run = make_run(tmp_path, capsule=False)
    (run / name).unlink()
    view = inspect_run(root, verify=True)
    assert view["audit_recovery"]["audit_recovery_resolved"] is False
    assert name in view["missing_evidence"]
    assert not (run / name).exists()


@pytest.mark.parametrize("key", list(NAMES))
def test_every_admission_binding_is_required_even_without_verify(tmp_path, key):
    root, run = make_run(tmp_path)
    change(
        root,
        run,
        "admission.json",
        lambda d: d["bindings"][key].update(sha256="0" * 64),
        bind=False,
    )
    recovery = inspect_run(root)["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is False
    assert "admission.binding." + key in recovery["metadata_conflicts"]


def test_candidate_binding_is_required(tmp_path):
    root, run = make_run(tmp_path)
    change(
        root,
        run,
        "admission.json",
        lambda d: d["admission_candidate"].update(sha256="0" * 64),
        bind=False,
    )
    assert inspect_run(root)["audit_recovery"]["audit_recovery_resolved"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("retained_A1_integrity_certification", False),
        ("retained_original_numerical_certification", False),
        ("original_A1_admission_failed", False),
        ("original_C1_coordinator_failed", False),
        ("original_C1_child_exit_code", 0),
        ("original_A1_failure_flags_preserved", False),
    ],
)
def test_retained_failure_and_certification_flags_cannot_be_rewritten(tmp_path, field, value):
    root, run = make_run(tmp_path)
    change(root, run, NAMES["result"], lambda d: d.update({field: value}))
    assert inspect_run(root, verify=True)["audit_recovery"]["audit_recovery_resolved"] is False


@pytest.mark.parametrize("value", [1, False])
def test_new_scientific_work_is_strict_integer_zero(tmp_path, value):
    root, run = make_run(tmp_path)
    change(root, run, NAMES["result"], lambda d: d["new_scientific_work"].update(neural_rows=value))
    assert inspect_run(root, verify=True)["audit_recovery"]["audit_recovery_resolved"] is False


@pytest.mark.parametrize(
    "name,field",
    [(NAMES["result"], "protocol_sha256"), ("source_manifest.json", "protocol_sha256")],
)
def test_protocol_hash_contracts_are_distinct_and_checked(tmp_path, name, field):
    root, run = make_run(tmp_path)
    change(root, run, name, lambda d: d.update({field: "0" * 64}))
    assert inspect_run(root)["audit_recovery"]["audit_recovery_resolved"] is False


def test_three_frozen_source_files_are_checked_without_generic_config_requirements(tmp_path):
    root, run = make_run(tmp_path, capsule=False)
    assert not (run / "config.json").exists()
    (run / "source/watchdog.py").write_text("# changed\n")
    view = inspect_run(root)
    assert view["audit_recovery"]["audit_recovery_resolved"] is False
    assert "watchdog.py" in view["hash_mismatches"]
    assert "config.json" not in view["missing_evidence"]


def test_bounded_metadata_and_no_scientific_imports_or_raw_reads(tmp_path, monkeypatch):
    root, run = make_run(tmp_path)
    raw = root / "raw.bin"
    raw.write_bytes(b"unopened predecessor payload")
    original_import, original_open = builtins.__import__, Path.open

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "mujoco", "numpy", "h5py", "jax"}:
            raise AssertionError("Scientific stack imported")
        return original_import(name, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        if path.suffix == ".bin":
            raise AssertionError("Raw predecessor evidence opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(Path, "open", guarded_open)
    view = inspect_run(root, verify=True)
    assert view["audit_recovery"]["audit_recovery_resolved"] is True
    (run / "result/certificate_result.json").write_text(" " * (4 * 1024**2 + 1))
    assert inspect_run(root, verify=True)["audit_recovery"]["audit_recovery_resolved"] is False


def test_unrelated_protocol_does_not_replace_generic_run_inspection(tmp_path):
    run = tmp_path / "generic"
    save(run / "protocol.json", dict(schema="unrelated-protocol"))
    save(run / "config.json", dict(schema="generic-config"))
    save(run / "manifest.json", dict(files=[]))
    save(run / "input_manifest.json", dict(files={}))
    save(run / "result.json", dict(status="generic-result"))
    view = inspect_run(run)
    assert view["config"]["schema"] == "generic-config"
    assert "audit_recovery" not in view
