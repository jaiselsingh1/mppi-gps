"""Recovery admission inspection retains the original failed audit history."""

from __future__ import annotations

import builtins
import hashlib
import json

import pytest

from src.gps.mppi_gps.inspection import format_inspection, inspect_run

SCIENCE_KEYS = (
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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def binding(path, relative):
    payload = path.read_bytes()
    return dict(file=relative, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def seal_capsule(root):
    if root.name == "capsule":
        files = {
            str(path.relative_to(root)): binding(path, str(path.relative_to(root)))
            for path in root.rglob("*")
            if path.is_file() and path.name != "capsule.json"
        }
        save(root / "capsule.json", dict(files=files, run_id="synthetic-recovery"))


def make_run(tmp_path, *, capsule=True):
    root = tmp_path / ("capsule" if capsule else "full-run")
    run = root / "run" if capsule else root
    config = run / "config.json"
    save(config, dict(schema="walker-c1-audit-recovery-protocol-v1", limits=dict(read_bytes=10000)))
    save(run / "manifest.json", dict(files=[], config_sha256=binding(config, "")["sha256"]))
    save(run / "input_manifest.json", dict(files={}))
    save(
        run / "result/recovery.json",
        dict(
            schema="walker-c1-audit-recovery-result-v1",
            recovery_checks_passed=True,
            admitted_original_scientific_checks=True,
            retained_numerical_certification=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
            original_audit_coordinator_failed=True,
            new_scientific_work=dict.fromkeys(SCIENCE_KEYS, 0),
            accounting=dict(read_bytes=100, wall_seconds=0.1),
            verified_file_count=12,
        ),
    )
    save(
        run / "exit.json",
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
        ),
    )
    for filename in ("coordinator_exit.json", "finalizer_exit.json"):
        save(
            run / filename,
            dict(
                exit_code=0,
                leader_reaped=True,
                stopping_reason=None,
                process_group_kill_requested=False,
                unknown_partial_work_possible=False,
                pid=4242,
                wall_seconds=1,
            ),
        )
    save(
        run / "admission.json",
        dict(
            schema="walker-c1-audit-recovery-admission-v1",
            audit_recovery_resolved=True,
            coordinator_pid=4242,
            retained_numerical_certification=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
            bindings={
                name: binding(run / relative, relative)
                for name, relative in (
                    ("result", "result/recovery.json"),
                    ("config", "config.json"),
                    ("manifest", "manifest.json"),
                    ("exit", "exit.json"),
                    ("coordinator_exit", "coordinator_exit.json"),
                )
            },
        ),
    )
    seal_capsule(root)
    return root, run


def change_record(root, run, relative, mutate, *, rebind=True):
    path = run / relative
    value = json.loads(path.read_text())
    mutate(value)
    save(path, value)
    if rebind and relative != "admission.json":
        admission = json.loads((run / "admission.json").read_text())
        key = next(k for k, v in admission["bindings"].items() if v["file"] == relative)
        admission["bindings"][key] = binding(path, relative)
        save(run / "admission.json", admission)
    seal_capsule(root)


@pytest.mark.parametrize("capsule", [False, True])
def test_recovery_inspection_keeps_new_admission_and_original_failure_separate(tmp_path, capsule):
    root, _ = make_run(tmp_path, capsule=capsule)
    view = inspect_run(root, verify=True)
    recovery = view["audit_recovery"]
    assert recovery["status"] == "REPORTED_RECOVERY_ACCEPTED_METADATA_VERIFIED"
    assert recovery["audit_recovery_resolved"] is True
    assert recovery["retained_numerical_certification"] is True
    assert recovery["original_overall_audit_closure_passed"] is False
    assert recovery["original_child_exit_code"] is None
    assert recovery["original_child_exit_unknown"] is True
    assert recovery["original_audit_coordinator_failed"] is True
    assert recovery["recorded_recovery_outer_exit_code"] == 0
    assert view["result_file"] == "result/recovery.json"
    assert view["bulk_evidence"] == {}
    assert view["missing_evidence"] == view["hash_mismatches"] == view["errors"] == []
    assert "audit_recovery:" in format_inspection(view)
    assert "Numerical certification is retained" in recovery["verification_scope"]
    assert view["costs"]["new_scientific_work"] == dict.fromkeys(SCIENCE_KEYS, 0)


def test_default_inspection_does_not_claim_capsule_wide_verification(tmp_path):
    root, _ = make_run(tmp_path)
    recovery = inspect_run(root)["audit_recovery"]
    assert recovery["status"] == "REPORTED_RECOVERY_ACCEPTED"
    assert recovery["copied_metadata_verified"] is None
    assert recovery["admission_bindings_match"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 1),
        ("leader_reaped", False),
        ("stopping_reason", "rss_limit"),
        ("unknown_partial_work_possible", True),
        ("process_group_kill_requested", True),
    ],
)
def test_failed_or_unknown_recovery_exit_cannot_admit_prior_checks(tmp_path, field, value):
    root, run = make_run(tmp_path)
    change_record(root, run, "exit.json", lambda record: record.update({field: value}))
    recovery = inspect_run(root, verify=True)["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is False
    assert "exit." + field in recovery["metadata_conflicts"]


@pytest.mark.parametrize(
    "missing",
    [
        "exit.json",
        "coordinator_exit.json",
        "finalizer_exit.json",
        "admission.json",
        "result/recovery.json",
    ],
)
def test_missing_recovery_metadata_is_not_fabricated(tmp_path, missing):
    root, run = make_run(tmp_path, capsule=False)
    (run / missing).unlink()
    view = inspect_run(root, verify=True)
    assert view["audit_recovery"]["audit_recovery_resolved"] is False
    assert missing in view["missing_evidence"]
    assert not (run / missing).exists()


@pytest.mark.parametrize("key", ["result", "config", "manifest", "exit", "coordinator_exit"])
def test_each_admission_binding_is_checked_even_without_capsule_verify(tmp_path, key):
    root, run = make_run(tmp_path)
    change_record(
        root,
        run,
        "admission.json",
        lambda record: record["bindings"][key].update(sha256="0" * 64),
    )
    recovery = inspect_run(root)["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is False
    assert "admission.binding." + key in recovery["metadata_conflicts"]


@pytest.mark.parametrize(
    "field,value",
    [("original_child_exit_code", 0), ("original_overall_audit_closure_passed", True)],
)
def test_recovery_cannot_rewrite_original_failure(tmp_path, field, value):
    root, run = make_run(tmp_path)
    change_record(root, run, "result/recovery.json", lambda record: record.update({field: value}))
    assert inspect_run(root, verify=True)["audit_recovery"]["audit_recovery_resolved"] is False


@pytest.mark.parametrize(
    "field", ["neural_rows", "native_controls", "optimizer_steps", "checkpoint_tensor_loads"]
)
def test_nonzero_scientific_work_cannot_be_presented_as_zero_work_recovery(tmp_path, field):
    root, run = make_run(tmp_path)
    change_record(
        root,
        run,
        "result/recovery.json",
        lambda record: record["new_scientific_work"].update({field: 1}),
    )
    recovery = inspect_run(root, verify=True)["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is False
    assert "work." + field in recovery["metadata_conflicts"]


def test_recovery_inspection_has_no_execution_stack_imports_or_writes(tmp_path, monkeypatch):
    root, run = make_run(tmp_path)
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"numpy", "torch", "mujoco", "h5py"}:
            raise AssertionError("Unexpected execution-stack import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    view = inspect_run(root, verify=True)
    assert view["audit_recovery"]["audit_recovery_resolved"] is True
    assert before == {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert not (run / "audit/exit.json").exists()


@pytest.mark.parametrize("file", ["coordinator_exit.json", "finalizer_exit.json"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 1),
        ("leader_reaped", False),
        ("stopping_reason", "coordinator_exception"),
        ("unknown_partial_work_possible", True),
    ],
)
def test_external_stage_failure_cannot_be_promoted_by_an_admission_file(
    tmp_path, file, field, value
):
    root, run = make_run(tmp_path)
    change_record(
        root,
        run,
        file,
        lambda record: record.update({field: value}),
        rebind=file != "finalizer_exit.json",
    )
    recovery = inspect_run(root, verify=True)["audit_recovery"]
    assert recovery["audit_recovery_resolved"] is False
    assert file.removesuffix(".json") + "." + field in recovery["metadata_conflicts"]


def test_admission_written_before_finalizer_exit_is_not_yet_accepted(tmp_path):
    root, run = make_run(tmp_path, capsule=False)
    (run / "finalizer_exit.json").unlink()
    recovery = inspect_run(root)["audit_recovery"]
    assert recovery["recorded_recovery_resolved"] is True
    assert recovery["audit_recovery_resolved"] is False
