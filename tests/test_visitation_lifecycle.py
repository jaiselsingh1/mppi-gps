"""V authorization, source ownership and closing receipts use synthetic metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.gps.mppi_gps.experiments import student_visitation as study
from src.gps.mppi_gps.provenance import sha

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def clean(pid):
    return dict(
        pid=pid,
        command=["synthetic"],
        wall_seconds=0.1,
        exit_code=0,
        leader_reaped=True,
        stopping_reason=None,
        process_group_kill_requested=False,
        unknown_partial_work_possible=False,
    )


def test_execution_recipe_preserves_the_reviewed_science_and_carry_forward():
    config = json.loads(study.RECIPE.read_text())
    review = json.loads((ROOT / config["review_recipe"]["file"]).read_text())
    study.validate_config(config)
    assert all(
        config[k] == review[k] for k in ("validity", "parents", "reference", "normalization")
    )
    carry = config["combined_carry_forward"]
    assert (
        carry["V_worker_stage_reservation_seconds"] + carry["L_remaining_worker_stage_seconds"]
        == 21600
    )
    assert (
        carry["V_audit_stage_reservation_seconds"] + carry["L_remaining_audit_stage_seconds"]
        == 7200
    )
    assert (
        carry["V_global_output_reservation_bytes"] + carry["L_remaining_global_output_bytes"]
        == 16 * 1024**3
    )
    assert carry["L_authorized"] is False


@pytest.mark.parametrize("change", ["authorize_L", "automatic", "walking_gate", "budget"])
def test_no_authorization_or_scientific_gate_drift(change):
    config = json.loads(study.RECIPE.read_text())
    if change == "authorize_L":
        config["conditional_learning_authorized"] = True
    elif change == "automatic":
        config["automatic_followup"] = True
    elif change == "walking_gate":
        config["validity"]["positive_control_min_walking_each16panel"] = 14
    else:
        config["limits"]["worker_stage_wall_seconds"] += 1
    with pytest.raises(ValueError):
        study.validate_config(config)


def test_bounded_child_ownership_cannot_be_inherited_from_another_launch(tmp_path, monkeypatch):
    save(tmp_path / "config.json", {})
    save(tmp_path / "manifest.json", {})
    owner = dict(
        nonce="owned",
        phase="prepare",
        coordinator_pid=os.getppid(),
        config_sha256=sha(tmp_path / "config.json"),
        source_manifest_sha256=sha(tmp_path / "manifest.json"),
    )
    save(tmp_path / "prepare_ownership.json", owner)
    monkeypatch.setenv("S12V_PHASE_NONCE", "other")
    with pytest.raises(ValueError, match="owner"):
        study._owner(tmp_path, "prepare")
    monkeypatch.setenv("S12V_PHASE_NONCE", "owned")
    assert study._owner(tmp_path, "prepare") == owner
    save(tmp_path / "config.json", {"changed": True})
    with pytest.raises(ValueError, match="owner"):
        study._owner(tmp_path, "prepare")


@pytest.mark.parametrize("fault", ["pid", "kill", "unknown", "exit_bool"])
def test_coordinator_receipt_requires_actual_pid_and_clean_typed_exit(tmp_path, fault):
    save(tmp_path / "worker_ownership.json", dict(coordinator_pid=100))
    receipt = clean(100)
    if fault == "pid":
        receipt["pid"] = 101
    elif fault == "kill":
        receipt["process_group_kill_requested"] = True
    elif fault == "unknown":
        receipt["unknown_partial_work_possible"] = True
    else:
        receipt["exit_code"] = False
    path = tmp_path / "outer.json"
    save(path, receipt)
    with pytest.raises(ValueError, match="coordinator"):
        study._check_coordinator(tmp_path, "worker", path)


def test_child_receipt_is_bound_to_its_worker_and_parent(tmp_path):
    save(tmp_path / "config.json", {})
    save(tmp_path / "worker_ownership.json", dict(coordinator_pid=100))
    process = dict(worker_pid=101, coordinator_pid=100, config_sha256=sha(tmp_path / "config.json"))
    save(tmp_path / "worker_result.json", process)
    save(tmp_path / "exit.json", clean(101))
    assert study._check_child(
        tmp_path, "worker", tmp_path / "exit.json", tmp_path / "worker_result.json"
    )
    process["coordinator_pid"] = 102
    save(tmp_path / "worker_result.json", process)
    with pytest.raises(ValueError, match="child"):
        study._check_child(
            tmp_path, "worker", tmp_path / "exit.json", tmp_path / "worker_result.json"
        )


def test_passing_preparation_still_requires_its_actual_finalizer(tmp_path):
    save(tmp_path / "config.json", {})
    save(tmp_path / "input_manifest.json", {})
    admission = dict(
        passed=True,
        config_sha256=sha(tmp_path / "config.json"),
        input_manifest_sha256=sha(tmp_path / "input_manifest.json"),
        finalizer_pid=200,
        bindings={},
    )
    save(tmp_path / "preparation_admission.json", admission)
    save(tmp_path / "preparation_finalizer_exit.json", clean(201))
    with pytest.raises(ValueError, match="admitted"):
        study._require_preparation_admission(tmp_path)
    save(tmp_path / "preparation_finalizer_exit.json", clean(200))
    study._require_preparation_admission(tmp_path)


def test_no_learning_parser_or_dispatch_exists():
    with pytest.raises(SystemExit) as error:
        study.main(["learn"])
    assert error.value.code == 2


def test_missing_or_nonfinite_exit_fields_cannot_establish_clean_closure():
    receipt = clean(100)
    receipt.pop("stopping_reason")
    assert study._clean_exit(receipt) is False
    receipt = clean(100)
    receipt["wall_seconds"] = float("nan")
    assert study._clean_exit(receipt) is False


def test_audit_output_guard_precedes_directory_creation(tmp_path, monkeypatch):
    run, old = tmp_path / "new_run", tmp_path / "old_c1"
    old.mkdir()
    config = dict(
        locations=dict(artifacts=str(tmp_path / "new_artifacts")),
        input_preparation=dict(c1_run=str(old)),
    )
    monkeypatch.setattr(study, "_verify", lambda _: config)
    monkeypatch.setattr(study, "_capacity", lambda *_: {})
    monkeypatch.setattr(study, "_verify_inputs", lambda *_: {})
    monkeypatch.setattr(study, "_require_preparation_admission", lambda *_: None)
    destination = old / "must_not_be_created"
    with pytest.raises(ValueError, match="namespace"):
        study.launch_phase(run, "audit", destination)
    assert not destination.exists()


def test_source_freeze_marks_V_as_nonfitting(tmp_path, monkeypatch):
    from src.gps.mppi_gps import lifecycle

    source = tmp_path / "source_checkout"
    (source / "src").mkdir(parents=True)
    (source / "src/example.py").write_text("value = 1\n")
    monkeypatch.setattr(lifecycle, "ROOT", source)
    out = tmp_path / "new_run"
    lifecycle.freeze(out, {}, "student-visitation-s1.2-v")
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["policy_action_fitting"] is False
    assert manifest["raw_search_protected"] is True
    assert (out / "source/src/example.py").read_bytes() == b"value = 1\n"
    with pytest.raises(FileExistsError):
        lifecycle.freeze(out, {}, "student-visitation-s1.2-v")
