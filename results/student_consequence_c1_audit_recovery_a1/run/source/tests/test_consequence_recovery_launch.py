"""Synthetic recovery ownership/admission tests; no scientific execution."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

from src.gps.mppi_gps import lifecycle, proposal_launch
from src.gps.mppi_gps.experiments import consequence_recovery as study


def recipe():
    return json.loads(study.RECIPE.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def closed_run(tmp_path):
    config = recipe()
    save(tmp_path / "config.json", config)
    save(tmp_path / "input_manifest.json", {})
    save(tmp_path / "manifest.json", {"config_sha256": study._sha(tmp_path / "config.json")})
    save(
        tmp_path / "result/recovery.json",
        dict(
            schema="walker-c1-audit-recovery-result-v1",
            recovery_checks_passed=True,
            admitted_original_scientific_checks=True,
            retained_numerical_certification=True,
            original_overall_audit_closure_passed=False,
            original_child_exit_code=None,
            original_audit_coordinator_failed=True,
            protocol_sha256=hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest(),
            new_scientific_work=config["new_scientific_work"],
        ),
    )
    save(
        tmp_path / "exit.json",
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
        ),
    )
    return tmp_path


def test_declared_protocol_cannot_buy_science_or_reset_cumulative_budgets():
    study.validate_protocol(recipe())
    for key, value in (
        ("neural_rows", 1),
        ("native_controls", 1),
        ("optimizer_steps", 1),
        ("checkpoint_tensor_loads", 1),
        ("neural_rows", False),
    ):
        changed = recipe()
        changed["new_scientific_work"][key] = value
        with pytest.raises(ValueError):
            study.validate_protocol(changed)
    changed = recipe()
    changed["cumulative"]["prior_wall_upper_seconds"] = 6400
    with pytest.raises(ValueError):
        study.validate_protocol(changed)


def test_candidate_retains_original_failure_without_final_admission(tmp_path):
    run = closed_run(tmp_path)
    admitted = study.admit(run)
    assert admitted["audit_recovery_resolved"] is False
    assert admitted["candidate_checks_passed"] is True
    assert not (run / "admission.json").exists()
    assert admitted["original_overall_audit_closure_passed"] is False
    assert admitted["original_child_exit_code"] is None
    assert set(admitted["bindings"]) == {"result", "config", "manifest", "exit"}
    for row in admitted["bindings"].values():
        assert row["sha256"] == study._sha(run / row["file"])
    with pytest.raises(FileExistsError):
        study.admit(run)


@pytest.mark.parametrize(
    ("file", "field", "value"),
    [
        ("exit.json", "exit_code", 1),
        ("exit.json", "leader_reaped", False),
        ("exit.json", "unknown_partial_work_possible", True),
        ("exit.json", "stopping_reason", "coordinator_exception"),
        ("result/recovery.json", "original_child_exit_code", 0),
        ("result/recovery.json", "protocol_sha256", "0" * 64),
        ("manifest.json", "config_sha256", "0" * 64),
    ],
)
def test_failed_or_cross_bound_admission_is_not_published(tmp_path, file, field, value):
    run = closed_run(tmp_path)
    document = json.loads((run / file).read_text())
    document[field] = value
    save(run / file, document)
    with pytest.raises(ValueError):
        study.admit(run)
    assert not (run / "admission.json").exists()


def test_import_guard_refuses_inference_and_simulator_stacks():
    guard = study._NoScientificImports()
    for name in ("torch", "torch.nn", "mujoco", "gymnasium", "stable_baselines3", "jax"):
        with pytest.raises(RuntimeError):
            guard.find_spec(name)
    assert guard.find_spec("json") is None


def test_freeze_is_nonfitting_create_only_and_pins_inputs(tmp_path, monkeypatch):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "README.md").write_text("Synthetic source.\n")
    monkeypatch.setattr(lifecycle, "ROOT", source)
    out = tmp_path / "recovery"
    study.freeze_recovery(study.RECIPE, out, "a" * 40)
    manifest = json.loads((out / "manifest.json").read_text())
    inputs = json.loads((out / "input_manifest.json").read_text())
    assert manifest["policy_action_fitting"] is False
    assert inputs["protocol_sha256"] == study._sha(out / "config.json")
    assert inputs["source_commit"] == "a" * 40
    with pytest.raises(FileExistsError):
        study.freeze_recovery(study.RECIPE, out, "a" * 40)


def test_launcher_persists_exception_and_never_retries(tmp_path, monkeypatch):
    config = recipe()
    save(tmp_path / "config.json", config)
    save(tmp_path / "manifest.json", {})
    save(tmp_path / "input_manifest.json", {})
    monkeypatch.setattr(
        study,
        "_verify_source",
        lambda _, **kwargs: dict(explicit_source_verification_read_bytes=0, wall_seconds=0),
    )
    affinity = []
    monkeypatch.setattr(study.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
    monkeypatch.setattr(
        study.os, "sched_setaffinity", lambda _, value: affinity.append(set(value)), raising=False
    )
    retained = dict(exit_code=0, stopping_reason="coordinator_exception", leader_reaped=True)
    calls = []

    def fail(*args, **kwargs):
        calls.append(deepcopy(kwargs))
        error = RuntimeError("synthetic coordinator failure")
        error.process_receipt = retained
        raise error

    monkeypatch.setenv("PYTHONPATH", "/untrusted/inherited/source")
    monkeypatch.setattr(proposal_launch, "bounded_process", fail)
    with pytest.raises(RuntimeError):
        study.launch(tmp_path)
    assert json.loads((tmp_path / "exit.json").read_text()) == retained
    assert affinity == [{0}, {0, 1}]
    with pytest.raises(FileExistsError):
        study.launch(tmp_path)
    assert len(calls) == 1
    assert "PYTHONPATH" not in calls[0]["environment"]


def frozen_source(tmp_path, monkeypatch):
    run = tmp_path / "frozen"
    source = run / "source"
    source.mkdir(parents=True)
    payload = source / "example.py"
    payload.write_bytes(b"# bounded source\n")
    save(run / "config.json", recipe())
    save(
        run / "manifest.json",
        dict(
            command=study.COMMAND,
            input_sha256=None,
            config_sha256=study._sha(run / "config.json"),
            files=[dict(file="example.py", sha256=study._sha(payload))],
        ),
    )
    monkeypatch.setattr(study, "ROOT", source)
    return run, payload


def test_source_payload_is_stat_reserved_before_hashing(tmp_path, monkeypatch):
    run, payload = frozen_source(tmp_path, monkeypatch)
    opened = []
    original = study.os.open

    def record(path, *args, **kwargs):
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(study.os, "open", record)
    with pytest.raises(ValueError, match="pre-read reservation"):
        study._verify_source(run, source_export_bytes=1)
    assert payload not in opened


@pytest.mark.parametrize("kind", ["escape", "symlink", "duplicate"])
def test_frozen_source_paths_and_roster_cannot_escape_or_alias(tmp_path, monkeypatch, kind):
    run, payload = frozen_source(tmp_path, monkeypatch)
    manifest = json.loads((run / "manifest.json").read_text())
    if kind == "escape":
        manifest["files"][0]["file"] = "../config.json"
    elif kind == "duplicate":
        manifest["files"].append(deepcopy(manifest["files"][0]))
    else:
        payload.unlink()
        payload.symlink_to(run / "config.json")
    save(run / "manifest.json", manifest)
    with pytest.raises(ValueError):
        study._verify_source(run)


def test_bounded_reads_stop_before_oversized_or_expired_payload_access(tmp_path, monkeypatch):
    path = tmp_path / "payload"
    path.write_bytes(b"12345")
    opened = []
    original = study.os.open

    def record(target, *args, **kwargs):
        opened.append(target)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(study.os, "open", record)
    with pytest.raises(ValueError, match="read reservation"):
        study._ReadBudget(4, study.time.monotonic() + 10).read(path, 5)
    with pytest.raises(ValueError, match="deadline"):
        study._ReadBudget(10, 0).read(path, 5)
    assert opened == []


def test_parent_source_passes_share_one_read_debit(tmp_path, monkeypatch):
    run, _ = frozen_source(tmp_path, monkeypatch)
    budget = study._ReadBudget(1024**2, study.time.monotonic() + 10)
    first = study._verify_source(run, budget=budget)
    second = study._verify_source(run, budget=budget)
    assert (
        first["explicit_source_verification_read_bytes"]
        == second["explicit_source_verification_read_bytes"]
    )
    assert budget.read_bytes == first["explicit_source_verification_read_bytes"] * 2
    assert second["orchestration_read_bytes"] == budget.read_bytes


@pytest.mark.parametrize(
    "file,field",
    [
        ("exit.json", "stopping_reason"),
        ("result/recovery.json", "original_child_exit_code"),
        ("result/recovery.json", "schema"),
    ],
)
def test_required_nulls_and_result_schema_cannot_be_omitted(tmp_path, file, field):
    run = closed_run(tmp_path)
    value = json.loads((run / file).read_text())
    value.pop(field)
    save(run / file, value)
    with pytest.raises(ValueError):
        study.admit(run)
    assert not (run / "admission.json").exists()


@pytest.mark.parametrize("key", ["original_run", "original_artifacts"])
def test_freeze_rejects_output_inside_original_trees_before_creation(tmp_path, key):
    config = recipe()
    original = tmp_path / "original"
    config["locations"][key] = str(original)
    config_path = tmp_path / "protocol.json"
    save(config_path, config)
    target = original / "new-recovery"
    with pytest.raises(ValueError, match="original input tree"):
        study.freeze_recovery(config_path, target, "a" * 40)
    assert not original.exists()


def test_shared_nonchild_deadline_includes_postflight_and_never_admits(tmp_path, monkeypatch):
    config = recipe()
    save(tmp_path / "config.json", config)
    save(tmp_path / "manifest.json", {})
    save(tmp_path / "input_manifest.json", {})
    clock = [0.0]
    monkeypatch.setattr(study.time, "monotonic", lambda: clock[0])
    budgets = []

    def verify(_, **kwargs):
        budgets.append(kwargs["budget"])
        clock[0] += 20 if len(budgets) == 1 else 41
        return dict(explicit_source_verification_read_bytes=0, wall_seconds=0)

    monkeypatch.setattr(study, "_verify_source", verify)
    monkeypatch.setattr(study.os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(study.os, "sched_setaffinity", lambda *_: None, raising=False)

    def child(*args, **kwargs):
        clock[0] += 800
        return dict(exit_code=0, stopping_reason=None)

    monkeypatch.setattr(proposal_launch, "bounded_process", child)
    with pytest.raises(ValueError, match="deadline"):
        study.launch(tmp_path)
    assert len(budgets) == 2 and budgets[0] is budgets[1]
    assert json.loads((tmp_path / "exit.json").read_text())["exit_code"] == 0
    assert not (tmp_path / "admission.json").exists()


def external_closed_run(tmp_path, monkeypatch):
    run = closed_run(tmp_path)
    study.admit(run)
    save(run / "ownership.json", dict(coordinator_pid=4242))
    save(
        run / "coordinator_exit.json",
        dict(
            exit_code=0,
            leader_reaped=True,
            stopping_reason=None,
            process_group_kill_requested=False,
            unknown_partial_work_possible=False,
            pid=4242,
            wall_seconds=1.0,
        ),
    )
    monkeypatch.setattr(study, "ROOT", run / "source")
    return run


def test_final_admission_requires_clean_external_coordinator_and_five_bindings(
    tmp_path, monkeypatch
):
    run = external_closed_run(tmp_path, monkeypatch)
    admitted = study.finalize(run)
    assert admitted["audit_recovery_resolved"] is True
    assert admitted["original_overall_audit_closure_passed"] is False
    assert admitted["original_child_exit_code"] is None
    assert set(admitted["bindings"]) == {"result", "config", "manifest", "exit", "coordinator_exit"}
    for row in admitted["bindings"].values():
        assert row["sha256"] == study._sha(run / row["file"])
    assert admitted["admission_candidate"]["sha256"] == study._sha(run / "admission_candidate.json")
    with pytest.raises(FileExistsError):
        study.finalize(run)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 1),
        ("leader_reaped", False),
        ("stopping_reason", "coordinator_exception"),
        ("unknown_partial_work_possible", True),
        ("pid", 9999),
        ("wall_seconds", 861),
    ],
)
def test_failed_unclosed_or_wrong_external_coordinator_never_finalizes(
    tmp_path, monkeypatch, field, value
):
    run = external_closed_run(tmp_path, monkeypatch)
    outer = json.loads((run / "coordinator_exit.json").read_text())
    outer[field] = value
    save(run / "coordinator_exit.json", outer)
    with pytest.raises(ValueError):
        study.finalize(run)
    assert not (run / "admission.json").exists()


def test_candidate_before_external_exit_is_not_a_final_admission(tmp_path, monkeypatch):
    run = external_closed_run(tmp_path, monkeypatch)
    (run / "coordinator_exit.json").unlink()
    with pytest.raises(FileNotFoundError):
        study.finalize(run)
    assert not (run / "admission.json").exists()
    assert (
        json.loads((run / "admission_candidate.json").read_text())["audit_recovery_resolved"]
        is False
    )


def test_changed_candidate_is_not_accepted_after_clean_coordinator_exit(tmp_path, monkeypatch):
    run = external_closed_run(tmp_path, monkeypatch)
    candidate = json.loads((run / "admission_candidate.json").read_text())
    candidate["bindings"]["result"]["sha256"] = "0" * 64
    save(run / "admission_candidate.json", candidate)
    with pytest.raises(ValueError, match="candidate differs"):
        study.finalize(run)
    assert not (run / "admission.json").exists()
