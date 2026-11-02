"""Small saved-only C1 report fixtures; no research execution or raw reads."""

from __future__ import annotations

import builtins
import csv
import json
import subprocess
import sys
from copy import deepcopy

import pytest

from src.gps.mppi_gps.reporting import student_consequence as module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source(tmp_path, result=None):
    run = tmp_path / "run"
    run.mkdir(parents=True)
    save(
        run / "config.json",
        dict(
            locations=dict(
                remote_workspace="/external/workspace", run_id="c1", source_run="/external/source"
            ),
            limits=dict(worker_wall_seconds=21600),
        ),
    )
    if result is not None:
        save(run / "worker_result.json", result)
    return run


def curve(count, *, weighted=True):
    return [
        dict(
            update=2000 + i,
            fit_update=i,
            action_mse=1 / i,
            **({"weighted_action_mse": 2 / i} if weighted else {}),
            proximal_loss=0.0,
            total_loss=2 / i,
            unclipped_gradient_norm=3 / i,
        )
        for i in range(1, count + 1)
    ]


def summary(walking=0, survived=16, *, n=16):
    return dict(
        episodes=n,
        successes=walking,
        survived=survived,
        controls=n * 1000,
        mean_steps=1000,
        mean_cost=500,
        mean_velocity_mae=1.5,
    )


def test_prerequisite_stop_stays_missing_and_never_imports_stacks(tmp_path, monkeypatch):
    run = source(
        tmp_path,
        dict(
            status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE",
            error=dict(type="ValueError", message="replay mismatch"),
            costs=dict(
                requested=dict(native_controls=12),
                completed=dict(native_controls=11),
                unknown_partial=dict(native_controls=1),
            ),
        ),
    )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "mujoco", "matplotlib"}:
            raise AssertionError("Unexpected implementation/plot import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    out = tmp_path / "report"
    receipt = module.render(run, out)
    assert receipt["table_rows"] == dict(curves=0, outcomes=0, metrics=0)
    assert receipt["figures"] == []
    text = (out / "report.md").read_text()
    assert "prerequisite/infrastructure" in text
    assert "missing, not zero" in text
    assert "/external/workspace/runs/c1/worker/" in text
    costs = json.loads((out / "costs.json").read_text())
    assert costs["worker"]["unknown_partial"]["native_controls"] == 1
    assert read_csv(out / "curves.csv") == read_csv(out / "outcomes.csv") == []


def test_degenerate_gate_is_not_a_failed_learning_result(tmp_path):
    run = source(
        tmp_path,
        dict(
            status="STOPPED_DEGENERATE_PLANNED_CONTRAST",
            contrast_gate=dict(
                passed=False, reasons=[dict(stream="1", reason="constant_cost_effect_weights")]
            ),
            learning_measured=False,
            standalone_measured=False,
        ),
    )
    out = tmp_path / "report"
    module.render(run, out)
    text = (out / "report.md").read_text()
    assert "not a failed-learning result" in text
    assert "constant_cost_effect_weights" in text
    assert read_csv(out / "outcomes.csv") == []


def test_all_nine_complete_curves_retained_and_bound(tmp_path, monkeypatch):
    records = {
        f"s{s}_{arm}": dict(
            tag=f"s{s}_{arm}", fitting_evidence=dict(learning_curve=curve(2000), failed=False)
        )
        for s in range(3)
        for arm in ("uniform", "cost_effect", "shuffled")
    }
    run = source(
        tmp_path,
        dict(status="COMPLETED_FIXED_C1_ROSTER", fit_records=records, gates=dict(confirmed=False)),
    )
    save(run / "worker/learning/s0_uniform/fit.json", records["s0_uniform"])
    monkeypatch.setattr(module, "_figures", lambda *args: [])
    out = tmp_path / "report"
    receipt = module.render(run, out)
    rows = read_csv(out / "curves.csv")
    assert len(rows) == 18000
    assert rows[0]["action_mse"] == "1.0" and rows[0]["weighted_action_mse"] == "2.0"
    uniform = next(row for row in rows if row["actor"] == "s0_uniform")
    assert "worker/learning/s0_uniform/fit.json" in uniform["sources"]
    assert receipt["table_rows"]["curves"] == 18000
    assert str(run / "worker_result.json") in receipt["inputs"]
    assert str(run / "worker/learning/s0_uniform/fit.json") in receipt["inputs"]
    for name, descriptor in receipt["outputs"].items():
        assert module._sha(out / name) == descriptor["sha256"]
        assert (out / name).stat().st_size == descriptor["bytes"]


def test_partial_failure_json_preserves_points_without_fabricating_weighted_values(
    tmp_path, monkeypatch
):
    run = source(tmp_path, dict(status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"))
    save(
        run / "worker/learning/s1_uniform/failure.json",
        dict(
            tag="s1_uniform",
            fitting_evidence=dict(learning_curve=curve(3, weighted=False), failed=True),
        ),
    )
    monkeypatch.setattr(module, "_figures", lambda *args: [])
    out = tmp_path / "report"
    module.render(run, out)
    rows = read_csv(out / "curves.csv")
    assert len(rows) == 3
    assert all(row["weighted_action_mse"] == "" and row["failed"] == "True" for row in rows)


def test_conflicting_curve_copies_fail_instead_of_selecting_favorable_values(tmp_path):
    record = dict(tag="s0_uniform", fitting_evidence=dict(learning_curve=curve(2)))
    run = source(
        tmp_path, dict(status="COMPLETED_FIXED_C1_ROSTER", fit_records={"s0_uniform": record})
    )
    altered = deepcopy(record)
    altered["fitting_evidence"]["learning_curve"][1]["action_mse"] = 99
    save(run / "worker/learning/s0_uniform/fit.json", altered)
    with pytest.raises(ValueError, match="Conflicting"):
        module.render(run, tmp_path / "report")


def test_standing_is_not_walking_and_source_gaps_stay_paired(tmp_path, monkeypatch):
    panels = {
        p: {
            "0": {
                "parent": summary(),
                "uniform": summary(2),
                "cost_effect": summary(),
                "shuffled": summary(),
            }
        }
        for p in ("primary", "confirmation")
    }
    panels["source"] = {"0": {"uniform": summary(0, 4, n=4)}}
    gaps = [
        dict(
            actor="s0_uniform",
            source_tag="teacher_s0_independent_00",
            teacher_cost=5,
            student_cost=10,
            cost_gap=5,
            teacher_walking=True,
            student_walking=False,
            teacher_survived=True,
            student_survived=True,
        )
    ]
    run = source(
        tmp_path,
        dict(
            status="COMPLETED_FIXED_C1_ROSTER",
            student_outcomes=panels,
            teacher_student_gap=gaps,
            gates=dict(confirmed=False, uniform_parent_confirmed=True),
            metrics={
                "s0_uniform": dict(
                    own=dict(action_mse=0.1, weighted_action_mse=0.9, rows=4000, batch_size=1000),
                    historical=dict(validation=dict(action_mse=0.2, rows=2000, batch_size=1024)),
                )
            },
        ),
    )
    audit = tmp_path / "audit.json"
    save(
        audit,
        dict(
            schema="walker-c1-round-audit-v1",
            scientific_round_audit_complete=True,
            worker_result_sha256=module._sha(run / "worker_result.json"),
            config_sha256=module._sha(run / "config.json"),
        ),
    )
    monkeypatch.setattr(module, "_figures", lambda *args: [])
    out = tmp_path / "report"
    module.render(run, out, audit)
    rows = read_csv(out / "outcomes.csv")
    weighted = next(
        row for row in rows if row["panel"] == "primary" and row["arm"] == "cost_effect"
    )
    assert weighted["walking_successes"] == "0" and weighted["full_survivals"] == "16"
    text = (out / "report.md").read_text()
    assert "Uniform teaching produced" in text and "separate results" in text
    assert "matched source starting contexts" in text
    assert "| s0_uniform | teacher_s0_independent_00 | 5 | 10 | 5 |" in text
    metrics = read_csv(out / "metrics.csv")
    assert metrics[0]["action_mse"] == "0.1" and metrics[0]["weighted_action_mse"] == "0.9"


def test_saved_episode_raw_terminal_and_absorbing_costs_are_separate_without_raw_reads(
    tmp_path, monkeypatch
):
    run = source(tmp_path, dict(status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"))
    event = dict(
        tag="eval_s0_uniform_primary_00",
        actor="s0_uniform",
        stream=0,
        arm="uniform",
        panel="primary",
        seed=1,
        controls=2,
        walking=False,
        survived=False,
        native_cost=23,
        raw_cost=3,
        native_terminal=20,
        absorbing_cost=4990,
        cost=5013,
        mae=1.2,
        trace=dict(file="arrays.bin", checkpoint="bad.pt"),
    )
    path = run / "worker/evaluations.jsonl"
    path.parent.mkdir()
    path.write_text(json.dumps(event) + "\n")
    original = module._Reader.load

    def guarded(self, path, **kwargs):
        assert path.suffix in {".json", ".jsonl"}
        return original(self, path, **kwargs)

    monkeypatch.setattr(module._Reader, "load", guarded)
    out = tmp_path / "report"
    module.render(run, out)
    row = read_csv(out / "outcomes.csv")[0]
    assert (row["raw_cost"], row["native_terminal"], row["absorbing_cost"], row["cost"]) == (
        "3",
        "20",
        "4990",
        "5013",
    )


def test_structure_only_audit_does_not_confirm_scientific_result(tmp_path):
    run = source(tmp_path, dict(status="COMPLETED_FIXED_C1_ROSTER", gates=dict(confirmed=True)))
    audit = tmp_path / "audit.json"
    save(audit, dict(schema="walker-c1-saved-stream-audit-v1", complete=True))
    receipt = module.render(run, tmp_path / "report", audit)
    assert receipt["C1_audit_bound"] is False
    assert "claims remain provisional" in (tmp_path / "report/report.md").read_text()


def test_wrong_audit_result_hash_is_rejected(tmp_path):
    run = source(tmp_path, dict(status="COMPLETED_FIXED_C1_ROSTER", gates=dict(confirmed=True)))
    audit = tmp_path / "audit.json"
    save(
        audit,
        dict(
            schema="walker-c1-round-audit-v1",
            full_scientific_audit_passed=True,
            worker_result_sha256="0" * 64,
            config_sha256=module._sha(run / "config.json"),
        ),
    )
    with pytest.raises(ValueError, match="different worker result"):
        module.render(run, tmp_path / "report", audit)


def test_interrupted_evaluation_jsonl_keeps_complete_prefix_and_opaque_tail(tmp_path):
    run = source(tmp_path, dict(status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE"))
    path = run / "worker/evaluations.jsonl"
    path.parent.mkdir()
    path.write_text(
        json.dumps(dict(actor="s0_parent", controls=2, walking=False, survived=False, cost=4))
        + '\n{"actor":'
    )
    out = tmp_path / "report"
    receipt = module.render(run, out)
    assert len(read_csv(out / "outcomes.csv")) == 1
    assert (
        receipt["incomplete_JSONL_fragments"][0]["availability"]
        == "incomplete_final_JSONL_fragment"
    )
    assert "incomplete final JSONL fragment" in (out / "report.md").read_text()


def test_standalone_figures_keep_walking_and_survival_separate(tmp_path):
    panels = {
        p: {"0": {"parent": summary(), "uniform": summary(2)}} for p in ("primary", "confirmation")
    }
    run = source(tmp_path, dict(status="COMPLETED_FIXED_C1_ROSTER", student_outcomes=panels))
    receipt = module.render(run, tmp_path / "report")
    assert set(receipt["figures"]) == {"standalone_outcomes.svg", "standalone_outcomes.png"}


def test_saved_curve_figures_and_create_only_cli(tmp_path):
    run = source(
        tmp_path,
        dict(
            status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE",
            fit_records={"s0_cost_effect": dict(fitting_evidence=dict(learning_curve=curve(3)))},
        ),
    )
    out = tmp_path / "report"
    receipt = module.render(run, out)
    assert set(receipt["figures"]) == {"learning_curves.svg", "learning_curves.png"}
    assert (out / "learning_curves.svg").stat().st_size > 1000
    with pytest.raises(FileExistsError):
        module.render(run, out)
    empty = source(tmp_path / "other")
    subprocess.run(
        [
            sys.executable,
            "scripts/diagnostics/report_student_consequence.py",
            "--run",
            str(empty),
            "--out",
            str(tmp_path / "cli"),
        ],
        check=True,
    )
    assert (tmp_path / "cli/report.md").is_file()


def failed_audit_closure(tmp_path):
    run = source(
        tmp_path,
        dict(
            status="COMPLETED_FIXED_C1_ROSTER",
            gates=dict(confirmed=True, uniform_parent_confirmed=True),
            costs=dict(completed=dict(native_controls=17)),
            metrics={"s0_uniform": dict(own=dict(action_mse=0.125, rows=4000))},
        ),
    )
    audit = tmp_path / "audit.json"
    save(
        audit,
        dict(
            schema="walker-c1-round-audit-v1",
            full_scientific_audit_passed=True,
            worker_result_sha256=module._sha(run / "worker_result.json"),
            config_sha256=module._sha(run / "config.json"),
        ),
    )
    closure = tmp_path / "failure.json"
    save(
        closure,
        dict(
            schema="walker-c1-audit-launcher-failure-v1",
            scientific_receipt=dict(bytes=audit.stat().st_size, sha256=module._sha(audit)),
            overall_audit_closure_passed=False,
            audit_coordinator_failed=True,
            original_audit_exit_exists=False,
            original_child_exit_code=None,
            coordinator_exit_code=1,
        ),
    )
    return run, audit, closure


def test_failed_outer_closure_cannot_admit_passed_scientific_receipt(tmp_path):
    run, audit, closure = failed_audit_closure(tmp_path)
    original_audit = audit.read_bytes()
    original_result = (run / "worker_result.json").read_bytes()
    out = tmp_path / "report"
    receipt = module.render(run, out, audit, closure)
    assert receipt["C1_audit_bound"] is True
    assert receipt["scientific_receipt_checks_passed"] is True
    assert receipt["overall_audit_closure_passed"] is False
    assert receipt["audit_closure_status"] == "AUDIT_COORDINATOR_FAILED"
    assert receipt["inputs"][str(closure)]["sha256"] == module._sha(closure)
    text = (out / "report.md").read_text()
    assert "AUDIT_COORDINATOR_FAILED; outer closure incomplete" in text
    assert "saved scientific receipt reports passing checks" in text
    assert "confirm the declared weighting walking gain" not in text
    assert "Uniform teaching produced" not in text
    costs = json.loads((out / "costs.json").read_text())
    assert costs["audit_launcher_failure"] == json.loads(closure.read_text())
    assert costs["worker"]["completed"]["native_controls"] == 17
    assert audit.read_bytes() == original_audit
    assert (run / "worker_result.json").read_bytes() == original_result
    assert not (audit.parent / "exit.json").exists()


def test_capsule_failure_companion_is_detected_and_numerical_tables_unchanged(tmp_path):
    run, audit, closure = failed_audit_closure(tmp_path)
    clean = tmp_path / "clean"
    clean_receipt = module.render(run, clean, audit)
    assert "confirm the declared weighting walking gain" in (clean / "report.md").read_text()
    assert clean_receipt["overall_audit_closure_passed"] is None
    capsule_closure = run / "accounting/audit_launcher_failure.json"
    save(capsule_closure, json.loads(closure.read_text()))
    failed = tmp_path / "failed"
    receipt = module.render(run, failed, audit)
    assert receipt["overall_audit_closure_passed"] is False
    assert str(capsule_closure) in receipt["inputs"]
    for name in ("curves.csv", "outcomes.csv", "metrics.csv"):
        assert (clean / name).read_bytes() == (failed / name).read_bytes()


@pytest.mark.parametrize("field", ("sha256", "bytes"))
def test_failure_companion_must_bind_exact_scientific_receipt(tmp_path, field):
    run, audit, closure = failed_audit_closure(tmp_path)
    value = json.loads(closure.read_text())
    value["scientific_receipt"][field] = "0" * 64 if field == "sha256" else 1
    save(closure, value)
    with pytest.raises(ValueError, match="different scientific receipt"):
        module.render(run, tmp_path / "report", audit, closure)
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("schema", "not-a-c1-closure"),
        ("overall_audit_closure_passed", True),
        ("audit_coordinator_failed", False),
        ("original_audit_exit_exists", True),
        ("original_child_exit_code", 0),
        ("coordinator_exit_code", 0),
        ("overall_audit_closure_passed", 0),
    ),
)
def test_failure_companion_cannot_relax_original_closure_fields(tmp_path, field, invalid):
    run, audit, closure = failed_audit_closure(tmp_path)
    value = json.loads(closure.read_text())
    value[field] = invalid
    save(closure, value)
    with pytest.raises(ValueError, match="closure failure field differs"):
        module.render(run, tmp_path / "report", audit, closure)


def test_saved_report_cli_accepts_explicit_failed_closure(tmp_path):
    run, audit, closure = failed_audit_closure(tmp_path)
    out = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.gps.mppi_gps.reporting.student_consequence",
            "--run",
            str(run),
            "--out",
            str(out),
            "--audit",
            str(audit),
            "--closure",
            str(closure),
        ],
        check=True,
    )
    assert json.loads((out / "receipt.json").read_text())["overall_audit_closure_passed"] is False
