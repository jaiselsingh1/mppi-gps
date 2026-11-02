"""C1 clean-export checks use synthetic saved JSON/CSV, with zero research work."""

from __future__ import annotations

import builtins
import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_export import verify_student_consequence_report
from src.gps.mppi_gps.reporting import student_consequence


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def binding(path):
    payload = path.read_bytes()
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


@pytest.fixture
def saved_report(tmp_path, monkeypatch, request):
    root = tmp_path / "export"
    capsule = root / "results/student_consequence_c1_0"
    run = capsule / "run"
    source = root / "src/gps/mppi_gps/reporting/student_consequence.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(student_consequence.__file__).read_bytes())
    monkeypatch.setattr(student_consequence, "__file__", str(source))
    records = {
        f"s{s}_{arm}": dict(
            tag=f"s{s}_{arm}",
            fitting_evidence=dict(
                failed=False,
                learning_curve=[
                    dict(
                        update=2000 + index,
                        fit_update=index,
                        action_mse=0.25,
                        **({} if arm == "uniform" else dict(weighted_action_mse=0.2)),
                        proximal_loss=0.0,
                        total_loss=0.25 if arm == "uniform" else 0.2,
                        unclipped_gradient_norm=0.1,
                    )
                    for index in range(1, 2001)
                ],
            ),
        )
        for s in range(3)
        for arm in ("uniform", "cost_effect", "shuffled")
    }
    panels, metrics, evaluations = {}, {}, []
    for panel, count in (("primary", 16), ("confirmation", 16), ("source", 4)):
        panels[panel] = {}
        for stream in range(3):
            panels[panel][str(stream)] = {}
            for arm in ("parent", "uniform", "cost_effect", "shuffled"):
                alias = f"s{stream}_{arm}"
                panels[panel][str(stream)][arm] = dict(
                    episodes=count,
                    successes=0,
                    survived=count,
                    controls=count * 1000,
                    mean_steps=1000,
                    mean_cost=500,
                    mean_velocity_mae=1.5,
                )
                for case in range(count):
                    evaluations.append(
                        dict(
                            actor=alias,
                            stream=stream,
                            arm=arm,
                            panel=panel,
                            seed=case,
                            controls=1000,
                            walking=False,
                            survived=True,
                            cost=500,
                            native_cost=500,
                            raw_cost=500,
                            native_terminal=0,
                            absorbing_cost=0,
                            mae=1.5,
                            tag=f"{alias}_{panel}_{case}",
                        )
                    )
                metrics[alias] = dict(
                    own=dict(action_mse=0.25, weighted_action_mse=0.2, rows=4000),
                    historical=dict(
                        train=dict(action_mse=0.3, rows=4000),
                        validation=dict(action_mse=0.4, rows=2000),
                    ),
                )
    replay = dict(
        episodes=[
            dict(
                tag=f"teacher_{index}",
                walking=True,
                survived=True,
                controls=1000,
                cost=10,
                native_cost=10,
                raw_cost=10,
                native_terminal=0,
                absorbing_cost=0,
            )
            for index in range(12)
        ]
    )
    result = dict(
        status="COMPLETED_FIXED_C1_ROSTER",
        fit_records=records,
        student_outcomes=panels,
        metrics=metrics,
        gates=dict(confirmed=False),
        learning_measured=True,
        standalone_measured=True,
        costs=dict(completed=dict(optimizer_steps=18000)),
    )
    for name, value in {
        "config.json": dict(locations={}, limits={}),
        "manifest.json": {},
        "worker_result.json": result,
        "worker_exit.json": {},
        "input_manifest.json": {},
        "prerequisites.json": {},
        "prepared/preparation_receipt.json": {},
        "worker/replay_gate.json": replay,
        "worker/contrast_gate.json": dict(passed=True),
        "worker/metrics.json": metrics,
        "worker/actor_registry.json": {},
        "worker/fit_records.json": records,
        "worker/outcomes.json": dict(panels=panels),
        "worker/learning/s0_uniform/fit.json": records["s0_uniform"],
        "worker/learning/s0_uniform/initial.json": dict(scope="synthetic saved metadata only"),
    }.items():
        save(run / name, value)
    path = run / "worker/evaluations.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in evaluations))
    save(
        capsule / "capsule.json",
        dict(
            scientific_contrast_completed=True,
            full_scientific_audit_passed=True,
            preserved_learning_points=18000,
        ),
    )
    save(
        capsule / "audit/audit.json",
        dict(
            schema="walker-student-consequence-c1.0-audit-v1",
            passed=True,
            full_scientific_audit_passed=True,
            scientific_contrast_completed=True,
            worker_result_sha256=binding(run / "worker_result.json")["sha256"],
            config_sha256=binding(run / "config.json")["sha256"],
        ),
    )

    def figures(out, curves, outcomes, reader):
        names = [
            stem + suffix
            for stem in ("learning_curves", "standalone_outcomes")
            for suffix in (".png", ".svg")
        ]
        for name in names:
            student_consequence._save(out / name, b"synthetic bound figure placeholder")
        return names

    monkeypatch.setattr(student_consequence, "_figures", figures)
    original = builtins.__import__

    def no_science(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "mujoco", "gymnasium"}:
            raise AssertionError("Saved-only export validation imported a scientific stack")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_science)
    report = tmp_path / "rendered"
    closure_path = None
    if getattr(request, "param", None) == "failed_launcher":
        traceback = capsule / "operator/audit_traceback.txt"
        gnu_time = capsule / "operator/audit_gnu_time.txt"
        traceback.parent.mkdir(parents=True)
        traceback.write_bytes(b"synthetic RSS exception, not a real audit failure\n")
        gnu_time.write_bytes(b"synthetic coordinator exit status: 1\n")
        closure_path = capsule / "accounting/audit_launcher_failure.json"
        save(
            closure_path,
            dict(
                schema="walker-c1-audit-launcher-failure-v1",
                overall_audit_closure_passed=False,
                audit_coordinator_failed=True,
                original_audit_exit_exists=False,
                original_child_exit_code=None,
                coordinator_exit_code=1,
                scientific_receipt=binding(capsule / "audit/audit.json"),
                traceback=binding(traceback),
                gnu_time=binding(gnu_time),
                frozen_source_commit="f" * 40,
            ),
        )
        metadata = json.loads((capsule / "capsule.json").read_text())
        metadata["files"] = {
            str(path.relative_to(capsule)): binding(path)
            for path in (closure_path, traceback, gnu_time)
        }
        save(capsule / "capsule.json", metadata)
    receipt = student_consequence.render(
        capsule, report, capsule / "audit/audit.json", closure_path
    )
    return root, capsule, report, receipt


def test_saved_c1_export_checks_full_curves_audit_and_all_bindings(saved_report):
    root, capsule, report, receipt = saved_report
    verify_student_consequence_report(root, capsule, report)
    assert receipt["table_rows"] == dict(curves=18000, outcomes=480, metrics=36)
    assert receipt["C1_audit_bound"] and receipt["new_neural_rows"] == 0
    assert "retention" in (report / "report.md").read_text()


@pytest.mark.parametrize("saved_report", ["failed_launcher"], indirect=True)
def test_passed_science_and_failed_coordinator_remain_distinct_and_bound(saved_report):
    root, capsule, report, receipt = saved_report
    verify_student_consequence_report(root, capsule, report)
    assert receipt["scientific_receipt_checks_passed"] is True
    assert receipt["overall_audit_closure_passed"] is False
    assert not (capsule / "audit/operator_exit.json").exists()
    assert "AUDIT_COORDINATOR_FAILED" in (report / "report.md").read_text()


@pytest.mark.parametrize("saved_report", ["failed_launcher"], indirect=True)
@pytest.mark.parametrize("problem", ("missing_companion", "claimed_success", "missing_traceback"))
def test_failed_closure_cannot_be_omitted_or_promoted_to_success(saved_report, problem):
    root, capsule, report, receipt = saved_report
    if problem == "missing_companion":
        (capsule / "accounting/audit_launcher_failure.json").unlink()
    elif problem == "claimed_success":
        receipt["overall_audit_closure_passed"] = True
        save(report / "receipt.json", receipt)
    else:
        (capsule / "operator/audit_traceback.txt").unlink()
    with pytest.raises(ValueError, match="closure|failure evidence"):
        verify_student_consequence_report(root, capsule, report)


@pytest.mark.parametrize(
    "problem,message",
    (
        ("association", "audit association"),
        ("input", "input binding"),
        ("source", "source binding"),
        ("scientific", "zero scientific"),
        ("table", "table row counts"),
        ("curve_value", "curve value"),
        ("duplicate_curve", "point roster"),
        ("figure", "output binding"),
        ("read_census", "input-read census"),
    ),
)
def test_c1_export_rejects_conflicting_saved_receipt_or_dropped_points(
    saved_report, problem, message
):
    root, capsule, report, receipt = saved_report
    if problem == "association":
        receipt["C1_config_association"] = "config_hash_not_declared"
    elif problem == "input":
        receipt["inputs"][str(capsule / "run/config.json")]["sha256"] = "0" * 64
    elif problem == "source":
        receipt["source_code"]["sha256"] = "0" * 64
    elif problem == "scientific":
        receipt["new_neural_rows"] = 1
    elif problem == "table":
        receipt["table_rows"]["outcomes"] -= 1
    elif problem in {"curve_value", "duplicate_curve"}:
        path = report / "curves.csv"
        rows = path.read_text().splitlines(keepends=True)
        if problem == "curve_value":
            rows[1] = rows[1].replace(",0.25,", ",0.5,", 1)
        else:
            rows[2] = rows[1]
        path.write_text("".join(rows))
        receipt["outputs"]["curves.csv"] = binding(path)
        receipt["output_hash_read_bytes"] = sum(row["bytes"] for row in receipt["outputs"].values())
    elif problem == "figure":
        del receipt["outputs"]["learning_curves.png"]
    elif problem == "read_census":
        receipt["saved_JSON_read_bytes"] += 1
    save(report / "receipt.json", receipt)
    with pytest.raises(ValueError, match=message):
        verify_student_consequence_report(root, capsule, report)
