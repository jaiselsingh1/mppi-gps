from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.reporting import similarity as report


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fixture_run(tmp_path):
    root = tmp_path / "run"
    for name in ("config.json", "manifest.json", "input_manifest.json"):
        write(root / name, dict(schema="synthetic saved-only fixture"))
    write(
        root / "worker_result.json",
        dict(
            status="COMPLETED_J1_FIXED_VERSION",
            objective_passed=False,
            costs=dict(native_controls=99, model_controls=1000),
        ),
    )
    observed = dict(
        controls=1000, walking=False, survived=True, cost=10, native_cost=10, absorbing_cost=0
    )
    case = dict(
        case_id="protected_g0_l0_mppi",
        group="protected_g0",
        learner="l0",
        source="mppi",
        split="protected",
        eligible=True,
        teacher=dict(tag="teacher", controls=1000),
        incoming_bindings=dict(actor_hash="a" * 64),
        scores={name: index + 0.25 for index, name in enumerate(report.CANDIDATES)},
        predictions={name: dict(motion=0.5, walking_probability=0.2) for name in report.CANDIDATES},
        child_checkpoint=dict(file="worker/fits/case/child.pt"),
        evaluations=[
            observed,
            dict(
                observed, controls=20, survived=False, cost=4900, native_cost=0, absorbing_cost=4900
            ),
        ],
        walking_successes=0,
        walking_trials=2,
        post_motion=0.7,
        reproduced=False,
        reproduction=dict(controls=400, walking=False, survived=False, cost=3000),
        fit_receipt_file="worker/fits/case/fit_receipt.json",
    )
    write(root / "worker/cases.json", [case])
    write(
        root / "worker/teachers.json",
        dict(teacher=dict(tag="teacher", controls=1000, walking=True, survived=True, cost=20)),
    )
    write(
        root / "worker/score_definitions.json",
        dict(
            candidates={name: "Saved definition " + name for name in report.CANDIDATES},
            rejected_candidates=dict(
                fourier="Exact Parseval alias",
                deterministic_w2="MSE alias",
                explained_only="Contradictory labels ignored",
            ),
        ),
    )
    write(
        root / "worker/calibration.json",
        dict(
            passed=True,
            checks=dict(fourier_alias=True, contradiction_retained=True),
            C_identity_limitation="Synthetic nonlinear copy exposes proxy bias",
        ),
    )
    write(
        root / "worker/predictive_comparison.json",
        dict(
            motion_identified=True,
            walking_identified=False,
            objective_passed=False,
            score_table=[
                dict(
                    candidate=name,
                    motion_rmse=0.2,
                    walking_brier=0.04,
                    passed=False,
                    motion_simultaneous_lower=0.0,
                )
                for name in report.CANDIDATES
            ],
        ),
    )
    write(root / "worker/fits/case/fit_receipt.json", dict(updates=2))
    write(
        root / "worker/fits/case/learning_curve.json",
        dict(
            learning_curve=[
                dict(
                    update=i,
                    fit_update=i,
                    action_mse=1 / i,
                    total_loss=1 / i,
                    unclipped_gradient_norm=0.1,
                )
                for i in [1, 2]
            ]
        ),
    )
    (root / "worker/fits/case/child.pt").write_bytes(
        b"not a checkpoint; a reporter must never decode this"
    )
    return root, case


def test_empty_report_keeps_unattempted_labels_missing(tmp_path):
    run = tmp_path / "run"
    write(run / "config.json", {})
    result = report.report_run(run, tmp_path / "report")
    assert result["objective_passed"] is None and result["audit_passed"] is None
    assert result["actual_cases"] == result["learning_points"] == 0
    assert not any(result["new_scientific_work"].values())
    assert len(read_csv(tmp_path / "report/score_table.csv")) == 5
    assert "unmeasured" in (tmp_path / "report/report.md").read_text()


def test_case_outcomes_remain_separate_and_every_curve_point_is_preserved(tmp_path, monkeypatch):
    run, case = fixture_run(tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    result = report.report_run(run, tmp_path / "report")
    assert result["actual_cases"] == result["actual_groups"] == result["completed_fits"] == 1
    assert result["learning_points"] == 2 and result["protected_groups"] == 1
    rows = read_csv(tmp_path / "report/cases.csv")
    assert rows[0]["standalone_walking"] == "0"
    assert rows[0]["standalone_full_survival"] == "1"
    assert rows[0]["reproduced"] == "False"
    assert rows[0]["post_motion"] == "0.7"
    assert result["score_rows"] == 8
    assert result["score_outcome_pairs"] == 5
    assert len(read_csv(tmp_path / "report/learning_curves.csv")) == 2
    assert result["accounting"]["worker"]["model_controls"] == 1000
    assert (
        "Motion" not in (tmp_path / "report/report.md").read_text()
        or "partial" in (tmp_path / "report/report.md").read_text()
    )
    assert all((tmp_path / "report" / name).stat().st_size > 0 for name in result["figures"])
    assert not any(result["new_scientific_work"].values())
    with pytest.raises(ValueError, match="fresh"):
        report.report_run(run, tmp_path / "report")


def test_pending_sealed_case_is_not_a_failed_evaluation(tmp_path, monkeypatch):
    run, case = fixture_run(tmp_path)
    monkeypatch.setattr(report, "_figures", lambda *_: [])
    pending = {
        key: case[key]
        for key in [
            "case_id",
            "group",
            "learner",
            "split",
            "source",
            "eligible",
            "scores",
            "teacher",
            "incoming_bindings",
        ]
    }
    write(run / "worker/score_seals/protected.json", dict(cases=[pending]))
    write(run / "worker/cases.json", [])
    result = report.report_run(run, tmp_path / "report")
    row = read_csv(tmp_path / "report/cases.csv")[0]
    assert row["standalone_walking"] == row["standalone_attempts"] == row["reproduced"] == ""
    assert result["completed_fits"] == 0
    assert result["actual_cases"] == 1


def test_post_update_score_substitution_is_rejected_before_output(tmp_path):
    run, case = fixture_run(tmp_path)
    before = dict(case, scores=dict(case["scores"], A=999))
    write(run / "worker/score_seals/protected.json", dict(cases=[before]))
    with pytest.raises(ValueError, match="pre-update/completed"):
        report.report_run(run, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_audit_claim_requires_all_actual_input_associations(tmp_path, monkeypatch):
    run, _ = fixture_run(tmp_path)
    monkeypatch.setattr(report, "_figures", lambda *_: [])
    audit = tmp_path / "audit.json"
    fields = {
        key: hashlib.sha256((run / name).read_bytes()).hexdigest()
        for key, name in [
            ("config_sha256", "config.json"),
            ("source_manifest_sha256", "manifest.json"),
            ("input_manifest_sha256", "input_manifest.json"),
            ("worker_result_sha256", "worker_result.json"),
        ]
    }
    write(
        audit,
        dict(passed=True, full_scientific_audit_passed=True, **fields, work=dict(neural_rows=5)),
    )
    result = report.report_run(run, tmp_path / "good", audit=audit)
    assert result["audit_association_verified"] is True
    assert result["full_scientific_audit_passed"] is True
    assert result["accounting"]["audit"]["neural_rows"] == 5
    write(audit, dict(passed=True, **(fields | dict(worker_result_sha256="f" * 64))))
    with pytest.raises(ValueError, match="audit association"):
        report.report_run(run, tmp_path / "bad", audit=audit)


def test_no_fitting_or_checkpoint_import_even_with_corrupt_checkpoint_file(tmp_path):
    run = tmp_path / "run"
    write(run / "config.json", {})
    code = (
        "import sys; from pathlib import Path; "
        "from src.gps.mppi_gps.reporting.similarity import report_run; "
        f"report_run(Path({str(run)!r}), Path({str(tmp_path / 'report')!r})); "
        "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)


def test_source_escape_and_nonfinite_json_fail_before_report(tmp_path):
    run, case = fixture_run(tmp_path)
    case["fit_receipt_file"] = "../outside/fit_receipt.json"
    write(run / "worker/cases.json", [case])
    with pytest.raises(ValueError, match="escapes"):
        report.report_run(run, tmp_path / "report")
    bad = tmp_path / "bad.json"
    bad.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="Nonfinite"):
        report.SavedJSON().load(bad)


def test_failed_fit_curve_is_preserved_without_completed_child_or_evaluation(tmp_path, monkeypatch):
    run, case = fixture_run(tmp_path)
    monkeypatch.setattr(report, "_figures", lambda *_: [])
    pending = {
        key: case[key] for key in ("case_id", "group", "learner", "split", "source", "scores")
    }
    write(run / "worker/cases.json", [])
    write(run / "worker/score_seals/protected.json", dict(cases=[pending]))
    write(
        run / "worker/fits" / case["case_id"] / "fit_failure.json",
        dict(fit_evidence=dict(learning_curve=[dict(fit_update=1, update=1, action_mse=0.5)])),
    )
    result = report.report_run(run, tmp_path / "report")
    assert result["completed_fits"] == 0 and result["learning_points"] == 1
    assert read_csv(tmp_path / "report/learning_curves.csv")[0]["fit_status"] == "partial_failed"
    assert read_csv(tmp_path / "report/cases.csv")[0]["standalone_walking"] == ""


def test_all_64_learning_curves_and_actual_group_counts(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(report, "_figures", lambda *_: [])
    cases = []
    curve = dict(
        learning_curve=[dict(fit_update=i, update=i, action_mse=0.5) for i in range(1, 2001)]
    )
    for group in range(16):
        for learner in range(2):
            for source in ("mppi", "reference"):
                tag = f"g{group}_s{learner}_{source}"
                fit = f"worker/fits/{tag}/fit_receipt.json"
                cases.append(
                    dict(
                        case_id=tag,
                        group=f"g{group}",
                        learner=learner,
                        split="development" if group < 8 else "protected",
                        source=source,
                        child_checkpoint=dict(file="not_loaded.pt"),
                        fit_receipt_file=fit,
                    )
                )
                write(run / fit, dict(updates=2000))
                write((run / fit).with_name("learning_curve.json"), curve)
    write(run / "worker/cases.json", cases)
    result = report.report_run(run, tmp_path / "report")
    assert result["actual_cases"] == result["completed_fits"] == 64
    assert result["actual_groups"] == 16 and result["protected_groups"] == 8
    assert result["learning_cases_with_curves"] == 64 and result["learning_points"] == 128000
    assert not any(result["new_scientific_work"].values())


def test_read_and_output_reservations_stop_without_loosening_limits(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    write(path, dict(value="abc"))
    monkeypatch.setattr(report, "MAX_READ", 2)
    with pytest.raises(ValueError, match="read reservation"):
        report.SavedJSON().load(path)
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setattr(report, "MAX_OUTPUT", 1024**2 + 1)
    with pytest.raises(ValueError, match="output reservation"):
        report.Outputs(root).save("oversized.json", b"{}")
    assert not list(root.iterdir())
