"""Portable report bindings must survive host aliases without admitting escapes."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_export import verify_report_outputs, verify_teacher_proposal_report


def test_report_binding_accepts_a_symlinked_parent(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    path = actual / "report.pdf"
    path.write_bytes(b"saved figure")
    verify_report_outputs(
        alias,
        {path.name: {"bytes": 12, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
    )


def test_report_binding_rejects_a_symlink_escape(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    (report / "figure.pdf").symlink_to(outside)
    with pytest.raises(ValueError, match="binding differs"):
        verify_report_outputs(
            report,
            {"figure.pdf": {"bytes": 7, "sha256": hashlib.sha256(b"outside").hexdigest()}},
        )


def test_report_binding_rejects_changed_bytes(tmp_path):
    path = tmp_path / "figure.pdf"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="binding differs"):
        verify_report_outputs(
            tmp_path,
            {path.name: {"bytes": 7, "sha256": hashlib.sha256(b"initial").hexdigest()}},
        )


@pytest.fixture
def saved_h5_report(tmp_path):
    from src.gps.mppi_gps.reporting import teacher_proposal

    root = tmp_path / "export"
    capsule = root / "results/teacher_proposal_h5_0_round1"
    run = capsule / "run"
    (run / "learning").mkdir(parents=True)
    source = root / "src/gps/mppi_gps/reporting/teacher_proposal.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(teacher_proposal.__file__).read_bytes())
    inputs = {
        "config.json": dict(
            launch_authorized=False,
            eligibility=dict(useful_learning_prerequisite_satisfied=False),
        ),
        "worker_result.json": dict(
            schema="walker-teacher-proposal-round-v1",
            status="teacher_competence_negative_no_child_fits",
            execution_complete=True,
            work={},
        ),
        "call_census.json": dict(
            total=dict(
                model_controls=1,
                live_controls=1,
                physics_substeps=4,
                worker_neural_rows=1,
            ),
            strong_audit_neural_rows=1,
            worker_optimizer_steps=1,
        ),
        "learning/s0_parent.json": dict(
            completed_updates=1,
            learning_curve=[
                dict(
                    update=1,
                    fit_update=1,
                    action_mse=0.5,
                    proximal_loss=0.0,
                    total_loss=0.5,
                    unclipped_gradient_norm=1.0,
                )
            ],
        ),
    }
    for name, value in inputs.items():
        (run / name).write_text(json.dumps(value))
    (capsule / "capsule.json").write_text("{}")
    saved = dict(
        schema="walker-h5-saved-diagnostics-v1",
        source_ledger=dict(file="/external/ledger.jsonl", sha256="a" * 64, bytes=123),
        audit_receipt=dict(file="/external/audit/receipt.json", sha256="b" * 64),
        events=[],
    )
    diagnostics = capsule / "diagnostics.json"
    diagnostics.write_text(json.dumps(saved))
    report = tmp_path / "report"
    receipt = teacher_proposal.render(capsule, report, diagnostics=diagnostics)
    return root, capsule, report, receipt


def test_saved_h5_report_verifies_curves_tables_and_input_bindings(saved_h5_report):
    root, capsule, report, receipt = saved_h5_report
    verify_teacher_proposal_report(root, capsule, report, diagnostics=True)
    assert receipt["table_rows"]["learning_curves.csv"] == 1
    assert receipt["table_rows"]["student_outcomes.csv"] == 0
    assert receipt["table_rows"]["paired_gap.csv"] == 0
    assert "Useful learning was not measured" in (report / "report.md").read_text()


@pytest.mark.parametrize(
    "problem, message",
    (
        ("input", "input binding"),
        ("table", "table row count"),
        ("figure", "learning figure"),
        ("work", "zero scientific calls"),
    ),
)
def test_saved_h5_report_rejects_inconsistent_receipts(saved_h5_report, problem, message):
    root, capsule, report, receipt = saved_h5_report
    if problem == "input":
        receipt["inputs"]["saved_diagnostics"]["sha256"] = "c" * 64
    elif problem == "table":
        receipt["table_rows"]["student_outcomes.csv"] = 1
    elif problem == "figure":
        del receipt["outputs"]["learning_curves.pdf"]
    else:
        receipt["network_forwards"] = 1
    (report / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match=message):
        verify_teacher_proposal_report(root, capsule, report, diagnostics=True)
