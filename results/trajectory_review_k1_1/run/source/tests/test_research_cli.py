"""Small metadata/dispatch fixtures; no research acquisition or fitting."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import train_mppi_gps
from scripts.check_repository import check_binding
from src.gps.mppi_gps.inspection import format_inspection, inspect_run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "saved"
    (run / "source/src").mkdir(parents=True)
    (run / "inputs").mkdir()
    config = run / "config.json"
    config.write_text('{"schema":"synthetic-only","seed":17}')
    source = run / "source/src/example.py"
    source.write_text("# saved worker\n")
    input_file = run / "inputs/provenance.json"
    input_file.write_text('{"parent":"synthetic"}')

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (run / "manifest.json").write_text(
        json.dumps(
            {
                "files": [{"file": "src/example.py", "sha256": digest(source)}],
                "config_sha256": digest(config),
                "input_sha256": None,
            }
        )
    )
    (run / "input_manifest.json").write_text(
        json.dumps(
            {
                "files": {"inputs/provenance.json": digest(input_file)},
            }
        )
    )
    (run / "result.json").write_text(
        json.dumps(
            {
                "status": "synthetic_complete",
                "error": None,
                "loop_closed": False,
                "cases": {
                    "primary": {
                        "child": {
                            "episodes": 2,
                            "successes": 0,
                            "survived": 1,
                            "mean_cost": 42,
                            "costs": [40, 44],
                        }
                    }
                },
                "work": {"completed_optimizer_steps": 3},
            }
        )
    )
    return run


def test_inspector_verifies_saved_metadata_without_writes(saved_run):
    before = {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    report = inspect_run(saved_run, verify=True)
    assert report["outcomes"]["status"] == "synthetic_complete"
    assert report["student_outcomes"]["primary"]["child"]["survived"] == 1
    assert "costs" not in report["student_outcomes"]["primary"]["child"]
    assert report["costs"]["work"]["completed_optimizer_steps"] == 3
    assert report["hash_mismatches"] == report["errors"] == []
    assert "evidence/raw.h5" in report["missing_evidence"]
    assert before == {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    (saved_run / "source/src/example.py").write_text("# tampered\n")
    assert inspect_run(saved_run, verify=True)["hash_mismatches"] == ["src/example.py"]


def test_inspector_cli_has_no_numerical_or_simulator_imports(saved_run):
    code = """
import builtins, runpy, sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'mujoco', 'numpy', 'scipy'} or name.startswith('src.envs'):
        raise AssertionError('Unexpected runtime import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['scripts/train_mppi_gps.py', 'inspect', '--run', sys.argv[1], '--json', '--verify']
runpy.run_path('scripts/train_mppi_gps.py', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(saved_run)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["kind"] == "frozen_run"


def test_saved_metric_inspector_requires_inputs_without_inventing_raw_evidence(saved_run):
    (saved_run / "config.json").write_text('{"schema":"walker-trajectory-review-config-v1"}')
    manifest = json.loads((saved_run / "manifest.json").read_text())
    manifest["config_sha256"] = hashlib.sha256((saved_run / "config.json").read_bytes()).hexdigest()
    (saved_run / "manifest.json").write_text(json.dumps(manifest))
    (saved_run / "result.json").write_text(
        json.dumps(
            {
                "schema": "walker-trajectory-review-result-v1",
                "execution_complete": True,
                "reference": {"policy": "reference"},
                "analysis_counts": {"trajectories": 34, "rows": 30070, "bins": 304},
                "groups": [{"source": "synthetic", "count": 2}],
                "work": {"new_policy_forward_rows": 0},
            }
        )
    )
    before = {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    report = inspect_run(saved_run, verify=True)
    assert report["diagnostics"]["analysis_counts"]["bins"] == 304
    assert report["diagnostics"]["groups"] == [{"source": "synthetic", "count": 2}]
    assert report["bulk_evidence"] == {}
    assert report["missing_evidence"] == report["errors"] == report["hash_mismatches"] == []
    assert before == {p: p.read_bytes() for p in saved_run.rglob("*") if p.is_file()}
    (saved_run / "inputs/provenance.json").unlink()
    assert "inputs/provenance.json" in inspect_run(saved_run, verify=True)["missing_evidence"]


@pytest.mark.parametrize("name", train_mppi_gps.STUDIES)
def test_study_dispatch_preserves_worker_arguments(monkeypatch, name):
    seen = []
    module = SimpleNamespace(main=lambda args: seen.append(args) or 7)
    monkeypatch.setattr(train_mppi_gps.importlib, "import_module", lambda target: module)
    assert train_mppi_gps.main(["study", name, "--worker", "--out", "/fresh"]) == 7
    assert seen == [["--worker", "--out", "/fresh"]]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["study", "--help"],
        ["study", "guided-data", "--help"],
        ["study", "learnability", "--help"],
        ["study", "backoff", "--help"],
    ],
)
def test_public_help_commands(arguments):
    result = subprocess.run(
        [sys.executable, "-B", "scripts/train_mppi_gps.py", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_capsule_inspection_exposes_bulk_storage_and_all_original_hashes():
    report = inspect_run(ROOT / "results/adaptive_coupling_h4_1", verify=True)
    assert report["kind"] == "result_capsule"
    assert report["outcomes"]["status"] == "primary_negative"
    assert report["bulk_evidence"]["remote_copy_verified"] is False
    assert report["omitted_bulk_evidence"]
    assert report["hash_mismatches"] == report["errors"] == []
    assert all(item["verified"] for item in report["files"])


def test_capsule_binding_rejects_tampering_and_escape(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("original")
    binding = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": 8}
    errors = []
    check_binding(tmp_path, path.name, binding, errors)
    assert not errors
    path.write_text("tampered")
    check_binding(tmp_path, path.name, binding, errors)
    assert errors == [f"Hash mismatch: {path}"]
    errors = []
    check_binding(tmp_path, "../escaped", binding, errors)
    assert errors == ["Path escapes evidence root: ../escaped"]
    # Listed capsule copies must fail verification when omitted from an export;
    # intentionally external bulk evidence is reported separately.
    (tmp_path / "capsule.json").write_text(json.dumps({"files": {"missing.json": binding}}))
    report = inspect_run(tmp_path, verify=True)
    assert report["files"][0]["verified"] is False
    assert report["errors"] == [f"Missing bound evidence: {tmp_path / 'missing.json'}"]
    assert train_mppi_gps.main(["inspect", "--run", str(tmp_path), "--verify"]) == 1


def test_width_inspector_exposes_summary_and_gates_without_episode_arrays(tmp_path):
    run = tmp_path / "width"
    run.mkdir()
    (run / "config.json").write_text('{"schema":"walker-student-width-w1-v1"}')
    (run / "result.json").write_text(
        json.dumps(
            {
                "execution_complete": True,
                "status": "complete",
                "teacher_outcomes": {
                    "train": {"teacher": {"cost": 400, "success": True, "complete": True}}
                },
                "evaluations": {
                    "primary": {
                        "s0_w256_u10000": {
                            "summary": {
                                "episodes": 16,
                                "successes": 0,
                                "survived": 0,
                                "mean_steps": 80,
                                "mean_cost": 5000,
                            },
                            "episodes": [{"raw": "must not leak into compact metadata"}],
                        }
                    }
                },
                "gates": {
                    "primary": {
                        "reconstruction_screen": True,
                        "validation_guard": False,
                        "task_screen": False,
                    }
                },
            }
        )
    )
    report = inspect_run(run)
    assert report["outcomes"]["execution_complete"] is True
    assert report["teacher_outcomes"]["train"]["teacher"]["cost"] == 400
    assert report["student_outcomes"]["primary"]["s0_w256_u10000"]["episodes"] == 16
    assert report["gates"]["primary"]["validation_guard"] is False
    assert "must not leak" not in json.dumps(report)


def test_evaluation_inspector_describes_segments_and_inherited_costs_read_only(tmp_path):
    run = tmp_path / "evaluation"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(dict(schema="walker-student-source-s1.1-v1", budgets=dict(episodes=320)))
    )
    setup, episode = "evidence/loads", "evidence/episodes/000_primary_reference_00"
    for directory in (setup, episode):
        target = run / directory
        target.mkdir(parents=True)
        for name in ("raw.h5", "ledger.jsonl", "receipt.json"):
            (target / name).write_text("synthetic availability only")
    result = dict(
        schema="walker-student-source-s1.1-result-v1",
        execution_complete=False,
        status="stopped_partial",
        setup=dict(segment=setup),
        segments=[dict(segment=episode)],
        inherited_fit=dict(file="inputs/s1/fit_result.json", sha256="original-fit"),
        work=dict(completed_live_controls=1000, completed_optimizer_steps=0),
        inherited_work=dict(
            original_fit=dict(completed_optimizer_steps=12000),
            original_eval_known_prefix=dict(completed_live_controls=55607),
            original_eval_unknown_tail="Not quantified; not zero",
        ),
    )
    (run / "result.json").write_text(json.dumps(result))
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    report = inspect_run(run)
    assert report["costs"]["work"]["completed_optimizer_steps"] == 0
    assert report["costs"]["inherited_work"] == result["inherited_work"]
    assert report["provenance"]["inherited_fit"] == result["inherited_fit"]
    assert report["segmented_evidence"]["declared_episode_segments"] == 1
    assert report["segmented_evidence"]["planned_episodes"] == 320
    assert report["segmented_evidence"]["owner_exit_available"] is False
    assert set(report["bulk_evidence"]) == {
        f"{directory}/{name}"
        for directory in (setup, episode)
        for name in ("raw.h5", "ledger.jsonl", "receipt.json")
    }
    assert "evidence/raw.h5" not in report["missing_evidence"]
    assert "fresh S1.1 evaluation only" in format_inspection(report)
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    (run / episode / "raw.h5").unlink()
    assert episode + "/raw.h5" in inspect_run(run)["missing_evidence"]
    result["segments"].append(dict(segment="../escaped"))
    (run / "result.json").write_text(json.dumps(result))
    assert "escapes run" in inspect_run(run)["errors"][0]
