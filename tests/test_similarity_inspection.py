import json

from src.gps.mppi_gps.inspection import format_inspection, inspect_run


def test_closed_similarity_metadata_exposes_outcomes_without_opening_cases(tmp_path):
    (tmp_path / "worker").mkdir()
    for name, value in {
        "config.json": {
            "schema": "walker-similarity-j1-execution-v1",
            "limits": {"worker_stage_wall_seconds": 28800},
        },
        "manifest.json": {"files": []},
        "input_manifest.json": {"files": {}},
        "worker_result.json": {
            "status": "CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
            "cases": 32,
            "eligible_cases": 32,
            "declared_source_groups": 16,
            "acquired_source_groups": 8,
            "objective_passed": False,
            "protected_test_revealed": False,
            "costs": {"completed": {"optimizer_steps": 64000}},
        },
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    # Opening partial/future case labels would be a bug: it is invalid JSON.
    (tmp_path / "worker/cases.json").write_text("DO NOT READ PROTECTED LABELS")
    result = inspect_run(tmp_path)
    assert result["result_file"] == "worker_result.json"
    assert result["student_outcomes"]["acquired_source_groups"] == 8
    assert result["student_outcomes"]["declared_source_groups"] == 16
    assert result["gates"]["objective_passed"] is False
    assert result["gates"]["protected_test_revealed"] is False
    assert result["costs"]["worker"]["completed"]["optimizer_steps"] == 64000
    assert "fixed-update" in format_inspection(result)
    assert "result.json" not in result["missing_evidence"]
