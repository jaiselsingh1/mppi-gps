"""Synthetic summary records only: zero simulator or model calls."""
import copy
import json

import pytest

from scripts.diagnostics.audit_walker_qualification import audit_directory, audit_rows


def panel(expected=64):
    rows = []
    for arm in ("A", "B1", "B2", "C"):
        for i in range(expected):
            K = 256 if arm == "C" else 32
            n = 1000
            ledger = dict(executed_controls=n, candidate_controls=K*48*(n+1), planner_calls=n,
                          value_prediction_rows=123)
            if arm in ("A", "B1"):
                ledger["verification_controls"] = 48*n
                ledger["one_step_verification_controls"] = 2
            if arm == "A":
                ledger["memory_controls"] = 8*48*n
            rows.append(dict(job_id=f"{arm}:{i}", steps=n, survived=True, terminated=False,
                cost=350., velocity_mae=.15, seconds=1.,
                reset_profile="angle_velocity_perturbed" if i%2 else "standard", ledger=ledger))
    return rows


def test_complete_four_arm_panel_passes_without_counting_planning_calls_as_controls():
    rows = panel()
    result = audit_rows(rows, finished=True)
    assert result["status"] == "PASS"
    assert result["all_256_completion"]
    assert result["observed_C_rows"] == result["unique_C_rows"] == 64
    assert result["recorded_total_controls"] == sum(
        sum(row["ledger"].get(k, 0) for k in result["recorded_slot_totals"]) for row in rows)
    assert result["per_arm"]["A"]["one_step_verification_controls"] == 128
    assert result["per_arm"]["C"]["candidate_controls"] == 64*256*48*1001
    assert result["additional_physics_controls"] == 0
    assert not result["source_identity_verified"] and not result["raw_acquisition_states_certified"]


def test_pending_best_subset_never_passes_even_with_perfect_observed_survival():
    rows = [r for r in panel() if r["job_id"].startswith("C:")][:12]
    result = audit_rows(rows, finished=False)
    assert result["status"] == "PENDING"
    assert not result["qualification_passed"]
    assert result["per_arm"]["C"]["observed_survival_rate"] == 1.
    assert result["missing_job_ids"]
    assert audit_rows(panel(), finished=False)["status"] == "PENDING"


def test_missing_fall_cannot_be_hidden_by_completed_marker():
    rows = panel()
    rows = [r for r in rows if r["job_id"] != "C:63"]
    result = audit_rows(rows, finished=True)
    assert result["status"] == "FAIL"
    assert result["missing_job_ids"] == ["C:63"]
    assert result["unique_C_rows"] == 63


def test_duplicate_is_charged_but_does_not_replace_or_add_an_outcome():
    rows = panel()
    duplicate = copy.deepcopy(next(r for r in rows if r["job_id"] == "C:0"))
    rows.append(duplicate)
    result = audit_rows(rows, finished=True)
    assert result["status"] == "FAIL"
    assert result["duplicate_job_ids"] == {"C:0": 2}
    assert result["observed_C_rows"] == 65 and result["unique_C_rows"] == 64
    assert result["per_arm"]["C"]["executed_controls"] == 65000


@pytest.mark.parametrize("change", ["counter", "declared"])
def test_repaired_arm_rejects_accidental_k32_default(change):
    rows = panel()
    bad = next(r for r in rows if r["job_id"] == "C:0")
    if change == "counter":
        bad["ledger"]["candidate_controls"] = 32*48*1001
    else:
        bad["mppi_config"] = dict(K=32, H=48)
    result = audit_rows(rows, finished=True)
    assert result["status"] == "FAIL"
    assert any(e["category"] == "raw_recipe" for e in result["validation_errors"])


def test_failed_c_episode_stays_in_mae_and_all_paid_controls():
    rows = panel()
    bad = next(r for r in rows if r["job_id"] == "C:7")
    bad.update(steps=100, survived=False, terminated=True, velocity_mae=12., cost=3100.)
    bad["ledger"].update(executed_controls=100, planner_calls=100,
                         candidate_controls=256*48*101)
    result = audit_rows(rows, finished=True)
    assert result["status"] == "FAIL"  # Survival63/64passes; retained MAE fails.
    repaired = result["per_arm"]["C"]
    assert repaired["velocity_mae"] == pytest.approx((63*.15+12.)/64)
    assert repaired["executed_controls"] == 63100
    assert repaired["candidate_controls"] == (63*1001+101)*256*48
    assert repaired["falls"][0]["recorded_controls"] == 100+256*48*101
    assert repaired["falls"][0]["cost"] == 3100.
    assert repaired["falls"][0]["job_id"] == "C:7"


@pytest.mark.parametrize("field,value", [("cost", float("nan")), ("velocity_mae", float("inf")), ("steps", 2.5)])
def test_nonfinite_or_invalid_outcomes_fail(field, value):
    rows = panel()
    rows[-1][field] = value
    assert audit_rows(rows, finished=True)["status"] == "FAIL"


def test_any_arm_invalid_ledger_blocks_completed_qualification():
    rows = panel()
    rows[0]["ledger"]["memory_controls"] = -1
    assert audit_rows(rows, finished=True)["status"] == "FAIL"


def test_unexpected_rows_do_not_disappear_from_costs_and_invalid_falls_are_json_safe():
    rows = panel()
    baseline = audit_rows(rows, finished=True)["recorded_total_controls"]
    rows.append(dict(job_id="unknown:0", ledger=dict(executed_controls=3, candidate_controls=17)))
    bad = next(r for r in rows if r["job_id"] == "C:0")
    bad.update(terminated=True, survived=False, cost=float("nan"))
    result = audit_rows(rows, finished=True)
    assert result["recorded_total_controls"] == baseline+20
    assert result["unassigned_recorded_slot_totals"]["candidate_controls"] == 17
    assert result["per_arm"]["C"]["falls"][0]["cost"] is None
    json.dumps(result, allow_nan=False)


def test_directory_cli_api_uses_safe_json_and_new_timestamp_artifacts(tmp_path):
    (tmp_path/"records.json").write_text(json.dumps(panel()))
    (tmp_path/"status.json").write_text(json.dumps(dict(finished=True, terminated=False)))
    first, a = audit_directory(tmp_path)
    second, b = audit_directory(tmp_path)
    assert a != b and a.exists() and b.exists()
    assert first["status"] == second["status"] == "PASS"
    assert first["input_sha256"] == second["input_sha256"]


def test_terminated_process_without_finish_marker_is_explicit_failure(tmp_path):
    (tmp_path/"records.json").write_text("[]")
    (tmp_path/"status.json").write_text(json.dumps(dict(finished=False, terminated=True)))
    result, _ = audit_directory(tmp_path)
    assert result["status"] == "FAIL"
    assert result["process_terminated_without_finish_marker"]
