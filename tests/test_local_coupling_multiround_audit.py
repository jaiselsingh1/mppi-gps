"""Pure saved-evidence L3 safeguards; no physics, neural inference or fitting."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits import local_coupling_multiround as module


def outcome(*, controls=300, survived=True, mae=0.2, cost=100.0):
    return dict(controls=controls, full_survival=survived, velocity_mae=mae, complete_cost=cost)


def test_admission_uses_actual_task_behavior_not_agreement_or_fitting():
    good = outcome()
    good.update(action_mse=1000, spectral_distance=1000, local_walking=False)
    assert module.admission_decision(good, 100)
    assert not module.admission_decision(outcome(mae=0.31), 100)
    assert not module.admission_decision(outcome(controls=299, survived=False), 100)
    assert not module.admission_decision(outcome(cost=110.0001), 100)
    assert module.admission_decision(outcome(cost=110), 100)


def transition(*, admitted=False, same=False):
    old, attempt = {"path": "/old"}, {"path": "/attempt"}
    previous = dict(manifest=old, last_accepted_round=1, experience_fingerprint="old-experience")
    proposed = "old-experience" if same else "new-experience"
    saved = dict(
        admitted=admitted,
        fallback=not admitted,
        previous=old,
        attempt=attempt,
        chosen=attempt if admitted else old,
        last_accepted_round=3 if admitted else 1,
        age=0 if admitted else 2,
        previous_experience_fingerprint="old-experience",
        chosen_experience_fingerprint=proposed if admitted else "old-experience",
        new_experience=admitted and not same,
        new_rows=300 if admitted and not same else 0,
    )
    return dict(
        round_index=3,
        outcome=outcome() if admitted else outcome(controls=80, survived=False),
        anchor_cost=100,
        previous=previous,
        attempt=attempt,
        experience_fingerprint=proposed,
        saved=saved,
    )


def test_fallback_reuses_old_source_and_cannot_count_fresh_rows():
    audit = module.LearningAudit(neural_rows=0)
    case = transition()
    measured = module.retention_transition(audit, **case)
    assert measured["manifest"] == case["previous"]["manifest"]
    assert measured["last_accepted_round"] == 1
    case["saved"]["new_rows"] = 300
    with pytest.raises(module.AuditFailure, match="new_rows"):
        module.retention_transition(audit, **case)


def test_admitted_identical_episode_is_not_new_experience():
    case = transition(admitted=True, same=True)
    measured = module.retention_transition(module.LearningAudit(neural_rows=0), **case)
    assert measured["last_accepted_round"] == 3
    assert measured["manifest"] == case["attempt"]
    assert case["saved"]["new_rows"] == 0
    case["saved"]["new_experience"] = True
    with pytest.raises(module.AuditFailure, match="new_experience"):
        module.retention_transition(module.LearningAudit(neural_rows=0), **case)


def test_canonical_experience_hash_is_pre_action_and_includes_torques():
    raw = dict(states=np.zeros((301, 19)), actions=np.zeros((300, 6)))
    old = module._experience(raw)
    raw["states"][-1, 0] = 4
    assert module._experience(raw) == old
    raw["actions"][9, 2] = 0.1
    assert module._experience(raw) != old


def panels(*, positive=0, initial_positive=0, independent=0):
    result = {}
    for stream in range(3):
        for panel in ("primary", "confirmation"):
            result[(stream, panel)] = {
                ("independent", "initial"): dict(walking=0, full_survival=0),
                ("positive", "initial"): dict(
                    walking=initial_positive, full_survival=initial_positive
                ),
                ("independent", "final"): dict(walking=independent, full_survival=independent),
                ("positive", "final"): dict(walking=positive, full_survival=max(positive, 4)),
            }
    return result


def test_zero_walking_with_retention_remains_a_complete_negative_outcome():
    value = module.final_walking_comparison(panels())
    assert not value["passed"] and value["winning_streams"] == []
    assert all(row["difference_in_gain"] == 0 for row in value["details"])
    assert value["nonregression_all"]


def test_final_walking_gain_needs_same_streams_both_panels_and_difference_in_gain():
    assert module.final_walking_comparison(panels(positive=4, independent=1))["passed"]
    assert not module.final_walking_comparison(
        panels(positive=5, initial_positive=3, independent=2)
    )["passed"]
    value = panels(positive=4)
    value[(0, "primary")][("positive", "final")]["walking"] = 0
    value[(1, "confirmation")][("positive", "final")]["walking"] = 0
    assert not module.final_walking_comparison(value)["passed"]


def test_partial_reset_panels_are_not_zero_success_panels():
    with pytest.raises(module.AuditFailure, match="sixteen"):
        module.summarize_panel([outcome() for _ in range(15)])


def test_terminal_boundary_without_executed_action_is_missing():
    audit = module.LearningAudit(neural_rows=0)
    assert not module.observed_anchor(
        audit, dict(control=50, status="MISSING_AFTER_EXECUTED_PREFIX"), 50
    )
    assert module.observed_anchor(audit, dict(control=0, status="OBSERVED"), 50)
    with pytest.raises(module.AuditFailure, match="not imputed"):
        module.observed_anchor(
            audit, dict(control=50, status="MISSING_AFTER_EXECUTED_PREFIX", gap_action_mse=0), 50
        )
    with pytest.raises(module.AuditFailure, match="prefix"):
        module.observed_anchor(audit, dict(control=100, status="OBSERVED"), 50)


def test_seven_round_prefix_cannot_certify_eight_even_if_status_claims_complete(monkeypatch):
    audit = module.LearningAudit(neural_rows=0)
    saved = dict(
        schema="local-coupling-multiround-stream-1",
        status="COMPLETED_ALL_ROUNDS",
        stream=0,
        planned_rounds=8,
        rounds=[dict(round=i) for i in range(1, 8)],
    )
    monkeypatch.setattr(audit, "json", lambda _: saved)
    with pytest.raises(module.AuditFailure, match="Every planned round"):
        module._stream(
            audit,
            Path("/unused"),
            stream_root=Path("/unused"),
            stream=0,
            initial={},
            anchors={},
            prepared={},
            config=dict(rounds=8),
            config_binding={},
            out=Path("/unused"),
        )
    assert audit.completed_rows == 0


def test_failed_original_prefix_and_counts_survive_independent_audit_failure(tmp_path):
    base = tmp_path / "run"
    (base / "worker").mkdir(parents=True)
    original = dict(
        schema="local-coupling-multiround-l3-result-1",
        status="FAILED_STOPPED",
        error="resource exhausted",
        planned_rounds=8,
        accounting=dict(counts=dict(requested_model_controls=7)),
    )
    source = base / "worker/result.json"
    source.write_text(json.dumps(original))
    before = source.read_bytes()
    receipt = module.audit_run(base, tmp_path / "audit", neural_rows=0)
    assert not receipt["passed"]
    findings = json.loads((tmp_path / "audit/findings.json").read_text())
    assert findings["original_status"] == "FAILED_STOPPED"
    assert findings["declared_paid_counts"]["requested_model_controls"] == 7
    assert source.read_bytes() == before
    assert receipt["completed_neural_rows"] == receipt["native_controls"] == 0


def test_audit_reserves_neural_budget_before_any_deployed_head_execution():
    audit = module.LearningAudit(neural_rows=0)
    with pytest.raises(module.AuditFailure, match="budget"):
        audit.predict({}, np.zeros((1, 17), np.float32))
    assert audit.requested_rows == audit.completed_rows == audit.unknown_rows == 0


def test_importing_independent_auditor_constructs_no_simulator_or_policy():
    code = (
        "import sys; import src.gps.mppi_gps.audits.local_coupling_multiround; "
        "assert not any(name in sys.modules for name in "
        "('mujoco','torch','src.mppi.mppi','src.gps.mppi_gps.local_coupling_controller',"
        "'src.gps.mppi_gps.local_coupling_learning'))"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
