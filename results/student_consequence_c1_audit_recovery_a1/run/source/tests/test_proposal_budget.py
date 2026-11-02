"""H5.0 arithmetic and fake-call boundaries; no policies or native dynamics."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.gps.mppi_gps.proposal_budget import (
    Reservation,
    ReservationError,
    ReservationLedger,
    census,
    teacher_decision_bounds,
)


@pytest.fixture
def recipe():
    return json.loads(Path("configs/studies/teacher_proposal_h5_0.json").read_text())


def test_complete_census_recomputes_each_stage_and_strong_audit(recipe):
    result = census(recipe)
    assert result["per_episode"]["model_controls"] == 24_792_576
    assert result["per_episode"]["model_queries"] == 6002
    assert result["per_arm"]["model_controls"] == 446_266_368
    assert result["per_arm"]["guidance_forward_rows"] == 666_365_184
    assert result["total"]["model_controls"] == 1_338_799_104
    assert result["total"]["worker_neural_rows"] == 2_005_857_552
    assert result["strong_audit_neural_rows"] == 1_999_713_552
    assert result["combined_neural_rows"] == 4_005_571_104
    assert result["per_arm_combined_neural_rows"] == 1_334_590_368
    assert result["shared_parent_combined_neural_rows"] == 1_800_000
    assert result["worker_optimizer_steps"] == 24_000
    assert result["worker_backward_rows"] == 6_144_000
    assert result["standalone_live_controls"] == 402_000
    assert result["teacher_live_controls"] == 54_000
    assert result["total"]["physics_substeps"] == 5_357_020_416
    assert result["worker_episodes"] == 456
    assert result["checkpoint_loads"] == 12


def test_decision_sum_matches_episode_and_counts_initial_refinement(recipe):
    first = teacher_decision_bounds(recipe, 0)
    later = teacher_decision_bounds(recipe, 999)
    assert first["model_controls"] == (2 * 2 * 256 + 4) * 48
    assert later["model_controls"] == (2 * 256 + 4) * 48
    assert first["neural_rows"] == (2 * 256 + 515) * 48
    assert later["neural_rows"] == (256 + 515) * 48
    for key in ("model_controls", "neural_rows", "model_queries"):
        total = first[key] + 999 * later[key]
        expected = "guidance_forward_rows" if key == "neural_rows" else key
        assert total == census(recipe)["per_episode"][expected]
    assert first["prediction_queries"] == 4 and later["prediction_queries"] == 3
    with pytest.raises(ValueError):
        teacher_decision_bounds(recipe, 1000)


@pytest.mark.parametrize(
    "section,key", [("total", "model_controls"), ("per_arm", "worker_neural_rows")]
)
def test_recorded_census_cannot_hide_one_unbudgeted_unit(recipe, section, key):
    recipe["call_census"][section][key] -= 1
    with pytest.raises(ValueError, match="census differs"):
        census(recipe)


@pytest.mark.parametrize(
    "key", ["model_controls", "physics_substeps", "neural_rows", "audit_neural_rows"]
)
def test_global_budget_cannot_understate_any_paid_boundary(recipe, key):
    recipe["budgets"][key] -= 1
    with pytest.raises(ValueError, match="budget differs"):
        census(recipe)


def test_changed_roster_or_queries_require_versioning(recipe):
    for field, value in (("K", 255), ("H", 47), ("initial_iterations", True)):
        changed = deepcopy(recipe)
        changed["mppi"][field] = value
        with pytest.raises(ValueError):
            census(changed, validate_recorded=False)
    recipe["confirmation_seeds"][0] = recipe["primary_seeds"][0]
    with pytest.raises(ValueError, match="disjoint"):
        census(recipe)


def test_audit_mode_is_not_silently_treated_as_free(recipe):
    recipe["call_census"]["audit"]["mode"] = "pending"
    with pytest.raises(ValueError, match="audit mode"):
        census(recipe)


def test_arm_ceiling_includes_the_strong_audit_not_just_worker(recipe):
    recipe["budgets"]["per_arm_neural_row_ceiling"] = 700_000_000
    with pytest.raises(ValueError, match="worker plus strong audit"):
        census(recipe)


def test_pre_call_rejection_does_not_invoke_external_work():
    calls = []
    ledger = ReservationLedger(dict(neural_rows=4))
    with pytest.raises(ReservationError, match="Pre-call"):
        ledger.run("too_large", dict(neural_rows=5), lambda: calls.append(1), phase="audit")
    snap = ledger.snapshot()
    assert calls == []
    assert snap["requested"]["total"]["neural_rows"] == 0
    assert snap["errors"]["budget_errors"] == 1
    assert snap["stopped"] is True
    with pytest.raises(ReservationError, match="stopped"):
        ledger.run("retry", dict(neural_rows=1), lambda: calls.append(2), phase="audit")
    assert calls == []


def test_exact_boundary_and_all_scopes_are_checked_before_mutation():
    limits = dict(model_controls=5, neural_rows=5)
    ledger = ReservationLedger(
        limits,
        per_arm_limits={"proposal_guided": limits | dict(model_controls=3)},
        phase_limits={"teacher": limits},
    )
    assert (
        ledger.run(
            "first", dict(model_controls=3), lambda: 7, arm="proposal_guided", phase="teacher"
        )
        == 7
    )
    before = ledger.snapshot()
    with pytest.raises(ReservationError, match="arm:proposal_guided"):
        ledger.run(
            "second",
            dict(model_controls=1, neural_rows=2),
            lambda: None,
            arm="proposal_guided",
            phase="teacher",
        )
    after = ledger.snapshot()
    assert before["requested"] == after["requested"]
    assert before["completed"] == after["completed"]


def test_shared_parent_work_is_separate_from_arm_and_still_global():
    limits = dict(optimizer_steps=4)
    ledger = ReservationLedger(
        limits,
        per_arm_limits={"independent": limits},
        phase_limits={"shared_parent": limits, "child": limits},
    )
    ledger.run("parent_once", dict(optimizer_steps=2), lambda: None, phase="shared_parent")
    ledger.run("child", dict(optimizer_steps=2), lambda: None, phase="child", arm="independent")
    snap = ledger.snapshot()
    assert snap["requested"]["total"]["optimizer_steps"] == 4
    assert snap["requested"]["arm:independent"]["optimizer_steps"] == 2
    assert snap["requested"]["phase:shared_parent"]["optimizer_steps"] == 2


def test_failed_paid_call_keeps_full_reservation_and_stops_every_scope():
    ledger = ReservationLedger(dict(model_controls=10, model_queries=2))

    def broken():
        raise RuntimeError("native partial call")

    with pytest.raises(RuntimeError, match="native partial"):
        ledger.run("model", dict(model_controls=6, model_queries=1), broken, phase="teacher")
    snap = ledger.snapshot()
    assert snap["requested"]["total"] == dict(model_controls=6, model_queries=1)
    assert snap["completed"]["total"] == dict(model_controls=0, model_queries=0)
    assert snap["unknown_partial"]["total"] == dict(model_controls=6, model_queries=1)
    assert snap["errors"]["external_errors"] == 1
    with pytest.raises(ReservationError, match="stopped"):
        ledger.run("replacement", dict(model_controls=1), lambda: None, phase="teacher")


def test_import_failure_is_paid_and_separate_from_external_or_budget_errors():
    ledger = ReservationLedger(dict(imports=1))

    def missing():
        raise ImportError("deliberately missing synthetic module")

    with pytest.raises(ImportError):
        ledger.run("import", dict(imports=1), missing, phase="startup")
    snap = ledger.snapshot()
    assert snap["unknown_partial"]["total"]["imports"] == 1
    assert snap["errors"] == dict(
        input_errors=0, budget_errors=0, import_errors=1, external_errors=0
    )


@pytest.mark.parametrize("value", [-1, True, 1.0, float("inf")])
def test_invalid_work_is_rejected_before_operation(value):
    ledger = ReservationLedger(dict(neural_rows=4))
    calls = []
    with pytest.raises(ReservationError):
        ledger.run("invalid", dict(neural_rows=value), lambda: calls.append(1), phase="fit")
    assert calls == [] and ledger.snapshot()["errors"]["input_errors"] == 1


def test_unknown_work_scope_and_reused_request_are_not_retries():
    ledger = ReservationLedger(
        dict(storage_bytes=8), phase_limits={"report": dict(storage_bytes=8)}
    )
    ticket = ledger.reserve("one", dict(storage_bytes=8), phase="report")
    ledger.complete(ticket)
    with pytest.raises(ReservationError, match="no retry"):
        ledger.reserve("one", dict(storage_bytes=1), phase="report")
    assert ledger.snapshot()["requested"]["total"]["storage_bytes"] == 8
    with pytest.raises(ReservationError, match="settled"):
        ledger.complete(ticket)


@pytest.mark.parametrize("work", [{}, {"model_queries": 0}, {"unknown": 1}])
def test_zero_or_undeclared_external_work_has_no_free_call(work):
    ledger = ReservationLedger(dict(model_queries=1))
    calls = []
    with pytest.raises(ReservationError):
        ledger.run("unmetered", work, lambda: calls.append(1), phase="teacher")
    assert calls == [] and ledger.snapshot()["errors"]["input_errors"] == 1


def test_altered_ticket_cannot_reassign_arm_or_phase_ownership():
    ledger = ReservationLedger(dict(neural_rows=1))
    ticket = ledger.reserve("one", dict(neural_rows=1), phase="fit")
    altered = Reservation(ticket.identifier, ticket.name, "proposal_guided", ticket.phase)
    with pytest.raises(ReservationError, match="altered"):
        ledger.complete(altered)
    ledger.complete(ticket)


def test_time_budget_is_checked_before_call_and_after_paid_call():
    now = [5.0]

    def clock():
        return now[0]

    ledger = ReservationLedger(dict(seconds=2), wall_seconds=2, clock=clock)
    calls = []
    now[0] = 7.1
    with pytest.raises(ReservationError, match="Wall-time"):
        ledger.run("too_late", dict(seconds=1), lambda: calls.append(1), phase="fit")
    assert calls == []
    now[0] = 0.0
    ledger = ReservationLedger(dict(model_queries=1), wall_seconds=2, clock=clock)

    def returns_late():
        now[0] = 3.0
        return 2

    with pytest.raises(ReservationError, match="Wall-time"):
        ledger.run("late_return", dict(model_queries=1), returns_late, phase="teacher")
    snap = ledger.snapshot()
    assert snap["completed"]["total"]["model_queries"] == 1
    assert snap["unknown_partial"]["total"]["model_queries"] == 0


def test_snapshots_cannot_mutate_totals_or_grant_hdf_storage_proof():
    ledger = ReservationLedger(dict(storage_bytes=8))
    ledger.run("bytes", dict(storage_bytes=8), lambda: None, phase="report")
    snap = ledger.snapshot()
    snap["requested"]["total"]["storage_bytes"] = 0
    assert ledger.snapshot()["requested"]["total"]["storage_bytes"] == 8
    assert ledger.snapshot()["storage_proof"] is False


def test_import_is_stdlib_only_even_with_optional_dependencies_blocked():
    code = """
import sys
from src.gps.mppi_gps import proposal_budget
for prefix in ('numpy','scipy','torch','mujoco','h5py','matplotlib','src.envs','src.mppi'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules),prefix
print('stdlib-only')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "stdlib-only"


def test_streaming_event_sink_does_not_retain_unbounded_events_or_expose_mutable_counters():
    events = []
    ledger = ReservationLedger(dict(model_queries=2), event_sink=events.append, retain_events=False)
    ledger.run("one", dict(model_queries=1), lambda: None, phase="teacher")
    ledger.run("two", dict(model_queries=1), lambda: None, phase="teacher")
    assert [row["event"] for row in events] == ["reserved", "completed", "reserved", "completed"]
    assert "events" not in ledger.snapshot()
    assert ledger.snapshot(include_events=True)["events"] == []
    events[0]["work"]["model_queries"] = 1000
    assert ledger.snapshot()["requested"]["total"]["model_queries"] == 2


def test_event_sink_failure_stops_before_external_call_and_keeps_the_reservation():
    calls = []

    def broken_sink(_):
        raise OSError("synthetic durable ledger failure")

    ledger = ReservationLedger(dict(model_queries=1), event_sink=broken_sink, retain_events=False)
    with pytest.raises(ReservationError, match="event sink failed"):
        ledger.run("one", dict(model_queries=1), lambda: calls.append(1), phase="teacher")
    snap = ledger.snapshot()
    assert not calls and snap["stopped"]
    assert snap["requested"]["total"]["model_queries"] == 1
    assert snap["completed"]["total"]["model_queries"] == 0
    assert snap["unknown_partial"]["total"]["model_queries"] == 0
    assert snap["errors"]["evidence_errors"] == 1
