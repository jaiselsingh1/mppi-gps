"""Synthetic arithmetic only: no actor, simulator, or model imports.

The single forward test executes three rows of a fixed two-unit NumPy network.
All other fixtures describe recorded arrays without executing a controller.
"""
from copy import deepcopy
import math

import numpy as np
import pytest

from scripts.diagnostics import audit_walker_owned_signal as audit


def _one_control_suffix(*, fall=True):
    states = np.zeros((2, 19), dtype=np.float64)
    states[:, 2] = [1.25, .8 if fall else 1.25]
    states[:, 10] = 1.5
    states[1, 0] = .008
    warmstarts = np.zeros((2, 9), dtype=np.float64)
    observations = np.zeros((2, 17), dtype=np.float64)
    observations[:, 0] = states[:, 2]
    observations[:, 8] = 1.5
    # Target speed, zero vertical speed/angle/control leave only 5+20 on fall.
    raw_cost = 25. if fall else 0.
    trace = dict(states=states, warmstarts=warmstarts, observations=observations,
        actions=np.zeros((1, 6)), raw_costs=np.array([raw_cost]),
        cost_components=np.array([[0., 0., 0., 5. if fall else 0., 0., 20. if fall else 0.]]))
    context = dict(id="root0", remaining=3, role="root", ancestry={"reset_seed": 7},
        state=states[0].copy(), warmstart=warmstarts[0].copy(),
        snapshot_digest=audit.snapshot_hash(states[0], warmstarts[0]))
    result = dict(executed_steps=1, gamma=.5, remaining=3, complete=True, cutoff=False,
        continuation_owner="candidate", target_kind="complete_changed_actor_return",
        snapshot_digest=context["snapshot_digest"], terminated=fall, truncated=False,
        executed_cost=raw_cost, absorbing_cost=3.75 if fall else 0.,
        cost=28.75 if fall else 0., absorbing_steps=2 if fall else 0, unhealthy_cost=5.)
    return trace, result, context


def test_strict_height_boundary_charges_one_fall_and_finite_absorption():
    trace, result, context = _one_control_suffix()
    checked = audit.audit_trace(trace, result, context)
    assert not audit.healthy(trace["states"][-1])  # z == .8 is a fall.
    assert checked["terminated"] is True
    assert checked["executed_steps"] == 1
    assert checked["raw_costs"].tolist() == [25.]
    assert checked["cost_components"].tolist() == [[0., 0., 0., 5., 0., 20.]]
    # Two unexecuted controls: .5*5 + .5**2*5, no second terminal charge.
    assert checked["absorption"] == 3.75
    assert checked["cost"] == 28.75


def test_healthy_early_cutoff_cannot_supply_complete_candidate_credit():
    trace, result, context = _one_control_suffix(fall=False)
    with pytest.raises(ValueError, match="healthy incomplete suffix"):
        audit.audit_trace(trace, result, context)


def test_duplicate_terminal_charge_fails_even_when_all_reported_totals_agree():
    trace, result, context = _one_control_suffix()
    trace["cost_components"][0, -1] = 40.
    trace["raw_costs"][0] = 45.
    result.update(executed_cost=45., cost=48.75)
    with pytest.raises(ValueError, match="raw cost-component mismatch"):
        audit.audit_trace(trace, result, context)


@pytest.mark.parametrize("call_returned", [False, True])
def test_failed_control_attempt_is_charged_without_completed_transition(call_returned):
    _, _, context = _one_control_suffix()
    ref, owner = "failed-query", "synthetic-owner"
    task = dict(frame_skip=4, cost_style="target_velocity", target_velocity=1.5,
        vel_cost_weight=2., vertical_velocity_cost_weight=.5, angle_cost_weight=.1,
        unhealthy_cost_weight=5., ctrl_cost_weight=.001, terminal_unhealthy_weight=20.,
        apply_terminal_cost_on_done=True)
    events = [
        dict(kind="query_begin", owner_hash=owner, context_id=context["id"],
             ancestry=context["ancestry"], budget_controls=3, task_config=task),
        dict(kind="control_attempt", pre=dict(state=context["state"], warmstart=context["warmstart"]),
             action=np.zeros(6)),
        dict(kind="control_error", attempt_id=f"{ref}:1", call_returned=call_returned),
        dict(kind="query_end", status="failed"),
    ]
    for index, event in enumerate(events):
        event.update(id=f"{ref}:{index}", query_id=ref,
                     controls=int(event["kind"] == "control_attempt"))
    checked = audit.audit_journal(events, query_ref=ref, owner=owner, context=context)
    assert checked["charged_requested_controls"] == 1
    assert checked["completed_controls"] == 0
    assert checked["error_controls"] == 1
    assert checked["inflight_or_interrupted_controls"] == 0
    assert checked["confirmed_returned_controls"] == int(call_returned)
    assert checked["unknown_partial_error_controls"] == int(not call_returned)
    assert checked["query_ended"] is True
    assert checked["complete"] is False


def test_roles_have_equal_weight_with_unequal_context_counts():
    owner, gamma = "candidate-owner", .5
    contexts = [dict(id=name, role=role, remaining=1, ancestry={"reset_seed": 17})
                for name, role in (("r", "root"), ("g0", "guide"), ("g1", "guide"))]
    refs = ["q-r", "q-g0", "q-g1"]
    results = {ref: dict(owner_hash=owner, context_id=context["id"], remaining=1,
                         gamma=gamma, cost=cost)
               for ref, context, cost in zip(refs, contexts, (5., 10., 20.))}
    checked = audit.score_panel(refs, owner, contexts, results, gamma)
    np.testing.assert_array_equal(checked["normalized_returns"], [1., 2., 4.])
    assert checked["role_scores"] == {"guide": 3., "root": 1.}
    assert checked["score"] == 2.
    assert checked["score"] != pytest.approx(7. / 3.)
    assert checked["root_clusters"] == {"17": 2.}


def _fixed_small_network():
    arrays = {
        "fc1.weight": np.zeros((2, 17), dtype=np.float32),
        "fc1.bias": np.zeros(2, dtype=np.float32),
        "fc2.weight": np.eye(2, dtype=np.float32),
        "fc2.bias": np.zeros(2, dtype=np.float32),
        "fc_mean.weight": np.array([[1., 0.], [0., 1.], [1., 1.],
                                    [-1., 0.], [0., -1.], [.5, -.25]], dtype=np.float32),
        "fc_mean.bias": np.array([0., 0., .1, 0., 0., -.2], dtype=np.float32),
        "action_low": np.full(6, -1., dtype=np.float32),
        "action_high": np.ones(6, dtype=np.float32),
        "action_scale": np.ones(6, dtype=np.float32),
        "action_bias": np.zeros(6, dtype=np.float32),
        "obs_mean": np.zeros(17, dtype=np.float32),
        "obs_std": np.ones(17, dtype=np.float32),
    }
    arrays["fc1.weight"][0, 0] = 1.
    arrays["fc1.weight"][1, 1] = 1.
    return arrays


def test_fixed_small_numpy_network_verifies_three_synthetic_action_rows():
    arrays = _fixed_small_network()
    assert audit.validate_actor(arrays, hidden=2).shape == (60,)
    obs = np.zeros((3, 17), dtype=np.float64)
    obs[:, :2] = [[0., 0.], [.5, -.25], [-.5, .75]]
    actions = audit.numpy_actions(arrays, obs, batch_size=2)
    # Independent scalar closed forms for this fixed network, not another actor.
    expected = []
    for x, y in obs[:, :2]:
        u, v = math.tanh(math.tanh(x)), math.tanh(math.tanh(y))
        expected.append([math.tanh(u), math.tanh(v), math.tanh(u+v+.1),
                         math.tanh(-u), math.tanh(-v), math.tanh(.5*u-.25*v-.2)])
    np.testing.assert_allclose(actions, expected, rtol=2e-6, atol=5e-8)
    assert actions.shape == (3, 6)
    assert actions.dtype == np.dtype("float32")
    assert np.all(np.abs(actions) <= 1.)


@pytest.mark.parametrize("name", ["action_low", "action_high", "action_scale",
                                  "action_bias", "obs_mean", "obs_std"])
def test_changed_action_or_observation_buffer_is_rejected(name):
    arrays = deepcopy(_fixed_small_network())
    arrays[name][0] += np.float32(.25)
    with pytest.raises(ValueError, match="actor buffers/identity transform mismatch"):
        audit.validate_actor(arrays, hidden=2)


def test_selection_filters_root_regression_before_taking_best_mixture_score():
    checked = audit.select_feasible([1., .9, .95], [1., 1.01, .99])
    np.testing.assert_array_equal(checked["feasible_flags"], [True, False, True])
    assert checked["unconstrained_selected_index"] == 1
    assert checked["selected_index"] == 2


def test_selection_keeps_base_first_when_feasible_scores_tie():
    checked = audit.select_feasible([1., 1., 1.], [1., .99, 1.])
    np.testing.assert_array_equal(checked["feasible_flags"], [True, True, True])
    assert checked["selected_index"] == 0
    assert checked["unconstrained_selected_index"] == 0


def test_missing_config_still_inventories_durable_paid_work_without_inference(tmp_path, monkeypatch):
    import json

    def forbidden_forward(*args, **kwargs):
        pytest.fail("Missing configuration must not trigger action verification")

    monkeypatch.setattr(audit, "numpy_actions", forbidden_forward)
    query = tmp_path / "queries" / "unfinished-query"
    query.mkdir(parents=True)
    # These intentionally lack result/owner metadata: paid attempts remain
    # visible even when evidence cannot proceed to semantic validation.
    (query / "journal.jsonl").write_text(json.dumps({"kind": "control_attempt"}) + "\n")
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"kind": "query_reserved", "remaining": 3}) + "\n")
    (tmp_path / "inferences.jsonl").write_text(json.dumps({"kind": "inference_attempt"}) + "\n")
    checked = audit.audit(tmp_path, verify_actions=True,
                          audit_inference_journal=tmp_path / "inventory-audit.inferences.jsonl")
    assert checked["ready"] is False
    assert checked["phase"] == "PENDING"
    assert any("config.json" in reason for reason in checked["pending"])
    assert checked["work"]["observed_requested_control_slots"] == 1
    assert checked["work"]["charged_requested_controls"] == 1
    assert checked["work"]["observed_reserved_queries"] == 1
    assert checked["work"]["observed_reserved_controls"] == 3
    assert checked["work"]["observed_inference_attempt_rows"] == 1
    assert checked["action_verification_rows"] == 0


def test_action_verification_requires_explicit_new_journal_before_auditor_runs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid verification journal must reject before audit or forward execution")

    monkeypatch.setattr(audit.Auditor, "run", forbidden)
    monkeypatch.setattr(audit, "numpy_actions", forbidden)
    with pytest.raises(ValueError, match="NEW durable audit inference journal"):
        audit.audit(tmp_path, verify_actions=True)
    existing = tmp_path / "existing.inferences.jsonl"
    existing.write_text("existing paid-work evidence\n")
    with pytest.raises(FileExistsError):
        audit.audit(tmp_path, verify_actions=True, audit_inference_journal=existing)
    assert existing.read_text() == "existing paid-work evidence\n"


def test_cli_existing_inference_journal_rejects_before_audit_even_without_json_output(tmp_path, monkeypatch):
    output = tmp_path / "audit-result.json"
    journal = output.with_suffix(".inferences.jsonl")
    journal.write_text("preserved interrupted verification\n")

    def forbidden(*args, **kwargs):
        pytest.fail("CLI must reject an existing verification journal before invoking audit")

    monkeypatch.setattr(audit, "audit", forbidden)
    monkeypatch.setattr("sys.argv", ["audit_walker_owned_signal", str(tmp_path),
                                     "--verify-actions", "--out", str(output)])
    with pytest.raises(ValueError, match="audit inference journal already exists"):
        audit.main()
    assert not output.exists()
    assert journal.read_text() == "preserved interrupted verification\n"


def test_interrupted_verification_keeps_two_fsynced_requested_rows_and_enforces_cap(tmp_path, monkeypatch):
    import json

    path = tmp_path / "synthetic-verification.inferences.jsonl"
    durable_snapshots = []
    original_fsync = audit.os.fsync

    def observe_fsync(descriptor):
        original_fsync(descriptor)
        durable_snapshots.append(path.read_text() if path.exists() else "")

    def forbidden_forward(*args, **kwargs):
        pytest.fail("Reserving synthetic verification rows must not perform forward arithmetic")

    monkeypatch.setattr(audit.os, "fsync", observe_fsync)
    monkeypatch.setattr(audit, "numpy_actions", forbidden_forward)
    journal = audit.VerificationJournal(path, cap=2)
    try:
        identifier = journal.reserve("synthetic-owner", "interrupted-fixture", np.zeros((2, 17)))
        assert identifier == "v0000000"
        assert journal.rows == 2 and journal.attempts == 1
        # Inspect while still open: the requested batch has already been
        # flushed and fsynced before any forward or completion can occur.
        durable = [json.loads(line) for line in durable_snapshots[-1].splitlines()]
        assert [row["kind"] for row in durable] == ["audit_begin", "verification_attempt"]
        assert durable[-1]["requested_rows"] == 2
        assert durable[-1]["cumulative_requested_rows"] == 2
        assert durable[-1]["input_shape"] == [2, 17]
        before_rejection = path.read_bytes()
        with pytest.raises(ValueError, match="durable independent audit inference cap exhausted"):
            journal.reserve("synthetic-owner", "over-cap", np.zeros((1, 17)))
        assert path.read_bytes() == before_rejection
        assert journal.rows == 2 and journal.attempts == 1
    finally:
        journal.close("interrupted_or_exception")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["kind"] for row in rows] == ["audit_begin", "verification_attempt", "audit_end"]
    assert rows[-1]["status"] == "interrupted_or_exception"
    assert rows[-1]["requested_rows"] == 2
    assert rows[-1]["attempts"] == 1
    assert sum(row.get("requested_rows", 0) for row in rows
               if row["kind"] == "verification_complete") == 0
