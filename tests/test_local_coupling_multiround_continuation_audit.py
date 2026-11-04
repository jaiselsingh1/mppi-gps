"""Pure L3.1 matched-group checks: no real forward pass, physics or fitting."""

from __future__ import annotations

import numpy as np
import pytest

from src.gps.mppi_gps.audits import local_coupling_multiround_continuation as module
from src.gps.mppi_gps.audits.local_coupling_learning import compare_native_discrepancy
from src.gps.mppi_gps.audits.trajectory_pi import observations, typed_fingerprint


class GroupedFakeAudit(module.LearningAudit):
    def __init__(self):
        super().__init__(neural_rows=100)
        self.fake_calls = []

    def predict(self, payload, inputs):
        n = len(inputs)
        self.requested_rows += n
        self.completed_rows += n
        self.fake_calls.append(n)
        # A small-group and full-group float32 graph can differ without an
        # execution/controller change. This is explicitly a mocked action map.
        return np.full((n, 6), 2e-6 if n == 2 else 0, dtype=np.float64)


def case():
    initial = np.zeros(19)
    initial[2] = 1.2
    posts = np.broadcast_to(initial, (2, 2, 19)).copy()
    actions = np.zeros((2, 2, 6))
    mask = np.ones((2, 2), dtype=bool)
    bank = dict(
        initial_state=initial,
        posts=posts,
        actions=actions,
        mask=mask,
        C=np.ones(2),
        D=np.zeros(2),
        predictions=np.zeros((4, 6)),
    )
    states = np.broadcast_to(initial, (3, 19)).copy()
    obs = observations(states)
    replay = dict(states=states, observations=obs, actions=np.zeros((2, 6)))
    grouped = np.concatenate((obs[:-1], obs[:-1]))
    digest = typed_fingerprint(grouped)
    native = np.full((2, 6), 2e-6)
    d = float(0.5 * np.sum(native**2))
    tol = 1e-10 + 2 * replay["actions"].size * 1e-6
    prediction = dict(
        observations=obs[:-1],
        predictions=native,
        expected_predictions=np.zeros((2, 6)),
        matched_predictions=np.zeros((2, 6)),
        matching_group_predictions=np.zeros((4, 6)),
        matching_group_observation_fingerprint=np.asarray(digest),
        matching_group_rows=np.asarray(4),
        replacement_offset=np.asarray(2),
        candidate=np.asarray(1),
        comparison_version=np.asarray(module.COMPARISON_VERSION),
        D=np.asarray(d),
        matched_D=np.asarray(0.0),
        D_tolerance=np.asarray(tol),
        action_max_error=np.asarray(2e-6),
        matched_action_max_error=np.asarray(0.0),
    )
    row = dict(
        comparison_version=module.COMPARISON_VERSION,
        cross_group_is_diagnostic=True,
        candidate=1,
        matching_group_observation_fingerprint=digest,
        native_prefix_rows=2,
        matching_group_rows=4,
        added_matching_forward_rows=4,
        D=d,
        matched_D=0.0,
        D_tolerance=tol,
        action_max_error=2e-6,
        cross_group_action_max_error=2e-6,
        matched_action_max_error=0.0,
    )
    return dict(payload={}, bank=bank, replay=replay, prediction=prediction, row=row)


def test_amendment_keeps_old_failure_and_checks_like_grouping():
    inputs = case()
    audit = GroupedFakeAudit()
    with pytest.raises(module.AuditFailure):
        compare_native_discrepancy(
            audit,
            actions=inputs["replay"]["actions"],
            native_predictions=inputs["prediction"]["predictions"],
            bank_predictions=np.zeros((2, 6)),
            bank_discrepancy=0.0,
            saved_action_error=2e-6,
            saved_D_tolerance=inputs["row"]["D_tolerance"],
        )
    metrics, extra = module.audit_matched_predictions(audit, **inputs)
    assert metrics["action_max_error"] == 2e-6
    assert metrics["matched_action_max_error"] == 0
    assert metrics["original_cross_group_bound_passed"] is False
    assert extra == 4 and audit.fake_calls == [2, 4]
    assert audit.completed_rows == 6


def test_completion_certificate_forward_rows_are_not_charged_twice():
    metrics, extra = module.audit_matched_predictions(
        GroupedFakeAudit(), **case(), charge_matching_rows=False
    )
    assert extra == 0
    assert metrics["matching_group_rows"] == 4


@pytest.mark.parametrize(
    "field,bad",
    [
        ("matched_predictions", np.full((2, 6), 0.1)),
        ("matching_group_predictions", np.full((4, 6), 0.1)),
        ("replacement_offset", np.asarray(0)),
        ("matching_group_rows", np.asarray(3)),
        ("matching_group_observation_fingerprint", np.asarray("wrong")),
    ],
)
def test_real_computation_or_provenance_mismatch_still_fails(field, bad):
    value = case()
    value["prediction"][field] = bad
    with pytest.raises(module.AuditFailure):
        module.audit_matched_predictions(GroupedFakeAudit(), **value)


def test_post_action_inputs_cannot_masquerade_as_pre_action_inputs():
    value = case()
    value["replay"]["observations"] = value["replay"]["observations"].copy()
    value["replay"]["observations"][0, 0] += 0.1
    with pytest.raises(module.AuditFailure, match="pre-action"):
        module.audit_matched_predictions(GroupedFakeAudit(), **value)


def test_declaring_cross_group_as_passed_is_rejected():
    value = case()
    value["row"]["cross_group_is_diagnostic"] = False
    with pytest.raises(module.AuditFailure):
        module.audit_matched_predictions(GroupedFakeAudit(), **value)


def test_recorded_prefix_length_is_not_filled_after_fall():
    value = case()
    value["replay"]["actions"] = value["replay"]["actions"][:1]
    with pytest.raises(module.AuditFailure, match="whole executable"):
        module.audit_matched_predictions(GroupedFakeAudit(), **value)


def encoded(rows):
    import json

    return "".join(json.dumps(row) + "\n" for row in rows).encode()


def test_interrupted_calls_remain_unknown_and_charge_full_requested_budget():
    rows = [
        dict(id=0, event="requested", charges=dict(model_controls=12288, physics_substeps=49152)),
        dict(
            id=1,
            event="completed",
            request_id=0,
            charges=dict(model_controls=12288, physics_substeps=49152),
        ),
        dict(id=2, event="requested", charges=dict(neural_rows=8000)),
    ]
    result = module.interrupted_ledger_counts(encoded(rows))
    assert result["counts"]["requested_model_controls"] == 12288
    assert result["counts"]["completed_model_controls"] == 12288
    assert result["counts"]["unknown_partial_neural_rows"] == 8000
    assert result["outstanding_requests"] == [2]


def test_closed_receipt_cannot_be_double_charged_as_interrupted_overhead():
    with pytest.raises(module.AuditFailure, match="Closed jobs"):
        module.interrupted_ledger_counts(encoded([dict(id=0, event="closed")]))


@pytest.mark.parametrize(
    "data",
    [
        b'{"id":0',
        b"{bad}\n",
        encoded([dict(id=1, event="requested", charges=dict(native_controls=1))]),
        encoded([dict(id=0, event="completed", request_id=5, charges=dict(native_controls=1))]),
    ],
)
def test_corrupt_or_unbound_interruption_prefix_is_not_assigned_zero(data):
    with pytest.raises(module.AuditFailure):
        module.interrupted_ledger_counts(data)
