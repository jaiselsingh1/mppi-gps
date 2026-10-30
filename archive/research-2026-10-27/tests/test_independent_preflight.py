"""End-to-end sentinels; the toy result must never be task-level evidence."""
import json

import numpy as np
import pytest

from src.gps import independent_preflight as preflight


def test_measured_targets_decisions_executions_and_revocation(tmp_path):
    result = preflight.run_preflight(tmp_path)
    assert result["status"] == "passed"
    assert all(result["sentinels"].values())
    assert result["failed_sentinels"] == []
    assert "not main environment" in result["scope"]
    assert result["learning"]["trainable_actor"] is False
    np.testing.assert_allclose(result["learning"]["coefficients"], [0., 0., 8.], atol=1e-12)
    assert result["learning"]["heldout_max_absolute_error"] < 1e-12
    assert result["acquisition"]["reexecuted"]["success"]
    assert result["acquisition"]["selected_index"] == 3
    assert result["trajectories"]["baseline"]["return"] == pytest.approx(8.1)
    assert result["trajectories"]["learned_deployment"]["return"] == pytest.approx(.2)
    assert result["first_action_change"] == 1.
    assert not result["trust_events"][1]["promoted"]
    assert result["trust_events"][2]["promoted"]
    assert result["trust_events"][3]["revoked"]
    assert result["trajectories"]["fallback_after_revocation"]["actions"] == [0., 0., 0., 1.]
    assert json.loads((tmp_path / "preflight.json").read_text()) == result
    assert sorted(path.name for path in tmp_path.iterdir()) == ["preflight.json"]


def test_every_control_and_prediction_is_charged_once(tmp_path):
    result = preflight.run_preflight(tmp_path)
    counts = result["counts"]
    transitions = result["raw_transitions"]
    queries = result["raw_queries"]
    assert counts["total_controls"] == len(transitions) == 212
    assert counts["simulated_controls"] + counts["executed_controls"] == 212
    assert sum(query["controls"] for query in queries) == 212
    assert counts["failed_queries"] == 0
    assert counts["zero_control_terminal_target_queries"] == 1
    assert counts["prediction_rows"] == sum(row["rows"] for row in result["raw_predictions"])
    assert counts["prediction_rows"] == 68
    assert result["wall_seconds"] > 0.
    for query in queries:
        rows = [row for row in transitions if row["query_id"] == query["id"]]
        assert len(rows) == query["controls"]
        assert query["wall_stop_seconds"] >= query["wall_start_seconds"]
    for target in result["learning"]["targets"]:
        query = queries[target["continuation_query_id"]]
        assert query["controls"] == target["remaining"]
        assert query["continuation_identity"] == result["learning"]["continuation_identity"]
        assert query["actions"] == [0.] * target["remaining"]
        assert target["target"] == pytest.approx(8. * (2. - target["state"]) ** 2)


def test_existing_evidence_is_retained(tmp_path):
    result_path = tmp_path / "preflight.json"
    preflight.run_preflight(result_path)
    before = result_path.read_bytes()
    with pytest.raises(FileExistsError, match="Retaining existing"):
        preflight.run_preflight(tmp_path)
    assert result_path.read_bytes() == before


def test_central_failure_writes_negative_evidence_and_raises_without_retuning(tmp_path, monkeypatch):
    calls = []

    def broken_fit(states, targets):
        calls.append((list(states), list(targets)))
        return np.zeros(3)

    monkeypatch.setattr(preflight, "_fit_value", broken_fit)
    with pytest.raises(preflight.PreflightFailure) as failure:
        preflight.run_preflight(tmp_path)
    result = failure.value.result
    assert len(calls) == 1
    assert result["status"] == "failed"
    assert result["verdict"].startswith("Toy causal plumbing fails")
    assert "heldout_complete_return_prediction_valid" in result["failed_sentinels"]
    assert "executed_long_horizon_improvement" in result["failed_sentinels"]
    assert json.loads((tmp_path / "preflight.json").read_text()) == result
