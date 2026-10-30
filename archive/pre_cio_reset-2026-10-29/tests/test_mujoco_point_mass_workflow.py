"""Pure evidence contracts; no substitute environment or MuJoCo execution."""
import json

import numpy as np
import pytest

from scripts.runners import run_point_mass as runner


def test_fixed_qualification_roster_and_native_profile():
    config = runner.load_config()
    q = config["qualify"]
    assert q["reset_seeds"] == [310000001, 310000002]
    assert q["planner_seeds"] == [410000001, 410000002]
    assert q["steps"] == 300
    assert q["max_physics_controls_per_phase"] == 2 * 300 * (256 * 64 + 1) == 9831000
    assert q["max_queries_per_phase"] == 2 * 300 * 2 == 1200
    assert config["environment"]["use_warp"] is False
    assert config["environment"]["nthread"] == config["environment"]["frame_skip"] == 1
    config["mppi"]["K"] = 1
    with pytest.raises(ValueError, match="frozen"):
        runner.load_config_dict(config)


def test_create_only_freeze_preserves_existing_evidence(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    sentinel = out / "evidence.txt"
    sentinel.write_text("retain")
    with pytest.raises(FileExistsError):
        runner._freeze(out, runner.load_config(), "validate")
    assert sentinel.read_text() == "retain"
    assert list(out.iterdir()) == [sentinel]


def test_boundary_failure_preserves_reservation_and_unknown_partial_work(tmp_path):
    ledger = runner.BoundaryLedger(tmp_path / "failed", max_controls=3, max_queries=1)
    def failed_call():
        raise RuntimeError("native boundary did not return")
    try:
        with pytest.raises(RuntimeError, match="did not return"):
            ledger.query("model_bank", 3, {"actions": np.zeros((1, 3, 2))}, failed_call, lambda x: x)
    finally:
        ledger.close()
    rows = [json.loads(x) for x in ledger.ledger_path.read_text().splitlines()]
    assert [x["kind"] for x in rows] == ["query_requested", "query_error", "archive_close"]
    assert rows[1]["partial_physics_unknown"] is True
    assert ledger.counts["requested_physics_controls"] == ledger.counts["unknown_partial_controls"] == 3
    assert ledger.counts["completed_physics_controls"] == 0
    assert ledger.counts["failed_queries"] == 1
    assert (ledger.path / rows[0]["inputs"]["file"]).is_file()


def test_budget_rejection_precedes_external_call(tmp_path):
    ledger = runner.BoundaryLedger(tmp_path / "capped", max_controls=2, max_queries=1)
    invoked = []
    try:
        with pytest.raises(RuntimeError, match="budget"):
            ledger.query("model_bank", 3, {}, lambda: invoked.append(True), lambda x: {})
    finally:
        ledger.close()
    assert invoked == []
    assert ledger.counts["queries"] == ledger.counts["requested_physics_controls"] == 0
    assert ledger.counts["rejected_queries"] == 1
    assert json.loads(ledger.ledger_path.read_text().splitlines()[0])["kind"] == "query_rejected"
