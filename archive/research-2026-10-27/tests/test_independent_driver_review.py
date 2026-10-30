"""Driver regressions using mocked search; these run no physical transitions."""
from types import SimpleNamespace
import time

import numpy as np
import pytest

import scripts.training.run_independent_search as driver
from src.gps.independent_core import Memory


def test_curve_keeps_panel_and_cumulative_execution_counts_distinct(tmp_path):
    ev = driver.Evidence.__new__(driver.Evidence)
    ev.out = tmp_path
    ev.cfg = {"evaluation_roots": 1, "final_roots": 1}
    ev.curves = []
    ev.preflight = {"counts": {"total_controls": 212}, "wall_seconds": 0.}
    ev.arm_starts = {}
    ev.start = time.perf_counter()
    ev.episodes = []
    ev.episode = lambda *args, **kwargs: {
        "cost": 4., "success": True, "changed": 2, "actions": np.zeros((3, 1)),
    }
    ev.totals = lambda **kwargs: {
        "total_controls": 99 if kwargs["arm"] != "shared" else 12,
        "executed_controls": 20 if kwargs["arm"] != "shared" else 2,
    }
    controller = driver.Controller(Memory([]))
    row = ev.curve(controller, seed=0, arm="mc", cycle=0)
    assert row["executed_controls"] == 20
    assert row["evaluated_executed_controls"] == 3
    assert row["charged_controls"] == 323


def test_completed_search_work_survives_a_guidance_failure(tmp_path, monkeypatch):
    cfg = {"steps": 3, "goal": .45, "K": 4, "temperature": 3.}
    ev = driver.Evidence(tmp_path, cfg)
    ev.save_controller = lambda controller: None
    ev.task = SimpleNamespace(reset=lambda seed: np.array([-.5, 0.]))
    bank = {
        "plans": np.zeros((5, 3, 1)), "actions": np.zeros((5, 1)),
        "memory_row_ids": (),
    }

    class FakeSearch:
        def __init__(self, *args, **kwargs):
            self.ledger = SimpleNamespace(events=[])

        def plan(self, *args):
            return {
                "bank": bank, "baseline_index": 0,
                "query": {"query_id": 0, "seed": 1, "remaining": 3,
                          "independent_controls": 12, "verification_controls": 3,
                          "planner_calls": 1, "wall_seconds": 0.},
            }

    def failed_scorer(*args):
        raise RuntimeError("synthetic predictor failure before any scratch physics")

    monkeypatch.setattr(driver, "Search", FakeSearch)
    monkeypatch.setattr(driver, "score_guidance", failed_scorer)
    controller = driver.Controller(Memory([]), "mc", SimpleNamespace(digest="dummy"))
    try:
        with pytest.raises(RuntimeError, match="synthetic predictor"):
            ev.episode(controller, seed=0, arm="mc", cycle=0, role="collection", index=0)
        totals = ev.totals()
        assert totals["independent_controls"] == 12
        assert totals["verification_controls"] == 3
        assert totals["planner_calls"] == 1
        assert totals["failed_queries"] == 1
        assert totals["executed_controls"] == 0
        durable = (tmp_path / "queries.jsonl").read_text()
        assert '"kind":"plan"' in durable
        assert '"kind":"failed_guidance"' in durable
    finally:
        ev.handle.close()
