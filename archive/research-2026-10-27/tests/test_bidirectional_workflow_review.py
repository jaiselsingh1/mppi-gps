"""Small integration checks with injected synthetic transitions only."""
import json

import numpy as np

from src.gps.bidirectional.evidence import Archive, QueryExecutor
from src.gps.bidirectional.search import IndependentSearch, SearchConfig, TrajectoryMemory
from src.gps.bidirectional.workflow import execute_episode
from test_bidirectional_task_evidence import task, upright


def test_executed_independent_search_has_raw_bank_mean_and_live_queries(tmp_path):
    synthetic = task(T=4)
    initial = upright(synthetic)
    with Archive(tmp_path/"evidence") as archive:
        executor = QueryExecutor(synthetic, archive, scope="synthetic/review")
        cfg = SearchConfig(K=2, H=2, smooth_candidates=1, memory_candidates=1,
                           temperature=.1, sigma=.1)
        planner = IndependentSearch(executor, cfg, seed=19)
        trace, behavior, outcome = execute_episode(executor, initial, "independent-0",
            planner=planner, source={"synthetic": True, "reset_cluster": "train/0"})
        assert behavior is None and trace.complete and outcome["complete"]
        assert trace.execution_semantics == "executed_replanning"
        assert len(trace.query_ids) == 4 and len(set(trace.query_ids)) == 4
        totals = archive.totals()
        # Effective horizons: 2,2,2,1. Four bank candidates, one verified mean,
        # and one actually executed control per decision. All pass the executor.
        assert totals["completed_controls"] == 4*7+7+4 == synthetic.env.calls
        assert totals["queries"] == totals["completed_queries"] == 12
        assert totals["unknown_controls"] == 0
        events = [json.loads(row) for row in archive.ledger_path.read_text().splitlines()]
        requests = [row for row in events if row["kind"] == "query_requested"]
        assert [row["query_kind"] for row in requests].count("planner_bank") == 4
        assert [row["query_kind"] for row in requests].count("independent_mean_verification") == 4
        assert [row["query_kind"] for row in requests].count("executed_control") == 4
        physical = {row["query_id"]: row for row in requests}
        for qid in trace.query_ids:
            assert physical[qid]["owner"]["controller_hash"] == outcome["controller_hash"]
            assert physical[qid]["owner"]["semantics"] == "executed_replanning"
        for decision in (row for row in events if row["kind"] == "decision"):
            assert all(qid in physical for qid in decision["bank_query_ids"])
            with np.load(archive.path/decision["raw"]["file"], allow_pickle=False) as raw:
                assert not raw["nominal_before"].flags.writeable or raw["nominal_before"].shape == (2,1)
                assert raw["executed_plan"].shape[1] == 1


def test_repeat_proposals_from_same_private_state_are_bitwise_identical(tmp_path):
    synthetic = task(T=3)
    initial = upright(synthetic)
    with Archive(tmp_path/"evidence") as archive:
        executor = QueryExecutor(synthetic, archive)
        planner = IndependentSearch(executor, SearchConfig(K=3,H=2,smooth_candidates=2,
            memory_candidates=1,initial_iterations=2), seed=23, memory=TrajectoryMemory())
        before = planner.snapshot()
        first = planner.propose(initial)
        planner.restore(before)
        second = planner.propose(initial)
        assert first.digest == second.digest
        np.testing.assert_array_equal(first.actions, second.actions)
        np.testing.assert_array_equal(first.independent_plan, second.independent_plan)
        # Each initial proposal uses TWO complete bank+mean passes, retained.
        assert len(first.query_ids) == len(second.query_ids) == 4
        assert archive.totals()["queries"] == 8
        assert archive.totals()["completed_controls"] == 4*(6*2+2)
        np.testing.assert_array_equal(synthetic.snapshot().state, initial.state)
