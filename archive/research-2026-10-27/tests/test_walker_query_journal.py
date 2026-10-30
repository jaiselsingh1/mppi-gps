"""Synthetic journal tests only: no environment construction or physics calls."""
from copy import deepcopy
import json

import numpy as np
import pytest

from src.gps.walker_query_journal import (
    ControlBudgetExceeded, JournaledEnv, QueryJournal, journal_stats,
)


TASK = dict(cost_style="target_velocity", target_velocity=1.5, frame_skip=4,
    vel_cost_weight=2., vertical_velocity_cost_weight=.5, angle_cost_weight=.1,
    unhealthy_cost_weight=5., ctrl_cost_weight=.001, terminal_unhealthy_weight=20.,
    apply_terminal_cost_on_done=True)


class FakeWalker:
    def __init__(self):
        self.state = np.zeros(19)
        self.state[2], self.state[10], self.state[11] = 1.25, 1.3, .2
        self.warm = np.arange(9, dtype=float)/10

    def get_state(self):
        return self.state.copy()

    def get_warmstart(self):
        return self.warm.copy()


class FakeEnv:
    def __init__(self, path, *, fail=False, fall=False, bad_cost=False):
        self.task = deepcopy(TASK)
        self.walker = FakeWalker()
        self.path, self.calls = path, 0
        self.fail, self.fall, self.bad_cost = fail, fall, bad_cost
        self.closed = False

    def reset(self, **kwargs):
        return np.zeros(17), {"reset": kwargs}

    def step(self, action):
        # Independently open the file while the wrapper is in the fake call:
        # the charge and exact pre-state must already be durable and readable.
        rows = _rows(self.path)
        assert rows[-1]["kind"] == "control_attempt"
        assert rows[-1]["controls"] == 1
        assert rows[-1]["pre"]["state"] == self.walker.state.tolist()
        assert rows[-1]["pre"]["warmstart"] == self.walker.warm.tolist()
        assert rows[-1]["action"] == action.tolist()
        self.calls += 1
        if self.fail:
            self.walker.state[0] += .002  # unknown partial fake integration
            raise RuntimeError("synthetic partial call")
        self.walker.state[0] += .008
        self.walker.warm += .01
        if self.fall:
            self.walker.state[2] = .7
        state = self.walker.state
        raw = (2*abs(1.5-state[10])+.5*state[11]**2+.1*state[3]**2
               +.001*sum(action**2)+(25 if self.fall else 0))
        if self.bad_cost:
            raw += 3
        # Preserve the adapter reward/tail as data, not as candidate credit.
        tail = -10000. if self.fall else 0.
        return np.zeros(17), -raw+tail, self.fall, not self.fall, dict(
            task_cost=raw, raw_reward=-raw, failure_tail=tail,
            training_reward=-raw+tail, fall_cost=20. if self.fall else 0.,
            custom_array=np.array([2., 3.]))

    def close(self):
        self.closed = True


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _journal(path, *, budget=3, **kwargs):
    return QueryJournal(path, query_id="seed0/actor0/q0", owner_hash="candidate-hash",
        context_id="historical/30/500", ancestry={"reset_seed": 30, "step": 500},
        budget_controls=budget, task_config=TASK, **kwargs)


def test_charge_is_visible_before_call_and_all_raw_arrays_are_preserved(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        fake = FakeEnv(path)
        env = JournaledEnv(fake, journal)
        pre, warm = fake.walker.get_state(), fake.walker.get_warmstart()
        action = np.arange(6)/10
        result = env.step(action)
        env.close()
        assert fake.closed and not journal.stats["query_ended"]
        journal.finish(metadata={"analytic_absorbing_slots": 0, "own_return": 1.})
    rows = _rows(path)
    assert [row["kind"] for row in rows] == ["query_begin", "control_attempt", "control_complete", "query_end"]
    assert len({row["id"] for row in rows}) == 4
    attempt, complete = rows[1:3]
    assert attempt["pre"] == {"state": pre.tolist(), "warmstart": warm.tolist()}
    assert complete["post"] == {"state": fake.walker.state.tolist(), "warmstart": fake.walker.warm.tolist()}
    assert complete["attempt_id"] == attempt["id"]
    assert complete["info"]["custom_array"] == [2., 3.]
    assert complete["reward"] == result[1]
    assert complete["raw_task_cost"] == result[4]["task_cost"]
    assert sum(complete["cost_components"].values()) == pytest.approx(complete["raw_task_cost"])
    stats = journal_stats(path)
    assert stats["charged_requested_controls"] == stats["completed_controls"] == stats["controls"] == 1
    assert stats["inflight_or_interrupted_controls"] == stats["error_controls"] == 0


def test_thrown_step_remains_charged_unknown_partial_and_cannot_silently_retry(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        fake = FakeEnv(path, fail=True)
        env = JournaledEnv(fake, journal)
        with pytest.raises(RuntimeError, match="partial call"):
            env.step(np.zeros(6))
        assert journal.stats["charged_requested_controls"] == journal.stats["error_controls"] == 1
        assert journal.stats["completed_controls"] == 0
        with pytest.raises(RuntimeError, match="failed"):
            env.step(np.zeros(6))
        with pytest.raises(ValueError, match="cannot finish"):
            journal.finish()
        journal.finish(status="error")
    error = _rows(path)[2]
    assert error["partial_physics_execution_unknown"] and not error["call_returned"]
    assert fake.calls == 1 and error["controls"] == 0


def test_budget_rejects_before_fake_call_without_new_control_charge(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path, budget=1) as journal:
        fake = FakeEnv(path)
        env = JournaledEnv(fake, journal)
        env.step(np.zeros(6))
        with pytest.raises(ControlBudgetExceeded):
            env.step(np.zeros(6))
        assert fake.calls == 1
        assert journal.stats["controls"] == 1
        assert journal.stats["rejected_before_call"] == 1


def test_pending_attempt_is_charged_without_inventing_completed_physics(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        journal.before_control({"state": [0.]*19, "warmstart": [0.]*9}, np.zeros(6))
        assert journal.stats["charged_requested_controls"] == 1
        assert journal.stats["inflight_or_interrupted_controls"] == 1
        assert journal.stats["completed_controls"] == 0
    assert journal_stats(path)["inflight_or_interrupted_controls"] == 1
    assert _rows(path)[-1]["status"] == "closed_without_result"


def test_fall_cost_once_reward_tail_separate_and_analytic_end_metadata_free(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        fake = FakeEnv(path, fall=True)
        _, reward, terminated, truncated, info = JournaledEnv(fake, journal).step(np.zeros(6))
        journal.finish(metadata={"analytic_absorbing_slots": 499})
    row = _rows(path)[2]
    assert terminated and not truncated
    assert row["cost_components"]["fall"] == 20.
    assert row["raw_task_cost"] == info["task_cost"]
    assert row["reward"] == reward == -info["task_cost"]-10000.
    assert row["info"]["failure_tail"] == -10000.
    assert journal_stats(path)["controls"] == 1


def test_post_call_validation_failure_retains_returned_evidence_and_charge(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        fake = FakeEnv(path, bad_cost=True)
        with pytest.raises(ValueError, match="cost components"):
            JournaledEnv(fake, journal).step(np.zeros(6))
    row = _rows(path)[2]
    assert row["kind"] == "control_error" and row["call_returned"]
    assert not row["partial_physics_execution_unknown"]
    assert row["evidence"]["info"]["task_cost"] > 3.
    assert journal_stats(path)["error_controls"] == 1


@pytest.mark.parametrize("action", [np.zeros(5), np.full(6, np.nan), np.full(6, 1.01)])
def test_invalid_action_never_calls_or_charges(tmp_path, action):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        fake = FakeEnv(path)
        with pytest.raises(ValueError):
            JournaledEnv(fake, journal).step(action)
        assert fake.calls == journal.stats["controls"] == 0
        assert journal.stats["rejected_before_call"] == 1


def test_bad_snapshot_foreign_task_and_planner_argument_rejected_before_call(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        fake = FakeEnv(path)
        with pytest.raises(TypeError):
            JournaledEnv(fake, journal, planner="forbidden")
        fake.task["vel_cost_weight"] = 1.
        with pytest.raises(ValueError, match="task differs"):
            JournaledEnv(fake, journal)
        fake.task = deepcopy(TASK)
        fake.walker.warm = np.zeros(8)
        with pytest.raises(ValueError, match="warmstart"):
            JournaledEnv(fake, journal).step(np.zeros(6))
        assert fake.calls == journal.stats["controls"] == 0


def test_exclusive_create_and_metadata_are_immutable_finite_json(tmp_path):
    path = tmp_path/"q.jsonl"
    metadata = {"a": np.array([1., 2.])}
    with _journal(path, metadata=metadata) as journal:
        metadata["a"][0] = 999.
        task = journal.task_config
        task["target_velocity"] = 9.
        assert journal.task_config["target_velocity"] == 1.5
        assert _rows(path)[0]["metadata"] == {"a": [1., 2.]}
        with pytest.raises(FileExistsError):
            _journal(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _journal(path)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="finite JSON"):
        _journal(tmp_path/"invalid.jsonl", metadata={"bad": float("nan")})
    assert not (tmp_path/"invalid.jsonl").exists()


def test_journal_reports_truncated_final_line_without_erasing_pending_charge(tmp_path):
    path = tmp_path/"q.jsonl"
    with _journal(path) as journal:
        journal.before_control({"state": [0.]*19, "warmstart": [0.]*9}, np.zeros(6))
    with path.open("ab") as handle:
        handle.write(b'{"kind":"control_complete"')
    stats = journal_stats(path)
    assert stats["incomplete_final_line"]
    assert stats["charged_requested_controls"] == stats["inflight_or_interrupted_controls"] == 1
