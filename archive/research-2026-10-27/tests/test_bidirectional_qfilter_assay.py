"""Pure arrays/metadata and injected mocks; no Task.transition or neural call."""
from dataclasses import dataclass
from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

import src.gps.bidirectional.qfilter_assay as q


def test_paid_bounds_include_both_q_branches_all_aux_and_uniform_fresh_control():
    work = q.work_bound()
    assert work["maximum_unique_pairs"] == 200
    assert work["model_controls"] == 10000
    assert work["qualification_controls"] == 8800
    assert work["final_controls"] == 16000
    assert work["executed_controls"] == 24800
    assert work["total_controls"] == 34800
    assert work["q_actor_rows"] == 9800
    assert work["training_actor_rows"] == 27200
    assert work["actor_rows"] == 61800
    assert work["adam_steps"] == 56 and work["diagnostic_backward_passes"] == 2


def test_fixed_config_rejects_retuning_or_new_roots(tmp_path):
    changed = deepcopy(q.DEFAULT); changed["qualification_seed"] += 1
    path = tmp_path/"config.json"; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): q.load_config(path)
    with pytest.raises(ValueError): q.work_bound(changed)
    assert q.load_config() == q.DEFAULT


def test_closed_arm_allows_only_late_cost_addition_not_owner_or_data_changes():
    original = dict(parent_actor_hash="frozen", behavior_sha256="actual64", own_episodes=64)
    closed = dict(**original, costs=dict(new_own_arm=dict(actor_rows=100)))
    q.validate_arm_event(original, closed)
    changed = dict(closed, parent_actor_hash="candidate")
    with pytest.raises(ValueError): q.validate_arm_event(original, changed)
    changed = dict(closed, behavior_sha256="other64")
    with pytest.raises(ValueError): q.validate_arm_event(original, changed)


def test_exact_pair_key_reuses_only_same_physical_action_state_and_owner():
    state = np.array([3., .25]); action = np.array([.125])
    key = q.pair_key(state, action, "actor", "task")
    assert q.pair_key(state.copy(), action.copy(), "actor", "task") == key
    assert q.pair_key(state, np.nextafter(action, 1.), "actor", "task") != key
    assert q.pair_key(state, action, "new_actor", "task") != key
    assert q.pair_key(state, action, "actor", "new_task") != key
    with pytest.raises(ValueError): q.pair_key(np.array([25., .25]), action, "actor", "task")


def test_positive_resampling_retains_actual_rows_weights_and_zero_mapping():
    obs = np.arange(1600.).reshape(200, 8); actions = np.linspace(-1., 1., 200)[:, None]
    families = np.zeros(200, dtype=np.int64); weights = np.zeros(200)
    weights[[3, 11, 198]] = [.1, .2, .3]
    result = q.resample_positive(obs, actions, families, weights)
    expected = np.array([3, 11, 198])[np.linspace(0, 2, 200, dtype=np.int64)]
    np.testing.assert_array_equal(result["original_indices"], expected)
    np.testing.assert_array_equal(result["actions"], actions[expected])
    np.testing.assert_array_equal(result["weights"], weights[expected])
    assert np.all(result["weights"] > 0) and not result["no_auxiliary_signal"]
    empty = q.resample_positive(obs, actions, families, np.zeros(200))
    assert empty["no_auxiliary_signal"] and not np.any(empty["weights"])
    np.testing.assert_array_equal(empty["original_indices"], np.arange(200))
    np.testing.assert_array_equal(empty["actions"], actions)
    with pytest.raises(ValueError): q.resample_positive(obs, actions, families, -np.ones(200))


@dataclass(frozen=True)
class FakeSnapshot:
    state: np.ndarray
    elapsed: int
    context_id: str


class FakeArchive:
    def __init__(self): self.events = []; self.arrays = {}
    def append(self, kind, **payload):
        identifier = f"event-{len(self.events)}"
        self.events.append(dict(id=identifier, kind=kind, **payload)); return identifier
    def save_npz(self, name, **arrays):
        assert name not in self.arrays
        self.arrays[name] = {k: np.array(v, copy=True) for k, v in arrays.items()}
        return dict(file=name, sha256="fake_sha", bytes=0)


class FakeActor:
    def __init__(self): self.calls = 0; self.version = "parent"
    def act(self, observation, family, deterministic):
        assert family == 0 and deterministic is True
        self.calls += 1; return np.array([-.5])


class FakeExecutor:
    def __init__(self, fail_at=None):
        from src.ending.core import Task
        self.archive = FakeArchive(); self.calls = []; self.fail_at = fail_at; self.latest_query_id = None
        self.task = SimpleNamespace(dynamics=Task(), config=SimpleNamespace(T=25), task_hash="task",
            observation=lambda snapshot: np.array([snapshot.state[1], snapshot.elapsed/25, 1-snapshot.elapsed/25, 0, 0, 0, 0, 0]))
    def model_rollout(self, snapshot, actions, *, kind, metadata):
        assert actions.shape == (1, 1, 1) and kind == "qfilter_actor_completion_step"
        self.calls.append(dict(snapshot=snapshot, actions=actions.copy(), metadata=metadata))
        if self.fail_at == len(self.calls): raise RuntimeError("injected query error")
        self.latest_query_id = f"mock-query-{len(self.calls)}"
        next_state = np.array([snapshot.elapsed+1., snapshot.state[1]])
        return SimpleNamespace(states=next_state.reshape(1, 1, 2), total_costs=np.array([float(actions[0, 0, 0]**2)]))


def _mock_owners(monkeypatch):
    monkeypatch.setattr(q, "_policy_hash", lambda actor: actor.version)
    monkeypatch.setattr(q, "_controller_hash", lambda actor: "controller:"+actor.version)


def test_full_measured_q_owns_frozen_actor_tail_not_updated_actor_or_teacher_suffix(monkeypatch):
    _mock_owners(monkeypatch); executor = FakeExecutor(); actor = FakeActor()
    initial = FakeSnapshot(np.array([23., .2]), 23, "source")
    result = q.measure_pair(executor, actor, initial, np.array([0.]), "pair-0000")
    assert actor.calls == 3 and len(executor.calls) == 4  # Injected logical calls only.
    assert result["q_actor"] == .5 and result["q_teacher"] == .25 and result["weight"] == .25
    np.testing.assert_array_equal([r["actions"][0, 0, 0] for r in executor.calls], [-.5, -.5, 0., -.5])
    assert [r["snapshot"].elapsed for r in executor.calls] == [23, 24, 23, 24]
    assert all(r["metadata"]["actor_hash"] == "parent" for r in executor.calls)
    assert executor.archive.events[0]["kind"] == "qfilter_pair_requested"
    assert executor.archive.events[-1]["kind"] == "qfilter_pair_complete"
    saved = executor.archive.arrays[result["raw"]["file"]]
    np.testing.assert_array_equal(saved["branch_indices"], [0, 0, 1, 1])
    np.testing.assert_array_equal(saved["states"][:, 0], [24, 25, 24, 25])


def test_query_failure_is_preserved_and_propagated_without_zero_weight_fallback(monkeypatch):
    _mock_owners(monkeypatch); executor = FakeExecutor(fail_at=2); actor = FakeActor()
    with pytest.raises(RuntimeError, match="injected query error"):
        q.measure_pair(executor, actor, FakeSnapshot(np.array([23., .2]), 23, "source"), np.array([0.]), "pair-0001")
    failed = executor.archive.events[-1]
    assert failed["kind"] == "qfilter_pair_error"
    assert failed["completed_query_ids"] == ["mock-query-1"]
    assert len(executor.calls) == 2
    assert not any(r["kind"] == "qfilter_pair_complete" for r in executor.archive.events)


def _panels(mode):
    def rows(cost):
        return [SimpleNamespace(context_id=f"root:{i}", cost=cost, capture=True,
            complete=True, deterministic=mode) for i in range(10)]
    return {name: rows(cost) for name, cost in (
        ("parent", 1.02), ("raw", 1.01), ("guided", 1.), ("best_raw", 1.005), ("uniform_best_raw", 1.004))}


def test_fresh_uniform_best_control_is_required_and_can_falsify_qfilter_advantage():
    mean, noisy = _panels(True), _panels(False)
    assert q.assess_final(mean, noisy)["passed"]
    for row in mean["uniform_best_raw"]: row.cost = .999
    assert not q.assess_final(mean, noisy)["passed"]
    noisy.pop("uniform_best_raw")
    with pytest.raises(ValueError): q.assess_final(mean, noisy)
