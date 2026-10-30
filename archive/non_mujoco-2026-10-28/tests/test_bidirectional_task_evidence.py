"""Synthetic environment tests: no MuJoCo, Warp, or neural calls."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.bidirectional.task import (AcrobotCaptureTask, Snapshot, TaskConfig,
                                       post_action_tip, trajectory_family)
from src.gps.bidirectional.evidence import Archive, EpisodeTrace, QueryExecutor, ledger_totals


class FakeAcrobot:
    state_dim, action_dim, _dt, _frame_skip = 5, 1, .01, 1
    action_bounds = (np.array([-1.]), np.array([1.]))

    def __init__(self):
        self.state, self.warm = np.zeros(5), np.zeros(2)
        self.data = SimpleNamespace(sensordata=np.zeros(3))
        self.calls = 0
        self.before = None
        self.fail = False
        self._sensors()

    def _sensors(self):
        q0, q1 = self.state[1:3]
        self.data.sensordata = np.array([np.sin(q0)+np.sin(q0+q1), 0.,
                                        2+np.cos(q0)+np.cos(q0+q1)])

    def reset(self, state=None):
        self.state = np.zeros(5) if state is None else np.array(state, copy=True)
        self.warm = np.zeros(2)
        self._sensors()
        return self.state[1:].copy()

    def get_state(self):
        return self.state.copy()

    def get_warmstart(self):
        return self.warm.copy()

    def set_warmstart(self, value):
        self.warm = np.array(value, copy=True)

    def step(self, action):
        if self.before:
            self.before()
        self.calls += 1
        self.state[0] += .01
        self.state[3] = action[0]
        self.state[1] += .01*action[0]
        self.warm += [1., 2.]
        self._sensors()
        if self.fail:
            raise RuntimeError("synthetic failure after possible partial work")
        parts = self.weighted_cost_components(self.running_cost_components(
            self.state, action, self.data.sensordata))
        return self.state[1:].copy(), float(parts.total), False, {}

    def running_cost_components(self, states, actions, sensors):
        target = np.square(sensors[..., 2]-4.)
        vel = np.square(states[..., 3:5]).sum(axis=-1)
        zero = np.zeros_like(target)
        control = .1*np.square(actions).sum(axis=-1)
        return SimpleNamespace(target_cost=target, qvel_cost=vel,
                               qvel_excess_cost=zero, control_cost=control)

    def weighted_cost_components(self, c):
        return SimpleNamespace(**vars(c), total=c.target_cost+c.qvel_cost+c.qvel_excess_cost+c.control_cost)

    def terminal_cost(self, states, sensors):
        return 7.+np.square(states[..., 3:5]).sum(axis=-1)

    def batch_rollout(self, initial, actions, initial_warmstart=None):
        if self.before:
            self.before()
        if self.fail:
            self.calls += 1
            raise RuntimeError("synthetic batch failure")
        output, sensors = [], []
        for bank in actions:
            self.reset(initial)
            self.set_warmstart(initial_warmstart)
            states, sense = [], []
            for action in bank:
                self.step(action)
                states.append(self.get_state())
                sense.append(self.data.sensordata.copy())
            output.append(states)
            sensors.append(sense)
        states, sensors = np.array(output), np.array(sensors)
        parts = self.weighted_cost_components(self.running_cost_components(states, actions, sensors))
        return states, parts.total.sum(axis=1)+self.terminal_cost(states[:, -1], sensors[:, -1]), sensors

    def close(self):
        pass


def task(T=4, capture_steps=2):
    return AcrobotCaptureTask(TaskConfig(T=T, capture_steps=capture_steps, local_by=T),
                              env=FakeAcrobot(), seed=0)


def upright(t, *, elapsed=0, hold=0, context="root/train/0"):
    state = np.zeros(5)
    state[0] = .01*elapsed
    result = Snapshot(state, np.array([3., 4.]), elapsed, hold, t.task_hash, context)
    t.restore(result)
    return result


def test_reset_uses_own_stream_without_touching_global_rng():
    np.random.seed(123)
    before = np.random.get_state()
    a, b = task(), task()
    first, second = a.reset(seed=9), b.reset(seed=9)
    np.testing.assert_array_equal(first.state, second.state)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert abs(first.state[1]-np.pi) <= .1
    assert abs(first.state[2]) <= .1
    assert np.max(abs(first.state[3:])) <= .05


def test_snapshot_restore_is_exact_and_rejects_foreign_task():
    t = task()
    original = upright(t)
    t.step(np.array([.2]))
    restored = t.restore(original)
    np.testing.assert_array_equal(restored.state, original.state)
    np.testing.assert_array_equal(restored.warmstart, original.warmstart)
    assert (restored.elapsed, restored.hold_count, restored.context_id) == (0, 0, original.context_id)
    with pytest.raises(ValueError, match="different task"):
        t.restore(replace(original, task_hash="foreign"))
    with pytest.raises(ValueError, match="clock"):
        t.restore(replace(original, elapsed=1))


def test_terminal_cost_once_and_capture_final_consecutive_hold():
    t = task(T=3)
    start = upright(t)
    steps = [t.step(np.array([0.])) for _ in range(3)]
    assert [step.terminal_cost for step in steps] == [0., 0., 7.]
    assert [step.snapshot.hold_count for step in steps] == [1, 2, 2]
    assert steps[-1].done and steps[-1].captured
    assert sum(step.total_cost for step in steps) == 7.
    with pytest.raises(RuntimeError, match="after finite"):
        t.step(np.array([0.]))
    trace = EpisodeTrace.from_steps(start, steps, config=t.config, actor_owner="actor0",
        execution_semantics="actor-only-feedback", source={"root": "train/0"})
    assert trace.complete and trace.continuation_complete and trace.successful
    assert trace.total_cost == 7.


def test_capture_failure_charge_once_and_last_window_not_earlier_hit():
    t = task(T=3)
    upright(t)
    first = t.step(np.array([0.]))
    middle = t.step(np.array([1.]))
    last = t.step(np.array([0.]))
    assert first.snapshot.hold_count == 1 and middle.snapshot.hold_count == 0
    assert last.snapshot.hold_count == 1 and not last.captured
    assert [s.capture_failure for s in (first, middle, last)] == [0., 0., 1000.]


def test_model_preserves_live_state_hold_context_and_observations_are_preaction():
    t = task(T=4)
    live = upright(t, elapsed=1, hold=1, context="ordinary/train/3")
    imagined = replace(live, context_id="other-root")
    model = t.model_rollout(imagined, np.zeros((2, 2, 1)))
    after = t.snapshot()
    np.testing.assert_array_equal(after.state, live.state)
    np.testing.assert_array_equal(after.warmstart, live.warmstart)
    assert after.context_id == live.context_id and after.hold_count == live.hold_count
    assert model.observations.shape == (2, 2, 8)
    np.testing.assert_array_equal(model.observations[:, 0, 7], [.5, .5])
    np.testing.assert_array_equal(model.observations[:, 1, 7], [1., 1.])
    np.testing.assert_array_equal(model.terminal_costs, [0., 0.])
    np.testing.assert_array_equal(model.terminal_heuristic, [7., 7.])
    np.testing.assert_array_equal(model.total_costs, [7., 7.])
    assert all(not snapshot.warmstart_exact for snapshot in model.end_snapshots)


def test_model_cutoff_final_failure_and_no_terminal_double_count():
    t = task(T=3)
    initial = upright(t, elapsed=1, hold=1)
    model = t.model_rollout(initial, np.zeros((2, 7, 1)))
    assert model.requested_horizon == 7 and model.horizon == 2
    np.testing.assert_array_equal(model.terminal_costs, [7., 7.])
    np.testing.assert_array_equal(model.capture_failures, [0., 0.])
    np.testing.assert_array_equal(model.total_costs, [7., 7.])
    failed = t.model_rollout(initial, np.ones((1, 7, 1)))
    assert failed.capture_failures[0] == 1000.
    assert failed.total_costs[0] == pytest.approx(failed.stage_costs.sum()+failed.terminal_heuristic[0]+1000.)


def test_failed_model_also_restores_live_snapshot():
    t = task()
    before = upright(t, elapsed=1, hold=1)
    t.env.fail = True
    with pytest.raises(RuntimeError, match="batch failure"):
        t.model_rollout(before, np.zeros((2, 2, 1)))
    after = t.snapshot()
    np.testing.assert_array_equal(after.state, before.state)
    np.testing.assert_array_equal(after.warmstart, before.warmstart)
    assert after.elapsed == 1 and after.hold_count == 1


def test_executor_charges_before_call_and_separates_padding(tmp_path):
    t = task(T=3)
    initial = upright(t, elapsed=1, hold=1)
    with Archive(tmp_path/"raw") as archive:
        executor = QueryExecutor(t, archive, scope="seed0/acquisition", owner={"kind": "independent"})
        def check_before():
            totals = archive.totals()
            assert totals["requested_controls"] == 14
            assert totals["inflight_or_interrupted_controls"] == 4
        t.env.before = check_before
        executor.model_rollout(initial, np.zeros((2, 7, 1)), metadata={"root": "train/0"})
        totals = executor.ledger_totals
        assert totals["requested_controls"] == 14
        assert totals["completed_controls"] == 4
        assert totals["cutoff_skipped_controls"] == 10
        assert totals["requested_physics_frames"] == 4
        assert totals["unknown_controls"] == 0
        files = list((archive.path/"queries").glob("*.npz"))
        assert len(files) == 2
        for path in files:
            with np.load(path, allow_pickle=False) as arrays:
                assert not any(arrays[key].dtype.hasobject for key in arrays.files)


def test_thrown_live_call_remains_charged_and_owner_override_saved(tmp_path):
    t = task()
    upright(t)
    with Archive(tmp_path/"raw") as archive:
        executor = QueryExecutor(t, archive, owner={"actor_hash": "old"})
        t.env.fail = True
        with pytest.raises(RuntimeError, match="partial work"):
            executor.step(np.array([0.]), metadata={"owner": {"actor_hash": "candidate1"}})
        totals = archive.totals()
        assert totals["requested_controls"] == totals["unknown_error_controls"] == 1
        assert totals["completed_controls"] == 0 and totals["error_queries"] == 1
        requests = [json.loads(line) for line in archive.ledger_path.read_text().splitlines()
                    if json.loads(line)["kind"] == "query_requested"]
        assert requests[0]["owner"] == {"actor_hash": "candidate1"}


def test_budget_and_invalid_actions_never_cross_physics_boundary(tmp_path):
    t = task()
    initial = upright(t)
    with Archive(tmp_path/"raw") as archive:
        executor = QueryExecutor(t, archive, max_requested_controls=1)
        with pytest.raises(RuntimeError, match="budget"):
            executor.model_rollout(initial, np.zeros((2, 1, 1)))
        with pytest.raises(ValueError, match="within"):
            executor.step(np.array([1.01]))
        assert t.env.calls == 0
        assert archive.totals()["requested_controls"] == 0
        assert archive.totals()["rejected_queries"] == 2
        assert archive.totals()["query_attempts"] == 2


def test_rejected_mean_preserves_exact_numeric_input_and_attempt_slots(tmp_path):
    t=task();initial=upright(t)
    bad=np.array([[[np.nextafter(1.,2.)],[0.25],[-.1]]],dtype=np.float64)
    with Archive(tmp_path/"raw") as archive:
        executor=QueryExecutor(t,archive)
        with pytest.raises(ValueError,match="within"):
            executor.model_rollout(initial,bad,kind="independent_mean_verification")
        rows=[json.loads(line) for line in archive.ledger_path.read_text().splitlines()]
        rejected=next(row for row in rows if row["kind"]=="query_rejected")
        assert rejected["attempted_controls_unvalidated"]==3
        assert rejected["requested_controls"]==rejected["requested_physics_controls"]==0
        assert rejected["completed_controls"]==0 and rejected["raw"] is not None
        with np.load(archive.path/rejected["raw"]["file"],allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["actions"],bad)
            assert raw["actions"].dtype==bad.dtype
            assert json.loads(str(raw["initial_snapshot_json"]))["context_id"]==initial.context_id
        assert archive.totals()["query_attempts"]==1 and archive.totals()["queries"]==0
        assert archive.totals()["completed_controls"]==t.env.calls==0


def test_invalid_object_input_records_safe_descriptor_without_masking_error(tmp_path):
    class InvalidControl:
        def __float__(self):raise TypeError("original invalid control")
        def __repr__(self):raise RuntimeError("must not stringify object")
    t=task();initial=upright(t);bad=np.array([[[InvalidControl()]]],dtype=object)
    with Archive(tmp_path/"raw") as archive:
        executor=QueryExecutor(t,archive)
        with pytest.raises(TypeError,match="original invalid control"):
            executor.model_rollout(initial,bad)
        rejected=next(json.loads(line) for line in archive.ledger_path.read_text().splitlines()
                      if json.loads(line)["kind"]=="query_rejected")
        assert rejected["attempted_controls_unvalidated"]==1
        with np.load(archive.path/rejected["raw"]["file"],allow_pickle=False) as raw:
            assert "actions" not in raw.files
            assert not any(raw[name].dtype.hasobject for name in raw.files)
            descriptor=json.loads(str(raw["actions_descriptor_json"]))
            assert descriptor["dtype"]=="object" and not descriptor["numeric_actions_saved"]
        assert archive.totals()["query_attempts"]==1 and t.env.calls==0


def test_rejection_archive_failure_does_not_replace_original_validation(tmp_path,monkeypatch):
    t=task();upright(t)
    with Archive(tmp_path/"raw") as archive:
        executor=QueryExecutor(t,archive)
        def fail_write(*args,**kwargs):raise OSError("synthetic evidence write failure")
        monkeypatch.setattr(archive,"save_npz",fail_write)
        with pytest.raises(ValueError,match="within"):
            executor.step(np.array([1.01]))
        rejected=next(json.loads(line) for line in archive.ledger_path.read_text().splitlines()
                      if json.loads(line)["kind"]=="query_rejected")
        assert rejected["raw"] is None and "write failure" in rejected["rejected_evidence_error"]
        assert archive.totals()["query_attempts"]==1 and t.env.calls==0


def test_pending_request_is_not_a_completed_query(tmp_path):
    with Archive(tmp_path/"raw") as archive:
        archive.append("query_requested", query_id="external-synthetic", requested_controls=12,
            requested_physics_controls=8, requested_physics_frames=8)
        totals = ledger_totals(archive.ledger_path)
        assert totals["unknown_controls"] == 8
        assert totals["completed_controls"] == 0
        assert totals["cutoff_skipped_controls"] == 4


def test_episode_roundtrip_root_ancestry_and_failure_archiving(tmp_path):
    t = task(T=3)
    initial = upright(t)
    steps = [t.step(np.array([0.])) for _ in range(3)]
    kwargs = dict(config=t.config, actor_owner={"hash": "actor0"},
                  execution_semantics="candidate-owned-complete", source={"root": "train/0"},
                  query_ids=("q0", "q1", "q2"))
    trace = EpisodeTrace.from_steps(initial, steps, **kwargs)
    path = tmp_path/"trace.npz"
    trace.to_npz(path)
    restored = EpisodeTrace.from_npz(path)
    assert restored.total_cost == trace.total_cost and restored.query_ids == trace.query_ids
    assert restored.successful and restored.continuation_complete
    np.testing.assert_array_equal(restored.warmstarts, trace.warmstarts)
    with pytest.raises(FileExistsError):
        trace.to_npz(path)
    bad = replace(steps[1], initial_snapshot=replace(steps[1].initial_snapshot, context_id="held/0"))
    with pytest.raises(ValueError, match="ancestry"):
        EpisodeTrace.from_steps(initial, (steps[0], bad, steps[2]), **kwargs)
    with Archive(tmp_path/"raw") as archive:
        saved = trace.archive(archive, "episode-0")
        assert saved["file"].startswith("episodes/success/")
        failed = replace(trace, captured=False)
        assert failed.archive(archive, "failed-0")["file"].startswith("episodes/failures/")


def test_nonroot_continuation_is_complete_but_not_root_success():
    t = task(T=3)
    initial = upright(t, elapsed=1, hold=1)
    steps = [t.step(np.array([0.])) for _ in range(2)]
    trace = EpisodeTrace.from_steps(initial, steps, config=t.config, actor_owner="candidate",
        execution_semantics="complete-after-handover", source={"root": "train/0"})
    assert trace.continuation_complete and trace.captured
    assert not trace.complete and not trace.successful


def test_family_is_coherent_unwrapped_first_approach():
    positive = np.zeros((3, 5)); positive[:, 1] = [3.1, 4.5, 6.28]
    negative = np.zeros((3, 5)); negative[:, 1] = [3.1, 1.5, 0.]
    assert trajectory_family(positive, [4., 2., 0.]) == "positive"
    assert trajectory_family(negative, [4., 2., 0.]) == "negative"
    assert trajectory_family(negative, [4., 2., 1.]) == "unreached"


def test_configuration_frozen_hash_and_unknown_keys():
    config = TaskConfig()
    assert TaskConfig.from_dict(config.to_dict()).task_hash == config.task_hash
    assert replace(config, failure_penalty=999.).task_hash != config.task_hash
    with pytest.raises(TypeError):
        TaskConfig.from_dict({"invisible_change": 2})
    with pytest.raises(ValueError):
        TaskConfig(dt=.02)


class StaleSensorAcrobot(FakeAcrobot):
    """Synthetic RK4-style stale sensor; exactly the same state transition."""
    def step(self, action):
        prior_sensor = post_action_tip(self.state)
        obs, _, done, info = super().step(action)
        self.data.sensordata = prior_sensor.copy()
        parts = self.weighted_cost_components(self.running_cost_components(
            self.state, action, self.data.sensordata))
        return obs, float(parts.total), done, info


def test_v2_post_action_fk_repairs_sensor_timing_without_extra_controls(tmp_path):
    config = TaskConfig(T=2, capture_steps=1, local_by=2)
    synthetic = AcrobotCaptureTask(config, env=StaleSensorAcrobot())
    initial = Snapshot(np.array([0., .099, 0., 0., 0.]), np.zeros(2), 0, 0,
                       synthetic.task_hash, "timing/root0")
    synthetic.restore(initial)
    with Archive(tmp_path/"raw") as archive:
        executor = QueryExecutor(synthetic, archive)
        modeled = executor.model_rollout(initial, np.array([[[.4]]]))
        actual = executor.step(np.array([.4]))
        # The old pre-action tip is inside .2m; the true post-action tip is out.
        assert np.linalg.norm(actual.legacy_sensors-[0.,0.,4.]) < .2
        assert actual.tip_distance > .2 and actual.snapshot.hold_count == 0
        np.testing.assert_array_equal(actual.snapshot.state, modeled.states[0,0])
        np.testing.assert_array_equal(actual.boundary_sensors, modeled.boundary_sensors[0,0])
        assert actual.stage_cost == pytest.approx(modeled.stage_costs[0,0])
        assert actual.legacy_stage_cost != pytest.approx(actual.stage_cost)
        assert modeled.legacy_stage_costs[0,0] == pytest.approx(actual.legacy_stage_cost)
        assert modeled.cost_boundary_source == "post-action-full-state-analytic-fk"
        assert synthetic.env.calls == archive.totals()["completed_controls"] == 2
        with np.load(archive.path/"queries/q000000000_result.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw["boundary_sensors"], modeled.boundary_sensors)
            np.testing.assert_array_equal(raw["legacy_sensors"], modeled.legacy_sensors)
            assert str(raw["cost_boundary_source"]) == "post-action-full-state-analytic-fk"
