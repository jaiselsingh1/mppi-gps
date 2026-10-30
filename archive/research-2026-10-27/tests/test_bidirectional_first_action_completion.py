"""Pure mocked completion wiring; no simulator transition or real actor call."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from src.ending.core import Task
import src.gps.bidirectional.first_action_completion as completion


@dataclass(frozen=True)
class Snapshot:
    state: np.ndarray
    elapsed: int
    context_id: str = "mock-root"


class RecordingArchive:
    def __init__(self):
        self.events, self.arrays = [], {}

    def append(self, kind, **payload):
        identifier = f"event-{len(self.events):03d}"
        self.events.append(dict(id=identifier, kind=kind, **payload))
        return identifier

    def save_npz(self, name, **arrays):
        if name in self.arrays:
            raise AssertionError("Evidence must never overwrite an earlier attempt")
        self.arrays[name] = {key: np.asarray(value).copy() for key, value in arrays.items()}
        return dict(file=name, sha256="mock-content-sha", bytes=0)


class FakeActor:
    def __init__(self):
        self.version = "actor-v1"
        self.calls = []
        self.invalid_at = None
        self.raise_at = None
        self.invalid_hash = False

    def act(self, observation, *, family, deterministic):
        assert family == 0 and deterministic is True
        self.calls.append(observation.copy())
        if self.raise_at == len(self.calls):
            raise RuntimeError("mock actor execution failure")
        if self.invalid_at == len(self.calls):
            return np.array([np.nan])
        return np.array([-observation[0]])


class FakeContext:
    def __init__(self):
        # Construction is arithmetic-only. This Task's transition/rollout is
        # NEVER called: the test executor below returns a scripted tiny model.
        self.dynamics = Task(gate=2, delay=1)
        self.config = SimpleNamespace(T=4)
        self.task_hash = "mock-ending-task"
        self.state = np.array([0., .3])

    def observation(self, snapshot):
        result = np.zeros(8)
        result[0] = snapshot.state[1]
        result[1] = snapshot.state[0]/4
        result[2] = 1-result[1]
        return result


class FakeTrust:
    def __init__(self, state="promoted", weight=1.):
        self.weight, self.revocations, self.requests = weight, [], []
        self.records = [SimpleNamespace(actor_hash="actor-v1", actor_controller_hash="actor-v1-controller",
            task_hash="mock-ending-task", family_id=0, state=state, deterministic=True,
            total_steps=4, time_interval=(0, 3), remaining_interval=(1, 4),
            support_anchors=np.zeros((1, 8)), observation_scale=np.ones(8), support_radius=10.)]

    def weight_for(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        return self.weight

    def revoke_actor(self, actor_hash, reason):
        self.revocations.append((actor_hash, reason))
        self.records[0].state = "revoked"
        self.weight = 0.


class ScriptedExecutor:
    """Query mock: first control identifies a branch; ending cost is delayed.

    Deliberately not a claim about Ending physics or a learned controller.
    It tests ownership, complete continuation, evidence and exception wiring.
    """
    def __init__(self, *, equal_cost=False, fail_at=None, invalid_result_at=None):
        self.task, self.archive = FakeContext(), RecordingArchive()
        self.latest_query_id = None
        self.calls = []
        self.equal_cost, self.fail_at, self.invalid_result_at = equal_cost, fail_at, invalid_result_at

    def model_rollout(self, snapshot, actions, *, kind, metadata):
        assert kind == "actor_completion_step" and actions.shape == (1, 1, 1)
        qid = f"mock-q{len(self.calls):03d}"
        self.latest_query_id = qid
        record = dict(query_id=qid, state=snapshot.state.copy(), elapsed=snapshot.elapsed,
            actions=actions.copy(), metadata=metadata)
        self.calls.append(record)
        self.archive.append("mock_model_before_call", query_id=qid, requested_controls=1)
        if self.fail_at == len(self.calls):
            raise RuntimeError("mock failed physics call")
        # No Task transition is executed. A branch-specific delayed terminal
        # cost makes truncated-prefix rankings deliberately uninformative.
        y = float(actions[0, 0, 0]) if snapshot.elapsed == 0 else float(snapshot.state[1])
        cost_map = {-.3: 1., .8: 5., -.9: 0., .1: 2., .9: 10.}
        cost = (3. if self.equal_cost else cost_map[y]) if snapshot.elapsed == 3 else 0.
        result = SimpleNamespace(states=np.array([[[snapshot.elapsed+1., y]]]),
                                 total_costs=np.array([cost]))
        if self.invalid_result_at == len(self.calls):
            result.terminal_heuristic = np.array([0.])
        return result


def bank():
    actions = np.repeat(np.array([-.9, -.5, .1, .5, .9])[:, None, None], 2, axis=1)
    actions.flags.writeable = False
    independent = np.full((2, 1), .8); independent.flags.writeable = False
    return SimpleNamespace(actions=actions, independent_plan=independent,
        independent_score=-100., initial=Snapshot(np.array([0., .3]), 0),
        proposal_state_before=dict(rng=dict(state=123), nominal=independent),
        proposal_state_after=dict(rng=dict(state=456), nominal=independent))


@pytest.fixture
def mocked(monkeypatch):
    # No policy module, Torch network, engine or actual dynamics is imported by
    # these callbacks. They replace the version/meter seams before construction.
    def actor_hash(actor):
        if actor.invalid_hash:
            raise ValueError("mock nonfinite policy hash")
        return actor.version
    monkeypatch.setattr(completion, "policy_hash", actor_hash)
    monkeypatch.setattr(completion, "controller_hash", lambda actor, family: actor.version+"-controller")
    monkeypatch.setattr(completion, "attach_meter", lambda actor, archive: archive.append("mock_meter_attached"))
    executor, actor, trust = ScriptedExecutor(), FakeActor(), FakeTrust()
    return executor, actor, trust


def test_complete_delayed_cost_ranks_all_actions_without_using_horizon_score(mocked):
    executor, actor, trust = mocked
    native = bank()
    actions_before = native.actions.tobytes(); nominal_before = native.independent_plan.tobytes()
    selection = completion.FirstActionCompletionSelector(executor, actor, trust).select(native)
    np.testing.assert_array_equal(selection.plan, [[-.9]])
    assert selection.raw_score == selection.augmented_score == 0.
    assert selection.raw_score != native.independent_score
    assert selection.score_semantics == completion.SEMANTICS
    assert selection.changed and selection.influenced and selection.guidance_weight == 1.
    assert len(executor.calls) == len(selection.query_ids) == 5*4
    assert len(actor.calls) == 1+5*3
    assert native.actions.tobytes() == actions_before
    assert native.independent_plan.tobytes() == nominal_before
    assert native.proposal_state_before["rng"] == dict(state=123)
    assert native.proposal_state_after["rng"] == dict(state=456)
    np.testing.assert_array_equal(executor.task.state, [0., .3])
    event = next(r for r in executor.archive.events if r["kind"] == "completion_decision")
    raw = executor.archive.arrays[event["raw"]["file"]]
    np.testing.assert_array_equal(raw["candidate_first_actions"][:, 0], [-.3, .8, -.9, .1, .9])
    np.testing.assert_array_equal(raw["complete_candidate_costs"], [1., 5., 0., 2., 10.])
    np.testing.assert_array_equal(raw["states"].reshape(5, 4, 2)[:, -1, 0], np.full(5, 4.))
    assert len(set(raw["query_ids"])) == 20
    for i, call in enumerate(executor.calls):
        assert call["metadata"]["actor_hash"] == "actor-v1"
        assert call["metadata"]["semantics"] == completion.SEMANTICS
        assert call["metadata"]["owner"]["deterministic"] is True
        assert call["elapsed"] == i % 4
    # Every tail action observes its own branch state rather than replaying
    # actor actions from the independent trajectory or another candidate.
    for i, first in enumerate(raw["candidate_first_actions"][:, 0]):
        np.testing.assert_array_equal(raw["actions"].reshape(5, 4, 1)[i, 1:, 0], np.full(3, -first))
    with pytest.raises(ValueError):
        selection.plan.flags.writeable = True


def test_actor_first_exact_cost_ties_have_priority(mocked):
    _, actor, trust = mocked
    executor = ScriptedExecutor(equal_cost=True)
    result = completion.FirstActionCompletionSelector(executor, actor, trust).select(bank())
    np.testing.assert_array_equal(result.plan, [[-.3]])
    assert result.selected_candidate_id == "candidate-000" and result.raw_score == 3.


def test_exact_duplicate_actions_are_queried_once_with_aliases(mocked):
    executor, actor, trust = mocked
    native = bank(); native.actions = np.full((5, 2, 1), -.3)
    native.independent_plan = np.full((2, 1), -.3)
    result = completion.FirstActionCompletionSelector(executor, actor, trust).select(native)
    assert len(executor.calls) == 4 and len(actor.calls) == 4
    assert not result.changed
    event = next(r for r in executor.archive.events if r["kind"] == "completion_candidates")
    assert len(event["candidates"]) == 1
    assert [r["role"] for r in event["candidates"][0]["roles"]] == [
        "actor_first", "independent_champion", "bank_min", "bank_median", "bank_max"]


@pytest.mark.parametrize("reason", ["zero_weight", "revoked", "absent", "off_support"])
def test_unsupported_does_no_actor_or_model_work_and_preserves_baseline(mocked, reason):
    executor, actor, trust = mocked
    if reason == "zero_weight": trust.weight = 0.
    elif reason == "off_support": trust.records[0].support_radius = .001
    else: trust.records[0].state = reason
    native = bank()
    result = completion.FirstActionCompletionSelector(executor, actor, trust).select(native)
    np.testing.assert_array_equal(result.plan, native.independent_plan)
    assert not result.changed and not result.influenced and result.query_ids == ()
    assert result.guidance_weight == 0 and result.score_semantics == "independent_short_horizon_fallback"
    assert not actor.calls and not executor.calls and not executor.archive.events


def test_weight_one_trial_requires_eligible_current_record(mocked):
    executor, actor, trust = mocked
    trust.records[0].state = "eligible"
    selector = completion.FirstActionCompletionSelector(executor, actor, trust)
    assert not selector.select(bank()).influenced
    assert not actor.calls and not executor.calls
    result = selector.select(bank(), trial_weight=1.)
    assert result.influenced and trust.requests[-1][1]["trial_weight"] == 1.


def test_promoted_record_is_not_reused_as_an_eligible_trial(mocked):
    executor, actor, trust = mocked
    result = completion.FirstActionCompletionSelector(executor, actor, trust).select(bank(), trial_weight=1)
    assert not result.influenced and not executor.calls and not actor.calls


@pytest.mark.parametrize("weight", [.1, 10., True, float("nan")])
def test_other_trial_weights_fail_before_actor_or_model_work(mocked, weight):
    executor, actor, trust = mocked
    with pytest.raises(ValueError, match="exactly weight1"):
        completion.FirstActionCompletionSelector(executor, actor, trust).select(bank(), trial_weight=weight)
    assert not actor.calls and not executor.calls


@pytest.mark.parametrize("mutation", ["changed_version", "invalid_hash"])
def test_frozen_hash_failure_revokes_before_work_without_mutating_bank(mocked, mutation):
    executor, actor, trust = mocked
    native = bank(); before = native.actions.tobytes()
    selector = completion.FirstActionCompletionSelector(executor, actor, trust)
    if mutation == "changed_version": actor.version = "actor-v2"
    else: actor.invalid_hash = True
    result = selector.select(native)
    assert result.reason == "numerical_revocation" and not result.influenced
    assert trust.revocations[0][0] == "actor-v1"
    assert native.actions.tobytes() == before and not actor.calls and not executor.calls


def test_returned_invalid_actor_action_revokes_and_retains_partial_work(mocked):
    executor, actor, trust = mocked
    actor.invalid_at = 3
    result = completion.FirstActionCompletionSelector(executor, actor, trust).select(bank())
    assert result.reason == "numerical_revocation" and not result.influenced
    assert len(executor.calls) == len(result.query_ids) == 2
    assert trust.revocations
    event = next(r for r in executor.archive.events if r["kind"] == "completion_revocation")
    raw = executor.archive.arrays[event["raw"]["file"]]
    assert len(raw["raw_costs"]) == 2 and len(raw["complete_candidate_costs"]) == 0
    np.testing.assert_array_equal(result.plan, bank().independent_plan)


def test_failed_model_query_propagates_and_keeps_before_call_and_partial_refs(mocked):
    _, actor, trust = mocked
    executor = ScriptedExecutor(fail_at=3)
    with pytest.raises(RuntimeError, match="failed physics call"):
        completion.FirstActionCompletionSelector(executor, actor, trust).select(bank())
    assert len(executor.calls) == 3 and not trust.revocations
    before = [r for r in executor.archive.events if r["kind"] == "mock_model_before_call"]
    assert len(before) == 3
    event = next(r for r in executor.archive.events if r["kind"] == "completion_error")
    assert event["completed_query_ids"] == ["mock-q000", "mock-q001"]
    assert event["latest_executor_query_id"] == "mock-q002"
    raw = executor.archive.arrays[event["raw"]["file"]]
    assert raw["query_ids"].tolist() == ["mock-q000", "mock-q001"]
    assert not any(r["kind"] == "completion_decision" for r in executor.archive.events)


def test_actor_execution_exception_is_not_silently_revoked_into_fallback(mocked):
    executor, actor, trust = mocked
    actor.raise_at = 2
    with pytest.raises(RuntimeError, match="actor execution failure"):
        completion.FirstActionCompletionSelector(executor, actor, trust).select(bank())
    assert len(executor.calls) == 1 and not trust.revocations
    assert any(r["kind"] == "completion_error" for r in executor.archive.events)


def test_endpoint_heuristic_result_is_rejected_after_its_returned_query(mocked):
    _, actor, trust = mocked
    executor = ScriptedExecutor(invalid_result_at=1)
    with pytest.raises(RuntimeError, match="heuristic-bearing"):
        completion.FirstActionCompletionSelector(executor, actor, trust).select(bank())
    assert len(executor.calls) == 1 and not trust.revocations
    event = next(r for r in executor.archive.events if r["kind"] == "completion_error")
    assert event["completed_query_ids"] == ["mock-q000"]


def test_non_ending_task_cannot_claim_endpoint_free_completion(mocked):
    executor, actor, trust = mocked
    executor.task.dynamics = SimpleNamespace(T=4)
    with pytest.raises(ValueError, match="Ending task"):
        completion.FirstActionCompletionSelector(executor, actor, trust)
    assert not actor.calls and not executor.calls


def test_exact_one_decision_and_root_upper_bounds():
    assert completion.completion_work_bound(25) == dict(model_controls=125, model_queries=125,
        actor_rows=121, candidates=5)
    assert sum(completion.completion_work_bound(r)["model_controls"] for r in range(1, 26)) == 1625
    assert sum(completion.completion_work_bound(r)["actor_rows"] for r in range(1, 26)) == 1525
    assert completion.completion_work_bound(0)["actor_rows"] == 0
