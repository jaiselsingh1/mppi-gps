"""Exercise actual query wiring with synthetic actors/steps; zero engine imports.

Fixture counts are logical mock calls, not experimental Walker controls or
network inferences: success3 steps/actions; failure1 attempted step/action.
"""
from collections import Counter
from dataclasses import dataclass
import builtins
import importlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


TASK = dict(frame_skip=4, cost_style="target_velocity", target_velocity=1.5,
    vel_cost_weight=2., vertical_velocity_cost_weight=.5, angle_cost_weight=.1,
    unhealthy_cost_weight=5., ctrl_cost_weight=.001, terminal_unhealthy_weight=20.,
    apply_terminal_cost_on_done=True)


class FakeActor:
    def __init__(self, calls):
        self.calls = calls

    def state_dict(self):
        return {"constant": np.full(6, .1, dtype=np.float32)}

    def act(self, observation, deterministic=True):
        assert observation.shape == (17,) and deterministic is True
        self.calls["actor_act"] += 1
        return np.full(6, .1)


@pytest.fixture
def wired_query(tmp_path, monkeypatch):
    calls = Counter()
    fail = dict(step=False)
    real_import = builtins.__import__

    def forbidden_engine_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "mujoco", "warp", "mujoco_warp"}:
            raise AssertionError("This wiring test must never import an engine or Torch: "+name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_engine_import)
    runner = importlib.import_module("scripts.diagnostics.run_walker_owned_signal")
    owned = importlib.import_module("src.gps.walker_owned_credit")
    from src.gps.walker_signal_search import ReservationBudget, observations_from_states

    @dataclass(frozen=True)
    class Snapshot:
        state: np.ndarray
        warmstart: np.ndarray

    expert_module = ModuleType("src.gps.expert_continuation")
    expert_module.Snapshot = Snapshot
    monkeypatch.setitem(sys.modules, expert_module.__name__, expert_module)

    root, out = tmp_path/"source", tmp_path/"result"
    (root/"configs").mkdir(parents=True)
    (root/"configs/task.json").write_text(json.dumps(TASK))
    for name in ("queries", "actors"):
        (out/name).mkdir(parents=True)
    journal_path = out/"queries/q000000/journal.jsonl"

    class FakeWalker:
        _use_warp, _frame_skip, _dt, _nthread = False, 4, .008, 1
        model = SimpleNamespace(na=0, nmocap=0)

        def __init__(self):
            self.state, self.warm = np.zeros(19), np.zeros(9)

        def get_state(self):
            return self.state.copy()

        def get_warmstart(self):
            return self.warm.copy()

    class FakeEnv:
        def __init__(self, *, task_config, gamma, max_episode_steps):
            calls["environment_constructor"] += 1
            self.task = json.loads(Path(task_config).read_text())
            self.walker, self.elapsed, self.limit = FakeWalker(), 0, max_episode_steps

        def reset(self, *, options):
            # The actual RecordedEnv proxy must write reset intent before reaching this mock.
            assert json.loads(journal_path.read_text().splitlines()[-1])["kind"] == "reset_attempt"
            calls["reset"] += 1
            self.walker.state = np.array(options["state"], copy=True)
            self.walker.warm = np.array(options["warmstart"], copy=True)
            return observations_from_states(self.walker.state), {}

        def step(self, action):
            # Actual JournaledEnv must persist its charged attempt before this mock call.
            intent = json.loads(journal_path.read_text().splitlines()[-1])
            assert intent["kind"] == "control_attempt" and intent["controls"] == 1
            assert np.array_equal(intent["action"], action)
            calls["step_attempt"] += 1
            if fail["step"]:
                raise RuntimeError("injected synthetic step failure")
            self.walker.state[0] += .008
            self.walker.state[1] += .008*1.5
            self.walker.warm += .01
            self.elapsed += 1
            cost = .001*float(np.square(action).sum())
            return observations_from_states(self.walker.state), -cost, False, self.elapsed == self.limit, {
                "task_cost":cost, "fall_cost":0., "x_velocity":1.5}

        def close(self):
            calls["close"] += 1

    walker_module = ModuleType("src.rl.walker")
    walker_module.WalkerRL = FakeEnv
    monkeypatch.setitem(sys.modules, walker_module.__name__, walker_module)

    def fake_finite(snapshot, actor, *, remaining, gamma, reset_id, env_factory):
        env = env_factory(gamma=gamma, max_episode_steps=remaining)
        try:
            obs, _ = env.reset(options=dict(state=snapshot.state, warmstart=snapshot.warmstart))
            assert np.array_equal(env.walker.get_state(), snapshot.state)
            assert np.array_equal(env.walker.get_warmstart(), snapshot.warmstart)
            states, warms, observations = [env.walker.get_state()], [env.walker.get_warmstart()], [obs]
            actions, costs = [], []
            for _ in range(remaining):
                action = actor.act(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                actions.append(action); costs.append(info["task_cost"])
                states.append(env.walker.get_state()); warms.append(env.walker.get_warmstart()); observations.append(obs)
            cost = float(np.dot(gamma**np.arange(remaining), costs))
            ledger = dict(policy_controls=remaining, teacher_controls=0, teacher_candidate_controls=0,
                          teacher_planner_calls=0, teacher_optimization_iterations=0)
            prefix = SimpleNamespace(requested_steps=remaining, gamma=gamma, unhealthy_cost=5.,
                healthy=True, terminated=terminated, truncated=truncated, raw_costs=np.asarray(costs),
                steps=remaining, cost=cost, actions=np.asarray(actions), endpoint_state=states[-1],
                endpoint_warmstart=warms[-1], endpoint_obs=observations[-1], ledger=ledger)
            return SimpleNamespace(cost=cost, prefix=prefix, states=np.asarray(states), warmstarts=np.asarray(warms),
                observations=np.asarray(observations), ledger=ledger)
        finally:
            env.close()

    finite_module = ModuleType("src.gps.mppi_experience")
    finite_module.finite_policy_rollout = fake_finite
    monkeypatch.setitem(sys.modules, finite_module.__name__, finite_module)

    assay = runner.Assay.__new__(runner.Assay)  # Deliberately bypass Torch-using initialization.
    actor = FakeActor(calls)
    owner = owned.actor_owner_hash(actor)
    actor_path = out/"actors"/(owner+".npz")
    runner.write_npz(actor_path, actor.state_dict())
    assay.out, assay.root, assay.cfg, assay.task = out, root, dict(gamma=.999, task_path="configs/task.json"), TASK
    assay.actors = {owner:dict(actor=actor, path=str(actor_path.relative_to(out)), sha256=runner.sha(actor_path))}
    assay.owned_rollout, assay.query_cache, assay.blind_open = owned.candidate_owned_rollout, {}, False
    assay.clock = [0]
    assay.events = runner.EventLog(out/"events.jsonl", assay.clock)
    assay.inferences = runner.EventLog(out/"inferences.jsonl", assay.clock)
    assay.budget = ReservationBudget(controls=3, queries=1, inferences=3)
    assay.inference_completions = assay.completed_controls = assay.completed_queries = 0
    assay.timing = dict(query_seconds=0., simulator_call_seconds=0.)
    state = np.zeros(19); state[2], state[10] = 1.2, 1.5
    warm = np.linspace(0., .08, 9)
    digest = owned.snapshot_digest(Snapshot(state, warm))
    context = dict(id="synthetic_context", snapshot_digest=digest, state=state, warmstart=warm,
                   remaining=3, ancestry=dict(reset_seed=30, source_context_ids=["synthetic_source"]))
    try:
        yield SimpleNamespace(assay=assay, owner=owner, context=context, calls=calls, fail=fail,
                              out=out, journal_path=journal_path)
    finally:
        assay.events.close(); assay.inferences.close()
        (tmp_path/"logical_fixture_calls.json").write_text(json.dumps(dict(calls)))


def test_real_query_wiring_retains_complete_owned_raw_trace_and_exact_cache(wired_query):
    fixture = wired_query
    result = fixture.assay.query(fixture.owner, fixture.context, phase="selection_complete", seed=0, round_index=0)
    assert result["complete"] is True and result["continuation_owner"] == "candidate"
    assert result["owner_hash"] == fixture.owner and result["executed_steps"] == 3
    assert result["absorbing_steps"] == 0 and len(result["inference_ids"]) == 3
    assert fixture.assay.budget.reserved_controls == 3 and fixture.assay.budget.inference_attempts == 3
    assert result["journal_stats"]["charged_requested_controls"] == 3
    assert result["journal_stats"]["completed_controls"] == 3
    raw_path = fixture.out/result["raw_path"]
    with np.load(raw_path, allow_pickle=False) as raw:
        assert raw["states"].shape == (4, 19) and raw["warmstarts"].shape == (4, 9)
        assert np.array_equal(raw["states"][0], fixture.context["state"])
        assert np.array_equal(raw["warmstarts"][0], fixture.context["warmstart"])
        assert raw["actions"].shape == (3, 6) and raw["cost_components"].shape == (3, 6)
        assert np.allclose(raw["raw_costs"], raw["cost_components"].sum(axis=1))
    events = [json.loads(line) for line in fixture.assay.events.path.read_text().splitlines()]
    assert events[0]["kind"] == "query_reserved" and events[0]["remaining"] == 3
    inferences = [json.loads(line) for line in fixture.assay.inferences.path.read_text().splitlines()]
    assert [row["inference_id"] for row in inferences if row["kind"] == "inference_attempt"] == result["inference_ids"]
    before = fixture.calls.copy()
    cached = fixture.assay.query(fixture.owner, fixture.context, phase="selection_complete", seed=0, round_index=1)
    assert cached is result and fixture.calls == before and fixture.assay.budget.queries == 1
    assert fixture.calls == Counter(environment_constructor=1, reset=1, step_attempt=3, actor_act=3, close=1)


def test_real_query_wiring_preserves_failed_attempt_and_never_returns_fallback(wired_query):
    fixture = wired_query
    fixture.fail["step"] = True
    with pytest.raises(RuntimeError, match="injected synthetic step failure"):
        fixture.assay.query(fixture.owner, fixture.context, phase="selection_complete", seed=0, round_index=0)
    directory = fixture.out/"queries/q000000"
    assert not (directory/"result.json").exists()
    failure = json.loads((directory/"failure.json").read_text())
    assert failure["journal_stats"]["charged_requested_controls"] == 1
    assert failure["journal_stats"]["completed_controls"] == 0
    assert failure["journal_stats"]["error_controls"] == 1
    assert failure["journal_stats"]["unknown_partial_error_controls"] == 1
    assert fixture.assay.budget.reserved_controls == 3 and fixture.assay.budget.queries == 1
    assert fixture.assay.budget.inference_attempts == 1 and not fixture.assay.query_cache
    assert fixture.assay.completed_queries == fixture.assay.completed_controls == 0
    rows = [json.loads(line) for line in fixture.journal_path.read_text().splitlines()]
    intent = next(row for row in rows if row["kind"] == "control_attempt")
    error = next(row for row in rows if row["kind"] == "control_error")
    assert error["attempt_id"] == intent["id"] and error["call_returned"] is False
    assert fixture.calls == Counter(environment_constructor=1, reset=1, step_attempt=1, actor_act=1, close=1)
