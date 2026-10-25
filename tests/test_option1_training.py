"""Ownership and finite-cost regressions for independent-MPPI experience training."""

from types import SimpleNamespace
from dataclasses import replace
import hashlib
import inspect
import json

import numpy as np
import pytest

from scripts.training import train_mppi_experience as training
from src.gps.expert_continuation import Snapshot
from src.gps import mppi_experience as experience


def _prefix(*, steps=2, terminated=False, cost=3., gamma=.5):
    return SimpleNamespace(steps=steps, terminated=terminated, cost=cost, gamma=gamma,
        unhealthy_cost=5., healthy=not terminated, endpoint_obs=np.arange(17, dtype=float))


def test_candidate_score_uses_frozen_base_policy_action_for_continuation():
    prefix = _prefix()
    calls = []
    base_action = np.linspace(-.3, .3, 6)

    class BaseActor:
        def act(self, observation, deterministic=True):
            assert deterministic
            np.testing.assert_array_equal(observation, prefix.endpoint_obs)
            calls.append("base_actor")
            return base_action.copy()

    class Value:
        def predict(self, observations, actions, remaining):
            np.testing.assert_array_equal(observations, prefix.endpoint_obs[None])
            np.testing.assert_array_equal(actions, base_action[None])
            assert remaining == 6
            calls.append("base_continuation")
            return np.array([7.])

    assert training.candidate_score(prefix, total_steps=8, value=Value(),
                                    base_actor=BaseActor()) == pytest.approx(3. + .5 ** 2 * 7.)
    assert calls == ["base_actor", "base_continuation"]


@pytest.mark.parametrize("terminated,steps,raw_cost", [(True, 2, 26.), (False, 8, 3.)])
def test_fallen_and_full_horizon_candidates_do_not_query_policy_or_value(terminated, steps, raw_cost):
    poison = SimpleNamespace(
        act=lambda *a, **k: pytest.fail("An exact candidate queried a tail policy"),
        predict=lambda *a, **k: pytest.fail("An exact candidate queried a critic"))
    prefix = _prefix(steps=steps, terminated=terminated, cost=raw_cost)
    expected = raw_cost
    if terminated:
        expected += .5 ** steps * sum(5. * .5 ** j for j in range(8 - steps))
    assert training.candidate_score(prefix, total_steps=8, value=poison,
                                    base_actor=poison) == pytest.approx(expected)


def test_healthy_unfinished_candidate_requires_grounded_value():
    actor = SimpleNamespace(act=lambda *a, **k: np.zeros(6))
    with pytest.raises(RuntimeError, match="(ground|critic|value)"):
        training.candidate_score(_prefix(), total_steps=8, value=None, base_actor=actor)


class _FiniteWalker:
    """Tiny deterministic Walker adapter: raw costs1, then2+terminal20."""
    instances = []
    action_dim = 6
    _w_unhealthy = 5.

    def __init__(self, *, gamma, max_episode_steps):
        self.walker = self
        self.max_steps = max_episode_steps
        self.closed = False
        self.instances.append(self)

    def reset(self, *, seed=None, options=None):
        if options is not None:
            self.state = np.array(options["state"], copy=True)
            self.warmstart = np.array(options["warmstart"], copy=True)
        else:
            self.state = np.zeros(18); self.state[2] = 1.
            self.warmstart = np.arange(9, dtype=float) / 10.
        self.initial_state = self.state.copy()
        self.initial_warmstart = self.warmstart.copy()
        return self.state[1:].copy(), {}

    def get_state(self):
        return self.state.copy()

    def get_warmstart(self):
        return self.warmstart.copy()

    def _is_healthy(self, height, angle):
        return height > 0.

    def step(self, action):
        assert np.asarray(action).shape == (6,)
        self.state[0] += 1
        self.warmstart += .01
        terminated = self.state[0] == 2
        if terminated:
            self.state[2] = 0.
        truncated = not terminated and self.state[0] == self.max_steps
        raw = float(self.state[0] + (20. if terminated else 0.))
        return self.state[1:].copy(), -999., terminated, truncated, {"task_cost": raw}

    def close(self):
        self.closed = True


def _finite_snapshot(*, healthy=True):
    state = np.zeros(18); state[2] = float(healthy)
    return Snapshot(state, np.arange(9, dtype=float) / 10.)


def test_finite_policy_rollout_owns_raw_fall_cost_and_remaining_transitions():
    snapshot = _finite_snapshot()
    calls = []

    def act(observation, deterministic=True):
        assert deterministic
        calls.append(observation.copy())
        return np.zeros(6)

    result = experience.finite_policy_rollout(snapshot, SimpleNamespace(act=act),
        remaining=5, gamma=.5, reset_id=17, env_factory=_FiniteWalker)
    assert result.prefix.cost == 12.  # 1 + .5 * (2+20), no reward-adapter tail.
    assert result.cost == pytest.approx(12. + .5 ** 2 * (5. + 2.5 + 1.25))
    assert result.prefix.terminated and not result.prefix.healthy and result.prefix.steps == 2
    np.testing.assert_array_equal(result.transitions.costs, [1., 22.])
    np.testing.assert_array_equal(result.transitions.remaining, [5, 4])
    np.testing.assert_array_equal(result.transitions.terminated, [False, True])
    np.testing.assert_array_equal(result.transitions.source, ["policy", "policy"])
    np.testing.assert_array_equal(result.transitions.reset_ids, [17, 17])
    np.testing.assert_array_equal(result.states[0], snapshot.state)
    np.testing.assert_array_equal(result.warmstarts[0], snapshot.warmstart)
    np.testing.assert_array_equal(result.prefix.endpoint_state, result.states[-1])
    np.testing.assert_array_equal(result.prefix.endpoint_warmstart, result.warmstarts[-1])
    np.testing.assert_array_equal(result.transitions.next_observations, result.observations[1:])
    assert len(calls) == 2 and len(result.states) == len(result.warmstarts) == 3
    assert result.ledger["policy_controls"] == 2
    assert all(result.ledger[key] == 0 for key in (
        "teacher_controls", "teacher_candidate_controls", "teacher_planner_calls",
        "teacher_optimization_iterations"))
    assert _FiniteWalker.instances[-1].closed


def test_healthy_cutoff_and_already_fallen_start_do_not_add_a_new_terminal_charge():
    actor = SimpleNamespace(act=lambda *a, **k: np.zeros(6))
    cutoff = experience.finite_policy_rollout(_finite_snapshot(), actor,
        remaining=1, gamma=.5, env_factory=_FiniteWalker)
    assert cutoff.cost == 1. and cutoff.prefix.truncated and not cutoff.prefix.terminated
    assert cutoff.transitions.remaining.tolist() == [1]
    assert not cutoff.transitions.terminated.any()
    poison = SimpleNamespace(act=lambda *a, **k: pytest.fail("Already fallen policy executed"))
    fallen = experience.finite_policy_rollout(_finite_snapshot(healthy=False), poison,
        remaining=3, gamma=.5, env_factory=_FiniteWalker)
    assert fallen.cost == 5. + 2.5 + 1.25
    assert fallen.transitions is None and fallen.prefix.steps == 0
    assert fallen.ledger["policy_controls"] == 0


def test_teacher_collection_has_private_controller_memory_and_no_policy_argument():
    assert not {"actor", "policy", "value", "critic"} & set(
        inspect.signature(experience.collect_teacher_episode).parameters)
    planners = []

    class Planner:
        def __init__(self, walker, config, *, seed):
            self.calls = 0
            self.rng = np.random.default_rng(seed)
            planners.append(self)

        def plan_step(self, state, *, initial_warmstart):
            # This one planner survives across decisions and sees the exact
            # executed state/warmstart, including its own previous decision.
            np.testing.assert_array_equal(initial_warmstart,
                np.arange(9, dtype=float) / 10. + self.calls * .01)
            assert state[0] == self.calls
            self.calls += 1
            return np.full(6, self.rng.normal()), {"optimization_iterations": 2 if self.calls == 1 else 1}

    config = SimpleNamespace(K=4, H=3)
    kwargs = dict(reset_seed=71, total_steps=5, planner_seed=29, planner=config,
                  gamma=.5, env_factory=_FiniteWalker, planner_factory=Planner)
    first = experience.collect_teacher_episode(**kwargs)
    # Independent local RNG and no actor inputs make unrelated learner/global
    # random work irrelevant to this repeated teacher collection.
    np.random.default_rng(13).normal(size=100)
    second = experience.collect_teacher_episode(**kwargs)
    np.testing.assert_array_equal(first.prefix.actions, second.prefix.actions)
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.transitions.source, ["teacher", "teacher"])
    assert first.cost == second.cost == pytest.approx(14.1875)
    assert len(planners) == 2 and [p.calls for p in planners] == [2, 2]
    assert first.ledger == dict(policy_controls=0, teacher_controls=2,
        teacher_candidate_controls=4 * 3 * 3, teacher_planner_calls=2,
        teacher_optimization_iterations=3)


def _tiny(**changes):
    cfg = replace(training.Option1Config(), total_steps=8, handover_schedule=(4, 0),
        updates_per_stage=1, root_updates=1, pairs=2, teacher_train_episodes=2,
        teacher_holdout_episodes=1, root_train_starts=2, root_verify_starts=1,
        critic_prefix=2, critic_updates=5, critic_batch_size=4, hidden_dim=8,
        critic_hidden_dim=8, workers=1, eval_episodes=2, eval_every=2,
        teacher_K=4, teacher_H=2, teacher_initial_iterations=1)
    return replace(cfg, **changes)


def _json(path):
    return json.loads(path.read_text())


def _standalone_controls(directory):
    reports = [directory / "initial.json", directory / "final.json"]
    reports += sorted(directory.glob("standalone_*.json"))
    return sum(row["steps"] for path in reports if path.exists()
               for row in _json(path)["episodes"])


def _check_run(directory, cfg):
    status = _json(directory / "status.json")
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert status["phase"] == "complete" and status["source_unchanged"]
    assert status["completed_updates"] == len(rows) == 2
    assert status["walking_gate"] is None
    assert [row["handover"] for row in rows] == [4, 0]
    assert set(rows[0]["train_components"]) == set(rows[0]["heldout_components"]) == {"root", "guide"}
    assert set(rows[1]["train_components"]) == set(rows[1]["heldout_components"]) == {"root"}
    for row in rows:
        assert not row["teacher_return_targets"] and row["planner_takeovers"] == 0
        assert len(row["candidate_scores"]) == 2 * cfg.pairs
        assert np.isfinite(row["candidate_scores"]).all()
        fit = row["critic_fit"]
        assert fit["balanced_source_sampling"] and fit["training_teacher_rows"] > 0
        assert fit["training_policy_rows"] > 0
        assert fit["sampled_sources"] == {"teacher": 10, "policy": 10}
        assert row["learner_mc_calibration"]["episodes"] >= cfg.root_verify_starts
        assert np.isfinite(row["learner_mc_calibration"]["rmse"])
        valid = []
        for proposal in row["proposals"]:
            expected = (proposal["train_cost"] < row["train_base_cost"]
                        and proposal["heldout_cost"] < row["heldout_base_cost"])
            assert proposal["passes"] == expected
            if expected:
                valid.append(proposal)
        assert row["accepted"] == bool(valid)
        if valid:
            best = min(valid, key=lambda p: p["heldout_cost"])
            assert row["accepted_kind"] == best["kind"]
            assert row["heldout_current_cost"] == best["heldout_cost"]
        else:
            assert row["accepted_kind"] is None
            assert row["heldout_current_cost"] == row["heldout_base_cost"]
    assert (directory / "policy_initial.pt").exists() and (directory / "policy_final.pt").exists()
    assert len(list(directory.glob("policy_????.pt"))) == len(list(directory.glob("state_????.pt"))) == 2
    assert _json(directory / "final.json")["standalone"]
    assert status["ledger"]["standalone_evaluation_controls"] == _standalone_controls(directory)
    protocol = _json(directory / "protocol.json")
    for name, expected in protocol["source_sha256"].items():
        assert hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() == expected
    return status, rows


def test_tiny_real_cpu_run_uses_independent_teacher_mixed_q_and_policy_only_verification(tmp_path, monkeypatch):
    cfg = _tiny()
    out = tmp_path / "real"
    dispatched = []
    original_job = training._job

    def job(item):
        if item["kind"] == "teacher":
            assert not {"policy", "actor", "critic", "value"} & set(item)
        else:
            assert isinstance(item["policy"], bytes)
            assert item["kind"] in {"prefix", "policy"}
        result = original_job(item)
        dispatched.append((item, result))
        return result

    monkeypatch.setattr(training, "_job", job)
    training.run(out, cfg)
    status, rows = _check_run(out, cfg)
    teacher = [(item, result) for item, result in dispatched if item["kind"] == "teacher"]
    assert len(teacher) == cfg.teacher_train_episodes + cfg.teacher_holdout_episodes
    assert all(np.all(result.transitions.source == "teacher") for _, result in teacher)
    policy = [(item, result) for item, result in dispatched if item["kind"] == "policy"]
    assert policy and all(np.all(result.transitions.source == "policy") for _, result in policy)
    for key in ("policy_controls", "teacher_controls", "teacher_candidate_controls",
                "teacher_planner_calls", "teacher_optimization_iterations"):
        assert status["ledger"][key] == sum(result.ledger.get(key, 0) for _, result in dispatched)
    # Teacher data are collected once. All later search/verification owns its
    # complete learner suffix; it never asks the planner to finish a candidate.
    for key in ("teacher_controls", "teacher_candidate_controls", "teacher_planner_calls"):
        assert rows[0]["ledger_cumulative"][key] == rows[-1]["ledger_cumulative"][key]
    root_ids = {item["reset_id"] for item, _ in policy if item["remaining"] == cfg.total_steps}
    assert root_ids.isdisjoint(range(5_000_000, 5_000_200))
    assert {item["reset_seed"] for item, _ in teacher}.isdisjoint(root_ids)

    # A completed resume performs no physics, teacher queries, value fitting or
    # duplicate evaluation, and must preserve all previously measured controls.
    metrics = (out / "metrics.jsonl").read_bytes()
    before_jobs = len(dispatched)
    monkeypatch.setattr(training, "_standalone",
                        lambda *a, **k: pytest.fail("Completed resume repeated standalone evaluation"))
    monkeypatch.setattr(training.LearnerValueTrainer, "fit",
                        lambda *a, **k: pytest.fail("Completed resume refit a critic"))
    training.run(out, cfg, resume=True)
    assert len(dispatched) == before_jobs
    assert (out / "metrics.jsonl").read_bytes() == metrics
    assert _json(out / "status.json")["ledger"] == status["ledger"]
    with pytest.raises(ValueError, match="configuration"):
        training.run(out, replace(cfg, pairs=3), resume=True)
    with pytest.raises(FileExistsError):
        training.run(out, cfg)


def test_resume_after_initial_evaluation_accounts_existing_controls_exactly_once(tmp_path, monkeypatch):
    cfg = _tiny()
    out = tmp_path / "interrupted"
    original_job = training._job
    fail_once = True

    def job(item):
        nonlocal fail_once
        if fail_once and item["kind"] == "teacher":
            fail_once = False
            raise RuntimeError("Simulated interruption after initial evaluation")
        return original_job(item)

    monkeypatch.setattr(training, "_job", job)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        training.run(out, cfg)
    initial = (out / "initial.json").read_bytes()
    assert not (out / "metrics.jsonl").exists()
    assert _json(out / "status.json")["phase"] == "failed"
    training.run(out, cfg, resume=True)
    _check_run(out, cfg)
    assert (out / "initial.json").read_bytes() == initial
