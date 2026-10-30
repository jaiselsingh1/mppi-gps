"""Independent target-ownership and context-boundary checks without Walker.

These checks deliberately inspect complete measured returns and retained raw
state, rather than reimplementing a model's fitted predictions.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import src.gps.replanning_rollout as rollout
import scripts.training.train_replanning_value as driver


def _state(value=0.):
    state = np.zeros(19)
    state[2], state[4] = 1.25, value
    return state


def _raw(value=0., *, initial=True, seed=9):
    return dict(U=np.full((2, 6), value), initial_plan=initial,
                rng_state=copy.deepcopy(np.random.default_rng(seed).bit_generator.state))


def _cfg():
    return SimpleNamespace(total_steps=4, gamma=.9, memory_candidates=0,
                           query_stride=1)


def _fake_world(monkeypatch, *, fall_at=None):
    planners, envs = [], []

    class FakeEnv:
        def __init__(self, gamma, max_episode_steps):
            self.walker = self
            self.n = max_episode_steps
            self.controls = 0
            self.state, self.warm = _state(), np.zeros(9)
            self._w_unhealthy, self._v_target = 5., 1.5
            self.closed = False
            envs.append(self)

        def get_state(self):
            return self.state.copy()

        def get_warmstart(self):
            return self.warm.copy()

        def step(self, action):
            self.controls += 1
            self.state[0] += .008
            self.state[4] += action[0]
            self.warm[:] = self.state[4]
            fell = self.controls == fall_at
            cost = 1. + self.controls + action[0]
            return None, None, fell, self.controls == self.n and not fell, {
                "task_cost": cost, "x_velocity": self.state[4]}

        def close(self):
            self.closed = True

    class Raw:
        def __init__(self, seed):
            self.saved = _raw(seed=seed)

        def snapshot(self):
            return copy.deepcopy(self.saved)

        def restore(self, state):
            self.saved = copy.deepcopy(state)

    class Planner:
        def __init__(self, controller, seed, remaining):
            self.raw = Raw(seed)
            self.remaining = remaining
            self.value_target_version = controller.get("value_target_version", controller["version"])
            self.calls = []
            planners.append(self)

        def plan_step(self, state, initial_warmstart, prepare_successors=False):
            self.calls.append((state.copy(), initial_warmstart.copy(), self.raw.snapshot(), self.remaining))
            self.raw.saved["U"] += .1
            self.raw.saved["initial_plan"] = False
            self.remaining -= 1
            return np.full(6, .1), dict(guidance_used=False, model_guidance_used=False, action_changed=False,
                selected_source="sample", candidate_controls=10, memory_controls=2,
                verification_controls=3, one_step_verification_controls=4,
                value_prediction_rows=0)

    def restore(env, snapshot):
        env.state, env.warm = snapshot.state.copy(), snapshot.warmstart.copy()

    monkeypatch.setattr(rollout, "WalkerRL", FakeEnv)
    monkeypatch.setattr(rollout, "_reset", lambda env, reset_id, cfg: "fake")
    monkeypatch.setattr(rollout, "_restore", restore)
    monkeypatch.setattr(rollout, "_planner", lambda env, cfg, controller, seed, remaining:
                        Planner(controller, seed, remaining))
    return planners, envs


def test_dense_targets_own_actual_predecision_context_and_complete_suffix(monkeypatch):
    planners, envs = _fake_world(monkeypatch)
    cfg = _cfg()
    trace = rollout.rollout_episode(cfg, {"version": "C0"}, reset_id=42,
        planner_seed=10, dense_stride=1, source="train", group=42)
    assert envs[0].closed
    assert trace["steps"] == trace["ledger"]["executed_controls"] == 4
    assert trace["ledger"]["planner_calls"] == 4
    assert trace["ledger"]["one_step_verification_controls"] == 16
    costs = trace["costs"]
    expected = [sum(cfg.gamma**j * costs[i+j] for j in range(len(costs)-i))
                for i in range(len(costs))]
    np.testing.assert_allclose(trace["mc_returns"], expected)
    assert len(trace["records"]) == 4
    for i, record in enumerate(trace["records"]):
        assert record["remaining"] == 4-i
        assert record["controller_version"] == "C0" and record["group"] == 42
        assert record["cost"] == pytest.approx(expected[i])
        np.testing.assert_array_equal(record["context"]["state"], trace["states"][i])
        np.testing.assert_array_equal(record["context"]["warmstart"], trace["warmstarts"][i])
        np.testing.assert_array_equal(record["context"]["nominal"], planners[0].calls[i][2]["U"])
        assert record["context"]["initial_plan"] is (i == 0)


def test_forced_first_action_never_labels_the_offpolicy_predecision_state(monkeypatch):
    planners, _ = _fake_world(monkeypatch)
    context = dict(state=_state(2.), warmstart=np.full(9, 3.), nominal=np.full((2, 6), .7),
                   initial_plan=False, remaining=4, controller_version="C0")
    post = _raw(.7, initial=False, seed=900)
    trace = rollout.rollout_episode(_cfg(), {"version": "C0"}, reset_id=42,
        planner_seed=23, context=context, raw_snapshot=post, forced_action=np.full(6, .5),
        dense_stride=8, source="train", group=42)
    assert trace["ledger"]["planner_calls"] == 3
    assert trace["ledger"]["executed_controls"] == 4
    assert [row["step"] for row in trace["records"]] == [1]
    record = trace["records"][0]
    assert record["remaining"] == 3 and record["context"]["remaining"] == 3
    np.testing.assert_array_equal(record["context"]["state"], trace["states"][1])
    np.testing.assert_array_equal(record["context"]["warmstart"], trace["warmstarts"][1])
    np.testing.assert_array_equal(record["context"]["nominal"], post["U"])
    np.testing.assert_array_equal(planners[0].calls[0][2]["U"], post["U"])
    assert planners[0].calls[0][3] == 3
    assert record["cost"] == pytest.approx(trace["mc_returns"][1])
    assert record["cost"] != pytest.approx(trace["mc_returns"][0])
    expected_rng = np.random.default_rng(23).bit_generator.state
    assert planners[0].calls[0][2]["rng_state"] == expected_rng
    # A new branch stream must not rewrite the captured post-bank nominal/RNG.
    assert post["rng_state"] == np.random.default_rng(900).bit_generator.state


def test_fall_dense_targets_keep_exact_finite_absorption_and_no_fake_rows(monkeypatch):
    _fake_world(monkeypatch, fall_at=2)
    trace = rollout.rollout_episode(_cfg(), {"version": "C0"}, reset_id=42,
        planner_seed=10, dense_stride=1)
    assert trace["terminated"] and not trace["survived"]
    assert len(trace["records"]) == trace["ledger"]["executed_controls"] == 2
    expected_last = trace["costs"][-1] + .9*(5.+.9*5.)
    assert trace["records"][-1]["cost"] == pytest.approx(expected_last)
    assert trace["cost"] == pytest.approx(trace["costs"][0]+.9*expected_last)


def _point(actions):
    return dict(controller_version="C0", root=42, group=42, step=9, remaining=4,
        pair_id="captured-pair", snapshot=dict(state=_state(), warmstart=np.arange(9.)),
        raw_post=_raw(.7, initial=False), bank=dict(baseline_index=0,
            actions=np.asarray(actions), plans=np.repeat(np.asarray(actions)[:, None, :], 2, axis=1)))


def test_counterfactual_pairs_preserve_ancestry_and_match_private_future_streams(monkeypatch):
    planners, _ = _fake_world(monkeypatch)
    point = _point([np.full(6, .1), np.full(6, .5)])
    result = rollout.paired_first_action(_cfg(), {"version": "C0"}, point, 1,
        [300, 301], source="train", dense_stride=8)
    assert len(planners) == 4
    assert result["ledger"]["executed_controls"] == 16
    assert result["ledger"]["planner_calls"] == 12
    assert len(result["records"]) == 4
    assert all(row["group"] == 42 and row["controller_version"] == "C0"
               and row["remaining"] == 3 for row in result["records"])
    assert [row["side"] for row in result["records"]] == ["baseline", "candidate"]*2
    for replicate, seed in enumerate((300, 301)):
        left, right = planners[2*replicate:2*replicate+2]
        assert left.calls[0][2]["rng_state"] == right.calls[0][2]["rng_state"]
        assert left.calls[0][2]["rng_state"] == np.random.default_rng(seed).bit_generator.state
        np.testing.assert_array_equal(left.calls[0][2]["U"], point["raw_post"]["U"])
        np.testing.assert_array_equal(right.calls[0][2]["U"], point["raw_post"]["U"])


def test_equal_first_actions_measure_once_without_duplicate_work_or_fake_pair_sides(monkeypatch):
    planners, _ = _fake_world(monkeypatch)
    point = _point([np.full(6, .1), np.full(6, .1)])
    # Distinct hypothetical tails must not create another first-action experiment.
    point["bank"]["plans"][1, 1] = .9
    result = rollout.paired_first_action(_cfg(), {"version": "C0"}, point, 1,
        [300, 301], source="train", dense_stride=8)
    assert result["equal_first_actions"]
    assert len(planners) == 2
    assert result["ledger"]["executed_controls"] == 8
    assert result["ledger"]["planner_calls"] == 6
    assert result["differences"] == [0., 0.]
    assert len(result["records"]) == 2
    assert all(row["side"] == "baseline" for row in result["records"])
    for baseline, candidate in zip(result["baseline_traces"], result["candidate_traces"]):
        np.testing.assert_array_equal(baseline["states"], candidate["states"])
        np.testing.assert_array_equal(baseline["actions"], candidate["actions"])


def test_healthy_partial_trace_is_not_accepted_as_a_complete_return():
    with pytest.raises(ValueError, match="Healthy incomplete"):
        rollout.measured_suffix_returns([1., 2.], gamma=.9, remaining=4, terminated=False)


def test_fitting_records_reject_roles_and_other_controller_versions():
    good = dict(source="train", controller_version="C0")
    assert driver.training_records([{"records": [good]}], [], "C0") == [good]
    for invalid in (dict(good, source="calibration"), dict(good, source="evaluation"),
                    dict(good, controller_version="C1")):
        with pytest.raises(ValueError, match="cannot mix"):
            driver.training_records([{"records": [good]}], [{"records": [invalid]}], "C0")


def test_memory_refresh_rejects_nontraining_and_strips_full_bank_evidence():
    episode = dict(source="train", states=np.zeros((2, 19)), warmstarts=np.zeros((2, 9)),
        actions=np.zeros((1, 6)), costs=np.ones(1), gamma=.9, remaining_start=1,
        terminated=False, cost=1., points=[{"large_bank": object()}], records=[object()],
        planning_info=[object()], contexts=[object()])
    archived = driver.archive_episode(episode)
    assert set(archived) == {"states", "warmstarts", "actions", "costs", "gamma",
                             "remaining_start", "terminated", "cost"}
    assert episode["points"]  # original durable evidence remains available
    with pytest.raises(ValueError, match="Only training"):
        driver.archive_episode(dict(episode, source="calibration"))


def test_noise_resolved_calibration_requires_distinct_root_ancestry():
    cfg = driver.ReplanningConfig()
    def row(group, delta=-5., noise=0.):
        return dict(group=group, equal_first_actions=False, prediction=delta,
                    differences=[delta-noise, delta+noise], remaining=100)
    rejected = driver.calibration_summary([row(1), row(1), row(2, -.1, 100.)], cfg)
    assert not rejected["eligible"] and rejected["resolved_distinct_roots"] == 1
    accepted = driver.calibration_summary([row(1), row(2)], cfg)
    assert accepted["eligible"]


def _fake_driver(monkeypatch, calls):
    fits = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def map(self, function, items):
            return map(function, items)

    class Model:
        @classmethod
        def fit(cls, records, **kwargs):
            assert all(r["source"] == "train" and r["controller_version"] == kwargs["controller_version"]
                       for r in records)
            fits.append((copy.deepcopy(records), kwargs["controller_version"]))
            result = cls()
            result.target = kwargs["controller_version"]
            result.metrics = {"training_group_count": 2, "holdout_group_count": 2}
            return result
        def packet(self):
            return {"controller_version": self.target}

    def fake_job(job):
        calls.append((job["kind"], job.get("source"), job.get("reset_id")))
        cfg = driver.ReplanningConfig(**job["config"])
        if job["kind"] == "pair":
            point = job["point"]
            return dict(source=job["source"], group=point["group"], pair_id=point["pair_id"],
                remaining=cfg.total_steps, equal_first_actions=True, records=[],
                baseline_traces=[], candidate_traces=[], ledger={"executed_controls": 0})
        n = cfg.total_steps
        source, version, root = job["source"], job["controller"]["version"], job["reset_id"]
        record = dict(source=source, controller_version=version, group=root)
        point = _point([np.full(6, .1), np.full(6, .2)])
        point.update(group=root, root=root, controller_version=version, pair_id=str(root), source=source)
        return dict(reset_id=root, planner_seed=job["planner_seed"], reset_profile="fake", source=source,
            states=np.asarray([_state(i) for i in range(n+1)]), warmstarts=np.zeros((n+1, 9)),
            actions=np.zeros((n, 6)), costs=np.ones(n), gamma=cfg.gamma, remaining_start=n,
            terminated=False, steps=n, survived=True, duration_seconds=.008*n, velocity_mae=0.,
            cost=float(n), guidance_fraction=0., learned_guidance_fraction=0., action_change_fraction=0.,
            selected_sources={}, ledger={"executed_controls": n}, records=[record],
            points=[point] if job.get("collect_points") else [])

    def no_shadow_change(point, value, alpha, cfg):
        base = point["bank"]["baseline_index"]
        return dict(selected_index=base, baseline_index=base, guidance_used=False,
                    action_changed=False, supported=np.ones(2, dtype=bool), value_prediction_rows=2)

    monkeypatch.setattr(driver, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(driver, "_job", fake_job)
    monkeypatch.setattr(driver, "ReplanningValue", Model)
    monkeypatch.setattr(driver, "rerank_point", no_shadow_change)
    monkeypatch.setattr(driver, "SOURCES", [])
    monkeypatch.setattr(driver, "provenance", lambda: {})
    monkeypatch.setattr(driver, "_revision", lambda: "synthetic-integrity-check")
    return fits


def test_resume_recovers_durable_final_panel_without_reexecuting_it(monkeypatch, tmp_path):
    calls = []
    _fake_driver(monkeypatch, calls)
    cfg = replace(driver.ReplanningConfig(), iterations=1, total_steps=4, workers=1,
        bootstrap_episodes=1, collection_episodes=4, train_pairs=2,
        qualification_episodes=1, evaluation_episodes=1, final_episodes=1,
        eval_every=1, K=1, H=2, query_stride=2)
    study = tmp_path/"study"
    original_save = driver._save

    def interrupt_after_final(path, payload):
        if path.name == "checkpoint.pkl" and (study/"evaluations/final.json").exists():
            raise RuntimeError("synthetic interruption after final evidence")
        original_save(path, payload)

    monkeypatch.setattr(driver, "_save", interrupt_after_final)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        driver.run(study, cfg)
    final_work = json.loads((study/"evaluations/final.json").read_text())["work"]
    previous_calls = list(calls)
    monkeypatch.setattr(driver, "_save", original_save)
    driver.run(study, cfg, resume=True)
    status = json.loads((study/"study_status.json").read_text())
    assert status["status"] == "complete" and status["completed_iterations"] == 1
    assert status["work_is_lower_bound"]
    assert status["work"] == final_work
    assert calls == previous_calls
    assert len((study/"metrics.jsonl").read_text().splitlines()) == 1


def test_epoch_fits_never_accumulate_old_controller_returns_and_final_uses_current_archive(monkeypatch, tmp_path):
    calls = []
    fits = _fake_driver(monkeypatch, calls)
    cfg = replace(driver.ReplanningConfig(), iterations=2, total_steps=4, workers=1,
        bootstrap_episodes=1, collection_episodes=4, train_pairs=2,
        qualification_episodes=1, evaluation_episodes=1, final_episodes=1,
        eval_every=1, K=1, H=2, query_stride=2)
    study = tmp_path/"study"
    driver.run(study, cfg)
    assert len(fits) == 2
    assert [len(rows) for rows, version in fits] == [4, 4]
    first_roots = {row["group"] for row in fits[0][0]}
    next_roots = {row["group"] for row in fits[1][0]}
    assert first_roots.isdisjoint(next_roots)
    assert fits[0][1] != fits[1][1]  # refresh defines a new continuation target
    rows = [json.loads(line) for line in (study/"metrics.jsonl").read_text().splitlines()]
    final = json.loads((study/"evaluations/final.json").read_text())
    assert final["shared_archive_sha256"] == rows[-1]["shared_archive_sha256"]
    assert final["guided"]["archive_sha256"] == final["memory"]["archive_sha256"]
    for row in rows:
        assert row["target_archive_sha256"] == row["shared_archive_sha256"]
        assert row["qualification_records_added"] == row["qualification_archive_episodes_added"] == 0


def test_new_controller_keeps_scorer_target_identity_separate_from_runtime_identity():
    fitted = {"controller_version": "previous-frozen-controller", "weights": np.ones(2)}
    active = driver.controller(value=fitted, alpha=.25)
    assert active["value_target_version"] == "previous-frozen-controller"
    assert active["version"] != active["value_target_version"]
    assert active["value"]["controller_version"] == "previous-frozen-controller"
    assert "reference_controller" not in active


@pytest.mark.parametrize("calibration", [False, True])
def test_shadow_work_counts_every_reranked_bank_even_when_filtered_or_not_selected(monkeypatch, calibration):
    points = []
    for root in range(5):
        point = _point([np.full(6, .1), np.full(6, .2)])
        point.update(group=root, root=root, pair_id=f"point-{root}")
        points.append(point)
    reranked = []
    def rerank(point, value, alpha, cfg):
        root = point["root"]
        reranked.append(root)
        return dict(selected_index=1, baseline_index=0, guidance_used=root % 2 == 0,
                    action_changed=True, supported=np.ones(2, dtype=bool),
                    value_prediction_rows=root+1)
    monkeypatch.setattr(driver, "rerank_point", rerank)
    work = {"probe_value_prediction_rows": 7, "probe_rerank_calls": 2}
    selected = driver.select_points(points, 1, 2, value={"fitted": True}, cfg=_cfg(),
        calibration=calibration, work_ledger=work, work_category="probe")
    assert len(selected) == 1
    assert reranked == [0, 1, 2, 3, 4]
    assert work["probe_value_prediction_rows"] == 7+sum(range(1, 6))
    assert work["probe_rerank_calls"] == 2+5


def test_controller_version_is_canonical_across_mapping_order_aliases_layout_and_roundtrip():
    shared = np.arange(12., dtype=np.float64).reshape(3, 4)
    archive = {"second": shared, "first": shared}
    fitted = {"weights": {"b": np.array([3., 4.]), "a": np.array([1., 2.])},
              "controller_version": "measured-C0"}
    original = driver.controller(archive, fitted, .25)
    equivalent_archive = {"first": np.asfortranarray(shared), "second": shared.copy()}
    equivalent_value = {"controller_version": "measured-C0",
                        "weights": {"a": np.array([1., 2.]), "b": np.array([3., 4.])}}
    equivalent = driver.controller(equivalent_archive, equivalent_value, np.float64(.25))
    assert equivalent["version"] == original["version"]
    restored = pickle.loads(pickle.dumps(original, protocol=4))
    rebuilt = driver.controller(restored["archive"], restored["value"], restored["alpha"])
    assert rebuilt["version"] == original["version"]
    changed_archive = {name: array.copy() for name, array in equivalent_archive.items()}
    changed_archive["first"][0, 0] += 1
    assert driver.controller(changed_archive, equivalent_value, .25)["version"] != original["version"]

