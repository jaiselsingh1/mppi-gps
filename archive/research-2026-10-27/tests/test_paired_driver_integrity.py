"""Independent ownership, finite-return and qualification checks without Walker."""
from dataclasses import replace
import json

import numpy as np
import pytest

import scripts.training.train_paired_endpoint as driver
from src.gps.expert_continuation import Snapshot, finite_absorbing_cost


def _state(value=0.):
    result = np.zeros(19)
    result[2], result[4] = 1.25, value
    return result


def _point():
    plans = [np.full((3, 6), value) for value in (.1, .2)]
    return dict(snapshot=Snapshot(_state(), np.arange(9.)), group=31_000_000,
        step=0, remaining=10, H=3, gamma_h=.9**3, pair_id="measured-pair",
        equal_plans=False, mode="guided", acquisition_mode="guided",
        guidance_used=True, action_changed=True,
        candidate=dict(plan=plans[0], endpoint=_state(1.), prefix_score=2.,
                       predicted_tail=3., supported=True),
        baseline=dict(plan=plans[1], endpoint=_state(2.), prefix_score=1.,
                      predicted_tail=4., supported=True))


def test_prefix_fall_executes_once_and_charges_complete_finite_remainder(monkeypatch):
    instances = []

    class FakeWalker:
        def __init__(self, gamma, max_episode_steps):
            self.walker = self
            self.n = max_episode_steps
            self.controls = 0
            self.closed = False
            instances.append(self)

        def get_state(self):
            return _state(float(self.controls))

        def get_warmstart(self):
            return np.full(9, self.controls)

        def step(self, action):
            self.controls += 1
            return None, None, self.controls == 2, False, {
                "task_cost": [3., 9.][self.controls-1]}

        def close(self):
            self.closed = True

    restored = []
    monkeypatch.setattr(driver, "WalkerRL", FakeWalker)
    monkeypatch.setattr(driver, "_restore", lambda env, snapshot: restored.append(snapshot))
    cfg = replace(driver.PairedConfig(), H=3, gamma=.9)
    snapshot = Snapshot(_state(), np.arange(9.))
    result = driver.prefix(cfg, snapshot, np.zeros((3, 6)), 10)
    assert restored == [snapshot]
    assert instances[0].n == 10 and instances[0].controls == 2 and instances[0].closed
    assert result["fell"] and result["remaining"] == 8
    assert result["cost"] == pytest.approx(3. + .9*9.)
    assert result["known_full_cost"] == pytest.approx(
        3. + .9*9. + .9**2*finite_absorbing_cost(.9, 8))


def _patch_measurement(monkeypatch, *, candidate_falls=False):
    prefix_calls, suffix_calls = [], []

    def prefix(cfg, snapshot, plan, remaining):
        side = "candidate" if plan[0, 0] == .1 else "baseline"
        prefix_calls.append((side, snapshot, remaining))
        fell = candidate_falls and side == "candidate"
        h = 2 if fell else 3
        cost = 20. if fell else (2. if side == "candidate" else 1.)
        return dict(snapshot=Snapshot(_state(1. if side == "candidate" else 2.), np.arange(9.)),
            cost=cost, controls=h, fell=fell, remaining=remaining-h,
            known_full_cost=cost + cfg.gamma**h*finite_absorbing_cost(cfg.gamma, remaining-h)
                if fell else None)

    def episode(cfg, reference, **kwargs):
        suffix_calls.append((reference, kwargs))
        cost = 10. + kwargs["snapshot"].state[4] + kwargs["planner_seed"] % 2
        n = kwargs["remaining"]
        return dict(cost=cost, ledger=dict(executed_controls=n,
            candidate_controls=2*n, memory_controls=0, verification_controls=n, planner_calls=n))

    monkeypatch.setattr(driver, "prefix", prefix)
    monkeypatch.setattr(driver, "episode", episode)
    return prefix_calls, suffix_calls


def test_pair_matches_suffix_seed_per_replica_and_charges_every_execution(monkeypatch):
    prefixes, suffixes = _patch_measurement(monkeypatch)
    cfg = replace(driver.PairedConfig(), H=3, gamma=.9)
    point, reference = _point(), {"reference": "frozen"}
    result = driver.measure_pair(cfg, reference, point, oracle_seed=400, source="train")
    assert [side for side, _, _ in prefixes] == ["candidate", "baseline"]
    assert all(snapshot is point["snapshot"] and n == 10 for _, snapshot, n in prefixes)
    assert [args["planner_seed"] for _, args in suffixes] == [400, 401, 400, 401]
    assert all(ref is reference and args["remaining"] == 7 for ref, args in suffixes)
    assert [r["oracle_seed"] for r in result["records"]] == [400, 401, 400, 401]
    assert [r["side"] for r in result["records"]] == ["candidate"]*2 + ["baseline"]*2
    assert all(r["source"] == "train" and r["group"] == point["group"]
               and r["pair_id"] == point["pair_id"] for r in result["records"])
    assert result["ledger"]["prefix_controls"] == 6
    assert result["ledger"]["continuation_executed_controls"] == 28
    assert result["ledger"]["continuation_candidate_controls"] == 56
    assert len(result["archive_episodes"]) == len(result["records"]) == 4
    expected = (2. + .9**3*11.) - (1. + .9**3*12.)
    np.testing.assert_allclose(result["measured_differences"], expected)


def test_calibration_measurements_never_return_fit_or_archive_rows(monkeypatch):
    _, suffixes = _patch_measurement(monkeypatch)
    result = driver.measure_pair(replace(driver.PairedConfig(), H=3, gamma=.9),
        {"reference": "frozen"}, _point(), oracle_seed=400, source="calibration")
    assert len(suffixes) == 4  # evidence was actually measured, not silently skipped
    assert result["records"] == result["archive_episodes"] == []
    assert "records" not in driver.public_pair(result)
    assert "archive_episodes" not in driver.public_pair(result)


def test_prefix_fall_stays_in_pair_difference_without_suffix_or_invalid_label(monkeypatch):
    _, suffixes = _patch_measurement(monkeypatch, candidate_falls=True)
    cfg = replace(driver.PairedConfig(), H=3, gamma=.9)
    result = driver.measure_pair(cfg, {}, _point(), oracle_seed=400, source="train")
    fallen = result["branches"]["candidate"]
    expected = 20. + .9**2*finite_absorbing_cost(.9, 8)
    np.testing.assert_allclose(fallen["full_costs"], [expected, expected])
    assert fallen["suffix_costs"] == [None, None]
    assert len(suffixes) == 2 and len(result["records"]) == 2
    assert all(row["side"] == "baseline" for row in result["records"])
    assert result["ledger"]["prefix_controls"] == 5
    assert len(result["measured_differences"]) == 2


def test_deployed_pair_uses_original_selection_and_batch_prediction_not_shadow():
    class Planner:
        def __init__(self):
            plans = np.arange(3*3*6, dtype=float).reshape(3, 3, 6)
            self.last_decision = dict(selected_index=1, baseline_index=0, plans=plans,
                endpoints=np.array([_state(i) for i in range(3)]),
                local_costs=np.array([1., 2., 3.]), sources=["sample"]*3,
                scores=np.array([4., 2., 3.]),
                predicted_values=np.array([100., 50., 1.]), supported=np.array([True, True, False]),
                guidance_used=True, action_changed=True)

        def acquisition_pair(self, *args, **kwargs):
            return dict(candidate={"index":2}, baseline={"index":0}, remaining=10,
                        H=3, gamma_h=.9**3, acquisition_mode="guided",
                        local_costs=self.last_decision["local_costs"].copy(),
                        acquisition_scores=np.array([3., 2., 1.]))

    planner = Planner()
    point = driver._point(planner, Snapshot(_state(), np.zeros(9)), 31_000_000, 0,
        np.random.default_rng(0), object(), "deployed", .1)
    assert point["candidate"]["index"] == 1
    assert point["candidate"]["predicted_tail"] == 50.
    assert point["baseline"]["predicted_tail"] == 100.
    assert point["candidate"]["supported"]
    np.testing.assert_array_equal(point["bank_selection_scores"], planner.last_decision["scores"])
    np.testing.assert_array_equal(point["candidate"]["plan"], planner.last_decision["plans"][1])


def _calibration_row(group, *, delta=-5., se=0., predicted=-5., equal=False):
    return dict(group=group, equal_plans=equal, guidance_used=True,
        predicted_full_difference=predicted, measured_mean_difference=delta,
        measured_difference_se=se, remaining=100,
        branches={side:{"original_supported":True} for side in ("candidate", "baseline")})


def test_equal_plans_cannot_qualify_and_supported_changed_pairs_can():
    cfg = driver.PairedConfig()
    assert not driver.pair_calibration([
        _calibration_row(1, equal=True), _calibration_row(2, equal=True)], cfg)["eligible"]
    assert driver.pair_calibration([
        _calibration_row(1), _calibration_row(2)], cfg)["eligible"]


def test_noise_resolved_calibration_must_cover_two_distinct_roots():
    cfg = driver.PairedConfig()
    result = driver.pair_calibration([
        _calibration_row(1), _calibration_row(1),
        _calibration_row(2, delta=-.1, se=10., predicted=-.1)], cfg)
    assert not result["eligible"]
    assert result["resolved_distinct_roots"] == 1


def test_pair_selection_deduplicates_decisions_balances_roots_and_keeps_equal_pairs():
    points = [dict(pair_id=f"{group}:{i}", group=group, equal_plans=(i == 0), guidance_used=True)
              for group in (10, 20) for i in range(4)]
    selected = driver.select_pairs(points + [points[0]], 4, 7)
    assert len(selected) == len({row["pair_id"] for row in selected}) == 4
    assert [sum(row["group"] == g for row in selected) for g in (10, 20)] == [2, 2]
    assert len(driver.select_pairs(points, 8, 7)) == 8
    assert all(not row["equal_plans"] for row in driver.select_pairs(points, 8, 7, changed_only=True))


def test_resume_recovers_durable_final_evaluation_work_without_reexecuting(monkeypatch, tmp_path):
    """Interrupt between the final JSON and checkpoint; recover its known ledger."""
    class Pool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, items):
            return map(function, items)

    cfg = replace(driver.PairedConfig(), iterations=1, total_steps=4, workers=1,
        bootstrap_episodes=1, collection_episodes=2, train_pairs=2,
        qualification_episodes=1, evaluation_episodes=1, final_episodes=1,
        eval_every=1, K=1, H=2, query_stride=2)
    calls = []

    def fake_job(job):
        calls.append((job["kind"], job.get("reset_id")))
        if job["kind"] == "pair":
            point = job["point"]
            return dict(source="train", group=point["group"], pair_id=point["pair_id"],
                equal_plans=True, records=[], archive_episodes=[], ledger={"prefix_controls":4})
        n = cfg.total_steps
        points = []
        if job.get("collect_pairs"):
            point = _point()
            point.update(group=job["reset_id"], pair_id=str(job["reset_id"]), equal_plans=True)
            points.append(point)
        return dict(reset_id=job["reset_id"], planner_seed=job["planner_seed"],
            reset_profile="fake", states=np.array([_state() for _ in range(n+1)]),
            warmstarts=np.zeros((n+1, 9)), actions=np.zeros((n, 6)), costs=np.ones(n),
            gamma=cfg.gamma, remaining_start=n, terminated=False, steps=n,
            survived=True, duration_seconds=.008*n, velocity_mae=0., cost=float(n),
            guidance_fraction=0., action_change_fraction=0., points=points,
            selected_sources={}, ledger=dict(executed_controls=n, candidate_controls=2*n,
                memory_controls=0, verification_controls=n, planner_calls=n))

    monkeypatch.setattr(driver, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(driver, "_job", fake_job)
    monkeypatch.setattr(driver, "SOURCES", [])
    monkeypatch.setattr(driver, "provenance", lambda: {})
    monkeypatch.setattr(driver, "_revision", lambda: "test-only")
    original_save = driver._save
    study = tmp_path/"study"

    def interrupted_save(path, payload):
        if path.name == "checkpoint.pkl" and (study/"evaluations/final.json").exists():
            raise RuntimeError("synthetic interruption after durable final evaluation")
        original_save(path, payload)

    monkeypatch.setattr(driver, "_save", interrupted_save)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        driver.run(study, cfg)
    final_work = json.loads((study/"evaluations/final.json").read_text())["work"]
    assert final_work["evaluation_guided_executed_controls"] == 12
    calls_before_resume = list(calls)
    monkeypatch.setattr(driver, "_save", original_save)
    driver.run(study, cfg, resume=True)
    status = json.loads((study/"study_status.json").read_text())
    assert status["status"] == "complete" and status["completed_iterations"] == 1
    assert status["work"] == final_work
    assert status["work_is_lower_bound"]
    assert calls == calls_before_resume
    assert len((study/"metrics.jsonl").read_text().splitlines()) == 1
