"""Collector purity, seed accounting and tiny four-arm study integration."""

from dataclasses import replace
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.training import train_decoupled_study as training
from src.gps.policy import TanhMLPPolicy


def _actor():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return TanhMLPPolicy(17, -np.ones(6), np.ones(6)).eval()


def test_defaults_match_authorized_fifteen_iteration_four_arm_pilot():
    cfg = training.StudyConfig()
    training.validate(cfg)
    assert cfg.iterations == 15 and cfg.K == 256 and cfg.H == 32 and cfg.tail_steps == 100
    assert cfg.m_step_updates == cfg.value_updates == 2000
    assert cfg.value_gamma == .995 and cfg.value_steps == 1000 and cfg.value_episodes == 10
    assert cfg.eval_every == 5 and cfg.eval_steps == 1000 and cfg.eval_episodes == 100
    assert training.ARMS == ("Baseline-Anchor", "Baseline-Selective", "Split-Medoid", "Split-Medoid-Critic")


def test_value_collection_is_deterministic_current_policy_and_respects_cutoff():
    cfg = replace(training.StudyConfig(), value_episodes=1, value_steps=7)
    actor = _actor()
    job = training.collection_jobs(actor, cfg, 0, value=True)[0]
    assert job["noise"] == 0 and job["steps"] == 7
    first = training.collect_episode(job)
    second = training.collect_episode({**job, "noise_seed": job["noise_seed"] + 10})
    assert first["steps"] == 7 and not first["terminated"]
    assert first["obs"].shape == (8, 17) and first["states"].shape == (8, 19)
    for key in ("obs", "states", "warmstarts", "costs", "actions"):
        np.testing.assert_array_equal(first[key], second[key])
    for observation, action in zip(first["obs"][:-1], first["actions"]):
        np.testing.assert_array_equal(action, actor.act(observation, deterministic=True))


def test_noisy_collection_is_reproducible_but_noise_is_not_used_for_value_data():
    cfg = replace(training.StudyConfig(), episodes=1, collection_steps=6)
    actor = _actor()
    job = training.collection_jobs(actor, cfg, 0)[0]
    first, second = training.collect_episode(job), training.collect_episode(job)
    np.testing.assert_array_equal(first["actions"], second["actions"])
    other = training.collect_episode({**job, "noise_seed": job["noise_seed"] + 1})
    assert not np.array_equal(first["actions"], other["actions"])
    assert np.isfinite(first["costs"]).all() and (first["costs"] >= 0).all()


def test_collector_stops_at_first_fall_stores_raw_cost_and_closes_environment(monkeypatch):
    instances = []

    class FakeEnv:
        def __init__(self, *, max_episode_steps):
            self.deadline, self.closed = max_episode_steps, False
            self.walker = SimpleNamespace(action_bounds=(-np.ones(6), np.ones(6)),
                                          get_state=lambda: np.zeros(19), get_warmstart=lambda: np.zeros(9))
            instances.append(self)

        def reset(self, *, seed):
            self.t = 0
            return np.zeros(17), {}

        def step(self, action):
            self.t += 1
            fall = self.t == 2
            cost = 25.0 if fall else 2.0
            reward = -cost - (995 if fall else 0)
            return np.zeros(17), reward, fall, False, {"task_cost": cost, "x_velocity": 1.5}

        def close(self):
            self.closed = True

    monkeypatch.setattr(training, "WalkerRL", FakeEnv)
    monkeypatch.setattr(training, "ValueCreditTrainer", lambda *a, **k: pytest.fail("Collector loaded a critic"))
    row = training.collect_episode({"policy": _actor().policy_state(), "steps": 10,
                                    "reset_seed": 7, "noise_seed": 8, "noise": 0})
    assert instances[0].deadline == 10 and instances[0].closed
    assert row["steps"] == 2 and row["terminated"]
    np.testing.assert_array_equal(row["costs"], [2.0, 25.0])
    assert len(row["obs"]) == 3  # final observation retained, no postfall actions


def test_roots_exclude_final_observation_and_common_count_is_budget_matched():
    episodes = [{key: np.arange(7)[:, None] for key in ("obs", "states", "warmstarts")}]
    extracted = training.roots(episodes, 2)
    np.testing.assert_array_equal(extracted["states"][:, 0], [0, 2, 4])
    assert training.matched_root_count([257, 300, 1000, 900], 2000, 64) == 256
    assert training.matched_root_count([2500, 2500, 2500, 2500], 2000, 64) == 1984
    assert training.matched_root_count([1000, 1000, 63, 1000], 2000, 64) == 0


def test_all_arms_share_seed_schedule_and_value_resets_are_disjoint():
    actor, cfg = _actor(), training.StudyConfig()
    expected = training.collection_jobs(actor, cfg, 3)
    for _ in training.ARMS:
        actual = training.collection_jobs(actor, cfg, 3)
        assert [(j["reset_seed"], j["noise_seed"]) for j in actual] == [
            (j["reset_seed"], j["noise_seed"]) for j in expected]
    resets = {j["reset_seed"] for i in range(cfg.iterations) for j in training.collection_jobs(actor, cfg, i)}
    value_resets = {j["reset_seed"] for i in range(cfg.iterations) for j in training.collection_jobs(actor, cfg, i, value=True)}
    assert len(resets) == cfg.iterations * cfg.episodes
    assert len(value_resets) == cfg.iterations * cfg.value_episodes
    assert resets.isdisjoint(value_resets)
    assert resets.isdisjoint(range(cfg.eval_seed, cfg.eval_seed + cfg.eval_episodes))


@pytest.mark.parametrize("changes", [
    {"seed": 490},  # collection would consume reserved final resets5000000+
    {"eval_seed": 2 ** 32 - 1},
    {"value_episodes": 101},  # stride100 would alias the next iteration
    {"episodes": 1000},  # fifteen iterations overflow the per-training-seed block
    {"tail_steps": -1},
])
def test_invalid_seed_or_horizon_budgets_are_rejected_before_acquisition(changes):
    with pytest.raises(ValueError):
        training.validate(replace(training.StudyConfig(), **changes))


class InlinePool:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, rows):
        return list(map(function, rows))


@pytest.mark.skipif(importlib.util.find_spec("src.gps.decoupled_gpu") is None,
                    reason="Shared discounted-score helper is being implemented separately")
def test_two_iteration_cpu_study_preserves_four_arms_and_records_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(training.mp, "get_context", lambda _: SimpleNamespace(Pool=lambda _: InlinePool()))
    cfg = replace(training.StudyConfig(), iterations=2, backend="cpu", K=8, H=2, tail_steps=3,
                  states_per_batch=1, max_label_states=2, episodes=1, collection_steps=6,
                  value_episodes=1, value_holdout=0, value_steps=100, value_updates=2, m_step_updates=2,
                  batch_size=2, workers=1, eval_every=1, eval_episodes=1, eval_steps=8)
    out = tmp_path / "pilot"
    training.run(out, cfg)
    status = json.loads((out / "status.json").read_text())
    assert status["phase"] == "complete" and status["source_unchanged"]
    rows = {}
    for arm in training.ARMS:
        rows[arm] = [json.loads(line) for line in (out / arm / "metrics.jsonl").read_text().splitlines()]
        assert [r["iteration"] for r in rows[arm]] == [0, 1]
        for row in rows[arm]:
            assert row["label_states"] == 2 and row["eval_dev"]["episodes"] == 1
            if row["replay_positive_rows"] == 0:
                assert row["parameters_before"] == row["parameters_after"]
                assert row["m_step_applied_updates"] == 0
            assert (out / arm / f"policy_{row['iteration']:03d}.pt").exists()
    for i in range(2):
        assert len({rows[a][i]["label_states"] for a in training.ARMS}) == 1
        candidates = [rows[a][i]["ledger"]["logical"]["candidate_controls"] for a in training.ARMS]
        assert candidates[:3] == [80, 80, 80] and candidates[3] == 32
    critic_rows = rows[training.ARMS[-1]]
    assert critic_rows[0]["value_gate"]["bootstrap_bypass"]
    assert not critic_rows[1]["value_gate"]["bootstrap_bypass"]
    assert all(r["value_fit"]["cost_units"].startswith("discounted raw task cost") for r in critic_rows)
    assert all(r["ledger"]["value_acquisition_controls"] > 0 for r in critic_rows)
    assert all(rows[a][0]["ledger"]["value_acquisition_controls"] == 0 for a in training.ARMS[:-1])


def test_equal_cost_mean_retained_only_by_anchor_with_actual_targets_saved(tmp_path, monkeypatch):
    """A strict kernel tie rejection must not silently change old Anchor labels."""
    from src.gps import decoupled_cpu

    emitted, fitted = [], []

    class TiedSimulator:
        def __init__(self, env, cfg, tail_steps, states_per_batch):
            self.env, self.cfg = env, cfg

        def __call__(self, states, warmstarts, actor, noise_seeds, **kwargs):
            obs = self.env.rollout_states_to_obs(states)
            nominal = np.stack([actor.act(row) for row in obs])
            mean = np.clip(nominal + .1, -1, 1)
            emitted.append({"nominal": nominal.copy(), "mean": mean.copy(), **kwargs})
            count = len(states)
            counts = {key: 0 for key in ("accepted", "mean_rejected_count", "bimodal_collapse_count",
                      "medoid_rescued_collapse_count", "rescue_eligible_count", "rescued_count")}
            counts.update(states=count, mean_tied_count=count)
            return {"labels": nominal, "nominal_first": nominal, "mean_first": mean,
                    "nominal_cost": np.ones(count), "mean_cost": np.ones(count),
                    "plan_cost": np.ones(count), "source_codes": np.zeros(count, dtype=int),
                    "endpoint_obs": obs, "endpoint_healthy": np.ones(count, dtype=bool),
                    "n_eff": np.full(count, self.cfg.K * .2), "diagnostics": counts,
                    "ledger": {"logical": {"candidate_controls": count * self.cfg.K * self.cfg.H}}}

    def capture_fit(actor, observations, actions, cfg, rng, weights):
        fitted.append((actions.copy(), weights.copy()))
        return 0.

    monkeypatch.setattr(decoupled_cpu, "DecoupledCPUEStep", TiedSimulator)
    monkeypatch.setattr(training, "_fit", capture_fit)
    monkeypatch.setattr(training.mp, "get_context", lambda _: SimpleNamespace(Pool=lambda _: InlinePool()))
    cfg = replace(training.StudyConfig(), iterations=1, backend="cpu", K=8, H=2, tail_steps=3,
                  states_per_batch=1, max_label_states=2, episodes=1, collection_steps=6,
                  value_episodes=1, value_holdout=0, value_steps=100, value_updates=1,
                  m_step_updates=1, batch_size=2, workers=1, eval_episodes=1, eval_steps=8)
    out = tmp_path / "ties"
    training.run(out, cfg)
    assert len(emitted) == len(fitted) == len(training.ARMS)
    for i, arm in enumerate(training.ARMS):
        expected = emitted[i]["mean"] if i < 2 else emitted[i]["nominal"]
        np.testing.assert_array_equal(fitted[i][0], expected)
        np.testing.assert_array_equal(fitted[i][1], np.ones(2) if i == 0 else np.zeros(2))
        with np.load(out / arm / "labels_000.npz") as labels:
            np.testing.assert_array_equal(labels["actions"], fitted[i][0])
            np.testing.assert_array_equal(labels["weights"], fitted[i][1])
        assert emitted[i]["proposal_mode"] == ("seeded" if i < 2 else "split_proposal")
        assert emitted[i]["synthesis_mode"] == ("mean_verified" if i < 2 else "medoid_fallback")
