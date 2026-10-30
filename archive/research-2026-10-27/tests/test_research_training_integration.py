"""End-to-end research hooks, including zero-rollout-overhead regression."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
from typing import Any

import numpy as np
import pytest

from scripts.training import train_walker_mdgps as training
from scripts.training.train_policy_selective import PolicySelectiveConfig
from src.envs.walker2d import Walker2d
from src.gps.research import (IterationBaseline, ResearchConfig, ResearchSession,
                              deterministic_probe_mask)
from src.gps.health_monitor import EarlyAbortException


class InlinePool:
    """Run the real collector in process so simulator calls can be counted."""

    def __init__(self, workers: int) -> None:
        self.workers = workers

    def __enter__(self) -> InlinePool:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def map(self, function: Callable[[dict[str, Any]], dict[str, Any]],
            jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [function(job) for job in jobs]


def test_two_iteration_training_emits_all_metrics_and_ladder_artifacts(tmp_path) -> None:
    cfg = PolicySelectiveConfig(K=8, H=4, tail_steps=3, episodes_per_iter=2, steps=12,
                                iterations=2, m_step_updates=3, workers=1,
                                gpu_estep=False, ladder_episodes=2, ladder_steps=12,
                                eval_episodes=1, save_labels=True)
    output = tmp_path / "instrumented"
    training.main(output, cfg)
    rows = [json.loads(line) for line in (output / "telemetry.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    required = ["collect/mean_ep_length", "collect/fall_rate", "e_step/share_plans_worse",
                "e_step/median_action_delta", "e_step/sigma_k", "e_step/mean_n_eff",
                "e_step/median_lambda", "e_step/jensen_gap_median", "e_step/tail_var_ratio",
                "e_step/lambda_saturation_rate", "channel/L_target", "channel/L_policy",
                "channel/lipschitz_ratio", "probe/predicted_improvement", "probe/realized_improvement",
                "probe/improvement_realization_ratio", "m_step/holdout_drift"]
    for k, row in enumerate(rows):
        assert all(name in row for name in required)
        assert all(row[name] is None or np.isfinite(row[name]) for name in required)
        with np.load(output / f"labels_{k:03d}.npz") as archive:
            expected_gap = np.median(archive["diagnostics"][:, 3] - archive["expected_costs"])
            np.testing.assert_allclose(row["e_step/jensen_gap_median"], expected_gap)
            np.testing.assert_allclose(row["e_step/tail_var_ratio"], np.median(archive["tail_var_ratio"]))
            assert row["probe/states"] == int(np.ceil(.02 * len(archive["obs"])))
        assert (output / "logs" / f"iter_{k:03d}_limit_cycle.png").exists()
        assert (output / "logs" / f"iter_{k:03d}_limit_cycle.npz").exists()
        ladder = json.loads((output / "logs" / f"iter_{k:03d}_ladder.json").read_text())
        assert ladder["cold_cache_hit"] is bool(k)
    assert len((output / "telemetry.csv").read_text().splitlines()) == 3
    assert json.loads((output / "status.json").read_text())["complete"]


def test_deterministic_two_percent_probes_do_not_change_collection_rng() -> None:
    np.random.seed(19)
    before = np.random.get_state()
    first = deterministic_probe_mask(500, .02, 7, 12)
    after = np.random.get_state()
    np.testing.assert_array_equal(first, deterministic_probe_mask(500, .02, 7, 12))
    assert first.sum() == 10
    assert deterministic_probe_mask(12, .02, 7, 12).sum() == 1
    assert not deterministic_probe_mask(12, 0, 7, 12).any()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])


def test_holdout_drift_uses_the_exact_five_iteration_old_targets(tmp_path) -> None:
    observations = np.array([[1., 0.], [2., 0.]])
    target = np.zeros((2, 2))
    scale = 1.

    def act(batch: np.ndarray) -> np.ndarray:
        return np.asarray(batch) * scale

    with ResearchSession(tmp_path, ResearchConfig()) as session:
        for k in range(6):
            baseline = IterationBaseline(100., 0., np.ones(2), .3, 1.6, 2.)
            before = session.begin(k, observations=observations, targets=target,
                                   old_actions=act(observations), weights=np.ones(2),
                                   trajectory_ids=np.array([0, 1]), plan_cost=np.zeros(2),
                                   expected_cost=np.zeros(2), tail_var_ratio=np.zeros(2),
                                   lambda_saturated=np.zeros(2, bool), action_delta=np.ones(2),
                                   baseline=baseline, act=act)
            if k == 5:
                scale = 2.
            metrics = session.finish(k, before, act=act, probe=None, fit_loss=0.)
        assert metrics.values["m_step/holdout_source_iteration"] == 0
        assert metrics.values["m_step/holdout_states"] == 2
        np.testing.assert_allclose(metrics.values["m_step/holdout_drift"], 2.5)
        np.testing.assert_allclose(metrics.values["m_step/holdout_mse_before"], 1.25)
        np.testing.assert_allclose(metrics.values["m_step/holdout_mse_after"], 5.)


def test_research_diagnostics_add_no_simulator_rollouts_without_probes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(training, "Pool", InlinePool)
    original = Walker2d.batch_rollout
    calls = 0

    def counted(self: Walker2d, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Walker2d, "batch_rollout", counted)
    cfg = training.TrainConfig(K=8, H=4, tail_steps=3, iterations=6, episodes_per_iter=2,
                               steps=6, grad_steps=1, workers=1, eval_episodes=1,
                               probe_probability=0., research_telemetry=False, enable_early_abort=False)
    training.main(tmp_path / "without", cfg)
    without = calls
    calls = 0
    training.main(tmp_path / "with", replace(cfg, research_telemetry=True, enable_early_abort=True))
    assert calls == without
    rows = [json.loads(line) for line in (tmp_path / "with/telemetry.jsonl").read_text().splitlines()]
    assert rows[-1]["m_step/holdout_source_iteration"] == 0
    assert all(row["probe/states"] == 0 for row in rows)


def test_nonfinite_fit_stops_cleanly_with_a_checkpoint_and_postmortem(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(training, "Pool", InlinePool)

    def broken_fit(*args: Any, **kwargs: Any) -> float:
        raise FloatingPointError("Injected nonfinite optimizer loss")

    monkeypatch.setattr(training, "_fit", broken_fit)
    output = tmp_path / "abort"
    training.main(output, training.TrainConfig(K=8, H=4, steps=4, iterations=2,
                                              episodes_per_iter=2, workers=1))
    status = json.loads((output / "status.json").read_text())
    report = json.loads((output / "early_abort_report.json").read_text())
    assert status["early_aborted"] and not status["complete"]
    assert status["checkpoint"].endswith("policy_initial.pt")
    assert report["reason_code"] == "nonfinite_actions_or_weights"
    assert "Injected" in report["original_error"]


def test_broken_optional_writer_cannot_hide_the_health_abort(tmp_path, monkeypatch) -> None:
    from src.gps import research as module

    class BrokenWriter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def log(self, metrics: Any) -> None:
            raise RuntimeError("writer log failed")

        def close(self) -> None:
            raise RuntimeError("writer close failed")

    monkeypatch.setattr(module, "StepLogger", BrokenWriter)
    with pytest.raises(EarlyAbortException) as caught:
        with ResearchSession(tmp_path, ResearchConfig()) as session:
            for k in range(3):
                before = session.begin(k, observations=np.zeros((2, 2)), targets=np.zeros((2, 2)),
                                       old_actions=np.zeros((2, 2)), weights=np.ones(2),
                                       trajectory_ids=np.array([0, 1]), plan_cost=np.zeros(2),
                                       expected_cost=np.zeros(2), tail_var_ratio=np.zeros(2),
                                       lambda_saturated=np.ones(2, bool), action_delta=np.zeros(2),
                                       baseline=IterationBaseline(100., 0., np.ones(2), .3, 1.6, 2.),
                                       act=lambda obs: np.zeros_like(obs))
                session.finish(k, before, act=lambda obs: np.zeros_like(obs), probe=None,
                               fit_loss=0., defer_log=True)
    assert caught.value.reason_code == "trust_region_collapse"
    assert "writer log failed" in caught.value.report["telemetry_write_error"]
    assert "writer close failed" in caught.value.report["writer_close_error"]


def test_bad_resume_holdout_is_rejected_before_opening_external_writers(tmp_path, monkeypatch) -> None:
    from src.gps import research as module

    def must_not_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Optional logging backend opened before archive recovery succeeded")

    monkeypatch.setattr(module, "StepLogger", must_not_open)
    np.savez(tmp_path / "labels_000.npz", obs=np.array([[np.nan, 0.]]), labels=np.zeros((1, 2)))
    with pytest.raises(ValueError, match="finite"):
        ResearchSession(tmp_path, ResearchConfig(wandb_project="configured"), resume_at=5)
