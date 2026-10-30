"""Small CPU checks of ladder semantics, cache independence, and cycle evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import src.gps.eval_ladder as ladder
from src.envs.walker2d import Walker2d
from src.rl.walker import DEFAULT_TASK
from src.utils.config import MPPIConfig


def _zero_policy(observations: np.ndarray) -> np.ndarray:
    return np.zeros((len(observations), 6), dtype=float)


def _smoke_planner() -> MPPIConfig:
    return MPPIConfig(K=8, H=4, lam=3., noise_sigma=.3, initial_iterations=2,
                      lam_target_neff_fraction=.2)


@pytest.mark.parametrize("fraction", [0., 1., float("nan")])
def test_invalid_guided_effective_sample_fraction_is_rejected(fraction: float) -> None:
    with pytest.raises(ValueError, match="guided_neff_fraction"):
        ladder.LadderConfig(guided_neff_fraction=fraction)


def test_periodicity_refuses_constant_drift_and_short_prefixes() -> None:
    config = ladder.LadderConfig(transient_seconds=0.)
    for values in (np.ones((500, 3)), np.arange(500)[:, None] * np.array([[.1, -.2, .3]])):
        result = ladder.state_periodicity(values, dt=.008, config=config)
        assert result.score is None
        assert result.reason == "no_oscillation"
    short = ladder.state_periodicity(np.zeros((12, 3)), dt=.008, config=config)
    assert short.score is None
    assert short.reason == "insufficient_cycles"


def test_periodicity_identifies_physical_cycle_instead_of_low_lag_peak() -> None:
    dt = .008
    times = np.arange(750) * dt
    values = np.stack([np.sin(2 * np.pi * times / .5), np.cos(2 * np.pi * times / .5)], axis=1)
    result = ladder.state_periodicity(values, dt=dt, config=ladder.LadderConfig())
    assert result.score is not None and result.score > .95
    assert result.period_seconds == pytest.approx(.5, abs=dt)
    assert result.period_seconds >= .16


def test_cold_controller_never_calls_policy() -> None:
    def forbidden_policy(observations: np.ndarray) -> np.ndarray:
        raise AssertionError("Cold baseline must be independent of the neural policy")

    trace, _ = ladder._episode(controller="mppi_cold_online", act_batch=forbidden_policy,
                               config=ladder.LadderConfig(episodes=1, steps=5),
                               planner_cfg=_smoke_planner(), tail_steps=0,
                               task_config=DEFAULT_TASK, episode=0, guided_factory=None)
    assert trace.metrics.steps == 5
    assert trace.metrics.controller == "mppi_cold_online"


def test_guided_controller_discards_shifted_plans() -> None:
    env = Walker2d(**json.loads(DEFAULT_TASK.read_text()), nthread=1)
    try:
        observation = env.reset()
        state, warmstart = env.get_state(), env.get_warmstart()
        controller = ladder.CPUGuidedController(env=env, act_batch=_zero_policy,
                                               planner_cfg=_smoke_planner(), tail_steps=3, seed=41)
        controller.planner.U[:] = .8
        controller.planner._rng = np.random.default_rng(99)
        first = controller.action(observation=observation, state=state, warmstart=warmstart)
        controller.planner.U[:] = -.8
        controller.planner._rng = np.random.default_rng(99)
        second = controller.action(observation=observation, state=state, warmstart=warmstart)
        np.testing.assert_allclose(first, second, atol=0., rtol=0.)
        assert controller.planner.initial_iterations == 1
    finally:
        env.close()


def test_fall_cost_includes_one_terminal_and_remaining_absorbing_steps() -> None:
    def pushing_policy(observations: np.ndarray) -> np.ndarray:
        return -np.ones((len(observations), 6), dtype=float)

    config = ladder.LadderConfig(episodes=1, steps=100)
    trace, _ = ladder._episode(controller="pi_standalone", act_batch=pushing_policy,
                               config=config, planner_cfg=_smoke_planner(), tail_steps=3,
                               task_config=DEFAULT_TASK, episode=0, guided_factory=None)
    assert not trace.metrics.survived
    assert trace.metrics.steps < config.steps
    assert trace.metrics.executed_cost == pytest.approx(trace.costs.sum())
    assert trace.metrics.absorbing_cost == 5. * (config.steps - trace.metrics.steps)
    assert trace.metrics.total_cost == pytest.approx(trace.costs.sum() + trace.metrics.absorbing_cost)
    assert trace.metrics.return_value == -trace.metrics.total_cost
    assert trace.metrics.action_jerk == 0.


@pytest.fixture(scope="module")
def smoke_study(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    directory = tmp_path_factory.mktemp("ladder_smoke")
    config = ladder.LadderConfig(episodes=2, steps=12)
    first = ladder.evaluate_ladder_and_regularity(act_batch=_zero_policy, out_dir=directory,
                                                  iteration=0, planner_cfg=_smoke_planner(),
                                                  tail_steps=3, config=config)
    second = ladder.evaluate_ladder_and_regularity(act_batch=_zero_policy, out_dir=directory,
                                                   iteration=1, planner_cfg=_smoke_planner(),
                                                   tail_steps=3, config=config)
    return directory, first, second


def test_small_two_iteration_smoke_and_cold_cache(smoke_study: tuple[Path, dict[str, Any], dict[str, Any]]) -> None:
    directory, first, second = smoke_study
    assert first["cold_cache_hit"] is False and second["cold_cache_hit"] is True
    assert second["controller_seconds"]["mppi_cold_online"] == 0.
    assert first["summary"]["mppi_cold_online"] == second["summary"]["mppi_cold_online"]
    assert second["cold_original_elapsed_seconds"] > 0.
    for report in (first, second):
        assert len(report["episodes"]) == 8
        assert report["controller_configs"]["mppi_guided_online"]["planner"]["initial_iterations"] == 1
        assert report["controller_configs"]["mppi_guided_online"]["planner"]["lam_target_neff_fraction"] == .2
        assert report["controller_configs"]["mppi_cold_online"]["policy_tail_steps"] == 0
        assert set(report["summary"]) == set(ladder.CONTROLLERS)
        for summary in report["summary"].values():
            assert summary["periodicity_score"] is None
            assert summary["periodicity_valid_episodes"] == 0
            assert summary["episodes"] == 2
        for kind in ("json", "traces", "figure"):
            assert Path(report["artifact_paths"][kind]).stat().st_size > 0
        with np.load(report["artifact_paths"]["traces"], allow_pickle=False) as data:
            assert len(data["controllers"]) == 8
            assert data["step_offsets"][-1] == len(data["actions"])
            assert data["state_offsets"][-1] == len(data["states"])
            assert data["states"].dtype == np.float32
    history = json.loads((directory / "logs" / "ladder_history.json").read_text())
    assert [record["iteration"] for record in history["records"]] == [0, 1]


def test_cache_refuses_changed_evaluation_protocol(smoke_study: tuple[Path, dict[str, Any], dict[str, Any]]) -> None:
    directory, _, _ = smoke_study
    with pytest.raises(ValueError, match="fingerprint"):
        ladder.evaluate_ladder_and_regularity(act_batch=_zero_policy, out_dir=directory,
                                              iteration=4, planner_cfg=_smoke_planner(), tail_steps=3,
                                              config=ladder.LadderConfig(episodes=2, steps=13))


def test_cache_key_excludes_policy_and_invalid_cache_is_not_reused(tmp_path: Path) -> None:
    config = ladder.LadderConfig(episodes=1, steps=3)
    first = ladder.evaluate_ladder_and_regularity(act_batch=_zero_policy, out_dir=tmp_path,
                                                  iteration=0, planner_cfg=_smoke_planner(), tail_steps=0, config=config)

    def different_policy(observations: np.ndarray) -> np.ndarray:
        return np.full((len(observations), 6), .05)

    second = ladder.evaluate_ladder_and_regularity(act_batch=different_policy, out_dir=tmp_path,
                                                   iteration=4, planner_cfg=_smoke_planner(), tail_steps=0, config=config)
    assert first["fingerprint"] == second["fingerprint"]
    assert second["cold_cache_hit"]
    cache = tmp_path / "logs" / "mppi_cold_cache.npz"
    cache.write_bytes(cache.read_bytes() + b"damaged")
    with pytest.raises(ValueError, match="checksum"):
        ladder.evaluate_ladder_and_regularity(act_batch=different_policy, out_dir=tmp_path,
                                              iteration=9, planner_cfg=_smoke_planner(), tail_steps=0, config=config)


def test_cold_cache_must_start_at_iteration_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iteration 0"):
        ladder.evaluate_ladder_and_regularity(act_batch=_zero_policy, out_dir=tmp_path,
                                              iteration=4, planner_cfg=_smoke_planner(), tail_steps=0,
                                              config=ladder.LadderConfig(episodes=1, steps=2))
