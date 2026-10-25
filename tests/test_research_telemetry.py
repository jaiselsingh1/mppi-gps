"""Array-only diagnostic oracles, frozen holdouts, and optional logger outputs."""
import csv
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.gps.telemetry import (HoldoutRing, StepLogger, TelemetryMetrics,
                               diagnostics_from_arrays, holdout_metrics, spatial_lipschitz)


def test_jensen_and_provided_tail_diagnostics_match_independent_medians() -> None:
    values = diagnostics_from_arrays(np.array([4., 12., 7.]), np.array([2., 8., 6.]),
                                      tail_var_ratio=np.array([0., 1.2, .8]),
                                      lambda_saturation_rate=.25)
    assert values == {"e_step/jensen_gap_median": 2., "e_step/tail_var_ratio": .8,
                      "e_step/lambda_saturation_rate": .25}
    assert diagnostics_from_arrays(np.empty(0), np.empty(0))["e_step/jensen_gap_median"] is None
    assert diagnostics_from_arrays(np.array([np.inf]), np.array([0.]))["e_step/jensen_gap_median"] is None
    with pytest.raises(ValueError, match="shape"):
        diagnostics_from_arrays(np.zeros((3, 1)), np.zeros(3))


def test_lipschitz_uses_cross_trajectory_neighbors_and_exact_norm_formula() -> None:
    obs = np.array([[0.], [.001], [1.], [1.001]])
    values = spatial_lipschitz(obs, 2 * obs, obs, np.array([0, 0, 1, 1]))
    distance = np.array([1., .999, .999, 1.])
    expected_policy = np.median(distance / (distance + 1e-4))
    np.testing.assert_allclose(values["channel/L_policy"], expected_policy)
    np.testing.assert_allclose(values["channel/L_target"], 2 * expected_policy)
    np.testing.assert_allclose(values["channel/lipschitz_ratio"], 2.)
    assert values["channel/lipschitz_pairs"] == 4
    # Masking all rows from trajectory 1 leaves no eligible cross-trajectory pair.
    missing = spatial_lipschitz(obs, obs, obs, np.array([0, 0, 1, 1]),
                               eligible=np.array([True, True, False, False]))
    assert missing["channel/lipschitz_pairs"] == 0
    assert missing["channel/lipschitz_ratio"] is None


def test_lipschitz_sampling_is_deterministic_bounded_and_rng_neutral() -> None:
    obs = np.arange(600, dtype=float)[:, None]
    trajectories = np.arange(600) % 2
    before = np.random.get_state()
    first = spatial_lipschitz(obs, obs * 3, obs, trajectories)
    second = spatial_lipschitz(obs, obs * 3, obs, trajectories)
    after = np.random.get_state()
    assert first == second
    assert first["channel/lipschitz_pairs"] == 256
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    # Identical states from different trajectories are legal; a flat policy
    # makes a nonflat target/policy slope ratio undefined, represented as None.
    flat = spatial_lipschitz(np.zeros((2, 1)), np.array([[0.], [1.]]), np.zeros((2, 1)), np.array([0, 1]))
    assert flat["channel/L_target"] == 10_000.
    assert flat["channel/lipschitz_ratio"] is None


def test_holdout_is_exactly_five_iterations_old_copied_and_capped() -> None:
    ring = HoldoutRing()
    obs = np.arange(900, dtype=float).reshape(300, 3)
    labels = np.ones((300, 2))
    for iteration in range(6):
        ring.add(iteration, obs + iteration, labels * iteration)
    old = ring.get(5)
    assert old is not None and old.iteration == 0
    assert old.observations.shape == (256, 3)
    np.testing.assert_array_equal(old.observations[[0, -1]], obs[[0, -1]])
    obs[:] = 0
    assert old.observations[-1, -1] == 899
    with pytest.raises(ValueError, match="read-only"):
        old.targets[0] = 1
    ring.add(6, obs, labels)
    assert ring.get(6).iteration == 1
    assert ring.get(5) is None  # k=0 has now left the bounded ring.
    with pytest.raises(ValueError, match="increase strictly"):
        ring.add(6, obs, labels)


def test_holdout_does_not_substitute_a_nearby_available_iteration() -> None:
    ring = HoldoutRing()
    ring.add(0, np.ones((2, 3)), np.zeros((2, 1)))
    ring.add(2, np.ones((2, 3)), np.zeros((2, 1)))
    assert ring.get(5).iteration == 0
    assert ring.get(6) is None
    assert ring.get(7).iteration == 2


def test_holdout_drift_and_mse_match_manual_inference_comparison() -> None:
    pre = np.array([[1., 2.], [3., 4.]])
    post = np.array([[2., 2.], [1., 4.]])
    values = holdout_metrics(pre, post, np.zeros((2, 2)))
    assert values == {"m_step/holdout_drift": 2.5, "m_step/holdout_mse_before": 7.5,
                      "m_step/holdout_mse_after": 6.25, "m_step/holdout_mse_delta": -1.25,
                      "m_step/holdout_states": 2}


def test_logger_has_strict_json_dynamic_csv_and_resume(tmp_path: Path, capsys: Any) -> None:
    logger = StepLogger(tmp_path)
    logger.log(TelemetryMetrics(0, {"probe/predicted_improvement": None,
                                     "e_step/jensen_gap_median": np.float64(1.5)}))
    logger.log(TelemetryMetrics(1, {"probe/predicted_improvement": np.inf,
                                     "m_step/holdout_drift": .25}))
    logger.close()
    rows = [json.loads(line) for line in (tmp_path / "telemetry.jsonl").read_text().splitlines()]
    assert rows[1]["probe/predicted_improvement"] is None
    with (tmp_path / "telemetry.csv").open() as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 2 and table[0]["m_step/holdout_drift"] == ""
    assert table[1]["m_step/holdout_drift"] == "0.25"
    assert "telemetry k=1" in capsys.readouterr().out
    resumed = StepLogger(tmp_path, console=False)
    resumed.log(TelemetryMetrics(2, {"collect/mean_ep_length": 100.}))
    with pytest.raises(ValueError, match="increase strictly"):
        resumed.log(TelemetryMetrics(2, {}))
    resumed.close()
    with pytest.raises(RuntimeError, match="close"):
        resumed.log(TelemetryMetrics(3, {}))
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == 3


def test_optional_writers_are_lazy_and_receive_only_available_scalars(tmp_path: Path, monkeypatch: Any) -> None:
    imports: list[str] = []
    events: list[tuple[Any, ...]] = []

    class Writer:
        def add_scalar(self, *args: Any) -> None:
            events.append(("tensorboard", *args))

        def flush(self) -> None:
            events.append(("flush",))

        def close(self) -> None:
            events.append(("close",))

    class Run:
        def log(self, values: dict[str, Any], step: int) -> None:
            events.append(("wandb", values, step))

        def finish(self) -> None:
            events.append(("finish",))

    class FakeModule:
        def init(self, **kwargs: Any) -> Run:
            return Run()

        def SummaryWriter(self, **kwargs: Any) -> Writer:
            return Writer()

    def imported(name: str) -> FakeModule:
        imports.append(name)
        return FakeModule()

    monkeypatch.setattr(importlib, "import_module", imported)
    local = StepLogger(tmp_path / "local", console=False)
    local.log(TelemetryMetrics(0, {"x": 1.}))
    local.close()
    assert imports == []
    remote = StepLogger(tmp_path / "optional", console=False, wandb_project="research", tensorboard=True)
    remote.log(TelemetryMetrics(0, {"x": 1., "missing": None}))
    remote.close()
    remote.close()
    assert imports == ["wandb", "torch.utils.tensorboard"]
    assert ("wandb", {"x": 1.}, 0) in events
    assert ("tensorboard", "x", 1., 0) in events
    assert events.count(("finish",)) == events.count(("close",)) == 1


@pytest.mark.parametrize("failure_stage", ["import", "constructor"])
def test_partial_optional_initialization_finishes_started_wandb_run(
    tmp_path: Path, monkeypatch: Any, failure_stage: str,
) -> None:
    events: list[str] = []

    class Run:
        def finish(self) -> None:
            events.append("wandb_finish")

    class FakeModule:
        def init(self, **kwargs: Any) -> Run:
            events.append("wandb_init")
            return Run()

        def SummaryWriter(self, **kwargs: Any) -> None:
            raise RuntimeError("TensorBoard constructor failed")

    def imported(name: str) -> FakeModule:
        if name == "torch.utils.tensorboard" and failure_stage == "import":
            raise ImportError("TensorBoard import failed")
        return FakeModule()

    monkeypatch.setattr(importlib, "import_module", imported)
    with pytest.raises((ImportError, RuntimeError), match="TensorBoard"):
        StepLogger(tmp_path, console=False, wandb_project="research", tensorboard=True)
    assert events == ["wandb_init", "wandb_finish"]


@pytest.mark.parametrize("failed_backends", [{"tensorboard"}, {"wandb"}, {"tensorboard", "wandb"}])
def test_close_attempts_both_backends_and_marks_closed_even_on_failure(
    tmp_path: Path, failed_backends: set[str],
) -> None:
    events: list[str] = []

    class Writer:
        def close(self) -> None:
            events.append("tensorboard_close")
            if "tensorboard" in failed_backends:
                raise RuntimeError("TensorBoard close failed")

    class Run:
        def finish(self) -> None:
            events.append("wandb_finish")
            if "wandb" in failed_backends:
                raise RuntimeError("WandB finish failed")

    logger = StepLogger(tmp_path, console=False)
    logger._tensorboard, logger._wandb = Writer(), Run()
    with pytest.raises(RuntimeError):
        logger.close()
    assert events == ["tensorboard_close", "wandb_finish"]
    assert logger._closed
    logger.close()
    assert events == ["tensorboard_close", "wandb_finish"]
