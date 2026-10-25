"""Boundary, coverage, contiguity and post-mortem tests for early stopping."""
import json
from pathlib import Path

import pytest

from src.gps.health_monitor import EarlyAbortException, TrainingHealthMonitor
from src.gps.telemetry import TelemetryMetrics


def _metrics(iteration: int, *, length: float = 100., predicted: float | None = 1.,
             realized: float | None = 1., saturation: float | None = 0.) -> TelemetryMetrics:
    return TelemetryMetrics(iteration, {"collect/mean_ep_length": length,
                                        "probe/predicted_improvement": predicted,
                                        "probe/realized_improvement": realized,
                                        "e_step/lambda_saturation_rate": saturation})


def test_bootstrap_failure_uses_full_last_three_iteration_means_and_saves_report(tmp_path: Path) -> None:
    path = tmp_path / "postmortem.json"
    monitor = TrainingHealthMonitor(report_path=path)
    monitor.observe(_metrics(10, length=50., realized=1.))
    monitor.observe(_metrics(11, length=70., realized=-2.))
    with pytest.raises(EarlyAbortException) as caught:
        monitor.observe(_metrics(12, length=40., realized=0.))
    report = json.loads(path.read_text())
    assert caught.value.reason_code == "bootstrap_failure"
    assert report == caught.value.report
    assert report["details"]["last_three_mean_ep_length"] == pytest.approx(160 / 3)
    assert report["coverage"]["paired_probe_iterations"] == 3
    assert [row["iteration"] for row in report["history"]] == [10, 11, 12]


@pytest.mark.parametrize("missing_iteration", [10, 11, 12])
def test_missing_probe_never_causes_bootstrap_failure(missing_iteration: int) -> None:
    monitor = TrainingHealthMonitor()
    for k in (10, 11, 12):
        monitor.observe(_metrics(k, length=1., predicted=None if k == missing_iteration else 1.,
                                 realized=None if k == missing_iteration else -1.))
    assert monitor.coverage["paired_probe_iterations"] == 2


def test_bootstrap_requires_contiguous_iterations_and_strict_threshold() -> None:
    monitor = TrainingHealthMonitor()
    for k in (9, 11, 12):
        monitor.observe(_metrics(k, length=1., realized=-1.))
    boundary = TrainingHealthMonitor()
    for k in (10, 11, 12):
        boundary.observe(_metrics(k, length=60., realized=0.))


def test_chronic_disconnect_checks_five_paired_consecutive_iterations_after_warmup() -> None:
    monitor = TrainingHealthMonitor()
    for k in range(11, 15):
        monitor.observe(_metrics(k, realized=-1.))
    with pytest.raises(EarlyAbortException) as caught:
        monitor.observe(_metrics(15, realized=-1.))
    assert caught.value.reason_code == "m_step_disconnect"
    assert "holdout_drift" in str(caught.value)


def test_missing_and_gapped_iterations_break_disconnect_and_saturation_streaks() -> None:
    monitor = TrainingHealthMonitor()
    for k in (15, 16, 18, 19):
        monitor.observe(_metrics(k, realized=-1.))
    monitor.observe(_metrics(20, predicted=None, realized=None))
    for k in range(21, 25):
        monitor.observe(_metrics(k, realized=-1.))
    assert monitor.coverage["negative_realization_streak"] == 4
    saturated = TrainingHealthMonitor()
    for k in (0, 1, 3, 4):
        saturated.observe(_metrics(k, saturation=.71))
    assert saturated.coverage["lambda_saturation_streak"] == 2


def test_numerical_failure_aborts_immediately_independent_of_probe_coverage() -> None:
    monitor = TrainingHealthMonitor()
    with pytest.raises(EarlyAbortException) as caught:
        monitor.observe(_metrics(0, predicted=None, realized=None), numerical_valid=False)
    assert caught.value.reason_code == "nonfinite_actions_or_weights"
    assert caught.value.report["coverage"]["paired_probe_iterations"] == 0


def test_lambda_saturation_strict_boundary_and_three_consecutive_observations() -> None:
    monitor = TrainingHealthMonitor()
    monitor.observe(_metrics(0, saturation=.70))
    monitor.observe(_metrics(1, saturation=.71))
    monitor.observe(_metrics(2, saturation=.72))
    with pytest.raises(EarlyAbortException) as caught:
        monitor.observe(_metrics(3, saturation=.73))
    assert caught.value.reason_code == "trust_region_collapse"


def test_disabled_monitor_never_aborts_and_history_is_bounded() -> None:
    monitor = TrainingHealthMonitor(enabled=False, history_size=5)
    for k in range(30):
        monitor.observe(_metrics(k, length=1., realized=-1., saturation=1.), numerical_valid=False)
    assert monitor.coverage["iterations_observed"] == 30
    assert monitor.coverage["history_rows"] == 5
    with pytest.raises(ValueError, match="increase strictly"):
        monitor.observe(_metrics(29))


def test_no_probe_coverage_is_safe_across_the_entire_75_iteration_run() -> None:
    monitor = TrainingHealthMonitor()
    for k in range(75):
        monitor.observe(_metrics(k, length=1., predicted=None, realized=None, saturation=None))
    assert monitor.coverage["paired_probe_iterations"] == 0
    assert monitor.coverage["negative_realization_streak"] == 0
    assert monitor.coverage["history_rows"] == 20
