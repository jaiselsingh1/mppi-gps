"""Finite W1 roster and advancement boundaries; no real data or physics."""

import json
from pathlib import Path

import pytest

from src.gps.mppi_gps.experiments.student_width import WidthMeter, contrast, point_name, reservation
from src.gps.mppi_gps.runtime import BudgetExhausted


def config():
    return json.loads(
        (Path(__file__).parents[1] / "configs/studies/student_width.json").read_text()
    )


def panel(train_gains, validation_gains, walking_gains=None):
    points, evaluations = {}, {}
    for pair in range(3):
        for width in (128, 256):
            key = point_name(pair, width, 10000)
            points[key] = {
                "metrics": {
                    "train": {"mse": 1.0 if width == 128 else 1 - train_gains[pair]},
                    "validation": {"mse": 1.0 if width == 128 else 1 - validation_gains[pair]},
                }
            }
            evaluations[key] = {
                "summary": {
                    "mean_cost": 100.0 if width == 128 else 90.0,
                    "successes": 0
                    if width == 128 or walking_gains is None
                    else walking_gains[pair],
                    "survived": 0 if width == 128 or walking_gains is None else walking_gains[pair],
                    "mean_steps": 100 if width == 128 else 120,
                }
            }
    return points, evaluations


def test_exact_roster_includes_initial_curves_all_training_seeds_and_all_confirmation():
    work = reservation(config())
    assert work == dict(
        requested_controls=288000,
        neural_rows=15792000,
        backward_rows=15360000,
        optimizer_steps=60000,
        episodes=288,
        model_controls=0,
        diagnostic_forward_rows=144000,
    )
    assert all(
        value <= config()["budgets"][key]
        for key, value in work.items()
        if key in config()["budgets"]
    )


def test_training_gain_and_validation_guard_are_separate_and_zero_walking_cannot_advance():
    points, evaluations = panel([0.2, 0.2, 0.2], [-0.01, -0.01, -0.01])
    gate = contrast(points, evaluations, config())
    assert gate["reconstruction_screen"] and not gate["validation_guard"]
    assert not gate["fitting_screen"] and not gate["task_screen"]
    assert gate["task_checks"]["mean_cost_gain"]
    assert not gate["task_checks"]["walking_pairs"]


def test_walking_gain_must_replicate_and_no_seed_may_regress():
    points, evaluations = panel([0.2, 0.2, 0], [0, 0, 0], [2, 2, 0])
    assert contrast(points, evaluations, config())["task_screen"]
    evaluations[point_name(2, 256, 10000)]["summary"]["mean_steps"] = 99
    assert not contrast(points, evaluations, config())["task_screen"]
    points, evaluations = panel([0.2, 0.2, 0], [0, 0, 0], [16, 0, 0])
    assert not contrast(points, evaluations, config())["task_screen"]


def test_meter_rejects_excess_learning_before_call_and_forbids_unregistered_work(tmp_path):
    budgets = dict(
        requested_controls=0,
        neural_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        wall_seconds=60,
        storage_bytes=2**31,
    )
    entered = []
    with WidthMeter(tmp_path / "evidence", budgets) as meter:
        with pytest.raises(BudgetExhausted):
            meter.measure("optimizer_steps", 1, lambda: entered.append(True))
        with pytest.raises(BudgetExhausted):
            meter.measure("backward_rows", 1, lambda: entered.append(True))
        with pytest.raises(ValueError):
            meter.measure("extra_work", 1, lambda: entered.append(True))
        with pytest.raises(ValueError):
            meter._query("model", 0, 4, {}, lambda: None, lambda value: value)
        assert entered == []
        assert meter.totals()["requested_optimizer_steps"] == 0
