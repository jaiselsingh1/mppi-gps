"""Synthetic W2 endpoints, partial-benefit labels and serialized continuation."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.experiments.student_update_budget import (
    UpdateMeter,
    confirmation_required,
    contrast,
    point_name,
    reservation,
)
from src.gps.mppi_gps.policy import Policy, fit_policy
from src.gps.mppi_gps.runtime import BudgetExhausted


def config():
    return json.loads(
        (Path(__file__).parents[1] / "configs/studies/student_update_budget.json").read_text()
    )


def fixture():
    points, panel = {}, {}
    for pair in range(3):
        for update in (2000, 10000):
            early = update == 2000
            name = point_name(pair, update)
            points[name] = {
                "metrics": {
                    "train": {"mse": 0.2 if early else 0.1},
                    "validation": {"mse": 0.2 if early else 0.3},
                }
            }
            panel[name] = {
                "summary": {
                    "mean_steps": 200 if early else 100,
                    "mean_cost": 5000 if early else 4500,
                    "successes": 0,
                    "survived": 0,
                }
            }
    return points, panel


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def test_roster_counts_three_training_streams_and_both_fixed_endpoints():
    assert reservation(config()) == dict(
        requested_controls=228000,
        neural_rows=7962000,
        backward_rows=7680000,
        optimizer_steps=30000,
        episodes=228,
        model_controls=0,
        checkpoint_loads=3,
        diagnostic_forward_rows=54000,
    )


def test_retention_with_zero_walking_and_worse_cost_is_partial_not_competence():
    points, panel = fixture()
    gate = contrast(points, panel, config())
    assert gate["validation_screen"] and gate["retention_screen"]
    assert not gate["cost_screen"] and not gate["walking_screen"]
    assert gate["pairs"][0]["relative_train_mse_gain"] == -1
    assert gate["pairs"][0]["controls_delta"] == 100
    assert confirmation_required(gate, panel, config())


def test_validation_alone_never_triggers_confirmation_and_survival_loss_blocks_retention():
    points, panel = fixture()
    for row in panel.values():
        row["summary"]["mean_steps"] = 100
    gate = contrast(points, panel, config())
    assert gate["validation_screen"] and not confirmation_required(gate, panel, config())
    panel[point_name(0, 10000)]["summary"]["survived"] = 1
    for pair in range(3):
        panel[point_name(pair, 2000)]["summary"]["mean_steps"] = 200
    assert not contrast(points, panel, config())["retention_screen"]
    panel[point_name(0, 10000)]["summary"]["successes"] = 15
    assert confirmation_required(contrast(points, panel, config()), panel, config())


def test_checkpoint_load_quota_initializes_and_rejects_before_second_callable(tmp_path):
    budgets = dict(
        requested_controls=0,
        neural_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        checkpoint_loads=1,
        wall_seconds=60,
        storage_bytes=2**31,
    )
    entered = []
    with UpdateMeter(tmp_path / "evidence", budgets) as meter:
        assert meter.measure("checkpoint_loads", 1, lambda: entered.append("one")) is None
        with pytest.raises(BudgetExhausted):
            meter.measure("checkpoint_loads", 1, lambda: entered.append("two"))
        assert entered == ["one"]
        assert meter.totals()["completed_checkpoint_loads"] == 1
        assert meter.totals()["requested_checkpoint_loads"] == 1
        assert meter.totals()["rejected_measurements"] == 1


def test_real_serialization_boundary_preserves_default128_adam_rng_and_exact_future(tmp_path):
    rng = np.random.default_rng(71)
    observations = rng.normal(size=(12, 17))
    actions = np.tanh(rng.normal(size=(12, 6)))
    parent = Policy(seed=172)
    options = dict(batch_size=4, lr=0.0003, eta_theta=0, prox_reduction="mean", seed=108)
    whole, whole_state, _ = fit_policy(parent, observations, actions, updates=8, **options)
    early, early_state, _ = fit_policy(parent, observations, actions, updates=3, **options)
    before = early.hash()
    path = tmp_path / "early.pt"
    early.checkpoint(path, optimizer_state=early_state, metadata={"updates": 3})
    loaded, loaded_state, metadata = Policy.from_checkpoint(path)
    assert loaded.hash() == before and metadata["updates"] == 3
    equal_tree(early_state, loaded_state)
    final, final_state, _ = fit_policy(
        loaded,
        observations,
        actions,
        optimizer_state=loaded_state,
        updates=5,
        resume_fit=True,
        **options,
    )
    assert early.hash() == before and final.hash() == whole.hash()
    for key in (
        "adam",
        "rng",
        "updates",
        "fit_updates",
        "anchor",
        "anchor_policy_hash",
        "fit_config",
        "fit_data",
    ):
        equal_tree(final_state[key], whole_state[key])
