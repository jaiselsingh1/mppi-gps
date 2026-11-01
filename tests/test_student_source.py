"""S1 boundary and actual-initial-state checks, using small synthetic data."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from copy import deepcopy

import numpy as np
import pytest
import torch

from src.gps.mppi_gps.experiments.student_source import SourceMeter, combine_work, source_screen
from src.gps.mppi_gps.policy import Policy, fit_policy
from src.gps.mppi_gps.provenance import fingerprint
from src.gps.mppi_gps.recipes import load_recipe
from src.gps.mppi_gps.runtime import BudgetExhausted


def test_initial_observer_sees_actual_state_and_cannot_mutate_fit():
    rng = np.random.default_rng(91)
    observations = rng.normal(size=(32, 17))
    targets = np.tanh(observations[:, :6])
    parent = Policy(19)
    before = parent.hash()
    plain, plain_state, _ = fit_policy(
        parent, observations, targets, updates=5, eta_theta=0, seed=21
    )
    captured = []

    def observer(payload):
        captured.append(deepcopy(payload))
        for tensor in payload["state_dict"].values():
            tensor.zero_()
        payload["private_rng_state"].zero_()
        payload["adam"]["param_groups"][0]["lr"] = 999
        payload["rng"].clear()

    observed, observed_state, _ = fit_policy(
        parent, observations, targets, updates=5, eta_theta=0, seed=21, initial_observer=observer
    )
    assert len(captured) == 1
    assert captured[0]["adam"]["state"] == {}
    assert captured[0]["adam"]["param_groups"][0]["foreach"] is False
    assert captured[0]["rng"] == np.random.Generator(np.random.PCG64(21)).bit_generator.state
    assert torch.equal(captured[0]["private_rng_state"], parent._generator.get_state())
    assert observed.hash() == plain.hash()
    assert fingerprint(observed_state) == fingerprint(plain_state)
    assert parent.hash() == before


def test_source_walking_gate_is_separate_from_cost_and_competence():
    config = load_recipe("student_source")
    rows = {}
    for pair in range(3):
        rows[f"mppi_s{pair}"] = dict(
            summary=dict(successes=0, survived=0, mean_cost=5000, mean_steps=50)
        )
        rows[f"sac_reference_s{pair}"] = dict(
            summary=dict(successes=0, survived=0, mean_cost=2500, mean_steps=500)
        )
    assert not source_screen(rows, config)["walking_screen"]
    for pair in (0, 1):
        rows[f"sac_reference_s{pair}"]["summary"].update(successes=2, survived=2)
    assert source_screen(rows, config)["walking_screen"]
    rows["mppi_s2"]["summary"]["survived"] = 1
    assert not source_screen(rows, config)["walking_screen"]


def test_phase_work_is_not_double_charged_and_deadline_keeps_completed_work(tmp_path):
    config = load_recipe("student_source")
    meter = SourceMeter(tmp_path / "evidence/fit", config, "fit", time.monotonic() + 60)
    try:

        def operation():
            meter.deadline = time.monotonic() - 1
            return 7

        assert meter.measure("optimizer_steps", 1, operation) == 7
        meter.save_arrays("completed_output", values=np.array([7]))
        with pytest.raises(BudgetExhausted):
            meter.measure("optimizer_steps", 1, lambda: pytest.fail("No extra work"))
        assert meter.totals()["completed_optimizer_steps"] == 1
    finally:
        meter.close()
    assert (
        combine_work(
            dict(neural_rows=3180000, optimizer_steps=12000),
            dict(neural_rows=42, optimizer_steps=0),
        )["neural_rows"]
        == 3180042
    )


def test_evaluation_forbids_learning_and_model_queries(tmp_path):
    meter = SourceMeter(
        tmp_path / "evidence/eval", load_recipe("student_source"), "eval", time.monotonic() + 60
    )
    try:
        with pytest.raises(ValueError):
            meter.measure("backward_rows", 1, lambda: None)
        with pytest.raises(ValueError):
            meter._query("model", 1, 4)
        assert meter.totals()["requested_controls"] == 0
    finally:
        meter.close()


def test_source_cli_help_is_simulator_and_torch_free():
    program = """
import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','mujoco'} or name.startswith('src.envs'):
  raise AssertionError('Unexpected import '+name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=['scripts/train_mppi_gps.py','study','source','--help']
runpy.run_path('scripts/train_mppi_gps.py',run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert all(command in result.stdout for command in ("prepare", "fit", "evaluate"))


def test_actual_recipe_retains_proposal_seeds_gates_and_budgets():
    from pathlib import Path

    proposal = json.loads(Path("configs/studies/student_source_s1_proposal.json").read_text())
    config = load_recipe("student_source")
    for key in (
        "arms",
        "seed_pairs",
        "learning",
        "primary_seeds",
        "confirmation_seeds",
        "budgets",
        "walking_screen",
        "competence",
        "evaluation_order",
        "normalization",
    ):
        assert config[key] == proposal[key], key


def test_inspector_exposes_fit_only_result_and_separate_missing_phases(tmp_path):
    from src.gps.mppi_gps.inspection import inspect_run

    (tmp_path / "config.json").write_text(json.dumps(load_recipe("student_source")))
    (tmp_path / "fit_result.json").write_text(
        json.dumps(
            dict(
                schema="walker-student-source-s1-fit-v1",
                execution_complete=False,
                error={"type": "BudgetExhausted"},
                work={"completed_optimizer_steps": 53},
            )
        )
    )
    report = inspect_run(tmp_path)
    assert report["result_file"] == "fit_result.json"
    assert report["costs"]["work"]["completed_optimizer_steps"] == 53
    assert "result.json" in report["missing_evidence"]
    assert "evidence/fit/raw.h5" in report["missing_evidence"]
    assert "evidence/eval/raw.h5" in report["missing_evidence"]
    assert "evidence/raw.h5" not in report["missing_evidence"]
