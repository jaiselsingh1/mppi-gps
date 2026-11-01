"""Synthetic continuation and lifecycle boundaries, without teacher execution."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps.experiments.transfer_diagnosis import (
    TransferMeter,
    prefix_outcomes,
    replay_check,
)
from src.gps.mppi_gps.inspection import format_inspection, inspect_run
from src.gps.mppi_gps.recipes import load_recipe
from src.gps.mppi_gps.runtime import BudgetExhausted


def trace(rows=100, *, fall=False):
    costs = np.ones(rows)
    flags = np.zeros(rows, dtype=bool)
    terminals = np.zeros(rows)
    if fall:
        costs[-1] += 20
        flags[-1] = True
        terminals[-1] = 20
    return dict(
        costs=costs,
        done_flags=flags,
        fall_costs=terminals,
        states=np.zeros((rows + 1, 19)),
        warmstarts=np.zeros((rows + 1, 9)),
        integrations=np.zeros((rows + 1, 91)),
    )


def test_prefix_fall_components_and_no_postfall_state():
    branch, baseline = trace(25, fall=True), trace()
    branch["states"][25, 2] = 2
    rows = prefix_outcomes(branch, baseline, [1, 25, 50, 100])
    assert rows[0]["delta_cost"] == 0
    assert rows[1]["raw_cost"] == 25
    assert rows[1]["terminal_cost"] == 20
    assert rows[1]["absorbing_cost"] == 0
    assert rows[1]["state_rms"] == pytest.approx(2 / np.sqrt(19))
    assert rows[2]["complete_cost"] == 25 + 20 + 125
    assert rows[2]["state_rms"] is None
    assert rows[3]["delta_cost"] == 25 + 20 + 375 - 100


def test_fall_at_horizon_is_still_a_fall_and_terminal_is_charged_once():
    row = prefix_outcomes(trace(100, fall=True), trace(), [100])[0]
    assert row["done"] and row["executed"] == 100
    assert row["terminal_cost"] == 20 and row["absorbing_cost"] == 0
    assert row["complete_cost"] == 120
    assert row["state_rms"] == 0


def test_unchanged_trace_has_zero_effect_and_replay_gate_is_fixed():
    baseline = trace()
    assert all(r["delta_cost"] == 0 for r in prefix_outcomes(baseline, baseline, [1, 10, 100]))
    assert replay_check(baseline, baseline, atol=1e-10, rtol=0)["passed"]
    altered = {key: value.copy() for key, value in baseline.items()}
    altered["states"][1, 2] = 2e-10
    assert not replay_check(altered, baseline, atol=1e-10, rtol=0)["passed"]
    altered["states"] = altered["states"][:-1]
    assert not replay_check(altered, baseline, atol=1e-10, rtol=0)["passed"]


def test_zero_learning_and_model_work_and_query_limits(tmp_path):
    cfg = load_recipe("transfer_diagnosis")
    cfg["d1_budgets"]["support_queries"] = 3
    meter = TransferMeter(tmp_path, cfg)
    try:
        meter.begin_stage("d1")
        meter.support_queries(3)
        with pytest.raises(BudgetExhausted):
            meter.support_queries(1)
        with pytest.raises(ValueError):
            meter.measure("neural_rows", 1, lambda: pytest.fail("Must not execute"))
        with pytest.raises(ValueError):
            meter._query("model", 1, 4)
        assert meter.totals()["requested_controls"] == 0
        meter.begin_stage("d2")
        with pytest.raises(BudgetExhausted):
            meter._query("live", 24001, 4)
        assert meter.totals()["requested_physics_substeps"] == 0
    finally:
        meter.close()


def test_completed_native_output_survives_deadline_and_next_work_is_rejected(tmp_path):
    cfg = load_recipe("transfer_diagnosis")
    meter = TransferMeter(tmp_path, cfg)
    try:
        meter.begin_stage("d2")

        def completed_call():
            meter.stage_started -= cfg["d2_budgets"]["wall_seconds"] + 1
            return np.array([7.0])

        value = meter._query(
            "live", 1, 4, {"action": np.zeros(6)}, completed_call, lambda result: {"value": result}
        )
        assert value[0] == 7
        assert meter.totals()["completed_live_controls"] == 1
        meter.save_arrays("partial_trace_after_deadline", values=value)
        events = [json.loads(line) for line in meter.ledger_path.read_text().splitlines()]
        assert any(row["event"] == "query_complete" for row in events)
        with pytest.raises(BudgetExhausted):
            meter._query("live", 1, 4, {}, lambda: pytest.fail("No extra native work"), dict)
        assert meter.totals()["requested_live_controls"] == 1
    finally:
        meter.close()


def test_transfer_cli_help_has_no_policy_or_simulator_imports():
    code = """
import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','mujoco'} or name.startswith('src.envs'):
        raise AssertionError('Unexpected runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
sys.argv = ['scripts/train_mppi_gps.py','study','transfer-diagnosis','--help']
runpy.run_path('scripts/train_mppi_gps.py',run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "prepare" in result.stdout and "worker" in result.stdout


def test_frozen_diagnostic_does_not_claim_policy_fitting(tmp_path, monkeypatch):
    from src.gps.mppi_gps import lifecycle

    source = tmp_path / "checkout"
    (source / "src").mkdir(parents=True)
    (source / "src/example.py").write_text("# synthetic source\n")
    monkeypatch.setattr(lifecycle, "ROOT", source)
    for command, expected in (("transfer-diagnosis", False), ("student-update-w2", True)):
        out = lifecycle.freeze(tmp_path / command, {"schema": "synthetic"}, command)
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["policy_action_fitting"] is expected


def test_repository_check_accepts_sized_input_bindings(tmp_path):
    from scripts.check_repository import check_frozen, digest

    (tmp_path / "inputs").mkdir()
    config, evidence = tmp_path / "config.json", tmp_path / "inputs/example.txt"
    config.write_text('{"schema":"synthetic"}')
    evidence.write_text("sealed input\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(files=[], config_sha256=digest(config)))
    )
    (tmp_path / "input_manifest.json").write_text(
        json.dumps(
            dict(
                files={
                    "inputs/example.txt": dict(
                        sha256=digest(evidence), bytes=evidence.stat().st_size
                    )
                }
            )
        )
    )
    errors = []
    check_frozen(tmp_path, errors)
    assert errors == []
    evidence.write_text("changed\n")
    check_frozen(tmp_path, errors)
    assert any("Hash mismatch" in value for value in errors)


def test_inspector_exposes_partial_diagnostic_gate(tmp_path):
    for name, value in {
        "config.json": {"schema": "walker-transfer-diagnosis-v1"},
        "manifest.json": {},
        "input_manifest.json": {},
        "result.json": {
            "schema": "walker-transfer-diagnosis-result-v1",
            "status": "stopped_invalid",
            "error": {"message": "Replay mismatch"},
            "d1": {"query_count": 69701, "coverage_records": [{}]},
            "d2": {"baseline_gate_passed": False, "baselines": [{}], "branches": []},
        },
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    before = {p: p.read_bytes() for p in tmp_path.iterdir()}
    report = inspect_run(tmp_path)
    assert report["diagnostics"]["baseline_replays"] == 1
    assert not report["diagnostics"]["baseline_gate_passed"]
    assert "diagnostics:" in format_inspection(report)
    assert before == {p: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("mismatch", [False, True])
def test_all_baselines_precede_substitutions_and_failure_stops(monkeypatch, mismatch):
    from src.gps.mppi_gps import runtime, transfer_inputs
    from src.gps.mppi_gps.experiments.transfer_diagnosis import run_d2

    cfg = load_recipe("transfer_diagnosis")
    cfg["task"] = {"apply_terminal_cost_on_done": True}

    class FakeEnv:
        def __init__(self, **_):
            self.state = np.zeros(19)
            self.warmstart = np.zeros(9)
            self.integration = np.zeros(91)

        def _get_obs(self):
            return np.zeros(17)

        def step(self, action):
            self.state[0] += 1
            self.integration[0] += 1
            return self._get_obs(), 1.0, False, {"x_velocity": 1.5, "fall_cost": 0.0}

        def close(self):
            pass

    def snap(env):
        return dict(
            state=env.state.copy(),
            warmstart=env.warmstart.copy(),
            integration=env.integration.copy(),
            spec=np.array(1),
            dimensions=np.array([10, 9, 6]),
        )

    def restore(env, saved):
        env.state = saved["state"].copy()
        env.warmstart = saved["warmstart"].copy()
        env.integration = saved["integration"].copy()
        return snap(env)

    class FakeMeter:
        def __init__(self):
            self.events = []

        def record(self, event, **metadata):
            self.events.append((event, metadata))

        def start_branch(self, **metadata):
            self.record("start", **metadata)

        def instrument(self, env):
            return env

        def save_arrays(self, tag, **arrays):
            return {"file": "synthetic", "group": tag, "arrays": {}}

    anchors = []
    for teacher in cfg["teacher_ids"]:
        for step in cfg["anchors"]:
            expected = trace()
            expected["states"][:, 0] = step + np.arange(101)
            expected["integrations"][:, 0] = step + np.arange(101)
            anchors.append(
                dict(
                    teacher_id=teacher,
                    step=step,
                    observation=np.zeros(17),
                    initial=dict(
                        state=expected["states"][0],
                        warmstart=expected["warmstarts"][0],
                        integration=expected["integrations"][0],
                    ),
                    expected=expected,
                    teacher_actions=np.zeros((100, 6)),
                    branches=[dict(name=name, action=np.zeros(6)) for name in cfg["branch_names"]],
                )
            )
    if mismatch:
        anchors[0]["expected"]["costs"][0] += 1e-8
    monkeypatch.setitem(sys.modules, "src.envs.walker2d", SimpleNamespace(Walker2d=FakeEnv))
    monkeypatch.setattr(runtime, "snapshot", snap)
    monkeypatch.setattr(runtime, "restore", restore)
    monkeypatch.setattr(transfer_inputs, "load_fork_inputs", lambda *args, **kwargs: anchors)
    meter = FakeMeter()
    result = dict(baseline_gate_passed=False, baselines=[], branches=[])
    if mismatch:
        with pytest.raises(ValueError, match="unmodified replay gate"):
            run_d2(None, meter, cfg, result)
        assert len(result["baselines"]) == 1
        assert result["branches"] == [] and not result["baseline_gate_passed"]
    else:
        run_d2(None, meter, cfg, result)
        starts = [row for event, row in meter.events if event == "start"]
        assert len(starts) == 240
        assert all(row["baseline"] for row in starts[:20])
        assert not any(row["baseline"] for row in starts[20:])
        assert result["baseline_gate_passed"] and result["execution_complete"]
