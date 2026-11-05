"""Synthetic probes, bindings and candidate diagnostics; no model is constructed."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import acrobot_energy as objective
from src.gps.mppi_gps import acrobot_energy_probe as probe

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/studies/acrobot_energy_discovery_v1_1.json"


@pytest.fixture
def config():
    return json.loads(CONFIG.read_text())


def test_frozen_configuration_and_call_reservations(config):
    probe.validate_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["task"].update(balance_height=3.7),
        lambda c: c["parts"]["energy"].update(controls=500),
        lambda c: c["parts"]["energy"]["variants"][0].update(energy_weight=10),
        lambda c: c["parts"]["energy"]["variants"][0]["planner"].update(H=299),
        lambda c: c["parts"]["energy"]["variants"][1]["planner"].update(noise_std=[1]),
        lambda c: c["parts"]["energy"]["starts"].pop(),
        lambda c: c["preflight"].update(atol=1e-8),
        lambda c: c["preflight"]["profiles"][1].update(threads=2),
        lambda c: c["energy_cost"].update(energy_scale=29.43),
        lambda c: c["numerics"].update(cost_relative_fp64_ulps=128),
        lambda c: c.update(selection_rule="choose lowest cross-objective cost"),
        lambda c: c.update(kill_rule="continue until success"),
        lambda c: c["shared_limits"].update(cpu_seconds=20000),
    ],
)
def test_frozen_recipes_gates_and_limits_are_not_silently_changed(config, mutation):
    mutation(config)
    with pytest.raises(ValueError):
        probe.validate_config(config)


def rows(successes):
    return [
        dict(recipe_index=index, controls=600, success=trial < count, task_cost=100 - index)
        for index, count in enumerate(successes)
        for trial in range(2)
    ]


def test_kill_rule_counts_both_complete_episodes_and_does_not_fit(config):
    variants = config["parts"]["energy"]["variants"]
    outcome = probe.decision(rows([0, 1, 1]), variants)
    assert outcome["status"] == "TEACHER_PROBLEM_STOP"
    assert outcome["selected_variant"] is None
    assert not outcome["teacher_qualified"] and not outcome["learning_or_loop_launched"]
    outcome = probe.decision(rows([2, 2, 2]), variants)
    assert outcome["selected_variant"] == variants[0]["name"]
    assert not outcome["teacher_qualified"] and not outcome["learning_or_loop_launched"]
    incomplete = rows([0, 0, 0])
    incomplete[-1]["controls"] = 599
    with pytest.raises(probe.InvalidExecution, match="all six"):
        probe.decision(incomplete, variants)


def test_missing_prerequisite_stops_without_native_imports(tmp_path):
    with pytest.raises(probe.InvalidExecution, match="missing"):
        probe.verify_preflight(tmp_path, CONFIG, {})


@pytest.mark.parametrize("part", [None, "near-upright"])
def test_existing_output_is_never_modified(config, tmp_path, part):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        if part is None:
            probe.preflight(CONFIG, out)
        else:
            probe.execute(CONFIG, part, tmp_path / "missing", out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == {"keep": b"untouched"}


def test_cli_help_does_not_import_simulator_or_network():
    script = """
import sys
from src.gps.mppi_gps.acrobot_energy_probe import main
try: main(['--help'])
except SystemExit as e: assert e.code == 0
assert 'mujoco' not in sys.modules
assert 'torch' not in sys.modules
"""
    value = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True
    )
    assert value.returncode == 0, value.stderr


def test_native_metrics_separate_running_terminal_and_height_only_success():
    env = SimpleNamespace(terminal_cost=lambda state: np.array(7.0))
    states = np.zeros((600, 5))
    states[:, 0] = np.arange(1, 601) * 0.02
    states[:, 3:] = [3, 4]
    costs = np.ones(600)
    actions = np.zeros((600, 1))
    value = probe.native_metrics(env, states, costs, actions)
    assert value["success"] and value["task_cost"] == 607
    assert value["native_running_cost_sum"] == 600 and value["episode_terminal_cost"] == 7
    assert value["original_dense_cost_sum"] == 150
    assert value["time_balanced_seconds"] == 12
    assert value["first_complete_hold_start_seconds"] == 0.02
    states[-100, 1] = np.pi
    assert not probe.native_metrics(env, states, costs, actions)["success"]


def test_bank_diagnostics_preserve_all_candidates_and_two_selected_paths(tmp_path, monkeypatch):
    states = np.zeros((3, 100, 5))
    states[:, :, 0] = np.arange(1, 101) * 0.02
    states[1, :, 1] = 0.01
    states[2, :, 1] = np.pi
    actions = np.zeros((3, 100, 1))
    actions[0] = 0.9
    state = np.array([0, np.pi, 0, 0, 0], dtype=float)
    planner = SimpleNamespace(
        _last_states=states,
        _last_actions=actions,
        _last_costs=probe._dense_cost(states, actions).sum(axis=1),
        _rng=np.random.default_rng(19),
    )
    env = SimpleNamespace(terminal_cost=lambda value: np.zeros(np.asarray(value).shape[:-1]))
    context = dict(states=state.copy(), warmstarts=np.zeros(2), integration=np.zeros(13))
    monkeypatch.setattr(probe, "_capture", lambda mj, env: copy.deepcopy(context))
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace())
    budget = SimpleNamespace(check=lambda output_reservation=0: None)
    with h5py.File(tmp_path / "banks.h5", "x") as h:
        group = h.create_group("pass")
        before_rng = json.dumps(planner._rng.bit_generator.state)
        probe.record_bank(group, env, planner, state, budget)
        assert json.dumps(planner._rng.bit_generator.state) == before_rng
        components = group["components"]
        np.testing.assert_array_equal(components["selected_indices"][:], [1, 0])
        assert components["running_sums"].shape == (3, 6)
        assert components["selected_states"].shape == (2, 100, 5)
        assert components["selected_actions"].shape == (2, 100, 1)
        assert group.attrs["action_bank_sha256"] == hashlib.sha256(actions.tobytes()).hexdigest()
        np.testing.assert_array_equal(components["height_qualified"][:], [True, True, False])
        np.testing.assert_array_equal(components["primary_total_cost"][:], planner._last_costs)
        assert np.max(np.abs(components["component_grouping_error"][:])) < 1e-10
        np.testing.assert_array_equal(group["pre_context/states"][:], state)


def test_energy_bank_keeps_endpoint_once_and_canonical_components(tmp_path, monkeypatch):
    states = np.zeros((2, 100, 5))
    states[0, :, 1] = np.pi
    states[1, :, 1] = np.pi
    states[1, :, 4] = np.sqrt((58.86 - 19.62) / (0.5 * (objective.MODEL_PARAMETERS["I2"] + 0.25)))
    actions = np.zeros((2, 100, 1))
    weight = 160 / 9
    values = objective.components(states, actions, weight)
    env = SimpleNamespace(
        cost_components=lambda s, a: objective.components(s, a, weight),
        terminal_cost=objective.terminal_cost,
    )
    planner = SimpleNamespace(
        _last_states=states,
        _last_actions=actions,
        _last_costs=values["running"].sum(axis=1) + values["terminal"][:, -1],
        _rng=np.random.default_rng(12),
    )
    state = np.array([0, np.pi, 0, 0, 0], dtype=float)
    monkeypatch.setattr(
        probe,
        "_capture",
        lambda mj, env: dict(states=state, warmstarts=np.zeros(2), integration=np.zeros(13)),
    )
    monkeypatch.setitem(sys.modules, "mujoco", SimpleNamespace())
    budget = SimpleNamespace(check=lambda output_reservation=0: None)
    with h5py.File(tmp_path / "energy.h5", "x") as h:
        probe.record_bank(h, env, planner, state, budget)
        g = h["components"]
        np.testing.assert_allclose(g["terminal"][:], values["terminal"][:, -1])
        np.testing.assert_array_equal(g["energy_matched_low_height_controls"][:], [0, 100])
        np.testing.assert_array_equal(g["height_qualified"][:], [False, False])
