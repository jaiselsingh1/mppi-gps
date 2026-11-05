"""Saved/synthetic checks for the bounded teacher gate; no simulator is constructed."""

import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import acrobot_teacher as teacher


@pytest.fixture
def config():
    path = (
        Path(__file__).resolve().parents[1] / "configs/studies/acrobot_controller_teacher_v1_0.json"
    )
    return json.loads(path.read_text())


def test_frozen_config_is_accepted(config):
    teacher.validate_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c["task"].update(controls=499),
        lambda c: c.update(qualification_successes_required=8),
        lambda c: c.update(qualification_episode_count=9),
        lambda c: c.update(selection="pick the best-looking trial"),
        lambda c: c["qualification"].pop(),
        lambda c: c["qualification"].__setitem__(0, c["development"][0]),
        lambda c: c["preflight"].update(atol=1e-8),
        lambda c: c["limits"].update(cpu_seconds=10801),
        lambda c: c["recipes"][1]["planner"].update(noise_lowpass_sample_rate_hz=125),
    ],
)
def test_recipe_gate_seed_and_budget_changes_are_explicit(config, mutation):
    mutation(config)
    with pytest.raises(ValueError):
        teacher.validate_config(config)


def development_rows():
    rows = []
    for index in range(3):
        for trial in range(2):
            rows.append(
                dict(
                    recipe_index=index,
                    controls=500,
                    success=False,
                    task_cost=30 - 10 * index + trial,
                )
            )
    return rows


def test_development_selection_counts_failures_and_prefers_success_before_cost():
    rows = development_rows()
    assert teacher.choose_recipe(rows) == 2  # No proxy gate: qualification still follows.
    rows[0]["success"] = True
    assert teacher.choose_recipe(rows) == 0
    rows[2]["success"] = True
    assert teacher.choose_recipe(rows) == 1
    rows[0]["task_cost"] = rows[2]["task_cost"]
    rows[1]["task_cost"] = rows[3]["task_cost"]
    assert teacher.choose_recipe(rows) == 0
    rows.pop()
    with pytest.raises(teacher.InvalidExecution, match="six complete"):
        teacher.choose_recipe(rows)


def test_partial_development_episode_cannot_select_a_recipe():
    rows = development_rows()
    rows[-1]["controls"] = 225
    with pytest.raises(teacher.InvalidExecution, match="complete"):
        teacher.choose_recipe(rows)


def test_height_only_success_and_time_measurements_preserve_raw_branch():
    states = np.zeros((500, 5))
    states[:, 0] = np.arange(1, 501) * 0.02
    states[:, 1] = 2 * np.pi
    states[:, 3:] = 100  # Velocity is reported, not a hidden admission gate.
    result = teacher.episode_metrics(states, np.zeros(500))
    assert result["success"]
    assert result["first_complete_hold_start_seconds"] == 0.02
    assert result["time_balanced_seconds"] == 10
    assert result["final_window_velocity_rms"] == 100
    assert result["terminal_winding"] == [1, 0]
    states[400, 1] = np.pi
    assert not teacher.episode_metrics(states, np.zeros(500))["success"]
    with pytest.raises(teacher.InvalidExecution, match="all 500"):
        teacher.episode_metrics(states[:225], np.zeros(225))
    states[-1, -1] = np.nan
    with pytest.raises(teacher.InvalidExecution, match="Nonfinite"):
        teacher.episode_metrics(states, np.zeros(500))


def test_budget_reserves_attempted_and_completed_work_separately(config, tmp_path):
    limits = copy.deepcopy(config["limits"])
    limits["native_controls"] = 3
    budget = teacher.Budget(limits, tmp_path)
    budget.reserve("verification", 1, 2)
    budget.complete("verification", 1, 2)
    budget.reserve("native", 2, 2)  # Attempt fails before complete(): still paid and preserved.
    assert budget.work.physics_substeps_attempted == 6
    assert budget.work.physics_substeps_completed == 2
    assert budget.work.native_controls_completed == 0
    with pytest.raises(teacher.ResourceLimit, match="Native-control"):
        budget.reserve("native", 1, 2)
    assert budget.work.native_controls_attempted == 2
    limits["forecast_controls"] = 10
    budget.reserve("forecast", 10, 2)
    budget.complete("forecast", 10, 2)
    assert budget.work.model_queries_completed == 1
    with pytest.raises(teacher.ResourceLimit, match="Forecast-control"):
        budget.reserve("forecast", 1, 2)
    assert budget.work.physics_substeps_completed == 22
    assert asdict(budget.work)["optimizer_updates"] == 0


def test_budget_enforces_output_and_physics_before_paid_call(config, tmp_path):
    limits = copy.deepcopy(config["limits"])
    limits["physics_substeps"] = 3
    budget = teacher.Budget(limits, tmp_path)
    with pytest.raises(teacher.ResourceLimit, match="Physics-substep"):
        budget.reserve("forecast", 2, 2)
    assert budget.work.model_queries_attempted == 0
    limits["output_bytes"] = 260000
    with pytest.raises(teacher.ResourceLimit, match="output_bytes"):
        budget.check()


def test_import_and_cli_help_do_not_load_simulator():
    code = (
        "import sys; from src.gps.mppi_gps.acrobot_teacher import main; "
        "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules; "
        "main(['--help'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_invalid_input_binding_stops_before_native_import(config, tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    config["bindings"][0]["sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "result"
    result = teacher.run(path, out)
    assert result["status"] == "STOPPED_INVALID_EXECUTION"
    assert "Input binding changed" in result["error"]
    assert result["work"]["physics_substeps_attempted"] == 0
    assert not (out / "evidence.h5").exists()
    with pytest.raises(FileExistsError):
        teacher.run(path, out)


def test_source_manifest_checks_each_payload(config, tmp_path, monkeypatch):
    monkeypatch.setattr(teacher, "ROOT", tmp_path)
    for relative in (
        "src/gps/mppi_gps/acrobot_teacher.py",
        "src/gps/mppi_gps/provenance.py",
        "src/envs/acrobot.py",
        "src/envs/mujoco_env.py",
        "src/envs/base.py",
        "src/mppi/mppi.py",
        "src/utils/config.py",
        "src/utils/math.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    config["bindings"] = []
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("exact preserved bytes")
    manifest = {"files": [{"path": "frozen.txt", "sha256": teacher.sha(frozen)}]}
    (tmp_path / "source_manifest.json").write_text(json.dumps(manifest))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = teacher._source_bindings(config, config_path)
    assert result["frozen_source_manifest"]["verified_payload_count"] == 1
    frozen.write_text("changed")
    with pytest.raises(teacher.InvalidExecution, match="Frozen source payload changed"):
        teacher._source_bindings(config, config_path)


def test_strict_replay_contract_is_not_relative():
    teacher._assert_close(np.array([1e8]), np.array([1e8]), "state", 1e-10)
    with pytest.raises(teacher.InvalidExecution, match="Strict replay mismatch"):
        teacher._assert_close(np.array([1e8 + 1]), np.array([1e8]), "state", 1e-10)
    with pytest.raises(teacher.InvalidExecution, match="Nonfinite"):
        teacher._assert_close(np.array([np.nan]), np.array([np.nan]), "state", 1e-10)


def test_partial_trial_retains_contexts_all_passes_and_paid_failure(config, tmp_path, monkeypatch):
    import h5py

    class FakeEnv:
        _frame_skip, _dt = 2, 0.02

        def reset(self, *, seed):
            self.state = np.array([0, float(seed % 10), 0, 0, 0])
            self.executed = 0

        def _get_obs(self):
            return self.state[1:].copy()

        def batch_rollout(self, state, actions, **kwargs):
            states = np.broadcast_to(state, (*actions.shape[:2], 5)).copy()
            states[..., 0] = state[0] + np.arange(1, actions.shape[1] + 1) * self._dt
            return (
                states,
                np.ones(actions.shape[0]),
                np.empty((*actions.shape[:2], 0)),
            )

        def step(self, action):
            if self.executed == 3:
                raise RuntimeError("synthetic interrupted native call")
            self.executed += 1
            self.state[0] += 0.02
            return self._get_obs(), 0.1, False, {}

    class FakeMPPI:
        def __init__(self, env, cfg):
            self.env, self.cfg = env, cfg
            self.U = np.zeros((cfg.H, 1))
            self.first = True

        def reset(self, *, seed):
            self.seed = seed

        def _plan_iteration(self, state, prior, coupling, warm):
            actions = np.repeat(self.U[None], self.cfg.K, axis=0)
            _, self._last_costs, _ = self.env.batch_rollout(state, actions)
            self._last_weights = np.full(self.cfg.K, 1 / self.cfg.K)
            self.U[:] = 0.1
            return self.U[0].copy(), {"n_eff": float(self.cfg.K), "lam": self.cfg.lam}

        def plan_step(self, state, *, initial_warmstart):
            for _ in range(self.cfg.initial_iterations if self.first else 1):
                action, info = self._plan_iteration(state, None, None, initial_warmstart)
            self.first = False
            return action, info

    monkeypatch.setattr(
        teacher,
        "_capture",
        lambda mj, env: {
            "states": env.state.copy(),
            "warmstarts": np.zeros(2),
            "integration": env.state.copy(),
        },
    )
    recipe = {"name": "synthetic", "planner": {"H": 3, "K": 2, "initial_iterations": 2}}
    budget = teacher.Budget(config["limits"], tmp_path)
    with h5py.File(tmp_path / "partial.h5", "x") as evidence:
        group = evidence.create_group("trial")
        with pytest.raises(RuntimeError, match="synthetic interrupted"):
            teacher._run_trial(
                None,
                FakeEnv(),
                FakeMPPI,
                group,
                budget,
                recipe,
                config["development"][0],
                "development",
                0,
                0,
                tmp_path,
            )
        assert group.attrs["status"] == "STOPPED"
        assert group["states"].shape == (4, 5)
        assert group["actions"].shape == (3, 1)
        assert group["pre_action_observations"].shape == (3, 4)
        np.testing.assert_array_equal(group["states"][:-1, 1:], group["pre_action_observations"][:])
        assert len(group["planning/control_0000"]) == 3  # pre-observation plus two passes
        assert len(group["planning/control_0001"]) == 2
        np.testing.assert_array_equal(
            group["planning/control_0000/pass_000/weights"][:], [0.5, 0.5]
        )
        assert budget.work.native_controls_attempted == 4
        assert budget.work.native_controls_completed == 3
        assert budget.work.model_queries_attempted == budget.work.model_queries_completed == 5
        assert budget.work.forecast_controls_completed == 30
        assert budget.work.physics_substeps_attempted == 68
        assert budget.work.physics_substeps_completed == 66
