"""Synthetic controller lifecycle checks: no simulator construction or integration."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import acrobot_attractor_controller as controller
from src.gps.mppi_gps import acrobot_teacher as teacher
from src.utils.math import effective_sample_size, softmin_weights, temperature_for_neff_fraction

ROOT = Path(__file__).resolve().parents[1]


def frozen_config():
    return json.loads((ROOT / "configs/studies/acrobot_attractor_v1_0.json").read_text())


class FakeEnv:
    """A known synthetic transition to upright, used only to exercise recording."""

    _dt, _frame_skip = 0.02, 2

    def __init__(self):
        self.state = np.zeros(5)
        self.reset_calls = self.step_calls = 0

    def reset(self, state=None, seed=None):
        del seed
        self.reset_calls += 1
        self.state = np.zeros(5) if state is None else state.copy()
        return self._get_obs()

    def get_state(self):
        return self.state.copy()

    def _get_obs(self):
        return self.state[1:5].copy()

    def step(self, action):
        assert action.shape == (1,)
        self.step_calls += 1
        self.state[0] += self._dt
        self.state[1:] = 0
        return self._get_obs(), 5.0, False, {}

    def terminal_cost(self, state):
        return np.zeros(np.asarray(state).shape[:-1]) + 7.0

    def batch_rollout(self, state, actions, **kwargs):
        del kwargs
        k, h = actions.shape[:2]
        states = np.broadcast_to(state, (k, h, 5)).copy()
        states[..., 0] = state[0] + np.arange(1, h + 1) * self._dt
        return states, np.arange(k, dtype=float), np.zeros((k, h, 1))


class FakeMPPI:
    """Records real counter hooks but uses no native model or real search."""

    def __init__(self, env, cfg):
        self.env, self.cfg = env, cfg
        self.K, self.H = cfg.K, cfg.H
        self.U = np.zeros((self.H, 1))
        self._rng = None
        self.initial = True

    def reset(self, seed):
        self._rng = np.random.default_rng(seed)
        self.U[:] = 0
        self.initial = True

    def _plan_iteration(self, state, prior_cost, coupling, warmstart):
        del prior_cost, coupling, warmstart
        self._last_actions = np.broadcast_to(self.U, (self.K, self.H, 1)).copy()
        self._last_states, self._last_costs, _ = self.env.batch_rollout(state, self._last_actions)
        fraction = self.cfg.lam_target_neff_fraction
        lam = temperature_for_neff_fraction(self._last_costs, fraction) if fraction else 3.0
        self._last_weights = softmin_weights(self._last_costs, lam)
        return self.U[0].copy(), {"n_eff": effective_sample_size(self._last_weights), "lam": lam}

    def plan_step(self, state, initial_warmstart=None):
        for _ in range(self.cfg.initial_iterations if self.initial else 1):
            action, info = self._plan_iteration(state, None, None, initial_warmstart)
        self.initial = False
        return action, info


def capture(mj, env):
    del mj
    return dict(
        states=env.get_state(), warmstarts=np.zeros(2), integration=np.r_[env.state, np.zeros(23)]
    )


def budget(tmp_path):
    limits = dict(
        cpu_seconds=30,
        wall_seconds=60,
        output_bytes=32 * 1024**2,
        rss_bytes=4 * 1024**3,
        native_controls=2000,
        forecast_controls=10_000_000,
        physics_substeps=30_000_000,
    )
    return teacher.Budget(limits, tmp_path)


def test_import_remains_lazy():
    run = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.acrobot_attractor_controller; "
            'assert "mujoco" not in sys.modules; assert "torch" not in sys.modules',
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr


def test_stage_changes_only_temperature_and_rejects_unrelated_recipe_change():
    stages = frozen_config()["stages"]
    for name, stage in stages.items():
        controller.validate_stage(stage, swing=name != "near")
    altered = copy.deepcopy(stages["near"])
    altered["profile"]["planner"]["noise_sigma"] *= 2
    with pytest.raises(ValueError, match="temperature recipe"):
        controller.validate_stage(altered, swing=False)
    altered = copy.deepcopy(stages["swing-development"])
    altered["lqr_steps"] = 100
    with pytest.raises(ValueError, match="suffix"):
        controller.validate_stage(altered, swing=True)


def test_stop_hook_sees_durably_recorded_complete_context_and_default_is_unchanged(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(teacher, "_capture", capture)
    recipe = dict(name="synthetic", planner=dict(H=3, K=2, initial_iterations=1))
    outputs = []
    for label, callback in (("default", None), ("false", lambda e, g, n: False), ("stop", "stop")):
        out = tmp_path / label
        out.mkdir()
        env = FakeEnv()
        seen = []

        def stop(current, group, n):
            assert len(group["actions"]) == n
            assert len(group["states"]) == len(group["integration"]) == n + 1
            assert np.array_equal(group["states"][-1], current.get_state())
            seen.append(n)
            return n == 2

        with h5py.File(out / "evidence.h5", "x") as f:
            group = f.create_group("trial")
            row = teacher._run_trial(
                object(),
                env,
                FakeMPPI,
                group,
                budget(out),
                recipe,
                dict(reset_seed=1, planner_seed=10001),
                "synthetic",
                0,
                0,
                out,
                controls=4,
                initial_state=np.zeros(5),
                metrics_fn=controller.trace_metrics,
                stop_after_action=stop if callback == "stop" else callback,
            )
            outputs.append(
                {
                    key: group[key][:]
                    for key in (
                        *controller.CONTEXT_FIELDS,
                        "actions",
                        "costs",
                        "pre_action_observations",
                    )
                }
            )
        assert row["controls"] == env.step_calls == (2 if label == "stop" else 4)
        if label == "stop":
            assert seen == [1, 2]
    for key in outputs[0]:
        assert np.array_equal(outputs[0][key], outputs[1][key])


def test_first_bank_pairing_fails_before_any_execution_and_preserves_receipt(tmp_path):
    baseline = tmp_path / "baseline.h5"
    with h5py.File(baseline, "x") as f:
        old = f.create_group("trial/planning/control_0000/pass_000")
        old["candidate_costs"] = np.arange(4, dtype=float)
        old["nominal_before"] = np.zeros((3, 1))
    with h5py.File(tmp_path / "new.h5", "x") as f:
        group = f.create_group("bank")
        group["candidate_costs"] = np.arange(4, dtype=float)
        group["nominal_before"] = np.zeros((3, 1))
        binding = dict(h5_path=str(baseline), group="trial")
        assert controller.verify_first_bank(group, binding)["status"] == "PASSED"
        group["candidate_costs"][0] = 0.1
        with pytest.raises(teacher.InvalidExecution, match="first-pass bank mismatch"):
            controller.verify_first_bank(group, binding)
        assert (
            json.loads(group.attrs["baseline_first_bank_comparison"])["fields"]["candidate_costs"][
                "passed"
            ]
            is False
        )


def test_swing_join_keeps_context_no_reset_and_charges_only_one_final_terminal(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(teacher, "_capture", capture)
    monkeypatch.setattr(controller, "_capture", capture)
    stage = copy.deepcopy(frozen_config()["stages"]["swing-development"])
    for start in stage["starts"]:
        fake = FakeEnv()
        fake.reset(state=np.asarray(start["initial_state"]))
        start["initial_context"] = {
            key: value.tolist() for key, value in capture(None, fake).items()
        }
    env = FakeEnv()
    work = budget(tmp_path)
    mj = SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=8191))
    result = controller.run_swingup(
        stage,
        tmp_path,
        work,
        dict(K=np.zeros((1, 4)), P=np.eye(4), rho=0.01),
        env=env,
        mj=mj,
        mppi_type=FakeMPPI,
    )
    assert result["successes"] == result["joins"] == 2
    assert result["weighting_validated"] and result["bank_count"] == 6
    assert env.reset_calls == work.work.resets == 4  # Two checks and two prefixes; no join resets.
    assert env.step_calls == work.work.native_controls_completed == 402
    assert work.work.model_queries_completed == 6
    with h5py.File(tmp_path / "evidence.h5", "r") as f:
        for i, row in enumerate(result["trials"]):
            assert (
                row["mppi_controls"] == 1
                and row["lqr_controls"] == 200
                and row["total_controls"] == 201
            )
            assert (
                row["mppi_prefix_metrics"]["task_cost"] == 12
            )  # Five running plus prefix diagnostic terminal7.
            assert row["task_cost"] == 1012  # 201*5 +one terminal7; never1019.
            group = f[f"trials/variant_0/trial_{i}"]
            suffix = group["lqr_suffix"]
            assert len(suffix["actions"]) == 200 and len(suffix["states"]) == 201
            for key in controller.CONTEXT_FIELDS:
                assert np.array_equal(group[key][-1], suffix[key][0])
            assert np.array_equal(suffix["pre_action_observations"][:], suffix["states"][:-1, 1:5])
            assert group.attrs["teacher_success"] and suffix.attrs["success"]
    assert len((tmp_path / "trials.jsonl").read_text().splitlines()) == 2
    assert len((tmp_path / "hybrid_trials.jsonl").read_text().splitlines()) == 2


def test_no_join_is_a_task_failure_not_a_shortened_or_invalid_episode():
    prefix = dict(
        controls=600,
        native_running_cost_sum=100,
        episode_terminal_cost=7,
        original_dense_cost_sum=80,
    )
    row = controller.hybrid_metrics(prefix, None, dict(control=None, V=None, rho=0.1))
    assert not row["success"] and not row["joined"]
    assert row["mppi_controls"] == row["total_controls"] == 600 and row["lqr_controls"] == 0
    assert row["task_cost"] == 107


def test_unreachable_minimum_ties_are_explicit_and_not_called_validated():
    rows = [dict(target_reached=False, unreachable_ties=True, ess=256, temperature=1)]
    summary = controller._weighting_summary(rows)
    assert not summary["weighting_validated"] and summary["unreachable_tie_banks"] == 1
