"""Step 18: a context re-weights MPPI's finished samples and never enters its search."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.diagnostics.context_rescore_check import _episode, report, rescored_action
from scripts.diagnostics.walker_headroom import _episode as raw_mppi_episode
from scripts.training.train_walker_mdgps import _batch_act, _new_actor, policy_state
from src.envs.walker2d import Walker2d
from src.gps.policy_tail import WalkerPolicyTail
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _planner(K: int = 8, H: int = 6) -> tuple[Walker2d, MPPI]:
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    cfg = replace(MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text())), K=K, H=H)
    np.random.seed(5)
    env.reset()
    return env, MPPI(env, cfg, seed=3)


@pytest.mark.parametrize("rule", ["temperature", "neff"])
def test_without_a_tail_the_rescored_action_is_mppis_own(rule):
    env, planner = _planner()
    action, info = planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
    rollout = planner.get_rollout_data()
    chosen, weights = rescored_action(rollout, np.zeros(planner.K), info["lam"], rule, info["n_eff"])
    np.testing.assert_allclose(weights, rollout["weights"], atol=1e-9)
    np.testing.assert_allclose(chosen, action, atol=1e-9)


def test_rescoring_follows_the_tail():
    rng = np.random.default_rng(0)
    rollout = {"costs": np.full(16, 10.0), "actions": rng.uniform(-1.0, 1.0, (16, 4, 6))}
    tail = np.full(16, 100.0)
    tail[7] = 0.0
    chosen, weights = rescored_action(rollout, tail, lam=3.0)
    assert weights.argmax() == 7
    np.testing.assert_allclose(chosen, rollout["actions"][7, 0], atol=1e-9)


def test_the_neff_rule_keeps_mppis_own_effective_sample_size():
    rng = np.random.default_rng(1)
    rollout = {"costs": rng.gamma(2.0, 2.0, 32), "actions": rng.uniform(-1.0, 1.0, (32, 4, 6))}
    tail = rng.gamma(2.0, 20.0, 32)
    own = effective_sample_size(np.exp(-(rollout["costs"] - rollout["costs"].min()) / 3.0)
                                / np.exp(-(rollout["costs"] - rollout["costs"].min()) / 3.0).sum())
    _, kept = rescored_action(rollout, tail, 3.0, "neff", own)
    _, sharper = rescored_action(rollout, tail, 3.0, "temperature")
    np.testing.assert_allclose(effective_sample_size(kept), own, rtol=1e-3)
    assert effective_sample_size(sharper) < own


def test_the_context_changes_the_executed_action_but_not_mppi_or_the_simulator():
    env, planner = _planner()
    twin_env, twin = _planner()
    state = env.get_state()
    action, info = planner.plan_step(state, initial_warmstart=env.get_warmstart())
    twin.plan_step(twin_env.get_state(), initial_warmstart=twin_env.get_warmstart())
    rollout = planner.get_rollout_data()
    tail = WalkerPolicyTail(env, _batch_act(_new_actor(0)), 10)(rollout["states"], rollout["actions"])
    chosen, weights = rescored_action(rollout, tail, info["lam"], "neff", info["n_eff"])
    np.testing.assert_array_equal(planner.U, twin.U)
    np.testing.assert_array_equal(env.get_state(), state)
    assert not np.allclose(weights, rollout["weights"])
    assert not np.allclose(chosen, action)


def test_no_context_is_raw_mppi():
    job = {"K": 8, "H": 6, "steps": 15, "reset_seed": 70_000, "planner_seed": 95_000}
    ours = _episode({**job, "context": "none", "rule": None, "context_path": None, "tail_steps": 0})
    raw = raw_mppi_episode({**job, "neff_fraction": None, "tail_policy": None, "tail_steps": 0})
    for key in ("steps", "survived", "mean_vx", "velocity_mae"):
        assert ours[key] == raw[key]
    assert ours["median_own_neff"] == raw["median_neff"]


@pytest.mark.parametrize("rule", ["temperature", "neff"])
def test_a_context_episode_runs_and_records_its_reweighting(tmp_path, rule):
    path = tmp_path / "context.pt"
    torch.save(policy_state(_new_actor(0)), path)
    row = _episode({"K": 8, "H": 6, "steps": 5, "reset_seed": 70_000, "planner_seed": 95_000,
                    "context": "random", "rule": rule, "context_path": str(path), "tail_steps": 4})
    assert row["steps"] >= 1 and row["median_used_neff"] is not None and row["median_action_shift"] > 0.0


def test_report_pairs_each_context_with_no_context_and_the_other_rule():
    arms = [("none", None)] + [(c, r) for c in ("sac_800k", "step15_student") for r in ("neff", "temperature")]
    rows = [{"H": h, "context": c, "rule": r, "reset_seed": s, "survived": bool((s + h) % 3),
             "velocity_mae": 0.3 + 0.01 * s, "steps": 500 + s, "median_own_neff": 2.0,
             "median_used_neff": None if c == "none" else 1.5, "median_action_shift": 0.0, "mean_plan_seconds": 0.1}
            for h in (16, 48) for c, r in arms for s in range(6)]
    out = report(rows)
    assert set(out["arms"]) == {"H16_none", "H48_none"} | {
        f"H{h}_{c}_{r}" for h in (16, 48) for c, r in arms if c != "none"}
    assert {"H16_sac_800k_neff:H16_none", "H16_sac_800k_neff:H48_none", "H48_sac_800k_neff:H48_none",
            "H48_sac_800k_neff:H48_step15_student_neff", "H48_sac_800k_neff:H48_sac_800k_temperature",
            "H16_step15_student_temperature:H48_none"} <= set(out["paired"])
    assert "H48_sac_800k_neff:H16_none" not in out["paired"]
