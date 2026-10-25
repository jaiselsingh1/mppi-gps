"""Selection-to-replay-to-fit integration using real CPU Walker dynamics."""
import json

import numpy as np
import pytest
import torch

from scripts.training.train_policy_selective import PolicySelectiveConfig
from scripts.training.train_walker_mdgps import _batch_act, _evaluate, _fit, _new_actor, main
from src.gps.policy import TanhMLPPolicy
from src.rl.sac import load_policy


def test_selective_preset_matches_the_declared_experiment() -> None:
    cfg = PolicySelectiveConfig()
    assert (cfg.K, cfg.H, cfg.tail_steps, cfg.neff_fraction) == (256, 32, 100, .2)
    assert (cfg.iterations, cfg.episodes_per_iter, cfg.steps, cfg.label_every) == (75, 10, 400, 2)
    assert (cfg.replay_iterations, cfg.batch_size, cfg.lr, cfg.m_step_updates) == (2, 256, 3e-4, 2000)
    assert cfg.selection_mode == "selective" and cfg.policy_architecture == "tanh"
    assert cfg.gpu_estep and cfg.sigma_decay_iterations == 25


@pytest.mark.parametrize("mode", ["anchor", "selective"])
def test_two_iteration_selection_replay_and_standalone_checkpoint(tmp_path, mode: str) -> None:
    cfg = PolicySelectiveConfig(K=8, H=4, tail_steps=3, iterations=2, episodes_per_iter=2,
                                steps=12, workers=1, m_step_updates=3, batch_size=8,
                                eval_every=1, eval_episodes=1, probe_probability=1.,
                                selection_mode=mode, gpu_estep=False, save_labels=True, research_ladder=False)
    directory = tmp_path / mode
    main(directory, cfg)
    rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    archives = [np.load(directory / f"labels_{iteration:03d}.npz") for iteration in range(2)]
    for iteration, (row, archive) in enumerate(zip(rows, archives)):
        for name in ("fit_loss", "share_plans_worse", "median_action_delta", "sigma_k",
                     "mean_n_eff", "median_lambda", "test_return_standalone"):
            assert np.isfinite(row[name]), (name, row)
        assert row["probe_improvement_ratio"] is None or np.isfinite(row["probe_improvement_ratio"])
        np.testing.assert_allclose(row["sigma_k"], .3 / np.sqrt(1 + iteration / 25))
        delta = np.linalg.norm(archive["proposals"] - archive["old_actions"], axis=1)
        improvement = archive["diagnostics"][:, 2] - archive["diagnostics"][:, 3]
        if mode == "selective":
            expected = np.where(improvement > 0, np.tanh(np.maximum(improvement, 0) / cfg.tau_s)
                                * np.exp(-delta ** 2 / (2 * row["sigma_k"] ** 2)), 0)
            np.testing.assert_allclose(archive["weights"], expected, atol=1e-14)
            np.testing.assert_array_equal(archive["labels"], archive["proposals"])
        else:
            expected = np.where((improvement < 0)[:, None], archive["old_actions"], archive["proposals"])
            np.testing.assert_array_equal(archive["labels"], expected)
            np.testing.assert_array_equal(archive["weights"], np.ones(len(expected)))
        expected_mass = sum(np.sum(item["weights"]) for item in archives[:iteration + 1])
        np.testing.assert_allclose(row["replay_weight_mass"], expected_mass)
    assert rows[1]["replay_size"] == sum(len(item["weights"]) for item in archives)
    payload = torch.load(directory / "policy_001.pt", map_location="cpu", weights_only=True)
    deployed = load_policy(payload)
    assert isinstance(deployed, TanhMLPPolicy)
    # Evaluation of the saved artifact cannot instantiate a training teacher.
    from unittest.mock import patch
    with patch("src.mppi.mppi.MPPI.__init__", side_effect=AssertionError("planner at inference")):
        assert np.isfinite(_evaluate(deployed, episodes=1, seed=81_000)["training_return"])


def test_zero_weight_selection_skips_adam_even_with_weight_decay(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    actor = _new_actor(0, "tanh")
    before = {name: tensor.clone() for name, tensor in actor.state_dict().items()}
    cfg = PolicySelectiveConfig(m_step_updates=2, weight_decay=.5)
    loss = _fit(actor, np.zeros((2, 17)), np.ones((2, 6)), cfg,
                np.random.default_rng(0), np.zeros(2))
    assert loss == 0
    for name, tensor in actor.state_dict().items():
        assert torch.equal(before[name], tensor)


def test_fit_uses_weighted_action_norm_and_ignores_zero_rows(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    actor = _new_actor(1, "tanh")
    obs = np.zeros((2, 17))
    target = np.array([[.25] * 6, [-1.] * 6])
    prediction = _batch_act(actor)(obs)
    expected = np.sum((prediction[0] - target[0]) ** 2)
    loss = _fit(actor, obs, target, PolicySelectiveConfig(m_step_updates=1),
                np.random.default_rng(0), np.array([.2, 0.]))
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
