"""labeler="teacher": MPPI runs every episode with action noise and labels its own states."""
from dataclasses import replace

import numpy as np

from scripts.training.train_walker_mdgps import TrainConfig, _collect, _new_actor, _reweighted_label
from src.utils.math import effective_sample_size, softmin_weights, temperature_for_neff_fraction


def _job(cfg):
    actor = _new_actor(0)
    return {"cfg": cfg, "actor": {k: v.cpu() for k, v in actor.state_dict().items()}, "reset_seed": 7,
            "planner_seed": 11, "noise_seed": 13, "execute_expert": True,
            "probe_probability": 0.0, "neff_fraction": 0.2}


def test_teacher_without_noise_is_the_executing_expert():
    dagger = TrainConfig(labeler="dagger", K=8, H=6, steps=12, workers=1)
    expert = _collect(_job(dagger))
    teacher = _collect(_job(replace(dagger, labeler="teacher", exec_noise=0.0)))
    np.testing.assert_array_equal(teacher["obs"], expert["obs"])
    np.testing.assert_array_equal(teacher["labels"], expert["labels"])


def test_teacher_noise_moves_the_states_not_the_first_label():
    cfg = TrainConfig(labeler="teacher", K=8, H=6, steps=12, workers=1)
    quiet = _collect(_job(replace(cfg, exec_noise=0.0)))
    noisy = _collect(_job(cfg))
    np.testing.assert_array_equal(noisy["labels"][0], quiet["labels"][0])
    assert not np.allclose(noisy["obs"][1:3], quiet["obs"][1:3])


def test_reweighted_label_is_the_smoother_sample_average():
    rng = np.random.default_rng(0)
    rollout = {"costs": rng.gamma(2.0, 3.0, 32), "actions": rng.uniform(-1.0, 1.0, (32, 6, 6))}
    weights = softmin_weights(rollout["costs"], temperature_for_neff_fraction(rollout["costs"], 0.2))
    np.testing.assert_allclose(effective_sample_size(weights), 0.2 * 32, rtol=1e-3)
    np.testing.assert_allclose(_reweighted_label(rollout, 0.2), weights @ rollout["actions"][:, 0])


def test_smoother_labels_leave_execution_unchanged():
    cfg = TrainConfig(labeler="teacher", K=16, H=6, steps=12, workers=1, exec_noise=0.0)
    plain = _collect(_job(cfg))
    smooth = _collect(_job(replace(cfg, label_neff_fraction=0.2)))
    np.testing.assert_array_equal(smooth["obs"], plain["obs"])
    np.testing.assert_array_equal(smooth["diagnostics"][:, 0], plain["diagnostics"][:, 0])
    assert not np.allclose(smooth["labels"], plain["labels"])
    np.testing.assert_allclose(smooth["diagnostics"][:, 4], np.linalg.norm(smooth["labels"] - plain["labels"], axis=1))
