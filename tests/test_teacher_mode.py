"""labeler="teacher": MPPI runs every episode with action noise and labels its own states."""
from dataclasses import replace

import numpy as np

from scripts.training.train_walker_mdgps import TrainConfig, _collect, _new_actor


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
