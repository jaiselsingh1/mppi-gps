"""Averaged iterates: action averaging and weight averaging of policies."""
import numpy as np
import torch

from scripts.training.train_walker_mdgps import _new_actor, policy_state
from src.gps.averaging import AveragedPolicy, weight_average
from src.rl.sac import load_policy


def test_averaged_policy_acts_with_the_mean_action():
    actors = [_new_actor(seed) for seed in (0, 1)]
    obs = np.linspace(-1.0, 1.0, actors[0].obs_dim)
    expected = (actors[0].act(obs) + actors[1].act(obs)) / 2
    np.testing.assert_allclose(AveragedPolicy(actors).act(obs), expected, rtol=1e-6)
    np.testing.assert_array_equal(AveragedPolicy(actors[:1]).act(obs), actors[0].act(obs))


def test_weight_average_loads_and_averages_parameters():
    actors = [_new_actor(seed) for seed in (0, 1)]
    averaged = load_policy(weight_average([policy_state(actor) for actor in actors]))
    for name, value in averaged.state_dict().items():
        torch.testing.assert_close(value, (actors[0].state_dict()[name] + actors[1].state_dict()[name]) / 2)
    same = load_policy(weight_average([policy_state(actors[0])] * 3))
    obs = np.zeros(actors[0].obs_dim)
    # Averaging three float32 copies can round by one ULP. Near-zero actions
    # require an absolute tolerance as well as relative error (observed on
    # Linux x86_64: 7.45e-9 absolute error for a 3.23e-4 action).
    np.testing.assert_allclose(same.act(obs), actors[0].act(obs), rtol=1e-6, atol=1e-7)
