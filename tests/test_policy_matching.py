"""The network-matching term scores each MPPI sample against the policy at the states where it acts."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from scripts.training.train_walker_mdgps import _batch_act, _new_actor
from src.envs.walker2d import Walker2d
from src.gps.policy_matching import PolicyMatching
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _walker() -> Walker2d:
    return Walker2d(**json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text()), nthread=1)


def test_the_term_compares_each_action_with_the_policy_at_the_state_where_it_is_applied():
    env, act = _walker(), _batch_act(_new_actor(0))
    rng = np.random.default_rng(0)
    np.random.seed(1)
    env.reset()
    current = env.get_state()
    states = current + 0.05 * rng.standard_normal((4, 5, current.size))
    before = np.concatenate([np.broadcast_to(current, (4, 1, current.size)), states[:, :-1]], axis=1)
    on_policy = act(env.rollout_states_to_obs(before).reshape(20, -1)).reshape(4, 5, -1)
    matching = PolicyMatching(env, act, alpha=2.0)
    matching.state = current
    np.testing.assert_allclose(matching(states, on_policy), 0.0, atol=1e-10)
    np.testing.assert_allclose(matching(states, on_policy + 0.1), 0.5 * 2.0 * 5 * on_policy.shape[-1] * 0.01)


def test_alpha_must_be_nonnegative_and_the_state_set():
    env, act = _walker(), _batch_act(_new_actor(0))
    with pytest.raises(ValueError):
        PolicyMatching(env, act, alpha=-1.0)
    with pytest.raises(RuntimeError):
        PolicyMatching(env, act, alpha=1.0)(np.zeros((2, 3, 19)), np.zeros((2, 3, 6)))


def test_a_strong_term_pulls_mppis_action_toward_the_policy():
    target = np.array([0.6, -0.6, 0.6, -0.6, 0.6, -0.6])
    constant = lambda obs: np.tile(target, (len(obs), 1))

    def first_action(alpha: float) -> np.ndarray:
        env = _walker()
        np.random.seed(3)
        env.reset()
        cfg = replace(MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text())), K=64, H=8)
        planner = MPPI(env, cfg, seed=5)
        matching = PolicyMatching(env, constant, alpha) if alpha else None
        if matching is not None:
            matching.state = env.get_state()
        return planner.plan_step(env.get_state(), prior_cost=matching, initial_warmstart=env.get_warmstart())[0]

    assert np.linalg.norm(first_action(10.0) - target) < np.linalg.norm(first_action(0.0) - target)
