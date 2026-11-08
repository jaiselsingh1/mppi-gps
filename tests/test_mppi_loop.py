"""Small checks of the loop's pieces that need no saved runs."""

import copy

import numpy as np
import torch

from src.mppi_loop.fit import mixed_targets
from src.mppi_loop.policy import Policy, tanh_network
from src.mppi_loop.similarity import medoid, reference_trees, set_distance, similarity_cost
from src.mppi_loop.spec import WALKER


def _policy(seed=0):
    torch.manual_seed(seed)
    return Policy(dict(mean=np.zeros(17), std=np.ones(17)), tanh_network(17, 6))


def test_similarity_cost_is_zero_on_the_reference_and_grows_away_from_it():
    rng = np.random.default_rng(0)
    obs, sd = WALKER.observed, np.ones(17)
    ref = rng.normal(size=(50, 19))
    prior = similarity_cost(reference_trees([(ref, 1.0)], obs, sd), 2.0, obs, sd)
    on = prior(ref[None, :10], None)
    off = prior(ref[None, :10] + 0.5, None)
    assert on.shape == (1,) and np.allclose(on, 0.0)
    assert off[0] > 0.0


def test_reference_shares_weight_the_cost():
    rng = np.random.default_rng(1)
    obs, sd = WALKER.observed, np.ones(17)
    a, b = rng.normal(size=(20, 19)), rng.normal(size=(20, 19)) + 3.0
    states = a[None, :5]
    only_a = similarity_cost(reference_trees([(a, 1.0), (b, 0.0)], obs, sd), 1.0, obs, sd)
    only_b = similarity_cost(reference_trees([(a, 0.0), (b, 1.0)], obs, sd), 1.0, obs, sd)
    assert only_a(states, None)[0] < only_b(states, None)[0]


def test_medoid_picks_the_walk_between_the_others():
    base = np.zeros((30, 19))
    walks = [base + 0.0, base + 1.0, base + 2.0]
    m, d = medoid(walks, WALKER.observed, np.ones(17))
    assert m == 1
    assert np.isclose(d, set_distance(walks[0], walks[1], WALKER.observed, np.ones(17)))


def test_policy_predicts_bounded_actions_and_state_dict_round_trips():
    p = _policy()
    x = np.random.default_rng(2).normal(size=(4, 17)).astype(np.float32)
    u = p.predict(x)
    assert u.shape == (4, 6) and np.all(np.abs(u) <= 1.0)
    q = _policy(seed=5)
    q.load_state_dict(copy.deepcopy(p.state_dict()))
    assert np.array_equal(q.predict(x), u)


def test_mixed_targets_at_s_one_are_the_anchor_itself():
    p = _policy()
    rng = np.random.default_rng(3)
    X = rng.normal(size=(8, 19))
    U, K, SC = rng.normal(size=(8, 6)), rng.normal(size=(8, 6, 18)), np.ones((8, 18))
    Um, Km = mixed_targets(p, 1.0, WALKER, X, U, K, SC)
    own = p.predict(WALKER.observation(X).astype(np.float32))
    assert np.allclose(Um, own, atol=1e-6)
    Um0, Km0 = mixed_targets(p, 0.0, WALKER, X, U, K, SC)
    assert np.allclose(Um0, U) and np.allclose(Km0, K)
