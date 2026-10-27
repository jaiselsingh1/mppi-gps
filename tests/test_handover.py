"""Earned channels (src/gps/handover.py): nothing enters MPPI where the policy is untrusted."""
from dataclasses import replace

import numpy as np

from scripts.training.train_walker_mdgps import _walker
from src.gps.handover import HandoverTerminal, TrustPricedMatching
from src.gps.trust import TrustField
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def _field(trusted: bool) -> TrustField:
    """Probes spread widely around the Walker's start, all labelled the same way."""
    env = _walker()
    np.random.seed(3)
    centre = env.reset()
    obs = centre + np.random.default_rng(0).normal(scale=2.0, size=(400, 17))
    return TrustField(obs, np.full(400, trusted), k=20, familiar_quantile=1.0)


def _act(obs):
    return np.full(np.shape(obs)[:-1] + (6,), 0.3)


def _plan(coupled: bool, trusted: bool):
    env = _walker()
    np.random.seed(3)
    obs = env.reset()
    planner = MPPI(env, replace(MPPIConfig.load("walker2d"), K=8, H=6), seed=11)
    matching = TrustPricedMatching(env, _act, _field(trusted), alpha0=3.0)
    handover = HandoverTerminal(env, _act, _field(trusted), window=5, candidates=4)
    actions = []
    for _ in range(3):
        state = env.get_state()
        prior = matching.for_decision(state, obs) if coupled else None
        action, _ = planner.plan_step(state, prior_cost=prior, coupling=handover if coupled else None,
                                      initial_warmstart=env.get_warmstart())
        actions.append(action)
        obs, _, _, _ = env.step(action)
    return np.array(actions), planner.U.copy(), handover.tail_controls


def test_untrusted_channels_leave_mppi_bitwise_unchanged():
    plain_actions, plain_plan, _ = _plan(coupled=False, trusted=False)
    actions, plan, tail_controls = _plan(coupled=True, trusted=False)
    assert np.array_equal(actions, plain_actions) and np.array_equal(plan, plain_plan)
    assert tail_controls == 0


def test_trusted_channels_change_the_scores():
    plain_actions, _, _ = _plan(coupled=False, trusted=True)
    actions, _, tail_controls = _plan(coupled=True, trusted=True)
    assert tail_controls > 0 and not np.array_equal(actions, plain_actions)


def test_matching_weight_is_priced_by_trust():
    env = _walker()
    np.random.seed(3)
    obs = env.reset()
    field = _field(True)
    matching = TrustPricedMatching(env, _act, field, alpha0=2.0)
    prior = matching.for_decision(env.get_state(), obs)
    assert prior is not None and prior.alpha == 2.0 * field(obs)
    assert TrustPricedMatching(env, _act, _field(False), 2.0).for_decision(env.get_state(), obs) is None
