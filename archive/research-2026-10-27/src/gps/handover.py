"""Earned channels from the policy into independent MPPI (docs/trust_priced_consensus.md).

Both channels are priced by the trust field and add exactly nothing where the
policy has not earned trust. MPPI keeps its own plan and its own samples; these
only change how it scores them.

- TrustPricedMatching (c1): the network-matching cost with
  alpha = alpha0 * tau_hat(s) at the decision state. Used as `prior_cost`.
- HandoverTerminal (c3): among the `candidates` lowest-score samples that stay
  healthy, those ending at a trusted state s_H are charged
  tau_hat(s_H) * (V(s_H) - median V), where V is the policy's exact closed-loop
  cost over `window` controls from s_H. MPPI then prefers to hand the robot over
  where the policy has shown it can continue. Used as `coupling`.
"""
from collections.abc import Callable

import numpy as np

from src.envs.walker2d import Walker2d
from src.gps.policy_matching import PolicyMatching
from src.gps.policy_tail import WalkerPolicyTail
from src.gps.trust import TrustField


class TrustPricedMatching:
    def __init__(self, env: Walker2d, act_batch: Callable[[np.ndarray], np.ndarray],
                 trust: TrustField, alpha0: float) -> None:
        self.matching = PolicyMatching(env, act_batch, alpha0)
        self.trust, self.alpha0 = trust, alpha0

    def for_decision(self, state: np.ndarray, obs: np.ndarray):
        """The prior cost for this decision, or None where tau_hat(s) is 0."""
        tau = self.trust(obs)
        if tau <= 0.0:
            return None
        self.matching.alpha = self.alpha0 * tau
        self.matching.state = state
        return self.matching


class HandoverTerminal:
    def __init__(self, env: Walker2d, act_batch: Callable[[np.ndarray], np.ndarray],
                 trust: TrustField, window: int, candidates: int = 16) -> None:
        if candidates < 1:
            raise ValueError("candidates must be positive.")
        self.env, self.trust, self.candidates = env, trust, candidates
        self.tail = WalkerPolicyTail(env, act_batch, window)
        self.window = window
        self.tail_controls = 0

    def __call__(self, *, states, actions, costs, base_score, lam) -> dict:
        score = np.asarray(base_score, dtype=float).copy()
        healthy = np.all(self.env._is_healthy(states[..., 2], states[..., 3]), axis=1) & np.isfinite(score)
        rows = np.flatnonzero(healthy)
        top = rows[np.argsort(score[rows])[:self.candidates]]
        tau = np.atleast_1d(self.trust(self.env.rollout_states_to_obs(states[top, -1]))) if len(top) else np.zeros(0)
        use, tau = top[tau > 0.0], tau[tau > 0.0]
        if len(use) == 0:
            return {"score": score, "info": {"handover_evaluated": 0.0}}
        value = self.tail(states[use], actions[use])
        self.tail_controls += len(use) * self.window
        term = tau * (value - np.median(value))
        score[use] += term
        return {"score": score, "info": {"handover_evaluated": float(len(use)),
                                         "handover_term_spread": float(np.ptp(term))}}
