"""MPPI-GPS as mirror descent, method (B) of docs/objective.md.

E-step: MPPI samples around the policy's own closed-loop rollout (the policy
is the base measure) and takes one KL-bounded step; its first action is the
label. M-step: regress the policy onto labels at states the policy visits.
"""
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from src.envs.mujoco_env import MuJoCoEnv
from src.mppi.mppi import MPPI

_FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


def closed_loop_nominal(
    env: MuJoCoEnv,
    state: np.ndarray,
    act: Callable[[np.ndarray], np.ndarray],
    horizon: int,
    warmstart: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll `act(full_state) -> action` closed-loop for `horizon` control steps.

    Uses a scratch MjData, so the live simulator is untouched. Replaying the
    returned actions open-loop from the same state and warm start reproduces
    the returned states.
    """
    data = mujoco.MjData(env.model)
    mujoco.mj_setState(env.model, data, state, _FULLPHYSICS)
    if warmstart is not None:
        data.qacc_warmstart[:] = warmstart
    low, high = env.action_bounds
    current = state.copy()
    actions = np.empty((horizon, env.action_dim))
    states = np.empty((horizon, len(state)))
    for t in range(horizon):
        actions[t] = np.clip(act(current), low, high)
        data.ctrl[:] = actions[t]
        for _ in range(env._frame_skip):
            mujoco.mj_step(env.model, data)
        mujoco.mj_getState(env.model, data, current, _FULLPHYSICS)
        states[t] = current
    return actions, states


def mdgps_step_multiplier(predicted: float, actual: float) -> float:
    """MDGPS "global" step rule (Montgomery & Levine 2016, §4.2).

    Fitting the cost change as a*eps^2 + b*eps gives eps' = eps * pred /
    2(pred - actual), with pred and actual the predicted and realized
    improvement of the policy. As in the reference GPS implementation, the
    denominator is floored at 1e-4 and the multiplier clipped to [0.1, 5]:
    realizing at least the prediction grows the step 5x, and predicting no
    improvement shrinks it 10x.
    """
    return float(np.clip(predicted / (2.0 * max(1e-4, predicted - actual)), 0.1, 5.0))


@dataclass
class EStepResult:
    label: np.ndarray
    lam: float
    n_eff: float
    nominal_cost: float
    sample_mean_cost: float
    expected_cost: float
    plan_cost: float
    clip_fraction: float

    @property
    def sample_improvement(self) -> float:
        """E_p[S] - E_q*[S] over the noisy samples: fact 1, always >= 0."""
        return self.sample_mean_cost - self.expected_cost

    @property
    def predicted_improvement(self) -> float:
        """S(policy plan) - S(improved mean plan): MDGPS's prediction for a
        deterministic policy that reproduces the improved plan."""
        return self.nominal_cost - self.plan_cost


def e_step(
    planner: MPPI,
    env: MuJoCoEnv,
    state: np.ndarray,
    nominal: np.ndarray,
    warmstart: np.ndarray | None = None,
    terminal=None,
) -> EStepResult:
    """One policy-centred MPPI step from `state`; requires initial_iterations=1.

    `terminal` is an optional cost-to-go hook with the WalkerPolicyTail
    interface (called on rollout states and actions, exposing `last_tail`).
    Every cost reported here, for samples, nominal and plan alike, is the
    H-step cost plus that cost-to-go, so the bookkeeping scores one objective.
    """
    if planner.initial_iterations != 1:
        raise ValueError("The E-step takes exactly one KL-bounded step per state.")
    kwargs = {} if warmstart is None else {"initial_warmstart": warmstart}
    action, info = planner.plan_step(state, nominal=nominal, prior_cost=terminal, **kwargs)
    rollout = planner.get_rollout_data()
    sample_costs = rollout["costs"] + (terminal.last_tail if terminal is not None else 0.0)

    def total_cost(sequence: np.ndarray) -> float:
        states, cost, _ = env.batch_rollout(state, sequence[None], **kwargs)
        extra = terminal(states, sequence[None])[0] if terminal is not None else 0.0
        return float(cost[0] + extra)

    low, high = env.action_bounds
    sampled = rollout["actions"]
    finite = np.isfinite(sample_costs)
    return EStepResult(
        label=action,
        lam=info["lam"],
        n_eff=info["n_eff"],
        nominal_cost=total_cost(nominal),
        sample_mean_cost=float(np.mean(sample_costs[finite])),
        expected_cost=float(np.sum(rollout["weights"][finite] * sample_costs[finite])),
        plan_cost=total_cost(planner.last_plan),
        clip_fraction=float(np.mean((sampled <= low) | (sampled >= high))),
    )
