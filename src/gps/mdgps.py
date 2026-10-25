"""MPPI-GPS as mirror descent, method (B) of docs/objective.md.

E-step: MPPI samples around the policy's own closed-loop rollout (the policy
is the base measure) and takes one weighted sample update; its first action is
the label. Optional temperature selection targets empirical sample ESS, not a
population or neural-policy KL bound. M-step: regress onto visited-state labels.
"""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np
from jaxtyping import Float

from src.envs.mujoco_env import MuJoCoEnv
from src.mppi.mppi import MPPI
from src.utils.math import effective_sample_size, softmin_weights

_FULLPHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS


class PolicyTail(Protocol):
    """Cost-to-go hook exposing its most recent per-sample tail costs."""

    last_tail: Float[np.ndarray, "samples"]

    def __call__(
        self,
        states: Float[np.ndarray, "samples horizon nstate"],
        actions: Float[np.ndarray, "samples horizon action"],
    ) -> Float[np.ndarray, "samples"]: ...


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
    label: Float[np.ndarray, "action"]
    lam: float
    n_eff: float
    nominal_cost: float
    sample_mean_cost: float
    expected_cost: float
    plan_cost: float
    clip_fraction: float
    tail_var_ratio: float | None = None
    lambda_saturated: bool | None = None
    numerical_valid: bool = True

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
    terminal: PolicyTail | None = None,
) -> EStepResult:
    """One policy-centred MPPI step from `state`; requires initial_iterations=1.

    `terminal` is an optional cost-to-go hook with the WalkerPolicyTail
    interface (called on rollout states and actions, exposing `last_tail`).
    Every cost reported here, for samples, nominal and plan alike, is the
    H-step cost plus that cost-to-go, so the bookkeeping scores one objective.
    """
    if planner.initial_iterations != 1:
        raise ValueError("The E-step takes exactly one MPPI update per state.")
    kwargs = {} if warmstart is None else {"initial_warmstart": warmstart}
    action, info = planner.plan_step(state, nominal=nominal, prior_cost=terminal, **kwargs)
    rollout = planner.get_rollout_data()
    # Snapshot existing sample tails before nominal/mean verification reuses
    # the tail hook and overwrites its diagnostic arrays.
    sample_tails = terminal.last_tail.copy() if terminal is not None else np.zeros_like(rollout["costs"])
    sample_costs = rollout["costs"] + sample_tails
    tail_actions_valid = bool(getattr(terminal, "last_actions_finite", True))
    numerical_valid = bool(np.isfinite(rollout["weights"]).all()
                           and np.isfinite(rollout["actions"]).all()
                           and np.isfinite(action).all()
                           and np.isfinite(nominal).all()
                           and np.isfinite(planner.last_plan).all()
                           and tail_actions_valid
                           and _raw_weights_finite(sample_costs, info["lam"]))

    def total_cost(sequence: np.ndarray) -> float:
        nonlocal numerical_valid
        states, cost, _ = env.batch_rollout(state, sequence[None], **kwargs)
        extra = terminal(states, sequence[None])[0] if terminal is not None else 0.0
        numerical_valid = numerical_valid and bool(getattr(terminal, "last_actions_finite", True))
        return float(cost[0] + extra)

    low, high = env.action_bounds
    sampled = rollout["actions"]
    finite = np.isfinite(sample_costs)
    nominal_cost = total_cost(nominal)
    plan_cost = total_cost(planner.last_plan)
    return EStepResult(
        label=action,
        lam=info["lam"],
        n_eff=info["n_eff"],
        nominal_cost=nominal_cost,
        sample_mean_cost=float(np.mean(sample_costs[finite])),
        expected_cost=float(np.sum(rollout["weights"][finite] * sample_costs[finite])),
        plan_cost=plan_cost,
        clip_fraction=float(np.mean((sampled <= low) | (sampled >= high))),
        tail_var_ratio=_tail_variance_ratio(sample_tails, sample_costs),
        lambda_saturated=_temperature_saturated(sample_costs, info["lam"], planner.lam_target_neff_fraction),
        numerical_valid=numerical_valid,
    )


def _tail_variance_ratio(
    tails: Float[np.ndarray, "samples"],
    totals: Float[np.ndarray, "samples"],
) -> float | None:
    """Tail/total population variance on MPPI's existing finite support."""
    finite = np.isfinite(tails) & np.isfinite(totals)
    if not finite.any():
        return None
    return float(np.var(tails[finite]) / (np.var(totals[finite]) + 1e-6))


def _temperature_saturated(
    scores: Float[np.ndarray, "samples"],
    temperature: float,
    target_fraction: float | None,
) -> bool | None:
    """Inspect the scalar solver's actual initial bracket and limit cases.

    Fixed-temperature planners have no bisection and return None. Checking
    endpoint ESS also detects boundary roots where final ESS meets target.
    """
    if target_fraction is None:
        return None
    finite = scores[np.isfinite(scores)]
    if not len(finite) or not np.isfinite(temperature) or temperature <= 0:
        return True
    spread = float(np.max(finite) - np.min(finite))
    if spread == 0.0:
        return True
    lower, upper = spread * 1e-8, spread * 1e8
    target = target_fraction * len(scores)
    low_ess = effective_sample_size(softmin_weights(scores, lower))
    high_ess = effective_sample_size(softmin_weights(scores, upper))
    return bool(low_ess >= target or high_ess <= target
                or temperature <= lower * (1.0 + 1e-7)
                or temperature >= upper * (1.0 - 1e-7))


def _raw_weights_finite(
    scores: Float[np.ndarray, "samples"],
    temperature: float,
) -> bool:
    """Check CPU MPPI's normalized weights before its legacy fallback.

    The planner may replace a numerically failed normalization with a finite
    one-hot vector. Reconstruct that inexpensive normalization from cached
    scores so the failure remains visible to training health checks.
    """
    finite = np.isfinite(scores)
    if not finite.any():
        return False
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        shifted = np.where(finite, scores - np.min(scores[finite]), np.inf)
        raw = np.exp(-shifted / temperature)
        normalized = raw / np.sum(raw)
    return bool(np.isfinite(normalized).all())
