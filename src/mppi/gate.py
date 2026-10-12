"""Task-first Walker gate, evaluated in isolation from the MPPI controller.

This module scores suggestions; it never selects the action sent to the live
environment. S3's caller must always execute raw MPPI's action. Acceptance is
only a finite-horizon comparison, not an episode-level safety guarantee.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from src.envs.walker2d import Walker2d


class Policy(Protocol):
    def act(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray: ...


class InvalidBaselineError(RuntimeError):
    """The reference plan cannot be scored; stop instead of claiming fallback."""


def gate_accepts(
    baseline_cost: float,
    policy_cost: float | None,
    baseline_fell: bool,
    policy_fell: bool | None,
) -> bool:
    """The fixed tau=0 rule: no task-cost increase and no additional fall."""
    if not np.isfinite(baseline_cost):
        raise InvalidBaselineError("The baseline task cost must be finite.")
    return bool(
        policy_cost is not None
        and np.isfinite(policy_cost)
        and policy_fell is not None
        and policy_cost <= baseline_cost
        and (not policy_fell or baseline_fell)
    )


@dataclass(frozen=True)
class GateDecision:
    baseline_cost: float
    policy_cost: float | None
    baseline_fell: bool
    policy_fell: bool | None
    accepted: bool
    reason: str
    baseline_plan: np.ndarray
    candidate_plan: np.ndarray
    # Initial state followed by completed, valid closed-loop transitions.
    # The falling transition is included; no policy is queried afterward.
    candidate_states: np.ndarray
    candidate_prefix_cost: float

    @property
    def candidate_steps(self) -> int:
        return len(self.candidate_states) - 1


class TaskCostGate:
    """Compare an actual MPPI plan with one deterministic policy trajectory.

    The private CPU simulator starts both trajectories at the same full state
    and solver warmstart. It shares neither live MuJoCo data nor the planner's
    nominal sequence/RNG. Task costs are the unchanged Walker rollout costs,
    including its absorbing suffix after a fall, not SAC training rewards.
    """

    def __init__(self, task: dict, horizon: int, policy: Policy):
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 2:
            raise ValueError("horizon must be an integer >= 2.")
        if task.get("cost_style") != "target_velocity":
            raise ValueError("The S3 gate requires the frozen target_velocity task.")
        if task.get("use_warp", False):
            raise ValueError("The S3 gate requires CPU MuJoCo rollouts.")
        options = dict(task, nthread=1, use_warp=False)
        self._env = Walker2d(**options)
        self.horizon = int(horizon)
        self.policy = policy
        self._low, self._high = self._env.action_bounds
        self._closed = False

    def _valid_actions(self, actions: np.ndarray, shape: tuple[int, ...]) -> bool:
        return bool(actions.shape == shape and np.isfinite(actions).all()
                    and (actions >= self._low).all() and (actions <= self._high).all())

    def _prefix_status(self, states: np.ndarray) -> tuple[bool, bool]:
        """Return (fell, finite through first fall); ignore unused suffixes."""
        healthy = self._env._is_healthy(states[:, 2], states[:, 3])
        end = int(np.argmax(~healthy)) + 1 if not healthy.all() else len(states)
        return bool(not healthy.all()), bool(np.isfinite(states[:end]).all())

    def decide(self, state: np.ndarray, warmstart: np.ndarray,
               baseline_plan: np.ndarray) -> GateDecision:
        if self._closed:
            raise RuntimeError("The gate has been closed.")
        try:
            state = np.asarray(state, dtype=float).copy()
            warmstart = np.asarray(warmstart, dtype=float).copy()
            baseline_plan = np.asarray(baseline_plan, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise InvalidBaselineError("State, warmstart and plan must be numeric arrays.") from exc
        if state.shape != (self._env.state_dim,) or not np.isfinite(state).all():
            raise InvalidBaselineError("Invalid initial simulator state.")
        if warmstart.shape != (self._env.model.nv,) or not np.isfinite(warmstart).all():
            raise InvalidBaselineError("Invalid initial solver warmstart.")
        if not self._env._is_healthy(state[2], state[3]):
            raise InvalidBaselineError("The gate must not be called after an episode has fallen.")
        plan_shape = (self.horizon, self._env.action_dim)
        if not self._valid_actions(baseline_plan, plan_shape):
            raise InvalidBaselineError("Baseline plan has invalid shape, values or actuator bounds.")

        self._env.reset(state=state)
        self._env.set_warmstart(warmstart)
        candidate = np.zeros(plan_shape, dtype=float)
        prefix = [state.copy()]
        prefix_cost = 0.0
        invalid_reason = None
        for step in range(self.horizon):
            # The policy sees s_t, never the result of the action being chosen.
            obs = self._env.rollout_states_to_obs(prefix[-1])
            try:
                action = self.policy.act(obs, deterministic=True)
            except FloatingPointError:
                invalid_reason = "nonfinite_candidate"
                break
            try:
                action = np.asarray(action, dtype=float)
            except (TypeError, ValueError):
                invalid_reason = "invalid_policy_action"
                break
            if not self._valid_actions(action, (self._env.action_dim,)):
                invalid_reason = "invalid_policy_action"
                break
            candidate[step] = action
            try:
                _, cost, done, _ = self._env.step(action)
            except FloatingPointError:
                invalid_reason = "nonfinite_candidate"
                break
            following = self._env.get_state()
            if not np.isfinite(following).all() or not np.isfinite(cost):
                invalid_reason = "nonfinite_candidate"
                break
            prefix.append(following)
            prefix_cost += float(cost)
            if done:
                break

        # Score the actual optimized reference, not its best sampled rollout
        # or a weighted average of sample costs. Candidate suffix zeros are
        # immaterial after a fall because Walker.rollout_cost ignores them.
        plans = np.stack((baseline_plan, candidate))
        states, costs, _ = self._env.batch_rollout(
            state, plans, initial_warmstart=warmstart)
        baseline_fell, baseline_finite = self._prefix_status(states[0])
        baseline_cost = float(costs[0])
        if not baseline_finite or not np.isfinite(baseline_cost):
            raise InvalidBaselineError("The baseline rollout has a non-finite executable prefix or cost.")

        candidate_states = np.stack(prefix)
        policy_cost, policy_fell = None, None
        if invalid_reason is None:
            fell, finite = self._prefix_status(states[1])
            if not finite or not np.isfinite(costs[1]):
                invalid_reason = "nonfinite_candidate"
            elif not np.array_equal(states[1, :len(prefix) - 1], candidate_states[1:]):
                invalid_reason = "candidate_replay_mismatch"
            else:
                policy_cost, policy_fell = float(costs[1]), fell

        accepted = gate_accepts(baseline_cost, policy_cost, baseline_fell, policy_fell)
        if invalid_reason is not None:
            reason = invalid_reason
        elif policy_fell and not baseline_fell:
            reason = "predicted_fall"
        else:
            reason = "accepted" if accepted else "task_cost"
        return GateDecision(
            baseline_cost=baseline_cost, policy_cost=policy_cost,
            baseline_fell=baseline_fell, policy_fell=policy_fell,
            accepted=accepted, reason=reason, baseline_plan=baseline_plan,
            candidate_plan=candidate, candidate_states=candidate_states,
            candidate_prefix_cost=prefix_cost,
        )

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
