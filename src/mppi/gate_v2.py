"""S3 v2: compare one action followed by an equally scored MPPI continuation.

This is a diagnostic scorer, not an execution wrapper or safety certificate.
The planner still optimizes its frozen H-step horizon; the gate scores only
the first H-1 continuation actions, so each option covers exactly H controls.
Future repeated replanning remains approximated by one optimized open-loop
plan. No learned value, planner recentering, or policy rollout is introduced.
"""

import copy
from dataclasses import dataclass

import numpy as np

from src.envs.walker2d import Walker2d
from src.mppi.gate import InvalidBaselineError, gate_accepts
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


@dataclass(frozen=True)
class ContinuationScore:
    cost: float
    fell: bool
    action: np.ndarray
    immediate_cost: float
    immediate_fell: bool
    successor_state: np.ndarray
    successor_warmstart: np.ndarray
    continuation_cost: float
    # None/empty arrays after immediate termination: MPPI was not called.
    optimized_plan: np.ndarray | None
    scored_plan: np.ndarray
    continuation_states: np.ndarray
    planner_snapshot_after: dict | None


@dataclass(frozen=True)
class ContinuationDecision:
    baseline_score: ContinuationScore
    candidate_score: ContinuationScore | None
    accepted: bool
    reason: str

    @property
    def baseline_cost(self) -> float:
        return self.baseline_score.cost

    @property
    def policy_cost(self) -> float | None:
        return None if self.candidate_score is None else self.candidate_score.cost

    @property
    def baseline_fell(self) -> bool:
        return self.baseline_score.fell

    @property
    def policy_fell(self) -> bool | None:
        return None if self.candidate_score is None else self.candidate_score.fell


class ContinuationGate:
    """Score isolated paired continuations from the original post-call memory.

``post_snapshot`` must be taken after the live MPPI planning call, when its
nominal has already shifted and its random stream has advanced. Both scratch
branches restore that same snapshot; only the first action and consequently
the reached physical state/warmstart differ. The caller keeps ownership of
its live environment and planner, neither of which is accessed here.
    """

    def __init__(self, task: dict, cfg: MPPIConfig):
        if task.get("cost_style") != "target_velocity":
            raise ValueError("The continuation gate requires the target_velocity task.")
        if task.get("use_warp", False):
            raise ValueError("The continuation gate requires CPU MuJoCo rollouts.")
        if type(cfg.H) is not int or cfg.H < 2:
            raise ValueError("The continuation horizon must be an integer >= 2.")
        self.cfg = copy.deepcopy(cfg)
        self.horizon = cfg.H
        self._env = Walker2d(**dict(task, nthread=1, use_warp=False))
        try:
            self._planner = MPPI(self._env, self.cfg, seed=0)
        except Exception:
            self._env.close()
            raise
        self._low, self._high = self._env.action_bounds
        self._closed = False

    def _valid_actions(self, actions: np.ndarray, shape: tuple[int, ...]) -> bool:
        return bool(actions.shape == shape and np.isfinite(actions).all()
                    and (actions >= self._low).all() and (actions <= self._high).all())

    def _validated_start(self, state, warmstart, snapshot, baseline_action):
        try:
            state = np.asarray(state, dtype=float).copy()
            warmstart = np.asarray(warmstart, dtype=float).copy()
            baseline_action = np.asarray(baseline_action, dtype=float).copy()
        except (TypeError, ValueError) as exc:
            raise InvalidBaselineError("State, warmstart and baseline action must be numeric.") from exc
        if state.shape != (self._env.state_dim,) or not np.isfinite(state).all():
            raise InvalidBaselineError("Invalid initial simulator state.")
        if warmstart.shape != (self._env.model.nv,) or not np.isfinite(warmstart).all():
            raise InvalidBaselineError("Invalid initial solver warmstart.")
        if not self._env._is_healthy(state[2], state[3]):
            raise InvalidBaselineError("The gate must not be called after a fall.")
        if not self._valid_actions(baseline_action, (self._env.action_dim,)):
            raise InvalidBaselineError("Invalid baseline action shape, values or actuator bounds.")
        try:
            snapshot = copy.deepcopy(snapshot)
            nominal = np.asarray(snapshot["U"], dtype=float)
            if not self._valid_actions(nominal, (self.horizon, self._env.action_dim)):
                raise ValueError("Invalid snapshot nominal.")
            initial = snapshot["initial_plan"]
            if not isinstance(initial, (bool, np.bool_)) or initial:
                raise ValueError("A post-planning snapshot must have initial_plan=False.")
            # Validate before simulation, even if the first action would fall.
            # This local seeded generator consumes no global random numbers.
            rng = np.random.default_rng(0)
            rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise InvalidBaselineError("Invalid post-planning snapshot.") from exc
        snapshot["U"] = nominal.copy()
        return state, warmstart, snapshot, baseline_action

    def _score_action(self, state, warmstart, post_snapshot, action) -> ContinuationScore:
        self._env.reset(state=state)
        self._env.set_warmstart(warmstart)
        self._planner.restore(post_snapshot)
        _, immediate, done, _ = self._env.step(action)
        successor = self._env.get_state()
        successor_warmstart = self._env.get_warmstart()
        if (not np.isfinite(immediate) or not np.isfinite(successor).all()
                or not np.isfinite(successor_warmstart).all()):
            raise FloatingPointError("Non-finite first-action transition.")
        if done:
            # Live step already charged the fall penalty exactly once.
            continuation = float((self.horizon - 1) * self._env._w_unhealthy)
            plan, after = None, None
            scored_plan = np.empty((0, self._env.action_dim))
            states = np.empty((0, self._env.state_dim))
            fell = True
        else:
            self._planner.plan_step(successor, initial_warmstart=successor_warmstart)
            plan = self._planner.last_plan
            if plan is None or not self._valid_actions(plan, (self.horizon, self._env.action_dim)):
                raise FloatingPointError("Non-finite or infeasible optimized continuation.")
            after = self._planner.snapshot()
            scored_plan = plan[:self.horizon - 1].copy()
            trajectories, costs, _ = self._env.batch_rollout(
                successor, scored_plan[None], initial_warmstart=successor_warmstart)
            states = trajectories[0].copy()
            healthy = self._env._is_healthy(states[:, 2], states[:, 3])
            fell = bool(not healthy.all())
            end = int(np.argmax(~healthy)) + 1 if fell else len(states)
            continuation = float(costs[0])
            if not np.isfinite(states[:end]).all() or not np.isfinite(continuation):
                raise FloatingPointError("Non-finite optimized continuation prefix or score.")
        total = float(immediate + continuation)
        if not np.isfinite(total):
            raise FloatingPointError("Non-finite total continuation score.")
        return ContinuationScore(
            cost=total, fell=fell, action=action.copy(), immediate_cost=float(immediate),
            immediate_fell=bool(done), successor_state=successor.copy(),
            successor_warmstart=successor_warmstart.copy(), continuation_cost=continuation,
            optimized_plan=plan, scored_plan=scored_plan, continuation_states=states,
            planner_snapshot_after=after,
        )

    def decide(self, state: np.ndarray, warmstart: np.ndarray, post_snapshot: dict,
               baseline_action: np.ndarray, candidate_action: np.ndarray) -> ContinuationDecision:
        if self._closed:
            raise RuntimeError("The continuation gate has been closed.")
        state, warmstart, post_snapshot, baseline_action = self._validated_start(
            state, warmstart, post_snapshot, baseline_action)
        try:
            baseline = self._score_action(state, warmstart, post_snapshot, baseline_action)
        except FloatingPointError as exc:
            raise InvalidBaselineError("The baseline continuation cannot be scored.") from exc
        try:
            candidate_action = np.asarray(candidate_action, dtype=float).copy()
        except (TypeError, ValueError):
            return ContinuationDecision(baseline, None, False, "invalid_policy_action")
        if not self._valid_actions(candidate_action, (self._env.action_dim,)):
            return ContinuationDecision(baseline, None, False, "invalid_policy_action")
        try:
            candidate = self._score_action(state, warmstart, post_snapshot, candidate_action)
        except FloatingPointError:
            return ContinuationDecision(baseline, None, False, "nonfinite_candidate")
        accepted = gate_accepts(baseline.cost, candidate.cost, baseline.fell, candidate.fell)
        if candidate.fell and not baseline.fell:
            reason = "predicted_fall"
        else:
            reason = "accepted" if accepted else "task_cost"
        return ContinuationDecision(baseline, candidate, accepted, reason)

    def close(self) -> None:
        if not self._closed:
            self._env.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
