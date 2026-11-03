"""Promoted full-sequence PI optimizer with frozen reconstruction scoring.

``FullPI`` is copied without algorithm changes from the archived SAC prototype.
The driver keeps its geometric noise schedule, incumbent sample and best-scored
nominal selection. Reconstruction changes the score, not the update or proposal.
No simulator or policy implementation is imported here.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np

from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.rollout_data import pre_action_states, pre_fall_action_mask

FINAL_SCALE = 0.1
PROTOTYPE_SEED_OFFSET = 20_000


class FullPI(MPPI):
    """MPPI's update on a fixed-start sequence: scaled noise, the current sequence kept as sample 0."""

    scale = 1.0

    def _sample_noise(self) -> np.ndarray:
        noise = super()._sample_noise() * self.scale
        noise[0] = 0.0
        return noise


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if int(value) < minimum:
        raise ValueError(name + " is below its minimum")
    return int(value)


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim or result.dtype.kind not in "fiu" or not result.size:
        raise ValueError(name + " has an invalid numeric shape")
    return result


def _same_bytes(first: np.ndarray, second: np.ndarray) -> bool:
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and first.tobytes() == second.tobytes()
    )


def _number(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(name + " must be a finite nonnegative scalar")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(name + " must be a finite nonnegative scalar")
    return result


class FrozenReconstructionCost:
    """Half summed torque residual on each candidate's executable prefix.

    ``actor.predict`` must be the caller's frozen deterministic physical action
    function, including its existing features/normalization/output bounds. It
    receives float32 observations. Residuals use the original float64 controls
    minus its predictions promoted to float64. The actor never sees a post-fall
    state. A falling candidate's prefix D is marked by ``last_complete`` and
    ``last_prefix_lengths``; it is not a fabricated full-length score.

    Alpha zero performs no policy evaluation and leaves ``last_D`` unknown.
    Native task/fall scoring and final teacher competence remain the caller's
    separately declared contracts. This class adds no feasibility filter.
    """

    def __init__(self, env: Any, actor: Any, initial_state: np.ndarray, alpha: float):
        self.env, self.actor = env, actor
        self._alpha = _number(alpha, "alpha")
        self.initial_state = _array(initial_state, "initial state", 1).copy()
        if len(self.initial_state) < 4 or not np.isfinite(self.initial_state).all():
            raise ValueError("A finite Walker-compatible initial state is required")
        self.initial_state.setflags(write=False)
        self.action_dim = _integer(env.action_dim, "action dimension", 1)
        if self.alpha and not callable(getattr(actor, "predict", None)):
            raise ValueError("Positive reconstruction requires actor.predict")
        self.last_D: np.ndarray | None = None
        self.last_matching: np.ndarray | None = None
        self.last_active_mask: np.ndarray | None = None
        self.last_complete: np.ndarray | None = None
        self.last_prefix_lengths: np.ndarray | None = None
        self.last_valid_candidates: np.ndarray | None = None
        self._counts = dict(
            calls=0, requested_actor_rows=0, completed_actor_rows=0, unknown_partial_actor_rows=0
        )

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def counts(self) -> dict[str, int]:
        return self._counts.copy()

    def __call__(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self.last_D = self.last_matching = self.last_active_mask = None
        self.last_complete = self.last_prefix_lengths = None
        self.last_valid_candidates = None
        post = _array(states, "post-action states", 3)
        controls = _array(actions, "candidate actions", 3)
        if (
            post.shape[:2] != controls.shape[:2]
            or controls.shape[-1] != self.action_dim
            or post.shape[-1] != len(self.initial_state)
        ):
            raise ValueError("Candidate state/action batch dimensions differ")
        healthy = np.asarray(self.env._is_healthy(post[..., 2], post[..., 3]))
        if healthy.shape != post.shape[:2] or healthy.dtype != np.bool_:
            raise ValueError("Health adapter must return bool[K,H]")
        mask = pre_fall_action_mask(healthy)
        pre = pre_action_states(self.initial_state, post)
        valid = np.all(
            (np.isfinite(post).all(axis=-1) & np.isfinite(controls).all(axis=-1)) | ~mask,
            axis=1,
        )
        executable = mask & valid[:, None]
        if not np.isfinite(pre[executable]).all():
            raise ValueError("Nonfinite executable pre-action state")
        low, high = self.env.action_bounds
        outside = ((controls < low) | (controls > high)) & np.isfinite(controls)
        if np.any(outside & mask[..., None]):
            raise ValueError("Candidate controls exceed physical bounds")
        self.last_active_mask = mask.copy()
        self.last_prefix_lengths = mask.sum(axis=1)
        self.last_valid_candidates = valid.copy()
        self.last_complete = healthy.all(axis=1) & valid
        self._counts["calls"] += 1
        if self.alpha == 0:
            self.last_matching = np.zeros(len(controls), dtype=float)
            return self.last_matching.copy()
        if not np.any(executable):
            self.last_D = np.full(len(controls), np.inf)
            self.last_matching = np.full(len(controls), np.inf)
            return self.last_matching.copy()
        observations = np.asarray(self.env.rollout_states_to_obs(pre[executable]), dtype=np.float32)
        if (
            observations.ndim != 2
            or len(observations) != int(executable.sum())
            or not np.isfinite(observations).all()
        ):
            raise ValueError("Invalid executable observation batch")
        rows = len(observations)
        self._counts["requested_actor_rows"] += rows
        try:
            prediction = np.asarray(self.actor.predict(observations))
            if prediction.shape != (rows, self.action_dim) or not np.isfinite(prediction).all():
                raise ValueError("Invalid actor prediction shape or values")
            if np.any(prediction < low) or np.any(prediction > high):
                raise ValueError("Actor predictions exceed physical bounds; do not clip silently")
        except BaseException:
            self._counts["unknown_partial_actor_rows"] += rows
            raise
        self._counts["completed_actor_rows"] += rows
        residual = controls[executable].astype(np.float64) - prediction.astype(np.float64)
        per_control = np.zeros(mask.shape, dtype=np.float64)
        per_control[executable] = 0.5 * np.sum(residual**2, axis=-1)
        self.last_D = np.sum(per_control, axis=1)
        self.last_D[~valid] = np.inf
        self.last_matching = self.alpha * self.last_D
        if not np.isfinite(self.last_matching[valid]).all():
            raise ValueError("Reconstruction score overflow")
        return self.last_matching.copy()


def _finite_scalar(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def optimize_full(
    env: Any,
    cfg: MPPIConfig,
    start: np.ndarray,
    warm: np.ndarray,
    initial: np.ndarray,
    seed: int,
    iterations: int,
    *,
    prior_cost: FrozenReconstructionCost | None = None,
    observer: Callable[[dict], None] | None = None,
) -> dict:
    """Reuse the prototype's fixed-start PI loop and retain by total objective.

    ``seed`` has the prototype's semantics: actual PCG seed is ``20000+seed``.
    There is exactly one private refinement pass per iteration, including the
    first; ``plan_step`` is never called and its initialization flag is untouched.
    The final weighted update is returned but remains unscored/unselected.

    Observer calls are synchronous. The before event precedes the native bank;
    the after event contains copied raw bank outputs and score components. The
    caller meters actual native work. Returned geometry counters cannot identify
    where a failed native call stopped and are subordinate to that native meter.
    """
    seed = _integer(seed, "seed")
    iterations = _integer(iterations, "iterations", 1)
    start = _array(start, "initial state", 1).copy()
    warm = _array(warm, "initial warmstart", 1).copy()
    init = _array(initial, "initial sequence", 2)
    if not np.isfinite(start).all() or not np.isfinite(warm).all() or not np.isfinite(init).all():
        raise ValueError("Initial context and sequence must be finite")
    if init.shape != (cfg.H, env.action_dim):
        raise ValueError("Initial sequence must have shape (H,action_dim)")
    if observer is not None and not callable(observer):
        raise ValueError("Observer must be callable")
    if prior_cost is not None and not isinstance(prior_cost, FrozenReconstructionCost):
        raise ValueError("A cached FrozenReconstructionCost is required")
    if prior_cost is not None and not _same_bytes(prior_cost.initial_state, start):
        raise ValueError("Reconstruction and native initial contexts differ")
    alpha = None if prior_cost is None else prior_cost.alpha
    planner = FullPI(env, cfg)
    planner.reset(seed=PROTOTYPE_SEED_OFFSET + seed)
    planner.U = np.clip(init.copy(), planner.action_low, planner.action_high)
    saved_start, saved_warm = start.copy(), warm.copy()
    best = planner.U.copy()
    best_C = best_D = best_states = None
    best_J, best_iteration = float("inf"), None
    curve = []
    work = dict(
        requested_passes=0,
        completed_passes=0,
        requested_model_controls=0,
        completed_model_controls=0,
        unknown_partial_model_controls=0,
    )
    for k in range(iterations):
        if prior_cost is not None and prior_cost.alpha != alpha:
            raise ValueError("Agreement weight changed during one solve")
        planner.scale = FINAL_SCALE ** (k / max(iterations - 1, 1))
        current = planner.U.copy()
        before = planner.snapshot()
        if observer is not None:
            observer(
                dict(
                    phase="before",
                    iteration=k,
                    scale=planner.scale,
                    planner_snapshot=deepcopy(before),
                    initial_state=start.copy(),
                    initial_warmstart=warm.copy(),
                    K=planner.K,
                    H=planner.H,
                    action_dim=planner.nu,
                    model_controls=planner.K * planner.H,
                    U=current.copy(),
                    planner_rng=deepcopy(before["rng_state"]),
                    initial_plan=before["initial_plan"],
                )
            )
        work["requested_passes"] += 1
        work["requested_model_controls"] += planner.K * planner.H
        try:
            _, info = planner._plan_iteration(start, prior_cost, None, warm)
        except BaseException as error:
            work["unknown_partial_model_controls"] += planner.K * planner.H
            error.trajectory_pi_work = deepcopy(work)
            raise
        work["completed_passes"] += 1
        work["completed_model_controls"] += planner.K * planner.H
        if not _same_bytes(saved_start, start) or not _same_bytes(saved_warm, warm):
            raise ValueError("Native planning mutated the fixed initial arrays")
        costs = np.asarray(planner._last_costs)
        matching = np.zeros_like(costs) if prior_cost is None else prior_cost.last_matching
        if costs.shape != (planner.K,) or matching is None or matching.shape != costs.shape:
            raise ValueError("Missing native/reconstruction bank scores")
        D = None if prior_cost is None or prior_cost.last_D is None else prior_cost.last_D.copy()
        scores = costs + matching
        if scores[0] < best_J:
            best, best_J, best_iteration = current, float(scores[0]), k
            best_C, best_D = float(costs[0]), None if D is None else float(D[0])
            best_states = planner._last_states[0].copy()
        population_index = int(np.argmin(scores))
        row = dict(
            iteration=k,
            scale=planner.scale,
            C=_finite_scalar(costs[0]),
            D=None if D is None else _finite_scalar(D[0]),
            J=_finite_scalar(scores[0]),
            best_J=_finite_scalar(best_J),
            best_iteration=best_iteration,
            population_best_index=population_index,
            population_best_C=_finite_scalar(costs[population_index]),
            population_best_D=None if D is None else _finite_scalar(D[population_index]),
            population_best_J=_finite_scalar(scores[population_index]),
            lam=float(info["lam"]),
            ess=float(info["n_eff"]),
            maximum_weight=float(np.max(planner._last_weights)),
            update_rms=float(np.sqrt(np.mean((planner.U - current) ** 2))),
        )
        curve.append(row)
        if observer is not None:
            event = dict(
                phase="after",
                **row,
                current_C=row["C"],
                current_D=row["D"],
                current_J=row["J"],
                current_actions=current.copy(),
                post_actions=planner.U.copy(),
                native_costs=costs.copy(),
                matching=matching.copy(),
                weights=planner._last_weights.copy(),
                states=planner._last_states.copy(),
                actions=planner._last_actions.copy(),
                active_mask=None if prior_cost is None else prior_cost.last_active_mask.copy(),
                valid_candidates=(
                    None if prior_cost is None else prior_cost.last_valid_candidates.copy()
                ),
                planner_after=planner.snapshot(),
                info=deepcopy(info),
                counts=deepcopy(work),
            )
            event.update(C=costs.copy(), D=D, J=scores.copy(), n_eff=float(info["n_eff"]))
            event["lambda"] = float(info["lam"])
            observer(event)
    return dict(
        status="COMPLETED" if best_iteration is not None else "NO_FINITE_SCORED_NOMINAL",
        best_actions=best.copy(),
        best_cost=best_C,
        best_D=best_D,
        best_J=_finite_scalar(best_J),
        best_iteration=best_iteration,
        best_states=best_states,
        curve=curve,
        final_actions=planner.U.copy(),
        final_update_scored=False,
        planner_snapshot=planner.snapshot(),
        planner_seed=PROTOTYPE_SEED_OFFSET + seed,
        counts=work,
        reconstruction_work=None if prior_cost is None else prior_cost.counts,
    )
