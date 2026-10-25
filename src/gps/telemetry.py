"""Research diagnostics from existing GPS arrays, with optional scalar writers.

All helpers are inference/array operations: none constructs a simulator or
performs a rollout. Missing diagnostics are represented by ``None`` in logs.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

Scalar: TypeAlias = float | int | bool | None
MetricValues: TypeAlias = dict[str, Scalar]
FloatArray: TypeAlias = NDArray[np.float64]


def _scalar(value: Any) -> Scalar:
    """Convert NumPy scalars and replace unavailable nonfinite values by None."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    raise TypeError(f"Telemetry values must be numeric scalars or None, got {type(value).__name__}.")


@dataclass(frozen=True)
class TelemetryMetrics:
    """Flat scalar ``values`` for one nonnegative outer ``iteration``."""

    iteration: int
    values: MetricValues

    def __post_init__(self) -> None:
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, int) or self.iteration < 0:
            raise ValueError("iteration must be a nonnegative integer.")
        if any(not isinstance(key, str) or not key or key == "iteration" for key in self.values):
            raise ValueError("Metric names must be nonempty strings distinct from 'iteration'.")
        object.__setattr__(self, "values", {key: _scalar(value) for key, value in self.values.items()})

    def as_dict(self) -> MetricValues:
        """Return a standalone JSON-safe flat row, including its iteration."""
        return {"iteration": self.iteration, **self.values}


class StepLogger:
    """Write console, JSONL and evolving-column CSV telemetry every iteration.

    Optional backends are imported only when configured. Existing JSONL rows
    are retained when resuming; subsequent iteration numbers must increase.
    CSV columns expand when periodic diagnostics first become available.
    """

    def __init__(self, out_dir: Path, *, console: bool = True,
                 wandb_project: str | None = None, wandb_run_name: str | None = None,
                 tensorboard: bool = False) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "telemetry.jsonl"
        self.csv_path = self.out_dir / "telemetry.csv"
        self.console = console
        self._rows: list[MetricValues] = []
        self._columns: list[str] = ["iteration"]
        self._wandb: Any = None
        self._tensorboard: Any = None
        self._closed = False
        if self.jsonl_path.exists():
            for line in self.jsonl_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    iteration = row.pop("iteration")
                    self._rows.append(TelemetryMetrics(iteration, row).as_dict())
            for row in self._rows:
                self._columns.extend(key for key in row if key not in self._columns)
            if self._rows:
                self._write_csv()
        try:
            if wandb_project is not None:
                wandb = importlib.import_module("wandb")
                self._wandb = wandb.init(project=wandb_project, name=wandb_run_name, dir=str(self.out_dir))
            if tensorboard:
                module = importlib.import_module("torch.utils.tensorboard")
                self._tensorboard = module.SummaryWriter(log_dir=str(self.out_dir / "tensorboard"))
        except BaseException as error:
            # Construction can fail before an enclosing context manager owns
            # this logger. Release any backend that already started, while
            # preserving the original initialization failure.
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"Optional telemetry cleanup also failed: {cleanup_error}")
            raise

    def _write_csv(self) -> None:
        """Rewrite the small scalar table when its column schema expands."""
        temporary = self.csv_path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(self._rows)
        temporary.replace(self.csv_path)

    def log(self, metrics: TelemetryMetrics) -> None:
        """Append one finite/nullable scalar row to every configured backend."""
        if self._closed:
            raise RuntimeError("Cannot log after StepLogger.close().")
        if self._rows and metrics.iteration <= int(self._rows[-1]["iteration"]):
            raise ValueError("Logged iterations must increase strictly.")
        row = metrics.as_dict()
        new_columns = [key for key in row if key not in self._columns]
        self._columns.extend(new_columns)
        self._rows.append(row)
        with self.jsonl_path.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if new_columns or not self.csv_path.exists():
            self._write_csv()
        else:
            with self.csv_path.open("a", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self._columns).writerow(row)
        available = {key: value for key, value in metrics.values.items() if value is not None}
        if self._wandb is not None:
            self._wandb.log(available, step=metrics.iteration)
        if self._tensorboard is not None:
            for key, value in available.items():
                self._tensorboard.add_scalar(key, value, metrics.iteration)
            self._tensorboard.flush()
        if self.console:
            values = " ".join(f"{key}={value:.6g}" if isinstance(value, float)
                              else f"{key}={value}" for key, value in metrics.values.items())
            print(f"[telemetry k={metrics.iteration}] {values}", flush=True)

    def close(self) -> None:
        """Flush and close optional writers; repeated calls are harmless."""
        if not self._closed:
            try:
                try:
                    if self._tensorboard is not None:
                        self._tensorboard.close()
                finally:
                    if self._wandb is not None:
                        self._wandb.finish()
            finally:
                self._closed = True


def _matrix(value: np.ndarray, name: str) -> FloatArray:
    """Validate a finite [rows, features] array with a nonempty feature axis."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array [rows, features] with features > 0.")
    return array


def _median(value: float | np.ndarray | None) -> float | None:
    """Summarize an available finite scalar or per-state vector."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return float(np.median(array)) if array.size and np.isfinite(array).all() else None


def diagnostics_from_arrays(plan_cost: np.ndarray, expected_cost: np.ndarray, *,
                            tail_var_ratio: float | np.ndarray | None = None,
                            lambda_saturation_rate: float | None = None) -> MetricValues:
    """Summarize existing plan/weighted sample costs [M], without a rollout.

    ``tail_var_ratio`` may already be a scalar median or a [M] vector from the
    E-step. The saturation rate must come from the solver's actual boundary
    masks; it cannot be inferred reliably from an arbitrary lambda threshold.
    """
    plan, expected = np.asarray(plan_cost, dtype=np.float64), np.asarray(expected_cost, dtype=np.float64)
    if plan.ndim != 1 or expected.shape != plan.shape:
        raise ValueError("plan_cost and expected_cost must have matching shape [M].")
    gap = plan - expected
    return {"e_step/jensen_gap_median": _median(gap),
            "e_step/tail_var_ratio": _median(tail_var_ratio),
            "e_step/lambda_saturation_rate": _scalar(lambda_saturation_rate)}


def spatial_lipschitz(observations: np.ndarray, targets: np.ndarray, old_actions: np.ndarray,
                     trajectory_ids: np.ndarray, *, eligible: np.ndarray | None = None,
                     max_states: int = 256) -> MetricValues:
    """Compare target/policy local slopes using at most 256 existing states.

    Inputs are [M, O], [M, A], [M, A] and trajectory IDs [M]. ``eligible[M]``
    can exclude zero-weight labels. Sampling is deterministic and evenly
    spaced across eligible rows. Neighbors exclude *every* same-trajectory
    pair, a stronger mask than only removing temporal neighbors within ±3
    steps. No qualifying cross-trajectory pairs produces nullable slopes.
    A zero policy slope gives ratio 0 when the target slope is also 0, or
    None when a nonzero target slope makes the ratio mathematically undefined.
    """
    obs, target, old = (_matrix(observations, "observations"), _matrix(targets, "targets"),
                        _matrix(old_actions, "old_actions"))
    ids = np.asarray(trajectory_ids)
    if target.shape != old.shape or len(target) != len(obs) or ids.shape != (len(obs),):
        raise ValueError("Targets/old actions must match [M, A], and trajectory_ids must have shape [M].")
    if np.issubdtype(ids.dtype, np.number) and not np.isfinite(ids).all():
        raise ValueError("trajectory_ids must be finite.")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or not 1 <= max_states <= 256:
        raise ValueError("max_states must be an integer in [1, 256].")
    valid = np.ones(len(obs), dtype=bool) if eligible is None else np.asarray(eligible)
    if valid.shape != (len(obs),) or valid.dtype != np.bool_:
        raise ValueError("eligible must be a boolean mask with shape [M].")
    candidates = np.flatnonzero(valid)
    indices = candidates[np.linspace(0, len(candidates) - 1, min(max_states, len(candidates)), dtype=int)]
    if len(indices) < 2:
        return {"channel/L_target": None, "channel/L_policy": None,
                "channel/lipschitz_ratio": None, "channel/lipschitz_pairs": 0}
    obs, target, old, ids = obs[indices], target[indices], old[indices], ids[indices]
    distance = np.linalg.norm(obs[:, None] - obs[None, :], axis=2)
    distance[ids[:, None] == ids[None, :]] = np.inf
    qualifying = np.isfinite(distance).any(axis=1)
    rows = np.flatnonzero(qualifying)
    if not len(rows):
        return {"channel/L_target": None, "channel/L_policy": None,
                "channel/lipschitz_ratio": None, "channel/lipschitz_pairs": 0}
    neighbors = distance[rows].argmin(axis=1)
    denominator = distance[rows, neighbors] + 1e-4
    target_slope = float(np.median(np.linalg.norm(target[rows] - target[neighbors], axis=1) / denominator))
    policy_slope = float(np.median(np.linalg.norm(old[rows] - old[neighbors], axis=1) / denominator))
    ratio = target_slope / policy_slope if policy_slope else (0.0 if not target_slope else None)
    return {"channel/L_target": _scalar(target_slope), "channel/L_policy": _scalar(policy_slope),
            "channel/lipschitz_ratio": _scalar(ratio), "channel/lipschitz_pairs": len(rows)}


@dataclass(frozen=True)
class HoldoutSnapshot:
    """Copied, read-only observations [Q,O] and targets [Q,A] from one iteration."""

    iteration: int
    observations: FloatArray
    targets: FloatArray


class HoldoutRing:
    """Keep at most delay+1 small snapshots; retrieve exactly iteration-delay."""

    def __init__(self, delay: int = 5, max_states: int = 256) -> None:
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1:
            raise ValueError("delay must be a positive integer.")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or not 1 <= max_states <= 256:
            raise ValueError("max_states must be an integer in [1, 256].")
        self.delay, self.max_states = delay, max_states
        self._snapshots: dict[int, HoldoutSnapshot] = {}
        self._latest = -1

    def add(self, iteration: int, observations: np.ndarray, targets: np.ndarray) -> None:
        """Freeze an evenly sampled set of up to max_states existing labels."""
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration <= self._latest:
            raise ValueError("Holdout iterations must be nonnegative and increase strictly.")
        obs, target = _matrix(observations, "observations"), _matrix(targets, "targets")
        if len(obs) != len(target):
            raise ValueError("Holdout observations and targets must have the same number of rows.")
        indices = np.linspace(0, len(obs) - 1, min(self.max_states, len(obs)), dtype=int)
        obs, target = obs[indices].copy(), target[indices].copy()
        obs.setflags(write=False)
        target.setflags(write=False)
        self._snapshots[iteration] = HoldoutSnapshot(iteration, obs, target)
        self._snapshots = {key: value for key, value in self._snapshots.items() if key >= iteration - self.delay}
        self._latest = iteration

    def get(self, iteration: int) -> HoldoutSnapshot | None:
        """Return exactly k-delay, never a different available historical batch."""
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("iteration must be a nonnegative integer.")
        return self._snapshots.get(iteration - self.delay)


def holdout_metrics(pre_actions: np.ndarray, post_actions: np.ndarray, targets: np.ndarray) -> MetricValues:
    """Compare inference [Q,A] before/after M-step against frozen [Q,A] targets.

    Drift is mean squared Euclidean action distance. Validation MSE averages
    squared error over both examples and action components. No rollout occurs.
    """
    pre, post, target = (_matrix(pre_actions, "pre_actions"), _matrix(post_actions, "post_actions"),
                         _matrix(targets, "targets"))
    if pre.shape != post.shape or pre.shape != target.shape:
        raise ValueError("Holdout predictions and targets must share shape [Q, A].")
    if not len(pre):
        return {"m_step/holdout_drift": None, "m_step/holdout_mse_before": None,
                "m_step/holdout_mse_after": None, "m_step/holdout_mse_delta": None,
                "m_step/holdout_states": 0}
    before, after = float(np.mean((pre - target) ** 2)), float(np.mean((post - target) ** 2))
    return {"m_step/holdout_drift": float(np.mean(np.sum((post - pre) ** 2, axis=1))),
            "m_step/holdout_mse_before": before, "m_step/holdout_mse_after": after,
            "m_step/holdout_mse_delta": after - before, "m_step/holdout_states": len(pre)}
