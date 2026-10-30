"""Training hooks that reuse GPS arrays and policy inference without simulation."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.gps.health_monitor import EarlyAbortException, TrainingHealthMonitor
from src.gps.telemetry import (HoldoutRing, HoldoutSnapshot, MetricValues, StepLogger,
                               TelemetryMetrics, diagnostics_from_arrays, holdout_metrics,
                               spatial_lipschitz)

ActionFunction = Callable[[np.ndarray], np.ndarray]


def deterministic_probe_mask(size: int, probability: float, seed: int, iteration: int) -> np.ndarray:
    """Select ceil(probability*M) state indices [M] without touching collection RNG.

    The subset is deterministic for the training seed, iteration and batch size.
    Positive fractions retain one probe in small smoke batches; zero disables it.
    """
    if size < 0 or not 0 <= probability <= 1 or seed < 0 or iteration < 0:
        raise ValueError("Require nonnegative size/seed/iteration and probability in [0,1].")
    mask = np.zeros(size, dtype=bool)
    count = min(size, int(np.ceil(probability * size)))
    if count:
        rng = np.random.default_rng(np.random.SeedSequence([seed, iteration, 0x50524F42]))
        mask[rng.choice(size, size=count, replace=False)] = True
    return mask


@dataclass(frozen=True)
class ResearchConfig:
    """Scalar writers and compute-saving health checks for a future run."""

    enabled: bool = True
    enable_early_abort: bool = True
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    tensorboard: bool = False


@dataclass
class IterationDiagnostics:
    """Pre-M-step scalars and at most 256 historical policy observations."""

    values: MetricValues
    holdout: HoldoutSnapshot | None
    holdout_pre: np.ndarray | None
    observations: np.ndarray  # [M, observation]
    targets: np.ndarray  # [M, action]
    eligible: np.ndarray  # [M] positive-weight rows


@dataclass(frozen=True)
class IterationBaseline:
    """Existing collection/trust summaries and per-state nominal costs [M]."""

    mean_ep_length: float
    fall_rate: float
    nominal_costs: np.ndarray
    sigma: float
    mean_n_eff: float
    median_lambda: float | None


class ResearchSession:
    """Keep logging/health/holdout bookkeeping outside the training algorithm."""

    def __init__(self, out_dir: Path, config: ResearchConfig, resume_at: int = 0) -> None:
        self.config = config
        path = out_dir / "telemetry.jsonl"
        prior: list[TelemetryMetrics] = []
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            keep = [row for row in rows if row["iteration"] < resume_at]
            orphaned = [row for row in rows if row["iteration"] >= resume_at]
            if orphaned:
                backup = out_dir / f"telemetry_orphaned_at_resume_{resume_at}.jsonl"
                with backup.open("a") as stream:
                    stream.writelines(json.dumps(row, allow_nan=False) + "\n" for row in orphaned)
                path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in keep))
                if not keep:
                    (out_dir / "telemetry.csv").unlink(missing_ok=True)
            prior = [TelemetryMetrics(row["iteration"], {key: value for key, value in row.items()
                                                        if key != "iteration"}) for row in keep]
        self.monitor = TrainingHealthMonitor(enabled=False, report_path=out_dir / "early_abort_report.json")
        for metrics in prior[-20:]:
            self.monitor.observe(metrics)
        self.monitor.enabled = config.enable_early_abort
        self.holdouts = HoldoutRing(delay=5, max_states=256)
        # Recover holdout snapshots when label archives exist; otherwise report
        # unavailable drift until five new iterations have accumulated.
        for k in range(max(0, resume_at - 5), resume_at):
            archive = out_dir / f"labels_{k:03d}.npz"
            if archive.exists():
                with np.load(archive) as data:
                    eligible = data["weights"] > 0 if "weights" in data else np.ones(len(data["obs"]), bool)
                    self.holdouts.add(k, data["obs"][eligible], data["labels"][eligible])
        # Open external writers only after pure resume recovery has succeeded.
        self.logger = (StepLogger(out_dir, wandb_project=config.wandb_project,
                                  wandb_run_name=config.wandb_run_name,
                                  tensorboard=config.tensorboard) if config.enabled else None)

    def check_numerics(self, iteration: int, *arrays: np.ndarray, valid: bool = True) -> None:
        """Abort before regression if existing actions/weights contain NaN or Inf."""
        if not valid or any(not np.isfinite(value).all() for value in arrays):
            self.monitor.observe(TelemetryMetrics(iteration, {"numerical/actions_weights_finite": False}),
                                 numerical_valid=False)
            raise FloatingPointError("Nonfinite actions or weights; regression cannot continue.")

    def begin(self, iteration: int, *, observations: np.ndarray, targets: np.ndarray,
              old_actions: np.ndarray, weights: np.ndarray, trajectory_ids: np.ndarray,
              plan_cost: np.ndarray, expected_cost: np.ndarray,
              tail_var_ratio: np.ndarray | None, lambda_saturated: np.ndarray | None,
              action_delta: np.ndarray, baseline: IterationBaseline, act: ActionFunction
              ) -> IterationDiagnostics:
        """Summarize arrays [M,...] and predict on k-5 states before optimization."""
        eligible = weights > 0
        saturation = None if lambda_saturated is None else float(np.mean(lambda_saturated))
        values: MetricValues = {
            "collect/mean_ep_length": baseline.mean_ep_length,
            "collect/fall_rate": baseline.fall_rate,
            "e_step/share_plans_worse": float(np.mean(plan_cost > baseline.nominal_costs)),
            "e_step/median_action_delta": float(np.median(action_delta)),
            "e_step/sigma_k": baseline.sigma,
            "e_step/mean_n_eff": baseline.mean_n_eff,
            "e_step/median_lambda": baseline.median_lambda,
            "m_step/positive_weight_fraction": float(np.mean(eligible)) if len(eligible) else 0.,
            "m_step/holdout_drift": None, "m_step/holdout_mse_before": None,
            "m_step/holdout_mse_after": None, "m_step/holdout_mse_delta": None,
            "m_step/holdout_states": 0,
        }
        values.update(diagnostics_from_arrays(plan_cost, expected_cost, tail_var_ratio=tail_var_ratio,
                                              lambda_saturation_rate=saturation))
        values.update(spatial_lipschitz(observations, targets, old_actions, trajectory_ids, eligible=eligible))
        holdout = self.holdouts.get(iteration)
        pre = act(holdout.observations) if holdout is not None and len(holdout.observations) else None
        return IterationDiagnostics(values, holdout, pre, observations, targets, eligible)

    def finish(self, iteration: int, diagnostics: IterationDiagnostics, *, act: ActionFunction,
               probe: Mapping[str, object] | None, fit_loss: float,
               extra: MetricValues | None = None, defer_log: bool = False) -> TelemetryMetrics:
        """Log after fitting; detect health failures only after preserving the row."""
        values = dict(diagnostics.values)
        predicted = None if probe is None else float(probe["predicted_improvement"])
        realized = None if probe is None else float(probe["actual_improvement"])
        denominator = None if predicted is None else predicted + 1e-6
        values.update({"probe/predicted_improvement": predicted,
                       "probe/realized_improvement": realized,
                       "probe/improvement_realization_ratio": (
                           None if denominator is None or denominator == 0 or realized is None else realized / denominator),
                       "probe/states": 0 if probe is None else int(probe["probes"]),
                       "m_step/weighted_loss": fit_loss})
        snapshot = diagnostics.holdout
        if snapshot is not None and diagnostics.holdout_pre is not None:
            post = act(snapshot.observations)
            values.update(holdout_metrics(diagnostics.holdout_pre, post, snapshot.targets))
            values["m_step/holdout_source_iteration"] = snapshot.iteration
        else:
            values["m_step/holdout_source_iteration"] = None
        self.holdouts.add(iteration, diagnostics.observations[diagnostics.eligible],
                          diagnostics.targets[diagnostics.eligible])
        if extra:
            values.update(extra)
        metrics = TelemetryMetrics(iteration, values)
        if not defer_log and self.logger is not None:
            self.logger.log(metrics)
        try:
            self.monitor.observe(metrics, numerical_valid=bool(np.isfinite(fit_loss)))
        except EarlyAbortException as exc:
            if defer_log and self.logger is not None:
                try:
                    self.logger.log(metrics)
                except Exception as logging_error:
                    exc.report["telemetry_write_error"] = repr(logging_error)
            raise
        return metrics

    def log(self, metrics: TelemetryMetrics, extra: MetricValues | None = None) -> None:
        """Add separately timed ladder scalars after the health check has passed."""
        combined = TelemetryMetrics(metrics.iteration, {**metrics.values, **(extra or {})})
        if self.logger is not None:
            self.logger.log(combined)

    def close(self) -> None:
        """Close optional scalar backends after normal completion or an abort."""
        if self.logger is not None:
            self.logger.close()

    def __enter__(self) -> ResearchSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        try:
            self.close()
        except Exception as logging_error:
            if exc is None:
                raise
            if isinstance(exc, EarlyAbortException):
                exc.report["writer_close_error"] = repr(logging_error)
            elif isinstance(exc, BaseException):
                exc.add_note(f"Research writer cleanup also failed: {logging_error!r}")
