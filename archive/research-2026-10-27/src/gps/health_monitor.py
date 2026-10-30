"""Bounded, coverage-aware early stopping for future MPPI-GPS training runs."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.gps.telemetry import TelemetryMetrics


class EarlyAbortException(RuntimeError):
    """Clean training-stop signal containing a JSON-safe diagnostic report."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.report = report
        self.reason_code: str = report["reason_code"]
        self.iteration: int = report["iteration"]
        super().__init__(report["diagnosis"])


class TrainingHealthMonitor:
    """Observe scalar telemetry without adding simulator calls.

    Missing probes never count as failed improvement. Iteration gaps reset
    streaks; duplicate or reversed iteration numbers raise ValueError. At k=12,
    bootstrap checks means over fully covered contiguous k=10,11,12. Chronic
    disconnect requires five contiguous paired probes and can trigger at k>=15.
    The latest 20 rows are retained for post-mortem reporting.
    """

    def __init__(self, enabled: bool = True, report_path: Path | None = None,
                 history_size: int = 20) -> None:
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 5:
            raise ValueError("history_size must be an integer >= 5.")
        self.enabled = enabled
        self.report_path = None if report_path is None else Path(report_path)
        self._history: deque[TelemetryMetrics] = deque(maxlen=history_size)
        self._last_iteration: int | None = None
        self._negative_streak = 0
        self._saturation_streak = 0
        self._seen = 0
        self._paired_probes = 0

    @property
    def coverage(self) -> dict[str, int]:
        """Expose actual probe coverage and streaks, never inferred failures."""
        return {"iterations_observed": self._seen, "paired_probe_iterations": self._paired_probes,
                "history_rows": len(self._history), "negative_realization_streak": self._negative_streak,
                "lambda_saturation_streak": self._saturation_streak}

    def _abort(self, metrics: TelemetryMetrics, reason: str, diagnosis: str,
               details: dict[str, Any] | None = None) -> None:
        """Persist the bounded report before raising a clean early-stop signal."""
        report: dict[str, Any] = {
            "iteration": metrics.iteration, "reason_code": reason, "diagnosis": diagnosis,
            "created_utc": datetime.now(timezone.utc).isoformat(), "coverage": self.coverage,
            "details": {} if details is None else details,
            "history": [item.as_dict() for item in self._history],
            "thresholds": {"bootstrap_iteration": 12, "bootstrap_window": 3, "bootstrap_length": 60,
                           "disconnect_first_iteration": 15, "disconnect_consecutive": 5,
                           "lambda_saturation_rate": .70, "lambda_saturation_consecutive": 3},
        }
        if self.report_path is not None:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.report_path.with_suffix(self.report_path.suffix + ".tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.report_path)
        raise EarlyAbortException(report)

    def observe(self, metrics: TelemetryMetrics, numerical_valid: bool = True) -> None:
        """Check a completed iteration; caller validates existing actions/weights.

        ``numerical_valid`` must be False if any E-step/sample/selected action or
        weight is NaN/Inf. This check is immediate and independent of probe
        coverage. Disabling the monitor records history but raises no aborts.
        """
        k = metrics.iteration
        if self._last_iteration is not None and k <= self._last_iteration:
            raise ValueError("Health-monitor iterations must increase strictly.")
        if self._last_iteration is not None and k != self._last_iteration + 1:
            self._negative_streak = self._saturation_streak = 0
        self._last_iteration = k
        self._seen += 1
        self._history.append(metrics)
        values = metrics.values
        predicted = values.get("probe/predicted_improvement")
        actual = values.get("probe/realized_improvement")
        paired = predicted is not None and actual is not None
        self._paired_probes += int(paired)
        self._negative_streak = (self._negative_streak + 1 if paired and predicted > 0 and actual < 0 else 0)
        saturation = values.get("e_step/lambda_saturation_rate")
        self._saturation_streak = (self._saturation_streak + 1 if saturation is not None and saturation > .70 else 0)
        if not self.enabled:
            return
        if not numerical_valid:
            self._abort(metrics, "nonfinite_actions_or_weights",
                        "Numerical collapse: nonfinite actions or weights detected; inspect E-step scores and policy outputs.")
        if self._saturation_streak >= 3:
            self._abort(metrics, "trust_region_collapse",
                        "Trust-region bisection collapsed: >70% of states cannot achieve n_eff = rho K. "
                        "Cost distribution across K samples is degenerate.")
        if k == 12:
            recent = list(self._history)[-3:]
            complete = len(recent) == 3 and [item.iteration for item in recent] == [10, 11, 12]
            lengths = [item.values.get("collect/mean_ep_length") for item in recent]
            realizations = [item.values.get("probe/realized_improvement") for item in recent]
            predictions = [item.values.get("probe/predicted_improvement") for item in recent]
            if complete and all(value is not None for value in lengths + realizations + predictions):
                mean_length, mean_realized = float(np.mean(lengths)), float(np.mean(realizations))
                if mean_length < 60 and mean_realized <= 0:
                    self._abort(metrics, "bootstrap_failure",
                                "Bootstrap failed: Policy unable to escape immediate falls; "
                                "check tail fall penalty scaling or initial sigma_0.",
                                {"last_three_mean_ep_length": mean_length,
                                 "last_three_mean_realized_improvement": mean_realized})
        if k >= 15 and self._negative_streak >= 5:
            self._abort(metrics, "m_step_disconnect",
                        "M-Step Disconnect: E-step finds local improvements, but M-step updates degrade closed-loop performance. "
                        "Check e_step/jensen_gap_median (multimodal averaging), channel/lipschitz_ratio (target aliasing), "
                        "or m_step/holdout_drift (2,000-step Adam overfitting).")
