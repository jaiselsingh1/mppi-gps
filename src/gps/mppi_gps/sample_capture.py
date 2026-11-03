"""Original final-pass MPPI bank capture helper; no recording launch command.

Preserves RecordingMPPI from the exploratory sample_bank source verbatim.
It retains eight float16 candidate controls and float32 costs/weights/plan,
not complete sample trajectories, planner snapshots or replay certificates.
Use the existing pure sample_bank module for saved-array labeling and review.
"""

from __future__ import annotations

import numpy as np

from src.mppi.mppi import MPPI

PREFIX = 8


class RecordingMPPI(MPPI):
    """Unchanged MPPI that keeps the last pass's samples, costs and weights."""

    def _sample_noise(self) -> np.ndarray:
        self._noise = super()._sample_noise()
        self._nominal_first = self.U[0].copy()
        return self._noise

    def plan_step(self, state, **kwargs):
        action, info = super().plan_step(state, **kwargs)
        self.record = dict(
            nominal_first=self._nominal_first,
            raw_first=(self._nominal_first + self._noise[:, 0]).astype(np.float32),
            prefix=self._last_actions[:, :PREFIX].astype(np.float16),
            costs=self._last_costs.astype(np.float32),
            weights=self._last_weights.astype(np.float32),
            plan=self._last_plan.astype(np.float32),
        )
        return action, info
