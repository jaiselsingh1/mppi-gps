"""Saved episode summaries with the historical schema variants preserved."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def legacy_summary(episodes: Sequence[Any]) -> dict[str, Any]:
    return dict(
        episodes=len(episodes),
        successes=sum(bool(e.success) for e in episodes),
        survived=sum(bool(e.survived) for e in episodes),
        mean_cost=float(np.mean([e.cost for e in episodes])),
        mean_velocity_mae=float(np.mean([e.mae for e in episodes])),
        costs=[float(e.cost) for e in episodes],
        tags=[e.tag for e in episodes],
        mean_steps=float(np.mean([e.steps for e in episodes])),
    )


def summary(episodes: Sequence[Any]) -> dict[str, Any]:
    return dict(
        episodes=len(episodes),
        tags=[e.tag for e in episodes],
        successes=sum(bool(e.success) for e in episodes),
        survived=sum(bool(e.survived) for e in episodes),
        mean_cost=float(np.mean([e.cost for e in episodes])),
        costs=[float(e.cost) for e in episodes],
        mean_steps=float(np.mean([e.steps for e in episodes])),
        steps=[int(e.steps) for e in episodes],
        mean_velocity_mae=float(np.mean([e.mae for e in episodes])),
    )
