"""Tasks for the loop. Each setup(args) returns a TaskSetup.

TaskSetup fields:
- env: the MuJoCo env (task cost, health rule);
- spec: spec.Task (observation, observed state indices, clipping, success rule);
- starts: training start states (one MPPI walk from each per round);
- current: [(controls, states)] the starting walks (set the similarity's scale and the cost
  normalizer c0);
- policy: the starting policy;
- c0: the starting walks' costs (the task score's normalizer);
- judge(policy) -> dict of test measures; "standing_walks" (Walker) or "success" is what the
  loop's readings and kill rule count;
- task_score(policy) -> c0's mean over the policy's own mean cost from fixed starts, at most 1;
- legs: Section's leg indices for the gait measures, or None to skip them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TaskSetup:
    env: Any
    spec: Any
    starts: list
    current: list
    policy: Any
    c0: Any
    judge: Callable
    task_score: Callable
    legs: dict | None
