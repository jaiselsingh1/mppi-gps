"""Independent continuous-control ending-cost mechanism experiment."""

from .core import (
    Actor, IndependentMPPI, Task, actor_features, capacity_witness_theta,
    local_lower_bound, pure_actor_actions,
)

__all__ = [
    "Actor", "IndependentMPPI", "Task", "actor_features", "capacity_witness_theta",
    "local_lower_bound", "pure_actor_actions",
]
