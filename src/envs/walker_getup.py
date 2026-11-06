"""Walker2d without the fall rule: the get-up task's dense cost, and the plain task cost unended.

WalkerNoFall (general, any task): the Walker2d task's own running cost over the whole rollout, no
absorbing failure after a fall (the unhealthy penalty still applies per control while unhealthy),
so a planner started from a fallen state still sees which motions lead back toward the task.
WalkerGetUp (hand-made, 2026-11-06; parked after the user's point that a task per failure does not
carry to other envs):

Same model, state layout, reset and stepping as Walker2d; only the task differs. The cost per
control is getup_probe's (2026-11-05): 10 max(0, 1.25 - z)^2 + pitch^2 + 0.1 vx^2 + 1e-3 |u|^2
(torso at standing height, upright, standing still, little effort). Nothing is ever "unhealthy",
so a rollout is scored over its whole length (lying down is where the task starts).
"""

import numpy as np

from src.envs.walker2d import Walker2d

STAND_Z, W_H, W_UP, W_V, W_U = 1.25, 10.0, 1.0, 0.1, 1e-3


def getup_cost(states, actions):
    """(..., T) cost per control: torso height, uprightness, standing still, effort."""
    z, pitch, vx = states[..., 2], states[..., 3], states[..., 10]
    return (
        W_H * np.maximum(0.0, STAND_Z - z) ** 2
        + W_UP * pitch**2
        + W_V * vx**2
        + W_U * np.sum(actions**2, axis=-1)
    )


def stood_up(states):
    """(..., T, n) -> (...): torso above 1.0 m and |pitch| below 0.5 rad over the final 1 s."""
    tail = states[..., -125:, :]
    return ((tail[..., 2] > 1.0) & (np.abs(tail[..., 3]) < 0.5)).all(-1)


class WalkerNoFall(Walker2d):
    def rollout_cost(self, states, actions, sensordata=None):
        costs = self.running_cost(states, actions, sensordata).sum(axis=1)
        valid = np.all(np.isfinite(states), axis=(-1, -2)) & np.all(np.isfinite(actions), axis=(-1, -2))
        return np.where(valid, costs, np.inf)


class WalkerGetUp(Walker2d):
    @staticmethod
    def _is_healthy(z, angle):
        return np.ones(np.shape(z), dtype=bool)

    def running_cost(self, states, actions, sensordata=None):
        return getup_cost(states, actions)

    def terminal_cost(self, states, sensordata=None):
        return np.zeros(states.shape[:-1])

    def rollout_cost(self, states, actions, sensordata=None):
        costs = getup_cost(states, actions).sum(axis=1)
        valid = np.all(np.isfinite(states), axis=(-1, -2)) & np.all(np.isfinite(actions), axis=(-1, -2))
        return np.where(valid, costs, np.inf)
