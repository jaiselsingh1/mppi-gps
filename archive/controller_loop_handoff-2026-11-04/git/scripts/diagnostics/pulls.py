"""Pulls of a trajectory toward the policy other than action matching (shared by pull_bench and
controller_loop). Each returns similarity(states (N, T, n), controls (N, T, m)) -> (N, T), the
per-control pull cost for the trajectory search (similarity_loop.search's `similarity`)."""

from __future__ import annotations

import numpy as np

from scripts.diagnostics.feedback_rollouts import step

CARRY_EVERY, CARRY_HORIZON, FALLEN = 100, 125, 3.0


def policy_states(env, spec, policy, x0, length):
    """The policy alone from x0: states after each control and whether it is still up."""
    x, seen, up, alive = x0[None].copy(), [], [], True
    for _ in range(length):
        u = np.clip(policy.predict(spec.observation(x).astype(np.float32)).astype(float), -1, 1)
        x = step(env, x, u)
        alive = alive and bool(env._is_healthy(x[:, 2], x[:, 3])[0])
        seen.append(x[0])
        up.append(alive)
    return np.array(seen), np.array(up)


def obs_distance(a, b, scale, spec):
    """rms distance (sd units, observed coordinates) along the last axis."""
    d = (a[..., spec.observed] - b[..., spec.observed]) / scale[spec.observed - 1]
    return np.sqrt((d**2).mean(-1))


def motion_pull(env, spec, policy, start, scale, length):
    """Toward where the policy goes: squared distance to its own rollout while it stays up."""
    ref, up = policy_states(env, spec, policy, start, length)

    def sim(s, c):
        return np.where(up[None], obs_distance(s, ref[None], scale, spec) ** 2, 0.0)

    return sim


def carry_on_pull(env, spec, policy, start, scale, periodic=False):
    """Toward states the policy can carry on from: at points along the trajectory, the policy
    alone runs CARRY_HORIZON controls; the cost at the point is its mean distance (sd) to the
    trajectory over that horizon, FALLEN once it has fallen. periodic: the trajectory is one
    stride of a cycle, so it is continued by itself (points every quarter stride)."""

    def sim(s, c):
        N, T = s.shape[:2]
        before = np.concatenate([np.repeat(start[None, None], N, 0), s[:, :-1]], 1)
        if periodic:
            reps = 1 + -(-CARRY_HORIZON // T)
            ahead = np.concatenate([s] * (reps + 1), 1)
            points = list(range(0, T, max(1, T // 4)))
        else:
            ahead = s
            points = list(range(0, T - CARRY_HORIZON, CARRY_EVERY))
        if not points:
            return np.zeros((N, T))
        x = before[:, points].reshape(-1, s.shape[-1]).copy()
        alive = np.ones(len(x), bool)
        dist = np.zeros((len(x), CARRY_HORIZON))
        ref = np.stack([ahead[:, t : t + CARRY_HORIZON] for t in points], 1).reshape(
            len(x), CARRY_HORIZON, -1
        )
        for h in range(CARRY_HORIZON):
            u = np.clip(policy.predict(spec.observation(x).astype(np.float32)).astype(float), -1, 1)
            x = step(env, x, u)
            alive &= env._is_healthy(x[:, 2], x[:, 3])
            dist[:, h] = np.where(alive, obs_distance(x, ref[:, h], scale, spec), FALLEN)
        out = np.zeros((N, T))
        out[:, points] = dist.mean(1).reshape(N, len(points))
        return out

    return sim
