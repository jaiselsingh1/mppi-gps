"""Batched stepping, finite-difference linearization, LQR gains and per-control task costs."""

from __future__ import annotations

import numpy as np

EPS = 1e-6  # finite-difference step (scaled state units / action units)
R = 1.0  # LQR control weight
CAP = 3.0  # largest singular value kept in an LQR gain (the slope targets)


def step(env, states, controls):
    """One control (frame_skip physics steps) from each of a batch of full states."""
    out, _ = env._rollout_ctx.rollout(
        env.model, env._data_pool, states, np.repeat(controls[:, None], env._frame_skip, axis=1)
    )
    return out[:, -1]


def linearize(env, xbar, ubar, scale):
    """Scaled one-control Jacobians A (T,n,n), B (T,n,m) by central differences (n = state size - 1)."""
    T, n, m = len(ubar), xbar.shape[1] - 1, ubar.shape[1]
    rows_x, rows_u = [], []
    for t in range(T):
        for i in range(n):
            for s in (1.0, -1.0):
                x = xbar[t].copy()
                x[1 + i] += s * EPS * scale[i]
                rows_x.append((x, ubar[t]))
        for j in range(m):
            for s in (1.0, -1.0):
                u = ubar[t].copy()
                u[j] += s * EPS
                rows_u.append((xbar[t], u))
    nx = step(env, np.array([r[0] for r in rows_x]), np.array([r[1] for r in rows_x]))
    nu = step(env, np.array([r[0] for r in rows_u]), np.array([r[1] for r in rows_u]))
    nx = nx[:, 1:].reshape(T, n, 2, n) / scale
    nu = nu[:, 1:].reshape(T, m, 2, n) / scale
    A = ((nx[:, :, 0] - nx[:, :, 1]) / (2 * EPS)).transpose(0, 2, 1)
    B = ((nu[:, :, 0] - nu[:, :, 1]) / (2 * EPS)).transpose(0, 2, 1)
    return A, B


def lqr(A, B, r, free=(0,)):
    """Finite-horizon LQR on scaled deviations; Q identity except the free coordinates (forward position)."""
    n, m = A.shape[1], B.shape[2]
    Q = np.eye(n)
    Q[list(free), list(free)] = 0.0
    P, K = Q.copy(), np.zeros((len(A), m, n))
    for t in range(len(A) - 1, -1, -1):
        BtP = B[t].T @ P
        K[t] = np.linalg.solve(r * np.eye(m) + BtP @ B[t], BtP @ A[t])
        P = Q + A[t].T @ P @ (A[t] - B[t] @ K[t])
        P = 0.5 * (P + P.T)
    return K


def capped(K, cap):
    """K_t with singular values clipped to at most cap."""
    if cap is None:
        return K
    u, s, vt = np.linalg.svd(K, full_matrices=False)
    return np.einsum("tij,tj,tjk->tik", u, np.minimum(s, cap), vt)


def closed_loop(env, start, xbar, ubar, K, scale, feedforward):
    """Run feedforward controls with LQR feedback toward x_bar from start (one row per feedforward)."""
    n = len(feedforward)
    states = np.repeat(start[None], n, axis=0)
    seen, used = [], []
    for t in range(len(ubar)):
        dx = (states[:, 1:] - xbar[t, 1:]) / scale
        u = np.clip(feedforward[:, t] - np.einsum("ij,nj->ni", K[t], dx), -1.0, 1.0)
        states = step(env, states, u)
        seen.append(states)
        used.append(u)
    return np.stack(seen, 1), np.stack(used, 1)


def step_costs(env, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """The env's rollout cost split by control: (K, T), summing to rollout_cost."""
    healthy = env._is_healthy(states[..., 2], states[..., 3])
    active = np.ones_like(healthy, dtype=bool)
    active[:, 1:] = np.logical_and.accumulate(healthy[:, :-1], axis=1)
    cost = np.where(active, env.running_cost(states, actions), env._w_unhealthy)
    if env._apply_terminal_cost_on_done:
        fell = ~healthy.all(axis=1)
        rows = np.flatnonzero(fell)
        first = np.argmax(~healthy[rows], axis=1)
        cost[rows, first] += env.terminal_cost(states[rows, first])
    return cost


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
