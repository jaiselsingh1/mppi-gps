"""Shared numpy pieces for PREREG_handover.md (rss_checks/rl; imported from both venvs, numpy + mujoco only).
States are full-physics vectors x = [t, qpos(9), qvel(9)] (19,)."""
import numpy as np

GAMMA = 0.995


def obs17(X):
    """(..., 19) full states -> (..., 17) the policy's observation (qpos[1:], qvel clipped at +-10)."""
    return np.concatenate([X[..., 2:10], np.clip(X[..., 10:19], -10.0, 10.0)], -1)


def healthy(X):
    return (X[..., 2] > 0.8) & (X[..., 2] < 2.0) & (np.abs(X[..., 3]) < 1.0)


def rewards(X, U=None):
    """X (..., T, 19): states after each control; U (..., T, 6) or None (drops the 0.001 |u|^2 term).
    our_walker's reward r = 5 - c - 20 [fall], and r = 0 after the first unhealthy state (episode over).
    Returns (r (..., T), alive at the end (...))."""
    h = healthy(X)
    c = 2.0 * np.abs(1.5 - X[..., 10]) + 0.5 * X[..., 11] ** 2 + 0.1 * X[..., 3] ** 2 + 5.0 * (~h)
    if U is not None:
        c = c + 1e-3 * np.sum(U ** 2, -1)
    r = 5.0 - c - 20.0 * (~h)
    ok = np.cumprod(h, -1).astype(bool)
    prev_ok = np.concatenate([np.ones_like(ok[..., :1]), ok[..., :-1]], -1)
    return np.where(prev_ok, r, 0.0), ok[..., -1]


def disc(r):
    return (GAMMA ** np.arange(r.shape[-1]) * r).sum(-1)


def _mlp(layers, h):
    for i, (W, b) in enumerate(layers):
        z = h @ W + b
        h = z / (1.0 + np.exp(-z)) if i < len(layers) - 1 else z
    return h


def critic_fn(C):
    """C: dict-like with kind ('mlp' | 'basin'), n_nets, n_layers, n{j}_W{i}, n{j}_b{i}, in_mean, in_std, a, b, kappa.
    Per net v = a * out + b (basin: a * sigmoid(out) + b); with several nets V = mean - kappa * std.
    Returns V: (N, 19) states -> (N,) value in reward units (discounted, gamma = 0.995)."""
    kind, n, L = str(C["kind"]), int(C["n_nets"]), int(C["n_layers"])
    nets = [[(np.asarray(C[f"n{j}_W{i}"], np.float64), np.asarray(C[f"n{j}_b{i}"], np.float64)) for i in range(L)] for j in range(n)]
    m, s = np.asarray(C["in_mean"], np.float64), np.asarray(C["in_std"], np.float64)
    a, b, kap = float(C["a"]), float(C["b"]), float(C["kappa"])

    def V(X):
        h0 = (obs17(np.asarray(X, np.float64)) - m) / s
        o = np.stack([_mlp(net, h0)[..., 0] for net in nets])
        if kind == "basin":
            o = 1.0 / (1.0 + np.exp(-o))
        v = a * o + b
        return v[0] if n == 1 else v.mean(0) - kap * v.std(0)
    return V


def policy_obs_fn(P):
    """Brax PPO policy exported by policy_export.export_policy -> deterministic action tanh(loc) from (N, 17) obs."""
    layers = [(P[f"PW{i}"], P[f"Pb{i}"]) for i in range(5)]
    m, s = P["obs_mean"], P["obs_std"]

    def pi(o):
        return np.tanh(_mlp(layers, (o - m) / s)[:, :6])
    return pi
