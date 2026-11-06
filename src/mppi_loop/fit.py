"""The policy fit: actions plus LQR-gain slopes, targets mixed with the deployed policy's own."""

from __future__ import annotations

import numpy as np
import torch
from torch.func import jacrev, vmap

from src.mppi_loop.spec import WALKER

UPDATES = 20_000
BATCH = 256
LR = 3e-4
SLOPE = 3.0  # weight of the slope term against the action term
CHUNK = 32


def fit_rows(
    policy, xbar, u, K, scale, lam, updates=UPDATES, P=None, seed=0, spec=WALKER, lr_decay=False
):
    """Adam on |pi - u|^2 + lam |J + K|^2 over (x_bar_t, u_t, K_t, scale) rows (GPU if present).

    lr_decay: the learning rate falls linearly from LR to 0 over the updates, so the fit ends
    at the minimum it is heading for rather than anywhere in minibatch noise."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)
    obs = torch.tensor(spec.observation(xbar).astype(np.float32), device=device)
    target = torch.tensor(u.astype(np.float32), device=device)
    gains = torch.tensor(K[:, :, spec.observed - 1].astype(np.float32), device=device)
    P = None if P is None else torch.tensor(P.astype(np.float32), device=device)
    factor = torch.tensor(spec.slope_factor(xbar, scale).astype(np.float32), device=device)
    mean, std = policy.observation_mean, policy.observation_std

    def act(o):
        return policy.network((o - mean) / std)

    jac = vmap(jacrev(act))
    optimizer = torch.optim.Adam(policy.network.parameters(), lr=LR)
    generator = torch.Generator().manual_seed(seed)
    policy.train()
    for n in range(updates):
        if lr_decay:
            for group in optimizer.param_groups:
                group["lr"] = LR * (1.0 - n / updates)
        i = torch.randint(len(obs), (BATCH,), generator=generator).to(device)
        loss = ((act(obs[i]) - target[i]) ** 2).mean()
        if lam > 0:
            J = jac(obs[i]) * factor[i][:, None, :]
            miss = J + gains[i]
            if P is not None:
                miss = miss @ P[i]
            loss = loss + lam * (miss**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    policy.eval()
    policy.to("cpu")
    return policy


def mixed_targets(anchor, s, spec, X, U, K, SC):
    """Targets (1 - s) MPPI + s anchor for actions and slopes (s = the anchor's task score).

    For squared losses this is the same as weighting the fit to MPPI's rows by 1 - s and the fit
    to the anchor policy's own actions and slopes by s. Slopes are compared in fit_rows' units:
    the fit drives J * factor toward -K on the observed coordinates, so the anchor's K is
    -(its J * factor)."""
    obs = torch.tensor(spec.observation(X).astype(np.float32))
    factor = torch.tensor(spec.slope_factor(X, SC).astype(np.float32))
    mean, std = anchor.observation_mean.cpu(), anchor.observation_std.cpu()
    net = anchor.network.cpu()

    def act(o):
        return net((o - mean) / std)

    with torch.no_grad():
        own = act(obs).numpy().astype(np.float64)
    J = torch.cat([vmap(jacrev(act))(obs[i : i + 4096]) for i in range(0, len(obs), 4096)])
    J = (J * factor[:, None, :]).detach().numpy().astype(np.float64)
    K = K.copy()
    K[:, :, spec.observed - 1] = (1.0 - s) * K[:, :, spec.observed - 1] - s * J
    return (1.0 - s) * U + s * own, K


def pre_states(start, states):
    return np.concatenate([np.repeat(start[None, None], len(states), 0), states[:, :-1]], 1)


def discrepancy(policy, start, states, controls, spec=WALKER) -> np.ndarray:
    """Per-control 1/2 |u_t - pi(h(x_t))|^2 along each trajectory's own pre-action states."""
    x = pre_states(start, states)
    out = np.empty(controls.shape[:-1])
    for i in range(0, len(x), CHUNK):
        xs = x[i : i + CHUNK][..., None, :]
        us = np.clip(controls[i : i + CHUNK][..., None, :], -1.0, 1.0)
        obs = spec.observation(xs.reshape(-1, xs.shape[-1])).astype(np.float32)
        p = policy.predict(obs).reshape(us.shape).astype(np.float64)
        out[i : i + CHUNK] = 0.5 * ((us - p) ** 2).sum(-1).mean(-1)
    return out


def gap(policy, start, states, controls, spec):
    """Per-action MSE between a trajectory's actions and the policy along its own states."""
    return float(
        2.0
        * discrepancy(policy, start, states[None], controls[None], spec=spec).sum()
        / controls.size
    )
