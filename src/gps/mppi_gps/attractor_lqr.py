"""Fixed-point Acrobot LQR and finite basin probes, without simulator imports.

Coordinates are unscaled radians/radians-per-second, excluding simulator time.
The locked controller uses the period-one specialization of periodic LQR. Its
quadratic is the original dense task's upright Taylor expansion, not a tuning
sweep: tip horizontal error is 2*q0+q1; vertical squared error is fourth order.
All nonlinear-basin statements require recorded native execution. An ellipsoid
or a finite set of successful starts alone is not a nonlinear basin proof.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

from src.gps.mppi_gps.acrobot_energy import MODEL_PARAMETERS, MODEL_SHA256


def dense_quadratic() -> tuple[np.ndarray, np.ndarray]:
    """Return x'Qx+u'Ru for the unchanged dense objective near upright."""
    q = np.diag([0.0, 0.0, 0.01, 0.01])
    q[:2, :2] = np.outer([2.0, 1.0], [2.0, 1.0])
    return q, np.array([[0.01]])


def upright_continuous_linearization() -> tuple[np.ndarray, np.ndarray]:
    """Analytic mechanical Jacobian at q=v=u=0 for the bound native model.

    The returned input is normalized motor control, so the XML's gear=2 is
    included exactly once. The native one-control Jacobian must be checked
    independently before executing this controller.
    """
    p = MODEL_PARAMETERS
    lower = p["I2"] + p["m2"] * p["c2"] ** 2
    cross = p["m2"] * p["l1"] * p["c2"]
    mass = np.array(
        [
            [
                p["I1"] + p["m1"] * p["c1"] ** 2 + lower + p["m2"] * p["l1"] ** 2 + 2 * cross,
                lower + cross,
            ],
            [lower + cross, lower],
        ]
    )
    a = p["m1"] * p["c1"] + p["m2"] * p["l1"]
    b = p["m2"] * p["c2"]
    gravity = p["gravity"] * np.array([[a + b, b], [b, b]])
    damping = np.diag([p["damping1"], p["damping2"]])
    state = np.block(
        [
            [np.zeros((2, 2)), np.eye(2)],
            [np.linalg.solve(mass, gravity), np.linalg.solve(mass, -damping)],
        ]
    )
    control = np.concatenate(
        [np.zeros((2, 1)), np.linalg.solve(mass, np.array([[0.0], [p["gear"]]]))]
    )
    return state, control


def upright_linearization(
    *, timestep: float = 0.01, frame_skip: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the native RK4 map analytically at its fixed point.

    For linear dynamics with held input, augmented RK4 is a fourth-order matrix
    polynomial. Repeating its native timestep matches frame_skip, rather than
    substituting an exact matrix exponential for the simulator's integrator.
    """
    if not np.isfinite(timestep) or timestep <= 0 or type(frame_skip) is not int or frame_skip < 1:
        raise ValueError("A finite positive timestep and positive integer frame_skip are required.")
    a, b = upright_continuous_linearization()
    augmented = np.zeros((5, 5))
    augmented[:4, :4], augmented[:4, 4:] = a, b
    z = timestep * augmented
    one_step = np.eye(5) + z + z @ z / 2 + z @ z @ z / 6 + z @ z @ z @ z / 24
    transition = np.linalg.matrix_power(one_step, frame_skip)
    return transition[:4, :4], transition[:4, 4:]


def solve_lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray | None = None,
    r: np.ndarray | None = None,
    *,
    action_bound: float = 1.0,
) -> dict[str, Any]:
    """Solve the discrete Riccati equation with explicit stability diagnostics."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != (a.shape[0], 1):
        raise ValueError("Use a square state Jacobian and a single normalized motor input.")
    if q is None or r is None:
        if a.shape != (4, 4):
            raise ValueError("Default cost is defined for four Acrobot coordinates.")
        default_q, default_r = dense_quadratic()
        q, r = default_q if q is None else q, default_r if r is None else r
    q, r = np.asarray(q, dtype=float), np.asarray(r, dtype=float)
    if q.shape != a.shape or r.shape != (1, 1):
        raise ValueError("Quadratic dimensions differ from the linear map.")
    if not all(np.isfinite(x).all() for x in (a, b, q, r)):
        raise ValueError("Linearization and quadratics must be finite.")
    if not np.array_equal(q, q.T) or not np.array_equal(r, r.T):
        raise ValueError("Quadratics must be symmetric.")
    if np.linalg.eigvalsh(q).min() < -1e-12 or r[0, 0] <= 0:
        raise ValueError("Q must be positive semidefinite and R positive definite.")
    if not np.isfinite(action_bound) or action_bound <= 0:
        raise ValueError("action_bound must be positive and finite.")
    p = solve_discrete_are(a, b, q, r)
    p = 0.5 * (p + p.T)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    closed = a - b @ k
    eigenvalues = np.linalg.eigvals(closed)
    residual = p - (q + a.T @ p @ (a - b @ k))
    relative_residual = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(p)))
    radius = float(np.max(np.abs(eigenvalues)))
    if np.linalg.eigvalsh(p).min() <= 0 or radius >= 1 or relative_residual > 1e-8:
        raise ValueError(
            "Riccati solution did not establish positive P and local linear stability."
        )
    input_value = float((k @ np.linalg.solve(p, k.T))[0, 0])
    return dict(
        A=a,
        B=b,
        Q=q,
        R=r,
        P=p,
        K=k,
        closed_loop_eigenvalues_real_imag=np.stack([eigenvalues.real, eigenvalues.imag], axis=1),
        spectral_radius=radius,
        dare_relative_residual=relative_residual,
        unsaturated_rho=float(action_bound**2 / input_value),
        action_bound=float(action_bound),
        model_sha256=MODEL_SHA256,
    )


def wrapped_error(states: np.ndarray) -> np.ndarray:
    """Shortest-angle error about the upright, leaving velocities unchanged."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim < 1 or states.shape[-1] != 5 or not np.isfinite(states).all():
        raise ValueError("Finite full states [...,time,q0,q1,v0,v1] are required.")
    error = states[..., 1:].copy()
    error[..., :2] = (error[..., :2] + np.pi) % (2 * np.pi) - np.pi
    return error


def lqr_action(states: np.ndarray, gain: np.ndarray, *, clip: bool = True) -> np.ndarray:
    """Normalized torque u=clip(-K*error,-1,1); no policy-input changes."""
    gain = np.asarray(gain, dtype=np.float64)
    if gain.shape != (1, 4) or not np.isfinite(gain).all():
        raise ValueError("The Acrobot gain must be finite and shaped (1,4).")
    raw = -np.einsum("ij,...j->...i", gain, wrapped_error(states))
    return np.clip(raw, -1.0, 1.0) if clip else raw


def lyapunov_value(states: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The local linear-controller value V=error'P*error, not nonlinear proof."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (4, 4) or not np.isfinite(p).all():
        raise ValueError("The Acrobot value matrix must be finite and shaped (4,4).")
    error = wrapped_error(states)
    return np.einsum("...i,ij,...j->...", error, p, error)


def ellipsoid_contains(states: np.ndarray, p: np.ndarray, rho: float) -> np.ndarray:
    """Membership in a declared local level set; call it a basin only after tests."""
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("The level-set threshold must be finite and positive.")
    return lyapunov_value(states, p) <= rho


def ellipsoid_coordinate_bounds(p: np.ndarray, rho: float) -> np.ndarray:
    """Marginal coordinate projections; these are NOT a jointly captured box."""
    p = np.asarray(p, dtype=float)
    if p.shape != (4, 4) or not np.isfinite(p).all() or rho <= 0 or not np.isfinite(rho):
        raise ValueError("A finite positive level and four-coordinate value matrix are required.")
    return np.sqrt(rho * np.diag(np.linalg.inv(p)))


def probe_directions() -> np.ndarray:
    """Eight axes plus all sixteen orthant diagonals in Lyapunov coordinates."""
    axes = np.concatenate([np.eye(4), -np.eye(4)])
    corners = np.asarray(list(itertools.product((-1.0, 1.0), repeat=4))) / 2.0
    return np.concatenate([axes, corners])


def independent_directions(seed: int, count: int = 32) -> np.ndarray:
    """Private reproducible held-out directions; freeze seed/count before use."""
    if type(seed) is not int or seed < 0 or type(count) is not int or count < 1:
        raise ValueError("Nonnegative integer seed and positive count are required.")
    values = np.random.Generator(np.random.PCG64(seed)).normal(size=(count, 4))
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def ellipsoid_starts(p: np.ndarray, rho: float, directions: np.ndarray) -> np.ndarray:
    """Full time-zero states on V=rho for unit Lyapunov-coordinate directions."""
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 4 or not np.isfinite(directions).all():
        raise ValueError("Finite directions must have shape (N,4).")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Ellipsoid directions must be unit length.")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")
    transform = np.linalg.cholesky(np.asarray(p, dtype=float)).T
    errors = np.linalg.solve(transform, (np.sqrt(rho) * directions).T).T
    return np.column_stack([np.zeros(len(errors)), errors])


def basin_probe_roster(
    p: np.ndarray,
    unsaturated_rho: float,
    *,
    shell_factors: tuple[float, ...] = (0.25, 1.0, 4.0, 16.0, 64.0, 256.0),
    angle_marginals: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5),
    velocity_marginals: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
) -> list[dict[str, Any]]:
    """A finite descriptive roster, not certification of its convex hull.

    The 197 default starts comprise equilibrium, 144 shell probes and 52
    single-coordinate marginals. Failures remain rows; testing them does not
    enlarge any admitted join level without separately frozen verification.
    """
    rows: list[dict[str, Any]] = [dict(id="equilibrium", kind="equilibrium", state=[0.0] * 5)]
    directions = probe_directions()
    for factor in shell_factors:
        states = ellipsoid_starts(p, factor * unsaturated_rho, directions)
        for i, state in enumerate(states):
            rows.append(
                dict(
                    id=f"shell_{factor:g}_{i}",
                    kind="shell",
                    factor=factor,
                    direction=i,
                    rho=float(factor * unsaturated_rho),
                    state=state.tolist(),
                )
            )
    for coordinate in range(4):
        magnitudes = angle_marginals if coordinate < 2 else velocity_marginals
        for magnitude in magnitudes:
            if not np.isfinite(magnitude) or magnitude <= 0:
                raise ValueError("Marginal magnitudes must be finite and positive.")
            for sign in (-1, 1):
                state = np.zeros(5)
                state[1 + coordinate] = sign * magnitude
                rows.append(
                    dict(
                        id=f"axis_{coordinate}_{sign * magnitude:g}",
                        kind="marginal",
                        coordinate=coordinate,
                        value=float(sign * magnitude),
                        state=state.tolist(),
                    )
                )
    return rows
