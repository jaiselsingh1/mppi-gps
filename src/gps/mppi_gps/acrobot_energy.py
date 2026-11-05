"""Batched Acrobot energy and the versioned swing-up objective, without physics.

Geometry and inertia match the preserved ``assets/acrobot.xml`` and the saved
2026-10-27 native parameter packet. Zero angles are upright. The old catches
used a deficit-only energy term and a 29.43 J normalization from (pi, pi).
This version uses the full signed squared error, the true 39.24 J
hanging-to-upright difference, and the old proximity gate to hand off to balance.
The original dense-cost Acrobot and its failed prerequisite remain unchanged.
"""

from __future__ import annotations

import numpy as np

MODEL_SHA256 = "324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe"
MODEL_PARAMETERS = {
    "m1": 1.0,
    "m2": 1.0,
    "c1": 0.5,
    "c2": 0.5,
    "l1": 1.0,
    "I1": 0.0955703125,
    "I2": 0.09530365719649561,
    "gravity": 9.81,
    "pivot_height": 2.0,
    "damping1": 0.05,
    "damping2": 0.05,
    "armature1": 0.0,
    "armature2": 0.0,
    "gear": 2.0,
}
UPRIGHT_ENERGY = 58.86
HANGING_ENERGY = 19.62
ENERGY_SCALE = 39.24
ENERGY_WEIGHTS = (160.0 / 9.0, 320.0 / 9.0)
TARGET = np.array([0.0, 0.0, 4.0])


def _states(value: np.ndarray) -> np.ndarray:
    states = np.asarray(value, dtype=np.float64)
    if states.ndim < 1 or states.shape[-1] != 5:
        raise ValueError("Acrobot full states require final dimension five.")
    if not np.isfinite(states).all():
        raise ValueError("Acrobot cost requires finite full states.")
    return states


def tip_position(states: np.ndarray) -> np.ndarray:
    """Exact world tip coordinates from full states (..., 5)."""
    states = _states(states)
    shoulder, absolute_elbow = states[..., 1], states[..., 1] + states[..., 2]
    horizontal = np.sin(shoulder) + np.sin(absolute_elbow)
    height = 2.0 + np.cos(shoulder) + np.cos(absolute_elbow)
    return np.stack([horizontal, np.zeros_like(height), height], axis=-1)


def energy_components(states: np.ndarray) -> dict[str, np.ndarray]:
    """Mechanical energy in joules, vectorized over arbitrary batch axes."""
    states = _states(states)
    shoulder, elbow = states[..., 1], states[..., 2]
    shoulder_velocity, elbow_velocity = states[..., 3], states[..., 4]
    lower = MODEL_PARAMETERS["I2"] + 0.25
    cross = 0.5 * np.cos(elbow)
    mass00 = MODEL_PARAMETERS["I1"] + 0.25 + lower + 1.0 + 2.0 * cross
    mass01 = lower + cross
    kinetic = 0.5 * (
        mass00 * shoulder_velocity**2
        + 2.0 * mass01 * shoulder_velocity * elbow_velocity
        + lower * elbow_velocity**2
    )
    potential = 9.81 * (4.0 + 1.5 * np.cos(shoulder) + 0.5 * np.cos(shoulder + elbow))
    return {
        "kinetic": kinetic,
        "potential": potential,
        "total_energy": kinetic + potential,
    }


def _pose_balance(states: np.ndarray) -> dict[str, np.ndarray]:
    tip = tip_position(states)
    distance = np.linalg.norm(tip - TARGET, axis=-1)
    reward = np.exp(np.log(0.1) * (np.maximum(distance - 0.2, 0.0) / 2.0) ** 2)
    gate = reward**2
    pose = 1.0 - np.clip(tip[..., 2] / 4.0, 0.0, 1.0) + 0.05 * tip[..., 0] ** 2
    velocity = states[..., 3:5]
    scaled_velocity = np.sum((velocity / np.array([2.5, 5.0])) ** 2, axis=-1)
    excess = np.maximum(np.abs(velocity) - 8.0, 0.0) / 8.0
    return {
        "tip_distance": distance,
        "tip_height": tip[..., 2],
        "proximity_reward": reward,
        "gate": gate,
        "pose": pose,
        "scaled_velocity_squared": scaled_velocity,
        "excess_velocity_squared": np.sum(excess**2, axis=-1),
    }


def components(states: np.ndarray, actions: np.ndarray, weight: float) -> dict[str, np.ndarray]:
    """Cost components, with non-additive raw values and explicit weighted terms.

    ``running`` is pose + weighted_energy + weighted_balance + weighted_excess.
    The action coefficient stays zero, matching the earlier energy catches.
    Aliases distance_cost/energy_cost/velocity_cost/total serve recorder readers;
    aliases must not be added again to the canonical weighted components.
    """
    states = _states(states)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != states.shape[:-1] + (1,) or not np.isfinite(actions).all():
        raise ValueError("One finite Acrobot action must match every full state.")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("Energy weight must be finite and nonnegative.")
    values = {**energy_components(states), **_pose_balance(states)}
    error = (values["total_energy"] - UPRIGHT_ENERGY) / ENERGY_SCALE
    weighted_energy = float(weight) * (1.0 - values["gate"]) * error**2
    weighted_balance = 0.15 * values["gate"] * values["scaled_velocity_squared"]
    weighted_excess = 0.05 * values["excess_velocity_squared"]
    running = values["pose"] + weighted_energy + weighted_balance + weighted_excess
    terminal = 100.0 * values["pose"] + 25.0 * values["gate"] * values["scaled_velocity_squared"]
    return {
        **values,
        "energy_error": error,
        "weighted_energy": weighted_energy,
        "weighted_balance": weighted_balance,
        "weighted_excess": weighted_excess,
        "running": running,
        "terminal": terminal,
        "control_cost": np.zeros_like(running),
        "distance_cost": values["pose"],
        "energy_cost": weighted_energy,
        "velocity_cost": weighted_balance,
        "total": running,
    }


def running_cost(states: np.ndarray, actions: np.ndarray, weight: float) -> np.ndarray:
    """Post-action running cost, without integrating or querying a simulator."""
    return components(states, actions, weight)["running"]


def terminal_cost(states: np.ndarray) -> np.ndarray:
    """The earlier catch objective's fixed pose and proximity-gated balance tail."""
    values = _pose_balance(_states(states))
    return 100.0 * values["pose"] + 25.0 * values["gate"] * values["scaled_velocity_squared"]
