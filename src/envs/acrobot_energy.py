"""Additive Acrobot cost adapter; the original dense task is not modified."""

from __future__ import annotations

import hashlib
from typing import Any

import mujoco
import numpy as np

from src.envs.acrobot import MODEL_PATH, Acrobot
from src.gps.mppi_gps import acrobot_energy as objective


class EnergyAcrobot(Acrobot):
    """The same native model, reset law and controls with a declared energy cost."""

    def __init__(
        self,
        frame_skip: int = 2,
        nthread: int | None = None,
        seed: int | None = None,
        reset_std: float = 0.05,
        weight: float = objective.ENERGY_WEIGHTS[0],
    ) -> None:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Energy weight must be finite and nonnegative.")
        self.energy_weight = float(weight)
        super().__init__(frame_skip=frame_skip, nthread=nthread, seed=seed, reset_std=reset_std)

    def cost_components(self, states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        return objective.components(states, actions, self.energy_weight)

    def components(self, states: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        return self.cost_components(states, actions)

    def running_cost(
        self, states: np.ndarray, actions: np.ndarray, sensordata: np.ndarray | None = None
    ) -> np.ndarray:
        return objective.running_cost(states, actions, self.energy_weight)

    def terminal_cost(self, states: np.ndarray, sensordata: np.ndarray | None = None) -> np.ndarray:
        return objective.terminal_cost(states)


def _inertia_y(model: Any, body: int) -> float:
    w, x, y, z = np.asarray(model.body_iquat[body], dtype=np.float64)
    row_y = np.array([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)])
    return float(np.dot(row_y**2, model.body_inertia[body]))


def native_energy_validation(
    env: Acrobot, atol: float = 1e-10, states: np.ndarray | None = None
) -> dict[str, Any]:
    """Called only inside a frozen native prerequisite; performs zero timesteps.

    The default twelve states, or an explicitly supplied frozen state roster,
    test analytic versus engine potential/kinetic energy. N+1 mj_forward calls
    include restoration of the incoming complete integration context. Pos/Vel
    energy routines are each called N times.
    The caller charges those forwards/routines separately from physics steps.
    """
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != objective.MODEL_SHA256:
        raise ValueError("The preserved Acrobot XML binding changed.")
    model = env.model
    if (model.nq, model.nv, model.nu) != (2, 2, 1):
        raise ValueError("Energy validation requires the preserved Acrobot dimensions.")
    upper, lower = model.body("upper_arm").id, model.body("lower_arm").id
    measured = {
        "m1": float(model.body_mass[upper]),
        "m2": float(model.body_mass[lower]),
        "c1": float(model.body_ipos[upper, 2]),
        "c2": float(model.body_ipos[lower, 2]),
        "l1": float(model.body_pos[lower, 2]),
        "I1": _inertia_y(model, upper),
        "I2": _inertia_y(model, lower),
        "gravity": float(-model.opt.gravity[2]),
        "pivot_height": float(model.body_pos[upper, 2]),
        "damping1": float(model.dof_damping[0]),
        "damping2": float(model.dof_damping[1]),
        "armature1": float(model.dof_armature[0]),
        "armature2": float(model.dof_armature[1]),
        "gear": float(model.actuator_gear[0, 0]),
    }
    for name, expected in objective.MODEL_PARAMETERS.items():
        if not np.isclose(measured[name], expected, rtol=0, atol=1e-12):
            raise ValueError(f"Compiled Acrobot energy parameter changed: {name}.")
    for body in (upper, lower):
        if not np.allclose(model.body_ipos[body, :2], 0, rtol=0, atol=1e-12):
            raise ValueError("Acrobot COM is no longer on the link axis.")
    if not np.allclose(model.opt.gravity[:2], 0, rtol=0, atol=1e-12):
        raise ValueError("Acrobot gravity is no longer vertical.")
    if not np.allclose(model.jnt_axis, [[0, 1, 0], [0, 1, 0]], rtol=0, atol=1e-12):
        raise ValueError("Acrobot joint axes changed.")
    spin = np.sqrt(
        2 * (objective.UPRIGHT_ENERGY - 29.43) / (objective.MODEL_PARAMETERS["I2"] + 0.25)
    )
    if states is None:
        states = np.array(
            [
                [0, 0, 0, 0, 0],
                [0, np.pi, 0, 0, 0],
                [0, np.pi, np.pi, 0, 0],
                [0, np.pi, np.pi, 0, spin],
                [0, 0.7, -1.1, 1.2, -0.3],
                [0, -2, 2.4, -0.5, 2],
                [0, 3, 4, 2.5, -3],
                [0, 2 * np.pi + 0.1, -4 * np.pi - 0.2, 3, -2],
                [0, np.pi / 2, 0, 0, 0],
                [0, np.pi / 2, -np.pi / 2, 1, -1],
                [0, 0, 0, 2, 3],
                [0, np.pi, 0, -5, 8],
            ],
            dtype=np.float64,
        )
    else:
        states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 5 or len(states) == 0:
        raise ValueError("Native energy validation needs a nonempty (N, 5) state roster.")
    if not np.isfinite(states).all():
        raise ValueError("Native energy validation states must be finite.")
    integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    original = np.empty(mujoco.mj_stateSize(model, integration_spec))
    mujoco.mj_getState(model, env.data, original, integration_spec)
    native = []
    try:
        for state in states:
            mujoco.mj_setState(model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            mujoco.mj_forward(model, env.data)
            mujoco.mj_energyPos(model, env.data)
            mujoco.mj_energyVel(model, env.data)
            native.append(env.data.energy.copy())
    finally:
        mujoco.mj_setState(model, env.data, original, integration_spec)
        mujoco.mj_forward(model, env.data)
        mujoco.mj_setState(model, env.data, original, integration_spec)
    restored = np.empty_like(original)
    mujoco.mj_getState(model, env.data, restored, integration_spec)
    if restored.tobytes() != original.tobytes():
        raise ValueError("Energy validation failed byte-exact incoming-context restoration.")
    native = np.asarray(native)
    calculated = objective.energy_components(states)
    analytic = np.stack([calculated["potential"], calculated["kinetic"]], axis=-1)
    if not np.allclose(analytic, native, rtol=0, atol=atol):
        raise ValueError("Analytic energy disagrees with native potential/kinetic energy.")
    return {
        "status": "PASSED",
        "states": states.tolist(),
        "analytic_potential_kinetic": analytic.tolist(),
        "native_potential_kinetic": native.tolist(),
        "maximum_absolute_error": float(np.max(np.abs(analytic - native))),
        "model_parameters": measured,
        "model_sha256": objective.MODEL_SHA256,
        "rtol": 0,
        "atol": atol,
        "new_physics_controls": 0,
        "physics_substeps": 0,
        "native_forwards": len(states) + 1,
        "native_energy_pos_calls": len(states),
        "native_energy_vel_calls": len(states),
        "native_state_sets": len(states) + 2,
        "native_state_gets": 2,
        "incoming_context_restored_exactly": True,
    }
