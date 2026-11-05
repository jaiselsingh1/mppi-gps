"""Source-derived Drake Acrobot reference, explicitly adapted to our XML.

This is not Drake's stock controller: that controller owns the default plant
and clamps at 20 Nm. Here the model is parameterized and the clamp is 2 Nm.
Imports of Drake are confined to the preparation entry point. No integration
or controller tuning occurs in this module's preparation command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from fractions import Fraction as F
from pathlib import Path

import numpy as np

from src.gps.mppi_gps.acrobot_energy import MODEL_PARAMETERS, MODEL_SHA256


def mechanics(error: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mass, bias, energy in our upright-zero coordinates; physical torque."""
    q0, q1, v0, v1 = np.asarray(error, dtype=float)
    p = MODEL_PARAMETERS
    a = p["m2"] * p["l1"] * p["c2"]
    lower = p["I2"] + p["m2"] * p["c2"] ** 2
    upper = p["I1"] + p["m1"] * p["c1"] ** 2
    mass = np.array(
        [
            [upper + lower + p["m2"] * p["l1"] ** 2 + 2 * a * np.cos(q1), lower + a * np.cos(q1)],
            [lower + a * np.cos(q1), lower],
        ]
    )
    shoulder = p["m1"] * p["c1"] + p["m2"] * p["l1"]
    end = p["m2"] * p["c2"]
    gravity = p["gravity"] * np.array(
        [shoulder * np.sin(q0) + end * np.sin(q0 + q1), end * np.sin(q0 + q1)]
    )
    coriolis = a * np.sin(q1) * np.array([-2 * v0 * v1 - v1 * v1, v0 * v0])
    bias = coriolis - gravity + np.array([p["damping1"] * v0, p["damping2"] * v1])
    energy = float(
        0.5 * np.array([v0, v1]) @ mass @ np.array([v0, v1])
        + p["gravity"] * (shoulder * np.cos(q0) + end * np.cos(q0 + q1))
    )
    return mass, bias, energy


def spong_action(
    full_state: np.ndarray, gain: np.ndarray, value: np.ndarray, balance_level: float
) -> float:
    return spong_control(full_state, gain, value, balance_level)["action"]


def spong_control(
    full_state: np.ndarray, gain: np.ndarray, value: np.ndarray, balance_level: float
) -> dict:
    """Official Spong equation/gains with our plant, torque cap, certified join.

    Return normalized motor input. The energy zero differs from MuJoCo's by
    a constant; E-Edesired is unchanged. Balance uses continuous-time CARE.
    """
    error = np.asarray(full_state, dtype=float)[1:].copy()
    error[:2] = (error[:2] + np.pi) % (2 * np.pi) - np.pi
    local_value = float(error @ value @ error)
    balanced = local_value <= balance_level
    _, _, energy = mechanics(np.asarray(full_state, dtype=float)[1:])
    if balanced:
        torque = -float((gain @ error).item())
    else:
        # Drake's PFL uses the original (unwrapped) elbow coordinate.
        original = np.asarray(full_state, dtype=float)[1:]
        mass, bias, energy = mechanics(original)
        inverse = np.linalg.inv(mass)
        desired = MODEL_PARAMETERS["gravity"] * (
            MODEL_PARAMETERS["m1"] * MODEL_PARAMETERS["c1"]
            + MODEL_PARAMETERS["m2"] * (MODEL_PARAMETERS["l1"] + MODEL_PARAMETERS["c2"])
        )
        energy_torque = -5.0 * (energy - desired) * original[3]
        desired_acceleration = -50.0 * original[1] - 5.0 * original[3]
        pfl_torque = (inverse[0, 1] * bias[0] + desired_acceleration) / inverse[1, 1] + bias[1]
        torque = energy_torque + pfl_torque
    return {
        "action": float(np.clip(torque, -2.0, 2.0) / 2.0),
        "raw_torque": float(torque),
        "energy": energy,
        "balance": balanced,
        "local_value": local_value,
    }


def _matrix(value):
    return [[F(float(x)) for x in row] for row in np.asarray(value)]


def _transpose(a):
    return [list(x) for x in zip(*a, strict=True)]


def _multiply(a, b):
    return [
        [sum(x * y for x, y in zip(row, col, strict=True)) for col in zip(*b, strict=True)]
        for row in a
    ]


def _inverse(a):
    n = len(a)
    rows = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for i in range(n):
        if rows[i][i] == 0:
            swap = next(j for j in range(i + 1, n) if rows[j][i])
            rows[i], rows[swap] = rows[swap], rows[i]
        pivot = rows[i][i]
        rows[i] = [x / pivot for x in rows[i]]
        for j in range(n):
            if j != i:
                factor = rows[j][i]
                rows[j] = [x - factor * y for x, y in zip(rows[j], rows[i], strict=True)]
    return [row[n:] for row in rows]


def _ldl(a):
    n = len(a)
    lower = [[F(int(i == j)) for j in range(n)] for i in range(n)]
    diagonal = []
    for j in range(n):
        d = a[j][j] - sum(lower[j][k] ** 2 * diagonal[k] for k in range(j))
        if d <= 0:
            raise ValueError("Exact rational LDL positivity failed")
        diagonal.append(d)
        for i in range(j + 1, n):
            lower[i][j] = (
                a[i][j] - sum(lower[i][k] * lower[j][k] * diagonal[k] for k in range(j))
            ) / d
    return lower, diagonal


def local_certificate(gain: np.ndarray, value: np.ndarray) -> dict:
    """Exact rational check of a conservative nonlinear continuous-time ROA.

    The remainder bound uses |sin z-z|<=|z|^3/6, |cos z-1|<=z^2/2.
    The resulting quadratic SOS Gram is rational and checked by exact LDL.
    It certifies continuously recomputed, unsaturated angle feedback only;
    it does not certify 50Hz sample-and-hold or MuJoCo's RK4 numerical map.
    """
    p = MODEL_PARAMETERS
    # Binary64 model parameters and saved controller coefficients define the
    # rational nominal model exactly; physical parameter uncertainty is absent.
    a = F(p["m2"]) * F(p["l1"]) * F(p["c2"])
    lower_inertia = F(p["I2"]) + F(p["m2"]) * F(p["c2"]) ** 2
    m00 = (
        F(p["I1"])
        + F(p["m1"]) * F(p["c1"]) ** 2
        + lower_inertia
        + F(p["m2"]) * F(p["l1"]) ** 2
        + 2 * a
    )
    mass = [[m00, lower_inertia + a], [lower_inertia + a, lower_inertia]]
    inv = _inverse(mass)
    shoulder = F(p["m1"]) * F(p["c1"]) + F(p["m2"]) * F(p["l1"])
    end = F(p["m2"]) * F(p["c2"])
    g = F(p["gravity"])
    k = _matrix(gain)[0]
    force = [
        [g * (shoulder + end), g * end, -F(p["damping1"]), F(0)],
        [g * end - k[0], g * end - k[1], -k[2], -F(p["damping2"]) - k[3]],
    ]
    acceleration = _multiply(inv, force)
    acl = [[F(0), F(0), F(1), F(0)], [F(0), F(0), F(0), F(1)], *acceleration]
    v = _matrix(value)
    # Freeze a symmetric rational value matrix, not a solver-specific asymmetry.
    v = [[(v[i][j] + v[j][i]) / 2 for j in range(4)] for i in range(4)]
    _ldl(v)
    left = _multiply(_transpose(acl), v)
    qcl = [[-left[i][j] - left[j][i] for j in range(4)] for i in range(4)]
    _ldl(qcl)
    # CARE residual bounds Qcl >= (1 - ||E||inf) I, since Q>=I, K'K>=0.
    target = [
        [F(int(i == j)) * (10 if i < 2 else 1) + k[i] * k[j] for j in range(4)] for i in range(4)
    ]
    residual = max(sum(abs(target[i][j] - qcl[i][j]) for j in range(4)) for i in range(4))
    qlower = 1 - residual
    if qlower <= 0:
        raise ValueError("CARE residual too large for bound")
    mass_lower = (mass[0][0] * mass[1][1] - mass[0][1] ** 2) / (mass[0][0] + mass[1][1])
    delta = 2 * a  # ||M(q)-M(0)||2 <= 2a ||x||2^2.
    # Triangle/L1 bound: |q0+q1| <= 2||x||; two endpoint terms
    # give 16*end/6, but Euclidean triangle gives sqrt(2)*(sqrt(2))^3
    # =4*end/6. Use the looser 8*end/6 as an explicit safety margin.
    gravity_remainder = g * (shoulder + 8 * end) / 6
    coriolis_remainder = 2 * a
    acceleration_norm = sum(abs(x) for row in acceleration for x in row)
    pbottom = sum(abs(v[i][j]) for i in range(4) for j in (2, 3))
    radius = F(1, 10)
    for _ in range(40):
        denominator = mass_lower - delta * radius**2
        if denominator > 0:
            remainder = (
                gravity_remainder + coriolis_remainder + delta * acceleration_norm
            ) / denominator
            gamma = 2 * pbottom * remainder * radius**2
            if gamma <= qlower / 2 and sum(abs(x) for x in k) * radius <= F(9, 5):
                break
        radius /= 2
    else:
        raise ValueError("No local remainder certificate within frozen 40 halvings")
    gram = [[qcl[i][j] - (gamma if i == j else 0) for j in range(4)] for i in range(4)]
    lower, diagonal = _ldl(gram)
    vinverse = _inverse(v)
    vinverse_norm = max(sum(abs(x) for x in row) for row in vinverse)
    level = radius**2 / vinverse_norm

    def scalar(x):
        return {"rational": str(x), "float": float(x)}

    return {
        "claim": "exact continuous nominal EOM, continuously recomputed local linear feedback; not sampled execution",
        "radius": scalar(radius),
        "level": scalar(level),
        "remainder_B": scalar(remainder),
        "gamma": scalar(gamma),
        "Qcl_lower": scalar(qlower),
        "CARE_residual_infinity_bound": scalar(residual),
        "mass_eigenvalue_lower": scalar(mass_lower),
        "torque_bound": scalar(sum(abs(x) for x in k) * radius),
        "gram_rational": [[str(x) for x in row] for row in gram],
        "gram_float": [[float(x) for x in row] for row in gram],
        "gram_LDL_diagonal": [str(x) for x in diagonal],
        "gram_LDL_lower": [[str(x) for x in row] for row in lower],
        "P_rational": [[str(x) for x in row] for row in v],
        "exact_LDL_pass": True,
    }


def prepare(output: Path, config_path: Path) -> None:
    """One bounded Drake CARE + one quadratic SDP; no physics integration."""
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text())
    version = importlib.metadata.version("drake")
    if version != config["drake_version"]:
        raise RuntimeError("Installed Drake differs from frozen dependency binding")
    from pydrake.examples import AcrobotPlant
    from pydrake.solvers import CsdpSolver, MathematicalProgram
    from pydrake.systems.controllers import LinearQuadraticRegulator
    from pydrake.systems.primitives import Linearize

    plant = AcrobotPlant()
    context = plant.CreateDefaultContext()
    params = plant.get_mutable_parameters(context)
    mapping = {
        "m1": "m1",
        "m2": "m2",
        "l1": "l1",
        "lc1": "c1",
        "lc2": "c2",
        "Ic1": "I1",
        "Ic2": "I2",
        "b1": "damping1",
        "b2": "damping2",
        "gravity": "gravity",
    }
    for name, key in mapping.items():
        getattr(params, "set_" + name)(MODEL_PARAMETERS[key])
    params.set_l2(1.0)
    context.get_mutable_continuous_state_vector().SetFromVector([np.pi, 0.0, 0.0, 0.0])
    plant.get_input_port(0).FixValue(context, [0.0])
    linear = Linearize(plant, context)
    q = np.diag([10.0, 10.0, 1.0, 1.0])
    r = np.array([[1.0]])
    gain, value = LinearQuadraticRegulator(linear.A(), linear.B(), q, r)
    value = (value + value.T) / 2
    cert = local_certificate(gain, value)
    solver = CsdpSolver()
    if not solver.available() or not solver.enabled():
        raise RuntimeError("CSDP unavailable; stop before native validation")
    program = MathematicalProgram()
    gram = program.NewSymmetricContinuousVariables(4, "gram")
    for i in range(4):
        for j in range(i + 1):
            program.AddLinearConstraint(gram[i, j] == cert["gram_float"][i][j])
    program.AddPositiveSemidefiniteConstraint(gram)
    result = solver.Solve(program)
    if not result.is_success():
        raise RuntimeError("Quadratic SOS feasibility failed")
    numerical = result.GetSolution(gram)
    residual = float(np.max(np.abs(numerical - np.array(cert["gram_float"]))))
    mineigen = float(np.linalg.eigvalsh(numerical).min())
    if residual > 1e-7 or mineigen < -1e-7:
        raise RuntimeError("SOS numerical validation failed")
    record = {
        "protocol": "acrobot-drake-reference-v" + config["version"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_sha256": MODEL_SHA256,
        "model_parameters": MODEL_PARAMETERS,
        "drake_version": importlib.metadata.version("drake"),
        "state_map": "theta1=q0+pi, theta2=q1, same velocities",
        "input_map": "physical torque=2*normalized action",
        "Q": q.tolist(),
        "R": r.tolist(),
        "K_physical": gain.tolist(),
        "P": value.tolist(),
        "A": linear.A().tolist(),
        "B_physical": linear.B().tolist(),
        "certificate": cert,
        "SOS": {
            "solver": "CSDP",
            "status": str(result.get_solution_result()),
            "max_coefficient_residual": residual,
            "minimum_gram_eigenvalue": mineigen,
        },
        "counts": {"native_controls": 0, "physics_substeps": 0, "CARE_solves": 1, "SDP_solves": 1},
        "elapsed_wall_seconds": time.monotonic() - started,
    }
    with (output / "preparation.json").open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/acrobot_drake_reference_v1_1.json")
    )
    args = parser.parse_args()
    prepare(args.output, args.config)
