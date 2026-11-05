"""Bounded identical-model CasADi reference, with a separate native verifier.

The symbolic equations use physical elbow torque. Solver outputs are candidates,
never recorded as executed experience. This module does not import a simulator
or CasADi until the corresponding explicit command is selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MODEL_SHA = "324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe"


def write_json(path: Path, value: Any) -> None:
    """Create receipts without overwriting an earlier attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def symbolic_dynamics(ca: Any) -> tuple[Any, Any]:
    """Coulom capsule equations in the XML's upright-zero coordinates."""
    x, torque = ca.MX.sym("x", 4), ca.MX.sym("torque")
    q0, q1, w0, w1 = (x[i] for i in range(4))
    lower = 0.09530365719649561 + 0.25
    cross = 0.5 * ca.cos(q1)
    mass = ca.vertcat(
        ca.horzcat(0.0955703125 + 0.25 + lower + 1 + 2 * cross, lower + cross),
        ca.horzcat(lower + cross, lower),
    )
    coriolis = ca.vertcat(-0.5 * ca.sin(q1) * (2 * w0 * w1 + w1**2), 0.5 * ca.sin(q1) * w0**2)
    gravity = -9.81 * ca.vertcat(1.5 * ca.sin(q0) + 0.5 * ca.sin(q0 + q1), 0.5 * ca.sin(q0 + q1))
    acceleration = ca.solve(
        mass, ca.vertcat(0, torque) - coriolis - gravity - 0.05 * ca.vertcat(w0, w1)
    )
    f = ca.Function("dynamics", [x, torque], [ca.vertcat(w0, w1, acceleration)])
    dt = ca.MX.sym("dt")
    k1 = f(x, torque)
    k2 = f(x + dt * k1 / 2, torque)
    k3 = f(x + dt * k2 / 2, torque)
    k4 = f(x + dt * k3, torque)
    step = ca.Function("rk4", [x, torque, dt], [x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6])
    return f, step


def task_cost(ca: Any, x: Any, torque: Any) -> Any:
    horizontal = ca.sin(x[0]) + ca.sin(x[0] + x[1])
    vertical = ca.cos(x[0]) + ca.cos(x[0] + x[1]) - 2
    return horizontal**2 + vertical**2 + 0.01 * ca.sumsqr(x[2:]) + 0.01 * (torque / 2) ** 2


def initial_guess(n: int, duration: float, shape: int) -> tuple[np.ndarray, np.ndarray]:
    """Three prospective smooth guesses, not previously optimized references."""
    phase = np.linspace(0, 1, n + 1)
    envelope = np.sin(np.pi * phase) ** 2
    q0 = np.pi * (1 - phase) + (0, 2, -2)[shape] * np.sin(2 * np.pi * phase) * envelope
    q1 = (0, 3, -3)[shape] * np.sin(2 * np.pi * phase) * envelope
    q = np.stack([q0, q1])
    velocity = np.gradient(q, duration / n, axis=1)
    x = np.vstack([q, velocity])
    x[:, 0] = [np.pi, 0, 0, 0]
    x[:, -1] = 0
    return x, np.zeros(n + 1)


def solver_options(max_iter: int, cpu_seconds: int) -> dict[str, Any]:
    return {
        "ipopt.max_iter": max_iter,
        "ipopt.max_cpu_time": float(cpu_seconds),
        "ipopt.tol": 1e-8,
        "ipopt.constr_viol_tol": 1e-8,
        "ipopt.print_level": 5,
        "print_time": True,
    }


def solve_attempt(
    ca: Any,
    out: Path,
    config: dict,
    *,
    duration: float,
    shape: int,
    fine: dict | None = None,
    cost_objective: bool = False,
) -> dict:
    """Save debug iterates and counts even if IPOPT fails."""
    out.mkdir(exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    f, step = symbolic_dynamics(ca)
    opti = ca.Opti()
    if fine is None:
        n = config["coarse_intervals"]
        total_time = opti.variable()
        opti.subject_to(
            opti.bounded(*[config["duration_bounds"][0], total_time, config["duration_bounds"][1]])
        )
        opti.set_initial(total_time, duration)
        dt = total_time / n
    else:
        n = fine["intervals"]
        total_time = n * 0.01
        dt = 0.01
    x = opti.variable(4, n + 1)
    u = opti.variable(1, n + 1 if fine is None else n)
    opti.subject_to(x[:, 0] == np.asarray(config["start"]))
    opti.subject_to(x[:, -1] == np.zeros(4))
    opti.subject_to(opti.bounded(-2, u, 2))
    effort, cost = 0, 0
    for i in range(n):
        if fine is None:
            left, right = f(x[:, i], u[i]), f(x[:, i + 1], u[i + 1])
            midpoint = (x[:, i] + x[:, i + 1]) / 2 + dt * (left - right) / 8
            defect = (
                x[:, i + 1]
                - x[:, i]
                - dt / 6 * (left + right + 4 * f(midpoint, (u[i] + u[i + 1]) / 2))
            )
            effort += dt * (u[i] ** 2 + u[i + 1] ** 2) / 2
            cost += dt * (task_cost(ca, x[:, i], u[i]) + task_cost(ca, x[:, i + 1], u[i + 1])) / 2
        else:
            defect = x[:, i + 1] - step(x[:, i], u[i], dt)
            effort += dt * u[i] ** 2
            cost += dt * task_cost(ca, x[:, i + 1], u[i])
        opti.subject_to(defect == 0)
    if fine is None:
        opti.minimize(total_time + 0.0001 * effort)
        guess_x, guess_u = initial_guess(n, duration, shape)
    else:
        opti.minimize(cost if cost_objective else effort)
        guess_x, guess_u = fine["x"], fine["u"]
    opti.set_initial(x, guess_x)
    opti.set_initial(u, guess_u)
    key = "coarse" if fine is None else "fine"
    opti.solver("ipopt", solver_options(config[f"{key}_max_iter"], config[f"{key}_cpu_seconds"]))
    error = None
    try:
        sol = opti.solve()
        value = sol.value
    except RuntimeError as exc:
        error = str(exc)[-3000:]
        value = opti.debug.value
    stats = opti.stats()
    try:
        xx = np.asarray(value(x)).reshape(4, n + 1)
        uu = np.asarray(value(u)).reshape(-1)
        duration_value = float(value(total_time)) if fine is None else float(total_time)
        gg = np.asarray(value(opti.g)).reshape(-1)
        lower_g = np.asarray(value(opti.lbg)).reshape(-1)
        upper_g = np.asarray(value(opti.ubg)).reshape(-1)
        violation = float(np.max(np.maximum(lower_g - gg, gg - upper_g), initial=0))
        finite = bool(
            np.isfinite(xx).all() and np.isfinite(uu).all() and np.isfinite(duration_value)
        )
        np.savez_compressed(
            out / "candidate.npz",
            states=xx.T,
            torques_nm=uu,
            duration=duration_value,
            constraints=gg,
            lower=lower_g,
            upper=upper_g,
        )
    except RuntimeError as exc:
        finite, violation, duration_value = False, None, None
        error = (error or "") + str(exc)[-1000:]
    feasible = bool(
        finite and violation is not None and violation <= config[f"{key}_constraint_atol"]
    )
    result = dict(
        schema="acrobot-collocation-attempt-v1",
        intervals=n,
        method=key,
        objective="task_cost" if cost_objective else "time_effort" if fine is None else "effort",
        initial_duration=duration,
        initial_shape=shape,
        duration=duration_value,
        feasible=feasible,
        constraint_max_violation=violation,
        solver_success=bool(stats.get("success", False)),
        error=error,
        solver_stats=stats,
        cpu_seconds=time.process_time() - cpu_started,
        wall_seconds=time.monotonic() - started,
        symbolic_dynamics_per_constraint_evaluation=3 * n if fine is None else 4 * n,
        native_physics_steps=0,
        global_optimality_established=False,
    )
    write_json(out / "result.json", result)
    return result


def solve(out: Path, config: dict) -> dict:
    import casadi as ca

    out.mkdir(exist_ok=False)
    write_json(
        out / "runtime.json",
        dict(casadi=ca.__version__, numpy=np.__version__, python=os.sys.version, config=config),
    )
    records = []
    for duration in config["initial_durations"]:
        for shape in config["initial_shapes"]:
            name = f"coarse_t{int(duration)}_shape{shape}"
            result = solve_attempt(ca, out / name, config, duration=duration, shape=shape)
            records.append(dict(path=name, **result))
    feasible = [r for r in records if r["feasible"]]
    selected = min(feasible, key=lambda r: (r["duration"], r["path"])) if feasible else None
    fine_records = []
    if selected is not None:
        coarse = np.load(out / selected["path"] / "candidate.npz")
        n = math.ceil(selected["duration"] / 0.01)
        times = np.linspace(0, selected["duration"], len(coarse["states"]))
        fine_times = np.arange(n + 1) * 0.01
        guess = dict(
            intervals=n,
            x=np.stack([np.interp(fine_times, times, row) for row in coarse["states"].T]),
            u=np.interp(fine_times[:-1], times, coarse["torques_nm"]),
        )
        for objective in (False, True):
            name = "fine_task_cost" if objective else "fine_effort"
            result = solve_attempt(
                ca,
                out / name,
                config,
                duration=n * 0.01,
                shape=selected["initial_shape"],
                fine=guess,
                cost_objective=objective,
            )
            fine_records.append(dict(path=name, **result))
            if not result["feasible"]:
                break
            previous = np.load(out / name / "candidate.npz")
            guess["x"], guess["u"] = previous["states"].T, previous["torques_nm"]
    result = dict(
        status="FEASIBLE_FOR_NATIVE_VERIFICATION"
        if any(r["feasible"] for r in fine_records)
        else "NO_NATIVE_VERIFIABLE_SOLUTION",
        coarse=records,
        selected_coarse=selected,
        fine=fine_records,
        native_physics_steps=0,
        claim="Local collocation candidates only; no minimum-time lower bound or global cost optimum.",
    )
    write_json(out / "result.json", result)
    return result


def native_verify(solve_dir: Path, out: Path, config: dict) -> dict:
    import mujoco

    model_path = ROOT / "assets/acrobot.xml"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA:
        raise ValueError("Exact XML binding failed")
    out.mkdir(exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if model.opt.timestep != 0.01 or int(model.opt.integrator) != int(
        mujoco.mjtIntegrator.mjINT_RK4
    ):
        raise ValueError("Reference native model must remain .01 RK4")
    if not np.array_equal(model.actuator_gear[:, 0], [2]) or not np.array_equal(
        model.actuator_ctrlrange, [[-1, 1]]
    ):
        raise ValueError("Physical ±2 Nm binding differs")
    from src.gps.mppi_gps.acrobot_energy import MODEL_PARAMETERS

    def rhs(x: np.ndarray, torque: float) -> np.ndarray:
        q0, q1, w0, w1 = x
        lower = MODEL_PARAMETERS["I2"] + 0.25
        cross = 0.5 * np.cos(q1)
        mass = np.array(
            [
                [MODEL_PARAMETERS["I1"] + 0.25 + lower + 1 + 2 * cross, lower + cross],
                [lower + cross, lower],
            ]
        )
        bias = np.array(
            [
                -0.5 * np.sin(q1) * (2 * w0 * w1 + w1 * w1)
                - 9.81 * (1.5 * np.sin(q0) + 0.5 * np.sin(q0 + q1))
                + 0.05 * w0,
                0.5 * np.sin(q1) * w0 * w0 - 4.905 * np.sin(q0 + q1) + 0.05 * w1,
            ]
        )
        return np.r_[x[2:], np.linalg.solve(mass, np.array([0, torque]) - bias)]

    def step(x: np.ndarray, torque: float) -> np.ndarray:
        k1 = rhs(x, torque)
        k2 = rhs(x + 0.005 * k1, torque)
        k3 = rhs(x + 0.005 * k2, torque)
        k4 = rhs(x + 0.01 * k3, torque)
        return x + 0.01 * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    rng, parity = np.random.default_rng(2026110501), []
    requested = 0
    for index in range(32):
        x = rng.uniform([-np.pi, -np.pi, -5, -5], [np.pi, np.pi, 5, 5])
        torque = float(rng.uniform(-2, 2))
        data = mujoco.MjData(model)
        data.qpos[:], data.qvel[:] = x[:2], x[2:]
        mujoco.mj_forward(model, data)
        data.ctrl[0] = torque / 2
        requested += 1
        mujoco.mj_step(model, data)
        actual, expected = np.r_[data.qpos, data.qvel], step(x, torque)
        error = float(np.max(np.abs(actual - expected)))
        parity.append(
            dict(
                index=index,
                state=x.tolist(),
                torque_nm=torque,
                native=actual.tolist(),
                analytic=expected.tolist(),
                max_error=error,
            )
        )
    write_json(out / "parity.json", dict(cases=parity, native_steps=requested))
    if max(r["max_error"] for r in parity) > config["native_parity_atol"]:
        result = dict(status="STOPPED_PARITY", native_steps=requested)
        write_json(out / "result.json", result)
        return result
    summary = json.loads((solve_dir / "result.json").read_text())
    records = []
    for candidate in summary["fine"]:
        if not candidate["feasible"]:
            continue
        values = np.load(solve_dir / candidate["path"] / "candidate.npz")
        controls = np.r_[values["torques_nm"], np.zeros(round(config["hold_seconds"] / 0.01))]
        if requested + len(controls) > config["native_max_steps"]:
            raise ValueError("Native reference budget exhausted before trajectory")
        if np.max(np.abs(controls)) > 2 + 1e-8:
            raise ValueError("Solver returned controls outside physical limit")
        data = mujoco.MjData(model)
        data.qpos[:] = np.asarray(config["start"])[:2]
        data.qvel[:] = np.asarray(config["start"])[2:]
        mujoco.mj_forward(model, data)
        states, energy, contexts = [np.r_[data.qpos, data.qvel]], [], []
        for torque in controls:
            data.ctrl[0] = np.clip(torque / 2, -1, 1)
            requested += 1
            mujoco.mj_step(model, data)
            state = np.r_[data.qpos, data.qvel]
            if not np.isfinite(state).all() or np.max(np.abs(state[2:])) > 50:
                raise ValueError("Invalid native reference execution")
            states.append(state)
            energy.append(data.energy.copy())
            context = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, context, mujoco.mjtState.mjSTATE_INTEGRATION)
            contexts.append(context)
        states = np.array(states)
        height = 2 + np.cos(states[:, 0]) + np.cos(states[:, 0] + states[:, 1])
        horizontal = np.sin(states[:, 0]) + np.sin(states[:, 0] + states[:, 1])
        distance = np.sqrt(horizontal**2 + (height - 4) ** 2)
        post = states[1:]
        cost = (
            distance[1:] ** 2 + 0.01 * np.sum(post[:, 2:] ** 2, axis=1) + 0.01 * (controls / 2) ** 2
        )
        # dm_control's Gaussian tolerance: bounds[0,.2], margin1, value_at_margin=.1.
        rewards = np.exp(math.log(0.1) * np.maximum(distance[1:] - 0.2, 0) ** 2)
        hold = round(config["hold_seconds"] / 0.01)
        errors = (states[-hold:, :2] + np.pi) % (2 * np.pi) - np.pi
        captured = bool(
            np.min(height[-hold:]) >= 3.8
            and np.max(np.abs(errors)) <= 0.02
            and np.max(np.abs(states[-hold:, 2:])) <= 0.05
        )
        replay_error = float(np.max(np.abs(states[: len(values["states"])] - values["states"])))
        np.savez_compressed(
            out / (candidate["path"] + ".npz"),
            states=states,
            torques_nm=controls,
            energy=energy,
            integration=contexts,
            costs=cost,
            dm_control_reward=rewards,
            heights=height,
        )
        records.append(
            dict(
                candidate=candidate["path"],
                captured=captured,
                minimum_hold_height=float(np.min(height[-hold:])),
                final_state=states[-1].tolist(),
                dense_cost_sum=float(np.sum(cost)),
                dense_cost_integral=float(0.01 * np.sum(cost)),
                mean_dm_control_reward=float(np.mean(rewards)),
                native_solver_state_max_error=replay_error,
                steps=len(controls),
            )
        )
    result = dict(
        status="REFERENCE_EXECUTES"
        if any(r["captured"] for r in records)
        else "NO_EXECUTABLE_REFERENCE",
        records=records,
        native_steps=requested,
        mujoco_version=mujoco.__version__,
        model_sha256=MODEL_SHA,
        standard_start_evaluation=False,
        global_optimality_established=False,
    )
    write_json(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["solve", "native"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--solve-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "solve":
        result = solve(args.out, config)
    else:
        if args.solve_dir is None:
            parser.error("native requires --solve-dir")
        result = native_verify(args.solve_dir, args.out, config)
    print(json.dumps(dict(status=result["status"]), sort_keys=True))


if __name__ == "__main__":
    main()
