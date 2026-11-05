"""Bounded native fixed-point locking, basin probes and MPPI-to-LQR joins.

The execution stages share the existing native recorder and MPPI optimizer.
Native imports occur only inside execution. No learner is constructed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.gps.mppi_gps.acrobot_energy_probe import _finalize, resource_timers
from src.gps.mppi_gps.acrobot_teacher import (
    Budget,
    InvalidExecution,
    ResourceLimit,
    _capture,
    _environment,
    _source_bindings,
)
from src.gps.mppi_gps.attractor_lqr import (
    basin_probe_roster,
    ellipsoid_coordinate_bounds,
    ellipsoid_starts,
    independent_directions,
    lqr_action,
    solve_lqr,
    upright_linearization,
    wrapped_error,
)
from src.gps.mppi_gps.provenance import sha, write

ROOT = Path(__file__).resolve().parents[3]


def plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def capture_metrics(states: np.ndarray, actions: np.ndarray, raw: np.ndarray) -> dict:
    """Keep height success distinct from empirical fixed-point convergence."""
    x = np.asarray(states)[1:]
    height = 2 + np.cos(x[:, 1]) + np.cos(x[:, 1] + x[:, 2])
    error = wrapped_error(x)
    height_success = len(x) >= 100 and bool((height[-100:] >= 3.8).all())
    angle = float(np.max(np.abs(error[-100:, :2])))
    velocity = float(np.max(np.abs(error[-100:, 2:])))
    return dict(
        controls=len(actions),
        height_success=height_success,
        captured=bool(height_success and angle <= 0.02 and velocity <= 0.05),
        maximum_height=float(height.max()),
        final_minimum_height=float(height[-100:].min()),
        final_angle_max=angle,
        final_velocity_max=velocity,
        clipped_fraction=float(np.mean(np.abs(raw) > 1)),
        initial_error=wrapped_error(states[:1])[0].tolist(),
        final_error=error[-1].tolist(),
    )


def select_shell(rows: list[dict], factors: list[float]) -> float | None:
    selected = None
    for factor in factors:
        found = [r for r in rows if r.get("kind") == "shell" and r.get("factor") == factor]
        if len(found) != 24 or not all(r["captured"] for r in found):
            break
        selected = factor
    return selected


def record_lqr(env: Any, mj: Any, group: Any, budget: Budget, gain: np.ndarray, steps: int) -> dict:
    """Record the bounded torque controller with complete boundary contexts."""
    seen = {key: [value] for key, value in _capture(mj, env).items()}
    actions, raw, observations, costs = [], [], [], []
    group.attrs["status"] = "RUNNING"
    try:
        for _ in range(steps):
            before = seen["states"][-1]
            observation = env._get_obs()
            if not np.array_equal(observation, before[1:]):
                raise InvalidExecution("LQR pre-action observation differs from its state.")
            observations.append(observation)
            raw.append(lqr_action(before, gain, clip=False))
            action = np.clip(raw[-1], -1, 1)
            budget.reserve("native", 1, env._frame_skip)
            _, cost, done, _ = env.step(action)
            budget.complete("native", 1, env._frame_skip)
            context = _capture(mj, env)
            if (
                done
                or not np.isfinite(cost)
                or not all(np.isfinite(v).all() for v in context.values())
            ):
                raise InvalidExecution("Invalid LQR native execution.")
            if not np.isclose(context["states"][0] - before[0], 0.02, rtol=0, atol=1e-10):
                raise InvalidExecution("LQR native time failed to advance.")
            actions.append(action)
            costs.append(cost)
            for key, value in context.items():
                seen[key].append(value)
        group.attrs["status"] = "COMPLETED"
    except BaseException:
        group.attrs["status"] = "STOPPED"
        raise
    finally:
        arrays = dict(
            seen,
            actions=actions,
            raw_actions=raw[: len(actions)],
            pre_action_observations=observations[: len(actions)],
            costs=costs,
        )
        for key, value in arrays.items():
            array = np.asarray(value)
            budget.check(output_reservation=array.nbytes + 4096)
            group.create_dataset(key, data=array)
        group.attrs["completed_controls"] = len(actions)
        group.file.flush()
    return capture_metrics(np.asarray(seen["states"]), np.asarray(actions), np.asarray(raw))


def native_linearization(env: Any, mj: Any, group: Any, budget: Budget, spec: dict) -> dict:
    starts, actions, predictions = [np.zeros(5)], [np.zeros(1)], []
    initial_contexts, final_contexts = [], []
    for delta in spec["fd_steps"]:
        for j in range(5):
            for sign in (1, -1):
                state, control = np.zeros(5), np.zeros(1)
                if j < 4:
                    state[j + 1] = sign * delta
                else:
                    control[0] = sign * delta
                starts.append(state)
                actions.append(control)
    rng = np.random.Generator(np.random.PCG64(spec["heldout_linearization_seed"]))
    held = rng.normal(0, spec["heldout_scale"], (16, 5))
    for row in held:
        starts.append(np.r_[0.0, row[:4]])
        actions.append(row[4:])
    for state, control in zip(starts, actions):
        env.reset(state=state)
        budget.work.resets += 1
        budget.work.forwards += 1
        if mj is not None:
            initial_contexts.append(_capture(mj, env))
        budget.reserve("verification", 1, 2)
        env.step(control)
        budget.complete("verification", 1, 2)
        predictions.append(env.get_state())
        if mj is not None:
            final_contexts.append(_capture(mj, env))
    starts, actions, predictions = map(np.asarray, (starts, actions, predictions))
    for key, array in dict(
        probe_states=starts, probe_actions=actions, next_states=predictions
    ).items():
        group.create_dataset(key, data=array)
    for label, contexts in (("probe", initial_contexts), ("next", final_contexts)):
        if contexts:
            for field in ("integration", "warmstarts"):
                group.create_dataset(
                    label + "_" + field, data=np.asarray([x[field] for x in contexts])
                )
    group.file.flush()
    if np.max(np.abs(predictions[0, 1:])) > spec["equilibrium_atol"]:
        raise InvalidExecution("Upright is not a native fixed point.")
    matrices = []
    for offset, delta in zip((1, 11), spec["fd_steps"]):
        columns = (
            predictions[offset : offset + 10 : 2, 1:]
            - predictions[offset + 1 : offset + 10 : 2, 1:]
        ) / (2 * delta)
        matrices.append((columns[:4].T, columns[4:, :].T))
    analytic = upright_linearization()
    errors = [float(np.max(np.abs(a - b))) for pair in matrices for a, b in zip(pair, analytic)]
    if max(errors) > spec["linearization_atol"]:
        raise InvalidExecution("Native finite difference differs from analytic mechanical map.")
    a, b = matrices[0]
    estimate = starts[21:, 1:] @ a.T + actions[21:] @ b.T
    held_error = float(np.max(np.abs(estimate - predictions[21:, 1:])))
    if held_error > spec["linearization_atol"]:
        raise InvalidExecution("Held-out local map prediction failed.")
    answer = solve_lqr(a, b)
    answer.update(
        analytic_A=analytic[0],
        analytic_B=analytic[1],
        finite_difference_max_errors=errors,
        heldout_prediction_max_error=held_error,
    )
    return answer


def replay_prefix(env: Any, mj: Any, group: Any, budget: Budget) -> dict:
    count = 12
    mj.mj_setState(env.model, env.data, group["integration"][0], mj.mjtState.mjSTATE_INTEGRATION)
    restored = _capture(mj, env)
    if any(restored[k].tobytes() != group[k][0].tobytes() for k in restored):
        raise InvalidExecution("Native integration restoration was not byte exact.")
    budget.work.restores_attempted += 1
    budget.work.restores_completed += 1
    errors = {k: 0.0 for k in restored}
    replay = {k: [v] for k, v in restored.items()}
    replay_costs = []
    for t in range(count):
        budget.reserve("verification", 1, 2)
        _, cost, _, _ = env.step(group["actions"][t])
        budget.complete("verification", 1, 2)
        for key, value in _capture(mj, env).items():
            errors[key] = max(errors[key], float(np.max(np.abs(value - group[key][t + 1]))))
            replay[key].append(value)
        replay_costs.append(cost)
        errors["costs"] = max(errors.get("costs", 0.0), abs(cost - group["costs"][t]))
    if max(errors.values()) > 1e-10:
        raise InvalidExecution("Native LQR prefix replay mismatch.")
    saved = group.file.create_group("replay")
    for key, value in replay.items():
        saved.create_dataset(key, data=np.asarray(value))
    saved.create_dataset("actions", data=group["actions"][:count])
    saved.create_dataset("costs", data=np.asarray(replay_costs))
    return dict(status="PASSED", controls=count, rtol=0, atol=1e-10, maximum_errors=errors)


def lqr_stage(config: dict, out: Path, budget: Budget) -> dict:
    import h5py
    import mujoco

    from src.envs.acrobot import Acrobot

    spec = config["lqr"]
    env = Acrobot(frame_skip=2, nthread=2)
    rows = []
    result = dict(trials=rows, join_validation_passed=False, success=False)
    try:
        with h5py.File(out / "evidence.h5", "x") as raw:
            raw.attrs["integration_spec"] = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            locked = native_linearization(
                env, mujoco, raw.create_group("linearization"), budget, spec
            )
            write(out / "lqr.json", plain(locked))
            roster = [
                dict(
                    id=f"historical_{i}",
                    kind="historical",
                    state=x["initial_state"],
                    initial_context=x["initial_context"],
                )
                for i, x in enumerate(spec["historical_starts"])
            ]
            roster += basin_probe_roster(
                locked["P"],
                locked["unsaturated_rho"],
                shell_factors=tuple(spec["shell_factors"]),
                angle_marginals=tuple(spec["axis_angles"]),
                velocity_marginals=tuple(spec["axis_velocities"]),
            )
            roster += [
                dict(id=f"joint_grid_{i}", kind="joint_grid", state=x)
                for i, x in enumerate(spec["grid_states"])
            ]
            traces = raw.create_group("trials")

            def execute(case: dict) -> None:
                env.reset(state=np.asarray(case["state"]))
                budget.work.resets += 1
                budget.work.forwards += 1
                if "initial_context" in case:
                    current = _capture(mujoco, env)
                    if any(
                        current[k].tobytes() != np.asarray(v, dtype=float).tobytes()
                        for k, v in case["initial_context"].items()
                    ):
                        raise InvalidExecution("Historical near-upright context changed.")
                group = traces.create_group(case["id"])
                group.attrs["case_json"] = json.dumps(case)
                metrics = record_lqr(env, mujoco, group, budget, locked["K"], 500)
                row = {k: v for k, v in case.items() if k != "initial_context"} | metrics
                rows.append(row)
                group.attrs["captured"] = metrics["captured"]
                group.attrs["height_success"] = metrics["height_success"]
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(plain(row)) + "\n")
                if case["id"] == "historical_0":
                    result["replay"] = replay_prefix(env, mujoco, group, budget)
                if len(rows) % 24 == 0:
                    print(
                        json.dumps(
                            dict(
                                event="basin_progress",
                                cases=len(rows),
                                captured=sum(x["captured"] for x in rows),
                                resources=budget.observations(),
                            )
                        ),
                        flush=True,
                    )

            for case in roster:
                execute(case)
            factor = select_shell(rows, spec["shell_factors"])
            if factor is None:
                result.update(status="NO_VERIFIED_LOCAL_BASIN", lqr=plain(locked))
                return result
            rho = factor * locked["unsaturated_rho"]
            boundary = ellipsoid_starts(
                locked["P"], rho, independent_directions(spec["confirmation_boundary_seed"], 64)
            )
            interior = ellipsoid_starts(
                locked["P"], rho, independent_directions(spec["confirmation_interior_seed"], 64)
            )
            radii = (
                np.random.Generator(np.random.PCG64(spec["confirmation_radius_seed"])).uniform(
                    size=64
                )
                ** 0.25
            )
            interior[:, 1:] *= radii[:, None]
            for kind, starts in (("confirm_boundary", boundary), ("confirm_interior", interior)):
                for i, state in enumerate(starts):
                    execute(dict(id=f"{kind}_{i}", kind=kind, state=state.tolist()))
            confirmed = all(x["captured"] for x in rows if x["kind"].startswith("confirm_"))
            locked.update(
                rho=rho,
                selected_rho=rho,
                selected_shell_factor=factor,
                coordinate_projections=ellipsoid_coordinate_bounds(locked["P"], rho),
                join_validation_passed=confirmed,
                shells=[
                    dict(
                        factor=f,
                        captures=sum(x["captured"] for x in rows if x.get("factor") == f),
                        count=24,
                    )
                    for f in spec["shell_factors"]
                ],
            )
            write(out / "lqr.json", plain(locked))
            result.update(
                status="VERIFIED_EMPIRICAL_JOIN_REGION"
                if confirmed
                else "FAILED_BASIN_CONFIRMATION",
                join_validation_passed=confirmed,
                success=confirmed,
                historical_captures=sum(x["captured"] for x in rows if x["kind"] == "historical"),
                lqr=plain(locked),
            )
    finally:
        env.close()
    return result


def run(config_path: Path, stage: str, out: Path, lqr_path: Path | None = None) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("protocol") != "acrobot-transient-attractor" or config.get("version") != "1.0":
        raise ValueError("Unknown frozen attractor protocol.")
    spec = config["lqr"] if stage == "lqr" else config["stages"][stage]
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_bytes(config_path.read_bytes())
    budget = Budget(spec["limits"], out)
    result = dict(
        protocol=config["protocol"], version="1.0", stage=stage, status="STARTING", policy_fits=0
    )
    try:
        with resource_timers(spec["limits"], budget):
            inputs = _source_bindings(config, config_path)
            if not inputs.get("frozen_source_manifest"):
                raise InvalidExecution("Execution requires a frozen source export.")
            write(out / "inputs.json", inputs)
            write(out / "environment.json", _environment(dict(config, threads=spec["threads"])))
            if stage == "lqr":
                result.update(lqr_stage(config, out, budget))
            else:
                from src.gps.mppi_gps.acrobot_attractor_controller import run_near, run_swingup

                if lqr_path is None:
                    raise InvalidExecution(
                        "Bind the passing fixed-point lock before MPPI execution."
                    )
                locked = json.loads(lqr_path.read_text())
                if not locked.get("join_validation_passed"):
                    raise InvalidExecution(
                        "The local join region did not pass its frozen validation."
                    )
                result["lqr_binding"] = dict(path=str(lqr_path), sha256=sha(lqr_path))
                lqr_result = json.loads((lqr_path.parent / "result.json").read_text())
                if locked != lqr_result.get("lqr") or sha(lqr_path.parent / "config.json") != sha(
                    config_path
                ):
                    raise InvalidExecution(
                        "The LQR prerequisite differs from its recorded result or recipe."
                    )
                base = lqr_path.parent.parent
                if stage.startswith("swing"):
                    near = json.loads((base / "near/result.json").read_text())
                    if near.get("status") != "COMPLETED_ADAPTIVE_TEMPERATURE_PROBE" or not near.get(
                        "weighting_validated"
                    ):
                        raise InvalidExecution(
                            "Complete the validated temperature probe before acquisition."
                        )
                    if sha(base / "near/config.json") != sha(config_path):
                        raise InvalidExecution(
                            "The temperature prerequisite used another configuration."
                        )
                if stage == "swing-qualification":
                    development = json.loads((base / "swing-development/result.json").read_text())
                    if (
                        development.get("status") != "PASSED_JOIN_GATE"
                        or development.get("successes") != 2
                    ):
                        raise InvalidExecution(
                            "Fresh qualification requires two successful development joins and captures."
                        )
                    if sha(base / "swing-development/config.json") != sha(config_path):
                        raise InvalidExecution("Development used another configuration.")
                if stage == "near":
                    result.update(run_near(spec, out, budget))
                    result["status"] = "COMPLETED_ADAPTIVE_TEMPERATURE_PROBE"
                else:
                    result.update(run_swingup(spec, out, budget, locked))
                    successes = sum(bool(x["success"]) for x in result["trials"])
                    required = 2 if stage == "swing-development" else 9
                    result.update(
                        status="PASSED_JOIN_GATE" if successes >= required else "FAILED_JOIN_GATE",
                        successes=successes,
                        required=required,
                    )
    except ResourceLimit as error:
        result.update(status="STOPPED_RESOURCES", error=f"{type(error).__name__}: {error}")
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="STOPPED_INVALID_EXECUTION", error=f"{type(error).__name__}: {error}")
    result["additional_evidence"] = {
        name: dict(sha256=sha(out / name), bytes=(out / name).stat().st_size)
        for name in ("lqr.json", "hybrid_trials.jsonl")
        if (out / name).is_file()
    }
    return _finalize(out, result, budget)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("lqr", "near", "swing-development", "swing-qualification"),
        required=True,
    )
    parser.add_argument("--lqr", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.stage, args.out, args.lqr)
    print(json.dumps(plain(result)), flush=True)
    return 2 if result["status"].startswith("STOPPED_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
