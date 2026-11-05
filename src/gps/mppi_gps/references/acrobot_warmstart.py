"""Two frozen saved-prefix initializations of the existing Acrobot optimizer.

Preparation reads saved arrays only. Solver and native stages are independent,
create-only commands; neither constructs or changes an MPPI controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .acrobot_collocation import MODEL_SHA, ROOT, solve_attempt, symbolic_dynamics, write_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_file(binding: dict) -> Path:
    path = ROOT / binding["path"]
    if sha256(path) != binding["sha256"]:
        raise ValueError(f"Frozen file changed: {binding['path']}")
    return path


def wrap_state(states: np.ndarray) -> np.ndarray:
    wrapped = np.asarray(states, dtype=float).copy()
    wrapped[..., :2] = (wrapped[..., :2] + np.pi) % (2 * np.pi) - np.pi
    return wrapped


def select_prefix(states: np.ndarray, p: np.ndarray) -> tuple[int, np.ndarray]:
    wrapped = wrap_state(states)
    scores = np.einsum("ni,ij,nj->n", wrapped, p, wrapped)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite saved selection scores")
    return int(np.argmin(scores)), scores


def prepare(out: Path, config_path: Path, config: dict) -> dict:
    """Export deterministic saved-only inputs, including all selection scores."""
    import h5py

    started, cpu = time.monotonic(), time.process_time()
    binding_keys = (
        "saved_evidence",
        "saved_lqr",
        "saved_start_config",
        "reused_native_parity",
        "reused_drake_parity",
    )
    paths = {key: bound_file(config[key]) for key in binding_keys}
    lqr = json.loads(paths["saved_lqr"].read_text())
    p = np.asarray(lqr["P"], dtype=float)
    if lqr["model_sha256"] != MODEL_SHA or not np.linalg.eigvalsh(p).min() > 0:
        raise ValueError("Saved P binding failed")
    native = json.loads(paths["reused_native_parity"].read_text())
    drake = json.loads(paths["reused_drake_parity"].read_text())
    if (
        native["native_steps"] != 32
        or len(native["cases"]) != 32
        or max(row["max_error"] for row in native["cases"]) > 1e-9
        or not drake["passed"]
        or len(drake["rows"]) != 32
        or drake["cases_sha256"] != config["reused_native_parity"]["sha256"]
    ):
        raise ValueError("Reused equation/parity receipts failed")
    starts = json.loads(paths["saved_start_config"].read_text())["stages"]["swing-development"][
        "starts"
    ]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    with h5py.File(paths["saved_evidence"], "r") as saved:
        for index, episode in enumerate(config["episodes"]):
            group = saved[episode["group"]]
            states = np.asarray(group["states"])
            actions = np.asarray(group["actions"]).reshape(-1)
            context = starts[episode["start_index"]]["initial_context"]
            for name in ("states", "warmstarts", "integration"):
                if not np.array_equal(group[name][0], context[name]):
                    raise ValueError(f"Saved full initial context differs: {name}")
            if (
                states.shape != (len(actions) + 1, 5)
                or not np.isfinite(states).all()
                or not np.isfinite(actions).all()
                or np.max(np.abs(actions)) > 1
            ):
                raise ValueError("Invalid saved episode arrays")
            if not np.allclose(np.diff(states[:, 0]), 0.02, atol=1e-12, rtol=0):
                raise ValueError("Saved action rate differs from frozen 50 Hz")
            count, scores = select_prefix(states[:, 1:], p)
            if count != episode["expected_prefix_controls"] or count < 1:
                raise ValueError("Prospectively declared prefix selection differs")
            terminal = np.r_[2 * np.pi * np.rint(states[count, 1:3] / (2 * np.pi)), [0.0, 0.0]]
            path = out / f"prefix_{index}.npz"
            np.savez_compressed(
                path,
                states=states[: count + 1, 1:],
                saved_times=states[: count + 1, 0],
                saved_actions=actions[:count],
                torques_nm=np.repeat(2 * actions[:count], 2),
                initial_integration=np.asarray(context["integration"]),
                initial_warmstarts=np.asarray(context["warmstarts"]),
                terminal_state=terminal,
                selection_scores=scores,
            )
            records.append(
                dict(
                    path=path.name,
                    sha256=sha256(path),
                    group=episode["group"],
                    prefix_controls=count,
                    intervals=2 * count,
                    duration=0.02 * count,
                    selected_score=float(scores[count]),
                    selected_saved_state=states[count].tolist(),
                    initial_context=context,
                    terminal_state=terminal.tolist(),
                    source_episode_states=len(states),
                    source_episode_controls=len(actions),
                )
            )
    source_paths = [
        "src/__init__.py",
        "src/gps/__init__.py",
        "src/gps/mppi_gps/__init__.py",
        "src/gps/mppi_gps/references/__init__.py",
        "src/gps/mppi_gps/references/acrobot_collocation.py",
        "src/gps/mppi_gps/references/acrobot_warmstart.py",
        "assets/acrobot.xml",
        "assets/common/visual.xml",
        "assets/common/skybox.xml",
        "assets/common/materials.xml",
        "docs/acrobot_collocation_warmstart_protocol.md",
        str(config_path.resolve().relative_to(ROOT)),
        *[config[key]["path"] for key in binding_keys],
    ]
    manifest = dict(
        schema="acrobot-warmstart-frozen-inputs-v2.0",
        model_sha256=MODEL_SHA,
        files=[
            dict(path=name, sha256=sha256(ROOT / name), bytes=(ROOT / name).stat().st_size)
            for name in dict.fromkeys(source_paths)
        ],
        prefixes=records,
        reused_native_steps=32,
        reused_drake_eom_cases=32,
        new_native_steps=0,
        new_solver_attempts=0,
        cpu_seconds=time.process_time() - cpu,
        wall_seconds=time.monotonic() - started,
    )
    write_json(out / "manifest.json", manifest)
    return manifest


def verify_preparation(preparation: Path, config: dict, authorization: Path) -> dict:
    manifest_path = preparation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["model_sha256"] != MODEL_SHA or config["model_sha256"] != MODEL_SHA:
        raise ValueError("Model binding differs")
    for item in manifest["files"]:
        bound_file(item)
    for item in manifest["prefixes"]:
        if sha256(preparation / item["path"]) != item["sha256"]:
            raise ValueError("Exported prefix changed")
    launch = json.loads(authorization.read_text())
    if (
        launch.get("approved_manifest_sha256") != sha256(manifest_path)
        or launch.get("authorized") is not True
        or re.fullmatch(r"[0-9a-f]{40}", launch.get("source_commit", "")) is None
    ):
        raise ValueError("A committed-source launch authorization is required")
    for key in (
        "total_wall_seconds",
        "total_output_bytes",
        "native_max_steps",
        "solver_total_cpu_seconds",
    ):
        if launch.get("limits", {}).get(key) != config[key]:
            raise ValueError(f"Launch supervisor must enforce frozen {key}")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError(f"Set {name}=1 before Python starts")
    return manifest


def fine_guess(step: Any, prefix: Any) -> tuple[dict, dict]:
    saved, controls = prefix["states"], prefix["torques_nm"]
    n = len(controls)
    states = np.empty((n + 1, 4))
    states[::2] = saved
    for i in range(len(saved) - 1):
        states[2 * i + 1] = np.asarray(step(saved[i], controls[2 * i], 0.01)).reshape(4)
    defects = np.array(
        [
            states[i + 1] - np.asarray(step(states[i], controls[i], 0.01)).reshape(4)
            for i in range(n)
        ]
    )
    diagnostics = dict(
        physics_defect_max=float(np.max(np.abs(defects))),
        initial_state_max_error=float(np.max(np.abs(states[0] - saved[0]))),
        terminal_state_max_error=float(np.max(np.abs(states[-1] - prefix["terminal_state"]))),
        torque_max_abs=float(np.max(np.abs(controls))),
        native_endpoints_preserved=bool(np.array_equal(states[::2], saved)),
    )
    return dict(intervals=n, x=states.T, u=controls), diagnostics


def output_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def solve(out: Path, preparation: Path, config: dict, authorization: Path) -> dict:
    manifest = verify_preparation(preparation, config, authorization)
    import casadi as ca

    if ca.__version__ != config["casadi_version"]:
        raise ValueError("Frozen CasADi version differs")
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    write_json(
        out / "runtime.json",
        dict(
            casadi=ca.__version__,
            numpy=np.__version__,
            python=os.sys.version,
            manifest_sha256=sha256(preparation / "manifest.json"),
            authorization_sha256=sha256(authorization),
            config=config,
        ),
    )
    _, step = symbolic_dynamics(ca)
    records, stop = [], None
    for index, selected in enumerate(manifest["prefixes"]):
        elapsed = time.monotonic() - started + manifest["wall_seconds"]
        if (
            index >= config["solver_max_attempts"]
            or elapsed >= config["total_wall_seconds"]
            or output_bytes(out.parent) >= config["total_output_bytes"]
        ):
            stop = "FROZEN_RESOURCE_LIMIT"
            break
        with np.load(preparation / selected["path"], allow_pickle=False) as prefix:
            guess, diagnostics = fine_guess(step, prefix)
            np.savez_compressed(
                out / f"initial_guess_{index}.npz",
                states=guess["x"].T,
                torques_nm=guess["u"],
                terminal_state=prefix["terminal_state"],
            )
            write_json(out / f"initial_guess_{index}.json", diagnostics)
            attempt_config = dict(config, start=prefix["states"][0].tolist())
            result = solve_attempt(
                ca,
                out / f"prefix_{index}",
                attempt_config,
                duration=selected["duration"],
                shape=-1,
                fine=guess,
                terminal_state=prefix["terminal_state"],
                strict_torque_bounds=True,
            )
        # Feasibility alone never permits even a tiny out-of-bounds native control.
        with np.load(out / f"prefix_{index}" / "candidate.npz", allow_pickle=False) as candidate:
            strict = bool(
                np.isfinite(candidate["torques_nm"]).all()
                and np.max(np.abs(candidate["torques_nm"])) <= 2
            )
        records.append(
            dict(
                path=f"prefix_{index}",
                prefix_index=index,
                strictly_bounded_torques=strict,
                native_eligible=bool(result["feasible"] and strict),
                **result,
            )
        )
        if not np.isfinite(result["duration"]) or result["constraint_max_violation"] is None:
            stop = "INVALID_SOLVER_ARRAYS"
            break
    summary = dict(
        schema="acrobot-warmstart-solves-v2.0",
        records=records,
        stopped=stop,
        status="FEASIBLE_FOR_NATIVE_VERIFICATION"
        if any(r["native_eligible"] for r in records)
        else "NO_NATIVE_VERIFIABLE_SOLUTION",
        solver_attempts=len(records),
        native_steps=0,
        wall_seconds=time.monotonic() - started,
        cpu_seconds=time.process_time() - cpu,
        output_bytes=output_bytes(out.parent),
        global_optimality_established=False,
    )
    write_json(out / "result.json", summary)
    return summary


def native_verify(
    out: Path, preparation: Path, solve_dir: Path, config: dict, authorization: Path
) -> dict:
    manifest = verify_preparation(preparation, config, authorization)
    solved = json.loads((solve_dir / "result.json").read_text())
    if solved["stopped"] is not None:
        raise ValueError("Solver route stopped; native execution is prohibited")
    import mujoco

    if mujoco.__version__ != config["native_mujoco_version"]:
        raise ValueError("Frozen native MuJoCo version differs")
    model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/acrobot.xml"))
    if (
        model.opt.timestep != 0.01
        or int(model.opt.integrator) != int(mujoco.mjtIntegrator.mjINT_RK4)
        or not np.array_equal(model.actuator_gear[:, 0], [2])
        or not np.array_equal(model.actuator_ctrlrange, [[-1, 1]])
    ):
        raise ValueError("Frozen RK4/torque binding differs")
    out.mkdir(exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    steps, records, stop = 0, [], None
    hold = round(config["hold_seconds"] / 0.01)
    for candidate in solved["records"]:
        if not candidate["native_eligible"]:
            continue
        index = candidate["prefix_index"]
        with np.load(preparation / f"prefix_{index}.npz", allow_pickle=False) as prefix:
            context, start = prefix["initial_integration"].copy(), prefix["states"][0].copy()
            warmstart = prefix["initial_warmstarts"].copy()
        values = np.load(solve_dir / candidate["path"] / "candidate.npz", allow_pickle=False)
        controls = np.r_[values["torques_nm"], np.zeros(hold)]
        if (
            not np.isfinite(controls).all()
            or np.max(np.abs(controls)) > 2
            or steps + len(controls) > config["native_max_steps"]
        ):
            raise ValueError("Strict physical torque or native step budget failed")
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, context, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, context, mujoco.mjtState.mjSTATE_INTEGRATION)
        restored = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(model, data, restored, mujoco.mjtState.mjSTATE_INTEGRATION)
        if (
            restored.tobytes() != context.tobytes()
            or np.asarray(data.qacc_warmstart).tobytes() != warmstart.tobytes()
            or np.r_[data.qpos, data.qvel].tobytes() != start.tobytes()
        ):
            raise ValueError("Native full-context start differs")
        states, contexts = [start], [context]
        for torque in controls:
            wall = manifest["wall_seconds"] + solved["wall_seconds"] + time.monotonic() - started
            if (
                wall >= config["total_wall_seconds"]
                or output_bytes(out.parent) >= config["total_output_bytes"]
            ):
                stop = "FROZEN_RESOURCE_LIMIT"
                break
            data.ctrl[0] = torque / 2
            steps += 1
            mujoco.mj_step(model, data)
            state = np.r_[data.qpos, data.qvel]
            states.append(state)
            current = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, current, mujoco.mjtState.mjSTATE_INTEGRATION)
            contexts.append(current)
            if not np.isfinite(state).all() or np.max(np.abs(state[2:])) > 50:
                stop = "INVALID_NATIVE_PHYSICS"
                break
        xx = np.asarray(states)
        applied = controls[: len(xx) - 1]
        np.savez_compressed(
            out / f"prefix_{index}.npz",
            states=xx,
            torques_nm=applied,
            integration=np.asarray(contexts),
        )
        if stop:
            records.append(
                dict(
                    candidate=candidate["path"],
                    captured=False,
                    executed_steps=len(applied),
                    stopped=stop,
                )
            )
            break
        height = 2 + np.cos(xx[:, 0]) + np.cos(xx[:, 0] + xx[:, 1])
        distance_squared = (np.sin(xx[:, 0]) + np.sin(xx[:, 0] + xx[:, 1])) ** 2 + (height - 4) ** 2
        costs = (
            distance_squared[1:]
            + 0.01 * np.sum(xx[1:, 2:] ** 2, axis=1)
            + 0.01 * (applied / 2) ** 2
        )
        rewards = np.exp(np.log(0.1) * np.maximum(np.sqrt(distance_squared[1:]) - 0.2, 0) ** 2)
        angle = float(np.max(np.abs(wrap_state(xx[-hold:])[:, :2])))
        speed = float(np.max(np.abs(xx[-hold:, 2:])))
        minimum_height = float(np.min(height[-hold:]))
        records.append(
            dict(
                candidate=candidate["path"],
                prefix_index=index,
                executed_steps=len(applied),
                captured=bool(
                    minimum_height >= config["minimum_hold_height"]
                    and angle <= config["maximum_hold_angle_error"]
                    and speed <= config["maximum_hold_speed"]
                ),
                minimum_hold_height=minimum_height,
                maximum_hold_angle_error=angle,
                maximum_hold_speed=speed,
                final_state=xx[-1].tolist(),
                terminal_state=xx[len(values["torques_nm"])].tolist(),
                native_solver_state_max_error=float(
                    np.max(np.abs(xx[: len(values["states"])] - values["states"]))
                ),
                torque_effort_integral=float(0.01 * np.sum(applied**2)),
                dense_cost_sum=float(np.sum(costs)),
                dense_cost_integral=float(0.01 * np.sum(costs)),
                mean_dm_control_reward=float(np.mean(rewards)),
                sum_dm_control_reward=float(np.sum(rewards)),
            )
        )
        values.close()
    result = dict(
        schema="acrobot-warmstart-native-v2.0",
        records=records,
        stopped=stop,
        status="REFERENCE_EXECUTES"
        if any(r["captured"] for r in records)
        else "NO_EXECUTABLE_REFERENCE",
        native_steps=steps,
        new_parity_native_steps=0,
        reused_parity_native_steps=32,
        model_sha256=MODEL_SHA,
        mujoco_version=mujoco.__version__,
        wall_seconds=time.monotonic() - started,
        cpu_seconds=time.process_time() - cpu,
        total_wall_seconds=manifest["wall_seconds"]
        + solved["wall_seconds"]
        + time.monotonic()
        - started,
        output_bytes=output_bytes(out.parent),
        standard_start_evaluation=False,
        global_optimality_established=False,
        certificate_established=False,
        interpretation="A passing replay is a feasible fixed-duration upper bound only.",
    )
    write_json(out / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "solve", "native"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preparation", type=Path)
    parser.add_argument("--solve-dir", type=Path)
    parser.add_argument("--authorization", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "prepare":
        result = prepare(args.out, args.config, config)
    else:
        if args.preparation is None or args.authorization is None:
            parser.error("solve/native require --preparation and --authorization")
        if args.stage == "solve":
            result = solve(args.out, args.preparation, config, args.authorization)
        else:
            if args.solve_dir is None:
                parser.error("native requires --solve-dir")
            result = native_verify(
                args.out, args.preparation, args.solve_dir, config, args.authorization
            )
    print(json.dumps(dict(status=result.get("status", "PREPARED")), sort_keys=True))


if __name__ == "__main__":
    main()
