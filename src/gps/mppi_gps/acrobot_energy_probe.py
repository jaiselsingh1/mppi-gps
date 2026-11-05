"""Versioned Acrobot balance/discovery probes, reusing the bounded teacher recorder.

The preflight runs once. Two separately budgeted worker parts can then execute
concurrently from the same immutable source export and preflight evidence. No
policy, fitting, qualification expansion or alternating loop is launched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import signal
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.gps.mppi_gps.acrobot_teacher import (
    Budget,
    InvalidExecution,
    ResourceLimit,
    _capture,
    _environment,
    _preflight,
    _run_trial,
    _save_array,
    _source_bindings,
)
from src.gps.mppi_gps.provenance import fingerprint, sha, write
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "acrobot-energy-discovery-probe"
VERSION = "1.1"
RUNNING_NAMES = (
    "pose",
    "weighted_energy",
    "weighted_balance",
    "weighted_excess",
    "control_cost",
    "running",
)
ENERGY_NAMES = (
    "potential",
    "kinetic",
    "total_energy",
    "energy_error",
    "gate",
    "scaled_velocity_squared",
    "excess_velocity_squared",
)
PART_DESIGNS = {
    "near-upright": (2, 500, 4, ((150, 256, 3.0, 0.6, None),)),
    "energy": (
        4,
        600,
        2,
        (
            (300, 512, 0.15, 0.18553280373949055, 160 / 9),
            (400, 512, 0.15, 0.18553280373949055, 160 / 9),
            (400, 512, 0.15, 0.18553280373949055, 320 / 9),
        ),
    ),
}


def validate_config(config: dict[str, Any]) -> None:
    """Require the declared finite roster, success metric, controls and caps."""
    if config.get("protocol") != PROTOCOL or config.get("version") != VERSION:
        raise ValueError("Unknown Acrobot energy probe version.")
    expected_task = dict(
        frame_skip=2, dt=0.02, reset_std=0.05, balance_height=3.8, balance_seconds=2.0
    )
    if config.get("task") != expected_task or config.get("math_threads") != 1:
        raise ValueError("Task timing, reset scale, height test and math threads are frozen.")
    if set(config.get("parts", {})) != set(PART_DESIGNS):
        raise ValueError("Declare exactly the near-upright and energy parts.")
    all_seeds = []
    for name, (threads, controls, starts, variants) in PART_DESIGNS.items():
        part = config["parts"][name]
        if (part.get("threads"), part.get("controls"), len(part.get("starts", []))) != (
            threads,
            controls,
            starts,
        ):
            raise ValueError(f"Frozen {name} part dimensions changed.")
        if len(part.get("variants", [])) != len(variants):
            raise ValueError(f"Frozen {name} variant roster changed.")
        for variant, expected in zip(part["variants"], variants):
            planner = MPPIConfig(**variant["planner"])
            actual = (
                planner.H,
                planner.K,
                planner.lam,
                planner.noise_sigma,
                variant.get("energy_weight"),
            )
            if actual != expected:
                raise ValueError("Planner and energy-weight recipes are frozen.")
            if (
                planner.initial_iterations != 3
                or planner.use_is_correction
                or planner.noise_lowpass_cutoff_hz is not None
                or planner.lam_target_neff_fraction is not None
                or planner.noise_std is not None
                or planner.noise_cov is not None
            ):
                raise ValueError("Use the declared white-noise, fixed-temperature MPPI recipe.")
            if variant.get("kind") != ("dense" if name == "near-upright" else "energy"):
                raise ValueError("Cost profile differs from the declared part.")
        for start in part["starts"]:
            state = np.asarray(start.get("initial_state"), dtype=np.float64)
            if state.shape != (5,) or not np.isfinite(state).all() or state[0] != 0:
                raise ValueError("Every start needs a finite explicit five-component state.")
            seed = start.get("reset_seed")
            if type(seed) is not int or seed < 0 or start.get("planner_seed") != seed + 10000:
                raise ValueError("Frozen private reset/planner seed pairs are required.")
            all_seeds.append(seed)
        expected_cpu = 1200 if name == "near-upright" else 9600
        expected_output = (128 if name == "near-upright" else 640) * 1024**2
        if part["limits"]["cpu_seconds"] != expected_cpu:
            raise ValueError("Part CPU caps are frozen.")
        if (
            part["limits"].get("wall_seconds") != (1800 if name == "near-upright" else 5400)
            or part["limits"].get("rss_bytes") != 4 * 1024**3
        ):
            raise ValueError("Part wall and memory caps are frozen.")
        if part["limits"]["output_bytes"] != expected_output:
            raise ValueError("Part output caps are frozen.")
        queries = starts * sum(
            controls + v["planner"]["initial_iterations"] - 1 for v in part["variants"]
        )
        forecast = starts * sum(
            (controls + v["planner"]["initial_iterations"] - 1)
            * v["planner"]["K"]
            * v["planner"]["H"]
            for v in part["variants"]
        )
        native = starts * len(variants) * controls
        if queries <= 0 or (
            part["limits"]["native_controls"] != native
            or part["limits"]["forecast_controls"] != forecast
            or part["limits"]["physics_substeps"] != 2 * (native + forecast)
        ):
            raise ValueError("Part census does not cover exactly the frozen call roster.")
    if len(set(all_seeds)) != 6:
        raise ValueError("The two parts must have six disjoint reset seeds.")
    preflight = config.get("preflight", {})
    if (
        preflight.get("controls") != 12
        or preflight.get("atol") != 1e-10
        or preflight.get("threads") != 2
        or preflight.get("energy_validation_random_states") != 28
        or len(preflight.get("profiles", [])) != 3
    ):
        raise ValueError("Replay tolerance and the three-profile preflight are frozen.")
    profiles = [(x.get("kind"), x.get("energy_weight")) for x in preflight["profiles"]]
    if profiles != [("dense", None), ("energy", 160 / 9), ("energy", 320 / 9)]:
        raise ValueError("Preflight must bind all three native cost profiles.")
    if (
        preflight["limits"]["native_controls"] != 108
        or preflight["limits"]["physics_substeps"] != 216
    ):
        raise ValueError("Preflight physics reservation is frozen.")
    states = np.asarray(preflight.get("energy_validation_states"), dtype=float)
    if states.shape != (32, 5) or not np.isfinite(states).all():
        raise ValueError("Freeze all32 analytic/native energy validation states before launch.")
    if [profile.get("threads") for profile in preflight["profiles"]] != [2, 4, 4]:
        raise ValueError("Preflight profile threads must match their execution workers.")
    expected_cost = dict(
        energy_upright=58.86,
        energy_hanging=19.62,
        energy_scale=39.24,
        energy_error="signed squared normalized error",
        energy_gate="1 - proximity_reward**2",
        pose="1-clip(z/4,0,1)+0.05*x**2",
        proximity=dict(target_radius=0.2, margin=2.0, value_at_margin=0.1),
        balance_weight=0.15,
        velocity_scale=[2.5, 5.0],
        excess_weight=0.05,
        excess_threshold=8.0,
        control_weight=0.0,
        terminal_pose_weight=100.0,
        terminal_velocity_weight=25.0,
    )
    if config.get("energy_cost") != expected_cost:
        raise ValueError("The energy objective and balance handoff are frozen.")
    if config.get("numerics") != dict(
        replay_rtol=0.0, replay_atol=1e-10, cost_absolute_floor=1e-10, cost_relative_fp64_ulps=64
    ):
        raise ValueError("Replay and component-roundoff bounds are frozen.")
    if (
        config.get("selection_rule")
        != "first energy variant in declared roster with both successes; no cross-objective cost selection"
    ):
        raise ValueError("First successful variant rule must not select on cross-objective costs.")
    if config.get("diagnostics") != dict(
        energy_match_normalized_error=0.05,
        low_height=3.0,
        saved_candidate_traces=["minimum_task_cost", "maximum_height"],
        all_candidate_components=True,
    ):
        raise ValueError("Candidate recording and energy-degeneracy diagnostics are frozen.")
    if preflight["limits"] != dict(
        cpu_seconds=120,
        wall_seconds=600,
        output_bytes=16 * 1024**2,
        rss_bytes=4 * 1024**3,
        native_controls=108,
        forecast_controls=1,
        physics_substeps=216,
    ):
        raise ValueError("All preflight resource reservations are frozen.")
    shared = config.get("shared_limits", {})
    if (
        shared.get("cpu_seconds") != 11160
        or shared.get("output_bytes") != 2 * 1024**3
        or shared.get("physics_substeps") != 1510411416
        or shared.get("network_forwards") != 0
        or shared.get("optimizer_updates") != 0
    ):
        raise ValueError("Shared work/output ceilings are frozen.")
    if "no energy variant succeeds on both" not in config.get("kill_rule", ""):
        raise ValueError("The two-of-two teacher kill rule must be declared.")
    if "original nine-of-ten" not in config.get("admission", ""):
        raise ValueError("Exploratory two-of-two must not replace original qualification.")


def decision(rows: list[dict[str, Any]], variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify all six complete attempts; no proxy screens or automatic expansion."""
    successful = []
    counts = {}
    for index, variant in enumerate(variants):
        found = [row for row in rows if row["recipe_index"] == index]
        if len(found) != 2 or any(row["controls"] != 600 for row in found):
            raise InvalidExecution("The kill decision requires all six complete energy episodes.")
        count = sum(bool(row["success"]) for row in found)
        counts[variant["name"]] = count
        if count == 2:
            successful.append(variant["name"])
    return dict(
        status="PROMISING_TEACHER_PROBE_ONLY" if successful else "TEACHER_PROBLEM_STOP",
        successes_by_variant=counts,
        promising_variants=successful,
        selected_variant=successful[0] if successful else None,
        teacher_qualified=False,
        learning_or_loop_launched=False,
        interpretation="Exploratory two-of-two only; original fresh nine-of-ten qualification remains required."
        if successful
        else "No energy variant met two-of-two; this is an Acrobot teacher problem, not a loop result.",
    )


@contextmanager
def resource_timers(limits: dict[str, int], budget: Budget):
    """Bound one process; separate part caps sum to the declared shared ceiling."""
    used_resources = budget.observations()
    cpu_remaining = limits["cpu_seconds"] - used_resources["cpu_seconds"]
    wall_remaining = limits["wall_seconds"] - used_resources["wall_seconds"]
    if cpu_remaining <= 0 or wall_remaining <= 0:
        raise ResourceLimit("Source preparation exhausted the frozen process reservation.")
    previous_alarm = signal.signal(signal.SIGALRM, _timer_failure)
    previous_cpu_handler = previous_cpu_limits = None
    signal.alarm(max(1, math.ceil(wall_remaining)))
    if sys.platform == "linux":
        previous_cpu_handler = signal.signal(signal.SIGXCPU, _timer_failure)
        previous_cpu_limits = resource.getrlimit(resource.RLIMIT_CPU)
        used = resource.getrusage(resource.RUSAGE_SELF)
        soft = math.ceil(used.ru_utime + used.ru_stime + cpu_remaining)
        if previous_cpu_limits[1] != resource.RLIM_INFINITY:
            soft = min(soft, previous_cpu_limits[1])
        resource.setrlimit(resource.RLIMIT_CPU, (soft, previous_cpu_limits[1]))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if previous_cpu_handler is not None:
            signal.signal(signal.SIGXCPU, previous_cpu_handler)
        if previous_cpu_limits is not None:
            resource.setrlimit(resource.RLIMIT_CPU, previous_cpu_limits)


def _timer_failure(signum: int, frame: Any) -> None:
    del frame
    raise ResourceLimit(f"Frozen process resource timer signalled {signum}.")


def _setup(config_path: Path, out: Path, budget: Budget, threads: int):
    config = json.loads(config_path.read_text())
    validate_config(config)
    (out / "config.json").write_bytes(config_path.read_bytes())
    (out / "trials.jsonl").touch(exist_ok=False)
    inputs = _source_bindings(config, config_path)
    if not inputs.get("frozen_source_manifest"):
        raise InvalidExecution("Native execution requires an immutable source export manifest.")
    write(out / "inputs.json", inputs)
    write(out / "environment.json", _environment(dict(config, threads=threads)))
    return config, budget


def _finalize(out: Path, result: dict[str, Any], budget: Budget) -> dict[str, Any]:
    evidence_path = out / "evidence.h5"
    if evidence_path.exists():
        import h5py

        with h5py.File(evidence_path, "r+") as evidence:
            evidence.attrs["status"] = result["status"]
    result["work"] = asdict(budget.work)
    if result.get("part") == "preflight":
        result["possible_partial_energy_validation"] = bool(
            result.get("native_energy_validation_attempted")
            and not result.get("native_energy_validation_complete")
        )
    result["unknown_partial_work"] = {
        kind: getattr(budget.work, kind + "_attempted") - getattr(budget.work, kind + "_completed")
        for kind in (
            "native_controls",
            "verification_controls",
            "forecast_controls",
            "model_queries",
            "physics_substeps",
            "restores",
        )
    }
    result["evidence"] = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()
    }
    result["resources"] = budget.observations()
    write(out / "result.json", result)
    return result


def _new_env(profile: dict[str, Any], threads: int):
    from src.envs.acrobot import Acrobot
    from src.envs.acrobot_energy import EnergyAcrobot

    if profile["kind"] == "dense":
        return Acrobot(frame_skip=2, nthread=threads, reset_std=0.05)
    return EnergyAcrobot(
        weight=profile["energy_weight"], frame_skip=2, nthread=threads, reset_std=0.05
    )


def preflight(config_path: Path, out: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    validate_config(config)
    limits = config["preflight"]["limits"]
    result = dict(
        protocol=PROTOCOL, version=VERSION, part="preflight", status="STARTING", profiles=[]
    )
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(limits, out)
    try:
        config, budget = _setup(config_path, out, budget, config["preflight"]["threads"])
        with resource_timers(limits, budget):
            import h5py
            import mujoco

            from src.envs.acrobot_energy import native_energy_validation

            with h5py.File(out / "evidence.h5", "x") as evidence:
                evidence.attrs["integration_spec"] = int(mujoco.mjtState.mjSTATE_INTEGRATION)
                env = _new_env(config["preflight"]["profiles"][1], 4)
                try:
                    states = np.asarray(
                        config["preflight"]["energy_validation_states"], dtype=np.float64
                    )
                    budget.work.restores_attempted += 1
                    result["native_energy_validation_attempted"] = True
                    result["native_energy_validation_complete"] = False
                    validation = native_energy_validation(env, atol=1e-10, states=states)
                    group = evidence.create_group("energy_validation")
                    metadata = {}
                    for name, value in validation.items():
                        if isinstance(value, np.ndarray) or name in (
                            "states",
                            "analytic_potential_kinetic",
                            "native_potential_kinetic",
                        ):
                            _save_array(group, name, np.asarray(value), budget)
                        else:
                            metadata[name] = value
                    _save_array(group, "validation_states", states, budget)
                    budget.work.forwards += validation["native_forwards"]
                    budget.work.restores_completed += 1
                    write(out / "energy_validation.json", metadata)
                    result["energy_validation"] = metadata
                    result["native_energy_validation_complete"] = True
                finally:
                    env.close()
                for profile in config["preflight"]["profiles"]:
                    env = _new_env(profile, profile["threads"])
                    try:
                        row = _preflight(
                            mujoco,
                            env,
                            evidence.create_group("profiles/" + profile["name"]),
                            budget,
                            config["preflight"],
                            include_terminal_cost=profile["kind"] == "energy",
                        )
                        result["profiles"].append(
                            dict(name=profile["name"], threads=profile["threads"], **row)
                        )
                    finally:
                        env.close()
                evidence.flush()
        result["status"] = "PASSED"
    except ResourceLimit as error:
        result.update(status="STOPPED_RESOURCES", error=f"{type(error).__name__}: {error}")
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="STOPPED_INVALID_EXECUTION", error=f"{type(error).__name__}: {error}")
    return _finalize(out, result, budget)


def verify_preflight(
    preflight_path: Path, config_path: Path, inputs: dict[str, Any]
) -> dict[str, Any]:
    """Bind already-paid replay to this exact configuration and source export."""
    required = ("result.json", "inputs.json", "environment.json", "config.json", "evidence.h5")
    if any(not (preflight_path / name).is_file() for name in required):
        raise InvalidExecution("Required preserved preflight evidence is missing.")
    result = json.loads((preflight_path / "result.json").read_text())
    if (
        result.get("protocol") != PROTOCOL
        or result.get("version") != VERSION
        or result.get("part") != "preflight"
        or result.get("status") != "PASSED"
        or len(result.get("profiles", [])) != 3
        or any(row.get("status") != "PASSED" for row in result["profiles"])
    ):
        raise InvalidExecution("The complete three-profile preflight must pass before execution.")
    config = json.loads(config_path.read_text())
    expected_profiles = [(p["name"], p["threads"]) for p in config["preflight"]["profiles"]]
    if [(p.get("name"), p.get("threads")) for p in result["profiles"]] != expected_profiles:
        raise InvalidExecution("Preflight cost/thread profile roster changed.")
    expected_work = dict(
        verification_controls=108,
        physics_substeps=216,
        restores=4,
        native_controls=0,
        forecast_controls=0,
        model_queries=0,
    )
    for field, number in expected_work.items():
        if any(
            result.get("work", {}).get(field + "_" + suffix) != number
            for suffix in ("attempted", "completed")
        ):
            raise InvalidExecution("Preflight work census is incomplete: " + field)
    if any(
        result.get("work", {}).get(key) != 0
        for key in ("network_forwards", "backward_calls", "optimizer_updates")
    ):
        raise InvalidExecution("Preflight contains unauthorized learning work.")
    energy = result.get("energy_validation", {})
    if (
        energy.get("status") != "PASSED"
        or not energy.get("incoming_context_restored_exactly")
        or energy.get("native_forwards") != 33
        or not result.get("native_energy_validation_complete")
    ):
        raise InvalidExecution("The complete32-state native energy validation must pass.")
    previous_inputs = json.loads((preflight_path / "inputs.json").read_text())
    if sha(preflight_path / "config.json") != sha(config_path):
        raise InvalidExecution("Preflight configuration binding changed.")
    for key in ("config_sha256", "files", "frozen_source_manifest"):
        if previous_inputs.get(key) != inputs.get(key):
            raise InvalidExecution(f"Preflight source binding changed: {key}.")
    environment = json.loads((preflight_path / "environment.json").read_text())
    if (
        environment.get("versions") != config["expected_environment"]
        or environment.get("physics_threads") != 2
        or set(environment.get("math_threads", {}).values()) != {"1"}
    ):
        raise InvalidExecution("Preflight native runtime binding changed.")
    for name, binding in result["evidence"].items():
        target = preflight_path / name
        if (
            not target.is_file()
            or sha(target) != binding["sha256"]
            or target.stat().st_size != binding["bytes"]
        ):
            raise InvalidExecution(f"Preserved preflight output changed: {name}.")
    if any(result.get("unknown_partial_work", {}).values()):
        raise InvalidExecution("Preflight includes unknown partial native work.")
    return dict(
        result_sha256=sha(preflight_path / "result.json"),
        evidence_sha256=sha(preflight_path / "evidence.h5"),
        config_sha256=sha(config_path),
        new_physics_repeated=0,
    )


def _dense_cost(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    from src.gps.mppi_gps.acrobot_energy import tip_position

    position = np.sum((tip_position(states) - [0, 0, 4]) ** 2, axis=-1)
    velocity = 0.01 * np.sum(states[..., 3:5] ** 2, axis=-1)
    control = 0.01 * np.sum(actions**2, axis=-1)
    return position + velocity + control


def native_metrics(
    env: Any, states: np.ndarray, costs: np.ndarray, actions: np.ndarray
) -> dict[str, Any]:
    from src.gps.mppi_gps.acrobot_energy import energy_components, tip_position

    n = len(costs)
    if states.shape != (n, 5) or actions.shape != (n, 1) or n not in (500, 600):
        raise InvalidExecution("An episode needs its complete frozen native trace.")
    if not all(np.isfinite(x).all() for x in (states, costs, actions)):
        raise InvalidExecution("Nonfinite native evidence is not an ordinary task failure.")
    height = tip_position(states)[:, 2]
    high = height >= 3.8
    hold = 100
    above = np.flatnonzero(high)
    starts = np.flatnonzero(np.convolve(high.astype(int), np.ones(hold, int), "valid") == hold)
    boundaries = np.flatnonzero(np.r_[False, high, False][1:] != np.r_[False, high, False][:-1])
    longest = int(np.max(boundaries[1::2] - boundaries[::2], initial=0))
    energies = energy_components(states)
    matched = np.abs((energies["total_energy"] - 58.86) / 39.24) <= 0.05
    terminal = float(np.asarray(env.terminal_cost(states[-1])).item())
    running = float(costs.sum())
    return dict(
        controls=n,
        success=bool(high[-hold:].all()),
        task_cost=running + terminal,
        native_running_cost_sum=running,
        episode_terminal_cost=terminal,
        original_dense_cost_sum=float(_dense_cost(states, actions).sum()),
        maximum_tip_height=float(height.max()),
        final_window_minimum_height=float(height[-hold:].min()),
        first_tip_above_threshold_seconds=float((above[0] + 1) * 0.02) if len(above) else None,
        first_complete_hold_start_seconds=float((starts[0] + 1) * 0.02) if len(starts) else None,
        time_balanced_seconds=float(high.sum() * 0.02),
        longest_height_interval_seconds=longest * 0.02,
        final_window_velocity_rms=float(np.sqrt(np.mean(states[-hold:, 3:5] ** 2))),
        terminal_angles_raw=states[-1, 1:3].tolist(),
        terminal_winding=np.floor((states[-1, 1:3] + np.pi) / (2 * np.pi)).astype(int).tolist(),
        terminal_potential=float(energies["potential"][-1]),
        terminal_kinetic=float(energies["kinetic"][-1]),
        terminal_total_energy=float(energies["total_energy"][-1]),
        energy_matched_low_height_controls=int((matched & (height < 3.0)).sum()),
        energy_matched_controls=int(matched.sum()),
        seconds_before_first_below_height=float(np.flatnonzero(~high)[0] * 0.02)
        if (~high).any()
        else n * 0.02,
    )


def record_bank(group: Any, env: Any, planner: Any, state: np.ndarray, budget: Budget) -> None:
    """Add saved-only diagnostics; no samples, simulation or random draws are generated."""
    from src.gps.mppi_gps.acrobot_energy import energy_components, tip_position

    states, actions = planner._last_states, planner._last_actions
    if states.ndim != 3 or states.shape[-1] != 5 or actions.shape != states.shape[:2] + (1,):
        raise InvalidExecution("Candidate bank has incompatible native dimensions.")
    if hasattr(env, "cost_components"):
        parts = env.cost_components(states, actions)
    else:
        energy = energy_components(states)
        zero = np.zeros(states.shape[:-1])
        pose = np.sum((tip_position(states) - [0, 0, 4]) ** 2, axis=-1)
        velocity = 0.01 * np.sum(states[..., 3:5] ** 2, axis=-1)
        control = 0.01 * np.sum(actions**2, axis=-1)
        parts = dict(
            energy,
            pose=pose,
            weighted_energy=zero,
            weighted_balance=velocity,
            weighted_excess=zero,
            control_cost=control,
            running=_dense_cost(states, actions),
            energy_error=(energy["total_energy"] - 58.86) / 39.24,
            gate=zero,
            scaled_velocity_squared=np.sum((states[..., 3:5] / [2.5, 5]) ** 2, axis=-1),
            excess_velocity_squared=np.sum(
                (np.maximum(np.abs(states[..., 3:5]) - 8, 0) / 8) ** 2, axis=-1
            ),
        )
    terminal = np.asarray(env.terminal_cost(states[:, -1]))
    running_sum = np.asarray(parts["running"]).sum(axis=1)
    primary = running_sum + terminal
    if not np.allclose(primary, planner._last_costs, rtol=0, atol=1e-10):
        raise InvalidExecution("Native primary running/terminal candidate cost binding failed.")
    additive = sum(np.asarray(parts[key]).sum(axis=1) for key in RUNNING_NAMES[:-1]) + terminal
    allowance = 1e-10 + 64 * np.finfo(np.float64).eps * np.maximum(1, np.abs(primary))
    grouping_error = additive - primary
    if np.any(np.abs(grouping_error) > allowance):
        raise InvalidExecution("Candidate component grouping exceeds the declared FP64 bound.")
    subgroup = group.create_group("components")
    subgroup.attrs["running_names"] = json.dumps(RUNNING_NAMES)
    subgroup.attrs["running_sums_columns"] = json.dumps(RUNNING_NAMES)
    subgroup.attrs["energy_names"] = json.dumps(ENERGY_NAMES)
    subgroup.attrs["energy_sums_columns"] = json.dumps(ENERGY_NAMES)
    values = dict(
        running_sums=np.stack(
            [np.asarray(parts[key]).sum(axis=1) for key in RUNNING_NAMES], axis=1
        ),
        energy_sums=np.stack([np.asarray(parts[key]).sum(axis=1) for key in ENERGY_NAMES], axis=1),
        primary_running_sum=running_sum,
        terminal=terminal,
        primary_total_cost=primary,
        component_grouping_error=grouping_error,
        component_grouping_allowance=allowance,
    )
    height = tip_position(states)[..., 2]
    values.update(
        peak_height=height.max(axis=1), final_window_minimum_height=height[:, -100:].min(axis=1)
    )
    values["height_qualified"] = (height[:, -100:] >= 3.8).all(axis=1)
    for key in ("potential", "kinetic", "total_energy", "energy_error", "gate"):
        values["terminal_" + key] = np.asarray(parts[key])[:, -1]
    matched = np.abs(parts["energy_error"]) <= 0.05
    values["energy_matched_controls"] = matched.sum(axis=1)
    values["energy_matched_low_height_controls"] = (matched & (height < 3.0)).sum(axis=1)
    for name, value in values.items():
        _save_array(subgroup, name, value, budget)
    group.attrs["action_bank_sha256"] = hashlib.sha256(
        np.ascontiguousarray(actions).tobytes()
    ).hexdigest()
    subgroup.attrs["candidate_actions_sha256"] = group.attrs["action_bank_sha256"]
    group.attrs["action_bank_fingerprint"] = fingerprint(np.asarray(actions))
    group.attrs["action_bank_shape"] = json.dumps(actions.shape)
    group.attrs["action_bank_dtype"] = str(actions.dtype)
    group.attrs["rng_state_after_fingerprint"] = fingerprint(planner._rng.bit_generator.state)
    group.attrs["rng_state_after_json"] = json.dumps(planner._rng.bit_generator.state)
    indices = np.array([np.argmin(primary), np.argmax(height.max(axis=1))], dtype=np.int64)
    _save_array(subgroup, "selected_indices", indices, budget)
    _save_array(subgroup, "selected_states", states[indices], budget)
    _save_array(subgroup, "selected_actions", actions[indices], budget)
    subgroup.attrs["selected_roles"] = json.dumps(["minimum_task_cost", "maximum_height"])
    group.attrs["candidate_height_scope"] = (
        "Final100 POST boundaries of the planning horizon, not full-episode admission"
    )
    mujoco = sys.modules.get("mujoco")
    if mujoco is None:
        raise InvalidExecution("Native bank recording requires the initialized simulator module.")
    context = group.create_group("pre_context")
    captured = _capture(mujoco, env)
    if captured["states"].tobytes() != np.asarray(state).tobytes():
        raise InvalidExecution("Candidate diagnostics changed their native pre-action context.")
    for name, value in captured.items():
        _save_array(context, name, value, budget)


def execute(config_path: Path, part_name: str, prerequisite: Path, out: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    validate_config(config)
    if part_name not in PART_DESIGNS:
        raise ValueError("Unknown probe part.")
    part = config["parts"][part_name]
    result = dict(protocol=PROTOCOL, version=VERSION, part=part_name, status="STARTING", trials=[])
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(part["limits"], out)
    try:
        config, budget = _setup(config_path, out, budget, part["threads"])
        inputs = json.loads((out / "inputs.json").read_text())
        result["preflight_binding"] = verify_preflight(prerequisite, config_path, inputs)
        with resource_timers(part["limits"], budget):
            import h5py
            import mujoco

            from src.mppi.mppi import MPPI

            with h5py.File(out / "evidence.h5", "x") as evidence:
                evidence.attrs["integration_spec"] = int(mujoco.mjtState.mjSTATE_INTEGRATION)
                for index, variant in enumerate(part["variants"]):
                    env = _new_env(variant, part["threads"])
                    try:
                        for trial_index, start in enumerate(part["starts"]):
                            row = _run_trial(
                                mujoco,
                                env,
                                MPPI,
                                evidence.create_group(
                                    f"trials/variant_{index}/trial_{trial_index}"
                                ),
                                budget,
                                variant,
                                {key: start[key] for key in ("reset_seed", "planner_seed")},
                                part_name,
                                index,
                                trial_index,
                                out,
                                controls=part["controls"],
                                initial_state=np.asarray(start["initial_state"], dtype=float),
                                metrics_fn=native_metrics,
                                bank_callback=record_bank,
                            )
                            result["trials"].append(row)
                    finally:
                        env.close()
                evidence.flush()
        if part_name == "energy":
            result.update(decision(result["trials"], part["variants"]))
        else:
            result.update(
                status="COMPLETED_NEAR_UPRIGHT_PROBE",
                successes=sum(x["success"] for x in result["trials"]),
                teacher_qualified=False,
                learning_or_loop_launched=False,
            )
    except ResourceLimit as error:
        result.update(status="STOPPED_RESOURCES", error=f"{type(error).__name__}: {error}")
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="STOPPED_INVALID_EXECUTION", error=f"{type(error).__name__}: {error}")
    return _finalize(out, result, budget)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "execute"):
        item = sub.add_parser(command)
        item.add_argument("--config", type=Path, required=True)
        item.add_argument("--out", type=Path, required=True)
        if command == "execute":
            item.add_argument("--part", choices=tuple(PART_DESIGNS), required=True)
            item.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        preflight(args.config, args.out)
        if args.command == "preflight"
        else execute(args.config, args.part, args.preflight, args.out)
    )
    print(json.dumps(result, allow_nan=False), flush=True)
    return 2 if result["status"].startswith("STOPPED_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
