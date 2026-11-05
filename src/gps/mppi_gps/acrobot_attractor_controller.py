"""Native teacher stages for Acrobot's transient-to-upright-controller probe.

The existing MPPI recorder owns sampling, refinement and complete contexts.
This module changes only its declared temperature setting and adds an explicit
handoff to a previously validated fixed-point LQR controller. Native imports
remain lazy. No fitting, policy, optimizer implementation or automatic next
stage is present here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from src.gps.mppi_gps.acrobot_energy import energy_components, tip_position
from src.gps.mppi_gps.acrobot_teacher import (
    Budget,
    InvalidExecution,
    _capture,
    _run_trial,
    _save_array,
)
from src.gps.mppi_gps.attractor_lqr import lqr_action, lyapunov_value
from src.utils.config import MPPIConfig

CONTEXT_FIELDS = ("states", "warmstarts", "integration")
NEAR_PLANNER = MPPIConfig(
    H=150,
    K=256,
    lam=3.0,
    noise_sigma=0.6,
    initial_iterations=3,
    lam_target_neff_fraction=0.125,
)
SWING_PLANNER = MPPIConfig(
    H=300,
    K=512,
    lam=0.15,
    noise_sigma=0.18553280373949055,
    initial_iterations=3,
    lam_target_neff_fraction=0.0625,
)


def validate_stage(stage: dict[str, Any], *, swing: bool) -> None:
    """Reject undeclared interventions; callers freeze starts and full budgets."""
    name = stage["name"]
    expected_count = {"near": 4, "swing-development": 2, "swing-qualification": 10}
    if name not in expected_count or (name != "near") != swing:
        raise ValueError("Unknown or inconsistent attractor-controller stage.")
    expected = SWING_PLANNER if swing else NEAR_PLANNER
    if asdict(MPPIConfig(**stage["profile"]["planner"])) != asdict(expected):
        raise ValueError("Only the declared adaptive-ESS temperature recipe is allowed.")
    if (
        stage["controls"] != (600 if swing else 500)
        or stage["threads"] != (4 if swing else 2)
        or stage["lqr_steps"] != (200 if swing else 0)
        or len(stage["starts"]) != expected_count[name]
    ):
        raise ValueError("Stage horizon, suffix, threads and start roster are frozen.")
    profile = stage["profile"]
    if profile["kind"] != ("energy" if swing else "dense"):
        raise ValueError("The declared task cost changed.")
    if swing and profile.get("energy_weight") != 160 / 9:
        raise ValueError("Use the existing matched six-second energy objective.")
    for start in stage["starts"]:
        initial = np.asarray(start["initial_state"], dtype=np.float64)
        if initial.shape != (5,) or not np.isfinite(initial).all() or initial[0] != 0:
            raise ValueError("Starts require finite complete time-zero states.")
        if (
            type(start["reset_seed"]) is not int
            or start["planner_seed"] != start["reset_seed"] + 10000
        ):
            raise ValueError("Private reset and planner seed bindings changed.")
        if name != "swing-qualification" and "initial_context" not in start:
            raise ValueError("Historical starts require the original complete context.")
        if name == "near" and "baseline" not in start:
            raise ValueError("Near-upright starts require the historical first-pass bank.")


def trace_metrics(
    env: Any, states: np.ndarray, costs: np.ndarray, actions: np.ndarray
) -> dict[str, Any]:
    """POST-state measurements for complete or explicitly stopped MPPI prefixes."""
    states = np.asarray(states)
    costs = np.asarray(costs)
    actions = np.asarray(actions)
    n = len(costs)
    if n < 1 or states.shape != (n, 5) or actions.shape != (n, 1):
        raise InvalidExecution("Each paid control needs its complete native outcome.")
    if not all(np.isfinite(x).all() for x in (states, costs, actions)):
        raise InvalidExecution("Nonfinite evidence is an execution failure, not a task miss.")
    height = tip_position(states)[:, 2]
    high = height >= 3.8
    boundaries = np.flatnonzero(np.r_[False, high, False][1:] != np.r_[False, high, False][:-1])
    longest = int(np.max(boundaries[1::2] - boundaries[::2], initial=0))
    energy = energy_components(states)
    terminal = float(np.asarray(env.terminal_cost(states[-1])).item())
    running = float(costs.sum())
    position = tip_position(states) - np.array([0.0, 0.0, 4.0])
    dense = np.sum(position * position, axis=1)
    dense += 0.01 * np.sum(states[:, 3:5] ** 2, axis=1)
    dense += 0.01 * np.sum(actions * actions, axis=1)
    return dict(
        controls=n,
        success=bool(n >= 100 and high[-100:].all()),
        height_hold_success=bool(n >= 100 and high[-100:].all()),
        task_cost=running + terminal,
        native_running_cost_sum=running,
        episode_terminal_cost=terminal,
        original_dense_cost_sum=float(dense.sum()),
        maximum_tip_height=float(height.max()),
        final_window_minimum_height=float(height[-100:].min()),
        longest_height_interval_seconds=longest * 0.02,
        time_balanced_seconds=float(high.sum() * 0.02),
        final_window_velocity_rms=float(np.sqrt(np.mean(states[-100:, 3:5] ** 2))),
        terminal_total_energy=float(energy["total_energy"][-1]),
        terminal_potential=float(energy["potential"][-1]),
        terminal_kinetic=float(energy["kinetic"][-1]),
    )


def verify_first_bank(group: Any, baseline: dict[str, Any]) -> dict[str, Any]:
    """Compare the one bank whose contexts, zero plan and innovations are paired."""
    import h5py

    with h5py.File(baseline["h5_path"], "r") as source:
        old = source[baseline["group"] + "/planning/control_0000/pass_000"]
        receipt: dict[str, Any] = dict(
            baseline_h5=baseline["h5_path"],
            baseline_group=old.name,
            rtol=0.0,
            atol=1e-10,
            fields={},
        )
        for name in ("candidate_costs", "nominal_before"):
            actual, expected = group[name][:], old[name][:]
            same_shape = actual.shape == expected.shape
            error = float(np.max(np.abs(actual - expected), initial=0)) if same_shape else None
            passed = same_shape and np.isfinite(actual).all() and np.isfinite(expected).all()
            passed = bool(passed and error <= 1e-10)
            receipt["fields"][name] = dict(maximum_absolute_error=error, passed=passed)
            if not passed:
                group.attrs["baseline_first_bank_comparison"] = json.dumps(receipt)
                raise InvalidExecution("Historical first-pass bank mismatch: " + name)
    receipt["status"] = "PASSED"
    group.attrs["baseline_first_bank_comparison"] = json.dumps(receipt)
    return receipt


def _weighting_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise InvalidExecution("No candidate banks were recorded.")
    return dict(
        bank_count=len(rows),
        target_ess=32.0,
        weighting_validated=all(row["target_reached"] for row in rows),
        unreachable_tie_banks=sum(row["unreachable_ties"] for row in rows),
        ess_minimum=min(row["ess"] for row in rows),
        ess_maximum=max(row["ess"] for row in rows),
        temperature_minimum=min(row["temperature"] for row in rows),
        temperature_maximum=max(row["temperature"] for row in rows),
    )


def _bank_recorder(
    baseline: dict[str, Any] | None = None, weighting: list[dict[str, Any]] | None = None
):
    def record(group: Any, env: Any, planner: Any, state: np.ndarray, budget: Budget) -> None:
        del env, state, budget
        actions = np.ascontiguousarray(planner._last_actions)
        group.attrs["action_bank_sha256"] = hashlib.sha256(actions.tobytes()).hexdigest()
        group.attrs["action_bank_shape"] = json.dumps(actions.shape)
        group.attrs["action_bank_dtype"] = str(actions.dtype)
        group.attrs["rng_state_after_json"] = json.dumps(planner._rng.bit_generator.state)
        group.attrs["candidate_cost_std"] = float(np.std(planner._last_costs))
        scores = np.asarray(planner._last_costs)
        minimum_count = int((scores == scores.min()).sum())
        unreachable = minimum_count > 32
        neff = float(group.attrs["neff"])
        reached = abs(neff - 32.0) <= 1e-6
        group.attrs["ess32_target_reached"] = reached
        group.attrs["ess32_unreachable_minimum_ties"] = unreachable
        if not reached and not unreachable:
            raise InvalidExecution(
                "Adaptive temperature did not attain its reachable ESS32 target."
            )
        if weighting is not None:
            weighting.append(
                dict(
                    ess=neff,
                    temperature=float(group.attrs["lam"]),
                    target_reached=reached,
                    unreachable_ties=unreachable,
                )
            )
        if baseline is not None and group.name.endswith("/control_0000/pass_000"):
            verify_first_bank(group, baseline)

    return record


def _dependencies(stage: dict[str, Any], env: Any, mj: Any, mppi_type: Any):
    if mj is None:
        import mujoco as mj
    if mppi_type is None:
        from src.mppi.mppi import MPPI

        mppi_type = MPPI
    owned = env is None
    if env is None:
        from src.gps.mppi_gps.acrobot_energy_probe import _new_env

        env = _new_env(stage["profile"], stage["threads"])
    if env._dt != 0.02 or env._frame_skip != 2:
        raise InvalidExecution("Use the unchanged fifty-Hz native physics controller.")
    return env, mj, mppi_type, owned


def _precheck_starts(mj: Any, env: Any, evidence: Any, stage: dict, budget: Budget) -> None:
    checks = evidence.create_group("initial_context_checks")
    for index, start in enumerate(stage["starts"]):
        if "initial_context" not in start:
            continue
        budget.check()
        env.reset(state=np.asarray(start["initial_state"], dtype=np.float64))
        budget.work.resets += 1
        budget.work.forwards += 1
        actual = _capture(mj, env)
        group = checks.create_group(f"start_{index}")
        for key in CONTEXT_FIELDS:
            expected = np.asarray(start["initial_context"][key], dtype=np.float64)
            _save_array(group, key, actual[key], budget)
            if actual[key].shape != expected.shape or actual[key].tobytes() != expected.tobytes():
                raise InvalidExecution("Historical complete initial context mismatch: " + key)
        group.attrs["status"] = "PASSED_BYTE_EXACT"
    evidence.file.flush()


def _new_evidence(mj: Any, out: Path, stage: dict[str, Any]):
    import h5py

    (out / "trials.jsonl").touch(exist_ok=False)
    evidence = h5py.File(out / "evidence.h5", "x")
    evidence.attrs.update(
        protocol="acrobot-transient-attractor",
        stage=stage["name"],
        status="RUNNING",
        integration_spec=int(mj.mjtState.mjSTATE_INTEGRATION),
    )
    return evidence


def run_near(
    stage: dict[str, Any],
    out: Path,
    budget: Budget,
    *,
    env: Any = None,
    mj: Any = None,
    mppi_type: Any = None,
) -> dict[str, Any]:
    """Run all four unchanged near starts, changing only adaptive temperature."""
    validate_stage(stage, swing=False)
    env, mj, mppi_type, owned = _dependencies(stage, env, mj, mppi_type)
    rows, weighting = [], []
    try:
        with _new_evidence(mj, out, stage) as evidence:
            try:
                _precheck_starts(mj, env, evidence, stage, budget)
                for index, start in enumerate(stage["starts"]):
                    group = evidence.create_group(f"trials/variant_0/trial_{index}")
                    rows.append(
                        _run_trial(
                            mj,
                            env,
                            mppi_type,
                            group,
                            budget,
                            stage["profile"],
                            {key: start[key] for key in ("reset_seed", "planner_seed")},
                            stage["name"],
                            0,
                            index,
                            out,
                            controls=stage["controls"],
                            initial_state=np.asarray(start["initial_state"], dtype=np.float64),
                            metrics_fn=trace_metrics,
                            bank_callback=_bank_recorder(start["baseline"], weighting),
                        )
                    )
                evidence.attrs["weighting_summary"] = json.dumps(_weighting_summary(weighting))
                evidence.attrs["status"] = "COMPLETED_NEAR_CONTROLLER_PROBE"
            except BaseException as error:
                evidence.attrs.update(status="STOPPED", error=f"{type(error).__name__}: {error}")
                raise
    finally:
        if owned:
            env.close()
    return dict(
        status="COMPLETED_NEAR_CONTROLLER_PROBE",
        trials=rows,
        successes=sum(row["success"] for row in rows),
        **_weighting_summary(weighting),
        learning_or_loop_launched=False,
    )


def record_lqr_trace(
    env: Any,
    group: Any,
    budget: Budget,
    gain: np.ndarray,
    steps: int,
    *,
    mj: Any,
    p: np.ndarray,
    prefix: Any,
) -> dict[str, Any]:
    """Execute the locked suffix in the existing context, without reset or restore."""
    if type(steps) is not int or steps < 100:
        raise ValueError("The locked suffix must contain a complete two-second hold window.")
    start = _capture(mj, env)
    for key in CONTEXT_FIELDS:
        if start[key].tobytes() != np.asarray(prefix[key][-1]).tobytes():
            raise InvalidExecution("The LQR handoff changed complete native context: " + key)
    values: dict[str, list] = {key: [value] for key, value in start.items()}
    values.update(pre_action_observations=[], actions=[], costs=[], unclipped_actions=[])
    group.attrs.update(status="RUNNING", controller="upright_period_one_lqr", reset_at_join=False)
    datasets = {}
    for key, rows in values.items():
        shape = (
            np.asarray(rows[0]).shape
            if rows
            else {
                "pre_action_observations": (4,),
                "actions": (1,),
                "costs": (),
                "unclipped_actions": (1,),
            }[key]
        )
        datasets[key] = group.create_dataset(
            key,
            shape=(0, *shape),
            maxshape=(None, *shape),
            dtype="f8",
            chunks=(32, *shape),
        )

    def append(name: str, value: Any) -> None:
        value = np.asarray(value)
        budget.check(output_reservation=value.nbytes + 4096)
        dataset = datasets[name]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    for key, value in start.items():
        append(key, value)
    try:
        for t in range(steps):
            budget.check()
            before = _capture(mj, env)
            observation = env._get_obs()
            if not np.array_equal(observation, before["states"][1:5]):
                raise InvalidExecution("LQR PRE observation differs from its current state.")
            raw_action = lqr_action(before["states"], gain, clip=False)
            action = np.clip(raw_action, -1.0, 1.0)
            budget.reserve("native", 1, env._frame_skip)
            _, cost, done, _ = env.step(action)
            budget.complete("native", 1, env._frame_skip)
            after = _capture(mj, env)
            if (
                done
                or not np.isfinite(cost)
                or not all(np.isfinite(x).all() for x in after.values())
            ):
                raise InvalidExecution("Invalid native locked-controller execution.")
            if not np.isclose(
                after["states"][0], before["states"][0] + env._dt, rtol=0, atol=1e-10
            ):
                raise InvalidExecution("LQR execution stopped advancing physical time.")
            for key, value in dict(
                pre_action_observations=observation,
                actions=action,
                costs=cost,
                unclipped_actions=raw_action,
                **after,
            ).items():
                append(key, value)
            group.attrs["completed_controls"] = t + 1
            if (t + 1) % 10 == 0:
                group.file.flush()
        states, costs, actions = (
            datasets["states"][1:],
            datasets["costs"][:],
            datasets["actions"][:],
        )
        metrics = trace_metrics(env, states, costs, actions)
        all_values = lyapunov_value(datasets["states"][:], p)
        _save_array(group, "lyapunov_values", all_values, budget)
        metrics.update(
            saturation_controls=int(
                (np.abs(datasets["unclipped_actions"][:]) > 1).any(axis=1).sum()
            ),
            initial_V=float(all_values[0]),
            final_V=float(all_values[-1]),
            maximum_V=float(all_values.max()),
        )
        group.attrs.update(
            status="COMPLETED", success=metrics["success"], task_cost=metrics["task_cost"]
        )
        group.file.flush()
        return metrics
    except BaseException as error:
        group.attrs.update(status="STOPPED", error=f"{type(error).__name__}: {error}")
        group.file.flush()
        raise


def hybrid_metrics(prefix: dict, suffix: dict | None, join: dict) -> dict[str, Any]:
    """Use basin entry as acquisition success; never add the prefix terminal twice."""
    joined = join.get("control") is not None
    success = bool(joined and suffix is not None and suffix["success"])
    final = suffix if suffix is not None else prefix
    running = prefix["native_running_cost_sum"] + (
        suffix["native_running_cost_sum"] if suffix else 0
    )
    return dict(
        success=success,
        joined=joined,
        join_control=join.get("control"),
        join_V=join.get("V"),
        join_rho=join["rho"],
        mppi_controls=prefix["controls"],
        lqr_controls=suffix["controls"] if suffix else 0,
        total_controls=prefix["controls"] + (suffix["controls"] if suffix else 0),
        task_cost=running + final["episode_terminal_cost"],
        native_running_cost_sum=running,
        episode_terminal_cost=final["episode_terminal_cost"],
        original_dense_cost_sum=prefix["original_dense_cost_sum"]
        + (suffix["original_dense_cost_sum"] if suffix else 0),
        lqr_final_height_hold=bool(suffix and suffix["success"]),
        mppi_prefix_metrics=prefix,
        lqr_suffix_metrics=suffix,
    )


def run_swingup(
    stage: dict[str, Any],
    out: Path,
    budget: Budget,
    lqr_data: dict[str, Any],
    *,
    env: Any = None,
    mj: Any = None,
    mppi_type: Any = None,
) -> dict[str, Any]:
    """One explicit development OR qualification stage; never launch the next one."""
    validate_stage(stage, swing=True)
    gain, p = np.asarray(lqr_data["K"]), np.asarray(lqr_data["P"])
    rho = float(lqr_data["rho"])
    if (
        gain.shape != (1, 4)
        or p.shape != (4, 4)
        or not all(np.isfinite(x).all() for x in (gain, p))
    ):
        raise ValueError("A finite, previously verified LQR controller is required.")
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("A positive previously verified join level is required.")
    for start in stage["starts"]:
        if float(lyapunov_value(np.asarray(start["initial_state"]), p)) <= rho:
            raise InvalidExecution("A bound hanging start is already in the join region.")
    env, mj, mppi_type, owned = _dependencies(stage, env, mj, mppi_type)
    (out / "hybrid_trials.jsonl").touch(exist_ok=False)
    rows, weighting = [], []
    try:
        with _new_evidence(mj, out, stage) as evidence:
            try:
                _precheck_starts(mj, env, evidence, stage, budget)
                for index, start in enumerate(stage["starts"]):
                    group = evidence.create_group(f"trials/variant_0/trial_{index}")
                    join: dict[str, Any] = dict(control=None, V=None, rho=rho)

                    def stop(current_env: Any, trace: Any, completed: int) -> bool:
                        del current_env
                        value = float(lyapunov_value(trace["states"][-1], p))
                        if value <= rho:
                            join.update(control=completed, V=value)
                            trace.attrs.update(join_control=completed, join_V=value, join_rho=rho)
                            return True
                        return False

                    def prefix_metrics(
                        current_env: Any, states: np.ndarray, costs: np.ndarray, actions: np.ndarray
                    ) -> dict:
                        metrics = trace_metrics(current_env, states, costs, actions)
                        metrics["success"] = join["control"] is not None
                        metrics["success_scope"] = (
                            "MPPI acquisition reaches the verified LQR join level"
                        )
                        return metrics

                    prefix = _run_trial(
                        mj,
                        env,
                        mppi_type,
                        group,
                        budget,
                        stage["profile"],
                        {key: start[key] for key in ("reset_seed", "planner_seed")},
                        stage["name"],
                        0,
                        index,
                        out,
                        controls=stage["controls"],
                        initial_state=np.asarray(start["initial_state"], dtype=np.float64),
                        metrics_fn=prefix_metrics,
                        bank_callback=_bank_recorder(weighting=weighting),
                        stop_after_action=stop,
                    )
                    group.attrs["measurement_scope"] = (
                        "Root trace is MPPI prefix; success means basin entry, teacher_success includes LQR suffix"
                    )
                    suffix = None
                    if join["control"] is not None:
                        suffix = record_lqr_trace(
                            env,
                            group.create_group("lqr_suffix"),
                            budget,
                            gain,
                            stage["lqr_steps"],
                            mj=mj,
                            p=p,
                            prefix=group,
                        )
                    row = dict(
                        stage=stage["name"],
                        trial_index=index,
                        reset_seed=start["reset_seed"],
                        planner_seed=start["planner_seed"],
                        **hybrid_metrics(prefix, suffix, join),
                    )
                    group.attrs["teacher_success"] = row["success"]
                    group.attrs["hybrid_task_cost"] = row["task_cost"]
                    group.file.flush()
                    with (out / "hybrid_trials.jsonl").open("a") as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(
                        json.dumps(dict(event="hybrid_trial", **row), allow_nan=False), flush=True
                    )
                    rows.append(row)
                evidence.attrs["status"] = "COMPLETED_SWINGUP_STAGE"
                evidence.attrs["weighting_summary"] = json.dumps(_weighting_summary(weighting))
            except BaseException as error:
                evidence.attrs.update(status="STOPPED", error=f"{type(error).__name__}: {error}")
                raise
    finally:
        if owned:
            env.close()
    return dict(
        status="COMPLETED_SWINGUP_STAGE",
        trials=rows,
        successes=sum(row["success"] for row in rows),
        joins=sum(row["joined"] for row in rows),
        **_weighting_summary(weighting),
        learning_or_loop_launched=False,
    )
