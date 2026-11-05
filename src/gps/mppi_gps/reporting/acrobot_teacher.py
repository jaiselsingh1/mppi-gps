"""Independently recompute saved Acrobot teacher evidence; never run a simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ATOL = 1e-10


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def tip_geometry(states: np.ndarray) -> np.ndarray:
    """Reimplement the preserved +z link geometry, independently of the task."""
    states = np.asarray(states, dtype=np.float64)
    _require(states.shape[-1] == 5, "Full Acrobot state has dimension five")
    first, second = states[..., 1], states[..., 1] + states[..., 2]
    return np.stack(
        [np.sin(first) + np.sin(second), np.zeros_like(first), 2 + np.cos(first) + np.cos(second)],
        axis=-1,
    )


def _longest_true(values: np.ndarray) -> int:
    bounded = np.r_[False, np.asarray(values, dtype=bool), False]
    boundaries = np.flatnonzero(bounded[1:] != bounded[:-1])
    return int(np.max(boundaries[1::2] - boundaries[::2], initial=0))


def episode_metrics(
    states: np.ndarray,
    actions: np.ndarray,
    saved_costs: np.ndarray,
    *,
    dt: float = 0.02,
    horizon: int = 500,
    hold_seconds: float = 2.0,
    minimum_height: float = 3.8,
    observations: np.ndarray | None = None,
) -> dict:
    """Validate N+1 boundaries and recompute cost/success from executed arrays.

    Incomplete evidence remains explicitly incomplete and cannot pass the task.
    No post-terminal states or controls are invented. Height-only balance is
    distinguished from slow capture; speed is reported without changing success.
    """
    states, actions = np.asarray(states), np.asarray(actions)
    saved_costs = np.asarray(saved_costs)
    n = len(actions)
    _require(0 < n <= horizon, "Nonempty executed prefix within frozen horizon required")
    _require(states.shape == (n + 1, 5), "N+1 pre/post state indexing required")
    _require(actions.shape == (n, 1), "One bounded elbow action per control required")
    _require(saved_costs.shape == (n,), "One saved native cost per control required")
    _require(np.isfinite(states).all(), "Non-finite saved physics")
    _require(np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1, "Invalid native actions")
    _require(np.isfinite(saved_costs).all(), "Non-finite native costs")
    _require(np.isfinite(dt) and dt > 0, "Positive control period required")
    _require(np.isfinite(hold_seconds) and hold_seconds > 0, "Positive hold window required")
    _require(
        np.allclose(np.diff(states[:, 0]), dt, rtol=0, atol=ATOL),
        "Saved time boundaries differ from frozen control period",
    )
    if observations is not None:
        _require(
            np.array_equal(observations, states[:-1, 1:]),
            "Saved observations must be the N PRE-action raw states",
        )
    post = states[1:]
    tip = tip_geometry(post)
    position_cost = np.sum((tip - [0, 0, 4]) ** 2, axis=1)
    velocity_cost = 0.01 * np.sum(post[:, 3:5] ** 2, axis=1)
    action_cost = 0.01 * np.sum(actions**2, axis=1)
    recomputed_costs = position_cost + velocity_cost + action_cost
    errors = np.abs(recomputed_costs - saved_costs)
    _require(np.all(errors <= ATOL), "Saved POST-action cost differs from independent geometry")
    above = tip[:, 2] >= minimum_height
    hold = math.ceil(hold_seconds / dt)
    complete = n == horizon
    first = np.flatnonzero(above)
    sustained = (
        np.flatnonzero(np.convolve(above.astype(int), np.ones(hold, int), "valid") == hold)
        if n >= hold
        else np.array([], dtype=int)
    )
    window_velocity = np.linalg.norm(post[-min(n, hold) :, 3:5], axis=1)
    return {
        "controls": n,
        "complete": complete,
        "success": bool(complete and n >= hold and above[-hold:].all()),
        "task_cost": float(recomputed_costs.sum()),
        "position_cost": float(position_cost.sum()),
        "velocity_cost": float(velocity_cost.sum()),
        "action_cost": float(action_cost.sum()),
        "max_cost_error": float(errors.max(initial=0)),
        "first_height_crossing_seconds": None if not len(first) else float((first[0] + 1) * dt),
        "first_complete_hold_start_seconds": (
            None if not len(sustained) else float((sustained[0] + 1) * dt)
        ),
        "height_seconds": float(above.sum() * dt),
        "longest_height_seconds": float(_longest_true(above) * dt),
        "final_height": float(tip[-1, 2]),
        "minimum_tip_distance": float(np.min(np.linalg.norm(tip - [0, 0, 4], axis=1))),
        "final_tip_distance": float(np.linalg.norm(tip[-1] - [0, 0, 4])),
        "final_velocity_norm": float(np.linalg.norm(post[-1, 3:5])),
        "final_window_mean_velocity_norm": float(window_velocity.mean()),
        "final_window_max_velocity_norm": float(window_velocity.max()),
        "final_window_minimum_height": float(tip[-min(n, hold) :, 2].min()),
        "final_window_velocity_rms": float(np.sqrt(np.mean(post[-min(n, hold) :, 3:5] ** 2))),
        "terminal_angles_raw": post[-1, 1:3].tolist(),
        "terminal_winding": (np.floor((post[-1, 1:3] + np.pi) / (2 * np.pi)).astype(int).tolist()),
        "height_only_success": True,
        "missing_controls": horizon - n,
    }


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selection_from_rows(rows: list[dict], config: dict) -> dict:
    """Recompute development selection and the disjoint 9/10 qualification gate."""
    development = [row for row in rows if row["phase"] == "development"]
    qualification = [row for row in rows if row["phase"] == "qualification"]
    ranked = []
    for index, recipe in enumerate(config["recipes"]):
        subset = [row for row in development if row["recipe"] == recipe["name"]]
        expected = {(int(x["reset_seed"]), int(x["planner_seed"])) for x in config["development"]}
        actual = {(int(x["reset_seed"]), int(x["planner_seed"])) for x in subset}
        _require(
            actual == expected and len(subset) == len(expected), "Development roster incomplete"
        )
        _require(all(row["complete"] for row in subset), "Incomplete development episode")
        ranked.append(
            {
                "recipe": recipe["name"],
                "index": index,
                "successes": sum(row["success"] for row in subset),
                "mean_cost": float(np.mean([row["task_cost"] for row in subset])),
            }
        )
    ranked.sort(key=lambda row: (-row["successes"], row["mean_cost"], row["index"]))
    selected = ranked[0]["recipe"]
    expected = {(int(x["reset_seed"]), int(x["planner_seed"])) for x in config["qualification"]}
    actual = {(int(x["reset_seed"]), int(x["planner_seed"])) for x in qualification}
    _require(
        len(expected) == config["qualification_episode_count"], "Frozen qualification count differs"
    )
    _require(
        actual == expected and len(qualification) == len(expected),
        "Qualification roster incomplete",
    )
    _require(all(row["recipe"] == selected for row in qualification), "Wrong qualified recipe")
    _require(all(row["complete"] for row in qualification), "Incomplete qualification episode")
    development_seeds = {x["reset_seed"] for x in config["development"]}
    _require(
        not development_seeds & {x["reset_seed"] for x in config["qualification"]}, "Seed leakage"
    )
    successes = sum(row["success"] for row in qualification)
    return {
        "selected_recipe": selected,
        "ranking": ranked,
        "qualification_successes": successes,
        "qualification_episodes": len(qualification),
        "gate_passed": successes >= config["qualification_successes_required"],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(run: Path, config: Path, out: Path, cpu_seconds: float = 180) -> dict:
    """Recompute the frozen worker's HDF5/JSON evidence without new execution."""
    import h5py

    started = time.process_time()
    run, config, out = run.resolve(), config.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    checks: list[str] = []
    consumed: list[dict] = []

    def check(condition: Any, label: str) -> None:
        _require(time.process_time() - started <= cpu_seconds, "Report CPU ceiling exhausted")
        _require(condition, label)
        checks.append(label)

    def same(actual: Any, expected: Any, label: str, tolerance: float = ATOL) -> None:
        a, b = np.asarray(actual), np.asarray(expected)
        check(a.shape == b.shape, label + ": shape")
        check(
            np.isfinite(a).all()
            and np.isfinite(b).all()
            and np.allclose(a, b, rtol=0, atol=tolerance),
            label + ": values",
        )

    def consume(path: Path) -> Path:
        record = {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}
        if record not in consumed:
            consumed.append(record)
        return path

    def load_json(path: Path) -> dict:
        return json.loads(consume(path).read_text())

    limitations = [
        "Saved-array audit only: no physics, neural inference, fitting or MPPI queries were repeated.",
        "Height-only final-two-second success is not a hidden low-speed capture requirement.",
        "Full candidate state/action arrays were not recorded; candidate task costs and weighted "
        "nominal updates cannot be independently re-executed from these compact banks.",
        "Attempt counters are worker receipts. Completed calls are corroborated by saved arrays; "
        "unobserved partial work cannot be certified or assigned fabricated measurements.",
    ]
    audit: dict = {"status": "STARTING", "limitations": limitations}
    rows, trajectories, passes_rows = [], [], []
    try:
        spec = load_json(config)
        saved_config = load_json(run / "config.json")
        check(spec == saved_config, "Run configuration equals frozen configuration")
        result = load_json(run / "result.json")
        inputs = load_json(run / "inputs.json")
        environment = load_json(run / "environment.json")
        check(inputs["config_sha256"] == _sha(config), "Source receipt binds frozen configuration")
        check(
            environment["versions"] == spec["expected_environment"],
            "Pinned native package versions",
        )
        check(environment["physics_threads"] == spec["threads"], "Pinned native physics threads")
        check(
            set(environment["math_threads"].values()) == {str(spec["math_threads"])},
            "Pinned math thread environment",
        )
        source_roots = [run / "source", run.parent / "source", config.parents[2]]
        sources = {}
        for binding in inputs["files"]:
            original = Path(binding["path"])
            suffix = next(
                (
                    str(original).split(marker, 1)[1]
                    for marker in ("/src/", "/assets/")
                    if marker in str(original)
                ),
                None,
            )
            marker = "src" if "/src/" in str(original) else "assets"
            relative = Path(marker) / suffix if suffix is not None else Path(original.name)
            candidates = [root / relative for root in source_roots]
            if original.exists():
                candidates.append(original)
            matching = [
                path for path in candidates if path.is_file() and _sha(path) == binding["sha256"]
            ]
            check(bool(matching), "Available frozen source bytes: " + str(relative))
            source = consume(matching[0])
            check(source.stat().st_size == binding["bytes"], "Source byte count: " + str(relative))
            sources[str(relative)] = binding["sha256"]
        for binding in spec["bindings"]:
            check(
                sources.get(binding["path"]) == binding["sha256"],
                "Frozen asset: " + binding["path"],
            )
        for name, binding in result["evidence"].items():
            path = consume(run / name)
            check(
                _sha(path) == binding["sha256"] and path.stat().st_size == binding["bytes"],
                "Worker output digest: " + name,
            )
        complete_status = result["status"] in ("QUALIFIED", "FAILED_TEACHER_GATE")
        check(complete_status, "All teacher stages completed before saved-only certification")
        dt = 0.01 * spec["task"]["frame_skip"]
        horizon = spec["task"]["controls"]
        query_count, forecast_count, native_count = 0, 0, 0
        evidence_path = consume(run / "evidence.h5")
        with h5py.File(evidence_path, "r") as evidence:
            check(evidence.attrs["status"] == result["status"], "HDF5 completion status binding")
            preflight = evidence["preflight"]
            check(preflight.attrs["status"] == "PASSED", "Preflight worker gate passed")
            if "geometry_states" in preflight:
                same(
                    preflight["geometry_native_tip"][:],
                    tip_geometry(preflight["geometry_states"][:]),
                    "Native site versus independent analytic geometry",
                    tolerance=1e-14,
                )
            for name in ("states", "warmstarts", "integration", "costs"):
                same(
                    preflight["restored"][name][:],
                    preflight["recorded"][name][:],
                    "Preflight replay " + name,
                )
            recorded_states = preflight["recorded/states"][:]
            recorded_costs = preflight["recorded/costs"][:]
            episode_metrics(
                recorded_states,
                preflight["actions"][:],
                recorded_costs,
                dt=dt,
                horizon=spec["preflight"]["controls"],
            )
            checks.append("Preflight independent geometry costs and N+1 indexing")
            same(preflight["batch_states"][:], recorded_states[1:], "Preflight native/batch states")
            same(
                preflight["batch_cost"][:],
                np.array([recorded_costs.sum()]),
                "Preflight native/batch cost",
            )
            for phase in ("development", "qualification"):
                for recipe_index, recipe in enumerate(spec["recipes"]):
                    for trial_index, seeds in enumerate(spec[phase]):
                        path = (
                            f"development/recipe_{recipe_index}/trial_{trial_index}"
                            if phase == "development"
                            else f"qualification/trial_{trial_index}"
                        )
                        if (
                            phase == "qualification"
                            and recipe_index != result["selected_recipe_index"]
                        ):
                            continue
                        group = evidence[path]
                        check(group.attrs["status"] == "COMPLETED", path + ": completed")
                        for key, expected in {
                            "stage": phase,
                            "recipe": recipe["name"],
                            **seeds,
                        }.items():
                            check(group.attrs[key] == expected, path + ": binding " + key)
                        states, actions, costs = (
                            group["states"][:],
                            group["actions"][:],
                            group["costs"][:],
                        )
                        metrics = episode_metrics(
                            states,
                            actions,
                            costs,
                            dt=dt,
                            horizon=horizon,
                            hold_seconds=spec["task"]["balance_seconds"],
                            minimum_height=spec["task"]["balance_height"],
                            observations=group["pre_action_observations"][:],
                        )
                        checks.append(
                            path + ": independent geometry/native cost, N+1 and PRE observations"
                        )
                        for name in ("warmstarts", "integration"):
                            values = group[name][:]
                            check(
                                len(values) == horizon + 1 and np.isfinite(values).all(),
                                path + ": contexts " + name,
                            )
                        check(
                            group.attrs["completed_controls"] == horizon,
                            path + ": native completion receipt",
                        )
                        check(
                            bool(group.attrs["success"]) == metrics["success"],
                            path + ": success receipt",
                        )
                        same(
                            group.attrs["task_cost"],
                            metrics["task_cost"],
                            path + ": task cost receipt",
                        )
                        row = dict(
                            phase=phase,
                            recipe=recipe["name"],
                            recipe_index=recipe_index,
                            trial_index=trial_index,
                            **seeds,
                            **metrics,
                        )
                        native_count += metrics["controls"]
                        previous = None
                        banks = group["planning"]
                        check(len(banks) == horizon, path + ": one bank per control")
                        planner = recipe["planner"]
                        for t in range(horizon):
                            bank = banks[f"control_{t:04d}"]
                            same(
                                bank["pre_action_observation"][:],
                                states[t, 1:],
                                path + f": bank PRE state {t}",
                                tolerance=0,
                            )
                            pass_count = planner["initial_iterations"] if t == 0 else 1
                            check(len(bank) - 1 == pass_count, path + f": optimization passes {t}")
                            for p in range(pass_count):
                                found = bank[f"pass_{p:03d}"]
                                scores, weights = found["candidate_costs"][:], found["weights"][:]
                                before, after = (
                                    found["nominal_before"][:],
                                    found["nominal_after"][:],
                                )
                                check(
                                    scores.shape == weights.shape == (planner["K"],),
                                    path + ": bank sample count",
                                )
                                check(
                                    np.isfinite(scores).all() and np.isfinite(weights).all(),
                                    path + ": finite bank",
                                )
                                same(
                                    found.attrs["lam"],
                                    planner["lam"],
                                    path + ": temperature",
                                    tolerance=0,
                                )
                                exponent = np.exp(-(scores - scores.min()) / planner["lam"])
                                expected_weights = exponent / exponent.sum()
                                same(weights, expected_weights, path + ": Gibbs weights")
                                same(
                                    found.attrs["neff"],
                                    1 / np.sum(weights**2),
                                    path + ": effective sample size",
                                )
                                check(
                                    before.shape == after.shape == (planner["H"], 1),
                                    path + ": nominal dimensions",
                                )
                                check(
                                    np.isfinite(after).all() and (np.abs(after) <= 1).all(),
                                    path + ": bounded nominal",
                                )
                                expected_before = (
                                    np.zeros_like(before) if previous is None else previous
                                )
                                same(
                                    before,
                                    expected_before,
                                    path + ": retained/shifted plan",
                                    tolerance=0,
                                )
                                previous = after
                                query_count += 1
                                forecast_count += planner["K"] * planner["H"]
                                passes_rows.append(
                                    dict(
                                        phase=phase,
                                        recipe=recipe["name"],
                                        trial=trial_index,
                                        control=t,
                                        refinement=p,
                                        neff=float(found.attrs["neff"]),
                                        min_candidate_cost=float(scores.min()),
                                        mean_candidate_cost=float(scores.mean()),
                                    )
                                )
                            same(
                                actions[t],
                                previous[0],
                                path + f": deployed optimized action {t}",
                                tolerance=0,
                            )
                            previous = np.concatenate([previous[1:], previous[-1:]])
                        rows.append(row)
                        trajectories.append((row, states, actions))
        selection = selection_from_rows(rows, spec)
        checks.append(
            "Complete development and qualification rosters; disjoint seeds; independent selection and gate"
        )
        chosen = load_json(run / "selection.json")
        check(
            spec["recipes"][chosen["selected_recipe_index"]]["name"]
            == selection["selected_recipe"],
            "Prequalification selection agrees",
        )
        check(
            result["selected_recipe"]["name"] == selection["selected_recipe"],
            "Result selection agrees",
        )
        check(
            result["qualification_successes"] == selection["qualification_successes"],
            "Independent success census agrees",
        )
        check(
            (result["status"] == "QUALIFIED") == selection["gate_passed"],
            "Qualification decision agrees",
        )
        saved_rows = result["development"] + result["qualification"]
        jsonl = [
            json.loads(line)
            for line in consume(run / "trials.jsonl").read_text().splitlines()
            if line
        ]
        check(saved_rows == jsonl, "All trial receipts agree with completed result roster")
        for row in rows:
            receipt = next(
                x
                for x in saved_rows
                if x["stage"] == row["phase"]
                and x["recipe_index"] == row["recipe_index"]
                and x["trial_index"] == row["trial_index"]
            )
            same(
                receipt["task_cost"],
                row["task_cost"],
                "JSON/HDF5 independently recomputed task cost",
            )
            check(
                receipt["success"] == row["success"], "JSON/HDF5 independently recomputed success"
            )
            for field in (
                "first_complete_hold_start_seconds",
                "final_window_minimum_height",
                "final_window_velocity_rms",
                "terminal_angles_raw",
                "terminal_winding",
            ):
                if row[field] is None:
                    check(receipt[field] is None, "Missing complete hold remains missing")
                else:
                    same(receipt[field], row[field], "Independent episode metric: " + field)
            if row["first_height_crossing_seconds"] is None:
                check(
                    receipt["first_tip_above_threshold_seconds"] is None,
                    "No height crossing remains missing",
                )
            else:
                same(
                    receipt["first_tip_above_threshold_seconds"],
                    row["first_height_crossing_seconds"],
                    "Independent first height crossing",
                )
            same(
                receipt["time_balanced_seconds"],
                row["height_seconds"],
                "Independent time above threshold",
            )
        verification = 3 * spec["preflight"]["controls"]
        recomputed = dict(
            native_controls=native_count,
            verification_controls=verification,
            forecast_controls=forecast_count,
            model_queries=query_count,
            physics_substeps=spec["task"]["frame_skip"]
            * (native_count + verification + forecast_count),
            restores=1,
        )
        for field, number in recomputed.items():
            for suffix in ("attempted", "completed"):
                check(
                    result["work"][field + "_" + suffix] == number,
                    "Saved completed work census: " + field + "_" + suffix,
                )
        for field in ("network_forwards", "backward_calls", "optimizer_updates"):
            check(result["work"][field] == 0, "No unauthorized learning work: " + field)
        check(
            set(result["unknown_partial_work"].values()) == {0},
            "No uncorroborated partial calls reported",
        )
        limits = spec["limits"]
        check(native_count + verification <= limits["native_controls"], "Native budget respected")
        for field in ("forecast_controls", "physics_substeps"):
            check(recomputed[field] <= limits[field], "Control budget respected: " + field)
        for field in ("cpu_seconds", "wall_seconds", "rss_bytes", "output_bytes"):
            check(
                result["resources"][field] < limits[field],
                "Worker resource receipt within frozen ceiling: " + field,
            )
        _write_csv(out / "episodes.csv", rows)
        _write_csv(out / "planner_passes.csv", passes_rows)
        _plot(out / "figure.png", trajectories, dt, spec["task"]["balance_height"])
        audit.update(
            status="PASSED",
            selection=selection,
            recomputed_work=recomputed,
            worker_resources=result["resources"],
            maximum_cost_error=max(x["max_cost_error"] for x in rows),
            scientific_verdict="teacher qualified"
            if selection["gate_passed"]
            else "teacher prerequisite failed; no similarity mechanism or learning tested",
        )
        _report(out / "report.md", rows, audit)
    except Exception as error:
        audit.update(status="FAILED_OR_INCOMPLETE", error=f"{type(error).__name__}: {error}")
        (out / "report.md").write_text(
            "# Acrobot saved-evidence review\n\nSaved evidence could not be certified: "
            + audit["error"]
            + ". No new execution was performed.\n"
        )
    audit.update(
        check_count=len(checks),
        checks=checks,
        inputs=consumed,
        report_cpu_seconds=time.process_time() - started,
        new_physics_controls=0,
        new_network_forwards=0,
        new_fitting_updates=0,
    )
    _write_json(out / "independent_audit.json", audit)
    return audit


def _plot(path: Path, trajectories: list[tuple], dt: float, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(11, 8), sharex="col")
    for column, phase in enumerate(("development", "qualification")):
        for row, states, actions in trajectories:
            if row["phase"] != phase:
                continue
            t = np.arange(1, len(actions) + 1) * dt
            label = f"{row['recipe']}, seed {row['reset_seed']}"
            line = axes[0, column].plot(
                t, tip_geometry(states[1:])[:, 2], label=label, alpha=0.8, lw=0.8
            )[0]
            color = line.get_color()
            axes[1, column].plot(t, states[1:, 1], color=color, alpha=0.65, lw=0.7)
            axes[1, column].plot(t, states[1:, 2], color=color, alpha=0.65, lw=0.7, linestyle="--")
            axes[2, column].plot(t, actions[:, 0], color=color, alpha=0.65, lw=0.7)
        axes[0, column].axhline(threshold, color="black", linestyle=":", lw=1)
        axes[0, column].axvspan(t[-1] - 2, t[-1], color="gray", alpha=0.08)
        axes[0, column].set_ylim(-0.1, 4.2)
        axes[0, column].set_title(phase.capitalize())
        axes[2, column].set_xlabel("Seconds from hanging start")
    for index, name in enumerate(
        ("Tip height (m)", "Raw angles (rad)\nsolid shoulder; dashed elbow", "Elbow control")
    ):
        axes[index, 0].set_ylabel(name)
    axes[0, 0].legend(fontsize=5, loc="lower right")
    figure.suptitle("Acrobot teacher prerequisite: all completed trials, including failures")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _report(path: Path, rows: list[dict], audit: dict) -> None:
    selection = audit["selection"]
    lines = [
        "# Acrobot teacher prerequisite",
        "",
        "**" + audit["scientific_verdict"].capitalize() + ".**",
        "",
        "Three declared controller recipes were compared on two paired development starts each; "
        "selection used successes, then native task cost, then declared recipe order. The selected "
        "recipe was tested on ten unused starts. No policy or similarity term was used.",
        "",
        "| Development recipe | Balance successes | Mean cost |",
        "|---|---:|---:|---:|",
    ]
    for ranked in sorted(selection["ranking"], key=lambda x: x["index"]):
        lines.append(
            f"| {ranked['recipe']} | {ranked['successes']}/2 | {ranked['mean_cost']:.3f} |"
        )
    chosen = [row for row in rows if row["phase"] == "qualification"]
    lines.extend(
        [
            "",
            f"Selected **{selection['selected_recipe']}**: **{selection['qualification_successes']}/10** "
            "qualification episodes kept the tip at least 3.8 m high for the entire final two seconds "
            "of a ten-second episode; the frozen gate requires 9/10. This is height-only balance; "
            "velocities are reported separately.",
            "",
            "| Qualification seed | Balance | Cost | First crossing (s) | Time above (s) | Longest interval (s) | Final speed norm |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in chosen:
        first = row["first_height_crossing_seconds"]
        lines.append(
            f"| {row['reset_seed']} | {int(row['success'])} | {row['task_cost']:.2f} | "
            f"{'—' if first is None else f'{first:.2f}'} | {row['height_seconds']:.2f} | "
            f"{row['longest_height_seconds']:.2f} | {row['final_velocity_norm']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Saved-array verification recomputed every executed cost, success, selection, raw pre-action "
            "binding, all optimization pass weights and completed-work accounting. It compared the saved "
            "same-worker replay and batch path at the unchanged 1e-10 absolute tolerance. Full candidate "
            "physics arrays were not saved; no new physics was run for this review.",
            "",
            "The figure shows every trial’s tip height, both raw joint angles and torques. "
            "Complete per-episode and per-pass measurements are in episodes.csv and planner_passes.csv. "
            "Costs, source bindings, limitations and work census are in independent_audit.json.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cpu-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    result = render(args.run, args.config, args.out, args.cpu_seconds)
    summary = {
        key: result[key]
        for key in ("status", "check_count", "selection", "scientific_verdict", "error")
        if key in result
    }
    summary["audit"] = str(args.out / "independent_audit.json")
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
