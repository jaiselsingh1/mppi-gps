"""Saved-only verification of the Acrobot balance and energy-discovery probes.

This module does not import simulator or controller implementations. It derives
geometry, mechanical energy, objectives and height outcomes from recorded arrays.
White innovations may be regenerated to check the sampled update without physics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

REPLAY_ATOL = 1e-10
ROUNDING_FACTOR = 64.0 * np.finfo(np.float64).eps
RUNNING_COLUMNS = (
    "pose",
    "weighted_energy",
    "weighted_balance",
    "weighted_excess",
    "control_cost",
    "running",
)
ENERGY_COLUMNS = (
    "potential",
    "kinetic",
    "total_energy",
    "energy_error",
    "gate",
    "scaled_velocity_squared",
    "excess_velocity_squared",
)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def independent_components(
    states: np.ndarray, actions: np.ndarray, *, kind: str, weight: float = 0.0
) -> dict[str, np.ndarray]:
    """Independent full-state geometry/energy and the two declared objectives."""
    states, actions = np.asarray(states, dtype=np.float64), np.asarray(actions, dtype=np.float64)
    require(states.shape[-1] == 5 and actions.shape[-1] == 1, "Expected full state5/action1")
    require(states.shape[:-1] == actions.shape[:-1], "Cost state/action boundaries differ")
    require(np.isfinite(states).all() and np.isfinite(actions).all(), "Nonfinite cost inputs")
    q0, q1, v0, v1 = (states[..., i] for i in range(1, 5))
    x = np.sin(q0) + np.sin(q0 + q1)
    z = 2 + np.cos(q0) + np.cos(q0 + q1)
    d2 = x * x + (z - 4) ** 2
    potential = 9.81 * (4 + 1.5 * np.cos(q0) + 0.5 * np.cos(q0 + q1))
    m00 = 0.0955703125 + 0.09530365719649561 + 1.5 + np.cos(q1)
    m01 = 0.09530365719649561 + 0.25 + 0.5 * np.cos(q1)
    m11 = 0.09530365719649561 + 0.25
    kinetic = 0.5 * (m00 * v0 * v0 + 2 * m01 * v0 * v1 + m11 * v1 * v1)
    energy = potential + kinetic
    error = (energy - 58.86) / 39.24
    reward = np.exp(np.log(0.1) * (np.maximum(np.sqrt(d2) - 0.2, 0) / 2) ** 2)
    gate = reward * reward
    scaled_velocity = (v0 / 2.5) ** 2 + (v1 / 5) ** 2
    excess_velocity = np.sum((np.maximum(np.abs(states[..., 3:5]) - 8, 0) / 8) ** 2, axis=-1)
    if kind == "dense":
        pose = d2
        energy_cost = np.zeros_like(d2)
        balance = 0.01 * (v0 * v0 + v1 * v1)
        excess = np.zeros_like(d2)
        control = 0.01 * np.sum(actions * actions, axis=-1)
        terminal = np.zeros_like(d2)
    elif kind == "energy":
        pose = 1 - np.clip(z / 4, 0, 1) + 0.05 * x * x
        energy_cost = weight * (1 - gate) * error * error
        balance = 0.15 * gate * scaled_velocity
        excess = 0.05 * excess_velocity
        control = np.zeros_like(d2)
        terminal = 100 * pose + 25 * gate * scaled_velocity
    else:
        raise ValueError("Unknown declared objective")
    if kind == "dense":
        gate = np.zeros_like(gate)  # The prior objective has no energy/balance handoff.
    running = pose + energy_cost + balance + excess + control
    return dict(
        height=z,
        distance_squared=d2,
        potential=potential,
        kinetic=kinetic,
        total_energy=energy,
        energy_error=error,
        gate=gate,
        scaled_velocity_squared=scaled_velocity,
        excess_velocity_squared=excess_velocity,
        pose=pose,
        weighted_energy=energy_cost,
        weighted_balance=balance,
        weighted_excess=excess,
        control_cost=control,
        running=running,
        terminal=terminal,
        original_dense=d2 + 0.01 * (v0 * v0 + v1 * v1) + 0.01 * np.sum(actions * actions, axis=-1),
    )


def longest_true(values: np.ndarray) -> int:
    padded = np.r_[False, np.asarray(values, dtype=bool), False]
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(boundaries[1::2] - boundaries[::2], initial=0))


def episode_measurements(
    states: np.ndarray,
    actions: np.ndarray,
    costs: np.ndarray,
    *,
    controls: int,
    kind: str,
    weight: float = 0.0,
    dt: float = 0.02,
    observations: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Require an N+1 complete trace; costs use POST states and actions use PRE observations."""
    states, actions, costs = np.asarray(states), np.asarray(actions), np.asarray(costs)
    n = len(actions)
    require(0 < n <= controls, "Empty or over-budget episode")
    require(states.shape == (n + 1, 5), "Episode requires N+1 complete boundaries")
    require(actions.shape == (n, 1) and costs.shape == (n,), "One action/cost per control")
    require(np.isfinite(states).all() and np.isfinite(costs).all(), "Invalid native arrays")
    require(np.isfinite(actions).all() and np.max(np.abs(actions)) <= 1, "Invalid bounded controls")
    require(np.allclose(np.diff(states[:, 0]), dt, rtol=0, atol=REPLAY_ATOL), "Time not advancing")
    if observations is not None:
        require(np.array_equal(observations, states[:-1, 1:]), "PRE observation binding failed")
    components = independent_components(states[1:], actions, kind=kind, weight=weight)
    error = np.abs(costs - components["running"])
    bound = REPLAY_ATOL + ROUNDING_FACTOR * np.maximum(1, np.abs(components["running"]))
    require(np.all(error <= bound), "Independent running cost differs beyond declared FP64 bound")
    high = components["height"] >= 3.8
    hold = math.ceil(2 / dt)
    complete = n == controls
    crossing = np.flatnonzero(high)
    starts = (
        np.flatnonzero(np.convolve(high.astype(int), np.ones(hold, int), "valid") == hold)
        if n >= hold
        else np.array([], dtype=int)
    )
    matched_low = (np.abs(components["energy_error"]) <= 0.05) & (components["height"] < 3)
    terminal = float(components["terminal"][-1])
    row = dict(
        controls=n,
        complete=complete,
        success=bool(complete and n >= hold and high[-hold:].all()),
        task_cost=float(components["running"].sum()) + terminal,
        native_running_cost_sum=float(components["running"].sum()),
        episode_terminal_cost=terminal,
        original_dense_cost_sum=float(components["original_dense"].sum()),
        maximum_height=float(components["height"].max()),
        final_window_minimum_height=float(components["height"][-min(n, hold) :].min()),
        first_tip_above_threshold_seconds=None
        if not len(crossing)
        else float((crossing[0] + 1) * dt),
        first_complete_hold_start_seconds=None if not len(starts) else float((starts[0] + 1) * dt),
        time_above_height_seconds=float(high.sum() * dt),
        longest_height_hold_seconds=float(longest_true(high) * dt),
        terminal_velocity_norm=float(np.linalg.norm(states[-1, 3:])),
        peak_total_energy=float(components["total_energy"].max()),
        terminal_total_energy=float(components["total_energy"][-1]),
        terminal_potential_energy=float(components["potential"][-1]),
        terminal_kinetic_energy=float(components["kinetic"][-1]),
        mean_balance_gate=float(components["gate"].mean()),
        target_energy_low_height_fraction=float(matched_low.mean()),
        maximum_running_cost_error=float(error.max(initial=0)),
        maximum_running_cost_error_in_bound=float(np.max(error / bound, initial=0)),
        missing_controls=controls - n,
    )
    return row, components


def decision(rows: list[dict], spec: dict) -> dict:
    """Only full native height outcomes determine the two-seed kill rule."""
    near = [row for row in rows if row["part"] == "near-upright"]
    near_expected = spec["parts"]["near-upright"]
    require(len(near) == len(near_expected["starts"]), "Near-upright roster incomplete")
    near_pairs = {(x["reset_seed"], x["planner_seed"]) for x in near_expected["starts"]}
    require(
        {(x["reset_seed"], x["planner_seed"]) for x in near} == near_pairs,
        "Near-upright seed roster differs",
    )
    require(all(x["complete"] for x in near), "Incomplete near-upright episodes")
    energy = [row for row in rows if row["part"] == "energy"]
    require(len(rows) == len(near) + len(energy), "Unknown or duplicated probe part")
    require(
        len(energy)
        == len(spec["parts"]["energy"]["variants"]) * len(spec["parts"]["energy"]["starts"]),
        "Energy study roster differs",
    )
    decisions = []
    for variant in spec["parts"]["energy"]["variants"]:
        found = [row for row in energy if row["variant"] == variant["name"]]
        expected = {(x["reset_seed"], x["planner_seed"]) for x in spec["parts"]["energy"]["starts"]}
        actual = {(x["reset_seed"], x["planner_seed"]) for x in found}
        require(
            len(found) == len(expected) and actual == expected, "Energy variant roster incomplete"
        )
        require(all(x["complete"] for x in found), "Incomplete energy episodes cannot qualify")
        successes = sum(x["success"] for x in found)
        decisions.append(
            dict(
                variant=variant["name"],
                successes=successes,
                episodes=len(found),
                both_seeds_succeed=successes == len(expected),
            )
        )
    promising = [x["variant"] for x in decisions if x["both_seeds_succeed"]]
    return dict(
        near_upright_successes=sum(x["success"] for x in near),
        near_upright_episodes=len(near),
        variants=decisions,
        promising_variants=promising,
        selected_variant=promising[0] if promising else None,
        scientific_decision="EXPLORATORY_TEACHER_SCREEN_PASSED"
        if promising
        else "TEACHER_PROBLEM_STOP",
        loop_tested=False,
        teacher_nine_of_ten_qualified=False,
    )


class SavedAudit:
    """Bounded saved-array checks with explicit numeric contracts and raw bindings."""

    def __init__(self, base: Path, cpu_seconds: float) -> None:
        self.base, self.cpu_seconds = base, cpu_seconds
        self.started = time.process_time()
        self.checks: list[str] = []
        self.inputs: list[dict] = []
        self.max_error = 0.0
        self.max_error_in_bound = 0.0
        self.regeneration_cpu_seconds = 0.0

    def check(self, condition: Any, label: str) -> None:
        require(
            time.process_time() - self.started <= self.cpu_seconds,
            "Saved audit CPU budget exhausted",
        )
        require(condition, label)
        self.checks.append(label)

    def same(
        self,
        actual: Any,
        expected: Any,
        label: str,
        *,
        replay: bool = False,
        exact: bool = False,
        absolute_tolerance: float | None = None,
    ) -> None:
        a, b = np.asarray(actual), np.asarray(expected)
        self.check(a.shape == b.shape, label + ": shape")
        self.check(np.isfinite(a).all() and np.isfinite(b).all(), label + ": finite")
        if exact:
            self.check(np.array_equal(a, b), label + ": exact")
            return
        if absolute_tolerance is not None:
            bound = absolute_tolerance
        else:
            bound = (
                REPLAY_ATOL if replay else REPLAY_ATOL + ROUNDING_FACTOR * np.maximum(1, np.abs(b))
            )
        error = np.abs(a - b)
        self.max_error = max(self.max_error, float(error.max(initial=0)))
        self.max_error_in_bound = max(
            self.max_error_in_bound, float(np.max(error / bound, initial=0))
        )
        self.check(np.all(error <= bound), label + ": numeric")

    def consume(self, path: Path) -> Path:
        self.check(True, "CPU budget before reading " + path.name)
        record = dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)
        if record not in self.inputs:
            self.inputs.append(record)
        return path

    def load(self, path: Path) -> dict:
        return json.loads(self.consume(path).read_text())


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), "No completed rows for publication")
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _names(group: Any, dataset: str, expected: tuple[str, ...]) -> list[str]:
    values = group[dataset].attrs.get("columns", group.attrs.get(dataset + "_columns", expected))
    if isinstance(values, str):
        values = json.loads(values)
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def _check_sources(audit: SavedAudit, run: Path, config: Path, spec: dict, result: dict) -> None:
    inputs = audit.load(run / "inputs.json")
    environment = audit.load(run / "environment.json")
    audit.check(inputs["config_sha256"] == sha(config), "Frozen configuration digest")
    audit.check(environment["versions"] == spec["expected_environment"], "Native runtime versions")
    audit.check(
        set(environment["math_threads"].values()) == {str(spec["math_threads"])}, "Math threads"
    )
    source_roots = [audit.base / "source", run / "source", config.parents[2]]
    for record in inputs.get("files", []):
        original = Path(record["path"])
        text = str(original)
        relative = next(
            (
                Path(marker) / text.split("/" + marker + "/", 1)[1]
                for marker in ("src", "assets")
                if "/" + marker + "/" in text
            ),
            Path(original.name),
        )
        candidates = [root / relative for root in source_roots] + [original]
        found = next((p for p in candidates if p.is_file() and sha(p) == record["sha256"]), None)
        audit.check(found is not None, "Frozen source available: " + str(relative))
        if found is not None:
            audit.consume(found)
            audit.check(found.stat().st_size == record["bytes"], "Source byte count")
    frozen = inputs.get("frozen_source_manifest")
    audit.check(isinstance(frozen, dict), "Frozen source-manifest receipt present")
    if isinstance(frozen, dict):
        candidates = [root / "source_manifest.json" for root in source_roots]
        manifest_path = next(
            (p for p in candidates if p.is_file() and sha(p) == frozen["sha256"]), None
        )
        audit.check(manifest_path is not None, "Frozen source manifest available")
        if manifest_path is not None:
            manifest = audit.load(manifest_path)
            audit.check(manifest["config_sha256"] == sha(config), "Manifest configuration digest")
            audit.check(
                len(manifest["files"]) == frozen["verified_payload_count"],
                "Full source payload count",
            )
            for entry in manifest["files"]:
                path = manifest_path.parent / entry["path"]
                audit.check(
                    path.is_file()
                    and path.stat().st_size == entry["bytes"]
                    and sha(path) == entry["sha256"],
                    "Full frozen source payload: " + entry["path"],
                )
    for name, record in result.get("evidence", {}).items():
        path = audit.consume(run / name)
        audit.check(
            sha(path) == record["sha256"] and path.stat().st_size == record["bytes"],
            "Worker output digest: " + name,
        )


def _check_work(
    audit: SavedAudit,
    result: dict,
    expected: dict[str, int],
    limits: dict,
    *,
    historical_storage_stop: bool = False,
) -> None:
    work = result["work"]
    for name, number in expected.items():
        for suffix in ("attempted", "completed"):
            audit.check(work[name + "_" + suffix] == number, "Work census: " + name + "_" + suffix)
    for key in ("network_forwards", "backward_calls", "optimizer_updates"):
        audit.check(work[key] == 0, "No learning work: " + key)
    partial = result.get("unknown_partial_work")
    audit.check(isinstance(partial, dict) and bool(partial), "Partial-work receipt present")
    if isinstance(partial, dict):
        audit.check(
            all(value == 0 for value in partial.values()), "No unresolved partial native calls"
        )
    for key in ("cpu_seconds", "wall_seconds", "output_bytes", "rss_bytes"):
        if key == "output_bytes" and historical_storage_stop:
            prefix = "ResourceLimit: Frozen output_bytes budget exhausted: "
            error = str(result.get("error", ""))
            audit.check(
                result["status"] == "STOPPED_RESOURCES" and error.startswith(prefix),
                "Preserved original storage failure",
            )
            failed_bytes = int(error.removeprefix(prefix).split(" >= ", 1)[0])
            audit.check(failed_bytes >= limits[key], "Historical storage ceiling was exceeded")
        else:
            audit.check(result["resources"][key] < limits[key], "Resource ceiling: " + key)


def _verify_preflight(audit: SavedAudit, spec: dict, config: Path) -> dict:
    import h5py

    run = audit.base / "preflight"
    result = audit.load(run / "result.json")
    audit.check(result["status"] == "PASSED", "All shared prerequisites passed")
    audit.check(audit.load(run / "config.json") == spec, "Prerequisite configuration binding")
    _check_sources(audit, run, config, spec, result)
    with h5py.File(audit.consume(run / "evidence.h5"), "r") as raw:
        audit.check(raw.attrs["status"] == "PASSED", "Prerequisite HDF5 completion binding")
        audit.check(
            len(result["profiles"]) == len(spec["preflight"]["profiles"]),
            "Complete replay-profile roster",
        )
        for profile in spec["preflight"]["profiles"]:
            receipt = next(x for x in result["profiles"] if x["name"] == profile["name"])
            audit.check(
                receipt["status"] == "PASSED" and receipt["threads"] == profile["threads"],
                "Frozen replay profile thread count",
            )
            group = raw["profiles/" + profile["name"]]
            audit.check(
                group.attrs["status"] == "PASSED", "Native replay profile: " + profile["name"]
            )
            for key in ("states", "warmstarts", "integration", "costs"):
                audit.same(
                    group["recorded/" + key][:],
                    group["restored/" + key][:],
                    "Strict native replay " + profile["name"] + ": " + key,
                    replay=True,
                )
            states, actions = group["recorded/states"][:], group["actions"][:]
            comp = independent_components(
                states[1:], actions, kind=profile["kind"], weight=profile.get("energy_weight", 0.0)
            )
            audit.same(group["recorded/costs"][:], comp["running"], "Preflight cost profile")
            audit.same(
                group["batch_states"][:], states[1:], "Native versus batch states", replay=True
            )
            total = comp["running"].sum() + comp["terminal"][-1]
            audit.same(
                group["batch_cost"][:],
                np.array([total]),
                "Batch running plus terminal",
                replay=True,
            )
            if "terminal_cost" in group:
                audit.same(
                    group["terminal_cost"][:],
                    np.array([comp["terminal"][-1]]),
                    "One terminal cost in replay",
                    replay=True,
                )
            if "geometry_states" in group:
                s = group["geometry_states"][:]
                x = np.sin(s[:, 1]) + np.sin(s[:, 1] + s[:, 2])
                z = 2 + np.cos(s[:, 1]) + np.cos(s[:, 1] + s[:, 2])
                audit.same(
                    group["geometry_native_tip"][:],
                    np.stack([x, np.zeros_like(x), z], 1),
                    "Native tip geometry",
                    absolute_tolerance=1e-14,
                )
        energy = raw["energy_validation"]
        states = energy["states"][:]
        audit.same(
            states,
            spec["preflight"]["energy_validation_states"],
            "Frozen energy-validation states",
            exact=True,
        )
        comp = independent_components(states, np.zeros((len(states), 1)), kind="energy")
        analytic = np.stack([comp["potential"], comp["kinetic"]], 1)
        native_name = "native_potential_kinetic"
        audit.same(energy[native_name][:], analytic, "Native energy validation", replay=True)
        audit.check(
            result["energy_validation"]["incoming_context_restored_exactly"],
            "Energy-validation incoming context restored",
        )
        audit.check(
            not result.get("possible_partial_energy_validation", True),
            "Complete native energy-validation work",
        )
    verify = len(spec["preflight"]["profiles"]) * 3 * spec["preflight"]["controls"]
    expected = dict(
        native_controls=0,
        forecast_controls=0,
        verification_controls=verify,
        model_queries=0,
        physics_substeps=verify * spec["task"]["frame_skip"],
        restores=len(spec["preflight"]["profiles"]) + 1,
    )
    _check_work(audit, result, expected, spec["preflight"]["limits"])
    return dict(result=result, expected=expected)


def _verify_bank(
    audit: SavedAudit,
    group: Any,
    variant: dict,
    rng: Any,
    previous: np.ndarray | None,
    start: np.ndarray,
    pass_rows: list[dict],
    identity: dict,
    context: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    cfg = variant["planner"]
    if context is not None:
        for name, expected_context in context.items():
            audit.same(
                group["pre_context/" + name][:],
                expected_context,
                "Candidate complete PRE context: " + name,
                exact=True,
            )
    audit.check(cfg.get("noise_lowpass_cutoff_hz") is None, "Declared white innovation model")
    K, H = cfg["K"], cfg["H"]
    scores, weights = group["candidate_costs"][:], group["weights"][:]
    before, after = group["nominal_before"][:], group["nominal_after"][:]
    audit.check(scores.shape == weights.shape == (K,), "Bank K")
    audit.check(np.isfinite(scores).all() and np.isfinite(weights).all(), "Finite candidate bank")
    audit.same(
        before,
        np.zeros((H, 1)) if previous is None else previous,
        "Nominal continuation",
        exact=True,
    )
    audit.check(
        after.shape == (H, 1) and np.isfinite(after).all() and (np.abs(after) <= 1).all(),
        "Bounded optimized nominal",
    )
    audit.same(group.attrs["lam"], cfg["lam"], "Fixed temperature", exact=True)
    w = np.exp(-(scores - scores.min()) / cfg["lam"])
    w /= w.sum()
    audit.same(weights, w, "Gibbs weights")
    audit.same(group.attrs["neff"], 1 / np.sum(weights * weights), "Effective sample size")
    regeneration_started = time.process_time()
    standard = rng.standard_normal((K, H, 1))
    chol = np.linalg.cholesky(np.array([[cfg["noise_sigma"] ** 2]]))
    noise = np.einsum("khi,ji->khj", standard, chol)
    actions = np.clip(before[None] + noise, -1, 1)
    expected = np.clip(before + np.einsum("k,kha->ha", weights, actions - before[None]), -1, 1)
    regenerated_hash = array_sha(actions)
    audit.regeneration_cpu_seconds += time.process_time() - regeneration_started
    audit.check(audit.regeneration_cpu_seconds <= 60, "White-innovation regeneration CPU cap")
    audit.same(after, expected, "Reconstructed white-noise weighted update")
    diagnostics = group["components"]
    if "rng_state_after_json" in group.attrs:
        audit.check(
            json.loads(group.attrs["rng_state_after_json"]) == rng.bit_generator.state,
            "Exact regenerated PCG64 stream endpoint",
        )
    columns = _names(diagnostics, "running_sums", RUNNING_COLUMNS)
    audit.check(columns == list(RUNNING_COLUMNS), "Running component column contract")
    running = diagnostics["running_sums"][:]
    audit.check(
        running.shape == (K, len(columns)) and np.isfinite(running).all(),
        "Candidate running components",
    )
    terminal = diagnostics["terminal"][:]
    audit.same(running[:, :5].sum(1), running[:, 5], "Decomposed running sums")
    audit.same(running[:, 5] + terminal, scores, "All candidate running plus terminal cost")
    ecolumns = _names(diagnostics, "energy_sums", ENERGY_COLUMNS)
    audit.check(ecolumns == list(ENERGY_COLUMNS), "Energy component column contract")
    energies = diagnostics["energy_sums"][:]
    audit.check(
        energies.shape == (K, len(ecolumns)) and np.isfinite(energies).all(),
        "Candidate energy summaries",
    )
    audit.same(energies[:, 0] + energies[:, 1], energies[:, 2], "Potential plus kinetic summary")
    peak = diagnostics["peak_height"][:]
    endmin = diagnostics["final_window_minimum_height"][:]
    qualifies = diagnostics["height_qualified"][:]
    audit.check(peak.shape == endmin.shape == qualifies.shape == (K,), "Candidate height summaries")
    audit.check(np.array_equal(qualifies.astype(bool), endmin >= 3.8), "Candidate height flag")
    indices = diagnostics["selected_indices"][:].astype(int)
    audit.same(
        indices,
        np.array([np.argmin(scores), np.argmax(peak)]),
        "Declared saved-candidate selectors",
        exact=True,
    )
    saved_states = diagnostics["selected_states"][:]
    saved_actions = diagnostics["selected_actions"][:]
    audit.check(
        saved_states.shape == (2, H, 5) and saved_actions.shape == (2, H, 1),
        "Two retained complete candidates",
    )
    audit.same(saved_actions, actions[indices], "Retained candidate action binding")
    audit.same(
        saved_states[:, :, 0],
        np.broadcast_to(start[0] + np.arange(1, H + 1)[None] * 0.02, (2, H)),
        "Candidate control times",
        replay=True,
    )
    comp = independent_components(
        saved_states, saved_actions, kind=variant["kind"], weight=variant.get("energy_weight", 0)
    )
    audit.same(
        comp["running"].sum(1) + comp["terminal"][:, -1],
        scores[indices],
        "Retained candidate objective",
    )
    audit.same(comp["height"].max(1), peak[indices], "Retained candidate peak height")
    audit.same(
        comp["height"][:, -100:].min(1), endmin[indices], "Retained candidate terminal height"
    )
    for index, name in enumerate(RUNNING_COLUMNS):
        audit.same(
            comp[name].sum(1), running[indices, index], "Retained candidate component: " + name
        )
    for index, name in enumerate(ENERGY_COLUMNS):
        audit.same(
            comp[name].sum(1), energies[indices, index], "Retained candidate energy: " + name
        )
    native_hash = diagnostics.attrs.get("candidate_actions_sha256", None)
    hash_equal = None if native_hash is None else str(native_hash) == regenerated_hash
    pass_rows.append(
        dict(
            **identity,
            neff=float(group.attrs["neff"]),
            candidate_height_qualified_fraction=float(qualifies.mean()),
            weight_on_height_qualified=float(weights[qualifies.astype(bool)].sum()),
            maximum_candidate_height=float(peak.max()),
            minimum_candidate_cost=float(scores.min()),
            candidate_action_hash_matches_regeneration=hash_equal,
        )
    )
    return after


def _verify_part(
    audit: SavedAudit,
    part: str,
    spec: dict,
    config: Path,
    rows: list[dict],
    traces: list[tuple],
    pass_rows: list[dict],
    *,
    run_name: str | None = None,
    trial_indices: tuple[int, ...] | None = None,
    accepted_status: str | None = None,
    partial_index: int | None = None,
    limits_override: dict | None = None,
) -> dict:
    import h5py

    run = audit.base / (run_name or part)
    result = audit.load(run / "result.json")
    valid_statuses = (
        {"COMPLETED_NEAR_UPRIGHT_PROBE"}
        if part == "near-upright"
        else {"TEACHER_PROBLEM_STOP", "PROMISING_TEACHER_PROBE_ONLY"}
    )
    if accepted_status is not None:
        valid_statuses = {accepted_status}
    audit.check(result["status"] in valid_statuses, "Declared part status: " + run.name)
    audit.check(audit.load(run / "config.json") == spec, "Part configuration: " + part)
    _check_sources(audit, run, config, spec, result)
    binding = result.get("preflight_binding", {})
    audit.check(
        binding.get("result_sha256") == sha(audit.base / "preflight/result.json"),
        "Bound preserved prerequisite result",
    )
    audit.check(
        binding.get("evidence_sha256") == sha(audit.base / "preflight/evidence.h5"),
        "Bound preserved prerequisite arrays",
    )
    audit.check(binding.get("config_sha256") == sha(config), "Prerequisite configuration hash")
    audit.check(binding.get("new_physics_repeated") == 0, "No repeated prerequisite physics")
    part_cfg = spec["parts"][part]
    selected_indices = (
        tuple(range(len(part_cfg["starts"]))) if trial_indices is None else trial_indices
    )
    completed_indices = tuple(x for x in selected_indices if x != partial_index)
    environment = audit.load(run / "environment.json")
    audit.check(environment["physics_threads"] == part_cfg["threads"], "Physics threads: " + part)
    native, forecast, queries = 0, 0, 0
    receipts = result.get("trials", result.get("episodes", []))
    audit.check(
        len(receipts) == len(part_cfg["variants"]) * len(completed_indices),
        "Complete trial-receipt roster",
    )
    partial_measurement = None
    incomplete_banks = []
    with h5py.File(audit.consume(run / "evidence.h5"), "r") as raw:
        audit.check(raw.attrs["status"] == result["status"], "HDF5 completion binding: " + part)
        for vi, variant in enumerate(part_cfg["variants"]):
            for si, seeds in enumerate(part_cfg["starts"]):
                if si not in selected_indices:
                    continue
                path = f"trials/variant_{vi}/trial_{si}"
                group = raw[path]
                partial = si == partial_index
                audit.check(
                    group.attrs["status"] == ("STOPPED" if partial else "COMPLETED"),
                    path + ": preserved episode status",
                )
                for key, value in (
                    ("recipe", variant["name"]),
                    ("reset_seed", seeds["reset_seed"]),
                    ("planner_seed", seeds["planner_seed"]),
                ):
                    audit.check(group.attrs[key] == value, path + ": " + key)
                states, actions, costs = group["states"][:], group["actions"][:], group["costs"][:]
                audit.same(states[0], seeds["initial_state"], "Frozen initial state", exact=True)
                measurement, components = episode_measurements(
                    states,
                    actions,
                    costs,
                    controls=part_cfg["controls"],
                    kind=variant["kind"],
                    weight=variant.get("energy_weight", 0),
                    observations=group["pre_action_observations"][:],
                )
                audit.check(
                    measurement["complete"] != partial,
                    "Full episode before classification or explicitly excluded partial prefix",
                )
                if partial:
                    audit.check(len(actions) == 429, "Original paid partial-prefix length")
                context_arrays = {}
                for key in ("warmstarts", "integration"):
                    array = group[key][:]
                    context_arrays[key] = array
                    audit.check(
                        len(array) == len(actions) + 1 and np.isfinite(array).all(),
                        "Complete context " + key,
                    )
                if not partial:
                    audit.same(
                        group.attrs["task_cost"],
                        measurement["task_cost"],
                        "Episode objective receipt",
                    )
                    audit.check(
                        bool(group.attrs["success"]) == measurement["success"],
                        "Height outcome receipt",
                    )
                row = dict(
                    part=part,
                    evidence_run=run.name,
                    variant=variant["name"],
                    variant_index=vi,
                    trial_index=si,
                    reset_seed=seeds["reset_seed"],
                    planner_seed=seeds["planner_seed"],
                    **measurement,
                )
                if receipts and not partial:
                    receipt = next(
                        x for x in receipts if x["recipe_index"] == vi and x["trial_index"] == si
                    )
                    audit.check(receipt["success"] == row["success"], "JSON native success receipt")
                    audit.same(receipt["task_cost"], row["task_cost"], "JSON objective receipt")
                planner = variant["planner"]
                previous = None
                rng = np.random.default_rng(seeds["planner_seed"])
                banks = group["planning"]
                audit.check(
                    len(banks) == len(actions) + int(partial),
                    "Executed plans plus declared unexecuted partial bank",
                )
                for t in range(len(actions)):
                    bank = banks[f"control_{t:04d}"]
                    audit.same(
                        bank["pre_action_observation"][:],
                        states[t, 1:],
                        "Bank PRE state",
                        exact=True,
                    )
                    count = planner["initial_iterations"] if t == 0 else 1
                    audit.check(
                        len([x for x in bank if x.startswith("pass_")]) == count,
                        "All refinement passes saved",
                    )
                    for p in range(count):
                        identity = dict(
                            part=part,
                            evidence_run=run.name,
                            variant=variant["name"],
                            trial_index=si,
                            control=t,
                            refinement=p,
                        )
                        previous = _verify_bank(
                            audit,
                            bank[f"pass_{p:03d}"],
                            variant,
                            rng,
                            previous,
                            states[t],
                            pass_rows,
                            identity,
                            context={
                                "states": states[t],
                                **{name: value[t] for name, value in context_arrays.items()},
                            },
                        )
                        queries += 1
                        forecast += planner["K"] * planner["H"]
                    audit.same(actions[t], previous[0], "Executed optimized action", exact=True)
                    previous = np.concatenate([previous[1:], previous[-1:]])
                native += len(actions)
                if partial:
                    # The worker charged one additional complete native batch before the
                    # output guard fired. Its partially written diagnostic bank is preserved,
                    # but neither its update nor an unexecuted action is fabricated here.
                    pending = banks[f"control_{len(actions):04d}"]
                    inventory = {}
                    pending.visititems(
                        lambda name, value: inventory.update(
                            {name: dict(shape=list(value.shape), dtype=str(value.dtype))}
                        )
                        if isinstance(value, h5py.Dataset)
                        else None
                    )
                    incomplete_banks.append(
                        dict(
                            path=pending.name, datasets=inventory, numerical_update_certified=False
                        )
                    )
                    queries += 1
                    forecast += planner["K"] * planner["H"]
                    partial_measurement = row
                else:
                    rows.append(row)
                    traces.append((row, states, actions, components))
    expected = dict(
        native_controls=native,
        forecast_controls=forecast,
        verification_controls=0,
        model_queries=queries,
        physics_substeps=(native + forecast) * spec["task"]["frame_skip"],
        restores=0,
    )
    _check_work(
        audit,
        result,
        expected,
        limits_override or part_cfg["limits"],
        historical_storage_stop=partial_index is not None,
    )
    return dict(
        result=result,
        expected=expected,
        partial_measurement=partial_measurement,
        incomplete_banks=incomplete_banks,
    )


def verify_repeated_prefix(
    audit: SavedAudit, previous: Any, completion: Any, receipt: dict
) -> dict:
    """Check a fresh deterministic repeat, not a restored continuation claim."""
    audit.check(
        receipt.get("status") == "PASSED"
        and receipt.get("controls") == 429
        and receipt.get("previous_paid_queries") == 432
        and receipt.get("rtol") == 0
        and receipt.get("atol") == REPLAY_ATOL,
        "Frozen deterministic-repeat contract",
    )
    errors = {}
    for name in (
        "states",
        "warmstarts",
        "integration",
        "actions",
        "costs",
        "pre_action_observations",
    ):
        count = 430 if name in ("states", "warmstarts", "integration") else 429
        old, new = np.asarray(previous[name][:count]), np.asarray(completion[name][:count])
        audit.check(len(old) == len(new) == count, "Repeated-prefix rows: " + name)
        audit.same(new, old, "Repeated-prefix strict replay: " + name, replay=True)
        maximum = float(np.abs(new - old).max(initial=0))
        field = receipt["fields"][name]
        audit.check(
            field["passed"]
            and field["rows"] == count
            and field["shape"] == list(old.shape)
            and field["dtype"] == str(old.dtype)
            and field["previous_array_raw_bytes_sha256"] == array_sha(old)
            and field["completion_array_raw_bytes_sha256"] == array_sha(new),
            "Repeated-prefix recorded array bindings: " + name,
        )
        audit.same(
            field["maximum_absolute_error"], maximum, "Repeated-prefix error receipt", exact=True
        )
        errors[name] = maximum
    return errors


def verify_recovery_input_alias(
    audit: SavedAudit, previous: dict, completion: dict, original_config_path: str
) -> None:
    """Accept only the frozen relative path and its exact native-source absolute spelling."""
    source_root = PurePosixPath(previous["frozen_source_manifest"]["path"]).parent
    relative = PurePosixPath(original_config_path)
    audit.check(not relative.is_absolute(), "Frozen original configuration path is relative")
    audit.check(
        previous["config_path"] == original_config_path
        and completion["config_path"] == str(source_root / relative),
        "Exact relative-to-native-source configuration path alias",
    )
    normalized = dict(completion, config_path=previous["config_path"])
    audit.check(normalized == previous, "Every other original input binding remains exact")


def _verify_near(
    audit: SavedAudit,
    spec: dict,
    config: Path,
    rows: list[dict],
    traces: list[tuple],
    passes: list[dict],
    recovery_config: Path | None = None,
) -> dict:
    """Compose complete measurements while retaining all failed and repeated work."""
    import h5py

    completion_dir = audit.base / "near-upright-completion"
    if not completion_dir.exists():
        audit.check(recovery_config is None, "No recovery configuration without saved completion")
        return _verify_part(audit, "near-upright", spec, config, rows, traces, passes)
    audit.check(
        recovery_config is not None,
        "Storage completion requires explicit frozen recovery configuration",
    )
    assert recovery_config is not None
    amendment = audit.load(completion_dir / "amendment.json")
    audit.check(
        audit.load(recovery_config) == amendment
        and sha(recovery_config) == sha(completion_dir / "amendment.json"),
        "Saved completion binds the separately frozen recovery protocol",
    )
    audit.check(
        amendment["protocol"] == "acrobot-near-upright-storage-completion"
        and amendment["version"] == "1.1R"
        and amendment["retained_trial_indices"] == [0, 1]
        and amendment["completion_trial_indices"] == [2, 3]
        and amendment["partial_controls"] == 429
        and amendment["prefix_atol"] == REPLAY_ATOL
        and amendment["no_new_scientific_conditions"],
        "Explicit storage-only completion protocol",
    )
    audit.check(
        amendment["limits"]
        == dict(
            cpu_seconds=635,
            wall_seconds=1400,
            output_bytes=128 * 1024**2,
            rss_bytes=4 * 1024**3,
            native_controls=1000,
            forecast_controls=38553600,
            physics_substeps=77109200,
        )
        and amendment["expected_previous_controls"] == 1429
        and amendment["expected_previous_queries"] == 1436,
        "Predeclared remaining budget and previous-work bindings",
    )
    original_dir = audit.base / "near-upright"
    audit.check(
        sha(config) == amendment["original_config_sha256"], "Original scientific inputs unchanged"
    )
    audit.check(
        sha(original_dir / "result.json") == amendment["previous_result_sha256"]
        and sha(original_dir / "evidence.h5") == amendment["previous_evidence_sha256"],
        "Original failure bytes preserved",
    )
    manifest = audit.load(audit.base / "source/source_manifest.json")
    audit.check(
        sha(audit.base / "source/source_manifest.json")
        == amendment["frozen_source_manifest_sha256"]
        and manifest["source_commit"] == amendment["frozen_worker_source_commit"],
        "Completion executes the original frozen worker, separate from current reviewer",
    )
    original = _verify_part(
        audit,
        "near-upright",
        spec,
        config,
        rows,
        traces,
        passes,
        trial_indices=(0, 1, 2),
        accepted_status="STOPPED_RESOURCES",
        partial_index=2,
    )
    completion = _verify_part(
        audit,
        "near-upright",
        spec,
        config,
        rows,
        traces,
        passes,
        run_name="near-upright-completion",
        trial_indices=(2, 3),
        accepted_status="COMPLETED_NEAR_UPRIGHT_RECOVERY",
        limits_override=amendment["limits"],
    )
    old, new = original["result"], completion["result"]
    audit.check(
        new["original_failure_preserved"]
        and new["retained_original_trial_indices"] == [0, 1]
        and new["completed_trial_indices"] == [2, 3]
        and new["prior_work"] == old["work"]
        and new["prior_resources"] == old["resources"],
        "Overlay retains completed old cases and preserves the paid failed prefix",
    )
    verify_recovery_input_alias(
        audit,
        audit.load(original_dir / "inputs.json"),
        audit.load(completion_dir / "inputs.json"),
        amendment["original_config_path"],
    )
    for filename in ("environment.json", "config.json"):
        audit.check(
            audit.load(completion_dir / filename) == audit.load(original_dir / filename),
            "Original native source/runtime bindings unchanged: " + filename,
        )
    bindings = audit.load(completion_dir / "recovery_bindings.json")
    for name in (
        "frozen_source_manifest_sha256",
        "previous_result_sha256",
        "previous_evidence_sha256",
    ):
        audit.check(bindings[name] == amendment[name], "Recovery binding: " + name)
    audit.check(
        bindings["protocol_sha256"] == sha(completion_dir / "amendment.json")
        and bindings["wrapper_sha256"] == sha(completion_dir / "wrapper_source.py")
        and bindings["remaining_limits"] == amendment["limits"],
        "Separately preserved wrapper and remaining limits",
    )
    prefix_receipt = audit.load(completion_dir / "prefix_check.json")
    with (
        h5py.File(original_dir / "evidence.h5", "r") as previous_raw,
        h5py.File(completion_dir / "evidence.h5", "r") as completion_raw,
    ):
        prefix_errors = verify_repeated_prefix(
            audit,
            previous_raw["trials/variant_0/trial_2"],
            completion_raw["trials/variant_0/trial_2"],
            prefix_receipt,
        )
    paid = {
        key: original["expected"][key] + completion["expected"][key] for key in original["expected"]
    }
    duplicate = dict(
        native_controls=429,
        forecast_controls=16588800,
        model_queries=432,
        verification_controls=0,
        physics_substeps=33178458,
        restores=0,
    )
    audit.check(
        new["partial_repeat"]
        == {
            key: duplicate[key]
            for key in ("native_controls", "forecast_controls", "model_queries", "physics_substeps")
        },
        "Repeated-prefix paid census",
    )
    for name, number in old["work"].items():
        audit.check(
            new["cumulative_paid_work"][name] == number + new["work"][name],
            "Cumulative paid work includes original failure: " + name,
        )
    shared = amendment["shared_limits"]
    audit.check(
        shared["original_physics_substeps"] == spec["shared_limits"]["physics_substeps"]
        and shared["additional_repeated_physics_substeps"] == duplicate["physics_substeps"]
        and shared["amended_physics_substeps"]
        == shared["original_physics_substeps"] + duplicate["physics_substeps"],
        "Explicit repeated-work amendment, original resource ceiling retained",
    )
    audit.check(
        shared["cpu_seconds"] == spec["shared_limits"]["cpu_seconds"]
        and shared["output_bytes"] == spec["shared_limits"]["output_bytes"],
        "Shared time and storage ceilings remain unchanged",
    )
    resources = {
        name: max(old["resources"][name], new["resources"][name])
        if name == "rss_bytes"
        else old["resources"][name] + new["resources"][name]
        for name in old["resources"]
    }
    for name in ("cpu_seconds", "wall_seconds"):
        audit.check(
            resources[name] < spec["parts"]["near-upright"]["limits"][name],
            "Completion does not reset the original paid time budget: " + name,
        )
    return dict(
        result=dict(status=new["status"], resources=resources),
        expected=paid,
        canonical_work={key: paid[key] - duplicate[key] for key in paid},
        recovery=dict(
            protocol="1.1R",
            original_status=old["status"],
            original_failure_preserved=True,
            original_error=old["error"],
            original_work=original["expected"],
            completion_work=completion["expected"],
            repeated_partial_work=duplicate,
            canonical_work={key: paid[key] - duplicate[key] for key in paid},
            paid_work=paid,
            prefix_maximum_absolute_errors=prefix_errors,
            original_partial_measurements=original["partial_measurement"],
            original_incomplete_banks=original["incomplete_banks"],
            original_resource_receipt=old["resources"],
            completion_resource_receipt=new["resources"],
            shared_limits=shared,
        ),
    )


def render(
    base: Path,
    config: Path,
    out: Path,
    cpu_seconds: float = 180,
    *,
    recovery_config: Path | None = None,
) -> dict:
    """Verify a closed saved study; failures retain a non-certifying result."""
    base, config, out = base.resolve(), config.resolve(), out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    audit = SavedAudit(base, cpu_seconds)
    result: dict[str, Any] = dict(
        status="STARTING",
        new_physics_controls=0,
        new_network_forwards=0,
        new_fitting_updates=0,
        limitations=[
            "Saved-array verification only; no physics or neural inference was repeated.",
            "Two seeds per energy recipe are exploratory; they do not qualify a nine-of-ten teacher or a learning loop.",
            "Height-based success is unchanged and is not a hidden velocity criterion.",
            "All candidate scalar summaries are recorded, but only two complete candidate trajectories per bank are retained; unretained physics cannot be independently replayed.",
            "Cross-host RNG hashes are reported separately from declared numerical update comparisons; no byte identity is inferred from a tolerance pass.",
            "The near-upright diagnostic keeps the previous planner; energy recipes also change planner settings and duration, so this is not an isolated cost-only causal comparison.",
        ],
    )
    rows, traces, passes = [], [], []
    try:
        require(0 <= cpu_seconds <= 180, "Saved audit CPU allocation exceeds frozen ceiling")
        spec = audit.load(config)
        require(
            spec["protocol"] == "acrobot-energy-discovery-probe" and spec["version"] == "1.1",
            "Unknown frozen probe",
        )
        preflight = _verify_preflight(audit, spec, config)
        parts = dict(
            near_upright=_verify_near(audit, spec, config, rows, traces, passes, recovery_config),
            energy=_verify_part(audit, "energy", spec, config, rows, traces, passes),
        )
        selected = decision(rows, spec)
        expected_status = (
            "PROMISING_TEACHER_PROBE_ONLY"
            if selected["promising_variants"]
            else "TEACHER_PROBLEM_STOP"
        )
        audit.check(
            parts["energy"]["result"]["status"] == expected_status,
            "Independent energy kill decision",
        )
        audit.check(
            parts["energy"]["result"]["selected_variant"] == selected["selected_variant"],
            "First-declared promising recipe selection",
        )
        totals = {
            key: preflight["expected"][key] + sum(x["expected"][key] for x in parts.values())
            for key in (
                "native_controls",
                "forecast_controls",
                "verification_controls",
                "model_queries",
                "physics_substeps",
            )
        }
        recovery = parts["near_upright"].get("recovery")
        physics_ceiling = (
            recovery["shared_limits"]["amended_physics_substeps"]
            if recovery
            else spec["shared_limits"]["physics_substeps"]
        )
        audit.check(
            totals["physics_substeps"] <= physics_ceiling, "Explicit shared paid physics cap"
        )
        if recovery:
            result["storage_completion"] = recovery
            result["limitations"].append(
                "The original near-upright output cap failed and remains recorded. Complete cases0/1 "
                "plus separately repeated cases2/3 form the canonical roster. The paid429-control "
                "prefix and its unexecuted partially written bank are billed separately; the "
                "incomplete bank is inventoried, not numerically certified."
            )
        write_csv(out / "episodes.csv", rows)
        write_csv(out / "planner_passes.csv", passes)
        _plot(out / "figure.png", traces)
        result.update(
            status="PASSED",
            decision=selected,
            recomputed_work=totals,
            part_work={k: v["expected"] for k, v in parts.items()},
            resource_receipts={k: v["result"]["resources"] for k, v in parts.items()},
            maximum_cost_error=max(x["maximum_running_cost_error"] for x in rows),
        )
        _report(out / "report.md", rows, result)
    except Exception as error:
        result.update(status="FAILED_OR_INCOMPLETE", error=f"{type(error).__name__}: {error}")
        (out / "report.md").write_text(
            "# Acrobot energy probe\n\nSaved evidence could not be certified: "
            + result["error"]
            + "\n"
        )
    result.update(
        check_count=len(audit.checks),
        checks=audit.checks,
        inputs=audit.inputs,
        maximum_numeric_error=audit.max_error,
        maximum_error_in_declared_bound=audit.max_error_in_bound,
        white_regeneration_cpu_seconds=audit.regeneration_cpu_seconds,
        white_regeneration_cpu_limit_seconds=60,
        report_cpu_seconds=time.process_time() - audit.started,
        cost_roundoff_contract="1e-10 + 64*eps64*max(1,abs(expected)); replay remains rtol0/atol1e-10",
    )
    write_json(out / "independent_audit.json", result)
    return result


def _plot(path: Path, traces: list[tuple]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(row["variant"] for row, *_ in traces))
    figure, axes = plt.subplots(
        3, len(names), figsize=(4 * len(names), 8), sharex="col", squeeze=False
    )
    for column, name in enumerate(names):
        for row, states, actions, comp in traces:
            if row["variant"] != name:
                continue
            t = np.arange(1, len(actions) + 1) * 0.02
            label = str(row["reset_seed"])
            line = axes[0, column].plot(t, comp["height"], label=label, lw=1)[0]
            color = line.get_color()
            axes[1, column].plot(t, comp["total_energy"], color=color, lw=1)
            axes[2, column].plot(t, comp["potential"], color=color, lw=1)
            axes[2, column].plot(t, comp["kinetic"], color=color, lw=1, ls="--")
        axes[0, column].axhline(3.8, color="black", ls=":", lw=1)
        axes[0, column].axvspan(t[-1] - 2, t[-1], color="gray", alpha=0.1)
        axes[0, column].set_ylim(-0.1, 4.2)
        axes[1, column].axhline(58.86, color="black", ls=":", lw=1)
        axes[0, column].set_title(name.replace("_", " "), fontsize=9)
        axes[0, column].legend(title="Reset seed", fontsize=6, title_fontsize=6)
        axes[2, column].set_xlabel("Time (s)")
    for i, label in enumerate(("Tip height (m)", "Total energy (J)", "PE solid / KE dashed (J)")):
        axes[i, 0].set_ylabel(label)
    figure.suptitle("Acrobot probes: every complete native execution; no learning loop")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def _report(path: Path, rows: list[dict], audit: dict) -> None:
    d = audit["decision"]
    near = d["near_upright_successes"]
    n = d["near_upright_episodes"]
    verdict = (
        "No energy recipe passed both seeds: Acrobot remains a teacher-discovery/control problem."
        if not d["promising_variants"]
        else "At least one energy recipe passed both seeds; this is an exploratory teacher screen."
    )
    lines = [
        "# Acrobot near-upright and energy probes",
        "",
        f"**{verdict}**",
        f"Near upright, the unchanged prior controller held from **{near}/{n}** starts. "
        "No policy was fitted and no alternating loop ran.",
        "",
        "| Part / recipe | Height successes | Maximum height | Mean objective cost | Mean original dense cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for part, name in dict.fromkeys((x["part"], x["variant"]) for x in rows):
        selected = [x for x in rows if x["part"] == part and x["variant"] == name]
        lines.append(
            f"| {part}: {name} | {sum(x['success'] for x in selected)}/{len(selected)} | "
            f"{max(x['maximum_height'] for x in selected):.3f} m | "
            f"{np.mean([x['task_cost'] for x in selected]):.3f} | "
            f"{np.mean([x['original_dense_cost_sum'] for x in selected]):.3f} |"
        )
    lines += [
        "",
        "Success is unchanged: tip height at least 3.8 m at every final-two-second control boundary. "
        "Near-upright runs last ten seconds; energy runs last twelve seconds. Energy regulation alone "
        "does not count as capture. Objective costs use different teacher costs and are not directly comparable.",
        "",
        "The figure shows all native heights, total energy, potential and kinetic energy. Per-episode "
        "tables separately report holds, speeds and target-energy/low-height fractions. Candidate height "
        "statistics explain search behavior; they are not admission gates.",
        "",
        "The energy recipes reuse the historical pose, proximity and velocity terms, with a corrected "
        "39.24 J hanging-to-upright normalization, signed energy-error penalty and explicit balance handoff. "
        "The old catches were shorter holds and do not establish the present result.",
        "",
        "Saved verification recomputed costs, physical outcomes, source/input bindings, weights, white-noise "
        "updates, selected candidate costs, executed-action bindings and complete work counts. "
        "Full unselected candidate trajectories were not saved; no physics was repeated.",
        "",
        "If no energy recipe passes 2/2, the declared stop applies to this teacher probe. A positive "
        "2/2 result still requires a fresh 9/10 teacher qualification before any learning-loop claim.",
        "",
    ]
    if "storage_completion" in audit:
        recovery = audit["storage_completion"]
        lines += [
            "The original near-upright run exceeded its storage limit after two complete episodes "
            "and429 controls of the third. It remains an unchanged failure. A separately frozen "
            "storage-only completion repeated the third start, matched all429 saved controls and "
            "contexts at the original strict tolerance, then ran the fourth start. This is a fresh "
            "deterministic repeat, not a resumed simulator state.",
            "",
            "Accounting appendix: near-upright canonical episodes use "
            f"{recovery['canonical_work']['native_controls']:,} executed controls; all paid near-upright "
            f"work includes{recovery['paid_work']['native_controls']:,} executed controls and "
            f"{recovery['paid_work']['physics_substeps']:,} physics substeps. The repeated partial "
            "prefix is charged once in the paid census and excluded from the canonical outcome roster. "
            "Its final unexecuted, partially written bank remains available but uncertified.",
            "",
        ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--recovery-config",
        type=Path,
        help="Separately frozen1.1R amendment; required for a saved near-upright-completion overlay",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    result = render(
        args.base, args.config, args.out, args.cpu_seconds, recovery_config=args.recovery_config
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "decision", "error", "check_count", "report_cpu_seconds")
                if key in result
            },
            indent=2,
        )
    )
    return 0 if result["status"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
