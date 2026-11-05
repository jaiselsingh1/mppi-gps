"""Bounded Acrobot teacher qualification; native imports occur only at execution.

This stage runs six development episodes, selects one declared MPPI recipe by
success count / native cost / recipe order, and runs ten fresh qualification
episodes. Failure to balance is data. Physics, binding or resource failures stop
the stage while keeping partial HDF5 evidence. There is no learner in this stage.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.gps.mppi_gps.provenance import sha, write
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[3]
REQUIRED_TASK = {
    "reset_std": 0.05,
    "frame_skip": 2,
    "controls": 500,
    "balance_seconds": 2.0,
    "balance_height": 3.8,
}
SELECTION = "maximize development successes, then minimize mean native task cost, then recipe index"


class InvalidExecution(RuntimeError):
    """A binding or deterministic execution contract failed."""


class ResourceLimit(RuntimeError):
    """A frozen resource reservation cannot be satisfied."""


def validate_config(config: dict[str, Any]) -> None:
    """Reject changes to the frozen stage design before constructing a simulator."""
    if config.get("protocol") != "acrobot-controller-teacher-prerequisite":
        raise ValueError("Unknown Acrobot teacher protocol.")
    if config.get("version") != "1.0" or config.get("task") != REQUIRED_TASK:
        raise ValueError("Teacher stage 1.0 task values are frozen.")
    if config.get("threads") != 2 or config.get("math_threads") != 1:
        raise ValueError("Use exactly two physics threads and one math thread.")
    if config.get("selection") != SELECTION:
        raise ValueError("Development selection rule is frozen.")
    if config.get("qualification_successes_required") != 9:
        raise ValueError("The qualification gate is nine successes; it cannot be relaxed.")
    if config.get("qualification_episode_count") != 10:
        raise ValueError("Qualification requires ten complete episodes.")
    recipes = config.get("recipes", [])
    if len(recipes) != 3 or len({r["name"] for r in recipes}) != 3:
        raise ValueError("Declare exactly three distinct development recipes.")
    for recipe in recipes:
        planner = MPPIConfig(**recipe["planner"])
        if not (type(planner.H) is int and planner.H >= 2):
            raise ValueError("Planner horizon must be an integer >= 2.")
        if not (type(planner.K) is int and planner.K >= 1):
            raise ValueError("Planner sample count must be a positive integer.")
        if type(planner.initial_iterations) is not int or planner.initial_iterations < 1:
            raise ValueError("Initial refinement passes must be a positive integer.")
        if not np.isfinite([planner.lam, planner.noise_sigma]).all():
            raise ValueError("Planner temperature and noise must be finite.")
        if planner.lam <= 0 or planner.noise_sigma <= 0 or planner.use_is_correction:
            raise ValueError("Require positive planner temperature/noise and no IS correction.")
        if planner.noise_lowpass_cutoff_hz is not None:
            if planner.noise_lowpass_sample_rate_hz != 50.0:
                raise ValueError("Filtered Acrobot noise must use the native 50 Hz rate.")
            if not 0 < planner.noise_lowpass_cutoff_hz < 25:
                raise ValueError("Filtered noise cutoff must lie below native Nyquist.")
    reset_seeds, planner_seeds = [], []
    for stage, count in (("development", 2), ("qualification", 10)):
        trials = config.get(stage, [])
        if len(trials) != count:
            raise ValueError(f"{stage} requires exactly {count} seed pairs.")
        for trial in trials:
            reset, planner = trial["reset_seed"], trial["planner_seed"]
            if type(reset) is not int or reset < 0 or planner != reset + 10000:
                raise ValueError("Use private nonnegative reset seeds with planner seed +10000.")
            reset_seeds.append(reset)
            planner_seeds.append(planner)
    if len(set(reset_seeds)) != 12 or len(set(planner_seeds)) != 12:
        raise ValueError("Development and qualification seeds must be unique and disjoint.")
    preflight = config.get("preflight", {})
    if preflight.get("controls") != 12 or preflight.get("atol") != 1e-10:
        raise ValueError("Preflight requires 12 controls and the original 1e-10 tolerance.")
    if type(preflight.get("seed")) is not int or preflight["seed"] in reset_seeds:
        raise ValueError("Preflight needs a distinct private reset seed.")
    limits = config.get("limits", {})
    ceilings = {
        "cpu_seconds": 10800,
        "wall_seconds": 7200,
        "output_bytes": 512 * 1024**2,
        "rss_bytes": 4 * 1024**3,
    }
    for key in (*ceilings, "native_controls", "forecast_controls", "physics_substeps"):
        if type(limits.get(key)) is not int or limits[key] <= 0:
            raise ValueError(f"Missing positive integer resource limit: {key}.")
        if key in ceilings and limits[key] > ceilings[key]:
            raise ValueError(f"The frozen {key} ceiling cannot be increased.")
    if limits["native_controls"] < 16 * 500 + 36:
        raise ValueError("Native reservation must cover all 16 episodes plus preflight.")
    for binding in config.get("bindings", []):
        if set(binding) != {"path", "sha256"} or len(binding["sha256"]) != 64:
            raise ValueError("Each input binding needs an exact path and SHA-256.")


def choose_recipe(rows: list[dict[str, Any]], recipe_count: int = 3) -> int:
    """Select using development outcomes only, including unsuccessful episodes."""
    keys = []
    for index in range(recipe_count):
        selected = [row for row in rows if row["recipe_index"] == index]
        if len(selected) != 2 or not all(row["controls"] == 500 for row in selected):
            raise InvalidExecution(
                "Recipe selection requires all six complete development episodes."
            )
        cost = float(np.mean([row["task_cost"] for row in selected]))
        if not np.isfinite(cost):
            raise InvalidExecution("Recipe selection received a nonfinite cost.")
        successes = sum(bool(row["success"]) for row in selected)
        keys.append((-successes, cost, index))
    return min(keys)[2]


def episode_metrics(states: np.ndarray, costs: np.ndarray, dt: float = 0.02) -> dict[str, Any]:
    """Measure complete POST-action trajectories, without a hidden velocity gate."""
    states, costs = np.asarray(states), np.asarray(costs)
    if states.shape != (500, 5) or costs.shape != (500,):
        raise InvalidExecution("A qualified episode requires all 500 executed controls.")
    if not np.isfinite(states).all() or not np.isfinite(costs).all():
        raise InvalidExecution(
            "Nonfinite native measurements cannot be classified as task failure."
        )
    height = 2 + np.cos(states[:, 1]) + np.cos(states[:, 1] + states[:, 2])
    high = height >= REQUIRED_TASK["balance_height"]
    hold = math.ceil(REQUIRED_TASK["balance_seconds"] / dt)
    first = np.flatnonzero(high)
    sustained = np.flatnonzero(np.convolve(high.astype(int), np.ones(hold, int), "valid") == hold)
    return {
        "controls": len(costs),
        "success": bool(high[-hold:].all()),
        "task_cost": float(costs.sum()),
        "first_tip_above_threshold_seconds": float((first[0] + 1) * dt) if len(first) else None,
        "first_complete_hold_start_seconds": float((sustained[0] + 1) * dt)
        if len(sustained)
        else None,
        "time_balanced_seconds": float(high.sum() * dt),
        "final_window_minimum_height": float(height[-hold:].min()),
        "final_window_velocity_rms": float(np.sqrt(np.mean(states[-hold:, 3:5] ** 2))),
        "terminal_angles_raw": states[-1, 1:3].tolist(),
        "terminal_winding": np.floor((states[-1, 1:3] + np.pi) / (2 * np.pi)).astype(int).tolist(),
    }


@dataclass
class Work:
    native_controls_attempted: int = 0
    native_controls_completed: int = 0
    verification_controls_attempted: int = 0
    verification_controls_completed: int = 0
    forecast_controls_attempted: int = 0
    forecast_controls_completed: int = 0
    model_queries_attempted: int = 0
    model_queries_completed: int = 0
    physics_substeps_attempted: int = 0
    physics_substeps_completed: int = 0
    restores_attempted: int = 0
    restores_completed: int = 0
    resets: int = 0
    forwards: int = 0
    network_forwards: int = 0
    backward_calls: int = 0
    optimizer_updates: int = 0


class Budget:
    """Reserve whole native calls before execution; never hide failed paid attempts."""

    def __init__(self, limits: dict[str, int], out: Path) -> None:
        self.limits, self.out, self.work = limits, out, Work()
        self.cpu_start, self.wall_start = time.process_time(), time.monotonic()

    def observations(self) -> dict[str, float | int]:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != "darwin":
            rss *= 1024
        return {
            "cpu_seconds": time.process_time() - self.cpu_start,
            "wall_seconds": time.monotonic() - self.wall_start,
            "rss_bytes": int(rss),
            "output_bytes": sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file()),
        }

    def check(self, output_reservation: int = 0) -> None:
        values = self.observations()
        # Keep enough room to close the HDF5 file and preserve a resource-stop receipt.
        values["output_bytes"] += output_reservation + 256 * 1024
        for key, value in values.items():
            if value >= self.limits[key]:
                raise ResourceLimit(
                    f"Frozen {key} budget exhausted: {value} >= {self.limits[key]}."
                )

    def reserve(self, kind: str, controls: int, frame_skip: int) -> None:
        self.check()
        if kind not in ("native", "verification", "forecast") or controls < 1:
            raise ValueError("Unknown work type or nonpositive native reservation.")
        native = self.work.native_controls_attempted + self.work.verification_controls_attempted
        if (
            kind in ("native", "verification")
            and native + controls > self.limits["native_controls"]
        ):
            raise ResourceLimit("Native-control reservation exhausted.")
        if kind == "forecast" and (
            self.work.forecast_controls_attempted + controls > self.limits["forecast_controls"]
        ):
            raise ResourceLimit("Forecast-control reservation exhausted.")
        physics = controls * frame_skip
        if self.work.physics_substeps_attempted + physics > self.limits["physics_substeps"]:
            raise ResourceLimit("Physics-substep reservation exhausted.")
        field = f"{kind}_controls_attempted"
        setattr(self.work, field, getattr(self.work, field) + controls)
        self.work.physics_substeps_attempted += physics
        if kind == "forecast":
            self.work.model_queries_attempted += 1

    def complete(self, kind: str, controls: int, frame_skip: int) -> None:
        field = f"{kind}_controls_completed"
        setattr(self.work, field, getattr(self.work, field) + controls)
        self.work.physics_substeps_completed += controls * frame_skip
        if kind == "forecast":
            self.work.model_queries_completed += 1


def _assert_close(actual: np.ndarray, expected: np.ndarray, label: str, atol: float) -> None:
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise InvalidExecution(f"Nonfinite {label}.")
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=atol):
        raise InvalidExecution(f"Strict replay mismatch: {label} (rtol=0, atol={atol}).")


def _source_bindings(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    records = []
    source_paths = (
        "src/gps/mppi_gps/acrobot_teacher.py",
        "src/gps/mppi_gps/provenance.py",
        "src/envs/acrobot.py",
        "src/envs/mujoco_env.py",
        "src/envs/base.py",
        "src/mppi/mppi.py",
        "src/utils/config.py",
        "src/utils/math.py",
    )
    for path in source_paths:
        target = ROOT / path
        records.append({"path": str(target), "sha256": sha(target), "bytes": target.stat().st_size})
    for binding in config.get("bindings", []):
        target = Path(binding["path"])
        if not target.is_absolute():
            target = ROOT / target
        if sha(target) != binding["sha256"]:
            raise InvalidExecution(f"Input binding changed: {binding['path']}.")
        records.append({"path": str(target), "sha256": sha(target), "bytes": target.stat().st_size})
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    source_manifest = ROOT / "source_manifest.json"
    verified_manifest = None
    if source_manifest.exists():
        manifest = json.loads(source_manifest.read_text())
        if manifest.get("config_sha256", sha(config_path)) != sha(config_path):
            raise InvalidExecution("Frozen source manifest binds a different configuration.")
        entries = manifest.get("files", manifest.get("payloads", []))
        if isinstance(entries, dict):
            entries = [dict(path=path, **record) for path, record in entries.items()]
        for entry in entries:
            relative = entry.get("path", entry.get("relative_path"))
            if relative is None or "sha256" not in entry:
                raise InvalidExecution("Malformed frozen source-manifest entry.")
            target = ROOT / relative
            if sha(target) != entry["sha256"]:
                raise InvalidExecution(f"Frozen source payload changed: {relative}.")
        if not entries:
            raise InvalidExecution("Frozen source manifest contains no payloads.")
        verified_manifest = {
            "path": str(source_manifest),
            "sha256": sha(source_manifest),
            "verified_payload_count": len(entries),
        }
    return {
        "git_head": head,
        "config_path": str(config_path),
        "config_sha256": sha(config_path),
        "files": records,
        "frozen_source_manifest": verified_manifest,
    }


def _environment(config: dict[str, Any]) -> dict[str, Any]:
    versions = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "mujoco", "h5py"):
        versions[name] = importlib.metadata.version(name)
    expected = config.get("expected_environment", {})
    if expected and versions != expected:
        raise InvalidExecution(f"Native environment changed: expected {expected}; got {versions}.")
    math_threads = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    if any(value != "1" for value in math_threads.values()):
        raise InvalidExecution(
            "Freeze OMP/MKL/OPENBLAS_NUM_THREADS=1 before importing native libraries."
        )
    return {
        "versions": versions,
        "platform": platform.platform(),
        "executable": sys.executable,
        "math_threads": math_threads,
        "physics_threads": config["threads"],
    }


def _capture(mj: Any, env: Any) -> dict[str, np.ndarray]:
    spec = mj.mjtState.mjSTATE_INTEGRATION
    integration = np.empty(mj.mj_stateSize(env.model, spec))
    mj.mj_getState(env.model, env.data, integration, spec)
    return {
        "states": env.get_state(),
        "warmstarts": env.get_warmstart(),
        "integration": integration,
    }


def _save_array(group: Any, name: str, value: np.ndarray, budget: Budget) -> None:
    value = np.asarray(value)
    budget.check(output_reservation=value.nbytes + 4096)
    group.create_dataset(name, data=value)


def _preflight(
    mj: Any,
    env: Any,
    group: Any,
    budget: Budget,
    config: dict[str, Any],
    *,
    include_terminal_cost: bool = False,
) -> dict[str, Any]:
    """36 controls: original, complete-context replay, and one native batch replay."""
    from src.envs.acrobot import tip_position

    geometry_states = np.array([[0, 0, 0, 0, 0], [0, np.pi, 0, 0, 0], [0, 0.6, -1.2, 0, 0]])
    native_positions = []
    for state in geometry_states:
        env.reset(state=state)
        budget.work.resets += 1
        budget.work.forwards += 1
        native_positions.append(env.data.site("tip").xpos.copy())
    native_positions = np.asarray(native_positions)
    _assert_close(native_positions, tip_position(geometry_states), "native tip geometry", 1e-14)
    _save_array(group, "geometry_states", geometry_states, budget)
    _save_array(group, "geometry_native_tip", native_positions, budget)
    env.reset(seed=config["seed"])
    budget.work.resets += 1
    budget.work.forwards += 2
    original = _capture(mj, env)
    actions = np.linspace(-0.7, 0.8, 12)[:, None]
    expected, actual = {}, {}
    for label, measurements in (("recorded", expected), ("restored", actual)):
        if label == "restored":
            budget.work.restores_attempted += 1
            mj.mj_setState(
                env.model, env.data, original["integration"], mj.mjtState.mjSTATE_INTEGRATION
            )
            restored = _capture(mj, env)
            for key in original:
                if (
                    restored[key].shape != original[key].shape
                    or restored[key].dtype != original[key].dtype
                    or restored[key].tobytes() != original[key].tobytes()
                ):
                    raise InvalidExecution(f"Byte-exact restoration failed: {key}.")
            mj.mj_forward(env.model, env.data)
            budget.work.forwards += 1
            # Forward recomputes derived quantities; restore the exact captured integration again.
            mj.mj_setState(
                env.model, env.data, original["integration"], mj.mjtState.mjSTATE_INTEGRATION
            )
            budget.work.restores_completed += 1
        snapshots = [original]
        costs = []
        for action in actions:
            budget.reserve("verification", 1, env._frame_skip)
            _, cost, done, _ = env.step(action)
            budget.complete("verification", 1, env._frame_skip)
            if done:
                raise InvalidExecution("Acrobot unexpectedly terminated during replay.")
            snapshots.append(_capture(mj, env))
            costs.append(cost)
        for key in original:
            measurements[key] = np.stack([snapshot[key] for snapshot in snapshots])
        measurements["costs"] = np.asarray(costs)
        subgroup = group.create_group(label)
        for key, value in measurements.items():
            _save_array(subgroup, key, value, budget)
    atol = config["atol"]
    for key in expected:
        _assert_close(actual[key], expected[key], f"sequential {key}", atol)
    budget.reserve("verification", 12, env._frame_skip)
    states, cost, _ = env.batch_rollout(original["states"], actions[None], original["warmstarts"])
    budget.complete("verification", 12, env._frame_skip)
    _save_array(group, "batch_states", states[0], budget)
    _save_array(group, "batch_cost", cost, budget)
    _save_array(group, "actions", actions, budget)
    _assert_close(states[0], expected["states"][1:], "batch states", atol)
    expected_total = expected["costs"].sum(keepdims=True)
    if include_terminal_cost:
        terminal = np.asarray(env.terminal_cost(expected["states"][-1:]), dtype=np.float64)
        _save_array(group, "terminal_cost", terminal, budget)
        expected_total = expected_total + terminal
    _assert_close(cost, expected_total, "batch total cost", atol)
    group.attrs["status"] = "PASSED"
    return {
        "status": "PASSED",
        "controls": 36,
        "physics_substeps": 36 * env._frame_skip,
        "rtol": 0,
        "atol": atol,
        "restoration_atol": 0,
    }


def _run_trial(
    mj: Any,
    env: Any,
    mppi_type: Any,
    group: Any,
    budget: Budget,
    recipe: dict[str, Any],
    seed_pair: dict[str, int],
    stage: str,
    recipe_index: int,
    trial_index: int,
    out: Path,
    *,
    controls: int = 500,
    initial_state: np.ndarray | None = None,
    metrics_fn: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], dict[str, Any]] | None = None,
    bank_callback: Callable[[Any, Any, Any, np.ndarray, Budget], None] | None = None,
) -> dict[str, Any]:
    """Record every optimization pass and every executed control, including failed trials."""
    import h5py

    group.attrs.update(
        status="RUNNING",
        stage=stage,
        recipe=recipe["name"],
        reset_seed=seed_pair["reset_seed"],
        planner_seed=seed_pair["planner_seed"],
    )
    if initial_state is None:
        env.reset(seed=seed_pair["reset_seed"])
    else:
        env.reset(state=np.asarray(initial_state, dtype=np.float64))
        group.attrs["initialization"] = "frozen_full_state"
    budget.work.resets += 1
    budget.work.forwards += 2 if initial_state is None else 1
    if initial_state is not None:
        expected_state = np.asarray(initial_state, dtype=np.float64)
        if env.get_state().tobytes() != expected_state.tobytes():
            raise InvalidExecution("The frozen initial state was not restored byte-exactly.")
    context = _capture(mj, env)
    lengths = {key: value.shape for key, value in context.items()}
    datasets = {}
    for key, shape in {
        **lengths,
        "pre_action_observations": (4,),
        "actions": (1,),
        "costs": (),
    }.items():
        datasets[key] = group.create_dataset(
            key, shape=(0, *shape), maxshape=(None, *shape), dtype="f8", chunks=(32, *shape)
        )

    def append(key: str, value: Any) -> None:
        value = np.asarray(value)
        budget.check(output_reservation=value.nbytes + 4096)
        dataset = datasets[key]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    for key, value in context.items():
        append(key, value)
    plans = group.create_group("planning")
    original_batch = env.batch_rollout
    active = {"group": None, "pass": 0}

    def counted_batch(state: np.ndarray, actions: np.ndarray, **kwargs: Any) -> Any:
        controls = int(actions.shape[0] * actions.shape[1])
        budget.reserve("forecast", controls, env._frame_skip)
        result = original_batch(state, actions, **kwargs)
        budget.complete("forecast", controls, env._frame_skip)
        if not all(np.isfinite(value).all() for value in result):
            raise InvalidExecution("Nonfinite candidate trajectory, cost, or sensor measurements.")
        expected_times = state[0] + np.arange(1, actions.shape[1] + 1) * env._dt
        if not np.allclose(result[0][..., 0], expected_times[None], rtol=0, atol=1e-10):
            raise InvalidExecution("A native candidate stopped advancing physical time.")
        return result

    class RecordedMPPI(mppi_type):
        def _plan_iteration(
            self,
            state: np.ndarray,
            prior_cost: Any,
            coupling: Any,
            initial_warmstart: np.ndarray | None,
        ) -> Any:
            before = self.U.copy()
            result = super()._plan_iteration(state, prior_cost, coupling, initial_warmstart)
            pass_group = active["group"].create_group(f"pass_{active['pass']:03d}")
            for name, value in (
                ("candidate_costs", self._last_costs),
                ("weights", self._last_weights),
                ("nominal_before", before),
                ("nominal_after", self.U),
            ):
                _save_array(pass_group, name, value, budget)
            pass_group.attrs["neff"] = result[1]["n_eff"]
            pass_group.attrs["lam"] = result[1]["lam"]
            if bank_callback is not None:
                bank_callback(pass_group, env, self, state, budget)
            active["pass"] += 1
            return result

    env.batch_rollout = counted_batch
    try:
        planner = RecordedMPPI(env, MPPIConfig(**recipe["planner"]))
        planner.reset(seed=seed_pair["planner_seed"])
        for t in range(controls):
            budget.check()
            before = _capture(mj, env)
            observation = env._get_obs()
            _assert_close(observation, before["states"][1:5], "pre-action observation", 0.0)
            active["group"] = plans.create_group(f"control_{t:04d}")
            active["pass"] = 0
            _save_array(active["group"], "pre_action_observation", observation, budget)
            action, _ = planner.plan_step(before["states"], initial_warmstart=before["warmstarts"])
            if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise InvalidExecution("MPPI returned an invalid bounded action.")
            budget.reserve("native", 1, env._frame_skip)
            _, cost, done, _ = env.step(action)
            budget.complete("native", 1, env._frame_skip)
            after = _capture(mj, env)
            if (
                done
                or not np.isfinite(cost)
                or not all(np.isfinite(v).all() for v in after.values())
            ):
                raise InvalidExecution("Invalid native Acrobot execution.")
            if not np.isclose(
                after["states"][0], before["states"][0] + env._dt, rtol=0, atol=1e-10
            ):
                raise InvalidExecution("Native Acrobot execution stopped advancing physical time.")
            append("pre_action_observations", observation)
            append("actions", action)
            append("costs", cost)
            for key, value in after.items():
                append(key, value)
            group.attrs["completed_controls"] = t + 1
            if (t + 1) % 10 == 0:
                group.file.flush()
            if (t + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "trial_progress",
                            "stage": stage,
                            "recipe": recipe["name"],
                            "trial_index": trial_index,
                            "controls": t + 1,
                            "task_cost": float(datasets["costs"][:].sum()),
                            "work": asdict(budget.work),
                            "resources": budget.observations(),
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
        if metrics_fn is None:
            metrics = episode_metrics(datasets["states"][1:], datasets["costs"][:], env._dt)
        else:
            metrics = metrics_fn(
                env, datasets["states"][1:], datasets["costs"][:], datasets["actions"][:]
            )
        result = {
            "stage": stage,
            "recipe_index": recipe_index,
            "recipe": recipe["name"],
            "trial_index": trial_index,
            **seed_pair,
            **metrics,
        }
        group.attrs["status"] = "COMPLETED"
        group.attrs["success"] = metrics["success"]
        group.attrs["task_cost"] = metrics["task_cost"]
        group.file.flush()
        with (out / "trials.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)
        return result
    except BaseException as error:
        group.attrs["status"] = "STOPPED"
        group.attrs["error"] = f"{type(error).__name__}: {error}"
        group.file.flush()
        raise
    finally:
        env.batch_rollout = original_batch
        if not isinstance(group.file, h5py.File):
            raise InvalidExecution("Evidence handle unexpectedly changed.")


def run(config_path: Path, out: Path) -> dict[str, Any]:
    """Execute only this finite teacher prerequisite; no later stage is launched."""
    config = json.loads(config_path.read_text())
    validate_config(config)
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_bytes(config_path.read_bytes())
    (out / "trials.jsonl").touch(exist_ok=False)
    budget = Budget(config["limits"], out)
    result: dict[str, Any] = {
        "protocol": config["protocol"],
        "version": config["version"],
        "status": "STARTING",
        "development": [],
        "qualification": [],
    }
    env = None
    old_alarm = old_cpu_handler = None
    old_cpu_limit = None

    def interrupted(signum: int, frame: Any) -> None:
        del frame
        raise ResourceLimit(f"Frozen resource timer signalled {signum}.")

    try:
        write(out / "inputs.json", _source_bindings(config, config_path))
        write(out / "environment.json", _environment(config))
        old_alarm = signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(config["limits"]["wall_seconds"])
        if sys.platform == "linux":
            old_cpu_handler = signal.signal(signal.SIGXCPU, interrupted)
            old_cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
            used = resource.getrusage(resource.RUSAGE_SELF)
            soft = math.ceil(used.ru_utime + used.ru_stime + config["limits"]["cpu_seconds"])
            hard = old_cpu_limit[1]
            if hard != resource.RLIM_INFINITY:
                soft = min(soft, hard)
            resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
        import h5py
        import mujoco

        from src.envs.acrobot import Acrobot
        from src.mppi.mppi import MPPI

        env = Acrobot(frame_skip=2, nthread=2, reset_std=0.05)
        if env._dt != 0.02:
            raise InvalidExecution("Native Acrobot control period changed.")
        with h5py.File(out / "evidence.h5", "x") as evidence:
            evidence.attrs["protocol"] = config["protocol"]
            evidence.attrs["version"] = config["version"]
            evidence.attrs["integration_spec"] = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            evidence.attrs["status"] = "RUNNING"
            result["preflight"] = _preflight(
                mujoco, env, evidence.create_group("preflight"), budget, config["preflight"]
            )
            evidence.flush()
            for index, recipe in enumerate(config["recipes"]):
                for trial_index, seeds in enumerate(config["development"]):
                    group = evidence.create_group(f"development/recipe_{index}/trial_{trial_index}")
                    result["development"].append(
                        _run_trial(
                            mujoco,
                            env,
                            MPPI,
                            group,
                            budget,
                            recipe,
                            seeds,
                            "development",
                            index,
                            trial_index,
                            out,
                        )
                    )
            selected = choose_recipe(result["development"])
            result["selected_recipe_index"] = selected
            result["selected_recipe"] = copy.deepcopy(config["recipes"][selected])
            write(
                out / "selection.json",
                {
                    "rule": SELECTION,
                    "selected_recipe_index": selected,
                    "recipe": result["selected_recipe"],
                    "development": result["development"],
                },
            )
            recipe = config["recipes"][selected]
            for trial_index, seeds in enumerate(config["qualification"]):
                group = evidence.create_group(f"qualification/trial_{trial_index}")
                result["qualification"].append(
                    _run_trial(
                        mujoco,
                        env,
                        MPPI,
                        group,
                        budget,
                        recipe,
                        seeds,
                        "qualification",
                        selected,
                        trial_index,
                        out,
                    )
                )
            successes = sum(row["success"] for row in result["qualification"])
            result["qualification_successes"] = successes
            result["status"] = "QUALIFIED" if successes >= 9 else "FAILED_TEACHER_GATE"
            evidence.attrs["status"] = result["status"]
    except ResourceLimit as error:
        result.update(status="STOPPED_RESOURCES", error=f"{type(error).__name__}: {error}")
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="STOPPED_INVALID_EXECUTION", error=f"{type(error).__name__}: {error}")
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                result.update(
                    status="STOPPED_INVALID_EXECUTION",
                    close_error=f"{type(error).__name__}: {error}",
                )
        evidence_path = out / "evidence.h5"
        if evidence_path.exists():
            try:
                import h5py

                with h5py.File(evidence_path, "r+") as evidence:
                    evidence.attrs["status"] = result["status"]
            except Exception as error:
                result["evidence_status_error"] = f"{type(error).__name__}: {error}"
        signal.alarm(0)
        if old_alarm is not None:
            signal.signal(signal.SIGALRM, old_alarm)
        if old_cpu_handler is not None:
            signal.signal(signal.SIGXCPU, old_cpu_handler)
        if old_cpu_limit is not None:
            resource.setrlimit(resource.RLIMIT_CPU, old_cpu_limit)
        result["work"] = asdict(budget.work)
        result["unknown_partial_work"] = {
            "model_queries": budget.work.model_queries_attempted
            - budget.work.model_queries_completed,
            "native_controls": budget.work.native_controls_attempted
            - budget.work.native_controls_completed,
            "verification_controls": budget.work.verification_controls_attempted
            - budget.work.verification_controls_completed,
            "forecast_controls": budget.work.forecast_controls_attempted
            - budget.work.forecast_controls_completed,
            "physics_substeps": budget.work.physics_substeps_attempted
            - budget.work.physics_substeps_completed,
        }
        result["evidence"] = {
            p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in out.iterdir()
            if p.is_file()
        }
        result["resources"] = budget.observations()
        write(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.out)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] in ("QUALIFIED", "FAILED_TEACHER_GATE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
