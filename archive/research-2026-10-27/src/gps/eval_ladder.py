"""Dedicated four-controller Walker evaluation, separate from training telemetry.

The policy callback maps (N, 17) observations to (N, 6) native actions. The
default controller implementation uses CPU MuJoCo/MPPI; a guided controller
factory can supply an accelerated implementation. Cold MPPI never calls the
policy and is evaluated only at iteration zero, then reused after fingerprint
and artifact-integrity checks. Regularity uses the executed traces only.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.envs.walker2d import Walker2d
from src.gps.mdgps import closed_loop_nominal
from src.gps.policy_tail import WalkerPolicyTail
from src.mppi.mppi import MPPI
from src.rl.walker import DEFAULT_TASK, WalkerRL
from src.utils.config import MPPIConfig

FloatArray: TypeAlias = NDArray[np.float64]
PolicyBatch: TypeAlias = Callable[[np.ndarray], np.ndarray]
CONTROLLERS = ("pi_standalone", "pi_noisy", "mppi_guided_online", "mppi_cold_online")
_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = 1


@dataclass(frozen=True)
class LadderConfig:
    """Fixed reset panel and regularity settings for dedicated evaluations."""

    episodes: int = 10
    steps: int = 400
    reset_seed_base: int = 8_300_000
    planner_seed_base: int = 9_300_000
    exec_noise: float = .1
    guided_neff_fraction: float | None = .2
    min_period_seconds: float = .16
    max_period_seconds: float = 1.2
    min_cycles: int = 2
    transient_seconds: float = .25
    oscillation_rms_min: float = .001

    def __post_init__(self) -> None:
        """Reject invalid horizons, seeds, and unidentifiable periodicity windows."""
        for name in ("episodes", "steps", "min_cycles"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_cycles < 2:
            raise ValueError("Periodicity requires at least two cycles")
        if (self.guided_neff_fraction is not None
                and (not math.isfinite(self.guided_neff_fraction) or not 0 < self.guided_neff_fraction < 1)):
            raise ValueError("guided_neff_fraction must be None or lie strictly between 0 and 1")
        for name in ("reset_seed_base", "planner_seed_base"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 2**32 - self.episodes:
                raise ValueError(f"{name} must leave room for the full reset panel")
        for name in ("exec_noise", "transient_seconds", "oscillation_rms_min"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (not math.isfinite(self.min_period_seconds) or not math.isfinite(self.max_period_seconds)
                or not 0 < self.min_period_seconds <= self.max_period_seconds):
            raise ValueError("Require 0 < min_period_seconds <= max_period_seconds")


class GuidedController(Protocol):
    """One episode's policy-centred controller; actions have shape (6,)."""

    def action(self, *, observation: FloatArray, state: FloatArray,
               warmstart: FloatArray) -> FloatArray:
        """Return a native action from the current live state and solver state."""
        ...

    def close(self) -> None:
        """Release controller-owned resources, without closing the live environment."""
        ...


class GuidedControllerFactory(Protocol):
    """Optional accelerated controller constructor, called separately per episode."""

    def __call__(self, *, env: Walker2d, act_batch: PolicyBatch, planner_cfg: MPPIConfig,
                 tail_steps: int, seed: int) -> GuidedController:
        """Construct a memoryless policy-nominal controller using the pure task score."""
        ...


class CPUGuidedController:
    """Fresh closed-loop policy nominal and one MPPI update at every decision."""

    def __init__(self, *, env: Walker2d, act_batch: PolicyBatch, planner_cfg: MPPIConfig,
                 tail_steps: int, seed: int) -> None:
        """Keep the policy frozen; the planner's shifted plan is never reused."""
        self.env: Walker2d = env
        self.act_batch: PolicyBatch = act_batch
        self.planner: MPPI = MPPI(env, replace(planner_cfg, initial_iterations=1), seed=seed)
        self.tail: WalkerPolicyTail | None = (
            WalkerPolicyTail(env, self._batch_action, tail_steps) if tail_steps else None)

    def _batch_action(self, observations: np.ndarray) -> FloatArray:
        """Map (N, 17) observations to checked (N, 6) policy actions."""
        return _policy_actions(self.act_batch, observations, self.env.action_dim)

    def _state_action(self, state: np.ndarray) -> FloatArray:
        """Map one (nstate,) full simulator state to one (6,) policy action."""
        return self._batch_action(self.env.rollout_states_to_obs(state)[None])[0]

    def action(self, *, observation: FloatArray, state: FloatArray,
               warmstart: FloatArray) -> FloatArray:
        """Discard shifted plans and resample around the current policy rollout."""
        self.planner.reset()
        nominal, _ = closed_loop_nominal(self.env, state, self._state_action,
                                         self.planner.H, warmstart)
        action, _ = self.planner.plan_step(state, nominal=nominal,
                                           prior_cost=self.tail, initial_warmstart=warmstart)
        return np.asarray(action, dtype=np.float64)

    def close(self) -> None:
        """The live environment owns the CPU rollout resources."""
        return None


@dataclass(frozen=True)
class PeriodicityResult:
    """Autocorrelation evidence; absent evidence is represented by null scores."""

    score: float | None
    period_seconds: float | None
    channels_used: int
    samples: int
    reason: str | None


def state_periodicity(states: FloatArray, *, dt: float, config: LadderConfig) -> PeriodicityResult:
    """Score an executed (T, D) torso/joint position trace, with x excluded.

    Remove a linear trend from each channel, discard nearly constant channels,
    and seek a positive *local* autocorrelation peak after a physical minimum
    period. Require at least ``min_cycles`` at the candidate lag. Thus a smooth
    short prefix, stationary pose, or monotonic drift cannot be called a cycle.
    """
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("Periodicity requires a finite (T, D) position trace")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    values = values[int(math.ceil(config.transient_seconds / dt)):]
    count = len(values)
    minimum = max(2, int(math.ceil(config.min_period_seconds / dt)))
    maximum = min(int(math.floor(config.max_period_seconds / dt)), (count - 1) // config.min_cycles)
    if maximum < minimum or count < 4:
        return PeriodicityResult(None, None, 0, count, "insufficient_cycles")
    times = np.linspace(-1., 1., count)
    centered = values - values.mean(axis=0)
    centered -= times[:, None] * (times @ centered / (times @ times))[None]
    active = np.sqrt(np.mean(centered**2, axis=0)) > max(config.oscillation_rms_min, 1e-12)
    used = int(np.sum(active))
    if not used:
        return PeriodicityResult(None, None, 0, count, "no_oscillation")
    centered = centered[:, active]
    correlations = np.empty(maximum + 2, dtype=np.float64)
    correlations[0] = 1.
    for lag in range(1, maximum + 2):
        first, second = centered[:-lag], centered[lag:]
        denominator = np.sqrt(np.sum(first**2, axis=0) * np.sum(second**2, axis=0))
        per_channel = np.divide(np.sum(first * second, axis=0), denominator,
                                out=np.zeros(used), where=denominator > 1e-20)
        correlations[lag] = float(np.mean(per_channel))
    peaks = [lag for lag in range(minimum, maximum + 1)
             if correlations[lag] > .1 and correlations[lag] > correlations[lag - 1]
             and correlations[lag] >= correlations[lag + 1]]
    if not peaks:
        return PeriodicityResult(None, None, used, count, "no_positive_periodic_peak")
    # Prefer the first comparably strong peak over an exact higher harmonic.
    # E.g. a period of 62.5 samples should not be reported as 125 samples just
    # because the second peak aligns more exactly with the discrete grid.
    strongest = max(correlations[lag] for lag in peaks)
    best = min(lag for lag in peaks if correlations[lag] >= strongest - .05)
    return PeriodicityResult(float(np.clip(correlations[best], -1., 1.)), best * dt,
                             used, count, None)


@dataclass(frozen=True)
class EpisodeMetrics:
    """Finite-horizon task accounting and executed-trace regularity."""

    controller: str
    reset_seed: int
    steps: int
    survived: bool
    executed_cost: float
    absorbing_cost: float
    total_cost: float
    return_value: float
    velocity_mae: float
    action_jerk: float | None
    periodicity_score: float | None
    period_seconds: float | None
    periodicity_reason: str | None
    periodicity_channels: int
    mean_action_seconds: float
    wall_time_seconds: float


@dataclass
class EpisodeTrace:
    """Ragged executed data: states (T+1,S), actions (T,A), scalar arrays (T,)."""

    metrics: EpisodeMetrics
    states: FloatArray
    actions: FloatArray
    sensors: FloatArray  # (T, sensor_dim), possibly zero-width
    costs: FloatArray
    velocities: FloatArray
    action_seconds: FloatArray
    simulation_seconds: FloatArray


def _policy_actions(act_batch: PolicyBatch, observations: np.ndarray, action_dim: int) -> FloatArray:
    """Validate the frozen policy's (N, A) output before any simulator use."""
    observations = np.asarray(observations, dtype=np.float64)
    actions = np.asarray(act_batch(observations), dtype=np.float64)
    if actions.shape != (len(observations), action_dim) or not np.all(np.isfinite(actions)):
        raise FloatingPointError("Policy must return finite (N, action_dim) actions")
    return actions


def _episode(*, controller: str, act_batch: PolicyBatch, config: LadderConfig,
             planner_cfg: MPPIConfig, tail_steps: int, task_config: Path, episode: int,
             guided_factory: GuidedControllerFactory | None) -> tuple[EpisodeTrace, list[str]]:
    """Execute one controller once; all cost and regularity use that same trace."""
    started = time.perf_counter()
    env = WalkerRL(task_config, max_episode_steps=config.steps)
    guided: GuidedController | None = None
    backends: list[str] = []
    seed = config.reset_seed_base + episode
    planner_seed = config.planner_seed_base + episode
    try:
        walker = env.walker
        observation, _ = env.reset(seed=seed)
        planner = MPPI(walker, planner_cfg, seed=planner_seed) if controller == "mppi_cold_online" else None
        if controller == "mppi_guided_online":
            guided_cfg = replace(planner_cfg, initial_iterations=1,
                                 lam_target_neff_fraction=config.guided_neff_fraction)
            constructor = CPUGuidedController if guided_factory is None else guided_factory
            try:
                guided = constructor(env=walker, act_batch=act_batch, planner_cfg=guided_cfg,
                                     tail_steps=tail_steps, seed=planner_seed)
            except (ImportError, RuntimeError) as exc:
                if guided_factory is None:
                    raise
                backends.append(f"factory failed, CPU fallback: {type(exc).__name__}: {exc}")
                guided = CPUGuidedController(env=walker, act_batch=act_batch, planner_cfg=guided_cfg,
                                             tail_steps=tail_steps, seed=planner_seed)
            backends.append(f"{type(guided).__module__}.{type(guided).__qualname__}")
        elif planner is not None:
            backends.append("src.mppi.mppi.MPPI (raw task score, no policy)")
        else:
            backends.append("policy callback")
        noise = np.random.default_rng(np.random.SeedSequence([planner_seed, 2]))
        low, high = walker.action_bounds
        states = [walker.get_state()]
        actions, sensors, costs, velocities, action_seconds, simulation_seconds = [], [], [], [], [], []
        terminated = False
        for step in range(config.steps):
            decision_started = time.perf_counter()
            if guided is not None:
                try:
                    action = guided.action(observation=observation.copy(), state=states[-1].copy(),
                                           warmstart=walker.get_warmstart())
                except (ImportError, RuntimeError) as exc:
                    if isinstance(guided, CPUGuidedController):
                        raise
                    guided.close()
                    backends.append(f"action failed at step {step}, CPU fallback: {type(exc).__name__}: {exc}")
                    guided = CPUGuidedController(env=walker, act_batch=act_batch, planner_cfg=guided_cfg,
                                                 tail_steps=tail_steps, seed=planner_seed)
                    action = guided.action(observation=observation.copy(), state=states[-1].copy(),
                                           warmstart=walker.get_warmstart())
            elif planner is not None:
                action, _ = planner.plan_step(states[-1], initial_warmstart=walker.get_warmstart())
            else:
                action = _policy_actions(act_batch, observation[None], walker.action_dim)[0]
                if controller == "pi_noisy":
                    action = action + config.exec_noise * noise.standard_normal(walker.action_dim)
            action = np.asarray(action, dtype=np.float64)
            if action.shape != (walker.action_dim,) or not np.all(np.isfinite(action)):
                raise FloatingPointError("Controller produced a non-finite or incorrectly shaped action")
            action = np.clip(action, low, high)
            action_seconds.append(time.perf_counter() - decision_started)
            simulation_started = time.perf_counter()
            observation, _, terminated, truncated, info = env.step(action)
            simulation_seconds.append(time.perf_counter() - simulation_started)
            actions.append(action.copy())
            states.append(walker.get_state())
            sensors.append(walker.data.sensordata.copy())
            costs.append(info["task_cost"])
            velocities.append(info["x_velocity"])
            if terminated or truncated:
                break
        state_array, action_array = np.asarray(states), np.asarray(actions)
        sensor_array = np.asarray(sensors)
        # Reuse the task's first-fall/terminal accounting on executed states.
        # The physical fall transition is charged once, then only the unexecuted
        # suffix receives the absorbing rate. There is no rollout here.
        executed_cost = float(walker.rollout_cost(state_array[None, 1:], action_array[None], sensor_array[None])[0])
        if not np.isclose(executed_cost, np.sum(costs), atol=1e-8, rtol=1e-10):
            raise RuntimeError("Executed trace scoring differs from live task costs")
        absorbing = walker._w_unhealthy * (config.steps - len(actions)) if terminated else 0.
        total_cost = executed_cost + absorbing
        # qpos = [x, torso_z, torso_angle, joints...]; remove translating x.
        positions = walker.state_qpos(state_array)[:, 1:]
        regularity = state_periodicity(positions, dt=walker._dt, config=config)
        jerk = float(np.mean(np.sum(np.diff(action_array, axis=0)**2, axis=1))) if len(actions) > 1 else None
        metrics = EpisodeMetrics(controller, seed, len(actions), not terminated and len(actions) == config.steps,
                                 executed_cost, float(absorbing), total_cost, -total_cost,
                                 float(np.mean(np.abs(walker._v_target - np.asarray(velocities)))), jerk,
                                 regularity.score, regularity.period_seconds, regularity.reason,
                                 regularity.channels_used, float(np.mean(action_seconds)), time.perf_counter() - started)
        return EpisodeTrace(metrics, state_array, action_array, sensor_array, np.asarray(costs),
                            np.asarray(velocities), np.asarray(action_seconds), np.asarray(simulation_seconds)), backends
    finally:
        if guided is not None:
            guided.close()
        env.close()


def _summary(traces: list[EpisodeTrace]) -> dict[str, float | int | None]:
    """Equal-weight episode means, retaining failed episodes and missing cycles."""
    result: dict[str, float | int | None] = {"episodes": len(traces)}
    for field in ("executed_cost", "absorbing_cost", "total_cost", "return_value", "velocity_mae",
                  "action_jerk", "periodicity_score", "period_seconds", "mean_action_seconds", "wall_time_seconds"):
        values = [getattr(trace.metrics, field) for trace in traces if getattr(trace.metrics, field) is not None]
        result["return" if field == "return_value" else field] = float(np.mean(values)) if values else None
    result["survival_rate"] = float(np.mean([trace.metrics.survived for trace in traces]))
    result["mean_steps"] = float(np.mean([trace.metrics.steps for trace in traces]))
    result["periodicity_valid_episodes"] = sum(trace.metrics.periodicity_score is not None for trace in traces)
    return result


def _paired_differences(by_controller: dict[str, list[EpisodeTrace]]) -> dict[str, Any]:
    """Descriptive policy-minus-planner changes on exactly matched reset pairs."""
    result: dict[str, Any] = {}
    candidate = by_controller["pi_standalone"]
    for reference_name in ("mppi_guided_online", "mppi_cold_online"):
        reference = by_controller[reference_name]
        if [trace.metrics.reset_seed for trace in candidate] != [trace.metrics.reset_seed for trace in reference]:
            raise ValueError("Ladder comparison requires exactly paired reset seeds")
        metrics: dict[str, Any] = {}
        for name in ("survived", "steps", "total_cost", "return_value", "velocity_mae", "action_jerk", "periodicity_score"):
            differences = [float(getattr(first.metrics, name)) - float(getattr(second.metrics, name))
                           for first, second in zip(candidate, reference)
                           if getattr(first.metrics, name) is not None and getattr(second.metrics, name) is not None]
            metric = {"survived": "survival_rate", "return_value": "return"}.get(name, name)
            metrics[metric] = {"mean": float(np.mean(differences)) if differences else None, "paired_episodes": len(differences)}
        result[f"pi_standalone_minus_{reference_name}"] = metrics
    return result


def _fingerprint(*, config: LadderConfig, planner_cfg: MPPIConfig, task_config: Path) -> tuple[str, dict[str, Any]]:
    """Identify policy-independent cold physics, scoring, randomness, and evaluation."""
    sources = ("src/gps/eval_ladder.py", "src/gps/mdgps.py", "src/gps/policy_tail.py",
               "src/mppi/mppi.py", "src/envs/base.py", "src/envs/mujoco_env.py", "src/envs/walker2d.py",
               "src/rl/walker.py", "src/utils/config.py", "src/utils/math.py", "assets/walker2d.xml")
    payload: dict[str, Any] = {
        "schema": _SCHEMA, "controller": "raw MPPI, shifted receding plan; no policy or policy tail",
        "evaluation": asdict(config), "cold_planner": asdict(planner_cfg),
        "task_sha256": hashlib.sha256(task_config.read_bytes()).hexdigest(),
        "sources": {relative: hashlib.sha256((_ROOT / relative).read_bytes()).hexdigest() for relative in sources},
        "runtime": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "mujoco")}},
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return digest, payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically replace an owned JSON artifact, disallowing NaN and infinity."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object used for history or a fingerprinted cold cache."""
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save_traces(path: Path, traces: list[EpisodeTrace]) -> None:
    """Save ragged numeric arrays without object arrays or pickle requirements."""
    lengths = [trace.metrics.steps for trace in traces]
    arrays: dict[str, np.ndarray] = {
        name: np.concatenate([getattr(trace, name) for trace in traces], axis=0).astype(np.float32)
        for name in ("states", "actions", "sensors", "costs", "velocities", "action_seconds", "simulation_seconds")}
    arrays.update(state_offsets=np.cumsum([0, *[length + 1 for length in lengths]]),
                  step_offsets=np.cumsum([0, *lengths]),
                  controllers=np.asarray([trace.metrics.controller for trace in traces]),
                  reset_seeds=np.asarray([trace.metrics.reset_seed for trace in traces], dtype=np.uint32))
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_traces(path: Path, episodes: list[dict[str, Any]]) -> list[EpisodeTrace]:
    """Load and validate the pickle-free cold trace cache against its JSON rows."""
    traces: list[EpisodeTrace] = []
    with np.load(path, allow_pickle=False) as data:
        step_offsets, state_offsets = data["step_offsets"], data["state_offsets"]
        if (len(step_offsets) != len(episodes) + 1 or len(state_offsets) != len(episodes) + 1
                or step_offsets[0] != 0 or state_offsets[0] != 0
                or len(data["states"]) != state_offsets[-1] or len(data["actions"]) != step_offsets[-1]):
            raise ValueError("Cold trace cache has invalid offsets")
        for index, row in enumerate(episodes):
            steps = row["steps"]
            if (step_offsets[index + 1] - step_offsets[index] != steps
                    or state_offsets[index + 1] - state_offsets[index] != steps + 1
                    or str(data["controllers"][index]) != "mppi_cold_online"
                    or int(data["reset_seeds"][index]) != row["reset_seed"]):
                raise ValueError("Cold trace cache disagrees with episode metadata")
            state_slice = slice(state_offsets[index], state_offsets[index + 1])
            step_slice = slice(step_offsets[index], step_offsets[index + 1])
            arrays = [np.asarray(data[name][state_slice if name == "states" else step_slice], dtype=np.float64)
                      for name in ("states", "actions", "sensors", "costs", "velocities", "action_seconds", "simulation_seconds")]
            if not all(np.all(np.isfinite(array)) for array in arrays):
                raise ValueError("Cold trace cache contains non-finite data")
            traces.append(EpisodeTrace(EpisodeMetrics(**row), *arrays))
    return traces


def _figure(path: Path, traces: list[EpisodeTrace], history: list[dict[str, Any]]) -> None:
    """Plot a preselected reset's phase portrait and the four-controller history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"pi_standalone": "#2465a7", "pi_noisy": "#72a8d1", "mppi_guided_online": "#d96b25", "mppi_cold_online": "#777777"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for controller in ("pi_standalone", "mppi_guided_online"):
        trace = next(trace for trace in traces if trace.metrics.controller == controller)
        # Walker full state is [time, qpos(9), qvel(9)]; primary thigh is qpos[3].
        axes[0].plot(trace.states[:, 4], trace.states[:, 13], color=colors[controller],
                     lw=1.2, alpha=.85, label=f"{controller} ({trace.metrics.steps} steps)")
        axes[0].scatter(trace.states[0, 4], trace.states[0, 13], color=colors[controller], s=20)
    axes[0].set(title="Primary thigh phase portrait · first fixed reset", xlabel="Thigh angle (rad)", ylabel="Thigh angular velocity (rad/s)")
    for controller in CONTROLLERS:
        axes[1].plot([row["iteration"] for row in history], [row["summary"][controller]["return"] for row in history],
                     color=colors[controller], marker="o", markersize=3, lw=1.5, label=controller,
                     ls="--" if controller == "mppi_cold_online" else "-")
    axes[1].set(title="Four-controller ladder · failures retained", xlabel="Training iteration (zero-based)", ylabel="Return = − fixed-horizon task cost")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("Executed traces only · cold MPPI cached from iteration 0")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_ladder_and_regularity(*, act_batch: PolicyBatch, out_dir: Path, iteration: int,
                                  planner_cfg: MPPIConfig, tail_steps: int = 100,
                                  config: LadderConfig | None = None,
                                  task_config: Path = DEFAULT_TASK,
                                  guided_factory: GuidedControllerFactory | None = None) -> dict[str, Any]:
    """Evaluate a frozen policy snapshot and save compact ladder/regularity artifacts.

    Call at iteration zero, then every fifth completed iteration and the final
    iteration. The caller owns scheduling. ``planner_cfg`` must stay fixed for
    the experiment, including cold MPPI's standard initial refinement passes;
    guided MPPI always uses one policy-centred pass at each decision. Evaluation
    timings are returned separately and must not be counted as training time.
    """
    if type(iteration) is not int or iteration < 0 or type(tail_steps) is not int or tail_steps < 0:
        raise ValueError("Require a nonnegative integer iteration and tail_steps")
    config = LadderConfig() if config is None else config
    out_dir, task_config = Path(out_dir), Path(task_config)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stem = f"iter_{iteration:03d}"
    report_path = logs / f"{stem}_ladder.json"
    if report_path.exists():
        raise FileExistsError(f"Ladder iteration already evaluated: {report_path}")
    fingerprint, fingerprint_payload = _fingerprint(config=config, planner_cfg=planner_cfg, task_config=task_config)
    history_path = logs / "ladder_history.json"
    history = _read_json(history_path) if history_path.exists() else {"schema": _SCHEMA, "fingerprint": fingerprint, "records": []}
    if history.get("fingerprint") != fingerprint or history.get("schema") != _SCHEMA:
        raise ValueError("Ladder history fingerprint differs; start a new experiment instead of mixing protocols")
    if history["records"] and iteration <= history["records"][-1]["iteration"]:
        raise ValueError("Ladder iterations must be strictly increasing")
    cache_json, cache_npz = logs / "mppi_cold_cache.json", logs / "mppi_cold_cache.npz"
    if cache_json.exists() != cache_npz.exists():
        raise ValueError("Cold cache is incomplete; refusing a partial cache")
    started = time.perf_counter()
    cache_hit = cache_json.exists()
    cold_backends: list[str] = []
    if cache_hit:
        cached = _read_json(cache_json)
        if cached.get("fingerprint") != fingerprint or cached.get("schema") != _SCHEMA:
            raise ValueError("Cold cache fingerprint differs; refusing stale baseline reuse")
        if hashlib.sha256(cache_npz.read_bytes()).hexdigest() != cached["trace_sha256"]:
            raise ValueError("Cold cache trace checksum differs")
        cold = _load_traces(cache_npz, cached["episodes"])
        cold_seconds = 0.
        cold_backends = cached["backends"]
    else:
        if iteration != 0:
            raise ValueError("Initialize the policy-independent cold MPPI baseline at iteration 0")
        cold_started = time.perf_counter()
        cold = []
        for episode in range(config.episodes):
            trace, backends = _episode(controller="mppi_cold_online", act_batch=act_batch, config=config,
                                       planner_cfg=planner_cfg, tail_steps=0, task_config=task_config,
                                       episode=episode, guided_factory=None)
            cold.append(trace)
            cold_backends.extend(backends)
        cold_seconds = time.perf_counter() - cold_started
        _save_traces(cache_npz, cold)
        cached = {"schema": _SCHEMA, "fingerprint": fingerprint, "fingerprint_payload": fingerprint_payload,
                  "iteration": 0, "elapsed_seconds": cold_seconds, "summary": _summary(cold),
                  "backends": sorted(set(cold_backends)), "episodes": [asdict(trace.metrics) for trace in cold],
                  "trace_sha256": hashlib.sha256(cache_npz.read_bytes()).hexdigest()}
        _write_json(cache_json, cached)
    by_controller: dict[str, list[EpisodeTrace]] = {"mppi_cold_online": cold}
    timings: dict[str, float] = {"mppi_cold_online": cold_seconds}
    backends_by_controller: dict[str, list[str]] = {"mppi_cold_online": sorted(set(cold_backends))}
    for controller in CONTROLLERS[:-1]:
        arm_started = time.perf_counter()
        traces, all_backends = [], []
        for episode in range(config.episodes):
            trace, backends = _episode(controller=controller, act_batch=act_batch, config=config,
                                       planner_cfg=planner_cfg, tail_steps=tail_steps, task_config=task_config,
                                       episode=episode, guided_factory=guided_factory)
            traces.append(trace)
            all_backends.extend(backends)
        by_controller[controller] = traces
        timings[controller] = time.perf_counter() - arm_started
        backends_by_controller[controller] = sorted(set(all_backends))
    traces = [trace for controller in CONTROLLERS for trace in by_controller[controller]]
    summaries = {controller: _summary(by_controller[controller]) for controller in CONTROLLERS}
    record = {"iteration": iteration, "summary": summaries, "cold_cache_hit": cache_hit, "controller_seconds": timings}
    records = [*history["records"], record]
    trace_path, figure_path = logs / f"{stem}_limit_cycle.npz", logs / f"{stem}_limit_cycle.png"
    _save_traces(trace_path, traces)
    _figure(figure_path, traces, records)
    report: dict[str, Any] = {
        "schema": _SCHEMA, "iteration": iteration, "config": asdict(config), "planner_config": asdict(planner_cfg),
        "controller_configs": {"mppi_cold_online": {"planner": asdict(planner_cfg), "policy_tail_steps": 0,
                                                        "nominal": "shifted receding-horizon raw MPPI plan"},
                               "mppi_guided_online": {"planner": asdict(replace(planner_cfg, initial_iterations=1,
                                                                                    lam_target_neff_fraction=config.guided_neff_fraction)),
                                                      "policy_tail_steps": tail_steps,
                                                      "nominal": "fresh policy rollout at each decision; no plan memory"}},
        "tail_steps": tail_steps, "fingerprint": fingerprint, "cold_cache_hit": cache_hit,
        "cold_original_elapsed_seconds": cached["elapsed_seconds"], "controller_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started, "summary": summaries,
        "metrics": {f"ladder/{controller}/{key}": value for controller, summary in summaries.items() for key, value in summary.items()},
        "episodes": [asdict(trace.metrics) for trace in traces], "controller_backends": backends_by_controller,
        "paired_differences": _paired_differences(by_controller),
        "artifact_paths": {"json": str(report_path), "traces": str(trace_path), "figure": str(figure_path), "history": str(history_path)},
        "cost_semantics": "Executed task cost through first fall, including one terminal cost, plus unhealthy_cost_weight for each remaining control. Return is its negative; no discounted infinite failure tail.",
        "regularity_semantics": "Linear-detrended torso/joint qpos excluding world x; positive local autocorrelation peak after the physical minimum period and at least two cycles. Missing evidence remains null; means use only identifiable episodes, whose count is reported.",
        "trace_layout": "Concatenated states include initial state plus T post-action states; actions/sensors/costs/velocities/timings contain T transitions. Use separate state_offsets and step_offsets. Full Walker states: time, qpos(9), qvel(9). Float32 storage; JSON metrics computed in float64.",
        "comparison_scope": "Matched evaluation resets for this frozen policy; descriptive differences do not prove superiority across independently trained seeds.",
        "compute_comparison": "Both online controllers use the supplied K/H/noise model. Guided MPPI uses one ESS-controlled update plus a learned-policy tail; cold MPPI uses raw task scores, its configured initial refinements, and no learned tail. Their rollout budgets and trust rules differ; measured controller/action times are reported.",
    }
    _write_json(report_path, report)
    _write_json(history_path, {"schema": _SCHEMA, "fingerprint": fingerprint, "records": records})
    return report
