"""Native full-sequence solves and standalone execution with complete evidence.

Imports of MuJoCo, the inherited optimizer and checkpoint decoding are lazy.
Workers never fit a policy. Every native forecast/control is reserved before
execution; all source contexts and missing post-fall tails remain explicit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import sha, write
from .runtime import restore, snapshot
from .trajectory_pi_accounting import Work
from .trajectory_pi_student import NumpyPolicy, observations


def same_bytes(first: Any, second: Any) -> bool:
    first, second = np.asarray(first), np.asarray(second)
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes() == second.tobytes()
    )


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def load_context(path: Path, step: int = 0) -> dict:
    """Read a true pre-action boundary from the frozen NPZ or a native trace."""
    import mujoco

    if type(step) is not int or step < 0:
        raise ValueError("Nonnegative context index required")
    if path.suffix == ".h5":
        import h5py

        with h5py.File(path, "r") as raw:
            value = {
                name: raw[name][step].copy() for name in ("states", "warmstarts", "integrations")
            }
    else:
        with np.load(path, allow_pickle=False) as raw:
            value = {
                name: raw[name][step].copy() for name in ("states", "warmstarts", "integrations")
            }
    return dict(
        state=value["states"],
        warmstart=value["warmstarts"],
        integration=value["integrations"],
        spec=np.array(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
        dimensions=np.array([9, 9, 6]),
    )


def load_trace(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as raw:
        return {key: raw[key][:] for key in raw}


def trace_outcome(states: np.ndarray, costs: np.ndarray, horizon: int, fell: bool) -> dict:
    if (
        states.shape != (len(costs) + 1, 19)
        or len(costs) > horizon
        or not np.isfinite(states).all()
        or not np.isfinite(costs).all()
    ):
        raise ValueError("Finite N+1 Walker states must match actual controls")
    velocities = states[1:, 10]
    survived = len(costs) == horizon and not fell
    mae = float(np.mean(np.abs(velocities - 1.5))) if len(velocities) else None
    tail = float(5 * (horizon - len(costs))) if fell else 0.0
    return dict(
        controls=len(costs),
        requested_controls=horizon,
        fell=fell,
        full_survival=survived,
        walking=bool(horizon == 1000 and survived and mae is not None and mae <= 0.3),
        velocity_mae=mae,
        mean_velocity=float(np.mean(velocities)) if len(velocities) else None,
        raw_native_cost=float(np.sum(costs)),
        absorbing_tail_cost=tail,
        complete_cost=float(np.sum(costs)) + tail,
    )


class Trace:
    """Append native boundaries durably so a paid failed prefix is retained."""

    def __init__(self, path: Path):
        import h5py

        self.file = h5py.File(path, "x")

    def append(self, **values: Any) -> None:
        for name, value in values.items():
            value = np.asarray(value)
            if name not in self.file:
                self.file.create_dataset(
                    name,
                    shape=(0, *value.shape),
                    maxshape=(None, *value.shape),
                    dtype=value.dtype,
                    chunks=(1, *value.shape),
                )
            dataset = self.file[name]
            dataset.resize(len(dataset) + 1, axis=0)
            dataset[-1] = value

    def boundary(self, env: Any) -> None:
        value = snapshot(env)
        self.append(
            states=value["state"],
            warmstarts=value["warmstart"],
            integrations=value["integration"],
            observations=env._get_obs(),
        )

    def flush(self) -> None:
        self.file.flush()
        os.fsync(self.file.id.get_vfd_handle())

    def close(self) -> None:
        self.flush()
        self.file.close()


def _env(job: dict, work: Work) -> Any:
    from src.envs.walker2d import Walker2d

    env = work.call(
        "environment",
        dict(environment_constructions=1),
        lambda: Walker2d(**job["task"], nthread=job["physics_threads"], use_warp=False),
    )
    if (env.model.nq, env.model.nv, env.model.nu) != (9, 9, 6):
        raise ValueError("Frozen Walker dimensions differ")
    batch, step = env.batch_rollout, env.step
    env._pi_in_solve = False

    def forecast(initial, actions, **kwargs):
        controls = int(np.prod(np.shape(actions)[:2]))
        charges = dict(model_controls=controls, physics_substeps=controls * 4, model_queries=1)
        if env._pi_in_solve:
            charges["pi_passes"] = 1
        return work.call("native_forecast", charges, lambda: batch(initial, actions, **kwargs))

    def execute(action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Bounded float64 physical torques required")
        return work.call(
            "native_control", dict(native_controls=1, physics_substeps=4), lambda: step(action)
        )

    env.batch_rollout, env.step = forecast, execute
    return env


def _restore(env: Any, value: dict, work: Work) -> None:
    actual = work.call(
        "restore", dict(restores=1, external_mj_forward_calls=1), lambda: restore(env, value)
    )
    if any(
        not same_bytes(actual[key], value[key]) for key in ("state", "warmstart", "integration")
    ):
        raise ValueError("Complete simulator context did not restore byte-exactly")
    if np.any(env.data.qfrc_applied) or np.any(env.data.xfrc_applied) or env.model.na:
        raise ValueError("Batch forecast supports only the frozen zero-external-force Walker")


def execute_trace(
    env: Any,
    work: Work,
    path: Path,
    horizon: int,
    *,
    actor: NumpyPolicy | None = None,
    actions: np.ndarray | None = None,
) -> dict:
    if (actor is None) == (actions is None):
        raise ValueError("Choose exactly one physical execution controller")
    trace = Trace(path)
    states, costs, fell = [env.get_state()], [], False
    try:
        trace.boundary(env)
        for index in range(horizon):
            if actor is not None:
                action = actor.predict(env._get_obs(), meter=work).astype(np.float64)
            else:
                action = np.asarray(actions[index], dtype=np.float64)
            _, cost, done, info = env.step(action)
            trace.append(actions=action, costs=cost, native_fall_cost=info["fall_cost"])
            trace.boundary(env)
            trace.flush()
            states.append(env.get_state())
            costs.append(cost)
            if done:
                fell = True
                break
    finally:
        trace.close()
    return trace_outcome(np.asarray(states), np.asarray(costs), horizon, fell)


class SolveRecorder:
    """All nominal paths/statistics; predeclared full population banks only."""

    def __init__(
        self,
        directory: Path,
        actor: NumpyPolicy,
        env: Any,
        work: Work,
        context: dict,
        full_banks: list[int],
    ):
        import h5py

        self.directory, self.actor, self.env, self.work = directory, actor, env, work
        self.context, self.full_banks = context, full_banks
        self.file = h5py.File(directory / "iterations.h5", "x")
        self.before = None

    def __call__(self, event: dict) -> None:
        index = event["iteration"]
        if event["phase"] == "before":
            self.before = event
            return
        mask = event["active_mask"]
        if mask is None:
            from src.utils.rollout_data import pre_fall_action_mask

            mask = pre_fall_action_mask(
                self.env._is_healthy(event["states"][..., 2], event["states"][..., 3])
            )
        group = self.file.create_group(f"iteration_{index:04d}")
        for name, value in dict(
            nominal_before=event["current_actions"],
            nominal_after=event["post_actions"],
            nominal_post_states=event["states"][0],
            native_costs=event["C"],
            total_objectives=event["J"],
            weights=event["weights"],
            active_prefix_packed=np.packbits(mask.ravel()),
        ).items():
            group.create_dataset(
                name, data=value, compression="gzip", compression_opts=1, shuffle=True
            )
        healthy = self.env._is_healthy(event["states"][..., 2], event["states"][..., 3])
        complete = healthy.all(axis=1) & np.isfinite(event["states"]).all(axis=(1, 2))
        walking = complete & (np.mean(np.abs(event["states"][..., 10] - 1.5), axis=1) <= 0.3)
        if hasattr(self.env, "running_cost"):
            running = self.env.running_cost(event["states"], event["actions"])
            prefix = np.where(mask, running, 0.0).sum(axis=1)
            terminal = (~healthy.all(axis=1)).astype(float) * self.env._w_term
            absorbing = (~mask).sum(axis=1) * self.env._w_unhealthy
            for name, value in dict(
                prefix_running_cost=prefix,
                fall_terminal_cost=terminal,
                absorbing_tail_cost=absorbing,
            ).items():
                group.create_dataset(name, data=value)
            self.work.counts["derived_cost_component_rows"] = self.work.counts.get(
                "derived_cost_component_rows", 0
            ) + int(mask.size)
            self.work.record("derived_cost_components", rows=int(mask.size))
        group.create_dataset("prefix_lengths", data=mask.sum(axis=1))
        group.create_dataset("complete_candidates", data=complete)
        group.create_dataset("walking_candidates", data=walking)
        group.attrs["best_walking_population_J"] = (
            float(np.min(event["J"][walking])) if walking.any() else np.nan
        )
        if event["D"] is not None:
            group.create_dataset("reconstruction", data=event["D"])
            nominal_d = float(event["D"][0])
        elif np.isfinite(event["C"][0]):
            pre = np.concatenate((self.context["state"][None], event["states"][0, :-1]))
            nominal_mask = mask[0]
            prediction = self.actor.predict(
                observations(pre[nominal_mask]), meter=self.work
            ).astype(np.float64)
            nominal_d = float(
                0.5 * np.sum((event["current_actions"][nominal_mask] - prediction) ** 2)
            )
        else:
            nominal_d = np.nan
        group.attrs["nominal_D"] = nominal_d
        group.attrs["nominal_D_available"] = bool(np.isfinite(nominal_d))
        group.attrs["population_D_measured"] = event["D"] is not None
        group.attrs["lambda"] = event["lambda"]
        group.attrs["n_eff"] = event["n_eff"]
        group.attrs["scale"] = self.before["scale"]
        group.attrs["rng_before"] = json.dumps(self.before["planner_rng"], sort_keys=True)
        group.attrs["initial_plan_before"] = bool(self.before["initial_plan"])
        if "planner_after" in event:
            group.attrs["rng_after"] = json.dumps(
                event["planner_after"]["rng_state"], sort_keys=True
            )
        group.attrs["maximum_weight"] = float(np.max(event["weights"]))
        group.attrs["update_rms"] = float(
            np.sqrt(np.mean((event["post_actions"] - event["current_actions"]) ** 2))
        )
        if index in self.full_banks:
            self.work.check(reserve_output=56 * 1024**2)
            with (self.directory / f"bank_{index:04d}.npz").open("xb") as stream:
                np.savez_compressed(stream, states=event["states"], actions=event["actions"])
        self.file.flush()
        os.fsync(self.file.id.get_vfd_handle())
        current = snapshot(self.env)
        if any(
            not same_bytes(current[key], self.context[key])
            for key in ("state", "warmstart", "integration")
        ):
            raise ValueError("Whole-sequence refinement mutated its fixed live context")

    def close(self) -> None:
        self.file.close()


def run_job(job: dict) -> dict:
    """Create-only native worker; all inputs are already bound by the launcher."""
    directory = Path(job["out"])
    directory.mkdir(parents=True, exist_ok=False)
    work = Work(directory, job["limits"])
    write(directory / "job.json", job)
    env, recorder = None, None
    try:
        env = _env(job, work)
        actor_path = Path(job["checkpoint"])
        if binding(actor_path) != job["checkpoint_binding"]:
            raise ValueError("Incoming policy checkpoint changed")
        actor = work.call(
            "numpy_checkpoint",
            dict(checkpoint_loads=1, numpy_constructions=1),
            lambda: NumpyPolicy.from_checkpoint(actor_path),
        )
        if job["kind"] == "actor":
            previous = np.random.get_state()
            try:
                np.random.seed(job["reset_seed"])
                work.call("paired_reset", dict(resets=1, external_mj_forward_calls=2), env.reset)
            finally:
                np.random.set_state(previous)
            context = snapshot(env)
        else:
            source = Path(job["context_file"])
            if binding(source) != job["context_binding"]:
                raise ValueError("Incoming complete context changed")
            context = load_context(source, job.get("context_step", 0))
            _restore(env, context, work)
        result: dict[str, Any] = dict(kind=job["kind"], policy_hash=actor.policy_hash)
        if job["kind"] == "solve":
            from src.utils.config import MPPIConfig

            from .trajectory_pi_optimizer import FrozenReconstructionCost, optimize_full

            plan_path = Path(job["initial_plan_file"])
            if binding(plan_path) != job["initial_plan_binding"]:
                raise ValueError("Incoming complete nominal changed")
            with np.load(plan_path, allow_pickle=False) as source:
                initial = source["actions"].astype(np.float64, copy=True)
            if initial.shape != (job["horizon"], 6):
                raise ValueError("Full-sequence initial plan shape differs")

            class MeteredActor:
                def predict(self, obs):
                    return actor.predict(obs, meter=work)

            cost = FrozenReconstructionCost(env, MeteredActor(), context["state"], job["alpha"])
            recorder = SolveRecorder(directory, actor, env, work, context, job["full_banks"])
            env._pi_in_solve = True
            solved = optimize_full(
                env,
                MPPIConfig(**job["planner"]),
                context["state"],
                context["warmstart"],
                initial,
                job["planner_seed"],
                job["iterations"],
                prior_cost=cost if job["alpha"] else None,
                observer=recorder,
            )
            env._pi_in_solve = False
            recorder.close()
            recorder = None
            if solved["status"] != "COMPLETED" or solved["best_iteration"] is None:
                raise ValueError("Full PI produced no finite scored nominal")
            with (directory / "solution.npz").open("xb") as stream:
                np.savez_compressed(
                    stream,
                    actions=solved["best_actions"],
                    batch_post_states=solved["best_states"],
                    final_unselected_actions=solved["final_actions"],
                )
            last_states, last_costs, _ = env.batch_rollout(
                context["state"],
                solved["final_actions"][None],
                initial_warmstart=context["warmstart"],
            )
            last_matching = (
                cost(last_states, solved["final_actions"][None]) if job["alpha"] else np.zeros(1)
            )
            with (directory / "last_mean.npz").open("xb") as stream:
                np.savez_compressed(stream, states=last_states[0], actions=solved["final_actions"])
            _restore(env, context, work)
            outcome = execute_trace(
                env, work, directory / "trace.h5", job["horizon"], actions=solved["best_actions"]
            )
            actual = load_trace(directory / "trace.h5")
            if not np.allclose(
                actual["states"][1:],
                solved["best_states"][: outcome["controls"]],
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError("Selected batch/native physical trajectory mismatch")
            prediction = actor.predict(actual["observations"][:-1], meter=work).astype(np.float64)
            reconstruction = float(0.5 * np.sum((actual["actions"] - prediction) ** 2))
            if not np.isclose(outcome["complete_cost"], solved["best_cost"], rtol=0, atol=1e-9):
                raise ValueError("Selected batch/native complete cost mismatch")
            if not np.isclose(
                outcome["complete_cost"] + job["alpha"] * reconstruction,
                solved["best_J"],
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError("Selected batch/native total objective mismatch")
            with (directory / "selected_predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, predictions=prediction)
            result.update(
                outcome=outcome,
                selected_D=reconstruction,
                selected_J=outcome["complete_cost"] + job["alpha"] * reconstruction,
                selected_iteration=solved["best_iteration"],
                selected_task_batch=solved["best_cost"],
                selected_total_batch=solved["best_J"],
                final_mean_task=float(last_costs[0]) if np.isfinite(last_costs[0]) else None,
                final_mean_total=float(last_costs[0] + last_matching[0])
                if np.isfinite(last_costs[0] + last_matching[0])
                else None,
                final_mean_score_available=bool(np.isfinite(last_costs[0] + last_matching[0])),
                optimizer_counts=solved["counts"],
                curve=solved["curve"],
            )
        elif job["kind"] == "actor":
            result["outcome"] = execute_trace(
                env, work, directory / "trace.h5", job["horizon"], actor=actor
            )
        else:
            raise ValueError("Unknown native worker kind")
        result["trace"] = binding(directory / "trace.h5")
        result["policy_checkpoint"] = job["checkpoint_binding"]
        write(directory / "result.json", result)
        receipt = work.close("COMPLETED", result_binding=binding(directory / "result.json"))
        return dict(path=str(directory), result=result, receipt=receipt)
    except BaseException as error:
        work.close(
            "FAILED", error=repr(error), partial_native_trace=(directory / "trace.h5").exists()
        )
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()
