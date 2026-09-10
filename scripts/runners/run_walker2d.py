import json
import hashlib
import inspect
import platform
import time
from importlib.metadata import version
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

# Allow both `python scripts/runners/run_walker2d.py` and module execution.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI 
from src.utils.config import MPPIConfig



def _foot_contacts(model, data, floor_id, right_foot_id, left_foot_id):
    right_contact = False
    left_contact = False
    normal_forces = np.zeros(2)
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        pair = {contact.geom1, contact.geom2}
        right_contact |= pair == {floor_id, right_foot_id}
        left_contact |= pair == {floor_id, left_foot_id}
        for foot, geom_id in enumerate((right_foot_id, left_foot_id)):
            if pair == {floor_id, geom_id}:
                mujoco.mj_contactForce(model, data, index, force)
                normal_forces[foot] += max(0.0, force[0])
    return (right_contact, left_contact), normal_forces


def _gait_metrics(contact_samples):
    contacts = np.asarray(contact_samples, dtype=bool)
    right, left = contacts[:, 0], contacts[:, 1]
    onsets = contacts[1:] & ~contacts[:-1]
    single_onset_labels = [
        int(np.argmax(row)) for row in onsets if np.sum(row) == 1
    ]
    alternation = (
        float(np.mean(np.diff(single_onset_labels) != 0))
        if len(single_onset_labels) > 1 else 0.0
    )
    return {
        "right_foot_onsets": int(np.sum(onsets[:, 0])),
        "left_foot_onsets": int(np.sum(onsets[:, 1])),
        "single_support_fraction": float(np.mean(right ^ left)),
        "double_support_fraction": float(np.mean(right & left)),
        "flight_fraction": float(np.mean(~right & ~left)),
        "onset_alternation": alternation,
    }



def _save_report(cfg, task, provenance, results, summary, traces, elapsed, *,
                 traces_output, metrics_output, steps, episodes, seed, planner_seed):
    if traces_output:
        path = traces_output
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: np.concatenate([episode[key] for episode in traces])
            for key in ("states", "warmstarts", "actions", "contacts", "contact_normal_forces", "costs")
        }
        lengths = [len(episode["actions"]) for episode in traces]
        arrays["episode_step_offsets"] = np.cumsum([0, *lengths])
        arrays["episode_state_offsets"] = np.cumsum([0, *[n + 1 for n in lengths]])
        arrays["reset_seeds"] = np.array([r["reset_seed"] for r in results])
        arrays["planner_seeds"] = np.array([r["planner_seed"] for r in results])
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    if metrics_output:
        report = {
            "task": {**task, "max_steps": steps,
                     "velocity_cost_weight": task["vel_cost_weight"]},
            "mppi_config": asdict(cfg), "episodes": results, "summary": summary,
            "run": {"requested_episodes": episodes, "complete": len(results) == episodes,
                    "elapsed_seconds": elapsed, "reset_seed_base": seed,
                    "planner_seed_base": planner_seed},
            "cost_semantics": {
                "total_cost": "Executed step costs through first fall, including fall cost once. Failed episodes are shorter, so total costs alone cannot rank controllers.",
                "planning_cost": "Running cost through first predicted fall, one terminal fall cost, and an absorbing failure cost for each remaining horizon step.",
                "absorbing_failure_cost_per_remaining_step": task["unhealthy_cost_weight"],
            },
            "provenance": provenance,
            "trace_layout": {
                "states": "Flat mjSTATE_FULLPHYSICS; episode_state_offsets delimit initial + post-action states.",
                "warmstarts": "Solver state aligned with states.",
                "actions": "episode_step_offsets delimit actions, costs and post-action contacts.",
                "contacts": "Right, left; raw floor-foot contact presence.",
                "contact_normal_forces": "Right, left; summed nonnegative floor-foot normal force in newtons.",
            },
        }
        path = metrics_output
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

def _argument_error(message: str) -> None:
    print(f"run_walker2d: error: {message}", file=sys.stderr)
    raise SystemExit(2)


def main(
    config: Path = _REPO_ROOT / "configs/walker2d_best.json",
    task_config: Path = _REPO_ROOT / "configs/walker2d_task.json",
    planner_seed: int = 10_000,
    traces_output: Path | None = None,
    episodes: int = 5,
    steps: int = 500,
    seed: int = 0,
    render: bool = False,
    output: Path = Path("runs/walker2d_teacher/walker2d.mp4"),
    metrics_output: Path | None = None,
    log_every: int = 20,
) -> None:
    """Run a seeded Walker2d MPPI baseline.

    Args:
        planner_seed: First planner seed, independent of reset seed.
        log_every: Control steps between logs; 0 disables step logs.
    """
    if episodes <= 0 or steps <= 0:
        _argument_error("--episodes and --steps must be positive.")
    if log_every < 0:
        _argument_error("--log-every must be nonnegative (0 disables step logs).")
    if any(value < 0 or value + episodes > 2**32 for value in (seed, planner_seed)):
        _argument_error("Reset and planner seeds must remain in [0, 2**32).")
    outputs = [p.resolve() for p in (metrics_output, traces_output,
               output if render else None) if p is not None]
    if len(set(outputs)) != len(outputs) or set(outputs) & {config.resolve(), task_config.resolve()}:
        _argument_error("Output paths must be distinct and must not overwrite configs.")
    task = {name: p.default for name, p in inspect.signature(Walker2d).parameters.items()
            if p.default is not inspect.Parameter.empty}
    task.update(json.loads(task_config.read_text()))
    cfg = MPPIConfig(**json.loads(config.read_text()))
    source_paths = ("scripts/runners/run_walker2d.py", "src/envs/walker2d.py",
                    "src/envs/mujoco_env.py", "src/mppi/mppi.py", "src/utils/config.py",
                    "assets/walker2d.xml")
    provenance = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
        "config_path": str(config.resolve()),
        "task_config_path": str(task_config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "task_config_sha256": hashlib.sha256(task_config.read_bytes()).hexdigest(),
        "source_sha256": {p: hashlib.sha256((_REPO_ROOT / p).read_bytes()).hexdigest()
                          for p in source_paths},
    }
    env = Walker2d(**task)
    renderer = None
    try:
        controller = MPPI(env, cfg)

        renderer = (
            mujoco.Renderer(env.model, height=480, width=640)
            if render else None
        )
        frames = []
        traces = []
        started = time.perf_counter()
        provenance.update(control_dt=env._dt, physics_dt=float(env.model.opt.timestep),
                          frame_skip=env._frame_skip, nthread=env._nthread,
                          backend="warp" if env._use_warp else "mujoco_cpu",
                          video_fps=1.0 / (env._dt * 4), video_frame_stride=4)
        episode_results = []
        floor_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        right_foot_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_geom"
        )
        left_foot_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_left_geom"
        )

        print(f"task={task} config={cfg}", flush=True)

        for episode in range(episodes):
            reset_seed = seed + episode
            episode_planner_seed = planner_seed + episode
            np.random.seed(reset_seed)
            env.reset()
            controller.reset(seed=episode_planner_seed)
            state = env.get_state()
            start_x = float(env.data.qpos[0])
            velocities = []
            actions = []
            contact_samples = []
            effective_sample_sizes = []
            states = [state.copy()] if traces_output else []
            warmstarts = [env.get_warmstart()] if traces_output else []
            normal_forces, costs, plan_times = [], [], []
            total_fall_cost = 0.0
            total_cost = 0.0
            done = False

            for t in range(steps):
                plan_started = time.perf_counter()
                action, plan_info = controller.plan_step(
                    state,
                    initial_warmstart=env.get_warmstart(),
                )
                plan_times.append(time.perf_counter() - plan_started)
                _, cost, done, step_info = env.step(action)
                state = env.get_state()
                velocities.append(step_info["x_velocity"])
                actions.append(action.copy())
                contacts, forces = _foot_contacts(env.model, env.data, floor_id, right_foot_id, left_foot_id)
                contact_samples.append(contacts)
                if traces_output:
                    states.append(state.copy())
                    warmstarts.append(env.get_warmstart())
                    normal_forces.append(forces)
                    costs.append(cost)
                total_fall_cost += step_info["fall_cost"]
                effective_sample_sizes.append(plan_info["n_eff"])
                total_cost += cost

                if renderer is not None and t % 4 == 0:
                    renderer.update_scene(env.data, camera="track")
                    frames.append(renderer.render().copy())

                if log_every > 0 and t % log_every == 0:
                    pos = env.data.qpos.copy()
                    vel = env.data.qvel.copy()
                    print(
                        f"episode={episode:2d}  step={t:4d}  "
                        f"cost_min={plan_info['cost_min']:.3f}  "
                        f"pos=({pos[0]:.3f}, {pos[1]:.3f})  "
                        f"vel=({vel[0]:.3f}, {vel[1]:.3f})"
                    )

                if done:
                    break

            velocities_array = np.asarray(velocities)
            actions_array = np.asarray(actions)
            action_delta_rms = (
                float(np.sqrt(np.mean(np.sum(np.diff(actions_array, axis=0) ** 2, axis=1))))
                if len(actions_array) > 1 else 0.0
            )
            gait = _gait_metrics(contact_samples)
            saturated = np.isclose(actions_array, controller.action_low) | np.isclose(
                actions_array, controller.action_high
            )
            result = {
                "reset_seed": reset_seed,
                "planner_seed": episode_planner_seed,
                "steps": len(velocities),
                "survived": not done,
                "distance": float(env.data.qpos[0] - start_x),
                "mean_vx": float(np.mean(velocities_array)),
                "velocity_mae": float(
                    np.mean(np.abs(env._v_target - velocities_array))
                ),
                "total_cost": float(total_cost),
                "fall_cost": float(total_fall_cost),
                "mean_running_cost": float((total_cost - total_fall_cost) / len(velocities)),
                "mean_plan_seconds": float(np.mean(plan_times)),
                "action_delta_rms": action_delta_rms,
                "action_saturation_fraction": float(np.mean(saturated)),
                "action_near_limit_fraction": float(np.mean(np.abs(actions_array) >= 0.95)),
                "mean_n_eff": float(np.mean(effective_sample_sizes)),
                **gait,
            }
            episode_results.append(result)
            if traces_output:
                traces.append({"states": np.asarray(states), "warmstarts": np.asarray(warmstarts),
                               "actions": actions_array, "contacts": np.asarray(contact_samples, dtype=bool),
                               "contact_normal_forces": np.asarray(normal_forces), "costs": np.asarray(costs)})
            print(
                f"episode={episode:2d}  reset_seed={reset_seed}  "
                f"planner_seed={episode_planner_seed}  steps={result['steps']}/{steps}  "
                f"survived={result['survived']}  distance={result['distance']:.3f}  "
                f"mean_vx={result['mean_vx']:.3f}  "
                f"velocity_mae={result['velocity_mae']:.3f}  "
                f"cost={result['total_cost']:.3f}  "
                f"action_delta_rms={result['action_delta_rms']:.3f}  "
                f"foot_onsets=({result['right_foot_onsets']},"
                f"{result['left_foot_onsets']})  "
                f"alternation={result['onset_alternation']:.3f}"
            )

            summary = {
                "episodes": len(episode_results),
                "survival_rate": float(np.mean([r["survived"] for r in episode_results])),
                "mean_steps": float(np.mean([r["steps"] for r in episode_results])),
                "mean_vx": float(np.mean([r["mean_vx"] for r in episode_results])),
                "velocity_mae": float(np.mean([r["velocity_mae"] for r in episode_results])),
                "action_delta_rms": float(
                    np.mean([r["action_delta_rms"] for r in episode_results])
                ),
                "action_saturation_fraction": float(
                    np.mean([r["action_saturation_fraction"] for r in episode_results])
                ),
                "mean_plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in episode_results])),
                "mean_n_eff": float(np.mean([r["mean_n_eff"] for r in episode_results])),
                "single_support_fraction": float(
                    np.mean([r["single_support_fraction"] for r in episode_results])
                ),
                "double_support_fraction": float(
                    np.mean([r["double_support_fraction"] for r in episode_results])
                ),
                "flight_fraction": float(
                    np.mean([r["flight_fraction"] for r in episode_results])
                ),
                "onset_alternation": float(
                    np.mean([r["onset_alternation"] for r in episode_results])
                ),
            }
            _save_report(cfg, task, provenance, episode_results, summary, traces,
                         time.perf_counter() - started, traces_output=traces_output,
                         metrics_output=metrics_output, steps=steps, episodes=episodes,
                         seed=seed, planner_seed=planner_seed)

        print(
            f"summary episodes={episodes}  "
            f"survival_rate={summary['survival_rate']:.3f}  "
            f"mean_steps={summary['mean_steps']:.1f}  "
            f"mean_vx={summary['mean_vx']:.3f}  "
            f"velocity_mae={summary['velocity_mae']:.3f}  "
            f"action_delta_rms={summary['action_delta_rms']:.3f}  "
            f"saturation={summary['action_saturation_fraction']:.3f}  "
            f"single_support={summary['single_support_fraction']:.3f}  "
            f"flight={summary['flight_fraction']:.3f}  "
            f"alternation={summary['onset_alternation']:.3f}"
        )

        if metrics_output:
            print(f"metrics={metrics_output.resolve()}")
        if traces_output:
            print(f"traces={traces_output.resolve()}")

        if renderer is not None:
            import mediapy
            output.parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(output, frames, fps=1.0 / (env._dt * 4))
            print(f"video={output.resolve()}")
    finally:
        if renderer is not None:
            renderer.close()
        env.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
        
