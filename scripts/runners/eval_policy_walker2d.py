"""Evaluate saved S2 policies on the frozen local task, without MPPI."""

import hashlib
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.runners.run_walker2d import _foot_contacts, _gait_metrics
from src.rl.sac import load_policy
from src.rl.walker import DEFAULT_TASK, WalkerRL


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def provenance() -> dict:
    paths = ("src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py", "src/rl/walker.py",
             "src/rl/sac.py", "scripts/runners/eval_policy_walker2d.py",
             "scripts/runners/run_walker2d.py", "assets/walker2d.xml")
    return {"python": platform.python_version(),
            "packages": {name: version(name) for name in ("mujoco", "numpy", "torch", "gymnasium")},
            "source_sha256": {p: sha256(_ROOT / p) for p in paths}}


def evaluate_policy(policy, *, task_config: Path = DEFAULT_TASK, gamma: float = .995,
                    episodes: int = 30, steps: int = 1000, seed: int = 50_000,
                    video_path: Path | None = None, traces_path: Path | None = None) -> dict:
    """Use frozen-runner metric definitions, retaining failures in all averages."""
    if episodes < 1 or steps < 1 or seed < 0 or seed + episodes > 2**32:
        raise ValueError("Require positive episodes/steps and valid consecutive reset seeds.")
    if traces_path is not None and traces_path.exists():
        raise FileExistsError(traces_path)
    env = WalkerRL(task_config, gamma, steps)
    renderer = None
    frames, results, traces = [], [], []
    started = time.perf_counter()
    try:
        walker = env.walker
        if video_path is not None:
            if video_path.exists():
                raise FileExistsError(video_path)
            renderer = mujoco.Renderer(walker.model, height=480, width=640)
        geom_ids = [mujoco.mj_name2id(walker.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                    for name in ("floor", "foot_geom", "foot_left_geom")]
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            start_x = float(walker.data.qpos[0])
            velocities, actions, contacts, action_seconds = [], [], [], []
            states = [walker.get_state()] if traces_path else []
            warmstarts = [walker.get_warmstart()] if traces_path else []
            costs, rewards = [], []
            total_cost = fall_cost = training_return = 0.0
            terminated = truncated = False
            for step in range(steps):
                begin = time.perf_counter()
                action = policy.act(obs, deterministic=True)
                action_seconds.append(time.perf_counter() - begin)
                obs, reward, terminated, truncated, info = env.step(action)
                velocities.append(info["x_velocity"])
                actions.append(action.copy())
                contact, _ = _foot_contacts(walker.model, walker.data, *geom_ids)
                contacts.append(contact)
                total_cost += info["task_cost"]
                fall_cost += info["fall_cost"]
                training_return += reward
                if traces_path:
                    states.append(walker.get_state())
                    warmstarts.append(walker.get_warmstart())
                    costs.append(info["task_cost"])
                    rewards.append(reward)
                if renderer is not None and episode == 0 and step % 4 == 0:
                    renderer.update_scene(walker.data, camera="track")
                    frames.append(renderer.render().copy())
                if terminated or truncated:
                    break
            action_array = np.asarray(actions)
            vx = np.asarray(velocities)
            low, high = walker.action_bounds
            results.append({
                "reset_seed": seed + episode, "steps": len(actions),
                "survived": not terminated and len(actions) == steps,
                "terminated": terminated, "truncated": truncated,
                "distance": float(walker.data.qpos[0] - start_x),
                "mean_vx": float(vx.mean()),
                "velocity_mae": float(np.mean(np.abs(walker._v_target - vx))),
                "total_cost": total_cost, "fall_cost": fall_cost,
                "mean_running_cost": (total_cost - fall_cost) / len(actions),
                "training_return": training_return,
                "mean_action_seconds": float(np.mean(action_seconds)),
                "action_delta_rms": float(np.sqrt(np.mean(np.sum(np.diff(action_array, axis=0)**2, axis=1)))) if len(actions) > 1 else 0.,
                "action_saturation_fraction": float(np.mean(np.isclose(action_array, low) | np.isclose(action_array, high))),
                "action_near_limit_fraction": float(np.mean(np.abs(action_array) >= .95)),
                **_gait_metrics(contacts),
            })
            if traces_path:
                traces.append({"states": np.asarray(states), "warmstarts": np.asarray(warmstarts),
                               "actions": action_array, "costs": np.asarray(costs),
                               "training_rewards": np.asarray(rewards), "contacts": np.asarray(contacts)})
        if video_path is not None and frames:
            import mediapy
            video_path.parent.mkdir(parents=True, exist_ok=True)
            mediapy.write_video(str(video_path), frames, fps=1 / (walker._dt * 4))
    finally:
        if renderer is not None:
            renderer.close()
        env.close()
    summary = {key: float(np.mean([r[key] for r in results])) for key in results[0]
               if key not in {"reset_seed", "terminated", "truncated"}}
    summary["episodes"] = len(results)
    summary["survival_rate"] = summary.pop("survived")
    summary["mean_steps"] = summary.pop("steps")
    if traces_path:
        arrays = {key: np.concatenate([trace[key] for trace in traces]) for key in traces[0]}
        lengths = [r["steps"] for r in results]
        arrays.update(episode_step_offsets=np.cumsum([0, *lengths]),
                      episode_state_offsets=np.cumsum([0, *[n + 1 for n in lengths]]),
                      reset_seeds=np.arange(seed, seed + episodes))
        traces_path.parent.mkdir(parents=True, exist_ok=True)
        with traces_path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
    return {"episodes": results, "summary": summary,
            "run": {"requested_episodes": episodes, "complete": True, "steps": steps,
                    "reset_seed_base": seed, "elapsed_seconds": time.perf_counter() - started},
            "task": env.task,
            "traces": None if traces_path is None else {"path": str(traces_path), "sha256": sha256(traces_path),
                "layout": "States/warmstarts contain initial plus post-action states per episode; actions/costs/rewards/contacts contain transitions. Use separate state and step offsets."},
            "cost_semantics": {"total_cost": "Executed costs through first fall; not comparable across unequal episode lengths alone.",
                               "velocity_mae": "Per-episode mean over executed steps, then equal-weight episode mean; failures retained.",
                               "training_return": "Undiscounted sum of adjusted rewards, including analytic failure tail; not a primary gait metric."}}


def classify(report: dict) -> str:
    """Development labels only, fixed for 30 episodes of 1000 steps."""
    if len(report["episodes"]) != 30 or report["run"]["steps"] != 1000:
        return "unclassified"
    completed = sum(r["survived"] for r in report["episodes"])
    if completed < 15:
        return "weak"
    if completed >= 29 and report["summary"]["velocity_mae"] <= .20:
        return "strong"
    return "intermediate"


def main(checkpoint: Path, output: Path, episodes: int = 30, steps: int = 1000,
         seed: int = 50_000, task_config: Path = DEFAULT_TASK,
         video: Path | None = None, traces_output: Path | None = None) -> None:
    """Evaluate a local S2 checkpoint; never overwrite an existing report/video."""
    traces_output = traces_output or output.with_suffix(".npz")
    protected = {checkpoint.resolve(), task_config.resolve()}
    outputs = [p.resolve() for p in (output, traces_output, video) if p is not None]
    if len(outputs) != len(set(outputs)) or protected.intersection(outputs):
        raise ValueError("Outputs must be distinct from checkpoint, task and each other.")
    for path in (output, traces_output, video):
        if path is not None and path.exists():
            raise FileExistsError(path)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["task_sha256"] != sha256(task_config):
        raise ValueError("Checkpoint and evaluation task hashes differ.")
    for relative in ("src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py", "assets/walker2d.xml"):
        expected = payload["source_sha256"].get(relative)
        if expected is not None and expected != sha256(_ROOT / relative):
            raise ValueError(f"Checkpoint and evaluation physics differ: {relative}")
    policy = load_policy(payload["policy"], device="cpu")
    report = evaluate_policy(policy, task_config=task_config, gamma=payload["gamma"],
                             episodes=episodes, steps=steps, seed=seed, video_path=video,
                             traces_path=traces_output)
    report.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                  training_step=payload["step"], task_sha256=sha256(task_config),
                  provenance=provenance(), development_class=classify(report))
    write_json(output, report)
    print(json.dumps({"output": str(output), "summary": report["summary"],
                      "development_class": report["development_class"]}, indent=2))


def classify_ladder(run_dir: Path, task_config: Path = DEFAULT_TASK) -> dict:
    """Evaluate the complete saved ladder on the predeclared development seeds.

    Called only after training by --classify-at-end. No checkpoint is called
    strong based on its training age, and absent categories remain absent.
    """
    directory = run_dir / "classification"
    directory.mkdir(exist_ok=False)
    records = [json.loads(line) for line in (run_dir / "checkpoints.jsonl").read_text().splitlines()]
    rows, selected = [], {}
    for record in records:
        checkpoint = run_dir / "checkpoints" / f"step_{record['step']:07d}.pt"
        if sha256(checkpoint) != record["sha256"]:
            raise ValueError(f"Checkpoint hash differs from training manifest: {checkpoint}")
        output = directory / f"step_{record['step']:07d}.json"
        main(checkpoint, output, task_config=task_config)
        report = json.loads(output.read_text())
        label = report["development_class"]
        rows.append({"step": record["step"], "checkpoint": str(checkpoint),
                     "sha256": record["sha256"], "development_class": label,
                     "survival_rate": report["summary"]["survival_rate"],
                     "velocity_mae": report["summary"]["velocity_mae"], "report": str(output)})
        selected.setdefault(label, rows[-1])
        summary = {"complete": len(rows) == len(records), "checkpoints": rows,
                   "selected_earliest": selected,
                   "missing_categories": sorted({"weak", "intermediate", "strong"} - selected.keys()),
                   "reset_seeds": [50000, 50029], "episodes_per_checkpoint": 30, "steps": 1000,
                   "purpose": "Development selection only; no S4 held-out data used."}
        write_json(directory / "ladder.json", summary)
    return summary


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
