"""Train the S2 local-Walker ladder; each invocation owns a new run directory."""

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.runners.eval_policy_walker2d import classify_ladder, evaluate_policy, provenance, sha256, write_json
from src.rl.sac import ReplayBuffer, SACAgent, SACConfig
from src.rl.walker import DEFAULT_TASK, WalkerRL


def collect_transition(env, replay, obs, action):
    """Save the *final* observation before the caller can reset after a timeout.

    This function is the collector used by training and by target-alignment
    tests. Truncation does not become a terminal mask in replay.
    """
    next_obs, reward, terminated, truncated, info = env.step(action)
    replay.add(obs, action, reward, terminated, next_obs)
    return next_obs, reward, terminated, truncated, info


def append_record(path: Path, record: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def main(
    run_dir: Path,
    total_steps: int = 1_000_000,
    seed: int = 0,
    learning_starts: int = 5_000,
    save_every: int = 25_000,
    eval_every: int = 25_000,
    eval_episodes: int = 5,
    batch_size: int = 256,
    hidden_dim: int = 256,
    buffer_size: int = 1_000_000,
    max_episode_steps: int = 1_000,
    task_config: Path = DEFAULT_TASK,
    classify_at_end: bool = False,
) -> None:
    """Train fresh SAC on CPU; checkpoints are not exact training resumes.

    Evaluation uses development seeds 60000 onward. Training resets use only
    0..39999. The 30-seed checkpoint classification is a separate evaluation.
    """
    if min(total_steps, save_every, eval_every, eval_episodes, batch_size,
           hidden_dim, buffer_size, max_episode_steps) < 1:
        raise ValueError("Counts must be positive.")
    if not 0 <= seed < 2**32 or learning_starts < 0 or buffer_size < batch_size:
        raise ValueError("Invalid seed, learning_starts or buffer_size.")
    if eval_episodes > 10_000:
        raise ValueError("Monitoring reset seeds are reserved to 60000..69999.")
    # Refuse existing paths, including partial runs. Never overwrite evidence.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "evaluations").mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    action_rng = np.random.default_rng(seed)
    reset_rng = np.random.default_rng(seed + 1)
    cfg = SACConfig(batch_size=batch_size, hidden_dim=hidden_dim)
    env = WalkerRL(task_config, cfg.gamma, max_episode_steps)
    low, high = env.walker.action_bounds
    agent = SACAgent(17, low, high, config=cfg, device="cpu")
    replay = ReplayBuffer(buffer_size, 17, len(low), seed=seed + 2)
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    metadata = provenance()
    metadata["source_sha256"]["scripts/training/train_walker2d_sac.py"] = sha256(Path(__file__))
    metadata["source_sha256"]["docs/rl_ladder.md"] = sha256(_ROOT / "docs/rl_ladder.md")
    metadata.update(git_head=git_head, git_status=subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=_ROOT, text=True),
        task_sha256=sha256(task_config), frozen_mppi_config_sha256=sha256(_ROOT / "configs/walker2d_best.json"),
        control_dt=env.walker._dt, physics_dt=float(env.walker.model.opt.timestep),
        frame_skip=env.walker._frame_skip, torch_threads=1, simulator_threads=1)
    config = {"seed": seed, "total_steps": total_steps, "learning_starts": learning_starts,
              "save_every": save_every, "eval_every": eval_every, "eval_episodes": eval_episodes,
              "buffer_size": buffer_size, "max_episode_steps": max_episode_steps,
              "classify_at_end": classify_at_end,
              "sac": asdict(cfg), "task": env.task, "task_config": str(task_config.resolve()),
              "failure_tail": env.failure_tail, "observation_transform": "identity",
              "device": "cpu", "training_reset_seed_range": [0, 39999],
              "monitor_reset_seed_base": 60000, "classification_reset_seeds": [50000, 50029],
              "provenance": metadata, "pid": os.getpid(),
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_scope": "Network/optimizer snapshot; replay and simulator state NOT saved. No exact resume claim."}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "task.json", env.task)
    # Preserve exact executed source bytes even if working files change later.
    for relative in metadata["source_sha256"]:
        destination = run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_ROOT / relative).read_bytes())
    (run_dir / "source" / "pyproject.toml").write_bytes((_ROOT / "pyproject.toml").read_bytes())
    (run_dir / "source" / "uv.lock").write_bytes((_ROOT / "uv.lock").read_bytes())
    started = time.perf_counter()
    step = updates = episodes = 0
    last_metrics = {}

    def status(state: str, **extra):
        write_json(run_dir / "status.json", {"status": state, "step": step, "updates": updates,
                   "episodes": episodes, "pid": os.getpid(), "total_steps": total_steps,
                   "elapsed_seconds": time.perf_counter() - started,
                   "latest_update": last_metrics, **extra})

    def checkpoint():
        path = run_dir / "checkpoints" / f"step_{step:07d}.pt"
        payload = {"format_version": 1, "step": step, "seed": seed, "gamma": cfg.gamma,
                   "task_sha256": config["provenance"]["task_sha256"],
                   "policy": agent.policy_state(), "agent": agent.state_dict(),
                   "source_sha256": metadata["source_sha256"], "config": config}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        append_record(run_dir / "checkpoints.jsonl", {"step": step, "path": str(path),
                      "sha256": sha256(path)})
        return path

    def monitor(path):
        report = evaluate_policy(agent, task_config=task_config, gamma=cfg.gamma,
                                 episodes=eval_episodes, steps=max_episode_steps, seed=60000,
                                 traces_path=run_dir / "evaluations" / f"step_{step:07d}.npz")
        report.update(training_step=step, checkpoint=str(path), checkpoint_sha256=sha256(path),
                      purpose="development monitoring, not classification or held-out evaluation")
        write_json(run_dir / "evaluations" / f"step_{step:07d}.json", report)
        append_record(run_dir / "metrics.jsonl", {"type": "evaluation", "step": step,
                      "summary": report["summary"]})
        print(json.dumps({"step": step, "evaluation": report["summary"]}), flush=True)

    try:
        status("running")
        monitor(checkpoint())
        reset_seed = int(reset_rng.integers(0, 40000))
        obs, _ = env.reset(seed=reset_seed)
        episode_reward = episode_cost = 0.0
        for step in range(1, total_steps + 1):
            if step <= learning_starts:
                action = action_rng.uniform(low, high).astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = collect_transition(env, replay, obs, action)
            episode_reward += reward
            episode_cost += info["task_cost"]
            obs = next_obs
            if step > learning_starts and len(replay) >= batch_size:
                last_metrics = agent.update(replay, step)
                updates += 1
                if not all(np.isfinite(value) for value in last_metrics.values()):
                    raise FloatingPointError(f"Non-finite SAC update at step {step}: {last_metrics}")
            if terminated or truncated:
                episodes += 1
                append_record(run_dir / "episodes.jsonl", {"episode": episodes, "step": step,
                    "reset_seed": reset_seed, "steps": info["elapsed_steps"],
                    "terminated": terminated, "truncated": truncated,
                    "task_cost": episode_cost, "training_return": episode_reward,
                    "failure_tail": info["failure_tail"]})
                reset_seed = int(reset_rng.integers(0, 40000))
                obs, _ = env.reset(seed=reset_seed)
                episode_reward = episode_cost = 0.0
            if step % 1000 == 0:
                append_record(run_dir / "metrics.jsonl", {"type": "update", "step": step,
                    "updates": updates, "elapsed_seconds": time.perf_counter() - started, **last_metrics})
                status("running")
                print(f"step={step}/{total_steps} updates={updates} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            if step % save_every == 0 or step % eval_every == 0 or step == total_steps:
                saved_path = checkpoint()
                if step % eval_every == 0 or step == total_steps:
                    monitor(saved_path)
                status("running")
        if classify_at_end:
            status("classifying")
            classify_ladder(run_dir, task_config)
        status("complete")
    except BaseException as exc:
        status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=repr(exc))
        raise
    finally:
        env.close()


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
