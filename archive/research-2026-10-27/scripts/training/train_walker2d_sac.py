"""Train the S2 local-Walker ladder; each invocation owns a new run directory.

With --mppi-share > 0, that share of every --mppi-block executed controls comes
from MPPI episodes instead of the policy (src/rl/mppi_stream.py, Step 19). They
enter the same replay buffer, the budget counts them, and SAC still makes one
update per executed control. With --mppi-terminal actor, MPPI's terminal cost
is the current actor's simulated cost-to-go; MPPI's initialization, plans and
sampling stay its own. The default, --mppi-share 0, is plain SAC.
"""

import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.runners.eval_policy_walker2d import classify_ladder, evaluate_policy, provenance, sha256, write_json
from src.rl.mppi_stream import collect_block
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
    mppi_share: float = 0.0,
    mppi_block: int = 10_000,
    mppi_terminal: str = "none",
    mppi_K: int = 32,
    mppi_H: int = 32,
    mppi_tail_steps: int = 100,
    mppi_episode_cap: int = 500,
    mppi_workers: int = 4,
) -> None:
    """Train fresh SAC on CPU; checkpoints are not exact training resumes.

    Evaluation uses development seeds 60000 onward. Training resets use only
    0..39999; MPPI episodes reset from 200000 + 10000 * seed onward. The
    30-seed checkpoint classification is a separate evaluation.
    """
    if min(total_steps, save_every, eval_every, eval_episodes, batch_size,
           hidden_dim, buffer_size, max_episode_steps) < 1:
        raise ValueError("Counts must be positive.")
    if not 0 <= seed < 2**32 or learning_starts < 0 or buffer_size < batch_size:
        raise ValueError("Invalid seed, learning_starts or buffer_size.")
    if eval_episodes > 10_000:
        raise ValueError("Monitoring reset seeds are reserved to 60000..69999.")
    if not 0.0 <= mppi_share < 1.0 or mppi_terminal not in {"none", "actor"}:
        raise ValueError("mppi_share must lie in [0, 1) and mppi_terminal be 'none' or 'actor'.")
    mppi_steps = round(mppi_share * mppi_block)
    policy_steps = mppi_block - mppi_steps
    if mppi_share > 0 and (mppi_steps < 1 or policy_steps < 1 or min(mppi_episode_cap, mppi_workers) < 1
                           or mppi_episode_cap >= max_episode_steps):
        raise ValueError("An MPPI block needs policy and MPPI controls, and capped MPPI episodes.")
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
    if mppi_share > 0:
        metadata["source_sha256"]["src/rl/mppi_stream.py"] = sha256(_ROOT / "src/rl/mppi_stream.py")
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
              "mppi_stream": None if mppi_share == 0 else {
                  "share": mppi_share, "block": mppi_block, "mppi_steps_per_block": mppi_steps,
                  "policy_steps_per_block": policy_steps, "terminal": mppi_terminal, "K": mppi_K, "H": mppi_H,
                  "tail_steps": mppi_tail_steps if mppi_terminal == "actor" else 0,
                  "episode_cap": mppi_episode_cap, "workers": mppi_workers,
                  "reset_seed_base": 200_000 + 10_000 * seed, "planner_seed_base": 300_000 + 10_000 * seed},
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

    mppi_index = 0

    def next_mppi_seeds() -> dict:
        nonlocal mppi_index
        mppi_index += 1
        if mppi_index >= 10_000:
            raise RuntimeError("MPPI episode seeds would leave this run's range.")
        return {"reset_seed": 200_000 + 10_000 * seed + mppi_index,
                "planner_seed": 300_000 + 10_000 * seed + mppi_index}

    mppi_job = {"task_config": str(task_config), "gamma": cfg.gamma, "max_episode_steps": max_episode_steps,
                "K": mppi_K, "H": mppi_H, "tail_steps": mppi_tail_steps}
    pool = Pool(mppi_workers) if mppi_share > 0 and mppi_workers > 1 else None
    pending = deque()
    policy_steps_in_block = 0
    try:
        status("running")
        monitor(checkpoint())
        reset_seed = int(reset_rng.integers(0, 40000))
        obs, _ = env.reset(seed=reset_seed)
        episode_reward = episode_cost = 0.0
        for step in range(1, total_steps + 1):
            # MPPI's queued transitions are consumed one per executed control,
            # so updates, checkpoints and evaluations follow the same schedule.
            from_mppi = bool(pending)
            if from_mppi:
                replay.add(*pending.popleft())
            else:
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
            if not from_mppi and (terminated or truncated):
                episodes += 1
                append_record(run_dir / "episodes.jsonl", {"episode": episodes, "step": step,
                    "reset_seed": reset_seed, "steps": info["elapsed_steps"],
                    "terminated": terminated, "truncated": truncated,
                    "task_cost": episode_cost, "training_return": episode_reward,
                    "failure_tail": info["failure_tail"]})
                reset_seed = int(reset_rng.integers(0, 40000))
                obs, _ = env.reset(seed=reset_seed)
                episode_reward = episode_cost = 0.0
            if not from_mppi and mppi_share > 0:
                policy_steps_in_block += 1
                if policy_steps_in_block == policy_steps:
                    policy_steps_in_block = 0
                    policy = agent.policy_state() if mppi_terminal == "actor" else None
                    block, stats = collect_block(pool.map if pool else map, {**mppi_job, "policy": policy},
                                                 mppi_steps, mppi_episode_cap, next_mppi_seeds)
                    pending.extend(block)
                    for episode in stats.pop("per_episode"):
                        append_record(run_dir / "mppi_episodes.jsonl", {"step": step, **episode})
                    append_record(run_dir / "metrics.jsonl", {"type": "mppi_block", "step": step, **stats})
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
        if pool is not None:
            pool.close()
        env.close()


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
