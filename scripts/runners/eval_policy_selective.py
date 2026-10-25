"""Evaluate a raw MPPI-GPS checkpoint with a deterministic policy and no planner."""
from pathlib import Path

import torch
import tyro

from scripts.runners.eval_policy_walker2d import evaluate_policy, write_json
from src.rl.sac import load_policy


def main(checkpoint: Path, output: Path, episodes: int = 10, steps: int = 1000,
         seed: int = 81_000) -> None:
    """Load a saved standalone network; evaluation never runs a planner."""
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy = load_policy(payload)
    report = evaluate_policy(policy, episodes=episodes, steps=steps, seed=seed)
    report["checkpoint"] = str(checkpoint.resolve())
    write_json(output, report)
    print(f"standalone return {report['summary']['training_return']:.3f}; saved {output}")


if __name__ == "__main__":
    tyro.cli(main)
