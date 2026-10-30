"""Step 13c: how often is the E-step's improved plan worse than the policy's own?

For each checkpoint, states are collected the way training collects them
(policy + N(0, 0.1^2) on executed actions, every 2nd state) and labelled twice
by the batched GPU E-step at the training settings, with independent noise
draws. Gain = policy plan cost - improved plan cost on the E-step objective
(H controls plus the policy tail): the predicted improvement, at every state.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import tyro

from scripts.benchmarks.gpu_estep_check import _collect
from src.envs.walker2d import Walker2d
from src.gps.gpu_estep import WalkerGPUEStep
from src.rl.sac import load_policy
from src.utils.config import MPPIConfig

_ROOT = Path(__file__).resolve().parents[2]
_NOISE_SEEDS = (10_000_000, 20_000_000)


def main(out_dir: Path, checkpoints: tuple[str, ...], K: int = 256, H: int = 32, tail_steps: int = 100,
         neff_fraction: float = 0.2, episodes: int = 10, steps: int = 400, collect_seed: int = 123_000,
         states_per_batch: int = 128) -> None:
    """checkpoints: "name=path" entries."""
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = Walker2d(**json.loads((_ROOT / "configs/walker2d_task.json").read_text()), nthread=1)
    base = MPPIConfig(**json.loads((_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=K, H=H, initial_iterations=1, lam_target_neff_fraction=neff_fraction)
    report = {"K": K, "H": H, "tail_steps": tail_steps, "neff_fraction": neff_fraction, "episodes": episodes,
              "steps": steps, "collect_seed": collect_seed, "noise_seeds": _NOISE_SEEDS, "checkpoints": {}}
    for entry in checkpoints:
        name, path = entry.split("=", 1)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actor = load_policy(payload.get("policy", payload), device="cpu")

        @torch.no_grad()
        def act(obs):
            mean, _ = actor(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
            out = (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)
            return out if np.ndim(obs) > 1 else out[0]

        states, warmstarts = _collect(env, act, episodes, steps, 2, 0.1, collect_seed)
        estep = WalkerGPUEStep(env, load_policy(payload.get("policy", payload), device="cuda"), cfg,
                               tail_steps, states_per_batch=states_per_batch)
        draws = [estep(states, warmstarts, [seed + i for i in range(len(states))]) for seed in _NOISE_SEEDS]
        gains = [draw.nominal_cost - draw.plan_cost for draw in draws]
        worse = gains[0] < 0
        moved = np.linalg.norm(draws[0].labels - act(env.rollout_states_to_obs(states)), axis=1)
        median = lambda x: float(np.median(x)) if len(x) else None
        report["checkpoints"][name] = {
            "path": path, "states": int(len(states)),
            "gain_negative_fraction": [float(np.mean(gain < 0)) for gain in gains],
            "gain_median": [float(np.median(gain)) for gain in gains],
            "nominal_cost_median": float(np.median(draws[0].nominal_cost)),
            "sign_agreement": float(np.mean((gains[0] < 0) == (gains[1] < 0))),
            "label_distance_median": {"gain_negative": median(moved[worse]), "gain_positive": median(moved[~worse])},
        }
        print(name, json.dumps(report["checkpoints"][name]), flush=True)
        del estep
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
