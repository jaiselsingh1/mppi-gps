"""Step 4: where does raw MPPI on Walker leave headroom, and does a policy tail use it?

Grid mode runs raw MPPI (frozen config except K and H) for eight-second
episodes. Tail mode runs one (K, H) cell with a frozen SAC checkpoint's
closed-loop cost-to-go added as MPPI's terminal cost (src/gps/policy_tail.py),
plus a no-tail arm; pass --neff-fraction so every arm uses the same
n_eff-targeted temperature and only the tail's information differs.
Metric definitions follow scripts/runners/eval_policy_walker2d.py: completion
means no fall in 1,000 controls; speed MAE averages |1.5 - vx| over executed
controls, failures included.
"""
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LADDER = _REPO_ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints"
TAIL_POLICIES = {
    "weak": _LADDER / "step_0000000.pt",
    "intermediate": _LADDER / "step_0200000.pt",
    "strong": _LADDER / "step_0800000.pt",
}


def _episode(job: dict) -> dict:
    import torch

    from src.envs.walker2d import Walker2d
    from src.gps.policy_tail import WalkerPolicyTail
    from src.mppi.mppi import MPPI
    from src.rl.sac import load_policy
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=job["K"], H=job["H"], lam_target_neff_fraction=job["neff_fraction"])
    prior = None
    if job["tail_policy"] is not None:
        payload = torch.load(TAIL_POLICIES[job["tail_policy"]], map_location="cpu", weights_only=False)
        actor = load_policy(payload["policy"], device="cpu")

        @torch.no_grad()
        def act_batch(obs: np.ndarray) -> np.ndarray:
            mean, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
            return (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)

        prior = WalkerPolicyTail(env, act_batch, job["tail_steps"])
    np.random.seed(job["reset_seed"])
    env.reset()
    planner = MPPI(env, cfg, seed=job["planner_seed"])
    velocities, neff, plan_seconds = [], [], []
    survived = True
    for _ in range(job["steps"]):
        started = time.perf_counter()
        action, info = planner.plan_step(env.get_state(), prior_cost=prior,
                                         initial_warmstart=env.get_warmstart())
        plan_seconds.append(time.perf_counter() - started)
        neff.append(info["n_eff"])
        _, _, done, step_info = env.step(action)
        velocities.append(step_info["x_velocity"])
        if done:
            survived = False
            break
    vx = np.asarray(velocities)
    return {
        **{key: job[key] for key in ("K", "H", "neff_fraction", "tail_policy", "tail_steps",
                                     "reset_seed", "planner_seed")},
        "steps": len(vx),
        "survived": survived,
        "mean_vx": float(vx.mean()),
        "velocity_mae": float(np.mean(np.abs(env._v_target - vx))),
        "median_neff": float(np.median(neff)),
        "mean_plan_seconds": float(np.mean(plan_seconds)),
    }


def _summarize(rows: list[dict]) -> list[dict]:
    cells = {}
    for row in rows:
        key = (row["K"], row["H"], row["neff_fraction"], row["tail_policy"], row["tail_steps"])
        cells.setdefault(key, []).append(row)
    summary = []
    for (k, h, rho, tail, tail_steps), cell in sorted(cells.items(), key=lambda c: (str(c[0][3]), c[0][1], c[0][0])):
        summary.append({
            "K": k, "H": h, "neff_fraction": rho, "tail_policy": tail, "tail_steps": tail_steps,
            "episodes": len(cell),
            "completed": int(sum(r["survived"] for r in cell)),
            "completion_rate": float(np.mean([r["survived"] for r in cell])),
            "velocity_mae": float(np.mean([r["velocity_mae"] for r in cell])),
            "mean_steps": float(np.mean([r["steps"] for r in cell])),
            "mean_plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in cell])),
            "median_neff": float(np.median([r["median_neff"] for r in cell])),
        })
    return summary


def select_cell(report: dict, low: float = 0.2, high: float = 0.6, target: float = 0.4) -> dict:
    """The preregistered rule: completion closest to `target` within [low, high],
    ties to the smaller K*H; with no cell in the band, the nearest to it."""
    cells = [c for c in report["cells"] if c["tail_policy"] is None]
    # Rates are multiples of 1/episodes; round so ties really tie.
    rate = lambda c: round(c["completion_rate"], 9)
    in_band = [c for c in cells if low <= rate(c) <= high]
    if in_band:
        chosen = min(in_band, key=lambda c: (round(abs(rate(c) - target), 9), c["K"] * c["H"]))
    else:
        distance = lambda c: round(max(low - rate(c), rate(c) - high, 0.0), 9)
        chosen = min(cells, key=lambda c: (distance(c), c["K"] * c["H"]))
    return {**chosen, "in_band": bool(in_band), "neff_fraction": chosen["median_neff"] / chosen["K"]}


def main(
    out_dir: Path,
    mode: str = "grid",
    horizons: tuple[int, ...] = (8, 12, 16, 24, 32, 48),
    samples: tuple[int, ...] = (32, 64, 128, 256),
    tail_policies: tuple[str, ...] = ("weak", "intermediate", "strong"),
    tail_steps: int = 100,
    neff_fraction: float | None = None,
    episodes: int = 10,
    steps: int = 1000,
    reset_seed: int = 80_000,
    planner_seed: int = 90_000,
    workers: int = max(1, (os.cpu_count() or 2) - 4),
) -> None:
    if mode not in {"grid", "tail"}:
        raise ValueError("mode must be 'grid' or 'tail'.")
    out_dir.mkdir(parents=True, exist_ok=False)
    jobs = []
    for h in horizons:
        for k in samples:
            arms = [None] if mode == "grid" else [None, *tail_policies]
            for tail in arms:
                for episode in range(episodes):
                    jobs.append({"K": k, "H": h, "tail_policy": tail, "neff_fraction": neff_fraction,
                                 "tail_steps": tail_steps if tail else 0, "steps": steps,
                                 "reset_seed": reset_seed + episode,
                                 "planner_seed": planner_seed + episode})
    # Longest jobs first keeps the pool busy.
    jobs.sort(key=lambda j: -(j["K"] * (j["H"] + j["tail_steps"])))
    started = time.time()
    with Pool(workers) as pool:
        rows = []
        for row in pool.imap_unordered(_episode, jobs):
            rows.append(row)
            (out_dir / "episodes.jsonl").open("a").write(json.dumps(row) + "\n")
    report = {"mode": mode, "jobs": len(jobs), "workers": workers,
              "wall_time_s": time.time() - started, "cells": _summarize(rows)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    for cell in report["cells"]:
        print(json.dumps(cell))


def print_selection(report_path: Path) -> None:
    """Print the selected cell of a finished grid report as one JSON line."""
    print(json.dumps(select_cell(json.loads(report_path.read_text()))))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"run": main, "select": print_selection})
