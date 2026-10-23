"""Step 5 comparison: policy-alone completion and speed MAE on held-out resets.

Every arm is scored on the same held-out reset seeds (70000+). Policies run
alone through scripts/runners/eval_policy_walker2d.evaluate_policy (no planner
at test time); raw MPPI runs at the Step 4 cell through walker_headroom's
episode function. Completion gets Wilson intervals and paired Newcombe
differences against each reference; speed MAE gets paired bootstrap intervals.
"""
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from src.utils.stats import paired_bootstrap_mean_difference, paired_proportion_difference, wilson_interval

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _evaluate_checkpoint(job: dict) -> dict:
    import torch

    from scripts.runners.eval_policy_walker2d import evaluate_policy
    from src.gps.averaging import AveragedPolicy
    from src.rl.sac import load_policy

    torch.set_num_threads(1)
    policies = []
    for path in str(job["path"]).split("+"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        policies.append(load_policy(payload.get("policy", payload), device="cpu"))
    policy = policies[0] if len(policies) == 1 else AveragedPolicy(policies)
    episodes = evaluate_policy(policy, episodes=job["episodes"], seed=job["seed"])["episodes"]
    return {"arm": job["arm"], "source": str(job["path"]),
            "survived": [bool(e["survived"]) for e in episodes],
            "velocity_mae": [float(e["velocity_mae"]) for e in episodes],
            "steps": [int(e["steps"]) for e in episodes]}


def _mppi_episode(job: dict) -> dict:
    from scripts.diagnostics.walker_headroom import _episode

    return _episode(job)


def _arm_summary(survived: np.ndarray, mae: np.ndarray, steps: np.ndarray | None = None) -> dict:
    low, high = wilson_interval(int(survived.sum()), len(survived))
    summary = {"episodes": int(len(survived)), "completed": int(survived.sum()),
               "completion_rate": float(survived.mean()), "completion_wilson95": [low, high],
               "velocity_mae": float(mae.mean())}
    if steps is not None:
        summary["mean_seconds_alive"] = float(steps.mean() * 0.008)
    return summary


def main(
    out_dir: Path,
    checkpoints: tuple[str, ...],
    mppi_K: int,
    mppi_H: int,
    episodes: int = 50,
    reset_seed: int = 70_000,
    planner_seed: int = 95_000,
    pairs: tuple[str, ...] = (),
    references: tuple[str, ...] = ("sac_800k",),
    workers: int = max(1, (os.cpu_count() or 2) - 4),
) -> None:
    """checkpoints: "arm=path" entries; repeat an arm name for its training seeds.
    A path of several checkpoints joined by "+" is evaluated as the policy that
    acts with the mean of their actions.

    pairs: "candidate:reference" arm names compared seed by seed, the i-th
    checkpoint of one against the i-th of the other, paired by reset seed.
    references: single-checkpoint arms that every seed of every other arm is
    paired against, like raw MPPI.
    """
    policy_jobs = []
    for entry in checkpoints:
        arm, path = entry.split("=", 1)
        policy_jobs.append({"arm": arm, "path": Path(path), "episodes": episodes, "seed": reset_seed})
    for name in references:
        if sum(job["arm"] == name for job in policy_jobs) > 1:
            raise ValueError(f"Reference arm {name!r} must have one checkpoint.")
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    mppi_jobs = [{"K": mppi_K, "H": mppi_H, "neff_fraction": None, "tail_policy": None, "tail_steps": 0,
                  "steps": 1000, "reset_seed": reset_seed + e, "planner_seed": planner_seed + e}
                 for e in range(episodes)]
    with Pool(workers) as pool:
        mppi_rows = pool.map(_mppi_episode, mppi_jobs)
        policy_rows = pool.map(_evaluate_checkpoint, policy_jobs)
    mppi_rows.sort(key=lambda r: r["reset_seed"])
    reference = {"raw_mppi": (np.array([r["survived"] for r in mppi_rows]),
                              np.array([r["velocity_mae"] for r in mppi_rows]))}
    mppi_steps = np.array([r["steps"] for r in mppi_rows])
    for row in policy_rows:
        if row["arm"] in references:
            reference[row["arm"]] = (np.array(row["survived"]), np.array(row["velocity_mae"]))
    arms = {}
    for row in policy_rows:
        arms.setdefault(row["arm"], []).append(row)
    report = {"held_out_reset_seeds": [reset_seed, reset_seed + episodes - 1],
              "raw_mppi_cell": {"K": mppi_K, "H": mppi_H},
              "raw_mppi": _arm_summary(*reference["raw_mppi"], mppi_steps), "arms": {}}
    for arm, rows in arms.items():
        survived = np.array([s for r in rows for s in r["survived"]])
        mae = np.array([m for r in rows for m in r["velocity_mae"]])
        steps = np.array([n for r in rows for n in r["steps"]])
        entry = {"pooled": _arm_summary(survived, mae, steps), "per_seed": [], "paired": {}}
        for r in rows:
            entry["per_seed"].append({"source": r["source"], **_arm_summary(
                np.array(r["survived"]), np.array(r["velocity_mae"]), np.array(r["steps"]))})
            for name, (ref_survived, ref_mae) in reference.items():
                if name == arm:
                    continue
                entry["paired"].setdefault(name, []).append({
                    "source": r["source"],
                    "completion_difference": paired_proportion_difference(
                        np.array(r["survived"]).astype(int), ref_survived.astype(int)),
                    "velocity_mae_difference": paired_bootstrap_mean_difference(
                        np.array(r["velocity_mae"]), ref_mae),
                })
        report["arms"][arm] = entry
    report["arm_pairs"] = {}
    for pair in pairs:
        candidate, reference_arm = pair.split(":", 1)
        rows = []
        for cand, ref in zip(arms[candidate], arms[reference_arm], strict=True):
            rows.append({
                "candidate": cand["source"], "reference": ref["source"],
                "completion_difference": paired_proportion_difference(
                    np.array(cand["survived"]).astype(int), np.array(ref["survived"]).astype(int)),
                "velocity_mae_difference": paired_bootstrap_mean_difference(
                    np.array(cand["velocity_mae"]), np.array(ref["velocity_mae"])),
                # Exploratory, not preregistered: time to fall, in seconds.
                "seconds_alive_difference": paired_bootstrap_mean_difference(
                    0.008 * np.array(cand["steps"], dtype=float), 0.008 * np.array(ref["steps"], dtype=float)),
            })
        report["arm_pairs"][pair] = rows
    report["wall_time_s"] = time.time() - started
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "episodes.json").write_text(json.dumps({"raw_mppi": mppi_rows, "policies": policy_rows}))
    print(json.dumps({"raw_mppi": report["raw_mppi"],
                      **{arm: v["pooled"] for arm, v in report["arms"].items()}}, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
