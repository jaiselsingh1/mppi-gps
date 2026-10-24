"""Step 25: does MPPI still walk without the plan it carries in?

Steps 22-24 found that MPPI's label at a state is mostly set by its carried-over
plan, so a policy that sees only the state cannot fit it. A memoryless MPPI
re-plans from the zero plan at every decision (planner.reset() before each
plan_step, so every decision runs initial_iterations passes from zero). Its
label then depends only on the state and its samples. Nothing from a policy is
involved.

- Episodes: raw MPPI and memoryless arms on the same development resets,
  paired by reset. Completion means no fall in 1,000 controls; speed MAE
  averages |1.5 - vx| over executed controls, failures included (as in
  walker_headroom.py and Step 18).
- Labels: at Step 22's 256 teacher anchors, re-recorded with Step 22's seeds
  and checked against its saved labels, the variance of a memoryless label over
  planner seeds is its whole floor, because the warm start is always the zero
  plan.
"""
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from scripts.diagnostics.label_variance_split import (LABELS, SplitConfig, _provenance, _resamples, _with_interval,
                                                       query, record_episode)
from src.utils.stats import paired_bootstrap_mean_difference, paired_proportion_difference, wilson_interval

_REPO_ROOT = Path(__file__).resolve().parents[2]
# name: (K, passes per decision, memoryless). Raw MPPI runs its passes only at
# an episode's first decision.
ARMS = {
    "raw": (32, 2, False),
    "memoryless_k32": (32, 2, True),
    "memoryless_k256": (256, 2, True),
    "memoryless_k64_x8": (64, 8, True),
}
H = 48


def _cfg(K: int, passes: int, horizon: int = H):
    from src.utils.config import MPPIConfig

    base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    return replace(base, K=K, H=horizon, initial_iterations=passes)


def run_episode(K: int, passes: int, memoryless: bool, reset_seed: int, planner_seed: int, steps: int,
                horizon: int = H) -> dict:
    """One episode on Step 18's runner (no action noise), optionally memoryless."""
    from scripts.training.train_walker_mdgps import _walker
    from src.mppi.mppi import MPPI

    env = _walker()
    np.random.seed(reset_seed)
    env.reset()
    planner = MPPI(env, _cfg(K, passes, horizon), seed=planner_seed)
    velocities, seconds, neff = [], [], []
    survived = True
    for _ in range(steps):
        if memoryless:
            planner.reset()
        started = time.perf_counter()
        action, info = planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
        seconds.append(time.perf_counter() - started)
        neff.append(info["n_eff"])
        _, _, done, step_info = env.step(action)
        velocities.append(step_info["x_velocity"])
        if done:
            survived = False
            break
    vx = np.asarray(velocities)
    return {"reset_seed": reset_seed, "planner_seed": planner_seed, "steps": len(vx), "survived": survived,
            "mean_vx": float(vx.mean()), "velocity_mae": float(np.mean(np.abs(env._v_target - vx))),
            "median_neff": float(np.median(neff)), "mean_plan_seconds": float(np.mean(seconds))}


def _episode_job(job: dict) -> dict:
    K, passes, memoryless = ARMS[job["arm"]]
    return {"arm": job["arm"]} | run_episode(K, passes, memoryless, job["reset_seed"], job["planner_seed"],
                                             job["steps"])


def single_start_components(labels: np.ndarray) -> dict[str, float]:
    """labels (N, R, nu) from one fixed warm start: floor = mean Var_r, plus the
    state component and the share of label variance it explains."""
    R = labels.shape[1]
    floor = labels.var(axis=1, ddof=1)
    state = float((labels.mean(axis=1).var(axis=0, ddof=1) - floor.mean(axis=0) / R).mean())
    total = state + float(floor.mean())
    return {"floor": float(floor.mean()), "state": state, "total": total, "explainable_share": state / total,
            "rms_label": float(np.sqrt(np.mean(labels ** 2)))}


def _label_job(job: dict) -> dict:
    """Re-record one Step 22 teacher episode and query memoryless arms at its anchors."""
    from scripts.training.train_walker_mdgps import _walker
    from src.mppi.mppi import MPPI

    split_cfg, episode, anchors, queries = job["split_cfg"], job["episode"], job["anchors"], job["queries"]
    data = record_episode(split_cfg, "teacher", episode)
    for anchor, t, saved in anchors:
        if not np.array_equal(data["labels"][t], saved):
            raise RuntimeError(f"re-recorded episode {episode} differs from Step 22 at decision {t}")
    out = {"anchor": [a for a, _, _ in anchors]}
    memoryless = [name for name, (_, _, m) in ARMS.items() if m]
    for arm_index, arm in enumerate(memoryless):
        K, passes, _ = ARMS[arm]
        planner = MPPI(_walker(), _cfg(K, passes), seed=0)
        cells = []
        for anchor, t, _ in anchors:
            for r in range(queries):
                seed = 8_000_000 + (anchor * len(memoryless) + arm_index) * queries + r
                snapshot = {"U": np.zeros((H, planner.nu)), "initial_plan": True,
                            "rng_state": np.random.default_rng(seed).bit_generator.state}
                cells.append(query(split_cfg, planner, data["states"][t], data["warmstarts"][t], snapshot))
        for label in LABELS:
            out[f"{arm}_{label}"] = np.asarray([c[label] for c in cells]).reshape(len(anchors), queries, -1)
    return out


def main(out_dir: Path, step22_raw: Path = Path("runs/2026-10-24_label_variance_001/raw.npz"),
         reset_seed: int = 70_000, planner_seed: int = 95_000, episodes: int = 50, expensive_episodes: int = 25,
         steps: int = 1000, queries: int = 8, bootstrap: int = 2000,
         workers: int = max(1, (os.cpu_count() or 2) - 2)) -> None:
    raw = np.load(_REPO_ROOT / step22_raw)
    out_dir.mkdir(parents=True, exist_ok=False)
    config = {"arms": ARMS, "H": H, "reset_seed": reset_seed, "planner_seed": planner_seed, "episodes": episodes,
              "expensive_episodes": expensive_episodes, "steps": steps, "queries": queries,
              "step22_raw": str(step22_raw)}
    (out_dir / "config.json").write_text(json.dumps(config | _provenance(), indent=2))
    count = {arm: episodes if K * passes <= 64 else expensive_episodes for arm, (K, passes, _) in ARMS.items()}
    jobs = [{"arm": arm, "reset_seed": reset_seed + e, "planner_seed": planner_seed + e, "steps": steps}
            for arm in ARMS for e in range(count[arm])]
    jobs.sort(key=lambda j: -ARMS[j["arm"]][0] * ARMS[j["arm"]][1])
    split_cfg = SplitConfig()
    by_episode = {}
    for anchor, (e, t) in enumerate(zip(raw["teacher_episode"], raw["teacher_t"])):
        by_episode.setdefault(int(e), []).append((anchor, int(t), raw["teacher_recorded_label"][anchor]))
    label_jobs = [{"split_cfg": split_cfg, "episode": e, "anchors": anchors, "queries": queries}
                  for e, anchors in by_episode.items()]
    started = time.time()
    with Pool(workers) as pool:
        label_results = pool.map_async(_label_job, label_jobs)
        rows = pool.map(_episode_job, jobs)
        label_results = label_results.get()
    result = {"wall_time_s": time.time() - started, "arms": {}, "paired_with_raw": {}, "labels": {}}
    arms = {arm: sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["reset_seed"]) for arm in ARMS}
    for arm, arm_rows in arms.items():
        survived = np.array([r["survived"] for r in arm_rows])
        result["arms"][arm] = {
            "episodes": len(arm_rows), "completed": int(survived.sum()),
            "completion_wilson95": list(wilson_interval(int(survived.sum()), len(arm_rows))),
            "mean_seconds_alive": float(np.mean([r["steps"] for r in arm_rows]) * 0.008),
            "velocity_mae": float(np.mean([r["velocity_mae"] for r in arm_rows])),
            "median_neff": float(np.median([r["median_neff"] for r in arm_rows])),
            "mean_plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in arm_rows]))}
        if arm != "raw":
            reference = arms["raw"][:len(arm_rows)]
            field = lambda rs, key: np.array([r[key] for r in rs], dtype=float)
            result["paired_with_raw"][arm] = {
                "completion_difference": paired_proportion_difference(
                    survived.astype(int), np.array([r["survived"] for r in reference]).astype(int)),
                "seconds_alive_difference": paired_bootstrap_mean_difference(
                    0.008 * field(arm_rows, "steps"), 0.008 * field(reference, "steps")),
                "velocity_mae_difference": paired_bootstrap_mean_difference(
                    field(arm_rows, "velocity_mae"), field(reference, "velocity_mae"))}
    order = np.argsort(np.concatenate([res["anchor"] for res in label_results]))
    episodes_of = raw["teacher_episode"]
    resamples = list(_resamples(episodes_of, bootstrap, 8_100_000))
    saved = {}
    for arm in [name for name, (_, _, m) in ARMS.items() if m]:
        for label in LABELS:
            labels = np.concatenate([res[f"{arm}_{label}"] for res in label_results])[order]
            saved[f"{arm}_{label}"] = labels
            result["labels"][f"{arm}_{label}"] = _with_interval(
                single_start_components(labels), [single_start_components(labels[i]) for i in resamples])
    np.savez_compressed(out_dir / "labels.npz", **saved)
    with (out_dir / "episodes.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    for arm, summary in result["arms"].items():
        print(arm, f"{summary['completed']}/{summary['episodes']}", round(summary["mean_seconds_alive"], 2), "s",
              "MAE", round(summary["velocity_mae"], 3), "plan s", round(summary["mean_plan_seconds"], 3))
    for name, parts in result["labels"].items():
        print(name, {key: round(value["mean"], 4) for key, value in parts.items()})


if __name__ == "__main__":
    tyro.cli(main)
