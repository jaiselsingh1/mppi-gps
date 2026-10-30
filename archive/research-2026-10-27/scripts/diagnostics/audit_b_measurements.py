"""Measured numbers for the Step 2 audit of MPPI-GPS (B) (docs/objective.md).

1. n_eff/K under each frozen planner config: how large a KL step a fixed
   temperature takes (plug-in KL <= log(K/n_eff)).
2. Label noise at fixed Walker states: the spread of MPPI's first action across
   planner seeds from the same state, warm start and nominal, for the frozen
   temperature and for n_eff-targeted temperatures. This is the jitter a
   regression target carries.
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import tyro

from src.envs.acrobot import Acrobot
from src.envs.point_mass import PointMass
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _walker() -> Walker2d:
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    return Walker2d(**task)


def neff_trace(env, cfg: MPPIConfig, reset_seed: int, planner_seed: int, steps: int) -> dict:
    np.random.seed(reset_seed)
    env.reset()
    planner = MPPI(env, cfg, seed=planner_seed)
    fractions = []
    for _ in range(steps):
        action, info = planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
        fractions.append(info["n_eff"] / cfg.K)
        if env.step(action)[2]:
            break
    fractions = np.asarray(fractions)
    return {
        "K": cfg.K,
        "lam": cfg.lam,
        "decisions": int(len(fractions)),
        "neff_fraction_median": float(np.median(fractions)),
        "neff_fraction_p10": float(np.quantile(fractions, 0.1)),
        "neff_median": float(np.median(fractions) * cfg.K),
        "kl_bound_median_nats": float(np.log(1.0 / np.median(fractions))),
    }


def label_spread(
    env: Walker2d,
    cfg: MPPIConfig,
    reset_seed: int,
    planner_seed: int,
    decisions: tuple[int, ...],
    n_seeds: int,
    targets: tuple[float | None, ...],
) -> dict:
    np.random.seed(reset_seed)
    env.reset()
    planner = MPPI(env, cfg, seed=planner_seed)
    saved = {}
    for t in range(max(decisions) + 1):
        state, warmstart = env.get_state(), env.get_warmstart()
        if t in decisions:
            saved[t] = (state, warmstart, planner.snapshot()["U"])
        action, _ = planner.plan_step(state, initial_warmstart=warmstart)
        if env.step(action)[2]:
            raise RuntimeError(f"Walker fell before decision {max(decisions)}.")

    rows = {}
    for target in targets:
        setting = replace(cfg, lam_target_neff_fraction=target)
        spreads, fractions, lams = [], [], []
        for state, warmstart, nominal in saved.values():
            actions = []
            for seed in range(n_seeds):
                probe = MPPI(env, setting, seed=50_000 + seed)
                probe.U = nominal.copy()
                probe._initial_plan = False
                action, info = probe.plan_step(state, initial_warmstart=warmstart)
                actions.append(action)
                fractions.append(info["n_eff"] / cfg.K)
                lams.append(info["lam"])
            spreads.append(np.std(np.asarray(actions), axis=0).mean())
        rows["frozen_lam" if target is None else f"neff_fraction_{target:g}"] = {
            "first_action_std_across_seeds": float(np.mean(spreads)),
            "neff_fraction_median": float(np.median(fractions)),
            "lam_median": float(np.median(lams)),
        }
    return {"decisions": list(decisions), "seeds_per_state": n_seeds, "settings": rows}


def main(out_dir: Path, walker_steps: int = 150, other_steps: int = 50, n_seeds: int = 16) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    walker_cfg = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    report = {
        "neff_under_frozen_configs": {
            "walker2d": neff_trace(_walker(), walker_cfg, 30, 20030, walker_steps),
            "acrobot": neff_trace(Acrobot(), MPPIConfig.load("acrobot"), 0, 10_000, other_steps),
            "point_mass": neff_trace(
                PointMass(random_goal=False), MPPIConfig.load("point_mass"), 0, 10_000, other_steps
            ),
        },
        "walker_label_spread": label_spread(
            _walker(), walker_cfg, 30, 20030, (20, 50, 80, 110, 140), n_seeds, (None, 0.05, 0.2)
        ),
    }
    report["wall_time_s"] = time.time() - started
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
