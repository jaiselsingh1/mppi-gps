"""Step 3 oracle: MPPI-GPS (B) with a linear policy must recover the LQR gain.

Each iteration executes the current policy u = -G x from random starts, runs
one policy-centred MPPI step (docs/objective.md, method B) at every visited
state, and refits G by least squares on the labels. The answer is known in
closed form: G* = K from scipy.linalg.solve_discrete_are (LQRPointMass).
"""
import json
import time
from pathlib import Path

import numpy as np
import tyro

from src.envs.lqr_point_mass import LQRPointMass
from src.gps.mdgps import closed_loop_nominal, e_step
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig


def run_setting(
    samples: int,
    neff_fraction: float | None,
    fixed_lam: float,
    seed: int,
    start_at_optimum: bool,
    iterations: int,
    episodes: int,
    episode_len: int,
    horizon: int,
    sigma: float,
) -> dict:
    env = LQRPointMass(nthread=4)
    cfg = MPPIConfig(
        K=samples, H=horizon, lam=fixed_lam, noise_sigma=sigma,
        lam_target_neff_fraction=neff_fraction,
    )
    planner = MPPI(env, cfg, seed=seed)
    rng = np.random.default_rng(seed + 1)
    gain = env.K_lqr.copy() if start_at_optimum else np.zeros_like(env.K_lqr)
    scale = np.linalg.norm(env.K_lqr)
    errors, neff, sample_gain, plan_gain, clip = [], [], [], [], []
    for iteration in range(iterations + 1):
        errors.append(float(np.linalg.norm(gain - env.K_lqr) / scale))
        if iteration == iterations:
            break
        act = lambda s, g=gain: -g @ env.lqr_state(s)
        xs, labels, stats = [], [], []
        for _ in range(episodes):
            start = env.full_state(rng.uniform(-0.05, 0.05, 4))
            _, visited = closed_loop_nominal(env, start, act, episode_len - 1)
            for state in np.concatenate([start[None], visited]):
                nominal, _ = closed_loop_nominal(env, state, act, horizon)
                result = e_step(planner, env, state, nominal)
                xs.append(env.lqr_state(state))
                labels.append(result.label)
                stats.append(result)
        gain = -np.linalg.lstsq(np.asarray(xs), np.asarray(labels), rcond=None)[0].T
        neff.append(float(np.median([r.n_eff for r in stats]) / samples))
        sample_gain.append(float(np.min([r.sample_improvement for r in stats])))
        plan_gain.append(float(np.mean([r.predicted_improvement > 0 for r in stats])))
        clip.append(float(np.mean([r.clip_fraction for r in stats])))
    return {
        "K": samples,
        "neff_fraction_target": neff_fraction,
        "fixed_lam": None if neff_fraction is not None else fixed_lam,
        "seed": seed,
        "start_at_optimum": start_at_optimum,
        "relative_gain_error": errors,
        "median_neff_fraction": neff,
        "min_sample_improvement": sample_gain,
        "fraction_plans_improving": plan_gain,
        "clip_fraction": clip,
    }


def main(
    out_dir: Path,
    iterations: int = 12,
    episodes: int = 6,
    episode_len: int = 25,
    horizon: int = 16,
    sigma: float = 0.3,
    seeds: tuple[int, ...] = (0, 1),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    settings = [(k, rho, 1.0, False) for k in (64, 256, 1024) for rho in (0.05, 0.2, 0.5)]
    settings += [(256, None, 0.01, False), (256, 0.2, 1.0, True)]
    env = LQRPointMass(nthread=1)
    rows = []
    for samples, rho, lam, at_opt in settings:
        for seed in seeds:
            row = run_setting(samples, rho, lam, seed, at_opt, iterations, episodes,
                              episode_len, horizon, sigma)
            rows.append(row)
            label = f"K={samples:5d} rho={rho} lam={row['fixed_lam']} start_opt={at_opt} seed={seed}"
            errs = row["relative_gain_error"]
            print(f"{label}: err {errs[0]:.3f} -> {errs[len(errs) // 2]:.3f} -> {errs[-1]:.4f}", flush=True)
    report = {
        "question": "Does method B with a linear policy converge to the LQR gain?",
        "K_lqr": env.K_lqr.tolist(),
        "settings": {"iterations": iterations, "episodes": episodes, "episode_len": episode_len,
                     "horizon": horizon, "sigma": sigma, "seeds": list(seeds)},
        "runs": rows,
        "wall_time_s": time.time() - started,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
