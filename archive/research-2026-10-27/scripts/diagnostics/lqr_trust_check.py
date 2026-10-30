"""Step 14b: exact checks of docs/theory.md on the LQR oracle.

Step 3's setup (linear policy u = -G x, one policy-centred E-step at every
visited state, least-squares M-step), with four checks:
  A. With Sigma fixed, the noise floor scales as sigma / sqrt(n_eff)
     (Consequence 2).
  B. Shrinking sigma as k^(-1/2) removes that floor (Consequence 2).
  C. The trust step (Prop. 2), with the exact LQR controller as the planner,
     takes the large step's speed and the small step's floor.
  D. The predictive check (Prop. 4) lowers the floor.
Every arm labels with `estep_label`, which reproduces `e_step`'s label for the
fixed n_eff rule (tests/test_trust.py).
"""
import json
import time
from pathlib import Path

import numpy as np
import tyro

from src.envs.lqr_point_mass import LQRPointMass
from src.gps.mdgps import closed_loop_nominal
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.math import effective_sample_size, softmin_weights, temperature_for_budget, temperature_for_neff_fraction


def estep_label(env, state, nominal, noise, rho, planner_plan=None, rho_cap=None, check=False):
    """One policy-centred E-step with the given noise; returns (label, info).

    rho alone is the working method's step (n_eff = rho K). With planner_plan,
    the temperature makes the weighted mean cost meet the planner's noisy
    expected cost (same noise), clipped between the rho_cap step and the rho
    step. With check, the label is the policy's own action whenever the
    improved plan costs more than the policy's plan.
    """
    low, high = env.action_bounds
    nominal = np.clip(nominal, low, high)
    sampled = np.clip(nominal[None] + noise, low, high)
    _, scores, _ = env.batch_rollout(state, sampled)
    lam = temperature_for_neff_fraction(scores, rho)
    if planner_plan is not None:
        _, planner_scores, _ = env.batch_rollout(state, np.clip(planner_plan[None] + noise, low, high))
        cap = temperature_for_neff_fraction(scores, rho_cap)
        lam = float(np.clip(temperature_for_budget(scores, float(np.mean(planner_scores))), cap, lam))
    weights = softmin_weights(scores, lam)
    plan = np.clip(nominal + np.einsum("k,kha->ha", weights, sampled - nominal[None]), low, high)
    info = {"neff_fraction": effective_sample_size(weights) / len(scores), "rejected": False}
    if check:
        _, costs, _ = env.batch_rollout(state, np.stack([nominal, plan]))
        if costs[1] > costs[0]:
            info["rejected"] = True
            return nominal[0].copy(), info
    return plan[0].copy(), info


def run(env, start_at_optimum, seed, iterations, samples, rho, sigma_at, rho_cap=None, check=False,
        episodes=6, episode_len=25, horizon=16) -> dict:
    rng = np.random.default_rng(seed + 1)
    gain = env.K_lqr.copy() if start_at_optimum else np.zeros_like(env.K_lqr)
    scale = np.linalg.norm(env.K_lqr)
    lqr = lambda s: -env.K_lqr @ env.lqr_state(s)
    errors, neff, rejected = [], [], []
    for k in range(iterations + 1):
        errors.append(float(np.linalg.norm(gain - env.K_lqr) / scale))
        if k == iterations:
            break
        planner = MPPI(env, MPPIConfig(K=samples, H=horizon, noise_sigma=sigma_at(k)), seed=1000 * seed + k)
        act = lambda s, g=gain: -g @ env.lqr_state(s)
        xs, labels, infos = [], [], []
        for _ in range(episodes):
            start = env.full_state(rng.uniform(-0.05, 0.05, 4))
            _, visited = closed_loop_nominal(env, start, act, episode_len - 1)
            for state in np.concatenate([start[None], visited]):
                nominal, _ = closed_loop_nominal(env, state, act, horizon)
                plan = closed_loop_nominal(env, state, lqr, horizon)[0] if rho_cap is not None else None
                label, info = estep_label(env, state, nominal, planner._sample_noise(), rho, plan, rho_cap, check)
                xs.append(env.lqr_state(state))
                labels.append(label)
                infos.append(info)
        gain = -np.linalg.lstsq(np.asarray(xs), np.asarray(labels), rcond=None)[0].T
        neff.append(float(np.median([i["neff_fraction"] for i in infos])))
        rejected.append(float(np.mean([i["rejected"] for i in infos])))
    return {"relative_gain_error": errors, "median_neff_fraction": neff, "rejected_fraction": rejected}


def first_below(errors, level=0.10):
    return next((k for k, e in enumerate(errors) if e < level), None)


def main(out_dir: Path, seeds: tuple[int, ...] = (0, 1)) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    env = LQRPointMass(nthread=4)
    fixed = lambda sigma: (lambda k: sigma)
    shrinking = lambda k: 0.3 / np.sqrt(1.0 + k / 5.0)
    report = {"seeds": list(seeds), "A": [], "B": [], "C": [], "D": []}

    for samples, rho, sigma in [(256, r, s) for s in (0.15, 0.3, 0.6) for r in (0.05, 0.2, 0.5)] + [(64, 0.2, 0.3), (1024, 0.2, 0.3)]:
        for seed in seeds:
            row = run(env, True, seed, 10, samples, rho, fixed(sigma))
            floor = float(np.median(row["relative_gain_error"][1:]))
            report["A"].append({"K": samples, "rho": rho, "sigma": sigma, "seed": seed, "floor": floor,
                                "n_eff": rho * samples, **row})
            print(f"A K={samples} rho={rho} sigma={sigma} seed={seed}: floor {floor:.4f}", flush=True)
    a = report["A"]
    design = np.column_stack([np.log([r["sigma"] for r in a]), np.log([r["n_eff"] for r in a]), np.ones(len(a))])
    slopes = np.linalg.lstsq(design, np.log([r["floor"] for r in a]), rcond=None)[0]
    report["A_fit"] = {"sigma_exponent": float(slopes[0]), "n_eff_exponent": float(slopes[1])}
    print("A fit", report["A_fit"], flush=True)

    for start_opt in (False, True):
        for name, sigma_at in (("fixed", fixed(0.3)), ("shrinking", shrinking)):
            for seed in seeds:
                row = run(env, start_opt, seed, 40, 256, 0.2, sigma_at)
                errors = row["relative_gain_error"]
                summary = {"late_median": float(np.median(errors[31:])), "first_below_0.10": first_below(errors)}
                report["B"].append({"arm": name, "start_at_optimum": start_opt, "seed": seed, **summary, **row})
                print(f"B {name} start_opt={start_opt} seed={seed}: {summary}", flush=True)

    for name, rho, cap, check in (("fixed_0.05", 0.05, None, False), ("fixed_0.2", 0.2, None, False),
                                  ("trust", 0.2, 0.05, False), ("check", 0.2, None, True)):
        for seed in seeds:
            row = run(env, False, seed, 30, 256, rho, fixed(0.3), rho_cap=cap, check=check)
            errors = row["relative_gain_error"]
            summary = {"first_below_0.10": first_below(errors), "floor": float(np.median(errors[20:])),
                       "neff_first3": float(np.median(row["median_neff_fraction"][:3])),
                       "neff_late": float(np.median(row["median_neff_fraction"][20:])),
                       "rejected_late_min": float(np.min(row["rejected_fraction"][20:]))}
            report["D" if name == "check" else "C"].append({"arm": name, "seed": seed, **summary, **row})
            print(f"{'D' if name == 'check' else 'C'} {name} seed={seed}: {summary}", flush=True)

    report["wall_time_s"] = time.time() - started
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
