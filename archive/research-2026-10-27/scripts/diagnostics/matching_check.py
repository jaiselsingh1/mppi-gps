"""Step 22a: does a network-matching term in MPPI's cost pull MPPI toward a good
policy's gait while it still does the task? No training.

MPPI (frozen config except K and H) keeps its own warm start and its own
samples; for each alpha its score adds the matching term of
src/gps/policy_matching.py toward a fixed policy (alpha = 0 is raw MPPI). The
policy also runs alone from each reset, as the reference gait.
Per episode: survival over `steps` controls, speed MAE, the mean squared gap
between the executed action and the policy's action at the same state, the
joint-angle regularity, and the joint-angle distance to the policy's own
episode and to raw MPPI's from the same reset (scripts/diagnostics/joint_plots.py).
"""
import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from scripts.diagnostics.joint_plots import JOINTS, distance, regularity

_REPO_ROOT = Path(__file__).resolve().parents[2]
SAC_800K = "runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_0800000.pt"


def _episode(job: dict) -> dict:
    import torch

    from src.envs.walker2d import Walker2d
    from src.gps.policy_matching import PolicyMatching
    from src.mppi.mppi import MPPI
    from src.rl.sac import load_policy
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    payload = torch.load(_REPO_ROOT / job["policy"], map_location="cpu", weights_only=False)
    actor = load_policy(payload.get("policy", payload), device="cpu")

    @torch.no_grad()
    def act_batch(obs: np.ndarray) -> np.ndarray:
        mean, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
        return (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)

    np.random.seed(job["reset_seed"])
    obs = env.reset()
    planner = matching = None
    if job["alpha"] is not None:
        base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
        planner = MPPI(env, replace(base, K=job["K"], H=job["H"]), seed=job["planner_seed"])
        matching = PolicyMatching(env, act_batch, job["alpha"]) if job["alpha"] > 0 else None
    angles, velocities, gaps, survived = [], [], [], True
    for _ in range(job["steps"]):
        policy_action = act_batch(obs[None])[0]
        if planner is None:
            action = policy_action
        else:
            if matching is not None:
                matching.state = env.get_state()
            action, _ = planner.plan_step(env.get_state(), prior_cost=matching,
                                          initial_warmstart=env.get_warmstart())
        gaps.append(float(np.mean((action - policy_action) ** 2)))
        obs, _, done, info = env.step(action)
        angles.append(env.data.qpos[3:9].copy())
        velocities.append(info["x_velocity"])
        if done:
            survived = False
            break
    angles = np.asarray(angles)
    return {"alpha": job["alpha"], "reset_seed": job["reset_seed"], "steps": len(angles), "survived": survived,
            "velocity_mae": float(np.mean(np.abs(env._v_target - np.asarray(velocities)))),
            "action_gap": float(np.mean(gaps)), "regularity": regularity(angles, env._dt),
            "angles": angles.tolist(), "dt": env._dt}


def main(
    out_dir: Path,
    alphas: tuple[float, ...] = (0.0, 0.03, 0.1, 0.3, 1.0),
    policy: str = SAC_800K,
    resets: int = 5,
    reset_seed: int = 81_000,
    planner_seed: int = 95_000,
    K: int = 32,
    H: int = 48,
    steps: int = 1000,
    workers: int = 8,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if 0.0 not in alphas:
        raise ValueError("alphas must include 0, raw MPPI.")
    out_dir.mkdir(parents=True, exist_ok=False)
    jobs = [{"alpha": alpha, "policy": policy, "reset_seed": reset_seed + e, "planner_seed": planner_seed + e,
             "K": K, "H": H, "steps": steps}
            for e in range(resets) for alpha in (None, *alphas)]
    with Pool(workers) as pool:
        rows = pool.map(_episode, jobs)
    by = {(r["reset_seed"], r["alpha"]): r for r in rows}
    for r in rows:
        angles = np.asarray(r["angles"])
        r["distance_to_policy"] = distance(angles, np.asarray(by[(r["reset_seed"], None)]["angles"]), r["dt"])
        r["distance_to_raw_mppi"] = distance(angles, np.asarray(by[(r["reset_seed"], 0.0)]["angles"]), r["dt"])
    summary = []
    for alpha in (None, *alphas):
        cell = [r for r in rows if r["alpha"] == alpha]
        mean = lambda key, rows_: float(np.nanmean([r[key] for r in rows_])) if rows_ else float("nan")
        up = [r for r in cell if r["survived"]]
        summary.append({"controller": "policy alone" if alpha is None else f"MPPI, alpha = {alpha:g}",
                        "completed": len(up), "episodes": len(cell), "velocity_mae": mean("velocity_mae", cell),
                        "action_gap": mean("action_gap", cell), "regularity": mean("regularity", up),
                        "distance_to_policy": mean("distance_to_policy", up),
                        "distance_to_raw_mppi": mean("distance_to_raw_mppi", up)})
    first = [by[(reset_seed, alpha)] for alpha in (*alphas, None)]
    fig, axes = plt.subplots(1, len(first), figsize=(4.2 * len(first), 3.4), sharey=True, squeeze=False)
    for ax, row in zip(axes[0], first):
        angles = np.asarray(row["angles"])
        t = row["dt"] * np.arange(1, len(angles) + 1)
        for j, name in enumerate(JOINTS):
            ax.plot(t, angles[:, j], lw=1.1, label=name)
        name = "policy alone" if row["alpha"] is None else f"MPPI, alpha = {row['alpha']:g}"
        note = (f"regularity {row['regularity']:.2f}, to policy {row['distance_to_policy']:.2f}" if row["survived"]
                else f"fell at {row['steps'] * row['dt']:.1f} s")
        ax.set_title(f"{name}\n{note}", fontsize=10)
        ax.set_xlim(0, steps * row["dt"])
        ax.set_xlabel("time (s)")
    axes[0][0].set_ylabel("joint angle (rad)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "matching_check.png", dpi=150)
    with (out_dir / "episodes.jsonl").open("w") as stream:
        for r in rows:
            stream.write(json.dumps({k: v for k, v in r.items() if k != "angles"}) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"policy": policy, "resets": [reset_seed, reset_seed + resets - 1],
                                                      "K": K, "H": H, "steps": steps, "summary": summary}, indent=2))
    for row in summary:
        print(json.dumps(row))


if __name__ == "__main__":
    tyro.cli(main)
