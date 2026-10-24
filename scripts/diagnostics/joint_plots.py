"""Joint-angle trajectories of MPPI and of learned policies, after Mordatch &
Todorov (2014), Figure 3: under MPC the joints wander, under a good policy they
cycle.

Each panel runs one controller from the same reset for `steps` controls and
plots the six actuated joint angles. Regularity is the mean over joints of the
angle's autocorrelation at its dominant period between 0.4 and 1.5 s, after
the first second: 1 for a gait that repeats exactly. Distance to the first
controller is the mean over joints of the Wasserstein-1 distance (rad) between
the two angle distributions after the first second: how far a policy's posture
and swing are from the MPPI it should be distilled from.

controllers: "label=mppi:K:H" or "label=mppi:K:H:planner_seed" for raw MPPI
(frozen config otherwise), or "label=policy:path" for a checkpoint acting
deterministically.
"""
import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
JOINTS = ("right thigh", "right leg", "right foot", "left thigh", "left leg", "left foot")


def regularity(angles: np.ndarray, dt: float, skip: float = 1.0, periods: tuple = (0.4, 1.5)) -> float:
    x = angles[int(round(skip / dt)):]
    x = x - x.mean(axis=0)
    lo, hi = int(round(periods[0] / dt)), int(round(periods[1] / dt))
    if len(x) < 2 * hi:
        return float("nan")
    scores = []
    for v in x.T:
        lags = [float(np.dot(v[:-lag], v[lag:]) / np.sqrt(np.dot(v[:-lag], v[:-lag]) * np.dot(v[lag:], v[lag:])))
                for lag in range(lo, hi + 1)]
        scores.append(max(lags))
    return float(np.mean(scores))


def distance(angles: np.ndarray, reference: np.ndarray, dt: float, skip: float = 1.0) -> float:
    from scipy.stats import wasserstein_distance

    a, b = angles[int(round(skip / dt)):], reference[int(round(skip / dt)):]
    if not len(a) or not len(b):
        return float("nan")
    return float(np.mean([wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])]))


def _run(job: dict) -> dict:
    import torch

    from src.envs.walker2d import Walker2d
    from src.mppi.mppi import MPPI
    from src.rl.sac import load_policy
    from src.utils.config import MPPIConfig

    torch.set_num_threads(1)
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    np.random.seed(job["reset_seed"])
    obs = env.reset()
    kind, _, rest = job["spec"].partition(":")
    if kind == "mppi":
        K, H, *seed = (int(n) for n in rest.split(":"))
        base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
        planner = MPPI(env, replace(base, K=K, H=H), seed=seed[0] if seed else job["planner_seed"])
        act = lambda _: planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())[0]
    elif kind == "policy":
        payload = torch.load(_REPO_ROOT / rest, map_location="cpu", weights_only=False)
        actor = load_policy(payload.get("policy", payload), device="cpu")
        act = lambda o: actor.act(o, deterministic=True)
    else:
        raise ValueError(f"unknown controller {job['spec']!r}")
    angles, velocities, fell = [], [], False
    for _ in range(job["steps"]):
        obs, _, done, info = env.step(act(obs))
        angles.append(env.data.qpos[3:9].copy())
        velocities.append(info["x_velocity"])
        if done:
            fell = True
            break
    angles = np.asarray(angles)
    return {"label": job["label"], "spec": job["spec"], "angles": angles.tolist(), "fell": fell,
            "seconds": len(angles) * env._dt, "mean_vx": float(np.mean(velocities)),
            "regularity": regularity(angles, env._dt), "dt": env._dt}


def main(
    out_dir: Path,
    controllers: tuple[str, ...],
    reset_seed: int = 81_000,
    planner_seed: int = 95_000,
    steps: int = 500,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    jobs = [{"label": label, "spec": spec, "reset_seed": reset_seed, "planner_seed": planner_seed, "steps": steps}
            for label, spec in (entry.rsplit("=", 1) for entry in controllers)]
    out_dir.mkdir(parents=True, exist_ok=False)
    with Pool(min(len(jobs), 8)) as pool:
        rows = pool.map(_run, jobs)
    reference = np.asarray(rows[0]["angles"])
    for row in rows:
        row["distance_to_first"] = distance(np.asarray(row["angles"]), reference, row["dt"])
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.4), sharey=True, squeeze=False)
    for ax, row in zip(axes[0], rows):
        angles = np.asarray(row["angles"])
        t = row["dt"] * np.arange(1, len(angles) + 1)
        for j, name in enumerate(JOINTS):
            ax.plot(t, angles[:, j], lw=1.1, label=name)
        note = f"fell at {row['seconds']:.1f} s" if row["fell"] else f"regularity {row['regularity']:.2f}"
        ax.set_title(f"{row['label']}\n{note}, {row['mean_vx']:.2f} m/s, "
                     f"distance {row['distance_to_first']:.2f}", fontsize=10)
        ax.set_xlim(0, steps * row["dt"])
        ax.set_xlabel("time (s)")
    axes[0][0].set_ylabel("joint angle (rad)")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "joint_plots.png", dpi=150)
    summary = [{k: row[k] for k in ("label", "spec", "fell", "seconds", "mean_vx", "regularity", "distance_to_first")}
               for row in rows]
    (out_dir / "summary.json").write_text(json.dumps({"reset_seed": reset_seed, "planner_seed": planner_seed,
                                                      "steps": steps, "controllers": summary}, indent=2))
    for row in summary:
        print(json.dumps(row))


if __name__ == "__main__":
    tyro.cli(main)
