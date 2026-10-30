"""Does a near-optimal policy know its own trajectories? No training.

Kendall's test of step 5: with a policy that is close to optimal and an MPPI
that also does the task, do their trajectories match, and can the policy tell
which trajectories it would not produce?

Trajectory sets, each from the same development resets:
- the policy alone, deterministic, and with N(0, sigma^2) added to the actions
  it executes (its own behaviour, perturbed);
- raw MPPI, and MPPI with the network-matching term toward the policy
  (src/gps/policy_matching.py) at each alpha.
Two scores per trajectory, higher meaning less like the policy's own:
- action gap: mean squared difference between the executed action and the
  policy's deterministic action at the same state;
- unfamiliarity: median distance from the trajectory's observations to the
  nearest observation in a reference set of the policy's own deterministic
  rollouts from other resets, after standardizing each dimension by the
  reference set's spread.
AUC(own, other) is the probability that a trajectory from `other` scores higher
than one from `own` (ties count half).
"""
import json
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
SAC_800K = "runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_0800000.pt"


def auc(own: np.ndarray, other: np.ndarray) -> float:
    own, other = np.asarray(own, dtype=float), np.asarray(other, dtype=float)
    greater = (other[:, None] > own[None, :]).mean()
    ties = (other[:, None] == own[None, :]).mean()
    return float(greater + 0.5 * ties)


def unfamiliarity(observations: np.ndarray, reference: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    scale = reference.std(axis=0) + 1e-8
    distances, _ = cKDTree(reference / scale).query(observations / scale)
    return float(np.median(distances))


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
    rng = np.random.default_rng(job["noise_seed"])
    low, high = env.action_bounds
    planner = matching = None
    if job["kind"] == "mppi":
        base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
        planner = MPPI(env, replace(base, K=job["K"], H=job["H"]), seed=job["planner_seed"])
        matching = PolicyMatching(env, act_batch, job["alpha"]) if job["alpha"] > 0 else None
    observations, gaps, survived = [], [], True
    for _ in range(job["steps"]):
        own = act_batch(obs[None])[0]
        if planner is None:
            action = np.clip(own + job["sigma"] * rng.standard_normal(own.shape), low, high)
        else:
            if matching is not None:
                matching.state = env.get_state()
            action, _ = planner.plan_step(env.get_state(), prior_cost=matching,
                                          initial_warmstart=env.get_warmstart())
        observations.append(obs.copy())
        gaps.append(float(np.mean((action - own) ** 2)))
        obs, _, done, _ = env.step(action)
        if done:
            survived = False
            break
    return {"set": job["set"], "reset_seed": job["reset_seed"], "steps": len(gaps), "survived": survived,
            "action_gap": float(np.mean(gaps)), "observations": np.asarray(observations)}


def main(
    out_dir: Path,
    policy: str = SAC_800K,
    sigmas: tuple[float, ...] = (0.1, 0.3, 0.5),
    alphas: tuple[float, ...] = (0.3, 1.0),
    resets: int = 10,
    reset_seed: int = 81_000,
    reference_resets: int = 20,
    reference_seed: int = 82_000,
    planner_seed: int = 95_000,
    K: int = 32,
    H: int = 48,
    steps: int = 1000,
    workers: int = 8,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = ([("policy", {"kind": "policy", "sigma": 0.0})]
            + [(f"policy, noise {s:g}", {"kind": "policy", "sigma": s}) for s in sigmas]
            + [("MPPI", {"kind": "mppi", "alpha": 0.0})]
            + [(f"MPPI, pulled {a:g}", {"kind": "mppi", "alpha": a}) for a in alphas])
    common = {"policy": policy, "K": K, "H": H, "steps": steps}
    jobs = [{**common, **spec, "set": name, "reset_seed": reset_seed + e, "planner_seed": planner_seed + e,
             "noise_seed": 970_000 + e} for e in range(resets) for name, spec in sets]
    jobs += [{**common, "kind": "policy", "sigma": 0.0, "set": "reference", "reset_seed": reference_seed + e,
              "planner_seed": 0, "noise_seed": 0} for e in range(reference_resets)]
    out_dir.mkdir(parents=True, exist_ok=False)
    jobs.sort(key=lambda j: j["kind"] != "mppi")
    with Pool(workers) as pool:
        rows = pool.map(_episode, jobs)
    reference = np.concatenate([r["observations"] for r in rows if r["set"] == "reference"])
    rows = [r for r in rows if r["set"] != "reference"]
    for r in rows:
        r["unfamiliarity"] = unfamiliarity(r["observations"], reference)
    names = [name for name, _ in sets]
    cells = {name: sorted((r for r in rows if r["set"] == name), key=lambda r: r["reset_seed"]) for name in names}
    own = cells["policy"] + cells[names[1]]
    summary = {name: {"completed": sum(r["survived"] for r in cell), "episodes": len(cell),
                      "median_action_gap": float(np.median([r["action_gap"] for r in cell])),
                      "median_unfamiliarity": float(np.median([r["unfamiliarity"] for r in cell]))}
               for name, cell in cells.items()}
    aucs = {score: {name: auc([r[score] for r in own], [r[score] for r in cells[name]])
                    for name in names[2:]} for score in ("action_gap", "unfamiliarity")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, score, label in zip(axes, ("action_gap", "unfamiliarity"),
                                ("action gap to the policy", "unfamiliarity of states")):
        for i, name in enumerate(names):
            values = [r[score] for r in cells[name]]
            ax.scatter(np.full(len(values), i) + np.linspace(-0.15, 0.15, len(values)), values, s=14)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "own_trajectories.png", dpi=150)
    with (out_dir / "episodes.jsonl").open("w") as stream:
        for r in rows:
            stream.write(json.dumps({k: v for k, v in r.items() if k != "observations"}) + "\n")
    result = {"policy": policy, "resets": [reset_seed, reset_seed + resets - 1],
              "reference_resets": [reference_seed, reference_seed + reference_resets - 1],
              "reference_states": int(len(reference)), "own_sets": names[:2], "summary": summary, "auc": aucs}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
