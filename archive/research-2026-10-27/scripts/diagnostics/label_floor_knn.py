"""Diagnostic C: the label floor under natural histories, from nearby states.

Step 22 (diagnostic A) regenerated MPPI's warm start at fixed states, but a
replayed planner never chose the actions it plans around, so its plans can be
stale. Here every label comes from a planner that drove its own episode, as in
Step 15 (MPPI executes its action + N(0, exec_noise^2)), and the floor is
estimated from how labels differ between nearby states of different episodes.
Held-out episodes are disjoint from training episodes; observations are
standardized with the training mean and std.

- k-NN regression: held-out squared error per action dimension, for several k.
  At this data size the best k gives an upper estimate of the floor.
- Nugget: half the squared label difference to the nearest training state,
  regressed on that state's distance over the closest pairs and extrapolated
  to distance 0. It estimates Var(label | state) directly.
- The students' network (2 x hidden MLP, tanh-squashed, MSE, Adam, batch 256,
  lr 3e-4) fit to the training episodes: held-out squared error by number of
  training episodes, at the students' width and a wider one.
"""
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = ("executed", "smoothed")


@dataclass
class FloorConfig:
    episodes: int = 400
    held_out: int = 80
    K: int = 32
    H: int = 48
    steps: int = 400
    exec_noise: float = 0.1
    label_neff_fraction: float = 0.2
    ks: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200)
    # Fraction of held-out states, closest first, used to extrapolate the nugget.
    nugget_fraction: float = 0.25
    train_episodes: tuple[int, ...] = (40, 80, 160, 320)
    widths: tuple[int, ...] = (256, 1024)
    grad_steps: int = 20_000
    seed_base: int = 7_500_000
    bootstrap: int = 1000
    workers: int = max(1, (os.cpu_count() or 2) - 2)


def record_teacher(cfg: FloorConfig, episode: int) -> dict:
    """One Step 15 teacher episode with both labels at every decision."""
    from scripts.training.train_walker_mdgps import _reweighted_label, _walker
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    env = _walker()
    base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    planner = MPPI(env, replace(base, K=cfg.K, H=cfg.H), seed=cfg.seed_base + 100_000 + episode)
    rng = np.random.default_rng(cfg.seed_base + 200_000 + episode)
    low, high = env.action_bounds
    np.random.seed(cfg.seed_base + episode)
    obs = env.reset()
    record = {"obs": [], "executed": [], "smoothed": []}
    done = False
    for _ in range(cfg.steps):
        label, _ = planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
        for key, value in (("obs", obs), ("executed", label),
                           ("smoothed", _reweighted_label(planner.get_rollout_data(), cfg.label_neff_fraction))):
            record[key].append(np.copy(value))
        action = np.clip(label + cfg.exec_noise * rng.standard_normal(env.action_dim), low, high)
        obs, _, done, _ = env.step(action)
        if done:
            break
    return {key: np.asarray(values) for key, values in record.items()} | {"episode": episode, "fell": bool(done)}


def knn_errors(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
               ks) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Per-state held-out squared error (mean over action dims) of k-NN regression
    for each k, plus each test state's nearest distance and half squared label gap."""
    from scipy.spatial import cKDTree

    distance, index = cKDTree(train_x).query(test_x, k=max(ks), workers=-1)
    distance, index = distance.reshape(len(test_x), -1), index.reshape(len(test_x), -1)
    errors = {k: ((train_y[index[:, :k]].mean(axis=1) - test_y) ** 2).mean(axis=1) for k in ks}
    half_gap = 0.5 * ((train_y[index[:, 0]] - test_y) ** 2).mean(axis=1)
    return errors, distance[:, 0], half_gap


def nugget(distance: np.ndarray, half_gap: np.ndarray, fraction: float) -> float:
    """Intercept at distance 0 of half the squared label gap, fit over the closest pairs."""
    closest = np.argsort(distance)[:max(2, int(fraction * len(distance)))]
    slope, intercept = np.polyfit(distance[closest], half_gap[closest], 1)
    return float(intercept)


def fit_mlp(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray,
            width: int, steps: int, seed: int) -> dict:
    """The students' fit (scripts/training/train_walker_mdgps.py::_fit) for `steps` Adam steps."""
    import torch

    from scripts.training.train_walker_mdgps import _walker
    from src.rl.sac import SacActor

    torch.manual_seed(seed)
    actor = SacActor(train_x.shape[1], *_walker().action_bounds, width)
    params = [p for name, p in actor.named_parameters() if not name.startswith("fc_logstd")]
    optimizer = torch.optim.Adam(params, lr=3e-4)
    x, y = torch.as_tensor(train_x, dtype=torch.float32), torch.as_tensor(train_y, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    predict = lambda inputs: torch.tanh(actor(inputs)[0]) * actor.action_scale + actor.action_bias
    losses = []
    for _ in range(steps):
        index = torch.as_tensor(rng.integers(0, len(x), size=min(256, len(x))))
        loss = torch.mean((predict(x[index]) - y[index]) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        held_out = float(torch.mean((predict(torch.as_tensor(test_x, dtype=torch.float32))
                                     - torch.as_tensor(test_y, dtype=torch.float32)) ** 2))
    return {"train_loss": float(np.mean(losses[-100:])), "held_out": held_out}


def _mlp_job(job: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    return job["key"] | fit_mlp(job["train_x"], job["train_y"], job["test_x"], job["test_y"],
                                job["key"]["width"], job["steps"], job["seed"])


def main(out_dir: Path, cfg: FloorConfig) -> None:
    if not 0 < cfg.held_out < cfg.episodes or max(cfg.train_episodes) > cfg.episodes - cfg.held_out:
        raise ValueError("held_out and train_episodes must fit within episodes.")
    from scripts.diagnostics.label_variance_split import _provenance

    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg) | _provenance(), indent=2))
    started = time.time()
    with Pool(cfg.workers) as pool:
        recorded = pool.starmap(record_teacher, [(cfg, e) for e in range(cfg.episodes)])
    order = np.random.default_rng(cfg.seed_base + 300_000).permutation(cfg.episodes)
    test_eps, train_eps = order[:cfg.held_out], order[cfg.held_out:]
    stack = lambda eps, key: np.concatenate([recorded[e][key] for e in eps])
    episode_of = lambda eps: np.concatenate([[e] * len(recorded[e]["obs"]) for e in eps])
    # k-NN needs comparable scales; the MLP sees raw observations, as the students did.
    train_raw, test_raw = stack(train_eps, "obs"), stack(test_eps, "obs")
    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0) + 1e-8
    train_x, test_x = (train_raw - mean) / std, (test_raw - mean) / std
    test_episode = episode_of(test_eps)
    result = {"states": {"train": int(len(train_x)), "held_out": int(len(test_x))},
              "falls": int(sum(d["fell"] for d in recorded)), "labels": {}}
    rng = np.random.default_rng(cfg.seed_base + 400_000)
    groups = [np.flatnonzero(test_episode == e) for e in test_eps]
    resamples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                 for _ in range(cfg.bootstrap)]
    interval = lambda values: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
    jobs = []
    for label in LABELS:
        train_y, test_y = stack(train_eps, label), stack(test_eps, label)
        errors, distance, half_gap = knn_errors(train_x, train_y, test_x, test_y, cfg.ks)
        result["labels"][label] = {
            "label_variance": float(test_y.var(axis=0).mean()),
            "knn": {str(k): {"mean": float(e.mean()), "ci95": interval([e[i].mean() for i in resamples])}
                    for k, e in errors.items()},
            "nugget": {"mean": nugget(distance, half_gap, cfg.nugget_fraction),
                       "ci95": interval([nugget(distance[i], half_gap[i], cfg.nugget_fraction)
                                         for i in resamples])},
            "nearest_distance_median": float(np.median(distance)),
        }
        for n in cfg.train_episodes:
            subset = np.isin(episode_of(train_eps), train_eps[:n])
            for width in cfg.widths:
                jobs.append({"key": {"label": label, "train_episodes": n, "width": width},
                             "train_x": train_raw[subset], "train_y": train_y[subset], "test_x": test_raw,
                             "test_y": test_y, "steps": cfg.grad_steps, "seed": cfg.seed_base + 500_000})
    with Pool(cfg.workers) as pool:
        fits = pool.map(_mlp_job, jobs)
    for fit in fits:
        result["labels"][fit["label"]].setdefault("mlp", []).append(
            {key: fit[key] for key in ("train_episodes", "width", "train_loss", "held_out")})
    result["wall_time_s"] = time.time() - started
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out_dir / "raw.npz", **{f"{key}_{e}": recorded[e][key] for e in range(cfg.episodes)
                                                 for key in ("obs", *LABELS)})
    for label, summary in result["labels"].items():
        best = min(summary["knn"].items(), key=lambda item: item[1]["mean"])
        print(label, "variance", round(summary["label_variance"], 4), "best k-NN", best[0],
              round(best[1]["mean"], 4), "nugget", round(summary["nugget"]["mean"], 4))


if __name__ == "__main__":
    tyro.cli(main)
