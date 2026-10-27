"""Step 34: Stage 1 of docs/trust_priced_consensus.md, the trust map and the channel test. No training.

Phase A: plain K256 MPPI (configs/walker2d_best.json) from fresh resets. Records the
states and warm starts at the probe times. Also Stage 1b's shared baseline arm.

Phase B (1a, trust map): for each Selective seed-0 checkpoint:
- its own deterministic episodes from the same resets;
- the policy probed alone for `window` controls from MPPI's states and from its own states, at the probe times;
- trust fields cross-fitted by reset parity, reporting held-out AUC, Brier score and the trusted share of MPPI's states.

Phase C (1b, channel test): K256 MPPI with an earned channel toward a learner, on the
same resets and planner seeds as Phase A. Channels: c1 (matching priced by trust), c3
(handover terminal), and c1+c3. Measures:
- held-out state-only R^2 of the executed actions (Step 33's k-NN estimator);
- walking and speed error;
- |delta| = |a - pi(s)|;
- trusted share along MPPI's path;
- compute.

Every job writes its own file, so a rerun into the same directory resumes.
"""
import json
import pickle
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class StageConfig:
    selective_run: str = "runs/2026-10-24_policy_selective_75_seed0_001/selective"
    checkpoints: tuple[int, ...] = (9, 19, 29, 74)
    learners: tuple[int, ...] = (29, 74)
    alphas: tuple[float, ...] = (0.3, 3.0)
    episodes: int = 24
    steps: int = 1000
    reset_seed: int = 58_000_000
    planner_seed: int = 889_000_000
    probe_first: int = 25
    probe_every: int = 50
    window: int = 250
    max_speed_error: float = 0.30
    trust_k: int = 20
    trust_z: float = 1.0
    familiar_quantile: float = 0.95
    trusted_threshold: float = 0.8
    candidates: int = 16
    ks: tuple[int, ...] = (5, 10, 20, 50, 100)
    folds: int = 3
    walking_from: int = 96
    workers: int = 28


def _probe_times(cfg: StageConfig) -> range:
    return range(cfg.probe_first, cfg.steps, cfg.probe_every)


def _load_act(cfg: StageConfig, checkpoint: int):
    import torch

    from scripts.training.train_walker_mdgps import _batch_act
    from src.rl.sac import load_policy

    torch.set_num_threads(1)
    path = _REPO_ROOT / cfg.selective_run / f"policy_{checkpoint:03d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    actor = load_policy(payload.get("policy", payload), device="cpu")
    actor.eval()
    return _batch_act(actor)


def _env_and_reset(cfg: StageConfig, episode: int):
    from scripts.training.train_walker_mdgps import _walker

    env = _walker()
    np.random.seed(cfg.reset_seed + episode)
    return env, env.reset()


def _trust_field(cfg: StageConfig, obs: np.ndarray, labels: np.ndarray):
    from src.gps.trust import TrustField

    return TrustField(obs, labels, cfg.trust_k, cfg.trust_z, cfg.familiar_quantile)


def mppi_episode(job: dict) -> dict:
    """K256 MPPI from one reset, optionally with an earned channel toward a learner."""
    from src.gps.handover import HandoverTerminal, TrustPricedMatching
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    cfg, episode, start = job["cfg"], job["episode"], time.time()
    env, obs = _env_and_reset(cfg, episode)
    planner_cfg = MPPIConfig.load("walker2d")
    if (planner_cfg.K, planner_cfg.H) != (256, 48):
        raise RuntimeError(f"expected the tuned K256 H48 teacher, got K{planner_cfg.K} H{planner_cfg.H}")
    planner = MPPI(env, planner_cfg, seed=cfg.planner_seed + episode)
    act = trust = matching = handover = None
    if job.get("learner") is not None:
        act = _load_act(cfg, job["learner"])
        trust = _trust_field(cfg, job["trust_obs"], job["trust_labels"])
        if job.get("alpha"):
            matching = TrustPricedMatching(env, act, trust, job["alpha"])
        if job.get("handover"):
            handover = HandoverTerminal(env, act, trust, cfg.window, cfg.candidates)
    probe_times = set(_probe_times(cfg))
    record = {k: [] for k in ("obs", "action", "speed_error", "tau", "policy_action")}
    probes, passes, done = [], 0, False
    for t in range(cfg.steps):
        state, warm = env.get_state(), env.get_warmstart()
        if t in probe_times:
            probes.append({"t": t, "state": state.copy(), "warmstart": warm.copy()})
        prior = matching.for_decision(state, obs) if matching is not None else None
        action, info = planner.plan_step(state, prior_cost=prior, coupling=handover, initial_warmstart=warm)
        passes += int(info["optimization_iterations"])
        record["obs"].append(obs.copy())
        record["action"].append(action.copy())
        if act is not None:
            record["tau"].append(trust(obs))
            record["policy_action"].append(act(obs))
        obs, _, done, step_info = env.step(action)
        record["speed_error"].append(step_info["velocity_error"])
        if done:
            break
    out = {k: np.asarray(v) for k, v in record.items()}
    return out | {
        "episode": episode, "arm": job["arm"], "survived": not done, "steps": len(out["action"]), "probes": probes,
        "planner_controls": passes * planner_cfg.K * planner_cfg.H,
        "tail_controls": handover.tail_controls if handover is not None else 0,
        "seconds": time.time() - start,
    }


def own_episode(job: dict) -> dict:
    """A checkpoint alone and deterministic from one reset, then probed from its own states."""
    from src.gps.trust import probe

    cfg, episode, start = job["cfg"], job["episode"], time.time()
    act = _load_act(cfg, job["checkpoint"])
    env, obs = _env_and_reset(cfg, episode)
    low, high = env.action_bounds
    probe_times, states, errors, done = set(_probe_times(cfg)), [], [], False
    for t in range(cfg.steps):
        if t in probe_times:
            states.append({"t": t, "state": env.get_state(), "warmstart": env.get_warmstart()})
        obs, _, done, info = env.step(np.clip(act(obs), low, high))
        errors.append(info["velocity_error"])
        if done:
            break
    results = [probe(env, act, s["state"], s["warmstart"], cfg.window) for s in states]
    return {"checkpoint": job["checkpoint"], "episode": episode, "survived": not done, "steps": len(errors),
            "speed_mae": float(np.mean(errors)), "probes": _probe_rows(env, states, results, "own"),
            "probe_controls": sum(r.steps for r in results), "seconds": time.time() - start}


def mppi_state_probes(job: dict) -> dict:
    """A checkpoint probed alone from one MPPI episode's states at the probe times."""
    from src.gps.trust import probe
    from scripts.training.train_walker_mdgps import _walker

    cfg, start = job["cfg"], time.time()
    act, env = _load_act(cfg, job["checkpoint"]), _walker()
    results = [probe(env, act, s["state"], s["warmstart"], cfg.window) for s in job["states"]]
    return {"checkpoint": job["checkpoint"], "episode": job["episode"],
            "probes": _probe_rows(env, job["states"], results, "mppi"),
            "probe_controls": sum(r.steps for r in results), "seconds": time.time() - start}


def _probe_rows(env, states: list[dict], results, source: str) -> list[dict]:
    rows = []
    for s, r in zip(states, results):
        obs = env.reset(state=s["state"])
        rows.append({"t": s["t"], "obs": obs, "source": source, "survived": r.survived,
                     "steps": r.steps, "speed_error": r.speed_error})
    return rows


def _call(item: tuple[str, int, dict]) -> tuple[int, dict]:
    name, index, job = item
    return index, globals()[name](job)


def _run(fn, jobs: list[dict], paths: list[Path], workers: int) -> list[dict]:
    """Run the jobs whose output file is missing, then load every job's output.

    Each result is written as soon as it finishes, so a rerun resumes.
    """
    todo = [(job, path) for job, path in zip(jobs, paths) if not path.exists()]
    if todo:
        with Pool(min(workers, len(todo))) as pool:
            items = [(fn.__name__, i, job) for i, (job, _) in enumerate(todo)]
            for index, result in pool.imap_unordered(_call, items):
                path = todo[index][1]
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(pickle.dumps(result))
                tmp.replace(path)
    return [pickle.loads(path.read_bytes()) for path in paths]


def _arms(cfg: StageConfig) -> list[dict]:
    arms = []
    for learner in cfg.learners:
        arms += [{"arm": f"L{learner}_c1_a{a:g}", "learner": learner, "alpha": a, "handover": False}
                 for a in cfg.alphas]
        arms.append({"arm": f"L{learner}_c3", "learner": learner, "alpha": 0.0, "handover": True})
        arms.append({"arm": f"L{learner}_c1c3_a{max(cfg.alphas):g}", "learner": learner,
                     "alpha": max(cfg.alphas), "handover": True})
    return arms


def _labels(rows: list[dict], cfg: StageConfig, survival_only: bool) -> np.ndarray:
    limit = None if survival_only else cfg.max_speed_error
    return np.array([r["survived"] and (limit is None or r["speed_error"] <= limit) for r in rows])


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN without both classes."""
    from scipy.stats import rankdata

    label = np.asarray(label, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def trust_map(cfg: StageConfig, rows: list[dict], survival_only: bool) -> dict:
    """Cross-fitted by reset parity: fit on one parity's probes, score the other's."""
    labels = _labels(rows, cfg, survival_only)
    obs = np.array([r["obs"] for r in rows])
    parity = np.array([r["episode"] % 2 for r in rows])
    tau = np.full(len(rows), np.nan)
    for held in (0, 1):
        train, test = parity != held, parity == held
        if train.sum() > cfg.trust_k and test.any():
            tau[test] = _trust_field(cfg, obs[train], labels[train])(obs[test])
    scored = np.isfinite(tau)
    mppi = scored & np.array([r["source"] == "mppi" for r in rows])
    return {
        "probes": len(rows), "trusted_rate_mppi": float(labels[[r["source"] == "mppi" for r in rows]].mean()),
        "trusted_rate_own": (float(labels[[r["source"] == "own" for r in rows]].mean())
                             if any(r["source"] == "own" for r in rows) else float("nan")),
        "auc": auc(tau[scored], labels[scored]),
        "brier": float(np.mean((tau[scored] - labels[scored]) ** 2)) if scored.any() else float("nan"),
        "trusted_share_mppi": float(np.mean(tau[mppi] >= cfg.trusted_threshold)) if mppi.any() else float("nan"),
    }


def held_out_r2(episodes: list[dict], cfg: StageConfig) -> dict:
    """Step 33's estimator: contiguous episode folds, k-NN on standardized observations."""
    from scripts.diagnostics.label_floor_knn import knn_errors

    folds = [[e for e in episodes if e["episode"] * cfg.folds // cfg.episodes == f] for f in range(cfg.folds)]
    cat = lambda group, key: np.concatenate([e[key] for e in group])
    natural, walking = {k: [] for k in cfg.ks}, {k: [] for k in cfg.ks}
    for f in range(cfg.folds):
        train = [e for g in range(cfg.folds) if g != f for e in folds[g]]
        x, y = cat(train, "obs"), cat(train, "action")
        tx, ty = cat(folds[f], "obs"), cat(folds[f], "action")
        tt = np.concatenate([np.arange(len(e["action"])) for e in folds[f]])
        mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
        errors, _, _ = knn_errors((x - mean) / std, y, (tx - mean) / std, ty, cfg.ks)
        for keep, store in ((np.ones(len(ty), bool), natural), (tt >= cfg.walking_from, walking)):
            variance = float(ty[keep].var(axis=0).mean())
            for k in cfg.ks:
                store[k].append(1.0 - float(errors[k][keep].mean()) / variance)
    natural = {k: float(np.mean(v)) for k, v in natural.items()}
    walking = {k: float(np.mean(v)) for k, v in walking.items()}
    best = max(natural, key=natural.get)
    return {"r2": natural[best], "r2_walking": walking[best], "best_k": best, "r2_by_k": natural}


def arm_summary(episodes: list[dict], cfg: StageConfig) -> dict:
    controls = sum(e["steps"] for e in episodes)
    return {
        "survived": int(sum(e["survived"] for e in episodes)), "episodes": len(episodes),
        "speed_mae": float(np.mean([e["speed_error"].mean() for e in episodes])),
        **held_out_r2(episodes, cfg),
        "planner_controls_per_control": sum(e["planner_controls"] for e in episodes) / controls,
        "tail_controls_per_control": sum(e["tail_controls"] for e in episodes) / controls,
        "worker_seconds": float(sum(e["seconds"] for e in episodes)),
    }


def path_trust(episodes: list[dict], field, cfg: StageConfig) -> float:
    """Share of decisions along MPPI's executed path where tau_hat >= the trusted threshold."""
    tau = np.concatenate([np.atleast_1d(field(e["obs"])) for e in episodes])
    return float(np.mean(tau >= cfg.trusted_threshold))


def main(out_dir: Path, cfg: StageConfig = StageConfig(), phases: str = "ABC") -> None:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("phaseA", "phaseB", "phaseC"):
        (out_dir / sub).mkdir(exist_ok=True)
    config_path = out_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["cfg"] != json.loads(json.dumps(asdict(cfg))):
        raise RuntimeError("config.json differs from this configuration; use a new directory.")
    config_path.write_text(json.dumps({"cfg": asdict(cfg), "phases": phases}, indent=2))
    summary: dict = {}

    plain = _run(mppi_episode, [{"cfg": cfg, "episode": i, "arm": "plain"} for i in range(cfg.episodes)],
                 [out_dir / "phaseA" / f"plain_{i:02d}.pkl" for i in range(cfg.episodes)], cfg.workers)
    summary["plain"] = arm_summary(plain, cfg)
    if "B" not in phases:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return

    own_jobs, own_paths, state_jobs, state_paths = [], [], [], []
    for c in cfg.checkpoints:
        for e in plain:
            own_jobs.append({"cfg": cfg, "checkpoint": c, "episode": e["episode"]})
            own_paths.append(out_dir / "phaseB" / f"own_{c:03d}_{e['episode']:02d}.pkl")
            state_jobs.append({"cfg": cfg, "checkpoint": c, "episode": e["episode"], "states": e["probes"]})
            state_paths.append(out_dir / "phaseB" / f"mppi_{c:03d}_{e['episode']:02d}.pkl")
    own = _run(own_episode, own_jobs, own_paths, cfg.workers)
    states = _run(mppi_state_probes, state_jobs, state_paths, cfg.workers)
    rows = {c: [r | {"episode": job["episode"]} for job, out in zip(own_jobs + state_jobs, own + states)
                if job["checkpoint"] == c for r in out["probes"]] for c in cfg.checkpoints}
    maps = {c: {"mae": trust_map(cfg, rows[c], False), "survival_only": trust_map(cfg, rows[c], True),
                "own_survived": int(sum(o["survived"] for o in own if o["checkpoint"] == c)),
                "own_mean_steps": float(np.mean([o["steps"] for o in own if o["checkpoint"] == c]))}
            for c in cfg.checkpoints}
    survival_only = maps[29]["mae"]["trusted_share_mppi"] < 0.05 if 29 in maps else False
    summary["trust_map"] = {str(c): m for c, m in maps.items()}
    summary["stage2_uses_survival_only"] = bool(survival_only)
    summary["probe_controls"] = int(sum(o["probe_controls"] for o in own + states))
    if "C" not in phases:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return

    fields = {L: (np.array([r["obs"] for r in rows[L]]), _labels(rows[L], cfg, survival_only)) for L in cfg.learners}
    arm_jobs, arm_paths = [], []
    for arm in _arms(cfg):
        obs, labels = fields[arm["learner"]]
        for i in range(cfg.episodes):
            arm_jobs.append(arm | {"cfg": cfg, "episode": i, "trust_obs": obs, "trust_labels": labels})
            arm_paths.append(out_dir / "phaseC" / f"{arm['arm']}_{i:02d}.pkl")
    results = _run(mppi_episode, arm_jobs, arm_paths, cfg.workers)
    arms: dict = {}
    for arm in _arms(cfg):
        episodes = [r for r in results if r["arm"] == arm["arm"]]
        field = _trust_field(cfg, *fields[arm["learner"]])
        arms[arm["arm"]] = arm_summary(episodes, cfg) | {
            "learner": arm["learner"],
            "delta": float(np.mean(np.concatenate([np.linalg.norm(e["action"] - e["policy_action"], axis=1)
                                                   for e in episodes]))),
            "path_trust": path_trust(episodes, field, cfg),
            "plain_path_trust": path_trust(plain, field, cfg),
        }
    summary["arms"] = arms
    summary["decisions"] = decisions(cfg, summary["plain"], arms)
    summary["wall_seconds_this_invocation"] = time.time() - start
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def decisions(cfg: StageConfig, plain: dict, arms: dict) -> dict:
    """The preregistered rules (docs/experiment_log.md, Step 34)."""
    plain_falls = cfg.episodes - plain["survived"]
    by_channel: dict = {}
    for name, a in arms.items():
        channel = name.split("_", 1)[1]
        by_channel.setdefault(channel, []).append(a)
    joins = [channel for channel, rows in by_channel.items()
             if any(a["path_trust"] - a["plain_path_trust"] >= 0.10 for a in rows)
             and all(cfg.episodes - a["survived"] <= plain_falls + 2 for a in rows)]
    learnable = [name for name, a in arms.items() if a["learner"] == 29 and a["r2"] >= 0.30
                 and a["survived"] >= 20 and a["speed_mae"] <= 0.25]
    return {"mppi_actions_learnable_arms": learnable, "add_mppi_action_arm_to_stage2": bool(learnable),
            "channels_joining_stage2": joins}


if __name__ == "__main__":
    tyro.cli(main)
