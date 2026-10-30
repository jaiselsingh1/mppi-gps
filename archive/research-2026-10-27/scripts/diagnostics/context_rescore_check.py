"""Step 18: does a policy's cost-to-go, attached after MPPI's search, pick better actions?

MPPI (frozen config except K and H) plans with its own warm start and its own
samples; nothing from a policy enters its search. After each search, every
final sample is scored as "this sample's H controls, then the context policy
for tail_steps controls" (src/gps/policy_tail.py: Step 4's tail, run after the
search instead of as MPPI's terminal cost). The executed action is the first
action of those samples re-weighted by that total, under one of two rules:
"temperature" keeps MPPI's temperature (the weights MPPI would give these
samples if it could see the context); "neff" keeps MPPI's own effective sample
size at each decision, so only the context's ranking of the samples differs.
MPPI's own update, and so its warm start, still uses its own scores only.
With no context an episode is raw MPPI, as in walker_headroom.py.
Completion means no fall in 1,000 controls; speed MAE averages |1.5 - vx| over
executed controls, failures included.
"""
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tyro

from src.utils.math import softmin_weights, temperature_for_neff_fraction
from src.utils.stats import paired_bootstrap_mean_difference, paired_proportion_difference, wilson_interval

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = {
    "none": None,
    "sac_800k": _REPO_ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_0800000.pt",
    "step15_student": _REPO_ROOT / "runs/2026-10-23_step15_teacher_seed0/policy_032.pt",
}
RULES = ("neff", "temperature")


def rescored_action(rollout: dict, tail: np.ndarray, lam: float, rule: str = "temperature",
                    own_neff: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """First action of MPPI's final samples re-weighted by their score plus a tail cost.

    rule "temperature" keeps MPPI's temperature lam; "neff" solves for the
    temperature that keeps MPPI's own effective sample size own_neff.
    """
    scores = rollout["costs"] + tail
    if rule == "neff":
        lam = temperature_for_neff_fraction(scores, min(own_neff / len(scores), 1.0 - 1e-9))
    elif rule != "temperature":
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}.")
    weights = softmin_weights(scores, lam)
    return np.einsum("k,ka->a", weights, rollout["actions"][:, 0]), weights


def _episode(job: dict) -> dict:
    import torch

    from src.envs.walker2d import Walker2d
    from src.gps.policy_tail import WalkerPolicyTail
    from src.mppi.mppi import MPPI
    from src.rl.sac import load_policy
    from src.utils.config import MPPIConfig
    from src.utils.math import effective_sample_size

    torch.set_num_threads(1)
    task = json.loads((_REPO_ROOT / "configs/walker2d_task.json").read_text())
    env = Walker2d(**task, nthread=1)
    base = MPPIConfig(**json.loads((_REPO_ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=job["K"], H=job["H"])
    tail = None
    if job["context_path"] is not None:
        payload = torch.load(job["context_path"], map_location="cpu", weights_only=False)
        actor = load_policy(payload.get("policy", payload), device="cpu")

        @torch.no_grad()
        def act_batch(obs: np.ndarray) -> np.ndarray:
            mean, _ = actor(torch.as_tensor(obs, dtype=torch.float32))
            return (torch.tanh(mean) * actor.action_scale + actor.action_bias).numpy().astype(float)

        tail = WalkerPolicyTail(env, act_batch, job["tail_steps"])
    np.random.seed(job["reset_seed"])
    env.reset()
    planner = MPPI(env, cfg, seed=job["planner_seed"])
    velocities, plan_seconds, own_neff, used_neff, shift = [], [], [], [], []
    survived = True
    for _ in range(job["steps"]):
        started = time.perf_counter()
        action, info = planner.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
        own_neff.append(info["n_eff"])
        if tail is not None:
            rollout = planner.get_rollout_data()
            chosen, weights = rescored_action(rollout, tail(rollout["states"], rollout["actions"]), info["lam"],
                                              job["rule"], info["n_eff"])
            used_neff.append(effective_sample_size(weights))
            shift.append(float(np.linalg.norm(chosen - action)))
            action = chosen
        plan_seconds.append(time.perf_counter() - started)
        _, _, done, step_info = env.step(action)
        velocities.append(step_info["x_velocity"])
        if done:
            survived = False
            break
    vx = np.asarray(velocities)
    return {
        **{key: job[key] for key in ("K", "H", "context", "rule", "tail_steps", "reset_seed", "planner_seed")},
        "steps": len(vx),
        "survived": survived,
        "mean_vx": float(vx.mean()),
        "velocity_mae": float(np.mean(np.abs(env._v_target - vx))),
        "median_own_neff": float(np.median(own_neff)),
        "median_used_neff": float(np.median(used_neff)) if used_neff else None,
        "median_action_shift": float(np.median(shift)) if shift else 0.0,
        "mean_plan_seconds": float(np.mean(plan_seconds)),
    }


def _summary(rows: list[dict]) -> dict:
    survived = np.array([r["survived"] for r in rows])
    low, high = wilson_interval(int(survived.sum()), len(rows))
    used = [r["median_used_neff"] for r in rows if r["median_used_neff"] is not None]
    return {"episodes": len(rows), "completed": int(survived.sum()), "completion_wilson95": [low, high],
            "velocity_mae": float(np.mean([r["velocity_mae"] for r in rows])),
            "mean_seconds_alive": float(np.mean([r["steps"] for r in rows]) * 0.008),
            "median_own_neff": float(np.median([r["median_own_neff"] for r in rows])),
            "median_used_neff": float(np.median(used)) if used else None,
            "median_action_shift": float(np.median([r["median_action_shift"] for r in rows])),
            "mean_plan_seconds": float(np.mean([r["mean_plan_seconds"] for r in rows]))}


def _paired(candidate: list[dict], reference: list[dict]) -> dict:
    if [r["reset_seed"] for r in candidate] != [r["reset_seed"] for r in reference]:
        raise ValueError("paired arms must share reset seeds")
    field = lambda rows, key: np.array([r[key] for r in rows], dtype=float)
    survived = lambda rows: np.array([r["survived"] for r in rows]).astype(int)
    return {"completion_difference": paired_proportion_difference(survived(candidate), survived(reference)),
            "velocity_mae_difference": paired_bootstrap_mean_difference(field(candidate, "velocity_mae"),
                                                                        field(reference, "velocity_mae")),
            "seconds_alive_difference": paired_bootstrap_mean_difference(0.008 * field(candidate, "steps"),
                                                                         0.008 * field(reference, "steps"))}


def _arm_name(h: int, context: str, rule: str | None) -> str:
    return f"H{h}_none" if context == "none" else f"H{h}_{context}_{rule}"


def report(rows: list[dict]) -> dict:
    """Per-arm summaries and paired differences.

    Each context arm is paired with no context at its own and at the longest
    horizon, with the other context under the same rule, and with itself under
    the other rule.
    """
    arms = {}
    for row in sorted(rows, key=lambda r: r["reset_seed"]):
        arms.setdefault((row["H"], row["context"], row["rule"]), []).append(row)
    order = sorted(arms, key=lambda k: (k[0], k[1], k[2] or ""))
    longest = max(h for h, _, _ in arms)
    contexts = sorted({c for _, c, _ in arms if c != "none"})
    out = {"arms": {_arm_name(*k): _summary(arms[k]) for k in order}, "paired": {}}

    def pair(a: tuple, b: tuple) -> None:
        if a in arms and b in arms:
            out["paired"][f"{_arm_name(*a)}:{_arm_name(*b)}"] = _paired(arms[a], arms[b])

    for h, context, rule in order:
        if context == "none":
            continue
        pair((h, context, rule), (h, "none", None))
        if h != longest:
            pair((h, context, rule), (longest, "none", None))
        for other in contexts:
            if other > context:
                pair((h, context, rule), (h, other, rule))
        if rule == "neff":
            pair((h, context, "neff"), (h, context, "temperature"))
    return out


def main(
    out_dir: Path,
    horizons: tuple[int, ...] = (16, 32, 48),
    contexts: tuple[str, ...] = ("none", "sac_800k", "step15_student"),
    rules: tuple[str, ...] = RULES,
    K: int = 32,
    tail_steps: int = 100,
    episodes: int = 50,
    steps: int = 1000,
    reset_seed: int = 70_000,
    planner_seed: int = 95_000,
    workers: int = max(1, (os.cpu_count() or 2) - 2),
) -> None:
    unknown = set(contexts) - set(CONTEXTS)
    if unknown or "none" not in contexts:
        raise ValueError(f"contexts must include 'none' and come from {sorted(CONTEXTS)}; got {contexts}.")
    if not rules or set(rules) - set(RULES):
        raise ValueError(f"rules must come from {RULES}; got {rules}.")
    missing = [str(CONTEXTS[c]) for c in contexts if CONTEXTS[c] is not None and not CONTEXTS[c].exists()]
    if missing:
        raise FileNotFoundError(f"missing context checkpoints: {missing}")
    out_dir.mkdir(parents=True, exist_ok=False)
    arms = [(c, None) if CONTEXTS[c] is None else (c, r) for c in contexts for r in rules]
    jobs = [{"K": K, "H": h, "context": c, "rule": r,
             "context_path": None if CONTEXTS[c] is None else str(CONTEXTS[c]),
             "tail_steps": 0 if CONTEXTS[c] is None else tail_steps, "steps": steps,
             "reset_seed": reset_seed + e, "planner_seed": planner_seed + e}
            for h in horizons for c, r in dict.fromkeys(arms) for e in range(episodes)]
    # Longest jobs first keeps the pool busy.
    jobs.sort(key=lambda j: -(j["H"] + j["tail_steps"]))
    started = time.time()
    rows = []
    with Pool(workers) as pool:
        for row in pool.imap_unordered(_episode, jobs):
            rows.append(row)
            with (out_dir / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    result = {"held_out_reset_seeds": [reset_seed, reset_seed + episodes - 1], "K": K, "tail_steps": tail_steps,
              "rules": list(rules),
              "contexts": {c: None if CONTEXTS[c] is None else str(CONTEXTS[c].relative_to(_REPO_ROOT))
                           for c in contexts},
              "workers": workers, "wall_time_s": time.time() - started, **report(rows)}
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    for name, arm in result["arms"].items():
        print(name, json.dumps(arm))


if __name__ == "__main__":
    tyro.cli(main)
