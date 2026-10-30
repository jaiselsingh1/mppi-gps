"""Bounded, descriptive Walker probe of an independent MPPI teacher.

This does not train a policy or calibrate a Q function. Raw MPPI's full-plan
scores and forced-first-action scores answer different questions and are
reported separately. Policies are loaded only after teacher collection.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import tyro

from src.mppi.mppi import MPPI
from src.rl.sac import load_policy
from src.rl.walker import DEFAULT_TASK, WalkerRL
from src.utils.config import MPPIConfig

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/2026-10-24_policy_selective_75_seed0_001"
RESERVED_RESETS = (5_000_000, 5_000_199)
HORIZON = 48
LONG_HORIZON = 1000


class InstrumentedMPPI(MPPI):
    """Observe each pass's incoming nominal without changing its computation."""

    def _plan_iteration(self, state, prior_cost, coupling, initial_warmstart):
        self.pass_nominals.append(self.U.copy())
        return super()._plan_iteration(state, prior_cost, coupling, initial_warmstart)

    def plan_step(self, state, **kwargs):
        if any(kwargs.get(key) is not None for key in ("nominal", "nominal_first", "prior_cost", "coupling")):
            raise ValueError("This independent teacher accepts no policy inputs.")
        self.pass_nominals = []
        return super().plan_step(state, **kwargs)


def teacher_config() -> MPPIConfig:
    base = MPPIConfig(**json.loads((ROOT / "configs/walker2d_best.json").read_text()))
    return replace(base, K=32, H=HORIZON, initial_iterations=2, lam_target_neff_fraction=None)


def validate_protocol(episodes: int, teacher_steps: int, reset_seed: int,
                      planner_seed: int, anchor_steps: tuple[int, ...]) -> None:
    if not 1 <= episodes <= 3 or not 1 <= teacher_steps <= 300:
        raise ValueError("This bounded probe permits 1..3 episodes and 1..300 teacher controls.")
    for name, seed in (("reset", reset_seed), ("planner", planner_seed)):
        if seed < 0 or seed + episodes > 2**32:
            raise ValueError(f"Invalid {name} seed range.")
    first, last = reset_seed, reset_seed + episodes - 1
    if first <= RESERVED_RESETS[1] and last >= RESERVED_RESETS[0]:
        raise ValueError("The final reserved reset panel must remain untouched.")
    if not anchor_steps or tuple(sorted(set(anchor_steps))) != anchor_steps or anchor_steps[0] < 0:
        raise ValueError("Anchor controls must be nonnegative, sorted and unique.")
    if len(anchor_steps) > 4:
        raise ValueError("This bounded probe permits at most four anchors per teacher episode.")
    if anchor_steps[-1] >= teacher_steps:
        raise ValueError("Every planned anchor must precede the collection cutoff.")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(initial: Path, final: Path) -> dict:
    paths = ("scripts/diagnostics/independent_teacher_probe.py", "src/mppi/mppi.py",
             "src/utils/math.py", "src/utils/config.py", "src/envs/base.py",
             "src/envs/mujoco_env.py", "src/envs/walker2d.py", "src/rl/walker.py",
             "src/rl/sac.py", "src/gps/policy.py", "assets/walker2d.xml",
             "configs/walker2d_task.json", "configs/walker2d_best.json")
    return {"python": platform.python_version(),
            "packages": {key: version(key) for key in ("mujoco", "numpy", "scipy", "torch")},
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
            "source_sha256": {key: sha256(ROOT / key) for key in paths},
            "checkpoints": {"initial": {"path": str(initial), "sha256": sha256(initial)},
                            "selective_074": {"path": str(final), "sha256": sha256(final)}}}


def collect_teacher(episodes: int, steps: int, reset_seed: int, planner_seed: int,
                    anchor_steps: tuple[int, ...], cfg: MPPIConfig) -> tuple[list[dict], list[dict], list[dict]]:
    """Collect raw MPPI only; absent anchors remain absent after the first fall."""
    snapshots, rows, missing = [], [], []
    for episode in range(episodes):
        env = WalkerRL(max_episode_steps=steps)
        try:
            obs, _ = env.reset(seed=reset_seed + episode)
            planner = InstrumentedMPPI(env.walker, cfg, seed=planner_seed + episode)
            velocities, costs = [], []
            terminated = truncated = False
            for t in range(steps):
                snapshot = None
                if t in anchor_steps:
                    snapshot = {"id": len(snapshots), "episode": episode, "teacher_t": t,
                                "reset_seed": reset_seed + episode, "planner_seed": planner_seed + episode,
                                "state": env.walker.get_state(), "warmstart": env.walker.get_warmstart(),
                                "obs": obs.copy(), "planner": planner.snapshot()}
                action, info = planner.plan_step(env.walker.get_state(),
                                                 initial_warmstart=env.walker.get_warmstart())
                if snapshot is not None:
                    snapshot["recorded_action"] = action.copy()
                    snapshot["recorded_sample_costs"] = planner.get_rollout_data()["costs"].copy()
                    snapshot["optimization_iterations"] = info["optimization_iterations"]
                    snapshots.append(snapshot)
                obs, _, terminated, truncated, step_info = env.step(action)
                velocities.append(step_info["x_velocity"])
                costs.append(step_info["task_cost"])
                if terminated or truncated:
                    break
            observed = {s["teacher_t"] for s in snapshots if s["episode"] == episode}
            missing.extend({"episode": episode, "teacher_t": t, "reason": "teacher_fell_before_anchor"}
                           for t in anchor_steps if t not in observed)
            rows.append({"episode": episode, "reset_seed": reset_seed + episode,
                         "planner_seed": planner_seed + episode, "executed_controls": len(costs),
                         "fell": bool(terminated), "healthy_at_collection_cutoff": not terminated,
                         "administrative_cutoff": bool(truncated), "executed_task_cost": sum(costs),
                         "mean_vx": float(np.mean(velocities)),
                         "velocity_mae": float(np.mean(np.abs(env.walker._v_target - np.asarray(velocities))))})
        finally:
            env.close()
    return snapshots, rows, missing


def save_snapshots(directory: Path, snapshots: list[dict]) -> None:
    """Numeric physics/plans in NPZ; lossless RNG state and indexing in JSON."""
    np.savez_compressed(directory / "snapshots.npz",
                        states=np.asarray([s["state"] for s in snapshots]),
                        warmstarts=np.asarray([s["warmstart"] for s in snapshots]),
                        observations=np.asarray([s["obs"] for s in snapshots]),
                        planner_U=np.asarray([s["planner"]["U"] for s in snapshots]),
                        initial_plan=np.asarray([s["planner"]["initial_plan"] for s in snapshots]),
                        recorded_actions=np.asarray([s["recorded_action"] for s in snapshots]),
                        recorded_sample_costs=np.asarray([s["recorded_sample_costs"] for s in snapshots]))
    write_json(directory / "snapshots.json", {
        "numeric_file": "snapshots.npz", "numeric_sha256": sha256(directory / "snapshots.npz"),
        "snapshots": [{key: s[key] for key in ("id", "episode", "teacher_t", "reset_seed", "planner_seed",
                                                    "optimization_iterations")} |
                      {"rng_state": s["planner"]["rng_state"]} for s in snapshots]})


def restore_snapshot(env: WalkerRL, snapshot: dict) -> np.ndarray:
    obs, _ = env.reset(options={"state": snapshot["state"], "warmstart": snapshot["warmstart"]})
    return obs


def query_teacher(env: WalkerRL, planner: InstrumentedMPPI, snapshot: dict) -> dict:
    restore_snapshot(env, snapshot)
    planner.restore(copy.deepcopy(snapshot["planner"]))
    action, info = planner.plan_step(snapshot["state"], initial_warmstart=snapshot["warmstart"])
    data = planner.get_rollout_data()
    return {"action": action.copy(), "mean_plan": planner.last_plan,
            "final_pass_nominal": planner.pass_nominals[-1].copy(),
            "pass_nominals": np.asarray(planner.pass_nominals),
            "sample_plans": data["actions"].copy(), "sample_costs": data["costs"].copy(),
            "sample_weights": data["weights"].copy(), "optimization_iterations": info["optimization_iterations"]}


def replay_difference(reference: dict, other: dict) -> dict:
    fields = ("action", "sample_costs", "sample_plans", "mean_plan", "final_pass_nominal", "sample_weights")
    differences = {f"max_abs_{key}_difference": float(np.max(np.abs(reference[key] - other[key])))
                   for key in fields}
    return {**differences, "exact": all(np.array_equal(reference[key], other[key]) for key in fields)}


def score_full_plans(env: WalkerRL, snapshot: dict, query: dict) -> tuple[dict, np.ndarray]:
    best_index = int(np.argmin(query["sample_costs"]))
    names = ("weighted_mean", "final_pass_incoming_nominal", "best_final_pass_sample")
    plans = np.asarray([query["mean_plan"], query["final_pass_nominal"], query["sample_plans"][best_index]])
    _, costs, _ = env.walker.batch_rollout(snapshot["state"], plans,
                                          initial_warmstart=snapshot["warmstart"])
    if not np.isfinite(costs).all():
        raise FloatingPointError("Non-finite full-plan score.")
    return {"horizon_controls": plans.shape[1], "semantics": "Entire open-loop plan, no actor continuation or tail.",
            "costs": dict(zip(names, costs.tolist())),
            "ranking_low_cost_first": [names[i] for i in np.argsort(costs, kind="stable")],
            "mean_minus_nominal": float(costs[0] - costs[1]),
            "mean_minus_best_sample": float(costs[0] - costs[2]),
            "best_sample_index": best_index,
            "best_sample_stored_cost": float(query["sample_costs"][best_index]),
            "best_sample_rescore_difference": float(costs[2] - query["sample_costs"][best_index])}, plans


def _finite_metrics(costs: list[float], velocities: list[float], terminated: bool,
                    horizon: int, target: float, dt: float, unhealthy_cost: float) -> dict:
    used = min(len(costs), horizon)
    # Costs already contain the falling transition's terminal charge. Add only
    # the absorbing cost for unexecuted controls, never another terminal charge.
    fell_within = bool(terminated and len(costs) <= horizon)
    absorbing = unhealthy_cost * (horizon - used) if fell_within else 0.0
    return {"requested_controls": horizon, "executed_controls": used,
            "survived": not fell_within and used == horizon, "fell": fell_within,
            "executed_task_cost": float(sum(costs[:used])), "absorbing_cost": float(absorbing),
            "finite_horizon_task_cost": float(sum(costs[:used]) + absorbing),
            "time_alive_seconds": float(used * dt),
            "mean_vx": float(np.mean(velocities[:used])),
            "velocity_mae": float(np.mean(np.abs(target - np.asarray(velocities[:used]))))}


def rollout_forced_action(snapshot: dict, first_action: np.ndarray, actor, *, total_steps: int = LONG_HORIZON,
                          short_steps: int = HORIZON, env_factory=WalkerRL) -> tuple[dict, dict]:
    """One forced action, then the named actor; no planner or analytic reward tail."""
    if total_steps < short_steps or short_steps < 1:
        raise ValueError("Require total_steps >= short_steps >= 1.")
    env = env_factory(max_episode_steps=total_steps)
    costs, velocities, actions, states, warmstarts = [], [], [], [], []
    terminated = truncated = False
    try:
        obs = restore_snapshot(env, snapshot)
        states.append(env.walker.get_state())
        warmstarts.append(env.walker.get_warmstart())
        for t in range(total_steps):
            action = np.asarray(first_action if t == 0 else actor.act(obs, deterministic=True), dtype=float)
            obs, _, terminated, truncated, info = env.step(action)
            costs.append(float(info["task_cost"]))
            velocities.append(float(info["x_velocity"]))
            actions.append(action.copy())
            states.append(env.walker.get_state())
            warmstarts.append(env.walker.get_warmstart())
            if terminated or truncated:
                break
        if len(costs) < total_steps and not terminated:
            raise RuntimeError("Unexpected administrative truncation before the requested horizon.")
        if not np.isfinite(costs).all() or not np.isfinite(velocities).all():
            raise FloatingPointError("Non-finite forced-action rollout.")
        metrics = {"short": _finite_metrics(costs, velocities, terminated, short_steps,
                                            env.walker._v_target, env.walker._dt, env.walker._w_unhealthy),
                   "long": _finite_metrics(costs, velocities, terminated, total_steps,
                                           env.walker._v_target, env.walker._dt, env.walker._w_unhealthy),
                   "administrative_cutoff": bool(truncated)}
        arrays = {"states": np.asarray(states), "warmstarts": np.asarray(warmstarts),
                  "actions": np.asarray(actions), "task_costs": np.asarray(costs),
                  "velocities": np.asarray(velocities)}
        return metrics, arrays
    finally:
        env.close()


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return load_policy(payload.get("policy", payload), device="cpu").eval()


def main(out_dir: Path, initial_checkpoint: Path = STUDY / "policy_initial.pt",
         final_checkpoint: Path = STUDY / "selective/policy_074.pt", episodes: int = 3,
         teacher_steps: int = 300, reset_seed: int = 8_900_000, planner_seed: int = 8_910_000,
         anchor_steps: tuple[int, ...] = (0, 75, 150, 225)) -> None:
    """Run a new, bounded development diagnostic; existing outputs are refused."""
    validate_protocol(episodes, teacher_steps, reset_seed, planner_seed, anchor_steps)
    initial_checkpoint, final_checkpoint = initial_checkpoint.resolve(), final_checkpoint.resolve()
    fingerprints = provenance(initial_checkpoint, final_checkpoint)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cfg = teacher_config()
    config = {"purpose": "Small descriptive horizon-ranking/continuation probe; no training or sample-efficiency claim.",
              "teacher": asdict(cfg), "episodes": episodes, "teacher_steps": teacher_steps,
              "reset_seed_base": reset_seed, "planner_seed_base": planner_seed,
              "planned_anchor_steps": anchor_steps, "short_horizon": HORIZON, "long_horizon": LONG_HORIZON,
              "task_config": str(DEFAULT_TASK), "provenance": fingerprints,
              "predeclared_checks": ["Exact raw MPPI replay after initial/final actor loading and external rollouts.",
                                      "Score weighted mean, final-pass incoming nominal and best final-pass sample over H48.",
                                      "Compare raw MPPI first action versus each actor's first action with that same actor continuing.",
                                      "Report every missing anchor after a teacher fall; do not replace episodes."],
              "cost_semantics": "Live task costs include terminal20 on first fall; add5 per missing control. No discounted/infinite failure reward tail.",
              "limitations": "Three teacher episodes, up to12 correlated anchors. Descriptive only; no Q function, calibration, causal exploration or sample-efficiency conclusion."}
    write_json(out_dir / "protocol.json", config)
    write_json(out_dir / "status.json", {"status": "collecting_teacher", "complete": False})
    started = time.perf_counter()
    # No actor has been loaded or called at this point.
    snapshots, collection, missing = collect_teacher(episodes, teacher_steps, reset_seed, planner_seed, anchor_steps, cfg)
    save_snapshots(out_dir, snapshots)
    write_json(out_dir / "teacher_collection.json", {"episodes": collection, "missing_anchors": missing})
    cases, arrays = [], {}
    query_env = WalkerRL(max_episode_steps=LONG_HORIZON)
    try:
        planner = InstrumentedMPPI(query_env.walker, cfg, seed=planner_seed)
        for snapshot in snapshots:
            case_id = snapshot["id"]
            write_json(out_dir / "status.json", {"status": "probing", "complete": False,
                                                 "completed_anchors": len(cases), "anchors": len(snapshots)})
            reference = query_teacher(query_env, planner, snapshot)
            recorded = {"max_abs_action_difference": float(np.max(np.abs(reference["action"] - snapshot["recorded_action"]))),
                        "max_abs_entire_sample_cost_vector_difference": float(np.max(np.abs(reference["sample_costs"] - snapshot["recorded_sample_costs"]))),
                        "exact": bool(np.array_equal(reference["action"], snapshot["recorded_action"]) and
                                      np.array_equal(reference["sample_costs"], snapshot["recorded_sample_costs"]))}
            plan_report, plans = score_full_plans(query_env, snapshot, reference)
            for key in ("mean_plan", "final_pass_nominal", "pass_nominals", "sample_plans", "sample_costs", "sample_weights"):
                arrays[f"anchor_{case_id}_{key}"] = reference[key]
            arrays[f"anchor_{case_id}_scored_full_plans"] = plans
            branches, interference = {}, {}
            for name, checkpoint in (("initial", initial_checkpoint), ("selective_074", final_checkpoint)):
                # Loading and executing each actor happens in separate environments
                # between raw-teacher queries restored from the identical snapshot.
                actor = load_checkpoint(checkpoint)
                actor_action = actor.act(snapshot["obs"], deterministic=True)
                for first_name, first_action in (("raw_mppi", reference["action"]), ("actor", actor_action)):
                    key = f"{first_name}_then_{name}"
                    metrics, trace = rollout_forced_action(snapshot, first_action, actor)
                    branches[key] = {"forced_first_action_source": first_name, "continuation_policy": name,
                                     "forced_first_action": first_action.tolist(), **metrics}
                    arrays.update({f"anchor_{case_id}_{key}_{k}": v for k, v in trace.items()})
                interference[f"after_{name}_load_and_rollouts"] = replay_difference(
                    reference, query_teacher(query_env, planner, snapshot))
            ranking = {horizon: sorted(branches, key=lambda key: branches[key][horizon]["finite_horizon_task_cost"])
                       for horizon in ("short", "long")}
            comparisons = {name: {horizon: branches[f"raw_mppi_then_{name}"][horizon]["finite_horizon_task_cost"] -
                                          branches[f"actor_then_{name}"][horizon]["finite_horizon_task_cost"]
                                  for horizon in ("short", "long")}
                           for name in ("initial", "selective_074")}
            cases.append({"anchor": case_id, "episode": snapshot["episode"], "teacher_t": snapshot["teacher_t"],
                          "recorded_replay": recorded, "policy_interference_checks": interference,
                          "full_open_loop_plan_scores": plan_report, "forced_first_action_scores": branches,
                          "forced_action_rankings_low_cost_first": ranking,
                          "raw_first_minus_actor_first_cost_same_continuation": comparisons})
            write_json(out_dir / "cases.json", {"cases": cases})
    finally:
        query_env.close()
    np.savez_compressed(out_dir / "query_and_branch_traces.npz", **arrays)
    exact = all(case["recorded_replay"]["exact"] and
                all(row["exact"] for row in case["policy_interference_checks"].values()) for case in cases)
    report = {"complete": True, "validity_checks_passed": exact, "wall_seconds": time.perf_counter() - started,
              "teacher_collection": collection, "observed_anchors": len(cases), "missing_anchors": missing,
              "cases": cases, "provenance": fingerprints,
              "interpretation": "Full-planH48 costs and first-action-with-named-actor-continuation costs are different estimands. No Q or learning claim.",
              "artifacts": {name: sha256(out_dir / name) for name in
                            ("protocol.json", "snapshots.npz", "snapshots.json", "teacher_collection.json",
                             "cases.json", "query_and_branch_traces.npz")}}
    write_json(out_dir / "report.json", report)
    write_json(out_dir / "status.json", {"status": "complete" if exact else "validity_failed", "complete": True,
                                         "validity_checks_passed": exact, "observed_anchors": len(cases)})
    print(json.dumps({"out_dir": str(out_dir), "observed_anchors": len(cases), "missing_anchors": len(missing),
                      "validity_checks_passed": exact, "wall_seconds": report["wall_seconds"]}))
    if not exact:
        raise RuntimeError("Raw MPPI replay independence failed; retain the complete diagnostic report.")


if __name__ == "__main__":
    tyro.cli(main)
