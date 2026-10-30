"""Calibrate a frozen historical SAC critic using paired simulator continuations.

This is a bounded diagnostic, not a training run. A forced first action has no
entropy term; all subsequent actions come from the frozen stochastic learner.
The environment's adjusted fall reward already includes its absorbing tail.
Neither a critic bootstrap nor another failure tail is added by this module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from scripts.runners.eval_policy_walker2d import provenance, write_json
from src.rl.sac import SacActor, SoftQNetwork, load_policy
from src.rl.walker import DEFAULT_TASK, WalkerRL


ROOT = Path(__file__).resolve().parents[2]
ACTION_NAMES = ("independent_mppi", "actor_mean")
MAX_STATES, MAX_REPEATS, MAX_CONTROLS = 8, 8, 2500
DEFAULT_CHECKPOINT = ROOT / "runs/2026-10-23_step19_ctx_seed1/checkpoints/step_0200000.pt"
DEFAULT_TRACES = (
    ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/evaluations/step_0000000.npz",
    ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/evaluations/step_0800000.npz",
)
SOURCES = (
    "scripts/diagnostics/critic_continuation_probe.py", "src/rl/sac.py", "src/rl/walker.py",
    "src/envs/base.py", "src/envs/mujoco_env.py", "src/envs/walker2d.py",
    "src/mppi/mppi.py", "configs/walker2d_task.json", "configs/walker2d_best.json",
    "assets/walker2d.xml",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class FrozenCritic:
    actor: SacActor
    q1: SoftQNetwork
    q2: SoftQNetwork
    gamma: float
    alpha: float
    checkpoint_metadata: dict[str, Any]


def load_frozen(checkpoint: Path, task_config: Path = DEFAULT_TASK) -> FrozenCritic:
    """Load only frozen actor/online critics, leaving optimizers unconstructed."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    saved = payload["agent"]
    gamma = float(payload["gamma"])
    if gamma != float(saved["config"]["gamma"]) or not 0 <= gamma < 1:
        raise ValueError("Checkpoint discount settings disagree or are invalid.")
    if payload.get("task_sha256") != sha256(task_config):
        raise ValueError("Checkpoint task hash does not match the selected task.")
    policy_state = payload["policy"]
    if any(not torch.equal(value, saved["policy"]["actor"][key])
           for key, value in policy_state["actor"].items()):
        raise ValueError("Checkpoint actor snapshots disagree.")
    with torch.random.fork_rng(devices=[]):
        actor = load_policy(policy_state, "cpu")
        if not isinstance(actor, SacActor):
            raise ValueError("Soft-Q calibration requires a stochastic SAC actor.")
        q1 = SoftQNetwork(actor.obs_dim, actor.act_dim, saved["config"]["hidden_dim"])
        q2 = SoftQNetwork(actor.obs_dim, actor.act_dim, saved["config"]["hidden_dim"])
        q1.load_state_dict(saved["q1"])
        q2.load_state_dict(saved["q2"])
    for model in (actor, q1, q2):
        model.eval().requires_grad_(False)
        if any(not bool(torch.isfinite(value).all()) for value in model.state_dict().values()):
            raise FloatingPointError("Checkpoint contains non-finite model values.")
    alpha = float(saved["log_alpha"].exp())
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Checkpoint entropy coefficient is invalid.")
    return FrozenCritic(actor, q1, q2, gamma, alpha, {
        "path": str(checkpoint.resolve()), "sha256": sha256(checkpoint),
        "step": payload.get("step"), "seed": payload.get("seed"),
        "source_sha256": payload.get("source_sha256"),
        "training_mppi_stream": payload.get("config", {}).get("mppi_stream"),
    })


def uniform_indices(length: int, count: int) -> np.ndarray:
    """Select by input index alone, before actions, scores or outcomes are read."""
    if length < 1 or count < 1:
        raise ValueError("Need nonempty inputs and a positive state count.")
    return np.linspace(0, length - 1, min(length, count), dtype=int)


def load_anchors(path: Path, state_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read independent_teacher_probe's numeric snapshots without pickle."""
    with np.load(path, allow_pickle=False) as arrays:
        states, warmstarts = arrays["states"], arrays["warmstarts"]
        actions = arrays["recorded_actions"]
        if states.ndim != 2 or warmstarts.shape != (len(states), 9) or actions.shape != (len(states), 6):
            raise ValueError("Require states[N,S], warmstarts[N,9], recorded_actions[N,6].")
        if states.shape[1] != 19 or not all(np.isfinite(a).all() for a in (states, warmstarts, actions)):
            raise ValueError("Require finite Walker full states, warm starts and actions.")
        indices = uniform_indices(len(states), state_count)
        selected = [(states[i].copy(), warmstarts[i].copy(), actions[i].copy()) for i in indices]
    sidecar = path.with_suffix(".json")
    manifest = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    if manifest.get("numeric_sha256", sha256(path)) != sha256(path):
        raise ValueError("Anchor numeric hash differs from its sidecar manifest.")
    metadata = manifest.get("snapshots", [])
    if metadata and len(metadata) != len(states):
        raise ValueError("Anchor sidecar row count differs from numeric rows.")
    anchors = [{"id": str(metadata[int(i)].get("id", int(i))) if metadata else str(int(i)),
                "source": "recorded_independent_teacher", "input_row": int(i),
                "input_metadata": metadata[int(i)] if metadata else {},
                "state": state, "warmstart": warm, "teacher_action": action}
               for i, (state, warm, action) in zip(indices, selected)]
    inputs = {"kind": "independent_teacher_snapshots", "path": str(path.resolve()),
              "sha256": sha256(path), "selected_rows": indices.tolist(),
              "selection": "Uniform input-row indices, fixed before critic/action/continuation outcomes.",
              "sidecar_sha256": sha256(sidecar) if sidecar.exists() else None}
    protocol = path.parent / "protocol.json"
    if protocol.exists():
        inputs["teacher_protocol_sha256"] = sha256(protocol)
    return anchors, inputs


def load_trace_anchors(paths: tuple[Path, ...], state_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fallback: uniform pre-action states from explicitly identified trajectories."""
    if not paths or len(paths) > state_count:
        raise ValueError("Need at least one trace and no more traces than selected states.")
    anchors, inputs = [], []
    quotas = [state_count // len(paths) + int(i < state_count % len(paths)) for i in range(len(paths))]
    for source_index, (path, quota) in enumerate(zip(paths, quotas)):
        with np.load(path, allow_pickle=False) as arrays:
            states, warmstarts = arrays["states"], arrays["warmstarts"]
            offsets = arrays["episode_state_offsets"]
            if states.ndim != 2 or states.shape[1] != 19 or warmstarts.shape != (len(states), 9):
                raise ValueError("Trace must contain Walker states and aligned solver warm starts.")
            if not (np.isfinite(states).all() and np.isfinite(warmstarts).all()):
                raise ValueError("Trace contains non-finite states.")
            if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(states):
                raise ValueError("Invalid episode state offsets.")
            if not np.issubdtype(offsets.dtype, np.integer) or np.any(np.diff(offsets) < 2):
                raise ValueError("Each recorded episode needs initial plus post-action states.")
            eligible = np.concatenate([np.arange(a, b - 1) for a, b in zip(offsets[:-1], offsets[1:])])
            chosen = eligible[uniform_indices(len(eligible), quota)]
            for index in chosen:
                anchors.append({"id": f"trace{source_index}_row{index}",
                                "source": str(path.resolve()), "input_row": int(index),
                                "state": states[index].copy(), "warmstart": warmstarts[index].copy(),
                                "teacher_action": None, "input_metadata": {}})
        inputs.append({"path": str(path.resolve()), "sha256": sha256(path), "selected_rows": chosen.tolist()})
    return anchors, {"kind": "policy_trace_fallback", "traces": inputs,
                     "selection": "Uniform pre-action row indices in each supplied trace; no outcome selection.",
                     "teacher_query": "Fresh zero-plan K32/H48 MPPI per state (cold query, no carried history)."}


@torch.no_grad()
def query_q(frozen: FrozenCritic, observation: np.ndarray, action: np.ndarray) -> dict[str, float]:
    obs = torch.as_tensor(observation, dtype=torch.float32)[None]
    act = torch.as_tensor(action, dtype=torch.float32)[None]
    q1, q2 = float(frozen.q1(obs, act).item()), float(frozen.q2(obs, act).item())
    if not np.isfinite([q1, q2]).all():
        raise FloatingPointError("Non-finite critic query.")
    return {"q1": q1, "q2": q2, "min_q": min(q1, q2), "twin_disagreement": abs(q1 - q2)}


@torch.no_grad()
def rollout_soft_return(env: WalkerRL, actor: SacActor, state: np.ndarray,
                        warmstart: np.ndarray, first_action: np.ndarray, *,
                        gamma: float, alpha: float, max_steps: int, seed: int) -> dict[str, Any]:
    """Force a0, then sample frozen pi; return an unbootstrapped soft-Q estimate."""
    if not 0 <= gamma < 1 or not np.isfinite(alpha) or alpha < 0 or max_steps < 1:
        raise ValueError("Invalid rollout discount, entropy coefficient or horizon.")
    observation, _ = env.reset(options={"state": state, "warmstart": warmstart})
    if not np.array_equal(env.walker.get_state(), state) or not np.array_equal(env.walker.get_warmstart(), warmstart):
        raise RuntimeError("Simulator did not restore the declared full state and solver warm start.")
    total = discounted_reward = entropy_bonus = 0.0
    partial = {}
    horizons = sorted({min(1000, max_steps), max_steps})
    terminated = truncated = False
    failure_tail = 0.0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for t in range(max_steps):
            log_prob = 0.0
            if t == 0:
                action = np.asarray(first_action, dtype=np.float64).copy()
            else:
                action_tensor, log_tensor, _ = actor.get_action(torch.as_tensor(observation, dtype=torch.float32)[None])
                action = action_tensor[0].cpu().numpy().copy()
                log_prob = float(log_tensor.item())
            if not np.isfinite(action).all() or not np.isfinite(log_prob):
                raise FloatingPointError("Non-finite continuation action/log probability.")
            observation, reward, terminated, truncated, info = env.step(action)
            expected_reward = -float(info["task_cost"]) + float(info["failure_tail"])
            if not np.isfinite(reward) or not np.isclose(reward, expected_reward, rtol=1e-12, atol=1e-12):
                raise RuntimeError("Environment reward differs from task-plus-failure-tail convention.")
            discount = gamma ** t
            discounted_reward += discount * reward
            entropy_bonus -= discount * alpha * log_prob
            total = discounted_reward + entropy_bonus
            failure_tail += float(info["failure_tail"])
            if t + 1 in horizons:
                partial[str(t + 1)] = total
            if terminated or truncated:
                if truncated and t + 1 != max_steps:
                    raise RuntimeError("Unexpected administrative cutoff before diagnostic horizon.")
                break
    steps = t + 1
    if terminated:
        # The reward on the fall already accounts for the complete absorbing future.
        for horizon in horizons:
            partial.setdefault(str(horizon), total)
    if not np.isfinite([total, discounted_reward, entropy_bonus, failure_tail]).all():
        raise FloatingPointError("Non-finite rollout accounting.")
    return {"seed": seed, "steps": steps, "terminated": bool(terminated), "truncated": bool(truncated),
            "soft_return": total, "discounted_adjusted_rewards": discounted_reward,
            "discounted_entropy_bonus": entropy_bonus, "failure_tail_charged_by_environment": failure_tail,
            "partial_soft_returns": partial, "residual_discount_weight": 0.0 if terminated else gamma ** steps}


def mc_summary(values: list[float]) -> dict[str, Any]:
    samples = np.asarray(values, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Need finite, nonempty Monte Carlo samples.")
    return {"mean": float(samples.mean()),
            "mc_standard_error": float(samples.std(ddof=1) / np.sqrt(len(samples))) if len(samples) > 1 else None,
            "replicates": len(samples)}


def paired_ranking(actor_returns: list[float], teacher_returns: list[float], q_delta: float) -> dict[str, Any]:
    if len(actor_returns) != len(teacher_returns):
        raise ValueError("Paired continuation counts differ.")
    delta = np.asarray(actor_returns) - np.asarray(teacher_returns)
    summary = mc_summary(delta.tolist())
    mean = summary["mean"]
    return {"difference": "actor_mean minus independent_mppi", "critic_delta": q_delta,
            "mc_delta": mean, "paired_mc_standard_error": summary["mc_standard_error"],
            "empirical_ranking_agrees": bool(np.sign(q_delta) == np.sign(mean)),
            "interpretation": "A finite-rollout paired ranking, not a population or infinite-horizon certificate."}


def _cold_teacher_action(env: WalkerRL, state: np.ndarray, warmstart: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    from src.mppi.mppi import MPPI
    from src.utils.config import MPPIConfig

    base = MPPIConfig(**json.loads((ROOT / "configs/walker2d_best.json").read_text()))
    cfg = replace(base, K=32, H=48, lam=3.0, lam_target_neff_fraction=None)
    planner = MPPI(env.walker, cfg, seed=seed)
    action, info = planner.plan_step(state, initial_warmstart=warmstart)
    return action, cfg.K * cfg.H * info["optimization_iterations"]


def run(*, checkpoint: Path, output_dir: Path, anchors_path: Path | None = None,
        trace_paths: tuple[Path, ...] = DEFAULT_TRACES, state_count: int = 8,
        repeats: int = 8, max_steps: int = 2500, seed_base: int = 9_420_000,
        task_config: Path = DEFAULT_TASK) -> dict[str, Any]:
    """Write a new protocol, raw paired outcomes and a calibration summary."""
    for name, value, upper in (("state_count", state_count, MAX_STATES), ("repeats", repeats, MAX_REPEATS),
                               ("max_steps", max_steps, MAX_CONTROLS)):
        if type(value) is not int or not 1 <= value <= upper:
            raise ValueError(f"{name} must lie in [1, {upper}].")
    if type(seed_base) is not int or not 0 <= seed_base < 2**32 - state_count * repeats:
        raise ValueError("Continuation seed base is invalid.")
    started = time.perf_counter()
    frozen = load_frozen(checkpoint, task_config)
    anchors, inputs = (load_anchors(anchors_path, state_count) if anchors_path is not None
                       else load_trace_anchors(trace_paths, state_count))
    output_dir.mkdir(parents=True, exist_ok=False)
    fingerprints = {path: sha256(ROOT / path) for path in SOURCES}
    protocol = {
        "purpose": "Frozen historical critic calibration; no learning and no independent-loop efficacy claim.",
        "checkpoint": frozen.checkpoint_metadata, "inputs": inputs,
        "states": [{key: value for key, value in anchor.items() if key not in {"state", "warmstart", "teacher_action"}}
                   for anchor in anchors],
        "state_count": len(anchors), "repeats": repeats, "max_steps": max_steps,
        "gamma": frozen.gamma, "alpha": frozen.alpha, "seed_base": seed_base,
        "pairing": "Identical torch seed per state/replicate for both first actions; fresh frozen-pi draws at t>=1.",
        "max_continuation_controls": len(anchors) * len(ACTION_NAMES) * repeats * max_steps,
        "score": "r0 + sum_{t>=1} gamma**t * (r_t - alpha*log pi(a_t|s_t)); no entropy at fixed a0.",
        "failure": "Adjusted environment reward contains the analytic absorbing tail exactly once; no postfall bootstrap/entropy.",
        "truncation": "No critic tail. Report residual discount weight, not a bound on missing return.",
        "source_sha256": fingerprints, "environment": provenance(),
    }
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "status.json", {"complete": False, "phase": "evaluating", "cases_completed": 0})
    cases, simulated_controls, teacher_controls = [], 0, 0
    torch.set_num_threads(1)
    env = WalkerRL(task_config, frozen.gamma, max_steps)
    try:
        for index, anchor in enumerate(anchors):
            state, warmstart = anchor["state"], anchor["warmstart"]
            observation, _ = env.reset(options={"state": state, "warmstart": warmstart})
            low, high = env.walker.action_bounds
            if not (np.array_equal(low.astype(np.float32), frozen.actor.action_low.numpy())
                    and np.array_equal(high.astype(np.float32), frozen.actor.action_high.numpy())):
                raise ValueError("Checkpoint and environment action bounds differ.")
            teacher_action = anchor["teacher_action"]
            if teacher_action is None:
                teacher_action, cost = _cold_teacher_action(env, state, warmstart, seed_base + index)
                teacher_controls += cost
            if teacher_action.shape != (6,) or not np.isfinite(teacher_action).all() or np.any(teacher_action < low) or np.any(teacher_action > high):
                raise ValueError("Invalid independent teacher action.")
            actions = {"independent_mppi": teacher_action, "actor_mean": frozen.actor.act(observation, deterministic=True)}
            branches = {}
            for name, action in actions.items():
                q = query_q(frozen, observation, action)
                results = [rollout_soft_return(env, frozen.actor, state, warmstart, action,
                           gamma=frozen.gamma, alpha=frozen.alpha, max_steps=max_steps,
                           seed=seed_base + index * repeats + replicate) for replicate in range(repeats)]
                summary = mc_summary([result["soft_return"] for result in results])
                summary.update(bias_min_q_minus_mc=q["min_q"] - summary["mean"],
                               truncated_replicates=sum(result["truncated"] for result in results),
                               largest_residual_discount_weight=max(result["residual_discount_weight"] for result in results))
                simulated_controls += sum(result["steps"] for result in results)
                branches[name] = {"first_action": action.tolist(), **q, "monte_carlo": summary, "rollouts": results}
            paired = paired_ranking([r["soft_return"] for r in branches["actor_mean"]["rollouts"]],
                                    [r["soft_return"] for r in branches["independent_mppi"]["rollouts"]],
                                    branches["actor_mean"]["min_q"] - branches["independent_mppi"]["min_q"])
            cases.append({"id": anchor["id"], "source": anchor["source"], "input_row": anchor["input_row"],
                          "action_distance": float(np.linalg.norm(actions["actor_mean"] - actions["independent_mppi"])),
                          "branches": branches, "paired_action_ranking": paired})
            write_json(output_dir / "cases.json", {"cases": cases})
            write_json(output_dir / "status.json", {"complete": False, "phase": "evaluating", "cases_completed": len(cases)})
        unchanged = fingerprints == {path: sha256(ROOT / path) for path in SOURCES}
        if not unchanged:
            raise RuntimeError("Executed diagnostic source changed during evaluation.")
        summary = {name: {
            "mean_bias_min_q_minus_mc": float(np.mean([c["branches"][name]["monte_carlo"]["bias_min_q_minus_mc"] for c in cases])),
            "rmse_against_finite_mc_means": float(np.sqrt(np.mean([c["branches"][name]["monte_carlo"]["bias_min_q_minus_mc"] ** 2 for c in cases]))),
        } for name in ACTION_NAMES}
        report = {"protocol": protocol, "cases": cases, "summary": summary, "source_unchanged": unchanged,
                  "resource_ledger": {"continuation_controls": simulated_controls, "cold_teacher_search_controls": teacher_controls,
                                      "wall_seconds": time.perf_counter() - started},
                  "limits": ["Historical actor-tail-trained critic, not a critic trained in the proposed independent replay loop.",
                             "Monte Carlo variability and finite-rollout tail error both matter; twin-Q agreement is not certification.",
                             "No policy updates, sample-efficiency experiment, or comparison across training seeds."]}
        write_json(output_dir / "comparison.json", report)
        lines = ["# Frozen critic continuation diagnostic", "", protocol["purpose"], "",
                 f"States: {len(cases)}; paired repeats: {repeats}; maximum controls: {max_steps}; gamma: {frozen.gamma}; alpha: {frozen.alpha:.8g}.", "",
                 "| State | Action | min Q | MC soft return | MC SE | Q minus MC | Truncated |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for case in cases:
            for name in ACTION_NAMES:
                branch = case["branches"][name]
                mc = branch["monte_carlo"]
                se = "unavailable" if mc["mc_standard_error"] is None else f"{mc['mc_standard_error']:.3f}"
                lines.append(f"| {case['id']} | {name} | {branch['min_q']:.3f} | {mc['mean']:.3f} | {se} | {mc['bias_min_q_minus_mc']:.3f} | {mc['truncated_replicates']} |")
        lines += ["", *report["limits"], "", "A forced initial action has no entropy term. Continuing Q is compared with unbootstrapped stochastic-learner rollouts, not MPPI continuations."]
        (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
        write_json(output_dir / "status.json", {"complete": True, "phase": "complete", "cases_completed": len(cases),
                                                "source_unchanged": unchanged, "resource_ledger": report["resource_ledger"]})
        return report
    except BaseException as exc:
        write_json(output_dir / "status.json", {"complete": False, "phase": "failed", "cases_completed": len(cases), "error": repr(exc)})
        raise
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, help="independent_teacher_probe snapshots.npz; preferred over cold queries")
    parser.add_argument("--trace", type=Path, action="append", help="Fallback saved-policy trace NPZ; repeat for multiple sources")
    parser.add_argument("--state-count", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed-base", type=int, default=9_420_000)
    args = parser.parse_args()
    report = run(checkpoint=args.checkpoint, output_dir=args.output_dir, anchors_path=args.anchors,
                 trace_paths=tuple(args.trace) if args.trace else DEFAULT_TRACES,
                 state_count=args.state_count, repeats=args.repeats, max_steps=args.max_steps, seed_base=args.seed_base)
    print(json.dumps({"states": len(report["cases"]), "summary": report["summary"],
                      "resource_ledger": report["resource_ledger"]}, allow_nan=False))


if __name__ == "__main__":
    main()
