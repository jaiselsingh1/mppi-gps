"""Single-action MPPI seed sensitivity, with a shared original continuation.

Alternative seeds change only the proposal at one saved decision. Their planner
memory is discarded: every continuation retains the original post-call nominal
and random stream. This is a sensitivity control, not a zero-centered or
action-magnitude-matched null distribution.
"""

import copy
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics.walker_gate_counterfactual import (
    FROZEN_SOURCES, _json_snapshot, _save_trace, compare_reference,
    load_decision, run_branch,
)
from scripts.runners.eval_policy_walker2d import sha256, write_json
from src.envs.walker2d import Walker2d
from src.mppi.mppi import MPPI
from src.utils.trace_replay import compare_fields, decision_snapshot


def _validate_seed(seed: int | None) -> None:
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("Proposal seed must be a nonnegative integer, or None for the original RNG.")


def generate_alternative(start: dict, seed: int | None) -> tuple[np.ndarray, dict]:
    """Generate one action from the original PRE-call memory, without executing.

    ``None`` preserves the saved random stream and is the exact-recovery audit.
    An integer changes only that stream, not U, refinement, state or warmstart.
    Neither this helper nor its returned post-call snapshot is used to seed the
    downstream branch. That branch always starts from ``start['snapshot']``.
    """
    _validate_seed(seed)
    before = copy.deepcopy(decision_snapshot(start["trace"], start["decision_step"]))
    if seed is not None:
        before["rng_state"] = copy.deepcopy(np.random.default_rng(seed).bit_generator.state)
    env = Walker2d(**start["task"])
    try:
        env.reset(state=start["state"].copy())
        env.set_warmstart(start["warmstart"].copy())
        controller = MPPI(env, start["cfg"])
        controller.restore(copy.deepcopy(before))
        state, warmstart = env.get_state(), env.get_warmstart()
        action, info = controller.plan_step(state, initial_warmstart=warmstart)
        if not (np.array_equal(env.get_state(), state)
                and np.array_equal(env.get_warmstart(), warmstart)):
            raise RuntimeError("Proposal planning mutated the source simulator state or warmstart.")
        low, high = env.action_bounds
        if (action.shape != (env.action_dim,) or not np.isfinite(action).all()
                or (action < low).any() or (action > high).any()):
            raise ValueError("Proposal produced an invalid action.")
        difference = action - start["baseline_action"]
        return action.copy(), {
            "proposal_seed": seed,
            "original_rng_preserved": seed is None,
            "action": action.tolist(),
            "action_l2_difference": float(np.linalg.norm(difference)),
            "action_linf_difference": float(np.max(np.abs(difference))),
            "pre_call_snapshot": _json_snapshot(before),
            "discarded_post_call_snapshot": _json_snapshot(controller.snapshot()),
            "optimized_plan": controller.last_plan.tolist(),
            "optimization_iterations": int(info["optimization_iterations"]),
        }
    finally:
        env.close()


def window_metrics(trace: dict, summary: dict, task: dict, horizon: int = 48) -> dict:
    """Fixed-window costs, padding after a fall but never after truncation."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("Metric horizon must be a positive integer.")
    observed = min(len(trace["costs"]), horizon)
    fell = summary["first_fall_step"] is not None and summary["first_fall_step"] <= horizon
    complete = observed == horizon or fell
    cost = (float(trace["costs"][:observed].sum()
                  + (horizon - observed) * task["unhealthy_cost_weight"])
            if complete else None)
    vx = trace["states"][1:observed + 1, 10]
    return {"horizon_steps": horizon, "observed_steps": observed,
            "fixed_horizon_task_cost": cost, "fell": fell,
            "velocity_mae": float(np.mean(np.abs(vx - task["target_velocity"]))) if observed else None,
            "mean_velocity": float(vx.mean()) if observed else None,
            "complete_window_or_absorbing_tail": complete}


def _difference(summary: dict, raw: dict) -> dict:
    def subtract(left, right):
        return None if left is None or right is None else left - right

    return {
        "fixed_horizon_task_cost": summary["fixed_horizon_task_cost"] - raw["fixed_horizon_task_cost"],
        "velocity_mae_executed_prefix": summary["velocity_mae"] - raw["velocity_mae"],
        "window48_task_cost": subtract(summary["window48"]["fixed_horizon_task_cost"],
                                        raw["window48"]["fixed_horizon_task_cost"]),
        "window48_velocity_mae_executed_prefix": subtract(summary["window48"]["velocity_mae"],
                                                           raw["window48"]["velocity_mae"]),
        "additional_fall": summary["first_fall_step"] is not None and raw["first_fall_step"] is None,
        "prevented_fall": summary["first_fall_step"] is None and raw["first_fall_step"] is not None,
    }


def main(shadow_dir: Path, policy_label: str, decision_step: int, out_dir: Path,
         proposal_seeds: tuple[int, ...] = (31000, 31001, 31002, 31003, 31004),
         steps: int = 250) -> None:
    """Run one raw branch, one identical-action control, and seeded alternatives."""
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer including the first action.")
    if not proposal_seeds or len(set(proposal_seeds)) != len(proposal_seeds):
        raise ValueError("Specify at least one distinct proposal seed.")
    for seed in proposal_seeds:
        _validate_seed(seed)
        if seed is None:
            raise ValueError("CLI proposal seeds must be integers; original-RNG recovery runs automatically.")
    start = load_decision(Path(shadow_dir), policy_label, decision_step)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = FROZEN_SOURCES + (
        "src/utils/trace_replay.py", "scripts/runners/eval_policy_walker2d.py",
        "scripts/diagnostics/walker_gate_counterfactual.py",
        "scripts/diagnostics/walker_action_controls.py",
    )
    source_hashes = {relative: sha256(_ROOT / relative) for relative in source_files}
    for relative in source_files:
        target = out_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)
    inputs = out_dir / "inputs"
    inputs.mkdir()
    for relative, name in (("manifest.json", "shadow_manifest.json"),
                           ("artifact_hashes.json", "shadow_artifact_hashes.json"),
                           ("report.json", "shadow_report.json"),
                           ("inputs/task.json", "task.json"),
                           ("inputs/mppi.json", "mppi.json")):
        shutil.copyfile(Path(shadow_dir) / relative, inputs / name)
    pre = decision_snapshot(start["trace"], decision_step)
    _save_trace(inputs / "decision.npz", {
        "state": start["state"], "warmstart": start["warmstart"],
        "baseline_action": start["baseline_action"], "policy_action": start["candidate_action"],
        "pre_plan_nominal": pre["U"], "post_plan_nominal": start["snapshot"]["U"],
    })
    manifest = {
        "shadow_dir": start["shadow_dir"], "shadow_artifact_hashes_sha256": start["artifact_hashes_sha256"],
        "policy_label": policy_label, "decision_step": decision_step, "steps": steps,
        "proposal_seeds": list(proposal_seeds), "task": start["task"], "mppi_config": asdict(start["cfg"]),
        "gate_decision": start["decision"], "pre_call_snapshot": _json_snapshot(pre),
        "original_post_call_snapshot": _json_snapshot(start["snapshot"]),
        "provenance": {"python": platform.python_version(),
                       "packages": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
                       "source_sha256": source_hashes},
        "intervention": "Only first action changes. All continuations use original post-call U, RNG and refinement flag; alternative proposal planner memory is discarded.",
        "trace_layout": "states/warmstarts include initial state; planned_here[0]=False; first pre-plan snapshot is original POST-call snapshot.",
    }
    write_json(out_dir / "manifest.json", manifest)
    started = time.perf_counter()
    report = {
        "status": "running", "error": None, "single_intervention_only": True,
        "branches": {}, "proposals": {}, "raw_reference_comparison": None,
        "identity_comparison": None, "original_rng_recovery_exact": None,
        "cost_semantics": "Fixed-horizon totals include each executed task cost and one unhealthy_cost_weight per remaining control after a fall. No repeated terminal penalty. window48 covers exactly 48 controls, or is null if the requested run truncates earlier without a fall. MAE is over executed steps only; prefix lengths are reported.",
        "limits": "Single-action MPPI seed sensitivity, not a zero-centered, magnitude-matched null or population-level effect. Compare action differences and paired costs together.",
    }

    def progress(active: str | None = None) -> None:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(out_dir / "report.json", report)
        write_json(out_dir / "status.json", {
            "status": report["status"], "active_branch": active,
            "completed_branches": list(report["branches"]),
            "total_branches": len(proposal_seeds) + 2,
            "elapsed_seconds": report["elapsed_seconds"], "error": report["error"],
        })

    progress("original_rng_recovery")
    try:
        recovered, recovery = generate_alternative(start, None)
        original_plan = start["trace"]["plans"][decision_step]
        exact = (np.array_equal(recovered, start["baseline_action"])
                 and np.array_equal(recovery["optimized_plan"], original_plan)
                 and recovery["discarded_post_call_snapshot"] == _json_snapshot(start["snapshot"]))
        report["original_rng_recovery_exact"] = exact
        write_json(out_dir / "original_rng_recovery.json", recovery)
        if not exact:
            raise RuntimeError("Original pre-call RNG does not exactly recover the original planning output.")
        raw_trace = None
        for label, seed in [("raw", None), ("identity", None)] + [(f"seed_{seed}", seed) for seed in proposal_seeds]:
            progress(label)
            if seed is None:
                action = start["baseline_action"].copy()
            else:
                action, proposal = generate_alternative(start, seed)
                report["proposals"][label] = proposal
                write_json(out_dir / "proposals.json", report["proposals"])
            trace, summary = run_branch(start["task"], start["cfg"], start["state"], start["warmstart"],
                                        copy.deepcopy(start["snapshot"]), action.copy(), steps)
            _save_trace(out_dir / f"{label}.npz", trace)
            difference = action - start["baseline_action"]
            summary.update(window48=window_metrics(trace, summary, start["task"]),
                           first_action=action.tolist(), proposal_seed=seed,
                           action_l2_difference=float(np.linalg.norm(difference)),
                           action_linf_difference=float(np.max(np.abs(difference))))
            report["branches"][label] = summary
            if label == "raw":
                raw_trace = trace
                comparison = compare_reference(trace, start["trace"], decision_step, steps)
                report["raw_reference_comparison"] = comparison
                if not comparison["exact"]:
                    raise RuntimeError("Raw continuation differs from the saved shadow prefix.")
            else:
                summary["difference_from_raw"] = _difference(summary, report["branches"]["raw"])
                if label == "identity":
                    fields = compare_fields(raw_trace, trace, tuple(raw_trace))
                    report["identity_comparison"] = {"exact": all(row["exact"] for row in fields.values()), "fields": fields}
                    if not report["identity_comparison"]["exact"]:
                        raise RuntimeError("Same-action control is not byte-identical across every trace field.")
            progress()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        progress()
        hashes = {str(path.relative_to(out_dir)): sha256(path)
                  for path in sorted(out_dir.rglob("*"))
                  if path.is_file() and path.name != "artifact_hashes.json"}
        write_json(out_dir / "artifact_hashes.json", hashes)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
