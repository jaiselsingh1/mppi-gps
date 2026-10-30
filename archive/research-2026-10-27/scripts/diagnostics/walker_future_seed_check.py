"""Repeat fixed first actions under paired, fresh future-planner randomness.

Every branch retains the ORIGINAL post-planning nominal/refinement flag.
Only the future RNG is changed. This diagnoses seed sensitivity of the
existing execution rule, independently of any nominal-memory ablation.
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

from scripts.diagnostics.walker_action_controls import window_metrics
from scripts.diagnostics.walker_gate_counterfactual import (
    FROZEN_SOURCES, _json_snapshot, _save_trace, load_decision, run_branch,
)
from scripts.diagnostics.walker_gate_v2_diagnostic import (
    _load_trace, _validate_start, _validate_trace_start, read_bundle,
)
from scripts.runners.eval_policy_walker2d import sha256, write_json
from src.utils.trace_replay import compare_fields


FUTURE_SEEDS = tuple(range(41000, 41008))
PROPOSAL_SEEDS = tuple(range(31000, 31005))
STEPS = 250
SCORE_HORIZON = 48
CASES = (("success", "intermediate", 96), ("success", "strong", 10),
         ("failure", "intermediate", 126), ("failure", "strong", 0))
DEFAULT_CONTROLS = _ROOT / "runs/2026-10-12_s3_action_controls_001"
DEFAULT_SHADOW = _ROOT / "runs/2026-10-11_s3_shadow_001"


def future_snapshot(original: dict, seed: int) -> dict:
    """Change only future RNG, preserving the original shifted nominal."""
    if type(seed) is not int or seed < 0:
        raise ValueError("Future seed must be a nonnegative integer.")
    if original.get("initial_plan") is not False:
        raise ValueError("Require the original post-planning snapshot.")
    snapshot = copy.deepcopy(original)
    snapshot["rng_state"] = copy.deepcopy(np.random.default_rng(seed).bit_generator.state)
    return snapshot


def verify_pair(raw: dict, candidate: dict, state: np.ndarray, warmstart: np.ndarray,
                snapshot: dict) -> dict:
    """Check common control-boundary randomness through either first fall."""
    for trace in (raw, candidate):
        if (not len(trace["actions"]) or not np.array_equal(trace["states"][0], state)
                or not np.array_equal(trace["warmstarts"][0], warmstart)
                or not np.array_equal(trace["pre_plan_nominals"][0], snapshot["U"])
                or not np.array_equal(trace["post_plan_nominals"][0], snapshot["U"])
                or bool(trace["initial_plan_flags"][0]) or bool(trace["planned_here"][0])
                or json.loads(str(trace["planner_rng_states"][0])) != snapshot["rng_state"]
                or json.loads(str(trace["post_planner_rng_states"][0])) != snapshot["rng_state"]):
            raise RuntimeError("Branch does not start at the declared common state/post-call memory.")
        if len(trace["states"]) != len(trace["actions"]) + 1 or len(trace["warmstarts"]) != len(trace["states"]):
            raise RuntimeError("Invalid branch state/action alignment.")
        if (not np.array_equal(trace["planned_here"], np.arange(len(trace["actions"])) > 0)
                or np.any(trace["initial_plan_flags"]) or np.any(trace["post_initial_plan_flags"])):
            raise RuntimeError("A branch changed the declared planning schedule/refinement flag.")
    length = min(len(raw["actions"]), len(candidate["actions"]))
    fields = ("planner_rng_states", "post_planner_rng_states", "initial_plan_flags",
              "post_initial_plan_flags", "planned_here")
    compared = compare_fields({key: raw[key][:length] for key in fields},
                             {key: candidate[key][:length] for key in fields}, fields)
    if not all(value["exact"] for value in compared.values()):
        raise RuntimeError("Paired branches consumed different future planner random streams.")
    return {"exact": True, "checked_controls": length, "fields": list(fields)}


def first_transition(trace: dict) -> dict:
    return {"action": trace["actions"][0].copy(), "state": trace["states"][1].copy(),
            "warmstart": trace["warmstarts"][1].copy(), "cost": float(trace["costs"][0])}


def verify_first_transition(trace: dict, expected: dict) -> None:
    actual = first_transition(trace)
    if (any(not np.array_equal(actual[key], expected[key]) for key in ("action", "state", "warmstart"))
            or actual["cost"] != expected["cost"]):
        raise RuntimeError("Future randomness changed the fixed first transition from saved evidence.")


def validate_branch_summary(trace: dict, summary: dict, task: dict, steps: int) -> None:
    """Do not infer absorption merely from an unchecked summary flag."""
    count = len(trace["actions"])
    if not 1 <= count <= steps or len(trace["states"]) != count + 1:
        raise RuntimeError("Invalid observed branch duration.")
    states = trace["states"][1:]
    healthy = ((states[:, 2] > .8) & (states[:, 2] < 2.)
               & (states[:, 3] > -1.) & (states[:, 3] < 1.))
    falls = np.flatnonzero(~healthy)
    fell = bool(len(falls))
    if (not np.isfinite(states).all() or not np.isfinite(trace["costs"]).all()
            or len(trace["costs"]) != count or (fell and not np.array_equal(falls, [count - 1]))
            or (count < steps and not fell)):
        raise RuntimeError("Branch has a non-finite prefix, post-fall execution or unexplained truncation.")
    expected_fall = count if fell else None
    cost = float(trace["costs"].sum())
    if (summary["requested_steps"] != steps or summary["completed_steps"] != count
            or summary["first_fall_step"] != expected_fall
            or summary["completed_without_fall"] != (count == steps and not fell)
            or summary["task_cost"] != cost
            or summary["fixed_horizon_task_cost"] != cost + (steps - count) * task["unhealthy_cost_weight"]):
        raise RuntimeError("Branch summary disagrees with observed termination/costs.")


def load_actions(start: dict, controls_dir: Path, policy_dir: Path) -> tuple[dict, dict, dict]:
    """Read fixed actions only; do not select them using repeat-run outcomes."""
    manifest, report, hashes = read_bundle(controls_dir)
    policy_manifest, _, policy_hashes = read_bundle(policy_dir)
    _validate_start(start, manifest)
    _validate_start(start, policy_manifest)
    if (manifest["proposal_seeds"] != list(PROPOSAL_SEEDS)
            or not report["identity_comparison"]["exact"] or not report["original_rng_recovery_exact"]):
        raise ValueError("Require all five declared proposals and exact control audits.")
    labels = [f"seed_{seed}" for seed in PROPOSAL_SEEDS]
    if set(report["branches"]) != {"raw", "identity", *labels}:
        raise ValueError("Seed-control branch set differs from the declared five proposals.")
    raw = _load_trace(controls_dir, "raw", hashes)
    identity = _load_trace(controls_dir, "identity", hashes)
    _validate_trace_start(start, raw)
    if not all(item["exact"] for item in compare_fields(raw, identity, tuple(raw)).values()):
        raise ValueError("Saved identity control differs from raw trace.")
    if not np.array_equal(raw["actions"][0], start["baseline_action"]):
        raise ValueError("Saved raw action differs from original MPPI action.")
    policy_raw = _load_trace(policy_dir, "raw", policy_hashes)
    if not all(item["exact"] for item in compare_fields(raw, policy_raw, tuple(raw)).values()):
        raise ValueError("Policy and seed controls have different saved raw continuations.")
    policy_trace = _load_trace(policy_dir, "intervention", policy_hashes)
    _validate_trace_start(start, policy_trace)
    if not np.array_equal(policy_trace["actions"][0], start["candidate_action"]):
        raise ValueError("Saved policy first action differs from the frozen suggestion.")
    actions = {"raw": start["baseline_action"].copy(), "policy": start["candidate_action"].copy()}
    transitions = {"raw": first_transition(raw), "policy": first_transition(policy_trace)}
    for seed, label in zip(PROPOSAL_SEEDS, labels):
        trace = _load_trace(controls_dir, label, hashes)
        _validate_trace_start(start, trace)
        verify_pair(raw, trace, start["state"], start["warmstart"], start["snapshot"])
        proposal = report["proposals"][label]
        action = trace["actions"][0].copy()
        if (proposal["proposal_seed"] != seed
                or not np.array_equal(action, proposal["action"])
                or not np.array_equal(action, report["branches"][label]["first_action"])):
            raise ValueError("Saved proposal action/seed disagrees with executed control.")
        actions[label] = action
        transitions[label] = first_transition(trace)
    if any(action.shape != (6,) or not np.isfinite(action).all()
           or np.any(np.abs(action) > 1) for action in actions.values()):
        raise ValueError("A fixed first action is invalid.")
    return actions, {"controls_dir": str(controls_dir.resolve()),
                     "artifact_hashes_sha256": sha256(controls_dir / "artifact_hashes.json"),
                     "policy_dir": str(policy_dir.resolve()),
                     "policy_artifact_hashes_sha256": sha256(policy_dir / "artifact_hashes.json"),
                     "manifest": manifest}, transitions


def _distribution(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Cannot summarize empty/non-finite paired differences.")
    return {"n": len(data), "mean": float(data.mean()),
            "sample_sd": float(data.std(ddof=1)) if len(data) > 1 else None,
            "min": float(data.min()), "max": float(data.max()),
            "improved": int(np.sum(data < 0)), "tied": int(np.sum(data == 0)),
            "worsened": int(np.sum(data > 0))}


def summarize_pairs(rows: list[dict]) -> dict:
    """Describe repeat seeds within each fixed intervention; no pooled CI."""
    labels = list(dict.fromkeys(row["intervention"] for row in rows))
    result = {}
    for label in labels:
        group = [row for row in rows if row["intervention"] == label]
        result[label] = {
            "future_seeds": [row["future_seed"] for row in group],
            "score_horizon_delta": _distribution([row["score_horizon_delta"] for row in group]),
            "continuation_delta": _distribution([row["continuation_delta"] for row in group]),
            "candidate_falls_at_score_horizon": sum(row["candidate_window"]["fell"] for row in group),
            "raw_falls_at_score_horizon": sum(row["raw_window"]["fell"] for row in group),
            "additional_falls_at_score_horizon": sum(row["candidate_window"]["fell"] and not row["raw_window"]["fell"] for row in group),
            "prevented_falls_at_score_horizon": sum(row["raw_window"]["fell"] and not row["candidate_window"]["fell"] for row in group),
            "candidate_falls_at_continuation": sum(row["candidate_fell"] for row in group),
            "raw_falls_at_continuation": sum(row["raw_fell"] for row in group),
            "additional_falls_at_continuation": sum(row["candidate_fell"] and not row["raw_fell"] for row in group),
            "prevented_falls_at_continuation": sum(row["raw_fell"] and not row["candidate_fell"] for row in group),
        }
    return result


def summarize_raw(branches: dict) -> dict:
    """Raw MPPI variability across future seeds, distinct from paired effects."""
    group = [summary for name, summary in branches.items() if name.endswith("/raw")]
    if not group:
        return {}
    # These are absolute costs, so negative/positive counts are not improvements.
    def absolute(values):
        return {key: value for key, value in _distribution(values).items()
                if key in {"n", "mean", "sample_sd", "min", "max"}}
    return {"score_horizon_cost": absolute([row["score_window"]["fixed_horizon_task_cost"] for row in group]),
            "continuation_cost": absolute([row["fixed_horizon_task_cost"] for row in group]),
            "falls_at_score_horizon": sum(row["score_window"]["fell"] for row in group),
            "falls_at_continuation": sum(row["first_fall_step"] is not None for row in group)}


def _guard_output(out_dir: Path, input_dirs: tuple[Path, ...]) -> None:
    if any(out_dir.resolve().is_relative_to(directory.resolve()) for directory in input_dirs):
        raise ValueError("Output must be outside input evidence bundles.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(f"Output is not empty: {out_dir}")


def run_case(start: dict, actions: dict, out_dir: Path, *, future_seeds: tuple[int, ...],
             steps: int, score_horizon: int, expected_first_transitions: dict, progress=None) -> dict:
    """Run paired repeats; the public CLI fixes seeds, actions and horizons."""
    _guard_output(out_dir, (Path(start["shadow_dir"]),))
    if type(steps) is not int or steps < 1 or type(score_horizon) is not int or not 1 <= score_horizon <= steps:
        raise ValueError("Require positive steps and score_horizon <= steps.")
    if not future_seeds or len(set(future_seeds)) != len(future_seeds):
        raise ValueError("Require distinct future seeds.")
    for seed in future_seeds:
        future_snapshot(start["snapshot"], seed)
    if "raw" not in actions or not np.array_equal(actions["raw"], start["baseline_action"]):
        raise ValueError("Raw action must be the original MPPI action.")
    for label, action in actions.items():
        if not label or Path(label).name != label or label in {".", ".."}:
            raise ValueError("Intervention names must be single directory names.")
        if (np.asarray(action).shape != (6,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1)):
            raise ValueError("Invalid fixed first action.")
    # Retain immutable reference copies even if a called helper were changed.
    state, warmstart = start["state"].copy(), start["warmstart"].copy()
    snapshot = copy.deepcopy(start["snapshot"])
    fixed_actions = {label: action.copy() for label, action in actions.items()}
    if set(expected_first_transitions) != set(fixed_actions):
        raise ValueError("Require saved first-transition evidence for every fixed action.")
    expected = copy.deepcopy(expected_first_transitions)
    if any(not np.array_equal(fixed_actions[label], expected[label]["action"]) for label in fixed_actions):
        raise ValueError("First-transition evidence actions differ from fixed actions.")
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_trace(out_dir / "inputs.npz", {"state": state, "warmstart": warmstart,
                "original_post_nominal": snapshot["U"], "actions": np.stack(list(fixed_actions.values())),
                "action_labels": np.asarray(list(fixed_actions)),
                "expected_successor_states": np.stack([expected[label]["state"] for label in fixed_actions]),
                "expected_successor_warmstarts": np.stack([expected[label]["warmstart"] for label in fixed_actions]),
                "expected_first_costs": np.asarray([expected[label]["cost"] for label in fixed_actions])})
    write_json(out_dir / "manifest.json", {"decision_step": start["decision_step"],
        "policy_label": start["policy_label"], "original_post_call_snapshot": _json_snapshot(snapshot),
        "future_seeds": list(future_seeds), "steps": steps, "score_horizon": score_horizon,
        "task": start["task"], "mppi_config": asdict(start["cfg"]),
        "first_actions": {label: action.tolist() for label, action in fixed_actions.items()}})
    report = {"status": "running", "error": None, "completed_branches": 0,
              "requested_branches": len(actions) * len(future_seeds), "pairs": [],
              "branches": {}, "summary": {}, "raw_summary": {}, "original_nominal_only": True}
    started = time.perf_counter()

    def save(active=None):
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(out_dir / "report.json", report)
        write_json(out_dir / "status.json", {key: report[key] for key in
                   ("status", "error", "completed_branches", "requested_branches", "elapsed_seconds")} | {"active_branch": active})
        if progress is not None:
            progress(report["completed_branches"], active)

    try:
        save()
        for seed in future_seeds:
            shared = future_snapshot(snapshot, seed)
            seed_dir = out_dir / f"future_{seed}"
            seed_dir.mkdir()
            raw_trace, raw_summary, raw_window = None, None, None
            for label in ["raw"] + [name for name in fixed_actions if name != "raw"]:
                save(f"future_{seed}/{label}")
                trace, summary = run_branch(start["task"], start["cfg"], state.copy(), warmstart.copy(),
                    copy.deepcopy(shared), fixed_actions[label].copy(), steps)
                verify_pair(trace, trace, state, warmstart, shared)
                verify_first_transition(trace, expected[label])
                validate_branch_summary(trace, summary, start["task"], steps)
                if not np.array_equal(trace["actions"][0], fixed_actions[label]):
                    raise RuntimeError("A branch changed its fixed first action.")
                window = window_metrics(trace, summary, start["task"], score_horizon)
                if not window["complete_window_or_absorbing_tail"]:
                    raise RuntimeError("Branch did not complete the declared scoring window.")
                _save_trace(seed_dir / f"{label}.npz", trace)
                summary["score_window"] = window
                report["branches"][f"future_{seed}/{label}"] = summary
                report["completed_branches"] += 1
                if label == "raw":
                    raw_trace, raw_summary, raw_window = trace, summary, window
                else:
                    pairing = verify_pair(raw_trace, trace, state, warmstart, shared)
                    report["pairs"].append({
                        "future_seed": seed, "intervention": label,
                        "score_horizon_delta": window["fixed_horizon_task_cost"] - raw_window["fixed_horizon_task_cost"],
                        "continuation_delta": summary["fixed_horizon_task_cost"] - raw_summary["fixed_horizon_task_cost"],
                        "candidate_window": window, "raw_window": raw_window,
                        "candidate_fell": summary["first_fall_step"] is not None,
                        "raw_fell": raw_summary["first_fall_step"] is not None,
                        "candidate_first_fall_step": summary["first_fall_step"],
                        "raw_first_fall_step": raw_summary["first_fall_step"],
                        "candidate_executed_steps": summary["completed_steps"],
                        "raw_executed_steps": raw_summary["completed_steps"],
                        "action_l2_difference": float(np.linalg.norm(fixed_actions[label] - fixed_actions["raw"])),
                        "saved_first_transition_exact": True,
                        "shared_future_rng_prefix": pairing})
                save()
        report["summary"] = summarize_pairs(report["pairs"])
        report["raw_summary"] = summarize_raw(report["branches"])
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
        write_json(out_dir / "artifact_hashes.json", {str(path.relative_to(out_dir)): sha256(path)
                   for path in sorted(out_dir.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"})
    return report


def main(out_dir: Path, preregistration: Path, controls_root: Path = DEFAULT_CONTROLS,
         shadow_root: Path = DEFAULT_SHADOW) -> None:
    """Run the fixed 4 cases × 8 future seeds × 7 actions, with original U only."""
    _guard_output(out_dir, (controls_root, shadow_root))
    if not preregistration.is_file() or not preregistration.read_text().strip():
        raise ValueError("A nonempty preregistration file is required before any run.")
    loaded = {}
    # Validate every fixed input before producing any experimental outcome.
    for episode, label, step in CASES:
        name = f"{episode}_{label}_{step}"
        start = load_decision(shadow_root / episode, label, step)
        actions, metadata, transitions = load_actions(start, controls_root / name, shadow_root / "branches" / name)
        if start["cfg"].H != SCORE_HORIZON:
            raise ValueError("The declared 48-control scoring horizon differs from the archived planner.")
        loaded[name] = (start, actions, metadata, transitions)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = FROZEN_SOURCES + (
        "src/utils/trace_replay.py", "src/mppi/gate_v2.py",
        "scripts/runners/eval_policy_walker2d.py",
        "scripts/diagnostics/walker_gate_counterfactual.py",
        "scripts/diagnostics/walker_action_controls.py",
        "scripts/diagnostics/walker_gate_v2_diagnostic.py",
        "scripts/diagnostics/walker_future_seed_check.py")
    fingerprints = {relative: sha256(_ROOT / relative) for relative in sources}
    for relative in sources:
        destination = out_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, destination)
    shutil.copyfile(preregistration, out_dir / "preregistration.md")
    write_json(out_dir / "manifest.json", {
        "future_seeds": list(FUTURE_SEEDS), "fixed_proposal_seeds": list(PROPOSAL_SEEDS),
        "steps": STEPS, "score_horizon": SCORE_HORIZON,
        "requested_branches": len(CASES) * len(FUTURE_SEEDS) * (2 + len(PROPOSAL_SEEDS)),
        "preregistration": str(preregistration.resolve()), "preregistration_sha256": sha256(preregistration),
        "cases": {name: metadata for name, (_, _, metadata, _) in loaded.items()},
        "provenance": {"python": platform.python_version(),
                       "packages": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
                       "source_sha256": fingerprints},
        "intervention": "Fixed original first actions, original post-call U and refinement flag. Change only future RNG; pair raw/intervention within each future seed. No nominal-mode choice based on the preceding ablation.",
        "limits": "Eight repeated future seeds at four fixed states; not independent episodes or a population CI. Descriptive within-intervention sample SD and signs only. No policy training or active gate."})
    reports, status, error = {}, "running", None
    finished_branches = 0
    total_completed = 0

    def progress(name, current, active):
        nonlocal total_completed
        total_completed = max(total_completed, finished_branches + current)
        write_json(out_dir / "status.json", {"status": status, "error": error,
            "completed_cases": list(reports), "completed_branches": total_completed,
            "requested_branches": len(CASES) * len(FUTURE_SEEDS) * (2 + len(PROPOSAL_SEEDS)),
            "active_case": name, "active_branch": active})

    try:
        for name, (start, actions, _, transitions) in loaded.items():
            reports[name] = run_case(start, actions, out_dir / name, future_seeds=FUTURE_SEEDS,
                steps=STEPS, score_horizon=SCORE_HORIZON, expected_first_transitions=transitions,
                progress=lambda current, active, case=name: progress(case, current, active))
            finished_branches += reports[name]["completed_branches"]
        status = "complete"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out_dir / "report.json", {"status": status, "error": error,
            "cases": {name: {"status": result["status"], "summary": result["summary"],
                             "raw_summary": result["raw_summary"]}
                      for name, result in reports.items()},
            "original_nominal_only": True, "no_pooled_confidence_interval": True})
        progress(None, 0, None)
        write_json(out_dir / "artifact_hashes.json", {str(path.relative_to(out_dir)): sha256(path)
                   for path in sorted(out_dir.rglob("*")) if path.is_file() and path != out_dir / "artifact_hashes.json"})


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
