"""Compare fixed v2 predictions with saved first-action continuations.

No controller executes here: the four policy outcomes and twenty seed-control
outcomes are read-only calibration data. Only isolated scoring probes run.
"""

import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics.walker_action_controls import window_metrics
from scripts.diagnostics.walker_gate_counterfactual import (
    FROZEN_SOURCES, _json_snapshot, _save_trace, load_decision,
)
from scripts.runners.eval_policy_walker2d import sha256, write_json
from src.mppi.gate_v2 import ContinuationGate
from src.utils.trace_replay import compare_fields

DEFAULT_SHADOW = _ROOT / "runs/2026-10-11_s3_shadow_001"
DEFAULT_CONTROLS = _ROOT / "runs/2026-10-12_s3_action_controls_001"


def read_bundle(directory: Path) -> tuple[dict, dict, dict]:
    """Reject incomplete, changed or path-escaping evidence before scoring."""
    hashes = json.loads((directory / "artifact_hashes.json").read_text())
    if not {"manifest.json", "report.json", "raw.npz"}.issubset(hashes):
        raise ValueError("Evidence bundle lacks required hashed files.")
    for relative, expected in hashes.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or Path(relative).is_absolute():
            raise ValueError("Evidence path escapes its bundle.")
        if sha256(path) != expected:
            raise ValueError(f"Evidence hash mismatch: {path}")
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "complete" or not report["raw_reference_comparison"]["exact"]:
        raise ValueError("Require complete, exact raw-continuation evidence.")
    for relative, expected in manifest["provenance"]["source_sha256"].items():
        if hashes.get(f"source/{relative}") != expected:
            raise ValueError(f"Archived source disagrees with manifest: {relative}")
    for relative in FROZEN_SOURCES:
        expected = manifest["provenance"]["source_sha256"].get(relative)
        if expected != sha256(_ROOT / relative):
            raise ValueError(f"Frozen source differs from continuation evidence: {relative}")
    return manifest, report, hashes


def _load_trace(directory: Path, label: str, hashes: dict) -> dict:
    relative = f"{label}.npz"
    if relative not in hashes:
        raise ValueError(f"Unhashed continuation trace: {relative}")
    with np.load(directory / relative, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def prediction_agreement(predicted: float | None, actual: float) -> str:
    if predicted is None:
        return "invalid"
    if predicted == 0 or actual == 0:
        return "tie"
    return "same_sign" if (predicted < 0) == (actual < 0) else "opposite_sign"


def _verify_probe(score, trace: dict) -> None:
    """The probe must reproduce the actual next call, not just its first state."""
    if not (np.array_equal(score.action, trace["actions"][0])
            and np.array_equal(score.successor_state, trace["states"][1])
            and np.array_equal(score.successor_warmstart, trace["warmstarts"][1])
            and score.immediate_cost == trace["costs"][0]):
        raise RuntimeError("Scoring probe differs from saved first transition.")
    if score.immediate_fell:
        if len(trace["actions"]) != 1 or score.optimized_plan is not None:
            raise RuntimeError("Immediate-fall probe or saved continuation is inconsistent.")
    else:
        if len(trace["actions"]) < 2:
            raise ValueError("Need a second recorded action to verify the successor planner.")
        after = score.planner_snapshot_after
        if not (np.array_equal(score.optimized_plan[0], trace["actions"][1])
                and np.array_equal(after["U"], trace["post_plan_nominals"][1])
                and after["rng_state"] == json.loads(str(trace["post_planner_rng_states"][1]))
                and after["initial_plan"] == bool(trace["post_initial_plan_flags"][1])):
            raise RuntimeError("Scoring probe successor planner differs from saved continuation.")


def _save_probe(directory: Path, name: str, score) -> dict | None:
    if score is None:
        return None
    arrays = {key: getattr(score, key) for key in (
        "action", "successor_state", "successor_warmstart", "scored_plan", "continuation_states")}
    arrays["optimized_plan"] = (np.empty((0, len(score.action))) if score.optimized_plan is None
                                else score.optimized_plan)
    _save_trace(directory / f"{name}.npz", arrays)
    return {"cost": score.cost, "fell": score.fell,
            "immediate_cost": score.immediate_cost, "immediate_fell": score.immediate_fell,
            "continuation_cost": score.continuation_cost,
            "planner_snapshot_after": None if score.planner_snapshot_after is None else
                                      _json_snapshot(score.planner_snapshot_after)}


def _validate_start(start: dict, manifest: dict) -> None:
    if (manifest["decision_step"] != start["decision_step"]
            or manifest["policy_label"] != start["policy_label"]
            or manifest["shadow_artifact_hashes_sha256"] != start["artifact_hashes_sha256"]):
        raise ValueError("Continuation bundle belongs to another decision or shadow revision.")
    if manifest["task"] != start["task"] or manifest["mppi_config"] != asdict(start["cfg"]):
        raise ValueError("Continuation task or planner configuration differs.")
    snapshot = manifest.get("original_post_call_snapshot", manifest.get("post_call_snapshot"))
    if snapshot != _json_snapshot(start["snapshot"]):
        raise ValueError("Continuation planner snapshot differs from the original post-call snapshot.")


def _validate_trace_start(start: dict, trace: dict) -> None:
    if not (np.array_equal(trace["states"][0], start["state"])
            and np.array_equal(trace["warmstarts"][0], start["warmstart"])
            and np.array_equal(trace["pre_plan_nominals"][0], start["snapshot"]["U"])
            and json.loads(str(trace["planner_rng_states"][0])) == start["snapshot"]["rng_state"]
            and not bool(trace["initial_plan_flags"][0])
            and not bool(trace["planned_here"][0])):
        raise ValueError("Saved continuation starts at a different simulator/planner boundary.")


def score_case(start: dict, controls_dir: Path, policy_dir: Path, out_dir: Path) -> dict:
    """Score known policy, identity and seed actions without re-executing them."""
    for directory in (controls_dir, policy_dir, Path(start["shadow_dir"])):
        if out_dir.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Output must be outside all input evidence bundles.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(out_dir)
    cm, cr, ch = read_bundle(controls_dir)
    pm, pr, ph = read_bundle(policy_dir)
    for manifest in (cm, pm):
        _validate_start(start, manifest)
    if not cr["identity_comparison"]["exact"] or not cr["original_rng_recovery_exact"]:
        raise ValueError("Seed-control identity or proposal recovery did not pass.")
    if cm["steps"] != pm["steps"]:
        raise ValueError("Policy and control continuation durations differ.")
    raw = _load_trace(controls_dir, "raw", ch)
    prior_raw = _load_trace(policy_dir, "raw", ph)
    if not all(r["exact"] for r in compare_fields(raw, prior_raw, tuple(raw)).values()):
        raise ValueError("Policy and seed controls have different raw baselines.")
    _validate_trace_start(start, raw)
    raw_summary = cr["branches"]["raw"]
    raw_window = window_metrics(raw, raw_summary, start["task"], start["cfg"].H)
    if not raw_window["complete_window_or_absorbing_tail"]:
        raise ValueError("Raw continuation is too short for the scoring horizon.")
    options = [("policy", "policy", policy_dir, "intervention", ph, pr["branches"]["intervention"])]
    options += [(label, "identity" if label == "identity" else "seed_control",
                 controls_dir, label, ch, summary) for label, summary in cr["branches"].items() if label != "raw"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    with ContinuationGate(start["task"], start["cfg"]) as gate:
        for name, category, directory, label, hashes, summary in options:
            trace = _load_trace(directory, label, hashes)
            _validate_trace_start(start, trace)
            action = trace["actions"][0]
            if category == "policy" and not np.array_equal(action, start["candidate_action"]):
                raise ValueError("Saved policy action differs from the frozen shadow candidate.")
            decision = gate.decide(start["state"], start["warmstart"], start["snapshot"],
                                   start["baseline_action"], action)
            _verify_probe(decision.baseline_score, raw)
            if decision.candidate_score is not None:
                _verify_probe(decision.candidate_score, trace)
            window = window_metrics(trace, summary, start["task"], start["cfg"].H)
            if not window["complete_window_or_absorbing_tail"]:
                raise ValueError("Candidate continuation is too short for the scoring horizon.")
            predicted = None if decision.policy_cost is None else decision.policy_cost - decision.baseline_cost
            actual = window["fixed_horizon_task_cost"] - raw_window["fixed_horizon_task_cost"]
            later = summary["fixed_horizon_task_cost"] - raw_summary["fixed_horizon_task_cost"]
            actual_accept = actual <= 0 and (not window["fell"] or raw_window["fell"])
            probe_dir = out_dir / name
            probe_dir.mkdir()
            row = {"name": name, "category": category, "accepted": decision.accepted,
                   "reason": decision.reason, "predicted_delta": predicted,
                   "actual_score_horizon_delta": actual, "actual_continuation_delta": later,
                   "score_horizon": start["cfg"].H, "continuation_controls": cm["steps"],
                   "actual_horizon_accepts": actual_accept,
                   "false_acceptance_at_score_horizon": decision.accepted and not actual_accept,
                   "false_rejection_at_score_horizon": not decision.accepted and actual_accept,
                   "horizon_sign_agreement": prediction_agreement(predicted, actual),
                   "continuation_sign_agreement": prediction_agreement(predicted, later),
                   "action_l2_difference": float(np.linalg.norm(action - start["baseline_action"])),
                   "actual_window": window, "raw_window": raw_window,
                   "actual_completion": summary["completed_without_fall"],
                   "raw_completion": raw_summary["completed_without_fall"],
                   "baseline_probe": _save_probe(probe_dir, "baseline", decision.baseline_score),
                   "candidate_probe": _save_probe(probe_dir, "candidate", decision.candidate_score)}
            if category == "policy":
                row["v1_predicted_delta"] = start["decision"]["policy_cost"] - start["decision"]["baseline_cost"]
            if category == "identity" and (predicted != 0 or actual != 0 or later != 0 or not decision.accepted):
                raise RuntimeError("The identical-action scoring control did not tie exactly.")
            rows.append(row)
    report = {"status": "complete", "calibration_only": True,
              "decision_step": start["decision_step"], "policy_label": start["policy_label"],
              "elapsed_seconds": time.perf_counter() - started, "rows": rows,
              "input_hashes": {"controls": sha256(controls_dir / "artifact_hashes.json"),
                               "policy_continuations": sha256(policy_dir / "artifact_hashes.json")},
              "successor_replay_exact": True}
    write_json(out_dir / "report.json", report)
    return report


def summarize(rows: list[dict]) -> dict:
    return {category: {"n": len(group), "accepted": sum(r["accepted"] for r in group),
                       "false_acceptances_at_score_horizon": sum(r["false_acceptance_at_score_horizon"] for r in group),
                       "false_rejections_at_score_horizon": sum(r["false_rejection_at_score_horizon"] for r in group),
                       **{field: {value: sum(r[field] == value for r in group)
                                  for value in ("same_sign", "opposite_sign", "tie", "invalid")}
                          for field in ("horizon_sign_agreement", "continuation_sign_agreement")}}
            for category in ("policy", "seed_control", "identity")
            if (group := [r for r in rows if r["category"] == category])}


def main(out_dir: Path, controls_root: Path = DEFAULT_CONTROLS,
         shadow_root: Path = DEFAULT_SHADOW,
         selection: Path = DEFAULT_SHADOW / "branch_selection.json") -> None:
    """Read audited continuations and evaluate v2; never execute a coupled policy."""
    if out_dir.resolve().is_relative_to(shadow_root.resolve()) or out_dir.resolve().is_relative_to(controls_root.resolve()):
        raise ValueError("Output must be outside the input evidence roots.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(out_dir)
    selected = json.loads(selection.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = FROZEN_SOURCES + (
        "src/mppi/gate_v2.py", "src/utils/trace_replay.py",
        "scripts/diagnostics/walker_gate_counterfactual.py",
        "scripts/diagnostics/walker_action_controls.py",
        "scripts/diagnostics/walker_gate_v2_diagnostic.py",
        "scripts/runners/eval_policy_walker2d.py", "docs/gate_v2.md")
    fingerprints = {relative: sha256(_ROOT / relative) for relative in sources}
    for relative in sources:
        destination = out_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, destination)
    write_json(out_dir / "manifest.json", {"selection": selected, "selection_sha256": sha256(selection),
        "controls_root": str(controls_root.resolve()), "shadow_root": str(shadow_root.resolve()),
        "source_sha256": fingerprints, "tau": 0.0, "score": "one action + H-1 successor-plan actions"})
    rows, cases = [], {}
    status, error = "failed", None
    try:
        for case in selected["branches"]:
            label, step = case["policy_label"], case["decision_step"]
            name = f"{case['episode']}_{label}_{step}"
            start = load_decision(shadow_root / case["episode"], label, step)
            report = score_case(start, controls_root / name, shadow_root / "branches" / name, out_dir / name)
            cases[name] = report
            rows.extend({**row, "case": name} for row in report["rows"])
            write_json(out_dir / "status.json", {"status": "running", "completed_cases": list(cases)})
            print(f"scored {name}: successor replay exact", flush=True)
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = {"status": status, "error": error, "calibration_only": True,
                  "cases": cases, "summary": summarize(rows),
                  "limits": "Known policy outcomes plus declared seed controls; not held-out validation. One replan with open-loop tail remains an approximation; sign agreement at H does not guarantee 250-control benefit."}
        write_json(out_dir / "report.json", report)
        write_json(out_dir / "status.json", {"status": status, "completed_cases": list(cases), "error": error})
        hashes = {str(path.relative_to(out_dir)): sha256(path)
                  for path in sorted(out_dir.rglob("*")) if path.is_file()}
        write_json(out_dir / "artifact_hashes.json", hashes)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
