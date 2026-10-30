"""Ablate continuation nominal memory without changing action or future RNG.

For each archived seed action, reuse its original-U branch and run one branch
with only U replaced by that proposal's shifted plan. This is not an active
policy gate, a new action proposal, or a change to the solver warmstart.
"""

import copy
import json
import platform
import shutil
import sys
import time
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
from src.utils.trace_replay import compare_fields, decision_snapshot


def matched_snapshot(original: dict, proposal: dict) -> dict:
    """Replace only nominal U; never inherit the proposal's advanced RNG."""
    result = copy.deepcopy(original)
    result["U"] = np.asarray(proposal["discarded_post_call_snapshot"]["U"], dtype=float).copy()
    return result


def verify_pair(original: dict, matched: dict) -> dict:
    """Changing nominal memory must not change the first physical transition."""
    first = ("states", "warmstarts", "actions", "costs")
    reference = {k: original[k][:2 if k in {"states", "warmstarts"} else 1] for k in first}
    actual = {k: matched[k][:2 if k in {"states", "warmstarts"} else 1] for k in first}
    first_fields = compare_fields(reference, actual, first)
    if not all(row["exact"] for row in first_fields.values()):
        raise RuntimeError("Nominal ablation changed the first action or physical transition.")
    count = min(len(original["actions"]), len(matched["actions"]))
    rng_fields = ("planner_rng_states", "post_planner_rng_states", "initial_plan_flags", "post_initial_plan_flags", "planned_here")
    rng = compare_fields({k: original[k][:count] for k in rng_fields},
                         {k: matched[k][:count] for k in rng_fields}, rng_fields)
    if not all(row["exact"] for row in rng.values()):
        raise RuntimeError("Nominal ablation changed the common future planner random stream.")
    return {"first_transition_exact": True, "first_transition_fields": first_fields,
            "future_rng_prefix_exact": True, "future_rng_checked_steps": count, "future_rng_fields": rng}


def load_controls(directory: Path) -> dict:
    """Validate completed, hashed inputs and the proposal's shifted-plan identity."""
    manifest, report, hashes = read_bundle(directory)
    required = {"proposals.json", "identity.npz", "original_rng_recovery.json"}
    required.update(f"seed_{seed}.npz" for seed in range(31000, 31005))
    if not required.issubset(hashes):
        raise ValueError("Missing hashed proposal, recovery or continuation evidence.")
    if (manifest["steps"] != 250 or manifest["proposal_seeds"] != list(range(31000, 31005))
            or not report["identity_comparison"]["exact"] or not report["original_rng_recovery_exact"]):
        raise ValueError("Require the declared five-seed, 250-control audited experiment.")
    start = load_decision(Path(manifest["shadow_dir"]), manifest["policy_label"], manifest["decision_step"])
    _validate_start(start, manifest)
    if start["cfg"].H != 48:
        raise ValueError("This preregistered ablation requires the frozen H=48 task.")
    pre = _json_snapshot(decision_snapshot(start["trace"], start["decision_step"]))
    if manifest["pre_call_snapshot"] != pre:
        raise ValueError("Original pre-call snapshot differs from the shadow decision.")
    recovery = json.loads((directory / "original_rng_recovery.json").read_text())
    if (recovery["pre_call_snapshot"] != pre
            or recovery["discarded_post_call_snapshot"] != _json_snapshot(start["snapshot"])
            or not np.array_equal(recovery["action"], start["baseline_action"])
            or not np.array_equal(recovery["optimized_plan"], start["trace"]["plans"][start["decision_step"]])):
        raise ValueError("Archived original-RNG recovery is inconsistent.")
    traces = {name: _load_trace(directory, name, hashes)
              for name in ("raw", "identity", *(f"seed_{seed}" for seed in manifest["proposal_seeds"]))}
    if not all(row["exact"] for row in compare_fields(traces["raw"], traces["identity"], tuple(traces["raw"])).values()):
        raise ValueError("Archived identical-action control is not exact.")
    proposals = json.loads((directory / "proposals.json").read_text())
    if set(proposals) != {f"seed_{seed}" for seed in manifest["proposal_seeds"]}:
        raise ValueError("Proposal labels differ from the declared seeds.")
    for name, trace in traces.items():
        _validate_trace_start(start, trace)
        if name in {"raw", "identity"} and not np.array_equal(trace["actions"][0], start["baseline_action"]):
            raise ValueError("Archived raw action differs from the original baseline action.")
        if not np.array_equal(trace["post_plan_nominals"][0], start["snapshot"]["U"]):
            raise ValueError("Archived branch changed nominal memory during its first action.")
        summary = report["branches"][name]
        count = len(trace["actions"])
        healthy = ((trace["states"][1:, 2] > .8) & (trace["states"][1:, 2] < 2.)
                   & (trace["states"][1:, 3] > -1.) & (trace["states"][1:, 3] < 1.))
        if (count < 1 or len(trace["states"]) != count + 1 or len(trace["warmstarts"]) != count + 1
                or not healthy[:-1].all() or summary["completed_steps"] != count
                or summary["first_fall_step"] != (None if healthy[-1] else count)
                or (healthy[-1] and count != 250)):
            raise ValueError("Archived termination or prefix length disagrees with the physical trace.")
        for horizon in (48, 250):
            metric = window_metrics(trace, summary, start["task"], horizon)
            expected = summary["window48"]["fixed_horizon_task_cost"] if horizon == 48 else summary["fixed_horizon_task_cost"]
            if not metric["complete_window_or_absorbing_tail"] or metric["fixed_horizon_task_cost"] != expected:
                raise ValueError("Archived continuation metrics disagree with its trace.")
    for seed in manifest["proposal_seeds"]:
        name = f"seed_{seed}"
        proposal = proposals[name]
        plan = np.asarray(proposal["optimized_plan"], dtype=float)
        action = np.asarray(proposal["action"], dtype=float)
        after = proposal["discarded_post_call_snapshot"]
        before = proposal["pre_call_snapshot"]
        # Frozen Walker actuators use six normalized controls in [-1, 1].
        if (plan.shape != (48, 6) or not np.isfinite(plan).all()
                or (np.abs(plan) > 1).any() or action.shape != (6,)
                or not np.array_equal(plan[0], action)
                or not np.array_equal(action, traces[name]["actions"][0])
                or not np.array_equal(after["U"], np.concatenate((plan[1:], plan[-1:])))
                or after["initial_plan"] is not False):
            raise ValueError(f"Proposal action or shifted nominal is inconsistent: {name}")
        if (proposal["proposal_seed"] != seed or before["U"] != pre["U"]
                or before["initial_plan"] != pre["initial_plan"]
                or before["rng_state"] != np.random.default_rng(seed).bit_generator.state):
            raise ValueError(f"Proposal changed more than its declared random stream: {name}")
        expected_iterations = start["cfg"].initial_iterations if pre["initial_plan"] else 1
        if proposal["optimization_iterations"] != expected_iterations:
            raise ValueError("Proposal refinement count differs from the frozen planner.")
    return {"start": start, "manifest": manifest, "report": report, "hashes": hashes,
            "proposals": proposals, "traces": traces}


def _metrics(trace: dict, summary: dict, task: dict) -> dict:
    return {str(h): window_metrics(trace, summary, task, h) for h in (48, 250)}


def _deltas(left: dict, right: dict) -> dict:
    return {h: {"task_cost": left[h]["fixed_horizon_task_cost"] - right[h]["fixed_horizon_task_cost"],
                "additional_fall": left[h]["fell"] and not right[h]["fell"],
                "prevented_fall": right[h]["fell"] and not left[h]["fell"],
                "velocity_mae_executed_prefix": left[h]["velocity_mae"] - right[h]["velocity_mae"]}
            for h in ("48", "250")}


def summarize(branches: dict) -> dict:
    """Descriptive paired deltas for this state; no inferential statistics."""
    result = {}
    for contrast in ("matched_minus_original", "matched_minus_raw"):
        result[contrast] = {}
        for horizon in ("48", "250"):
            rows = [branch[contrast][horizon] for branch in branches.values()]
            costs = np.asarray([row["task_cost"] for row in rows])
            result[contrast][horizon] = {
                "n": len(rows), "lower_cost": int(np.count_nonzero(costs < 0)),
                "higher_cost": int(np.count_nonzero(costs > 0)), "ties": int(np.count_nonzero(costs == 0)),
                "min": float(costs.min()) if len(rows) else None,
                "median": float(np.median(costs)) if len(rows) else None,
                "max": float(costs.max()) if len(rows) else None,
                "additional_falls": sum(row["additional_fall"] for row in rows),
                "prevented_falls": sum(row["prevented_fall"] for row in rows),
            }
    return result


def main(controls_dir: Path, out_dir: Path, preregistration: Path) -> None:
    """Run the five declared nominal-memory ablations for one archived case."""
    controls_dir, out_dir, preregistration = map(Path, (controls_dir, out_dir, preregistration))
    if out_dir.resolve().is_relative_to(controls_dir.resolve().parent):
        raise ValueError("Output must be outside the audited controls root.")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise FileExistsError(out_dir)
    if not preregistration.is_file() or not preregistration.read_text().strip():
        raise ValueError("A nonempty preregistration file is required before simulation.")
    inputs = load_controls(controls_dir)
    start, source_report = inputs["start"], inputs["report"]
    if (out_dir.resolve().is_relative_to(Path(start["shadow_dir"]).resolve().parent)
            or preregistration.resolve().is_relative_to(out_dir.resolve())):
        raise ValueError("Output must be outside audited shadow inputs and cannot contain the preregistration.")
    out_dir.mkdir(parents=True, exist_ok=True)
    archived = out_dir / "inputs"
    archived.mkdir()
    for relative in ("manifest.json", "report.json", "artifact_hashes.json", "proposals.json",
                     "original_rng_recovery.json", "raw.npz", "identity.npz",
                     *(f"seed_{seed}.npz" for seed in inputs["manifest"]["proposal_seeds"])):
        shutil.copyfile(controls_dir / relative, archived / relative)
    shutil.copyfile(preregistration, archived / "preregistration.md")
    sources = FROZEN_SOURCES + (
        "src/utils/trace_replay.py", "src/mppi/gate_v2.py",
        "scripts/runners/eval_policy_walker2d.py", "scripts/diagnostics/walker_gate_counterfactual.py",
        "scripts/diagnostics/walker_action_controls.py", "scripts/diagnostics/walker_gate_v2_diagnostic.py",
        "scripts/diagnostics/walker_nominal_ablation.py",
    )
    for relative in sources:
        target = out_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)
    write_json(out_dir / "manifest.json", {
        "controls_dir": str(controls_dir.resolve()), "controls_hash_manifest_sha256": sha256(controls_dir / "artifact_hashes.json"),
        "preregistration": str(preregistration.resolve()), "preregistration_sha256": sha256(preregistration),
        "input_manifest": inputs["manifest"], "original_post_call_snapshot": _json_snapshot(start["snapshot"]),
        "steps": 250, "measurement_horizons": [48, 250],
        "intervention": "Only continuation U changes to proposal shifted plan. First action, initial physical state, initial solver warmstart, original future planner RNG and refinement flag stay fixed.",
        "provenance": {"python": platform.python_version(),
                       "packages": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
                       "source_sha256": {relative: sha256(_ROOT / relative) for relative in sources}},
    })
    report = {"status": "running", "error": None, "raw_replay_comparison": None, "branches": {},
              "primary_comparison": "matched nominal minus the same action with original nominal",
              "limits": "Development ablation on preselected seed actions, not a policy gate or population qualification. Negative cost difference favors matched nominal; MAE covers executed prefixes only."}
    begun = time.perf_counter()

    def progress(active=None):
        report["elapsed_seconds"] = time.perf_counter() - begun
        report["summary"] = summarize(report["branches"])
        write_json(out_dir / "report.json", report)
        write_json(out_dir / "status.json", {"status": report["status"], "error": report["error"],
                   "active_branch": active, "completed_branches": list(report["branches"]),
                   "elapsed_seconds": report["elapsed_seconds"]})

    try:
        progress("raw_replay")
        raw, raw_summary = run_branch(start["task"], start["cfg"], start["state"], start["warmstart"],
                                      copy.deepcopy(start["snapshot"]), start["baseline_action"].copy(), 250)
        _save_trace(out_dir / "raw_replay.npz", raw)
        fields = compare_fields(inputs["traces"]["raw"], raw, tuple(raw))
        report["raw_replay_comparison"] = {"exact": all(row["exact"] for row in fields.values()), "fields": fields}
        if not report["raw_replay_comparison"]["exact"]:
            raise RuntimeError("Raw/original-U replay is not exact across all trace fields.")
        raw_metrics = _metrics(raw, raw_summary, start["task"])
        for name, proposal in inputs["proposals"].items():
            progress(name)
            snapshot = matched_snapshot(start["snapshot"], proposal)
            action = np.asarray(proposal["action"], dtype=float)
            matched, summary = run_branch(start["task"], start["cfg"], start["state"], start["warmstart"],
                                          snapshot, action.copy(), 250)
            _save_trace(out_dir / f"{name}.npz", matched)
            checks = verify_pair(inputs["traces"][name], matched)
            if (not np.array_equal(matched["pre_plan_nominals"][0], snapshot["U"])
                    or not np.array_equal(matched["post_plan_nominals"][0], snapshot["U"])
                    or bool(matched["post_initial_plan_flags"][0])):
                raise RuntimeError("Matched branch did not use the declared proposal nominal.")
            matched_metrics = _metrics(matched, summary, start["task"])
            original_metrics = _metrics(inputs["traces"][name], source_report["branches"][name], start["task"])
            report["branches"][name] = {
                "proposal_seed": proposal["proposal_seed"], "action": action.tolist(),
                "start_snapshot": _json_snapshot(snapshot), "checks": checks, "matched_summary": summary,
                "matched_metrics": matched_metrics, "original_metrics": original_metrics, "raw_metrics": raw_metrics,
                "matched_minus_original": _deltas(matched_metrics, original_metrics),
                "matched_minus_raw": _deltas(matched_metrics, raw_metrics),
                "original_minus_raw": _deltas(original_metrics, raw_metrics),
            }
            progress()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        progress()
        write_json(out_dir / "artifact_hashes.json", {
            str(path.relative_to(out_dir)): sha256(path) for path in sorted(out_dir.rglob("*"))
            if path.is_file() and path != out_dir / "artifact_hashes.json"})


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
