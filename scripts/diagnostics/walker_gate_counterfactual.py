"""S3 single-intervention branches from an audited shadow decision.

Both branches reuse the planner snapshot AFTER the original planning call.
Only their first action differs; every subsequent action comes from raw MPPI.
This is a development diagnostic, not an active-gate or non-inferiority study.
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

from scripts.runners.eval_policy_walker2d import sha256, write_json
from src.envs.walker2d import Walker2d
from src.mppi.gate import gate_accepts
from src.mppi.mppi import MPPI
from src.utils.config import MPPIConfig
from src.utils.trace_replay import compare_fields

FROZEN_SOURCES = (
    "src/envs/base.py", "src/envs/walker2d.py", "src/envs/mujoco_env.py",
    "assets/walker2d.xml", "src/mppi/mppi.py", "src/mppi/gate.py",
    "src/utils/config.py", "src/utils/math.py",
)
POST_FIELDS = ("post_plan_nominals", "post_planner_rng_states", "post_initial_plan_flags")
COMPARISON_FIELDS = ("states", "warmstarts", "actions", "costs") + POST_FIELDS


def _artifact_path(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or Path(relative).is_absolute():
        raise ValueError(f"Artifact path escapes its directory: {relative}")
    return path


def load_decision(shadow_dir: Path, policy_label: str, decision_step: int) -> dict:
    """Validate a completed shadow bundle before extracting a branch point."""
    if isinstance(decision_step, bool) or not isinstance(decision_step, int) or decision_step < 0:
        raise ValueError("decision_step must be a nonnegative integer.")
    if not policy_label or Path(policy_label).name != policy_label:
        raise ValueError("policy_label must be a single directory name.")
    shadow_dir = Path(shadow_dir)
    hashes = json.loads((shadow_dir / "artifact_hashes.json").read_text())
    required = {"manifest.json", "report.json", "traces.npz", "inputs/task.json", "inputs/mppi.json",
                f"{policy_label}/decisions.json", f"{policy_label}/candidates.npz"}
    if not required.issubset(hashes):
        raise ValueError(f"Missing hashed shadow inputs: {sorted(required - hashes.keys())}")
    for relative, expected in hashes.items():
        path = _artifact_path(shadow_dir, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Shadow artifact hash mismatch: {relative}")

    manifest = json.loads((shadow_dir / "manifest.json").read_text())
    report = json.loads((shadow_dir / "report.json").read_text())
    if (report.get("status") != "complete" or report.get("reference_trace_exact") is not True
            or report.get("shadow_only") is not True or report.get("tau") != 0.0):
        raise ValueError("Counterfactuals require a completed, exact, tau=0 shadow audit.")
    if policy_label not in manifest["selection"]["checkpoints"]:
        raise ValueError("Policy is absent from the frozen selection.")
    task_path, config_path = shadow_dir / "inputs/task.json", shadow_dir / "inputs/mppi.json"
    if sha256(task_path) != manifest["task_sha256"] or sha256(config_path) != manifest["config_sha256"]:
        raise ValueError("Archived task/config bytes disagree with the shadow manifest.")
    task = json.loads(task_path.read_text())
    cfg = MPPIConfig(**json.loads(config_path.read_text()))
    if task != manifest["task"] or asdict(cfg) != manifest["mppi_config"]:
        raise ValueError("Parsed task/config disagrees with the shadow manifest.")
    sources = manifest["provenance"]["source_sha256"]
    for relative, expected in sources.items():
        archived = f"source/{relative}"
        if hashes.get(archived) != expected:
            raise ValueError(f"Archived source hash mismatch: {relative}")
    for relative in FROZEN_SOURCES:
        if sources.get(relative) != sha256(_ROOT / relative):
            raise ValueError(f"Current frozen source differs from shadow source: {relative}")
    if manifest["provenance"]["python"] != platform.python_version():
        raise ValueError("Python version differs from the shadow run.")
    for package, expected in manifest["provenance"]["packages"].items():
        if version(package) != expected:
            raise ValueError(f"Package version differs from shadow run: {package}")

    with np.load(shadow_dir / "traces.npz", allow_pickle=False) as data:
        trace = {key: data[key].copy() for key in data.files}
    with np.load(shadow_dir / policy_label / "candidates.npz", allow_pickle=False) as data:
        candidate_plans, candidate_steps = data["plans"].copy(), data["steps"].copy()
    records = json.loads((shadow_dir / policy_label / "decisions.json").read_text())["decisions"]
    count = len(trace["actions"])
    if not 0 <= decision_step < count:
        raise ValueError("decision_step is outside the saved shadow episode.")
    if (not np.array_equal(trace["episode_step_offsets"], [0, count])
            or not np.array_equal(trace["episode_state_offsets"], [0, count + 1])
            or len(trace["states"]) != count + 1 or len(trace["warmstarts"]) != count + 1
            or report["completed_steps"] != count):
        raise ValueError("Shadow state/transition offsets are inconsistent.")
    if len(records) != count or len(candidate_plans) != count or len(candidate_steps) != count:
        raise ValueError("Candidate and baseline decision counts differ.")
    decision = records[decision_step]
    if decision["step"] != decision_step or decision.get("accepted") is not True:
        raise ValueError("Choose a saved, accepted shadow decision.")
    if not gate_accepts(decision["baseline_cost"], decision["policy_cost"],
                        decision["baseline_fell"], decision["policy_fell"]):
        raise ValueError("Saved acceptance is inconsistent with the fixed gate rule.")
    if not 1 <= candidate_steps[decision_step] <= cfg.H or candidate_steps[decision_step] != decision["candidate_steps"]:
        raise ValueError("Invalid candidate prefix length.")
    plan = trace["plans"][decision_step]
    candidate = candidate_plans[decision_step]
    snapshot = {"U": trace["post_plan_nominals"][decision_step].copy(),
                "rng_state": json.loads(str(trace["post_planner_rng_states"][decision_step])),
                "initial_plan": bool(trace["post_initial_plan_flags"][decision_step])}
    # The shift repeats the last optimized action. A pre-call snapshot would
    # spend the same random samples twice and invalidate the intervention.
    shifted = np.concatenate((plan[1:], plan[-1:]))
    if snapshot["initial_plan"] or not np.array_equal(snapshot["U"], shifted):
        raise ValueError("Saved post-call snapshot does not match the optimized-plan shift.")
    if not np.array_equal(trace["actions"][decision_step], plan[0]):
        raise ValueError("The saved baseline action differs from the optimized plan.")
    if decision_step + 1 < count:
        if (not np.array_equal(snapshot["U"], trace["pre_plan_nominals"][decision_step + 1])
                or snapshot["rng_state"] != json.loads(str(trace["planner_rng_states"][decision_step + 1]))):
            raise ValueError("Post-call snapshot differs from the next reference decision.")
    env = Walker2d(**task)
    try:
        low, high = env.action_bounds
        for name, actions in (("baseline", plan), ("candidate", candidate), ("post-call nominal", snapshot["U"])):
            if (actions.shape != (cfg.H, env.action_dim) or not np.isfinite(actions).all()
                    or (actions < low).any() or (actions > high).any()):
                raise ValueError(f"Invalid {name} shape, values or actuator bounds.")
    finally:
        env.close()
    return {"task": task, "cfg": cfg, "trace": trace, "decision": decision,
            "state": trace["states"][decision_step].copy(),
            "warmstart": trace["warmstarts"][decision_step].copy(),
            "snapshot": snapshot, "baseline_action": plan[0].copy(),
            "candidate_action": candidate[0].copy(), "manifest": manifest,
            "decision_step": decision_step, "policy_label": policy_label,
            "shadow_dir": str(shadow_dir.resolve()),
            "artifact_hashes_sha256": sha256(shadow_dir / "artifact_hashes.json")}


def _json_snapshot(snapshot: dict) -> dict:
    return {**copy.deepcopy(snapshot), "U": snapshot["U"].tolist()}


def run_branch(task: dict, cfg: MPPIConfig, state: np.ndarray, warmstart: np.ndarray,
               post_snapshot: dict, first_action: np.ndarray, steps: int) -> tuple[dict, dict]:
    """Execute one supplied action, then continue independently with raw MPPI."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer including the first intervention.")
    started = time.perf_counter()
    env = Walker2d(**task)
    try:
        state, warmstart, first_action = (np.asarray(value, dtype=float).copy()
                                        for value in (state, warmstart, first_action))
        low, high = env.action_bounds
        if (state.shape != (env.state_dim,) or not np.isfinite(state).all()
                or warmstart.shape != (env.model.nv,) or not np.isfinite(warmstart).all()
                or not env._is_healthy(state[2], state[3])):
            raise ValueError("Branch requires a finite, healthy state and matching warmstart.")
        if (first_action.shape != (env.action_dim,) or not np.isfinite(first_action).all()
                or (first_action < low).any() or (first_action > high).any()):
            raise ValueError("Invalid initial branch action.")
        env.reset(state=state)
        env.set_warmstart(warmstart)
        controller = MPPI(env, cfg)
        controller.restore(copy.deepcopy(post_snapshot))
        if controller.snapshot()["initial_plan"]:
            raise ValueError("A post-call snapshot must have initial_plan=False.")
        saved = {key: [] for key in ("actions", "costs", "pre_plan_nominals", "planner_rng_states",
                                      "initial_plan_flags", "planned_here") + POST_FIELDS}
        saved.update(states=[env.get_state()], warmstarts=[env.get_warmstart()])
        initial_snapshot = controller.snapshot()
        fell = False
        for step in range(steps):
            before = controller.snapshot()
            if step == 0:
                action = first_action.copy()  # Do not consume another planning call here.
            else:
                action, _ = controller.plan_step(env.get_state(), initial_warmstart=env.get_warmstart())
            after = controller.snapshot()
            _, cost, fell, _ = env.step(action)
            if not np.isfinite(cost):
                raise FloatingPointError("Non-finite branch task cost.")
            for key, value in {
                "states": env.get_state(), "warmstarts": env.get_warmstart(), "actions": action.copy(),
                "costs": cost, "planned_here": step > 0, "pre_plan_nominals": before["U"],
                "planner_rng_states": json.dumps(before["rng_state"]), "initial_plan_flags": before["initial_plan"],
                "post_plan_nominals": after["U"], "post_planner_rng_states": json.dumps(after["rng_state"]),
                "post_initial_plan_flags": after["initial_plan"],
            }.items():
                saved[key].append(value)
            if fell:
                break
        trace = {key: np.asarray(values) for key, values in saved.items()}
        completed = len(trace["actions"])
        vx = trace["states"][1:, 1 + env.model.nq]
        summary = {"requested_steps": steps, "completed_steps": completed,
                   "completed_without_fall": completed == steps and not fell,
                   "first_fall_step": completed if fell else None,
                   "velocity_mae": float(np.mean(np.abs(vx - task["target_velocity"]))),
                   "mean_velocity": float(vx.mean()), "task_cost": float(trace["costs"].sum()),
                   "fixed_horizon_task_cost": float(trace["costs"].sum()
                        + (steps - completed) * task["unhealthy_cost_weight"]),
                   "planner_calls": int(trace["planned_here"].sum()),
                   "start_snapshot": _json_snapshot(initial_snapshot),
                   "end_snapshot": _json_snapshot(controller.snapshot()),
                   "elapsed_seconds": time.perf_counter() - started}
        return trace, summary
    finally:
        env.close()


def compare_reference(trace: dict, source: dict, decision_step: int, steps: int) -> dict:
    """Check the raw branch against every available reference control boundary."""
    length = min(steps, len(source["actions"]) - decision_step)
    expected = {key: source[key][decision_step:decision_step + length + (key in {"states", "warmstarts"})]
                for key in COMPARISON_FIELDS}
    actual = {key: trace[key][:length + (key in {"states", "warmstarts"})] for key in COMPARISON_FIELDS}
    fields = compare_fields(expected, actual, COMPARISON_FIELDS)
    return {"checked_steps": length, "exact": all(row["exact"] for row in fields.values()), "fields": fields}


def _save_trace(path: Path, trace: dict) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **trace)


def main(shadow_dir: Path, policy_label: str, decision_step: int, out_dir: Path,
         steps: int = 250) -> None:
    """Branch once at a preselected accepted suggestion; do not run an active gate."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    start = load_decision(shadow_dir, policy_label, decision_step)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_files = FROZEN_SOURCES + ("src/utils/trace_replay.py", "scripts/runners/eval_policy_walker2d.py",
                                    "scripts/diagnostics/walker_gate_counterfactual.py")
    source_hashes = {relative: sha256(_ROOT / relative) for relative in source_files}
    for relative in source_files:
        target = out_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)
    manifest = {"shadow_dir": start["shadow_dir"], "shadow_artifact_hashes_sha256": start["artifact_hashes_sha256"],
                "policy_label": policy_label, "decision_step": decision_step, "steps": steps,
                "gate_decision": start["decision"], "task": start["task"], "mppi_config": asdict(start["cfg"]),
                "post_call_snapshot": _json_snapshot(start["snapshot"]),
                "provenance": {"python": platform.python_version(),
                               "packages": {name: version(name) for name in ("mujoco", "numpy", "scipy")},
                               "source_sha256": source_hashes},
                "trace_layout": "states/warmstarts are initial plus post-action states. planned_here[0]=False: first pre-plan snapshot is already POST original planning call. All later rows are ordinary raw-MPPI calls."}
    write_json(out_dir / "manifest.json", manifest)
    summaries = {}
    comparison = None
    status, error = "failed", None
    try:
        for label, action in (("raw", start["baseline_action"]), ("intervention", start["candidate_action"])):
            trace, summary = run_branch(start["task"], start["cfg"], start["state"], start["warmstart"],
                                        start["snapshot"], action, steps)
            _save_trace(out_dir / f"{label}.npz", trace)
            summaries[label] = summary
            if label == "raw":
                comparison = compare_reference(trace, start["trace"], decision_step, steps)
                if not comparison["exact"]:
                    raise RuntimeError("Raw continuation does not exactly reproduce the saved shadow prefix.")
        status = "complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(out_dir / "report.json", {"status": status, "error": error,
                   "single_intervention_only": True, "gate_decision": start["decision"],
                   "branches": summaries, "raw_reference_comparison": comparison,
                   "cost_semantics": "task_cost covers executed steps and includes the one-time fall penalty. fixed_horizon_task_cost adds unhealthy_cost_weight for each remaining control after a fall, without another terminal penalty. velocity_mae covers executed steps only.",
                   "limits": "A paired deterministic single intervention, not repeated active-gate execution or population-level benefit. Unequal-length executed task totals are not directly comparable; use the fixed-horizon absorbing total alongside completion and the MAE prefix lengths."})
        hashes = {str(path.relative_to(out_dir)): sha256(path)
                  for path in sorted(out_dir.rglob("*")) if path.is_file()}
        write_json(out_dir / "artifact_hashes.json", hashes)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
