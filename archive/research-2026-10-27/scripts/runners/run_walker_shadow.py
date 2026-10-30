"""S3: score frozen policies alongside raw MPPI, without executing suggestions.

Re-run a complete reference episode continuously from its reset/planner seeds.
Compare every saved baseline field, including planner snapshots, at the end.
Policy scores are development diagnostics, not evidence of improved control.
"""

import copy
import json
import shutil
import sys
import time
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.runners.eval_policy_walker2d import provenance, sha256, write_json
from scripts.runners.run_walker2d import _foot_contacts
from src.envs.walker2d import Walker2d
from src.mppi.gate import TaskCostGate
from src.mppi.mppi import MPPI
from src.rl.sac import load_policy
from src.utils.config import MPPIConfig
from src.utils.trace_replay import compare_fields, load_episode

DEFAULT_SELECTION = (_ROOT / "runs/2026-10-11_s2_sac_ladder_seed0_001/"
                     "review_001/frozen_selection.json")
TRACE_FIELDS = ("states", "warmstarts", "actions", "costs", "contacts",
                "contact_normal_forces", "pre_plan_nominals", "plans",
                "planner_rng_states", "initial_plan_flags")
PHYSICS_SOURCES = ("src/envs/base.py", "src/envs/walker2d.py",
                   "src/envs/mujoco_env.py", "assets/walker2d.xml")


def load_selection(selection_path: Path, task_config: Path) -> tuple[dict, dict]:
    """Load the hash-identified S2 ladder; never substitute the latest policy."""
    selection = json.loads(selection_path.read_text())
    task_hash = sha256(task_config)
    if selection["task_sha256"] != task_hash or selection["obs_transform"] != "identity":
        raise ValueError("Selection task or observation transform differs.")
    if set(selection["checkpoints"]) != {"weak", "intermediate", "strong"}:
        raise ValueError("Selection must identify weak, intermediate and strong checkpoints.")
    policies = {}
    for label, record in selection["checkpoints"].items():
        path = _ROOT / record["checkpoint"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {label}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["task_sha256"] != task_hash or payload["step"] != record["step"]:
            raise ValueError(f"Checkpoint task or training step mismatch: {label}")
        for relative in PHYSICS_SOURCES:
            if payload["source_sha256"].get(relative) != sha256(_ROOT / relative):
                raise ValueError(f"Checkpoint physics mismatch: {label}: {relative}")
        policies[label] = load_policy(payload["policy"], device="cpu")
    return policies, selection


def _gate_summary(records: list[dict]) -> dict:
    valid = [r for r in records if r["policy_cost"] is not None]
    accepted = sum(r["accepted"] for r in records)
    return {"decisions": len(records), "accepted": accepted,
            "acceptance_rate": accepted / len(records) if records else None,
            "reasons": dict(Counter(r["reason"] for r in records)),
            "baseline_predicted_falls": sum(r["baseline_fell"] for r in records),
            "policy_predicted_falls": sum(r["policy_fell"] is True for r in records),
            "mean_policy_minus_baseline_cost": float(np.mean([
                r["policy_cost"] - r["baseline_cost"] for r in valid])) if valid else None,
            "mean_gate_seconds": float(np.mean([r["seconds"] for r in records])) if records else None}


def run_shadow(reference: dict, task: dict, cfg: MPPIConfig, policies: dict,
               steps: int, on_progress=None) -> tuple[dict, dict, dict]:
    """Return baseline trace, per-policy candidate arrays, and diagnostics.

    on_progress(trace, candidates, report) receives owned copies every 100
    completed controls and at completion/error. Never restore the reference
    states during this run: doing so would hide cumulative side effects.
    """
    if steps < 1 or not policies:
        raise ValueError("Require positive steps and at least one frozen policy.")
    if any(key not in reference for key in TRACE_FIELDS):
        raise ValueError("Reference must include full S1 decision instrumentation.")
    saved = {key: [] for key in TRACE_FIELDS}
    # Post-call snapshots are separate: future single-intervention branches
    # must resume AFTER the call that produced the candidate comparison.
    saved.update(post_plan_nominals=[], post_planner_rng_states=[], post_initial_plan_flags=[])
    candidates = {label: [] for label in policies}
    records = {label: [] for label in policies}
    started = time.perf_counter()
    completed, fell, planner_seconds = 0, False, 0.0
    pending = None

    def result(status: str, error: str | None = None):
        arrays = {key: np.asarray(values) for key, values in saved.items()}
        arrays.update(episode_step_offsets=np.array([0, completed]),
                      episode_state_offsets=np.array([0, completed + 1]),
                      reset_seeds=np.array([reference["reset_seed"]]),
                      planner_seeds=np.array([reference["planner_seed"]]))
        candidate_arrays = {label: {"plans": np.asarray(plans),
                            "steps": np.asarray([r["candidate_steps"] for r in records[label]])}
                            for label, plans in candidates.items()}
        report = {"status": status, "shadow_only": True, "tau": 0.0,
                  "requested_steps": steps, "completed_steps": completed, "fell": fell,
                  "elapsed_seconds": time.perf_counter() - started,
                  "planner_seconds": planner_seconds, "error": error,
                  "policies": {label: _gate_summary(rows) for label, rows in records.items()},
                  "decisions": copy.deepcopy(records),
                  "pending_decision": copy.deepcopy(pending),
                  "limits": "Only raw MPPI executed. Acceptance measures finite-horizon suggestions on MPPI states; it is not an improvement, safety or non-inferiority result."}
        if status == "complete":
            length = min(steps, len(reference["actions"]))
            expected = {key: reference[key][:length + (key in {"states", "warmstarts"})]
                        for key in TRACE_FIELDS}
            fields = compare_fields(expected, arrays, TRACE_FIELDS)
            report.update(reference_fields=fields,
                          reference_trace_exact=all(field["exact"] for field in fields.values()))
            if not report["reference_trace_exact"]:
                report["status"] = "parity_failed"
        if on_progress is not None:
            on_progress(arrays, candidate_arrays, report)
        return arrays, candidate_arrays, report

    try:
        with ExitStack() as stack:
            env = Walker2d(**task)
            stack.callback(env.close)
            controller = MPPI(env, cfg)
            gates = {label: stack.enter_context(TaskCostGate(task, cfg.H, policy))
                     for label, policy in policies.items()}
            np.random.seed(reference["reset_seed"])
            env.reset()
            controller.reset(seed=reference["planner_seed"])
            saved["states"].append(env.get_state())
            saved["warmstarts"].append(env.get_warmstart())
            geom_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                        for name in ("floor", "foot_geom", "foot_left_geom")]
            for step in range(steps):
                state, warmstart = env.get_state(), env.get_warmstart()
                before = controller.snapshot()
                begin = time.perf_counter()
                action, _ = controller.plan_step(state, initial_warmstart=warmstart)
                planner_seconds += time.perf_counter() - begin
                plan = controller.last_plan
                after = controller.snapshot()
                pending = {"step": step, "state": state.tolist(), "warmstart": warmstart.tolist(),
                           "baseline_plan": plan.tolist(), "decisions": {},
                           "before": {**before, "U": before["U"].tolist()},
                           "after": {**after, "U": after["U"].tolist()}}
                step_decisions = {}
                for label, gate in gates.items():
                    begin = time.perf_counter()
                    decision = gate.decide(state, warmstart, plan)
                    step_decisions[label] = decision
                    pending["decisions"][label] = {
                        "step": step, "baseline_cost": decision.baseline_cost,
                        "policy_cost": decision.policy_cost,
                        "baseline_fell": decision.baseline_fell, "policy_fell": decision.policy_fell,
                        "accepted": decision.accepted, "reason": decision.reason,
                        "candidate_steps": decision.candidate_steps,
                        "candidate_prefix_cost": decision.candidate_prefix_cost,
                        "first_action_l2_difference": float(np.linalg.norm(decision.candidate_plan[0] - action)),
                        "seconds": time.perf_counter() - begin}
                # This is the only live control action, irrespective of decisions.
                _, cost, fell, _ = env.step(action)
                contacts, forces = _foot_contacts(env.model, env.data, *geom_ids)
                for label, decision in step_decisions.items():
                    records[label].append(pending["decisions"][label])
                    candidates[label].append(decision.candidate_plan)
                for key, value in {
                    "states": env.get_state(), "warmstarts": env.get_warmstart(),
                    "actions": action.copy(), "costs": cost, "contacts": contacts,
                    "contact_normal_forces": forces, "plans": plan,
                    "pre_plan_nominals": before["U"],
                    "planner_rng_states": json.dumps(before["rng_state"]),
                    "initial_plan_flags": before["initial_plan"],
                    "post_plan_nominals": after["U"],
                    "post_planner_rng_states": json.dumps(after["rng_state"]),
                    "post_initial_plan_flags": after["initial_plan"],
                }.items():
                    saved[key].append(value)
                completed += 1
                pending = None
                if completed % 100 == 0:
                    result("running")
                if fell:
                    break
        return result("complete")
    except Exception as exc:
        result("failed", f"{type(exc).__name__}: {exc}")
        raise


def _write_npz(path: Path, arrays: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def archive_inputs(out_dir: Path, config: Path, task_config: Path, policy_selection: Path) -> None:
    """Preserve byte-exact inputs, verifying any already-recorded manifest."""
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    inputs = out_dir / "inputs"
    inputs.mkdir(exist_ok=True)
    for source, name, key in ((config, "mppi.json", "config_sha256"),
                              (task_config, "task.json", "task_sha256"),
                              (policy_selection, "selection.json", "selection_sha256")):
        if manifest is not None and sha256(source) != manifest[key]:
            raise ValueError(f"Input no longer matches manifest: {source}")
        destination = inputs / name
        if destination.exists():
            if sha256(destination) != sha256(source):
                raise ValueError(f"Existing archived input differs: {destination}")
        else:
            shutil.copyfile(source, destination)


def hash_artifacts(out_dir: Path) -> None:
    hashes = {str(path.relative_to(out_dir)): sha256(path)
              for path in sorted(out_dir.rglob("*"))
              if path.is_file() and path != out_dir / "artifact_hashes.json"}
    write_json(out_dir / "artifact_hashes.json", hashes)


def main(reference: Path, out_dir: Path, reference_episode: int = 0, steps: int = 1000,
         config: Path = _ROOT / "configs/walker2d_best.json",
         task_config: Path = _ROOT / "configs/walker2d_task.json",
         policy_selection: Path = DEFAULT_SELECTION) -> None:
    """Run one audited S3 shadow episode using all three frozen S2 policies."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    torch.set_num_threads(1)
    expected = load_episode(reference, reference_episode)
    policies, selection = load_selection(policy_selection, task_config)
    task = json.loads(task_config.read_text())
    cfg = MPPIConfig(**json.loads(config.read_text()))
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_inputs(out_dir, config, task_config, policy_selection)
    sources = provenance()
    sources["packages"]["scipy"] = version("scipy")
    for relative in ("src/mppi/gate.py", "src/mppi/mppi.py", "src/utils/config.py",
                     "src/utils/math.py", "src/utils/trace_replay.py", "scripts/runners/run_walker_shadow.py"):
        sources["source_sha256"][relative] = sha256(_ROOT / relative)
    for relative in sources["source_sha256"]:
        target = out_dir / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_ROOT / relative, target)
    manifest = {"reference": str(reference.resolve()), "reference_episode": reference_episode,
                "reference_sha256": sha256(reference), "selection": selection,
                "selection_sha256": sha256(policy_selection), "task": task,
                "task_sha256": sha256(task_config), "mppi_config": asdict(cfg),
                "config_sha256": sha256(config), "provenance": sources,
                "reset_seed": expected["reset_seed"], "planner_seed": expected["planner_seed"]}
    write_json(out_dir / "manifest.json", manifest)

    def progress(trace, candidates, report):
        _write_npz(out_dir / "traces.npz", trace)
        for label, arrays in candidates.items():
            directory = out_dir / label
            directory.mkdir(exist_ok=True)
            _write_npz(directory / "candidates.npz", arrays)
            write_json(directory / "decisions.json", {"decisions": report["decisions"][label]})
        summary = {key: value for key, value in report.items() if key != "decisions"}
        write_json(out_dir / "status.json", summary)
        if report["status"] in {"complete", "failed", "parity_failed"}:
            write_json(out_dir / "report.json", summary)
        print(json.dumps({"status": report["status"], "steps": report["completed_steps"],
                          "elapsed_seconds": report["elapsed_seconds"],
                          "accepted": {k: v["accepted"] for k, v in report["policies"].items()}}), flush=True)

    _, _, report = run_shadow(expected, task, cfg, policies, steps, progress)
    hash_artifacts(out_dir)
    if not report["reference_trace_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
