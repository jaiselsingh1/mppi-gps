"""Independent saved-only verification of worker-native teacher baselines.

RV captures and repeats existing controls; it neither acquires teachers nor
fits a policy. This auditor compares complete saved native contexts and the
separately recorded K=1 batch states/costs. It imports no simulator or producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from .trajectory_pi import Audit, _plain, audit_ledger, batch_cost, load_arrays
from .trajectory_pi_replay_validation import (
    ARRAYS,
    CONTEXT,
    compare_arrays,
    compare_saved_comparisons,
    compare_trace,
    original_arrays,
    physical_outcome,
    trace_arrays,
)

HORIZON = 1000
NATIVE_CONTROLS = 4000
MODEL_CONTROLS = 2000
PHYSICS_SUBSTEPS = 24000
ATOL = 1e-10
COST_ATOL = 1e-9


def baseline_comparisons(
    a: Audit, original: dict, capture: dict, replay: dict, candidate: dict
) -> dict:
    """Verify native repeatability and K=1 equivalence, without replaying physics."""
    a.need(
        len(capture["actions"]) == len(replay["actions"]) == HORIZON,
        "Each capture and repeat has exactly 1000 executed controls",
    )
    initial_bytes = {
        name: (
            original[name][0].dtype == capture[name][0].dtype
            and original[name][0].tobytes() == capture[name][0].tobytes()
        )
        for name in CONTEXT
    }
    original_actions = (
        original["actions"].dtype == capture["actions"].dtype
        and original["actions"].tobytes() == capture["actions"].tobytes()
    )
    a.need(
        all(initial_bytes.values()) and original_actions,
        "Worker-native capture preserves the original complete initial context and physical U",
    )
    repeat = compare_trace(a, replay, capture, (0.0, ATOL))
    a.need(repeat["passed"], "Worker-native N+1 repeat agrees at strict source tolerance")
    captured_outcome, repeated_outcome = physical_outcome(a, capture), physical_outcome(a, replay)
    a.need(
        captured_outcome["walking"]
        and repeated_outcome["walking"]
        and captured_outcome["native_cost_formula_matches"]
        and repeated_outcome["native_cost_formula_matches"],
        "Both complete native traces walk and retain the unchanged task cost",
    )
    post, controls, costs = candidate["states"], candidate["actions"], candidate["costs"]
    a.need(
        post.shape == (1, HORIZON, 19)
        and controls.shape == (1, HORIZON, 6)
        and costs.shape == (1,)
        and post.dtype == controls.dtype == costs.dtype == np.float64,
        "Single-candidate batch evidence has declared physical float64 shapes",
    )
    a.equal(controls[0], original["actions"], "K=1 forecast uses exact original physical U")
    for key, context in (
        ("initial_state", "states"),
        ("initial_warmstart", "warmstarts"),
        ("initial_integration", "integrations"),
    ):
        a.equal(candidate[key], original[context][0], "K=1 bound original initial context: " + key)
    for key, initial in (
        ("final_state", "initial_state"),
        ("final_warmstart", "initial_warmstart"),
        ("final_integration", "initial_integration"),
    ):
        a.equal(
            candidate[key],
            candidate[initial],
            "K=1 forecast leaves the complete live context unchanged: " + key,
        )
    post_comparison = compare_arrays(post[0], capture["states"][1:], (0.0, ATOL))
    a.need(post_comparison["passed"], "K=1 model post-action states match native capture")
    own_cost, mask, valid = batch_cost(post, controls)
    a.need(valid[0] and mask.all(), "Finite complete K=1 executable state sequence")
    a.close(costs, own_cost, "Independent K=1 task-cost arithmetic", atol=COST_ATOL)
    a.close(
        costs[0],
        captured_outcome["complete_cost"],
        "K=1/native complete cost equivalence",
        atol=COST_ATOL,
    )
    a.need(np.mean(np.abs(post[0, :, 10] - 1.5)) <= 0.3, "K=1 model sequence also walks")
    original_difference = {
        name: compare_arrays(capture[name], original[name], (0.0, ATOL)) for name in ARRAYS
    }
    return dict(
        passed=True,
        original_initial_bytes=initial_bytes,
        original_actions_exact_bytes=original_actions,
        repeat=repeat,
        candidate_post_states=post_comparison,
        candidate_cost=float(costs[0]),
        native_capture_outcome=captured_outcome,
        native_replay_outcome=repeated_outcome,
        original_to_worker_difference=original_difference,
        source_scope="Original controls and initial contexts are retained. New native trajectories/costs carry worker-native provenance; historical state paths are not overwritten.",
        batch_context_scope="Complete live context before/after the batch is checked. Per-step batch warmstart/integration arrays are unavailable and are never fabricated.",
    )


def _same_file_bytes(first: dict, second: dict) -> bool:
    return all(first[key] == second[key] for key in ("bytes", "sha256"))


def _source_roster(a: Audit, source: dict, root: Path) -> None:
    a.need(source["schema"] == "trajectory-pi-source-1", "Canonical frozen P1.1 source roster")
    encoded = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    a.need(
        hashlib.sha256(encoded).hexdigest() == source["roster_sha256"],
        "RV frozen source roster hash",
    )
    names = []
    for row in source["files"]:
        name = row["path"]
        a.need(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and name not in names
            and (root / name).resolve().is_relative_to(root.resolve()),
            "Contained unique RV source path",
        )
        names.append(name)
        a.bind(root / name, row)
    a.need(
        names == sorted(names)
        and "src/gps/mppi_gps/audits/trajectory_pi_native_baselines.py" in names,
        "Auditor included in sorted immutable source roster",
    )
    a.need(
        (root / "src/gps/mppi_gps/audits/trajectory_pi_native_baselines.py").resolve()
        == Path(__file__).resolve(),
        "RV audit executes its frozen source version",
    )


def _origin_proof(a: Audit, proof: dict, p1: dict) -> dict:
    values = {
        name: a.json(a.bound(proof[name]))
        for name in ("result", "audit_receipt", "audit_findings", "external_summary")
    }
    a.need(
        proof["origin_replay_passed"] is True
        and values["result"]["status"] == "ORIGIN_REPLAY_PASS"
        and values["audit_receipt"]["passed"] is True
        and values["audit_receipt"]["origin_replay_passed"] is True
        and values["audit_findings"]["origin_replay_passed"] is True
        and all(row["passed"] for row in values["result"]["source_comparisons"].values()),
        "Existing audited origin-host replay passed; no origin physics is repeated",
    )
    names = ("result", "audit_receipt", "audit_findings", "external_summary")
    a.need(
        set(p1["lineage"]["origin_replay"]["bindings"]) == set(names),
        "All four canonical origin replay proof bindings are frozen",
    )
    for key in names:
        a.need(
            _same_file_bytes(p1["lineage"]["origin_replay"]["bindings"][key], proof[key]),
            "Frozen origin proof identity: " + key,
        )
    return values


def audit_evidence(a: Audit, run: Path) -> dict:
    """Audit a closed raw RV directory, before its proof manifest is assembled."""
    launch = a.json(run / "launch.json")
    a.need(
        launch["schema"] == "trajectory-pi-native-baselines-launch-1"
        and Path(launch["output_directory"]).resolve() == run.resolve(),
        "Frozen create-only RV launch",
    )
    files = {name: a.bound(launch[name]) for name in ("config", "source_manifest", "bindings")}
    config, source, inputs = (
        a.json(files[name]) for name in ("config", "source_manifest", "bindings")
    )
    a.need(
        config["schema"] == "trajectory-pi-native-baselines-1",
        "Frozen worker-native baseline configuration",
    )
    root = Path(launch["source_directory"])
    _source_roster(a, source, root)
    p1_path = root / config["p1_config_relative"]
    a.need(
        p1_path.resolve().is_relative_to(root.resolve()), "Contained P1.1 scientific configuration"
    )
    p1_binding = a.bind(p1_path)
    a.need(
        p1_binding["sha256"] == config["p1_config_sha256"],
        "RV binds the exact subsequent P1.1 recipe",
    )
    p1 = a.json(p1_path)
    a.need(
        p1["version"] == "P1.1"
        and p1["input_mode"] == "worker-native"
        and p1["launch_ready"] is True,
        "Versioned native-input stage",
    )
    a.need(
        config["horizon"] == HORIZON
        and config["physics_threads"] == 8
        and config["task"] == p1["task"],
        "RV uses the exact frozen full-horizon scientific task",
    )
    a.need(
        config["tolerances"]
        == dict(
            native=dict(rtol=0, atol=ATOL),
            candidate_states=dict(rtol=0, atol=ATOL),
            candidate_complete_cost=dict(rtol=0, atol=COST_ATOL),
        )
        and config["one_attempt"] is True
        and config["phase_census"]
        == dict(captures=2, native_replays=2, single_candidate_rollouts=2),
        "Exactly one bounded RV assay with separate frozen equivalence tolerances",
    )
    a.equal(
        config["global_limits_unchanged"],
        p1["limits"],
        "RV does not enlarge the subsequent global loop ceilings",
    )
    previous_path = a.bound(inputs["previous_prepared_manifest"])
    previous_config_path = a.bound(inputs["previous_config"])
    previous = a.json(previous_path)
    a.need(
        previous["config_sha256"] == inputs["previous_config"]["sha256"]
        and _same_file_bytes(
            p1["lineage"]["previous_prepared_manifest"], inputs["previous_prepared_manifest"]
        )
        and _same_file_bytes(p1["lineage"]["previous_config"], inputs["previous_config"]),
        "RV retains the exact original P1.0 prepared/configuration lineage",
    )
    old_config = a.json(previous_config_path)
    a.need(
        old_config["version"] == "P1.0",
        "Original failure belongs to P1.0, not an overwritten version",
    )
    origin = _origin_proof(a, inputs["origin_replay"], p1)
    environment = a.json(a.bound(inputs["worker_environment"]))
    a.need(
        environment["machine"] == "x86_64"
        and environment["python"] == p1["resources"]["runtime"]["python"],
        "Worker-native capture stays on the declared native CPU runtime",
    )
    a.need(
        set(environment["versions"]) == {"numpy", "mujoco", "scipy", "h5py"},
        "Complete declared native worker dependency roster",
    )
    for name, version in environment["versions"].items():
        a.need(
            version == p1["resources"]["runtime"][name], "Pinned worker native dependency: " + name
        )
        if "expected_runtime" in config:
            a.need(version == config["expected_runtime"][name], "RV worker dependency pin: " + name)
    a.bound(environment["interpreter"])
    a.need(len(environment["binary_files"]) >= 2, "Explicit worker native binary roster")
    for item in environment["binary_files"]:
        a.bound(item)
    limits = config["limits"]
    a.need(
        limits["wall_seconds"] == 600
        and limits["output_bytes"] == 64 * 1024**2
        and limits["native_controls"] == NATIVE_CONTROLS
        and limits["model_controls"] == MODEL_CONTROLS
        and limits["physics_substeps"] == PHYSICS_SUBSTEPS
        and limits["environment_constructions"] == 1
        and limits["restores"] == limits["external_mj_forward_calls"] == 6
        and limits["model_queries"] == 2
        and limits["compiled_model_saves"] == 1,
        "Finite two-capture/two-repeat/two-K1-query ceilings",
    )
    for unit in ("neural_rows", "backward_rows", "optimizer_steps", "pi_passes"):
        a.need(limits[unit] == 0, "No new learning, policy or PI intervention: " + unit)
    result = a.json(run / "result.json")
    a.need(
        result["schema"] == "trajectory-pi-native-baselines-result-1",
        "Closed raw worker-native baseline result",
    )
    a.bind(run / "launch.json", result["launch"])
    a.need(
        result["config_sha256"] == launch["config"]["sha256"]
        and result["p1_config_sha256"] == config["p1_config_sha256"]
        and result["source_manifest_sha256"] == launch["source_manifest"]["sha256"]
        and result["previous_prepared_manifest"] == inputs["previous_prepared_manifest"]
        and result["origin_replay"] == inputs["origin_replay"],
        "Raw RV result retains every frozen source/input seal",
    )
    ledger = audit_ledger(a, run)
    a.equal(result["work"], ledger["counts"], "RV raw result and durable paid census")
    for unit, cap in limits.items():
        if unit not in ("wall_seconds", "memory_bytes", "output_bytes"):
            a.need(
                ledger["counts"].get("requested_" + unit, 0) <= cap, "RV finite work limit: " + unit
            )
    a.need(
        ledger["receipt"]["wall_seconds"] <= limits["wall_seconds"]
        and ledger["receipt"]["peak_process_rss_bytes"] <= limits["memory_bytes"],
        "RV wall/RSS ceilings",
    )
    for prefix in ("requested_", "completed_", "unknown_partial_"):
        a.need(
            ledger["counts"].get(prefix + "physics_substeps", 0)
            == 4
            * (
                ledger["counts"].get(prefix + "native_controls", 0)
                + ledger["counts"].get(prefix + "model_controls", 0)
            ),
            "Actual native and batch physics charged separately at four substeps per control",
        )
        for unit in ("neural_rows", "backward_rows", "optimizer_steps", "pi_passes"):
            a.need(ledger["counts"].get(prefix + unit, 0) == 0, "No hidden prohibited RV work")
    events = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
    model_events = [row for row in events if row["event"] == "native_model_bound"]
    calls = [
        row
        for row in events
        if row["event"] == "requested"
        and (row["charges"].get("native_controls", 0) or row["charges"].get("model_controls", 0))
    ]
    for row in calls:
        charges = row["charges"]
        if charges.get("native_controls", 0):
            a.need(
                charges["native_controls"] == 1 and charges["physics_substeps"] == 4,
                "Each native capture/repeat control has its own exact reservation",
            )
        else:
            a.need(
                charges["model_controls"] == HORIZON
                and charges["physics_substeps"] == 4 * HORIZON
                and charges["model_queries"] == 1,
                "Each K=1 full-horizon forecast is charged once",
            )
    a.need(
        len(model_events) == 1
        and result["model_bindings"] == model_events[0]["model_bindings"]
        and (not calls or model_events[0]["id"] < calls[0]["id"]),
        "Compiled native model/options were bound before every capture, repeat and batch call",
    )
    model_file = a.bound(result["model_bindings"])
    model = a.json(model_file)
    a.need(
        model["dimensions"] == [9, 9, 6] and type(model["spec"]) is int and model["spec"] > 0,
        "Frozen Walker and full integration-state contract",
    )
    for name in ("compiled", "options"):
        a.bound(model[name])
    a.need(result["status"] in ("PASS", "FAIL_STOPPED"), "No incomplete RV is silently promoted")
    episodes = []
    if result["status"] == "PASS":
        a.need(
            ledger["receipt"]["status"] == "COMPLETED"
            and not ledger["failed"]
            and not ledger["rejected"]
            and len(result["episodes"]) == len(inputs["episodes"]) == 2,
            "All requested baseline validations completed without retry",
        )
        for supplied, saved in zip(inputs["episodes"], result["episodes"], strict=True):
            slot = supplied["slot"]
            a.need(
                slot == saved["slot"]
                and saved["original_episode"] == supplied["original_episode"]
                and saved["original_prepared"] == supplied["original_prepared"],
                "Exact original slot ownership",
            )
            for key in ("original_episode", "original_prepared", "capture", "replay", "candidate"):
                a.bound(saved[key])
            old = next(row for row in previous["teachers"] if row["slot"] == slot)
            scientific_teacher = next(row for row in p1["teachers"] if row["slot"] == slot)
            a.need(
                _same_file_bytes(saved["original_episode"], old["source"])
                and _same_file_bytes(saved["original_episode"], scientific_teacher["episode"])
                and old["file_relative"] == scientific_teacher["prepared_path"],
                "No replacement or reacquisition of original H4 episode",
            )
            old_prepared = previous_path.parent / old["file_relative"]
            a.bind(old_prepared, saved["original_prepared"])
            original = original_arrays(a, Path(saved["original_episode"]["path"]))
            capture, replay = (
                trace_arrays(a, Path(saved[key]["path"])) for key in ("capture", "replay")
            )
            candidate = load_arrays(a, Path(saved["candidate"]["path"]))
            measured = baseline_comparisons(a, original, capture, replay, candidate)
            if "replay_comparisons" in saved:
                compare_saved_comparisons(a, saved["replay_comparisons"], measured["repeat"])
            if "replay_outcome" in saved:
                for key in ("controls", "walking", "full_survival"):
                    a.need(
                        saved["replay_outcome"][key] == measured["native_replay_outcome"][key],
                        "Independently reported repeat walking/survival outcome",
                    )
            if "candidate_state_maximum_error" in saved:
                a.close(
                    saved["candidate_state_maximum_error"],
                    measured["candidate_post_states"]["maximum_absolute_error"],
                    "Reported candidate post-state equivalence error",
                    atol=0,
                )
            if "candidate_cost_error" in saved:
                a.close(
                    saved["candidate_cost_error"],
                    abs(
                        float(candidate["costs"][0])
                        - measured["native_capture_outcome"]["complete_cost"]
                    ),
                    "Reported candidate/native aggregate cost error",
                    atol=0,
                )
            for key in (
                "native_replay_passed",
                "single_candidate_passed",
                "initial_context_exact",
                "actions_exact",
            ):
                a.need(
                    saved[key] is True, "Worker result preserves every independent baseline gate"
                )
            for key in ("controls", "walking", "full_survival"):
                a.need(
                    saved["outcome"][key] == measured["native_capture_outcome"][key],
                    "Native capture outcome identity",
                )
            for key in ("complete_cost", "velocity_mae"):
                a.close(
                    saved["outcome"][key],
                    measured["native_capture_outcome"][key],
                    "Native capture outcome arithmetic",
                    atol=ATOL,
                )
            episodes.append(
                dict(
                    slot=slot,
                    **{
                        key: saved[key]
                        for key in (
                            "original_episode",
                            "original_prepared",
                            "capture",
                            "replay",
                            "candidate",
                        )
                    },
                    outcome=measured["native_capture_outcome"],
                    native_replay_passed=True,
                    single_candidate_passed=True,
                    initial_context_exact=True,
                    actions_exact=True,
                    measurements=measured,
                )
            )
        a.need([row["slot"] for row in episodes] == [0, 1], "Two predeclared original source slots")
        for unit, amount in (
            ("native_controls", NATIVE_CONTROLS),
            ("model_controls", MODEL_CONTROLS),
            ("physics_substeps", PHYSICS_SUBSTEPS),
            ("environment_constructions", 1),
            ("restores", 6),
            ("external_mj_forward_calls", 6),
            ("model_queries", 2),
            ("compiled_model_saves", 1),
        ):
            a.need(
                ledger["counts"].get("requested_" + unit)
                == ledger["counts"].get("completed_" + unit)
                == amount,
                "Complete exact RV paid work: " + unit,
            )
        a.need(
            all(
                not key.startswith("unknown_partial_") or value == 0
                for key, value in ledger["counts"].items()
            ),
            "No unknown partial work in a qualifying baseline",
        )
    else:
        a.need(ledger["receipt"]["status"] == "FAILED", "Original RV failure remains failed")
    for file in run.rglob("*"):
        if file.is_file():
            a.bind(file)
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    a.need(size <= limits["output_bytes"], "Complete RV output ceiling")
    return dict(
        schema="trajectory-pi-native-baselines-audit-findings-1",
        status="PASS_SAVED_EVIDENCE",
        native_validation_passed=result["status"] == "PASS",
        original_status=result["status"],
        result_binding=dict(
            path=str((run / "result.json").resolve()),
            **{key: a.bind(run / "result.json")[key] for key in ("bytes", "sha256")},
        ),
        config_sha256=launch["config"]["sha256"],
        p1_config_sha256=config["p1_config_sha256"],
        source_manifest_sha256=launch["source_manifest"]["sha256"],
        previous_prepared_manifest=inputs["previous_prepared_manifest"],
        origin_replay=inputs["origin_replay"],
        episodes=episodes,
        work=ledger["counts"],
        model_bindings=model,
        environment=environment,
        origin_work_already_paid=origin["result"]["work"],
        study_output_bytes_before_audit=size,
        scope="Saved array equivalence and original-input/runtime lineage only. Zero new physics, inference or optimizer arithmetic. Original P1.0 failure remains historical evidence.",
    )


def audit_baselines(run: Path, out: Path, *, seconds: float = 60) -> dict:
    run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not np.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("RV saved audit seconds must lie in (0,60]")
    a = Audit(seconds=seconds, neural_rows=0, output_bytes=64 * 1024**2, memory_bytes=2 * 1024**3)
    findings, error = {}, None
    try:
        a.bind(Path(__file__))
        findings = audit_evidence(a, run)
        forbidden = (
            "mujoco",
            "src.envs.",
            "src.mppi.",
            "src.gps.mppi_gps.policy",
            "src.gps.mppi_gps.trajectory_pi_native_baselines",
            "src.gps.mppi_gps.trajectory_pi_native",
        )
        a.need(
            not any(name.startswith(forbidden) for name in sys.modules),
            "Independent RV audit imports no producer or simulator",
        )
        for path in list(a.files):
            a.bind(Path(path))
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    for name, value in (("findings.json", findings), ("file_bindings.json", a.files)):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, default=_plain)
            stream.write("\n")
    findings_file = out / "findings.json"
    receipt = dict(
        schema="trajectory-pi-native-baselines-independent-audit-1",
        passed=error is None,
        native_validation_passed=findings.get("native_validation_passed", False),
        error=error,
        findings_binding=dict(
            path=str(findings_file),
            bytes=findings_file.stat().st_size,
            sha256=hashlib.sha256(findings_file.read_bytes()).hexdigest(),
        ),
        checks=a.checks,
        bound_files=len(a.files),
        hashed_file_bytes=sum(value["bytes"] for value in a.files.values()),
        new_native_controls=0,
        new_model_controls=0,
        new_physics_substeps=0,
        new_neural_rows=0,
        new_gradients=0,
        new_optimizer_updates=0,
        wall_seconds=time.monotonic() - a.started,
        cpu_seconds=time.process_time() - a.cpu_started,
    )
    with (out / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    a.need(
        sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) <= a.output_limit,
        "RV independent audit output cap",
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args(argv)
    receipt = audit_baselines(args.run, args.out, seconds=args.seconds)
    print(json.dumps(receipt, allow_nan=False))
    return int(not receipt["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
