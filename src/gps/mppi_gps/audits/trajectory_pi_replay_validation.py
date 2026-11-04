"""Independent, saved-array-only audit of the single-attempt R0 origin replay.

The original terminal context, new origin trace and already-paid Linux trace
are compared without importing MuJoCo, executing a policy, or restoring RNGs.
An audit pass confirms the saved evidence; the scientific replay gate has its
own explicit pass/fail field. Opaque compiled models and native libraries are
byte-bound, not loaded or recompiled by this audit.
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

from .trajectory_pi import (
    Audit,
    AuditFailure,
    _plain,
    audit_ledger,
    healthy,
    load_arrays,
    observations,
    running_cost,
)

ATOL = 1e-10
HORIZON = 1000
PHYSICS_SUBSTEPS = 4000
ARRAYS = ("states", "warmstarts", "integrations", "actions", "costs", "observations")
CONTEXT = ("states", "warmstarts", "integrations")


def original_arrays(a: Audit, path: Path) -> dict[str, np.ndarray]:
    """Append separately recorded terminal context; preserve all original bytes."""
    raw = load_arrays(a, path)
    result = {key: raw[key] for key in ("actions", "costs", "observations")}
    for key, final, width in (
        ("states", "terminal_state", 19),
        ("warmstarts", "terminal_warmstarts", 9),
        ("integrations", "terminal_integrations", 91),
    ):
        a.need(
            raw[key].shape == (HORIZON, width) and raw[final].shape == (width,),
            "Original full N+1 context is available: " + key,
        )
        result[key] = np.concatenate((raw[key], raw[final][None]))
    a.need(
        result["actions"].shape == (HORIZON, 6)
        and result["costs"].shape == (HORIZON,)
        and result["observations"].shape == (HORIZON, 17),
        "Original complete control/label roster",
    )
    for key, value in result.items():
        a.need(
            value.dtype == np.float64 and np.isfinite(value).all(),
            "Original finite float64 source array: " + key,
        )
    a.need(np.all(np.abs(result["actions"]) <= 1), "Original bounded physical torques")
    a.close(
        result["observations"],
        observations(result["states"][:-1]),
        "Original labels use exact pre-action states",
        atol=0,
    )
    result["observations"] = observations(result["states"])
    return result


def trace_arrays(a: Audit, path: Path) -> dict[str, np.ndarray]:
    raw = load_arrays(a, path)
    n = len(raw["actions"])
    a.need(
        0 <= n <= HORIZON and raw["actions"].shape == (n, 6) and raw["costs"].shape == (n,),
        "Actual saved replay control prefix",
    )
    for key, width in (
        ("states", 19),
        ("warmstarts", 9),
        ("integrations", 91),
        ("observations", 17),
    ):
        a.need(raw[key].shape == (n + 1, width), "True N+1 replay boundary array: " + key)
    for key in ARRAYS:
        a.need(raw[key].dtype == np.float64, "Actual float64 replay array: " + key)
    if "native_fall_cost" in raw:
        a.need(raw["native_fall_cost"].shape == (n,), "Saved native terminal ledger length")
    return raw


def _first_row(mask: np.ndarray) -> int | None:
    rows = np.flatnonzero(np.any(mask.reshape(len(mask), -1), axis=1)) if len(mask) else []
    return int(rows[0]) if len(rows) else None


def compare_arrays(
    actual: np.ndarray, expected: np.ndarray, thresholds: tuple[float, ...] = (0.0, ATOL)
) -> dict:
    """Describe differences only on recorded rows; never invent missing states."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    width_matches = actual.shape[1:] == expected.shape[1:]
    same_shape = actual.shape == expected.shape
    common = min(len(actual), len(expected)) if width_matches else 0
    left, right = actual[:common], expected[:common]
    finite = np.isfinite(left) & np.isfinite(right)
    with np.errstate(invalid="ignore", over="ignore"):
        difference = np.abs(left - right) if width_matches else np.empty((0,))
    finite_difference = difference[finite] if width_matches else np.empty((0,))
    maximum = float(np.max(finite_difference)) if finite_difference.size else None
    byte_equal = (
        same_shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    )
    crossings = (
        {format(value, ".12g"): _first_row((difference > value) | ~finite) for value in thresholds}
        if width_matches
        else {}
    )
    passed = bool(same_shape and finite.all() and maximum is not None and maximum <= ATOL)
    return dict(
        actual_shape=list(actual.shape),
        expected_shape=list(expected.shape),
        same_shape=same_shape,
        same_dtype=actual.dtype == expected.dtype,
        exact_bytes=byte_equal,
        compared_rows=common,
        missing_expected_rows=max(len(expected) - len(actual), 0),
        unexpected_actual_rows=max(len(actual) - len(expected), 0),
        nonfinite_compared_elements=int(np.sum(~finite)),
        maximum_absolute_error=maximum,
        first_nonidentical_row=crossings.get("0"),
        first_above_threshold_row=crossings,
        rtol=0,
        atol=ATOL,
        passed=passed,
    )


def physical_outcome(a: Audit, raw: dict[str, np.ndarray]) -> dict:
    """Walking and native/absorbing costs are distinct from saved-path equality."""
    states, controls, costs = raw["states"], raw["actions"], raw["costs"]
    n = len(controls)
    finite = all(np.isfinite(raw[key]).all() for key in ARRAYS)
    if not finite or not n:
        return dict(
            controls=n,
            complete_trace=False,
            finite=finite,
            fell=None,
            full_survival=False,
            walking=False,
            velocity_mae=None,
            mean_velocity=None,
            raw_native_cost=None,
            absorbing_tail_cost=None,
            complete_cost=None,
            native_cost_formula_matches=False,
            early_stop_reason="Nonfinite or empty recorded prefix",
        )
    alive = healthy(states[1:])
    a.need(alive[:-1].all(), "No physical states fabricated after the first fall")
    fell = not bool(alive[-1])
    actual_obs = raw["observations"][:n]
    a.close(
        actual_obs, observations(states[:-1]), "Saved physical pre-action observation map", atol=0
    )
    native = running_cost(states[1:], controls) + 20 * ~alive
    formula_matches = bool(np.allclose(costs, native, rtol=0, atol=ATOL))
    if "native_fall_cost" in raw:
        a.close(
            raw["native_fall_cost"],
            20 * ~alive,
            "One native fall terminal, independently checked",
            atol=0,
        )
    mae = float(np.mean(np.abs(states[1:, 10] - 1.5)))
    survived = n == HORIZON and not fell
    tail = float(5 * (HORIZON - n)) if fell else 0.0
    return dict(
        controls=n,
        complete_trace=n == HORIZON,
        finite=True,
        fell=fell,
        full_survival=survived,
        walking=bool(survived and mae <= 0.3),
        velocity_mae=mae,
        mean_velocity=float(np.mean(states[1:, 10])),
        raw_native_cost=float(costs.sum()),
        absorbing_tail_cost=tail,
        complete_cost=float(costs.sum()) + tail,
        native_cost_formula_matches=formula_matches,
        early_stop_reason=None if n == HORIZON else "Fall" if fell else "Incomplete live prefix",
    )


def compare_trace(
    a: Audit,
    actual: dict[str, np.ndarray],
    original: dict[str, np.ndarray],
    thresholds: tuple[float, ...],
) -> dict:
    report = {}
    for name in ARRAYS:
        report[name] = compare_arrays(actual[name], original[name], thresholds)
    initial = {
        name: (
            actual[name][0].dtype == original[name][0].dtype
            and actual[name][0].tobytes() == original[name][0].tobytes()
        )
        for name in CONTEXT
    }
    actions_exact = report["actions"]["exact_bytes"]
    return dict(
        arrays=report,
        initial_context_exact_bytes=initial,
        initial_context_passed=all(initial.values()),
        actions_exact_bytes=actions_exact,
        strict_arrays_passed=all(row["passed"] for row in report.values()),
        passed=bool(
            all(initial.values())
            and actions_exact
            and all(row["passed"] for row in report.values())
        ),
    )


def diagnosis(origin: dict, linux: dict, origin_outcome: dict) -> dict:
    replay_passed = bool(
        origin["passed"]
        and origin_outcome["walking"]
        and origin_outcome["native_cost_formula_matches"]
    )
    label = (
        "ORIGIN_REPLAY_CONFIRMED_LINUX_DIVERGES"
        if replay_passed and not linux["passed"]
        else "ORIGIN_AND_LINUX_REPLAYS_CONFIRMED"
        if replay_passed
        else "ORIGIN_REPLAY_NOT_CONFIRMED"
    )
    return dict(
        origin_replay_passed=replay_passed,
        linux_replay_passed=linux["passed"],
        interpretation=label,
        scope="Origin-pass/Linux-divergence supports a practical platform/runtime replay distinction; it does not identify a unique numerical cause.",
    )


def validate_inputs(a: Audit, run: Path) -> tuple[dict, dict, dict, dict]:
    launch = a.json(run / "launch.json")
    a.need(
        launch.get("schema") == "trajectory-pi-replay-launch-1"
        and launch["mode"] == "origin"
        and Path(launch["output_directory"]).resolve() == run.resolve(),
        "Frozen single origin-host launch",
    )
    files = {name: a.bound(launch[name]) for name in ("config", "source_manifest", "bindings")}
    config, source, inputs = (
        a.json(files[name]) for name in ("config", "source_manifest", "bindings")
    )
    a.need(
        config["schema"] == "trajectory-pi-replay-r0-1"
        and config["horizon"] == HORIZON
        and config["tolerance"] == dict(rtol=0, atol=ATOL)
        and config["physics_threads"] == 8,
        "Unchanged source horizon, strict tolerance and declared native CPU threads",
    )
    a.need(
        source["schema"] == "trajectory-pi-replay-source-1"
        and inputs["schema"] == "trajectory-pi-replay-bindings-1",
        "Frozen R0 source and input schemas",
    )
    roster = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    a.need(
        hashlib.sha256(roster).hexdigest() == source["roster_sha256"], "Exact frozen source roster"
    )
    source_root, names = Path(launch["source_directory"]), []
    for row in source["files"]:
        name = row["path"]
        path = source_root / name
        a.need(
            not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and path.resolve().is_relative_to(source_root.resolve())
            and name not in names,
            "Unique contained frozen source path",
        )
        names.append(name)
        a.bind(path, row)
    required = {
        "src/gps/mppi_gps/trajectory_pi_replay_validation.py",
        "src/gps/mppi_gps/audits/trajectory_pi_replay_validation.py",
        "src/gps/mppi_gps/audits/trajectory_pi.py",
        "src/gps/mppi_gps/trajectory_pi_native.py",
        "src/gps/mppi_gps/trajectory_pi_accounting.py",
        "src/gps/mppi_gps/runtime.py",
        "src/envs/mujoco_env.py",
        "src/envs/walker2d.py",
        "assets/walker2d.xml",
    }
    a.need(
        names == sorted(names) and required <= set(names),
        "Complete sorted relevant replay/audit source roster",
    )
    a.need(
        (source_root / "src/gps/mppi_gps/audits/trajectory_pi_replay_validation.py").resolve()
        == Path(__file__).resolve(),
        "Audit executes its frozen source version",
    )
    for name in ("source_recording", "linux_trace", "origin_launch_receipt", "environment"):
        a.bound(inputs[name])
        if name != "environment":
            a.need(
                all(
                    inputs[name][key] == config["inputs"][name][key] for key in ("bytes", "sha256")
                ),
                "No replacement of frozen evidence bytes; exact copied paths remain explicit: "
                + name,
            )
    environment = a.json(Path(inputs["environment"]["path"]))
    a.need(
        environment["schema"] == "trajectory-pi-replay-environment-1"
        and environment["machine"] == "arm64"
        and environment["python"] == "3.11.15"
        and environment["versions"]
        == dict(numpy="2.4.3", mujoco="3.6.0", scipy="1.17.1", h5py="3.16.0"),
        "Bound declared origin-host runtime; audit does not import native libraries",
    )
    a.bound(environment["interpreter"])
    binaries = environment["binary_files"]
    a.need(
        len(binaries) >= 2 and len({item["path"] for item in binaries}) == len(binaries),
        "Explicit distinct native-library binary roster",
    )
    for item in binaries:
        a.need(item["bytes"] > 0, "Nonempty native binary")
        a.bound(item)
    for name, value in config.get("expected_environment", {}).items():
        a.need(environment.get(name) == value, "Frozen expected environment: " + name)
    for name, value in config.get("expected_runtime", {}).items():
        actual = environment.get(name, environment["versions"].get(name))
        a.need(actual == value, "Frozen runtime pin: " + name)
    original_receipt = a.json(Path(inputs["origin_launch_receipt"]["path"]))
    hardware = original_receipt["hardware"]
    a.need(
        hardware["machine"] == environment["machine"]
        and hardware["python"].split()[0] == environment["python"]
        and all(
            hardware["dependency_versions"][name] == version
            for name, version in environment["versions"].items()
        ),
        "Origin receipt agrees on machine, Python and declared native dependency versions",
    )
    task = dict(
        frame_skip=4,
        cost_style="target_velocity",
        target_velocity=1.5,
        vel_cost_weight=2.0,
        vertical_velocity_cost_weight=0.5,
        angle_cost_weight=0.1,
        unhealthy_cost_weight=5.0,
        ctrl_cost_weight=0.001,
        terminal_unhealthy_weight=20.0,
        apply_terminal_cost_on_done=True,
    )
    a.need(config["task"] == task, "Independent cost formula matches the frozen Walker task")
    limits = config["limits"]
    a.need(
        limits["wall_seconds"] == 600
        and limits["output_bytes"] == 64 * 1024**2
        and limits["memory_bytes"] == 2 * 1024**3
        and limits["native_controls"] == HORIZON
        and limits["physics_substeps"] == PHYSICS_SUBSTEPS
        and limits["environment_constructions"] == 1,
        "Exact one-attempt R0 ceilings",
    )
    for unit in ("model_controls", "neural_rows", "backward_rows", "optimizer_steps", "pi_passes"):
        a.need(limits[unit] == 0, "No new learning or planner work: " + unit)
    return launch, config, inputs, environment


def compare_saved_comparisons(a: Audit, saved: dict, own: dict) -> None:
    a.need(set(saved) == set(ARRAYS), "Every source/replay array comparison is reported")
    for name in ARRAYS:
        actual, expected = saved[name], own["arrays"][name]
        a.need(
            actual["passed"] is expected["passed"]
            and actual["exact_bytes"] is expected["exact_bytes"]
            and actual["source_shape"] == expected["expected_shape"]
            and actual["replay_shape"] == expected["actual_shape"],
            "Independent strict array comparison: " + name,
        )
        if expected["maximum_absolute_error"] is None:
            a.need(actual["maximum_absolute_error"] is None, "Empty comparison stays missing")
        else:
            a.close(
                actual["maximum_absolute_error"],
                expected["maximum_absolute_error"],
                "Independent maximum replay error: " + name,
                atol=0,
            )
        a.need(
            actual["first_above_atol"] == expected["first_above_threshold_row"]["1e-10"],
            "Independent first strict tolerance crossing: " + name,
        )


def compare_saved_outcome(a: Audit, saved: dict, own: dict) -> None:
    for name in ("controls", "fell", "full_survival", "walking"):
        a.need(saved[name] == own[name], "Independent physical replay outcome: " + name)
    a.need(saved["requested_controls"] == HORIZON, "Declared complete native execution horizon")
    for name in (
        "velocity_mae",
        "mean_velocity",
        "raw_native_cost",
        "absorbing_tail_cost",
        "complete_cost",
    ):
        if own[name] is None:
            a.need(saved.get(name) is None, "Undefined physical outcome remains missing: " + name)
        else:
            a.close(
                saved[name],
                own[name],
                "Independent physical outcome arithmetic: " + name,
                atol=ATOL,
            )


def replay_findings(a: Audit, run: Path) -> dict:
    launch, config, inputs, environment = validate_inputs(a, run)
    result = a.json(run / "result.json")
    a.need(
        result["schema"] == "trajectory-pi-replay-result-1"
        and result["status"] in ("ORIGIN_REPLAY_PASS", "FAIL_STOPPED"),
        "Closed R0 scientific result",
    )
    a.bind(run / "launch.json", result["launch"])
    ledger = audit_ledger(a, run)
    a.equal(result["work"], ledger["counts"], "Saved result and independently reconciled paid work")
    a.need(not ledger["rejected"], "No rejected call is omitted from a complete replay result")
    counts, limits = ledger["counts"], config["limits"]
    for unit, maximum in limits.items():
        if unit not in ("wall_seconds", "output_bytes", "memory_bytes"):
            a.need(
                counts.get("requested_" + unit, 0) <= maximum,
                "Finite one-attempt work limit: " + unit,
            )
    a.need(
        counts.get("requested_environment_constructions", 0) == 1
        and counts.get("requested_restores", 0) <= 1,
        "No environment replacement or restore retry",
    )
    for prefix in ("requested_", "completed_", "unknown_partial_"):
        a.need(
            counts.get(prefix + "physics_substeps", 0)
            == 4 * counts.get(prefix + "native_controls", 0),
            "Exactly four physics substeps per native control; no model physics",
        )
        for unit in (
            "model_controls",
            "neural_rows",
            "backward_rows",
            "optimizer_steps",
            "pi_passes",
        ):
            a.need(
                counts.get(prefix + unit, 0) == 0,
                "Forbidden extra work is absent: " + prefix + unit,
            )
    a.need(
        ledger["receipt"]["wall_seconds"] <= limits["wall_seconds"]
        and ledger["receipt"]["peak_process_rss_bytes"] <= limits["memory_bytes"],
        "Internal finite wall/RSS ceilings",
    )
    rows = [json.loads(line) for line in (run / "ledger.jsonl").read_text().splitlines()]
    native = [
        row
        for row in rows
        if row["event"] == "requested" and row["charges"].get("native_controls", 0)
    ]
    for row in native:
        a.need(
            row["charges"]["native_controls"] == 1 and row["charges"]["physics_substeps"] == 4,
            "Each actual replay control has its own reservation",
        )
    model_events = [row for row in rows if row["event"] == "native_model_bound"]
    model = None
    if native or result.get("model_bindings"):
        a.need(
            len(model_events) == 1
            and result.get("model_bindings") == model_events[0]["model_bindings"],
            "Compiled model/options are bound once, before native execution",
        )
        a.need(
            not native or model_events[0]["id"] < native[0]["id"],
            "No control precedes compiled-model binding",
        )
        model_file = a.bound(result["model_bindings"])
        a.need(
            model_file.resolve() == (run / "model_bindings.json").resolve(),
            "Local model-binding artifact",
        )
        model = a.json(model_file)
        a.need(
            model["dimensions"] == [9, 9, 6] and type(model["spec"]) is int and model["spec"] > 0,
            "Bound Walker and integration-state dimensions",
        )
        for name, filename in (("compiled", "compiled.mjb"), ("options", "model_options.json")):
            a.need(
                a.bound(model[name]).resolve() == (run / filename).resolve(),
                "Bound local native artifact: " + name,
            )
        a.need(
            model["compiled"]["bytes"] > 0 and bool(a.json(run / "model_options.json")),
            "Nonempty compiled model/options captured; not loaded by audit",
        )
    original = original_arrays(a, a.bound(inputs["source_recording"]))
    linux = trace_arrays(a, a.bound(inputs["linux_trace"]))
    source_outcome = physical_outcome(a, original)
    linux_outcome = physical_outcome(a, linux)
    linux_comparison = compare_trace(a, linux, original, (0.0, ATOL))
    trace_present = (run / "trace.h5").is_file()
    if trace_present:
        if result.get("trace"):
            a.bind(run / "trace.h5", result["trace"])
        actual = trace_arrays(a, run / "trace.h5")
        origin_comparison = compare_trace(a, actual, original, (0.0, ATOL))
        origin_outcome = physical_outcome(a, actual)
        a.need(
            len(actual["actions"]) == counts.get("completed_native_controls", 0),
            "Every completed native control has one preserved physical action and terminal boundary",
        )
        if result.get("source_comparisons"):
            compare_saved_comparisons(a, result["source_comparisons"], origin_comparison)
            compare_saved_comparisons(a, result["linux_comparisons"], linux_comparison)
            a.need(
                result["initial_context_exact"] is origin_comparison["initial_context_passed"]
                and result["actions_exact"] is origin_comparison["actions_exact_bytes"],
                "Exact initial bytes/actions are independent of tolerance",
            )
            compare_saved_outcome(a, result["outcome"], origin_outcome)
        else:
            a.need(
                result["status"] == "FAIL_STOPPED" and result.get("error"),
                "Partial trace lacks completed comparisons only after a preserved execution failure",
            )
    else:
        a.need(
            result["status"] == "FAIL_STOPPED"
            and result.get("error")
            and not counts.get("completed_native_controls", 0),
            "Missing trace only before completed native execution",
        )
        origin_comparison = dict(passed=False, arrays={}, reason="No native trace preserved")
        origin_outcome = dict(walking=False, native_cost_formula_matches=False, controls=0)
    conclusion = diagnosis(origin_comparison, linux_comparison, origin_outcome)
    if result["status"] == "ORIGIN_REPLAY_PASS":
        a.need(
            conclusion["origin_replay_passed"]
            and ledger["receipt"]["status"] == "COMPLETED"
            and counts.get("completed_native_controls") == HORIZON
            and counts.get("completed_physics_substeps") == PHYSICS_SUBSTEPS
            and not ledger["failed"]
            and all(
                not key.startswith("unknown_partial_") or value == 0
                for key, value in counts.items()
            ),
            "Scientific replay pass requires complete clean finite work",
        )
    else:
        a.need(
            ledger["receipt"]["status"] == "FAILED" and result.get("error"),
            "Original scientific or execution failure remains failed",
        )
        conclusion["origin_replay_passed"] = False
    expected_diagnosis = (
        "SUPPORTS_PLATFORM_RUNTIME_REPLAYABILITY_DIFFERENCE"
        if conclusion["origin_replay_passed"] and not linux_comparison["strict_arrays_passed"]
        else "UNRESOLVED_ORIGIN_REPLAY"
    )
    a.need(result["diagnosis"] == expected_diagnosis, "No stronger platform/runtime causal claim")
    external_path = run / "external_receipt.json"
    external = a.json(external_path) if external_path.is_file() else None
    if external is not None:
        a.need(
            external["schema"] == "trajectory-pi-phase-1" and external["phase"] == "origin-replay",
            "One external origin-replay phase receipt",
        )
        a.equal(
            external,
            a.json(run / "supervisor/origin-replay.json"),
            "External supervisor receipt identity",
        )
        a.need(
            external["wall_seconds"] <= limits["wall_seconds"] + 1,
            "External finite deadline, allowing one supervisor shutdown interval",
        )
        if result["status"] == "ORIGIN_REPLAY_PASS":
            a.need(
                external["exit_code"] == 0
                and external["limit_error"] is None
                and external.get("remaining_group_terminated") is False,
                "A clean scientific pass has no external timeout, termination or error",
            )
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    a.need(size <= limits["output_bytes"], "R0 total saved output ceiling")
    return dict(
        schema="trajectory-pi-replay-audit-findings-1",
        status="PASS_SAVED_EVIDENCE",
        original_status=result["status"],
        origin_replay_passed=conclusion["origin_replay_passed"],
        conclusion=conclusion,
        original=source_outcome,
        origin=origin_outcome,
        linux=linux_outcome,
        source_comparison=origin_comparison,
        linux_comparison=linux_comparison,
        work=counts,
        ledger_closed=True,
        native_model_bindings=model,
        environment=environment,
        external_receipt=external,
        study_output_bytes_before_audit=size,
        original_terminal_observation="Derived by the unchanged state feature map from the recorded terminal physical state; no terminal state is fabricated",
        limitations=[
            "The audit compares saved arrays and opaque binary/model hashes without dynamics or inference.",
            "Bound current native binaries/options do not prove that every unrecorded original native setting was identical.",
            "An origin pass plus Linux mismatch supports practical replayability separation, not a unique floating-point or platform cause.",
            "An evidence-audit pass alone does not authorize the conditional next stage; origin_replay_passed is the separate scientific gate.",
        ],
    )


def audit_replay(run: Path, out: Path, *, seconds: float = 60) -> dict:
    """Create one independent saved-only audit; preserve failures without retry."""
    run, out = Path(run).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not np.isfinite(seconds) or not 0 < seconds <= 60:
        raise ValueError("Saved R0 audit seconds must lie in (0,60]")
    a = Audit(seconds=seconds, neural_rows=0, output_bytes=64 * 1024**2, memory_bytes=2 * 1024**3)
    findings, error = {}, None
    try:
        a.bind(Path(__file__))
        findings = replay_findings(a, run)
        forbidden = (
            "mujoco",
            "src.envs.",
            "src.mppi.",
            "src.gps.mppi_gps.policy",
            "src.gps.mppi_gps.trajectory_pi_replay_validation",
            "src.gps.mppi_gps.trajectory_pi_native",
            "src.gps.mppi_gps.reporting.trajectory_pi_replay",
        )
        a.need(
            not any(name.startswith(forbidden) for name in sys.modules),
            "Independent R0 audit imported no simulator, producer or trainable policy",
        )
        for path in list(a.files):
            a.bind(Path(path))
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    receipt = dict(
        schema="trajectory-pi-replay-independent-audit-1",
        passed=error is None,
        status="PASS_SAVED_EVIDENCE" if error is None else "FAIL_AUDIT",
        origin_replay_passed=findings.get("origin_replay_passed", False),
        error=error,
        checks=a.checks,
        bound_files=len(a.files),
        hashed_file_bytes=sum(v["bytes"] for v in a.files.values()),
        new_native_controls=0,
        new_model_controls=0,
        new_physics_substeps=0,
        new_neural_rows=0,
        new_gradients=0,
        new_optimizer_updates=0,
        wall_seconds=time.monotonic() - a.started,
        cpu_seconds=time.process_time() - a.cpu_started,
    )
    for name, value in (
        ("findings.json", findings),
        ("file_bindings.json", a.files),
        ("receipt.json", receipt),
    ):
        encoded = json.dumps(value, indent=2, allow_nan=False, default=_plain) + "\n"
        if (
            sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) + len(encoded.encode())
            > a.output_limit
        ):
            raise AuditFailure("R0 audit output ceiling exhausted")
        with (out / name).open("x") as stream:
            stream.write(encoded)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args(argv)
    receipt = audit_replay(args.run, args.out, seconds=args.seconds)
    print(json.dumps(receipt, allow_nan=False))
    return int(not receipt["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
