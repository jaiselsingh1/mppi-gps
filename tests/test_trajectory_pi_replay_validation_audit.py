"""Synthetic saved-only R0 audit fixtures: no physics, policy or fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits import trajectory_pi_replay_validation as module


def complete_arrays():
    n = 1000
    states = np.zeros((n + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(n + 1, dtype=float) * 0.008
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    return dict(
        states=states,
        warmstarts=np.zeros((n + 1, 9)),
        integrations=np.zeros((n + 1, 91)),
        actions=np.zeros((n, 6)),
        costs=np.zeros(n),
        observations=module.observations(states),
        native_fall_cost=np.zeros(n),
    )


def original_arrays(raw):
    return {key: value.copy() for key, value in raw.items() if key != "native_fall_cost"}


def write_original(path, raw):
    values = {key: value.copy() for key, value in raw.items() if key in ("actions", "costs")}
    values["observations"] = raw["observations"][:-1]
    for name, terminal in (
        ("states", "terminal_state"),
        ("warmstarts", "terminal_warmstarts"),
        ("integrations", "terminal_integrations"),
    ):
        values[name], values[terminal] = raw[name][:-1], raw[name][-1]
    np.savez(path, **values)


def test_original_join_uses_separately_preserved_terminal_context(tmp_path):
    raw = complete_arrays()
    raw["warmstarts"][-1] = np.arange(9)
    raw["integrations"][-1] = np.arange(91)
    path = tmp_path / "teacher.npz"
    write_original(path, raw)
    result = module.original_arrays(module.Audit(neural_rows=0), path)
    for name in module.CONTEXT:
        np.testing.assert_array_equal(result[name], raw[name])
    assert result["observations"].shape == (1001, 17)


def test_strict_matching_and_exact_initial_action_contracts_are_distinct():
    raw = complete_arrays()
    source = original_arrays(raw)
    result = module.compare_trace(module.Audit(neural_rows=0), raw, source, (0.0, 1e-10))
    assert result["passed"] and result["actions_exact_bytes"]
    raw["actions"][10, 0] = 5e-12
    result = module.compare_trace(module.Audit(neural_rows=0), raw, source, (0.0, 1e-10))
    assert result["strict_arrays_passed"] and not result["actions_exact_bytes"]
    assert not result["passed"]
    raw = complete_arrays()
    raw["integrations"][0, 0] = 5e-12
    result = module.compare_trace(module.Audit(neural_rows=0), raw, source, (0.0, 1e-10))
    assert result["strict_arrays_passed"] and not result["initial_context_passed"]
    assert not result["passed"]


def test_crossing_rows_have_declared_strict_tolerance_and_no_rescaling():
    expected = np.zeros((10, 19))
    actual = expected.copy()
    actual[2, 0] = 1e-12
    actual[4, 1] = 2e-10
    result = module.compare_arrays(actual, expected)
    assert not result["passed"] and result["rtol"] == 0 and result["atol"] == 1e-10
    assert result["first_nonidentical_row"] == 2
    assert result["first_above_threshold_row"]["1e-10"] == 4
    assert result["maximum_absolute_error"] == 2e-10


def test_missing_postfall_rows_are_reported_missing_without_zero_fill():
    raw = complete_arrays()
    original = original_arrays(raw)
    for name in module.CONTEXT + ("observations",):
        raw[name] = raw[name][:4]
    for name in ("actions", "costs", "native_fall_cost"):
        raw[name] = raw[name][:3]
    raw["states"][-1, 2] = 0.5
    raw["observations"] = module.observations(raw["states"])
    raw["costs"] = module.running_cost(raw["states"][1:], raw["actions"])
    raw["costs"][-1] += 20
    raw["native_fall_cost"][-1] = 20
    a = module.Audit(neural_rows=0)
    outcome = module.physical_outcome(a, raw)
    comparison = module.compare_trace(a, raw, original, (0.0, 1e-10))
    assert outcome["fell"] and not outcome["walking"]
    assert outcome["raw_native_cost"] == 25 and outcome["absorbing_tail_cost"] == 4985
    assert outcome["complete_cost"] == 5010
    assert comparison["arrays"]["states"]["compared_rows"] == 4
    assert comparison["arrays"]["states"]["missing_expected_rows"] == 997
    assert not comparison["passed"]


def test_linux_still_walking_does_not_make_path_replay_pass():
    raw = complete_arrays()
    source = original_arrays(raw)
    origin = module.compare_trace(module.Audit(neural_rows=0), raw, source, (0.0, 1e-10))
    linux_raw = copy.deepcopy(raw)
    linux_raw["states"][10:, 10] += 0.0001
    linux_raw["observations"] = module.observations(linux_raw["states"])
    linux_raw["costs"] = module.running_cost(linux_raw["states"][1:], linux_raw["actions"])
    linux = module.compare_trace(module.Audit(neural_rows=0), linux_raw, source, (0.0, 1e-10))
    outcome = module.physical_outcome(module.Audit(neural_rows=0), linux_raw)
    assert outcome["walking"] and outcome["native_cost_formula_matches"]
    assert not linux["passed"]
    result = module.diagnosis(origin, linux, outcome)
    assert result["origin_replay_passed"] and not result["linux_replay_passed"]
    assert result["interpretation"] == "ORIGIN_REPLAY_CONFIRMED_LINUX_DIVERGES"
    assert "does not identify" in result["scope"]


def test_nonfinite_saved_array_cannot_pass_comparison():
    source = np.zeros((10, 19))
    damaged = source.copy()
    damaged[3, 8] = np.nan
    result = module.compare_arrays(damaged, source)
    assert not result["passed"] and result["nonfinite_compared_elements"] == 1
    assert result["first_above_threshold_row"]["1e-10"] == 3


def test_actual_trace_requires_true_n_plus_one_headers(tmp_path):
    import h5py

    raw = complete_arrays()
    path = tmp_path / "trace.h5"
    with h5py.File(path, "x") as target:
        for name, value in raw.items():
            target.create_dataset(name, data=value)
    loaded = module.trace_arrays(module.Audit(neural_rows=0), path)
    assert loaded["states"].shape == (1001, 19)
    with h5py.File(path, "a") as target:
        del target["integrations"]
        target.create_dataset("integrations", data=np.zeros((1000, 91)))
    with pytest.raises(module.AuditFailure, match="N\\+1"):
        module.trace_arrays(module.Audit(neural_rows=0), path)


def test_r0_audit_import_does_not_load_simulator_policy_or_producer():
    command = (
        "from src.gps.mppi_gps.audits import trajectory_pi_replay_validation; import sys; "
        "assert not any(n.startswith(('mujoco','src.envs.','src.mppi.',"
        "'src.gps.mppi_gps.policy','src.gps.mppi_gps.trajectory_pi_native',"
        "'src.gps.mppi_gps.trajectory_pi_preflight')) for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def binding(path):
    return dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def write_trace(path, raw):
    import h5py

    with h5py.File(path, "x") as target:
        for name, value in raw.items():
            target.create_dataset(name, data=value)


def producer_comparisons(actual, source):
    own = module.compare_trace(module.Audit(neural_rows=0), actual, source, (0.0, 1e-10))
    return {
        name: dict(
            passed=row["passed"],
            exact_bytes=row["exact_bytes"],
            source_shape=row["expected_shape"],
            replay_shape=row["actual_shape"],
            maximum_absolute_error=row["maximum_absolute_error"],
            first_above_atol=row["first_above_threshold_row"]["1e-10"],
        )
        for name, row in own["arrays"].items()
    }


def saved_replay_fixture(run, *, scientific_pass=True):
    """Construct exact R0 filesystem contracts, without any native call."""
    run.mkdir()
    root = Path(module.__file__).resolve().parents[4]
    source_names = sorted(
        {
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
    )
    roster = [
        dict(
            path=name,
            bytes=(root / name).stat().st_size,
            sha256=hashlib.sha256((root / name).read_bytes()).hexdigest(),
        )
        for name in source_names
    ]
    manifest = dict(
        schema="trajectory-pi-replay-source-1",
        git_commit="0" * 40,
        files=roster,
        roster_sha256=hashlib.sha256(
            json.dumps(roster, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    put(run / "source_manifest.json", manifest)
    original = complete_arrays()
    write_original(run / "original.npz", original)
    linux = copy.deepcopy(original)
    linux["states"][10:, 10] += 1e-4
    linux["observations"] = module.observations(linux["states"])
    linux["costs"] = module.running_cost(linux["states"][1:], linux["actions"])
    write_trace(run / "paid_linux.h5", linux)
    environment = dict(
        schema="trajectory-pi-replay-environment-1",
        platform="synthetic-MacOS",
        machine="arm64",
        python="3.11.15",
        versions=dict(numpy="2.4.3", mujoco="3.6.0", scipy="1.17.1", h5py="3.16.0"),
    )
    for name in ("python_binary", "native_mujoco.so", "native_numpy.so"):
        (run / name).write_bytes(b"synthetic immutable binary bytes")
    environment["interpreter"] = binding(run / "python_binary")
    environment["binary_files"] = [
        binding(run / name) for name in ("native_mujoco.so", "native_numpy.so")
    ]
    put(run / "environment.json", environment)
    put(
        run / "origin_launch_receipt.json",
        dict(
            hardware=dict(
                machine="arm64", python="3.11.15 Clang", dependency_versions=environment["versions"]
            )
        ),
    )
    inputs = dict(
        schema="trajectory-pi-replay-bindings-1",
        source_recording=binding(run / "original.npz"),
        linux_trace=binding(run / "paid_linux.h5"),
        origin_launch_receipt=binding(run / "origin_launch_receipt.json"),
        environment=binding(run / "environment.json"),
    )
    put(run / "bindings.json", inputs)
    config = json.loads((root / "configs/studies/trajectory_pi_replay_r0_1.json").read_text())
    config["inputs"] = {
        name: inputs[name] for name in ("source_recording", "linux_trace", "origin_launch_receipt")
    }
    config["output_directory"] = str(run)
    put(run / "config.json", config)
    launch = dict(
        schema="trajectory-pi-replay-launch-1",
        mode="origin",
        source_directory=str(root),
        output_directory=str(run),
        config=binding(run / "config.json"),
        source_manifest=binding(run / "source_manifest.json"),
        bindings=binding(run / "bindings.json"),
    )
    put(run / "launch.json", launch)
    (run / "compiled.mjb").write_bytes(b"synthetic opaque compiled model")
    put(run / "model_options.json", dict(timestep=0.002, gravity=[0.0, 0.0, -9.81]))
    put(
        run / "model_bindings.json",
        dict(
            compiled=binding(run / "compiled.mjb"),
            options=binding(run / "model_options.json"),
            dimensions=[9, 9, 6],
            spec=16383,
        ),
    )
    actual = copy.deepcopy(original)
    if not scientific_pass:
        actual["states"][5:, 10] += 1e-4
        actual["observations"] = module.observations(actual["states"])
        actual["costs"] = module.running_cost(actual["states"][1:], actual["actions"])
    write_trace(run / "trace.h5", actual)
    rows, counts = [], {}

    def call(name, charges):
        token = len(rows)
        rows.append(dict(id=token, event="requested", operation=name, charges=charges))
        rows.append(
            dict(id=token + 1, event="completed", request_id=token, operation=name, charges=charges)
        )
        for unit, amount in charges.items():
            for prefix in ("requested_", "completed_"):
                counts[prefix + unit] = counts.get(prefix + unit, 0) + amount

    call("environment", dict(environment_constructions=1))
    call("freeze_compiled_model", dict(compiled_model_saves=1))
    rows.append(
        dict(
            id=len(rows),
            event="native_model_bound",
            model_bindings=binding(run / "model_bindings.json"),
        )
    )
    call("restore", dict(restores=1, external_mj_forward_calls=1))
    for _ in range(1000):
        call("native_control", dict(native_controls=1, physics_substeps=4))
    receipt = dict(
        status="COMPLETED" if scientific_pass else "FAILED",
        counts=counts,
        wall_seconds=2.0,
        cpu_seconds=1.0,
        peak_process_rss_bytes=1000000,
        output_bytes_before_receipt=0,
        physical_disk_io_bytes=None,
    )
    rows.append(dict(id=len(rows), event="closed", receipt=receipt))
    (run / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    put(run / "receipt.json", receipt)
    own = module.physical_outcome(module.Audit(neural_rows=0), actual)
    outcome = {
        key: own[key]
        for key in (
            "controls",
            "fell",
            "walking",
            "full_survival",
            "velocity_mae",
            "mean_velocity",
            "raw_native_cost",
            "absorbing_tail_cost",
            "complete_cost",
        )
    }
    outcome["requested_controls"] = 1000
    result = dict(
        schema="trajectory-pi-replay-result-1",
        status="ORIGIN_REPLAY_PASS" if scientific_pass else "FAIL_STOPPED",
        diagnosis="SUPPORTS_PLATFORM_RUNTIME_REPLAYABILITY_DIFFERENCE"
        if scientific_pass
        else "UNRESOLVED_ORIGIN_REPLAY",
        launch=binding(run / "launch.json"),
        model_bindings=binding(run / "model_bindings.json"),
        outcome=outcome,
        source_comparisons=producer_comparisons(actual, original),
        linux_comparisons=producer_comparisons(linux, original),
        initial_context_exact=True,
        actions_exact=True,
        trace=binding(run / "trace.h5"),
        work=counts,
    )
    if not scientific_pass:
        result["error"] = "Frozen strict origin-host replay requirement failed"
    put(run / "result.json", result)
    external = dict(
        schema="trajectory-pi-phase-1",
        phase="origin-replay",
        wall_seconds=3.0,
        exit_code=0 if scientific_pass else 1,
        limit_error=None,
        remaining_group_terminated=False,
    )
    put(run / "external_receipt.json", external)
    put(run / "supervisor/origin-replay.json", external)
    return result


@pytest.mark.parametrize("scientific_pass", [True, False])
def test_full_saved_adapter_keeps_evidence_pass_separate_from_replay_pass(
    tmp_path, scientific_pass
):
    run = tmp_path / "r0"
    saved_replay_fixture(run, scientific_pass=scientific_pass)
    findings = module.replay_findings(module.Audit(neural_rows=0), run)
    assert findings["status"] == "PASS_SAVED_EVIDENCE"
    assert findings["origin_replay_passed"] is scientific_pass
    assert findings["origin"]["walking"]
    assert findings["work"]["completed_native_controls"] == 1000
    assert findings["work"]["completed_physics_substeps"] == 4000


def test_adapter_rejects_tampered_compiled_model_and_no_retry_contract(tmp_path):
    run = tmp_path / "r0"
    saved_replay_fixture(run)
    (run / "compiled.mjb").write_bytes(b"different compiled bytes")
    with pytest.raises(module.AuditFailure, match="Binding differs"):
        module.replay_findings(module.Audit(neural_rows=0), run)


def test_cli_saved_adapter_is_create_only_zero_science_and_accepts_bounded_seconds(tmp_path):
    run = tmp_path / "r0"
    saved_replay_fixture(run)
    out = tmp_path / "audit"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.gps.mppi_gps.audits.trajectory_pi_replay_validation",
            "--run",
            str(run),
            "--out",
            str(out),
            "--seconds",
            "20",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["passed"] and receipt["origin_replay_passed"]
    assert all(
        receipt[name] == 0
        for name in (
            "new_native_controls",
            "new_model_controls",
            "new_physics_substeps",
            "new_neural_rows",
            "new_gradients",
            "new_optimizer_updates",
        )
    )
    with pytest.raises(FileExistsError):
        module.audit_replay(run, out)
