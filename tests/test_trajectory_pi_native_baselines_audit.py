"""Saved worker-native baseline fixtures; no native execution or fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits import trajectory_pi_native_baselines as module
from src.gps.mppi_gps.audits.trajectory_pi import AuditFailure, observations, running_cost


def arrays():
    states = np.zeros((1001, 19), dtype=np.float64)
    states[:, 0] = np.arange(1001) * 0.008
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    return dict(
        states=states,
        warmstarts=np.zeros((1001, 9)),
        integrations=np.zeros((1001, 91)),
        observations=observations(states),
        actions=np.zeros((1000, 6)),
        costs=np.zeros(1000),
        native_fall_cost=np.zeros(1000),
    )


def fixture():
    original = arrays()
    capture = copy.deepcopy(original)
    capture["states"][1:, 4] += 0.01 * np.sin(np.arange(1, 1001) * 0.008)
    capture["observations"] = observations(capture["states"])
    replay = copy.deepcopy(capture)
    candidate = dict(
        states=capture["states"][None, 1:].copy(),
        actions=original["actions"][None].copy(),
        costs=np.zeros(1),
        initial_state=original["states"][0].copy(),
        initial_warmstart=original["warmstarts"][0].copy(),
        initial_integration=original["integrations"][0].copy(),
        final_state=original["states"][0].copy(),
        final_warmstart=original["warmstarts"][0].copy(),
        final_integration=original["integrations"][0].copy(),
    )
    return original, capture, replay, candidate


def test_native_baseline_can_differ_from_historical_path_but_not_context_or_actions():
    original, capture, replay, candidate = fixture()
    result = module.baseline_comparisons(
        module.Audit(neural_rows=0), original, capture, replay, candidate
    )
    assert result["passed"]
    assert not result["original_to_worker_difference"]["states"]["passed"]
    assert result["repeat"]["passed"] and result["candidate_post_states"]["passed"]
    assert result["original_actions_exact_bytes"] and all(result["original_initial_bytes"].values())
    assert (
        result["native_capture_outcome"]["walking"] and result["native_replay_outcome"]["walking"]
    )


@pytest.mark.parametrize("field", ["states", "warmstarts", "integrations", "observations", "costs"])
def test_native_repeat_strict_tolerance_cannot_be_relaxed(field):
    original, capture, replay, candidate = fixture()
    if replay[field].ndim == 1:
        replay[field][3] += 2e-10
    else:
        replay[field][3, 0] += 2e-10
    with pytest.raises(AuditFailure, match="N\\+1 repeat"):
        module.baseline_comparisons(
            module.Audit(neural_rows=0), original, capture, replay, candidate
        )


def test_k1_post_states_are_post_action_not_shifted_pre_action_states():
    original, capture, replay, candidate = fixture()
    candidate["states"][0] = capture["states"][:-1]
    with pytest.raises(AuditFailure, match="post-action states"):
        module.baseline_comparisons(
            module.Audit(neural_rows=0), original, capture, replay, candidate
        )


def test_candidate_aggregate_cost_has_its_separate_frozen_tolerance():
    original, capture, replay, candidate = fixture()
    candidate["costs"][0] = 5e-10
    result = module.baseline_comparisons(
        module.Audit(neural_rows=0), original, capture, replay, candidate
    )
    assert result["passed"]
    candidate["costs"][0] = 2e-9
    with pytest.raises(AuditFailure, match="task-cost arithmetic"):
        module.baseline_comparisons(
            module.Audit(neural_rows=0), original, capture, replay, candidate
        )


@pytest.mark.parametrize(
    "part", ["capture_initial", "capture_actions", "batch_initial", "batch_actions", "batch_final"]
)
def test_original_inputs_are_byte_exact_even_below_the_state_tolerance(part):
    original, capture, replay, candidate = fixture()
    if part == "capture_initial":
        capture["integrations"][0, 0] += 1e-12
    if part == "capture_actions":
        capture["actions"][0, 0] += 1e-12
    if part == "batch_initial":
        candidate["initial_warmstart"][0] += 1e-12
    if part == "batch_actions":
        candidate["actions"][0, 0, 0] += 1e-12
    if part == "batch_final":
        candidate["final_integration"][0] += 1e-12
    with pytest.raises(AuditFailure):
        module.baseline_comparisons(
            module.Audit(neural_rows=0), original, capture, replay, candidate
        )


def test_standing_survival_does_not_qualify_as_a_worker_native_walking_baseline():
    original, capture, replay, candidate = fixture()
    for raw in (capture, replay):
        raw["states"][1:, 10] = 0
        raw["observations"] = observations(raw["states"])
        raw["costs"] = running_cost(raw["states"][1:], raw["actions"])
    candidate["states"] = capture["states"][None, 1:]
    candidate["costs"][0] = 3000
    with pytest.raises(AuditFailure, match="complete native traces walk"):
        module.baseline_comparisons(
            module.Audit(neural_rows=0), original, capture, replay, candidate
        )


def test_batch_missing_integration_is_missing_proof_not_invented_context():
    original, capture, replay, candidate = fixture()
    del candidate["initial_integration"]
    with pytest.raises(KeyError):
        module.baseline_comparisons(
            module.Audit(neural_rows=0), original, capture, replay, candidate
        )


def test_native_auditor_import_is_lazy_and_failed_receipt_performs_zero_science(tmp_path):
    command = (
        "from src.gps.mppi_gps.audits import trajectory_pi_native_baselines; import sys; "
        "assert not any(n.startswith(('mujoco','src.envs.','src.mppi.',"
        "'src.gps.mppi_gps.policy','src.gps.mppi_gps.trajectory_pi_native_baselines',"
        "'src.gps.mppi_gps.trajectory_pi_native')) for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
    out = tmp_path / "audit"
    receipt = module.audit_baselines(tmp_path / "missing", out)
    assert not receipt["passed"] and not receipt["native_validation_passed"]
    assert json.loads((out / "receipt.json").read_text())["new_native_controls"] == 0
    assert receipt["new_neural_rows"] == receipt["new_optimizer_updates"] == 0
    with pytest.raises(FileExistsError):
        module.audit_baselines(tmp_path / "missing", out)


def binding(path):
    return dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def save_trace(path, raw):
    import h5py

    with h5py.File(path, "x") as target:
        for key, value in raw.items():
            target.create_dataset(key, data=value)


def saved_worker_fixture(base):
    """A small frozen source export and exact RV raw contracts, with no execution."""
    run, source, previous = base / "rv", base / "source", base / "previous"
    run.mkdir()
    source.mkdir()
    previous.mkdir()
    root = Path(module.__file__).resolve().parents[4]
    names = [
        "src/__init__.py",
        "src/gps/__init__.py",
        "src/gps/mppi_gps/__init__.py",
        "src/gps/mppi_gps/audits/__init__.py",
        "src/gps/mppi_gps/audits/trajectory_pi.py",
        "src/gps/mppi_gps/audits/trajectory_pi_replay_validation.py",
        "src/gps/mppi_gps/audits/trajectory_pi_native_baselines.py",
    ]
    for name in names:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, source / name)
    original, capture, replay, candidate = fixture()
    inputs_episodes, saved_episodes, teachers, old_files = [], [], [], []
    for slot in (0, 1):
        path = base / f"original_{slot}.npz"
        values = {key: original[key] for key in ("actions", "costs")}
        values["observations"] = original["observations"][:-1]
        for key, terminal in (
            ("states", "terminal_state"),
            ("warmstarts", "terminal_warmstarts"),
            ("integrations", "terminal_integrations"),
        ):
            values[key], values[terminal] = original[key][:-1], original[key][-1]
        np.savez(path, **values)
        old_path = previous / "teachers" / f"slot{slot}.npz"
        old_path.parent.mkdir(exist_ok=True)
        np.savez(old_path, **original)
        old_relative = str(old_path.relative_to(previous))
        teachers.append(dict(slot=slot, source=binding(path), file_relative=old_relative))
        old_files.append(
            dict(
                path=old_relative, bytes=old_path.stat().st_size, sha256=binding(old_path)["sha256"]
            )
        )
        episode = dict(
            slot=slot, original_episode=binding(path), original_prepared=binding(old_path)
        )
        inputs_episodes.append(episode)
        for kind, raw in (("capture", capture), ("replay", replay)):
            save_trace(run / f"{kind}_{slot}.h5", raw)
        np.savez(run / f"candidate_{slot}.npz", **candidate)
        outcome = module.physical_outcome(module.Audit(neural_rows=0), capture)
        saved_episodes.append(
            dict(
                episode,
                capture=binding(run / f"capture_{slot}.h5"),
                replay=binding(run / f"replay_{slot}.h5"),
                candidate=binding(run / f"candidate_{slot}.npz"),
                outcome=outcome,
                native_replay_passed=True,
                single_candidate_passed=True,
                initial_context_exact=True,
                actions_exact=True,
            )
        )
    old_cfg = json.loads((root / "configs/studies/trajectory_pi_p1.json").read_text())
    for row in old_cfg["teachers"]:
        slot = row["slot"]
        row["episode"] = inputs_episodes[slot]["original_episode"]
        row["prepared_path"] = teachers[slot]["file_relative"]
    put(base / "old_config.json", old_cfg)
    put(
        previous / "manifest.json",
        dict(
            schema="trajectory-pi-prepared-1",
            config_sha256=binding(base / "old_config.json")["sha256"],
            teachers=teachers,
            files=old_files,
        ),
    )
    r0_result = dict(
        schema="trajectory-pi-replay-result-1",
        status="ORIGIN_REPLAY_PASS",
        work={
            "requested_native_controls": 1000,
            "completed_native_controls": 1000,
            "requested_physics_substeps": 4000,
            "completed_physics_substeps": 4000,
        },
        source_comparisons={"states": dict(passed=True)},
    )
    put(base / "r0_result.json", r0_result)
    put(base / "r0_receipt.json", dict(passed=True, origin_replay_passed=True))
    put(base / "r0_findings.json", dict(origin_replay_passed=True))
    put(base / "r0_external.json", dict(replay=dict(exit_code=0), audit=dict(exit_code=0)))
    origin = dict(
        result=binding(base / "r0_result.json"),
        audit_receipt=binding(base / "r0_receipt.json"),
        audit_findings=binding(base / "r0_findings.json"),
        external_summary=binding(base / "r0_external.json"),
        origin_replay_passed=True,
    )
    p1 = copy.deepcopy(old_cfg)
    p1.update(
        version="P1.1",
        input_mode="worker-native",
        launch_ready=True,
        lineage=dict(
            previous_config=binding(base / "old_config.json"),
            previous_prepared_manifest=binding(previous / "manifest.json"),
            origin_replay=dict(
                bindings={
                    name: origin[name]
                    for name in ("result", "audit_receipt", "audit_findings", "external_summary")
                }
            ),
        ),
    )
    p1_relative = "configs/studies/p1_1.json"
    put(source / p1_relative, p1)
    for name in ("python_binary", "mujoco_binary.so", "numpy_binary.so"):
        (base / name).write_bytes(b"synthetic native artifact bytes")
    environment = dict(
        machine="x86_64",
        python=p1["resources"]["runtime"]["python"],
        versions={
            key: p1["resources"]["runtime"][key] for key in ("numpy", "mujoco", "scipy", "h5py")
        },
        interpreter=binding(base / "python_binary"),
        binary_files=[binding(base / name) for name in ("mujoco_binary.so", "numpy_binary.so")],
    )
    put(base / "environment.json", environment)
    limits = dict(
        wall_seconds=600,
        output_bytes=64 * 1024**2,
        memory_bytes=2 * 1024**3,
        native_controls=4000,
        model_controls=2000,
        physics_substeps=24000,
        environment_constructions=1,
        restores=6,
        external_mj_forward_calls=6,
        model_queries=2,
        compiled_model_saves=1,
        neural_rows=0,
        backward_rows=0,
        optimizer_steps=0,
        pi_passes=0,
    )
    cfg = dict(
        schema="trajectory-pi-native-baselines-1",
        horizon=1000,
        physics_threads=8,
        task=p1["task"],
        p1_config_relative=p1_relative,
        p1_config_sha256=binding(source / p1_relative)["sha256"],
        limits=limits,
        tolerances=dict(
            native=dict(rtol=0, atol=1e-10),
            candidate_states=dict(rtol=0, atol=1e-10),
            candidate_complete_cost=dict(rtol=0, atol=1e-9),
        ),
        one_attempt=True,
        phase_census=dict(captures=2, native_replays=2, single_candidate_rollouts=2),
        global_limits_unchanged=p1["limits"],
    )
    put(source / "configs/studies/rv.json", cfg)
    names += [p1_relative, "configs/studies/rv.json"]
    files = [
        dict(
            path=name, bytes=(source / name).stat().st_size, sha256=binding(source / name)["sha256"]
        )
        for name in sorted(names)
    ]
    manifest = dict(
        schema="trajectory-pi-source-1",
        git_commit="0" * 40,
        files=files,
        roster_sha256=hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    put(base / "source_manifest.json", manifest)
    inputs = dict(
        previous_prepared_manifest=binding(previous / "manifest.json"),
        previous_config=binding(base / "old_config.json"),
        origin_replay=origin,
        worker_environment=binding(base / "environment.json"),
        episodes=inputs_episodes,
    )
    put(base / "bindings.json", inputs)
    launch = dict(
        schema="trajectory-pi-native-baselines-launch-1",
        source_directory=str(source),
        output_directory=str(run),
        config=binding(source / "configs/studies/rv.json"),
        source_manifest=binding(base / "source_manifest.json"),
        bindings=binding(base / "bindings.json"),
    )
    put(run / "launch.json", launch)
    (run / "compiled.mjb").write_bytes(b"synthetic compiled model")
    put(run / "model_options.json", dict(timestep=0.002))
    put(
        run / "model_bindings.json",
        dict(
            compiled=binding(run / "compiled.mjb"),
            options=binding(run / "model_options.json"),
            dimensions=[9, 9, 6],
            spec=16383,
        ),
    )
    rows, counts = [], {}

    def call(name, charges):
        token = len(rows)
        rows.extend(
            [
                dict(id=token, event="requested", operation=name, charges=charges),
                dict(
                    id=token + 1,
                    event="completed",
                    request_id=token,
                    operation=name,
                    charges=charges,
                ),
            ]
        )
        for key, value in charges.items():
            for prefix in ("requested_", "completed_"):
                counts[prefix + key] = counts.get(prefix + key, 0) + value

    call("environment", dict(environment_constructions=1))
    call("freeze_compiled_model", dict(compiled_model_saves=1))
    rows.append(
        dict(
            id=len(rows),
            event="native_model_bound",
            model_bindings=binding(run / "model_bindings.json"),
        )
    )
    for _ in range(6):
        call("restore", dict(restores=1, external_mj_forward_calls=1))
    for _ in range(4000):
        call("native_control", dict(native_controls=1, physics_substeps=4))
    for _ in range(2):
        call("native_forecast", dict(model_controls=1000, physics_substeps=4000, model_queries=1))
    receipt = dict(
        status="COMPLETED",
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
    result = dict(
        schema="trajectory-pi-native-baselines-result-1",
        status="PASS",
        model_bindings=binding(run / "model_bindings.json"),
        work=counts,
        episodes=saved_episodes,
        launch=binding(run / "launch.json"),
        config_sha256=launch["config"]["sha256"],
        p1_config_sha256=cfg["p1_config_sha256"],
        source_manifest_sha256=launch["source_manifest"]["sha256"],
        previous_prepared_manifest=inputs["previous_prepared_manifest"],
        origin_replay=origin,
    )
    put(run / "result.json", result)
    return run, source


def test_complete_frozen_saved_rv_adapter_and_cli_match_exact_work_without_physics(tmp_path):
    run, source = saved_worker_fixture(tmp_path)
    out = run / "audit"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.gps.mppi_gps.audits.trajectory_pi_native_baselines",
            "--run",
            str(run),
            "--out",
            str(out),
            "--seconds",
            "20",
        ],
        cwd=source,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads((out / "receipt.json").read_text())
    finding = json.loads((out / "findings.json").read_text())
    assert receipt["passed"] and receipt["native_validation_passed"]
    assert finding["work"]["completed_native_controls"] == 4000
    assert finding["work"]["completed_model_controls"] == 2000
    assert finding["work"]["completed_physics_substeps"] == 24000
    assert [row["slot"] for row in finding["episodes"]] == [0, 1]
    assert finding["result_binding"] == binding(run / "result.json")
    assert receipt["findings_binding"] == binding(out / "findings.json")
    assert all(
        receipt[key] == 0
        for key in (
            "new_native_controls",
            "new_model_controls",
            "new_physics_substeps",
            "new_neural_rows",
            "new_gradients",
            "new_optimizer_updates",
        )
    )


@pytest.mark.parametrize(
    "changed", [None, "result", "audit_receipt", "audit_findings", "external_summary"]
)
def test_real_p1_1_config_origin_proof_uses_all_four_canonical_binding_names(monkeypatch, changed):
    """Use the actual tracked config contract; no ignored run files required."""
    root = Path(module.__file__).resolve().parents[4]
    p1 = json.loads((root / "configs/studies/trajectory_pi_p1_1.json").read_text())
    names = ("result", "audit_receipt", "audit_findings", "external_summary")
    frozen = p1["lineage"]["origin_replay"]["bindings"]
    assert set(frozen) == set(names)
    proof = {name: dict(path=f"/frozen/{name}.json", **frozen[name]) for name in names}
    proof["origin_replay_passed"] = True
    values = dict(
        result=dict(status="ORIGIN_REPLAY_PASS", source_comparisons={"states": dict(passed=True)}),
        audit_receipt=dict(passed=True, origin_replay_passed=True),
        audit_findings=dict(origin_replay_passed=True),
        external_summary=dict(replay=dict(exit_code=0), audit=dict(exit_code=0)),
    )
    a = module.Audit(neural_rows=0)
    monkeypatch.setattr(a, "bound", lambda item: Path(item["path"]))
    monkeypatch.setattr(a, "json", lambda path: values[path.stem])
    if changed is not None:
        proof[changed]["sha256"] = "0" * 64
        with pytest.raises(AuditFailure, match="Frozen origin proof identity"):
            module._origin_proof(a, proof, p1)
    else:
        result = module._origin_proof(a, proof, p1)
        assert set(result) == set(names)
    assert a.requested_rows == a.completed_rows == 0
