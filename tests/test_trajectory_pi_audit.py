"""Independent small saved-output audit fixtures; no task dynamics or fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits import trajectory_pi as audit


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def bind(path):
    return dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def states(n):
    value = np.zeros((n, 19), dtype=np.float64)
    value[:, 0] = np.arange(n) * 0.008
    value[:, 2] = 1.25
    value[:, 10] = 1.5
    return value


def payload(action=0.0):
    state = {
        "observation_mean": np.zeros(17, np.float32),
        "observation_std": np.ones(17, np.float32),
    }
    for layer, out, inp in ((0, 128, 17), (2, 128, 128), (4, 6, 128)):
        state[f"network.{layer}.weight"] = np.zeros((out, inp), np.float32)
        state[f"network.{layer}.bias"] = np.zeros(out, np.float32)
    state["network.4.bias"][:] = np.arctanh(np.float32(action))
    return dict(
        schema="walker-tanh-policy-v1",
        obs_dim=17,
        hidden_dim=128,
        seed=0,
        state_dict=state,
        policy_hash=audit.policy_hash(state),
        optimizer_state=None,
        private_rng_state=np.array([1, 2], dtype=np.uint8),
    )


def trace(horizon, fall=None):
    n = horizon if fall is None else fall + 1
    physical = states(n + 1)
    if fall is not None:
        physical[-1, 2] = 0.5
    actions = np.zeros((n, 6), dtype=np.float64)
    terminal = 20.0 * ~audit.healthy(physical[1:])
    return dict(
        states=physical,
        actions=actions,
        observations=audit.observations(physical),
        warmstarts=np.zeros((n + 1, 9)),
        integrations=np.zeros((n + 1, 91)),
        costs=audit.running_cost(physical[1:], actions) + terminal,
        native_fall_cost=terminal,
    )


def test_independent_head_is_batch_invariant_and_bounded():
    a = audit.Audit()
    rng = np.random.Generator(np.random.PCG64(91))
    p = payload()
    for name, value in p["state_dict"].items():
        if name.startswith("network"):
            value[:] = rng.normal(0, 0.03, value.shape)
    obs = rng.normal(size=(43, 17)).astype(np.float64)
    batch = a.predict(p, obs)
    grouped = np.concatenate([a.predict(p, row[None]) for row in obs])
    np.testing.assert_array_equal(batch, grouped)
    assert batch.dtype == np.float32
    assert np.isfinite(batch).all() and np.max(np.abs(batch)) <= 1
    assert a.completed_rows == a.requested_rows == 86


def test_numpy_audit_imports_no_producer_or_simulator_in_fresh_process():
    command = (
        "from src.gps.mppi_gps.audits import trajectory_pi; import sys; "
        "assert not any(n.startswith(('mujoco','src.envs.','src.mppi.',"
        "'src.gps.mppi_gps.policy','src.gps.mppi_gps.trajectory_pi_optimizer',"
        "'src.gps.mppi_gps.trajectory_pi_student')) for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def test_prefix_cost_includes_fall_once_and_ignores_missing_suffix():
    a = audit.Audit()
    raw = trace(1000, fall=2)
    result = audit.outcome(a, raw, 1000)
    assert result["controls"] == 3
    assert result["fell"] and not result["walking"] and not result["full_survival"]
    assert result["raw_native_cost"] == 25
    assert result["absorbing_tail_cost"] == 5 * 997
    post = np.concatenate((raw["states"][1:], np.full((997, 19), np.nan)))[None]
    actions = np.concatenate((raw["actions"], np.full((997, 6), np.nan)))[None]
    cost, mask, valid = audit.batch_cost(post, actions)
    assert cost[0] == result["complete_cost"]
    assert mask.sum() == 3 and valid[0]
    D, rows = audit.reconstruction(a, payload(0.25), raw["states"][0], post, actions)
    assert rows == 3 and np.isfinite(D).all()


def test_reconstruction_is_pre_action_float64_residual_and_invalid_skips_head():
    a = audit.Audit()
    p = payload(0.1)
    post = np.broadcast_to(states(3)[None], (2, 3, 19)).copy()
    actions = np.full((2, 3, 6), 0.2, dtype=np.float64)
    post[1, 0, 10] = np.nan
    D, rows = audit.reconstruction(a, p, states(1)[0], post, actions)
    prediction = a.predict(p, audit.observations(states(1)))[0].astype(np.float64)
    assert D[0] == 0.5 * 3 * np.sum((0.2 - prediction) ** 2)
    assert np.isinf(D[1]) and rows == 3
    assert a.completed_rows == 4


def test_walk_requires_complete_horizon_and_velocity_accuracy():
    a = audit.Audit()
    assert audit.outcome(a, trace(1000), 1000)["walking"]
    assert not audit.outcome(a, trace(999), 999)["walking"]
    raw = trace(1000)
    raw["states"][:, 10] = 0
    raw["observations"] = audit.observations(raw["states"])
    raw["costs"] = audit.running_cost(raw["states"][1:], raw["actions"])
    result = audit.outcome(a, raw, 1000)
    assert result["full_survival"] and not result["walking"]


def test_double_terminal_or_fabricated_postfall_state_fails():
    a = audit.Audit()
    raw = trace(1000, 2)
    raw["costs"][-1] += 20
    with pytest.raises(audit.AuditFailure, match="fall terminal"):
        audit.outcome(a, raw, 1000)
    raw = trace(5)
    raw["states"][2, 2] = 0.5
    with pytest.raises(audit.AuditFailure, match="first actual fall"):
        audit.outcome(a, raw, 5)


def test_budget_stops_before_head_and_does_not_change_rng():
    a = audit.Audit(neural_rows=2)
    old = np.random.get_state()
    with pytest.raises(audit.AuditFailure, match="row budget"):
        a.predict(payload(), np.zeros((3, 17)))
    assert a.requested_rows == a.completed_rows == 0
    assert audit.typed_fingerprint(np.random.get_state()) == audit.typed_fingerprint(old)


def test_distinct_file_policy_array_and_typed_hashes(tmp_path):
    p = payload()
    value = p["state_dict"]["network.4.bias"]
    path = tmp_path / "array.bin"
    path.write_bytes(value.tobytes())
    hashes = {
        bind(path)["sha256"],
        audit.array_hash(value),
        audit.typed_fingerprint(value),
        p["policy_hash"],
    }
    assert len(hashes) == 4
    assert audit.typed_fingerprint([1, 2]) != audit.typed_fingerprint((1, 2))
    assert audit.typed_fingerprint({1: "x"}) != audit.typed_fingerprint({"1": "x"})


def ledger(directory, charges, *, fail=False):
    rows, counts = [], {}
    for operation, values in charges:
        request = len(rows)
        rows.append(dict(id=request, event="requested", operation=operation, charges=values))
        rows.append(
            dict(
                id=request + 1,
                event="failed" if fail else "completed",
                request_id=request,
                operation=operation,
                charges=values,
            )
        )
        for name, amount in values.items():
            for prefix in ("requested_", "unknown_partial_" if fail else "completed_"):
                counts[prefix + name] = counts.get(prefix + name, 0) + amount
    receipt = dict(
        status="FAILED" if fail else "COMPLETED",
        counts=counts,
        wall_seconds=0.01,
        cpu_seconds=0.001,
        peak_process_rss_bytes=1000,
        output_bytes_before_receipt=0,
        physical_disk_io_bytes=None,
    )
    rows.append(dict(id=len(rows), event="closed", receipt=receipt))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    put(directory / "receipt.json", receipt)
    return rows


def test_ledger_recomputes_paid_unknown_and_does_not_erase_failure(tmp_path):
    ledger(tmp_path, [("native", dict(model_controls=100, physics_substeps=400))], fail=True)
    result = audit.audit_ledger(audit.Audit(), tmp_path)
    assert result["failed"] == 1 and result["receipt"]["status"] == "FAILED"
    assert result["counts"]["unknown_partial_model_controls"] == 100
    assert result["counts"].get("completed_model_controls", 0) == 0


def test_ledger_missing_duplicate_or_changed_charge_fails(tmp_path):
    rows = ledger(tmp_path, [("head", dict(neural_rows=17))])
    rows[1]["charges"]["neural_rows"] = 18
    (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(audit.AuditFailure):
        audit.audit_ledger(audit.Audit(), tmp_path)


def test_gibbs_ignores_invalid_candidates_and_temperature_is_own_algebra():
    scores = np.array([2.0, 3.0, 5.0, np.inf, np.nan])
    lam = audit.temperature(scores, 0.4)
    weights, ess = audit.gibbs(scores, lam)
    assert weights[3] == weights[4] == 0
    np.testing.assert_allclose(ess, 2.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        weights[:3],
        np.exp(-(scores[:3] - 2) / lam) / np.exp(-(scores[:3] - 2) / lam).sum(),
        rtol=0,
        atol=1e-16,
    )


def test_binding_changed_after_first_read_fails(tmp_path):
    path = tmp_path / "evidence"
    path.write_text("before")
    a = audit.Audit()
    a.bind(path)
    path.write_text("after")
    with pytest.raises(audit.AuditFailure, match="changed during"):
        a.bind(path)


def criteria_fixture():
    config = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/studies/trajectory_pi_p1.json").read_text()
    )
    history = []
    confirmation = {}
    for round_index in range(21):
        record = dict(round=round_index, streams={}, parents={})
        for stream in range(3):
            source = dict(outcome=dict(walking=True, full_survival=True, complete_cost=400.0))
            arms = {}
            for arm in ("independent", "coupled"):
                gap = 1.0 if arm == "independent" or round_index < 16 else 0.6
                walks = 3 if arm == "coupled" else 0
                arms[arm] = dict(
                    bank=[copy.deepcopy(source), copy.deepcopy(source)],
                    raw_candidates=[
                        dict(result=copy.deepcopy(source)),
                        dict(result=copy.deepcopy(source)),
                    ],
                    teacher_after_fit=dict(rms=gap),
                    student=dict(summary=dict(walking=walks, full_survival=5)),
                    queries={
                        str(anchor): dict(present=True, qualified=True, action_gap=gap)
                        for anchor in (0, 100)
                    },
                )
            record["streams"][str(stream)] = arms
            record["parents"][str(stream)] = dict(summary=dict(walking=0, full_survival=5))
            confirmation[str(stream)] = {
                arm: dict(summary=dict(walking=3 if arm == "coupled" else 0, full_survival=5))
                for arm in ("independent", "coupled", "parent")
            }
        history.append(record)
    return config, history, confirmation


def test_independent_criteria_require_same_streams_walking_gaps_and_teacher():
    config, history, confirmation = criteria_fixture()
    result = audit.independent_criteria(history, confirmation, config)
    assert result["verdict"] == "HELPED_FINITE_ALTERNATING_LOOP"
    assert result["joint_streams"] == [0, 1, 2]
    assert not result["full_project_success"]
    history[18]["streams"]["0"]["independent"]["queries"]["100"] = dict(
        present=False, qualified=False, action_gap=None
    )
    history[19]["streams"]["1"]["coupled"]["queries"]["100"]["qualified"] = False
    result = audit.independent_criteria(history, confirmation, config)
    assert result["verdict"] == "PARTIAL_WALKING_GAIN_WITH_UNRESOLVED_GAP"
    assert not result["per_stream"]["0"]["student_gap"]["100"]["resolved"]
    assert result["joint_streams"] == [2]


def test_teacher_deterioration_or_standing_is_not_success():
    config, history, confirmation = criteria_fixture()
    for record in history:
        for stream in range(3):
            record["streams"][str(stream)]["coupled"]["student"]["summary"]["walking"] = 0
            confirmation[str(stream)]["coupled"]["summary"]["walking"] = 0
            if record["round"] >= 16:
                for item in record["streams"][str(stream)]["coupled"]["bank"]:
                    item["outcome"]["complete_cost"] = 600
    result = audit.independent_criteria(history, confirmation, config)
    assert result["verdict"] == "NO_CONFIRMED_HELP"
    assert not result["passed"]


def test_cli_failure_receipt_is_preserved_create_only_without_science(tmp_path):
    out = tmp_path / "audit"
    assert audit.main(["--run", str(tmp_path / "missing"), "--out", str(out)]) == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert not receipt["passed"]
    assert receipt["new_model_controls"] == receipt["new_native_controls"] == 0
    assert receipt["new_gradients"] == receipt["new_optimizer_updates"] == 0
    with pytest.raises(FileExistsError):
        audit.main(["--run", str(tmp_path / "missing"), "--out", str(out)])


def native_fixture(directory, alpha):
    """Write a tiny complete saved job, without running a producer or simulator."""
    import h5py
    import torch

    directory.mkdir()
    h, k, iterations, seed = 3, 4, 2, 71
    p = payload()
    p["state_dict"] = {name: torch.from_numpy(value) for name, value in p["state_dict"].items()}
    p["private_rng_state"] = torch.tensor([1, 2], dtype=torch.uint8)
    checkpoint = directory / "incoming.pt"
    torch.save(p, checkpoint)
    raw = trace(h)
    context = directory / "incoming.npz"
    np.savez(context, **raw)
    job = dict(
        kind="solve",
        out=str(directory),
        horizon=h,
        checkpoint=str(checkpoint),
        checkpoint_binding=bind(checkpoint),
        context_file=str(context),
        context_binding=bind(context),
        context_step=0,
        initial_plan_file=str(context),
        initial_plan_binding=bind(context),
        planner_seed=seed,
        iterations=iterations,
        planner=dict(H=h, K=k, use_is_correction=False, lam_target_neff_fraction=0.75),
        alpha=alpha,
        full_banks=[0, 1],
    )
    current = np.zeros((h, 6))
    rng = np.random.Generator(np.random.PCG64(seed + 20000))
    curves = []
    with h5py.File(directory / "iterations.h5", "x") as groups:
        for i in range(iterations):
            bank_U = np.clip(current[None] + np.array([0, -0.1, 0.2, 0.3])[:, None, None], -1, 1)
            bank_states = np.broadcast_to(raw["states"][None, 1:], (k, h, 19)).copy()
            C = 0.001 * np.sum(bank_U**2, axis=(1, 2))
            D = 0.5 * np.sum(bank_U**2, axis=(1, 2))
            J = C + alpha * D
            lam = audit.temperature(J, 0.75)
            weights, ess = audit.gibbs(J, lam)
            after = np.clip(
                current + np.einsum("k,kha->ha", weights, bank_U - current[None]), -1, 1
            )
            group = groups.create_group(f"iteration_{i:04d}")
            for name, value in dict(
                nominal_before=current,
                nominal_after=after,
                nominal_post_states=bank_states[0],
                native_costs=C,
                total_objectives=J,
                weights=weights,
                active_prefix_packed=np.packbits(np.ones(k * h, dtype=bool)),
                prefix_lengths=np.full(k, h),
                complete_candidates=np.ones(k, dtype=bool),
                walking_candidates=np.ones(k, dtype=bool),
                prefix_running_cost=C,
                fall_terminal_cost=np.zeros(k),
                absorbing_tail_cost=np.zeros(k),
            ).items():
                group.create_dataset(name, data=value)
            if alpha:
                group.create_dataset("reconstruction", data=D)
            group.attrs.update(
                nominal_D=D[0],
                nominal_D_available=True,
                population_D_measured=bool(alpha),
                **{"lambda": lam},
                n_eff=ess,
                scale=0.1 ** (i / (iterations - 1)),
                rng_before=json.dumps(rng.bit_generator.state),
                maximum_weight=weights.max(),
                update_rms=np.sqrt(np.mean((after - current) ** 2)),
                best_walking_population_J=float(J.min()),
            )
            np.savez(directory / f"bank_{i:04d}.npz", states=bank_states, actions=bank_U)
            curves.append(
                dict(
                    C=float(C[0]),
                    D=float(D[0]) if alpha else None,
                    J=float(J[0]),
                    best_J=0.0,
                    best_iteration=0,
                )
            )
            current = after
            rng.standard_normal((k, h, 6))
    np.savez(
        directory / "solution.npz",
        actions=raw["actions"],
        batch_post_states=raw["states"][1:],
        final_unselected_actions=current,
    )
    np.savez(directory / "last_mean.npz", states=raw["states"][1:], actions=current)
    np.savez(directory / "selected_predictions.npz", predictions=np.zeros((h, 6)))
    with h5py.File(directory / "trace.h5", "x") as output:
        for name, value in raw.items():
            output.create_dataset(name, data=value)
    measured = audit.outcome(audit.Audit(), raw, h)
    model = iterations * k * h + h
    neural = iterations * (k * h if alpha else h) + (h if alpha else 0) + h
    charges = dict(
        environment_constructions=1,
        checkpoint_loads=1,
        numpy_constructions=1,
        native_controls=h,
        neural_rows=neural,
        restores=2,
        external_mj_forward_calls=2,
        model_queries=iterations + 1,
        pi_passes=iterations,
        model_controls=model,
        physics_substeps=4 * (model + h),
    )
    job["limits"] = dict(charges, wall_seconds=10, output_bytes=1024**2, memory_bytes=1024**3)
    put(directory / "job.json", job)
    final_C = float(0.001 * np.sum(current**2))
    final_D = float(0.5 * np.sum(current**2))
    result = dict(
        kind="solve",
        policy_hash=p["policy_hash"],
        policy_checkpoint=bind(checkpoint),
        trace=bind(directory / "trace.h5"),
        outcome=measured,
        curve=curves,
        selected_D=0.0,
        selected_J=0.0,
        selected_iteration=0,
        selected_task_batch=0.0,
        selected_total_batch=0.0,
        final_mean_task=final_C,
        final_mean_total=final_C + alpha * final_D,
        final_mean_score_available=True,
        optimizer_counts=dict(
            requested_passes=iterations,
            completed_passes=iterations,
            requested_model_controls=iterations * k * h,
            completed_model_controls=iterations * k * h,
            unknown_partial_model_controls=0,
        ),
    )
    put(directory / "result.json", result)
    rows = ledger(directory, [("synthetic_saved_job", charges)])
    rows.insert(-1, dict(id=2, event="derived_cost_components", rows=iterations * k * h))
    rows[-1]["id"] = 3
    receipt = rows[-1]["receipt"]
    receipt["counts"]["derived_cost_component_rows"] = iterations * k * h
    receipt["result_binding"] = bind(directory / "result.json")
    put(directory / "receipt.json", receipt)
    (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return job


@pytest.mark.parametrize("alpha", [0.0, 0.12])
def test_complete_native_saved_job_joins_scores_contexts_census_and_sparse_banks(tmp_path, alpha):
    directory = tmp_path / "job"
    native_fixture(directory, alpha)
    a = audit.Audit()
    result = audit.audit_native(a, directory)
    assert result["nominal_paths"] == result["full_banks"] == 2
    assert result["selected_iteration"] == 0
    assert result["outcome"]["full_survival"] and not result["outcome"]["walking"]
    assert result["counts"]["derived_cost_component_rows"] == 24
    assert a.completed_rows == (36 if alpha else 9)


def test_corrupt_saved_population_weight_is_rejected_without_physics(tmp_path):
    import h5py

    directory = tmp_path / "job"
    native_fixture(directory, 0.12)
    with h5py.File(directory / "iterations.h5", "a") as output:
        output["iteration_0000/weights"][0] = 0.99
    with pytest.raises(audit.AuditFailure, match="Gibbs weights"):
        audit.audit_native(audit.Audit(), directory)


def test_open_ledger_preserves_unresolved_request_without_fake_completion(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text(
        json.dumps(
            dict(id=0, event="requested", operation="forecast", charges=dict(model_controls=256000))
        )
        + "\n"
    )
    result = audit.audit_ledger(audit.Audit(), tmp_path, allow_open=True)
    assert result["receipt"] is None
    assert result["counts"] == dict(requested_model_controls=256000)
    assert result["unresolved_request_upper_charges"] == dict(model_controls=256000)
    assert not result["original_closure_present"]


def test_saved_p0_replay_and_operative_continuation_file_contract(tmp_path, monkeypatch):
    """Exercise real saved-array joins and expected P0 filenames, decoding mocked."""
    import h5py

    preflight_dir, prepared_dir = tmp_path / "preflight", tmp_path / "prepared"
    preflight_dir.mkdir()
    prepared_dir.mkdir()
    teacher_rows = []
    for slot in (0, 1):
        raw = trace(1000)
        relative = f"slot{slot}.npz"
        source = dict(raw, observations=raw["observations"][:-1])
        np.savez(prepared_dir / relative, **source)
        with h5py.File(preflight_dir / f"teacher_{slot}_replay.h5", "x") as target:
            for name, value in raw.items():
                target.create_dataset(name, data=value)
        teacher_rows.append(dict(slot=slot, file_relative=relative))
    zero = trace(16)
    with h5py.File(preflight_dir / "zero_residual_native.h5", "x") as target:
        for name, value in zero.items():
            target.create_dataset(name, data=value)
    np.savez(
        preflight_dir / "zero_residual_forecast.npz",
        states=zero["states"][None, 1:],
        costs=np.zeros(1),
        D=np.zeros(1),
        matching=np.zeros(1),
    )
    np.savez(
        preflight_dir / "action_function_parity.npz",
        observations=zero["observations"][:-1],
        numpy_actions=zero["actions"],
        torch_actions=zero["actions"],
    )
    for name, alpha in (("independent", 0.0), ("coupled", 0.12)):
        native_fixture(preflight_dir / ("solve_" + name), alpha)
    ledger(preflight_dir, [("saved_paid_work", dict(native_controls=2016, optimizer_steps=8))])
    required = (
        "full_context_replay",
        "pre_action_index",
        "full_sequence_no_shift",
        "score_formula",
        "checkpoint_continuation",
        "fall_accounting",
        "budget_enforcement",
        "zero_residual_reproduction",
    )
    preflight = dict(
        status="PASS",
        checks=[dict(name=name, passed=True) for name in required],
        work=json.loads((preflight_dir / "receipt.json").read_text())["counts"],
    )
    launch = dict(
        preflight=dict(path=str(preflight_dir / "preflight.json")),
        prepared_directory=str(prepared_dir),
    )
    config = dict(
        arms=[dict(name="independent", alpha=0.0), dict(name="coupled", alpha=0.12)],
        planner=dict(lam_target_neff_fraction=0.75),
        prerequisites=dict(action_function_parity_atol=1e-6),
    )
    prepared = dict(teachers=teacher_rows, parents=[dict(directory="initial/stream0")])
    parent = payload()
    full = dict(
        parent,
        optimizer_state=dict(
            schema="walker-adam-proximal-fit-v1",
            parent_policy_hash="initial",
            updates=2004,
            adam=dict(moment=[0.1, 0.2]),
        ),
    )
    resumed = copy.deepcopy(full)
    resumed["optimizer_state"]["parent_policy_hash"] = "middle"
    expected_directories = []

    def decoded_stream(_audit, directory):
        expected_directories.append(directory.name)
        assert directory.name in ("continuation_full", "continuation_resumed")
        return (
            full if directory.name == "continuation_full" else resumed,
            dict(global_rng_fingerprint="identical"),
        )

    a = audit.Audit()
    monkeypatch.setattr(a, "checkpoint", lambda path: parent)
    monkeypatch.setattr(audit, "audit_stream", decoded_stream)
    result = audit.audit_preflight(a, launch, config, prepared, preflight)
    assert result["status"] == "PASS" and result["repeated_physics"] == 0
    assert expected_directories == ["continuation_full", "continuation_resumed"]
    resumed["optimizer_state"]["adam"]["moment"][0] += 0.01
    with pytest.raises(audit.AuditFailure, match="operative Adam"):
        audit.audit_preflight(a, launch, config, prepared, preflight)


@pytest.mark.parametrize("status", ["NOT_ATTEMPTED", "EXTERNAL_STOPPED"])
def test_missing_preflight_closure_never_fabricates_paid_counts(tmp_path, status):
    preflight = dict(
        status=status,
        accounting_complete=False,
        work={},
        checks=[],
        external_stop=dict(stage="capacity" if status == "NOT_ATTEMPTED" else "preflight"),
    )
    launch = dict(preflight=dict(path=str(tmp_path / "preflight.json")))
    result = audit.audit_preflight(audit.Audit(), launch, {}, {}, preflight)
    assert not result["original_closure_present"]
    assert result["accounting_complete"] is (status == "NOT_ATTEMPTED")
    assert result["work"] == {}


def native_lineage_fixture(base):
    """P1.1 lineage proof with synthetic files; no checkpoint construction."""
    import h5py

    old_dir, new_dir = base / "previous", base / "prepared"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()
    root = Path(__file__).resolve().parents[1]
    old_config = json.loads((root / "configs/studies/trajectory_pi_p1.json").read_text())
    parents, files, declared = [], [], []
    for stream in range(3):
        relative = f"initial/stream{stream}"
        parent = dict(
            stream=stream,
            directory=relative,
            source=dict(path=f"/original/s{stream}.pt", bytes=1, sha256="a" * 64),
            policy_hash=f"p{stream}",
            optimizer_fingerprint=f"adam{stream}",
            global_rng_fingerprint=f"global{stream}",
        )
        parents.append(parent)
        for filename in ("policy.pt", "global_rng.pt", "stream.json"):
            old_path = old_dir / relative / filename
            old_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.write_bytes(f"immutable stream{stream} {filename}".encode())
            new_path = new_dir / relative / filename
            new_path.parent.mkdir(parents=True, exist_ok=True)
            new_path.write_bytes(old_path.read_bytes())
            bound = bind(old_path)
            files.append(
                dict(path=f"{relative}/{filename}", bytes=bound["bytes"], sha256=bound["sha256"])
            )
            declared.append(
                dict(stream=stream, filename=filename, bytes=bound["bytes"], sha256=bound["sha256"])
            )
    teacher_rows, episodes = [], []
    for slot in range(2):
        raw = trace(1000)
        original = base / f"original_{slot}.npz"
        np.savez(original, **raw)
        runtime_copy = bind(original)
        source = dict(runtime_copy, path=f"/unavailable/historical/mac/H4/teacher_{slot}.npz")
        old_config["teachers"][slot]["episode"] = source
        (new_dir / "originals").mkdir(exist_ok=True)
        (new_dir / "originals" / f"teacher_{slot}.npz").write_bytes(original.read_bytes())
        old_native = old_dir / f"slot{slot}.npz"
        np.savez(old_native, **raw)
        capture = base / f"capture_{slot}.h5"
        with h5py.File(capture, "x") as target:
            for name, value in raw.items():
                target.create_dataset(name, data=value)
        replay = base / f"replay_{slot}.h5"
        replay.write_bytes(capture.read_bytes())
        candidate = base / f"candidate_{slot}.npz"
        np.savez(candidate, states=raw["states"][None, 1:])
        relative = f"teachers/slot{slot}.npz"
        (new_dir / "teachers").mkdir(exist_ok=True)
        canonical = {
            key: raw[key] for key in ("states", "warmstarts", "integrations", "actions", "costs")
        }
        canonical["observations"] = raw["observations"][:-1]
        np.savez(new_dir / relative, **canonical)
        outcome = dict(
            controls=1000, walking=True, full_survival=True, complete_cost=0.0, velocity_mae=0.0
        )
        teacher_rows.append(
            dict(
                slot=slot,
                source=source,
                native_capture=bind(capture),
                file_relative=relative,
                complete_cost=0.0,
                walking=True,
            )
        )
        episodes.append(
            dict(
                slot=slot,
                original_episode=runtime_copy,
                original_prepared=bind(old_native),
                capture=bind(capture),
                replay=bind(replay),
                candidate=bind(candidate),
                outcome=outcome,
                native_replay_passed=True,
                single_candidate_passed=True,
                initial_context_exact=True,
                actions_exact=True,
            )
        )
    put(base / "previous_config.json", old_config)
    previous = dict(
        schema="trajectory-pi-prepared-1",
        config_sha256=bind(base / "previous_config.json")["sha256"],
        parents=parents,
        files=files,
        teachers=[],
    )
    put(old_dir / "manifest.json", previous)
    r0_work = dict(
        requested_native_controls=1000,
        completed_native_controls=1000,
        requested_physics_substeps=4000,
        completed_physics_substeps=4000,
    )
    r0_result = dict(
        schema="trajectory-pi-replay-result-1",
        status="ORIGIN_REPLAY_PASS",
        work=r0_work,
        source_comparisons={"states": dict(passed=True)},
    )
    r0_receipt = dict(passed=True, origin_replay_passed=True)
    put(base / "r0_result.json", r0_result)
    put(base / "r0_receipt.json", r0_receipt)
    put(base / "r0_findings.json", dict(origin_replay_passed=True))
    put(base / "r0_external.json", dict(status="CLOSED"))
    origin = dict(
        result=bind(base / "r0_result.json"),
        audit_receipt=bind(base / "r0_receipt.json"),
        audit_findings=bind(base / "r0_findings.json"),
        external_summary=bind(base / "r0_external.json"),
        origin_replay_passed=True,
    )
    cfg = copy.deepcopy(old_config)
    cfg.update(
        version="P1.1",
        input_mode="worker-native",
        lineage=dict(
            previous_config=bind(base / "previous_config.json"),
            previous_prepared_manifest=bind(old_dir / "manifest.json"),
            parent_stream_files=dict(files=declared),
            origin_replay=dict(
                bindings={
                    name: origin[name]
                    for name in ("result", "audit_receipt", "audit_findings", "external_summary")
                }
            ),
        ),
    )
    put(base / "new_config.json", cfg)
    rv_work = {
        prefix + unit: value
        for unit, value in (
            ("native_controls", 4000),
            ("model_controls", 2000),
            ("physics_substeps", 24000),
        )
        for prefix in ("requested_", "completed_")
    }
    put(base / "rv_result.json", dict(status="PASS", work=rv_work))
    put(base / "rv_receipt.json", dict(status="COMPLETED", counts=rv_work))
    validation = dict(
        schema="trajectory-pi-native-validation-1",
        status="PASS",
        config_sha256="b" * 64,
        p1_config_sha256=bind(base / "new_config.json")["sha256"],
        source_manifest_sha256="c" * 64,
        previous_prepared_manifest=bind(old_dir / "manifest.json"),
        origin_replay=origin,
        result=bind(base / "rv_result.json"),
        receipt=bind(base / "rv_receipt.json"),
        episodes=episodes,
    )
    finding = dict(
        native_validation_passed=True,
        result_binding=validation["result"],
        config_sha256=validation["config_sha256"],
        p1_config_sha256=validation["p1_config_sha256"],
        source_manifest_sha256=validation["source_manifest_sha256"],
        episodes=episodes,
        work=rv_work,
    )
    put(base / "rv_findings.json", finding)
    put(
        base / "rv_audit.json",
        dict(
            passed=True,
            native_validation_passed=True,
            findings_binding=bind(base / "rv_findings.json"),
        ),
    )
    validation["native_audit"] = dict(
        receipt=bind(base / "rv_audit.json"), findings=bind(base / "rv_findings.json")
    )
    put(base / "validation_manifest.json", validation)
    prepared = dict(
        version="P1.1",
        input_mode="worker-native",
        new_global_envelopes_created=0,
        config_sha256=validation["p1_config_sha256"],
        parents=copy.deepcopy(parents),
        teachers=teacher_rows,
        previous_config=bind(base / "previous_config.json"),
        previous_prepared_manifest=bind(old_dir / "manifest.json"),
        validation_manifest=bind(base / "validation_manifest.json"),
        imported_work=[
            dict(phase="r0", receipt=origin["result"], counts=r0_work),
            dict(phase="rv", receipt=validation["receipt"], counts=rv_work),
        ],
    )
    return cfg, prepared, new_dir


def test_p1_1_native_lineage_keeps_all_nine_parent_files_and_new_capture_provenance(tmp_path):
    cfg, prepared, data = native_lineage_fixture(tmp_path)
    result = audit.audit_native_input_lineage(audit.Audit(), cfg, prepared, data)
    assert result["input_mode"] == "worker-native"
    assert len(result["copied_incoming_files"]) == 9
    assert result["new_parent_refits"] == 0
    assert result["r0_work"]["completed_native_controls"] == 1000
    assert result["worker_native_validation_work"]["completed_model_controls"] == 2000
    assert prepared["teachers"][0]["source"]["path"].startswith("/unavailable/historical/mac/")
    assert (
        prepared["teachers"][0]["source"]["path"]
        != json.loads(Path(prepared["validation_manifest"]["path"]).read_text())["episodes"][0][
            "original_episode"
        ]["path"]
    )


@pytest.mark.parametrize(
    "bad", ["gate", "parent", "capture", "import", "origin_findings", "origin_external"]
)
def test_p1_1_native_lineage_rejects_recipe_or_provenance_drift(tmp_path, bad):
    cfg, prepared, data = native_lineage_fixture(tmp_path)
    if bad == "gate":
        cfg["admission"]["maximum_cost_ratio_to_initial"] += 0.01
    if bad == "parent":
        (data / "initial/stream2/global_rng.pt").write_bytes(b"reseeded")
    if bad == "capture":
        path = data / "teachers/slot1.npz"
        with np.load(path, allow_pickle=False) as raw:
            values = {key: raw[key].copy() for key in raw.files}
        values["states"][1, 2] += 0.01
        np.savez(path, **values)
    if bad == "import":
        prepared["imported_work"].append(prepared["imported_work"][0])
    if bad == "origin_findings":
        cfg["lineage"]["origin_replay"]["bindings"]["audit_findings"]["sha256"] = "0" * 64
    if bad == "origin_external":
        cfg["lineage"]["origin_replay"]["bindings"]["external_summary"]["sha256"] = "0" * 64
    with pytest.raises(audit.AuditFailure):
        audit.audit_native_input_lineage(audit.Audit(), cfg, prepared, data)


def test_phase_accounting_keeps_r0_rv_p0_and_loop_counters_separate(tmp_path):
    cfg, prepared, data = native_lineage_fixture(tmp_path)
    lineage = audit.audit_native_input_lineage(audit.Audit(), cfg, prepared, data)
    p0 = dict(
        requested_native_controls=2016,
        completed_native_controls=2016,
        requested_model_controls=1024016,
        completed_model_controls=1024016,
        requested_physics_substeps=4104128,
        completed_physics_substeps=4104128,
    )
    loop = dict(
        requested_native_controls=100,
        completed_native_controls=100,
        requested_model_controls=256000,
        completed_model_controls=256000,
        requested_physics_substeps=1024400,
        completed_physics_substeps=1024400,
    )
    total = {}
    for values in [lineage["r0_work"], lineage["worker_native_validation_work"], p0, loop]:
        for key, value in values.items():
            total[key] = total.get(key, 0) + value
    coordinator = dict(antecedents=prepared["imported_work"], prerequisite_counts=p0, counts=total)
    result = audit.phase_accounting(audit.Audit(), coordinator, dict(work=p0), lineage)
    assert result["phases"]["study"] == loop
    assert result["combined"]["completed_native_controls"] == 7116
    assert result["imports_counted_once"]
    coordinator["prerequisite_counts"] = {
        key: value + lineage["r0_work"].get(key, 0) for key, value in p0.items()
    }
    with pytest.raises(audit.AuditFailure, match="not cumulative"):
        audit.phase_accounting(audit.Audit(), coordinator, dict(work=p0), lineage)


def test_coordinator_ledger_imports_origin_and_rv_once_without_overwriting_p0(tmp_path):
    _, prepared, _ = native_lineage_fixture(tmp_path)
    directory = tmp_path / "coordinator"
    directory.mkdir()
    p0 = dict(
        requested_native_controls=2016,
        completed_native_controls=2016,
        requested_physics_substeps=8064,
        completed_physics_substeps=8064,
    )
    rows = []
    for paid in prepared["imported_work"]:
        rows.append(dict(id=len(rows), event="antecedent_work_included", **paid))
    rows.append(dict(id=len(rows), event="prerequisite_work_included", counts=p0))
    total = {}
    for values in [p0, *[row["counts"] for row in prepared["imported_work"]]]:
        for key, value in values.items():
            total[key] = total.get(key, 0) + value
    receipt = dict(status="COMPLETED", counts=total)
    rows.append(dict(id=len(rows), event="closed", receipt=receipt))
    put(directory / "receipt.json", receipt)
    (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    saved = audit.audit_ledger(audit.Audit(), directory, coordinator=True)
    assert saved["counts"]["completed_native_controls"] == 7016
    assert saved["antecedents"] == prepared["imported_work"]
    assert saved["prerequisite_counts"] == p0
    rows.insert(-1, dict(rows[0], id=len(rows) - 1))
    rows[-1]["id"] = len(rows) - 1
    (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(audit.AuditFailure, match="exactly once"):
        audit.audit_ledger(audit.Audit(), directory, coordinator=True)
