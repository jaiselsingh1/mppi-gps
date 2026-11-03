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
