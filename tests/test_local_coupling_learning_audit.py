"""Synthetic saved-only proof fixtures; no physics or fitting."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps.audits import local_coupling_learning as audit
from src.gps.mppi_gps.audits.trajectory_pi import running_cost


def physical(n):
    state = np.zeros((n, 19), np.float64)
    state[:, 2] = 1.25
    state[:, 10] = 1.5
    return state


def episode(n):
    states = physical(n + 1)
    return dict(
        states=states,
        observations=audit.observations(states),
        actions=np.full((n, 6), 0.2, np.float64),
    )


def incoming():
    state = {
        "observation_mean": np.zeros(17, np.float32),
        "observation_std": np.ones(17, np.float32),
    }
    for layer, out, inp in ((0, 128, 17), (2, 128, 128), (4, 6, 128)):
        state[f"network.{layer}.weight"] = np.zeros((out, inp), np.float32)
        state[f"network.{layer}.bias"] = np.zeros(out, np.float32)
    rng = np.random.Generator(np.random.PCG64(17))
    optimizer = dict(
        updates=2000,
        rng=copy.deepcopy(rng.bit_generator.state),
        adam=dict(
            param_groups=[dict(lr=0.0003, params=[0])],
            state={
                0: dict(
                    step=np.float32(2000),
                    exp_avg=np.ones((2,), np.float32),
                    exp_avg_sq=np.ones((2,), np.float32),
                )
            },
        ),
    )
    return dict(
        state_dict=state,
        optimizer_state=optimizer,
        private_rng_state=np.arange(5, dtype=np.uint8),
        policy_hash=audit.policy_hash(state),
    )


def continuation(rows=5, updates=3):
    parent = incoming()
    child = copy.deepcopy(parent)
    old = parent["optimizer_state"]
    initial = dict(
        state_dict=copy.deepcopy(parent["state_dict"]),
        adam=copy.deepcopy(old["adam"]),
        rng=copy.deepcopy(old["rng"]),
        private_rng_state=parent["private_rng_state"].copy(),
    )
    sampler = np.random.Generator(np.random.PCG64(0))
    sampler.bit_generator.state = old["rng"]
    tape = []
    proof = []
    for _ in range(updates):
        before = copy.deepcopy(sampler.bit_generator.state)
        index = sampler.integers(0, rows, size=256)
        tape.append(index)
        proof.append(
            dict(
                rng_before=before,
                rng_after=copy.deepcopy(sampler.bit_generator.state),
                indices_fingerprint=audit.typed_fingerprint(index),
            )
        )
    child["optimizer_state"]["rng"] = copy.deepcopy(sampler.bit_generator.state)
    child["optimizer_state"]["updates"] += updates
    child["optimizer_state"]["adam"]["state"][0]["step"] += np.float32(updates)
    return dict(
        parent=parent,
        child=child,
        initial=initial,
        parent_globals={"x": [1, 2]},
        child_globals={"x": [1, 2]},
        indices=np.asarray(tape),
        rng_rows=proof,
        corpus_rows=rows,
        updates=updates,
    )


def test_common_prefix_corpus_uses_every_restart_in_frozen_order():
    independent = {"b": episode(2), "a": episode(4)}
    fixed = {"a": episode(1), "b": episode(3)}
    result = audit.common_prefix_corpora(audit.Audit(), independent, fixed, ["b", "a"])
    assert result["rows"] == 3
    np.testing.assert_array_equal(result["row_events"], [[0, 0], [0, 1], [1, 0]])
    assert result["arms"]["fixed"]["actions"].dtype == np.float64
    np.testing.assert_array_equal(
        result["arms"]["independent"]["observations"], result["arms"]["fixed"]["observations"]
    )


def test_missing_restart_or_post_action_observation_blocks_corpus():
    raw = episode(2)
    with pytest.raises(audit.AuditFailure, match="Every frozen restart"):
        audit.common_prefix_corpora(audit.Audit(), {"a": raw}, {"a": raw}, ["a", "b"])
    raw = episode(2)
    raw["observations"][0, 0] += 1
    with pytest.raises(audit.AuditFailure, match="pre-action"):
        audit.common_prefix_corpora(audit.Audit(), {"a": raw}, {"a": episode(2)}, ["a"])


def test_low_candidate_survival_is_measured_without_a_gate():
    post = np.broadcast_to(physical(3)[None], (2, 3, 19)).copy()
    post[1, 0, 2] = 0.5
    candidates = np.zeros((2, 3, 6))
    initial = np.zeros((3, 6))
    c, mask, _ = audit.batch_cost(post, candidates)
    d = np.array([0.3, 0.4])
    score = c + 0.12 * d
    weights, _ = audit.gibbs(score, 3.0)
    result = audit.compare_bank(
        audit.Audit(),
        initial_actions=initial,
        post_actions=initial,
        candidates=candidates,
        post_states=post,
        saved_cost=c,
        saved_discrepancy=d,
        saved_score=score,
        saved_weights=weights,
        temperature=3.0,
        alpha=0.12,
        saved_lengths=mask.sum(1),
    )
    assert result["full_survivors"] == 1
    assert result["executable_prediction_rows"] == 4
    assert result["D_independently_recomputed"] is False


def test_wrong_total_score_or_shifted_plan_fails():
    post = physical(3)[None]
    actions = np.zeros((1, 3, 6))
    c, mask, _ = audit.batch_cost(post, actions)
    kwargs = dict(
        initial_actions=np.zeros((3, 6)),
        post_actions=np.zeros((3, 6)),
        candidates=actions,
        post_states=post,
        saved_cost=c,
        saved_discrepancy=np.ones(1),
        saved_score=c + 0.12,
        saved_weights=np.ones(1),
        temperature=3.0,
        alpha=0.12,
        saved_lengths=mask.sum(1),
    )
    kwargs["saved_score"] = c.copy()
    with pytest.raises(audit.AuditFailure, match="total score"):
        audit.compare_bank(audit.Audit(), **kwargs)


def test_continuation_accepts_variable_corpus_size_and_checks_exact_tape():
    kwargs = continuation()
    result = audit.check_continuation(audit.Audit(), **kwargs)
    assert result["updates"] == 3 and not result["gradient_arithmetic_reproduced"]
    kwargs["indices"][1, 0] = (kwargs["indices"][1, 0] + 1) % 5
    with pytest.raises(audit.AuditFailure, match="minibatch draw"):
        audit.check_continuation(audit.Audit(), **kwargs)


@pytest.mark.parametrize("changed", ["adam", "normalization", "global", "private"])
def test_continuation_rejects_resets_or_representation_change(changed):
    kwargs = continuation()
    if changed == "adam":
        kwargs["initial"]["adam"]["state"][0]["exp_avg"][:] = 0
    elif changed == "normalization":
        kwargs["child"]["state_dict"]["observation_std"][:] = 2
    elif changed == "global":
        kwargs["child_globals"] = {"x": [2, 3]}
    else:
        kwargs["child"]["private_rng_state"][0] = 99
    with pytest.raises(audit.AuditFailure):
        audit.check_continuation(audit.Audit(), **kwargs)


def test_r1_pair_identity_includes_adam_and_private_rng():
    parent = incoming()
    other = copy.deepcopy(parent)
    audit.check_paired_incoming(audit.Audit(), parent, other)
    other["optimizer_state"]["rng"]["state"]["state"] += 1
    with pytest.raises(audit.AuditFailure, match="Paired incoming"):
        audit.check_paired_incoming(audit.Audit(), parent, other)


def local(walking=6, survival=8, cost=100):
    return dict(episodes=8, local_walking=walking, full_survival=survival, mean_complete_cost=cost)


def panel(walking=0, survival=16):
    return dict(episodes=16, walking=walking, full_survival=survival)


def r1_panels():
    return {
        name: {
            str(stream): dict(
                parent=panel(), independent=panel(), positive=panel(4 if stream < 2 else 0)
            )
            for stream in range(3)
        }
        for name in ("primary", "confirmation")
    }


def r2_panels():
    return {
        name: {
            str(stream): dict(
                parent_positive=panel(4),
                parent_independent=panel(2),
                positive=panel(8),
                independent=panel(4),
            )
            for stream in range(3)
        }
        for name in ("primary", "confirmation")
    }


def test_teacher_selection_prefers_weak_without_mse_or_spectral_inputs():
    decision = audit.select_teacher_arm(
        local(), local(cost=110), local(walking=8, cost=90), {"weak": 0.01, "large": 0.2}
    )
    assert decision["selected"] == "weak"
    decision = audit.select_teacher_arm(
        local(), local(), local(walking=8, cost=90), {"weak": 0.009, "large": 0.2}
    )
    assert decision["selected"] == "large"
    decision = audit.select_teacher_arm(
        local(walking=5), local(), local(walking=8), {"weak": 0.1, "large": 0.2}
    )
    assert decision["selected"] is None


def test_r1_requires_same_two_streams_on_both_panels():
    panels = r1_panels()
    decision = audit.r1_walking_decision(panels)
    assert decision["passed"] and decision["joint_streams"] == ["0", "1"]
    panels["confirmation"]["0"]["positive"] = panel()
    panels["confirmation"]["2"]["positive"] = panel(4)
    assert not audit.r1_walking_decision(panels)["passed"]


def test_retention_without_walking_and_any_survival_regression_fail():
    panels = r1_panels()
    panels["primary"]["2"]["positive"] = panel(0, 15)
    assert not audit.r1_walking_decision(panels)["passed"]
    for rows in panels.values():
        for row in rows.values():
            row["positive"] = panel(0, 16)
            row["parent"] = panel(0, 10)
            row["independent"] = panel(0, 10)
    assert not audit.r1_walking_decision(panels)["passed"]


def test_r2_difference_in_differences_uses_distinct_incoming_parents():
    panels = r2_panels()
    decision = audit.r2_walking_decision(panels)
    assert decision["passed"]
    assert decision["per_stream"]["0"]["panels"]["primary"]["difference_in_differences"] == 2
    for rows in panels.values():
        for row in rows.values():
            row["positive"] = panel(6)
    assert not audit.r2_walking_decision(panels)["passed"]


def test_invalid_panel_and_audit_budget_stop_before_inference():
    panels = r1_panels()
    panels["primary"]["0"]["positive"] = panel(17)
    with pytest.raises(audit.AuditFailure, match="consistent outcomes"):
        audit.r1_walking_decision(panels)
    a = audit.LearningAudit(neural_rows=0)
    with pytest.raises(audit.AuditFailure, match="row budget"):
        a.predict(incoming(), np.zeros((1, 17)))
    assert a.requested_rows == a.completed_rows == 0


def test_stock_noise_is_private_and_does_not_zero_the_first_sample():
    config = dict(
        K=3,
        H=4,
        noise_sigma=0.3,
        noise_lowpass_cutoff_hz=23.4375,
        noise_lowpass_sample_rate_hz=125.0,
        noise_lowpass_order=2,
    )
    sampler = audit.StockSampler(config)
    rng = np.random.Generator(np.random.PCG64(51))
    before = copy.deepcopy(rng.bit_generator.state)
    incoming = np.zeros((4, 6))
    actions, after = sampler.candidates_from(incoming, before)
    second, second_after = sampler.candidates_from(incoming, before)
    np.testing.assert_array_equal(actions, second)
    assert audit.typed_fingerprint(after) == audit.typed_fingerprint(second_after)
    assert audit.typed_fingerprint(rng.bit_generator.state) == audit.typed_fingerprint(before)
    assert not np.array_equal(actions[0], incoming)
    assert np.max(np.abs(actions)) <= 1


def test_every_bank_update_can_be_audited_without_physics_or_actor():
    sampler = audit.StockSampler(dict(K=3, H=4, noise_sigma=0.3))
    before = np.random.Generator(np.random.PCG64(91)).bit_generator.state
    incoming = np.zeros((4, 6))
    candidates, after = sampler.candidates_from(incoming, before)
    c = np.array([3.0, 5.0, 7.0])
    d = np.array([2.0, 1.0, 0.5])
    scores = c + 0.12 * d
    weights, _ = audit.gibbs(scores, 3.0)
    updated = np.einsum("k,kha->ha", weights, candidates)
    kwargs = dict(
        incoming=incoming,
        updated=updated,
        cost=c,
        discrepancy=d,
        scores=scores,
        weights=weights,
        lengths=np.array([1, 2, 4]),
        temperature=3.0,
        alpha=0.12,
        rng_before=before,
        rng_after=after,
    )
    result = audit.compare_bank_statistics(audit.Audit(), sampler, **kwargs)
    assert not result["physics_or_discrepancy_recomputed"]
    kwargs["updated"] = np.roll(updated, 1, axis=0)
    with pytest.raises(audit.AuditFailure, match="carried-plan"):
        audit.compare_bank_statistics(audit.Audit(), sampler, **kwargs)


def file_binding(path):
    return dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def saved_corpus(tmp_path):
    cases = []
    source = {"a": [], "b": []}
    ids = []
    for slot in range(8):
        traces = []
        for arm, count in (("a", 2), ("b", 1)):
            raw = episode(count)
            raw["warmstarts"] = np.zeros((count + 1, 9), np.float64)
            raw["integrations"] = np.zeros((count + 1, 91), np.float64)
            path = tmp_path / f"{slot}_{arm}.npz"
            np.savez(path, **raw)
            traces.append(file_binding(path))
            source[arm].append(raw)
        cases.append(dict(slot=slot, controls=[2, 1], common=1, traces=traces))
        ids.append([slot, 0])
    files = {}
    for arm in ("a", "b"):
        raw = {
            key: np.concatenate([row[key][:1] for row in source[arm]])
            for key in ("states", "observations", "actions")
        }
        path = tmp_path / f"corpus_{arm}.npz"
        np.savez(path, **raw, row_ids=np.asarray(ids, dtype=np.int64))
        files[arm] = file_binding(path)
    manifest = dict(
        schema="local-coupling-corpora-1",
        rows=8,
        row_ids_fingerprint=audit.typed_fingerprint(np.asarray(ids, dtype=np.int64)),
        cases=cases,
        files=files,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_saved_corpus_adapter_requires_exact_physical_rows_not_only_fp32_fit_hash(tmp_path):
    path, manifest = saved_corpus(tmp_path)
    assert audit.audit_corpus(audit.Audit(), path)["rows"] == 8
    corpus_path = tmp_path / "corpus_b.npz"
    with np.load(corpus_path, allow_pickle=False) as data:
        raw = {key: data[key].copy() for key in data.files}
    raw["actions"][0, 0] += 1e-12
    np.savez(corpus_path, **raw)
    manifest["files"]["b"] = file_binding(corpus_path)
    path.write_text(json.dumps(manifest))
    with pytest.raises(audit.AuditFailure, match="physical fitting rows"):
        audit.audit_corpus(audit.Audit(), path)


def test_audit_module_and_cli_help_do_not_import_execution_or_torch():
    command = "from src.gps.mppi_gps.audits import local_coupling_learning as a; import sys; assert not any(n.startswith(('mujoco','torch','src.envs.','src.mppi.','src.gps.mppi_gps.policy','src.gps.mppi_gps.local_coupling_controller','src.gps.mppi_gps.local_coupling_loop')) for n in sys.modules); a.main(['--help'])"
    subprocess.run([sys.executable, "-c", command], check=True, capture_output=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_native(path, initial, post, actions, horizon):
    import h5py

    states = np.concatenate((initial[None], post))
    terminal = 20 * (~audit.healthy(post))
    raw = dict(
        states=states,
        observations=audit.observations(states),
        actions=actions,
        costs=running_cost(post, actions) + terminal,
        warmstarts=np.zeros((len(states), 9)),
        integrations=np.zeros((len(states), 91)),
        native_fall_cost=terminal,
    )
    with h5py.File(path, "w") as target:
        for name, value in raw.items():
            target[name] = value
    measured = audit.outcome(audit.Audit(), raw, horizon)
    return raw, measured


def write_ledger(directory, charges):
    charges = {key: int(value) for key, value in charges.items()}
    counts = {
        prefix + key: value
        for prefix in ("requested_", "completed_")
        for key, value in charges.items()
    }
    receipt = dict(status="COMPLETED", counts=counts, wall_seconds=0.1, cpu_seconds=0.1)
    rows = [
        dict(id=0, event="requested", operation="synthetic", charges=charges),
        dict(id=1, event="completed", request_id=0, operation="synthetic", charges=charges),
        dict(id=2, event="closed", receipt=receipt),
    ]
    (directory / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(directory / "receipt.json", receipt)
    return counts


def saved_acquisition(directory, invalid_first=False):
    """Synthetic saved arrays and mocked head only; no native/model execution."""
    import h5py

    directory.mkdir()
    payload = incoming()
    checkpoint = directory / "policy.pt"
    checkpoint.write_bytes(b"synthetic checkpoint decoder fixture")
    actor_binding = file_binding(checkpoint)
    context = dict(
        state=physical(1)[0],
        warmstart=np.zeros(9),
        integration=np.zeros(91),
        spec=np.asarray(16383),
        dimensions=np.asarray([9, 9, 6]),
    )
    restarts = directory / "restarts.npz"
    np.savez(restarts, **{"r0_" + k: v for k, v in context.items()})
    planner = dict(K=256, H=48, noise_sigma=0.3, lam=3.0, initial_iterations=2)
    sampler = audit.StockSampler(planner)
    rng = np.random.Generator(np.random.PCG64(91)).bit_generator.state
    before = np.zeros((48, 6))
    banks = []
    with h5py.File(directory / "banks.h5", "w") as stats:
        for inner in (0, 1):
            actions, after = sampler.candidates_from(before, rng)
            post = np.full((256, 48, 19), np.nan)
            post[:, 0] = physical(1)[0]
            post[:, 0, 2] = 0.5
            if invalid_first and inner == 0:
                post[0, 0, 10] = np.nan
            c, mask, valid = audit.batch_cost(post, actions)
            predictions = np.zeros((int(valid.sum()), 6), np.float64)
            d = 0.5 * np.sum(actions[:, 0] ** 2, axis=1)
            d[~valid] = np.inf
            j = c + 0.12 * d
            w, ess = audit.gibbs(j, 3.0)
            updated = np.clip(before + np.einsum("k,kha->ha", w, actions - before[None]), -1, 1)
            values = dict(
                C=c,
                D=d,
                J=j,
                weights=w,
                lengths=mask.sum(1),
                full_survival=np.zeros(256, dtype=bool),
                U_before=before,
                U_after=updated,
                initial_state=context["state"],
                initial_warmstart=context["warmstart"],
                initial_integration=context["integration"],
            )
            full = directory / f"full_t000_p{inner}.npz"
            np.savez(
                full, actions=actions, posts=post, predictions=predictions, mask=mask, **values
            )
            group = stats.create_group(f"{inner:06d}")
            for name, value in values.items():
                group[name] = value
            group.attrs.update(
                control=0,
                inner=inner,
                alpha=0.12,
                lambda_value=3.0,
                actor_hash=payload["policy_hash"],
                rng_before=json.dumps(rng),
                rng_after=json.dumps(after),
                ESS=ess,
                survival_fraction=0.0,
                fallen_weight=float(w.sum()),
                full_bank=full.name,
            )
            banks.append((full, actions, post, c, d))
            before = updated
            rng = after
    post = physical(1)
    post[0, 2] = 0.5
    _, outcome = synthetic_native(directory / "trace.h5", context["state"], post, before[:1], 300)
    outcome["local_walking"] = False
    candidates = []
    full, actions, posts, c, d = banks[0]
    for candidate in (0, 255):
        if not np.isfinite(c[candidate]):
            candidates.append(
                dict(
                    source=file_binding(full),
                    candidate=candidate,
                    status="UNAVAILABLE_INVALID_PREFIX",
                )
            )
            continue
        trace = directory / f"candidate_{candidate}.h5"
        raw, measured = synthetic_native(
            trace, context["state"], posts[candidate, :1], actions[candidate, :1], 48
        )
        pred = directory / f"candidate_{candidate}.npz"
        np.savez(
            pred,
            observations=raw["observations"][:-1],
            predictions=np.zeros((1, 6)),
            expected_predictions=np.zeros((1, 6)),
            D=np.asarray(d[candidate]),
            D_tolerance=np.asarray(1e-10 + 2 * 6 * 1e-6),
            action_max_error=np.asarray(0.0),
        )
        candidates.append(
            dict(
                source=file_binding(full),
                candidate=candidate,
                trace=file_binding(trace),
                predictions=file_binding(pred),
                outcome=measured,
                D=float(d[candidate]),
                D_tolerance=1e-10 + 2 * 6 * 1e-6,
                action_max_error=0.0,
            )
        )
    candidate_count = sum("trace" in row for row in candidates)
    charges = dict(
        model_controls=2 * 256 * 48,
        native_controls=1 + candidate_count,
        neural_rows=sum(np.isfinite(bank[3]).sum() for bank in banks) + candidate_count,
        physics_substeps=4 * (2 * 256 * 48 + 1 + candidate_count),
    )
    counts = write_ledger(directory, charges)
    job = dict(
        round=1,
        stream=-1,
        arm="weak",
        slot=0,
        alpha=0.12,
        seed=91,
        checkpoint=actor_binding,
        restarts=file_binding(restarts),
        planner=planner,
        replay_episode=False,
    )
    result = dict(
        schema="local-coupling-episode-1",
        status="COMPLETED",
        **{k: job[k] for k in ("round", "stream", "arm", "slot", "alpha", "seed", "checkpoint")},
        actor_hash=payload["policy_hash"],
        outcome=outcome,
        initial_context_fingerprint=audit.typed_fingerprint(context),
        banks=2,
        native_candidate_checks=candidates,
        trace=file_binding(directory / "trace.h5"),
        bank_stats=file_binding(directory / "banks.h5"),
        work=counts,
    )
    write_json(directory / "job.json", job)
    write_json(directory / "manifest.json", result)
    return payload, dict(controller=dict(planner=planner))


@pytest.mark.parametrize("invalid_first", [False, True])
def test_saved_acquisition_adapter_checks_two_first_passes_and_unused_nan_suffix(
    tmp_path, invalid_first
):
    payload, config = saved_acquisition(tmp_path / "acquisition", invalid_first=invalid_first)

    class FakeHead(audit.LearningAudit):
        def checkpoint(self, path):
            self.bind(path)
            return payload

        def predict(self, payload, inputs):
            return np.zeros((*np.shape(inputs)[:-1], 6), np.float32)

    result = audit.audit_acquisition(FakeHead(), tmp_path / "acquisition", {}, config)
    assert result["banks"] == 2 and result["full_banks"] == 2
    assert result["outcome"]["fell"] and result["outcome"]["controls"] == 1


def test_coordinator_accounting_imports_engineering_and_child_once(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    child_counts = write_ledger(child, dict(neural_rows=17))
    prepared = dict(engineering_work=dict(requested_optimizer_steps=8, completed_optimizer_steps=8))
    total = {**prepared["engineering_work"], **child_counts}
    spec = file_binding(child / "receipt.json")
    result = dict(accounting=dict(child_receipts=[spec], counts=total))
    (tmp_path / "study_ledger.jsonl").write_text(
        json.dumps(dict(event="closed_child_included", receipt=spec, counts=child_counts)) + "\n"
    )
    out = tmp_path / "out"
    out.mkdir()
    measured = audit._accounting(audit.Audit(), tmp_path, result, prepared, out, "final")
    assert measured["clean"] and measured["counts"] == total
    result["accounting"]["child_receipts"].append(spec)
    with pytest.raises(audit.AuditFailure, match="imported once"):
        audit._accounting(audit.Audit(), tmp_path, result, prepared, out, "final")


def test_public_audit_missing_evidence_closes_failure_without_forward_or_fabricated_zero(tmp_path):
    run = tmp_path / "missing"
    run.mkdir()
    receipt = audit.audit_run(run, tmp_path / "audit")
    assert not receipt["passed"]
    assert receipt["completed_neural_rows"] == receipt["requested_neural_rows"] == 0
    findings = json.loads((tmp_path / "audit/findings.json").read_text())
    assert "Missing evidence" in findings["error"]
    assert "original_status" not in findings


def test_native_discrepancy_uses_propagated_bounded_action_tolerance_without_forward():
    actions = -np.ones((48, 6))
    bank = np.ones_like(actions)
    native = bank - 9e-7
    tolerance = 1e-10 + 2 * actions.size * 1e-6
    error = float(np.max(np.abs(native - bank)))
    kwargs = dict(
        actions=actions,
        native_predictions=native,
        bank_predictions=bank,
        bank_discrepancy=float(0.5 * np.sum((actions - bank) ** 2)),
        saved_action_error=error,
        saved_D_tolerance=tolerance,
    )
    result = audit.compare_native_discrepancy(audit.Audit(), **kwargs)
    assert result["D_absolute_error"] > 1e-6
    assert result["D_absolute_error"] <= result["D_tolerance"] == tolerance
    kwargs["native_predictions"] = bank - 2e-6
    kwargs["saved_action_error"] = float(np.max(np.abs(kwargs["native_predictions"] - bank)))
    with pytest.raises(audit.AuditFailure, match="per-action prediction tolerance"):
        audit.compare_native_discrepancy(audit.Audit(), **kwargs)


def test_horizon_length_does_not_imply_survival_when_final_control_falls():
    post = np.broadcast_to(physical(48)[None], (2, 48, 19)).copy()
    post[1, -1, 2] = 0.5
    _, mask, _ = audit.batch_cost(post, np.zeros((2, 48, 6)))
    np.testing.assert_array_equal(mask.sum(axis=1), [48, 48])
    np.testing.assert_array_equal(audit.healthy(post).all(axis=1), [True, False])
