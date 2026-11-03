"""Independent J2 numerical gates treat a learner seed as the replication unit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits import jitter_diagnosis as audit
from src.gps.mppi_gps.audits.teacher_proposal import AuditError


def records() -> tuple[list[dict], list[int], list[str]]:
    seeds = list(audit.FRESH_SEEDS)
    conditions = [f"condition_{index}" for index in range(13)]
    rows = [
        dict(
            seed=seed,
            condition=condition,
            walking=2 * index,
            diagnostic_error=float(13 - index),
            offline_error=float(index + 1),
        )
        for seed in seeds
        for index, condition in enumerate(conditions)
    ]
    return rows, seeds, conditions


def test_tied_ranks_and_missing_correlations() -> None:
    assert audit.rank_correlation([1, 2, 3, 4], [0, 0, 1, 2]) == pytest.approx(0.9486832980505138)
    assert audit.rank_correlation([4, 3, 2, 1], [2, 1, 0, 0]) == pytest.approx(0.9486832980505138)
    assert audit.rank_correlation([1, 1], [0, 1]) is None
    with pytest.raises(AuditError, match="Nonfinite"):
        audit.rank_correlation([float("nan"), 0], [0, 1])


def test_exact_sign_flip_includes_ties_and_does_not_replicate_one_seed() -> None:
    assert audit.sign_flip_oracle([1] * 6)["p_value"] == 1 / 64
    assert audit.sign_flip_oracle([0] * 6)["p_value"] == 1
    one_seed = audit.sign_flip_oracle([0, 0, 0, 0, 0, 1])
    assert one_seed["patterns"] == 64
    assert one_seed["p_value"] == 0.5
    with pytest.raises(AuditError, match="six whole-seed"):
        audit.sign_flip_oracle([1] * 78)


def test_primary_confirmation_and_all_failure_ineligibility() -> None:
    rows, seeds, conditions = records()
    result = audit.primary_oracle(rows, seeds, conditions)
    assert result["confirmed"] is True
    assert result["mean_diagnostic_spearman"] == pytest.approx(1)
    assert result["exact_sign_flip"]["patterns"] == 64
    assert result["streams_at_least_offline"] == 6
    assert result["automatic_followup_authorized"] is False
    for row in rows:
        row["walking"] = 0
    result = audit.primary_oracle(rows, seeds, conditions)
    assert result["eligible"] is False
    assert result["confirmed"] is False
    assert "exact_sign_flip" not in result
    assert result["secondary_pooled_auc"]["diagnostic"] is None


def test_missing_or_duplicate_case_cannot_change_the_comparison() -> None:
    rows, seeds, conditions = records()
    with pytest.raises(AuditError, match="case roster"):
        audit.primary_oracle(rows[:-1], seeds, conditions)
    with pytest.raises(AuditError, match="case roster"):
        audit.primary_oracle(rows + [rows[0]], seeds, conditions)
    rows[0]["walking"] = 33
    with pytest.raises(AuditError, match="out of 32"):
        audit.primary_oracle(rows, seeds, conditions)


def test_secondary_auc_defines_any_walking_and_handles_ties() -> None:
    assert audit.any_walking_auc([0, 1, 1, 2], [1, 32, 0, 0]) == 0.875
    assert audit.any_walking_auc([0, 1], [0, 0]) is None


def test_native_outcomes_preserve_fall_prefix_and_tail_separately() -> None:
    states = np.zeros((3, 19), dtype=np.float64)
    result = audit.episode_oracle(
        states,
        np.zeros((2, 6)),
        np.array([1.0, 21.0]),
        np.array([0.0, 20.0]),
        np.array([1.5, 1.5]),
        np.array([0, 1], dtype=np.uint8),
        1000,
    )
    assert result["walking"] is False
    assert result["native_terminal"] == 20
    assert result["absorbing_cost"] == 5 * 998
    assert result["complete_cost"] == 22 + 5 * 998
    with pytest.raises(AuditError, match="post-fall"):
        audit.episode_oracle(
            states,
            np.zeros((2, 6)),
            np.array([21.0, 1.0]),
            np.array([20.0, 0.0]),
            np.array([1.5, 1.5]),
            np.array([1, 0], dtype=np.uint8),
            1000,
        )


def test_import_does_not_load_simulator_or_torch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.gps.mppi_gps.audits import jitter_diagnosis; "
            "assert not {'mujoco','torch','warp'} & set(sys.modules)",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def trace_file(root: Path) -> Path:
    states = np.zeros((3, 19), dtype=np.float64)
    states[:, 0] = [0, 0.008, 0.016]
    states[:, 2] = 1.25
    states[:, 10] = 1.5
    warmstarts = np.zeros((3, 9), dtype=np.float64)
    integrations = np.zeros((3, 91), dtype=np.float64)
    integrations[:, :19] = states
    observations = np.hstack((states[:, 2:10], states[:, 10:19]))
    path = root / "traces.npz"
    np.savez_compressed(
        path,
        states=states,
        observations=observations,
        warmstarts=warmstarts,
        integrations=integrations,
        actions=np.zeros((2, 6)),
        reference_actions=np.full((2, 6), 0.25),
        has_reference=np.ones(2, dtype=bool),
        costs=np.zeros(2),
        fall_costs=np.zeros(2),
        velocities=np.full(2, 1.5),
        done=np.zeros(2, dtype=bool),
        episode_step_offsets=np.array([0, 2], dtype=np.int64),
        episode_state_offsets=np.array([0, 3], dtype=np.int64),
        episode_seeds=np.array([123], dtype=np.int64),
        episode_horizons=np.array([2], dtype=np.int64),
        episode_kinds=np.array(["probe"]),
    )
    return path


class ActorFixture:
    def __init__(self):
        self.verified_rows = 0

    def verify(self, binding, observations, expected, *, batch_one):
        assert batch_one
        assert np.array_equal(
            expected, np.full(expected.shape, 0.25 if binding == "reference" else 0)
        )
        self.verified_rows += len(observations)


def test_saved_trace_join_costs_and_probe_errors_without_physics(tmp_path: Path) -> None:
    import json

    task = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/walker2d_task.json").read_text()
    )
    trace_file(tmp_path)
    reader = audit.SavedReader(tmp_path, {})
    actors = ActorFixture()
    result = audit.verify_traces(
        reader, actors, "traces.npz", task, [("probe", 123, 2)], "student", "reference", {}
    )
    assert result["diagnostic_error"] == 0.0625
    assert result["diagnostic_short_error"] == 0.0625
    assert result["episodes"][0]["walking"] is False
    assert result["episodes"][0]["right_censored"] is True
    assert actors.verified_rows == 4
    assert result["snapshot_reads"] == 3
    with np.load(tmp_path / "traces.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["observations"][0, 0] += 1
    np.savez_compressed(tmp_path / "corrupt.npz", **arrays)
    with pytest.raises(AuditError, match="observation interface"):
        audit.verify_traces(
            reader, actors, "corrupt.npz", task, [("probe", 123, 2)], "student", "reference", {}
        )


def test_saved_reader_budget_hash_and_path_checks(tmp_path: Path) -> None:
    (tmp_path / "tiny.json").write_text("{}\n")
    reader = audit.SavedReader(tmp_path, {"audit_read_bytes": 2})
    with pytest.raises(AuditError, match="ceiling"):
        reader.json("tiny.json")
    assert reader.read_bytes == 0
    reader = audit.SavedReader(tmp_path, {})
    with pytest.raises(AuditError, match="byte binding"):
        reader.read(dict(file="tiny.json", bytes=3, sha256="incorrect"))
    (tmp_path / "alias.json").symlink_to(tmp_path / "tiny.json")
    with pytest.raises(AuditError, match="symlinked"):
        reader.read("alias.json")


def test_primary_duration_controls_remain_secondary() -> None:
    rows, seeds, conditions = records()
    for row in rows:
        row.update(mean_probe_controls=row["walking"] + 1.0, probe_falls=8 - row["walking"] / 4)
    result = audit.primary_oracle(rows, seeds, conditions)
    controls = result["per_seed"][0]["secondary_duration_controls"]
    assert controls["mean_probe_controls_spearman"] == pytest.approx(1)
    assert controls["negative_probe_fall_count_spearman"] == pytest.approx(1)
    assert controls["primary_gate_uses_these_controls"] is False


def test_primitives_include_reference_replays_and_single_endpoint_probes() -> None:
    traces = [
        dict(
            controls_by_kind=dict(replay=25, reference=1000),
            episodes=[dict(kind="replay"), dict(kind="reference")],
        ),
        dict(
            controls_by_kind=dict(probe=100, evaluation=1000),
            episodes=[dict(kind="probe"), dict(kind="evaluation")],
        ),
    ]
    census = audit.primitive_oracle(traces, fits=1, parents=1)
    assert census["native_controls"] == 2125
    assert census["physics_substeps"] == 8500
    assert census["neural_rows"] == 2560000 + 9000 + 16000 + 2100 + 100
    assert census["network_calls"] == 10000 + 1 + 2 + 2100 + 100
    assert census["snapshot_reads"] == 2125 + 4 + 1
    assert census["external_forwards"] == 7


def test_audit_cli_preserves_a_missing_evidence_failure(tmp_path: Path) -> None:
    import json

    run, output = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.gps.mppi_gps.audits.jitter_diagnosis",
            "--run",
            str(run),
            "--out",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 1
    saved = json.loads((output / "audit_result.json").read_text())
    assert saved["passed"] is False
    assert saved["new_physics_controls"] == saved["new_optimizer_steps"] == 0
    assert saved["new_neural_rows"] == 0
    assert "Missing" in saved["error"]["message"]


def test_synthetic_fit_package_binds_all_indices_and_first_loss(tmp_path: Path) -> None:
    """Construct metadata/tensors only: no optimizer, backward pass or fitting."""
    import hashlib
    import json

    import torch

    from src.gps.mppi_gps.policy import Policy

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seed = audit.FRESH_SEEDS[0]
        observations = np.zeros((9000, 17), dtype=np.float32)
        held = np.zeros((7000, 17), dtype=np.float32)
        labels = np.zeros((9000, 6), dtype=np.float32)
        corpus = dict(
            observations=observations,
            held_observations=held,
            held_clean=np.zeros((7000, 6)),
            labels__clean=labels,
        )
        parent = Policy(seed, "cpu", dict(mean=np.zeros(17), std=np.ones(17))).eval()
        global_torch = torch.random.get_rng_state().clone()
        global_numpy = dict(synthetic_fixture=True)
        metadata = dict(
            global_torch_rng_state=global_torch.tolist(), global_numpy_state=global_numpy
        )
        parent.checkpoint(tmp_path / f"parents/{seed}.pt", metadata=metadata)
        rng = np.random.Generator(np.random.PCG64(seed))
        initial_rng = json.loads(json.dumps(rng.bit_generator.state))
        tape = rng.integers(0, 9000, size=(10000, 256))
        stream = dict(
            indices=tape,
            global_torch_rng=global_torch.numpy(),
            global_numpy_state_json=np.array(json.dumps(global_numpy)),
        )
        group = dict(
            lr=0.0003,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
            fused=False,
            params=list(range(6)),
        )
        initial = dict(
            state_dict=parent.state_dict(),
            adam=dict(state={}, param_groups=[group]),
            rng=initial_rng,
            private_rng_state=parent._generator.get_state(),
            global_torch_rng=global_torch,
            global_numpy_state=global_numpy,
        )
        directory = f"cases/{seed}_clean"
        path = tmp_path / directory
        path.mkdir(parents=True)
        torch.save(initial, path / "initial.pt")
        config = dict(
            batch_size=256,
            lr=0.0003,
            eta_theta=0.0,
            gradient_clip=0.5,
            prox_reduction="mean",
            anchor_refresh=0,
        )
        fit_data = dict(
            observations_sha256=audit.array_fingerprint(observations),
            actions_sha256=audit.array_fingerprint(labels),
            rows=9000,
        )
        names = [name for name, _ in parent.named_parameters()]
        adam = dict(
            state={
                index: dict(
                    step=torch.tensor(10000.0),
                    exp_avg=torch.zeros_like(value),
                    exp_avg_sq=torch.zeros_like(value),
                )
                for index, (_, value) in enumerate(parent.named_parameters())
            },
            param_groups=[group],
        )
        state = dict(
            schema="walker-adam-proximal-fit-v1",
            adam=adam,
            rng=rng.bit_generator.state,
            updates=10000,
            fit_updates=10000,
            policy_hash=parent.hash(),
            parent_policy_hash=parent.hash(),
            anchor_policy_hash=parent.hash(),
            anchor=parent.state_dict(),
            parameter_names=names,
            fit_config=config,
            fit_data=fit_data,
        )
        checkpoint = parent.checkpoint(path / "child.pt", state, metadata)
        predictions = parent.predict(observations)
        held_predictions = parent.predict(held)
        first = parent.predict(observations[tape[0]])
        first_loss = float(torch.mean(torch.as_tensor(first) ** 2))
        curve = [
            dict(
                update=index + 1,
                fit_update=index + 1,
                action_mse=first_loss,
                proximal_loss=0.0,
                total_loss=first_loss,
                unclipped_gradient_norm=0.0,
            )
            for index in range(10000)
        ]
        tape_hash = hashlib.sha256(tape.tobytes()).hexdigest()
        hashes = np.stack(
            [np.frombuffer(hashlib.sha256(row.tobytes()).digest(), dtype=np.uint8) for row in tape]
        )
        np.savez_compressed(
            path / "fit_arrays.npz",
            observed_tape_update_hashes=hashes,
            held_predictions=held_predictions,
            incoming_predictions=predictions,
            train_predictions=predictions,
        )
        fit = dict(
            schema="walker-adam-proximal-fit-v1",
            initial_file="initial.pt",
            tape_file=f"streams/{seed}.npz",
            initial_package_sha256=hashlib.sha256((path / "initial.pt").read_bytes()).hexdigest(),
            initial_global_rng_sha256=hashlib.sha256(global_torch.numpy().tobytes()).hexdigest(),
            checkpoint=checkpoint,
            tape_sha256=tape_hash,
            observed_tape_sha256=tape_hash,
            parameter_count=19590,
            proximal_divisor=19590,
            fit_config=config,
            fit_data=fit_data,
            completed_updates=10000,
            cumulative_updates=10000,
            fit_updates=10000,
            failed=False,
            resume_fit=False,
            optimizer_continued=False,
            anchor_refreshes=[],
            parent_policy_hash=parent.hash(),
            anchor_policy_hash=parent.hash(),
            child_policy_hash=parent.hash(),
            learning_curve=curve,
            held_clean_mse=float(np.mean((held_predictions - corpus["held_clean"]) ** 2)),
            work={
                key: dict(requested=amount, completed=amount, unknown_partial=0)
                for key, amount in dict(
                    neural_rows=2560000, backward_rows=2560000, optimizer_steps=10000
                ).items()
            },
        )
        (path / "fit.json").write_text(json.dumps(fit))
        reader = audit.SavedReader(tmp_path, {})
        actors = audit.FunctionalActors(reader, 100000)
        payload = actors.load(f"parents/{seed}.pt")
        result = audit.verify_fit(
            reader, actors, directory, seed, "clean", corpus, stream, payload, {}
        )
        assert result["curve_points"] == 10000
        assert result["gradients_repeated"] is False
        assert result["all_curve_losses_recomputed"] is False
        assert actors.completed == 25256
        hashes[0, 0] ^= 1
        np.savez_compressed(
            path / "fit_arrays.npz",
            observed_tape_update_hashes=hashes,
            held_predictions=held_predictions,
            incoming_predictions=predictions,
            train_predictions=predictions,
        )
        reader = audit.SavedReader(tmp_path, {})
        actors = audit.FunctionalActors(reader, 100000)
        payload = actors.load(f"parents/{seed}.pt")
        with pytest.raises(AuditError, match="actual observed minibatch digest"):
            audit.verify_fit(reader, actors, directory, seed, "clean", corpus, stream, payload, {})
    finally:
        torch.set_num_threads(previous_threads)


def test_saved_source_corpus_and_strict_manifest_pin(tmp_path: Path) -> None:
    import hashlib
    import json

    import h5py

    from src.gps.mppi_gps import jitter_diagnosis as worker

    h5_path = tmp_path / "source.h5"
    with h5py.File(h5_path, "w") as handle:
        arrays = handle.create_group("arrays")
        for index in range(16):
            group = arrays.create_group(f"episode_k1_primary_reference_{index:02d}")
            states = np.zeros((1001, 19))
            states[:, 0] = np.arange(1001) * 0.008
            states[:, 1] = np.arange(1001) * 0.012
            states[:, 2], states[:, 10] = 1.25, 1.5
            integration = np.zeros((1001, 91))
            integration[:, :19] = states
            fields = dict(
                states=states,
                observations=np.hstack((states[:-1, 2:10], states[:-1, 10:19])),
                actions=np.zeros((1000, 6), dtype=np.float32),
                integrations=integration,
                warmstarts=np.zeros((1001, 9)),
                costs=np.zeros(1000),
            )
            for name, value in fields.items():
                group.create_dataset(name, data=value, compression="gzip")
    mppi_path = tmp_path / "mppi.npz"
    actions = np.tile(np.linspace(-0.5, 0.5, 1000)[:, None], (10, 6))
    np.savez_compressed(mppi_path, episode_step_offsets=np.arange(0, 10001, 1000), actions=actions)
    reference_path = tmp_path / "reference.pt"
    reference_path.write_bytes(b"synthetic source-binding fixture, not a decoded actor")
    inputs = {}
    for name, path in dict(
        reference_h5=h5_path, mppi_traces=mppi_path, reference_checkpoint=reference_path
    ).items():
        inputs[name] = dict(
            path=str(path),
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/studies/jitter_diagnosis_j2_0.json"
        ).read_text()
    )
    config["inputs"] = inputs
    worker.prepare(config, tmp_path / "inputs")
    manifest_path = tmp_path / "inputs/manifest.json"
    config["data"] = dict(prepared_manifest=worker.file_binding(manifest_path))
    reader = audit.SavedReader(tmp_path, {})
    corpus = audit.verify_corpus(reader, config)
    assert corpus["observations"].shape == (9000, 17)
    assert corpus["held_observations"].shape == (7000, 17)
    assert len([key for key in corpus if key.startswith("labels__")]) == 13
    assert reader.decoded_array_bytes > 0
    config["data"]["prepared_manifest"]["sha256"] = "0" * 64
    with pytest.raises(AuditError, match="manifest pin"):
        audit.verify_corpus(audit.SavedReader(tmp_path, {}), config)


def test_reference_failure_is_a_complete_stopping_result_without_fits(
    tmp_path: Path, monkeypatch
) -> None:
    import hashlib
    import json

    run = tmp_path / "run"
    run.mkdir()
    repository = Path(__file__).resolve().parents[1]
    source_name = "src/gps/mppi_gps/audits/jitter_diagnosis.py"
    source_bytes = (repository / source_name).read_bytes()
    trace_binding = dict(bytes=1, sha256="synthetic-trace-contract")
    replay = dict(controls_by_kind=dict(replay=1575), episodes=[dict(kind="replay")] * 63)
    reference_episodes = [dict(kind="reference", walking=False) for _ in range(32)]
    reference = dict(controls_by_kind=dict(reference=32000), episodes=reference_episodes)
    census = {
        key: value
        for key, value in audit.primitive_oracle([replay, reference], fits=0, parents=0).items()
        if value
    }
    work = dict(requested=census, completed=census, unknown_partial={}, wall_seconds=0.1)
    config = dict(
        schema="walker-jitter-diagnosis-j2-0",
        launch_ready=True,
        automatic_followup=False,
        seeds=dict(
            training=list(audit.FRESH_SEEDS),
            evaluation=list(range(32)),
            probe=list(range(100, 108)),
        ),
        conditions=[dict(id="clean")]
        + [
            dict(id=kind + "_" + str(share).replace(".", "p"))
            for kind in ("mppi", "white", "slow")
            for share in (0.9, 0.75, 0.5, 0.25)
        ],
        audit=dict(wall_seconds=60, neural_rows=100000),
        task={},
        locations=dict(source_export=str(repository)),
        source_bindings={
            source_name: dict(
                bytes=len(source_bytes), sha256=hashlib.sha256(source_bytes).hexdigest()
            )
        },
        limits=census,
    )
    worker = dict(
        status="STOPPED_REFERENCE_QUALIFICATION",
        actual_fits=0,
        cases=[],
        work=work,
        prerequisites=dict(
            held_replays=dict(episodes=63, traces=trace_binding),
            reference=dict(panel_walking=[0, 0], episodes=reference_episodes, traces=trace_binding),
        ),
    )
    for name, value in dict(config=config, result=worker, work=work).items():
        (run / (name + ".json")).write_text(json.dumps(value))

    def corpus(reader, _config):
        reader.files["inputs/reference.pt"] = dict(bytes=1, sha256="synthetic")
        return {
            key: np.zeros(
                (
                    16,
                    1001,
                    91
                    if key == "sac_integrations"
                    else 19
                    if key == "sac_states"
                    else 9
                    if key == "sac_warmstarts"
                    else 6
                    if key == "sac_actions"
                    else 1,
                )
            )
            for key in (
                "sac_states",
                "sac_integrations",
                "sac_warmstarts",
                "sac_actions",
                "sac_costs",
            )
        }

    def traces(reader, _actors, filename, *args, **kwargs):
        reader.files[filename] = trace_binding
        return replay if filename.endswith("replay.npz") else reference

    monkeypatch.setattr(audit, "verify_corpus", corpus)
    monkeypatch.setattr(audit, "verify_traces", traces)
    result = audit.audit_run(run, tmp_path / "audit")
    assert result["passed"] is True
    assert result["primary_comparison_executed"] is False
    assert result["records"] == []
    assert result["new_neural_rows"] == result["new_optimizer_steps"] == 0


def test_initial_replay_gate_rejects_signed_zero_despite_numeric_equality(tmp_path: Path) -> None:
    import json

    positive, negative = np.array([0.0]), np.array([-0.0])
    assert np.array_equal(positive, negative)
    assert audit._bytes_equal(positive, negative) is False
    assert audit._bytes_equal(positive, positive.astype(np.float32)) is False
    task = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/walker2d_task.json").read_text()
    )
    trace_file(tmp_path)
    with np.load(tmp_path / "traces.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["episode_kinds"] = np.array(["replay"])
    arrays["reference_actions"][:] = np.nan
    arrays["has_reference"][:] = False
    np.savez_compressed(tmp_path / "replay.npz", **arrays)
    source = {
        key: arrays[key].copy() for key in ("states", "warmstarts", "integrations", "actions")
    }
    source["costs"] = arrays["costs"].copy()
    reader = audit.SavedReader(tmp_path, {})
    actors = ActorFixture()
    expected = [("replay", 123, 2)]
    audit.verify_traces(
        reader, actors, "replay.npz", task, expected, None, None, {}, replay_sources={123: source}
    )
    source["warmstarts"][0, 0] = -0.0
    source["integrations"][0, 19] = -0.0
    assert np.allclose(arrays["warmstarts"], source["warmstarts"], rtol=0, atol=1e-10)
    with pytest.raises(AuditError, match="initial replay context is not byte-exact"):
        audit.verify_traces(
            reader,
            actors,
            "replay.npz",
            task,
            expected,
            None,
            None,
            {},
            replay_sources={123: source},
        )
    assert actors.verified_rows == 0


def test_paired_initial_context_hashes_reject_signed_zero(tmp_path: Path) -> None:
    import json

    task = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/walker2d_task.json").read_text()
    )
    trace_file(tmp_path)
    reader = audit.SavedReader(tmp_path, {})
    actors = ActorFixture()
    contexts = {}
    audit.verify_traces(
        reader, actors, "traces.npz", task, [("probe", 123, 2)], "student", "reference", contexts
    )
    with np.load(tmp_path / "traces.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["warmstarts"][0, 0] = -0.0
    arrays["integrations"][0, 19] = -0.0
    np.savez_compressed(tmp_path / "different_initial.npz", **arrays)
    with pytest.raises(AuditError, match="paired reset initial simulator contexts differ"):
        audit.verify_traces(
            reader,
            actors,
            "different_initial.npz",
            task,
            [("probe", 123, 2)],
            "student",
            "reference",
            contexts,
        )
