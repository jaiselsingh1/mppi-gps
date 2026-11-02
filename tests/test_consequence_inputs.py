"""Saved-source joins, bounded sparse reads and fail-closed C1 preparation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import consequence_inputs as inputs
from src.gps.mppi_gps.proposal_evidence import BinaryEvidenceMeter


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    return dict(bytes=path.stat().st_size, sha256=digest(path))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def make_source(root: Path, *, terminal_mismatch=False, padding=80):
    """Real existing-format groups/cache, with synthetic 12×1,000-row sources."""
    run = root / "old_run"
    run.mkdir(parents=True)
    for name in ("inputs", "checkpoints"):
        (run / name).mkdir()
    refs, episodes, parents = [], [], {}
    with BinaryEvidenceMeter(
        run / "evidence", max_controls=0, max_neural_rows=0, max_bytes=128 * inputs.MIB
    ) as meter:

        def save(tag, **arrays):
            ref = meter.save_arrays(tag, **arrays)
            refs.append(ref)
            return ref

        for _ in range(padding):
            save("padding", empty=np.empty((0, 6), dtype=np.float32))
        for stream in range(3):
            ckpt = run / f"checkpoints/s{stream}_parent.pt"
            ckpt.write_bytes(b"opaque checkpoint, never torch-decoded" + bytes([stream]))
            rng = save(f"s{stream}_global_rng", state=np.arange(32, dtype=np.uint8))
            parents[str(stream)] = {
                "checkpoint_record": dict(
                    file=str(ckpt.relative_to(run)),
                    **binding(ckpt),
                    actor_hash=str(stream + 1) * 64,
                    optimizer_sha256=str(stream + 4) * 64,
                    global_rng=dict(reference=rng, key="state"),
                )
            }
            for case in range(4):
                tag = f"teacher_s{stream}_independent_{case:02d}"
                observations = np.zeros((1000, 17), dtype="<f8")
                actions = np.full((1000, 6), 0.125, dtype="<f8")
                states = np.zeros((1001, 19), dtype="<f8")
                states[:, 0] = np.arange(1001) * 0.008
                warm = np.zeros((1001, 9), dtype="<f8")
                integration = np.zeros((1001, 91), dtype="<f8")
                integration[:, 0] = states[:, 0]
                costs = np.full(1000, 0.5, dtype="<f8")
                velocities = np.full(1000, 1.5, dtype="<f8")
                native_info = dict(x_velocity=1.5, fall_cost=0.0)
                query = f"q{stream * 4 + case:09d}"
                save(
                    query + "_inputs",
                    action=actions[-1],
                    state=states[-2],
                    warmstart=warm[-2],
                    integration=integration[-2],
                    spec=np.array(8191, dtype=np.int64),
                    dimensions=np.array([9, 9, 6], dtype=np.int64),
                )
                last = states[-1].copy()
                if terminal_mismatch and stream == case == 0:
                    last[1] += 1.0
                save(
                    query + "_result",
                    observation=np.full(17, 0.375, dtype="<f8"),
                    cost=np.array(0.5, dtype="<f8"),
                    done=np.array(False),
                    state=last,
                    warmstart=warm[-1],
                    integration=integration[-1],
                    info=native_info,
                )
                raw = save(
                    "episode_" + tag,
                    observations=observations,
                    actions=actions,
                    states=states,
                    warmstarts=warm,
                    integrations=integration,
                    costs=costs,
                    velocities=velocities,
                    infos={"rows": [dict(controller={}, native=native_info) for _ in range(1000)]},
                )
                prediction = np.zeros((1000, 6), dtype="<f4")
                disagreement = np.sum((actions - prediction) ** 2, axis=1)
                predicted = save(
                    tag + "_agreement",
                    policy_actions=prediction,
                    squared_disagreement=disagreement,
                )
                event = dict(
                    id=f"episode-{stream}-{case}",
                    tag=tag,
                    seed=2700000000 + 100 * stream + case,
                    steps=1000,
                    complete=True,
                    survived=True,
                    success=True,
                    raw=raw,
                )
                agreement = dict(
                    id=f"agreement-{stream}-{case}",
                    episode_raw=raw,
                    actor_hash=parents[str(stream)]["checkpoint_record"]["actor_hash"],
                    rows=1000,
                    predictions=dict(reference=predicted, key="policy_actions"),
                    squared_disagreement=dict(reference=predicted, key="squared_disagreement"),
                )
                episodes.append(
                    dict(
                        stream=stream,
                        case=case,
                        tag=tag,
                        source_reset_seed=event["seed"],
                        exact_episode_complete_event=event,
                        exact_teacher_agreement_complete_event=agreement,
                    )
                )
    cache = root / "closed-index.sqlite"
    with sqlite3.connect(cache) as database:
        database.execute("CREATE TABLE groups (name TEXT PRIMARY KEY,digest TEXT NOT NULL)")
        database.executemany(
            "INSERT INTO groups VALUES (?,?)",
            [(ref["group"], hashlib.sha256(canonical(ref)).hexdigest()) for ref in refs],
        )
    metadata = {}
    for name in ("config.json", "manifest.json", "input_manifest.json", "worker_result.json"):
        file = run / name
        file.write_text(json.dumps(dict(source=name)))
        metadata["run/" + name] = binding(file)
    config_inputs = {}
    for name, basename in [
        ("train", "train.npz"),
        ("validation", "validation.npz"),
        ("normalization", "normalization.npz"),
        ("task", "task.json"),
    ]:
        file = run / "inputs" / basename
        file.write_bytes(b"exact opaque archived input: " + basename.encode())
        config_inputs[name] = dict(file="inputs/" + basename, **binding(file))
    sources = dict(
        source_eligibility=dict(
            completed_R1_scientific_audit_passed=True,
            all_12_independent_sources_full_walking=True,
        ),
        external_evidence_files={
            "evidence/" + name: binding(run / "evidence" / name)
            for name in ("arrays.bin", "arrays.index.jsonl")
        },
        metadata_files=metadata,
        incoming_parents=parents,
        source_episodes_in_fixed_training_order=episodes,
        normalization_and_historical_inputs=dict(config_inputs=config_inputs),
    )
    source_file = root / "sources.json"
    source_file.write_text(json.dumps(sources, indent=2))
    return dict(
        run=run, source_file=source_file, cache=cache, sources=sources, cache_binding=binding(cache)
    )


@pytest.fixture(scope="module")
def frozen_source(tmp_path_factory):
    return make_source(tmp_path_factory.mktemp("synthetic_c1_source"))


@pytest.fixture(autouse=True)
def isolate_process_rss(monkeypatch):
    # Full-suite Torch imports elsewhere must not define this isolated phase's RSS.
    monkeypatch.setattr(inputs, "_peak_rss_bytes", lambda: 64 * inputs.MIB)


def prepare(source, out, **overrides):
    kwargs = dict(
        source_run=source["run"],
        source_bindings=source["source_file"],
        source_bindings_sha256=digest(source["source_file"]),
        audit_index=source["cache"],
        audit_index_binding=source["cache_binding"],
        out=out,
    )
    return inputs.prepare_inputs(**(kwargs | overrides))


def test_complete_preparation_uses_actual_terminal_return_and_preserves_old_bytes(
    frozen_source, tmp_path
):
    paths = [frozen_source["cache"], *frozen_source["run"].rglob("*")]
    before = {path: binding(path) for path in paths if path.is_file()}
    out = tmp_path / "prepared"
    catalog = prepare(frozen_source, out)
    assert len(catalog["episodes"]) == 12
    assert len(catalog["streams"]) == len(catalog["parents"]) == 3
    assert catalog["integration_spec"] == 8191
    assert catalog["dimensions"] == [9, 9, 6]
    episode = catalog["episodes"]["teacher_s0_independent_00"]
    with np.load(out / episode["file"], allow_pickle=False) as values:
        assert values["observations"].shape == (1001, 17)
        assert np.all(values["observations"][:-1] == 0)
        assert np.all(values["observations"][-1] == 0.375)
        assert values["policy_actions"].dtype == np.float32
        assert values["integrations"].shape == (1001, 91)
        assert np.all(values["fall_costs"] == 0)
    with np.load(out / catalog["streams"]["0"]["file"], allow_pickle=False) as values:
        assert values["observations"].shape == (4000, 17)
        assert values["observations"].dtype == values["actions"].dtype == np.float32
        np.testing.assert_array_equal(values["episode_index"], np.repeat(np.arange(4), 1000))
        np.testing.assert_array_equal(values["control_index"], np.tile(np.arange(1000), 4))
    assert (out / catalog["parents"]["0"]["file"]).read_bytes() == (
        frozen_source["run"] / "checkpoints/s0_parent.pt"
    ).read_bytes()
    assert (
        np.load(out / catalog["parents"]["0"]["global_torch_rng"]["file"], allow_pickle=False).dtype
        == np.uint8
    )
    receipt = json.loads((out / "preparation_receipt.json").read_text())
    assert receipt["work"]["source_read_bytes"] < 512 * inputs.MIB
    assert receipt["work"]["new_native_controls"] == receipt["work"]["new_neural_rows"] == 0
    assert receipt["native_replay_gate_passed"] is False
    assert all(binding(path) == value for path, value in before.items())


def test_sparse_index_lookup_authenticates_without_full_scan(tmp_path):
    source = make_source(tmp_path / "source", padding=2500)
    budget = inputs._Budget(512 * inputs.MIB, 256 * inputs.MIB, 600, 512 * inputs.MIB)
    archive = inputs._SelectedArchive(
        source["run"], source["sources"], source["cache"], source["cache_binding"], budget
    )
    try:
        target = archive.find_group(2300)
        assert target["group"].startswith("arrays/a000002300_")
        assert budget.read_by_kind["selected_index"] < archive.size // 2
        assert budget.read_by_kind.get("selected_array_C_bytes", 0) == 0
    finally:
        archive.close()


def test_changed_cache_rejected_before_selected_raw_reads(frozen_source, tmp_path):
    wrong = dict(frozen_source["cache_binding"], sha256="0" * 64)
    with pytest.raises(inputs.ConsequenceInputError, match="SHA256"):
        prepare(frozen_source, tmp_path / "prepared", audit_index_binding=wrong)
    receipt = json.loads((tmp_path / "prepared/preparation_failure.json").read_text())
    assert "selected_array_C_bytes" not in receipt["work"]["source_read_bytes_by_kind"]


def test_reference_not_in_closed_cache_fails_closed(frozen_source, tmp_path):
    altered = json.loads(frozen_source["source_file"].read_text())
    altered["source_episodes_in_fixed_training_order"][0]["exact_episode_complete_event"]["raw"][
        "arrays"
    ]["observations"]["sha256"] = "0" * 64
    changed = tmp_path / "changed_sources.json"
    changed.write_text(json.dumps(altered))
    with pytest.raises(inputs.ConsequenceInputError, match="join differs"):
        prepare(
            frozen_source,
            tmp_path / "prepared",
            source_bindings=changed,
            source_bindings_sha256=digest(changed),
        )


def test_canonical_group_digest_is_required_even_with_matching_source_label(frozen_source):
    budget = inputs._Budget(512 * inputs.MIB, 256 * inputs.MIB, 600, 512 * inputs.MIB)
    archive = inputs._SelectedArchive(
        frozen_source["run"],
        frozen_source["sources"],
        frozen_source["cache"],
        frozen_source["cache_binding"],
        budget,
    )
    try:
        reference = json.loads(
            json.dumps(
                frozen_source["sources"]["source_episodes_in_fixed_training_order"][0][
                    "exact_episode_complete_event"
                ]["raw"]
            )
        )
        reference["arrays"]["actions"]["offset"] += 8
        with pytest.raises(inputs.ConsequenceInputError, match="closed audit cache"):
            archive.verify(reference)
        assert budget.read_by_kind.get("selected_array_C_bytes", 0) == 0
    finally:
        archive.close()


def test_sparse_search_handles_first_last_and_both_index_record_kinds(frozen_source):
    budget = inputs._Budget(512 * inputs.MIB, 256 * inputs.MIB, 600, 512 * inputs.MIB)
    archive = inputs._SelectedArchive(
        frozen_source["run"],
        frozen_source["sources"],
        frozen_source["cache"],
        frozen_source["cache_binding"],
        budget,
    )
    try:
        rows = [json.loads(line) for line in archive.index.read_text().splitlines()]
        expected = [{k: v for k, v in row.items() if k != "event"} for row in rows[::2]]
        for ordinal, reference in enumerate(expected):
            assert archive.find_group(ordinal) == reference
        with pytest.raises(inputs.ConsequenceInputError, match="not found"):
            archive.find_group(len(expected) + 10)
    finally:
        archive.close()


def test_last_query_context_mismatch_is_not_replaced(tmp_path):
    source = make_source(tmp_path / "source", terminal_mismatch=True)
    with pytest.raises(inputs.ConsequenceInputError, match="Terminal live context differs"):
        prepare(source, tmp_path / "prepared")
    assert not (tmp_path / "prepared/catalog.json").exists()
    assert (tmp_path / "prepared/preparation_failure.json").exists()


def test_missing_or_reordered_source_stops_before_array_decode(frozen_source, tmp_path):
    changed = tmp_path / "changed.json"
    source = json.loads(frozen_source["source_file"].read_text())
    source["source_episodes_in_fixed_training_order"].reverse()
    changed.write_text(json.dumps(source))
    with pytest.raises(inputs.ConsequenceInputError, match="roster/order"):
        prepare(
            frozen_source,
            tmp_path / "prepared",
            source_bindings=changed,
            source_bindings_sha256=digest(changed),
        )
    work = json.loads((tmp_path / "prepared/preparation_failure.json").read_text())["work"]
    assert work["source_read_bytes_by_kind"].get("selected_array_C_bytes", 0) == 0


def test_read_ceiling_rejects_before_large_cache_read(frozen_source, tmp_path):
    with pytest.raises(inputs.ConsequenceInputError, match="read budget"):
        prepare(frozen_source, tmp_path / "prepared", max_read_bytes=100)
    assert not (tmp_path / "prepared/catalog.json").exists()


def test_declared_resource_ceilings_cannot_be_raised(frozen_source, tmp_path):
    with pytest.raises(inputs.ConsequenceInputError, match="512 MiB"):
        prepare(frozen_source, tmp_path / "prepared", max_read_bytes=513 * inputs.MIB)


def test_output_and_rss_limits_stop_without_complete_catalog(frozen_source, tmp_path, monkeypatch):
    with pytest.raises(inputs.ConsequenceInputError, match="output budget"):
        prepare(frozen_source, tmp_path / "small_output", max_output_bytes=65537)
    assert not (tmp_path / "small_output/catalog.json").exists()
    monkeypatch.setattr(inputs, "_peak_rss_bytes", lambda: 513 * inputs.MIB)
    with pytest.raises(inputs.ConsequenceInputError, match="RSS"):
        prepare(frozen_source, tmp_path / "rss")
    assert not (tmp_path / "rss/catalog.json").exists()


def test_existing_output_is_never_modified(frozen_source, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    (out / "keep").write_bytes(b"unchanged")
    with pytest.raises(inputs.ConsequenceInputError, match="already exists"):
        prepare(frozen_source, out)
    assert list(out.iterdir()) == [out / "keep"]
    assert (out / "keep").read_bytes() == b"unchanged"


def test_input_module_has_no_policy_or_simulator_imports():
    code = "import sys; import src.gps.mppi_gps.consequence_inputs; assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules; assert 'src.envs.walker2d' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
