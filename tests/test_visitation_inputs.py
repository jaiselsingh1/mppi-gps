"""S1.2 V saved-only source joins, context checks and preparation budgets."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import visitation_inputs as inputs
from src.gps.mppi_gps.proposal_evidence import BinaryEvidenceMeter


@pytest.fixture(autouse=True)
def isolate_preparation_process_rss(monkeypatch):
    # The API fixtures represent a small saved-only preparation process.
    # A shared pytest parent may retain unrelated Torch/Matplotlib peaks.
    monkeypatch.setattr(
        "src.gps.mppi_gps.consequence_inputs._peak_rss_bytes", lambda: 64 * inputs.MIB
    )


def binding(path: Path) -> dict:
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_json(path: Path, value) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return dict(path=str(path), **binding(path))


def source_fixture(
    root: Path, *, mismatch=False, missing_terminal=False, wrong_actor=False, physical=False
):
    """Ninety-six four-control prefixes, including their real synthetic falls."""
    run = root / "old"
    prepared = run / "prepared"
    (prepared / "parents").mkdir(parents=True)
    (prepared / "inputs").mkdir()
    parents, registry, files = {}, {}, []
    for stream in range(3):
        name = f"parents/s{stream}_parent.pt"
        path = prepared / name
        path.write_bytes(b"opaque complete checkpoint " + bytes([stream]))
        check = dict(file=name, **binding(path))
        files.append(check)
        rng = prepared / f"parents/s{stream}_global_torch_rng.npy"
        np.save(rng, np.arange(32, dtype="u1"), allow_pickle=False)
        sidecar = dict(file=str(rng.relative_to(prepared)), **binding(rng))
        files.append(sidecar)
        actor_hash = str(stream + 1) * 64
        parents[str(stream)] = dict(
            file="prepared/" + name,
            sha256=check["sha256"],
            actor_hash=actor_hash,
            stream=stream,
            arm="parent",
            continuation=dict(
                checkpoint_bytes=check["bytes"],
                incoming_updates=2000,
                incoming_fit_updates=2000,
                passed=True,
            ),
            global_cpu_sidecar=sidecar,
        )
        registry[f"s{stream}_parent"] = dict(
            file="prepared/" + name,
            sha256=check["sha256"],
            actor_hash=actor_hash,
            stream=stream,
            arm="parent",
        )
    norm = prepared / "inputs/normalization.npz"
    np.savez(norm, mean=np.zeros(17), std=np.ones(17))
    files.append(dict(file="inputs/normalization.npz", **binding(norm)))
    xml = root / "walker.xml"
    xml.write_bytes(b"frozen native XML bytes; never parsed in preparation")
    task = prepared / "inputs/task.json"
    task_values = dict(frame_skip=4)
    if physical:
        task_values.update(
            cost_style="target_velocity",
            target_velocity=1.5,
            vel_cost_weight=2.0,
            vertical_velocity_cost_weight=1.0,
            angle_cost_weight=0.0,
            unhealthy_cost_weight=0.0,
            ctrl_cost_weight=0.0,
            terminal_unhealthy_weight=20.0,
            apply_terminal_cost_on_done=True,
            nthread=16,
        )
    task.write_text(json.dumps(dict(task=task_values, xml_sha256=binding(xml)["sha256"])))
    files.append(dict(file="inputs/task.json", **binding(task)))
    reference = root / "reference.pt"
    reference.write_bytes(b"opaque fixed reference, no actor construction")
    records, source_rows = [], {str(stream): dict(train=[], held=[]) for stream in range(3)}
    with BinaryEvidenceMeter(
        run / "worker/evidence", max_controls=0, max_neural_rows=0, max_bytes=32 * inputs.MIB
    ) as meter:
        for stream in range(3):
            for panel, base in (("primary", 2810000000), ("confirmation", 2820000000)):
                for case in range(16):
                    tag = f"eval_s{stream}_parent_{panel}_{case:02d}"
                    state = np.zeros((5, 19), dtype="<f8")
                    state[:, 0] = np.arange(5) * 0.008
                    state[:, 2] = 1.2
                    if physical:
                        state[:, 10] = 1.5
                        state[:, 11] = 1.0
                        state[-1, 2] = 0.75
                    if mismatch and stream == 1 and panel == "primary" and case == 0:
                        state[0, 3] += 0.25
                    arrays = dict(
                        states=state,
                        observations=np.concatenate(
                            (state[:, 2:10], np.clip(state[:, 10:19], -10, 10)), axis=1
                        ),
                        warmstarts=np.zeros((5, 9), dtype="<f8"),
                        integrations=np.zeros((5, 91), dtype="<f8"),
                        actions=np.full((4, 6), 0.125, dtype="<f8"),
                        native_costs=np.array([1, 1, 1, 21], dtype="<f8"),
                        x_velocities=np.full(4, 1.5, dtype="<f8"),
                        fall_costs=np.array([0, 0, 0, 20], dtype="<f8"),
                        done=np.array([0, 0, 0, 1], dtype="u1"),
                        control_ordinals=np.arange(4, dtype="<u4"),
                    )
                    digest = hashlib.sha256()
                    for key in inputs.BOUNDARY_NAMES:
                        digest.update(arrays[key][0].tobytes())
                    previous = digest.digest()
                    arrays["initial_digest"] = np.frombuffer(previous, dtype="u1")
                    rows = []
                    for index in range(4):
                        digest = hashlib.sha256(previous)
                        for key in inputs.BOUNDARY_NAMES:
                            digest.update(arrays[key][index + 1].tobytes())
                        for key in inputs.ARRAY_NAMES[4:10]:
                            digest.update(arrays[key][index].tobytes())
                        previous = digest.digest()
                        rows.append(np.frombuffer(previous, dtype="u1"))
                    arrays["row_digests"] = np.array(rows, dtype="u1")
                    arrays["metadata"] = dict(
                        alias=tag,
                        actor=f"s{stream}_parent",
                        seed=base + case,
                        horizon=1000,
                        completed_controls=4,
                        availability="closed_actual_prefix",
                        unknown_partial_controls=0,
                        unknown_partial_physics_substeps=0,
                        integration_spec=16383,
                        dimensions=[9, 9, 6],
                    )
                    if physical:
                        arrays["metadata"]["absorbing_cost"] = 4980.0
                    if missing_terminal and stream == case == 0 and panel == "primary":
                        arrays["states"] = arrays["states"][:-1]
                    ref = meter.save_arrays(tag, **arrays)
                    records.append(
                        dict(
                            tag=tag,
                            trace=ref,
                            controls=4,
                            done=True,
                            actor=f"s{stream}_parent",
                            seed=base + case,
                            stream=stream,
                            arm="parent",
                            panel=panel,
                            case=case,
                            h=1000,
                        )
                    )
                    if physical:
                        records[-1].update(
                            raw_cost=4.0,
                            native_cost=24.0,
                            absorbing_cost=4980.0,
                            cost=5004.0,
                            native_terminal=20.0,
                            survived=False,
                            walking=False,
                            mae=0.0,
                        )
                    split = "held" if panel == "confirmation" and case >= 4 else "train"
                    source_rows[str(stream)][split].append(
                        dict(
                            tag=tag,
                            seed=base + case,
                            controls=4,
                            array_group=ref["group"],
                            selected_quota=4,
                        )
                    )
    if wrong_actor:
        registry["s1_parent"]["actor_hash"] = "f" * 64
    config = dict(xml_sha256=binding(xml)["sha256"])
    metadata_bindings = {}
    for name, relative in inputs.METADATA_PATHS.items():
        path = run / relative
        if name == "c1_evaluations":
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
        else:
            value = {
                "c1_config": config,
                "c1_actor_registry": registry,
                "c1_prerequisites": dict(passed=True),
                "c1_prepared_manifest": dict(files=files),
            }[name]
            write_json(path, value)
        metadata_bindings[name] = dict(file="irrelevant/capsule/" + relative, **binding(path))
    admission = write_json(
        root / "admission.json",
        dict(
            schema="walker-c1-certificate-admission-v1",
            audit_recovery_resolved=True,
            retained_A1_integrity_certification=True,
            retained_original_numerical_certification=True,
        ),
    )
    certified = write_json(
        root / "verified_files.json",
        dict(
            files={
                "run/worker/evidence/" + name: binding(run / "worker/evidence" / name)
                for name in ("arrays.bin", "arrays.index.jsonl")
            }
        ),
    )
    anchors = []
    for stream in range(3):
        for panel, source_panel, cases in (
            ("primary", "primary", range(4)),
            ("confirmation", "confirmation", range(4, 8)),
        ):
            for case in cases:
                tag = f"eval_s{stream}_parent_{source_panel}_{case:02d}"
                record = next(row for row in records if row["tag"] == tag)
                for ordinal in range(4):
                    anchors.append(
                        dict(
                            stream=stream,
                            panel=panel,
                            source_tag=tag,
                            source_case=case,
                            source_controls=4,
                            anchor_ordinal=ordinal,
                            control=ordinal,
                            array_group=record["trace"]["group"],
                            screen_role="secondary_initial"
                            if ordinal == 0
                            else "primary_noninitial",
                        )
                    )
    review = dict(
        schema="walker-student-visitation-s1.2-review-v2",
        c1_recovery_receipt=admission,
        input_bindings=metadata_bindings,
        parents=parents,
        source_episodes=source_rows,
        normalization=dict(file="inputs/normalization.npz", **binding(norm)),
        reference=dict(checkpoint_sha256=binding(reference)["sha256"], actor_hash="e" * 64),
        validity=dict(
            anchors=anchors,
            positive_control_reference_reset_seeds=(
                list(range(2810000000, 2810000016)) + list(range(2820000000, 2820000016))
            ),
        ),
        finite_census=dict(
            validity=dict(expected_unchanged_parent_replay_controls_from_bound_metadata=240)
        ),
    )
    recipe = write_json(root / "review.json", review)
    preparation = dict(
        c1_run=str(run),
        review_recipe=recipe,
        a1_verified_files=certified,
        recovery_admission=admission,
        reference=dict(path=str(reference), **binding(reference)),
        xml=dict(path=str(xml), **binding(xml)),
    )
    return dict(config=dict(input_preparation=preparation), source=run, records=records, root=root)


@pytest.fixture(scope="module")
def saved_source(tmp_path_factory):
    return source_fixture(tmp_path_factory.mktemp("visitation_saved_source"))


def test_prepare_preserves_full_prefixes_and_complete_opaque_checkpoints(saved_source, tmp_path):
    result = inputs.prepare(saved_source["config"], tmp_path / "new")
    catalog = inputs.verify_prepared(tmp_path / "new/prepared")
    assert result["passed"] is True
    assert len(catalog["initial_contexts"]) == 96
    assert len(catalog["canonical_contexts"]) == 32
    assert len(catalog["episodes"]) == 24
    assert all(set(row) == {"0", "1", "2"} for row in catalog["parent_initial_contexts"].values())
    first = catalog["episodes"]["eval_s0_parent_primary_00"]
    with np.load(tmp_path / "new/prepared" / first["file"], allow_pickle=False) as data:
        assert data["states"].shape == (5, 19)
        assert data["observations"].shape == (5, 17)
        assert data["done"].tolist() == [0, 0, 0, 1]
        assert int(data["spec"]) == 16383
    parent = catalog["parents"]["0"]
    assert (tmp_path / "new/prepared" / parent["file"]).read_bytes() == (
        saved_source["source"] / "prepared/parents/s0_parent.pt"
    ).read_bytes()
    assert catalog["_verification_accounting"]["new_neural_rows"] == 0
    assert result["accounting"]["source_read_bytes"] <= 512 * inputs.MIB


def test_preparation_is_create_only(saved_source, tmp_path):
    run = tmp_path / "existing"
    (run / "prepared").mkdir(parents=True)
    marker = run / "prepared/retained.txt"
    marker.write_bytes(b"retain")
    with pytest.raises(inputs.ConsequenceInputError, match="already exists"):
        inputs.prepare(saved_source["config"], run)
    assert marker.read_bytes() == b"retain"


@pytest.mark.parametrize(
    "option,reason",
    [
        ("mismatch", "initial contexts disagree"),
        ("missing_terminal", "shape/dtype/nonfinite"),
        ("wrong_actor", "row join differs"),
    ],
)
def test_invalid_sources_stop_and_preserve_failure(tmp_path, option, reason):
    source = source_fixture(tmp_path / "source", **{option: True})
    with pytest.raises(inputs.ConsequenceInputError, match=reason):
        inputs.prepare(source["config"], tmp_path / "new")
    failure = json.loads((tmp_path / "new/prepared/failure.json").read_text())
    assert failure["passed"] is False
    assert failure["scientific_work"] == 0


def test_sparse_array_tampering_stops_before_any_science(tmp_path):
    source = source_fixture(tmp_path / "source")
    descriptor = source["records"][0]["trace"]["arrays"]["states"]
    with (source["source"] / "worker/evidence/arrays.bin").open("r+b") as stream:
        stream.seek(descriptor["offset"])
        stream.write(b"x")
    with pytest.raises(inputs.ConsequenceInputError, match="C-byte SHA256"):
        inputs.prepare(source["config"], tmp_path / "new")


def test_budget_failure_retains_receipt_and_no_work(saved_source, tmp_path):
    config = copy.deepcopy(saved_source["config"])
    config["input_preparation"]["limits"] = dict(input_read_bytes=128)
    with pytest.raises(inputs.ConsequenceInputError, match="read budget"):
        inputs.prepare(config, tmp_path / "new")
    receipt = json.loads((tmp_path / "new/prepared/failure.json").read_text())
    assert receipt["accounting"]["new_native_controls"] == 0


def test_preparation_memory_guard_still_stops_before_input_reads(
    saved_source, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "src.gps.mppi_gps.consequence_inputs._peak_rss_bytes", lambda: 513 * inputs.MIB
    )
    with pytest.raises(inputs.ConsequenceInputError, match="peak RSS"):
        inputs.prepare(saved_source["config"], tmp_path / "new")
    assert not (tmp_path / "new/prepared/catalog.json").exists()


def test_prepared_tampering_is_detected(saved_source, tmp_path):
    inputs.prepare(saved_source["config"], tmp_path / "new")
    target = tmp_path / "new/prepared/parents/s0_parent.pt"
    target.write_bytes(b"changed")
    with pytest.raises(inputs.ConsequenceInputError, match="size changed"):
        inputs.verify_prepared(tmp_path / "new/prepared")


def test_input_module_imports_neither_torch_nor_simulator():
    command = [
        sys.executable,
        "-c",
        "import sys; from src.gps.mppi_gps import visitation_inputs; "
        "assert 'torch' not in sys.modules; assert 'mujoco' not in sys.modules; "
        "assert 'src.envs.walker2d' not in sys.modules",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
