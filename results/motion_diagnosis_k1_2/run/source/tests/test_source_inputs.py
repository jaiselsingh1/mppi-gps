"""Synthetic S1 lifecycle and lineage checks without neural or native execution."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from src.gps.mppi_gps import source_inputs as source
from src.gps.mppi_gps.provenance import sha, write
from src.gps.mppi_gps.transfer_inputs import array_binding


def native_rows(rows: int, identity: int, action_dtype="float64", endpoint=False):
    states = np.zeros((rows + int(endpoint), 19), dtype=np.float64)
    states[:, 0] = np.arange(len(states)) * 0.008
    states[:, 1] = identity * 0.01 + np.arange(len(states)) * 0.001
    states[:, 2] = 1.1 + identity * 0.001
    states[:, 3] = identity * 0.001
    states[:, 10] = 1.5 + np.arange(len(states)) * 0.0003
    observations = np.concatenate(
        (states[:rows, 2:10], np.clip(states[:rows, 10:19], -10, 10)), axis=1
    )
    return {
        "states": states,
        "observations": observations,
        "actions": np.full((rows, 6), 0.1, dtype=action_dtype),
    }


def fixture_source(tmp_path, monkeypatch, *, reference_dtype="float32", wrong_row_kind=False):
    """A tiny complete historical chain with the actual fixed episode roles."""
    monkeypatch.setattr(source, "EPISODE_ROWS", 3)
    diagnostic = tmp_path / "diagnostic"
    inputs = diagnostic / "inputs"
    inputs.mkdir(parents=True)
    files, trajectories, k1_teachers, k1_episodes, pairs_by_id = {}, [], [], [], {}
    w2_outcomes = {"train": {}, "validation": {}}
    reference_hash = "d" * 64
    task = {"frame_skip": 4, "target_velocity": 1.5}
    xml_hash = "a" * 64

    def put(name, values):
        files[name] = source._save(inputs / name, values)

    identity = 0
    for family, splits in source.ROSTER.items():
        for split, identifiers in splits.items():
            for identifier in identifiers:
                is_reference = family == "sac_reference"
                values = native_rows(
                    3,
                    identity,
                    reference_dtype if is_reference else "float64",
                    endpoint=is_reference,
                )
                identity += 1
                pairs_by_id[identifier] = {key: value[:3] for key, value in values.items()}
                name = f"k1/trajectories/{identifier}.npz"
                put(name, values)
                outcome = {
                    "complete": True,
                    "success": True,
                    "survived": True,
                    "cost": 2.0 + identity,
                    "mae": 0.1,
                }
                raw = {
                    "file": "raw.h5",
                    "group": identifier,
                    "arrays": {key: array_binding(value) for key, value in values.items()},
                }
                if is_reference:
                    seed = 2026103100 + int(identifier[-2:])
                    episode = {
                        **outcome,
                        "tag": identifier,
                        "policy": "reference",
                        "cohort": "primary",
                        "seed": seed,
                        "steps": 3,
                        "raw": raw,
                    }
                    outcome = {key: value for key, value in episode.items() if key != "raw"}
                    k1_episodes.append(episode)
                    group = "qualified_reference"
                else:
                    seed = (811000000 if split == "train" else 943000000) + int(
                        identifier[-2:] if split == "train" else identifier[-1]
                    )
                    group = "original_discovery" if split == "train" else "later_independent"
                    descriptor = {
                        "id": identifier,
                        "file": f"teachers/{identifier}.npz",
                        "seed": seed,
                        "steps": 3,
                        "outcome": outcome,
                        "source": {"raw": raw},
                        "extracted": {
                            "arrays": {key: array_binding(value) for key, value in values.items()}
                        },
                    }
                    k1_teachers.append(descriptor)
                    w2_outcomes[split][identifier] = outcome
                trajectories.append(
                    {"id": identifier, "group": group, "rows": 3, "file": name, "outcome": outcome}
                )
    benchmark = {}
    for split, identifiers in source.ROSTER["mppi"].items():
        values = {
            key: np.concatenate([pairs_by_id[name][key] for name in identifiers])
            for key in ("states", "observations", "actions")
        }
        kind = "executed_discovery" if split == "train" else "executed_added_independent"
        values.update(
            row_episode=np.repeat(identifiers, 3),
            row_step=np.tile(np.arange(3), len(identifiers)),
            row_kind=np.repeat("hypothetical" if wrong_row_kind else kind, 3 * len(identifiers)),
        )
        benchmark[split] = values
        put(f"{split}.npz", values)
    train_obs = benchmark["train"]["observations"]
    put(
        "normalization.npz",
        {
            "mean": train_obs.mean(axis=0).astype(np.float32),
            "std": np.maximum(train_obs.std(axis=0), 0.05).astype(np.float32),
        },
    )
    write(inputs / "task.json", {"task": task, "xml_sha256": xml_hash})
    files["task.json"] = {
        "sha256": sha(inputs / "task.json"),
        "bytes": (inputs / "task.json").stat().st_size,
    }
    checkpoint = inputs / "checkpoints/k1_reference.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"opaque-checkpoint-bytes-not-loaded")
    files["checkpoints/k1_reference.pt"] = {
        "sha256": sha(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }

    def metadata(study, kind, value):
        name = f"sources/{study}_{kind}.json"
        write(inputs / name, value)
        files[name] = {"sha256": sha(inputs / name), "bytes": (inputs / name).stat().st_size}
        return files[name]["sha256"]

    k1_cfg_hash = metadata("k1", "config", {"trajectory_roster": k1_teachers, "task": task})
    k1_manifest_hash = metadata("k1", "manifest", {"config_sha256": k1_cfg_hash})
    k1_seal_hash = metadata(
        "k1",
        "input_manifest",
        {
            "config_sha256": k1_cfg_hash,
            "source_manifest_sha256": k1_manifest_hash,
            "files": {
                **{
                    f"inputs/{row['file']}": files[f"k1/trajectories/{row['id']}.npz"]["sha256"]
                    for row in k1_teachers
                },
                "inputs/checkpoints/reference.pt": sha(checkpoint),
            },
        },
    )
    metadata(
        "k1",
        "result",
        {
            "execution_complete": True,
            "error": None,
            "protocol_sha256": k1_cfg_hash,
            "input_manifest_sha256": k1_seal_hash,
            "policy_hashes": {"reference": reference_hash},
            "stages": {"B": {"episodes": k1_episodes}},
        },
    )
    w2_cfg_hash = metadata("w2", "config", {"task": task, "xml_sha256": xml_hash})
    w2_manifest_hash = metadata("w2", "manifest", {"config_sha256": w2_cfg_hash})
    w2_seal_hash = metadata(
        "w2",
        "input_manifest",
        {
            "config_sha256": w2_cfg_hash,
            "source_manifest_sha256": w2_manifest_hash,
            "files": {
                f"inputs/{name}": files[name]["sha256"]
                for name in ("train.npz", "validation.npz", "normalization.npz", "task.json")
            },
        },
    )
    metadata(
        "w2",
        "result",
        {
            "execution_complete": True,
            "error": None,
            "config_sha256": w2_cfg_hash,
            "input_manifest_sha256": w2_seal_hash,
            "teacher_outcomes": w2_outcomes,
        },
    )
    write(
        inputs / "inventory.json",
        {
            "schema": "walker-transfer-inputs-d1d2-v1",
            "files": files,
            "policies": {
                "reference": {
                    "checkpoint_file": "checkpoints/k1_reference.pt",
                    "policy_hash": reference_hash,
                }
            },
            "k1_trajectories": trajectories,
            "missing_evidence": ["Original discovery context absent."],
        },
    )
    write(diagnostic / "config.json", {"schema": "fixture-diagnostic"})
    cfg_hash = sha(diagnostic / "config.json")
    (diagnostic / "source").mkdir()
    (diagnostic / "source/worker.py").write_text("# A frozen source fixture, never executed.\n")
    write(
        diagnostic / "manifest.json",
        {
            "config_sha256": cfg_hash,
            "files": [{"file": "worker.py", "sha256": sha(diagnostic / "source/worker.py")}],
        },
    )
    manifest_hash = sha(diagnostic / "manifest.json")
    write(
        diagnostic / "input_manifest.json",
        {
            "config_sha256": cfg_hash,
            "source_manifest_sha256": manifest_hash,
            "files": {
                str(path.relative_to(diagnostic)): {
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                }
                for path in inputs.rglob("*")
                if path.is_file()
            },
        },
    )
    seal_hash = sha(diagnostic / "input_manifest.json")
    write(
        diagnostic / "result.json",
        {
            "status": "diagnosis_complete",
            "execution_complete": True,
            "error": None,
            "config_sha256": cfg_hash,
            "input_manifest_sha256": seal_hash,
        },
    )
    write(
        diagnostic / "launch_receipt.json",
        {
            "config_sha256": cfg_hash,
            "source_manifest_sha256": manifest_hash,
            "input_manifest_sha256": seal_hash,
        },
    )
    published = tmp_path / "capsule.json"
    write(
        published,
        {
            "study": "transfer_diagnosis_d1d2",
            "files": {
                "run/" + name: {
                    "sha256": sha(diagnostic / name),
                    "bytes": (diagnostic / name).stat().st_size,
                }
                for name in (
                    "config.json",
                    "input_manifest.json",
                    "manifest.json",
                    "result.json",
                    "launch_receipt.json",
                    "inputs/inventory.json",
                )
            },
        },
    )
    monkeypatch.setattr(source, "CAPSULE_PATH", published)
    config = {
        "source_bindings": {
            "diagnostic_config_sha256": cfg_hash,
            "diagnostic_input_manifest_sha256": seal_hash,
            "diagnostic_result_sha256": sha(diagnostic / "result.json"),
            "k1_input_manifest_sha256": k1_seal_hash,
            "w2_input_manifest_sha256": w2_seal_hash,
            "reference_checkpoint_sha256": sha(checkpoint),
            "reference_policy_hash": reference_hash,
        },
        "arms": {
            family: {
                "training_episodes": list(splits["train"]),
                "held_episodes": list(splits["validation"]),
            }
            for family, splits in source.ROSTER.items()
        },
        "training_rows_per_arm": 12,
        "held_rows_per_arm": 6,
    }
    return diagnostic, config


def test_source_inputs_imports_no_simulator_or_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.gps.mppi_gps.source_inputs; assert not any(x in sys.modules for x in ('torch','mujoco','src.envs.walker2d'))",
        ],
        check=True,
    )


def test_complete_synthetic_preparation_preserves_roles_dtypes_and_exact_sources(
    tmp_path, monkeypatch
):
    diagnostic, config = fixture_source(tmp_path, monkeypatch)
    out = tmp_path / "new-inputs"
    inventory = source.prepare_inputs(diagnostic, out, config)
    assert inventory["sources"]["verified_source_files"] == 1
    assert len(inventory["teachers"]) == 12
    assert inventory["overlap_checks"]["passed"]
    assert len(inventory["overlap_checks"]["input_checks"]) == 25
    assert (
        inventory["new_forward_rows"]
        == inventory["new_physics_controls"]
        == inventory["new_optimizer_steps"]
        == 0
    )
    assert (out / "mppi_train.npz").read_bytes() == (diagnostic / "inputs/train.npz").read_bytes()
    assert (out / "normalization.npz").read_bytes() == (
        diagnostic / "inputs/normalization.npz"
    ).read_bytes()
    assert (out / "reference.pt").read_bytes() == b"opaque-checkpoint-bytes-not-loaded"
    with np.load(out / "sac_reference_train.npz", allow_pickle=False) as values:
        assert values["actions"].dtype == np.float32
        assert values["states"].dtype == np.float64
        assert values["row_episode"].dtype.kind == "U"
        assert np.array_equal(values["row_step"], np.tile(np.arange(3), 4))
        assert np.all(values["row_kind"] == "executed_reference")
    for teacher in inventory["teachers"].values():
        assert (out / teacher["audit_file"]).read_bytes() == (
            diagnostic / "inputs" / teacher["source_file"]
        ).read_bytes()
        assert teacher["split"] in ("train", "validation")


@pytest.mark.parametrize("relative", ["source/worker.py", "inputs/train.npz", "result.json"])
def test_published_source_input_and_result_tampering_are_rejected(tmp_path, monkeypatch, relative):
    diagnostic, config = fixture_source(tmp_path, monkeypatch)
    path = diagnostic / relative
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash differs"):
        source.prepare_inputs(diagnostic, tmp_path / "new-inputs", config)


def test_pin_mismatch_is_not_replaced_with_a_different_source(tmp_path, monkeypatch):
    diagnostic, config = fixture_source(tmp_path, monkeypatch)
    config["source_bindings"]["diagnostic_result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="diagnostic pin"):
        source.prepare_inputs(diagnostic, tmp_path / "new-inputs", config)


def test_source_role_change_is_rejected_before_copying(tmp_path, monkeypatch):
    diagnostic, config = fixture_source(tmp_path, monkeypatch)
    config["arms"]["sac_reference"]["held_episodes"][0] = "k1_primary_reference_00"
    with pytest.raises(ValueError, match="episode roles"):
        source.prepare_inputs(diagnostic, tmp_path / "new-inputs", config)
    assert not (tmp_path / "new-inputs").exists()


def test_reference_target_dtype_cannot_be_changed_even_with_coherent_source_hashes(
    tmp_path, monkeypatch
):
    diagnostic, config = fixture_source(tmp_path, monkeypatch, reference_dtype="float64")
    with pytest.raises(ValueError, match="dtype"):
        source.prepare_inputs(diagnostic, tmp_path / "new-inputs", config)


def test_row_kind_cannot_admit_hypothetical_pairs(tmp_path, monkeypatch):
    diagnostic, config = fixture_source(tmp_path, monkeypatch, wrong_row_kind=True)
    with pytest.raises(ValueError, match="classification"):
        source.prepare_inputs(diagnostic, tmp_path / "new-inputs", config)


def test_inputs_are_create_only_and_never_written_inside_frozen_source(tmp_path):
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError):
        source.prepare_inputs(tmp_path / "absent", out, {})
    with pytest.raises(ValueError, match="outside the frozen"):
        source.prepare_inputs(tmp_path / "frozen", tmp_path / "frozen/new", {})


def test_overlap_rejects_cross_source_inputs_but_not_repeated_actions():
    normalization = {"mean": np.zeros(17, dtype=np.float32), "std": np.ones(17, dtype=np.float32)}
    values = {
        family: {
            split: native_rows(2, index * 2 + offset)
            for offset, split in enumerate(("train", "validation"))
        }
        for index, family in enumerate(("mppi", "sac_reference"))
    }
    result = source.overlap_checks(values, normalization)
    assert result["passed"]
    assert all(
        row["repeated_action_rows"] == 1 for row in result["action_value_statistics"].values()
    )
    values["sac_reference"]["validation"]["observations"][0] = values["mppi"]["train"][
        "observations"
    ][0]
    result = source.overlap_checks(values, normalization)
    assert not result["passed"]
    assert (
        result["input_checks"]["mppi/train->sac_reference/validation:normalized_inputs:float32"][
            "cross_split_overlap"
        ]
        == 1
    )


def test_fitting_hash_converts_targets_without_changing_raw_hash_contract():
    original = np.array([[0.1, 0.2]], dtype=np.float64)
    converted = original.astype(np.float32)
    assert source.fitting_hash(original) == source.fitting_hash(converted)
    assert source.fitting_hash(original) != array_binding(converted)["sha256"]
    assert array_binding(original)["sha256"] != array_binding(converted)["sha256"]
