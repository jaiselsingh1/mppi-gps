"""Create the source-cloning benchmark from the published D1/D2 evidence.

Only saved bytes and NumPy arrays are read. The complete D1 source/input seals
are verified before selecting any example; no checkpoint is decoded, policy is
queried, simulator is imported, or original missing context is reconstructed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import sha, write
from .transfer_inputs import array_binding, load_arrays

SCHEMA = "walker-student-source-s1-inputs-v1"
EPISODE_ROWS = 1000
CAPSULE_PATH = Path(__file__).resolve().parents[3] / "results/transfer_diagnosis_d1d2/capsule.json"
ROSTER = {
    "mppi": {
        "train": tuple(f"discovery_{index:02d}" for index in range(4)),
        "validation": ("v2_r0_trial_raw_0", "v2_r0_trial_raw_1"),
    },
    "sac_reference": {
        "train": tuple(f"k1_primary_reference_{index:02d}" for index in range(4)),
        "validation": ("k1_primary_reference_04", "k1_primary_reference_05"),
    },
}
FIELDS = ("states", "observations", "actions", "row_episode", "row_step", "row_kind")


def _digest(binding: str | dict[str, Any]) -> str:
    return binding if isinstance(binding, str) else binding["sha256"]


def _path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Saved source path escapes its owner")
    return path


def _verify(path: Path, binding: str | dict[str, Any]) -> None:
    if sha(path) != _digest(binding):
        raise ValueError("Saved source hash differs: " + str(path))
    if isinstance(binding, dict) and "bytes" in binding and path.stat().st_size != binding["bytes"]:
        raise ValueError("Saved source size differs: " + str(path))


def _copy(source: Path, destination: Path, expected: str | dict[str, Any]) -> dict[str, Any]:
    _verify(source, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    _verify(destination, expected)
    result = {"sha256": sha(destination), "bytes": destination.stat().st_size}
    if destination.suffix == ".npz":
        with np.load(destination, allow_pickle=False) as stream:
            result["arrays"] = {key: array_binding(stream[key]) for key in stream.files}
    return result


def _save(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Object/pickle arrays are prohibited")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "arrays": {key: array_binding(value) for key, value in arrays.items()},
    }


def fitting_hash(array: np.ndarray) -> str:
    """Use the existing float32 fitting-data hash, not the raw-array contract."""
    value = np.ascontiguousarray(array, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(value.shape), str(value.dtype)]).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _published_source(diagnostic: Path, config: dict[str, Any]) -> dict[str, Any]:
    capsule = json.loads(CAPSULE_PATH.read_text())
    if capsule["study"] != "transfer_diagnosis_d1d2":
        raise ValueError("Expected the published D1/D2 capsule")
    metadata = {}
    for relative in (
        "config.json",
        "input_manifest.json",
        "manifest.json",
        "result.json",
        "launch_receipt.json",
        "inputs/inventory.json",
    ):
        binding = capsule["files"]["run/" + relative]
        path = _path(diagnostic, relative)
        _verify(path, binding)
        metadata[relative] = json.loads(path.read_text())
    pins = config["source_bindings"]
    for name, pin in (
        ("config.json", "diagnostic_config_sha256"),
        ("input_manifest.json", "diagnostic_input_manifest_sha256"),
        ("result.json", "diagnostic_result_sha256"),
    ):
        if sha(diagnostic / name) != pins[pin]:
            raise ValueError("S1 diagnostic pin differs: " + pin)
    seal, manifest = metadata["input_manifest.json"], metadata["manifest.json"]
    result, launch = metadata["result.json"], metadata["launch_receipt.json"]
    configuration = sha(diagnostic / "config.json")
    input_hash, source_hash = (
        sha(diagnostic / "input_manifest.json"),
        sha(diagnostic / "manifest.json"),
    )
    if (
        not result["execution_complete"]
        or result.get("error")
        or result["status"] != "diagnosis_complete"
    ):
        raise ValueError("Only the complete D1/D2 diagnosis is an eligible source")
    if any(owner["config_sha256"] != configuration for owner in (seal, manifest, result, launch)):
        raise ValueError("D1 configuration ownership differs")
    if (
        any(owner["source_manifest_sha256"] != source_hash for owner in (seal, launch))
        or result["input_manifest_sha256"] != input_hash
        or launch["input_manifest_sha256"] != input_hash
    ):
        raise ValueError("D1 source/input ownership differs")
    for entry in manifest["files"]:
        _verify(_path(diagnostic / "source", entry["file"]), entry)
    for name, binding in seal["files"].items():
        _verify(_path(diagnostic, name), binding)
    inventory = metadata["inputs/inventory.json"]
    if inventory["schema"] != "walker-transfer-inputs-d1d2-v1":
        raise ValueError("Expected the sealed portable D1 input inventory")
    for name, binding in inventory["files"].items():
        if binding["sha256"] != _digest(seal["files"]["inputs/" + name]):
            raise ValueError("D1 input inventory/seal binding differs")
    return {
        "metadata": metadata,
        "metadata_bindings": {name: capsule["files"]["run/" + name] for name in metadata},
        "inventory": inventory,
        "verified_source_files": len(manifest["files"]),
        "verified_input_files": len(seal["files"]),
        "capsule_sha256": sha(CAPSULE_PATH),
    }


def _lineage(diagnostic: Path, inventory: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for study in ("k1", "w2"):
        entries = {}
        for name in ("config", "input_manifest", "manifest", "result"):
            relative = f"sources/{study}_{name}.json"
            path = diagnostic / "inputs" / relative
            _verify(path, inventory["files"][relative])
            entries[name] = json.loads(path.read_text())
        seal = entries["input_manifest"]
        result_row = entries["result"]
        if (
            sha(diagnostic / "inputs/sources" / f"{study}_input_manifest.json")
            != config["source_bindings"][study + "_input_manifest_sha256"]
        ):
            raise ValueError("Original study input pin differs: " + study)
        configuration = sha(diagnostic / "inputs/sources" / f"{study}_config.json")
        if (
            seal["config_sha256"] != configuration
            or result_row.get("config_sha256", result_row.get("protocol_sha256")) != configuration
        ):
            raise ValueError("Original study configuration binding differs: " + study)
        if (
            result_row["input_manifest_sha256"]
            != config["source_bindings"][study + "_input_manifest_sha256"]
        ):
            raise ValueError("Original study result/input binding differs: " + study)
        if (
            sha(diagnostic / "inputs/sources" / f"{study}_manifest.json")
            != seal["source_manifest_sha256"]
        ):
            raise ValueError("Original study source manifest differs: " + study)
        if not result_row["execution_complete"] or result_row.get("error"):
            raise ValueError("Only completed original studies are eligible")
        result[study] = entries
    return result


def _matches(array: np.ndarray, description: dict[str, Any]) -> bool:
    binding = array_binding(array)
    return (
        binding["shape"] == description["shape"]
        and np.dtype(binding["dtype"]) == np.dtype(description["dtype"])
        and binding["sha256"] == description["sha256"]
    )


def _pairs(values: dict[str, np.ndarray], rows: int, action_dtype: str) -> None:
    expected = {"states": (rows, 19), "observations": (rows, 17), "actions": (rows, 6)}
    for key, shape in expected.items():
        array = values[key]
        dtype = np.dtype(action_dtype if key == "actions" else "float64")
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError("Original source shape/dtype/finiteness differs: " + key)
    native = np.concatenate(
        (values["states"][:, 2:10], np.clip(values["states"][:, 10:19], -10, 10)), axis=1
    )
    if not np.array_equal(native, values["observations"]):
        raise ValueError("Source observations are not original pre-action features")
    if np.any(np.abs(values["actions"]) > 1):
        raise ValueError("Source applied torques exceed their bounds")


def _row_keys(value: np.ndarray) -> set[bytes]:
    array = np.ascontiguousarray(value).copy()
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite overlap input")
    array[array == 0] = 0
    return {row.tobytes() for row in array}


def overlap_checks(
    datasets: dict[str, dict[str, dict[str, np.ndarray]]], normalization: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Audit input leakage; repeated action values alone are legitimate labels."""
    views = {}
    for source, splits in datasets.items():
        for split, values in splits.items():
            with np.errstate(over="raise", invalid="raise"):
                normalized = (
                    values["observations"].astype(np.float32) - normalization["mean"]
                ) / normalization["std"]
            views[(source, split)] = {
                "states:float64": values["states"].astype(np.float64),
                "states:float32": values["states"].astype(np.float32),
                "observations:float64": values["observations"].astype(np.float64),
                "observations:float32": values["observations"].astype(np.float32),
                "normalized_inputs:float32": normalized,
            }
    checks = {}
    for left, right in product(datasets, repeat=2):
        for coordinate in views[(left, "train")]:
            training = views[(left, "train")][coordinate]
            held = views[(right, "validation")][coordinate]
            first, second = _row_keys(training), _row_keys(held)
            checks[f"{left}/train->{right}/validation:{coordinate}"] = {
                "training_rows": len(training),
                "held_rows": len(held),
                "training_unique": len(first),
                "held_unique": len(second),
                "cross_split_overlap": len(first & second),
            }
    for coordinate in next(iter(views.values())):
        training = np.concatenate([views[(source, "train")][coordinate] for source in datasets])
        held = np.concatenate([views[(source, "validation")][coordinate] for source in datasets])
        first, second = _row_keys(training), _row_keys(held)
        checks[f"combined/train->combined/validation:{coordinate}"] = {
            "training_rows": len(training),
            "held_rows": len(held),
            "training_unique": len(first),
            "held_unique": len(second),
            "cross_split_overlap": len(first & second),
        }
    labels = {}
    for source, splits in datasets.items():
        for split, values in splits.items():
            unique = len(_row_keys(values["actions"]))
            labels[f"{source}/{split}"] = {
                "rows": len(values["actions"]),
                "unique_actions": unique,
                "repeated_action_rows": len(values["actions"]) - unique,
            }
    return {
        "passed": all(row["cross_split_overlap"] == 0 for row in checks.values()),
        "input_checks": checks,
        "action_value_statistics": labels,
        "equality": "Numerical row equality with signed zero canonicalized only for this diagnostic; original evidence bytes remain unchanged.",
        "scope": "All four source-train to source-validation combinations; action equality alone is not input leakage. Corpora are historical development evidence, not newly blinded data.",
    }


def prepare_inputs(diagnostic: Path, out: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Bind the approved S1 source intervention without learning or simulation."""
    diagnostic, out = Path(diagnostic).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("S1 inputs are create-only")
    if out.is_relative_to(diagnostic):
        raise ValueError("S1 inputs must remain outside the frozen D1/D2 run")
    for source, splits in ROSTER.items():
        for split, episodes in splits.items():
            field = "training_episodes" if split == "train" else "held_episodes"
            if config["arms"][source][field] != list(episodes):
                raise ValueError("S1 source/split episode roles differ")
    if (
        config["training_rows_per_arm"] != 4 * EPISODE_ROWS
        or config["held_rows_per_arm"] != 2 * EPISODE_ROWS
    ):
        raise ValueError("S1 corpus row budgets differ")
    verified = _published_source(diagnostic, config)
    original = verified["inventory"]
    lineage = _lineage(diagnostic, original, config)
    out.mkdir(parents=True)
    files = {}
    for name in ("config", "input_manifest", "manifest", "result", "launch_receipt"):
        relative = f"sources/diagnostic_{name}.json"
        files[relative] = _copy(
            diagnostic / (name + ".json"),
            out / relative,
            verified["metadata_bindings"][name + ".json"],
        )
    files["sources/diagnostic_inventory.json"] = _copy(
        diagnostic / "inputs/inventory.json",
        out / "sources/diagnostic_inventory.json",
        verified["metadata_bindings"]["inputs/inventory.json"],
    )
    for study in ("k1", "w2"):
        for name in ("config", "input_manifest", "manifest", "result"):
            relative = f"sources/{study}_{name}.json"
            files[relative] = _copy(
                diagnostic / "inputs" / relative, out / relative, original["files"][relative]
            )
    for source, target in (
        ("train.npz", "mppi_train.npz"),
        ("validation.npz", "mppi_validation.npz"),
        ("normalization.npz", "normalization.npz"),
        ("task.json", "task.json"),
    ):
        if original["files"][source]["sha256"] != _digest(
            lineage["w2"]["input_manifest"]["files"]["inputs/" + source]
        ):
            raise ValueError("Exact W2 benchmark container ancestry differs")
        files[target] = _copy(
            diagnostic / "inputs" / source, out / target, original["files"][source]
        )
    reference_source = original["policies"]["reference"]
    reference_file = reference_source["checkpoint_file"]
    if (
        original["files"][reference_file]["sha256"]
        != config["source_bindings"]["reference_checkpoint_sha256"]
        or reference_source["policy_hash"] != config["source_bindings"]["reference_policy_hash"]
    ):
        raise ValueError("Reference checkpoint/policy ownership differs")
    if (
        lineage["k1"]["result"]["policy_hashes"]["reference"] != reference_source["policy_hash"]
        or _digest(lineage["k1"]["input_manifest"]["files"]["inputs/checkpoints/reference.pt"])
        != config["source_bindings"]["reference_checkpoint_sha256"]
    ):
        raise ValueError("Original K1 reference checkpoint ancestry differs")
    files["reference.pt"] = _copy(
        diagnostic / "inputs" / reference_file,
        out / "reference.pt",
        original["files"][reference_file],
    )
    values = {
        "mppi": {
            split: load_arrays(out, f"mppi_{split}.npz", {"files": files})
            for split in ("train", "validation")
        },
        "sac_reference": {},
    }
    teachers = {}
    trajectories = {row["id"]: row for row in original["k1_trajectories"]}
    k1_teachers = {row["id"]: row for row in lineage["k1"]["config"]["trajectory_roster"]}
    k1_episodes = {row["tag"]: row for row in lineage["k1"]["result"]["stages"]["B"]["episodes"]}
    for source, splits in ROSTER.items():
        for split, episodes in splits.items():
            collected = []
            for identifier in episodes:
                row = trajectories[identifier]
                source_file = row["file"]
                saved = load_arrays(diagnostic / "inputs", source_file, original)
                n = EPISODE_ROWS
                pairs = {
                    key: saved[key][:n].copy() for key in ("states", "observations", "actions")
                }
                _pairs(pairs, n, "float64" if source == "mppi" else "float32")
                if row["rows"] != n or not all(
                    row["outcome"].get(key) is True for key in ("complete", "success", "survived")
                ):
                    raise ValueError("Selected source episode lacks a complete walking outcome")
                if (
                    not np.isfinite([row["outcome"]["cost"], row["outcome"]["mae"]]).all()
                    or row["outcome"]["cost"] < 0
                    or not 0 <= row["outcome"]["mae"] <= 0.3
                ):
                    raise ValueError("Selected teacher cost/velocity outcome differs")
                if source == "mppi":
                    owner = k1_teachers[identifier]
                    if (
                        row["group"] not in ("original_discovery", "later_independent")
                        or owner["outcome"] != row["outcome"]
                        or lineage["w2"]["result"]["teacher_outcomes"][split][identifier]
                        != row["outcome"]
                    ):
                        raise ValueError("MPPI source outcome/controller role differs")
                    expected_source_hash = _digest(
                        lineage["k1"]["input_manifest"]["files"]["inputs/" + owner["file"]]
                    )
                    if original["files"][source_file]["sha256"] != expected_source_hash:
                        raise ValueError("MPPI original source container differs")
                    for key, pair_array in pairs.items():
                        if not _matches(pair_array, owner["extracted"]["arrays"][key]):
                            raise ValueError("MPPI original pre-action array ancestry differs")
                    mask = values[source][split]["row_episode"] == identifier
                    if mask.sum() != n or not np.array_equal(
                        values[source][split]["row_step"][mask], np.arange(n)
                    ):
                        raise ValueError("MPPI source control order differs")
                    for key in pairs:
                        if not np.array_equal(values[source][split][key][mask], pairs[key]):
                            raise ValueError("MPPI benchmark/source ownership differs")
                    seed, source_raw = owner["seed"], owner["source"]["raw"]
                else:
                    owner = k1_episodes[identifier]
                    if (
                        row["group"] != "qualified_reference"
                        or owner["policy"] != "reference"
                        or owner["cohort"] != "primary"
                        or owner["steps"] != n
                    ):
                        raise ValueError("Reference source episode/controller role differs")
                    expected_outcome = {key: value for key, value in owner.items() if key != "raw"}
                    if expected_outcome != row["outcome"]:
                        raise ValueError("Reference original native outcome differs")
                    for key in ("states", "observations", "actions"):
                        if not _matches(saved[key], owner["raw"]["arrays"][key]):
                            raise ValueError("Reference original executed array ancestry differs")
                    pairs.update(
                        row_episode=np.repeat(identifier, n),
                        row_step=np.arange(n),
                        row_kind=np.repeat("executed_reference", n),
                    )
                    collected.append(pairs)
                    seed, source_raw = owner["seed"], owner["raw"]
                audit_file = f"teachers/{identifier}.npz"
                files[audit_file] = _copy(
                    diagnostic / "inputs" / source_file,
                    out / audit_file,
                    original["files"][source_file],
                )
                teachers[identifier] = {
                    "id": identifier,
                    "source": source,
                    "split": split,
                    "rows": n,
                    "seed": seed,
                    "outcome": row["outcome"],
                    "source_file": source_file,
                    "source_file_sha256": original["files"][source_file]["sha256"],
                    "audit_file": audit_file,
                    "source_raw": source_raw,
                    "array_bindings": {
                        key: array_binding(pairs[key])
                        for key in ("states", "observations", "actions")
                    },
                }
            if source == "sac_reference":
                values[source][split] = {
                    key: np.concatenate([item[key] for item in collected]) for key in FIELDS
                }
                relative = f"{source}_{split}.npz"
                files[relative] = _save(out / relative, values[source][split])
    normalization = load_arrays(out, "normalization.npz", {"files": files})
    expected_normalization = {
        "mean": values["mppi"]["train"]["observations"].mean(axis=0).astype(np.float32),
        "std": np.maximum(values["mppi"]["train"]["observations"].std(axis=0), 0.05).astype(
            np.float32
        ),
    }
    if set(normalization) != {"mean", "std"} or any(
        normalization[key].dtype != np.float32
        or not np.array_equal(normalization[key], expected_normalization[key])
        for key in expected_normalization
    ):
        raise ValueError("Shared original MPPI training-only normalization differs")
    task = json.loads((out / "task.json").read_text())
    if (
        task
        != {
            "task": lineage["w2"]["config"]["task"],
            "xml_sha256": lineage["w2"]["config"]["xml_sha256"],
        }
        or task["task"] != lineage["k1"]["config"]["task"]
    ):
        raise ValueError("Native source task/XML ownership differs")
    datasets = {}
    for source, splits in values.items():
        datasets[source] = {}
        for split, pairs in splits.items():
            episodes = ROSTER[source][split]
            if (
                set(pairs) != set(FIELDS)
                or not np.array_equal(pairs["row_episode"], np.repeat(episodes, EPISODE_ROWS))
                or not np.array_equal(
                    pairs["row_step"], np.tile(np.arange(EPISODE_ROWS), len(episodes))
                )
            ):
                raise ValueError("Exact source/split row roles differ")
            kind = (
                "executed_reference"
                if source == "sac_reference"
                else "executed_discovery"
                if split == "train"
                else "executed_added_independent"
            )
            if (
                pairs["row_episode"].dtype.kind != "U"
                or pairs["row_step"].dtype.kind not in "iu"
                or not np.all(pairs["row_kind"] == kind)
            ):
                raise ValueError("Original executed row classification differs")
            datasets[source][split] = {
                "file": f"{source}_{split}.npz",
                "rows": len(pairs["observations"]),
                "episodes": list(episodes),
                "fit_data": {
                    "rows": len(pairs["observations"]),
                    "observations_sha256": fitting_hash(pairs["observations"]),
                    "actions_sha256": fitting_hash(pairs["actions"]),
                },
            }
    overlap = overlap_checks(values, normalization)
    write(out / "overlap_checks.json", overlap)
    files["overlap_checks.json"] = {
        "sha256": sha(out / "overlap_checks.json"),
        "bytes": (out / "overlap_checks.json").stat().st_size,
    }
    if not overlap["passed"]:
        raise ValueError("Cross-split input leakage; preserve original IDs without replacement")
    inventory = {
        "schema": SCHEMA,
        "source_bindings": config["source_bindings"],
        "sources": {
            "diagnostic": str(diagnostic),
            "capsule_sha256": verified["capsule_sha256"],
            "verified_source_files": verified["verified_source_files"],
            "verified_input_files": verified["verified_input_files"],
            "scope": "Complete published D1 source/input file seals and selected original K1/W2 lineage; no historical network, optimizer or dynamics arithmetic replay.",
        },
        "files": files,
        "datasets": datasets,
        "teachers": teachers,
        "normalization": {
            "file": "normalization.npz",
            "training_only": True,
            "rows": 4 * EPISODE_ROWS,
            "source": "mppi/train",
            "rule": "Original float64 training mean/population std, floor0.05, thenfloat32; exact W2 buffer bytes.",
        },
        "reference": {
            "file": "reference.pt",
            "checkpoint_sha256": config["source_bindings"]["reference_checkpoint_sha256"],
            "policy_hash": reference_source["policy_hash"],
        },
        "overlap_checks": overlap,
        "missing_evidence": original["missing_evidence"]
        + [
            "Original discovery local terminal/warmstart/integration context remains unavailable; no context is reconstructed for S1 fitting.",
            "Both source corpora are previously examined development evidence; source/reset/gait/action-statistics differences are part of the whole-corpus intervention.",
        ],
        "new_forward_rows": 0,
        "new_physics_controls": 0,
        "new_optimizer_steps": 0,
    }
    write(out / "inventory.json", inventory)
    return inventory
