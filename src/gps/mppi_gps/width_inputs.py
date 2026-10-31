"""Create the W1 executed-only benchmark from already sealed K1 inputs.

This preparation reads small saved files only. It imports neither policies nor
optimizers nor simulator code, and never reconstructs absent teacher endpoints.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import write

TRAIN_TAGS = tuple(f"discovery_{index:02d}" for index in range(4))
VALIDATION_TAGS = tuple(f"v2_r0_trial_raw_{index}" for index in range(2))
FIELDS = ("states", "observations", "actions", "row_episode", "row_steps", "row_kind")


def audit_reset_seeds(repo: Path, seeds: list[int]) -> dict[str, Any]:
    """Census local run configurations and conservatively reserve root offsets.

    Exact integer matches are rejected anywhere in a configuration. Integer
    values in seed/root fields additionally reserve one million offsets. This
    is a metadata freshness guard, not proof about unrecorded reset events.
    """
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
    ):
        raise ValueError("A nonempty list of native uint32 reset seeds is required")
    selected = set(seeds)
    collisions: list[dict[str, Any]] = []
    configurations = []
    for seed in sorted(selected):
        if seeds.count(seed) > 1:
            collisions.append(
                dict(file=None, address="proposed_seeds", seed=seed, reason="duplicate proposal")
            )

    def visit(value: Any, address: str, owner: str) -> None:
        if type(value) is int and value in selected:
            collisions.append(
                dict(file=owner, address=address, seed=value, reason="exact recorded integer")
            )
        if (
            type(value) is int
            and 0 <= value < 2**32
            and any(word in address.lower() for word in ("seed", "root"))
        ):
            for seed in sorted(selected):
                if value < seed < min(value + 1_000_000, 2**32):
                    collisions.append(
                        dict(
                            file=owner,
                            address=address,
                            seed=seed,
                            conservative_root=value,
                            reserved_offsets=1_000_000,
                            reason="conservative root+offset reservation",
                        )
                    )
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, address + "/" + str(key), owner)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, address + f"/{index}", owner)

    paths = sorted((Path(repo).resolve() / "runs").glob("*/config.json"))
    if not paths:
        raise ValueError("No local historical run configurations are available")
    for path in paths:
        data = path.read_bytes()
        visit(json.loads(data), "", str(path))
        configurations.append(dict(file=str(path), sha256=_digest(data)))
    return dict(
        passed=not collisions,
        seeds=list(seeds),
        historical_configurations=configurations,
        collisions=collisions,
        scope="Local runs/*/config.json census, including earlier K1, with exact recorded integers and conservative one-million-offset native uint32 seed/root reservations. No ledger, bulk data, unavailable external run or unrecorded-reset audit; no global freshness guarantee.",
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_binding(value: np.ndarray) -> dict[str, Any]:
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        sha256=_digest(np.ascontiguousarray(value).tobytes()),
    )


def _fit_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float32)
    digest = hashlib.sha256(json.dumps([list(array.shape), str(array.dtype)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _load_arrays(data: bytes, binding: dict[str, Any]) -> dict[str, np.ndarray]:
    if len(data) != binding["bytes"] or _digest(data) != binding["file_sha256"]:
        raise ValueError("NPZ file differs from inventory binding")
    with np.load(io.BytesIO(data), allow_pickle=False) as stream:
        arrays = {key: np.array(stream[key]) for key in stream.files}
    if set(arrays) != set(binding["arrays"]):
        raise ValueError("NPZ array roster differs from inventory")
    for key, value in arrays.items():
        if _array_binding(value) != binding["arrays"][key]:
            raise ValueError("NPZ array differs from inventory: " + key)
    return arrays


def _validate_pairs(values: dict[str, np.ndarray], rows: int) -> None:
    expected = {"states": (rows, 19), "observations": (rows, 17), "actions": (rows, 6)}
    for key, shape in expected.items():
        value = values[key]
        if value.shape != shape or value.dtype != np.dtype("float64"):
            raise ValueError("Original float64 physical/target matrices required: " + key)
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite saved input: " + key)
    states = values["states"]
    observations = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
    if not np.array_equal(observations, values["observations"]):
        raise ValueError("Observations are not the native clipped pre-action features")
    if np.any(np.abs(values["actions"]) > 1):
        raise ValueError("Executed targets must be bounded without relabeling")


def _row_keys(value: np.ndarray) -> set[bytes]:
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite effective input in overlap audit")
    array = np.ascontiguousarray(value).copy()
    array[array == 0] = 0  # Numerical equality includes positive/negative zero.
    return {row.tobytes() for row in array}


def _check_overlap(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    normalization: dict[str, np.ndarray],
) -> dict[str, Any]:
    result = {}

    def check(name: str, first: np.ndarray, second: np.ndarray) -> None:
        left, right = _row_keys(first), _row_keys(second)
        record = dict(
            train_unique_rows=len(left),
            validation_unique_rows=len(right),
            cross_split_overlap=len(left & right),
        )
        if len(left) != len(first) or len(right) != len(second) or left & right:
            raise ValueError("Duplicate/overlapping benchmark rows: " + name)
        result[name] = record

    for key in ("states", "observations", "actions"):
        for dtype in (np.float64, np.float32):
            check(
                key + ":" + np.dtype(dtype).name,
                train[key].astype(dtype),
                validation[key].astype(dtype),
            )
    mean, std = normalization["mean"], normalization["std"]
    check(
        "normalized_inputs:float32",
        (train["observations"].astype(np.float32) - mean) / std,
        (validation["observations"].astype(np.float32) - mean) / std,
    )
    return result


def _save_arrays(path: Path, values: dict[str, np.ndarray]) -> dict[str, Any]:
    with path.open("xb") as stream:
        np.savez(stream, **values)
    data = path.read_bytes()
    return dict(
        file=path.name,
        file_sha256=_digest(data),
        bytes=len(data),
        arrays={key: _array_binding(value) for key, value in values.items()},
    )


def extract_benchmark(k1: Path, destination: Path) -> dict[str, Any]:
    """Create train/validation/normalization NPZs and a source-bound provenance record.

    The K1 launch receipt pins its input seal, which pins its inventory and the
    seven selected NPZ files. Only these small inputs and launch/config/source
    manifests are read; unrelated K1 checkpoints and historical raw are untouched.
    """
    k1, destination = Path(k1).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if destination.is_relative_to(k1):
        raise ValueError("Benchmark output must be outside the frozen K1 run")
    files: dict[str, dict[str, Any]] = {}
    contents: dict[str, bytes] = {}

    def read(name: str, expected: str | None = None) -> bytes:
        path = (k1 / name).resolve()
        if not path.is_relative_to(k1):
            raise ValueError("Source path escapes the K1 run")
        if name not in contents:
            contents[name] = path.read_bytes()
        data = contents[name]
        digest = _digest(data)
        if expected is not None and digest != expected:
            raise ValueError("K1 source file hash differs: " + name)
        files[name] = dict(path=str(path), bytes=len(data), sha256=digest)
        return data

    launch = json.loads(read("launch_receipt.json"))
    seal_bytes = read("input_manifest.json", launch["input_manifest_sha256"])
    seal = json.loads(seal_bytes)
    read("manifest.json", launch["source_manifest_sha256"])
    read("config.json", launch["config_sha256"])
    if (
        seal["source_manifest_sha256"] != launch["source_manifest_sha256"]
        or seal["config_sha256"] != launch["config_sha256"]
    ):
        raise ValueError("K1 input seal differs from launch source/config bindings")
    inventory_bytes = read("inputs/inventory.json", seal["files"]["inputs/inventory.json"])
    inventory = json.loads(inventory_bytes)
    dataset_name = "inputs/" + inventory["datasets"]["l1_training"]
    dataset = _load_arrays(
        read(dataset_name, seal["files"][dataset_name]),
        inventory["dataset_bindings"]["l1_training"],
    )
    if set(dataset) != set(FIELDS):
        raise ValueError("Unexpected L1 source array roster")
    _validate_pairs(dataset, 6096)
    if any(dataset[key].shape != (6096,) for key in ("row_episode", "row_steps", "row_kind")):
        raise ValueError("Unaligned source row provenance")
    if dataset["row_steps"].dtype.kind not in "iu":
        raise ValueError("Integer source row steps required")
    excluded = dataset["row_kind"][4000:4096]
    if (
        np.count_nonzero(excluded == "hypothetical_perturbed_resolve") != 64
        or np.count_nonzero(excluded == "executed_recovery") != 32
    ):
        raise ValueError("The excluded 96 rows must be hypothetical/recovery data")
    wanted = TRAIN_TAGS + VALIDATION_TAGS
    descriptors = {
        tag: [entry for entry in inventory["trajectories"] if entry["id"] == tag] for tag in wanted
    }
    if any(len(matches) != 1 for matches in descriptors.values()):
        raise ValueError("Exactly one source descriptor per selected teacher required")
    teachers = {tag: matches[0] for tag, matches in descriptors.items()}
    seeds, clusters = set(), set()
    for index, tag in enumerate(wanted):
        teacher = teachers[tag]
        discovery = index < 4
        start = index * 1000 if discovery else 4096 + (index - 4) * 1000
        selected = slice(start, start + 1000)
        kind = "executed_discovery" if discovery else "executed_added_independent"
        expected_seed = 811000000 + index if discovery else 943000000 + index - 4
        expected_group = "original_discovery" if discovery else "later_independent"
        ancestry = teacher["ancestry"]
        seed, cluster = teacher["seed"], ancestry["reset_cluster"]
        if (
            seed != expected_seed
            or ancestry["reset_seed"] != seed
            or cluster != f"native-reset-{seed}"
            or seed in seeds
            or cluster in clusters
        ):
            raise ValueError("Distinct source-bound teacher reset ancestry required")
        seeds.add(seed)
        clusters.add(cluster)
        outcome, controller = teacher["outcome"], teacher["controller"]
        if (
            teacher["group"] != expected_group
            or teacher["steps"] != 1000
            or not all(outcome[key] is True for key in ("complete", "success", "survived"))
            or not np.isfinite([outcome["cost"], outcome["mae"]]).all()
            or outcome["cost"] < 0
            or not 0 <= outcome["mae"] <= 0.3
            or controller["rho"] != 0
            or controller["effective_guidance_enabled"] is not False
            or controller["H"] != 48
            or controller["K"] != 256
            or controller["eta_u"] != 0.01
        ):
            raise ValueError("Selected teacher is not a complete successful independent episode")
        if (
            not np.all(dataset["row_episode"][selected] == tag)
            or not np.all(dataset["row_kind"][selected] == kind)
            or not np.array_equal(dataset["row_steps"][selected], np.arange(1000))
        ):
            raise ValueError("Exact episode/kind/step source row lineage changed")
        teacher_name = "inputs/" + teacher["file"]
        saved = _load_arrays(read(teacher_name, seal["files"][teacher_name]), teacher["extracted"])
        _validate_pairs(saved, 1000)
        if not np.array_equal(saved["row_steps"], np.arange(1000)):
            raise ValueError("Teacher row steps are not pre-action 0..999")
        for key in ("states", "observations", "actions"):
            if not np.array_equal(dataset[key][selected], saved[key]):
                raise ValueError(
                    "Dataset target lineage differs from executed teacher: " + tag + "/" + key
                )
        for key in ("observations", "actions"):
            if (
                _array_binding(saved[key])["sha256"]
                != teacher["source"]["raw"]["arrays"][key]["sha256"]
            ):
                raise ValueError("Teacher target differs from original executed descriptor")
    values = {}
    for split, selected in (("train", slice(0, 4000)), ("validation", slice(4096, 6096))):
        values[split] = {
            ("row_step" if key == "row_steps" else key): np.array(dataset[key][selected])
            for key in FIELDS
        }
    observations = values["train"]["observations"]
    normalization = dict(
        mean=observations.mean(axis=0).astype(np.float32),
        std=np.maximum(observations.std(axis=0), 0.05).astype(np.float32),
    )
    if not all(np.isfinite(value).all() for value in normalization.values()) or np.any(
        normalization["std"] <= 0
    ):
        raise ValueError("Invalid training-only normalization")
    overlap = _check_overlap(values["train"], values["validation"], normalization)
    destination.mkdir(parents=True, exist_ok=False)
    splits = {}
    for name, tags, indices in (
        ("train", TRAIN_TAGS, [0, 4000]),
        ("validation", VALIDATION_TAGS, [4096, 6096]),
    ):
        saved = _save_arrays(destination / f"{name}.npz", values[name])
        outcomes = {tag: teachers[tag]["outcome"] for tag in tags}
        saved.update(
            rows=len(values[name]["states"]),
            indices=dict(
                start=indices[0], stop=indices[1], convention="half-open source dataset indices"
            ),
            teachers=list(tags),
            outcomes=outcomes,
            teacher_mean_cost=float(np.mean([row["cost"] for row in outcomes.values()])),
            teacher_mean_mae=float(np.mean([row["mae"] for row in outcomes.values()])),
            fit_data=dict(
                observations_sha256=_fit_hash(values[name]["observations"]),
                actions_sha256=_fit_hash(values[name]["actions"]),
                rows=len(values[name]["states"]),
            ),
        )
        splits[name] = saved
    normalization_binding = _save_arrays(destination / "normalization.npz", normalization)
    normalization_binding.update(
        training_only=True,
        rows=4000,
        rule="float64 train mean -> float32; maximum(float64 train population std,0.05) -> float32",
    )
    for filename, data in (
        ("k1_inventory.json", inventory_bytes),
        ("k1_input_manifest.json", seal_bytes),
    ):
        with (destination / filename).open("xb") as stream:
            stream.write(data)
    provenance = dict(
        schema="walker-width-w1-executed-benchmark-v1",
        source=dict(
            run=str(k1),
            k1_input_manifest_sha256=_digest(seal_bytes),
            k1_inventory_sha256=_digest(inventory_bytes),
            files=files,
            verification_scope="Selected small K1 inputs only; unrelated input seal entries and historical raw/ledgers are not reread.",
        ),
        splits=splits,
        normalization=normalization_binding,
        teachers=teachers,
        excluded_rows=dict(start=4000, stop=4096, hypothetical=64, recovery=32),
        overlap_checks=overlap,
        reset_seeds=sorted(seeds),
        missing_evidence=[
            "Original discovery has 1000 pre-action physical states but no local terminal endpoint, warmstarts or integrations; only 999 adjacent poststates can be derived without reconstruction.",
            "Later independent episodes retain terminal_state, warmstarts/integrations and their terminal values, costs and velocities in the source teacher NPZs; these are bound here, not duplicated in train/validation targets.",
            "Validation is held out of W1 fitting, but is historical development evidence, not a newly blinded population sample.",
            "Recorded-teacher versus fresh-reset student performance gaps are descriptive and unpaired unless reset contexts are explicitly matched.",
        ],
        hash_contracts=dict(
            file="SHA256 of exact file bytes",
            arrays="SHA256 of original contiguous array bytes; shape and dtype are separate descriptors",
            fit_data="Cast matrix to contiguous float32; SHA256(default JSON [shape,dtype] followed by array bytes)",
        ),
        work=dict(new_forward_rows=0, new_controls=0, new_training_updates=0),
    )
    write(destination / "provenance.json", provenance)
    return provenance
