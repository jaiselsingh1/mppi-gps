"""Copy the exact W1 executed benchmark for a separate W2 update-count study.

Only small sealed metadata and NPZ inputs are read. No experiment, policy,
optimizer or simulator is imported, and no historical raw audit is repeated.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from . import width_inputs as benchmark
from .provenance import write

EXACT_COPIES = {
    "train.npz": "inputs/train.npz",
    "validation.npz": "inputs/validation.npz",
    "normalization.npz": "inputs/normalization.npz",
    "task.json": "inputs/task.json",
    "w1_provenance.json": "inputs/provenance.json",
    "k1_inventory.json": "inputs/k1_inventory.json",
    "k1_input_manifest.json": "inputs/k1_input_manifest.json",
    "w1_config.json": "config.json",
    "w1_manifest.json": "manifest.json",
    "w1_input_manifest.json": "input_manifest.json",
    "w1_launch_receipt.json": "launch_receipt.json",
}


def copy_benchmark(w1: Path, dest: Path) -> dict[str, Any]:
    """Create exact W1 copies and an augmented W2 provenance.json, without fitting.

    w1_provenance.json preserves the original provenance bytes. The returned
    record retains its splits, teacher descriptors and K1 ancestry, and adds
    W1 source/copy bindings and independently checked reuse conditions.
    """
    w1, dest = Path(w1).resolve(), Path(dest).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    if dest.is_relative_to(w1):
        raise ValueError("W2 inputs must be outside the frozen W1 run")
    contents: dict[str, bytes] = {}
    bindings: dict[str, dict[str, Any]] = {}

    def read(name: str, expected: str | None = None) -> bytes:
        path = (w1 / name).resolve()
        if not path.is_relative_to(w1):
            raise ValueError("Source path escapes the W1 run")
        if name not in contents:
            contents[name] = path.read_bytes()
        value = contents[name]
        digest = hashlib.sha256(value).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("W1 source hash differs: " + name)
        bindings[name] = dict(path=str(path), bytes=len(value), sha256=digest)
        return value

    launch = json.loads(read("launch_receipt.json"))
    seal = json.loads(read("input_manifest.json", launch["input_manifest_sha256"]))
    manifest = json.loads(read("manifest.json", launch["source_manifest_sha256"]))
    config = json.loads(read("config.json", launch["config_sha256"]))
    if (
        seal["config_sha256"] != launch["config_sha256"]
        or seal["source_manifest_sha256"] != launch["source_manifest_sha256"]
        or manifest["config_sha256"] != launch["config_sha256"]
        or config["schema"] != "walker-student-width-w1-v1"
        or config["rows"] != 4000
        or config["validation_rows"] != 2000
        or config["normalization_floor"] != 0.05
    ):
        raise ValueError("W1 source/configuration/input-seal contract changed")
    for source in EXACT_COPIES.values():
        if source.startswith("inputs/"):
            read(source, seal["files"][source])
    original = json.loads(contents["inputs/provenance.json"])
    if original["schema"] != "walker-width-w1-executed-benchmark-v1":
        raise ValueError("Expected the executed-only W1 benchmark provenance")
    if original["excluded_rows"] != dict(start=4000, stop=4096, hypothetical=64, recovery=32):
        raise ValueError("W1 auxiliary-row exclusion contract changed")
    task = json.loads(contents["inputs/task.json"])
    if task != dict(task=config["task"], xml_sha256=config["xml_sha256"]):
        raise ValueError("W1 task/XML binding differs from its configuration")
    source = original["source"]
    if (
        bindings["inputs/k1_inventory.json"]["sha256"] != source["k1_inventory_sha256"]
        or bindings["inputs/k1_input_manifest.json"]["sha256"] != source["k1_input_manifest_sha256"]
    ):
        raise ValueError("Original K1 source-chain hashes changed")
    inventory = json.loads(contents["inputs/k1_inventory.json"])
    k1_seal = json.loads(contents["inputs/k1_input_manifest.json"])
    if k1_seal["files"]["inputs/inventory.json"] != source["k1_inventory_sha256"]:
        raise ValueError("Original K1 inventory/input-seal binding changed")
    values = {}
    seeds = set()
    expected_fields = {"states", "observations", "actions", "row_episode", "row_step", "row_kind"}
    for split, rows, tags, indices, kind in (
        ("train", 4000, benchmark.TRAIN_TAGS, (0, 4000), "executed_discovery"),
        ("validation", 2000, benchmark.VALIDATION_TAGS, (4096, 6096), "executed_added_independent"),
    ):
        description = original["splits"][split]
        values[split] = benchmark._load_arrays(contents[f"inputs/{split}.npz"], description)
        arrays = values[split]
        benchmark._validate_pairs(arrays, rows)
        if (
            set(arrays) != expected_fields
            or description["rows"] != rows
            or description["teachers"] != list(tags)
            or (description["indices"]["start"], description["indices"]["stop"]) != indices
            or arrays["row_episode"].shape != (rows,)
            or not np.array_equal(arrays["row_episode"], np.repeat(tags, 1000))
            or arrays["row_kind"].shape != (rows,)
            or not np.all(arrays["row_kind"] == kind)
            or arrays["row_step"].dtype.kind not in "iu"
            or not np.array_equal(arrays["row_step"], np.tile(np.arange(1000), len(tags)))
        ):
            raise ValueError("Exact executed row lineage changed: " + split)
        fit_data = dict(
            observations_sha256=benchmark._fit_hash(arrays["observations"]),
            actions_sha256=benchmark._fit_hash(arrays["actions"]),
            rows=rows,
        )
        if description["fit_data"] != fit_data:
            raise ValueError("W1 float32 fitting-data hashes changed")
        for index, tag in enumerate(tags):
            teacher = original["teachers"][tag]
            matches = [entry for entry in inventory["trajectories"] if entry["id"] == tag]
            expected_seed = 811000000 + index if split == "train" else 943000000 + index
            outcome = teacher["outcome"]
            if (
                matches != [teacher]
                or teacher["steps"] != 1000
                or teacher["seed"] != expected_seed
                or teacher["seed"] in seeds
                or teacher["ancestry"]["reset_seed"] != expected_seed
                or teacher["ancestry"]["reset_cluster"] != f"native-reset-{expected_seed}"
                or not all(outcome[key] is True for key in ("complete", "success", "survived"))
                or not np.isfinite([outcome["cost"], outcome["mae"]]).all()
                or outcome["cost"] < 0
                or not 0 <= outcome["mae"] <= 0.3
                or description["outcomes"][tag] != outcome
            ):
                raise ValueError("Original successful teacher/outcome ancestry changed")
            seeds.add(teacher["seed"])
            selected = slice(index * 1000, (index + 1) * 1000)
            for key in ("states", "observations", "actions"):
                if (
                    benchmark._array_binding(arrays[key][selected])
                    != teacher["extracted"]["arrays"][key]
                ):
                    raise ValueError("W1 rows differ from original teacher array binding: " + tag)
        for key, aggregate in (("cost", "teacher_mean_cost"), ("mae", "teacher_mean_mae")):
            if description[aggregate] != float(
                np.mean([original["teachers"][tag]["outcome"][key] for tag in tags])
            ):
                raise ValueError("W1 teacher outcome aggregate changed")
    normalization = benchmark._load_arrays(
        contents["inputs/normalization.npz"], original["normalization"]
    )
    observations = values["train"]["observations"]
    expected_normalization = dict(
        mean=observations.mean(axis=0).astype(np.float32),
        std=np.maximum(observations.std(axis=0), 0.05).astype(np.float32),
    )
    if (
        set(normalization) != {"mean", "std"}
        or original["normalization"]["training_only"] is not True
        or original["normalization"]["rows"] != 4000
        or any(
            normalization[key].dtype != np.dtype("float32")
            or not np.array_equal(normalization[key], expected_normalization[key])
            for key in ("mean", "std")
        )
    ):
        raise ValueError("Exact training-only normalization with0.05 floor required")
    overlap = benchmark._check_overlap(values["train"], values["validation"], normalization)
    dest.mkdir(parents=True, exist_ok=False)
    copies = {}
    for destination, name in EXACT_COPIES.items():
        with (dest / destination).open("xb") as stream:
            stream.write(contents[name])
        copies[destination] = dict(source=name, **bindings[name])
    provenance = deepcopy(original)
    provenance["reuse_schema"] = "walker-update-w2-reused-benchmark-v1"
    provenance["w1_source"] = dict(
        run=str(w1),
        files=bindings,
        exact_copies=copies,
        scope="Selected sealed W1 input NPZs and metadata only. Original array/container bytes preserved; no historical raw/ledger or checkpoint read, actor prediction, optimizer operation or simulator reconstruction.",
    )
    provenance["reuse_overlap_checks"] = overlap
    provenance["work"] = dict(new_forward_rows=0, new_controls=0, new_training_updates=0)
    write(dest / "provenance.json", provenance)
    return provenance
