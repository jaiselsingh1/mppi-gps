"""Portable saved evidence for the D1/D2 transfer diagnostic.

Preparation validates and copies existing arrays and checkpoint bytes. It does
not import Torch, construct a simulator, recompute predictions or infer absent
integration context. File and raw-array SHA-256 contracts remain distinct.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import sha, write

SCHEMA = "walker-transfer-inputs-d1d2-v1"
TEACHER_IDS = ("v2_r0_trial_raw_0", "v2_r0_trial_raw_1")
EXPECTED_COUNTS = {
    "training_rows": 4000,
    "validation_rows": 2000,
    "k1_rows": 30070,
    "w2_episode_rows": 33631,
    "w2_episodes": 192,
    "w2_points": 6,
    "k1_policies": 9,
    "teachers": 2,
}


def array_binding(value: np.ndarray) -> dict[str, Any]:
    """Describe raw array bytes without reusing a checkpoint/fitting hash."""
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
    }


def _safe(root: Path, relative: str) -> Path:
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Portable evidence paths must remain under their owner")
    return path


def _identifier(value: str) -> str:
    if not value or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value
    ):
        raise ValueError("Invalid portable evidence identifier")
    return value


def _npz(path: Path, values: dict[str, np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError("Object/pickle evidence arrays are prohibited")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)
    return {
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "arrays": {key: array_binding(value) for key, value in values.items()},
    }


def _copy(source: Path, destination: Path, expected: str) -> dict[str, Any]:
    if sha(source) != expected:
        raise ValueError(f"Saved file hash changed: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copy2(source, destination)
    if sha(destination) != expected:
        raise ValueError("Copied saved file differs")
    binding = {"sha256": expected, "bytes": destination.stat().st_size}
    if destination.suffix == ".npz":
        with np.load(destination, allow_pickle=False) as stream:
            binding["arrays"] = {key: array_binding(stream[key]) for key in stream.files}
    return binding


def load_arrays(
    inputs: Path, relative: str, inventory: dict[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Load one portable NPZ and validate file and every raw-array binding."""
    inputs = Path(inputs)
    if inventory is None:
        inventory = json.loads((inputs / "inventory.json").read_text())
    binding = inventory["files"][relative]
    path = _safe(inputs, relative)
    if sha(path) != binding["sha256"] or path.stat().st_size != binding["bytes"]:
        raise ValueError("Portable input file differs: " + relative)
    with np.load(path, allow_pickle=False) as stream:
        result = {key: np.array(stream[key]) for key in stream.files}
    if set(result) != set(binding["arrays"]):
        raise ValueError("Portable input array roster changed")
    for key, value in result.items():
        if array_binding(value) != binding["arrays"][key]:
            raise ValueError("Portable raw-array binding changed: " + key)
    return result


def _read_raw(stream: Any, reference: dict[str, Any]) -> dict[str, np.ndarray]:
    values = {}
    for key, binding in reference["arrays"].items():
        dataset = stream[reference["group"]][key]
        value = np.asarray(dataset[()])
        dtype = np.dtype(binding["dtype"])
        if dtype.kind == "U" and value.dtype.kind == "S":
            value = np.char.decode(value, "utf-8").astype(dtype)
        else:
            value = value.astype(dtype, copy=False)
        actual = array_binding(value)
        if actual["shape"] != binding["shape"] or actual["sha256"] != binding["sha256"]:
            raise ValueError("Historical raw array changed: " + key)
        values[key] = value
    return values


def _physical(values: dict[str, np.ndarray], rows: int) -> None:
    states = values["states"][:rows]
    observations, actions = values["observations"], values["actions"]
    if states.shape != (rows, 19) or observations.shape != (rows, 17) or actions.shape != (rows, 6):
        raise ValueError("Saved physical row dimensions differ")
    if not all(np.isfinite(value).all() for value in (states, observations, actions)):
        raise ValueError("Nonfinite saved physical rows")
    native = np.concatenate((states[:, 2:10], np.clip(states[:, 10:19], -10, 10)), axis=1)
    if not np.array_equal(native, observations):
        raise ValueError("Saved observations do not match pre-action physical rows")
    if np.any(np.abs(actions) > 1):
        raise ValueError("Saved torques exceed their original bounds")


def _source(
    root: Path, out: Path, label: str, files: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = json.loads((root / "result.json").read_text())
    seal = json.loads((root / "input_manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    expected_config = result.get("config_sha256", result.get("protocol_sha256"))
    if sha(root / "config.json") != expected_config or expected_config != seal["config_sha256"]:
        raise ValueError("Historical configuration seal differs")
    if sha(root / "input_manifest.json") != result["input_manifest_sha256"]:
        raise ValueError("Historical input seal differs")
    if not result.get("execution_complete") or result.get("error"):
        raise ValueError("Only complete closed studies are admitted")
    for name in ("result.json", "config.json", "input_manifest.json", "manifest.json"):
        relative = f"sources/{label}_{name}"
        files[relative] = _copy(root / name, out / relative, sha(root / name))
    return result, seal, config


def prepare_inputs(k1: Path, w2: Path, out: Path) -> dict[str, Any]:
    """Create verified portable D1/D2 inputs; perform no diagnostic arithmetic."""
    import h5py

    k1, w2, out = Path(k1), Path(w2), Path(out)
    if out.exists():
        raise FileExistsError("Transfer inputs are create-only")
    out.mkdir(parents=True)
    files: dict[str, Any] = {}
    k1_result, k1_seal, k1_config = _source(k1, out, "k1", files)
    w2_result, w2_seal, _ = _source(w2, out, "w2", files)
    policy_names = list(k1_config["policy_order"])
    if len(set(policy_names)) != 9 or set(policy_names) != set(k1_result["policy_hashes"]):
        raise ValueError("K1 policy roster changed")
    for name in ("train.npz", "validation.npz", "normalization.npz", "task.json"):
        files[name] = _copy(w2 / "inputs" / name, out / name, w2_seal["files"]["inputs/" + name])
    train = load_arrays(out, "train.npz", {"files": files})
    validation = load_arrays(out, "validation.npz", {"files": files})
    _physical(train, 4000)
    _physical(validation, 2000)
    policies = {}
    for name in policy_names:
        relative = f"checkpoints/k1_{_identifier(name)}.pt"
        source = k1 / "inputs/checkpoints" / (name + ".pt")
        files[relative] = _copy(
            source, out / relative, k1_seal["files"][f"inputs/checkpoints/{name}.pt"]
        )
        policies[name] = {
            "policy_hash": k1_result["policy_hashes"][name],
            "checkpoint_file": relative,
            "family": "k1",
        }
    teachers = []
    k1_trajectories = []
    k1_inventory = json.loads((k1 / "inputs/inventory.json").read_text())
    if sha(k1 / "inputs/inventory.json") != k1_seal["files"]["inputs/inventory.json"]:
        raise ValueError("K1 source inventory changed")
    k1_episodes = {row["tag"]: row for row in k1_result["stages"]["B"]["episodes"]}
    with (
        h5py.File(k1 / "evidence/stage_a/raw.h5", "r") as stage_a_raw,
        h5py.File(k1 / "evidence/stage_b/raw.h5", "r") as stage_b_raw,
    ):
        entries = [
            (entry, stream)
            for stage, stream in (("A", stage_a_raw), ("B", stage_b_raw))
            for entry in k1_result["stages"][stage]["trajectories"]
        ]
        for entry, raw in entries:
            metadata = entry["trajectory"]
            identifier = _identifier(metadata["id"])
            predictions = _read_raw(raw, entry["raw"])
            if list(entry["policies"]) != policy_names:
                raise ValueError("Saved K1 prediction order differs")
            prediction = predictions["predictions"]
            n = metadata["steps"]
            if (
                prediction.shape != (n, 9, 6)
                or not np.isfinite(prediction).all()
                or np.any(np.abs(prediction) > 1)
            ):
                raise ValueError("Invalid saved K1 prediction matrix")
            relative = f"k1/trajectories/{identifier}.npz"
            if "file" in metadata:
                source_relative = "inputs/" + metadata["file"]
                files[relative] = _copy(
                    k1 / source_relative, out / relative, k1_seal["files"][source_relative]
                )
                values = load_arrays(out, relative, {"files": files})
                outcome = metadata["outcome"]
            else:
                episode = k1_episodes[identifier]
                values = _read_raw(raw, episode["raw"])
                files[relative] = _npz(out / relative, values)
                outcome = {key: value for key, value in episode.items() if key != "raw"}
            _physical(values, n)
            if not np.array_equal(
                values["observations"], predictions["observations"]
            ) or not np.array_equal(values["actions"], predictions["executed_actions"]):
                raise ValueError("K1 prediction/physical ownership join failed")
            prediction_file = f"k1/predictions/{identifier}.npz"
            files[prediction_file] = _npz(
                out / prediction_file,
                {"predictions": prediction, "policy_names": np.asarray(policy_names)},
            )
            k1_trajectories.append(
                {
                    "id": identifier,
                    "group": metadata["group"],
                    "rows": n,
                    "file": relative,
                    "predictions_file": prediction_file,
                    "outcome": outcome,
                    "source_mse": {
                        name: metric["mse"] for name, metric in entry["policies"].items()
                    },
                }
            )
            if identifier in TEACHER_IDS:
                teacher_file = f"teachers/{identifier}.npz"
                files[teacher_file] = _copy(
                    out / relative, out / teacher_file, files[relative]["sha256"]
                )
                mask = validation["row_episode"] == identifier
                if mask.sum() != n or not np.array_equal(
                    validation["row_step"][mask], np.arange(n)
                ):
                    raise ValueError("Held teacher control-index ownership differs")
                for key in ("states", "observations", "actions"):
                    if not np.array_equal(validation[key][mask], values[key]):
                        raise ValueError("Held teacher physical row ownership differs: " + key)
                teachers.append(
                    {
                        "id": identifier,
                        "file": teacher_file,
                        "validation_episode": identifier,
                        "rows": n,
                        "outcome": outcome,
                        "k1_predictions_file": prediction_file,
                    }
                )
    w2_points, w2_episodes = [], []
    with h5py.File(w2 / "evidence/raw.h5", "r") as raw:
        for point_id, point in w2_result["points"].items():
            if point["update"] not in (2000, 10000):
                continue
            relative = f"checkpoints/w2_{_identifier(point_id)}.pt"
            files[relative] = _copy(
                w2 / point["checkpoint"]["file"], out / relative, point["checkpoint"]["sha256"]
            )
            predictions = _read_raw(raw, point["predictions"])
            if predictions["train_predictions"].shape != (4000, 6) or predictions[
                "validation_predictions"
            ].shape != (2000, 6):
                raise ValueError("W2 reconstruction prediction row counts differ")
            prediction_file = f"w2/predictions/{point_id}.npz"
            files[prediction_file] = _npz(out / prediction_file, predictions)
            w2_points.append(
                {
                    "id": point_id,
                    "pair": point["pair"],
                    "update": point["update"],
                    "policy_hash": point["checkpoint"]["policy_hash"],
                    "checkpoint_file": relative,
                    "predictions_file": prediction_file,
                }
            )
            policies[point_id] = {
                "policy_hash": point["checkpoint"]["policy_hash"],
                "checkpoint_file": relative,
                "family": "w2",
            }
            for panel in ("primary", "confirmation"):
                for episode in w2_result["evaluations"][panel][point_id]["episodes"]:
                    values = _read_raw(raw, episode["raw"])
                    _physical(values, episode["steps"])
                    if (
                        values["states"].shape != (episode["steps"] + 1, 19)
                        or values["warmstarts"].shape != (episode["steps"] + 1, 9)
                        or values["integrations"].shape != (episode["steps"] + 1, 91)
                    ):
                        raise ValueError("W2 endpoint integration context is incomplete")
                    relative = f"w2/episodes/{_identifier(episode['tag'])}.npz"
                    files[relative] = _npz(out / relative, values)
                    w2_episodes.append(
                        {
                            "tag": episode["tag"],
                            "point": point_id,
                            "pair": point["pair"],
                            "update": point["update"],
                            "panel": panel,
                            "seed": episode["seed"],
                            "steps": episode["steps"],
                            "file": relative,
                            "outcome": {
                                key: value for key, value in episode.items() if key != "raw"
                            },
                        }
                    )
    counts = {
        "training_rows": len(train["observations"]),
        "validation_rows": len(validation["observations"]),
        "k1_rows": sum(row["rows"] for row in k1_trajectories),
        "w2_episode_rows": sum(row["steps"] for row in w2_episodes),
        "w2_episodes": len(w2_episodes),
        "w2_points": len(w2_points),
        "k1_policies": len(policy_names),
        "teachers": len(teachers),
    }
    if counts != EXPECTED_COUNTS or {row["id"] for row in teachers} != set(TEACHER_IDS):
        raise ValueError("Transfer source roster differs from frozen scope: " + str(counts))
    inventory = {
        "schema": SCHEMA,
        "counts": counts,
        "k1_policy_names": policy_names,
        "policies": policies,
        "k1_trajectories": k1_trajectories,
        "w2_episodes": w2_episodes,
        "w2_points": w2_points,
        "teachers": sorted(teachers, key=lambda row: row["validation_episode"]),
        "files": files,
        "sources": {"k1": str(k1.resolve()), "w2": str(w2.resolve())},
        "missing_evidence": k1_inventory["missing_evidence"],
        "new_forward_rows": 0,
        "new_physics_controls": 0,
        "new_optimizer_steps": 0,
    }
    write(out / "inventory.json", inventory)
    return inventory


def load_fork_inputs(inputs: Path, random_seed: int = 2026111001) -> list[dict[str, Any]]:
    """Build twenty fixed first-action substitutions entirely from saved rows."""
    if type(random_seed) is not int or random_seed < 0:
        raise ValueError("An explicit nonnegative private random seed is required")
    inputs = Path(inputs)
    inventory = json.loads((inputs / "inventory.json").read_text())
    if inventory["schema"] != SCHEMA:
        raise ValueError("Unexpected transfer input schema")
    validation = load_arrays(inputs, "validation.npz", inventory)
    names = inventory["k1_policy_names"]
    rng = np.random.Generator(np.random.PCG64(random_seed))
    points = sorted(inventory["w2_points"], key=lambda row: (row["pair"], row["update"]))
    point_predictions = {
        row["id"]: load_arrays(inputs, row["predictions_file"], inventory)["validation_predictions"]
        for row in points
    }
    anchors = []

    def provenance(relative: str, array: str, row: int, **extra: Any) -> dict[str, Any]:
        binding = inventory["files"][relative]
        return {
            "file": relative,
            "array": array,
            "row": row,
            "file_sha256": binding["sha256"],
            "array_sha256": binding["arrays"][array]["sha256"],
            **extra,
        }

    for teacher in sorted(inventory["teachers"], key=lambda row: row["validation_episode"]):
        values = load_arrays(inputs, teacher["file"], inventory)
        prediction = load_arrays(inputs, teacher["k1_predictions_file"], inventory)["predictions"]
        n = teacher["rows"]
        full = {}
        for key, terminal in (
            ("states", "terminal_state"),
            ("warmstarts", "terminal_warmstarts"),
            ("integrations", "terminal_integrations"),
        ):
            if key not in values or terminal not in values:
                raise ValueError("Teacher fork requires original complete context")
            full[key] = np.concatenate((values[key], values[terminal][None]), axis=0)
            width = {"states": 19, "warmstarts": 9, "integrations": 91}[key]
            if full[key].shape != (n + 1, width) or not np.isfinite(full[key]).all():
                raise ValueError("Teacher fork context dimensions differ")
        rows = np.flatnonzero(validation["row_episode"] == teacher["validation_episode"])
        if (
            len(rows) != n
            or not np.array_equal(validation["row_step"][rows], np.arange(n))
            or not np.array_equal(validation["observations"][rows], values["observations"])
        ):
            raise ValueError("Fork prediction observation join changed")
        for step in range(0, 1000, 100):
            branches = [
                {
                    "name": "teacher",
                    "action": values["actions"][step].copy(),
                    "provenance": provenance(teacher["file"], "actions", step),
                }
            ]
            for name in ("reference", "untrained_0", "untrained_1", "untrained_2"):
                branches.append(
                    {
                        "name": name,
                        "action": prediction[step, names.index(name)].copy(),
                        "policy_hash": inventory["policies"][name]["policy_hash"],
                        "provenance": provenance(
                            teacher["k1_predictions_file"],
                            "predictions",
                            step,
                            policy_column=names.index(name),
                            checkpoint_file=inventory["policies"][name]["checkpoint_file"],
                        ),
                    }
                )
            for point in points:
                branches.append(
                    {
                        "name": point["id"],
                        "action": point_predictions[point["id"]][rows[step]].copy(),
                        "policy_hash": point["policy_hash"],
                        "provenance": provenance(
                            point["predictions_file"],
                            "validation_predictions",
                            int(rows[step]),
                            checkpoint_file=point["checkpoint_file"],
                        ),
                    }
                )
            branches.append(
                {
                    "name": "uniform_random",
                    "action": rng.uniform(-1.0, 1.0, size=6),
                    "provenance": {
                        "generator": "PCG64",
                        "seed": random_seed,
                        "distribution": "independent uniform[-1,1]",
                        "draw_index": len(anchors),
                    },
                }
            )
            if len(branches) != 12 or any(
                not np.isfinite(branch["action"]).all() or np.any(np.abs(branch["action"]) > 1)
                for branch in branches
            ):
                raise ValueError("Fork branch roster or action bounds differ")
            anchors.append(
                {
                    "teacher_id": teacher["id"],
                    "step": step,
                    "observation": values["observations"][step].copy(),
                    "initial": {
                        "state": full["states"][step].copy(),
                        "warmstart": full["warmstarts"][step].copy(),
                        "integration": full["integrations"][step].copy(),
                    },
                    "expected": {
                        key: value[step : step + 101].copy() for key, value in full.items()
                    }
                    | {"costs": values["costs"][step : step + 100].copy()},
                    "teacher_actions": values["actions"][step : step + 100].copy(),
                    "branches": branches,
                }
            )
    if len(anchors) != 20:
        raise ValueError("Exactly twenty frozen anchors are required")
    return anchors
