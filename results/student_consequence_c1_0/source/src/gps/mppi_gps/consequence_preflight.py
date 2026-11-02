"""Saved-only C1 input, seed and capacity gates. No simulator imports."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import sha, write


def verify_prepared(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "prepared_input_manifest.json").read_text())
    for row in manifest["files"]:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("Missing/escaped C1 prepared input")
        if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
            raise ValueError("C1 prepared input changed: " + row["file"])
    catalog = json.loads((root / "catalog.json").read_text())
    if catalog["status"] != "complete" or len(catalog["episodes"]) != 12:
        raise ValueError("Incomplete C1 source preparation")
    return catalog


def seed_gate(
    workspace: Path,
    seeds: dict[str, list[int]],
    *,
    max_read_bytes: int = 64 * 1024**2,
    excluded_run_id: str = "2026-11-02_student_consequence_c1_001",
) -> dict[str, Any]:
    """Review all existing run configurations, not bulk arrays or ledger guesses.

    This is a metadata freshness check. It cannot establish absence from an
    unrecorded historical execution; that limit is retained in the receipt.
    """
    requested = [value for panel in seeds.values() for value in panel]
    if len(requested) != len(set(requested)):
        raise ValueError("C1 seed panels overlap")
    wanted = set(requested)
    files, collisions, read = [], [], 0

    def visit(value: Any, location: str) -> None:
        if type(value) is int and value in wanted:
            collisions.append(dict(location=location, seed=value))
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, location + "/" + str(key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, location + "/" + str(index))

    for path in sorted((workspace / "runs").glob("*/config.json")):
        if path.parent.name == excluded_run_id:
            continue
        size = path.stat().st_size
        if read + size > max_read_bytes:
            raise ValueError("C1 seed metadata read reservation exceeded")
        payload = path.read_bytes()
        read += len(payload)
        files.append(
            dict(file=str(path), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        )
        visit(json.loads(payload), str(path))
    if not files:
        raise ValueError("No historical run metadata was available for C1 seed review")
    result = dict(
        passed=not collisions,
        seeds=seeds,
        metadata_files=files,
        collisions=collisions,
        input_read_bytes=read,
        scope="Every top-level existing run config.json; no raw ledger scan; unrecorded executions cannot be excluded",
    )
    if collisions:
        error = ValueError("C1 proposed seeds occur in existing run metadata; stop/version")
        error.seed_receipt = result
        raise error
    return result


def arithmetic_gate(prepared: Path) -> dict[str, Any]:
    """Prove saved source reductions and unchanged historical input bindings."""
    catalog = verify_prepared(prepared)
    sources = json.loads((prepared / "metadata/source_bindings.json").read_text())
    with np.load(prepared / "inputs/train.npz", allow_pickle=False) as data:
        observations = data["observations"].copy()
    with np.load(prepared / "inputs/normalization.npz", allow_pickle=False) as norm:
        mean, std = norm["mean"].copy(), norm["std"].copy()
    if not np.array_equal(observations.mean(axis=0, dtype=np.float64).astype(np.float32), mean):
        raise ValueError("C1 inherited W1 mean cannot be reconstructed")
    expected_std = np.maximum(observations.std(axis=0, dtype=np.float64), 0.05).astype(np.float32)
    if not np.array_equal(expected_std, std):
        raise ValueError("C1 inherited W1 normalization rule differs")
    metrics = {}
    for tag, descriptor in catalog["episodes"].items():
        stream = str(descriptor["stream"])
        source = next(
            v for v in sources["source_episodes_in_fixed_training_order"] if v["tag"] == tag
        )
        with np.load(prepared / descriptor["file"], allow_pickle=False) as arrays:
            pred, target = arrays["policy_actions"], arrays["actions"]
            if pred.dtype != np.float32 or pred.shape != (1000, 6):
                raise ValueError("C1 saved batch geometry changed")
            native = np.sum((pred.astype(np.float64) - target) ** 2, axis=1)
            if not np.array_equal(native, arrays["squared_disagreement"]):
                raise ValueError("C1 native-target disagreement reduction differs")
            error = pred.astype(np.float64) - target.astype(np.float32).astype(np.float64)
            metrics[tag] = float(np.sum(error**2, dtype=np.float64) / error.size)
        if (
            source["exact_teacher_agreement_complete_event"]["actor_hash"]
            != catalog["parents"][stream]["actor_hash"]
        ):
            raise ValueError("C1 saved prediction parent ownership differs")
    historical = []
    for row in sources["reused_incoming_historical_metric_events"]:
        event = row["exact_saved_event"]
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if hashlib.sha256(payload).hexdigest() != row["canonical_event_sha256"]:
            raise ValueError("C1 historical metric event binding differs")
        stream = event["tag"].split("_")[0][1:]
        if (
            event["actor_hash"] != catalog["parents"][stream]["actor_hash"]
            or event["batch_size"] != 1024
        ):
            raise ValueError("C1 historical metric actor/batch binding differs")
        historical.append(event)
    if len(historical) != 6:
        raise ValueError("C1 needs six audited historical metric events")
    return dict(
        passed=True,
        own_corpus_episode_fitting_mse=metrics,
        historical_metrics=historical,
        reused_source_rows=12000,
        reused_historical_rows=18000,
        new_network_rows=0,
        historical_scope="Exact actor/input/batch events inherited from completed full H5 audit; no replacement inference",
    )


def capacity_gate(path: Path, *, required_bytes: int = 16 * 1024**3) -> dict[str, Any]:
    free = shutil.disk_usage(path).free
    if free < required_bytes + 64 * 1024**3:
        raise ValueError("C1 output envelope plus64GiB headroom unavailable")
    memory_available = None
    if Path("/proc/meminfo").is_file():
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        memory_available = int(fields["MemAvailable"].split()[0]) * 1024
        if memory_available < 16 * 1024**3:
            raise ValueError("C1 declared worker resident-memory capacity unavailable")
    return dict(
        passed=True,
        free_bytes=free,
        study_output_ceiling_bytes=required_bytes,
        required_extra_free_headroom_bytes=64 * 1024**3,
        checked_unix=time.time(),
        memory_available_bytes=memory_available,
    )


def save_prerequisites(
    path: Path, *, seed: dict, capacity: dict, arithmetic: dict, continuation: dict
) -> None:
    write(
        path,
        dict(
            schema="walker-c1-prelaunch-prerequisites-v1",
            passed=True,
            seed=seed,
            capacity=capacity,
            arithmetic=arithmetic,
            continuation=continuation,
            replay="Paid prelude; must pass all12 sources before main stage",
        ),
    )


def complete_prerequisites(prepared: Path, recipe: Path, out: Path) -> dict[str, Any]:
    """Finish bounded saved-only gates before the replay/main worker exists."""
    from .consequence_learning import validate_parent_checkpoint

    started = time.monotonic()
    config = json.loads(recipe.read_text())
    catalog = json.loads((prepared / "catalog.json").read_text())
    manifest = json.loads((prepared / "prepared_input_manifest.json").read_text())
    seed = seed_gate(Path(config["locations"]["remote_workspace"]), config["seeds"])
    # Two full checksum passes, partial NPZ reductions and three checkpoint
    # byte decodes are covered conservatively by four complete prepared reads
    # plus32MiB of headers/JSON/seed inventory. Python shared-library/code IO is
    # engineering overhead, outside this selected scientific-input byte census.
    upper = 4 * sum(v["bytes"] for v in manifest["files"]) + 32 * 1024**2 + seed["input_read_bytes"]
    if upper > config["preparation_reservations"]["remaining_metadata_read_bytes"]:
        raise ValueError(
            "C1 continuation/seed arithmetic cannot fit its preparation read reservation"
        )
    arithmetic = arithmetic_gate(prepared)
    normalization = {}
    with np.load(prepared / "inputs/normalization.npz", allow_pickle=False) as values:
        normalization = {key: values[key].copy() for key in values.files}
    continuation = {}
    for stream in range(3):
        descriptor = catalog["parents"][str(stream)]
        rng = np.load(prepared / descriptor["global_torch_rng"]["file"], allow_pickle=False)
        continuation[str(stream)] = validate_parent_checkpoint(
            prepared / descriptor["file"],
            descriptor,
            normalization,
            rng,
            expected_global_seed=config["inherited_global_torch_seed"],
        )
    capacity = capacity_gate(prepared)
    io = dict(line.split(":", 1) for line in Path("/proc/self/io").read_text().splitlines())
    if not {
        "rchar",
        "wchar",
        "syscr",
        "syscw",
        "read_bytes",
        "write_bytes",
        "cancelled_write_bytes",
    } <= set(io):
        raise ValueError("C1 process IO accounting prerequisite unavailable")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if peak > config["limits"]["prepare_peak_rss_bytes"]:
        raise ValueError("C1 saved-only preparation exceeded its resident-memory limit")
    save_prerequisites(
        out,
        seed=seed,
        capacity=capacity,
        arithmetic=arithmetic,
        continuation=dict(
            passed=True,
            parents=continuation,
            checkpoint_metadata_decodes=3,
            selected_metadata_read_upper_bytes=upper,
            peak_rss_bytes=peak,
            wall_seconds=time.monotonic() - started,
            new_network_rows=0,
            parent_refits=0,
        ),
    )
    return json.loads(out.read_text())
