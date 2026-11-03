"""Saved-only joins between a prototype sample bank and its recorded baseline.

This establishes byte agreements within the retained precision and evidence.
It neither repeats physics nor independently verifies reported survival.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import Any

MAX_UNCOMPRESSED_BYTES = 512 * 1024**2
MAX_EPISODES = 100
MAX_JSON_BYTES = 4 * 1024**2


def _binding(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Review input changed while hashing")
    return dict(path=str(path.resolve()), bytes=after.st_size, sha256=digest.hexdigest())


def _json(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Review JSON exceeds the 4 MiB limit")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Expected a saved JSON object")
    return value


def _reservation(path: Path) -> int:
    if path.stat().st_size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Compressed review input exceeds the memory reservation")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 128 or len({row.filename for row in members}) != len(members):
            raise ValueError("Oversized or duplicate NPZ member roster")
        size = sum(row.file_size for row in members)
    if size > MAX_UNCOMPRESSED_BYTES:
        raise ValueError("Uncompressed review input exceeds 512 MiB")
    return size


def verify_bank_reference(bank: Path, reference: Path, metrics: Path) -> dict[str, Any]:
    """Join every bank episode to complete saved source traces; no simulator imports."""
    import numpy as np

    bank, reference, metrics = bank.resolve(), reference.resolve(), metrics.resolve()
    reference_reservation = _reservation(reference)
    source_metrics = _json(metrics)
    inputs = dict(reference=_binding(reference), metrics=_binding(metrics), bank=[])
    with np.load(reference, allow_pickle=False) as archive:
        required = (
            "episode_step_offsets",
            "episode_state_offsets",
            "reset_seeds",
            "planner_seeds",
            "states",
            "warmstarts",
            "actions",
            "plans",
            "pre_plan_nominals",
            "initial_plan_flags",
            "costs",
        )
        source = {key: archive[key] for key in required}
    steps, states = source["episode_step_offsets"], source["episode_state_offsets"]
    count = len(steps) - 1
    if not 0 < count <= MAX_EPISODES or len(states) != count + 1:
        raise ValueError("Reference episode roster exceeds its finite limit or is incomplete")
    if any(value.dtype.kind not in "iu" or value.ndim != 1 for value in (steps, states)):
        raise ValueError("Reference offsets must be integer vectors")
    if steps[0] != 0 or states[0] != 0 or np.any(np.diff(steps) <= 0):
        raise ValueError("Invalid reference offsets")
    if not np.array_equal(np.diff(states), np.diff(steps) + 1):
        raise ValueError("Reference requires N+1 states in every episode")
    if steps[-1] != len(source["actions"]) or states[-1] != len(source["states"]):
        raise ValueError("Reference offsets do not cover the complete saved arrays")
    if len(source["reset_seeds"]) != count or len(source["planner_seeds"]) != count:
        raise ValueError("Reference seed roster differs from episode offsets")
    if len(set(source["reset_seeds"].tolist())) != count:
        raise ValueError("Reference reset seeds are ambiguous")
    for key in ("states", "warmstarts", "actions", "plans", "pre_plan_nominals", "costs"):
        if source[key].dtype.kind not in "fiu" or not np.isfinite(source[key]).all():
            raise ValueError("Reference contains nonfinite or nonnumeric recorded arrays")
    if source["states"].shape[1:] != (19,) or source["warmstarts"].shape != (
        len(source["states"]),
        9,
    ):
        raise ValueError("Reference physical state or warmstart interface differs")
    if source["actions"].shape[1:] != (6,) or source["costs"].shape != (len(source["actions"]),):
        raise ValueError("Reference control or cost interface differs")
    if (
        source["plans"].ndim != 3
        or source["plans"].shape[0] != len(source["actions"])
        or source["plans"].shape[-1] != 6
        or source["pre_plan_nominals"].shape != source["plans"].shape
    ):
        raise ValueError("Reference planner arrays differ from executed control rows")
    records = source_metrics["episodes"]
    if len(records) != count or source_metrics["run"]["requested_episodes"] != count:
        raise ValueError("Reference metrics and trace episode counts differ")
    maximum = int(source_metrics["task"]["max_steps"])
    target = float(source_metrics["task"]["target_velocity"])
    if maximum <= 0 or not np.isfinite(target):
        raise ValueError("Invalid recorded task metadata")
    paths = sorted(bank.glob("episode_*.npz"))
    if len(paths) > MAX_EPISODES:
        raise ValueError("Bank exceeds the 100-episode limit")
    summary_path = bank / "summary.json"
    summary = _json(summary_path) if summary_path.is_file() else None
    if summary is not None:
        inputs["bank_summary"] = _binding(summary_path)
    by_seed = {int(value): index for index, value in enumerate(source["reset_seeds"])}
    source_bytes = sum(value.nbytes for value in source.values())
    loaded_bytes, peak = source_bytes, reference_reservation
    results, unmatched, seen = [], [], set()

    def exact(left, right) -> bool:
        return (
            left.shape == right.shape
            and left.dtype.str == right.dtype.str
            and (np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes())
        )

    for path in paths:
        reservation = reference_reservation + _reservation(path)
        if reservation > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Reference plus one bank episode exceeds 512 MiB")
        peak = max(peak, reservation)
        inputs["bank"].append(_binding(path))
        match = re.fullmatch(r"episode_(\d+)\.npz", path.name)
        seed = int(match.group(1)) if match else None
        if seed not in by_seed or seed in seen:
            unmatched.append(path.name)
            continue
        seen.add(seed)
        with np.load(path, allow_pickle=False) as archive:
            episode = {key: archive[key] for key in archive.files}
        loaded_bytes += sum(value.nbytes for value in episode.values())
        required_bank = {
            "states",
            "warmstarts",
            "actions",
            "plan",
            "nominal_first",
            "cost",
            "survived",
            "walks",
        }
        absent = sorted(required_bank - episode.keys())
        if absent:
            results.append(
                dict(file=path.name, reset_seed=seed, passed=False, missing_required_arrays=absent)
            )
            del episode
            continue
        index = by_seed[seed]
        action_slice = slice(int(steps[index]), int(steps[index + 1]))
        state_slice = slice(int(states[index]), int(states[index + 1]))
        expected = {
            key: value[action_slice]
            for key, value in source.items()
            if key in {"actions", "plans", "pre_plan_nominals", "initial_plan_flags", "costs"}
        }
        expected_states = source["states"][state_slice]
        expected_warmstarts = source["warmstarts"][state_slice]
        controls = len(expected["actions"])
        metadata = records[index]
        initial = expected["initial_plan_flags"]
        if initial.dtype.kind != "b" or initial.shape != (controls,):
            raise ValueError("Source initial-pass flags are missing or malformed")
        total = 0.0
        for value in expected["costs"]:
            total += float(value)
        mae = float(np.mean(np.abs(expected_states[1:, 10] - target)))
        recorded_survival = metadata["survived"]
        if type(recorded_survival) is not bool:
            raise ValueError("Source survival must be an explicit recorded boolean")
        survived = recorded_survival and controls == maximum
        walking = survived and mae <= 0.3
        checks = dict(
            finite_recorded_arrays=all(
                value.dtype.kind in "fiub" and np.isfinite(value).all()
                for key, value in episode.items()
                if key in required_bank
            ),
            source_seed_and_length=(
                metadata["reset_seed"] == seed
                and metadata["planner_seed"] == int(source["planner_seeds"][index])
                and metadata["steps"] == controls
                and 0 < controls <= maximum
            ),
            pre_action_states=exact(episode["states"], expected_states[:-1]),
            pre_action_warmstarts=exact(episode["warmstarts"], expected_warmstarts[:-1]),
            executed_actions=exact(episode["actions"], expected["actions"]),
            stored_plan_precision=exact(episode["plan"], expected["plans"].astype(np.float32)),
            nominal_first_noninitial=episode["nominal_first"].shape
            == expected["pre_plan_nominals"][:, 0].shape
            and exact(
                episode["nominal_first"][~initial], expected["pre_plan_nominals"][~initial, 0]
            ),
            left_fold_executed_cost=exact(episode["cost"], np.asarray(total)),
            recorded_survival=exact(episode["survived"], np.asarray(survived)),
            walking_from_post_states_and_recorded_survival=exact(
                episode["walks"], np.asarray(walking)
            ),
        )
        if summary is not None:
            summary_rows = [row for row in summary["episodes"] if row["seed"] == seed]
            checks["bank_summary_seed_and_length"] = (
                len(summary_rows) == 1 and summary_rows[0]["controls"] == controls
            )
        missing = sorted(
            {"integrations", "planner_rng_states", "pre_plan_nominals"} - episode.keys()
        )
        prefix = episode.get("prefix")
        candidate_horizon = (
            int(prefix.shape[2]) if prefix is not None and prefix.ndim == 4 else None
        )
        results.append(
            dict(
                file=path.name,
                reset_seed=seed,
                source_planner_seed=int(source["planner_seeds"][index]),
                controls=controls,
                checks=checks,
                passed=all(checks.values()),
                nominal_first_initial_rows_excluded=int(initial.sum()),
                recorded_source_survived=recorded_survival,
                survival_independently_verified=False,
                rederived_walking=walking,
                velocity_mae=mae,
                left_fold_cost=total,
                missing_full_replay_context=missing,
                stored_candidate_horizon=candidate_horizon,
                full_planning_horizon=int(expected["plans"].shape[1]),
                stored_candidate_dtype=None if prefix is None else prefix.dtype.str,
                missing_candidate_evidence=[
                    "all optimization passes",
                    "original full-precision full-horizon candidate sequences",
                ],
            )
        )
        del episode, prefix
    missing_seeds = sorted(set(by_seed) - seen)
    return dict(
        schema="walker-bank-reference-join-v1",
        passed=(
            len(results) == count
            and not unmatched
            and not missing_seeds
            and source_metrics["run"].get("complete") is True
            and all(row["passed"] for row in results)
        ),
        scientific_audit=False,
        survival_independently_verified=False,
        scope="Exact saved-file joins at stored precision, conditional on recorded source survival; no replay, search, inference or learning.",
        expected_episodes=count,
        recorded_task_horizon=maximum,
        target_velocity=target,
        walking_velocity_mae_max=0.3,
        checked_episodes=len(results),
        missing_bank_seeds=missing_seeds,
        unexpected_bank_files=unmatched,
        bank_summary_available=summary is not None,
        bank_seed_authority="Episode filename and optional summary, joined to source reset/planner metadata; bank planner RNG is not stored.",
        inputs=inputs,
        episodes=results,
        work=dict(
            hash_read_bytes=sum(
                value["bytes"]
                for value in [
                    inputs["reference"],
                    inputs["metrics"],
                    *inputs["bank"],
                    *([inputs["bank_summary"]] if "bank_summary" in inputs else []),
                ]
            ),
            logical_array_bytes_loaded=loaded_bytes,
            maximum_uncompressed_reservation_bytes=peak,
            physical_disk_io_bytes=None,
            new_native_controls=0,
            new_neural_rows=0,
            new_optimizer_steps=0,
        ),
        limits=dict(
            reference_plus_one_bank_uncompressed_bytes=MAX_UNCOMPRESSED_BYTES,
            maximum_episodes=MAX_EPISODES,
            json_bytes=MAX_JSON_BYTES,
        ),
    )
