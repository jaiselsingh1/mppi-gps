"""Bounded J2 fresh-seed confirmation of an after-fitting walking diagnostic.

Preparation reads saved corpora only. Native simulation and Torch are imported
only by execution; scores do not guide fitting or execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

SCHEMA = "walker-jitter-diagnosis-j2-0"
SHARES = (0.9, 0.75, 0.5, 0.25)
KINDS = ("MPPI", "white", "slow")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def file_binding(path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed while hashing")
    return dict(bytes=after.st_size, sha256=digest.hexdigest())


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def global_numpy_state() -> dict:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return dict(
        algorithm=algorithm,
        keys=keys.tolist(),
        position=position,
        has_gauss=has_gauss,
        cached_gaussian=cached_gaussian,
    )


def condition_id(kind: str, share: float) -> str:
    return "clean" if share == 1 else f"{kind.lower()}_{str(share).replace('.', 'p')}"


def conditions() -> list[dict[str, Any]]:
    return [dict(id="clean", kind="clean", share=1.0)] + [
        dict(id=condition_id(kind, share), kind=kind, share=share)
        for kind in KINDS
        for share in SHARES
    ]


def validate_config(config: dict) -> None:
    if config["schema"] != SCHEMA:
        raise ValueError("Wrong J2 configuration schema")
    for name, count in (("training", 6), ("evaluation", 32), ("probe", 8)):
        values = config["seeds"][name]
        if (
            len(values) != count
            or len(set(values)) != count
            or any(type(value) is not int or not 0 <= value < 2**32 for value in values)
        ):
            raise ValueError("The fixed fresh-seed roster is incomplete")
    panels = [set(config["seeds"][name]) for name in ("training", "evaluation", "probe")]
    if any(panels[left] & panels[right] for left in range(3) for right in range(left)):
        raise ValueError("Seed roles must be disjoint")
    learning = config["learning"]
    if (learning["updates"], learning["batch_size"], learning["lr"]) != (10000, 256, 3e-4):
        raise ValueError("J2 retains the qualified positive-control fitting budget")
    for key, expected in dict(
        adam_betas=[0.9, 0.999],
        adam_eps=1e-8,
        gradient_clip=0.5,
        proximal_weight=0.0,
        anchor_refresh=0,
    ).items():
        if learning.get(key, expected) != expected:
            raise ValueError("J2 optimizer settings differ from the fixed recipe")
    if "prerequisites" in config:
        prerequisites = config["prerequisites"]
        clean = prerequisites["clean_children_gate"]
        if (
            clean["minimum_pooled_walking"],
            clean["pooled_attempts"],
            clean["minimum_each_stream"],
            clean["attempts_each_stream"],
        ) != (173, 192, 24, 32):
            raise ValueError("J2 clean positive-control gate differs")
        if prerequisites["reference_alone_gate"]["minimum_walking_each_panel"] != 15:
            raise ValueError("J2 reference qualification gate differs")
        replay = prerequisites["reference_replay"]
        if (replay["numeric_rtol"], replay["numeric_atol"], replay["continuation_controls"]) != (
            0.0,
            1e-10,
            25,
        ):
            raise ValueError("J2 replay tolerance or horizon differs")


def verify_launch(config: dict, prepared: Path) -> dict:
    """Check immutable source/data bindings before importing a science runtime."""
    validate_config(config)
    if config.get("launch_ready") is not True:
        raise ValueError("J2 launch prerequisites are not closed")
    manifest_path = prepared / "manifest.json"
    if file_binding(manifest_path) != config["data"]["prepared_manifest"]:
        raise ValueError("Prepared manifest differs from frozen configuration")
    source_root = Path(__file__).resolve().parents[3]
    bindings = config["source_bindings"]
    if "src/gps/mppi_gps/jitter_diagnosis.py" not in bindings:
        raise ValueError("Execution source binding missing")
    source_bytes = 0
    for name, binding in bindings.items():
        path = (source_root / name).resolve()
        if not path.is_relative_to(source_root) or file_binding(path) != binding:
            raise ValueError("Frozen execution source differs: " + name)
        source_bytes += binding["bytes"]
    manifest = json.loads(manifest_path.read_text())
    if manifest["source_inputs"] != config["inputs"]:
        raise ValueError("Prepared source inputs differ from frozen configuration")
    for name, binding in manifest["files"].items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared.resolve()) or file_binding(path) != binding:
            raise ValueError("Prepared input changed: " + name)
    declared = [
        (value["id"], value["jitter"], value["signal_share"]) for value in config["conditions"]
    ]
    expected = [(value["id"], value["kind"], value["share"]) for value in conditions()]
    if declared != expected or set(manifest["conditions"]) != {value[0] for value in expected}:
        raise ValueError("Frozen jitter-label roster differs")
    return dict(
        source_hash_read_bytes=source_bytes,
        prepared_hash_read_bytes=sum(value["bytes"] for value in manifest["files"].values()),
        prepared_manifest_read_bytes=manifest_path.stat().st_size,
        manifest=manifest,
    )


def make_labels(clean: np.ndarray, mppi_walks: np.ndarray) -> tuple[dict, dict]:
    """Recreate the original fixed jitter corpora, preserving draw/operation order."""
    from scipy import signal

    clean = np.asarray(clean, dtype=np.float64)
    mppi_walks = np.asarray(mppi_walks, dtype=np.float64)
    if clean.shape != (9000, 6) or mppi_walks.shape != (9, 1000, 6):
        raise ValueError("J2 requires the unchanged nine full reference and MPPI walks")
    if not np.isfinite(clean).all() or not np.isfinite(mppi_walks).all():
        raise ValueError("Nonfinite source labels")
    rng = np.random.default_rng(20261102)
    pooled = np.concatenate(mppi_walks)
    if (pooled.std(0) == 0).any():
        raise ValueError("Degenerate recorded MPPI jitter")
    sources = {"MPPI": [(walk - pooled.mean(0)) / pooled.std(0) for walk in mppi_walks]}
    sources["white"] = [rng.standard_normal((1000, 6)) for _ in range(9)]
    slow = [rng.standard_normal((1000, 6)) for _ in range(9)]
    sos = signal.butter(4, 2.0, fs=125.0, output="sos")
    slow = [signal.sosfiltfilt(sos, values, axis=0) for values in slow]
    pooled_slow = np.concatenate(slow)
    sources["slow"] = [values / pooled_slow.std(0) for values in slow]
    arrays, metadata = {}, {}
    for condition in conditions():
        kind, share = condition["kind"], condition["share"]
        source = sources["MPPI" if kind == "clean" else kind]
        scale = np.sqrt(clean.var(0) * (1 - share) / share) if share < 1 else 0.0
        unbounded = clean + scale * np.concatenate(source)
        values = np.clip(unbounded, -1, 1).astype(np.float32)
        arrays[condition["id"]] = values
        metadata[condition["id"]] = dict(
            **condition,
            sha256=array_hash(values),
            clipped_share=float(np.mean(np.abs(unbounded) > 1)),
        )
    return arrays, metadata


def load_reference(path: Path) -> dict[str, np.ndarray]:
    import h5py

    fields = ("states", "observations", "actions", "integrations", "warmstarts", "costs")
    result = {key: [] for key in fields}
    with h5py.File(path, "r") as handle:
        names = [
            name
            for name in sorted(handle["arrays"], key=lambda value: value.rsplit("_", 1)[-1])
            if "episode_k1_primary_reference_" in name
        ]
        if len(names) != 16:
            raise ValueError("Expected all sixteen original reference episodes")
        for name in names:
            for field in fields:
                values = handle["arrays"][name][field][:]
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite saved reference evidence")
                result[field].append(values)
    arrays = {key: np.stack(value) for key, value in result.items()}
    if arrays["actions"].shape != (16, 1000, 6):
        raise ValueError("All original reference episodes must contain 1000 controls")
    return arrays


def prepare(config: dict, out: Path) -> dict:
    """Bind old data and fixed label perturbations without constructing a simulator."""
    validate_config(config)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = {}
    for name, binding in config["inputs"].items():
        path = Path(binding["path"])
        actual = file_binding(path)
        if actual != {key: binding[key] for key in actual}:
            raise ValueError(f"Frozen input differs: {name}")
        inputs[name] = dict(path=str(path.resolve()), **actual)
    reference = load_reference(Path(config["inputs"]["reference_h5"]["path"]))
    with np.load(config["inputs"]["mppi_traces"]["path"], allow_pickle=False) as trace:
        offsets = trace["episode_step_offsets"]
        all_actions = trace["actions"]
        walks = [
            all_actions[int(start) : int(end)]
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
            if int(end - start) == 1000
        ][:9]
    if len(walks) != 9:
        raise ValueError("Missing nine original complete MPPI jitter sequences")
    observations = np.concatenate(reference["observations"][:9]).astype(np.float32)
    clean = np.concatenate(reference["actions"][:9]).astype(np.float64)
    if observations.shape != (9000, 17):
        raise ValueError("Training observations differ from the positive-control interface")
    labels, metadata = make_labels(clean, np.stack(walks))
    mean = observations.mean(0)
    std = np.maximum(observations.std(0), 1e-6)
    from .similarity_metrics import fit_state_explainer

    normalization = dict(mean=mean, std=std)
    explainer = fit_state_explainer(observations, clean, normalization, group_ids=range(9))
    values = dict(
        observations=observations,
        clean=clean,
        held_observations=np.concatenate(reference["observations"][9:]).astype(np.float32),
        held_clean=np.concatenate(reference["actions"][9:]).astype(np.float64),
        normalization_mean=mean,
        normalization_std=std,
        **{f"sac_{key}": value for key, value in reference.items()},
        **{f"labels__{key}": value for key, value in labels.items()},
    )
    np.savez_compressed(out / "corpus.npz", **values)
    shutil.copyfile(config["inputs"]["reference_checkpoint"]["path"], out / "reference.pt")
    write_json(out / "explainers.json", dict(clean=explainer))
    result = dict(
        schema=SCHEMA,
        source_inputs=inputs,
        conditions=metadata,
        unique_rows=9000,
        normalization={key: value.tolist() for key, value in normalization.items()},
        files={
            name: file_binding(out / name)
            for name in ("corpus.npz", "reference.pt", "explainers.json")
        },
        wall_seconds=time.monotonic() - started,
        reads=dict(
            source_hash_bytes=sum(value["bytes"] for value in inputs.values()),
            logical_hdf5_dataset_bytes=sum(value.nbytes for value in reference.values()),
            logical_npz_payload_bytes=offsets.nbytes + all_actions.nbytes,
            reference_copy_bytes=inputs["reference_checkpoint"]["bytes"],
            physical_disk_io_bytes=None,
        ),
        new_scientific_work=0,
    )
    write_json(out / "manifest.json", result)
    return result


class Counters:
    """Small actual-work census; failures retain charged prefixes and uncertainty."""

    def __init__(self, limits: dict, out: Path):
        self.limits, self.out = limits, out
        self.started = time.monotonic()
        self.requested: dict[str, int] = {}
        self.completed: dict[str, int] = {}
        self.unknown_partial: dict[str, int] = {}
        self.phase = "starting"

    def check(self) -> None:
        if time.monotonic() - self.started > self.limits["wall_seconds"]:
            raise RuntimeError("J2 wall budget exhausted")
        self.check_storage()

    def check_storage(self, reservation: int = 0) -> None:
        size = sum(path.stat().st_size for path in self.out.rglob("*") if path.is_file())
        if size + reservation > self.limits["output_bytes"]:
            raise RuntimeError("J2 storage budget exhausted")

    def reserve_storage(self, amount: int) -> None:
        self.check()
        self.check_storage(amount)

    def measure(self, kind: str, amount: int, operation: Callable, **_: Any) -> Any:
        counts = {kind: amount}
        if kind == "neural_rows":
            counts["network_calls"] = 1
        return self.paid(counts, operation)

    def paid(self, amounts: dict[str, int], operation: Callable) -> Any:
        if time.monotonic() - self.started > self.limits["wall_seconds"]:
            raise RuntimeError("J2 wall budget exhausted")
        for kind, amount in amounts.items():
            if type(amount) is not int or amount < 0:
                raise ValueError("Explicit nonnegative integer work required")
            if self.requested.get(kind, 0) + amount > self.limits.get(kind, math.inf):
                raise RuntimeError(f"J2 {kind} budget exhausted")
        for kind, amount in amounts.items():
            self.requested[kind] = self.requested.get(kind, 0) + amount
        try:
            result = operation()
        except BaseException:
            for kind, amount in amounts.items():
                self.unknown_partial[kind] = self.unknown_partial.get(kind, 0) + amount
            self.flush()
            raise
        for kind, amount in amounts.items():
            self.completed[kind] = self.completed.get(kind, 0) + amount
        return result

    def flush(self) -> None:
        write_json(
            self.out / "work.json",
            dict(
                phase=self.phase,
                requested=self.requested,
                completed=self.completed,
                unknown_partial=self.unknown_partial,
                wall_seconds=time.monotonic() - self.started,
            ),
        )


def episode_outcome(
    costs: np.ndarray, falls: np.ndarray, velocities: np.ndarray, done: np.ndarray, horizon: int
) -> dict:
    if not 0 < len(costs) <= horizon or any(
        len(value) != len(costs) for value in (falls, velocities, done)
    ):
        raise ValueError("Incomplete native execution evidence")
    if done.dtype.kind != "b" or not all(
        np.isfinite(value).all() for value in (costs, falls, velocities)
    ):
        raise ValueError("Native execution outcomes must be finite with boolean termination")
    if np.any(done[:-1]) or (not done[-1] and len(done) != horizon):
        raise ValueError("Missing or fabricated post-fall controls")
    survives = len(costs) == horizon and not bool(done[-1])
    native = float(np.sum(costs, dtype=np.float64))
    terminal = float(np.sum(falls, dtype=np.float64))
    tail = 5.0 * (horizon - len(costs)) if done[-1] else 0.0
    mae = float(np.mean(np.abs(velocities - 1.5)))
    return dict(
        controls=len(costs),
        horizon=horizon,
        done=bool(done[-1]),
        survived=survives,
        walking=bool(horizon == 1000 and survives and mae <= 0.3),
        mae=mae,
        native_cost=native,
        raw_cost=native - terminal,
        terminal_cost=terminal,
        absorbing_cost=tail,
        cost=native + tail,
        observed_seconds=0.008 * len(costs),
        right_censored=not bool(done[-1]),
    )


class TraceCollection:
    def __init__(self) -> None:
        self.episodes: list[dict] = []

    def execute(self, env: Any, controller: Any, meter: Counters, **kwargs: Any) -> dict:
        try:
            episode = execute_episode(env, controller, meter, **kwargs)
        except BaseException as error:
            if hasattr(error, "partial_trace"):
                self.episodes.append(error.partial_trace)
            raise
        self.episodes.append(episode)
        return episode

    def save(self, path: Path, meter: Counters | None = None) -> dict:
        if not self.episodes:
            return {}
        state_keys = ("states", "observations", "warmstarts", "integrations")
        step_keys = (
            "actions",
            "reference_actions",
            "has_reference",
            "costs",
            "fall_costs",
            "velocities",
            "done",
        )
        arrays = {
            key: np.concatenate([episode[key] for episode in self.episodes])
            for key in state_keys + step_keys
        }
        arrays.update(
            episode_step_offsets=np.cumsum(
                [0] + [len(e["actions"]) for e in self.episodes], dtype=np.int64
            ),
            episode_state_offsets=np.cumsum(
                [0] + [len(e["states"]) for e in self.episodes], dtype=np.int64
            ),
            episode_seeds=np.asarray([e["seed"] for e in self.episodes], dtype=np.int64),
            episode_kinds=np.asarray([e["kind"] for e in self.episodes]),
            episode_horizons=np.asarray([e["horizon"] for e in self.episodes], dtype=np.int64),
        )
        if meter is not None:
            reservation = sum(value.nbytes for value in arrays.values()) + 1024**2
            # Cleanup is permitted after wall exhaustion, under the reservation
            # obtained before this bounded stage began.
            meter.check_storage(reservation)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        return file_binding(path)


def execute_episode(
    env: Any,
    controller: Any,
    meter: Counters,
    *,
    seed: int,
    kind: str,
    horizon: int,
    reference: Any = None,
    replay: dict | None = None,
) -> dict:
    from .runtime import restore, snapshot

    if replay is None:
        saved_rng = np.random.get_state()
        try:
            np.random.seed(seed)
            observation = meter.paid({"resets": 1, "external_forwards": 2}, env.reset)
        finally:
            np.random.set_state(saved_rng)
    else:
        value = dict(
            state=replay["states"][0],
            warmstart=replay["warmstarts"][0],
            integration=replay["integrations"][0],
            spec=np.array(replay["spec"]),
            dimensions=np.array([9, 9, 6]),
        )
        meter.paid(
            {"restores": 1, "external_forwards": 1, "snapshot_reads": 1},
            lambda: restore(env, value),
        )
        observation = env._get_obs()
    current = meter.paid({"snapshot_reads": 1}, lambda: snapshot(env))
    saved = {
        key: [np.asarray(value).copy()]
        for key, value in dict(
            states=current["state"],
            observations=observation,
            warmstarts=current["warmstart"],
            integrations=current["integration"],
        ).items()
    }
    saved.update(
        {
            key: []
            for key in (
                "actions",
                "reference_actions",
                "has_reference",
                "costs",
                "fall_costs",
                "velocities",
                "done",
            )
        }
    )

    def finish() -> dict:
        arrays = {key: np.asarray(value) for key, value in saved.items()}
        for key in ("done", "has_reference"):
            arrays[key] = arrays[key].astype(bool)
        for key in ("actions", "reference_actions"):
            arrays[key] = arrays[key].reshape(-1, 6)
        arrays.update(kind=kind, seed=seed, horizon=horizon)
        return arrays

    try:
        for step in range(horizon):
            if replay is None:
                action = controller.predict(observation.astype(np.float32), meter=meter).astype(
                    np.float64
                )
            else:
                action = np.asarray(replay["actions"][step], dtype=np.float64)
            reference_action = (
                np.full(6, np.nan)
                if reference is None
                else reference.predict(observation.astype(np.float32), meter=meter).astype(
                    np.float64
                )
            )
            if action.shape != (6,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                raise ValueError("Direct torque controller returned invalid bounded controls")
            observation, cost, done, info = meter.paid(
                {"native_controls": 1, "physics_substeps": 4}, lambda: env.step(action)
            )
            current = meter.paid({"snapshot_reads": 1}, lambda: snapshot(env))
            for key, value in dict(
                states=current["state"],
                observations=observation,
                warmstarts=current["warmstart"],
                integrations=current["integration"],
                actions=action,
                reference_actions=reference_action,
                has_reference=reference is not None,
                costs=cost,
                fall_costs=info["fall_cost"],
                velocities=info["x_velocity"],
                done=done,
            ).items():
                saved[key].append(np.asarray(value).copy())
            if replay is not None:
                for key in ("states", "warmstarts", "integrations"):
                    if not np.allclose(saved[key][-1], replay[key][step + 1], rtol=0, atol=1e-10):
                        raise ValueError(f"Held teacher replay {key} mismatch")
                if not np.isclose(cost, replay["costs"][step], rtol=0, atol=1e-10):
                    raise ValueError("Held teacher replay native cost mismatch")
            if done:
                break
    except BaseException as error:
        error.partial_trace = finish()
        raise
    arrays = finish()
    arrays["outcome"] = dict(
        kind=kind,
        seed=seed,
        **episode_outcome(
            arrays["costs"], arrays["fall_costs"], arrays["velocities"], arrays["done"], horizon
        ),
    )
    meter.flush()
    return arrays


def _save_torch(path: Path, value: Any) -> dict:
    import torch

    with path.open("xb") as handle:
        torch.save(value, handle)
    return file_binding(path)


def run(config: dict, prepared: Path, out: Path) -> dict:
    """Execute the fixed roster once; prerequisites and clean controls precede jitter fits."""
    launch = verify_launch(config, prepared)
    import mujoco
    import torch

    from src.envs.walker2d import Walker2d

    from .diagnostic_policies import NATIVE_TASK, LegacySACActor
    from .policy import Policy, fit_policy
    from .similarity_metrics import action_mse, smoothed_action_mse, state_explained_score

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    meter = Counters(config["limits"], out)
    result: dict[str, Any] = dict(schema=SCHEMA, status="RUNNING", cases=[], prerequisites={})
    try:
        if (
            str(mujoco.__version__) != config["runtime"]["mujoco_version"]
            or str(torch.__version__) != config["runtime"]["torch_version"]
        ):
            raise ValueError("Pinned native/Torch version mismatch")
        torch.set_num_threads(1)
        if config["task"] != NATIVE_TASK:
            raise ValueError("Frozen native task differs")
        if config["runtime"]["torch_threads"] != 1:
            raise ValueError("J2 requires one CPU Torch thread")
        manifest = launch.pop("manifest")
        prepared_copy_bytes = sum(
            path.stat().st_size for path in prepared.rglob("*") if path.is_file()
        )
        meter.reserve_storage(prepared_copy_bytes + 1024**2)
        shutil.copytree(prepared, out / "inputs")
        with np.load(prepared / "corpus.npz", allow_pickle=False) as archive:
            corpus = {key: archive[key] for key in archive.files}
        for key, metadata in manifest["conditions"].items():
            if array_hash(corpus["labels__" + key]) != metadata["sha256"]:
                raise ValueError("Prepared label array differs from its manifest")
        if not np.array_equal(
            corpus["normalization_mean"], corpus["observations"].mean(0)
        ) or not np.array_equal(
            corpus["normalization_std"], np.maximum(corpus["observations"].std(0), 1e-6)
        ):
            raise ValueError("Prepared clean normalization differs")
        result["input_reads"] = dict(
            **launch,
            prepared_copy_bytes=prepared_copy_bytes,
            logical_npz_payload_bytes=sum(value.nbytes for value in corpus.values()),
            reference_checkpoint_bytes=manifest["files"]["reference.pt"]["bytes"],
            physical_disk_io_bytes=None,
        )
        explainers = json.loads((prepared / "explainers.json").read_text())
        env = meter.paid(
            {"environment_constructions": 1}, lambda: Walker2d(nthread=1, **NATIVE_TASK)
        )
        actor = meter.paid(
            {"checkpoint_loads": 1, "reference_actor_constructions": 1},
            lambda: LegacySACActor.from_checkpoint(prepared / "reference.pt"),
        )
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        prerequisite = out / "prerequisites"
        prerequisite.mkdir()
        replays = TraceCollection()
        meter.phase = "held_replay"
        meter.reserve_storage(8 * 1024**2)
        try:
            for episode in range(9, 16):
                for anchor in range(100, 1000, 100):
                    source = {
                        key: corpus[f"sac_{key}"][episode, anchor : anchor + 26]
                        for key in ("states", "integrations", "warmstarts")
                    }
                    source.update(
                        actions=corpus["sac_actions"][episode, anchor : anchor + 25],
                        costs=corpus["sac_costs"][episode, anchor : anchor + 25],
                        spec=spec,
                    )
                    replays.execute(
                        env,
                        None,
                        meter,
                        seed=episode * 1000 + anchor,
                        kind="replay",
                        horizon=25,
                        replay=source,
                    )
        finally:
            result["prerequisites"]["held_replays"] = dict(
                episodes=len(replays.episodes),
                traces=replays.save(prerequisite / "replay.npz", meter),
            )
        reference_traces = TraceCollection()
        meter.phase = "reference_qualification"
        meter.reserve_storage(64 * 1024**2)
        try:
            for seed in config["seeds"]["evaluation"]:
                reference_traces.execute(
                    env, actor, meter, seed=seed, kind="reference", horizon=1000
                )
        finally:
            result["prerequisites"]["reference_trace"] = reference_traces.save(
                prerequisite / "reference.npz", meter
            )
        reference_outcomes = [episode["outcome"] for episode in reference_traces.episodes]
        panels = [
            sum(row["walking"] for row in reference_outcomes[start : start + 16])
            for start in (0, 16)
        ]
        result["prerequisites"]["reference"] = dict(
            panel_walking=panels,
            episodes=reference_outcomes,
            traces=result["prerequisites"]["reference_trace"],
        )
        if any(count < 15 for count in panels):
            result["status"] = "STOPPED_REFERENCE_QUALIFICATION"
            result["closed_negative"] = True
            return result
        normalization = dict(mean=corpus["normalization_mean"], std=corpus["normalization_std"])
        parents, tapes, global_rngs, incoming_predictions = {}, {}, {}, {}
        (out / "parents").mkdir()
        (out / "streams").mkdir()
        meter.reserve_storage(160 * 1024**2)
        for seed in config["seeds"]["training"]:
            parent = meter.paid(
                {"policy_constructions": 1},
                lambda: Policy(seed, "cpu", normalization, hidden_dim=128).eval(),
            )
            parents[seed] = parent
            incoming_predictions[seed] = parent.predict(corpus["observations"], meter=meter)
            parent.checkpoint(
                out / "parents" / f"{seed}.pt",
                metadata=dict(
                    study=SCHEMA,
                    global_torch_rng_state=torch.random.get_rng_state().tolist(),
                    global_numpy_state=global_numpy_state(),
                ),
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            tapes[seed] = meter.paid(
                {"preallocated_index_values": 10000 * 256},
                lambda: rng.integers(0, 9000, size=(10000, 256)).astype(np.int64),
            )
            global_rngs[seed] = torch.random.get_rng_state().clone()
            np.savez_compressed(
                out / "streams" / f"{seed}.npz",
                indices=tapes[seed],
                global_torch_rng=global_rngs[seed].numpy(),
                global_numpy_state_json=np.array(json.dumps(global_numpy_state(), sort_keys=True)),
            )
        clean_cases = []
        for condition in conditions():
            for seed in config["seeds"]["training"]:
                meter.phase = f"case:{seed}:{condition['id']}"
                meter.reserve_storage(64 * 1024**2)
                case = out / "cases" / f"{seed}_{condition['id']}"
                case.mkdir(parents=True)
                parent = parents[seed]
                torch.random.set_rng_state(global_rngs[seed])
                labels = corpus["labels__" + condition["id"]]
                incoming = incoming_predictions[seed]
                score_rows = []
                for index in range(9):
                    rows = slice(index * 1000, (index + 1) * 1000)
                    score_rows.append(
                        dict(
                            A=action_mse(labels[rows], incoming[rows]),
                            B=smoothed_action_mse(labels[rows], incoming[rows]),
                            C=state_explained_score(
                                corpus["observations"][rows],
                                labels[rows],
                                incoming[rows],
                                explainers["clean"],
                            )["score"],
                        )
                    )
                pre_scores = {
                    key: float(np.mean([row[key] for row in score_rows])) for key in "ABC"
                }
                pre_scores.update(D=None, E=None)
                observed_hash = hashlib.sha256()
                update_hashes = []
                initial_binding = {}

                def initial(value: dict) -> None:
                    nonlocal initial_binding
                    value["global_torch_rng"] = torch.random.get_rng_state().clone()
                    value["global_numpy_state"] = global_numpy_state()
                    initial_binding = _save_torch(case / "initial.pt", value)

                def index(value: dict) -> None:
                    actual = np.asarray(value["indices"], dtype=np.int64)
                    meter.paid({"sampled_index_values": len(actual)}, lambda: None)
                    if not np.array_equal(actual, tapes[seed][value["fit_update"] - 1]):
                        raise ValueError("Actual minibatch stream differs from frozen seed tape")
                    payload = actual.tobytes()
                    observed_hash.update(payload)
                    update_hashes.append(
                        np.frombuffer(hashlib.sha256(payload).digest(), dtype=np.uint8).copy()
                    )
                    if value["fit_update"] % 250 == 0:
                        meter.flush()

                try:
                    child, optimizer, evidence = meter.paid(
                        {"fits": 1, "policy_constructions": 1, "optimizer_constructions": 1},
                        lambda: fit_policy(
                            parent,
                            corpus["observations"],
                            labels,
                            updates=10000,
                            batch_size=256,
                            lr=3e-4,
                            eta_theta=0.0,
                            seed=seed,
                            meter=meter,
                            prox_reduction="mean",
                            initial_observer=initial,
                            index_observer=index,
                        ),
                    )
                except BaseException as error:
                    write_json(
                        case / "fit_failure.json",
                        getattr(error, "fit_evidence", dict(error=repr(error))),
                    )
                    raise
                child.eval()
                checkpoint = child.checkpoint(
                    case / "child.pt",
                    optimizer,
                    dict(
                        study=SCHEMA,
                        condition=condition["id"],
                        seed=seed,
                        global_torch_rng_state=torch.random.get_rng_state().tolist(),
                        global_numpy_state=global_numpy_state(),
                    ),
                )
                held_predictions = child.predict(corpus["held_observations"], meter=meter)
                train_predictions = child.predict(corpus["observations"], meter=meter)
                offline_mse = float(np.mean((held_predictions - corpus["held_clean"]) ** 2))
                np.savez_compressed(
                    case / "fit_arrays.npz",
                    observed_tape_update_hashes=np.stack(update_hashes),
                    held_predictions=held_predictions,
                    incoming_predictions=incoming,
                    train_predictions=train_predictions,
                )
                evidence.update(
                    initial_package_sha256=initial_binding["sha256"],
                    initial_file="initial.pt",
                    tape_file=f"streams/{seed}.npz",
                    tape_sha256=array_hash(tapes[seed]),
                    observed_tape_sha256=observed_hash.hexdigest(),
                    initial_global_rng_sha256=array_hash(global_rngs[seed].numpy()),
                    checkpoint=checkpoint,
                    held_clean_mse=offline_mse,
                )
                write_json(case / "fit.json", evidence)
                traces = TraceCollection()
                try:
                    for reset in config["seeds"]["probe"]:
                        traces.execute(
                            env,
                            child,
                            meter,
                            seed=reset,
                            kind="probe",
                            horizon=100,
                            reference=actor,
                        )
                    for index_reset, reset in enumerate(config["seeds"]["evaluation"]):
                        episode = traces.execute(
                            env, child, meter, seed=reset, kind="evaluation", horizon=1000
                        )
                        reference_episode = reference_traces.episodes[index_reset]
                        for key in ("states", "observations", "warmstarts", "integrations"):
                            if not np.array_equal(episode[key][0], reference_episode[key][0]):
                                raise ValueError(
                                    "Paired evaluation initial simulator contexts differ"
                                )
                finally:
                    trace_binding = traces.save(case / "traces.npz", meter)
                probes = [value for value in traces.episodes if value["kind"] == "probe"]
                gaps = [
                    np.mean((value["actions"] - value["reference_actions"]) ** 2, axis=1)
                    for value in probes
                ]
                summary = dict(
                    case_id=case.name,
                    seed=seed,
                    condition=condition,
                    checkpoint=checkpoint,
                    traces=trace_binding,
                    episodes=[value["outcome"] for value in traces.episodes],
                    pre_update_scores=pre_scores,
                    pre_update_trajectory_scores=score_rows,
                    fitting=dict(
                        train_mse=float(np.mean((train_predictions - labels) ** 2)),
                        held_clean_mse=offline_mse,
                    ),
                    postfit=dict(
                        standing100=float(np.mean(np.concatenate(gaps))),
                        standing25=float(np.mean(np.concatenate([value[:25] for value in gaps]))),
                        offline_action_mse=offline_mse,
                        probe_mean_controls=float(
                            np.mean([len(value["actions"]) for value in probes])
                        ),
                        probe_falls=sum(bool(value["done"][-1]) for value in probes),
                    ),
                )
                write_json(case / "summary.json", summary)
                result["cases"].append(
                    dict(
                        case_id=case.name,
                        seed=seed,
                        condition=condition["id"],
                        summary=f"cases/{case.name}/summary.json",
                    )
                )
                if condition["id"] == "clean":
                    clean_cases.append(summary)
                meter.check()
                meter.flush()
                write_json(out / "result.json", result)
                print(
                    json.dumps(
                        dict(
                            event="case_completed",
                            case=case.name,
                            walking=sum(
                                row["walking"]
                                for row in summary["episodes"]
                                if row["kind"] == "evaluation"
                            ),
                        )
                    ),
                    flush=True,
                )
            if condition["id"] == "clean":
                walks = [
                    sum(row["walking"] for row in value["episodes"] if row["kind"] == "evaluation")
                    for value in clean_cases
                ]
                qualified = sum(walks) / (32 * len(walks)) >= 0.9 and min(walks) >= 24
                result["prerequisites"]["clean"] = dict(walking=walks, qualified=qualified)
                if not qualified:
                    result["status"] = "STOPPED_CLEAN_POSITIVE_CONTROL"
                    result["closed_negative"] = True
                    return result
        result["status"] = "COMPLETED_FIXED_J2_ROSTER"
        return result
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error), phase=meter.phase)
        raise
    finally:
        meter.flush()
        result["work"] = json.loads((out / "work.json").read_text())
        result["actual_fits"] = meter.completed.get("fits", 0)
        write_json(out / "result.json", result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.action == "prepare":
        print(json.dumps(prepare(config, args.out), indent=2))
    else:
        if args.prepared is None:
            parser.error("run requires --prepared")
        result = run(config, args.prepared, args.out)
        print(json.dumps(result, indent=2))
        if result.get("status") not in {
            "COMPLETED_FIXED_J2_ROSTER",
            "STOPPED_REFERENCE_QUALIFICATION",
            "STOPPED_CLEAN_POSITIVE_CONTROL",
        }:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
