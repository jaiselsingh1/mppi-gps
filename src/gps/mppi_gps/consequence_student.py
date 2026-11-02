"""C1's fixed three-arm learning roster, metrics and standalone panels."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .consequence_learning import (
    ARMS,
    contrast_gate,
    cost_effect_weights,
    fit_arm,
    fitting_mse,
    generate_expected_tape,
    learning_gates,
    shuffle_stratified,
)
from .consequence_native import load_npz
from .provenance import fingerprint, sha, write


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        episodes=len(rows),
        successes=sum(int(v["walking"]) for v in rows),
        survived=sum(int(v["survived"]) for v in rows),
        controls=sum(v["controls"] for v in rows),
        mean_cost=float(np.mean([v["cost"] for v in rows])),
        mean_steps=float(np.mean([v["controls"] for v in rows])),
        mean_velocity_mae=float(np.mean([v["mae"] for v in rows])),
    )


def _load_actor(path: Path, meter: Any):
    from .policy import Policy

    meter.authorize(dict(checkpoint_loads=1, policy_constructions=1))
    actor, state, payload = meter.measure(
        "checkpoint_loads",
        1,
        lambda: meter.measure(
            "policy_constructions",
            1,
            lambda: Policy.from_checkpoint(path, device="cpu"),
            purpose="checkpoint_actor_construction",
        ),
        purpose="fixed_checkpoint_endpoint",
    )
    return actor, state, payload


def _metric(
    actor: Any,
    data: dict,
    batch_size: int,
    alias: str,
    label: str,
    meter: Any,
    weights: np.ndarray | None = None,
) -> dict:
    predictions, references = [], []
    historical_squared_error = 0.0
    for batch, start in enumerate(range(0, len(data["actions"]), batch_size)):
        observations = data["observations"][start : start + batch_size]
        targets = data["actions"][start : start + batch_size]

        def persist(value):
            reference = meter.save_arrays(
                f"metric_{alias}_{label}_{batch}",
                observations=np.asarray(observations, dtype=np.float32),
                predictions=value,
                targets=np.asarray(targets, dtype=np.float32),
            )
            references.append(dict(start=start, stop=start + len(targets), reference=reference))

        meter.after_prediction = persist
        try:
            prediction = actor.predict(observations, meter=meter)
        finally:
            meter.after_prediction = None
        predictions.append(prediction)
        residual = prediction.astype(np.float64) - np.asarray(targets, dtype=np.float32).astype(
            np.float64
        )
        historical_squared_error += float(np.sum(residual**2, dtype=np.float64))
    prediction = np.concatenate(predictions)
    result = fitting_mse(prediction, data["actions"], weights)
    if label in {"train", "validation"}:
        # Retain the inherited full_dataset_mse accumulation order as well as
        # its exact prediction batches and float32 fitting targets.
        result["action_mse"] = historical_squared_error / np.asarray(data["actions"]).size
    return dict(
        **result,
        action_mse_reduction="float64_SSE_per_batch_then_ordered_sum"
        if label in {"train", "validation"}
        else "global_float64_mean",
        actor_hash=actor.hash(),
        batch_size=batch_size,
        rows=len(data["actions"]),
        batches=references,
    )


def run_learning_stage(
    out: Path, config: dict, meter: Any, native: Any, replay: dict, assays: list[dict]
) -> dict:
    import torch

    torch.set_num_threads(1)
    prepared, worker = out / "prepared", out / "worker"
    catalog = native.catalog
    weights_by_stream, weight_records = {}, {}
    (worker / "weights").mkdir()
    for stream in range(3):
        data = load_npz(prepared / catalog["streams"][str(stream)]["file"])
        rows = [v for v in assays if v["stream"] == stream]
        if len(rows) != 4000 or [(v["case"], v["t"]) for v in rows] != [
            (c, t) for c in range(4) for t in range(1000)
        ]:
            raise ValueError("C1 assay source/control ordering differs")
        delta = np.asarray([v["delta"] for v in rows], dtype=np.float64)
        treatment = cost_effect_weights(delta)

        def rng_observer(value):
            if value["event"] == "before":
                meter.authorize(dict(shuffle_calls=1, shuffle_entries=value["rows"]))
                meter.record(
                    "shuffle_requested",
                    stream=stream,
                    group=value["group"],
                    rows=value["rows"],
                    rng=fingerprint(value["rng"]),
                )
                meter.charge(dict(shuffle_calls=1, shuffle_entries=value["rows"]))
            else:
                meter.returned("shuffle_calls", 1)
                meter.returned("shuffle_entries", value["rows"])
                meter.record(
                    "shuffle_complete",
                    stream=stream,
                    group=value["group"],
                    rows=value["rows"],
                    rng=fingerprint(value["rng"]),
                )

        shuffled = shuffle_stratified(
            treatment["weights"],
            data["squared_disagreement"],
            config["seeds"]["shuffle"][stream],
            rng_observer,
        )
        path = worker / "weights" / f"s{stream}.npz"
        with path.open("xb") as output:
            np.savez(
                output,
                deltas=delta,
                raw=treatment["raw"],
                normalized=treatment["normalized"],
                cost_effect=treatment["weights"],
                shuffled=shuffled["shuffled"],
                source_indices=shuffled["source_indices"],
            )
            output.flush()
            os.fsync(output.fileno())
        group_file = worker / "weights" / f"s{stream}_groups.json"
        write(group_file, shuffled["groups"])
        weight_records[str(stream)] = dict(
            file=str(path.relative_to(out)),
            bytes=path.stat().st_size,
            sha256=sha(path),
            groups_file=str(group_file.relative_to(out)),
            groups_sha256=sha(group_file),
            mean_raw=treatment["mean_raw"],
            permutation_calls=shuffled["permutation_calls"],
            returned_entries=shuffled["returned_entries"],
            rng_initial=shuffled["rng_initial"],
            rng_terminal=shuffled["rng_terminal"],
        )
        weights_by_stream[stream] = dict(
            cost_effect=treatment["weights"], shuffled=shuffled["shuffled"]
        )
    gate = contrast_gate(weights_by_stream)
    write(worker / "contrast_gate.json", dict(**gate, weights=weight_records))
    if not gate["passed"]:
        return dict(
            status="STOPPED_DEGENERATE_PLANNED_CONTRAST",
            contrast_gate=gate,
            weights=weight_records,
            learning_measured=False,
            standalone_measured=False,
        )

    global_rng = np.load(
        prepared / catalog["parents"]["0"]["global_torch_rng"]["file"], allow_pickle=False
    )
    torch.set_rng_state(torch.from_numpy(global_rng.copy()))
    global_owner = fingerprint(torch.get_rng_state())
    actors, registry, fit_records, metrics = {}, {}, {}, {}
    historical = {
        label: load_npz(prepared / f"inputs/{label}.npz") for label in ("train", "validation")
    }
    sources = json.loads((prepared / "metadata/source_bindings.json").read_text())
    for stream in range(3):
        descriptor = catalog["parents"][str(stream)]
        parent, incoming, _ = _load_actor(prepared / descriptor["file"], meter)
        if (
            parent.hash() != descriptor["actor_hash"]
            or fingerprint(incoming) != descriptor["incoming_optimizer_sha256"]
        ):
            raise ValueError("C1 actual loaded parent continuation differs")
        parent_alias = f"s{stream}_parent"
        actors[parent_alias] = parent
        registry[parent_alias] = dict(
            file=str(Path("prepared") / descriptor["file"]),
            sha256=descriptor["sha256"],
            actor_hash=parent.hash(),
            stream=stream,
            arm="parent",
        )
        data = load_npz(prepared / catalog["streams"][str(stream)]["file"])
        metrics[parent_alias] = dict(
            own=fitting_mse(
                data["policy_actions"], data["actions"], weights_by_stream[stream]["cost_effect"]
            ),
            historical={},
        )
        for event in sources["reused_incoming_historical_metric_events"]:
            event = event["exact_saved_event"]
            if event["tag"].startswith(parent_alias + "/"):
                metrics[parent_alias]["historical"][event["tag"].split("/")[-1]] = dict(
                    action_mse=event["action_mse"], reused_audited_event=event["id"]
                )
        expected = generate_expected_tape(
            incoming, worker / "learning" / f"s{stream}_expected_indices.npy", meter, f"s{stream}"
        )
        for arm in config["arms"]:
            alias = f"s{stream}_{arm}"
            row_weights = None if arm == "uniform" else weights_by_stream[stream][arm]
            fitted = fit_arm(
                parent,
                incoming,
                data,
                row_weights,
                parent.seed,
                alias,
                worker / "learning",
                expected["tape"],
                meter,
            )
            record = fitted["record"]
            path = Path(fitted["directory"]) / "endpoint.pt"
            actor, state, _ = _load_actor(path, meter)
            if actor.hash() != fitted["actor"].hash() or fingerprint(state) != fingerprint(
                fitted["optimizer_state"]
            ):
                raise ValueError("C1 endpoint reload does not preserve complete continuation")
            actors[alias] = actor
            registry[alias] = dict(
                file=str(path.relative_to(out)),
                sha256=sha(path),
                actor_hash=actor.hash(),
                stream=stream,
                arm=arm,
            )
            fit_records[alias] = record
            metrics[alias] = dict(
                own=_metric(
                    actor, data, 1000, alias, "own", meter, weights_by_stream[stream]["cost_effect"]
                ),
                historical={},
            )
            for label, values in historical.items():
                metrics[alias]["historical"][label] = _metric(
                    actor, values, 1024, alias, label, meter
                )
        if fingerprint(torch.get_rng_state()) != global_owner:
            raise ValueError("C1 unexpectedly changed inherited global Torch RNG")
    write(worker / "actor_registry.json", registry)
    write(worker / "fit_records.json", fit_records)
    write(worker / "metrics.json", metrics)

    evaluations, panels = [], {name: {} for name in ("primary", "confirmation", "source")}
    with (worker / "evaluations.jsonl").open("x") as output:
        for stream in range(3):
            for panel in panels:
                panels[panel][str(stream)] = {}
            for arm in ARMS:
                alias = f"s{stream}_{arm}"
                actor = actors[alias]
                for panel in ("primary", "confirmation", "source"):
                    rows = []
                    seeds = (
                        config["seeds"][panel]
                        if panel != "source"
                        else [
                            catalog["episodes"][tag]["seed"]
                            for tag in catalog["streams"][str(stream)]["episode_tags"]
                        ]
                    )
                    for case, seed in enumerate(seeds):
                        source_tag = (
                            None
                            if panel != "source"
                            else catalog["streams"][str(stream)]["episode_tags"][case]
                        )
                        result = native.trace(
                            f"eval_{alias}_{panel}_{case:02d}",
                            actor=actor,
                            actor_alias=alias,
                            seed=seed,
                            source_tag=source_tag,
                            source=None if source_tag is None else native.episode(source_tag),
                        )
                        result.update(stream=stream, arm=arm, panel=panel, case=case)
                        output.write(
                            json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
                        )
                        output.flush()
                        rows.append(result)
                        evaluations.append(result)
                    panels[panel][str(stream)][arm] = _summary(rows)
    gates = learning_gates(panels["primary"], panels["confirmation"])
    gaps = []
    teachers = {row["source_tag"]: row for row in replay["episodes"]}
    for row in evaluations:
        if row["panel"] == "source":
            teacher = teachers[row["source_tag"]]
            gaps.append(
                dict(
                    actor=row["actor"],
                    source_tag=row["source_tag"],
                    teacher_cost=teacher["cost"],
                    student_cost=row["cost"],
                    cost_gap=row["cost"] - teacher["cost"],
                    teacher_walking=teacher["walking"],
                    student_walking=row["walking"],
                    teacher_survived=teacher["survived"],
                    student_survived=row["survived"],
                    teacher_controls=teacher["controls"],
                    student_controls=row["controls"],
                )
            )
    write(worker / "outcomes.json", dict(panels=panels, gates=gates, source_context_gaps=gaps))
    return dict(
        status="COMPLETED_FIXED_C1_ROSTER",
        contrast_gate=gate,
        weights=weight_records,
        learning_measured=True,
        standalone_measured=True,
        student_outcomes=panels,
        teacher_student_gap=gaps,
        gates=gates,
        metrics=metrics,
        checkpoints=registry,
        fit_records=fit_records,
    )
