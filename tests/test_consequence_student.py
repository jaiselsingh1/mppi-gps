"""Whole-roster C1 software fixture: zero scientific NN, physics or fitting.

The fake meter counts requested interface rows/calls only. All 9 fits, 3 tape
generations, 12 checkpoint loads and 432 native rollouts are replaced by fakes;
these counts are assertions of the orchestration contract, not research costs.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.gps.mppi_gps import consequence_student as student
from src.gps.mppi_gps.provenance import fingerprint


class FakeMeter:
    def __init__(self):
        self.work, self.events, self.predictions = {}, [], []
        self.after_prediction = None

    def authorize(self, work):
        assert all(amount >= 0 for amount in work.values())

    def charge(self, work):
        for key, value in work.items():
            self.work[key] = self.work.get(key, 0) + value

    def returned(self, key, value):
        pass

    def record(self, event, **fields):
        self.events.append(dict(event=event, **fields))

    def save_arrays(self, tag, **arrays):
        self.predictions.append((tag, {key: value.copy() for key, value in arrays.items()}))
        return dict(synthetic_only=True, tag=tag)

    def measure(self, kind, amount, operation, **kwargs):
        self.charge({kind: amount, "prediction_calls": int(kind == "neural_rows")})
        value = operation()
        if kind == "neural_rows" and self.after_prediction is not None:
            self.after_prediction(value)
        return value


class FakeActor:
    """Deterministic array arithmetic; no torch module or network is constructed."""

    def __init__(self, alias):
        self.alias, self.seed = alias, 7

    def hash(self):
        return fingerprint(self.alias)

    def predict(self, observations, meter):
        return meter.measure(
            "neural_rows",
            len(observations),
            lambda: np.zeros((len(observations), 6), dtype=np.float32),
        )


class FakeNative:
    def __init__(self, catalog):
        self.catalog, self.calls = catalog, []

    def episode(self, tag):
        return dict(synthetic_source=tag)

    def trace(self, tag, *, actor, actor_alias, seed, source_tag, source):
        self.calls.append(
            dict(tag=tag, actor=actor_alias, seed=seed, source_tag=source_tag, source=source)
        )
        controls = 3 + seed % 3
        return dict(
            actor=actor_alias,
            source_tag=source_tag,
            walking="cost_effect" in actor_alias,
            survived=True,
            controls=controls,
            cost=float(seed % 17),
            mae=0.2,
            executed_controls=list(range(controls)),
        )


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    before_global = torch.get_rng_state().clone()
    prepared, worker = tmp_path / "prepared", tmp_path / "worker"
    for directory in (
        prepared / "parents",
        prepared / "streams",
        prepared / "inputs",
        prepared / "metadata",
        worker,
    ):
        directory.mkdir(parents=True)
    global_rng = torch.Generator(device="cpu").manual_seed(7).get_state().numpy()
    np.save(prepared / "parents/global.npy", global_rng, allow_pickle=False)
    catalog = dict(parents={}, streams={}, episodes={})
    actors, state_by_path, fit_calls, tape_calls, load_calls = {}, {}, [], [], []
    for stream in range(3):
        alias = f"s{stream}_parent"
        path = prepared / "parents" / f"s{stream}.pt"
        path.write_bytes(b"synthetic checkpoint placeholder")
        actor, state = FakeActor(alias), dict(synthetic_stream=stream)
        actors[str(path)], state_by_path[str(path)] = actor, state
        catalog["parents"][str(stream)] = dict(
            file=str(path.relative_to(prepared)),
            actor_hash=actor.hash(),
            incoming_optimizer_sha256=fingerprint(state),
            sha256="synthetic-file-binding",
            global_torch_rng=dict(file="parents/global.npy"),
        )
        episode_tags = [f"s{stream}_case{case}" for case in range(4)]
        catalog["streams"][str(stream)] = dict(
            file=f"streams/s{stream}.npz", episode_tags=episode_tags
        )
        for case, tag in enumerate(episode_tags):
            catalog["episodes"][tag] = dict(seed=1000 + 4 * stream + case)
        np.savez(
            prepared / f"streams/s{stream}.npz",
            observations=np.zeros((4000, 17)),
            actions=np.zeros((4000, 6)),
            policy_actions=np.zeros((4000, 6), np.float32),
            squared_disagreement=np.zeros(4000),
        )
    for label, rows in (("train", 4000), ("validation", 2000)):
        np.savez(
            prepared / f"inputs/{label}.npz",
            observations=np.zeros((rows, 17)),
            actions=np.zeros((rows, 6)),
        )
    bindings = dict(
        reused_incoming_historical_metric_events=[
            dict(
                exact_saved_event=dict(
                    tag=f"s{stream}_parent/{label}",
                    action_mse=0.25,
                    id=f"synthetic-{stream}-{label}",
                )
            )
            for stream in range(3)
            for label in ("train", "validation")
        ]
    )
    (prepared / "metadata/source_bindings.json").write_text(json.dumps(bindings))

    def load_actor(path, meter):
        load_calls.append(str(path))
        return actors[str(path)], deepcopy(state_by_path[str(path)]), dict(synthetic_only=True)

    def expected(incoming, out, meter, tag):
        # No sampling: contents are deliberately unused by the fake fitting call.
        result = dict(tape=np.zeros((2000, 256), dtype=np.int64), binding=dict(synthetic_only=True))
        tape_calls.append(dict(tag=tag, incoming=deepcopy(incoming), result=result))
        return result

    def fit(parent, incoming, data, weights, seed, tag, out, expected_tape, meter):
        directory = Path(out) / tag
        directory.mkdir(parents=True)
        endpoint = directory / "endpoint.pt"
        endpoint.write_bytes(b"synthetic endpoint placeholder")
        actor, state = FakeActor(tag), dict(synthetic_endpoint=tag)
        actors[str(endpoint)], state_by_path[str(endpoint)] = actor, state
        fit_calls.append(
            dict(
                parent=parent.alias,
                incoming=deepcopy(incoming),
                tag=tag,
                weights=None if weights is None else weights.copy(),
                expected_tape=expected_tape,
                rows=len(data["actions"]),
            )
        )
        return dict(
            actor=actor,
            optimizer_state=state,
            directory=str(directory),
            record=dict(synthetic_only=True, fixed_updates=2000),
        )

    monkeypatch.setattr(student, "_load_actor", load_actor)
    monkeypatch.setattr(student, "generate_expected_tape", expected)
    monkeypatch.setattr(student, "fit_arm", fit)
    config = dict(
        arms=["uniform", "cost_effect", "shuffled"],
        seeds=dict(
            shuffle=[2800000000 + stream for stream in range(3)],
            primary=list(range(2000, 2016)),
            confirmation=list(range(3000, 3016)),
        ),
    )
    assays = [
        dict(stream=stream, case=case, t=t, delta=float(t % 200))
        for stream in range(3)
        for case in range(4)
        for t in range(1000)
    ]
    replay = dict(
        episodes=[
            dict(source_tag=tag, cost=100.0 + index, walking=True, survived=True, controls=1000)
            for index, tag in enumerate(catalog["episodes"])
        ]
    )
    result = dict(
        out=tmp_path,
        config=config,
        meter=FakeMeter(),
        native=FakeNative(catalog),
        replay=replay,
        assays=assays,
        fit_calls=fit_calls,
        tape_calls=tape_calls,
        load_calls=load_calls,
    )
    try:
        yield result
    finally:
        torch.set_rng_state(before_global)


def run(fixture):
    return student.run_learning_stage(
        **{key: fixture[key] for key in ("out", "config", "meter", "native", "replay", "assays")}
    )


def test_fixed_whole_roster_metrics_panels_source_pairs_and_no_auto_followup(lifecycle):
    result = run(lifecycle)
    assert result["status"] == "COMPLETED_FIXED_C1_ROSTER" and result["gates"]["confirmed"]
    assert len(lifecycle["fit_calls"]) == 9 and len(lifecycle["tape_calls"]) == 3
    assert len(lifecycle["load_calls"]) == 12 and len(result["checkpoints"]) == 12
    for stream in range(3):
        calls = lifecycle["fit_calls"][stream * 3 : (stream + 1) * 3]
        assert [row["tag"] for row in calls] == [
            f"s{stream}_{arm}" for arm in ("uniform", "cost_effect", "shuffled")
        ]
        assert calls[0]["weights"] is None and calls[1]["weights"].dtype == np.float32
        assert all(row["incoming"] == calls[0]["incoming"] and row["rows"] == 4000 for row in calls)
        assert all(
            row["expected_tape"] is lifecycle["tape_calls"][stream]["result"]["tape"]
            for row in calls
        )
    meter = lifecycle["meter"]
    assert meter.work["neural_rows"] == 90000 and meter.work["prediction_calls"] == 90
    assert len(meter.predictions) == 90
    assert [len(row[1]["targets"]) for row in meter.predictions[:10]] == [
        1000,
        1000,
        1000,
        1000,
        1024,
        1024,
        1024,
        928,
        1024,
        976,
    ]
    assert len(lifecycle["native"].calls) == 432
    for panel, count in (("primary", 192), ("confirmation", 192), ("source", 48)):
        assert (
            sum(
                actors[arm]["episodes"]
                for actors in result["student_outcomes"][panel].values()
                for arm in student.ARMS
            )
            == count
        )
    gaps = result["teacher_student_gap"]
    assert len(gaps) == 48
    teachers = {row["source_tag"]: row for row in lifecycle["replay"]["episodes"]}
    for gap in gaps:
        assert gap["teacher_cost"] == teachers[gap["source_tag"]]["cost"]
        assert gap["cost_gap"] == gap["student_cost"] - gap["teacher_cost"]
        assert gap["teacher_controls"] == 1000
    saved = [
        json.loads(line)
        for line in (lifecycle["out"] / "worker/evaluations.jsonl").read_text().splitlines()
    ]
    assert len(saved) == 432 and all(
        len(row["executed_controls"]) == row["controls"] for row in saved
    )
    assert sum(row["controls"] for row in saved) == sum(
        summary["controls"]
        for panel in result["student_outcomes"].values()
        for actors in panel.values()
        for summary in actors.values()
    )
    assert not (lifecycle["out"] / "worker/acquisition").exists()
    assert set(result) == {
        "status",
        "contrast_gate",
        "weights",
        "learning_measured",
        "standalone_measured",
        "student_outcomes",
        "teacher_student_gap",
        "gates",
        "metrics",
        "checkpoints",
        "fit_records",
    }


def test_degenerate_actual_weights_stop_every_fit_and_panel_after_all_assays(lifecycle):
    for row in lifecycle["assays"]:
        if row["stream"] == 2:
            row["delta"] = 0.0
    result = run(lifecycle)
    assert result["status"] == "STOPPED_DEGENERATE_PLANNED_CONTRAST"
    assert not result["learning_measured"] and not result["standalone_measured"]
    assert lifecycle["fit_calls"] == lifecycle["tape_calls"] == lifecycle["load_calls"] == []
    assert lifecycle["native"].calls == [] and lifecycle["meter"].predictions == []
    assert len(result["weights"]) == 3
    assert lifecycle["meter"].work == dict(shuffle_calls=624, shuffle_entries=12000)
    assert not (lifecycle["out"] / "worker/evaluations.jsonl").exists()


def test_missing_or_reordered_assay_rows_stop_without_fit(lifecycle):
    lifecycle["assays"][0], lifecycle["assays"][1] = lifecycle["assays"][1], lifecycle["assays"][0]
    with pytest.raises(ValueError, match="ordering"):
        run(lifecycle)
    assert lifecycle["fit_calls"] == lifecycle["native"].calls == []
