"""Synthetic lifecycle, frozen-input gates and finite-study decision rules."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import trajectory_pi_study as study

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def recipe():
    return json.loads((ROOT / "configs/studies/trajectory_pi_p1.json").read_text())


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def launch_fixture(tmp_path, recipe):
    """Metadata-only sealed export: no checkpoint decoding or task imports."""
    config = copy.deepcopy(recipe)
    config["launch_ready"] = True
    data = tmp_path / "prepared"
    data.mkdir()
    roster, parents, teachers = [], [], []
    for parent in config["parents"]:
        directory = data / parent["prepared_directory"]
        directory.mkdir(parents=True)
        for name in ("policy.pt", "global_rng.pt", "stream.json"):
            path = directory / name
            path.write_bytes(f"synthetic metadata {parent['stream']} {name}".encode())
            roster.append(dict(study.binding(path), path=str(path.relative_to(data))))
        parent["checkpoint"] = study.binding(directory / "policy.pt")
        parents.append(dict(stream=parent["stream"], directory=parent["prepared_directory"]))
    for teacher in config["teachers"]:
        path = data / teacher["prepared_path"]
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"bound metadata fixture only")
        roster.append(dict(study.binding(path), path=str(path.relative_to(data))))
        teachers.append(dict(slot=teacher["slot"], file_relative=teacher["prepared_path"]))
    dump(tmp_path / "config.json", config)
    files = [
        dict(
            study.binding(ROOT / "src/gps/mppi_gps/trajectory_pi_study.py"),
            path="src/gps/mppi_gps/trajectory_pi_study.py",
        )
    ]
    dump(
        tmp_path / "source.json",
        dict(
            schema="trajectory-pi-source-1",
            git_commit="synthetic",
            files=files,
            roster_sha256=hashlib.sha256(
                json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        ),
    )
    dump(
        tmp_path / "prepared.json",
        dict(
            schema="trajectory-pi-prepared-1",
            config_sha256=study.binding(tmp_path / "config.json")["sha256"],
            normalization_inherited=True,
            files=sorted(roster, key=lambda item: item["path"]),
            parents=parents,
            teachers=teachers,
        ),
    )
    paths = dict(
        config=tmp_path / "config.json",
        source_manifest=tmp_path / "source.json",
        prepared_manifest=tmp_path / "prepared.json",
    )
    seals = {name + "_sha256": study.binding(path)["sha256"] for name, path in paths.items()}
    dump(
        tmp_path / "preflight.json",
        dict(
            schema="trajectory-pi-preflight-1",
            status="PASS",
            checks=[dict(name=name, passed=True) for name in sorted(study.CHECKS)],
            work={"requested_optimizer_steps": 8, "completed_optimizer_steps": 8},
            **seals,
        ),
    )
    dump(
        tmp_path / "capacity.json",
        dict(
            schema="trajectory-pi-resource-observation-1",
            passed=True,
            observed_unix=time.time(),
            available_logical_cpus=28,
            available_memory_bytes=32 * 1024**3,
            free_disk_bytes=80 * 1024**3,
            versions=config["resources"]["runtime"],
            **seals,
        ),
    )
    paths.update(
        preflight=tmp_path / "preflight.json", resource_observation=tmp_path / "capacity.json"
    )
    launch = dict(
        schema="trajectory-pi-p1-launch-1",
        source_directory=str(ROOT),
        prepared_directory=str(data),
        output_directory=str(tmp_path / "unused-output"),
        **{name: study.binding(path) for name, path in paths.items()},
    )
    dump(tmp_path / "launch.json", launch)
    return tmp_path / "launch.json"


def change_artifact(launch_path, role, change):
    launch = json.loads(launch_path.read_text())
    path = Path(launch[role]["path"])
    value = json.loads(path.read_text())
    change(value)
    dump(path, value)
    launch[role] = study.binding(path)
    dump(launch_path, launch)


def test_sealed_launch_metadata_inspection_requires_no_neural_or_simulator_imports(
    tmp_path, recipe, monkeypatch
):
    path = launch_fixture(tmp_path, recipe)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "mujoco"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    config, launch, prepared, preflight = study.validate_launch(path)
    assert config["rounds"] == 20
    assert len(prepared["parents"]) == 3
    assert preflight["work"]["completed_optimizer_steps"] == 8
    assert not Path(launch["output_directory"]).exists()


@pytest.mark.parametrize(
    ("role", "change", "message"),
    [
        ("preflight", lambda row: row.update(status="FAILED"), "did not pass"),
        ("preflight", lambda row: row["checks"].pop(), "checks absent"),
        ("preflight", lambda row: row.update(config_sha256="wrong"), "seal differs"),
        (
            "preflight",
            lambda row: row["work"].update(unknown_partial_native_controls=1),
            "unknown partial",
        ),
        ("resource_observation", lambda row: row.update(observed_unix=time.time() - 301), "fresh"),
        ("resource_observation", lambda row: row.update(available_logical_cpus=27), "capacity"),
    ],
)
def test_missing_prerequisites_stop_before_any_worker_or_neural_import(
    tmp_path, recipe, monkeypatch, role, change, message
):
    path = launch_fixture(tmp_path, recipe)
    change_artifact(path, role, change)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "mujoco", "multiprocessing"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ValueError, match=message):
        study.run_loop(path)
    assert not (tmp_path / "unused-output").exists()


def test_bound_parent_bytes_cannot_be_replaced(tmp_path, recipe):
    path = launch_fixture(tmp_path, recipe)
    (tmp_path / "prepared/initial/stream0/policy.pt").write_bytes(b"different")
    with pytest.raises(ValueError, match="Bound input changed"):
        study.validate_launch(path)


def test_agreement_never_substitutes_for_teacher_competence():
    candidate = dict(
        result=dict(
            outcome=dict(walking=True, full_survival=True, complete_cost=120.0), selected_D=1e30
        )
    )
    assert study.admit(candidate, 100.0)
    candidate["result"]["outcome"]["complete_cost"] = 120.00001
    assert not study.admit(candidate, 100.0)
    candidate["result"]["outcome"].update(complete_cost=0.0, walking=False)
    assert not study.admit(candidate, 100.0)


def test_worker_reservations_include_policy_scoring_cost(recipe):
    independent = study.native_limits(recipe, "solve", 0, 100)
    coupled = study.native_limits(recipe, "solve", 0.12, 100)
    assert independent["model_controls"] == coupled["model_controls"]
    assert coupled["neural_rows"] == 100 * recipe["planner"]["K"] * 1000 + 2000
    assert independent["neural_rows"] == 102000
    assert coupled["physics_substeps"] == 4 * (coupled["model_controls"] + 1000)


def operation_fixture(tmp_path, recipe):
    operation = study.Operations.__new__(study.Operations)
    operation.c = recipe
    operation.out = tmp_path
    operation.alpha = {row["name"]: row["alpha"] for row in recipe["arms"]}
    directory = tmp_path / "parent"
    directory.mkdir()
    (directory / "policy.pt").write_bytes(b"metadata-only parent")
    template = tmp_path / "original-plan.npz"
    np.savez(template, actions=np.zeros((1000, 6)))
    operation.current = {(s, arm): directory for s in recipe["streams"] for arm in operation.alpha}
    operation.initial_bank = [
        dict(
            slot=slot,
            plan=str(template),
            plan_binding=study.binding(template),
            context=str(template),
            context_binding=study.binding(template),
            trace=str(template),
            trace_binding=study.binding(template),
            outcome=dict(complete_cost=100.0),
        )
        for slot in range(2)
    ]
    operation.bank = {key: list(operation.initial_bank) for key in operation.current}
    return operation


def test_acquisition_arms_share_planner_stream_and_failed_teachers_keep_incumbents(
    tmp_path, recipe
):
    operation = operation_fixture(tmp_path, recipe)
    jobs = []

    def execute(requested):
        jobs.extend(requested)
        return [
            dict(
                path=row["out"],
                result=dict(outcome=dict(walking=False, full_survival=False, complete_cost=50.0)),
            )
            for row in requested
        ]

    operation.execute = execute
    old = {
        key: dict(paths=[dict(D=2.0), dict(D=2.0)], policy_hash="same-parent")
        for key in operation.current
    }
    rows = operation.acquire(3, old)
    assert len(jobs) == 12
    for stream in recipe["streams"]:
        for slot in range(2):
            pair = [
                job
                for job in jobs
                if f"s{stream}/" in job["out"] and job["out"].endswith(f"slot{slot}")
            ]
            assert len(pair) == 2
            assert pair[0]["planner_seed"] == pair[1]["planner_seed"]
            assert (
                pair[0]["planner_seed"]
                == recipe["seeds"]["planner_argument_base"]
                + stream * recipe["seeds"]["stream_stride"]
                + 3 * recipe["seeds"]["round_stride"]
                + slot * recipe["seeds"]["slot_stride"]
            )
            assert pair[0]["initial_plan_binding"] == pair[1]["initial_plan_binding"]
            assert pair[0]["context_binding"] == pair[1]["context_binding"]
            assert {job["alpha"] for job in pair} == {0, 0.12}
            assert all(job["context_step"] == 0 and job["iterations"] == 100 for job in pair)
    assert all(not item["accepted"] for row in rows.values() for item in row["admission"])
    assert all(row["teacher_after_acquisition"]["mean_D"] == 2.0 for row in rows.values())
    assert all(bank == operation.initial_bank for bank in operation.bank.values())


def test_own_state_queries_bind_pre_action_anchors_without_shifting_template(tmp_path, recipe):
    import h5py

    operation = operation_fixture(tmp_path, recipe)
    actor_dir = tmp_path / "actor"
    actor_dir.mkdir()
    with h5py.File(actor_dir / "trace.h5", "w") as source:
        source["actions"] = np.zeros((101, 6))
    evaluations = {
        key: dict(
            jobs=[
                dict(
                    path=str(actor_dir),
                    result=dict(
                        trace=study.binding(actor_dir / "trace.h5"),
                        outcome=dict(complete_cost=200.0),
                    ),
                )
            ]
        )
        for key in operation.current
    }
    jobs = []

    def execute(requested):
        jobs.extend(requested)
        results = []
        for row in requested:
            path = Path(row["out"])
            path.mkdir(parents=True)
            np.savez(path / "solution.npz", actions=np.full((1000, 6), 0.25))
            results.append(
                dict(path=str(path), result=dict(outcome=dict(walking=True, complete_cost=100.0)))
            )
        return results

    operation.execute = execute
    result = operation.queries(4, evaluations)
    assert len(jobs) == 12
    for key, row in result.items():
        assert row["0"]["policy_cost_gap"] == 100.0
        assert row["100"]["policy_cost_gap"] is None
        assert row["100"]["action_gap"] == pytest.approx(np.sqrt(6) * 0.25)
        for anchor in (0, 100):
            selected = [
                job
                for job in jobs
                if f"s{key[0]}/{key[1]}/" in job["out"] and job["context_step"] == anchor
            ]
            assert len(selected) == 1
            job = selected[0]
            assert job["initial_plan_file"] == operation.initial_bank[0]["plan"]
            assert job["context_file"] == str(actor_dir / "trace.h5")
            assert job["iterations"] == 25 and job["horizon"] == 1000
    for stream in recipe["streams"]:
        for anchor in (0, 100):
            assert (
                len(
                    {
                        job["planner_seed"]
                        for job in jobs
                        if f"s{stream}/" in job["out"] and job["context_step"] == anchor
                    }
                )
                == 1
            )


def ledger_fixture(tmp_path, recipe):
    recipe = copy.deepcopy(recipe)
    recipe["limits"]["model_controls"] = 10
    recipe["resources"]["coordinator_ram_bytes"] = 1024**4
    return study.StudyWork(
        tmp_path, recipe, {"work": {"requested_optimizer_steps": 8, "completed_optimizer_steps": 8}}
    )


def test_reservation_fails_before_dispatch_and_keeps_first_reservation(tmp_path, recipe):
    ledger = ledger_fixture(tmp_path, recipe)
    first = dict(out=str(tmp_path / "one"), limits=dict(model_controls=6))
    ledger.reserve(first)
    with pytest.raises(ValueError, match="reserved budget"):
        ledger.reserve(dict(out=str(tmp_path / "two"), limits=dict(model_controls=5)))
    assert list(ledger.pending) == [first["out"]]
    ledger.unknown_pending()
    assert ledger.work.counts["unknown_partial_model_controls"] == 6
    assert ledger.work.counts["completed_optimizer_steps"] == 8
    ledger.work.close("FAILED")


def test_worker_paid_overrun_is_preserved_before_rejection(tmp_path, recipe):
    ledger = ledger_fixture(tmp_path, recipe)
    job = dict(out=str(tmp_path / "worker"), limits=dict(model_controls=5))
    ledger.reserve(job)
    with pytest.raises(ValueError, match="exceeded its reservation"):
        ledger.complete(
            job,
            dict(
                status="COMPLETED",
                counts=dict(requested_model_controls=6, completed_model_controls=6),
            ),
        )
    assert ledger.work.counts["requested_model_controls"] == 6
    assert ledger.work.counts["completed_model_controls"] == 6
    assert not ledger.pending
    ledger.work.close("FAILED")


def test_malformed_worker_receipt_charges_unknown_reservation(tmp_path, recipe):
    ledger = ledger_fixture(tmp_path, recipe)
    directory = tmp_path / "worker"
    ledger.reserve(dict(out=str(directory), limits=dict(model_controls=4)))
    directory.mkdir()
    (directory / "receipt.json").write_text("interrupted JSON")
    ledger.unknown_pending()
    assert ledger.work.counts["requested_model_controls"] == 4
    assert ledger.work.counts["unknown_partial_model_controls"] == 4
    ledger.work.close("FAILED")


class FakeOperations:
    def __init__(self):
        self.events = []
        self.current = {(s, a): 2000 for s in range(3) for a in ("independent", "coupled")}
        self.bank = {key: [] for key in self.current}

    def old_gaps(self):
        self.events.append(("gaps", tuple(self.current.values())))
        return dict.fromkeys(self.current, {"rms": 1.0})

    def acquire(self, round_index, old):
        self.events.append(("acquire", round_index))
        assert set(old) == set(self.current)
        return {key: {"incoming_updates": value} for key, value in self.current.items()}

    def fit(self, round_index, rows):
        self.events.append(("fit", round_index))
        for key in self.current:
            assert rows[key]["incoming_updates"] == self.current[key]
            self.current[key] += 100

    def evaluate(self, round_index, *, confirmation=False, initial=False):
        self.events.append(("confirm" if confirmation else "evaluate", round_index, initial))
        assert all(value == 2000 + round_index * 100 for value in self.current.values())
        return {key: {"version": value} for key, value in self.current.items()}

    def queries(self, round_index, evaluations):
        self.events.append(("query", round_index))
        assert all(row["version"] == self.current[key] for key, row in evaluations.items())
        return {}

    def close_round(self, round_index, rows, evaluations, queries):
        self.events.append(("close", round_index))
        return dict(round=round_index, versions=dict(self.current))


def test_small_lifecycle_alternates_acquisition_fit_evaluation_queries_and_continues():
    operations = FakeOperations()
    history, final = study.execute_schedule({"rounds": 2}, operations)
    assert [row["round"] for row in history] == [0, 1, 2]
    assert set(final) == set(operations.current)
    assert set(operations.current.values()) == {2200}
    names = [row[0] for row in operations.events]
    assert names == [
        "gaps",
        "evaluate",
        "query",
        "close",
        "gaps",
        "acquire",
        "fit",
        "evaluate",
        "query",
        "close",
        "gaps",
        "acquire",
        "fit",
        "evaluate",
        "query",
        "close",
        "confirm",
    ]


def fake_panel(walking, survival=16, controls=1000):
    return dict(
        summary=dict(
            walking=walking, full_survival=survival, mean_controls=controls, mean_cost=100.0
        )
    )


def evidence_fixture(recipe):
    bank = [
        dict(outcome=dict(complete_cost=100.0, walking=True, full_survival=True)) for _ in range(2)
    ]
    history = []
    for index in range(21):
        row = dict(round=index, streams={}, parents={})
        for stream in recipe["streams"]:
            row["parents"][str(stream)] = fake_panel(1)
            row["streams"][str(stream)] = {}
            for arm in ("independent", "coupled"):
                gap = 1.0 if arm == "independent" or index < 16 else 0.7
                row["streams"][str(stream)][arm] = dict(
                    teacher_after_fit=dict(rms=gap),
                    queries={
                        str(a): dict(present=True, qualified=True, action_gap=gap) for a in (0, 100)
                    },
                    bank=copy.deepcopy(bank),
                    raw_candidates=[
                        dict(result=dict(outcome=dict(walking=True))) for _ in range(2)
                    ],
                    student=fake_panel(4 if arm == "coupled" else 1),
                )
        history.append(row)
    confirm = {
        (s, arm): fake_panel(4 if arm == "coupled" else 1)
        for s in recipe["streams"]
        for arm in ("independent", "coupled", "parent")
    }
    return history, confirm


def test_joint_criteria_require_qualified_contraction_teacher_competence_and_confirmed_walking(
    recipe,
):
    history, confirm = evidence_fixture(recipe)
    result = study.criteria(history, confirm, recipe)
    assert result["passed"]
    assert result["joint_streams"] == [0, 1, 2]
    assert result["verdict"] == "HELPED_FINITE_ALTERNATING_LOOP"
    assert not result["full_project_success"]
    assert not result["per_stream"]["0"]["individual_competence"]


def test_missing_queries_are_unresolved_not_zero_or_failed_contraction(recipe):
    history, confirm = evidence_fixture(recipe)
    for stream in recipe["streams"]:
        history[17]["streams"][str(stream)]["coupled"]["queries"]["100"] = dict(
            present=False, qualified=False, action_gap=None
        )
    result = study.criteria(history, confirm, recipe)
    assert not result["passed"]
    assert result["verdict"] == "PARTIAL_WALKING_GAIN_WITH_UNRESOLVED_GAP"
    assert not result["per_stream"]["0"]["student_gap"]["100"]["resolved"]


def test_resolved_nonshrinking_gap_is_distinct_from_missing_query(recipe):
    history, confirm = evidence_fixture(recipe)
    for row in history:
        for stream in recipe["streams"]:
            row["streams"][str(stream)]["coupled"]["teacher_after_fit"]["rms"] = 1.0
    result = study.criteria(history, confirm, recipe)
    assert result["verdict"] == "PARTIAL_WALKING_GAIN_GAP_NOT_CLOSED"
    assert not result["passed"]


def test_a_single_stream_regression_blocks_other_streams_and_retention_is_not_walking(recipe):
    history, confirm = evidence_fixture(recipe)
    confirm[(2, "coupled")] = fake_panel(0, survival=15, controls=999)
    result = study.criteria(history, confirm, recipe)
    assert not result["passed"]
    assert not result["all_stream_nonregression"]
    assert result["joint_streams"] == [0, 1]


def test_teacher_deterioration_cannot_manufacture_a_pass(recipe):
    history, confirm = evidence_fixture(recipe)
    for row in history[16:]:
        for stream in recipe["streams"]:
            for item in row["streams"][str(stream)]["coupled"]["bank"]:
                item["outcome"]["complete_cost"] = 111
    result = study.criteria(history, confirm, recipe)
    assert not result["passed"]
    assert not any(row["teacher_competence"]["passed"] for row in result["per_stream"].values())


def test_undefined_gap_denominators_stay_unresolved():
    result = study._contraction([0.0], [0.0], [1.0], 0.2, 0.9)
    assert not result["resolved"]
    assert not result["passed"]
    assert result["contraction"] is None
