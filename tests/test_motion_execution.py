"""Synthetic HDF/NPZ decoding and frozen one-shot motion execution checks."""

import json
import os
import platform
import shutil
import signal
import subprocess
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import lifecycle, motion_inputs
from src.gps.mppi_gps.durability import durable_json, file_binding
from src.gps.mppi_gps.experiments import motion_diagnosis as execution
from src.gps.mppi_gps.provenance import sha


def empty_work():
    return dict.fromkeys(
        (
            "source_hash_bytes",
            "source_io_bytes",
            "dataset_attempts",
            "completed_datasets",
            "decoded_bytes",
            "failed_datasets",
            "physical_rows",
            "terminal_rows",
            "queries_requested",
            "queries_completed",
            "aligned_requested",
            "aligned_completed",
            *execution.ZERO_WORK,
        ),
        0,
    )


@pytest.fixture
def synthetic_catalog(tmp_path):
    made = []

    def make(storage="npz", terminal="full", change=None, count=150):
        directory = tmp_path / ("synthetic-arrays-" + str(len(made)))
        directory.mkdir()
        made.append(directory)
        clock = np.arange(count + 1) * 0.008
        all_states = np.zeros((count + 1, 19))
        all_states[:, 0] = clock
        all_states[:, 1] = clock * 1.5
        all_states[:, 2] = 1.2
        all_states[:, 10] = 12  # Native velocity, deliberately beyond observation clipping.
        physical = dict(
            states=all_states if terminal == "full" else all_states[:-1],
            row_steps=np.arange(count, dtype=np.int64),
        )
        if terminal == "separate":
            physical["terminal_state"] = all_states[-1]
        actions = np.tanh(clock[:-1, None] + np.arange(6) / 10)
        predictions = dict(
            observations=np.concatenate(
                (all_states[:-1, 2:10], np.clip(all_states[:-1, 10:19], -10, 10)), axis=1
            ),
            executed_actions=actions,
            predictions=np.repeat((actions * 0.5)[:, None, :], 9, axis=1),
        )
        if change:
            change(physical, predictions)
        physical_path = directory / ("physical." + ("npz" if storage == "npz" else "h5"))
        if storage == "npz":
            np.savez_compressed(physical_path, **physical)
            group = None
        else:
            with h5py.File(physical_path, "w") as stream:
                owner = stream.create_group("physical")
                for name, value in physical.items():
                    owner[name] = value
            group = "physical"
        prediction_path = directory / "predictions.h5"
        with h5py.File(prediction_path, "w") as stream:
            owner = stream.create_group("recorded")
            for name, value in predictions.items():
                owner[name] = value
        endpoint = None
        if terminal == "full":
            endpoint = dict(storage="last_row_of_full_state_array", row=count)
        elif terminal == "separate":
            endpoint = dict(
                storage="separate_npz_member",
                key="terminal_state",
                **motion_inputs.array_binding(physical["terminal_state"]),
            )
        physical_source = dict(
            file=str(physical_path),
            group=group,
            states=motion_inputs.array_binding(physical["states"]),
            row_steps=motion_inputs.array_binding(physical["row_steps"]),
            terminal=endpoint,
        )
        prediction_source = dict(
            file=str(prediction_path),
            group="recorded",
            reference_column=8,
            arrays={key: motion_inputs.array_binding(value) for key, value in predictions.items()},
        )
        catalog = dict(
            files=[
                dict(path=str(path), bytes=path.stat().st_size, recorded_sha256=sha(path))
                for path in (physical_path, prediction_path)
            ],
            roster=[
                dict(
                    id="synthetic-0",
                    index=0,
                    source="synthetic",
                    ancestry="synthetic-reset",
                    N=count,
                    physical_source=physical_source,
                    prediction_source=prediction_source,
                )
            ],
        )
        budgets = dict(
            source_hash_bytes=1024 * 1024,
            source_io_bytes=1024 * 1024,
            decoded_bytes=1024 * 1024,
            dataset_attempts=6 if terminal == "separate" else 5,
        )
        return catalog, budgets, physical, predictions

    return make


@pytest.mark.parametrize(
    "storage,terminal", [("npz", "full"), ("npz", "separate"), ("hdf", "full"), ("hdf", "missing")]
)
def test_bound_loader_decodes_only_declared_arrays_and_charges_reads(
    synthetic_catalog, storage, terminal
):
    catalog, budgets, physical, prediction = synthetic_catalog(storage, terminal)
    work, events = empty_work(), []
    rows, bindings = motion_inputs.read_sources(catalog, budgets, work, events.append)
    assert len(rows) == 1
    np.testing.assert_array_equal(rows[0]["states"], physical["states"][:150])
    np.testing.assert_array_equal(rows[0]["actions"], prediction["executed_actions"])
    np.testing.assert_array_equal(rows[0]["reference_predictions"], prediction["predictions"][:, 8])
    assert rows[0]["states"][0, 10] == 12
    assert (rows[0]["terminal_state"] is None) == (terminal == "missing")
    assert work["source_hash_bytes"] == sum(f["bytes"] for f in catalog["files"])
    assert 0 < work["source_io_bytes"] <= budgets["source_io_bytes"]
    assert work["dataset_attempts"] == work["completed_datasets"] == budgets["dataset_attempts"]
    assert work["decoded_bytes"] == sum(b["bytes"] for b in bindings.values())
    assert work["failed_datasets"] == 0 and work["physical_rows"] == 150
    assert work["terminal_rows"] == int(terminal != "missing")
    assert (
        len([e for e in events if e["event"] == "dataset_requested"]) == budgets["dataset_attempts"]
    )


@pytest.mark.parametrize("field,value", [("sha256", "wrong"), ("shape", [1, 19]), ("dtype", "<f4")])
def test_failed_decoded_array_is_still_charged(synthetic_catalog, field, value):
    catalog, budgets, physical, _ = synthetic_catalog()
    catalog["roster"][0]["physical_source"]["states"][field] = value
    work, events = empty_work(), []
    with pytest.raises(ValueError, match="contract"):
        motion_inputs.read_sources(catalog, budgets, work, events.append)
    assert work["dataset_attempts"] == work["failed_datasets"] == 1
    assert work["completed_datasets"] == 0
    assert work["decoded_bytes"] == physical["states"].nbytes
    assert events[-1]["event"] == "dataset_failed"


@pytest.mark.parametrize(
    "ceiling", ["source_hash_bytes", "source_io_bytes", "decoded_bytes", "dataset_attempts"]
)
def test_loader_budget_stops_preserve_paid_work(synthetic_catalog, ceiling):
    catalog, budgets, physical, _ = synthetic_catalog()
    budgets[ceiling] = 0
    work, events = empty_work(), []
    with pytest.raises(RuntimeError, match="ceiling"):
        motion_inputs.read_sources(catalog, budgets, work, events.append)
    if ceiling == "source_hash_bytes":
        assert work[ceiling] > 0 and work["dataset_attempts"] == 0
    elif ceiling == "source_io_bytes":
        assert work[ceiling] > 0
    elif ceiling == "decoded_bytes":
        assert work["decoded_bytes"] == physical["states"].nbytes
        assert work["failed_datasets"] == 1
    else:
        assert work["dataset_attempts"] == work["decoded_bytes"] == 0
        assert events[-1]["event"] == "dataset_rejected"


@pytest.mark.parametrize(
    "mutation", ["observation", "ordering", "terminal_clock", "terminal_row", "state_clock"]
)
def test_row_join_and_terminal_mismatches_fail_without_substitution(synthetic_catalog, mutation):
    def change(physical, prediction):
        if mutation == "observation":
            prediction["observations"][0, 0] += 0.1
        elif mutation == "ordering":
            physical["row_steps"][[0, 1]] = physical["row_steps"][[1, 0]]
        elif mutation == "terminal_clock":
            physical["states"][-1, 0] += 0.01
        elif mutation == "state_clock":
            physical["states"][2, 0] += 0.008

    catalog, budgets, _, _ = synthetic_catalog(change=change)
    if mutation == "terminal_row":
        catalog["roster"][0]["physical_source"]["terminal"]["row"] = 149
    with pytest.raises(ValueError, match="join|ordering|context|time"):
        motion_inputs.read_sources(catalog, budgets, empty_work(), lambda event: None)


def test_source_hash_and_inode_mutations_are_detected(synthetic_catalog):
    catalog, budgets, _, _ = synthetic_catalog()
    bad = deepcopy(catalog)
    bad["files"][0]["recorded_sha256"] = "foreign"
    with pytest.raises(ValueError, match="SHA"):
        motion_inputs.read_sources(bad, budgets, empty_work(), lambda event: None)
    path = Path(catalog["files"][0]["path"])
    changed = []

    def journal(event):
        if event["event"] == "dataset_completed" and not changed:
            info = path.stat()
            replacement = path.with_name("replacement.npz")
            shutil.copyfile(path, replacement)
            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
            os.replace(replacement, path)
            changed.append(True)

    with pytest.raises(ValueError, match="changed while decoding"):
        motion_inputs.read_sources(catalog, budgets, empty_work(), journal)
    assert changed


def test_later_failure_retains_earlier_accepted_rows(synthetic_catalog):
    catalog, budgets, _, _ = synthetic_catalog()
    second = deepcopy(catalog["roster"][0])
    second.update(id="synthetic-1", index=1)
    second["physical_source"]["states"]["sha256"] = "foreign"
    catalog["roster"].append(second)
    budgets["dataset_attempts"] = 10
    work, events = empty_work(), []
    with pytest.raises(ValueError, match="contract"):
        motion_inputs.read_sources(catalog, budgets, work, events.append)
    assert work["physical_rows"] == 150 and work["terminal_rows"] == 1
    assert work["completed_datasets"] == 5 and work["failed_datasets"] == 1
    assert work["dataset_attempts"] == 6
    assert [e["id"] for e in events if e["event"] == "row_completed"] == ["synthetic-0"]


@pytest.fixture
def prepared_motion(tmp_path, monkeypatch, synthetic_catalog):
    root = tmp_path / "synthetic-repository"
    (root / "scripts").mkdir(parents=True)
    (root / "metadata").mkdir()
    (root / "README.md").write_text("Synthetic-only frozen motion worker.\n")
    runs = []

    def prepare(changes=None, script="pass\n", source_change=None):
        catalog, budgets, physical, predictions = synthetic_catalog(change=source_change)
        errors = (predictions["executed_actions"] - predictions["predictions"][:, 8]) ** 2
        bins = [
            dict(
                start_control=s,
                stop_control_exclusive=min(s + 100, 150),
                rows=min(100, 150 - s),
                E=float(np.mean(errors[s : s + 100])),
                per_actuator_mse=np.mean(errors[s : s + 100], axis=0).tolist(),
            )
            for s in (0, 100)
        ]
        summary = dict(trajectories=[dict(id="synthetic-0", E=float(np.mean(errors)), bins=bins)])
        metadata = {}
        for role, value in (("motion_roster", catalog), ("k11_result", summary)):
            name = "metadata/" + role + ".json"
            (root / name).write_text(json.dumps(value))
            metadata[role] = dict(source=name, **file_binding(root / name))
        descriptors = [
            catalog["roster"][0]["physical_source"][key] for key in ("states", "row_steps")
        ]
        descriptors.extend(catalog["roster"][0]["prediction_source"]["arrays"].values())
        config = dict(
            schema="walker-motion-diagnosis-config-v1",
            metadata=metadata,
            expected=dict(
                trajectories=1,
                rows=150,
                curve_bins=6,
                segments=6,
                cross_trajectory_pairs=0,
                within_queries=3,
                cross_queries=0,
                pair_queries=3,
                aligned_state_pairs=75,
            ),
            source_files=2,
            source_file_bytes=sum(f["bytes"] for f in catalog["files"]),
            selected_datasets=5,
            selected_decode_bytes=sum(d["bytes"] for d in descriptors),
            terminal_rows=1,
            platform=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                h5py=h5py.__version__,
                device="cpu",
                threads=1,
            ),
            new_work=dict.fromkeys(execution.ZERO_WORK, 0),
            budgets=dict(
                budgets,
                wall_seconds=10,
                peak_rss_bytes=1024 * 1024 * 1024,
                storage_bytes=16 * 1024 * 1024,
                pair_queries=3,
                aligned_state_pairs=75,
            ),
        )
        if changes:
            changes(config)
        (root / "scripts/train_mppi_gps.py").write_text(script)
        monkeypatch.setattr(execution, "load_recipe", lambda name: deepcopy(config))
        monkeypatch.setattr(lifecycle, "ROOT", root)
        out = tmp_path / ("synthetic-run-" + str(len(runs)))
        execution.prepare(out, root)
        runs.append(out)
        monkeypatch.setattr(lifecycle, "ROOT", out / "source")
        monkeypatch.setattr(execution, "_assert_metadata_only", lambda: None)
        monkeypatch.setattr(execution, "_rss", lambda: 1024)
        return out, config, root

    return prepare


def launch_owner(out, config):
    return dict(
        schema="walker-motion-diagnosis-launch-v1",
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        source_manifest_sha256=sha(out / "manifest.json"),
        command=execution._command(out),
        retries_permitted=False,
        timeout_seconds=config["budgets"]["wall_seconds"],
    )


def synchronous_child(monkeypatch, out, after=None):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        execution.worker(out)
        if after:
            after(out / "result.json")
        return SimpleNamespace(pid=123456, wait=lambda timeout: 0, poll=lambda: 0)

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    return calls


def test_frozen_metadata_and_worker_lifecycle_are_create_only(prepared_motion):
    out, config, root = prepared_motion()
    assert execution.verify_inputs(out)[0] == config
    with pytest.raises(FileExistsError):
        execution.prepare(out, root)
    durable_json(out / "launch_receipt.json", launch_owner(out, config))
    result = execution.worker(out)
    assert result["execution_complete"] and result["error"] is None
    assert execution.valid_work(result["work"], config)
    assert result["motion_archive"]["sha256"] == sha(out / "evidence/motion.npz")
    with np.load(out / "evidence/motion.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"states_000", "actions_000", "reference_000", "terminal_000"}
        assert arrays["states_000"].shape == (150, 19)
    journal = [json.loads(line) for line in (out / "work.jsonl").read_text().splitlines()]
    assert sum(e.get("pairs", 0) for e in journal if e["event"] == "queries_requested") == 3
    before = sha(out / "result.json")
    with pytest.raises(FileExistsError):
        execution.worker(out)
    assert sha(out / "result.json") == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "foreign"),
        ("command", []),
        ("retries_permitted", True),
        ("timeout_seconds", 999),
        ("source_manifest_sha256", "foreign"),
    ],
)
def test_worker_ownership_rejects_before_decode(prepared_motion, field, value):
    out, config, _ = prepared_motion()
    receipt = launch_owner(out, config)
    receipt[field] = value
    durable_json(out / "launch_receipt.json", receipt)
    with pytest.raises(ValueError, match="owner"):
        execution.worker(out)
    assert not (out / "work.jsonl").exists()


@pytest.mark.parametrize("target", ["metadata", "source"])
def test_frozen_metadata_or_source_mutation_stops_before_worker(prepared_motion, target):
    out, _, _ = prepared_motion()
    path = out / (
        "input_metadata/motion_roster.json" if target == "metadata" else "source/README.md"
    )
    path.write_text(path.read_text() + "changed")
    with pytest.raises(ValueError, match="differs|changed"):
        execution.verify_inputs(out)


@pytest.mark.parametrize(
    "where,key,value",
    [
        ("platform", "python", "foreign"),
        ("platform", "numpy", "foreign"),
        ("platform", "h5py", "foreign"),
        ("platform", "device", "cuda"),
        ("platform", "threads", 2),
        ("new_work", "new_neural_rows", False),
        ("new_work", "new_physics_controls", 0.0),
    ],
)
def test_pinned_runtime_and_exact_zero_budgets_gate_all_source_reads(
    prepared_motion, where, key, value
):
    out, config, _ = prepared_motion(lambda c: c[where].update({key: value}))
    durable_json(out / "launch_receipt.json", launch_owner(out, config))
    result = execution.worker(out)
    assert not result["execution_complete"]
    assert result["work"]["source_hash_bytes"] == result["work"]["dataset_attempts"] == 0
    assert result["error"]["type"] == "ValueError"


def test_valid_work_requires_exact_integer_owned_census(prepared_motion):
    out, config, _ = prepared_motion()
    durable_json(out / "launch_receipt.json", launch_owner(out, config))
    work = execution.worker(out)["work"]
    assert execution.valid_work(work, config)
    for changed in (
        dict(work, foreign=0),
        dict(work, new_neural_rows=False),
        dict(work, source_io_bytes=1.0),
        dict(work, source_io_bytes=0),
        dict(work, failed_datasets=1),
        dict(work, queries_completed=2),
    ):
        assert not execution.valid_work(changed, config)
    assert not execution.valid_work({}, config)


def test_failed_decode_work_and_partial_query_request_remain_durable(prepared_motion, monkeypatch):
    from src.gps.mppi_gps import motion_diagnosis as core

    out, config, _ = prepared_motion()
    durable_json(out / "launch_receipt.json", launch_owner(out, config))

    def failure(*args):
        raise RuntimeError("synthetic query interruption")

    monkeypatch.setattr(core, "segment_distances", failure)
    result = execution.worker(out)
    assert not result["execution_complete"]
    assert result["work"]["dataset_attempts"] == result["work"]["completed_datasets"] == 5
    assert result["work"]["queries_requested"] == 3 and result["work"]["queries_completed"] == 0
    assert result["work"]["aligned_requested"] == 75 and result["work"]["aligned_completed"] == 0
    events = [json.loads(line) for line in (out / "work.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "worker_failed"
    assert events[-1]["work"] == result["work"]


def test_launch_closes_bound_success_and_rejects_second_attempt(prepared_motion, monkeypatch):
    out, _, _ = prepared_motion()
    calls = synchronous_child(monkeypatch, out)
    result = execution.launch(out)
    assert result["execution_complete"] and result["child_reaped"]
    assert result["result_error"] is None and not result["cleanup_errors"]
    assert result["result_sha256"] == sha(out / "result.json")
    assert (
        result["stored_bytes"] == execution._size(out) - (out / "worker_exit.json").stat().st_size
    )
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    with pytest.raises(FileExistsError):
        execution.launch(out)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["json", "hash", "counter"])
def test_launch_rejects_malformed_foreign_or_unowned_result(prepared_motion, monkeypatch, mutation):
    out, _, _ = prepared_motion()

    def after(path):
        value = json.loads(path.read_text())
        if mutation == "json":
            path.write_text("[]")
            return
        if mutation == "hash":
            value["source_manifest_sha256"] = "foreign"
        else:
            value["work"]["unowned_work"] = 0
        path.write_text(json.dumps(value))

    synchronous_child(monkeypatch, out, after)
    receipt = execution.launch(out)
    assert not receipt["execution_complete"]
    assert receipt["result_error"] is not None


def test_deadline_kills_and_reaps_one_actual_synthetic_child(prepared_motion, monkeypatch):
    out, _, _ = prepared_motion(
        lambda c: c["budgets"].update(wall_seconds=0.03), script="import time\ntime.sleep(10)\n"
    )
    children, original = [], subprocess.Popen

    def popen(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    receipt = execution.launch(out)
    assert not receipt["execution_complete"] and receipt["timeout"]
    assert receipt["child_reaped"] and receipt["kill_attempted"]
    assert receipt["exit_code"] == -signal.SIGKILL and not receipt["cleanup_errors"]
    assert len(children) == 1 and children[0].poll() == -signal.SIGKILL
    with pytest.raises(FileExistsError):
        execution.launch(out)
    assert len(children) == 1
