"""Synthetic lifecycle checks; no original K1 inputs or scientific arithmetic."""

import json
import platform
import signal
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import lifecycle, trajectory_review
from src.gps.mppi_gps.durability import durable_json, file_binding
from src.gps.mppi_gps.experiments import trajectory_review as execution
from src.gps.mppi_gps.provenance import sha


@pytest.fixture
def prepared_tree(tmp_path, monkeypatch):
    root = tmp_path / "synthetic-repository"
    (root / "scripts").mkdir(parents=True)
    (root / "evidence").mkdir()
    (root / "README.md").write_text("Synthetic frozen source.\n")
    roles = (
        "capsule",
        "config",
        "result",
        "matrix",
        "curves",
        "actuators",
        "saturation",
        "spread",
        "groups",
    )
    bindings = {}
    for role in roles:
        relative = "evidence/" + role + ".json"
        (root / relative).write_text(json.dumps(dict(role=role)))
        bindings[role] = dict(source=relative, **file_binding(root / relative))
    runs = []

    def prepare(changes=None, script="pass\n"):
        (root / "scripts/train_mppi_gps.py").write_text(script)
        config = dict(
            schema="walker-trajectory-review-config-v1",
            inputs=deepcopy(bindings),
            expected=dict(trajectories=1, rows=8, bins=1, actuator_entries=6),
            platform=dict(python=platform.python_version(), numpy=np.__version__, device="cpu"),
            budgets=dict(
                inputs=9,
                input_bytes=4096,
                storage_bytes=1024 * 1024,
                peak_rss_bytes=1024 * 1024 * 1024,
                wall_seconds=10,
                **{key: 0 for key in execution.ZERO_WORK},
            ),
        )
        if changes:
            changes(config)
        monkeypatch.setattr(execution, "load_recipe", lambda name: deepcopy(config))
        monkeypatch.setattr(lifecycle, "ROOT", root)
        out = tmp_path / ("run-" + str(len(runs)))
        execution.prepare(out, root)
        runs.append(out)
        monkeypatch.setattr(lifecycle, "ROOT", out / "source")
        return out, config, root

    return prepare


def owner_receipt(out, config):
    return dict(
        schema="walker-trajectory-review-launch-v1",
        config_sha256=sha(out / "config.json"),
        input_manifest_sha256=sha(out / "input_manifest.json"),
        source_manifest_sha256=sha(out / "manifest.json"),
        command=execution._command(out),
        timeout_seconds=config["budgets"]["wall_seconds"],
        retries_permitted=False,
    )


def synthetic_analysis(monkeypatch):
    calls = []

    def analyze(inputs, config):
        calls.append(inputs)
        return dict(
            schema="walker-trajectory-review-result-v1", trajectories=[dict(N=8, bins=[{}])]
        )

    monkeypatch.setattr(trajectory_review, "analyze", analyze)
    # The suite itself imports other research modules. Test the guard separately.
    monkeypatch.setattr(execution, "_assert_metadata_only", lambda: None)
    monkeypatch.setattr(execution, "_rss", lambda: 1024)
    return calls


def complete_in_process(monkeypatch, out, after=None):
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        execution.worker(out)
        if after:
            after(out / "result.json")
        kwargs["stdout"].write(b"synthetic worker\n")
        return SimpleNamespace(pid=123456, wait=lambda timeout: 0, poll=lambda: 0)

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    return calls


def test_prepare_is_create_only_and_copies_bound_inputs(prepared_tree):
    out, config, root = prepared_tree()
    saved = {role: (root / row["source"]).read_bytes() for role, row in config["inputs"].items()}
    checked, seal = execution.verify_inputs(out)
    assert checked == config
    for role, name in seal["roles"].items():
        assert (out / name).read_bytes() == saved[role]
        assert file_binding(out / name) == seal["files"][name]
    assert not any(out.rglob("*.pt"))
    before = sha(out / "input_manifest.json")
    with pytest.raises(FileExistsError):
        execution.prepare(out, root)
    assert sha(out / "input_manifest.json") == before
    assert saved == {
        role: (root / row["source"]).read_bytes() for role, row in config["inputs"].items()
    }


@pytest.mark.parametrize("target", ["source", "copied_input", "input_seal"])
def test_mutated_frozen_evidence_rejected(prepared_tree, target):
    out, _, _ = prepared_tree()
    if target == "source":
        path = out / "source/README.md"
    elif target == "copied_input":
        path = out / "inputs/matrix.json"
    else:
        path = out / "input_manifest.json"
        value = json.loads(path.read_text())
        value["budgets"]["raw_array_reads"] = 1
        path.write_text(json.dumps(value))
    if target != "input_seal":
        path.write_text(path.read_text() + "changed")
    with pytest.raises(ValueError, match="changed|differs"):
        execution.verify_inputs(out)
    assert not (out / "worker.lock").exists()


def test_original_mutation_after_copy_does_not_rebind_input(prepared_tree):
    out, config, root = prepared_tree()
    (root / config["inputs"]["matrix"]["source"]).write_text("changed original")
    assert execution.verify_inputs(out)[0] == config


def test_worker_is_create_only_and_accounts_exact_zero_work(prepared_tree, monkeypatch):
    out, config, _ = prepared_tree()
    calls = synthetic_analysis(monkeypatch)
    durable_json(out / "launch_receipt.json", owner_receipt(out, config))
    result = execution.worker(out)
    assert result["execution_complete"]
    assert result["error"] is None
    assert len(calls) == 1
    assert result["work"] == dict(
        input_files=9,
        input_bytes=sum(v["bytes"] for v in config["inputs"].values()),
        metric_records=1,
        bins=1,
        source_control_rows_covered=8,
        **{key: 0 for key in execution.ZERO_WORK},
    )
    assert (
        result["storage_bytes_before_result"]
        == execution._size(out) - (out / "result.json").stat().st_size
    )
    before = sha(out / "result.json")
    with pytest.raises(FileExistsError):
        execution.worker(out)
    assert len(calls) == 1 and sha(out / "result.json") == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "foreign"),
        ("command", []),
        ("timeout_seconds", 999),
        ("retries_permitted", True),
        ("source_manifest_sha256", "foreign"),
    ],
)
def test_worker_rejects_unbound_launch_before_analysis(prepared_tree, monkeypatch, field, value):
    out, config, _ = prepared_tree()
    calls = synthetic_analysis(monkeypatch)
    receipt = owner_receipt(out, config)
    receipt[field] = value
    durable_json(out / "launch_receipt.json", receipt)
    with pytest.raises(ValueError, match="owner"):
        execution.worker(out)
    assert not calls and not (out / "worker.lock").exists()


@pytest.mark.parametrize(
    "where,key,value",
    [
        ("platform", "python", "wrong"),
        ("platform", "numpy", "wrong"),
        ("platform", "device", "cuda"),
        ("budgets", "raw_array_reads", 1),
        ("budgets", "gpu_operations", False),
    ],
)
def test_runtime_and_zero_work_declarations_are_gated(
    prepared_tree, monkeypatch, where, key, value
):
    out, config, _ = prepared_tree(lambda recipe: recipe[where].update({key: value}))
    calls = synthetic_analysis(monkeypatch)
    durable_json(out / "launch_receipt.json", owner_receipt(out, config))
    result = execution.worker(out)
    assert not result["execution_complete"]
    assert result["error"]["type"] == "ValueError"
    assert not calls
    assert all(result["work"][key] == 0 for key in execution.ZERO_WORK)


def test_forbidden_import_guard_is_prefix_aware(monkeypatch):
    monkeypatch.setattr(
        execution, "sys", SimpleNamespace(modules={"numpy": object(), "torchlike": object()})
    )
    execution._assert_metadata_only()
    execution.sys.modules["src.gps.mppi_gps.policy.extra"] = object()
    with pytest.raises(RuntimeError, match="imported"):
        execution._assert_metadata_only()


def test_launcher_binds_success_streams_storage_and_never_retries(prepared_tree, monkeypatch):
    out, config, _ = prepared_tree()
    analyses = synthetic_analysis(monkeypatch)
    calls = complete_in_process(monkeypatch, out)
    receipt = execution.launch(out)
    assert receipt["execution_complete"] and receipt["exit_code"] == 0
    assert receipt["child_started"] and receipt["child_reaped"]
    assert not receipt["kill_attempted"] and not receipt["cleanup_errors"]
    assert receipt["result_error"] is None
    assert receipt["result_sha256"] == sha(out / "result.json")
    assert (
        receipt["stored_bytes"] == execution._size(out) - (out / "worker_exit.json").stat().st_size
    )
    assert receipt["streams"] == {
        name: file_binding(out / name) for name in ("worker.stdout", "worker.stderr")
    }
    command, kwargs = calls[0]
    assert command == execution._command(out)
    assert kwargs["start_new_session"] and kwargs["cwd"] == out / "source"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
    assert json.loads((out / "launch_receipt.json").read_text())["retries_permitted"] is False
    with pytest.raises(FileExistsError):
        execution.launch(out)
    assert len(calls) == len(analyses) == 1
    assert receipt["stored_bytes"] + 4096 <= config["budgets"]["storage_bytes"]


@pytest.mark.parametrize("mutation", ["json", "list", "hash", "work", "worker_error"])
def test_launcher_rejects_malformed_or_foreign_results(prepared_tree, monkeypatch, mutation):
    out, _, _ = prepared_tree()
    synthetic_analysis(monkeypatch)

    def after(path):
        result = json.loads(path.read_text())
        if mutation == "json":
            path.write_text("{broken")
            return
        if mutation == "list":
            result = []
        elif mutation == "hash":
            result["config_sha256"] = "foreign"
        elif mutation == "work":
            result["work"]["new_physics_controls"] = 1
        else:
            result["error"] = dict(type="SyntheticError", message="failed")
        path.write_text(json.dumps(result))

    calls = complete_in_process(monkeypatch, out, after)
    receipt = execution.launch(out)
    assert not receipt["execution_complete"] and len(calls) == 1
    assert receipt["child_reaped"]
    assert (receipt["result_error"] is None) == (mutation == "worker_error")
    assert (out / "worker_exit.json").exists()


def test_launcher_reports_storage_budget_failure(prepared_tree, monkeypatch):
    out, _, _ = prepared_tree(lambda config: config["budgets"].update(storage_bytes=1))
    synthetic_analysis(monkeypatch)
    complete_in_process(monkeypatch, out)
    receipt = execution.launch(out)
    assert not receipt["execution_complete"]
    assert receipt["stored_bytes"] > 1
    assert (
        receipt["stored_bytes"] == execution._size(out) - (out / "worker_exit.json").stat().st_size
    )


def test_timeout_kills_and_reaps_actual_synthetic_child_once(prepared_tree, monkeypatch):
    out, _, _ = prepared_tree(
        lambda config: config["budgets"].update(wall_seconds=0.03),
        script="import time\ntime.sleep(10)\n",
    )
    children, original = [], subprocess.Popen

    def popen(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    receipt = execution.launch(out)
    assert receipt["timeout"] and not receipt["execution_complete"]
    assert receipt["error"]["type"] == "TimeoutError"
    assert receipt["child_started"] and receipt["child_reaped"] and receipt["kill_attempted"]
    assert not receipt["cleanup_errors"]
    assert receipt["exit_code"] == -signal.SIGKILL
    assert len(children) == 1 and children[0].poll() == -signal.SIGKILL
    with pytest.raises(FileExistsError):
        execution.launch(out)
    assert len(children) == 1


def test_failed_cleanup_is_retained_without_claiming_reap(prepared_tree, monkeypatch):
    out, _, _ = prepared_tree()

    def wait(timeout):
        raise subprocess.TimeoutExpired("synthetic", timeout)

    def kill(*args):
        raise PermissionError("synthetic denied signal")

    monkeypatch.setattr(
        execution.subprocess,
        "Popen",
        lambda *a, **k: SimpleNamespace(pid=123, wait=wait, poll=lambda: None),
    )
    monkeypatch.setattr(execution.os, "killpg", kill)
    receipt = execution.launch(out)
    assert not receipt["execution_complete"] and not receipt["child_reaped"]
    assert [error["phase"] for error in receipt["cleanup_errors"]] == ["kill", "reap"]
    assert receipt["exit_code"] is None
    assert (out / "worker_exit.json").exists()
