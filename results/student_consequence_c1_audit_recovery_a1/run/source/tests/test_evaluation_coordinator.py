"""Synthetic off-host coordinator lifecycle; no network, HDF or control work."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.gps.mppi_gps import source_evaluation_coordinator as coordinator
from src.gps.mppi_gps.durability import durable_json, file_binding


@pytest.fixture
def exchange(tmp_path, monkeypatch):
    local, remote, mirror = (tmp_path / name for name in ("local", "remote", "mirror"))
    local.mkdir()
    remote.mkdir()
    config = dict(
        platform=dict(host="worker", native_interpreter="python"),
        budgets=dict(wall_seconds=100),
        evaluation_order=["mppi_s0"],
        primary_seeds=[7],
        confirmation_seeds=[8],
        checkpoints=dict(mppi_s0=dict(sha256="a" * 64, policy_hash="b" * 64)),
    )
    durable_json(local / "config.json", config)
    for name in ("input_manifest.json", "manifest.json"):
        durable_json(local / name, {})
    pins = {
        key: file_binding(local / name)["sha256"]
        for key, name in (
            ("config_sha256", "config.json"),
            ("input_manifest_sha256", "input_manifest.json"),
            ("source_manifest_sha256", "manifest.json"),
        )
    }
    launch = dict(
        boot_id="boot",
        owner=dict(pid=11, start_ticks="1", state="S"),
        deadline_monotonic=time.monotonic() + 1000,
        **pins,
    )
    durable_json(remote / "launch_receipt.json", launch)
    worker = dict(
        boot_id="boot",
        pid=22,
        launch_receipt_sha256=coordinator.sha(remote / "launch_receipt.json"),
        **pins,
    )
    durable_json(remote / "worker_started.json", worker)
    durable_json(
        remote / "worker_process.json", dict(boot_id="boot", pid=22, start_ticks="2", state="S")
    )
    state = dict(mode=None, closed=False, probes=0, acks=[], verified_segments=[], commands=[])
    publications = []
    cumulative = {}
    for index, role in enumerate(coordinator.publication_roster(config)):
        ordinal = index - 1
        identifier = role["identifier"]
        relative = "evidence/loads" if index == 0 else "evidence/episodes/" + identifier
        segment = remote / relative
        segment.mkdir(parents=True)
        (segment / "raw.h5").write_bytes(b"closed fake arrays " + identifier.encode())
        (segment / "ledger.jsonl").write_bytes(b"closed fake ledger\n")
        owner = {key: value for key, value in role.items() if key != "identifier"} | dict(
            ordinal=ordinal,
            boot_id="boot",
            boot_id_after="boot",
            owner_pid=22,
            launch_receipt_sha256=coordinator.sha(remote / "launch_receipt.json"),
            worker_started_sha256=coordinator.sha(remote / "worker_started.json"),
            cumulative_before=deepcopy(cumulative),
            checkpoint_sha256="a" * 64,
            policy_hash="b" * 64,
            optimizer_fingerprint="c" * 64,
            initial_context_fingerprint="d" * 64,
            **pins,
        )
        work = dict(
            episodes=int(index > 0),
            completed_checkpoint_loads=int(index == 0),
            completed_live_controls=int(index > 0),
        )
        cumulative = {key: cumulative.get(key, 0) + value for key, value in work.items()}
        receipt = dict(
            schema="walker-durable-loads-v1" if index == 0 else "walker-durable-episode-v1",
            ownership=owner,
            episode={},
            work=work,
            cumulative=deepcopy(cumulative),
            files={name: file_binding(segment / name) for name in ("raw.h5", "ledger.jsonl")},
        )
        durable_json(segment / "receipt.json", receipt)
        request = dict(
            schema="walker-s1.1-publication-v1",
            identifier=identifier,
            ordinal=ordinal,
            segment=relative,
            worker_boot_id="boot",
            receipt_sha256=coordinator.sha(segment / "receipt.json"),
            files={
                name: file_binding(segment / name)
                for name in ("raw.h5", "ledger.jsonl", "receipt.json")
            },
        )
        durable_json(remote / "publications" / (identifier + ".json"), request)
        publications.append(request)
    monkeypatch.setattr(coordinator, "read_boot_id", lambda: "boot")
    monkeypatch.setattr(coordinator, "process_alive", lambda identity: not state["closed"])
    monkeypatch.setattr(coordinator.socket, "gethostname", lambda: "offhost-Mac")
    monkeypatch.setattr(
        coordinator,
        "verify_segment",
        lambda path, sha: state["verified_segments"].append((path, sha)),
    )

    def finish():
        durable_json(remote / "result.json", dict(execution_complete=True))
        durable_json(
            remote / "worker_exit.json",
            dict(
                boot_id="boot",
                boot_id_after="boot",
                result_sha256=coordinator.sha(remote / "result.json"),
                exit_code=0,
            ),
        )
        durable_json(
            remote / "owner_exit.json",
            dict(
                boot_id="boot",
                boot_id_after="boot",
                worker_exit_sha256=coordinator.sha(remote / "worker_exit.json"),
                execution_complete=True,
                error=None,
            ),
        )
        state["closed"] = True

    def response(command, stdout="", code=0, stderr=""):
        return subprocess.CompletedProcess(command, code, stdout, stderr)

    def transport(command, *, input=None, capture_output, text, timeout):
        assert capture_output and text and timeout > 0
        state["commands"].append(command)
        if command[0] == "ssh":
            words = shlex.split(command[-1])
            identifier = words[words.index("--identifier") + 1]
            if "--probe" in words:
                if state["mode"] == "timeout" and not state["probes"]:
                    raise subprocess.TimeoutExpired(
                        command, timeout, output="paid partial", stderr="synthetic slow transport"
                    )
                if state["mode"] == "oserror" and not state["probes"]:
                    raise OSError("synthetic disconnected host")
                status = coordinator.remote_probe(remote, identifier, wait_seconds=0)
                state["probes"] += 1
                if state["probes"] == 2:
                    if state["mode"] == "boot":
                        status["boot_id"] = "new-boot"
                    elif state["mode"] == "vanished":
                        status["owner_alive"] = False
                    elif state["mode"] == "head":
                        status["metadata"]["worker_started.json"]["value"]["pid"] = 33
                return response(command, json.dumps(status))
            assert "--write-ack" in words
            if state["mode"] == "late_ack":
                monkeypatch.setattr(
                    coordinator.time, "monotonic", lambda: launch["deadline_monotonic"] + 1
                )
            coordinator.write_ack(remote, identifier, json.loads(input))
            state["acks"].append(identifier)
            if identifier == publications[-1]["identifier"]:
                finish()
            return response(command)
        assert command[0] == "rsync"
        if state["mode"] == "transport":
            return response(command, code=23, stderr="synthetic failed copy")
        source = Path(command[-2].split(":", 1)[1])
        target = Path(command[-1])
        if source == remote:
            assert "--ignore-existing" in command
            for path in remote.rglob("*"):
                relative = path.relative_to(remote)
                if relative.parts[0] in ("source", "inputs", "evidence"):
                    continue
                destination = target / relative
                if path.is_file() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
            if state["mode"] == "final":
                (target / "result.json").write_text('{"changed": true}\n')
        else:
            for path in source.iterdir():
                shutil.copy2(path, target / path.name)
            if state["mode"] == "tamper":
                (target / "raw.h5").write_bytes(b"changed by failed transport")
        return response(command)

    monkeypatch.setattr(coordinator.subprocess, "run", transport)
    return SimpleNamespace(
        local=local,
        remote=remote,
        mirror=mirror,
        config=config,
        state=state,
        publications=publications,
    )


def execute(exchange):
    return coordinator.coordinate(exchange.local, exchange.mirror, "worker", exchange.remote)


def test_full_synthetic_coordinator_closes_three_mirrored_packets_and_exact_head_chain(exchange):
    result = execute(exchange)
    assert result["execution_complete"] and result["error"] is None
    assert [row["identifier"] for row in result["records"]] == [
        row["identifier"] for row in exchange.publications
    ]
    assert len(exchange.state["acks"]) == 3 and len(exchange.state["verified_segments"]) == 2
    assert result["new_physics"] == result["new_neural_rows"] == result["new_optimizer_steps"] == 0
    for publication in exchange.publications:
        relative = publication["segment"]
        for name, expected in publication["files"].items():
            assert file_binding(exchange.mirror / relative / name) == expected
        identifier = publication["identifier"]
        ack = json.loads((exchange.remote / "acks" / (identifier + ".json")).read_text())
        commit = exchange.mirror / "local_coordinator/commits" / (identifier + ".json")
        assert ack["mirror_commit_sha256"] == coordinator.sha(commit)
    assert (exchange.mirror / "global/launch_receipt.json").read_bytes() == (
        exchange.remote / "launch_receipt.json"
    ).read_bytes()
    assert json.loads((exchange.mirror / "coordinator_exit.json").read_text()) == result


@pytest.mark.parametrize("mode", ["timeout", "oserror", "transport", "tamper", "late_ack"])
def test_transport_or_copy_failure_is_paid_preserved_and_never_acknowledged(exchange, mode):
    exchange.state["mode"] = mode
    result = execute(exchange)
    assert not result["execution_complete"] and result["error"]
    assert not exchange.state["acks"] and not result["records"]
    assert not (exchange.remote / "acks").exists()
    assert result["calls"]
    if mode == "timeout":
        assert result["calls"][0]["timeout"] is True
        assert result["calls"][0]["stdout_bytes"] == len("paid partial")
    elif mode == "oserror":
        assert result["calls"][0]["error_type"] == "OSError"
    elif mode == "transport":
        assert result["calls"][-1]["exit_code"] == 23
    elif mode == "tamper":
        assert "bytes differ" in result["error"]["message"]
    else:
        assert result["error"]["type"] == "TimeoutError"
    assert (exchange.mirror / "coordinator_exit.json").is_file()


@pytest.mark.parametrize("mode", ["boot", "vanished", "head"])
def test_changed_boot_owner_or_head_keeps_prior_packet_and_stops_without_restart(exchange, mode):
    exchange.state["mode"] = mode
    result = execute(exchange)
    assert not result["execution_complete"]
    assert exchange.state["acks"] == ["loads"]
    assert len(result["records"]) == 1
    first = exchange.publications[0]
    assert (
        file_binding(exchange.mirror / first["segment"] / "receipt.json")
        == first["files"]["receipt.json"]
    )
    assert not (
        exchange.remote / "acks" / (exchange.publications[1]["identifier"] + ".json")
    ).exists()
    assert exchange.state["probes"] == 2


@pytest.mark.parametrize(
    "field,value", [("seed", 99), ("checkpoint_sha256", "f" * 64), ("cumulative_before", {})]
)
def test_self_consistent_wrong_reset_actor_or_cumulative_chain_is_rejected(exchange, field, value):
    request = exchange.publications[1]
    segment = exchange.remote / request["segment"]
    path = segment / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["ownership"][field] = value
    path.unlink()
    durable_json(path, receipt)
    request["receipt_sha256"] = coordinator.sha(path)
    request["files"]["receipt.json"] = file_binding(path)
    request_path = exchange.remote / "publications" / (request["identifier"] + ".json")
    request_path.unlink()
    durable_json(request_path, request)
    result = execute(exchange)
    assert not result["execution_complete"]
    assert exchange.state["acks"] == ["loads"]
    assert len(result["records"]) == 1


def test_final_result_hash_failure_retains_all_previously_acknowledged_segments(exchange):
    exchange.state["mode"] = "final"
    result = execute(exchange)
    assert not result["execution_complete"]
    assert len(exchange.state["acks"]) == len(result["records"]) == 3
    assert "bindings differ" in result["error"]["message"]
    for publication in exchange.publications:
        assert (exchange.mirror / publication["segment"] / "receipt.json").is_file()


@pytest.mark.parametrize("directory", ["evidence/loads", "evidence"])
def test_off_host_fsync_failure_preserves_packet_without_commit_or_ack(
    exchange, monkeypatch, directory
):
    original = coordinator.fsync_directory

    def sync(path):
        copied = any(command[0] == "rsync" for command in exchange.state["commands"])
        if Path(path) == exchange.mirror / directory and copied:
            raise OSError("synthetic independent disk durability failure")
        return original(path)

    monkeypatch.setattr(coordinator, "fsync_directory", sync)
    result = execute(exchange)
    assert not result["execution_complete"]
    assert not result["records"] and not exchange.state["acks"]
    assert not (exchange.mirror / "local_coordinator/commits/loads.json").exists()
    assert not (exchange.remote / "acks").exists()
    first = exchange.publications[0]
    assert (
        file_binding(exchange.mirror / first["segment"] / "receipt.json")
        == first["files"]["receipt.json"]
    )
    assert "durability failure" in result["error"]["message"]
