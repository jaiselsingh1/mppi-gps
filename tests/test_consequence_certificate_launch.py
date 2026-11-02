"""Tiny A2 ownership and closure fixtures; no original payload or science."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from src.gps.mppi_gps.experiments import consequence_certificate_closure as module

RECIPE = (
    Path(__file__).resolve().parents[1]
    / "configs/studies/student_consequence_c1_certificate_closure_a2.json"
)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def binding(path):
    payload = path.read_bytes()
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def clean(pid=4242):
    return dict(
        exit_code=0,
        pid=pid,
        leader_reaped=True,
        stopping_reason=None,
        process_group_kill_requested=False,
        unknown_partial_work_possible=False,
        wall_seconds=0.01,
    )


def bundle(tmp_path, monkeypatch):
    run = tmp_path / "a2"
    source = run / "source"
    source.mkdir(parents=True)
    for name in module.SOURCE_FILES:
        (source / name).write_text("# synthetic frozen source\n")
    protocol = json.loads(RECIPE.read_text())
    for key in module.ROOT_KEYS:
        protocol["locations"][key] = str(tmp_path / key)
    save(run / "protocol.json", protocol)
    save(
        run / "source_manifest.json",
        dict(
            schema="walker-c1-certificate-source-v1",
            protocol_sha256=binding(run / "protocol.json")["sha256"],
            source_commit="a" * 40,
            files={name: binding(source / name) for name in module.SOURCE_FILES},
        ),
    )
    monkeypatch.setattr(module, "SOURCE_DIR", source)
    return run, protocol


def result(run, protocol):
    value = dict(module.RESULT_FLAGS)
    value.update(
        protocol_sha256=hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        new_scientific_work=protocol["new_scientific_work"],
    )
    save(run / "result/certificate_result.json", value)


def coordinated(tmp_path, monkeypatch):
    run, protocol = bundle(tmp_path, monkeypatch)
    calls = []

    def bounded(*args, **kwargs):
        calls.append(kwargs)
        kwargs["output"].write_bytes(b"")
        result(run, protocol)
        return clean()

    monkeypatch.setattr(module, "_components", lambda: (None, bounded))
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda *_: None, raising=False)
    monkeypatch.setenv("PYTHONPATH", "/not/the/frozen/bundle")
    assert module.coordinate(run) == 0
    owner = json.loads((run / "ownership.json").read_text())
    save(run / "coordinator_exit.json", clean(owner["coordinator_pid"]))
    return run, protocol, calls


def test_actual_recipe_is_zero_science_and_finite():
    protocol = json.loads(RECIPE.read_text())
    module.validate_protocol(protocol)
    for key in module.SCIENCE_KEYS:
        changed = deepcopy(protocol)
        changed["new_scientific_work"][key] = 1
        with pytest.raises(ValueError):
            module.validate_protocol(changed)


def test_coordinate_only_creates_candidate_and_finalization_binds_five_files(tmp_path, monkeypatch):
    run, _, calls = coordinated(tmp_path, monkeypatch)
    candidate = json.loads((run / "admission_candidate.json").read_text())
    assert candidate["audit_recovery_resolved"] is False
    assert not (run / "admission.json").exists()
    assert "PYTHONPATH" not in calls[0]["environment"]
    assert calls[0]["wall_seconds"] <= 30
    admission = module.finalize(run)
    assert admission["audit_recovery_resolved"] is True
    assert admission["original_A1_admission_failed"] is True
    assert admission["original_C1_child_exit_code"] is None
    assert set(admission["bindings"]) == {
        "result",
        "protocol",
        "source_manifest",
        "worker_exit",
        "coordinator_exit",
    }
    for value in admission["bindings"].values():
        assert binding(run / value["file"]) == {k: value[k] for k in ("bytes", "sha256")}
    assert not (run / "finalizer_exit.json").exists()  # external observer still must write it
    with pytest.raises(FileExistsError):
        module.finalize(run)
    with pytest.raises(FileExistsError):
        module.coordinate(run)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("exit_code", 1),
        ("leader_reaped", False),
        ("stopping_reason", "coordinator_exception"),
        ("unknown_partial_work_possible", True),
        ("pid", 99999),
        ("wall_seconds", 46),
    ],
)
def test_failed_or_wrong_external_coordinator_cannot_finalize(tmp_path, monkeypatch, field, value):
    run, _, _ = coordinated(tmp_path, monkeypatch)
    outer = json.loads((run / "coordinator_exit.json").read_text())
    outer[field] = value
    save(run / "coordinator_exit.json", outer)
    with pytest.raises(ValueError):
        module.finalize(run)
    assert not (run / "admission.json").exists()


def test_missing_coordinator_exit_never_becomes_an_admission(tmp_path, monkeypatch):
    run, _, _ = coordinated(tmp_path, monkeypatch)
    (run / "coordinator_exit.json").unlink()
    with pytest.raises(FileNotFoundError):
        module.finalize(run)
    assert not (run / "admission.json").exists()


@pytest.mark.parametrize("target", ["observer.py", "closure_checks.py", "watchdog.py"])
def test_changed_bundle_stops_before_creating_owner(tmp_path, monkeypatch, target):
    run, _ = bundle(tmp_path, monkeypatch)
    (run / "source" / target).write_text("changed source")
    with pytest.raises(ValueError, match="frozen source differs"):
        module.coordinate(run)
    assert not (run / "ownership.json").exists()


def test_bundle_symlinks_are_not_read(tmp_path, monkeypatch):
    run, _ = bundle(tmp_path, monkeypatch)
    target = run / "source/closure_checks.py"
    target.unlink()
    target.symlink_to(run / "protocol.json")
    with pytest.raises(ValueError, match="symlinks"):
        module.coordinate(run)
    assert not (run / "ownership.json").exists()


@pytest.mark.parametrize("key", module.ROOT_KEYS)
def test_all_four_preserved_roots_are_protected_before_write(tmp_path, monkeypatch, key):
    run, protocol = bundle(tmp_path, monkeypatch)
    protocol["locations"][key] = str(run.parent)
    save(run / "protocol.json", protocol)
    before = {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="preserved input tree"):
        module.coordinate(run)
    assert before == {str(p): p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_exception_receipt_is_saved_without_retry(tmp_path, monkeypatch):
    run, _ = bundle(tmp_path, monkeypatch)
    retained = clean()
    retained.update(stopping_reason="coordinator_exception", unknown_partial_work_possible=True)
    calls = []

    def bounded(*args, **kwargs):
        calls.append(True)
        error = RuntimeError("synthetic observer fault")
        error.process_receipt = retained
        raise error

    monkeypatch.setattr(module, "_components", lambda: (None, bounded))
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda *_: None, raising=False)
    with pytest.raises(RuntimeError):
        module.coordinate(run)
    assert json.loads((run / "worker_exit.json").read_text()) == retained
    assert not (run / "admission_candidate.json").exists()
    with pytest.raises(FileExistsError):
        module.coordinate(run)
    assert len(calls) == 1


def test_changed_candidate_or_missing_required_null_blocks_finalization(tmp_path, monkeypatch):
    run, _, _ = coordinated(tmp_path, monkeypatch)
    candidate = json.loads((run / "admission_candidate.json").read_text())
    candidate["original_C1_child_exit_code"] = 0
    save(run / "admission_candidate.json", candidate)
    with pytest.raises(ValueError, match="candidate differs"):
        module.finalize(run)
    assert not (run / "admission.json").exists()


def test_reader_reserves_before_payload_and_import_guard_rejects_science(tmp_path, monkeypatch):
    path = tmp_path / "tiny"
    path.write_bytes(b"12345")
    calls = []
    original = module.os.open

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module.os, "open", record)
    with pytest.raises(ValueError, match="read reservation"):
        module._Reader(4, 1).read(path)
    assert not calls
    guard = module._NoScience()
    for name in ("torch.nn", "mujoco", "gymnasium", "jax"):
        with pytest.raises(ValueError):
            guard.find_spec(name)
    assert guard.find_spec("json") is None


def test_standalone_cli_help_uses_no_repository_or_scientific_imports(tmp_path):
    script = tmp_path / "observer.py"
    script.write_bytes(Path(module.__file__).read_bytes())
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONPATH", None)
    child = subprocess.run(
        [sys.executable, "-B", str(script), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert child.returncode == 0 and "coordinate" in child.stdout and "finalize" in child.stdout


def test_cold_standalone_worker_runs_through_real_watchdog_without_original_inputs(
    tmp_path, monkeypatch
):
    from src.gps.mppi_gps import proposal_launch

    run, protocol = bundle(tmp_path, monkeypatch)
    (run / "source/observer.py").write_bytes(Path(module.__file__).read_bytes())
    watchdog = Path(proposal_launch.__file__).read_text().split("\ndef launch(", 1)[0]
    watchdog = watchdog.replace("from .provenance import sha, write\n", "")
    (run / "source/watchdog.py").write_text(watchdog)
    stub = f"""from pathlib import Path
import hashlib,json
def close_certificates(protocol,out,*,protocol_read_bytes=0):
    out=Path(out);out.mkdir(exist_ok=False)
    result={module.RESULT_FLAGS!r}
    result.update(protocol_sha256=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(),new_scientific_work=protocol["new_scientific_work"],work={{"read_bytes":protocol_read_bytes}})
    (out/"certificate_result.json").write_text(json.dumps(result)+"\\n")
    return result
"""
    (run / "source/closure_checks.py").write_text(stub)
    manifest = json.loads((run / "source_manifest.json").read_text())
    manifest["files"] = {name: binding(run / "source" / name) for name in module.SOURCE_FILES}
    save(run / "source_manifest.json", manifest)
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: {0}, raising=False)
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda *_: None, raising=False)
    assert module.coordinate(run) == 0
    assert json.loads((run / "worker_exit.json").read_text())["exit_code"] == 0
    prelude = json.loads((run / "worker_prelude.json").read_text())
    certificate = json.loads((run / "result/certificate_result.json").read_text())
    assert certificate["work"]["read_bytes"] == prelude["prelude_bytes_also_charged_by_core"]
    assert {p.name for p in (run / "source").iterdir()} == module.SOURCE_FILES
    assert not (run / "admission.json").exists()
    assert all(not Path(protocol["locations"][key]).exists() for key in module.ROOT_KEYS)
