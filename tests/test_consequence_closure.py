"""A2 reuses immutable certificates and cannot admit or rerun A1 itself."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.audits.consequence_closure import ClosureError, close_certificates


@pytest.fixture(autouse=True)
def fresh_worker_rss_context(monkeypatch):
    # API tests model the fresh, science-free A2 worker. The shared pytest
    # process can retain unrelated lifetime Torch/Matplotlib memory peaks.
    # The standalone CLI subprocess below still checks its real RSS.
    monkeypatch.setattr("src.gps.mppi_gps.audits.consequence_closure._rss", lambda: 32 * 1024**2)


def encode(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )


def binding(payload):
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def fixture(tmp_path):
    roots = {name: tmp_path / name for name in ("c1_run", "c1_artifacts", "a1_run", "a1_artifacts")}
    for root in roots.values():
        root.mkdir()
    anchors = {}

    def save(name, root, relative, value):
        payload = value if isinstance(value, bytes) else encode(value)
        path = roots[root] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        anchors[name] = dict(root=root, file=relative, **binding(payload))
        return binding(payload)

    original = dict(
        schema="walker-student-consequence-c1.0-audit-v1",
        passed=True,
        full_scientific_audit_passed=True,
        scientific_contrast_completed=True,
        evidence_integrity_passed=True,
        config_sha256="0" * 64,
        worker_result_sha256="1" * 64,
    )
    old_binding = save(
        "original_scientific_receipt", "c1_artifacts", "audit_001/result/audit.json", original
    )
    failure = dict(
        schema="walker-c1-audit-launcher-failure-v1",
        overall_audit_closure_passed=False,
        audit_coordinator_failed=True,
        original_audit_exit_exists=False,
        original_child_exit_code=None,
        coordinator_exit_code=1,
        scientific_receipt=old_binding,
    )
    failure_binding = save(
        "original_failure", "c1_artifacts", "audit_launcher_failure.json", failure
    )
    science = dict.fromkeys(
        (
            "neural_rows",
            "prediction_calls",
            "checkpoint_tensor_loads",
            "native_controls",
            "physics_substeps",
            "backward_rows",
            "optimizer_steps",
            "resets",
            "restores",
            "mppi_queries",
        ),
        0,
    )
    config = dict(
        schema="walker-c1-audit-recovery-protocol-v1",
        study="C1.0-A1",
        user_authorized=True,
        automatic_followup=False,
        new_scientific_work=science,
        anchors=dict(scientific_receipt=old_binding, failure=failure_binding),
    )
    config_binding = save("a1_config", "a1_run", "config.json", config)
    manifest = dict(
        command="student-consequence-audit-recovery-c1.0-a1",
        input_sha256=None,
        config_sha256=config_binding["sha256"],
        files=[dict(file="src/gps/mppi_gps/audits/consequence_recovery.py", sha256="2" * 64)],
    )
    manifest_binding = save("a1_manifest", "a1_run", "manifest.json", manifest)
    inputs = dict(
        schema="walker-c1-audit-recovery-inputs-v1",
        protocol_sha256=config_binding["sha256"],
        source_commit="a" * 40,
        original_inputs_are_read_only=True,
        anchors=config["anchors"],
    )
    input_binding = save("a1_input_manifest", "a1_run", "input_manifest.json", inputs)
    owner = dict(
        coordinator_pid=123,
        config_sha256=config_binding["sha256"],
        manifest_sha256=manifest_binding["sha256"],
        input_manifest_sha256=input_binding["sha256"],
    )
    save("a1_ownership", "a1_run", "ownership.json", owner)
    files = dict(
        schema="walker-c1-recovery-verified-files-v1",
        files={
            "run/config.json": dict(bytes=10, sha256="0" * 64),
            "run/worker_result.json": dict(bytes=20, sha256="1" * 64),
        },
    )
    map_binding = save("a1_verified_files", "a1_run", "result/verified_files.json", files)
    canonical_config = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    result = dict(
        schema="walker-c1-audit-recovery-result-v1",
        recovery_checks_passed=True,
        admitted_original_scientific_checks=True,
        retained_numerical_certification=True,
        original_audit_coordinator_failed=True,
        original_overall_audit_closure_passed=False,
        original_child_exit_code=None,
        new_scientific_work=science,
        original_scientific_receipt=old_binding,
        protocol_sha256=canonical_config,
        files_catalog=dict(file="verified_files.json", **map_binding),
        verified_file_count=2,
    )
    save("a1_result", "a1_run", "result/recovery.json", result)
    exit_value = dict(
        exit_code=0,
        leader_reaped=True,
        stopping_reason="coordinator_exception",
        process_group_kill_requested=True,
        unknown_partial_work_possible=True,
        error=dict(type="RuntimeError", message="Pinned Linux worker RSS could not be measured"),
        pid=124,
        command=[
            "/python",
            str(roots["a1_run"] / "source/scripts/train_mppi_gps.py"),
            "study",
            "student-consequence",
            "audit-recovery",
            "worker",
            "--run",
            str(roots["a1_run"]),
        ],
    )
    save("a1_worker_exit", "a1_run", "exit.json", exit_value)
    save(
        "a1_launcher_trace",
        "a1_run",
        "stdout.txt",
        b"Traceback: Pinned Linux worker RSS could not be measured\n",
    )
    save(
        "a1_external_failure",
        "a1_artifacts",
        "failure.json",
        dict(
            schema="walker-c1-a1-external-failure-v1",
            external_tool_exit_code=1,
            original_coordinator_exit_available=False,
            original_candidate_exists=False,
            original_admission_exists=False,
            worker_exit_recorded_zero=True,
            worker_coordinator_exception_flags_preserved=True,
            error_type="RuntimeError",
            error_message="Pinned Linux worker RSS could not be measured",
        ),
    )
    protocol = dict(
        schema="walker-c1-certificate-closure-protocol-v1",
        study="C1.0-A2",
        user_authorized=True,
        automatic_followup=False,
        locations={k: str(v) for k, v in roots.items()},
        anchors=anchors,
        new_scientific_work=science,
        expected_a1_verified_files=2,
        expected_a1_source_commit="a" * 40,
        expected_a1_auditor_sha256="2" * 64,
        limits=dict(
            wall_seconds=30,
            read_bytes=32 * 1024**2,
            rss_bytes=512 * 1024**2,
            output_bytes=4 * 1024**2,
            max_file_bytes=2 * 1024**2,
            max_files=32,
        ),
    )
    return protocol, roots


def reseal(protocol, roots, name, mutate):
    anchor = protocol["anchors"][name]
    path = roots[anchor["root"]] / anchor["file"]
    value = json.loads(path.read_bytes())
    mutate(value)
    payload = encode(value)
    path.write_bytes(payload)
    anchor.update(binding(payload))


def test_metadata_only_compensation_keeps_both_failed_gates_historical(tmp_path):
    protocol, roots = fixture(tmp_path)
    result = close_certificates(protocol, tmp_path / "new")
    assert result["certificate_checks_passed"], result.get("error")
    assert result["retained_A1_integrity_certification"]
    assert result["retained_original_numerical_certification"]
    assert result["original_A1_admission_failed"]
    assert result["original_A1_failure_flags_preserved"]
    assert result["new_scientific_work"] == protocol["new_scientific_work"]
    assert "recovery_resolved" not in result
    assert not (roots["a1_run"] / "admission.json").exists()
    assert not (roots["c1_artifacts"] / "audit_001/exit.json").exists()


@pytest.mark.parametrize(
    "name", ["a1_result", "a1_verified_files", "a1_worker_exit", "original_scientific_receipt"]
)
def test_tampering_any_preserved_certificate_fails_without_reexecution(tmp_path, name):
    protocol, roots = fixture(tmp_path)
    anchor = protocol["anchors"][name]
    path = roots[anchor["root"]] / anchor["file"]
    payload = path.read_bytes()
    path.write_bytes(b" " + payload[1:])
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]
    assert result["work"]["read_bytes"] > 0


@pytest.mark.parametrize(
    "fault", ["nonzero_exit", "not_reaped", "erased_failure", "wrong_command", "wrong_parent"]
)
def test_genuine_exit_and_preserved_failure_flags_are_both_required(tmp_path, fault):
    protocol, roots = fixture(tmp_path)

    def mutate(value):
        if fault == "nonzero_exit":
            value["exit_code"] = 1
        elif fault == "not_reaped":
            value["leader_reaped"] = False
        elif fault == "erased_failure":
            value["process_group_kill_requested"] = False
        elif fault == "wrong_command":
            value["command"][5] = "another-worker"
        else:
            value["pid"] = 123

    reseal(protocol, roots, "a1_worker_exit", mutate)
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]


def test_matching_anchor_alone_cannot_change_config_association(tmp_path):
    protocol, roots = fixture(tmp_path)
    reseal(protocol, roots, "a1_config", lambda value: value.update(automatic_followup=True))
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]


def test_wrong_source_commit_fails(tmp_path):
    protocol, _ = fixture(tmp_path)
    protocol["expected_a1_source_commit"] = "b" * 40
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]


@pytest.mark.parametrize(
    "name", ["admission_candidate.json", "admission.json", "coordinator_exit.json"]
)
def test_original_A1_missing_closure_is_never_manufactured_or_accepted(tmp_path, name):
    protocol, roots = fixture(tmp_path)
    (roots["a1_run"] / name).write_text("{}")
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]
    assert "replaced" in result["error"]["message"]


@pytest.mark.parametrize("root_name", ["c1_run", "c1_artifacts", "a1_run", "a1_artifacts"])
def test_all_original_roots_are_read_only_even_for_error_receipts(tmp_path, root_name):
    protocol, roots = fixture(tmp_path)
    out = roots[root_name] / "forbidden"
    with pytest.raises(ClosureError, match="inside preserved"):
        close_certificates(protocol, out)
    assert not out.exists()


def test_budget_failure_retains_paid_metadata_prefix(tmp_path):
    protocol, _ = fixture(tmp_path)
    protocol["limits"]["read_bytes"] = 100
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]
    assert result["work"]["read_bytes"] == 0
    assert (tmp_path / "new/certificate_result.json").is_file()


def test_symlink_and_escape_cannot_change_certificate_root(tmp_path):
    protocol, roots = fixture(tmp_path)
    anchor = protocol["anchors"]["a1_result"]
    path = roots["a1_run"] / anchor["file"]
    copy = tmp_path / "outside.json"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copy)
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]
    assert "Symlink" in result["error"]["message"]


# A small intermediary prevents Linux from charging an unrelated pytest peak to
# the measured CLI. Its own child is still subject to the unchanged 512 MiB cap.
_COLD_LAUNCHER = (
    "import subprocess,sys; raise SystemExit(subprocess.run(sys.argv[1:], timeout=20).returncode)"
)


def run_owned_process(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    """Bound and reap the complete synthetic process tree, including on failure."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def standalone_cli_command(protocol_path: Path, output_path: Path) -> list[str]:
    module = Path(__file__).resolve().parents[1] / "src/gps/mppi_gps/audits/consequence_closure.py"
    code = (
        """import builtins,runpy,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','numpy','mujoco','gymnasium','stable_baselines3','jax'}:
  raise AssertionError('scientific import:'+name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
sys.argv=[MODULE,'--protocol',PROTOCOL,'--out',OUT]
runpy.run_path(MODULE,run_name='__main__')
""".replace("MODULE", repr(str(module)))
        .replace("PROTOCOL", repr(str(protocol_path)))
        .replace("OUT", repr(str(output_path)))
    )
    return [sys.executable, "-I", "-c", code]


def test_import_and_cli_are_standalone_without_science_libraries(tmp_path):
    protocol, _ = fixture(tmp_path)
    path = tmp_path / "protocol.json"
    path.write_bytes(encode(protocol))
    out = tmp_path / "new"
    command = standalone_cli_command(path, out)
    process = run_owned_process(
        [sys.executable, "-I", "-c", _COLD_LAUNCHER, *command],
        timeout=25,
    )
    receipt = out / "certificate_result.json"
    assert process.returncode == 0, (
        f"CLI exited {process.returncode}\nstdout: {process.stdout}"
        f"\nstderr: {process.stderr}\nreceipt: "
        + (receipt.read_text() if receipt.is_file() else "missing")
    )
    actual = json.loads(receipt.read_text())
    assert json.loads(process.stdout) == dict(
        certificate_checks_passed=actual["certificate_checks_passed"], work=actual["work"]
    )
    assert actual["certificate_checks_passed"] is True
    assert actual["new_scientific_work"] == protocol["new_scientific_work"]
    assert 0 < actual["work"]["peak_rss_bytes"] < 512 * 1024**2


@pytest.mark.skipif(sys.platform != "linux", reason="Linux exec peak-RSS inheritance regression")
def test_cold_intermediary_isolates_cli_from_heavy_parent(tmp_path):
    protocol, _ = fixture(tmp_path)
    path = tmp_path / "protocol.json"
    path.write_bytes(encode(protocol))
    direct = standalone_cli_command(path, tmp_path / "direct")
    cold = [
        sys.executable,
        "-I",
        "-c",
        _COLD_LAUNCHER,
        *standalone_cli_command(path, tmp_path / "cold"),
    ]
    # Only this dedicated synthetic parent allocates memory. No policy, arrays,
    # checkpoint or simulator is imported. Both children see identical tiny data.
    pressure = """import json,resource,subprocess,sys
allocation=bytearray(580*1024**2)
results=[]
for command in json.loads(sys.argv[1]):
 process=subprocess.run(command,capture_output=True,text=True,timeout=22)
 results.append(dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
print(json.dumps(dict(parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     children=results)))
"""
    process = run_owned_process(
        [sys.executable, "-I", "-c", pressure, json.dumps([direct, cold])],
        timeout=28,
    )
    (tmp_path / "pressure_parent_stdout.txt").write_text(process.stdout)
    (tmp_path / "pressure_parent_stderr.txt").write_text(process.stderr)
    assert process.returncode == 0, f"stdout: {process.stdout}\nstderr: {process.stderr}"
    measured = json.loads(process.stdout)
    assert 512 * 1024**2 < measured["parent_peak_rss_bytes"] < 1024**3
    direct, cold = measured["children"]
    assert direct["returncode"] == 1, direct
    failed = json.loads((tmp_path / "direct/certificate_result.json").read_text())
    assert json.loads(direct["stdout"]) == dict(
        certificate_checks_passed=failed["certificate_checks_passed"], work=failed["work"]
    )
    assert failed["certificate_checks_passed"] is False
    assert failed["error"]["message"] == "Closure RSS limit exhausted"
    assert failed["work"]["peak_rss_bytes"] > 512 * 1024**2
    assert cold["returncode"] == 0, cold
    passed = json.loads((tmp_path / "cold/certificate_result.json").read_text())
    assert json.loads(cold["stdout"]) == dict(
        certificate_checks_passed=passed["certificate_checks_passed"], work=passed["work"]
    )
    assert passed["certificate_checks_passed"] is True
    assert 0 < passed["work"]["peak_rss_bytes"] < 512 * 1024**2
    assert failed["new_scientific_work"] == passed["new_scientific_work"]
    assert passed["new_scientific_work"] == protocol["new_scientific_work"]


def test_real_memory_limit_contract_still_rejects_an_overflow(tmp_path, monkeypatch):
    protocol, _ = fixture(tmp_path)
    monkeypatch.setattr(
        "src.gps.mppi_gps.audits.consequence_closure._rss", lambda: 512 * 1024**2 + 1
    )
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]
    assert "RSS limit" in result["error"]["message"]
    assert result["work"]["read_bytes"] == 0


@pytest.mark.parametrize(
    "fault", ["protocol_hash", "map_count", "new_science", "different_original"]
)
def test_passed_flags_cannot_override_cross_certificate_mismatches(tmp_path, fault):
    protocol, roots = fixture(tmp_path)

    def mutate(value):
        if fault == "protocol_hash":
            value["protocol_sha256"] = "f" * 64
        elif fault == "map_count":
            value["verified_file_count"] = 3
        elif fault == "new_science":
            value["new_scientific_work"]["neural_rows"] = 1
        else:
            value["original_scientific_receipt"]["sha256"] = "e" * 64

    reseal(protocol, roots, "a1_result", mutate)
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]


def test_external_failure_metadata_cannot_claim_a_clean_A1_observer(tmp_path):
    protocol, roots = fixture(tmp_path)
    reseal(
        protocol,
        roots,
        "a1_external_failure",
        lambda value: value.update(original_coordinator_exit_available=True),
    )
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]


def test_selected_source_anchor_is_tied_to_A1_manifest_without_importing_it(tmp_path):
    protocol, roots = fixture(tmp_path)
    source = b"raise AssertionError('this preserved source must not execute')\n"
    path = roots["a1_run"] / "source/auditor.py"
    path.parent.mkdir()
    path.write_bytes(source)
    protocol["anchors"]["a1_auditor_source"] = dict(
        root="a1_run", file="source/auditor.py", **binding(source)
    )
    result = close_certificates(protocol, tmp_path / "new")
    assert not result["certificate_checks_passed"]
    assert "selected source bytes" in result["error"]["message"]


def test_all_preserved_roots_are_protected_through_symlink_output_parents(tmp_path):
    protocol, roots = fixture(tmp_path)
    alias = tmp_path / "output-alias"
    alias.symlink_to(roots["a1_run"], target_is_directory=True)
    with pytest.raises(ClosureError, match="inside preserved"):
        close_certificates(protocol, alias / "must-not-exist")
    assert not (roots["a1_run"] / "must-not-exist").exists()
