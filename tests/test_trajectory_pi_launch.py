"""External supervision fixtures use metadata and disposable sleeping processes."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps import trajectory_pi_launch as launch
from src.gps.mppi_gps.provenance import write


@pytest.fixture
def config():
    path = Path(__file__).resolve().parents[1] / "configs/studies/trajectory_pi_p1.json"
    value = json.loads(path.read_text())
    value["launch_ready"] = True
    return value


def capacity(config, cores=32, memory=64):
    return dict(
        available_logical_cpus=cores,
        available_memory_bytes=memory * launch.GIB,
        free_disk_bytes=100 * launch.GIB,
        versions=config["resources"]["runtime"],
    )


def test_initial_and_main_capacity_are_distinct_and_runtime_is_exact(config):
    observed = capacity(config, cores=8, memory=8)
    assert launch.capacity_passed(observed, config, main=False)
    assert not launch.capacity_passed(observed, config, main=True)
    observed = capacity(config)
    assert launch.capacity_passed(observed, config, main=True)
    observed["versions"] = dict(observed["versions"], torch="wrong")
    assert not launch.capacity_passed(observed, config, main=True)


def proc_fixture(root, pid, own, reaped, children):
    directory = root / str(pid)
    (directory / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
    fields = ["0"] * 25
    fields[0] = "S"
    fields[11:15] = map(str, (*own, *reaped))
    (directory / "stat").write_text(f"{pid} (name with spaces) " + " ".join(fields))
    (directory / "statm").write_text("99 2 0")
    (directory / "task" / str(pid) / "children").write_text(" ".join(map(str, children)))


def test_tree_cpu_counts_live_and_reaped_descendants_once(tmp_path):
    proc_fixture(tmp_path, 100, (10, 20), (30, 40), [101])
    proc_fixture(tmp_path, 101, (5, 10), (15, 20), [])
    before = launch.tree_usage(100, tmp_path)
    assert before["cpu_seconds"] == 150 / os.sysconf("SC_CLK_TCK")
    # After parent waits for this child, its entire CPU moves into cutime.
    proc_fixture(tmp_path, 100, (10, 20), (80, 40), [])
    after = launch.tree_usage(100, tmp_path)
    assert after["cpu_seconds"] == before["cpu_seconds"]
    assert before["rss_bytes"] == 2 * after["rss_bytes"]
    assert after["observed_pids"] == [100]


class FakeGuard:
    def __init__(self):
        self.calls = 0

    def sample(self):
        return dict(cpu_seconds=float(self.calls), rss_bytes=100, peak_rss_bytes=100)

    def __call__(self):
        self.calls += 1
        return self.sample()


def test_external_deadline_kills_one_disposable_process_and_preserves_receipt(tmp_path):
    (tmp_path / "supervisor").mkdir()
    receipt = launch.run_phase(
        "fixture",
        [sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
        tmp_path,
        tmp_path,
        0.15,
        FakeGuard(),
        poll_seconds=0.01,
    )
    assert receipt["exit_code"] < 0
    assert "TimeoutError" in receipt["limit_error"]
    assert receipt["wall_seconds"] < 3
    assert Path(receipt["stdout"]).read_text() == "started\n"
    assert launch.read(tmp_path / "supervisor/fixture.json") == receipt
    assert receipt["process_tree_cpu_seconds"] > 0
    with pytest.raises(FileExistsError):
        launch.run_phase(
            "fixture",
            [sys.executable, "-c", "raise Exception('must not run')"],
            tmp_path,
            tmp_path,
            1,
            FakeGuard(),
        )


def test_no_remaining_time_does_not_start_a_process(tmp_path):
    (tmp_path / "supervisor").mkdir()
    receipt = launch.run_phase("fixture", ["DO_NOT_EXECUTE"], tmp_path, tmp_path, 0, FakeGuard())
    assert receipt["pid"] is None and receipt["exit_code"] is None
    assert "No total wall budget" in receipt["limit_error"]


def test_child_writes_no_source_bytecode_and_uses_frozen_thread_and_cache_settings(tmp_path):
    (tmp_path / "supervisor").mkdir()
    program = "import os,json; print(json.dumps({k:os.environ[k] for k in ['PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','MPLBACKEND','MPLCONFIGDIR','TMPDIR']}))"
    receipt = launch.run_phase(
        "environment",
        [sys.executable, "-c", program],
        tmp_path,
        tmp_path,
        3,
        FakeGuard(),
        poll_seconds=0.01,
    )
    assert receipt["exit_code"] == 0
    environment = json.loads(Path(receipt["stdout"]).read_text())
    assert {
        environment[key]
        for key in (
            "PYTHONDONTWRITEBYTECODE",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    } == {"1"}
    assert environment["MPLBACKEND"] == "Agg"
    assert Path(environment["MPLCONFIGDIR"]).is_relative_to(tmp_path)
    assert Path(environment["TMPDIR"]).is_relative_to(tmp_path)


def setup_base(
    tmp_path, config, monkeypatch, config_relative="configs/studies/trajectory_pi_p1.json"
):
    source = tmp_path / "source"
    write(source / config_relative, config)
    write(
        tmp_path / "source_manifest.json",
        dict(
            schema="metadata-fixture",
            files=[dict(launch.binding(source / config_relative), path=config_relative)],
        ),
    )
    write(tmp_path / "prepared/manifest.json", dict(schema="metadata-fixture"))
    monkeypatch.setattr(
        launch, "tree_usage", lambda pid: dict(cpu_seconds=0.0, rss_bytes=100, observed_pids=[pid])
    )
    return source


def fake_phase_factory(calls, p0_status="PASS", study_exit=0):
    def fake_phase(name, argv, base, source, timeout, guard):
        calls.append((name, argv))
        guard()
        if name == "preflight":
            paths = dict(
                config=Path(argv[argv.index("--config") + 1]),
                source_manifest=base / "source_manifest.json",
                prepared_manifest=base / "prepared/manifest.json",
            )
            write(
                base / "preflight/preflight.json",
                dict(
                    schema="trajectory-pi-preflight-1",
                    status=p0_status,
                    checks=[],
                    work={},
                    **{
                        key + "_sha256": launch.binding(path)["sha256"]
                        for key, path in paths.items()
                    },
                ),
            )
        if name == "study" and study_exit == 0:
            write(
                base / "worker/result.json",
                dict(
                    schema="trajectory-pi-p1-result-1",
                    status="COMPLETED_20_ROUNDS",
                    history=[],
                    full_project_success=False,
                ),
            )
        value = dict(
            schema="trajectory-pi-phase-1",
            phase=name,
            exit_code=study_exit if name == "study" else 0,
            limit_error=None,
            pid=100,
        )
        write(base / "supervisor" / f"{name}.json", value)
        return value

    return fake_phase


def test_failed_p0_is_preserved_and_only_saved_closure_follows(tmp_path, config, monkeypatch):
    setup_base(tmp_path, config, monkeypatch)
    calls = []
    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config))
    monkeypatch.setattr(launch, "run_phase", fake_phase_factory(calls, p0_status="FAIL_STOPPED"))
    result = launch.supervise(tmp_path)
    assert [name for name, _ in calls] == ["preflight", "audit", "report"]
    assert "--stopped" in calls[1][1]
    assert result["status"] == "STOPPED_PRESERVED"
    assert launch.read(tmp_path / "preflight/preflight.json")["status"] == "FAIL_STOPPED"
    assert launch.read(tmp_path / "worker/result.json")["status"] == "PRELAUNCH_STOPPED"
    assert launch.read(tmp_path / "worker/result.json")["history"] == []
    assert (tmp_path / "worker/external_supervisor.json").is_file()
    with pytest.raises(FileExistsError):
        launch.supervise(tmp_path)


def test_initial_capacity_failure_has_no_science_phase(tmp_path, config, monkeypatch):
    setup_base(tmp_path, config, monkeypatch)
    calls = []
    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config, cores=7))
    monkeypatch.setattr(launch, "run_phase", fake_phase_factory(calls))
    launch.supervise(tmp_path)
    assert [name for name, _ in calls] == ["audit", "report"]
    assert launch.read(tmp_path / "preflight/preflight.json")["status"] == "NOT_ATTEMPTED"


def test_passed_p0_is_not_relabelled_when_main_capacity_fails(tmp_path, config, monkeypatch):
    setup_base(tmp_path, config, monkeypatch)
    calls = []
    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config, cores=8))
    monkeypatch.setattr(launch, "run_phase", fake_phase_factory(calls))
    launch.supervise(tmp_path)
    assert [name for name, _ in calls] == ["preflight", "audit", "report"]
    assert launch.read(tmp_path / "preflight/preflight.json")["status"] == "PASS"
    assert launch.read(tmp_path / "worker/external_supervisor.json")["stage"] == "main_capacity"
    assert launch.read(tmp_path / "resource_observation.json")["passed"] is False


def test_complete_lifecycle_uses_frozen_public_cli_and_paired_launch_bindings(
    tmp_path, config, monkeypatch
):
    source = setup_base(tmp_path, config, monkeypatch)
    calls = []
    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config))
    monkeypatch.setattr(launch, "run_phase", fake_phase_factory(calls))
    result = launch.supervise(tmp_path)
    assert result["status"] == "CLOSED"
    assert [name for name, _ in calls] == ["preflight", "study", "audit", "report"]
    assert all(
        argv[:4] == [sys.executable, "scripts/train_mppi_gps.py", "study", "trajectory-pi"]
        for _, argv in calls
    )
    frozen = launch.read(tmp_path / "launch.json")
    assert frozen["source_directory"] == str(source)
    assert frozen["output_directory"] == str(tmp_path / "worker")
    assert frozen["config"] == launch.binding(source / "configs/studies/trajectory_pi_p1.json")
    assert "--stopped" not in calls[2][1]
    assert not (tmp_path / "worker/external_supervisor.json").exists()
    events = [
        json.loads(line)
        for line in (tmp_path / "supervisor/journal.jsonl").read_text().splitlines()
    ]
    study_start = next(
        row for row in events if row["event"] == "phase_start" and row["phase"] == "study"
    )
    assert study_start["reserved_closure_wall_seconds"] == 7500
    assert study_start["reserved_closure_cpu_seconds"] == 7500
    assert study_start["external_wall_seconds"] <= config["limits"]["execution_wall_seconds"]


def test_failed_main_has_no_retry_and_retains_external_stop(tmp_path, config, monkeypatch):
    setup_base(tmp_path, config, monkeypatch)
    calls = []
    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config))
    monkeypatch.setattr(launch, "run_phase", fake_phase_factory(calls, study_exit=-9))
    result = launch.supervise(tmp_path)
    assert result["status"] == "STOPPED_PRESERVED"
    assert [name for name, _ in calls] == ["preflight", "study", "audit", "report"]
    assert launch.read(tmp_path / "worker/result.json")["status"] == "FAILED_STOPPED"
    assert "--stopped" in calls[2][1]


def test_explicit_p1_1_configuration_is_source_bound_and_used_in_every_receipt(
    tmp_path, config, monkeypatch
):
    name = "configs/studies/trajectory_pi_p1_1.json"
    config.update(version="P1.1", input_mode="worker-native")
    source = setup_base(tmp_path, config, monkeypatch, config_relative=name)
    calls = []
    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config))
    monkeypatch.setattr(launch, "run_phase", fake_phase_factory(calls))
    result = launch.supervise(tmp_path, config_relative=name)
    assert result["status"] == "CLOSED"
    assert launch.read(tmp_path / "launch.json")["config"] == launch.binding(source / name)
    preflight_args = calls[0][1]
    assert preflight_args[preflight_args.index("--config") + 1] == str(source / name)
    assert launch.read(tmp_path / "worker/config.json")["version"] == "P1.1"


@pytest.mark.parametrize(
    "name",
    [
        "../outside.json",
        "/tmp/outside.json",
        "configs/../outside.json",
        "./configs/studies/trajectory_pi_p1.json",
    ],
)
def test_configuration_selector_cannot_escape_frozen_source(tmp_path, config, monkeypatch, name):
    setup_base(tmp_path, config, monkeypatch)
    with pytest.raises(ValueError, match="Contained normalized"):
        launch.supervise(tmp_path, config_relative=name)
    assert not (tmp_path / "supervisor").exists()


def test_selected_configuration_bytes_must_belong_to_frozen_source_roster(
    tmp_path, config, monkeypatch
):
    source = setup_base(tmp_path, config, monkeypatch)
    path = source / "configs/studies/trajectory_pi_p1.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source roster"):
        launch.supervise(tmp_path)
    assert not (tmp_path / "supervisor").exists()


def test_external_guard_enforces_mutable_output_ceiling_with_closure_headroom(
    tmp_path, config, monkeypatch
):
    monkeypatch.setattr(launch, "tree_usage", lambda _: dict(cpu_seconds=0.0, rss_bytes=1))
    guard = launch.Guard(tmp_path, config)
    assert guard.output_ceiling == config["limits"]["output_bytes"] == 64 * launch.GIB
    (tmp_path / "paid-prefix").write_bytes(b"x" * 20)
    guard()
    # Small synthetic ceilings exercise actual byte accounting without large files.
    guard.output_ceiling = 1024**2 + 19
    with pytest.raises(RuntimeError, match="total output/closure reservation"):
        guard()
    guard.output_ceiling = 1024**2 + 20
    guard()


def test_supervisor_reserves_execution_output_then_releases_audit_output(
    tmp_path, config, monkeypatch
):
    setup_base(tmp_path, config, monkeypatch)
    calls, observed = [], []
    fake = fake_phase_factory(calls)

    def record_ceiling(name, argv, base, source, timeout, guard):
        observed.append((name, guard.output_ceiling))
        return fake(name, argv, base, source, timeout, guard)

    monkeypatch.setattr(launch, "observe_capacity", lambda *a, **k: capacity(config))
    monkeypatch.setattr(launch, "run_phase", record_ceiling)
    assert launch.supervise(tmp_path)["status"] == "CLOSED"
    assert observed == [
        ("preflight", 56 * launch.GIB),
        ("study", 56 * launch.GIB),
        ("audit", 64 * launch.GIB),
        ("report", 64 * launch.GIB),
    ]
