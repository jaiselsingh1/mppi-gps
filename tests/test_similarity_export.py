"""J1 portable export validation uses synthetic saved files, never control or fitting."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.validate_export import (
    SIMILARITY_CLOSING_HASHES,
    SIMILARITY_SCIENCE_KEYS,
    similarity_export_commands,
    verify_similarity_report,
)
from src.gps.mppi_gps.reporting import similarity_capsule as decoder


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def binding(path: Path) -> dict:
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def tiny_capsule(root: Path, *, bomb: bool = False) -> dict[str, str]:
    original = b'{"learning_curve":[{"update":1,"action_mse":0.2}]}\n'
    path = root / "run/learning_curve.json.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(gzip.compress(b"X" * 1_000_000 if bomb else original, mtime=0))
    save(root / "run/config.json", dict(scope="synthetic"))
    save(
        root / "capsule.json",
        dict(
            mode="metadata",
            files={
                "run/learning_curve.json.gz": dict(
                    encoding="gzip_exact_original_bytes",
                    original_bytes=1 if bomb else len(original),
                    original_sha256=decoder.sha(b"X" if bomb else original),
                ),
            },
        ),
    )
    save(
        root / "files_manifest.json",
        dict(files={str(p.relative_to(root)): binding(p) for p in root.rglob("*") if p.is_file()}),
    )
    manifest_hash = binding(root / "files_manifest.json")["sha256"]
    save(root / "collection_exit.json", dict(passed=True, manifest_sha256=manifest_hash))
    return {name: binding(root / name)["sha256"] for name in SIMILARITY_CLOSING_HASHES}


def test_materialize_roundtrip_keeps_original_hashes_and_capsule_immutable(tmp_path, monkeypatch):
    monkeypatch.setattr(decoder, "rss_bytes", lambda: 1024)
    source = tmp_path / "capsule"
    hashes = tiny_capsule(source)
    before = {str(p.relative_to(source)): binding(p) for p in source.rglob("*") if p.is_file()}
    out = tmp_path / "replay"
    result = decoder.materialize(
        source, out, hashes["files_manifest.json"], hashes["collection_exit.json"]
    )
    assert result["passed"] and result["new_scientific_work"] == 0
    assert result["limit_enforcement"]["address_space_enforced"] is False
    assert result["limits"]["peak_rss_bytes"] == 512 * 1024**2
    assert (
        json.loads((out / "run/learning_curve.json").read_text())["learning_curve"][0]["update"]
        == 1
    )
    assert before == {
        str(p.relative_to(source)): binding(p) for p in source.rglob("*") if p.is_file()
    }
    with pytest.raises(FileExistsError):
        decoder.materialize(
            source, out, hashes["files_manifest.json"], hashes["collection_exit.json"]
        )


@pytest.mark.parametrize("problem", ["zipbomb", "missing", "corrupt", "rss"])
def test_materialize_stops_and_preserves_failure_receipt(tmp_path, monkeypatch, problem):
    monkeypatch.setattr(decoder, "rss_bytes", lambda: 1024)
    source = tmp_path / "capsule"
    hashes = tiny_capsule(source, bomb=problem == "zipbomb")
    if problem == "missing":
        (source / "run/config.json").unlink()
    elif problem == "corrupt":
        (source / "run/config.json").write_text("changed")
    elif problem == "rss":
        monkeypatch.setattr(decoder, "rss_bytes", lambda: decoder.MAX_MEMORY + 1)
    out = tmp_path / "replay"
    result = decoder.materialize(
        source, out, hashes["files_manifest.json"], hashes["collection_exit.json"]
    )
    assert result["passed"] is False and "error" in result
    assert (out / "materialization_receipt.json").is_file()
    if problem in {"zipbomb", "rss"}:
        assert not (out / "run/learning_curve.json").exists()


def test_darwin_limit_mode_never_attempts_linux_address_space_control(monkeypatch):
    monkeypatch.setattr(decoder.sys, "platform", "darwin")
    monkeypatch.setattr(
        decoder.resource, "setrlimit", lambda *_: pytest.fail("macOS AS must not be installed")
    )
    result = decoder.install_limits()
    assert result["mode"] == "darwin_rss_wall"
    assert result["address_space_enforced"] is False and result["address_space_bytes"] is None


def test_linux_limit_mode_installs_finite_address_space_and_retains_rss_guard(monkeypatch):
    monkeypatch.setattr(decoder.sys, "platform", "linux")
    monkeypatch.setattr(
        decoder.resource,
        "getrlimit",
        lambda *_: (decoder.resource.RLIM_INFINITY, decoder.resource.RLIM_INFINITY),
    )
    installed = []
    monkeypatch.setattr(
        decoder.resource, "setrlimit", lambda kind, values: installed.append((kind, values))
    )
    result = decoder.install_limits()
    assert installed == [(decoder.resource.RLIMIT_AS, (decoder.MAX_MEMORY, decoder.MAX_MEMORY))]
    assert result["address_space_enforced"] and result["mode"] == "linux_address_space_rss_wall"


def test_rss_sensor_uses_actual_platform_units(monkeypatch):
    monkeypatch.setattr(decoder.resource, "getrusage", lambda *_: SimpleNamespace(ru_maxrss=7))
    monkeypatch.setattr(decoder.sys, "platform", "darwin")
    assert decoder.rss_bytes() == 7
    monkeypatch.setattr(decoder.sys, "platform", "linux")
    assert decoder.rss_bytes() == 7 * 1024


def test_publication_cli_runs_in_fresh_process_without_scientific_imports(tmp_path):
    source = (tmp_path / "capsule").resolve()
    hashes = tiny_capsule(source)
    out = (tmp_path / "replay").resolve()
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.gps.mppi_gps.reporting.similarity_capsule",
            "--capsule",
            str(source),
            "--out",
            str(out),
            "--manifest-sha256",
            hashes["files_manifest.json"],
            "--exit-sha256",
            hashes["collection_exit.json"],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((out / "materialization_receipt.json").read_text())
    assert receipt["passed"] and receipt["new_scientific_work"] == 0
    assert receipt["limit_enforcement"]["mode"] in {
        "darwin_rss_wall",
        "linux_address_space_rss_wall",
    }
    imports = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from src.gps.mppi_gps.reporting import similarity_capsule; assert not {'torch','mujoco','gymnasium'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert imports.returncode == 0, imports.stderr


def test_j1_commands_are_conditional_and_use_portable_saved_only_module(tmp_path):
    root, outputs = tmp_path / "export", tmp_path / "outputs"
    assert similarity_export_commands(root, outputs) == []
    capsule = root / "results/similarity_j1"
    save(capsule / "capsule.json", {})
    with pytest.raises(ValueError, match="materializer"):
        similarity_export_commands(root, outputs)
    helper = capsule / "publication_tools/materialize.py"
    helper.parent.mkdir()
    helper.write_text("# sealed reference helper\n")
    commands = similarity_export_commands(root, outputs)
    assert len(commands) == 3
    assert commands[0][-2:] == ["similarity-investigation", "--help"]
    assert commands[1][:2] == ["-m", "src.gps.mppi_gps.reporting.similarity_capsule"]
    assert commands[2][:2] == ["-m", "src.gps.mppi_gps.reporting.similarity"]
    assert commands[1][-3:] == [
        SIMILARITY_CLOSING_HASHES["files_manifest.json"],
        "--exit-sha256",
        SIMILARITY_CLOSING_HASHES["collection_exit.json"],
    ]
    assert all("launch" not in command and "worker" not in command for command in commands)


@pytest.fixture
def saved_export(tmp_path):
    root, replay, report = (tmp_path / name for name in ("export", "replay", "report"))
    run, capsule = replay / "run", root / "results/similarity_j1"
    save(run / "config.json", dict(learning=dict(updates=2)))
    save(run / "manifest.json", {})
    save(run / "input_manifest.json", {})
    save(
        run / "worker_result.json",
        dict(status="CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE", objective_passed=False),
    )
    case = dict(
        case_id="case",
        group="g0",
        learner="l0",
        split="development",
        source="mppi",
        fit_receipt_file="worker/fits/case/fit_receipt.json",
    )
    save(run / "worker/cases.json", [case])
    save(run / case["fit_receipt_file"], dict(updates=2))
    points = [
        dict(
            update=i,
            fit_update=i,
            action_mse=1 / i,
            proximal_loss=0.0,
            total_loss=1 / i,
            unclipped_gradient_norm=0.1,
        )
        for i in (1, 2)
    ]
    save(run / "worker/fits/case/learning_curve.json", dict(learning_curve=points))
    audit = dict(passed=True, full_scientific_audit_passed=True)
    for key, name in (
        ("config_sha256", "config.json"),
        ("source_manifest_sha256", "manifest.json"),
        ("input_manifest_sha256", "input_manifest.json"),
        ("worker_result_sha256", "worker_result.json"),
    ):
        audit[key] = binding(run / name)["sha256"]
    save(replay / "audit/result/audit.json", audit)
    save(replay / "atlas/result/atlas_receipt.json", dict(passed=True))
    report.mkdir()
    fields = [
        "case_id",
        "group",
        "learner",
        "split",
        "source",
        "fit_status",
        "update",
        "fit_update",
        "action_mse",
        "proximal_loss",
        "total_loss",
        "unclipped_gradient_norm",
    ]
    with (report / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {key: case.get(key) for key in fields[:5]} | dict(fit_status="completed_fit") | point
            for point in points
        )
    for name in (
        "cases.csv",
        "outcome_summary.csv",
        "teachers.csv",
        "score_table.csv",
        "score_outcome_pairs.csv",
    ):
        (report / name).write_text("saved\nrow\n")
    for name in ("scores_vs_motion.png", "scores_vs_walking.png", "learning_curves.png"):
        (report / name).write_bytes(b"synthetic bound figure")
    save(report / "saved_measurements.json", {})
    (report / "report.md").write_text("Synthetic saved-only report\n")
    source = root / "src/gps/mppi_gps/reporting/similarity.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic reporter identity\n")
    receipt = dict(
        actual_cases=1,
        completed_fits=1,
        learning_cases_with_curves=1,
        learning_points=2,
        saved_only=True,
        no_followup_launched=True,
        audit_passed=True,
        full_scientific_audit_passed=True,
        audit_association_verified=True,
        status="CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
        objective_passed=False,
        new_scientific_work=dict.fromkeys(SIMILARITY_SCIENCE_KEYS, 0),
        source_sha256=binding(source)["sha256"],
        input_files={str(path.resolve()): binding(path) for path in replay.rglob("*.json")},
        outputs={path.name: binding(path) for path in report.iterdir()},
    )
    receipt["read_bytes"] = sum(value["bytes"] for value in receipt["input_files"].values())
    save(report / "report_receipt.json", receipt)
    for path in report.iterdir():
        target = replay / "report" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
    selected = {}
    for path in list(replay.rglob("*")):
        if path.is_file():
            name = str(path.relative_to(replay))
            selected[name] = dict(**binding(path), encoding="identity")
            target = capsule / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    save(capsule / "capsule.json", dict(mode="metadata", files=selected))
    (replay / "capsule.json").write_bytes((capsule / "capsule.json").read_bytes())
    save(
        capsule / "files_manifest.json",
        dict(
            files={
                str(p.relative_to(capsule)): binding(p) for p in capsule.rglob("*") if p.is_file()
            }
        ),
    )
    save(
        capsule / "collection_exit.json",
        dict(passed=True, manifest_sha256=binding(capsule / "files_manifest.json")["sha256"]),
    )
    hashes = {name: binding(capsule / name)["sha256"] for name in SIMILARITY_CLOSING_HASHES}
    for name in hashes:
        (replay / name).write_bytes((capsule / name).read_bytes())
    save(
        replay / "materialization_receipt.json",
        dict(
            passed=True,
            new_scientific_work=0,
            immutable_capsule_modified=False,
            manifest_sha256=hashes["files_manifest.json"],
            collection_exit_sha256=hashes["collection_exit.json"],
            files={
                str(p.relative_to(replay)): binding(p) for p in replay.rglob("*") if p.is_file()
            },
        ),
    )
    return root, capsule, replay, report, hashes


def validate_saved(fixture) -> None:
    root, capsule, replay, report, hashes = fixture
    verify_similarity_report(
        root, capsule, replay, report, expected_cases=1, expected_updates=2, closing_hashes=hashes
    )


def test_saved_export_checks_every_curve_row_and_all_bindings(saved_export):
    validate_saved(saved_export)


@pytest.mark.parametrize("problem", ["work", "point", "duplicate", "figure", "input"])
def test_saved_export_rejects_semantic_or_hash_corruption(saved_export, problem):
    _, _, _, report, _ = saved_export
    receipt_path = report / "report_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if problem == "work":
        receipt["new_scientific_work"]["neural_rows"] = 1
        save(receipt_path, receipt)
    elif problem == "input":
        next(iter(receipt["input_files"].values()))["sha256"] = "a" * 64
        save(receipt_path, receipt)
    elif problem == "figure":
        (report / "learning_curves.png").write_bytes(b"changed")
    else:
        path = report / "learning_curves.csv"
        lines = path.read_text().splitlines(keepends=True)
        if problem == "point":
            lines[1] = lines[1].replace("1.0", "9.0", 1)
        else:
            lines.append(lines[1])
        path.write_text("".join(lines))
    with pytest.raises(ValueError):
        validate_saved(saved_export)
