"""Saved-only J2 publication preserves stopped rosters and exact curve records."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.reporting import jitter_diagnosis as report


def write(path: Path, value: object) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value) + "\n").encode()
    path.write_bytes(payload)
    return dict(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def episode(seed: int, kind: str, walking: bool, survived: bool) -> dict:
    horizon = 1000 if kind == "evaluation" else 100
    controls = horizon if survived else 20
    native = 0.1 * controls + (0 if survived else 20)
    tail = 0 if survived else 5 * (horizon - controls)
    return dict(
        seed=seed,
        kind=kind,
        walking=walking,
        survived=survived,
        controls=controls,
        right_censored=survived,
        done=not survived,
        observed_seconds=0.008 * controls,
        native_cost=native,
        raw_cost=0.1 * controls,
        terminal_cost=0 if survived else 20,
        absorbing_cost=tail,
        cost=native + tail,
    )


def fixture(tmp_path: Path, status: str = "COMPLETED_FIXED_J2_ROSTER") -> tuple[Path, Path]:
    worker = tmp_path / "worker"
    bindings = {}
    seeds = list(range(3600003000, 3600003006))
    conditions = [dict(id="clean", jitter="clean", signal_share=1.0)] + [
        dict(id=f"{kind}_{str(share).replace('.', 'p')}", jitter=kind, signal_share=share)
        for kind in ("mppi", "white", "slow")
        for share in (0.9, 0.75, 0.5, 0.25)
    ]
    config = dict(
        seeds=dict(training=seeds, evaluation=list(range(32)), probe=list(range(100, 108))),
        learning=dict(updates=3),
        conditions=conditions,
        source_bindings={},
    )
    bindings["config.json"] = write(worker / "config.json", config)
    selected = (
        conditions
        if status == "COMPLETED_FIXED_J2_ROSTER"
        else conditions[:1]
        if status == "STOPPED_CLEAN_POSITIVE_CONTROL"
        else []
    )
    entries, records = [], []
    for index, condition in enumerate(selected):
        for seed in seeds:
            case_id = f"{seed}_{condition['id']}"
            checkpoint = dict(
                sha256=hashlib.sha256(case_id.encode()).hexdigest(),
                policy_hash=hashlib.sha256(("policy" + case_id).encode()).hexdigest(),
            )
            walking = max(0, 32 - 3 * index) if status == "COMPLETED_FIXED_J2_ROSTER" else 20
            survival = min(32, walking + 1)
            evaluations = [
                episode(reset, "evaluation", reset < walking, reset < survival)
                for reset in config["seeds"]["evaluation"]
            ]
            probes = [
                episode(reset, "probe", False, index < 5) for reset in config["seeds"]["probe"]
            ]
            postfit = dict(
                standing100=0.1 + 0.01 * index,
                standing25=0.09 + 0.01 * index,
                probe_mean_controls=sum(row["controls"] for row in probes) / 8,
                probe_falls=sum(row["done"] for row in probes),
            )
            summary = dict(
                case_id=case_id,
                seed=seed,
                condition=dict(
                    id=condition["id"], kind=condition["jitter"], share=condition["signal_share"]
                ),
                checkpoint=checkpoint,
                episodes=probes + evaluations,
                postfit=postfit,
                fitting=dict(train_mse=0.03, held_clean_mse=0.2 + 0.02 * index),
                pre_update_scores=dict(A=0.4, B=0.3, C=0.5, D=None, E=None),
            )
            name = f"cases/{case_id}/summary.json"
            bindings[name] = write(worker / name, summary)
            curve = [
                dict(
                    update=update,
                    fit_update=update,
                    action_mse=1 / update,
                    total_loss=1 / update,
                    proximal_loss=0,
                    unclipped_gradient_norm=0.1,
                    preserved_extra_scalar=update * 7,
                )
                for update in range(1, 4)
            ]
            name_fit = f"cases/{case_id}/fit.json"
            bindings[name_fit] = write(
                worker / name_fit, dict(checkpoint=checkpoint, learning_curve=curve)
            )
            (worker / f"cases/{case_id}/child.pt").write_bytes(b"MUST NOT DECODE THIS CHECKPOINT")
            (worker / f"cases/{case_id}/traces.npz").write_bytes(b"MUST NOT READ RAW ARRAYS")
            entries.append(
                dict(case_id=case_id, seed=seed, condition=condition["id"], summary=name)
            )
            records.append(
                dict(
                    case_id=case_id,
                    seed=seed,
                    condition=condition["id"],
                    walking=walking,
                    full_survival=survival,
                    actual_controls=sum(row["controls"] for row in evaluations),
                    complete_cost=sum(row["cost"] for row in evaluations),
                    native_cost=sum(row["native_cost"] for row in evaluations),
                    absorbing_cost=sum(row["absorbing_cost"] for row in evaluations),
                    diagnostic_error=postfit["standing100"],
                    diagnostic_short_error=postfit["standing25"],
                    offline_error=summary["fitting"]["held_clean_mse"],
                    mean_probe_controls=postfit["probe_mean_controls"],
                    probe_falls=postfit["probe_falls"],
                )
            )
    work = dict(
        requested=dict(fits=len(entries)),
        completed=dict(fits=len(entries)),
        unknown_partial={},
        wall_seconds=10,
    )
    bindings["work.json"] = write(worker / "work.json", work)
    bindings["result.json"] = write(
        worker / "result.json",
        dict(status=status, cases=entries, actual_fits=len(entries), work=work),
    )
    primary = (
        None
        if status != "COMPLETED_FIXED_J2_ROSTER"
        else dict(
            eligible=True,
            confirmed=False,
            per_seed=[
                dict(
                    seed=seed,
                    diagnostic_spearman=0.8,
                    offline_spearman=0.75,
                    improvement=0.05,
                    diagnostic_auc_any_walking=1.0,
                    offline_auc_any_walking=1.0,
                    secondary_duration_controls=dict(
                        mean_probe_controls_spearman=0.7,
                        negative_probe_fall_count_spearman=0.6,
                        primary_gate_uses_these_controls=False,
                    ),
                )
                for seed in seeds
            ],
        )
    )
    audit_path = tmp_path / "audit_result.json"
    write(
        audit_path,
        dict(
            passed=True,
            input_files=bindings,
            worker_status=status,
            records=records,
            primary=primary,
            primary_comparison_executed=primary is not None,
            wall_seconds=1,
            read_bytes=100,
            new_neural_rows=20,
            new_physics_controls=0,
            new_gradient_rows=0,
            new_optimizer_steps=0,
        ),
    )
    return worker, audit_path


def test_complete_report_keeps_every_case_curve_point_and_independent_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, audit = fixture(tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "fontcache"))
    out = tmp_path / "report"
    receipt = report.report_run(worker, audit, out)
    assert receipt["actual_cases"] == 78
    assert receipt["curve_points"] == 78 * 3
    assert receipt["primary"]["confirmed"] is False
    cases = read_csv(out / "cases.csv")
    assert len(cases) == 78 and len(read_csv(out / "per_seed_primary.csv")) == 6
    assert cases[-1]["walking"] == "0" and cases[-1]["full_survival"] == "1"
    assert cases[-1]["right_censored_episodes"] == "1"
    assert float(cases[-1]["absorbing_cost"]) > 0
    with gzip.open(out / "learning_curves.csv.gz", "rt", newline="") as stream:
        curves = list(csv.DictReader(stream))
    assert len(curves) == 234
    assert json.loads(curves[-1]["point_json"])["preserved_extra_scalar"] == 21
    assert all((out / name).stat().st_size > 100 for name in receipt["figures"])
    assert not any(receipt["new_scientific_work"].values())
    assert receipt["output_bytes"] == sum(path.stat().st_size for path in out.iterdir())
    assert (
        receipt["read_bytes"]
        == sum(binding["bytes"] for binding in receipt["input_files"].values())
        + receipt["reporter_source"]["bytes"]
    )
    assert "does not establish pre-update teachability" in (out / "report.md").read_text()
    with pytest.raises(ValueError, match="fresh output"):
        report.report_run(worker, audit, out)


@pytest.mark.parametrize(
    "status,actual", [("STOPPED_CLEAN_POSITIVE_CONTROL", 6), ("STOPPED_REFERENCE_QUALIFICATION", 0)]
)
def test_qualification_stop_keeps_unexecuted_statistics_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, actual: int
) -> None:
    worker, audit = fixture(tmp_path, status)
    monkeypatch.setattr(report, "_figures", lambda *args: [])
    receipt = report.report_run(worker, audit, tmp_path / "report")
    assert receipt["actual_cases"] == actual and receipt["curve_points"] == actual * 3
    assert receipt["primary"] is None and receipt["primary_comparison_executed"] is False
    rows = read_csv(tmp_path / "report/per_seed_primary.csv")
    assert all(row["comparison_executed"] == "False" and row["improvement"] == "" for row in rows)
    assert "qualification stop" in (tmp_path / "report/report.md").read_text()


def test_stale_audit_hash_and_inconsistent_measurement_are_rejected(tmp_path: Path) -> None:
    worker, audit_path = fixture(tmp_path, "STOPPED_CLEAN_POSITIVE_CONTROL")
    path = next((worker / "cases").glob("*/summary.json"))
    original = path.read_bytes()
    value = json.loads(original)
    value["postfit"]["standing100"] = 9
    write(path, value)
    with pytest.raises(ValueError, match="differs from independent audit"):
        report.report_run(worker, audit_path, tmp_path / "stale")
    audit = json.loads(audit_path.read_text())
    audit["input_files"][str(path.relative_to(worker))] = dict(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    write(audit_path, audit)
    with pytest.raises(ValueError, match="case/audit join differs"):
        report.report_run(worker, audit_path, tmp_path / "inconsistent")


def test_failed_audit_and_read_budget_stop_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, audit_path = fixture(tmp_path, "STOPPED_REFERENCE_QUALIFICATION")
    audit = json.loads(audit_path.read_text())
    audit["passed"] = False
    write(audit_path, audit)
    with pytest.raises(ValueError, match="passed independent audit"):
        report.report_run(worker, audit_path, tmp_path / "unadmitted")
    assert not (tmp_path / "unadmitted").exists()
    audit["passed"] = True
    write(audit_path, audit)
    monkeypatch.setattr(report, "MAX_READ_BYTES", 1)
    with pytest.raises(ValueError, match="read budget"):
        report.report_run(worker, audit_path, tmp_path / "overbudget")
    assert not (tmp_path / "overbudget").exists()


@pytest.mark.parametrize("corruption", ["missing_update", "different_checkpoint"])
def test_missing_curve_updates_and_wrong_endpoint_cannot_be_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    worker, audit_path = fixture(tmp_path, "STOPPED_CLEAN_POSITIVE_CONTROL")
    monkeypatch.setattr(report, "_figures", lambda *args: [])
    path = next((worker / "cases").glob("*/fit.json"))
    fit = json.loads(path.read_text())
    if corruption == "missing_update":
        fit["learning_curve"].pop()
    else:
        fit["checkpoint"]["sha256"] = "0" * 64
    binding = write(path, fit)
    audit = json.loads(audit_path.read_text())
    audit["input_files"][str(path.relative_to(worker))] = binding
    write(audit_path, audit)
    with pytest.raises(ValueError, match="curve"):
        report.report_run(worker, audit_path, tmp_path / "report")
    assert not (tmp_path / "report/receipt.json").exists()


def test_report_import_keeps_science_runtimes_unloaded() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from src.gps.mppi_gps.reporting import jitter_diagnosis; assert not {'torch','mujoco','warp'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stderr


def test_compact_gzip_mirror_keeps_decoded_audit_bindings_and_rejects_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker, audit_path = fixture(tmp_path, "STOPPED_CLEAN_POSITIVE_CONTROL")
    monkeypatch.setattr(report, "_figures", lambda *args: [])
    originals = {}
    for path in (worker / "cases").glob("*/fit.json"):
        payload = path.read_bytes()
        originals[str(path.relative_to(worker))] = payload
        path.with_name(path.name + ".gz").write_bytes(gzip.compress(payload, mtime=0))
        path.unlink()
    receipt = report.report_run(worker, audit_path, tmp_path / "report")
    assert receipt["curve_points"] == 18
    assert receipt["decoded_bytes"] == sum(map(len, originals.values()))
    assert receipt["logical_read_bytes"] == receipt["read_bytes"] + receipt["decoded_bytes"]
    assert (
        receipt["read_bytes"]
        == sum(value["bytes"] for value in receipt["stored_files"].values())
        + receipt["reporter_source"]["bytes"]
    )
    for name, payload in originals.items():
        assert receipt["input_files"][name] == dict(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
        )
        assert receipt["stored_files"][name]["compressed"] is True
        assert receipt["stored_files"][name]["path"].endswith("fit.json.gz")
    name, payload = next(iter(originals.items()))
    changed = json.loads(payload)
    changed["learning_curve"][0]["action_mse"] = 12
    (worker / (name + ".gz")).write_bytes(gzip.compress(json.dumps(changed).encode(), mtime=0))
    with pytest.raises(ValueError, match="differs from independent audit"):
        report.report_run(worker, audit_path, tmp_path / "altered")


def test_gzip_decoding_stops_at_declared_file_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = tmp_path / "worker"
    worker.mkdir()
    payload = json.dumps(dict(value="a" * 1000)).encode()
    (worker / "large.json.gz").write_bytes(gzip.compress(payload, mtime=0))
    monkeypatch.setattr(report, "MAX_FILE_BYTES", 100)
    reader = report.SavedInputs(worker)
    with pytest.raises(ValueError, match="decoded budget"):
        reader.load("large.json")
    assert reader.decoded_bytes == 100
    assert reader.read_bytes < 100
