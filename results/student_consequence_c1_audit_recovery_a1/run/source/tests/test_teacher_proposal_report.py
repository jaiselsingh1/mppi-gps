"""Small saved JSON fixtures test H5 reporting without research execution."""

from __future__ import annotations

import builtins
import csv
import json
from copy import deepcopy

import pytest

from src.gps.mppi_gps.proposal_budget import census
from src.gps.mppi_gps.provenance import sha
from src.gps.mppi_gps.recipes import load_recipe
from src.gps.mppi_gps.reporting import teacher_proposal as module


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source(tmp_path, *, capsule=False, result=None):
    run = tmp_path / "saved"
    root = run / "run" if capsule else run
    root.mkdir(parents=True)
    if capsule:
        save(run / "capsule.json", {})
    config = load_recipe("teacher_proposal_h5_0")
    save(root / "config.json", config)
    save(root / "call_census.json", census(config))
    save(
        root / ("worker_result.json" if result is not None else "result.json"),
        result
        or dict(
            status="prepared_unlaunched", execution_complete=False, work={}, launch_blockers=[]
        ),
    )
    return run, root


def curve(root, name="s0_parent", *, partial=False):
    rows = [
        dict(
            update=offset + index,
            fit_update=index,
            action_mse=1 / index,
            proximal_loss=0.0,
            total_loss=1 / index,
            unclipped_gradient_norm=2 / index,
        )
        for index, offset in ((1, 0), (2, 0))
    ]
    if "child" in name:
        for row in rows:
            row["update"] += 2000
    path = root / "learning" / (name + ("_partial" if partial else "") + ".json")
    save(path, dict(learning_curve=rows, completed_updates=len(rows), failed=partial))
    return path


def summary(cost=5012.0, steps=2, walking=False):
    return dict(
        episodes=1,
        successes=int(walking),
        survived=int(walking),
        controls=steps,
        mean_cost=cost,
        costs=[cost],
        complete=True,
    )


def diagnostics(tmp_path):
    raw = dict(file="arrays.bin", group="arrays/a000000001_teacher")
    own = dict(reference=dict(file="arrays.bin", group="arrays/a000000002_data"))
    metrics = []
    for index, split, owner, value in (
        (0, "incoming", "1" * 64, 2.0),
        (1, "endpoint", "2" * 64, 1.0),
        (2, "validation", "2" * 64, 3.0),
    ):
        metrics.append(
            dict(
                id=f"event-{index:09d}",
                event="policy_metric_complete",
                tag=f"s0_independent_child/{split}",
                actor_hash=owner,
                observations=dict(**own, key="observations"),
                targets=dict(**own, key="actions"),
                rows=4000 if split != "validation" else 2000,
                batch_size=1024,
                action_mse=value,
            )
        )
    episode = dict(
        id="event-000000003",
        event="episode_complete",
        tag="teacher_s0_independent_00",
        seed=100,
        steps=2,
        cost=5012.0,
        mae=1.4,
        success=False,
        survived=False,
        complete=True,
        terminal_cost=20.0,
        terminal_adjustment=0.0,
        absorbing_cost=4990.0,
        raw=raw,
    )
    agreement = dict(
        id="event-000000004",
        event="teacher_agreement_complete",
        actor_hash="1" * 64,
        episode_raw=raw,
        rows=2,
        action_agreement=6.0,
    )
    payload = dict(
        schema="walker-h5-saved-diagnostics-v1",
        source_ledger=dict(file="/external/ledger.jsonl", sha256="a" * 64, bytes=123),
        audit_receipt=dict(file="/external/audit.json", sha256="b" * 64),
        events=[*metrics, episode, agreement],
    )
    path = tmp_path / "diagnostics.json"
    save(path, payload)
    return path, payload


def test_metadata_only_report_does_not_import_plotting_or_execution_stacks(tmp_path, monkeypatch):
    run, root = source(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("torch", "mujoco", "matplotlib"):
            raise AssertionError("Unexpected implementation import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    receipt = module.render(run, tmp_path / "report")
    assert receipt["table_rows"]["learning_curves.csv"] == 0
    assert receipt["table_rows"]["student_outcomes.csv"] == 0
    assert not (tmp_path / "report/learning_curves.png").exists()
    assert read_csv(tmp_path / "report/student_outcomes.csv") == []
    assert "not zero" in (tmp_path / "report/report.md").read_text()
    assert "Incomplete execution" not in (tmp_path / "report/report.md").read_text()
    assert receipt["inputs"]["config.json"]["sha256"] == sha(root / "config.json")


def test_saved_metric_joins_and_fall_costs_preserve_reductions_and_missing_data(tmp_path):
    result = dict(
        schema="walker-teacher-proposal-round-v1",
        execution_complete=True,
        status="teacher_competence_negative_no_child_fits",
        work={},
        teacher_summaries={"0": {"independent": summary()}},
        evaluations={},
        checkpoint_records=[
            dict(name="s0_parent", actor_hash="1" * 64),
            dict(name="s0_independent_child", actor_hash="2" * 64),
        ],
    )
    run, root = source(tmp_path, result=result)
    path, _ = diagnostics(tmp_path)
    before = {str(p): sha(p) for p in run.rglob("*") if p.is_file()}
    out = tmp_path / "report"
    receipt = module.render(run, out, diagnostics=path)
    metrics = read_csv(out / "full_data_metrics.csv")
    assert [row["action_mse"] for row in metrics] == ["2.0", "1.0", "3.0"]
    assert [row["dataset"] for row in metrics] == [
        "own_acquisition",
        "own_acquisition",
        "held_trajectory",
    ]
    assert all(row["reduction"] == "mean_over_rows_and_six_actions" for row in metrics)
    agreement = read_csv(out / "teacher_action_agreement.csv")[0]
    assert agreement["action_agreement"] == "6.0"
    assert agreement["episode"] == "teacher_s0_independent_00"
    assert "sum_six" in agreement["reduction"]
    episode = read_csv(out / "episode_outcomes.csv")[0]
    assert episode["steps"] == "2" and episode["executed_prefix_cost"] == "22.0"
    assert episode["absorbing_cost"] == "4990.0" and episode["terminal_cost"] == "20.0"
    assert read_csv(out / "student_outcomes.csv") == []
    assert read_csv(out / "paired_gap.csv") == []
    assert "Useful learning was not measured" in (out / "report.md").read_text()
    assert receipt["source_ledger_binding"]["sha256"] == "a" * 64
    assert receipt["inputs"]["saved_diagnostics"]["sha256"] == sha(path)
    assert before == {str(p): sha(p) for p in run.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "problem", ("targets", "actor", "duplicate", "orphan", "row_count", "guide_actor")
)
def test_bad_saved_joins_fail_before_output_creation(tmp_path, problem):
    result = dict(
        status="worker_error",
        execution_complete=False,
        work={},
        checkpoint_records=[
            dict(name="s0_parent", actor_hash="1" * 64),
            dict(name="s0_independent_child", actor_hash="2" * 64),
        ],
    )
    run, _ = source(tmp_path, result=result)
    path, payload = diagnostics(tmp_path)
    if problem == "targets":
        payload["events"][1]["targets"]["key"] = "wrong_actions"
    elif problem == "actor":
        payload["events"][1]["actor_hash"] = "9" * 64
    elif problem == "duplicate":
        payload["events"].append(deepcopy(payload["events"][0]))
    elif problem == "orphan":
        payload["events"][-1]["episode_raw"] = dict(file="arrays.bin", group="unknown")
    elif problem == "guide_actor":
        payload["events"][-1]["actor_hash"] = "9" * 64
    else:
        payload["events"][-1]["rows"] = 1000
    save(path, payload)
    with pytest.raises(ValueError):
        module.render(run, tmp_path / "report", diagnostics=path)
    assert not (tmp_path / "report").exists()


def test_complete_and_partial_curves_keep_every_recorded_point_and_hash(tmp_path, monkeypatch):
    run, root = source(
        tmp_path,
        capsule=True,
        result=dict(
            status="worker_error",
            execution_complete=False,
            error=dict(type="SyntheticError", message="retained"),
            work={},
        ),
    )
    parent = curve(root)
    child = curve(root, "s0_proposal_guided_child", partial=True)
    captured = []
    original = module._figures

    def figures(rows, out):
        captured.extend(deepcopy(rows))
        original(rows, out)

    monkeypatch.setattr(module, "_figures", figures)
    out = tmp_path / "report"
    receipt = module.render(run, out)
    assert [row["update"] for row in captured] == [1, 2, 2001, 2002]
    assert [row["partial"] for row in captured] == [False, False, True, True]
    assert len(read_csv(out / "learning_curves.csv")) == 4
    assert (out / "learning_curves.png").stat().st_size > 1000
    assert (out / "learning_curves.pdf").read_bytes().startswith(b"%PDF")
    assert receipt["inputs"]["learning/" + parent.name]["sha256"] == sha(parent)
    assert receipt["inputs"]["learning/" + child.name]["sha256"] == sha(child)
    assert receipt["source_sha256"] == sha(module.__file__)
    assert receipt["source_read_bytes"] == sum(row["bytes"] for row in receipt["inputs"].values())
    for name, binding in receipt["outputs"].items():
        assert sha(out / name) == binding["sha256"]
        assert (out / name).stat().st_size == binding["bytes"]
    assert "Incomplete execution" in (out / "report.md").read_text()
    assert (
        receipt["network_forwards"]
        == receipt["physics_controls"]
        == receipt["optimizer_steps"]
        == 0
    )


def test_only_true_saved_gap_contexts_are_differenced(tmp_path):
    result = dict(
        status="learning_or_teacher_gate_negative",
        execution_complete=True,
        work={},
        teacher_summaries={"0": {"independent": summary(10, 1000, True)}},
        evaluations={"primary": {"0": {"parent": summary(100, 1000, True)}}},
        paired_teacher_student_gap={
            "0": {
                "independent": dict(
                    teacher=summary(15, 1000, True),
                    student=summary(22, 1000, True),
                    seeds=[123],
                    child_minus_teacher_cost=[7.0],
                    scope="exact reset",
                )
            }
        },
    )
    run, _ = source(tmp_path, result=result)
    module.render(run, tmp_path / "report")
    rows = read_csv(tmp_path / "report/paired_gap.csv")
    assert len(rows) == 1 and rows[0]["seed"] == "123"
    assert rows[0]["child_minus_teacher_cost"] == "7.0"
    assert len(read_csv(tmp_path / "report/teacher_outcomes.csv")) == 2
    assert len(read_csv(tmp_path / "report/student_outcomes.csv")) == 2


def test_error_without_worker_summary_retains_observed_episode_prefix(tmp_path):
    run, _ = source(tmp_path, result=dict(status="worker_error", execution_complete=False, work={}))
    path, _ = diagnostics(tmp_path)
    module.render(run, tmp_path / "report", diagnostics=path)
    row = read_csv(tmp_path / "report/teacher_outcomes.csv")[0]
    assert row["episodes"] == "1" and row["controls"] == "2"
    assert row["roster_status"] == "observed_prefix_only"
    assert read_csv(tmp_path / "report/student_outcomes.csv") == []
    text = (tmp_path / "report/report.md").read_text()
    assert "observed_prefix_only" in text and "saved_episode_events" in text


def test_launch_without_worker_result_is_explicitly_incomplete(tmp_path):
    run, root = source(tmp_path)
    save(root / "worker_exit.json", dict(exit_code=-9))
    out = tmp_path / "report"
    receipt = module.render(run, out)
    text = (out / "report.md").read_text()
    assert "launch_attempt_without_worker_result" in text
    assert "Incomplete execution" in text and '"completion_unknown": true' in text
    assert receipt["source_result_sha256"] == sha(root / "result.json")
    assert read_csv(out / "student_outcomes.csv") == []


def test_curve_count_mismatch_is_not_padded_or_interpolated(tmp_path):
    run, root = source(tmp_path)
    path = curve(root)
    saved = json.loads(path.read_text())
    saved["completed_updates"] = 2000
    save(path, saved)
    with pytest.raises(ValueError, match="update count"):
        module.render(run, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_optional_diagnostics_cli_uses_only_saved_json(tmp_path, capsys):
    run, _ = source(tmp_path)
    path, _ = diagnostics(tmp_path)
    assert (
        module.main(
            ["--run", str(run), "--out", str(tmp_path / "report"), "--saved-diagnostics", str(path)]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["raw_arrays_decoded"] == receipt["checkpoints_loaded"] == 0


def test_external_array_checkpoint_and_npz_paths_are_never_opened(tmp_path, monkeypatch):
    run, root = source(tmp_path)
    path, _ = diagnostics(tmp_path)
    forbidden = [root / "evidence/arrays.bin", root / "inputs/train.npz", root / "checkpoints/p.pt"]
    for target in forbidden:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"intentionally not a valid numerical artifact")
    original = type(root).read_bytes

    def guarded(target):
        if target in forbidden:
            raise AssertionError("Reporting opened numerical evidence")
        return original(target)

    monkeypatch.setattr(type(root), "read_bytes", guarded)
    receipt = module.render(run, tmp_path / "report", diagnostics=path)
    assert receipt["raw_arrays_decoded"] == receipt["checkpoints_loaded"] == 0
    assert "declared" in receipt["provenance_scope"]


def test_optional_mechanism_counts_preserve_exact_origins_without_learning_claim(tmp_path):
    run, _ = source(tmp_path)
    path, payload = diagnostics(tmp_path)
    counts = dict(
        decisions=3,
        selected_origin_counts={"bank:257:proposal_sample": 1, "protected_guard_fallback": 2},
        baseline_verification_status_counts={"ok": 3},
        guided_verification_status_counts={"ok": 1, "guard_fallback": 2},
    )
    payload["mechanism_summary"] = {"0": {"proposal_guided": counts}}
    save(path, payload)
    out = tmp_path / "report"
    receipt = module.render(run, out, diagnostics=path)
    row = read_csv(out / "mechanism_summary.csv")[0]
    assert json.loads(row["selected_origin_counts"]) == counts["selected_origin_counts"]
    assert row["protected_guard_fallback_count"] == "2"
    assert row["decisions"] == "3" and row["unique_selected_origins"] == "2"
    assert receipt["table_rows"]["mechanism_summary.csv"] == 1
    assert "No admission criterion is added" in (out / "report.md").read_text()
    payload["mechanism_summary"]["0"]["proposal_guided"]["decisions"] = 4
    save(path, payload)
    with pytest.raises(ValueError, match="count map"):
        module.render(run, tmp_path / "bad_report", diagnostics=path)
