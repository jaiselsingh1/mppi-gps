from __future__ import annotations

import copy
import csv
import gzip
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.gps.mppi_gps.reporting import similarity_diagnostic as diagnostic


def case(common: int = 100, prefix_mean: float = 4) -> dict:
    error = ((common / 1000) * prefix_mean + 4 * (1 - common / 1000)) ** 0.5
    terminal = 20 if common < 1000 else 0
    tail = 5 * (1000 - common)
    observed = dict(
        tag="synthetic",
        seed=1,
        controls=common,
        horizon=1000,
        done=common < 1000,
        raw_cost=500,
        native_cost=500 + terminal,
        native_terminal=terminal,
        cost=500 + terminal + tail,
        absorbing_cost=tail,
        walking=False,
        survived=common == 1000,
    )
    return dict(
        case_id="development_00_s0_mppi",
        group="development_00",
        learner="development_00_s0",
        source="mppi",
        split="development",
        teacher=dict(horizon=1000),
        reproduction=observed,
        evaluations=[observed],
        motion_components=dict(
            error=error, common_controls=common, capped_component_means=[prefix_mean] * 4
        ),
        post_motion=error,
        scores=dict.fromkeys(diagnostic.CANDIDATES, 0.5),
    )


def write_fixture(root: Path) -> None:
    source = case()
    rows, _ = diagnostic.analyze_cases([source])
    (root / "run/worker").mkdir(parents=True)
    (root / "report").mkdir()
    (root / "run/worker/cases.json").write_text(json.dumps([source]))
    (root / "run/config.json").write_text(
        json.dumps(
            dict(
                prediction=dict(minimum_motion_std=0.1),
                source_commit="synthetic",
                runtime=dict(control_dt=0.008),
            )
        )
    )
    table = dict(rows[0], post_motion=rows[0]["frozen_motion"])
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(table))
    writer.writeheader()
    writer.writerow(table)
    with gzip.open(root / "report/cases.csv.gz", "wt") as stream:
        stream.write(output.getvalue())


def test_frozen_score_can_be_duration_invariant_with_nonempty_overlap():
    values = []
    for common in (30, 100, 1000):
        values.append(diagnostic.decompose_motion(case(common)["motion_components"], 1000))
    assert [value["frozen_motion_reconstructed"] for value in values] == [2, 2, 2]
    assert [value["overlap_only_motion"] for value in values] == [2, 2, 2]
    assert values[0]["frozen_missing_term"] > values[-1]["frozen_missing_term"]


def test_overlap_measure_exposes_short_prefix_error_without_missing_penalty():
    value = diagnostic.decompose_motion(case(30, 1)["motion_components"], 1000)
    assert value["overlap_only_motion"] == 1
    assert value["frozen_motion_reconstructed"] == pytest.approx(3.91**0.5)


@pytest.mark.parametrize("common", [0, -1, 1001])
def test_empty_or_invalid_overlap_is_missing_not_zero(common):
    with pytest.raises(ValueError, match="nonempty actual overlap"):
        diagnostic.decompose_motion(
            dict(error=2, common_controls=common, capped_component_means=[1] * 4), 1000
        )


def test_mismatched_saved_motion_fails():
    value = case()["motion_components"]
    value["error"] = 1
    with pytest.raises(ValueError, match="reconstruct"):
        diagnostic.decompose_motion(value, 1000)


def test_missing_panel_evidence_fails_instead_of_zero_filling():
    value = case()
    del value["evaluations"][0]["absorbing_cost"]
    with pytest.raises(ValueError, match="Missing saved evidence: absorbing_cost"):
        diagnostic.analyze_cases([value])


def test_spearman_handles_ties_and_constant_scores():
    assert diagnostic.descriptive_spearman([1, 1, 2, 3], [4, 4, 2, 1]) == pytest.approx(-1)
    assert diagnostic.descriptive_spearman([1, 1], [2, 3]) is None


def test_duplicate_cases_and_case_table_join_fail():
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostic.analyze_cases([case(), copy.deepcopy(case())])
    rows, _ = diagnostic.analyze_cases([case()])
    with pytest.raises(ValueError, match="roster"):
        diagnostic._join_table(rows, [])


def test_actual_saved_decomposition_matches_review_without_promoting_it_to_prediction():
    capsule = Path(__file__).resolve().parents[1] / "results/similarity_j1"
    cases = json.loads((capsule / "run/worker/cases.json").read_text())
    rows, summary = diagnostic.analyze_cases(cases)
    mppi = summary["outcomes"]["mppi"]
    assert len(rows) == 32
    assert mppi["standalone_walking"] == 0
    assert mppi["standalone_mean_controls_range"] == [32.6875, 379.4375]
    assert mppi["frozen_motion_population_std"] == pytest.approx(0.03849315344778441)
    assert mppi["reproduction_aggregate_tail_fraction"] == pytest.approx(0.8601238691794663)
    assert mppi["standalone_aggregate_tail_fraction"] == pytest.approx(0.9044989299634594)
    assert mppi["descriptive_score_survival_spearman"]["B"] == pytest.approx(0.4088235294117647)
    assert summary["predictive_validation"] is False
    episodes = diagnostic.duration_rows(cases, 0.008)
    assert len(episodes) == 544
    assert sum(row["episode_kind"] == "standalone" for row in episodes) == 512
    assert sum(row["episode_kind"] == "reproduction" for row in episodes) == 32
    assert all(row["time_to_fall_seconds"] is None for row in episodes if row["right_censored"])


def test_saved_cli_emits_provenance_and_imports_no_simulator(tmp_path):
    root = tmp_path / "capsule"
    write_fixture(root)
    output = tmp_path / "supplement"
    program = (
        "import sys; from pathlib import Path; "
        "from src.gps.mppi_gps.reporting.similarity_diagnostic import supplement; "
        "supplement(Path(sys.argv[1]), Path(sys.argv[2])); "
        "assert not any(k.split('.')[0] in ('torch','mujoco','warp') for k in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(root), str(output)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["actual_cases"] == 1
    assert receipt["actual_episode_durations"] == 2
    assert all(value == 0 for value in receipt["new_scientific_work"].values())
    assert set(receipt["input_files"]) == {
        "run/worker/cases.json",
        "run/config.json",
        "report/cases.csv.gz",
    }
    assert (output / "motion_and_duration.png").stat().st_size > 1000
    assert (output / "episodes.csv").stat().st_size > 100
    with pytest.raises(FileExistsError):
        diagnostic.supplement(root, output)


def test_missing_saved_file_stops_before_output_creation(tmp_path):
    root = tmp_path / "capsule"
    root.mkdir()
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="Missing or unsafe saved evidence"):
        diagnostic.supplement(root, output)
    assert not output.exists()


def test_output_cannot_extend_immutable_capsule(tmp_path):
    root = tmp_path / "capsule"
    write_fixture(root)
    with pytest.raises(ValueError, match="outside the immutable capsule"):
        diagnostic.supplement(root, root / "new-output")
    assert not (root / "new-output").exists()


def test_duration_fall_and_right_censored_full_horizon_are_distinct():
    value = case(1000)
    value["reproduction"] = dict(
        value["reproduction"],
        done=True,
        survived=False,
        native_terminal=20,
        native_cost=520,
        cost=520,
    )
    rows = diagnostic.duration_rows([value], 0.008)
    assert [row["observed_seconds"] for row in rows] == [8, 8]
    assert [row["fall_event"] for row in rows] == [False, True]
    assert [row["right_censored"] for row in rows] == [True, False]
    assert [row["time_to_fall_seconds"] for row in rows] == [None, 8]
    assert rows[0]["walking"] is False  # Full survival alone is retention.
    assert rows[1]["native_terminal_cost"] == 20
    assert rows[1]["absorbing_tail_cost"] == 0


def test_short_duration_requires_saved_fall_event_and_complete_costs():
    value = case(100)
    rows = diagnostic.duration_rows([value], 0.008)
    assert rows[0]["time_to_fall_seconds"] == 0.8
    assert rows[0]["raw_cost"] == 500
    assert rows[0]["native_cost"] == 520
    assert rows[0]["native_terminal_cost"] == 20
    assert rows[0]["absorbing_tail_cost"] == 4500
    assert rows[0]["complete_cost"] == 5020
    value["reproduction"] = dict(value["reproduction"], done=False)
    with pytest.raises(ValueError, match="fall/survival/horizon"):
        diagnostic.duration_rows([value], 0.008)


def test_missing_duration_evidence_is_not_censored_or_fabricated():
    value = case()
    del value["reproduction"]["done"]
    with pytest.raises(ValueError, match="Missing saved duration evidence: done"):
        diagnostic.duration_rows([value], 0.008)


def test_invalid_duration_cost_decomposition_fails():
    value = case()
    value["reproduction"]["cost"] += 20
    with pytest.raises(ValueError, match="cost decomposition"):
        diagnostic.duration_rows([value], 0.008)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), True])
def test_invalid_control_interval_fails(interval):
    with pytest.raises(ValueError, match="control interval"):
        diagnostic.duration_rows([case()], interval)
