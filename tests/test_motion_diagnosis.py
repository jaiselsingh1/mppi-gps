"""Synthetic controls for a single frozen saved-motion segment diagnostic."""

import builtins
import json

import numpy as np
import pytest

from src.gps.mppi_gps import motion_diagnosis as motion


def row(count=150, index=0):
    controls = np.arange(count)
    clock = 0.008 * controls
    states = np.zeros((count, 19), dtype=float)
    states[:, 0] = clock
    states[:, 1] = clock * 1.5
    states[:, 2] = 1.2 + 0.05 * np.sin(clock)
    states[:, 3:10] = np.sin(clock[:, None] + np.arange(7))
    states[:, 10:] = np.cos(clock[:, None] + np.arange(9))
    actions = np.tanh(clock[:, None] + 0.1 * np.arange(6))
    predictions = actions * 0.5
    return dict(
        id="synthetic-" + str(index),
        source="synthetic",
        index=index,
        states=states,
        actions=actions,
        reference_predictions=predictions,
        terminal_state=None,
    )


def recorded(row):
    errors = (row["actions"] - row["reference_predictions"]) ** 2
    row["recorded_E"] = float(np.mean(errors))
    row["recorded_bins"] = []
    for start in range(0, len(errors), 100):
        stop = min(start + 100, len(errors))
        row["recorded_bins"].append(
            dict(
                start_control=start,
                stop_control_exclusive=stop,
                rows=stop - start,
                mse=float(np.mean(errors[start:stop])),
                per_actuator_mse=np.mean(errors[start:stop], axis=0).tolist(),
            )
        )
    return row


def test_distance_matches_independent_scalar_reduction_and_components():
    first = np.arange(2 * 25 * 17).reshape(2, 25, 17) / 37
    second = np.cos(first)
    actual, components = motion.segment_distances(first, second)
    expected = []
    for a, b in zip(first, second, strict=True):
        blocks = np.zeros(3)
        for x, y in zip(a, b, strict=True):
            blocks[0] += (x[0] - y[0]) ** 2
            for j in range(1, 8):
                wrapped = ((x[j] - y[j] + np.pi) % (2 * np.pi)) - np.pi
                blocks[1] += wrapped**2
            for j in range(8, 17):
                blocks[2] += (x[j] - y[j]) ** 2
        expected.append(blocks / (25 * 17))
    np.testing.assert_allclose(components, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(actual**2, np.sum(expected, axis=1), atol=1e-12)


def test_global_translation_and_self_controls_preserve_distance():
    first, translated = row(50), row(50, 1)
    translated["states"][:, 1] += 123.0
    result = motion.analyze([first, translated], {})
    assert result["cross_pairs"][0]["best"]["R"] == 0
    for a, b in zip(
        result["trajectories"][0]["curves"], result["trajectories"][1]["curves"], strict=True
    ):
        np.testing.assert_allclose(
            list(a["displacement"].values()), list(b["displacement"].values()), atol=1e-12
        )
        assert a["physical"] == b["physical"]
    physical = first["states"][:, 2:19][None]
    distance, contributions = motion.segment_distances(physical, physical)
    np.testing.assert_array_equal(distance, [0])
    np.testing.assert_array_equal(contributions, [[0, 0, 0]])


def test_hinge_wrap_is_invariant_but_native_plot_coordinates_are_retained():
    first, wrapped = row(25), row(25, 1)
    wrapped["states"][:, 3:10] += 2 * np.pi
    result = motion.analyze([first, wrapped], {})
    assert result["cross_pairs"][0]["best"]["R"] < 1e-14
    a, b = [r["curves"][0]["physical"]["mean"] for r in result["trajectories"]]
    np.testing.assert_allclose(np.array(b)[1:8] - np.array(a)[1:8], 2 * np.pi)
    assert result["representation"]["physical_columns"] == list(range(2, 19))
    assert result["representation"]["scales"] == [1.0] * 17


def test_ordered_segments_distinguish_same_marginal_states():
    sequence = np.zeros((1, 25, 17))
    sequence[0, :, 0] = np.arange(25) / 25
    reversed_sequence = sequence[:, ::-1]
    np.testing.assert_array_equal(np.sort(sequence, axis=1), np.sort(reversed_sequence, axis=1))
    assert motion.segment_distances(sequence, sequence)[0][0] == 0
    assert motion.segment_distances(sequence, reversed_sequence)[0][0] > 0


def test_within_mask_counts_and_lexicographic_ties():
    original = row(150)
    original["states"][:, 2:19] = 0
    result = motion.analyze([original], {})
    within = result["trajectories"][0]["within"]
    assert within["starts"] == [0, 25, 50, 75, 100, 125]
    assert within["eligible_pairs"] == 3
    matrix = within["R_matrix"]
    for a in range(6):
        for b in range(6):
            assert matrix[a][b] == (0.0 if abs(a - b) >= 4 else None)
    assert within["best"]["first_start_control"] == 0
    assert within["best"]["second_start_control"] == 100
    assert result["counts"]["pair_queries"] == 3
    assert result["counts"]["aligned_state_pairs"] == 75


def test_matched_action_and_reference_errors_use_exact_selected_segments():
    first, second = row(25), row(25, 1)
    first["actions"][:] = 0.5
    first["reference_predictions"][:] = 0.25
    second["actions"][:] = -0.5
    second["reference_predictions"][:] = -0.1
    best = motion.analyze([first, second], {})["cross_pairs"][0]["best"]
    assert best["action_mse"] == 1
    assert best["first_reference_E"] == 0.25**2
    assert best["second_reference_E"] == pytest.approx(0.4**2)
    assert sum(best["squared_contributions"].values()) == pytest.approx(best["R"] ** 2)


def test_curves_cover_tail_and_preserve_unclipped_velocity_and_errors():
    original = recorded(row(26))
    original["states"][:, 10] = 12.0  # Physical inspection never clips this velocity.
    summary = motion.analyze([original], {})["trajectories"][0]
    assert [b["rows"] for b in summary["curves"]] == [25, 1]
    assert summary["within"]["starts"] == [0]
    assert summary["curves"][-1]["stop_control_exclusive"] == 26
    assert summary["curves"][0]["physical"]["mean"][8] == 12
    assert not summary["terminal_state_available"]
    weighted = sum(b["rows"] * b["reference_error"]["E"] for b in summary["curves"]) / 26
    assert weighted == pytest.approx(summary["E"], abs=1e-12)
    last_errors = (original["actions"][-1] - original["reference_predictions"][-1]) ** 2
    np.testing.assert_allclose(summary["curves"][-1]["reference_error"]["max"], last_errors)


def test_short_prefixes_stay_distinct_and_are_never_padded():
    result = motion.analyze([row(8), row(14, 1), row(26, 2), row(50, 3)], {})
    for summary in result["trajectories"][:2]:
        assert summary["within"] == dict(starts=[0], eligible_pairs=0, R_matrix=[[None]], best=None)
        assert len(summary["curves"]) == 1
        assert summary["curves"][0]["rows"] == summary["N"]
    cross = result["cross_pairs"]
    assert [(p["kind"], p["H"]) for p in cross] == [
        ("partial", 8),
        ("partial", 8),
        ("partial", 8),
        ("partial", 14),
        ("partial", 14),
        ("full", 25),
    ]
    assert cross[-1]["eligible_pairs"] == 2
    assert result["counts"]["aligned_state_pairs"] == 8 + 8 + 16 + 14 + 28 + 50
    assert result["trajectories"][2]["N"] == 26


def test_existing_K1_scalar_and_bin_bindings_reconcile():
    original = recorded(row(126))
    summary = motion.analyze([original], {})["trajectories"][0]
    assert summary["E"] == original["recorded_E"]
    assert summary["first100_E"] == original["recorded_bins"][0]["mse"]
    original["recorded_E"] += 0.01
    with pytest.raises(ValueError, match="K1 full"):
        motion.analyze([original], {})


@pytest.mark.parametrize("mutation", ["missing", "reordered", "gapped", "error"])
def test_original_bin_binding_mismatch_stops(mutation):
    original = recorded(row(126))
    if mutation == "missing":
        original["recorded_bins"].pop()
    elif mutation == "reordered":
        original["recorded_bins"].reverse()
    elif mutation == "gapped":
        original["recorded_bins"][0]["stop_control_exclusive"] -= 1
    else:
        original["recorded_bins"][0]["mse"] += 0.1
    with pytest.raises(ValueError, match="K1"):
        motion.analyze([original], {})


@pytest.mark.parametrize(
    "mutation",
    [
        "state_shape",
        "action_shape",
        "reference_shape",
        "nonfinite",
        "clock",
        "action_units",
        "terminal_clock",
    ],
)
def test_missing_or_corrupt_array_evidence_stops(mutation):
    original = row(25)
    if mutation == "state_shape":
        original["states"] = original["states"][:, :-1]
    elif mutation == "action_shape":
        original["actions"] = original["actions"][:-1]
    elif mutation == "reference_shape":
        original["reference_predictions"] = original["reference_predictions"][:, :-1]
    elif mutation == "nonfinite":
        original["states"][0, 3] = np.nan
    elif mutation == "clock":
        original["states"][3, 0] += 0.008
    elif mutation == "action_units":
        original["actions"][0, 1] = 2
    else:
        original["terminal_state"] = original["states"][-1].copy()
    with pytest.raises(ValueError):
        motion.analyze([original], {})


def test_identity_and_terminal_availability_are_explicit():
    original = row(25)
    original["terminal_state"] = original["states"][-1].copy()
    original["terminal_state"][0] += 0.008
    assert motion.analyze([original], {})["trajectories"][0]["terminal_state_available"]
    with pytest.raises(ValueError, match="duplicate"):
        motion.analyze([original, original], {})
    with pytest.raises(ValueError, match="order"):
        motion.analyze([row(25, 1)], {})


@pytest.mark.parametrize(
    "config",
    [
        dict(window_controls=50),
        dict(stride_controls=1),
        dict(min_lag_controls=0),
        dict(control_seconds=0.01),
        dict(scales=[2.0] * 17),
        dict(scales=[1.0] * 4),
        dict(expected={}),
    ],
)
def test_probe_contract_cannot_silently_change(config):
    with pytest.raises(ValueError):
        motion.analyze([row(25)], config)


def test_progress_requests_precede_completed_work_and_respect_batch_cap(monkeypatch):
    batches, progress = [], []

    def distance(first, second):
        assert progress[-1]["event"] == "queries_requested"
        assert first.shape == second.shape
        batches.append(first.shape)
        return np.zeros(len(first)), np.zeros((len(first), 3))

    monkeypatch.setattr(motion, "segment_distances", distance)
    result = motion.analyze([row(1000), row(1000, 1)], {}, progress.append)
    assert [s[0] for s in batches] == [666, 666, 1536, 64]
    assert all(s[0] <= 1536 and s[1:] == (25, 17) for s in batches)
    assert [e["event"] for e in progress] == ["queries_requested", "queries_completed"] * 4
    for event in ("queries_requested", "queries_completed"):
        assert (
            sum(e["pairs"] for e in progress if e["event"] == event)
            == result["counts"]["pair_queries"]
        )
        assert (
            sum(e["aligned_state_pairs"] for e in progress if e["event"] == event)
            == result["counts"]["aligned_state_pairs"]
        )


def test_failed_query_retains_request_without_completed_event(monkeypatch):
    progress = []

    def failure(*args):
        raise RuntimeError("synthetic interruption")

    monkeypatch.setattr(motion, "segment_distances", failure)
    with pytest.raises(RuntimeError, match="interruption"):
        motion.analyze([row(150)], {}, progress.append)
    assert progress == [dict(event="queries_requested", pairs=3, aligned_state_pairs=75)]


def test_entire_roster_metadata_census_counts_every_evaluated_query(monkeypatch):
    """Exercise all query ownership with a stub, rather than doing new motion science."""
    actual = dict(pairs=0, aligned=0)

    def distance(first, second):
        actual["pairs"] += len(first)
        actual["aligned"] += first.shape[0] * first.shape[1]
        return np.zeros(len(first)), np.zeros((len(first), 3))

    monkeypatch.setattr(motion, "segment_distances", distance)
    expected = dict(
        trajectories=34,
        rows=30070,
        curve_bins=1205,
        segments=1204,
        cross_trajectory_pairs=561,
        within_queries=19980,
        cross_queries=700806,
        pair_queries=720786,
        aligned_state_pairs=17982374,
    )
    result = motion.analyze(
        [row(n, i) for i, n in enumerate([1000] * 30 + [8, 26, 22, 14])], dict(expected=expected)
    )
    assert result["counts"] == expected
    assert actual == dict(pairs=720786, aligned=17982374)
    assert len(result["cross_pairs"]) == 561
    assert all(
        p["best"]["first_start_control"] == p["best"]["second_start_control"] == 0
        for p in result["cross_pairs"]
    )
    json.dumps(result, allow_nan=False)


def test_analysis_does_not_import_simulator_network_or_raw_reader(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"mujoco", "torch", "h5py"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = motion.analyze([row(25)], {})
    assert result["counts"]["pair_queries"] == 0
