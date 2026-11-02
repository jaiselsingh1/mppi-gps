"""Finite H5 storage proofs use byte arithmetic and tiny synthetic records."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.proposal_evidence import BinaryEvidenceMeter
from src.gps.mppi_gps.proposal_storage import (
    CONTRACT,
    GIB,
    QUOTAS,
    bounded_json_payload,
    check_native_dimensions,
    episode_payload_bound,
    live_query_payload_bound,
    model_query_payload_bound,
    storage_census,
)


def recipe():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "configs/studies/teacher_proposal_h5_0.json").read_text())


def test_full_roster_payload_and_records_fit_without_compression():
    result = storage_census(recipe())
    assert result["payload_upper_bytes"] == 390_594_466_128
    assert result["array_groups_upper"] == 1_722_817
    assert result["ledger_records_upper"] == 3_403_015
    assert result["quotas"] == dict(
        payload=384 * GIB, ledger=80 * GIB, index=32 * GIB, external=16 * GIB
    )
    assert sum(result["quotas"].values()) == 512 * GIB
    assert result["payload_components"]["native_action_and_post_states"] == 267_759_820_800
    assert result["payload_components"]["guidance_predictions"] == 95_956_586_496
    assert result["payload_components"]["private_planner_unicode"] == 7_077_888_000
    assert result["payload_components"]["predecision_planner_unicode"] == 3_538_944_000
    assert result["payload_components"]["final_cache_weights"] == 221_184_000
    assert result["payload_components"]["unexpected_failure_context"] == 2048
    assert result["total_upper_bytes"] < 512 * GIB
    assert result["fits"]
    assert "Conditional" in result["proof_scope"]
    assert "second ReservationLedger event sink" in result["proof_scope"]


def test_index_has_two_lines_per_group_and_emergency_is_not_free():
    result = storage_census(recipe())
    assert result["index_upper_bytes"] == 1_722_817 * 2 * 8192
    assert result["ledger_upper_bytes"] == 3_403_015 * 16384
    assert result["ledger_components"]["fit_update_observer"] == 24_000
    assert result["ledger_components"]["fit_initial_state_observer"] == 12
    assert result["ledger_components"]["predecision_context_events"] == 108_000
    assert result["array_group_components"]["unexpected_failure_context_groups"] == 1
    assert result["ledger_components"]["emergency_events"] == 8
    assert result["ledger_components"]["query_request_and_result"] == 2 * (324_108 + 456_000)
    # ReservationLedger does not duplicate the canonical query/measurement log.
    assert "reservation_events" not in result["ledger_components"]


@pytest.mark.parametrize("trajectories", (1, 256))
def test_native_model_shapes_cover_exact_uncompressed_payload(trajectories):
    controls = trajectories * 48
    # Canonical input/output shapes, with the maximum integration context.
    arrays = (
        np.zeros(19),
        np.zeros((trajectories, 48, 6)),
        np.zeros(9),
        np.zeros(128),
        np.zeros((trajectories, 48, 19)),
        np.zeros(trajectories),
        np.zeros((trajectories, 48, 0)),
        np.array("native_planning_horizon_score"),
    )
    actual = sum(array.nbytes for array in arrays)
    assert model_query_payload_bound(controls, trajectories) >= actual
    assert (
        model_query_payload_bound(controls, trajectories)
        == controls * 200 + trajectories * 8 + 2048
    )


@pytest.mark.parametrize("controls,trajectories", ((47, 1), (96, 2), (0, 0), (48.0, 1), (48, True)))
def test_uncosted_native_batches_reject(controls, trajectories):
    with pytest.raises(ValueError):
        model_query_payload_bound(controls, trajectories)


def test_live_payload_counts_both_full_integration_snapshots():
    inputs = (6 + 19 + 9 + 128 + 1 + 3) * 8
    outputs = (17 + 1 + 19 + 9 + 128) * 8 + 1 + 1024 * 4
    assert live_query_payload_bound() == inputs + outputs == 6817


def test_complete_episode_counts_real_terminal_and_unicode_list_delimiters():
    controls = 1000
    numeric = (156 * 1001 + 25 * 1000) * 8
    infos = (1000 * 2048 + 999 + len('{"rows":[]}')) * 4
    assert episode_payload_bound(controls) == numeric + infos + 24
    assert infos == 8_196_040 < CONTRACT["max_array_bytes"]
    assert episode_payload_bound(0) == 156 * 8 + 11 * 4 + 24
    assert episode_payload_bound(1000) < 32 * 1024**2
    with pytest.raises(ValueError):
        episode_payload_bound(1001)


def test_json_census_charges_numpy_unicode_instead_of_utf8():
    text = json.dumps(dict(symbol="λ", rows=[1, 2]), sort_keys=True, separators=(",", ":"))
    assert text.isascii()
    expected = np.array(text).nbytes
    assert bounded_json_payload(text, text_bytes=len(text), purpose="test") == expected
    assert expected == 4 * len(text.encode("ascii"))
    with pytest.raises(ValueError, match="JSON text"):
        bounded_json_payload(text, text_bytes=len(text) - 1, purpose="test")
    with pytest.raises(ValueError, match="ASCII"):
        bounded_json_payload('"λ"', text_bytes=1024, purpose="test")
    with pytest.raises(json.JSONDecodeError):
        bounded_json_payload("invalid", text_bytes=1024, purpose="test")


@pytest.mark.parametrize("text", ("NaN", "Infinity", "-Infinity"))
def test_nonfinite_json_constants_cannot_bypass_finite_metadata_contract(text):
    with pytest.raises(ValueError, match="Nonfinite"):
        bounded_json_payload(text, text_bytes=1024, purpose="test")


@pytest.mark.parametrize("integration", (1, 91, 128))
def test_native_dimension_boundary_accepts_only_costed_model(integration):
    check_native_dimensions(state=19, action=6, integration=integration, sensors=0)


@pytest.mark.parametrize(
    "changed",
    (
        dict(state=20),
        dict(action=7),
        dict(integration=129),
        dict(integration=0),
        dict(sensors=1),
        dict(state=True),
    ),
)
def test_changed_model_dimensions_cannot_enter_old_census(changed):
    with pytest.raises(ValueError):
        check_native_dimensions(**(dict(state=19, action=6, integration=91, sensors=0) | changed))


def test_reduced_or_changed_envelope_fails_before_allocating_arrays(monkeypatch):
    changed = deepcopy(recipe())
    changed["budgets"]["storage_bytes"] -= 1
    with pytest.raises(ValueError, match="Version"):
        storage_census(changed)
    monkeypatch.setitem(QUOTAS, "payload", 1)
    monkeypatch.setitem(QUOTAS, "external", 512 * GIB - QUOTAS["ledger"] - QUOTAS["index"] - 1)
    with pytest.raises(ValueError, match="does not fit"):
        storage_census(recipe())


def test_actual_selection_array_names_fit_pinned_index_limit(tmp_path):
    # Current maximum-group schema, including four planner JSON arrays.
    names = (
        "raw_costs feasible proximity agreement_squared support predictions baseline_scores "
        "guided_scores selected_plan baseline_mean guided_mean baseline_mean_raw_cost "
        "guided_mean_raw_cost protected_before proposal_before protected_after proposal_after "
        "guided_mean_support guided_mean_predictions guided_mean_agreement_squared "
        "protected_final_weights proposal_final_weights"
    ).split()
    assert len(names) == 22 < CONTRACT["max_arrays_per_group"]
    with BinaryEvidenceMeter(
        tmp_path / "evidence",
        max_controls=0,
        max_neural_rows=0,
        max_bytes=1024**2,
        max_header_bytes=CONTRACT["max_index_line_bytes"],
        max_event_bytes=CONTRACT["max_event_line_bytes"],
        max_arrays_per_group=CONTRACT["max_arrays_per_group"],
    ) as owner:
        reference = owner.save_arrays("h5_event_000000000", **dict.fromkeys(names, np.array(0.0)))
        lines = owner.index_path.read_bytes().splitlines(keepends=True)
        assert len(lines) == 2
        assert all(len(line) <= 8192 for line in lines)
        assert len(reference["arrays"]) == 22


def test_storage_census_imports_no_array_policy_or_simulator_stack():
    code = """
import sys
from src.gps.mppi_gps.proposal_storage import QUOTAS
assert sum(QUOTAS.values()) == 512 * 1024**3
for prefix in ('numpy','torch','mujoco','h5py','scipy','matplotlib','src.envs','src.mppi'):
    assert not any(name == prefix or name.startswith(prefix+'.') for name in sys.modules),prefix
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)
