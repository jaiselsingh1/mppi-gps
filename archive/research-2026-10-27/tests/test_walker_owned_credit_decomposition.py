"""Synthetic arithmetic fixtures only: no raw experiment, actor or engine access."""
import hashlib

import numpy as np
import pytest

from scripts.diagnostics.decompose_walker_owned_credit import (
    BoundReader, GAMMA, absorption, split_return,
)


def test_two_falls_before48_have_identical_constant_far_tail():
    early = np.ones(10); early[-1] += 20.
    later = np.ones(30); later[-1] += 20.
    a, b = split_return(early, 1000, True), split_return(later, 1000, True)
    assert a["far_actual"] == b["far_actual"] == 0.
    assert a["far_absorbing"] == b["far_absorbing"] == absorption(48, 1000)
    assert a["near_absorbing"] > b["near_absorbing"] > 0.
    assert a["normalized"]["far_absorbing"] == b["normalized"]["far_absorbing"]
    for result in (a,b):
        assert result["near"]+result["far_actual"]+result["far_absorbing"] == pytest.approx(result["total"], abs=1e-10)


def test_actor_past48_has_measured_far_cost_separate_from_absorption():
    costs = np.full(70, 2.); costs[-1] += 20.
    result = split_return(costs, 1000, True)
    assert result["executed_beyond_H"] and not result["healthy_complete"]
    assert result["near_absorbing"] == 0.
    assert result["far_actual"] == pytest.approx(np.dot(GAMMA**np.arange(48,70), costs[48:]))
    assert result["far_actual"] > 0.
    assert result["far_absorbing"] == absorption(70,1000)


def test_healthy_complete_horizon_has_only_measured_near_and_far_cost():
    costs = np.full(1000, .4)
    result = split_return(costs, 1000, False)
    assert result["healthy_complete"] and result["executed_beyond_H"]
    assert result["near_absorbing"] == result["far_absorbing"] == 0.
    assert result["near"]+result["far_actual"] == pytest.approx(np.dot(GAMMA**np.arange(1000), costs))
    with pytest.raises(ValueError, match="incomplete"):
        split_return(costs[:50], 1000, False)


def test_descriptive_reader_refuses_unbound_or_changed_evidence(tmp_path):
    path = tmp_path/"raw.json"
    path.write_bytes(b'{"closed":true}\n')
    reader = BoundReader(tmp_path, {"raw.json":hashlib.sha256(path.read_bytes()).hexdigest()})
    assert reader.obj("raw.json") == {"closed":True}
    path.write_bytes(b'{"closed":false}\n')
    with pytest.raises(ValueError, match="hash changed"):
        reader.obj("raw.json")
    with pytest.raises(ValueError, match="absent"):
        reader.data("new_target.json")
