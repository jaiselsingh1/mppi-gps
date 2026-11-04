"""L3 ownership/continuation guards without a model, optimizer or simulator."""

from copy import deepcopy

import pytest

from src.gps.mppi_gps.experiments.local_coupling_multiround import include_reaped_cpu
from src.gps.mppi_gps.local_coupling_learning import incoming_lineage_matches


def recipe():
    return dict(
        schema="local-coupling-learning-l3-1",
        initial_cumulative_updates=4000,
        learning=dict(updates=2000),
        inputs=dict(
            initial_streams=[
                dict(stream=1, arm="positive", files={"stream.json": dict(sha256="old")})
            ]
        ),
    )


def test_exact_l2_incoming_package_is_continued_without_rewrapping():
    job = dict(
        round=1, stream=1, arm="positive", incoming=dict(sha256="old"), config=dict(sha256="new")
    )
    metadata = dict(round=1, stream=1, arm="positive", config_sha256="prior-l2")
    assert incoming_lineage_matches(recipe(), job, metadata, dict(updates=4000))
    for bad in ("other", "rewrapped"):
        changed = deepcopy(job)
        changed["incoming"]["sha256"] = bad
        assert not incoming_lineage_matches(recipe(), changed, metadata, dict(updates=4000))
    assert not incoming_lineage_matches(recipe(), job, metadata, dict(updates=2000))


@pytest.mark.parametrize("r", range(2, 9))
def test_every_round_uses_same_arm_immediately_previous_checkpoint(r):
    job = dict(round=r, stream=1, arm="positive", config=dict(sha256="new"))
    metadata = dict(round=r - 1, stream=1, arm="positive", config_sha256="new", fit_updates=2000)
    state = dict(updates=4000 + (r - 1) * 2000)
    assert incoming_lineage_matches(recipe(), job, metadata, state)
    for key, bad in (("round", 0), ("arm", "independent"), ("config_sha256", "old"), ("stream", 0)):
        changed = deepcopy(metadata)
        changed[key] = bad
        assert not incoming_lineage_matches(recipe(), job, changed, state)
    assert not incoming_lineage_matches(recipe(), job, metadata, dict(updates=4000))


def test_reaped_native_children_are_charged_once_while_coordinator_lives():
    usage = dict(cpu_seconds=17, observed_pids=[1, 2, 4])
    records = [dict(reaped_child_cpu_seconds=100), dict(reaped_child_cpu_seconds=200)]
    assert include_reaped_cpu(usage, 2, records)["cpu_seconds"] == 217
    assert include_reaped_cpu(usage, 99, records)["cpu_seconds"] == 17
    assert usage["cpu_seconds"] == 17
