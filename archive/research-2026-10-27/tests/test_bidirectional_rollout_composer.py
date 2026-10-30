"""Pure reference-level closure fixtures; no engine, optimizer or actor work."""
import pytest

from src.gps.bidirectional.rollout_audit import reference_linkage


def row(identifier, kind, request=None):
    value = dict(id=identifier, kind=kind)
    if request is not None:
        value["request_id"] = request
    return value


def test_reference_level_links_accept_failed_optimizer_and_complete_point():
    rows = [row("p", "rollout_point_requested"), row("pc", "rollout_point_complete", "p"),
        row("o", "rollout_optimizer_requested"), row("oe", "rollout_optimizer_error", "o")]
    assert reference_linkage(rows) == 2


@pytest.mark.parametrize("rows", [
    [row("o", "rollout_optimizer_requested")],
    [row("oe", "rollout_optimizer_error", "missing")],
    [row("oe", "rollout_optimizer_error", "o"), row("o", "rollout_optimizer_requested")],
    [row("o", "rollout_optimizer_requested"), row("pc", "rollout_point_complete", "o")],
    [row("o", "rollout_optimizer_requested"), row("oc", "rollout_optimizer_complete", "o"),
        row("oe", "rollout_optimizer_error", "o")],
    [row("p", "rollout_point_requested"), row("p", "rollout_point_requested")],
])
def test_reference_level_links_reject_unresolved_unbound_wrong_family_or_duplicates(rows):
    with pytest.raises(ValueError):
        reference_linkage(rows)
