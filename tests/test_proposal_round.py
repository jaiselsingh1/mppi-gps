"""Bounded H5 ordering/admission fixtures; no learner or native physics."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from src.gps.mppi_gps.proposal_round import ARMS, panel_gate, run_round
from src.gps.mppi_gps.recipes import load_recipe


class Operations:
    def __init__(self, *, fall=False, expensive=False):
        self.calls = []
        self.fall, self.expensive = fall, expensive

    @staticmethod
    def episode(seed, *, success=True, cost=1):
        return SimpleNamespace(
            seed=seed,
            complete=True,
            survived=success,
            success=success,
            steps=1000 if success else 10,
            cost=cost,
        )

    def parent(self, stream):
        self.calls.append(("parent", stream["id"]))
        return (stream["id"], "parent")

    def acquire(self, parent, stream, arm, index):
        self.calls.append(("acquire", stream["id"], arm, index))
        return self.episode(
            stream["acquisition_seeds"][index],
            success=not (self.fall and arm == "proposal_guided" and index == 0),
            cost=2 if self.expensive and arm != "independent" else 1,
        )

    def fit(self, parent, episodes, stream, arm):
        assert sum(c[0] == "acquire" for c in self.calls) == 36
        assert parent == (stream["id"], "parent")
        assert len(episodes) == 4 and all(e.success for e in episodes)
        self.calls.append(("fit", stream["id"], arm))
        return (stream["id"], arm)

    def evaluate(self, actor, seed, tag):
        self.calls.append(("evaluate", actor, seed, tag))
        return self.episode(seed, success=actor[1] == "proposal_guided")

    def gap_teacher(self, parent, stream, arm, index):
        self.calls.append(("gap_teacher", stream["id"], arm, index))
        return self.episode(stream["gap_seeds"][index])

    def paired(self, first, second):
        assert first.seed == second.seed

    def record(self, event, **fields):
        pass


def test_fixed_roster_and_all_panels_are_complete_without_automatic_round_two():
    ops = Operations()
    result = run_round(load_recipe("teacher_proposal_h5_0"), ops)
    assert sum(c[0] == "parent" for c in ops.calls) == 3
    assert sum(c[0] == "acquire" for c in ops.calls) == 36
    assert sum(c[0] == "fit" for c in ops.calls) == 9
    assert sum(c[0] == "evaluate" for c in ops.calls) == 402
    assert sum(c[0] == "gap_teacher" for c in ops.calls) == 18
    assert result["comparison_complete"] and result["confirmed_benefit"]
    assert result["second_round_review_may_be_prepared"]
    assert not result["next_round_eligible"] and not result["loop_closed"]


def test_teacher_fall_finishes_fixed_roster_then_stops_without_replacements_or_fits():
    ops = Operations(fall=True)
    result = run_round(load_recipe("teacher_proposal_h5_0"), ops)
    assert sum(c[0] == "acquire" for c in ops.calls) == 36
    assert not any(c[0] in {"fit", "evaluate", "gap_teacher"} for c in ops.calls)
    assert not result["comparison_complete"] and not result["confirmed_benefit"]


def test_cost_screen_does_not_reject_complete_walking_teaching_data_before_fitting():
    ops = Operations(expensive=True)
    result = run_round(load_recipe("teacher_proposal_h5_0"), ops)
    assert sum(c[0] == "fit" for c in ops.calls) == 9
    assert not result["complete_teacher_gate"] and not result["confirmed_benefit"]


def test_retention_without_walking_cannot_pass_and_regression_in_one_stream_blocks():
    panel = {
        str(s): {
            arm: dict(successes=0, survived=16 if arm == "proposal_guided" else 0)
            for arm in ("parent", *ARMS)
        }
        for s in range(3)
    }
    assert not panel_gate(panel)["passed"]
    for per_stream in panel.values():
        per_stream["proposal_guided"]["successes"] = 2
    assert panel_gate(panel)["passed"]
    panel["2"]["independent"]["successes"] = 3
    assert not panel_gate(panel)["passed"]


def test_joint_gain_is_required_in_same_streams_for_both_comparators():
    panel = {
        str(s): {arm: dict(successes=0, survived=16) for arm in ("parent", *ARMS)} for s in range(3)
    }
    for per_stream in panel.values():
        per_stream["proposal_guided"]["successes"] = 2
    panel["0"]["parent"]["successes"] = 1
    panel["1"]["selector_only"]["successes"] = 1
    assert not panel_gate(panel)["passed"]
    original = deepcopy(panel)
    panel_gate(panel)
    assert panel == original
