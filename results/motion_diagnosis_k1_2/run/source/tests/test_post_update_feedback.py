import pytest

from src.gps.mppi_gps.post_update import child_comparison, next_acquisition, teacher_eligibility


def score(cost, steps=1000.0, success=2, survived=2):
    return dict(
        episodes=2,
        mean_cost=cost,
        costs=[cost - 1, cost + 1],
        mean_steps=steps,
        successes=success,
        survived=survived,
    )


def test_successful_disagreeing_teacher_is_eligible():
    raw, guided = score(100), score(104)
    raw["action_disagreement"] = 1.0
    guided["action_disagreement"] = 1000.0
    assert teacher_eligibility(raw, guided)["eligible"]
    assert not teacher_eligibility(raw, score(106))["eligible"]


def test_child_must_beat_both_controls_and_not_shorten_survival():
    assert child_comparison(score(150), score(140), score(130))["passed"]
    assert not child_comparison(score(150), score(130), score(129))["passed"]
    assert not child_comparison(score(150), score(140), score(130, steps=999))["passed"]
    assert not child_comparison(score(150), score(140), score(130, success=1))["passed"]


def test_failed_walking_can_show_incremental_progress_without_qualification():
    result = child_comparison(
        score(4000, 200, 0, 0), score(4200, 180, 0, 0), score(3500, 300, 0, 0)
    )
    assert result["passed"]
    assert result["controls"]["parent"]["paired_cost_deltas_guided_minus_control"] == [
        -500.0,
        -500.0,
    ]


def test_confirmation_binds_next_planner_to_updated_actor_and_preserves_search():
    kwargs = dict(parent_hash="a" * 64, child_hash="b" * 64, rho=0.12)
    absent = next_acquisition(dict(passed=True), None, **kwargs)
    assert not absent["advance"] and absent["actor_hash"] == "a" * 64
    yes = next_acquisition(dict(passed=True), dict(passed=True), **kwargs)
    assert yes["advance"] and yes["guided_rho"] == 0.12
    assert yes["actor_hash"] == yes["optimizer_owner_hash"] == "b" * 64
    assert yes["independent_rho"] == 0 and not yes["independent_producer_policy_access"]
    assert yes["second_round_required_for_loop_claim"]


def test_incomplete_or_nonfinite_outcomes_are_rejected():
    with pytest.raises(ValueError):
        child_comparison(score(150), score(140), score(float("nan")))
    broken = score(100)
    broken["costs"] = [100]
    with pytest.raises(ValueError):
        teacher_eligibility(score(100), broken)
