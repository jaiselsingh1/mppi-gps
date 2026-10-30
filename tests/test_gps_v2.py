"""Pure gate checks; no teacher, fitted network or simulator is executed."""

from src.gps.mppi_gps import loop as fragment


def scores(cost, successes=2, survived=2, steps=1000.0, episodes=2):
    return dict(
        mean_cost=cost, successes=successes, survived=survived, mean_steps=steps, episodes=episodes
    )


def test_teacher_quality_guard_does_not_claim_cost_improvement():
    value = fragment._v2_teacher_gate(scores(100.0), scores(105.0), 2.0, 1.9, 1)
    assert value["eligible"]
    assert not value["strict_control_improvement"]
    assert not fragment._v2_teacher_gate(scores(100.0), scores(105.0001), 2.0, 1.9, 1)["eligible"]
    assert not fragment._v2_teacher_gate(scores(100.0), scores(99.95), 2.0, 1.9, 1)[
        "strict_control_improvement"
    ]
    assert fragment._v2_teacher_gate(scores(100.0), scores(99.89), 2.0, 1.9, 1)[
        "strict_control_improvement"
    ]


def test_actual_agreement_cannot_replace_success_or_an_executed_change():
    assert not fragment._v2_teacher_gate(scores(100.0), scores(90.0, successes=1), 2.0, 0.0, 1)[
        "eligible"
    ]
    assert not fragment._v2_teacher_gate(scores(100.0), scores(90.0), 2.0, 0.0, 0)["eligible"]
    assert not fragment._v2_teacher_gate(scores(100.0), scores(90.0), 2.0, 1.9001, 2)["eligible"]


def test_child_partial_progress_is_allowed_without_inventing_walking_success():
    parent = scores(100.0, successes=0, survived=0, steps=200.0, episodes=16)
    raw = scores(99.0, successes=0, survived=0, steps=250.0, episodes=16)
    guided = scores(98.0, successes=0, survived=0, steps=260.0, episodes=16)
    assert fragment._v2_child_gate(parent, raw, guided)
    assert not fragment._v2_child_gate(parent, raw, dict(guided, mean_steps=249.0))
    assert not fragment._v2_child_gate(parent, raw, dict(guided, mean_cost=99.0 * 0.999))


def test_child_cost_gain_does_not_compensate_for_success_or_survival_regression():
    parent = scores(100.0, successes=2, survived=3, steps=500.0, episodes=16)
    raw = scores(99.0, successes=1, survived=4, steps=600.0, episodes=16)
    assert not fragment._v2_child_gate(
        parent, raw, scores(80.0, successes=1, survived=4, steps=700.0, episodes=16)
    )
    assert not fragment._v2_child_gate(
        parent, raw, scores(80.0, successes=2, survived=3, steps=700.0, episodes=16)
    )


def test_weak_student_enters_both_feedback_rounds_without_standalone_gate():
    """Protocol doubles exercise admission/ownership; no dynamics are simulated."""
    from types import SimpleNamespace

    import numpy as np

    class Actor:
        def __init__(self, name):
            self.name = name

        def hash(self):
            return self.name

    class Meter:
        def record(self, *args, **kwargs):
            return "unit-event"

        def save_arrays(self, *args, **kwargs):
            return dict(unit_only=True)

    class Authority:
        def revoke(self, reason):
            pass

    calls = []
    fits = []

    def teacher(seed, planner_seed, tag, actor, rho, authority=None, provisional=False):
        calls.append((actor.hash(), rho, seed, planner_seed))
        trace = SimpleNamespace(
            success=True,
            survived=True,
            cost=102.0 if rho else 100.0,
            steps=1000,
            complete=True,
            tag=tag,
            mae=0.1,
            observations=np.zeros((1000, 17)),
            actions=np.zeros((1000, 6)),
        )
        return trace, int(rho > 0), 1.8 if rho else 2.0

    def fit(parent, state, data, tag, seed):
        fits.append((parent.hash(), seed, len(data[0])))
        return Actor(tag), dict(owner=tag)

    def evaluate(actor, root, count, tag):
        if actor.name == "initial":
            cost, steps = 5000.0, 200.0
        elif "r0_raw" in actor.name:
            cost, steps = 4900.0, 250.0
        elif "r0_guided" in actor.name:
            cost, steps = 4800.0, 260.0
        elif "r1_raw" in actor.name:
            cost, steps = 4700.0, 300.0
        else:
            cost, steps = 4600.0, 310.0
        return [], dict(episodes=16, successes=0, survived=0, mean_cost=cost, mean_steps=steps)

    cfg = dict(
        feedback_rounds=2,
        qualification_episodes=16,
        roots=dict(trial=100000, confirmation=200000, planner=300000, children=400000),
        learning=dict(seed=123),
        coupling=dict(teacher_cost_tolerance=0.05, minimum_agreement_gain=0.05),
        gates=dict(child_cost_gain=0.001, relative_cost_gain=0.001),
    )
    result = fragment.run_rounds(
        Actor("initial"),
        None,
        (np.zeros((2, 17)), np.zeros((2, 6))),
        cfg,
        Meter(),
        [0.01, 0.1],
        teacher,
        evaluate,
        fit,
        lambda actor, radius=None: Authority(),
    )
    assert result["error"] is None
    assert result["completed"] == 2
    assert len(calls) == 18
    assert len(fits) == 4
    assert fits[0][:2] == fits[1][:2] and fits[2][:2] == fits[3][:2]
    assert all(r["partial_progress_only"] for r in result["rounds"])
    assert result["control_improvedflag"] is False
    assert result["retained_actor_hash"] == "v2_r1_guided_child"
