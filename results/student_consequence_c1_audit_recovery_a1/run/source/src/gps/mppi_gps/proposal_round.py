"""One H5.0 roster, independent of native execution and policy implementations.

This module makes ordering, negative stops and admission testable with small
synthetic operations. Only an explicitly authorized native adapter may execute
the operations; importing this module constructs no policy or simulator.
"""

from __future__ import annotations

from typing import Any, Protocol

ARMS = ("independent", "selector_only", "proposal_guided")


class RoundOperations(Protocol):
    def parent(self, stream: dict[str, Any]) -> Any: ...

    def acquire(self, parent: Any, stream: dict[str, Any], arm: str, index: int) -> Any: ...

    def fit(self, parent: Any, episodes: list[Any], stream: dict[str, Any], arm: str) -> Any: ...

    def evaluate(self, actor: Any, seed: int, tag: str) -> Any: ...

    def gap_teacher(self, parent: Any, stream: dict[str, Any], arm: str, index: int) -> Any: ...

    def paired(self, first: Any, second: Any) -> None: ...

    def record(self, event: str, **fields: Any) -> None: ...


def outcome(episodes: list[Any]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("An empty cohort has no outcome")
    return dict(
        episodes=len(episodes),
        successes=sum(bool(e.success) for e in episodes),
        survived=sum(bool(e.survived) for e in episodes),
        controls=sum(int(e.steps) for e in episodes),
        mean_cost=sum(float(e.cost) for e in episodes) / len(episodes),
        costs=[float(e.cost) for e in episodes],
        complete=all(bool(e.complete) for e in episodes),
    )


def panel_gate(panel: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Require joint walking gains and every stream's walking/survival safety."""
    rows = []
    for stream, actors in panel.items():
        child = actors["proposal_guided"]
        gains = {
            arm: child["successes"] - actors[arm]["successes"]
            for arm in ("parent", "selector_only", "independent")
        }
        safe = all(
            child[key] >= actors[arm][key]
            for arm in ("parent", "selector_only", "independent")
            for key in ("successes", "survived")
        )
        rows.append(
            dict(
                stream=stream,
                gains=gains,
                joint_walking_gain=gains["parent"] >= 2 and gains["selector_only"] >= 2,
                walking_and_full_survival_nonregression=safe,
            )
        )
    passed = (
        len(rows) == 3
        and sum(row["joint_walking_gain"] for row in rows) >= 2
        and all(row["walking_and_full_survival_nonregression"] for row in rows)
    )
    return dict(passed=passed, streams=rows, retention_only_does_not_pass=True)


def run_round(config: dict[str, Any], operations: RoundOperations) -> dict[str, Any]:
    """Complete the fixed roster; no checkpoint, reset or follow-up selection."""
    if config["arms"] != list(ARMS) or len(config["streams"]) != 3:
        raise ValueError("H5.0 requires three paired streams and all three controls")
    result: dict[str, Any] = dict(
        schema="walker-teacher-proposal-round-v1",
        round=1,
        execution_complete=False,
        loop_closed=False,
        comparison_complete=False,
        parents={},
        teacher_summaries={},
        teacher_admission={},
        evaluations={},
        paired_teacher_student_gap={},
    )
    parents, teachers, actors = {}, {}, {}
    # Three common parents are constructed once, before any child or acquisition.
    for stream in config["streams"]:
        key = str(stream["id"])
        parents[key] = operations.parent(stream)
        actors[key] = {"parent": parents[key]}
    for stream in config["streams"]:
        key = str(stream["id"])
        teachers[key] = {arm: [] for arm in ARMS}
        for index in range(config["teacher_episodes_per_stream_arm"]):
            for arm in ARMS:
                episode = operations.acquire(parents[key], stream, arm, index)
                teachers[key][arm].append(episode)
            for arm in ARMS[1:]:
                operations.paired(teachers[key][ARMS[0]][index], teachers[key][arm][index])
        summaries = {arm: outcome(teachers[key][arm]) for arm in ARMS}
        result["teacher_summaries"][key] = summaries
        all_walk = all(
            e.complete and e.success and e.survived and e.steps == config["episode_steps"]
            for episodes in teachers[key].values()
            for e in episodes
        )
        base = summaries["independent"]["mean_cost"]
        result["teacher_admission"][key] = {
            arm: dict(
                walking=all(e.success for e in teachers[key][arm]),
                cost=summaries[arm]["mean_cost"]
                <= (1 + config["teacher_admission"]["aggregate_cost_tolerance_vs_independent"])
                * base,
            )
            for arm in ARMS
        }
        operations.record("teacher_stream_closed", stream=key, all_walk=all_walk)
    complete_rows = all(
        e.complete and e.success and e.survived and e.steps == config["episode_steps"]
        for per_stream in teachers.values()
        for episodes in per_stream.values()
        for e in episodes
    )
    if complete_rows:
        # A cost-screen failure cannot prevent fitting a complete walking roster.
        for stream in config["streams"]:
            key = str(stream["id"])
            for arm in ARMS:
                actors[key][arm] = operations.fit(parents[key], teachers[key][arm], stream, arm)
    else:
        result.update(
            execution_complete=True,
            status="teacher_competence_negative_no_child_fits",
            confirmed_benefit=False,
            next_round_eligible=False,
        )
        return result
    for cohort, seeds in (
        ("primary", config["primary_seeds"]),
        ("confirmation", config["confirmation_seeds"]),
    ):
        panel = {}
        for key, per_stream in actors.items():
            panel[key] = {}
            traces: dict[str, list[Any]] = {}
            for arm, actor in per_stream.items():
                traces[arm] = [
                    operations.evaluate(actor, seed, f"{cohort}_s{key}_{arm}_{index:02d}")
                    for index, seed in enumerate(seeds)
                ]
                panel[key][arm] = outcome(traces[arm])
            for arm in list(per_stream)[1:]:
                for parent_trace, child_trace in zip(traces["parent"], traces[arm]):
                    operations.paired(parent_trace, child_trace)
        result["evaluations"][cohort] = panel
    for stream in config["streams"]:
        key = str(stream["id"])
        result["paired_teacher_student_gap"][key] = {}
        for arm in ARMS:
            teacher, child = [], []
            for index, seed in enumerate(stream["gap_seeds"]):
                teacher.append(operations.gap_teacher(parents[key], stream, arm, index))
                child.append(
                    operations.evaluate(actors[key][arm], seed, f"gap_s{key}_{arm}_{index:02d}")
                )
                operations.paired(teacher[-1], child[-1])
            result["paired_teacher_student_gap"][key][arm] = dict(
                teacher=outcome(teacher),
                student=outcome(child),
                child_minus_teacher_cost=[c.cost - t.cost for c, t in zip(child, teacher)],
                seeds=stream["gap_seeds"],
                scope="Two fresh exactly paired contexts; separate from primary panels",
            )
    primary = panel_gate(result["evaluations"]["primary"])
    confirmation = panel_gate(result["evaluations"]["confirmation"])
    acquisition_teacher_pass = all(
        all(criteria.values())
        for per_stream in result["teacher_admission"].values()
        for criteria in per_stream.values()
    )
    gap_teacher_pass = all(
        value["teacher"]["successes"] == value["teacher"]["episodes"]
        and value["teacher"]["survived"] == value["teacher"]["episodes"]
        and value["teacher"]["mean_cost"]
        <= (1 + config["teacher_admission"]["aggregate_cost_tolerance_vs_independent"])
        * per_stream["independent"]["teacher"]["mean_cost"]
        for per_stream in result["paired_teacher_student_gap"].values()
        for value in per_stream.values()
    )
    teacher_pass = acquisition_teacher_pass and gap_teacher_pass
    confirmed = primary["passed"] and confirmation["passed"] and teacher_pass
    competence = {
        key: {
            arm: all(
                result["evaluations"][cohort][key][arm]["successes"] >= 15
                for cohort in ("primary", "confirmation")
            )
            for arm in per_stream
        }
        for key, per_stream in actors.items()
    }
    result.update(
        execution_complete=True,
        comparison_complete=True,
        status="confirmed_useful_learning" if confirmed else "learning_or_teacher_gate_negative",
        primary_gate=primary,
        confirmation_gate=confirmation,
        complete_teacher_gate=teacher_pass,
        acquisition_teacher_gate=acquisition_teacher_pass,
        gap_teacher_gate=gap_teacher_pass,
        confirmed_benefit=confirmed,
        individual_competence=competence,
        next_round_eligible=False,
        second_round_review_may_be_prepared=confirmed,
    )
    return result
