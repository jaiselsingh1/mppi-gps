"""Final L3 screens on synthetic saved files; no inference, fitting or physics."""

import json
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps.local_coupling_multiround_metrics import summarize
from src.gps.mppi_gps.provenance import sha, write


def binding(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def saved_json(path, value):
    write(path, value)
    return binding(path)


def panel(path, arm, kind, panel_name, *, walking=0, survival=0, seed_base=0):
    result = []
    for i in range(16):
        complete = i < max(walking, survival)
        value = dict(
            status="COMPLETED",
            reset_seed=seed_base + i,
            outcome=dict(
                walking=i < walking,
                full_survival=complete,
                controls=1000 if complete else 50,
                complete_cost=1000.0 if complete else 4800.0,
            ),
        )
        result.append(
            dict(
                arm=arm,
                kind=kind,
                panel=panel_name,
                reset_seed=seed_base + i,
                result=saved_json(path / f"{arm}_{kind}_{panel_name}_{i}.json", value),
            )
        )
    return result


def teacher_trace(path, action):
    states = np.zeros((301, 19), dtype=np.float64)
    states[:, 0] = np.arange(301) * 0.008
    states[:, 2], states[:, 10] = 1.2, 1.5
    with h5py.File(path, "x") as raw:
        raw.create_dataset("states", data=states, compression="gzip")
        raw.create_dataset("warmstarts", data=np.zeros((301, 9)), compression="gzip")
        raw.create_dataset("integrations", data=np.zeros((301, 91)), compression="gzip")
        raw.create_dataset("actions", data=np.full((300, 6), action), compression="gzip")
    return binding(path)


def probe(path, value, observed=(0, 50, 100)):
    return saved_json(
        path,
        dict(
            status="COMPLETED",
            anchors=[
                dict(
                    control=control,
                    status="OBSERVED" if control in observed else "MISSING_AFTER_EXECUTED_PREFIX",
                    **({"gap_action_mse": value + control / 10000} if control in observed else {}),
                )
                for control in (0, 50, 100)
            ],
        ),
    )


@pytest.fixture(scope="module")
def complete_saved_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("l3-synthetic-metrics")
    config = dict(
        rounds=8,
        seeds=dict(panel_size=16),
        criteria=dict(
            walking_gain_min=2,
            winning_streams_min=2,
            last_round_window=4,
            teacher_gap_reduction_min=0.2,
            fixed_anchor_gap_reduction_min=0.2,
            student_common_anchor_gap_reduction_min=0.2,
            feedback_effect_action_rms_min=0.01,
        ),
    )
    streams = []
    for stream_id in range(3):
        directory = root / f"s{stream_id}"
        directory.mkdir(parents=True)
        stream = dict(
            stream=stream_id,
            status="COMPLETED_ALL_ROUNDS",
            initial_evaluations=[],
            initial_probes=[],
            rounds=[],
            final_evaluations=[],
        )
        for arm in ("independent", "positive"):
            stream["initial_evaluations"].extend(panel(directory, arm, "initial", "primary"))
            stream["initial_probes"].append(
                dict(arm=arm, result=probe(directory / f"r0_{arm}_probe.json", 0.4))
            )
            for kind in ("initial", "final"):
                stream["final_evaluations"].extend(panel(directory, arm, kind, "confirmation"))
        for r in range(1, 9):
            block_dir = directory / f"r{r}"
            block_dir.mkdir()
            block = dict(
                round=r,
                status="COMPLETED",
                fits=[],
                acquisitions=[],
                admissions=[],
                evaluations=[],
                probes=[],
            )
            for arm in ("independent", "positive"):
                block["evaluations"].extend(panel(block_dir, arm, "child", "primary"))
                fit = dict(status="COMPLETED")
                for stage in ("before", "after"):
                    metric = 0.2 if stage == "before" else 0.2 - r * 0.01
                    fit["diagnostics_" + stage] = dict(
                        own=dict(action_mse=metric),
                        anchor=dict(action_mse=metric),
                        held=dict(action_mse=0.2 if stage == "before" else 0.2 + r * 0.01),
                    )
                block["fits"].append(
                    dict(arm=arm, result=saved_json(block_dir / f"{arm}_fit.json", fit))
                )
                block["probes"].append(
                    dict(arm=arm, result=probe(block_dir / f"{arm}_probe.json", 0.4 - r * 0.02))
                )
                trace = teacher_trace(
                    block_dir / f"{arm}.h5", 0.0 if arm == "independent" else 0.02 * r
                )
                for slot in range(8):
                    manifest = saved_json(
                        block_dir / f"{arm}_slot{slot}.json",
                        dict(
                            status="COMPLETED",
                            actor_hash=f"s{stream_id}-{arm}-r{r}",
                            seed=2961000000 + 100 * stream_id + slot,
                            trace=trace,
                        ),
                    )
                    block["acquisitions"].append(dict(arm=arm, slot=slot, manifest=manifest))
                    changed = arm == "positive"
                    block["admissions"].append(
                        dict(
                            arm=arm,
                            slot=slot,
                            admitted=True,
                            fallback=False,
                            new_experience=changed,
                            new_rows=300 if changed else 0,
                            age=0,
                        )
                    )
            stream["rounds"].append(block)
        streams.append(stream)
    return config, streams


def test_complete_negative_trajectory_is_scientifically_complete(complete_saved_fixture):
    config, streams = complete_saved_fixture
    result = summarize(config, streams)
    assert result["scientific_complete"] and result["missing"] == []
    assert not result["walking"]["passed"]
    assert all(row["difference_in_gain"] == 0 for row in result["walking"]["details"])
    assert result["feedback"]["active"]
    assert result["feedback"]["active_streams"] == [0, 1, 2]
    assert not result["convergence_claim"] and not result["superiority_claim"]
    for row in result["gaps"]["details"]:
        assert row["fixed_anchor"]["fractional_reduction"] == pytest.approx(0.4)
        assert row["fixed_anchor"]["last_four_postfit_slope"]["slope"] == pytest.approx(-0.01)
        assert row["held"]["fractional_reduction"] < 0


def replace_endpoint_panels(
    tmp_path, streams, *, positive_initial=0, positive_final=4, zero_final=1
):
    result = deepcopy(streams)
    for stream in result:
        directory = tmp_path / f"s{stream['stream']}"
        directory.mkdir(parents=True)
        for arm in ("independent", "positive"):
            initial = positive_initial if arm == "positive" else 0
            final = positive_final if arm == "positive" else zero_final
            stream["initial_evaluations"] = [
                r for r in stream["initial_evaluations"] if r["arm"] != arm
            ]
            stream["initial_evaluations"].extend(
                panel(directory, arm, "initial", "primary", walking=initial)
            )
            stream["rounds"][-1]["evaluations"] = [
                r for r in stream["rounds"][-1]["evaluations"] if r["arm"] != arm
            ]
            stream["rounds"][-1]["evaluations"].extend(
                panel(directory, arm, "child", "primary", walking=final)
            )
            stream["final_evaluations"] = [
                r for r in stream["final_evaluations"] if r["arm"] != arm
            ]
            for kind, count in (("initial", initial), ("final", final)):
                stream["final_evaluations"].extend(
                    panel(directory, arm, kind, "confirmation", walking=count)
                )
    return result


def test_final_walking_requires_both_panels_same_streams_and_did(tmp_path, complete_saved_fixture):
    config, streams = complete_saved_fixture
    changed = replace_endpoint_panels(tmp_path / "gain", streams)
    result = summarize(config, changed)
    assert result["walking"]["passed"]
    assert result["walking"]["winning_streams"] == [0, 1, 2]
    assert all(row["difference_in_gain"] == 3 for row in result["walking"]["details"])
    tied = replace_endpoint_panels(
        tmp_path / "did-zero", streams, positive_initial=3, positive_final=5, zero_final=2
    )
    result = summarize(config, tied)
    assert not result["walking"]["passed"]
    assert all(row["walking_gain_positive"] == 2 for row in result["walking"]["details"])
    assert all(row["difference_in_gain"] == 0 for row in result["walking"]["details"])


def test_survival_only_is_not_walking(tmp_path, complete_saved_fixture):
    config, streams = complete_saved_fixture
    streams = deepcopy(streams)
    folder = tmp_path / "survival"
    folder.mkdir()
    streams[0]["rounds"][-1]["evaluations"] = [
        row for row in streams[0]["rounds"][-1]["evaluations"] if row["arm"] != "positive"
    ]
    streams[0]["rounds"][-1]["evaluations"].extend(
        panel(folder, "positive", "child", "primary", survival=4)
    )
    result = summarize(config, streams)
    row = next(
        row
        for row in result["walking"]["details"]
        if row["stream"] == 0 and row["panel"] == "primary"
    )
    assert row["positive"]["full_survival"] == 4 and row["positive"]["walking"] == 0
    assert not result["walking"]["passed"]


def test_all_fallback_still_completes_without_fabricating_coupling(complete_saved_fixture):
    config, streams = complete_saved_fixture
    streams = deepcopy(streams)
    for stream in streams:
        for block in stream["rounds"]:
            for row in block["admissions"]:
                if row["arm"] == "positive":
                    row.update(
                        admitted=False,
                        fallback=True,
                        new_experience=False,
                        new_rows=0,
                        age=block["round"],
                    )
    result = summarize(config, streams)
    assert result["scientific_complete"]
    assert not result["feedback"]["active"]
    assert result["feedback"]["minimum_new_fraction_required"] is False
    positive = [row for row in result["feedback"]["activity"] if row["arm"] == "positive"]
    assert all(row["fallback_fraction"] == 1 for row in positive)
    assert all(row["new_fraction"] == 0 for row in positive)
    assert any(
        row["effective_attempt"] for row in result["feedback"]["events"] if row["arm"] == "positive"
    )


def test_missing_postfall_anchors_remain_missing_and_do_not_imply_contraction(
    tmp_path, complete_saved_fixture
):
    config, streams = complete_saved_fixture
    streams = deepcopy(streams)
    streams[0]["rounds"][-1]["probes"][1]["result"] = probe(
        tmp_path / "short-prefix.json", 0.1, observed=(0,)
    )
    result = summarize(config, streams)
    assert result["scientific_complete"]
    cell = next(
        row for row in result["gaps"]["details"] if row["stream"] == 0 and row["arm"] == "positive"
    )["student_common_anchor"]
    assert cell["initial_observed"] == 3 and cell["final_observed"] == 1
    assert cell["common_controls"] == [0]
    assert cell["fractional_reduction"] == pytest.approx(0.75)
    assert not cell["contraction_with_preserved_coverage"]
    assert cell["final"] == 0.1


def test_absent_panel_is_unknown_not_zero_success(complete_saved_fixture):
    config, streams = complete_saved_fixture
    streams = deepcopy(streams)
    streams[0]["rounds"][-1]["evaluations"].pop()
    result = summarize(config, streams)
    assert not result["scientific_complete"] and result["missing"]
    cell = next(
        row
        for row in result["walking"]["details"]
        if row["stream"] == 0 and row["panel"] == "primary"
    )
    assert cell["positive"]["episodes"] == 15
    assert cell["positive"]["walking"] is None
    assert cell["difference_in_gain"] is None


def test_corrupted_bound_measurement_is_rejected(tmp_path, complete_saved_fixture):
    config, streams = complete_saved_fixture
    streams = deepcopy(streams)
    spec = saved_json(tmp_path / "bad.json", dict(status="COMPLETED"))
    Path(spec["path"]).write_text("corrupted saved bytes")
    streams[0]["rounds"][0]["fits"][0]["result"] = spec
    with pytest.raises(ValueError, match="changed"):
        summarize(config, streams)


def test_missing_round_cannot_be_certified_complete(complete_saved_fixture):
    config, streams = complete_saved_fixture
    streams = deepcopy(streams)
    streams[1]["rounds"].pop()
    result = summarize(config, streams)
    assert not result["scientific_complete"]
    assert not result["walking"]["passed"]
    assert any(row["role"] == "complete trajectory" for row in result["missing"])


class PureAuditAssertions:
    """Only arithmetic assertions; no audit inference or physics pathways."""

    def need(self, value, message):
        assert value, message

    def equal(self, actual, expected, message):
        np.testing.assert_equal(actual, expected, err_msg=message)

    def close(self, actual, expected, message, atol=0):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, err_msg=message)


def independent_criteria_fixture(config, streams):
    """Adapt synthetic saved evidence to the auditor's independently measured view.

    This invokes only the auditor's scalar panel summary, feedback arithmetic
    and final algebra. It never invokes an audit checkpoint, predictor, physics
    verifier or model. The actual study auditor must still verify all evidence.
    """
    from src.gps.mppi_gps.audits import local_coupling_multiround as auditor

    def load(spec):
        return json.loads(Path(spec["path"]).read_text())

    def summary(rows):
        return auditor.summarize_panel([load(row["result"])["outcome"] for row in rows])

    def probes(rows):
        return {
            row["arm"]: dict(
                observed=[
                    point
                    for point in load(row["result"])["anchors"]
                    if point["status"] == "OBSERVED"
                ]
            )
            for row in rows
        }

    result = []
    for stream in sorted(streams, key=lambda row: row["stream"]):
        stream_id, rounds, previous = stream["stream"], [], None
        for block in stream["rounds"]:
            current = {}
            for row in block["acquisitions"]:
                manifest = load(row["manifest"])
                with h5py.File(manifest["trace"]["path"], "r") as raw:
                    values = {
                        key: raw[key][:]
                        for key in ("actions", "states", "warmstarts", "integrations")
                    }
                current[(row["arm"], row["slot"])] = dict(
                    actor_hash=manifest["actor_hash"], manifest=manifest, raw=values
                )
            diagnostics = {}
            for fit in block["fits"]:
                value = load(fit["result"])
                diagnostics[fit["arm"]] = {
                    stage: value["diagnostics_" + stage] for stage in ("before", "after")
                }
            feedback = (
                auditor._feedback_events(
                    PureAuditAssertions(),
                    block,
                    stream_id,
                    current,
                    previous,
                    block["admissions"],
                    config,
                )
                if previous is not None
                else []
            )
            rounds.append(
                dict(
                    round=block["round"],
                    diagnostics=diagnostics,
                    probes=probes(block["probes"]),
                    activity=block["admissions"],
                    feedback=feedback,
                    panels={
                        arm: summary([row for row in block["evaluations"] if row["arm"] == arm])
                        for arm in ("independent", "positive")
                    },
                )
            )
            previous = current
        panels = {}
        for panel_name in ("primary", "confirmation"):
            cells = {}
            for arm in ("independent", "positive"):
                for kind in ("initial", "final"):
                    rows = (
                        stream["initial_evaluations"]
                        if panel_name == "primary" and kind == "initial"
                        else stream["rounds"][-1]["evaluations"]
                        if panel_name == "primary"
                        else stream["final_evaluations"]
                    )
                    cells[(arm, kind)] = summary(
                        [
                            row
                            for row in rows
                            if row["arm"] == arm
                            and (panel_name == "primary" or row["kind"] == kind)
                        ]
                    )
            panels[(stream_id, panel_name)] = cells
        result.append(
            dict(
                stream=stream_id,
                panels=panels,
                rounds=rounds,
                initial_probes=probes(stream["initial_probes"]),
            )
        )
    return auditor._criteria(result, config)


def test_metrics_match_independent_final_algebra_on_full_saved_fixture(
    tmp_path, complete_saved_fixture
):
    from src.gps.mppi_gps.audits import local_coupling_multiround as auditor

    config, original = complete_saved_fixture
    positive = replace_endpoint_panels(tmp_path / "positive", original)
    missing_anchor = deepcopy(original)
    missing_anchor[0]["rounds"][-1]["probes"][1]["result"] = probe(
        tmp_path / "short-probe.json", 0.1, observed=(0,)
    )
    for streams in (original, positive, missing_anchor):
        actual = summarize(config, streams)
        expected = independent_criteria_fixture(config, streams)
        auditor._compare_subset(
            PureAuditAssertions(), actual, expected, "Independent final-measurement algebra"
        )
