"""V audit arithmetic, pairing, storage and no-simulator boundaries."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits.student_visitation import (
    AuditError,
    Reader,
    WorkChecks,
    compare_replay,
    positive_control_screen,
    reconcile_costs,
    reference_screen,
    trace_outcome,
    verify_outcome_roster,
)


def arrays(n=3, *, done=True):
    result = {
        "actions": np.zeros((n, 6), dtype="<f8"),
        "states": np.zeros((n + 1, 19), dtype="<f8"),
        "observations": np.zeros((n + 1, 17), dtype="<f8"),
        "warmstarts": np.zeros((n + 1, 9), dtype="<f8"),
        "integrations": np.zeros((n + 1, 91), dtype="<f8"),
        "native_costs": np.ones(n, dtype="<f8"),
        "x_velocities": np.full(n, 1.5, dtype="<f8"),
        "fall_costs": np.zeros(n, dtype="<f8"),
        "done": np.zeros(n, dtype="u1"),
    }
    if n and done:
        result["done"][-1] = 1
        result["fall_costs"][-1] = 20
        result["native_costs"][-1] += 20
    return result


def screen_rows():
    rows = []
    for panel in ("primary", "confirmation"):
        for stream in range(3):
            for case in range(4):
                for ordinal in range(4):
                    for kind in ("parent_replay", "reference_takeover"):
                        rows.append(
                            dict(
                                kind=kind,
                                stream=stream,
                                panel=panel,
                                source_tag=f"source_s{stream}_{panel}_{case}",
                                source_case=case,
                                anchor_ordinal=ordinal,
                                control=ordinal,
                                walking=False,
                                survived=False,
                                actor=f"s{stream}_parent"
                                if kind == "parent_replay"
                                else "reference",
                                seed=None,
                            )
                        )
    return rows


def mark_gain(rows, panel, streams):
    for row in rows:
        if (
            row["panel"] == panel
            and row["stream"] in streams
            and row["kind"] == "reference_takeover"
            and row["anchor_ordinal"] == 1
            and row["source_case"] in (0, 1)
        ):
            row.update(walking=True, survived=True)


def test_full_window_walking_and_single_terminal_tail_are_separate():
    fallen = trace_outcome(arrays())
    assert fallen == dict(
        controls=3,
        done=True,
        survived=False,
        walking=False,
        mae=0.0,
        native_cost=23.0,
        native_terminal=20.0,
        raw_cost=3.0,
        absorbing_cost=4985.0,
        cost=5008.0,
    )
    assert trace_outcome(arrays(1000, done=False))["walking"]
    slow = arrays(1000, done=False)
    slow["x_velocities"][:] = 0
    outcome = trace_outcome(slow)
    assert outcome["survived"] and not outcome["walking"]


def test_missing_prefix_never_becomes_walk_or_fabricated_fall():
    outcome = trace_outcome(arrays(0, done=False))
    assert outcome["mae"] is None and not outcome["walking"]
    assert outcome["absorbing_cost"] == 0


@pytest.mark.parametrize("fault", ["postfall", "double_terminal", "healthy_terminal", "nonfinite"])
def test_invalid_prefix_costs_are_rejected(fault):
    data = arrays()
    if fault == "postfall":
        data["done"][0] = 1
    elif fault == "double_terminal":
        data["fall_costs"][0] = 20
    elif fault == "healthy_terminal":
        data["fall_costs"][:] = 0
        data["fall_costs"][0] = 20
    else:
        data["x_velocities"][0] = np.nan
    with pytest.raises(AuditError):
        trace_outcome(data)


def test_parent_suffix_replay_keeps_the_recorded_fall_and_terminal_boundary():
    source = arrays(7)
    actual = {k: v[3:].copy() for k, v in source.items()}
    assert all(value == 0 for value in compare_replay(actual, source, 3).values())
    actual["states"][-1, 2] = 2e-10
    with pytest.raises(AuditError, match="replay mismatch"):
        compare_replay(actual, source, 3)


def test_initial_restore_bytes_are_stricter_than_later_replay_tolerance():
    source = arrays(7)
    actual = {k: v[3:].copy() for k, v in source.items()}
    actual["warmstarts"][0, 0] = 1e-12
    with pytest.raises(AuditError, match="byte exact"):
        compare_replay(actual, source, 3)


def test_initial_only_success_cannot_pass_reference_screen():
    rows = screen_rows()
    for row in rows:
        if row["kind"] == "reference_takeover" and row["anchor_ordinal"] == 0:
            row.update(walking=True, survived=True)
    assert not reference_screen(rows)["passed"]


def test_same_two_streams_must_gain_on_both_panels():
    rows = screen_rows()
    mark_gain(rows, "primary", (0, 1))
    mark_gain(rows, "confirmation", (1, 2))
    assert not reference_screen(rows)["passed"]
    mark_gain(rows, "confirmation", (0,))
    assert reference_screen(rows)["passed"]


def test_any_noninitial_survival_regression_blocks_a_walking_gain():
    rows = screen_rows()
    for panel in ("primary", "confirmation"):
        mark_gain(rows, panel, (0, 1))
    rows[0].update(survived=True)  # t0 is secondary and cannot block admission.
    assert reference_screen(rows)["passed"]
    next(
        row
        for row in rows
        if row["kind"] == "parent_replay" and row["stream"] == 2 and row["anchor_ordinal"] == 1
    ).update(survived=True)
    assert not reference_screen(rows)["passed"]


@pytest.mark.parametrize("fault", ["duplicate", "missing", "zero_noninitial"])
def test_incomplete_or_inconsistent_anchor_rosters_fail(fault):
    rows = screen_rows()
    if fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "missing":
        rows.pop()
    else:
        rows[2]["control"] = 0
    with pytest.raises(AuditError):
        reference_screen(rows)


def test_positive_control_requires_each_complete_disjoint_panel():
    rows = [
        dict(
            kind="reference_reset",
            seed=i,
            panel="primary" if i < 16 else "confirmation",
            walking=i not in (0, 16),
            survived=True,
        )
        for i in range(32)
    ]
    assert positive_control_screen(rows)["passed"]
    rows[1]["walking"] = False
    assert not positive_control_screen(rows)["passed"]
    rows[2]["seed"] = rows[3]["seed"]
    with pytest.raises(AuditError, match="seed repeated"):
        positive_control_screen(rows)


def test_rows_bind_actor_and_all_source_identifiers_before_forward_replay():
    rows = screen_rows()
    anchors = [
        dict(
            stream=r["stream"],
            panel=r["panel"],
            source_tag=r["source_tag"],
            source_case=r["source_case"],
            anchor_ordinal=r["anchor_ordinal"],
            control=r["control"],
        )
        for r in rows
        if r["kind"] == "parent_replay"
    ]
    config = dict(
        validity=dict(anchors=anchors, positive_control_reference_reset_seeds=list(range(32)))
    )
    verify_outcome_roster(rows, config, dict(anchors=anchors))
    rows[1]["actor"] = "s0_parent"
    with pytest.raises(AuditError, match="controller"):
        verify_outcome_roster(rows, config, dict(anchors=anchors))


def test_request_return_journal_rejects_missing_or_unowned_completions():
    work = WorkChecks()
    work.event(dict(event="runtime_open"))
    work.event(dict(event="work_requested", operation=1, category="neural_rows", amount=1))
    with pytest.raises(AuditError, match="lifecycle"):
        work.verify_complete()
    work.event(dict(event="runtime_close"))
    with pytest.raises(AuditError, match="uncertain"):
        work.verify_complete()
    work.event(dict(event="work_complete", operation=1, category="neural_rows", amount=1))
    work.verify_complete()
    assert work.requested["prediction_calls"] == work.completed["prediction_calls"] == 1
    with pytest.raises(AuditError, match="lacks request"):
        work.event(dict(event="work_complete", operation=1, category="neural_rows", amount=1))


def test_lifecycle_line_exception_is_not_extended_to_frequent_events():
    WorkChecks().event(dict(event="runtime_open", text="x" * 1000))
    with pytest.raises(AuditError, match="line bound"):
        WorkChecks().event(dict(event="other", text="x" * 1000))
    with pytest.raises(AuditError, match="batch-one"):
        WorkChecks().event(
            dict(event="work_requested", operation=1, category="neural_rows", amount=2)
        )


def test_reader_budget_precedes_reads_and_paths_cannot_escape(tmp_path):
    (tmp_path / "file.json").write_text("{}")
    reader = Reader(max_bytes=1)
    with pytest.raises(AuditError, match="reservation"):
        reader.read(tmp_path, "file.json")
    assert reader.read_bytes == 0
    with pytest.raises(AuditError, match="Escaped"):
        Reader().read(tmp_path, "../elsewhere")
    (tmp_path / "link.json").symlink_to(tmp_path / "file.json")
    with pytest.raises(AuditError, match="Symlink"):
        Reader().read(tmp_path, "link.json")


def test_selected_original_raw_ranges_are_bounded_and_hashed(tmp_path):
    values = np.arange(6, dtype="<f8").reshape(2, 3)
    payload = values.tobytes()
    (tmp_path / "raw").write_bytes(b"prefix" + payload)
    binding = dict(
        offset=6,
        byte_length=len(payload),
        shape=[2, 3],
        dtype="<f8",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert np.array_equal(Reader().selected_array(tmp_path, "raw", binding), values)
    binding["sha256"] = "0" * 64
    with pytest.raises(AuditError, match="original C1"):
        Reader().selected_array(tmp_path, "raw", binding)


def test_helper_census_and_missing_costs_cannot_be_hidden_by_good_outcomes():
    rows = [dict(kind="parent_replay", controls=2), dict(kind="reference_reset", controls=3)]
    expected = dict(
        native_controls=5,
        physics_substeps=20,
        neural_rows=5,
        prediction_calls=5,
        restores=1,
        resets=1,
        external_mj_forward_calls=3,
        snapshot_reads=12,
        checkpoint_loads=4,
        policy_constructions=4,
        environment_constructions=1,
        backward_rows=0,
        optimizer_steps=0,
    )
    worker = dict(
        costs=dict(
            requested=expected.copy(),
            completed=expected.copy(),
            unknown_partial={},
            returned_unpersisted={},
        )
    )
    work = WorkChecks()
    work.requested = Counter(expected)
    work.completed = Counter(expected)
    stream = dict(saved_controls=5, unknown_partial_controls=0)
    assert reconcile_costs(worker, rows, stream, work)["independently_derived"] == expected
    worker["costs"]["completed"]["snapshot_reads"] -= 1
    with pytest.raises(AuditError, match="snapshot_reads"):
        reconcile_costs(worker, rows, stream, work)


def test_audit_import_does_not_import_simulator_or_worker():
    source = """import builtins
old=builtins.__import__
def guarded(name,*a,**k):
 if any(part in name for part in ('mujoco','gymnasium','visitation_worker','visitation_native')):
  raise AssertionError(name)
 return old(name,*a,**k)
builtins.__import__=guarded
from src.gps.mppi_gps.audits import student_visitation
assert student_visitation.MAX_FORWARDS == 224000
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_npz_shape_claim_is_checked_before_numpy_can_allocate(tmp_path):
    import io
    import zipfile

    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header, dict(descr="<f8", fortran_order=False, shape=(100_000_000,))
    )
    path = tmp_path / "oversized.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("values.npy", header.getvalue())
    binding = dict(
        file=path.name,
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(AuditError, match="declared array"):
        Reader().npz(tmp_path, binding)


def test_independent_functional_replicas_match_both_real_policy_interfaces(tmp_path):
    import torch

    from src.gps.mppi_gps.audits.student_visitation import ActorVerifier
    from src.gps.mppi_gps.audits.teacher_proposal import AuditJournal
    from src.gps.mppi_gps.diagnostic_policies import LegacySACActor
    from src.gps.mppi_gps.policy import Policy

    torch.set_num_threads(1)
    actors = {f"s{stream}_parent": Policy(seed=stream, device="cpu") for stream in range(3)}
    actors["reference"] = LegacySACActor()
    registry = {}
    for alias, actor in actors.items():
        path = tmp_path / (alias + ".pt")
        if alias == "reference":
            torch.save(dict(policy=dict(actor=actor.state_dict())), path)
        else:
            actor.checkpoint(path)
        registry[alias] = dict(
            file=path.name,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            actor_hash=actor.hash(),
        )
    journal = AuditJournal(tmp_path / "audit_journal.jsonl", max_bytes=65536, max_line_bytes=512)
    try:
        verifier = ActorVerifier(Reader(), tmp_path, registry, journal)
        verifier.load_all()
        observation = np.linspace(-3, 3, 17, dtype=np.float64)
        for alias, actor in actors.items():
            action = actor.predict(observation).astype(np.float64)
            verifier.verify(alias, observation, action)
        assert verifier.loads == len(verifier.actors) == 4
        assert verifier.requested == verifier.completed == verifier.calls == 4
        action = actors["reference"].predict(observation).astype(np.float64)
        action[0] = np.nextafter(action[0], 1.0)
        with pytest.raises(AuditError, match="action differs"):
            verifier.verify("reference", observation, action)
        assert verifier.requested == verifier.completed == 5 and verifier.unknown == 0
        verifier.requested = 224000
        with pytest.raises(AuditError, match="reservation"):
            verifier.verify("reference", observation, action)
        assert verifier.calls == 5
    finally:
        journal.close()


def test_unpaired_source_names_cannot_pass_even_with_matching_counts():
    rows = screen_rows()
    for row in rows:
        if row["kind"] == "reference_takeover":
            row["source_tag"] += "_other"
    with pytest.raises(AuditError, match="not paired"):
        reference_screen(rows)


def test_parent_replay_actions_remain_exact_below_state_tolerance():
    source = arrays()
    actual = {key: value.copy() for key, value in source.items()}
    actual["actions"][0, 0] = float(np.float32(1e-12))
    assert np.allclose(actual["actions"], source["actions"], rtol=0, atol=1e-10)
    with pytest.raises(AuditError, match="action bytes"):
        compare_replay(actual, source, 0)


def test_complete_negative_validity_lifecycle_uses_real_binary_streams(tmp_path, monkeypatch):
    """128 tiny actual-prefix archives; synthetic callbacks pay no NN or physics.

    The separate functional-replica test covers real tensor replay. This fixture
    exercises the complete file, source, roster, gate and accounting protocol.
    Its four-control source corpus/config is explicitly synthetic, not a change
    to the production protocol's 11,426-control retained-source census.
    """
    import json

    from test_visitation_inputs import source_fixture

    from src.gps.mppi_gps import visitation_inputs
    from src.gps.mppi_gps.audits import student_visitation as audit
    from src.gps.mppi_gps.consequence_storage import C1TraceRecorder
    from src.gps.mppi_gps.visitation_worker import VisitationMeter

    # This fixture models a separate lightweight preparation process; the
    # shared pytest process may retain unrelated lifetime inference peaks.
    monkeypatch.setattr("src.gps.mppi_gps.consequence_inputs._peak_rss_bytes", lambda: 64 * 1024**2)
    old = source_fixture(tmp_path / "synthetic_source", physical=True)
    run = tmp_path / "synthetic_V"
    visitation_inputs.prepare(old["config"], run)
    catalog = visitation_inputs.verify_prepared(run / "prepared")
    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/studies/student_visitation_s1_2_v_1.json"
        ).read_text()
    )
    review = json.loads(
        Path(old["config"]["input_preparation"]["review_recipe"]["path"]).read_text()
    )
    config.update(
        study="SYNTHETIC_NO_NN_NO_PHYSICS",
        input_preparation=old["config"]["input_preparation"],
        parents=review["parents"],
        reference=review["reference"],
        normalization=review["normalization"],
        validity=review["validity"],
        expected_parent_replay_controls=240,
    )
    config["xml_sha256"] = old["config"]["input_preparation"]["xml"]["sha256"]
    config["task"] = json.loads((run / "prepared" / catalog["task"]["file"]).read_text())["task"]
    (run / "config.json").write_text(json.dumps(config, sort_keys=True))
    worker = run / "worker"
    worker.mkdir()
    registry = {}
    for alias in ("s0_parent", "s1_parent", "s2_parent", "reference"):
        record = catalog["reference"] if alias == "reference" else catalog["parents"][alias[1]]
        registry[alias] = dict(
            file="prepared/" + record["file"],
            bytes=record["bytes"],
            sha256=record["sha256"],
            actor_hash=record["actor_hash"],
        )
    (worker / "actor_registry.json").write_text(json.dumps(registry))

    class SyntheticRecordedActionVerifier:
        def __init__(self, *_):
            self.requested = self.completed = self.calls = self.loads = self.unknown = 0
            self.actors = {}

        def load_all(self):
            self.loads = 4
            self.actors = dict.fromkeys(registry)

        def verify(self, alias, observation, action):
            assert alias in registry and observation.shape == (17,)
            assert np.array_equal(action, np.full(6, 0.125))
            self.requested += 1
            self.completed += 1
            self.calls += 1

    monkeypatch.setattr(audit, "ActorVerifier", SyntheticRecordedActionVerifier)
    meter = VisitationMeter(
        worker / "evidence", wall_seconds=120, max_bytes=1024**3, rss_bytes=4 * 1024**3
    )
    recorder = C1TraceRecorder(meter, worker / "active")
    for _ in range(4):
        meter.measure("checkpoint_loads", 1, lambda: None, purpose="synthetic_opaque_load")
        meter.measure("policy_constructions", 1, lambda: None, purpose="synthetic_no_model")
    meter.reserve_work("environment_constructions")
    meter.record("environment_requested", mjdata_objects=17)
    meter.returned("environment_constructions", 1)
    meter.record("environment_complete", integration_size=91, mjdata_objects=17)
    rows = []

    def load(descriptor):
        with np.load(run / "prepared" / descriptor["file"], allow_pickle=False) as data:
            return {key: data[key].copy() for key in data.files}

    def branch(tag, kind, actor, source, t=0, seed=None, anchor=None, panel=None, case=None):
        reset = kind == "reference_reset"
        operation = "resets" if reset else "restores"
        meter.measure(operation, 1, lambda: None, purpose="synthetic_saved_context")
        meter.reserve_work("external_mj_forward_calls", 2 if reset else 1)
        meter.returned("external_mj_forward_calls", 2 if reset else 1)
        meter.reserve_work("snapshot_reads")
        meter.returned("snapshot_reads", 1)

        def snapshot(index):
            return dict(
                state=source["states"][index],
                warmstart=source["warmstarts"][index],
                integration=source["integrations"][index],
            )

        recorder.begin(tag, 1000, snapshot(t), source["observations"][t])
        meter.after_prediction = lambda value: recorder.prediction(
            value, actor_hash=registry[actor]["sha256"]
        )
        for index in range(t, 4):
            action = meter.measure(
                "neural_rows",
                1,
                lambda: np.full(6, 0.125, dtype=np.float32),
                purpose="synthetic_recorded_action",
            )
            meter.reserve_work("snapshot_reads", 2)
            meter.returned("snapshot_reads", 2)
            work = dict(native_controls=1, physics_substeps=4)
            meter.authorize(work)
            recorder.intent(
                action, snapshot=snapshot(index), observation=source["observations"][index]
            )
            meter.charge(work)
            meter.returned("native_controls", 1)
            meter.returned("physics_substeps", 4)
            recorder.completed(
                snapshot(index + 1),
                source["observations"][index + 1],
                cost=float(source["native_costs"][index]),
                velocity=float(source["x_velocities"][index]),
                fall=float(source["fall_costs"][index]),
                done=bool(source["done"][index]),
            )
        meter.after_prediction = None
        metadata = dict(
            kind=kind,
            actor=actor,
            source=None if reset else anchor["source_tag"],
            t=t,
            seed=seed,
            integration_spec=16383,
            dimensions=[9, 9, 6],
        )
        reference = recorder.close(5.0 * (1000 - (4 - t)), metadata)
        suffix = {
            key: values[t:].copy()
            for key, values in source.items()
            if key
            in (*audit.BOUNDARIES, "actions", "native_costs", "x_velocities", "fall_costs", "done")
        }
        measured = trace_outcome(suffix)
        row = dict(
            tag=tag,
            stage=kind,
            actor=actor,
            actor_hash=registry[actor]["actor_hash"],
            checkpoint_sha256=registry[actor]["sha256"],
            source_tag=None if reset else anchor["source_tag"],
            t=t,
            h=1000,
            seed=seed,
            trace=reference,
            stream=None if reset else anchor["stream"],
            panel=panel if reset else anchor["panel"],
            source_case=case if reset else anchor["source_case"],
            anchor_ordinal=None if reset else anchor["anchor_ordinal"],
            **measured,
        )
        rows.append(row)

    for ordinal, anchor in enumerate(catalog["anchors"]):
        branch(
            f"parent_{ordinal}",
            "parent_replay",
            f"s{anchor['stream']}_parent",
            load(catalog["episodes"][anchor["source_tag"]]),
            t=anchor["control"],
            anchor=anchor,
        )
    source = load(next(iter(catalog["episodes"].values())))
    for ordinal, seed in enumerate(config["validity"]["positive_control_reference_reset_seeds"]):
        branch(
            f"own_{ordinal}",
            "reference_reset",
            "reference",
            source,
            seed=seed,
            panel="primary" if ordinal < 16 else "confirmation",
            case=ordinal % 16,
        )
    meter.close()
    (worker / "outcomes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (worker / "parent_replay_gate.json").write_text(
        json.dumps(dict(passed=True, episodes=96, controls=240))
    )
    (worker / "reference_positive_control_gate.json").write_text(
        json.dumps(
            dict(
                passed=False,
                panels={
                    panel: dict(walking=0, survived=0) for panel in ("primary", "confirmation")
                },
            )
        )
    )
    worker_result = dict(
        status="STOPPED_REFERENCE_POSITIVE_CONTROL_FAILED",
        learning_launched=False,
        followup_launched=False,
        costs=meter.accounting(),
        config_sha256=hashlib.sha256((run / "config.json").read_bytes()).hexdigest(),
        prepared_manifest_sha256=hashlib.sha256(
            (run / "prepared/prepared_input_manifest.json").read_bytes()
        ).hexdigest(),
    )
    (run / "worker_result.json").write_text(json.dumps(worker_result))
    result = audit.audit_run(run, tmp_path / "synthetic_audit")
    assert result["passed"], result.get("error")
    assert result["closure_reason"] == "negative_reference_positive_controls"
    assert result["reference_screen_passed"] is None
    assert len(result["outcomes"]) == 128
    assert result["work"]["neural_rows"] == 368
    assert result["static_context_gate"]["selected_original_array_checks"] == 696
    assert result["complete_cost_accounting"]["independently_derived"]["restores"] == 96
    assert not result["automatic_learning_authorized"]
    assert not any(row["kind"] == "reference_takeover" for row in result["outcomes"])
    assert sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file()) < 32 * 1024**2
