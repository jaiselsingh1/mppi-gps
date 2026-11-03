"""Saved-only J1 audit oracles and deliberately corrupted lifecycle evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from src.gps.mppi_gps import similarity_metrics as metrics
from src.gps.mppi_gps.audits import similarity as audit
from src.gps.mppi_gps.audits.student_consequence import TRACE_KEYS

TASK = dict(
    cost_style="target_velocity",
    target_velocity=1.5,
    vel_cost_weight=2.0,
    vertical_velocity_cost_weight=0.5,
    angle_cost_weight=0.1,
    unhealthy_cost_weight=5.0,
    ctrl_cost_weight=0.001,
    terminal_unhealthy_weight=20.0,
)


def native_trace(controls=1000, *, fallen=False, horizon=1000):
    states = np.zeros((controls + 1, 19), dtype=np.float64)
    states[:, 0] = np.arange(controls + 1) * 0.008
    states[:, 1] = states[:, 0] * 1.5
    states[:, 2] = 1.2
    states[:, 10] = 1.5
    if fallen:
        states[-1, 2] = 0.7
    actions = np.zeros((controls, 6), dtype=np.float64)
    warm = np.zeros((controls + 1, 9), dtype=np.float64)
    integration = np.zeros((controls + 1, 91), dtype=np.float64)
    integration[:, :19], integration[:, 19:28] = states, warm
    integration[1:, 28:34] = actions
    native_costs = np.zeros(controls, dtype=np.float64)
    fall = np.zeros(controls, dtype=np.float64)
    done = np.zeros(controls, dtype="u1")
    if fallen:
        native_costs[-1], fall[-1], done[-1] = 25.0, 20.0, 1
    arrays = dict(
        observations=np.c_[states[:, 2:10], np.clip(states[:, 10:19], -10, 10)],
        states=states,
        warmstarts=warm,
        integrations=integration,
        actions=actions,
        native_costs=native_costs,
        x_velocities=states[1:, 10].copy(),
        fall_costs=fall,
        done=done,
        control_ordinals=np.arange(controls, dtype="u4"),
        row_digests=np.empty((controls, 32), dtype="u1"),
        metadata=np.array(
            json.dumps(
                dict(
                    alias="fixture",
                    horizon=horizon,
                    completed_controls=controls,
                    availability="closed_actual_prefix",
                    absorbing_cost=5.0 * (horizon - controls) if fallen else 0.0,
                    unknown_partial_controls=0,
                    unknown_partial_physics_substeps=0,
                    integration_spec=16383,
                    dimensions=[9, 9, 6],
                )
            )
        ),
    )
    digest = hashlib.sha256()
    for name in TRACE_KEYS[:4]:
        digest.update(arrays[name][0].tobytes())
    previous = digest.digest()
    arrays["initial_digest"] = np.frombuffer(previous, dtype="u1").copy()
    for index in range(controls):
        digest = hashlib.sha256(previous)
        for name in TRACE_KEYS[:4]:
            digest.update(arrays[name][index + 1].tobytes())
        for name in TRACE_KEYS[4:10]:
            digest.update(np.asarray(arrays[name][index]).tobytes())
        previous = digest.digest()
        arrays["row_digests"][index] = np.frombuffer(previous, dtype="u1")
    return arrays


def test_audit_import_is_lazy_and_missing_run_stays_missing(tmp_path):
    script = """import sys
from src.gps.mppi_gps.audits.similarity import native_outcome
assert not any(k in sys.modules for k in ['torch', 'mujoco', 'src.gps.mppi_gps.policy'])
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    run, out = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    result = audit.audit_run(run, out)
    assert result["passed"] is False
    assert result["new_optimizer_steps"] == result["new_physics_controls"] == 0
    assert result["work"]["neural_rows"] == 0
    assert "Missing/unexecuted evidence stays missing" in result["unavailable_outcomes"]
    with pytest.raises(FileExistsError):
        audit.audit_run(run, out)


def test_audit_association_hashes_match_the_launch_finalizer_contract_on_failure(tmp_path):
    """A later missing worker cannot erase already verified immutable bindings."""
    run, out = tmp_path / "run", tmp_path / "audit"
    (run / "source").mkdir(parents=True)
    (run / "prepared").mkdir()

    def write(path, value):
        path.write_text(json.dumps(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    protocol, xml = run / "source/protocol.md", run / "source/walker.xml"
    protocol.write_text("Synthetic frozen protocol")
    xml.write_text("Synthetic frozen model bytes")
    inputs = {}
    files = {}
    for role, filename in (("reference", "reference.pt"), ("normalization", "normalization.npz")):
        path = run / "prepared" / filename
        path.write_bytes(role.encode())
        inputs[role] = dict(
            bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        )
        files["prepared/" + filename] = inputs[role]
    inputs["xml"] = dict(file="walker.xml", sha256=hashlib.sha256(xml.read_bytes()).hexdigest())
    config = dict(
        schema="walker-similarity-j1-execution-v1",
        automatic_followup=False,
        guidance_launch_authorized=False,
        protocol="protocol.md",
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        inputs=inputs,
        limits=dict(
            audit_read_bytes=audit.MAX_READ,
            audit_stage_wall_seconds=audit.MAX_WALL,
            audit_wall_seconds=audit.MAX_WALL - 60,
            audit_output_bytes=audit.MAX_OUTPUT,
            audit_neural_rows=audit.MAX_NEURAL,
        ),
    )
    config_hash = write(run / "config.json", config)
    source_hash = write(
        run / "manifest.json",
        dict(
            config_sha256=config_hash,
            files=[
                dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                for path in (protocol, xml)
            ],
        ),
    )
    input_hash = write(run / "input_manifest.json", dict(files=files))
    result = audit.audit_run(run, out)
    assert result["passed"] is False
    assert result["config_sha256"] == config_hash
    assert result["source_manifest_sha256"] == source_hash
    assert result["input_manifest_sha256"] == input_hash
    assert result["worker_result_sha256"] is None
    assert result["work"]["neural_rows"] == 0


def test_independent_native_geometry_terminal_and_absorbing_accounting():
    full = native_trace()
    outcome = audit.native_outcome(full, task=TASK)
    assert outcome["walking"] and outcome["survived"] and outcome["cost"] == 0
    prefix = native_trace(11, fallen=True)
    outcome = audit.native_outcome(prefix, task=TASK)
    assert outcome["native_terminal"] == 20 and outcome["raw_cost"] == 5
    assert outcome["absorbing_cost"] == 4945 and outcome["cost"] == 4970
    assert not outcome["walking"] and not outcome["survived"]
    assert audit.verify_trace_arrays(prefix, task=TASK)["cost"] == outcome["cost"]


@pytest.mark.parametrize(
    "change",
    [
        "state",
        "warmstart",
        "torque",
        "observation",
        "cost",
        "terminal",
        "timing",
        "postfall",
        "short",
    ],
)
def test_native_oracle_detects_semantic_mutations_without_hashes(change):
    arrays = native_trace(3, fallen=True)
    if change == "state":
        arrays["integrations"][1, 2] += 0.1
    elif change == "warmstart":
        arrays["warmstarts"][1, 0] = 1
    elif change == "torque":
        arrays["integrations"][1, 28] = 0.1
    elif change == "observation":
        arrays["observations"][1, 0] += 0.1
    elif change == "cost":
        arrays["native_costs"][-1] += 0.1
    elif change == "terminal":
        arrays["fall_costs"][:] = 20
    elif change == "timing":
        arrays["states"][-1, 0] += 0.0001
        arrays["integrations"][-1, 0] += 0.0001
    elif change == "postfall":
        arrays["done"][0] = 1
    else:
        arrays = native_trace(3)
    with pytest.raises(audit.AuditError):
        audit.native_outcome(arrays, task=TASK)


def test_motion_and_all_five_score_oracles_agree_with_separate_production_formula():
    rng = np.random.default_rng(91)
    teacher = native_trace()
    teacher["actions"][:] = rng.uniform(-0.5, 0.5, size=(1000, 6))
    teacher["states"][:, 4:10] = rng.normal(0, 0.1, size=(1001, 6))
    teacher["observations"] = np.c_[teacher["states"][:, 2:10], teacher["states"][:, 10:19]]
    incoming = native_trace(41, fallen=True)
    prediction = rng.uniform(-0.4, 0.4, size=(1000, 6)).astype(np.float32)
    norm = dict(mean=np.zeros(17), std=np.ones(17))
    explainer = metrics.fit_state_explainer(teacher["observations"][:-1], teacher["actions"], norm)
    cloud = np.r_[
        np.c_[teacher["observations"][:-1], teacher["actions"]],
        np.c_[incoming["observations"][:-1], incoming["actions"]],
    ]
    occupancy = metrics.fit_occupancy_normalizer(cloud)
    models = dict(explainer=explainer, occupancy_normalizer=occupancy)
    actual = audit.score_oracle(teacher, incoming, prediction, models)
    expected = metrics.score_case(
        audit.metric_trace(teacher), audit.metric_trace(incoming), prediction, explainer, occupancy
    )["scores"]
    for key in metrics.SCORE_IDS:
        assert actual[key] == pytest.approx(expected[key], rel=0, abs=1e-9)
    actual_motion = audit.motion_oracle(teacher, incoming)
    expected_motion = metrics.motion_metrics(
        audit.metric_trace(teacher), audit.metric_trace(incoming)
    )
    assert actual_motion["error"] == pytest.approx(expected_motion["error"], rel=0, abs=1e-12)
    assert actual_motion["final_relative_progress_error"] is None
    assert actual_motion["reproduced"] is False


def test_planning_cost_oracle_excludes_postfall_rows_and_invalid_prefix():
    states = np.broadcast_to(native_trace(4)["states"][1:], (2, 4, 19)).copy()
    actions = np.zeros((2, 4, 6))
    states[0, 1:, 2] = 0.7
    states[0, 2:, 10] = 1e6
    value = audit.planning_cost_oracle(states, actions, TASK)
    assert value[0] == 35  # first fall 25, two missing controls charged five each
    assert value[1] == 0
    states[0, 3] = np.nan  # hypothetical postfall garbage remains outside the active prefix
    assert audit.planning_cost_oracle(states, actions, TASK)[0] == 35
    states[1, 0] = np.nan
    assert np.isinf(audit.planning_cost_oracle(states, actions, TASK)[1])


def test_reader_checks_bytes_typed_array_paths_and_combined_budget(tmp_path):
    root, out = tmp_path / "run", tmp_path / "out"
    root.mkdir()
    out.mkdir()
    path = root / "tape.npy"
    value = np.arange(20, dtype=np.int64).reshape(5, 4)
    np.save(path, value, allow_pickle=False)
    binding = dict(
        file="tape.npy",
        bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        array_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
        shape=[5, 4],
        dtype="int64",
    )
    reader = audit.Reader(root, out)
    np.testing.assert_array_equal(reader.npy(binding), value)
    with pytest.raises(audit.AuditError, match="binding"):
        reader.npy(dict(binding, array_sha256="0" * 64))
    with pytest.raises(audit.AuditError, match="escaped"):
        reader.read("../out/absent")
    (root / "link").symlink_to(path)
    with pytest.raises(audit.AuditError, match="escaped"):
        reader.read("link")
    limited = audit.Reader(root, out, read_limit=path.stat().st_size - 1)
    with pytest.raises(audit.AuditError, match="limit"):
        limited.npy(binding)
    assert limited.read_bytes == 0
    with pytest.raises(audit.AuditError, match="Duplicate"):
        audit._json('{"a":1,"a":2}')
    with pytest.raises(audit.AuditError, match="Nonfinite"):
        audit._json('{"a":NaN}')


def test_distinct_array_hash_contracts_remain_distinct():
    value = np.arange(12, dtype=np.float32).reshape(2, 6)
    raw = hashlib.sha256(value.tobytes()).hexdigest()
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    file_hash = hashlib.sha256(stream.getvalue()).hexdigest()
    typed = audit._policy_array_hash(value)
    assert len({raw, file_hash, typed}) == 3
    assert typed == audit._policy_array_hash(value.astype(np.float64))


def test_saved_actor_verification_respects_batch_shape_and_dtype(tmp_path):
    torch = pytest.importorskip("torch")
    from src.gps.mppi_gps.policy import Policy

    root, out = tmp_path / "run", tmp_path / "out"
    root.mkdir()
    out.mkdir()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        actor = Policy(seed=31, hidden_dim=128)
        binding = actor.checkpoint(root / "parent.pt")
        binding["file"] = "parent.pt"
        verifier = audit.Actors(audit.Reader(root, out), maximum=3000)
        observations = np.random.default_rng(81).normal(size=(1000, 17))
        expected = actor.predict(observations)
        verifier.verify(binding, observations, expected, batch_one=False)
        native = np.stack([actor.predict(row) for row in observations[:5]]).astype(np.float64)
        verifier.verify(binding, observations[:5], native, batch_one=True)
        assert verifier.completed == verifier.requested == 1005
        assert verifier.calls == 6 and verifier.loads == 1
        wrong = native.copy()
        wrong[0, 0] += 1e-12
        with pytest.raises(audit.AuditError, match="actions differ"):
            verifier.verify(binding, observations[:5], wrong, batch_one=True)
        with pytest.raises(audit.AuditError, match="dtype"):
            verifier.verify(binding, observations[:5], native.astype(np.float32), batch_one=True)
    finally:
        torch.set_num_threads(previous)


def test_saved_continuation_fixture_checks_adam_rng_tape_and_every_update_without_fitting(tmp_path):
    """A synthetic endpoint package checks the auditor; it is no learning result."""
    torch = pytest.importorskip("torch")
    from src.gps.mppi_gps.similarity_execution import JMeter, parent_prepare

    run, out = tmp_path / "run", tmp_path / "audit"
    out.mkdir()
    (run / "prepared").mkdir(parents=True)
    np.savez(run / "prepared/normalization.npz", mean=np.zeros(17), std=np.ones(17))
    meter = JMeter(run / "worker/evidence", {})
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        parent = parent_prepare(
            31, 37, dict(mean=np.zeros(17), std=np.ones(17)), run / "worker/parents/p0", meter
        )
        teacher = native_trace()
        package = copy.deepcopy(parent["package"])
        package.update(
            updates=2000,
            fit_updates=2000,
            rng=parent["bindings"]["terminal_pcg64"],
            fit_data=dict(
                observations_sha256=audit._policy_array_hash(teacher["observations"][:-1]),
                actions_sha256=audit._policy_array_hash(teacher["actions"]),
                rows=1000,
            ),
        )
        for index, parameter in enumerate(parent["actor"].parameters()):
            package["adam"]["state"][index] = dict(
                step=torch.tensor(2000.0),
                exp_avg=torch.zeros_like(parameter),
                exp_avg_sq=torch.zeros_like(parameter),
            )
        fitdir = run / "worker/fits/synthetic"
        child = parent["actor"].checkpoint(
            fitdir / "child.pt", optimizer_state=package, metadata=dict(synthetic_fixture=True)
        )
        child["file"] = str((fitdir / "child.pt").relative_to(run))
        receipt = dict(
            schema="walker-j1-learning-case-fit-v1",
            incoming_actor_hash=parent["actor"].hash(),
            child_actor_hash=parent["actor"].hash(),
            incoming_optimizer_fingerprint=parent["bindings"]["optimizer_fingerprint"],
            child_optimizer_fingerprint=audit.optimizer_fingerprint(package),
            checkpoint=child,
            updates=2000,
            tape_sha256=parent["bindings"]["tape"]["sha256"],
            observed_minibatch_hashes=[
                hashlib.sha256(row.tobytes()).hexdigest() for row in parent["tape"]
            ],
            all2000minibatches_and_rng_matched=True,
            parent_unchanged=True,
            global_rng_unchanged=True,
        )
        receipt_path = fitdir / "fit_receipt.json"
        receipt_path.write_text(json.dumps(receipt))
        (fitdir / "learning_curve.json").write_text(
            json.dumps(
                dict(
                    completed_updates=2000,
                    learning_curve=[
                        dict(
                            update=i + 1,
                            fit_update=i + 1,
                            action_mse=0.0,
                            proximal_loss=0.0,
                            total_loss=0.0,
                            unclipped_gradient_norm=0.0,
                        )
                        for i in range(2000)
                    ],
                )
            )
        )
        case = dict(
            case_id="synthetic",
            incoming_bindings=parent["bindings"],
            parent_checkpoint=parent["bindings"]["checkpoint"],
            child_checkpoint=child,
            tape_file=parent["bindings"]["tape"]["file"],
            fit_receipt_file=str(receipt_path.relative_to(run)),
        )
        actors = audit.Actors(audit.Reader(run, out))
        result = audit.verify_learning_case(actors.reader, actors, case, teacher)
        assert result["optimizer_rng_tape_verified"] and result["learning_curve_points"] == 2000
        assert result["fitting_repeated"] is False
        assert actors.requested == 0 and actors.loads == 2
        assert (
            meter.work.get("optimizer_steps", 0) == 0 and meter.work.get("native_controls", 0) == 0
        )
        changed = copy.deepcopy(case)
        changed["parent_checkpoint"] = dict(changed["parent_checkpoint"], sha256="0" * 64)
        with pytest.raises(audit.AuditError, match="checkpoint binding"):
            audit.verify_learning_case(actors.reader, actors, changed, teacher)
        receipt["observed_minibatch_hashes"][0] = "0" * 64
        bad_receipt_path = fitdir / "bad_receipt.json"
        bad_receipt_path.write_text(json.dumps(receipt))
        changed = dict(case, fit_receipt_file=str(bad_receipt_path.relative_to(run)))
        with pytest.raises(audit.AuditError, match="minibatches"):
            audit.verify_learning_case(actors.reader, actors, changed, teacher)
    finally:
        meter.close()
        torch.set_num_threads(previous)


def test_paid_ledger_census_includes_implicit_prediction_calls_and_nested_owners(tmp_path):
    root, out = tmp_path / "run", tmp_path / "out"
    root.mkdir()
    out.mkdir()
    ledger = audit.Ledger(audit.Reader(root, out), TASK, [], None, {})
    rows = [
        dict(event="runtime_open"),
        dict(event="paid_requested", operation=1, counts={"fits": 1}),
        dict(event="work_requested", operation=2, category="neural_rows", amount=1000),
        dict(event="work_complete", operation=2, category="neural_rows", amount=1000),
        dict(event="paid_complete", operation=1, counts={"fits": 1}),
        dict(event="runtime_close"),
    ]
    for row in rows:
        ledger.event(row)
    costs = dict(
        requested=dict(ledger.requested),
        completed=dict(ledger.completed),
        unknown_partial={},
        returned_unpersisted={},
    )
    result = ledger.finish(dict(costs=costs), dict(fits=1, neural_rows=1000, prediction_calls=1))
    assert result["completed"] == dict(fits=1, neural_rows=1000, prediction_calls=1)
    with pytest.raises(audit.AuditError, match="closure"):
        ledger.event(dict(event="work_requested", operation=3, category="fits", amount=1))


def test_paid_ledger_rejects_unknown_or_unfinished_work(tmp_path):
    reader = audit.Reader(tmp_path, tmp_path / "out")
    ledger = audit.Ledger(reader, TASK, [], None, {})
    ledger.event(dict(event="runtime_open"))
    ledger.event(dict(event="paid_requested", operation=1, counts={"model_controls": 10}))
    with pytest.raises(audit.AuditError, match="requested/completed"):
        ledger.event(dict(event="paid_complete", operation=1, counts={"model_controls": 9}))
    with pytest.raises(audit.AuditError, match="uncertain"):
        ledger.event(dict(event="paid_error", operation=1))


def test_worker_verdict_and_summary_cannot_bypass_independent_audit():
    cases = [dict(split="development", eligible=True, group="development_00")]
    teachers = {"development_00_mppi": dict(tag="development_00_mppi")}
    outcomes = {"development_00_mppi": dict(controls=1000, walking=True, survived=True, cost=50.0)}
    worker = dict(
        status="CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
        objective_passed=False,
        protected_test_revealed=False,
        cases=1,
        eligible_cases=1,
        declared_source_groups=16,
        independent_groups=1,
        acquired_source_groups=1,
        no_guidance_experiment=True,
        guidance_launched=False,
        automatic_followup=False,
        teacher_outcomes=outcomes,
    )
    checks = dict(objective_passed=False)
    assert audit.verify_worker_summary(worker, cases, teachers, outcomes, checks)["verdict_bound"]
    with pytest.raises(audit.AuditError, match="objective_passed"):
        audit.verify_worker_summary(
            dict(worker, objective_passed=True), cases, teachers, outcomes, checks
        )
    changed = copy.deepcopy(worker)
    changed["teacher_outcomes"]["development_00_mppi"]["walking"] = False
    with pytest.raises(audit.AuditError, match="walking"):
        audit.verify_worker_summary(changed, cases, teachers, outcomes, checks)


def test_complete_two_control_synthetic_stream_joins_intents_arrays_costs_and_contacts(tmp_path):
    from src.gps.mppi_gps.consequence_storage import C1TraceRecorder
    from src.gps.mppi_gps.similarity_execution import JMeter, paid

    run, out = tmp_path / "run", tmp_path / "audit"
    out.mkdir()
    meter = JMeter(run / "worker/evidence", {})
    arrays = native_trace(2, horizon=2)
    recorder = C1TraceRecorder(meter, run / "worker/active")

    def boundary(index):
        return dict(
            state=arrays["states"][index],
            warmstart=arrays["warmstarts"][index],
            integration=arrays["integrations"][index],
            spec=np.array(16383),
            dimensions=np.array([9, 9, 6]),
        )

    recorder.begin("fixture", 2, boundary(0), arrays["observations"][0])
    for index in range(2):
        recorder.intent(
            arrays["actions"][index],
            snapshot=boundary(index),
            observation=arrays["observations"][index],
        )
        paid(meter, dict(native_controls=1, physics_substeps=4), lambda: None)
        recorder.completed(
            boundary(index + 1),
            arrays["observations"][index + 1],
            cost=0.0,
            velocity=1.5,
            fall=0.0,
            done=False,
        )
    reference = recorder.close(meta=dict(integration_spec=16383, dimensions=[9, 9, 6]))
    contact = meter.save_arrays(
        "contacts",
        contacts=np.zeros((2, 2), dtype=bool),
        contact_normal_forces=np.zeros((2, 2)),
        control_ordinals=np.arange(2, dtype=np.int64),
    )
    outcome = dict(
        tag="fixture",
        trace=reference,
        contact_trace=contact,
        **audit.native_outcome(arrays, task=TASK, horizon=2),
    )
    meter.close()
    reader = audit.Reader(run, out)
    archive = audit.StreamArchive(meter.out, out / "index.sqlite", max_read_bytes=10 * 1024**2)
    reader.archive = archive
    try:
        archive.verify_index()
        ledger = audit.Ledger(reader, TASK, [outcome], None, {})
        for row in archive.ledger():
            ledger.event(row)
        census = ledger.finish(dict(costs=meter.accounting()), meter.limits)
        assert census["closed_traces"] == 1
        assert census["completed"]["native_controls"] == 2
        assert ledger.trace_outcomes["fixture"]["survived"] is True
        assert ledger.trace_outcomes["fixture"]["walking"] is False
        archive.verify_unchanged()
    finally:
        archive.close()


def test_independent_probe_selector_oracle_matches_native_array_controller_and_rejects_mutation():
    from src.gps.mppi_gps.search import CoupledSelector, IndependentMPPI
    from src.gps.mppi_gps.similarity_probe import IterationObserver
    from src.utils.config import MPPIConfig

    class Environment:
        action_dim = 6
        action_bounds = (-1.0, 1.0)

        def batch_rollout(self, state, actions, initial_warmstart=None):
            posts = np.broadcast_to(state, (*actions.shape[:2], 19)).copy()
            posts[:, :, 0] += np.arange(1, actions.shape[1] + 1) * 0.008
            return (
                posts,
                audit.planning_cost_oracle(posts, actions, TASK),
                np.empty((*actions.shape[:2], 0)),
            )

        def rollout_states_to_obs(self, states):
            return np.concatenate(
                (states[..., 2:10], np.clip(states[..., 10:19], -10, 10)), axis=-1
            )

        def _is_healthy(self, height, pitch):
            return (height > 0.8) & (height < 2) & (pitch > -1) & (pitch < 1)

    saved = {}

    class Meter:
        def save_arrays(self, tag, **arrays):
            saved[tag] = arrays
            return {"tag": tag}

        def record(self, event, **fields):
            return "event-000000001"

    state = native_trace(2, horizon=2)["states"][0]
    warm = np.zeros(9)
    config = dict(K=4, H=3, lam=3.0, noise_sigma=0.3, initial_iterations=1)
    planner = IndependentMPPI(Environment(), MPPIConfig(**config), 9)
    before = planner.private_state()
    with IterationObserver(planner._planner) as observer:
        bank = planner.propose(state, warm)
    plan, selection = CoupledSelector(alpha=0, eta_u=0.01).select(
        bank, actor=None, meter=Meter(), enabled=False
    )
    row = dict(selection=selection, planner_before=before, selected_plan=plan, action=plan[0])
    reader = SimpleNamespace(arrays=lambda reference: saved[reference["tag"]])
    context = dict(state=state, warmstart=warm)
    config = dict(mppi=config, selector=dict(alpha=0, eta_u=0.01), task=TASK)
    checks = audit.audit_probe_selection(reader, row, context, observer.passes[-1], config)
    assert checks["bank_candidates"] == 5 and checks["native_cost_rows"] == 15
    stored = saved[selection["evidence"]["tag"]]
    stored["bank_raw_costs"] = stored["bank_raw_costs"].copy()
    stored["bank_raw_costs"][0] += 1
    with pytest.raises(audit.AuditError, match="digest"):
        audit.audit_probe_selection(reader, row, context, observer.passes[-1], config)


def test_probe_roster_and_descriptive_independent_oracle():
    from src.gps.mppi_gps.similarity_probe import decision_roster, summarize_probe

    roster = audit.probe_roster_oracle()
    assert roster == decision_roster() and len(roster) == 208
    gain = np.random.default_rng(92).normal(size=(17, 6))
    rows = []
    for request in roster:
        obs = np.asarray(request["observation_delta"])
        action = obs @ gain + 0.01 * request["innovation"] + 0.02 * (request["nominal"] == "zero")
        action += (
            0.3 * request["time_delta"] + 0.2 * request["rootx_delta"] + request["warmstart_delta"]
        )
        presence = np.array([True, False])
        if request["role"] == "gain_axis" and request["coordinate"] == 0:
            presence[0] = False
        rows.append(
            dict(
                request=request,
                observation=obs,
                action=action,
                selected_plan=np.tile(action, (48, 1)),
                planner_after=dict(
                    U=np.zeros((48, 6)), rng_state={"source": 1}, initial_plan=False
                ),
                original_replay_verified=request["role"] == "original_repeat"
                or (
                    request["role"] == "factorial"
                    and request["nominal"] == "carried"
                    and request["innovation"] == 0
                ),
                contacts=dict(presence=presence),
            )
        )
    std = np.linspace(0.5, 2, 17)
    expected = summarize_probe(rows, normalization_std=std)
    actual = audit.probe_descriptive_oracle(rows, std)
    audit._same(actual, expected, "synthetic independent descriptive oracle", 1e-8)
    assert actual["original_replay_checks"] == 8
    assert all(f["axis_contact_regime_changes"] == 2 for f in actual["local_fits"])
    assert len(actual["hidden_input_contrasts"]) == 12
    assert not actual["optimal_gain_claim"]
    changed = copy.deepcopy(actual)
    changed["local_fits"][0]["axis_contact_regime_changes"] = 0
    with pytest.raises(audit.AuditError, match="integer differs"):
        audit._same(actual, changed, "corrupt contact-regime count", 1e-8)
