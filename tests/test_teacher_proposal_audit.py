"""Independent H5 evidence checks with bounded fake physics and fake actors."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps.audits.teacher_proposal import (
    AuditError,
    AuditJournal,
    RoundChecks,
    SavedActorVerifier,
    StreamArchive,
    agreement_values,
    audit_preparation,
    audit_stream_structure,
    audit_worker,
    healthy_prefix,
    native_cost,
    native_observations,
    optimizer_fingerprint,
)
from src.gps.mppi_gps.guided_proposals import TwoStreamMPPI
from src.gps.mppi_gps.proposal_evidence import BinaryEvidenceMeter
from src.utils.config import MPPIConfig


def meter(path):
    return BinaryEvidenceMeter(
        path, max_controls=100000, max_neural_rows=100000, max_bytes=64 * 1024**2
    )


def archive(path, output, **kwargs):
    return StreamArchive(path, output / "lookup.sqlite", max_read_bytes=256 * 1024**2, **kwargs)


def test_stream_exact_index_binding_and_structure(tmp_path):
    root = tmp_path / "evidence"
    m = meter(root)
    reference = m.save_arrays(
        "known", rows=np.arange(6, dtype=np.float64).reshape(2, 3), owner={"source": "synthetic"}
    )
    m.close()
    with archive(root, tmp_path / "audit") as reader:
        result = reader.verify_index()
        assert result["groups"] == 1 and result["arrays"] == 2
        assert np.array_equal(reader.array(reference, "rows"), np.arange(6).reshape(2, 3))
        assert reader.json_array(reference, "owner") == {"source": "synthetic"}
        verified = audit_stream_structure(reader)
        assert verified["structure_complete"] is True
        assert verified["scientific_round_audit_complete"] is False
        assert verified["new_network_rows"] == verified["new_physics_controls"] == 0


def test_reference_cannot_relabel_another_committed_group_or_byte_range(tmp_path):
    root = tmp_path / "evidence"
    m = meter(root)
    reference = m.save_arrays("known", rows=np.arange(3))
    m.close()
    forged = json.loads(json.dumps(reference))
    forged["arrays"]["rows"]["offset"] += 1
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        before = reader.read_bytes
        with pytest.raises(AuditError, match="committed index"):
            reader.array(forged, "rows")
        assert reader.read_bytes == before


@pytest.mark.parametrize(
    "problem", ["payload", "missing_completion", "extra_payload", "duplicate_key"]
)
def test_stream_corruption_missing_evidence_and_ambiguous_json_stop(tmp_path, problem):
    root = tmp_path / "evidence"
    m = meter(root)
    m.save_arrays("known", rows=np.arange(3))
    m.close()
    payload = root / "arrays.bin"
    index = root / "arrays.index.jsonl"
    if problem == "payload":
        data = bytearray(payload.read_bytes())
        data[0] ^= 1
        payload.write_bytes(data)
    elif problem == "extra_payload":
        with payload.open("ab") as stream:
            stream.write(b"unindexed")
    elif problem == "missing_completion":
        index.write_bytes(index.read_bytes().splitlines(keepends=True)[0])
    else:
        index.write_bytes(
            index.read_bytes().replace(
                b'"event":"arrays_requested"',
                b'"event":"arrays_requested","event":"arrays_requested"',
            )
        )
    with archive(root, tmp_path / "audit") as reader:
        with pytest.raises(AuditError):
            reader.verify_index()


def test_index_read_budget_stops_before_an_unbounded_payload(tmp_path):
    root = tmp_path / "evidence"
    m = meter(root)
    m.save_arrays("known", rows=np.arange(300))
    m.close()
    with StreamArchive(root, tmp_path / "audit.sqlite", max_read_bytes=10) as reader:
        with pytest.raises(AuditError):
            reader.verify_index()
        assert reader.read_bytes <= 10


def test_closed_work_counter_tampering_is_not_trusted(tmp_path):
    root = tmp_path / "evidence"
    m = meter(root)
    m.measure("neural_rows", 4, lambda: None)
    m.close()
    path = root / "ledger.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1]["totals"]["requested_neural_rows"] = 3
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        with pytest.raises(AuditError, match="closing counter"):
            audit_stream_structure(reader)


def test_preaction_masks_native_cost_and_disagreement_have_independent_known_values():
    initial = np.zeros(19)
    initial[2], initial[10] = 1.25, 1.5
    posts = np.broadcast_to(initial, (1, 4, 19)).copy()
    posts[0, :, 10] = [20, 1.5, -20, 1.5]
    posts[0, 1, 2] = 0.7
    posts[0, 3, :] = np.nan
    observations = native_observations(initial, posts)
    assert observations[0, 0, 8] == 1.5 and observations[0, 1, 8] == 10
    assert healthy_prefix(posts).tolist() == [[True, True, False, False]]
    actions = np.ones((1, 4, 6))
    d, mask = agreement_values(observations, actions, posts, np.array([30.0]), np.zeros((2, 6)))
    assert d.tolist() == [12] and mask.tolist() == [[True, True, False, False]]
    task = dict(
        vel_cost_weight=2,
        target_velocity=1.5,
        vertical_velocity_cost_weight=0.5,
        angle_cost_weight=0.1,
        unhealthy_cost_weight=5,
        ctrl_cost_weight=0.001,
        terminal_unhealthy_weight=20,
        apply_terminal_cost_on_done=True,
    )
    # Through first fall:37+0.006 +5+0.006;20 terminal;two5-cost missing controls.
    assert native_cost(posts, actions, task)[0] == pytest.approx(72.012)


class FakeActor:
    def __init__(self, *, corrupt=False):
        self.corrupt = corrupt

    def hash(self):
        return "0" * 64

    def predict(self, observations, meter=None):
        rows = int(np.prod(observations.shape[:-1]))

        def operation():
            return np.full(observations.shape[:-1] + (6,), float(self.corrupt), dtype=np.float32)

        return operation() if meter is None else meter.measure("neural_rows", rows, operation)


class FakePredictor:
    def __init__(self):
        self.neural_rows = 0

    def predict(self, owner, observations):
        assert owner == "0" * 64
        self.neural_rows += 1 if observations.ndim == 1 else len(observations)
        return np.zeros(observations.shape[:-1] + (6,), dtype=np.float32)

    def verify(self, owner, observations, saved):
        actual = self.predict(owner, observations)
        if actual.dtype != saved.dtype or actual.tobytes() != saved.tobytes():
            raise AuditError("synthetic prediction mismatch")


class FakeNative:
    action_bounds = (-np.ones(6), np.ones(6))
    action_dim = 6

    def __init__(self, recorder):
        self.recorder = recorder

    @staticmethod
    def _is_healthy(z, pitch):
        return (z > 0.8) & (z < 2) & (pitch > -1) & (pitch < 1)

    @staticmethod
    def rollout_states_to_obs(states):
        return np.concatenate((states[..., 2:10], np.clip(states[..., 10:19], -10, 10)), axis=-1)

    def batch_rollout(self, initial, actions, initial_warmstart=None):
        posts = np.broadcast_to(initial, actions.shape[:2] + (19,)).copy()
        posts[..., 0] += (np.arange(actions.shape[1]) + 1)[None] * 0.008
        costs = 0.001 * np.sum(actions**2, axis=(1, 2))
        inputs = dict(
            initial_state=initial,
            actions=actions,
            initial_warmstart=np.zeros(9) if initial_warmstart is None else initial_warmstart,
            live_integration_before=np.zeros(64),
        )
        return self.recorder._query(
            "model",
            int(np.prod(actions.shape[:2])),
            4,
            inputs,
            lambda: (posts, costs, np.empty(actions.shape[:2] + (0,))),
            lambda value: dict(states=value[0], raw_scores=value[1], sensordata=value[2]),
        )


def recipe():
    return json.loads(Path("configs/studies/teacher_proposal_h5_0.json").read_text())


def write_decision(
    path,
    arm,
    *,
    corrupt=False,
    altered_noise=False,
    steps=1,
    real_context=False,
    disturb_before_second=False,
):
    m = meter(path)
    env = FakeNative(m)
    stream = recipe()["streams"][0]
    controller = TwoStreamMPPI(
        env,
        MPPIConfig(**recipe()["mppi"]),
        stream["protected_planner_seeds"][0] if real_context else 1,
        stream["second_planner_seeds"][0] if real_context else 2,
    )
    initial = np.zeros(19)
    initial[2], initial[10] = 1.25, 1.5
    if altered_noise:
        controller.protected._planner.noise_chol *= 1.01
    if real_context:
        m.record("checkpoint", file="checkpoints/s0_parent.pt", actor_hash="0" * 64)
    for step in range(steps):
        m.context = dict(
            role="teacher",
            episode="teacher_s0_independent_00" if real_context else "synthetic",
            step=step,
        )
        if real_context:
            m.context.update(stream=0, arm="independent", actor_hash="0" * 64)
        if step == 1 and disturb_before_second:
            controller.protected._planner.U[0, 0] += 1e-3
        plan, _ = controller.decide(
            initial, FakeActor(corrupt=corrupt), m, arm=arm, warmstart=np.zeros(9)
        )
        m._query(
            "live",
            1,
            4,
            dict(action=plan[0], state=initial, warmstart=np.zeros(9), integration=np.zeros(64)),
            lambda: None,
            lambda _: dict(done=np.array(False)),
        )
    m.close()


@pytest.mark.parametrize("arm", ["I", "S", "P"])
def test_independent_audit_reconstructs_all_passes_means_scores_rng_and_guard(tmp_path, arm):
    root = tmp_path / "evidence"
    write_decision(root, arm)
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        checker = RoundChecks(reader, recipe(), FakePredictor(), run=tmp_path)
        structure = audit_stream_structure(reader, event_observer=checker.event)
        result = checker.summary()
        assert structure["work"]["model_queries"] == 8
        assert result["selection_checks"] == 1
        assert result["agreement_checks"] == 4
        assert result["independently_verified_prediction_rows"] == (2 * 256 + 515) * 48
        assert result["native_replay_controls"] == 0


def test_saved_scores_cannot_hide_predictions_from_another_actor(tmp_path):
    root = tmp_path / "evidence"
    write_decision(root, "P", corrupt=True)
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        checker = RoundChecks(reader, recipe(), FakePredictor(), run=tmp_path)
        with pytest.raises(AuditError, match="prediction mismatch"):
            audit_stream_structure(reader, event_observer=checker.event)


def test_audit_import_does_not_load_execution_policy_planner_or_simulator():
    code = """
import sys
from src.gps.mppi_gps.audits import teacher_proposal
for prefix in ('torch','mujoco','h5py','src.envs','src.mppi','src.gps.mppi_gps.policy','src.gps.mppi_gps.proposal_evidence','src.gps.mppi_gps.runtime','src.gps.mppi_gps.guided_proposals'):
    assert not any(name==prefix or name.startswith(prefix+'.') for name in sys.modules),prefix
print('independent-lazy')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "independent-lazy"


def write_short_episode(path, role, *, bad_solver=None):
    m = meter(path)
    initial = np.zeros(19)
    initial[2], initial[10] = 1.25, 1.5
    states = np.broadcast_to(initial, (4, 19)).copy()
    states[:, 0] = np.arange(4) * 0.008
    states[-1, 2] = 0.7
    warmstarts = np.arange(4 * 9, dtype=float).reshape(4, 9)
    integrations = np.arange(4 * 64, dtype=float).reshape(4, 64)
    observations = np.concatenate(
        (states[:-1, 2:10], np.clip(states[:-1, 10:19], -10, 10)), axis=-1
    )
    m.context = dict(role=role, actor_hash="0" * 64, episode="short")
    actor = FakeActor()
    actions, costs = [], [0.0, 0.0, 25.0]
    for index in range(3):
        action = (
            actor.predict(observations[index], meter=m)
            if role == "actor_only"
            else np.zeros(6, dtype=np.float32)
        )
        actions.append(action)
        m._query(
            "live",
            1,
            4,
            dict(
                state=states[index],
                action=action.astype(float),
                warmstart=warmstarts[index],
                integration=integrations[index],
            ),
            lambda: None,
            lambda _, index=index: dict(
                state=states[index + 1],
                cost=np.array(costs[index]),
                warmstart=warmstarts[index + 1],
                integration=integrations[index + 1],
            ),
        )
    actions = np.asarray(actions)
    infos = {"rows": [dict(controller=dict(policy_hash="0" * 64)) for _ in range(3)]}
    if bad_solver == "warmstart":
        warmstarts[-1, 0] += 1
    elif bad_solver == "integration":
        integrations[-1, 0] += 1
    reference = m.save_arrays(
        "episode_short",
        states=states,
        observations=observations,
        actions=actions,
        costs=np.asarray(costs),
        velocities=states[1:, 10],
        infos=infos,
        warmstarts=warmstarts,
        integrations=integrations,
        terminal_cost=np.array(20.0),
        terminal_adjustment=np.array(0.0),
        absorbing_cost=np.array(4985.0),
    )
    m._counts["episodes"] += 1
    m.record(
        "episode_complete",
        raw=reference,
        steps=3,
        tag="short",
        survived=False,
        success=False,
        mae=0.0,
        cost=5010.0,
        absorbing_cost=4985.0,
        terminal_adjustment=0.0,
    )
    if role == "teacher":
        prediction = actor.predict(observations, meter=m)
        squared = np.sum((actions - prediction) ** 2, axis=1)
        saved = m.save_arrays("posthoc", policy_actions=prediction, squared_disagreement=squared)
        m.record(
            "teacher_agreement_complete",
            actor_hash="0" * 64,
            episode_raw=reference,
            predictions=dict(reference=saved, key="policy_actions"),
            squared_disagreement=dict(reference=saved, key="squared_disagreement"),
            rows=len(observations),
            action_agreement=float(np.mean(squared)),
        )
    metric = actor.predict(observations.astype(np.float32), meter=m)
    assert not np.any(metric)
    m.record(
        "policy_metric_complete",
        actor_hash="0" * 64,
        observations=dict(reference=reference, key="observations"),
        targets=dict(reference=reference, key="actions"),
        rows=3,
        batch_size=1024,
        action_mse=0.0,
        tag="synthetic_metric",
    )
    m.close()


@pytest.mark.parametrize("role", ["actor_only", "teacher"])
def test_native_live_joins_full_cost_missing_tail_direct_posthoc_and_metric(tmp_path, role):
    root = tmp_path / "evidence"
    write_short_episode(root, role)
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        checker = RoundChecks(reader, recipe(), FakePredictor(), run=tmp_path)
        audit_stream_structure(reader, event_observer=checker.event)
        result = checker.finish()
        assert result["episodes"] == 1
        assert result["standalone_episode_checks"] == int(role == "actor_only")
        assert result["teacher_posthoc_checks"] == int(role == "teacher")
        assert result["full_metric_checks"] == 1
        assert result["independently_verified_prediction_rows"] == 6


def test_manual_network_replica_preserves_direct_vector_shape_and_rejects_changed_bytes(tmp_path):
    import torch

    state = dict(observation_mean=torch.zeros(17), observation_std=torch.ones(17))
    for layer, width, previous in ((0, 128, 17), (2, 128, 128), (4, 6, 128)):
        state[f"network.{layer}.weight"] = torch.zeros((width, previous))
        state[f"network.{layer}.bias"] = torch.zeros(width)
    owner = hashlib.sha256(b"walker-tanh-policy-v1")
    for name, tensor in sorted(state.items()):
        a = tensor.numpy()
        owner.update(
            json.dumps([name, list(a.shape), str(a.dtype)], separators=(",", ":")).encode()
        )
        owner.update(a.tobytes())
    key = owner.hexdigest()
    path = tmp_path / "initial.pt"
    torch.save(
        dict(
            schema="walker-tanh-policy-v1", policy_hash=key, state_dict=state, optimizer_state=None
        ),
        path,
    )
    registry = {
        key: dict(
            file=path.name,
            bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    }
    verifier = SavedActorVerifier(
        registry,
        tmp_path,
        max_neural_rows=3,
        max_scope_neural_rows={"shared_parent": 1, "independent": 2},
    )
    verifier.set_scope("shared_parent")
    verifier.verify(key, np.ones(17), np.zeros(6, dtype=np.float32))
    verifier.set_scope("independent")
    verifier.verify(key, np.ones((1, 17)), np.zeros((1, 6), dtype=np.float32))
    assert verifier.neural_rows == verifier.prediction_calls == 2
    assert verifier.checkpoint_loads == 1
    with pytest.raises(AuditError, match="prediction bytes differ"):
        verifier.verify(key, np.ones(17), np.ones(6, dtype=np.float32))
    with pytest.raises(AuditError, match="reservation exhausted"):
        verifier.predict(key, np.ones(17))
    assert verifier.neural_rows == 3
    assert verifier.scope_neural_rows == {"shared_parent": 1, "independent": 2}
    assert verifier.scope_completed_neural_rows == verifier.scope_neural_rows


def test_scoped_prediction_budget_rejects_unknown_or_overfull_arm_before_loading(tmp_path):
    verifier = SavedActorVerifier(
        {},
        tmp_path,
        max_neural_rows=100,
        max_scope_neural_rows={"independent": 0, "shared_parent": 2},
    )
    with pytest.raises(AuditError, match="Missing audit prediction budget scope"):
        verifier.predict("0" * 64, np.zeros(17))
    with pytest.raises(AuditError, match="Unknown audit prediction budget scope"):
        verifier.set_scope("not_an_arm")
    verifier.set_scope("independent")
    with pytest.raises(AuditError, match="scope neural reservation exhausted"):
        verifier.predict("0" * 64, np.zeros(17))
    assert verifier.neural_rows == verifier.prediction_calls == verifier.checkpoint_loads == 0
    assert verifier.scope_neural_rows == {}


def test_metric_budget_scope_uses_work_owner_even_for_incoming_parent_actor(tmp_path):
    verifier = SavedActorVerifier(
        {},
        tmp_path,
        max_neural_rows=100,
        max_scope_neural_rows={
            "independent": 10,
            "selector_only": 10,
            "proposal_guided": 10,
            "shared_parent": 10,
        },
    )
    checker = RoundChecks(None, recipe(), verifier, run=tmp_path)
    checker._metric_scope("s1_parent_initial/train")
    assert verifier.scope == "shared_parent"
    checker._metric_scope("s1_selector_only_child/incoming")
    assert verifier.scope == "selector_only"
    checker._metric_scope("s1_proposal_guided_child/endpoint")
    assert verifier.scope == "proposal_guided"
    with pytest.raises(AuditError, match="Unknown full-data metric audit budget scope"):
        checker._metric_scope("unknown/incoming")


def test_optimizer_tree_contract_distinguishes_lists_tuples_and_typed_arrays():
    assert optimizer_fingerprint([1, 2]) != optimizer_fingerprint((1, 2))
    assert optimizer_fingerprint(np.array([1, 2], dtype=np.int64)) != optimizer_fingerprint(
        np.array([1, 2], dtype=np.float64)
    )


def write_preparation(path, *, recipe_name=None, authorized=False):
    path.mkdir()
    config = recipe()
    if recipe_name is not None:
        config["recipe_name"] = recipe_name
    if authorized:
        config["launch_authorized"] = True
        config["eligibility"]["decision"] = "approved_single_round_exception"
    source_name = f"configs/studies/{recipe_name or 'teacher_proposal_h5_0'}.json"
    for label, binding in config["inputs"].items():
        payload = ("synthetic sealed opaque bytes " + label).encode()
        target = path / binding["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        binding.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    for name, value in (
        ("config.json", config),
        ("source/" + source_name, config),
        ("call_census.json", config["call_census"]),
        ("launch_receipt.json", dict(command="preparation only")),
        (
            "result.json",
            dict(
                preparation_complete=True,
                execution_complete=False,
                work=dict(
                    network_forwards=0, physics_controls=0, optimizer_steps=0, mppi_queries=0
                ),
            ),
        ),
    ):
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value))

    reseal_preparation(path)


def reseal_preparation(path):
    def digest(target):
        return hashlib.sha256(target.read_bytes()).hexdigest()

    manifest = dict(
        config_sha256=digest(path / "config.json"),
        files=[
            dict(file=str(source.relative_to(path / "source")), sha256=digest(source))
            for source in sorted((path / "source").rglob("*"))
            if source.is_file()
        ],
    )
    (path / "manifest.json").write_text(json.dumps(manifest))
    config = json.loads((path / "config.json").read_text())
    seal = dict(
        config_sha256=digest(path / "config.json"),
        source_manifest_sha256=digest(path / "manifest.json"),
        files={
            binding["file"]: dict(bytes=binding["bytes"], sha256=binding["sha256"])
            for binding in config["inputs"].values()
        },
    )
    (path / "input_manifest.json").write_text(json.dumps(seal))


def test_preparation_audit_is_opaque_hashing_and_never_decodes_or_launches(tmp_path):
    run = tmp_path / "prepared"
    write_preparation(run)
    before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    result = audit_preparation(run, tmp_path / "audit")
    assert result["preparation_audit_complete"]
    assert (
        result["array_decodes"]
        == result["new_network_rows"]
        == result["new_physics_controls"]
        == result["new_optimizer_steps"]
        == 0
    )
    assert result["launch_authorized"] is False
    assert before == {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in run.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        audit_preparation(run, tmp_path / "audit")


def test_authorized_preparation_audit_binds_the_separate_frozen_recipe(tmp_path):
    run = tmp_path / "authorized"
    write_preparation(run, recipe_name="teacher_proposal_h5_0_round1", authorized=True)
    result = audit_preparation(run, tmp_path / "audit")
    assert result["preparation_audit_complete"]
    assert result["launch_authorized"] is True
    assert result["eligibility_decision"] == "approved_single_round_exception"
    assert "source/configs/studies/teacher_proposal_h5_0_round1.json" in result["bindings"]
    assert "source/configs/studies/teacher_proposal_h5_0.json" not in result["bindings"]
    assert result["array_decodes"] == result["new_network_rows"] == 0
    assert result["new_physics_controls"] == result["new_optimizer_steps"] == 0


def test_authorized_preparation_rejects_a_different_sealed_recipe(tmp_path):
    run = tmp_path / "authorized"
    write_preparation(run, recipe_name="teacher_proposal_h5_0_round1", authorized=True)
    source = run / "source/configs/studies/teacher_proposal_h5_0_round1.json"
    changed = json.loads(source.read_text())
    changed["launch_authorized"] = False
    source.write_text(json.dumps(changed))
    reseal_preparation(run)
    with pytest.raises(AuditError, match="differs from its frozen checked-in recipe"):
        audit_preparation(run, tmp_path / "audit")


@pytest.mark.parametrize(
    "binding",
    ("../../outside", "teacher_proposal_h5_0_round2", "teacher_proposal_h5_0_round1.json"),
)
def test_authorized_preparation_rejects_unknown_recipe_binding_before_lookup(tmp_path, binding):
    run = tmp_path / "authorized"
    write_preparation(run, recipe_name="teacher_proposal_h5_0_round1", authorized=True)
    config = json.loads((run / "config.json").read_text())
    config["recipe_name"] = binding
    (run / "config.json").write_text(json.dumps(config))
    reseal_preparation(run)
    with pytest.raises(AuditError, match="Unknown H5.0 checked-in recipe binding"):
        audit_preparation(run, tmp_path / "audit")


def test_preparation_missing_binding_or_extra_source_cannot_pass(tmp_path):
    run = tmp_path / "prepared"
    write_preparation(run)
    (run / "source/injected.py").write_text("raise RuntimeError('unsealed')")
    with pytest.raises(AuditError, match="unsealed extra"):
        audit_preparation(run, tmp_path / "audit")


def test_durable_audit_journal_reserves_before_unrecorded_paid_work(tmp_path):
    journal = AuditJournal(tmp_path / "work.jsonl", max_bytes=80, max_line_bytes=80)
    journal.record(dict(event="requested", rows=1))
    before = journal.bytes
    with pytest.raises(AuditError, match="storage exhausted"):
        journal.record(dict(event="would_exceed" * 20, rows=1000))
    assert journal.bytes == before and journal.events == 1
    journal.close()
    assert json.loads((tmp_path / "work.jsonl").read_text())["rows"] == 1


def write_fake_parent_fit(run):
    """Fabricate optimizer/tape evidence; perform zero forward/backward/Adam steps."""
    import copy

    import torch

    from src.gps.mppi_gps.policy import Policy

    config = recipe()
    (run / "inputs").mkdir(parents=True)
    normalization = dict(mean=np.zeros(17, dtype=np.float32), std=np.ones(17, dtype=np.float32))
    normal_file = run / config["inputs"]["normalization"]["file"]
    np.savez(normal_file, **normalization)
    config["inputs"]["normalization"].update(
        bytes=normal_file.stat().st_size,
        sha256=hashlib.sha256(normal_file.read_bytes()).hexdigest(),
    )
    data = dict(observations_sha256="1" * 64, actions_sha256="2" * 64, rows=4000)
    config["inputs"]["train"]["fit_data"] = data
    incoming = Policy(config["streams"][0]["policy_seed"], normalization=normalization)
    child = copy.deepcopy(incoming)
    with torch.no_grad():
        child.network[-2].bias.add_(0.01)
    parent_hash, child_hash = incoming.hash(), child.hash()
    optimizer = torch.optim.Adam(child.parameters(), lr=0.0003, foreach=False, fused=False)
    adam = optimizer.state_dict()
    initial_adam = copy.deepcopy(adam)
    for index, parameter in enumerate(child.parameters()):
        adam["state"][index] = dict(
            step=torch.tensor(2000.0),
            exp_avg=torch.zeros_like(parameter),
            exp_avg_sq=torch.zeros_like(parameter),
        )
    generator = np.random.Generator(np.random.PCG64(config["streams"][0]["fit_seed"]))
    initial_pcg64 = copy.deepcopy(generator.bit_generator.state)
    tape = np.stack([generator.integers(0, 4000, size=256) for _ in range(2000)])
    fit_config = dict(
        batch_size=256,
        lr=0.0003,
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="mean",
        anchor_refresh=0,
    )
    state = dict(
        schema="walker-adam-proximal-fit-v1",
        adam=adam,
        rng=generator.bit_generator.state,
        updates=2000,
        fit_updates=2000,
        policy_hash=child_hash,
        parent_policy_hash=parent_hash,
        anchor_policy_hash=parent_hash,
        anchor=copy.deepcopy(incoming.state_dict()),
        parameter_names=[name for name, _ in incoming.named_parameters()],
        fit_config=fit_config,
        fit_data=data,
    )
    m = BinaryEvidenceMeter(
        run / "evidence", max_controls=1, max_neural_rows=1, max_bytes=64 * 1024**2
    )
    rng_ref = m.save_arrays(
        "global",
        state=torch.Generator(device="cpu")
        .manual_seed(config["learning"]["global_torch_seed"])
        .get_state()
        .numpy(),
    )
    tape_ref = m.save_arrays("tape", indices=tape)
    global_pointer = dict(reference=rng_ref, key="state")
    descriptors = []
    for actor, optimizer_state, name, fit_file, fit_tape in (
        (incoming, None, "s0_parent_initial", None, None),
        (child, state, "s0_parent", "learning/s0_parent.json", tape_ref),
    ):
        if optimizer_state is not None:
            m.record(
                "fit_initial_state",
                tag=name,
                parent_hash=parent_hash,
                policy_state_sha256=optimizer_fingerprint(incoming.state_dict()),
                adam_sha256=optimizer_fingerprint(initial_adam),
                rng=initial_pcg64,
                private_rng_sha256=optimizer_fingerprint(incoming._generator.get_state()),
                expected_tape=tape_ref,
            )
        point = run / "checkpoints" / (name + ".pt")
        descriptor = actor.checkpoint(
            point, optimizer_state=optimizer_state, metadata=dict(name=name)
        )
        binding = dict(
            actor_hash=actor.hash(),
            file=str(point.relative_to(run)),
            bytes=descriptor["bytes"],
            sha256=descriptor["sha256"],
            global_rng=global_pointer,
            optimizer_sha256=optimizer_fingerprint(optimizer_state),
            fitting_evidence=fit_file,
            fit_tape=fit_tape,
        )
        m.record("checkpoint", **binding)
        descriptors.append(binding)
    curve = [
        dict(
            update=index + 1,
            fit_update=index + 1,
            action_mse=0.0,
            proximal_loss=0.0,
            total_loss=0.0,
            unclipped_gradient_norm=0.0,
        )
        for index in range(2000)
    ]
    evidence = dict(
        fit_config=fit_config,
        fit_data=data,
        child_policy_hash=child_hash,
        incoming_optimizer_sha256=optimizer_fingerprint(None),
        initial_adam_sha256=optimizer_fingerprint(initial_adam),
        initial_private_rng_sha256=optimizer_fingerprint(incoming._generator.get_state()),
        initial_policy_state_sha256=optimizer_fingerprint(incoming.state_dict()),
        sampled_indices=tape_ref,
        resume_fit=False,
        completed_updates=2000,
        anchor_refreshes=[],
        expected_terminal_rng=state["rng"],
        learning_curve=curve,
        work={
            key: dict(requested=count, completed=count, unknown_partial=0)
            for key, count in (
                ("neural_rows", 512000),
                ("backward_rows", 512000),
                ("optimizer_steps", 2000),
            )
        },
    )
    (run / "learning").mkdir()
    (run / "learning/s0_parent.json").write_text(json.dumps(evidence))
    m.close()
    registry = {row["actor_hash"]: row for row in descriptors}
    return config, registry, curve, tape


def test_fit_owner_typed_adam_rng_anchor_and_actual_tape_checks_without_training(tmp_path):
    from src.gps.mppi_gps.audits.teacher_proposal import verify_fitting_ownership

    run = tmp_path / "synthetic"
    config, registry, curve, tape = write_fake_parent_fit(run)
    with archive(run / "evidence", tmp_path / "audit") as reader:
        reader.verify_index()
        verifier = SavedActorVerifier(registry, run, max_neural_rows=0)
        checker = RoundChecks(reader, config, verifier, run=run)
        audit_stream_structure(reader, event_observer=checker.event)
        # This unit fixture tests ownership arithmetic, not a full round. The
        # structural ledger/native-loop tests separately verify actual boundaries.
        for row, indices in zip(curve, tape, strict=True):
            checker.event(
                dict(event="fit_update", tag="s0_parent", indices=indices.tolist(), **row)
            )
        checker.fit_measurements["s0_parent"] = dict(
            neural_rows=512000, backward_rows=512000, optimizer_steps=2000
        )
        result = verify_fitting_ownership(verifier, checker, config)
        assert result["complete_fit_blocks"] == 1
        assert result["replayed_minibatch_indices"] == 512000
        assert result["training_forward_rows"] == result["adam_updates_reproduced"] == 0
        checker.fit_updates["s0_parent"][0]["indices"][0] ^= 1
        with pytest.raises(AuditError, match="Actual fit minibatch"):
            verify_fitting_ownership(verifier, checker, config)
        checker.fit_updates["s0_parent"][0]["indices"][0] ^= 1
        checker.fit_initial_states["s0_parent"]["policy_state_sha256"] = "0" * 64
        with pytest.raises(AuditError, match="different policy tensor tree"):
            verify_fitting_ownership(verifier, checker, config)


def test_numerical_audit_worker_cannot_bypass_live_parent_ownership(tmp_path, monkeypatch):
    import os

    output = tmp_path / "audit"
    output.mkdir()
    (output / "ownership.json").write_text(
        json.dumps(dict(coordinator_pid=os.getppid() + 99999, nonce="synthetic"))
    )
    monkeypatch.setenv("H5_AUDIT_NONCE", "synthetic")
    with pytest.raises(AuditError, match="authenticated coordinator"):
        audit_worker(tmp_path / "absent_run", output)


def test_prediction_deadline_and_journal_reserve_before_checkpoint_or_network(tmp_path):
    import time

    verifier = SavedActorVerifier({}, tmp_path, max_neural_rows=1, deadline=time.monotonic() - 1)
    with pytest.raises(AuditError, match="wall-time"):
        verifier.predict("0" * 64, np.zeros(17))
    assert verifier.neural_rows == verifier.checkpoint_loads == verifier.prediction_calls == 0
    journal = AuditJournal(tmp_path / "work.jsonl", max_bytes=0)
    verifier = SavedActorVerifier({}, tmp_path, max_neural_rows=1, event_sink=journal.record)
    with pytest.raises(AuditError, match="journal storage exhausted"):
        verifier.predict("0" * 64, np.zeros(17))
    assert verifier.neural_rows == verifier.checkpoint_loads == verifier.prediction_calls == 0
    journal.close()


@pytest.mark.parametrize("failure", [None, "acquisition", "gap_cost"])
def test_independent_roster_gate_covers_teacher_negative_and_all_gap_teachers(failure):
    from types import SimpleNamespace

    from src.gps.mppi_gps.audits.teacher_proposal import verify_outcome_roster
    from src.gps.mppi_gps.proposal_round import run_round

    config = recipe()
    episodes = {}

    def episode(tag, seed, success=True, survived=True, steps=1000, cost=10.0):
        row = dict(
            tag=tag,
            seed=seed,
            success=success,
            survived=survived,
            steps=steps,
            cost=cost,
            complete=True,
            raw=dict(tag=tag),
        )
        episodes[tag] = row
        return SimpleNamespace(**row)

    class Operations:
        def parent(self, stream):
            return stream["id"], "parent"

        def acquire(self, parent, stream, arm, index):
            fail = (
                failure == "acquisition"
                and stream["id"] == 0
                and arm == "independent"
                and index == 0
            )
            return episode(
                f"teacher_s{stream['id']}_{arm}_{index:02d}",
                stream["acquisition_seeds"][index],
                not fail,
                not fail,
                3 if fail else 1000,
                5010.0 if fail else 10.0,
            )

        def fit(self, parent, rows, stream, arm):
            return stream["id"], arm

        def evaluate(self, actor, seed, tag):
            return episode(tag, seed, actor[1] == "proposal_guided" and seed % 1000 < 2)

        def gap_teacher(self, parent, stream, arm, index):
            cost = 10.7 if failure == "gap_cost" and arm == "proposal_guided" else 10.0
            return episode(
                f"gap_teacher_s{stream['id']}_{arm}_{index:02d}",
                stream["gap_seeds"][index],
                cost=cost,
            )

        def paired(self, first, second):
            assert first.seed == second.seed

        def record(self, event, **fields):
            pass

    class InitialContext:
        @staticmethod
        def array(ref, key):
            width = dict(states=19, warmstarts=9, integrations=64, observations=17)[key]
            return np.zeros((1, width))

    result = run_round(config, Operations())
    negative = failure == "acquisition"
    checkpoints, metrics = [], {}
    for sid in range(3):
        for suffix in ("parent_initial", "parent"):
            point = f"s{sid}_{suffix}"
            checkpoints.append(dict(file=f"checkpoints/{point}.pt", actor_hash="0" * 64))
            for label in ("train", "validation"):
                filename = config["inputs"][label]["file"]
                metrics[point + "/" + label] = dict(
                    actor_hash="0" * 64,
                    observations=dict(file=filename, key="observations"),
                    targets=dict(file=filename, key="actions"),
                )
        if not negative:
            for arm in config["arms"]:
                point = f"s{sid}_{arm}_child"
                checkpoints.append(dict(file=f"checkpoints/{point}.pt", actor_hash="0" * 64))
                for suffix in ("incoming", "endpoint"):
                    metrics[point + "/" + suffix] = dict(
                        actor_hash="0" * 64,
                        observations=dict(reference=dict(marker=point), key="observations"),
                        targets=dict(reference=dict(marker=point), key="actions"),
                    )
                for label in ("train", "validation"):
                    filename = config["inputs"][label]["file"]
                    metrics[point + "/" + label] = dict(
                        actor_hash="0" * 64,
                        observations=dict(file=filename, key="observations"),
                        targets=dict(file=filename, key="actions"),
                    )
    teacher_tags = {
        tag for tag in episodes if tag.startswith("teacher_") or tag.startswith("gap_teacher_")
    }
    teacher_controls = sum(episodes[tag]["steps"] for tag in teacher_tags)
    checker = SimpleNamespace(
        recipe=config,
        episodes=episodes,
        archive=InitialContext(),
        checkpoints=checkpoints,
        fit_updates=dict.fromkeys(range(3 if negative else 12)),
        metric_datasets=metrics,
        teacher_posthoc_tags=teacher_tags,
        selection_checks=teacher_controls,
        teacher_controls=teacher_controls,
    )
    verified = verify_outcome_roster(checker, result)
    assert verified["confirmed_benefit"] is (failure is None)
    assert verified["standalone_panels_measured"] is (not negative)
    assert len(episodes) == (36 if negative else 456)
    assert verified["loop_closed"] is False
    if failure == "gap_cost":
        assert result["primary_gate"]["passed"] and result["confirmation_gate"]["passed"]
        assert not result["gap_teacher_gate"]


@pytest.mark.parametrize("solver", ["warmstart", "integration"])
def test_real_terminal_solver_rows_cannot_be_replaced_by_another_snapshot(tmp_path, solver):
    root = tmp_path / "evidence"
    write_short_episode(root, "actor_only", bad_solver=solver)
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        checker = RoundChecks(reader, recipe(), FakePredictor(), run=tmp_path)
        with pytest.raises(AuditError, match="solver context differs"):
            audit_stream_structure(reader, event_observer=checker.event)


def test_rng_state_advance_alone_cannot_certify_the_actual_proposal_distribution(tmp_path):
    root = tmp_path / "evidence"
    write_decision(root, "I", altered_noise=True)
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        checker = RoundChecks(reader, recipe(), FakePredictor(), run=tmp_path)
        with pytest.raises(AuditError, match="Protected samples differ"):
            audit_stream_structure(reader, event_observer=checker.event)


@pytest.mark.parametrize("disturbed", [False, True])
def test_fresh_seed_and_every_next_private_nominal_are_bound_across_decisions(tmp_path, disturbed):
    root = tmp_path / "evidence"
    write_decision(root, "I", steps=2, real_context=True, disturb_before_second=disturbed)
    with archive(root, tmp_path / "audit") as reader:
        reader.verify_index()
        checker = RoundChecks(reader, recipe(), FakePredictor(), run=tmp_path)
        if disturbed:
            with pytest.raises(AuditError, match="previous exact after"):
                audit_stream_structure(reader, event_observer=checker.event)
        else:
            audit_stream_structure(reader, event_observer=checker.event)
            assert checker.summary()["selection_checks"] == 2
            assert (
                checker.summary()["independently_recreated_standard_normal_draws"]
                == 2 * 3 * 256 * 48 * 6
            )
