"""Pure numerical/checker fixtures: no actual network, simulator or fitting.

The predictor below is a NumPy stub. Its Work rows exercise reservation/census
semantics only; they are not actual neural forwards or research physics.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import local_coupling_numeric_validation as numeric
from src.gps.mppi_gps.local_coupling_controller import local_outcome
from src.gps.mppi_gps.provenance import write
from src.gps.mppi_gps.trajectory_pi_accounting import Work
from src.gps.mppi_gps.trajectory_pi_native import binding
from src.gps.mppi_gps.trajectory_pi_student import observations


def limits(**changes):
    return dict(
        wall_seconds=60,
        output_bytes=64 * 1024**2,
        memory_bytes=6 * 1024**3,
        neural_rows=100_000,
        checkpoint_loads=10,
        policy_constructions=10,
        native_controls=0,
        model_controls=0,
        optimizer_steps=0,
        **changes,
    )


class GroupedPredictor:
    obs_dim, hidden_dim = 17, 128

    def __init__(self, *, group_error=0.0, invalid=False):
        self.group_error = group_error
        self.invalid = invalid
        self.called_rows = []

    def hash(self):
        return "fixture-actor"

    def eval(self):
        return self

    def predict(self, inputs, *, meter):
        n = len(inputs)

        def operation():
            self.called_rows.append(n)
            value = np.zeros((n, 6), dtype=np.float64)
            if n <= 48:
                value += 1.1e-6
            else:
                value += self.group_error
            if self.invalid:
                value[0, 0] = np.nan
            return value

        return meter.measure("neural_rows", n, operation)


def save_trace(path, raw):
    with h5py.File(path, "x") as target:
        for key, value in raw.items():
            target.create_dataset(key, data=value)


@pytest.fixture
def recorded(tmp_path):
    initial = np.zeros(19, dtype=np.float64)
    initial[2], initial[10] = 1.25, 1.5
    posts = np.broadcast_to(initial, (256, 48, 19)).copy()
    posts[0, 1:, 2] = 0.7
    controls = np.full((256, 48, 6), 0.2, dtype=np.float64)
    mask = np.ones((256, 48), dtype=bool)
    mask[0, 2:] = False
    cost = np.full(256, 48.0)
    cost[0] = 252.0
    predictions = np.zeros((int(mask.sum()), 6), dtype=np.float64)
    discrepancy = np.sum(0.5 * controls**2, axis=-1)
    discrepancy = np.sum(np.where(mask, discrepancy, 0), axis=1)
    bank = tmp_path / "bank.npz"
    np.savez_compressed(
        bank,
        actions=controls,
        posts=posts,
        mask=mask,
        C=cost,
        D=discrepancy,
        predictions=predictions,
        initial_state=initial,
        initial_warmstart=np.zeros(9, dtype=np.float64),
        initial_integration=np.zeros(91, dtype=np.float64),
    )
    raw = {}
    replays = {}
    for candidate in numeric.CANDIDATES:
        n = int(mask[candidate].sum())
        states = np.concatenate((initial[None], posts[candidate, :n]))
        costs = np.ones(n)
        fall_cost = np.zeros(n)
        if n < 48:
            costs[-1] += 20
            fall_cost[-1] = 20
        trace = dict(
            states=states,
            warmstarts=np.zeros((n + 1, 9), dtype=np.float64),
            integrations=np.zeros((n + 1, 91), dtype=np.float64),
            observations=observations(states),
            actions=controls[candidate, :n],
            costs=costs,
            native_fall_cost=fall_cost,
        )
        path = tmp_path / f"native_{candidate}.h5"
        save_trace(path, trace)
        raw[candidate], replays[candidate] = trace, path
    return dict(
        bank=bank,
        source=binding(bank),
        raw=raw,
        replays=replays,
        directory=tmp_path,
        controls=controls,
        posts=posts,
        cost=cost,
        discrepancy=discrepancy,
        group_rows=int(mask.sum()),
    )


def compare(recorded, tmp_path, actor=None, work_limits=None):
    directory = tmp_path / "verification"
    directory.mkdir()
    work = Work(directory, work_limits or limits())
    actor = actor or GroupedPredictor()
    result = numeric.compare_saved_candidate(
        source=recorded["source"],
        candidate=0,
        replay_path=recorded["replays"][0],
        predictions_path=directory / "predictions.npz",
        actor=actor,
        work=work,
    )
    return result, work, actor, directory


def test_matching_group_passes_without_assuming_cross_group_bit_invariance(recorded, tmp_path):
    result, work, actor, directory = compare(recorded, tmp_path)
    assert result["comparison_version"] == "MATCHED_GROUP_L3_1"
    assert result["action_max_error"] > 1e-6
    assert result["cross_group_action_max_error"] == result["action_max_error"]
    assert result["matched_action_max_error"] == 0
    assert result["D"] != result["matched_D"]
    assert result["matched_D"] == pytest.approx(recorded["discrepancy"][0], abs=1e-15)
    assert actor.called_rows == [2, recorded["group_rows"]]
    assert work.counts["completed_neural_rows"] == 2 + recorded["group_rows"]
    assert work.counts.get("requested_native_controls", 0) == 0
    with np.load(directory / "predictions.npz", allow_pickle=False) as proof:
        assert proof["matching_group_predictions"].shape == (recorded["group_rows"], 6)
        assert proof["replacement_offset"] == 0
        assert proof["source_sha256"] == recorded["source"]["sha256"]
        assert proof["matching_group_observation_fingerprint"].item()
    work.close("COMPLETED")


def test_actual_matched_prediction_error_stops_and_preserves_paid_outputs(recorded, tmp_path):
    directory = tmp_path / "bad_prediction"
    directory.mkdir()
    work = Work(directory, limits())
    actor = GroupedPredictor(group_error=1e-3)
    with pytest.raises(ValueError, match="matched-group deployed reconstruction mismatch"):
        numeric.compare_saved_candidate(
            source=recorded["source"],
            candidate=0,
            replay_path=recorded["replays"][0],
            predictions_path=directory / "predictions.npz",
            actor=actor,
            work=work,
        )
    with np.load(directory / "predictions.npz", allow_pickle=False) as proof:
        assert proof["matched_action_max_error"] == 1e-3
    assert work.counts["completed_neural_rows"] == 2 + recorded["group_rows"]
    work.close("FAILED")


@pytest.mark.parametrize("field", ("states", "costs", "observations", "actions", "warmstarts"))
def test_physics_cost_preaction_or_context_mismatch_stops_before_inference(
    recorded, tmp_path, field
):
    raw = {k: v.copy() for k, v in recorded["raw"][0].items()}
    if field == "states":
        raw[field][1, 4] += 1e-5
        raw["observations"] = observations(raw["states"])
    elif field == "warmstarts":
        raw[field][0, 0] = 1
    else:
        raw[field].flat[0] += 0.1
    bad = tmp_path / "corrupt.h5"
    save_trace(bad, raw)
    directory = tmp_path / "blocked"
    directory.mkdir()
    work = Work(directory, limits())
    actor = GroupedPredictor()
    with pytest.raises(ValueError):
        numeric.compare_saved_candidate(
            source=recorded["source"],
            candidate=0,
            replay_path=bad,
            predictions_path=directory / "predictions.npz",
            actor=actor,
            work=work,
        )
    assert actor.called_rows == []
    assert work.counts.get("requested_neural_rows", 0) == 0
    work.close("FAILED")


def test_matching_extra_rows_are_reserved_before_the_second_actor_call(recorded, tmp_path):
    directory = tmp_path / "limited"
    directory.mkdir()
    constrained = limits()
    constrained["neural_rows"] = 2
    work = Work(directory, constrained)
    actor = GroupedPredictor()
    with pytest.raises(RuntimeError, match="Work budget exhausted: neural_rows"):
        numeric.compare_saved_candidate(
            source=recorded["source"],
            candidate=0,
            replay_path=recorded["replays"][0],
            predictions_path=directory / "predictions.npz",
            actor=actor,
            work=work,
        )
    assert actor.called_rows == [2]
    assert work.counts["completed_neural_rows"] == 2
    work.close("FAILED")


def test_invalid_actor_value_remains_invalid_execution(recorded, tmp_path):
    directory = tmp_path / "invalid_actor"
    directory.mkdir()
    work = Work(directory, limits())
    actor = GroupedPredictor(invalid=True)
    with pytest.raises(ValueError, match="Invalid bounded deployed actor output"):
        numeric.compare_saved_candidate(
            source=recorded["source"],
            candidate=0,
            replay_path=recorded["replays"][0],
            predictions_path=directory / "predictions.npz",
            actor=actor,
            work=work,
        )
    assert (directory / "predictions.npz.invalid_prefix.npz").is_file()
    assert work.counts["completed_neural_rows"] == 2
    work.close("FAILED")


def test_future_verifier_runs_only_the_original_native_replay_roster(
    recorded, tmp_path, monkeypatch
):
    calls = []
    directory = tmp_path / "future"
    directory.mkdir()
    work = Work(directory, limits())
    actor = GroupedPredictor()
    monkeypatch.setattr(numeric, "_restore", lambda *args: None)

    def execute(env, meter, path, horizon, actions):
        candidate = numeric.CANDIDATES[len(calls)]
        assert horizon == 48
        assert np.array_equal(actions, recorded["controls"][candidate])
        save_trace(path, recorded["raw"][candidate])
        calls.append(candidate)
        return {}

    monkeypatch.setattr(numeric, "execute_trace", execute)
    result = numeric.verify_native(
        object(),
        work,
        directory,
        [
            dict(
                source=recorded["source"],
                context={},
                C=recorded["cost"][[0, 255]],
                actions=recorded["controls"][[0, 255]],
            )
        ],
        actor,
    )
    assert calls == [0, 255]
    assert len(result) == 2
    assert all(row["comparison_version"] == numeric.COMPARISON_VERSION for row in result)
    assert work.counts["completed_neural_rows"] == 2 + 48 + 2 * recorded["group_rows"]
    work.close("COMPLETED")


def original_failure(recorded, tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    (original / "policy.pt").write_bytes(b"nonexecuted fixture checkpoint")
    write(original / "config.json", dict(schema="local-coupling-learning-l3-1"))
    checkpoint, config = binding(original / "policy.pt"), binding(original / "config.json")
    (original / "full_t000_p0.npz").write_bytes(recorded["bank"].read_bytes())
    for candidate in numeric.CANDIDATES:
        (original / f"candidate_replay_0_{candidate}.h5").write_bytes(
            recorded["replays"][candidate].read_bytes()
        )
    initial = recorded["raw"][255]["states"][0]
    states = np.broadcast_to(initial, (301, 19)).copy()
    raw = dict(
        states=states,
        observations=observations(states),
        warmstarts=np.zeros((301, 9)),
        integrations=np.zeros((301, 91)),
        actions=np.full((300, 6), 0.2),
        costs=np.ones(300),
        native_fall_cost=np.zeros(300),
    )
    save_trace(original / "trace.h5", raw)
    with h5py.File(original / "banks.h5", "x"):
        pass
    counts = dict(requested_native_controls=349, completed_native_controls=349)
    write(original / "receipt.json", dict(status="FAILED", counts=counts))
    (original / "ledger.jsonl").write_text('{"event":"fixture_only"}\n')
    write(original / "job.json", dict(out=str(original), checkpoint=checkpoint, config=config))
    manifest = dict(
        schema="local-coupling-episode-1",
        status="FAILED_STOPPED",
        round=4,
        stream=0,
        arm="independent",
        slot=2,
        checkpoint=checkpoint,
        actor_hash="fixture-actor",
        error="ValueError('Candidate/native deployed reconstruction mismatch')",
        outcome=local_outcome(states, raw["costs"], 300, False),
        work=counts,
    )
    write(original / "manifest.json", manifest)
    return original, dict(
        out=str(tmp_path / "completion"),
        original_manifest=binding(original / "manifest.json"),
        checkpoint=checkpoint,
        config=config,
        limits=limits(),
    )


def test_certificate_reuses_completed_physics_preserves_failure_and_separates_counts(
    recorded, tmp_path, monkeypatch
):
    original, job = original_failure(recorded, tmp_path)
    before = {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()}
    monkeypatch.setattr(numeric, "_load_actor", lambda path: GroupedPredictor())
    value = numeric.certify_existing(job)
    assert value["result"]["status"] == "COMPLETED", value["result"].get("error")
    assert {p.name: p.read_bytes() for p in original.iterdir() if p.is_file()} == before
    adopted = json.loads(Path(value["result"]["adopted_manifest"]["path"]).read_text())
    assert adopted["status"] == "COMPLETED"
    assert adopted["work"]["completed_native_controls"] == 349
    assert adopted["original_validation_failure"] == job["original_manifest"]
    assert json.loads((original / "manifest.json").read_text())["status"] == "FAILED_STOPPED"
    receipt = value["receipt"]["counts"]
    assert receipt.get("completed_native_controls", 0) == 0
    assert receipt["completed_neural_rows"] == 2 + 48 + 2 * recorded["group_rows"]
    assert receipt["completed_checkpoint_loads"] == 1
    assert receipt["completed_policy_constructions"] == 1
    assert adopted["validation_completion"] == value["result"]["completion"]


@pytest.mark.parametrize("bad", ("physics", "unknown"))
def test_certificate_cannot_adopt_a_physics_failure_or_unknown_work(
    recorded, tmp_path, monkeypatch, bad
):
    original, job = original_failure(recorded, tmp_path)
    manifest = json.loads((original / "manifest.json").read_text())
    if bad == "physics":
        manifest["error"] = "ValueError('Candidate/native state replay mismatch')"
    else:
        receipt_path = original / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["counts"]["unknown_partial_neural_rows"] = 1
        manifest["work"] = receipt["counts"]
        receipt_path.write_text(json.dumps(receipt))
    (original / "manifest.json").write_text(json.dumps(manifest))
    job["original_manifest"] = binding(original / "manifest.json")
    monkeypatch.setattr(numeric, "_load_actor", lambda path: pytest.fail("Actor must not load"))
    value = numeric.certify_existing(job)
    assert value["result"]["status"] == "FAILED_STOPPED"
    assert value["receipt"]["counts"] == {}
    assert not (Path(job["out"]) / "adopted_manifest.json").exists()
