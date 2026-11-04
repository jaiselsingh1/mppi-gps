"""Saved-array-only P1.1 native input import; no simulator or checkpoint loading."""

from __future__ import annotations

import builtins
import copy
import json
import random
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.gps.mppi_gps import trajectory_pi_inputs as inputs
from src.gps.mppi_gps.provenance import fingerprint, write


def dataset():
    states = np.zeros((1001, 19), dtype=np.float64)
    states[:, 0] = np.arange(1001) * 0.008
    states[:, 2], states[:, 10] = 1.2, 1.5
    return dict(
        states=states,
        warmstarts=np.zeros((1001, 9)),
        integrations=np.zeros((1001, 91)),
        observations=inputs.observations(states),
        actions=np.full((1000, 6), 0.1),
        costs=np.ones(1000),
    )


def npz(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def h5(path, values):
    with h5py.File(path, "x") as target:
        for name, value in values.items():
            target[name] = value


@pytest.fixture
def frozen(tmp_path):
    source = tmp_path / "source"
    configs = source / "configs/studies"
    configs.mkdir(parents=True)
    prior = tmp_path / "prior/prepared"
    prior.mkdir(parents=True)
    recipe_path = Path(__file__).resolve().parents[1] / "configs/studies/trajectory_pi_p1.json"
    baseline = json.loads(recipe_path.read_text())
    parents, teachers, episodes = [], [], []
    rv = tmp_path / "validation"
    rv.mkdir()
    for spec in baseline["parents"]:
        directory = prior / spec["prepared_directory"]
        directory.mkdir(parents=True)
        for name in ("policy.pt", "global_rng.pt", "stream.json"):
            (directory / name).write_bytes(f"opaque existing {spec['stream']} {name}".encode())
        spec["checkpoint"] = inputs.binding(directory / "policy.pt")
        parents.append(
            dict(
                stream=spec["stream"],
                directory=spec["prepared_directory"],
                source=spec["checkpoint"],
                policy_hash=f"policy{spec['stream']}",
                optimizer_fingerprint=f"adam{spec['stream']}",
                global_rng_fingerprint=f"global{spec['stream']}",
            )
        )
    for spec in baseline["teachers"]:
        slot = spec["slot"]
        old = dataset()
        original = {
            name: old[name][:-1]
            for name in ("states", "warmstarts", "integrations", "observations")
        }
        original.update(
            actions=old["actions"],
            costs=old["costs"],
            terminal_state=old["states"][-1],
            terminal_warmstarts=old["warmstarts"][-1],
            terminal_integrations=old["integrations"][-1],
        )
        original_path = prior / "originals" / f"teacher_{slot}.npz"
        npz(original_path, original)
        spec["episode"] = inputs.binding(original_path)
        canonical = dict(old, observations=old["observations"][:-1])
        npz(prior / spec["prepared_path"], canonical)
        teachers.append(
            dict(slot=slot, file_relative=spec["prepared_path"], source=spec["episode"])
        )
        native = dataset()
        native["states"][1:, 4] += (slot + 1) * 1e-6
        native["costs"] += (slot + 1) * 0.01
        native["observations"] = inputs.observations(native["states"])
        capture, replay, candidate = (
            rv / f"capture{slot}.h5",
            rv / f"replay{slot}.h5",
            rv / f"candidate{slot}.npz",
        )
        h5(capture, native)
        h5(replay, native)
        npz(
            candidate,
            dict(states=native["states"][None, 1:], costs=np.array([native["costs"].sum()])),
        )
        outcome = dict(
            controls=1000,
            walking=True,
            full_survival=True,
            complete_cost=float(native["costs"].sum()),
            velocity_mae=0.0,
        )
        episodes.append(
            dict(
                slot=slot,
                original_episode=spec["episode"],
                original_prepared=inputs.binding(prior / spec["prepared_path"]),
                capture=inputs.binding(capture),
                replay=inputs.binding(replay),
                candidate=inputs.binding(candidate),
                outcome=outcome,
                native_replay_passed=True,
                single_candidate_passed=True,
                initial_context_exact=True,
                actions_exact=True,
            )
        )
    previous_config = configs / "trajectory_pi_p1.json"
    write(previous_config, baseline)
    config = copy.deepcopy(baseline)
    config.update(version="P1.1", input_mode="worker-native", launch_ready=True)
    config_path = configs / "trajectory_pi_p1_1.json"
    write(config_path, config)
    native_config = configs / "trajectory_pi_native_baselines_p1_1.json"
    write(native_config, dict(schema="trajectory-pi-native-baselines-1"))
    write(tmp_path / "source_manifest.json", dict(schema="metadata-only fixture"))
    roster = [
        dict(inputs.binding(path), path=str(path.relative_to(prior)))
        for path in sorted(prior.rglob("*"))
        if path.is_file()
    ]
    write(
        prior / "manifest.json",
        dict(
            schema="trajectory-pi-prepared-1",
            config_sha256=inputs.binding(previous_config)["sha256"],
            normalization_inherited=True,
            parent_refits=0,
            parents=parents,
            teachers=teachers,
            files=roster,
        ),
    )
    origin = {}
    for name, value in dict(
        result=dict(status="ORIGIN_REPLAY_PASS", work=dict(completed_native_controls=1000)),
        audit_receipt=dict(passed=True, origin_replay_passed=True),
        audit_findings=dict(origin_replay_passed=True),
        external_summary=dict(exit_code=0),
    ).items():
        path = rv / ("origin_" + name + ".json")
        write(path, value)
        origin[name] = inputs.binding(path)
    origin["origin_replay_passed"] = True
    write(rv / "result.json", dict(status="NATIVE_BASELINES_PASS"))
    write(
        rv / "receipt.json",
        dict(
            counts=dict(
                completed_native_controls=4000,
                completed_model_controls=2000,
                completed_physics_substeps=24000,
            )
        ),
    )
    proof = dict(
        schema="trajectory-pi-native-validation-1",
        status="PASS",
        config_sha256=inputs.binding(native_config)["sha256"],
        p1_config_sha256=inputs.binding(config_path)["sha256"],
        source_manifest_sha256=inputs.binding(tmp_path / "source_manifest.json")["sha256"],
        previous_prepared_manifest=inputs.binding(prior / "manifest.json"),
        origin_replay=origin,
        result=inputs.binding(rv / "result.json"),
        receipt=inputs.binding(rv / "receipt.json"),
        episodes=episodes,
    )
    findings = dict(
        schema="trajectory-pi-native-baselines-audit-findings-1",
        native_validation_passed=True,
        result_binding=proof["result"],
        **{
            key: copy.deepcopy(proof[key])
            for key in (
                "config_sha256",
                "p1_config_sha256",
                "source_manifest_sha256",
                "previous_prepared_manifest",
                "origin_replay",
                "episodes",
            )
        },
    )
    write(rv / "findings.json", findings)
    write(
        rv / "audit_receipt.json",
        dict(
            passed=True,
            native_validation_passed=True,
            findings_binding=inputs.binding(rv / "findings.json"),
        ),
    )
    proof["native_audit"] = dict(
        receipt=inputs.binding(rv / "audit_receipt.json"),
        findings=inputs.binding(rv / "findings.json"),
    )
    write(rv / "validation_manifest.json", proof)
    return dict(
        config=config_path,
        previous=prior,
        proof=rv / "validation_manifest.json",
        output=tmp_path / "prepared",
    )


def prepare(frozen):
    return inputs.prepare_native(
        frozen["config"], frozen["output"], frozen["proof"], frozen["previous"]
    )


def test_native_import_preserves_all_parent_bytes_globals_and_original_episodes_without_torch(
    frozen, monkeypatch
):
    before_python, before_numpy = random.getstate(), np.random.get_state()
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "mujoco"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = prepare(frozen)
    assert result["version"] == "P1.1" and result["input_mode"] == "worker-native"
    assert result["new_global_envelopes_created"] == 0
    assert result["preparation_work"]["checkpoint_loads"] == 0
    assert result["preparation_work"]["native_controls"] == 0
    assert fingerprint(random.getstate()) == fingerprint(before_python)
    assert fingerprint(np.random.get_state()) == fingerprint(before_numpy)
    assert result["imported_work"][1]["phase"] == "rv"
    assert result["imported_work"][1]["counts"]["completed_native_controls"] == 4000
    for parent in result["parents"]:
        for name in ("policy.pt", "global_rng.pt", "stream.json"):
            relative = Path(parent["directory"]) / name
            assert (frozen["output"] / relative).read_bytes() == (
                frozen["previous"] / relative
            ).read_bytes()
    for slot in (0, 1):
        relative = Path("originals") / f"teacher_{slot}.npz"
        assert (frozen["output"] / relative).read_bytes() == (
            frozen["previous"] / relative
        ).read_bytes()
        with (
            np.load(frozen["output"] / f"teachers/slot{slot}.npz") as native,
            np.load(frozen["previous"] / f"teachers/slot{slot}.npz") as old,
        ):
            assert native["states"].shape == (1001, 19)
            assert native["observations"].shape == (1000, 17)
            assert np.array_equal(native["actions"], old["actions"])
            assert not np.array_equal(native["states"][1:], old["states"][1:])
            assert np.array_equal(
                native["observations"], inputs.observations(native["states"][:-1])
            )
    with pytest.raises(FileExistsError):
        prepare(frozen)


@pytest.mark.parametrize(
    "change",
    [
        lambda proof: proof.update(status="FAIL"),
        lambda proof: proof.update(p1_config_sha256="wrong"),
        lambda proof: proof["origin_replay"].update(origin_replay_passed=False),
        lambda proof: proof["episodes"][0].update(actions_exact=False),
        lambda proof: proof["episodes"][0]["capture"].update(sha256="wrong"),
    ],
)
def test_failed_proof_stops_before_creating_or_copying_any_output(frozen, change):
    proof = json.loads(frozen["proof"].read_text())
    change(proof)
    frozen["proof"].write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        prepare(frozen)
    assert not frozen["output"].exists()


def test_valid_audit_cannot_be_mixed_with_a_different_episode_roster(frozen):
    proof = json.loads(frozen["proof"].read_text())
    proof["episodes"][0]["candidate"] = proof["episodes"][1]["candidate"]
    frozen["proof"].write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="audit does not certify"):
        prepare(frozen)
    assert not frozen["output"].exists()


def test_fitting_or_optimizer_recipe_change_stops_before_copy(frozen):
    config = json.loads(frozen["config"].read_text())
    config["learning"]["updates"] += 1
    frozen["config"].write_text(json.dumps(config))
    with pytest.raises(ValueError, match="scientific recipe changed"):
        prepare(frozen)
    assert not frozen["output"].exists()


def test_old_global_rng_package_mutation_stops_before_copy(frozen):
    (frozen["previous"] / "initial/stream0/global_rng.pt").write_bytes(b"new seed forbidden")
    with pytest.raises(ValueError, match="input changed"):
        prepare(frozen)
    assert not frozen["output"].exists()


def test_p1_1_cannot_use_legacy_preparation_that_creates_new_global_envelopes(frozen, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No actor construction or global RNG initialization permitted")

    monkeypatch.setattr(inputs, "freeze_legacy_parent", forbidden)
    with pytest.raises(ValueError, match="requires prepare_native"):
        inputs.prepare(frozen["config"], frozen["output"])
    assert not frozen["output"].exists()


def reseal_native_audit(frozen, proof, findings):
    finding_path = Path(proof["native_audit"]["findings"]["path"])
    finding_path.write_text(json.dumps(findings))
    proof["native_audit"]["findings"] = inputs.binding(finding_path)
    receipt_path = Path(proof["native_audit"]["receipt"]["path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["findings_binding"] = proof["native_audit"]["findings"]
    receipt_path.write_text(json.dumps(receipt))
    proof["native_audit"]["receipt"] = inputs.binding(receipt_path)
    frozen["proof"].write_text(json.dumps(proof))


def test_evidence_audit_pass_does_not_override_failed_native_validation(frozen):
    proof = json.loads(frozen["proof"].read_text())
    findings = inputs._json_bound(proof["native_audit"]["findings"])
    findings["native_validation_passed"] = False
    reseal_native_audit(frozen, proof, findings)
    with pytest.raises(ValueError, match="must both pass"):
        prepare(frozen)
    assert not frozen["output"].exists()


@pytest.mark.parametrize("array,index", [("actions", (0, 0)), ("warmstarts", (0, 0))])
def test_native_capture_cannot_change_selected_torques_or_initial_complete_context(
    frozen, array, index
):
    proof = json.loads(frozen["proof"].read_text())
    episode = proof["episodes"][0]
    with h5py.File(episode["capture"]["path"], "r+") as raw:
        raw[array][index] += 0.001
    episode["capture"] = inputs.binding(Path(episode["capture"]["path"]))
    findings = inputs._json_bound(proof["native_audit"]["findings"])
    findings["episodes"][0]["capture"] = episode["capture"]
    reseal_native_audit(frozen, proof, findings)
    with pytest.raises(ValueError, match="changed original torques or complete initial context"):
        prepare(frozen)
    assert not frozen["output"].exists()


def test_p0_metadata_gate_rechecks_native_and_origin_proofs_before_physics(frozen, monkeypatch):
    manifest = prepare(frozen)
    source = frozen["config"].resolve().parents[2].parent / "source_manifest.json"
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "mujoco"}
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    inputs.validate_native_prepared(frozen["config"], frozen["output"], manifest, source)
    proof = json.loads(frozen["proof"].read_text())
    proof["origin_replay"]["origin_replay_passed"] = False
    frozen["proof"].write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="input changed"):
        inputs.validate_native_prepared(frozen["config"], frozen["output"], manifest, source)


def test_p0_metadata_gate_rejects_new_global_envelope_even_with_updated_output_roster(frozen):
    manifest = prepare(frozen)
    source = frozen["config"].resolve().parents[2].parent / "source_manifest.json"
    path = frozen["output"] / "initial/stream0/global_rng.pt"
    path.write_bytes(b"different global stream")
    with pytest.raises(ValueError, match="differs from exact P1.0 continuation"):
        inputs.validate_native_prepared(frozen["config"], frozen["output"], manifest, source)


def test_p0_metadata_gate_rejects_omitted_antecedent_accounting(frozen):
    manifest = prepare(frozen)
    manifest["imported_work"] = []
    source = frozen["config"].resolve().parents[2].parent / "source_manifest.json"
    with pytest.raises(ValueError, match="Antecedent R0/RV accounting"):
        inputs.validate_native_prepared(frozen["config"], frozen["output"], manifest, source)
