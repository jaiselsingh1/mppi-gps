"""Create-only frozen inputs for the bounded full-trajectory PI comparison.

No teacher acquisition, policy forward or fitting occurs here. Recorded parent
packages are copied exactly; the historical Adam/private RNG states are retained.
New-study global RNG envelopes are declared separately from unavailable history.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .provenance import sha, write
from .trajectory_pi_student import freeze_legacy_parent, observations


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _bound(spec: dict) -> Path:
    path = Path(spec["path"])
    _need(path.is_absolute() and path.is_file(), "Required bound input unavailable")
    _need(
        type(spec["bytes"]) is int
        and path.stat().st_size == spec["bytes"]
        and sha(path) == spec["sha256"],
        "Bound native-preparation input changed",
    )
    return path


def _json_bound(spec: dict) -> dict:
    return json.loads(_bound(spec).read_text())


def _prior_files(directory: Path, manifest: dict) -> dict[str, Path]:
    paths = {}
    for spec in manifest["files"]:
        name = spec["path"]
        relative = Path(name)
        path = (directory / relative).resolve()
        _need(
            not relative.is_absolute()
            and str(relative) == name
            and ".." not in relative.parts
            and path.is_relative_to(directory.resolve())
            and name not in paths,
            "Unsafe or duplicate previous prepared path",
        )
        paths[name] = _bound(dict(spec, path=str(path)))
    _need(list(paths) == sorted(paths), "Previous file roster must be canonical")
    return paths


def _copy_exact(source: Path, target: Path) -> dict:
    before = binding(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as incoming, target.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    _need(
        {key: binding(target)[key] for key in ("bytes", "sha256")}
        == {key: before[key] for key in ("bytes", "sha256")}
        and binding(source) == before,
        "Preserved input changed while copying",
    )
    return before


def prepare(config_path: Path, directory: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["schema"] != "trajectory-pi-p1-1" or not config["launch_ready"]:
        raise ValueError("A committed frozen P1 configuration is required")
    if config.get("version") == "P1.1" or config.get("input_mode") == "worker-native":
        raise ValueError(
            "Worker-native P1.1 requires prepare_native; never reseed inherited streams"
        )
    directory.mkdir(parents=True, exist_ok=False)
    originals = directory / "originals"
    originals.mkdir()
    teachers, parents = [], []
    for spec in config["teachers"]:
        source_spec = spec["episode"]
        source = Path(source_spec["path"])
        if source.stat().st_size != source_spec["bytes"] or sha(source) != source_spec["sha256"]:
            raise ValueError("Recorded teacher source binding differs")
        shutil.copyfile(source, originals / f"teacher_{spec['slot']}.npz")
        copied = originals / f"teacher_{spec['slot']}.npz"
        if sha(copied) != source_spec["sha256"] or sha(source) != source_spec["sha256"]:
            raise ValueError("Teacher source changed during preparation")
        with np.load(source, allow_pickle=False) as raw:
            values = dict(
                states=np.concatenate((raw["states"], raw["terminal_state"][None])),
                warmstarts=np.concatenate((raw["warmstarts"], raw["terminal_warmstarts"][None])),
                integrations=np.concatenate(
                    (raw["integrations"], raw["terminal_integrations"][None])
                ),
                observations=raw["observations"].copy(),
                actions=raw["actions"].copy(),
                costs=raw["costs"].copy(),
            )
        if (
            values["states"].shape != (1001, 19)
            or values["actions"].shape != (1000, 6)
            or values["warmstarts"].shape != (1001, 9)
            or values["integrations"].shape != (1001, 91)
            or any(
                value.dtype != np.float64 or not np.isfinite(value).all()
                for value in values.values()
            )
        ):
            raise ValueError("Initial teacher must retain complete finite float64 contexts")
        if not np.array_equal(observations(values["states"][:-1]), values["observations"]):
            raise ValueError("Initial teacher pre-action observations differ")
        healthy = (
            (values["states"][1:, 2] > 0.8)
            & (values["states"][1:, 2] < 2)
            & (np.abs(values["states"][1:, 3]) < 1)
        )
        mae = float(np.mean(np.abs(values["states"][1:, 10] - 1.5)))
        if not healthy.all() or mae > 0.3:
            raise ValueError("Initial teaching sequence is not complete competent walking")
        target = directory / spec["prepared_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.savez_compressed(stream, **values)
        teachers.append(
            dict(
                slot=spec["slot"],
                file_relative=spec["prepared_path"],
                source=binding(source),
                complete_cost=float(values["costs"].sum()),
                walking=True,
                velocity_mae=mae,
            )
        )
    normalizations = []
    for spec in config["parents"]:
        source_spec = spec["checkpoint"]
        source = Path(source_spec["path"])
        if source.stat().st_size != source_spec["bytes"] or sha(source) != source_spec["sha256"]:
            raise ValueError("Recorded incoming policy/Adam package differs")
        target = directory / spec["prepared_directory"]
        policy, state, envelope = freeze_legacy_parent(
            source,
            target,
            global_seed=spec["new_global_envelope_seed"],
            metadata=dict(study="trajectory-pi-p1-1", stream=spec["stream"], parent_refitted=False),
        )
        if state["updates"] != 2000 or state["fit_updates"] != 2000:
            raise ValueError("Expected the recorded W2 2,000-update endpoint")
        normalizations.append(policy.normalization)
        parents.append(
            dict(
                stream=spec["stream"],
                directory=spec["prepared_directory"],
                source=binding(source),
                policy_hash=policy.hash(),
                optimizer_fingerprint=envelope["optimizer_fingerprint"],
                global_rng_fingerprint=envelope["global_rng_fingerprint"],
            )
        )
    if (
        len(parents) != 3
        or len(teachers) != 2
        or any(
            not np.array_equal(item[key], normalizations[0][key])
            for item in normalizations
            for key in ("mean", "std")
        )
    ):
        raise ValueError("Three matched W2 normalizations and two teachers required")
    roster = [
        dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    result = dict(
        schema="trajectory-pi-prepared-1",
        config_sha256=sha(config_path),
        parents=parents,
        teachers=teachers,
        files=roster,
        normalization_inherited=True,
        parent_refits=0,
        historical_global_rng_available=False,
        new_study_globals_frozen=True,
        preparation_work=dict(
            checkpoint_loads=3,
            policy_constructions=3,
            neural_rows=0,
            native_controls=0,
            optimizer_steps=0,
        ),
    )
    write(directory / "manifest.json", result)
    return result


def _native_proof(
    config_path: Path, validation_manifest: Path, prior_path: Path, source_manifest: Path
) -> tuple[dict, dict, dict, dict]:
    """Check saved R0/RV evidence and its independent seals without physics."""
    proof = json.loads(validation_manifest.read_text())
    _need(
        proof.get("schema") == "trajectory-pi-native-validation-1"
        and proof.get("status") == "PASS"
        and proof.get("p1_config_sha256") == sha(config_path)
        and proof.get("config_sha256")
        == sha(config_path.with_name("trajectory_pi_native_baselines_p1_1.json"))
        and proof.get("source_manifest_sha256") == sha(source_manifest)
        and proof.get("previous_prepared_manifest") == binding(prior_path),
        "Native validation proof does not bind this frozen P1.1/input/source",
    )
    for name in ("result", "receipt"):
        _bound(proof[name])
    validation_receipt = _json_bound(proof["receipt"])
    origin = proof["origin_replay"]
    origin_documents = {
        name: _json_bound(origin[name])
        for name in ("result", "audit_receipt", "audit_findings", "external_summary")
    }
    _need(
        origin.get("origin_replay_passed") is True
        and origin_documents["result"].get("status") == "ORIGIN_REPLAY_PASS"
        and origin_documents["audit_receipt"].get("passed") is True
        and origin_documents["audit_receipt"].get("origin_replay_passed") is True
        and origin_documents["audit_findings"].get("origin_replay_passed") is True,
        "R0 origin replay must pass independently; an evidence audit alone is insufficient",
    )
    audit = {name: _json_bound(proof["native_audit"][name]) for name in ("receipt", "findings")}
    _need(
        audit["receipt"].get("passed") is True
        and audit["receipt"].get("native_validation_passed") is True
        and audit["findings"].get("native_validation_passed") is True,
        "Worker-native validation and its independent audit must both pass",
    )
    episodes = {row["slot"]: row for row in proof["episodes"]}
    _need(
        len(proof["episodes"]) == len(episodes) == 2 and set(episodes) == {0, 1},
        "Two unique native teacher slots required",
    )
    findings = audit["findings"]
    audited_episodes = {row["slot"]: row for row in findings["episodes"]}
    _need(
        findings.get("schema") == "trajectory-pi-native-baselines-audit-findings-1"
        and findings.get("result_binding") == proof["result"]
        and all(
            findings.get(name) == proof[name]
            for name in (
                "config_sha256",
                "p1_config_sha256",
                "source_manifest_sha256",
                "previous_prepared_manifest",
                "origin_replay",
            )
        )
        and audit["receipt"].get("findings_binding") == proof["native_audit"]["findings"]
        and len(findings["episodes"]) == len(audited_episodes) == 2
        and set(audited_episodes) == set(episodes)
        and all(
            all(
                audited_episodes[slot].get(name) == row[name]
                for name in (
                    "original_episode",
                    "original_prepared",
                    "capture",
                    "replay",
                    "candidate",
                    "native_replay_passed",
                    "single_candidate_passed",
                    "initial_context_exact",
                    "actions_exact",
                )
            )
            for slot, row in episodes.items()
        ),
        "Native validation audit does not certify this exact result/source/episode roster",
    )
    _need(
        all(
            all(
                audited_episodes[slot]["outcome"][name] == row["outcome"][name]
                for name in ("controls", "walking", "full_survival")
            )
            and np.allclose(
                [
                    audited_episodes[slot]["outcome"][name]
                    for name in ("complete_cost", "velocity_mae")
                ],
                [row["outcome"][name] for name in ("complete_cost", "velocity_mae")],
                rtol=0,
                atol=1e-10,
            )
            for slot, row in episodes.items()
        ),
        "Audited worker-native outcomes differ from the imported baselines",
    )
    return proof, episodes, origin_documents, validation_receipt


def _imported_work(proof: dict, origin_documents: dict, validation_receipt: dict) -> list[dict]:
    return [
        dict(
            phase="r0",
            receipt=proof["origin_replay"]["result"],
            counts=origin_documents["result"]["work"],
        ),
        dict(phase="rv", receipt=proof["receipt"], counts=validation_receipt["counts"]),
    ]


def validate_native_prepared(
    config_path: Path, directory: Path, manifest: dict, source_manifest: Path
) -> None:
    """Metadata-only RV/R0 and inherited-stream gate, before P0 imports physics."""
    config = json.loads(config_path.read_text())
    _need(
        config.get("version") == manifest.get("version") == "P1.1"
        and config.get("input_mode") == manifest.get("input_mode") == "worker-native"
        and manifest.get("new_global_envelopes_created") == 0,
        "Explicit worker-native version and unchanged global envelopes required",
    )
    previous_config = _bound(manifest["previous_config"])
    prior_path = _bound(manifest["previous_prepared_manifest"])
    prior = json.loads(prior_path.read_text())
    _need(
        sha(previous_config) == sha(config_path.with_name("trajectory_pi_p1.json"))
        and prior["config_sha256"] == sha(previous_config),
        "P1.0 configuration/previous preparation binding differs",
    )
    proof, episodes, origin_documents, validation_receipt = _native_proof(
        config_path, _bound(manifest["validation_manifest"]), prior_path, source_manifest
    )
    old_files = _prior_files(prior_path.parent, prior)
    for parent in manifest["parents"]:
        for name in ("policy.pt", "global_rng.pt", "stream.json"):
            relative = parent["directory"] + "/" + name
            _need(
                parent["inherited_stream_files"][name] == binding(old_files[relative])
                and all(
                    binding(directory / relative)[key]
                    == parent["inherited_stream_files"][name][key]
                    for key in ("bytes", "sha256")
                ),
                "P0 incoming stream differs from exact P1.0 continuation",
            )
    for teacher in manifest["teachers"]:
        _need(
            teacher["native_capture"] == episodes[teacher["slot"]]["capture"],
            "P0 teacher does not bind the audited native capture",
        )
        _bound(teacher["native_capture"])
    _need(
        manifest["imported_work"] == _imported_work(proof, origin_documents, validation_receipt),
        "Antecedent R0/RV accounting differs from its bound receipts",
    )


def prepare_native(
    config_path: Path,
    directory: Path,
    validation_manifest: Path,
    previous_prepared: Path,
) -> dict:
    """Import audited worker-native executions and exact existing P1.0 streams.

    All prerequisites and source arrays are checked before creating output. This
    route never constructs an actor, decodes a checkpoint, or seeds any RNG.
    Original source episodes and the previous nine stream files remain exact.
    """
    import h5py

    config_path, directory = Path(config_path), Path(directory)
    validation_manifest, previous_prepared = Path(validation_manifest), Path(previous_prepared)
    config = json.loads(config_path.read_text())
    previous_config = config_path.with_name("trajectory_pi_p1.json")
    baseline = json.loads(previous_config.read_text())
    _need(
        config.get("schema") == "trajectory-pi-p1-1"
        and config.get("version") == "P1.1"
        and config.get("input_mode") == "worker-native"
        and config.get("launch_ready") is True,
        "Explicit launch-ready worker-native P1.1 configuration required",
    )
    for name in (
        "streams",
        "arms",
        "rounds",
        "horizon",
        "parents",
        "planner",
        "task",
        "learning",
        "optimizer",
        "score",
        "admission",
        "queries",
        "evaluation",
        "seeds",
        "criteria",
        "resources",
        "limits",
        "prerequisites",
        "stopping",
        "claim_scope",
    ):
        _need(config.get(name) == baseline.get(name), "P1.1 scientific recipe changed: " + name)
    _need(
        len(config["teachers"]) == len(baseline["teachers"]) == 2
        and all(
            all(
                new.get(key) == old.get(key)
                for key in ("slot", "episode", "prepared_path", "controls")
            )
            for new, old in zip(config["teachers"], baseline["teachers"], strict=True)
        ),
        "P1.1 changed the selected original teachers",
    )
    prior_path = previous_prepared / "manifest.json"
    prior = json.loads(prior_path.read_text())
    _need(
        prior.get("schema") == "trajectory-pi-prepared-1"
        and prior.get("config_sha256") == sha(previous_config)
        and prior.get("normalization_inherited") is True
        and prior.get("parent_refits") == 0,
        "Unchanged frozen P1.0 preparation required",
    )
    files = _prior_files(previous_prepared, prior)
    proof, episodes, origin_documents, validation_receipt = _native_proof(
        config_path,
        validation_manifest,
        prior_path,
        config_path.resolve().parents[2].parent / "source_manifest.json",
    )
    prior_parents = {row["stream"]: row for row in prior["parents"]}
    prior_teachers = {row["slot"]: row for row in prior["teachers"]}
    _need(set(prior_parents) == set(config["streams"]), "Previous parent roster differs")
    planned, teachers, parents = [], [], []
    for spec in config["teachers"]:
        slot, episode = spec["slot"], episodes[spec["slot"]]
        _need(
            all(
                episode.get(name) is True
                for name in (
                    "native_replay_passed",
                    "single_candidate_passed",
                    "initial_context_exact",
                    "actions_exact",
                )
            ),
            "Native replay/model/context/action gate failed",
        )
        original = files[f"originals/teacher_{slot}.npz"]
        old_teacher = files[spec["prepared_path"]]
        _need(
            prior_teachers[slot]["source"] == spec["episode"]
            and episode["original_prepared"] == binding(old_teacher)
            and all(
                episode["original_episode"][name] == spec["episode"][name]
                for name in ("bytes", "sha256")
            )
            and all(
                binding(original)[name] == spec["episode"][name] for name in ("bytes", "sha256")
            ),
            "Native execution must retain the exact selected original episode",
        )
        _bound(episode["original_episode"])
        for role in ("capture", "replay", "candidate"):
            _bound(episode[role])
        with np.load(old_teacher, allow_pickle=False) as raw:
            old = {name: raw[name].copy() for name in raw.files}
        with h5py.File(episode["capture"]["path"], "r") as raw:
            values = {
                name: raw[name][:]
                for name in (
                    "states",
                    "warmstarts",
                    "integrations",
                    "observations",
                    "actions",
                    "costs",
                )
            }
        expected_shapes = dict(
            states=(1001, 19),
            warmstarts=(1001, 9),
            integrations=(1001, 91),
            observations=(1001, 17),
            actions=(1000, 6),
            costs=(1000,),
        )
        _need(
            all(
                value.shape == expected_shapes[name]
                and value.dtype == np.float64
                and np.isfinite(value).all()
                for name, value in values.items()
            )
            and np.all(np.abs(values["actions"]) <= 1),
            "Complete finite float64 worker-native teacher required",
        )
        _need(
            values["actions"].tobytes() == old["actions"].tobytes()
            and all(
                values[name][0].tobytes() == old[name][0].tobytes()
                for name in ("states", "warmstarts", "integrations")
            ),
            "Native capture changed original torques or complete initial context",
        )
        _need(
            np.array_equal(values["observations"], observations(values["states"])),
            "Native teacher observations are not pre-action state features",
        )
        healthy = (
            (values["states"][1:, 2] > 0.8)
            & (values["states"][1:, 2] < 2)
            & (np.abs(values["states"][1:, 3]) < 1)
        )
        mae = float(np.mean(np.abs(values["states"][1:, 10] - 1.5)))
        cost = float(values["costs"].sum())
        _need(
            healthy.all()
            and mae <= 0.3
            and episode["outcome"]["walking"] is True
            and episode["outcome"]["full_survival"] is True
            and episode["outcome"]["controls"] == 1000
            and episode["outcome"]["complete_cost"] == cost,
            "Worker-native baseline is not complete competent walking",
        )
        values["observations"] = values["observations"][:-1].copy()
        planned.append((spec, original, values))
        teachers.append(
            dict(
                slot=slot,
                file_relative=spec["prepared_path"],
                source=spec["episode"],
                native_capture=episode["capture"],
                complete_cost=cost,
                walking=True,
                velocity_mae=mae,
            )
        )
    for spec in config["parents"]:
        old = prior_parents[spec["stream"]]
        _need(
            old["directory"] == spec["prepared_directory"] and old["source"] == spec["checkpoint"],
            "Original parent binding differs",
        )
        inherited = {
            name: binding(files[old["directory"] + "/" + name])
            for name in ("policy.pt", "global_rng.pt", "stream.json")
        }
        _need(
            all(
                inherited["policy.pt"][name] == spec["checkpoint"][name]
                for name in ("bytes", "sha256")
            ),
            "Original incoming policy package differs",
        )
        parents.append(
            dict(
                old,
                inherited_stream_files=inherited,
                source_prepared_directory=str((previous_prepared / old["directory"]).resolve()),
            )
        )
    directory.mkdir(parents=True, exist_ok=False)
    for spec, original, values in planned:
        _copy_exact(original, directory / "originals" / f"teacher_{spec['slot']}.npz")
        target = directory / spec["prepared_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.savez_compressed(stream, **values)
    for parent in parents:
        for name, spec in parent["inherited_stream_files"].items():
            _copy_exact(_bound(spec), directory / parent["directory"] / name)
    result = dict(
        schema="trajectory-pi-prepared-1",
        version="P1.1",
        input_mode="worker-native",
        config_sha256=sha(config_path),
        previous_config=binding(previous_config),
        previous_prepared_manifest=binding(prior_path),
        validation_manifest=binding(validation_manifest),
        parents=parents,
        teachers=teachers,
        files=[
            dict(binding(path), path=str(path.relative_to(directory)))
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        ],
        normalization_inherited=True,
        parent_refits=0,
        historical_global_rng_available=False,
        new_study_globals_frozen=True,
        new_global_envelopes_created=0,
        global_envelope_origin="P1.0 prepared stream files, copied byte-identically without reseeding",
        preparation_work=dict(
            checkpoint_loads=0,
            policy_constructions=0,
            optimizer_constructions=0,
            neural_rows=0,
            native_controls=0,
            model_controls=0,
            backward_rows=0,
            optimizer_steps=0,
            exact_file_copies=11,
            canonical_teacher_arrays_written=2,
        ),
        imported_work=_imported_work(proof, origin_documents, validation_receipt),
        imported_work_scope="Prior paid R0/RV phases, each imported once; no new preparation calls or duplicated P0 charges",
    )
    write(directory / "manifest.json", result)
    return result
