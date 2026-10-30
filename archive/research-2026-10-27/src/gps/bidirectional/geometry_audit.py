"""Independent saved-evidence arithmetic for the state-geometry Ending assay.

No producer, actor, Torch, task, engine or optimizer is imported. Recorded
hidden features are bound to source, owner, exact input rows and metered call
chronology; this audit does not recompute a neural layer or hidden feature.
The QR, SVD and statistical feature projections below are audit arithmetic.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BOOTSTRAP_SHA = "e415ae406f8d1db5cd829f64b042da9abbfa1fd7710bfe46dfd7e64832322c83"
BOOTSTRAP_ACTOR = "cb4f427fbd45c611e1ab256003319100fca744fc842f0333bea8aff500948158"
CHAINS = ("fixed", "ordinary", "raw", "long")
FEATURE_VERIFICATION_SOURCE = "bf2d03c71f0611653cbff575289c86dbd9fc2a6eec573d5e0ba89aabd62b9ad1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf"
        and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a, b, atol=2e-12, rtol=2e-12), message)


def exact(actual, expected, message):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and a.dtype == b.dtype
        and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes(), message)


def same_tree(actual, expected, message):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), message+": fields")
        for key in expected:
            same_tree(actual[key], expected[key], message+": "+str(key))
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message+": length")
        for a, b in zip(actual, expected):
            same_tree(a, b, message)
    elif type(expected) is bool:
        require(actual is expected, message)
    elif type(expected) in (int, float):
        same(actual, expected, message)
    else:
        require(actual == expected, message)


def declared_bound():
    training = 4*3*13*8*2*25
    return dict(calibration_controls=8*25, feature_forward_rows=288, training_controls=training,
        qualification_controls=5*16*25*2, final_controls=3*64*25*2, total_controls=76200,
        executed_controls=76200, actor_rows=76488, model_controls=0, adam_steps=0,
        diagnostic_backward_passes=0, training_actor_versions=156, search_iterations=12,
        training_episodes=2496, qualification_episodes=160, final_episodes=384, geometry_svd_calls=4)


def fixed_config():
    return dict(schema="bidirectional-ending-state-geometry-assay-v1",
        task=dict(gate=12, delay=12, move=.2, ending_y=-.1, effort=.02, greed=.2,
            detour=.2, detour_steps=5, penalty=20.), chains=list(CHAINS), search_steps=3,
        candidates_per_step=13, basis_seed=1701, head_weight_directions=4, bias_step=.1,
        log_std_step=.5, coefficient_l1_bound=3., calibration_roots=8, calibration_seed=621000,
        calibration_prefix_steps=12, feature_rows=288, geometry_rows=192, target_pre_squash_rms=.1,
        weight_axis_norm_cap=1., bias_axis_abs_cap=.5, singular_rms_cutoff=.001,
        ordinary_training_roots=8, ordinary_training_seed=622000, training_noise_seed=624000,
        controller_mode_weights=dict(mean=.5, stochastic=.5), source_return_weight=0.,
        teacher_action_labels=False, qualification_roots=16, qualification_seed=631000,
        qualification_noise_seed=633000, evaluation_roots=64, evaluation_seed=632000,
        evaluation_noise_seed=634000, minimum_complete_gain=.001, capture_min=.9,
        max_controls=76200, max_executed_controls=76200, max_model_controls=0,
        max_actor_rows=76488, max_adam_steps=0, max_diagnostic_backward_passes=0)


def make_basis():
    draws = np.random.Generator(np.random.PCG64(1701)).standard_normal((128, 4))
    q, r = np.linalg.qr(draws, mode="reduced")
    return (q*np.where(np.diag(r) < 0, -1., 1.)[None, :]).T


def calibration_indices(mode):
    require(mode in CHAINS, "Undeclared geometry calibration source")
    start = 0 if mode in ("fixed", "ordinary") else 96 if mode == "raw" else 192
    return np.r_[np.arange(96), np.arange(start, start+96)]


def geometry_arrays(hidden, basis, mode):
    """Reconstruct statistical geometry; never evaluate an actor's layers."""
    h, b = np.asarray(hidden, dtype=np.float64), np.asarray(basis, dtype=np.float64)
    require(h.shape == (192, 128) and b.shape == (4, 128) and np.isfinite(h).all()
        and np.isfinite(b).all() and np.allclose(b@b.T, np.eye(4), atol=1e-12, rtol=0)
        and mode in CHAINS, "Invalid fixed geometry support/basis/source")
    projected = h@b.T
    mu = projected.mean(axis=0)
    _, singular, right = np.linalg.svd(projected-mu, full_matrices=False)
    for vector in right:
        if vector[np.argmax(np.abs(vector))] < 0:
            vector *= -1
    rms = singular/math.sqrt(192)
    if mode == "fixed":
        scales = np.full(4, .1)
        active = np.ones(4, dtype=bool)
        weights, bias = .1*b, np.zeros(4)
        caps = ["fixed_old_random_basis"]*4
    else:
        active, scales, caps = rms >= .001, np.zeros(4), []
        for i in range(4):
            if not active[i]:
                caps.append("inactive_singular_rms_below_.001")
                continue
            offset = float(mu@right[i])
            limits = [.1/rms[i], 1., math.inf if offset == 0 else .5/abs(offset)]
            scales[i] = min(limits)
            caps.append(("target_rms_.1", "weight_norm1", "bias_abs_.5")[int(np.argmin(limits))])
        weights = (right@b)*scales[:, None]
        bias = -(mu@right.T)*scales
    response = h@weights.T+bias
    centered = response-response.mean(axis=0)
    diagnostics = dict(mode=mode, calibration_rows=192, pre_squash_feature_dim=128,
        projected_feature_dim=4, projected_mean=mu.tolist(), right_singular_vectors=right.tolist(),
        singular_values=singular.tolist(), singular_rms=rms.tolist(), active_axes=active.tolist(),
        axis_scales=scales.tolist(), limiting_caps=caps, weight_norms=np.linalg.norm(weights, axis=1).tolist(),
        bias_offsets=bias.tolist(), achieved_pre_squash_rms=np.sqrt(np.mean(response*response, axis=0)).tolist(),
        achieved_centered_pre_squash_rms=np.sqrt(np.mean(centered*centered, axis=0)).tolist(),
        response_means=response.mean(axis=0).tolist(), centered_response_gram=(centered.T@centered/192).tolist(),
        empirical_active_rank=int(np.sum(rms >= .001)), active_parameter_axes=int(np.sum(active)),
        singular_vectors_used_for_axes=mode != "fixed",
        scope="Empirical parent pre-squash feature geometry only; caps and inactive axes may prevent target RMS; no rescaling or retry")
    require(np.max(np.linalg.norm(weights, axis=1)) <= 1+1e-12 and np.max(abs(bias)) <= .5+1e-12,
        "Reconstructed geometry crossed declared caps")
    return dict(weight_axes=weights, bias_axes=bias, diagnostics=diagnostics)


def candidate_coordinates(center, step):
    c = np.asarray(center, dtype=np.float64)
    require(c.shape == (6,) and np.isfinite(c).all() and type(step) is int and 0 <= step < 3
        and np.abs(c).sum() <= step+1e-12, "Invalid geometry center/step reachability")
    values = [c.copy()]
    for axis in range(6):
        for sign in (-1., 1.):
            value = c.copy()
            value[axis] += sign
            values.append(value)
    return np.stack(values)


def verify_head_actor(parent, candidate, geometry, coordinates):
    c = np.asarray(coordinates, dtype=np.float64)
    w, b = np.asarray(geometry["weight_axes"], float), np.asarray(geometry["bias_axes"], float)
    require(c.shape == (6,) and np.isfinite(c).all() and np.abs(c).sum() <= 3+1e-12
        and w.shape == (4, 128) and b.shape == (4,) and np.isfinite(w).all() and np.isfinite(b).all()
        and np.max(np.linalg.norm(w, axis=1)) <= 1+1e-12 and np.max(abs(b)) <= .5+1e-12,
        "Invalid geometry coefficient/axis caps")
    for key in ("format_version", "policy_type", "obs_dim", "hidden_dim", "num_families", "action_low", "action_high"):
        require(candidate[key] == parent[key], "Geometry proposal changed architecture/bounds")
    require(parent["format_version"] == 1 and parent["policy_type"] == "bidirectional-family-gaussian-v1"
        and parent["obs_dim"] == 8 and parent["hidden_dim"] == 128
        and type(parent["num_families"]) is int and parent["num_families"] >= 1
        and parent["action_low"] == [-1.] and parent["action_high"] == [1.], "Wrong geometry parent architecture")
    require(set(parent["actor"]) == set(candidate["actor"]), "Geometry proposal changed tensor roster")
    delta = {"fc_mean.weight": c[1:5]@w, "fc_mean.bias": .1*c[0]+c[1:5]@b, "log_std": .5*c[5]}
    for name, value in parent["actor"].items():
        value = np.asarray(value)
        require(value.dtype.kind in "biuf" and np.isfinite(value).all(), "Nonfinite/nonnumeric parent tensor")
        expected = (value.astype(np.float64)+delta[name]).astype(value.dtype) if name in delta else value
        exact(candidate["actor"][name], expected, "Geometry proposal tensor bytes differ: "+name)
    require(np.asarray(parent["actor"]["fc_mean.weight"]).shape == (1, 128)
        and np.asarray(parent["actor"]["fc_mean.bias"]).shape == np.asarray(parent["actor"]["log_std"]).shape == (1,),
        "Geometry scalar head shape differs")
    exact(candidate["sampling_rng_state"], parent["sampling_rng_state"], "Geometry private RNG changed")
    require(np.all(np.asarray(candidate["actor"]["log_std"]) >= -5.)
        and np.all(np.asarray(candidate["actor"]["log_std"]) <= 1.), "Geometry active logstd bounds crossed")
    return True


def verify_feature_arrays(feature, inputs, parent):
    """Validate saved feature packet metadata, not the neural computation."""
    require(set(inputs) == {"observations", "states", "families", "row_references_json", "actor_hash", "teacher_action_labels"}
        and set(feature) == {"hidden", "observations", "states", "parent_pre_squash_means", "parent_log_std",
            "sampling_rng_before", "sampling_rng_after", "input_raw_json", "actor_hash"}, "Feature packet array roster differs")
    require(inputs["observations"].shape == (288, 8) and inputs["states"].shape == (288, 2)
        and inputs["families"].shape == (288,) and inputs["families"].dtype == np.dtype("int64")
        and np.all(inputs["families"] == 0) and not bool(inputs["teacher_action_labels"]), "Feature support/family/teacher-label scope differs")
    for key in ("observations", "states"):
        exact(feature[key], inputs[key], "Feature query changed frozen inputs: "+key)
        require(np.isfinite(inputs[key]).all(), "Nonfinite feature input")
    dtype = np.asarray(parent["actor"]["fc_mean.weight"]).dtype
    require(feature["hidden"].shape == (288, 128) and feature["hidden"].dtype == dtype
        and np.isfinite(feature["hidden"]).all() and np.max(abs(feature["hidden"])) <= 1., "Saved tanh-feature shape/range/dtype invalid")
    for key in ("parent_pre_squash_means", "parent_log_std"):
        require(feature[key].shape == (288, 1) and feature[key].dtype == dtype
            and np.isfinite(feature[key]).all(), "Saved parent output shape/dtype invalid")
    require(str(feature["actor_hash"]) == str(inputs["actor_hash"]) == BOOTSTRAP_ACTOR, "Feature query owner differs")
    exact(feature["sampling_rng_before"], parent["sampling_rng_state"], "Feature query did not use frozen parent RNG")
    exact(feature["sampling_rng_after"], parent["sampling_rng_state"], "Feature query consumed parent private RNG")
    return True


def select_from_scores(scores):
    require(len(scores) == 13, "Geometry selector requires all thirteen paid candidates")
    baseline = scores[0]["ordinary"]
    feasible, objectives, changes = [], [], []
    for score in scores:
        row = score["ordinary"]
        for mode in ("mean", "stochastic"):
            r = row[mode]
            require(math.isfinite(r["mean_cost"]) and type(r["captures"]) is int and 0 <= r["captures"] <= 8
                and len(r["capture_flags"]) == 8 and all(type(x) is bool for x in r["capture_flags"])
                and sum(r["capture_flags"]) == r["captures"] and len(r["context_ids"]) == len(set(r["context_ids"])) == 8
                and r["context_ids"] == baseline[mode]["context_ids"], "Invalid geometry complete ordinary score")
        feasible.append(all(row[m]["mean_cost"] <= baseline[m]["mean_cost"]
            and row[m]["captures"] >= baseline[m]["captures"] for m in ("mean", "stochastic")))
        objectives.append((row["mean"]["mean_cost"]+row["stochastic"]["mean_cost"])/50.)
        changes.append({m: dict(lost=sum(a and not b for a, b in zip(baseline[m]["capture_flags"], row[m]["capture_flags"])),
            gained=sum(b and not a for a, b in zip(baseline[m]["capture_flags"], row[m]["capture_flags"])))
            for m in ("mean", "stochastic")})
    require(feasible[0], "Geometry current center lost feasibility")
    selected = min((i for i, ok in enumerate(feasible) if ok), key=lambda i: (objectives[i], i))
    return dict(selected_index=selected, feasible_flags=feasible, objectives=objectives,
        actor_hash=scores[selected]["actor_hash"], source_return_weight=0., paired_capture_changes=changes,
        capture_nonregression_definition="Aggregate capture count in each separately executed matched mode; per-root losses/gains retained",
        scope="Only actual complete ordinary-root returns; cost and capture nonregression in each mode")


def assessment(mean, noisy, proposed, references):
    comparisons, passes = {}, []
    for mode, data in (("mean", mean), ("stochastic", noisy)):
        comparisons[mode] = {}
        for name in references:
            a, b = data[name], data[proposed]
            require(len(a) == len(b) > 0 and [x["context_id"] for x in a] == [x["context_id"] for x in b], "Unpaired geometry assessment")
            gains = [x["cost"]-y["cost"] for x, y in zip(a, b)]
            gain = math.fsum(gains)/len(gains)
            nonreg = sum(x["capture"] for x in b) >= sum(x["capture"] for x in a)
            value = dict(mean_cost_gain=gain, paired_cost_gains=gains, capture_nonregression=bool(nonreg),
                improved=bool(nonreg and gain >= .001), paired_capture_losses=sum(x["capture"] and not y["capture"] for x, y in zip(a, b)),
                paired_capture_gains=sum(y["capture"] and not x["capture"] for x, y in zip(a, b)),
                capture_nonregression_definition="Aggregate count in this matched mode; per-root loss is not forbidden")
            comparisons[mode][name] = value
            passes.append(value["improved"])
    capture = sum(x["capture"] for x in mean[proposed])/len(mean[proposed])
    return dict(comparisons=comparisons, deterministic_capture_rate=capture, passed=bool(capture >= .9 and all(passes)))


def bind_feature_verification(packet, e, result):
    """Bind the separately paid, source-pinned 288-row verification receipt.

    No array archive is decoded and no neural operation is performed. Missing
    artifacts return PENDING, retaining observed neural attempts. Malformed or
    conflicting closed evidence raises ValueError for the caller's FAIL path.
    The returned paid rows are additional to the producer's fixed row cap.
    """
    packet = Path(packet).resolve()
    root = Path(e.root).resolve()
    consumed, blobs, pending = {}, {}, []
    requested = completed = 0
    def file_bytes(path):
        path = Path(path).resolve()
        key = str(path)
        if key in consumed:
            return blobs[key]
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        evidence_key = str(path.relative_to(root)) if path.is_relative_to(root) else key
        require(evidence_key not in e.hashes or e.hashes[evidence_key] == sha,
            "Feature verification consumed-path hash conflicts: "+evidence_key)
        e.hashes[evidence_key] = sha
        consumed[key], blobs[key] = sha, data
        return data
    def finish(status, **extra):
        return dict(status=status, errors=[], pending=pending, actor_rows_requested=requested,
            actor_rows_completed=completed, actor_rows_unknown=requested-completed,
            new_physics_controls=0, new_scalar_controls=0, optimizer_updates=0, backward_calls=0,
            audit_actor_forward_rows=0, input_sha256=dict(consumed),
            paid_work_scope="Separate verification packet; never added to the producer cap or producer query totals",
            work_overlap="verification_requested and neural_requested name one attempt; only neural rows are charged",
            limits="Receipt/source/input/ledger binding only; this binder performs no neural replay", **extra)
    if not packet.is_dir():
        pending.append("Feature-verification packet directory absent")
        return finish("PENDING")
    names = ("verify.py", "manifest.json", "work_ledger.jsonl", "stage_complete.json", "verified_forward.npz")
    available = {}
    for name in names:
        path = packet/name
        if path.is_file():
            available[name] = file_bytes(path)
        else:
            pending.append("Missing feature-verification artifact: "+name)
    if "work_ledger.jsonl" in available:
        blob = available["work_ledger.jsonl"]
        require(blob.endswith(b"\n"), "Feature-verification work ledger has an incomplete final line")
        ledger = [json.loads(line) for line in blob.splitlines()]
        require(1 <= len(ledger) <= 3, "Feature-verification work roster exceeds the fixed one-forward declaration")
        for i, row in enumerate(ledger):
            require(row["id"] == f"work-{i:09d}" and type(row["wall_seconds"]) in (int, float)
                and math.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0
                and (i == 0 or ledger[i-1]["wall_seconds"] <= row["wall_seconds"]),
                "Feature-verification work IDs/order/clock invalid")
        start = ledger[0]
        require(start["kind"] == "verification_requested" and type(start["rows"]) is int and start["rows"] == 288
            and start["actor_hash"] == BOOTSTRAP_ACTOR and start["source_sha256"] == FEATURE_VERIFICATION_SOURCE,
            "Feature-verification before-call declaration differs")
        if len(ledger) >= 2:
            req = ledger[1]
            require(req["kind"] == "neural_requested" and type(req["rows"]) is int and req["rows"] == 288
                and req["actor_hash"] == BOOTSTRAP_ACTOR and req["scope"] == "independent_functional_feature_replay"
                and req["observation_shape"] == [288, 8] and req["extra_forward_rows"] == 0
                and req["operations"] == "twoLinearTanhplusmeanLinearandclampedlogstd",
                "Feature-verification neural request owner/shape/cap differs")
            requested = req["rows"]
        if len(ledger) == 3:
            done = ledger[2]
            require(done["kind"] in ("neural_complete", "neural_error") and done["request_id"] == ledger[1]["id"]
                and type(done["rows"]) is int and done["rows"] == 288,
                "Feature-verification resolution is unbound/duplicated")
            if done["kind"] == "neural_complete":
                require(done["returned"] is True, "Feature-verification forward did not return")
                completed = done["rows"]
        if requested != completed or len(ledger) < 3:
            pending.append("Feature-verification neural attempt is unresolved or incomplete")
    if pending:
        return finish("PENDING")
    manifest = json.loads(available["manifest.json"])
    receipt = json.loads(available["stage_complete.json"])
    source_sha = hashlib.sha256(available["verify.py"]).hexdigest()
    hashes = {name: consumed[str((packet/name).resolve())] for name in names}
    require(source_sha == FEATURE_VERIFICATION_SOURCE == manifest["source_sha256"] == receipt["source_sha256"],
        "Feature-verification source is not the fixed independently reviewed verifier")
    require(manifest["schema"] == "independent-paid-feature-verification-source-v1"
        and manifest["max_actor_rows"] == 288 and manifest["max_optimizer_steps"] == 0
        and manifest["max_physics_controls"] == manifest["max_scalar_controls"] == 0,
        "Feature-verification prospective schema/caps differ")
    require(receipt["schema"] == "independent-paid-feature-verification-v1" and receipt["passed"] is True
        and receipt["inputs_unchanged"] is True and receipt["blocked_import_attempts"] == []
        and receipt["actor_rows_requested"] == requested == receipt["actor_rows_completed"] == completed == 288
        and receipt["new_physics_controls"] == receipt["new_scalar_controls"] == receipt["optimizer_updates"] == receipt["backward_calls"] == 0,
        "Feature-verification receipt work/verdict differs")
    require(receipt["manifest_sha256"] == hashes["manifest.json"] and receipt["work_ledger_sha256"] == hashes["work_ledger.jsonl"]
        and receipt["output_sha256"] == hashes["verified_forward.npz"], "Feature-verification receipt artifact binding differs")
    require(type(receipt["wall_seconds"]) in (int, float) and math.isfinite(receipt["wall_seconds"])
        and receipt["wall_seconds"] >= ledger[-1]["wall_seconds"], "Feature-verification processing clock differs")
    require(set(receipt["checks"]) == {"hidden", "parent_pre_squash_means", "parent_log_std"}, "Feature-verification output check roster differs")
    for check in receipt["checks"].values():
        require(check["exactly_equal"] is True and check["max_abs_difference"] == 0.,
            "Feature-verification saved neural output differs")
    geometry = result["geometry"]
    require(geometry["inputs"]["file"] == "geometry/feature_inputs.npz" and geometry["features"]["file"] == "geometry/features.npz",
        "Feature-verification geometry artifact names changed")
    local = {
        "assay_result.json": None,
        "manifest.json": result["manifest_sha256"],
        "config.json": result["config_sha256"],
        "inputs/inputs/checkpoints/bootstrap.pt": BOOTSTRAP_SHA,
        "evidence/assay/geometry/feature_inputs.npz": geometry["inputs"]["sha256"],
        "evidence/assay/geometry/features.npz": geometry["features"]["sha256"],
    }
    bindings = manifest["input_sha256"]
    require(isinstance(bindings, dict) and len(bindings) == 6, "Feature-verification must bind exactly six unique inputs")
    prefixes = [Path(name).parent for name in bindings if Path(name).name == "assay_result.json"]
    require(len(prefixes) == 1 and set(bindings) == {str(prefixes[0]/name) for name in local},
        "Feature-verification inputs have duplicates, extras or mixed namespaces")
    for name, expected in local.items():
        data = file_bytes(root/name)
        sha = hashlib.sha256(data).hexdigest()
        require((expected is None or sha == expected) and bindings[str(prefixes[0]/name)] == sha,
            "Feature-verification consumed a different geometry input: "+name)
    require(json.loads(blobs[str((root/"assay_result.json").resolve())]) == result,
        "Feature-verification result is not the currently audited result")
    for key in ("inputs", "features"):
        require((root/"evidence/assay"/geometry[key]["file"]).stat().st_size == geometry[key]["bytes"],
            "Feature-verification geometry input byte count differs")
    native_manifest = json.loads(blobs[str((root/"manifest.json").resolve())])
    runtime = manifest["numerical_runtime"]
    require(runtime["threads"] == 1 and runtime["torch"] == native_manifest["dependency_versions"]["torch"]
        and runtime["numpy"] == native_manifest["dependency_versions"]["numpy"], "Feature-verification numerical runtime differs")
    return finish("PASS", receipt_sha256=hashes["stage_complete.json"], manifest_sha256=hashes["manifest.json"],
        source_sha256=source_sha, work_ledger_sha256=hashes["work_ledger.jsonl"], output_sha256=hashes["verified_forward.npz"],
        checked_output_names=sorted(receipt["checks"]), exact_output_comparisons=True,
        hidden_features_independently_verified_by_separate_paid_packet=True)


def validate_geometry_assay(e, base, rows, work, result, config, checkpoints, episodes, qreq, qdone, task_hash):
    from .ending_audit import checkpoint_numpy, policy_identity, controller_identity, utc, observations, transition_arithmetic, COMPONENTS
    from .ending_completion_audit import numeric_digest
    from .ending_precision_audit import _raw, _summary_check

    require(config == fixed_config(), "Undeclared state-geometry assay configuration")
    require(result["schema"] == config["schema"] and result["no_new_action_labels"] is True
        and result["teacher_information"] == "pre_gate_states_and_observations_only" and result["source_return_weight"] == 0.
        and result["critic_used"] is False and result["gradient_optimizer_used"] is False
        and result["parameter_return_search_used"] is True and result["full_feedback_established"] is False
        and result["efficiency_claim"] is False and result["ppo_comparison"] is False
        and result["no_retuning"] is True and result["old_negatives_unchanged"] is True, "Geometry mechanism/scope differs")
    for key, value in declared_bound().items():
        require(result["declared_work_bound"][key] == value, "Geometry prospective bound differs: "+key)
    require(all(q["mode"] == "live" for q in qreq.values()) and result["accounting"]["model_controls"] == 0,
        "Geometry assay made a model query")
    require(not any(r["kind"] in ("fit_work", "gradient_backward_requested", "diagnostic_backward_requested", "optimizer_requested") for r in rows)
        and not any(r["kind"] in ("optimizer_update_requested", "autograd_grad_requested", "autograd_backward_requested") for r in work),
        "Geometry assay performed optimizer/backward work")
    critical_geometry = {"source/src/gps/bidirectional/geometry_return.py", "source/src/gps/bidirectional/geometry_assay.py",
        "source/scripts/diagnostics/run_ending_geometry_assay.py"}
    require(critical_geometry <= set(e.hashes), "Frozen geometry producer/hook/launcher omitted")
    provenance = e.json("input_provenance.json")
    require(e.hashes["input_provenance.json"] == result["input_provenance_sha256"]
        and provenance["schema"] == "ending-state-geometry-input-provenance-v1"
        and provenance["bootstrap_sha256"] == BOOTSTRAP_SHA and provenance["bootstrap_actor_hash"] == BOOTSTRAP_ACTOR
        and provenance["teacher_action_labels"] is False and provenance["source_return_weight"] == 0., "Geometry source provenance differs")
    refs = {r["file"]: r for r in provenance["files"]}
    expected_inputs = {"manifest.json", "config.json", "loop_result.json", "audit_001.json", "input_provenance.json",
        "evidence/loop/ledger.jsonl", "work_ledger.jsonl", "inputs/checkpoints/bootstrap.pt"}
    expected_inputs |= {f"evidence/loop/episodes/early_initial_{role}_{i:03d}.npz" for role in ("short", "long") for i in range(8)}
    require(set(refs) == expected_inputs and len(refs) == len(provenance["files"]), "Geometry source packet roster differs")
    for name, ref in refs.items():
        path = e.file(Path("inputs")/name, ref["sha256"])
        require(path.stat().st_size == ref["bytes"], "Geometry ancestor bytes differ")
    old, old_audit, old_manifest = e.json("inputs/loop_result.json"), e.json("inputs/audit_001.json"), e.json("inputs/manifest.json")
    require(old_audit["status"] == "PASS" and old_audit["evidence_arithmetic_valid"] is True
        and old_audit["scientific_success"] is False and not old_audit["errors"] and not old_audit["pending"]
        and old["passed"] is False and old["schema"] == "bidirectional-ending-early-context-loop-v5"
        and old["stage"] == "failed_initial_data_all_actor_qualification", "Geometry ancestor negative changed")
    for name, ref in refs.items():
        if name != "audit_001.json":
            require(old_audit["input_sha256"].get(name) == ref["sha256"], "Geometry ancestor audit binding differs: "+name)
    for name, key in (("manifest.json", "manifest_sha256"), ("config.json", "config_sha256"),
            ("evidence/loop/ledger.jsonl", "query_ledger_sha256"), ("work_ledger.jsonl", "work_ledger_sha256"),
            ("input_provenance.json", "input_provenance_sha256")):
        require(refs[name]["sha256"] == old[key], "Geometry ancestor result binding differs")
    require(provenance["reused_early_ancestry"] == old["accounting"]
        and provenance["reused_v3_ancestry"] == old["cost_crosswalk"]["v3_ancestor"], "Geometry reused costs differ")
    same(provenance["reused_wall_seconds"], old["loop_wall_seconds"]+old["reused_wall_seconds"], "Geometry reused wall differs")
    old_sources = {r["file"]: r["sha256"] for r in old_manifest["files"]}
    critical = ("src/ending/core.py", "src/gps/bidirectional/ending_pilot.py", "src/gps/bidirectional/policy.py")
    require(set(provenance["critical_source_sha256"]) == set(critical), "Geometry task/actor source roster differs")
    for name, value in provenance["critical_source_sha256"].items():
        require(value == old_sources["source/"+name] == e.hashes["source/"+name], "Geometry reused task/actor source changed")
    source = checkpoint_numpy(e.file("inputs/inputs/checkpoints/bootstrap.pt", BOOTSTRAP_SHA))
    parent = source["learner"]["policy"]
    require(policy_identity(parent) == BOOTSTRAP_ACTOR, "Wrong exact geometry bootstrap actor")
    if result["stage"] == "execution_error":
        return dict(complete=False, scientific_success=False, partial_stage=True,
            paid_query_requests=len(qreq), paid_controls=sum(q["requested_controls"] for q in qreq.values()),
            analytic_reused_episode_transition_row_checks=0, audit_actor_forward_rows=0, audit_physics_controls=0,
            limits="Operationally incomplete; no full geometry/candidate/qualification assertion")
    loaded = checkpoints["geometry_source_parent"]
    require(numeric_digest(loaded["actor"]) == numeric_digest(parent) and loaded["source_checkpoint_sha256"] == BOOTSTRAP_SHA
        and loaded["source_full_learner_sha256"] == numeric_digest(source["learner"])
        and loaded["promotion_claim"] is False, "Geometry loaded parent/full RNG differs")
    require(np.all(np.asarray(parent["actor"]["log_std"])-1.5 >= -5.)
        and np.all(np.asarray(parent["actor"]["log_std"])+1.5 <= 1.), "Geometry reachable logstd domain crosses bounds")
    groups = defaultdict(list)
    for row in rows:
        groups[row["kind"]].append(row)
    def unique(kind, **keys):
        matches = [r for r in groups[kind] if all(r.get(k) == v for k, v in keys.items())]
        require(len(matches) == 1, "Geometry missing/duplicate event: "+kind+str(keys))
        return matches[0]
    checkpoint_events = {r["label"]: r for r in groups["checkpoint"]}
    require(len(checkpoint_events) == len(groups["checkpoint"]), "Repeated geometry checkpoint label")
    by_name = {ep["episode"]: ep for ep in episodes}
    require(len(by_name) == len(episodes), "Duplicated geometry episode name")
    used, all_ids = set(), set()
    def panel(prefix, n, det, owner, seed):
        values = [by_name[f"{prefix}_{i:03d}"] for i in range(n)]
        for i, ep in enumerate(values):
            require(ep["episode"] not in used and ep["context_id"] == f"ordinary:{seed+i}" and ep["controls"] == 25
                and ep["actor_hash"] == owner and ep["deterministic"] is det
                and ep["semantics"] == "actor_suffix" and ep["closed_outcome"], "Geometry ordinary panel ownership/completion differs")
            require(not all_ids.intersection(ep["query_ids"]), "Geometry physical query evidence reused")
            raw = e.npz(base, ep["raw_ref"])
            same(raw["initial_state"], [0., np.random.default_rng(seed+i).uniform(.2, .4)], "Geometry reset root differs")
            require(all(qreq[q]["mode"] == "live" for q in ep["query_ids"]), "Geometry ordinary panel made a model query")
            used.add(ep["episode"])
            all_ids.update(ep["query_ids"])
        return values
    calibration = panel("geometry_calibration", 8, True, BOOTSTRAP_ACTOR, 621000)
    expected_observations, expected_states, expected_row_refs = [], [], []
    for ep in calibration:
        raw = e.npz(base, ep["raw_ref"])
        expected_observations.extend(raw["observations"][:12])
        expected_states.extend(raw["states"][:12])
        expected_row_refs.extend(dict(role="ordinary", episode=ep["episode"], source_raw=ep["raw_ref"], source_state_index=t,
            source_query_id=ep["query_ids"][t], owner_actor_hash=BOOTSTRAP_ACTOR) for t in range(12))
    old_rows = e.rows("inputs/evidence/loop/ledger.jsonl")
    require(old_rows[-1]["kind"] == "archive_close", "Geometry acquisition ledger not closed")
    old_events = {r["episode"]: r for r in old_rows if r["kind"] == "episode_outcome"}
    reused_checks = 0
    for role, tag in (("raw", "short"), ("long", "long")):
        require(len(provenance["source_events"][role]) == 8, "Geometry source episode panel incomplete")
        for i, event in enumerate(provenance["source_events"][role]):
            name = f"early_initial_{tag}_{i:03d}"
            require(event == old_events[name] and event["episode"] == name, "Geometry exact source event differs")
            path = Path("inputs/evidence/loop")/event["raw"]["file"]
            ref = dict(file=str(path), sha256=event["raw"]["sha256"], bytes=event["raw"]["bytes"])
            raw = _raw(e, path, ref["sha256"], ref["bytes"])
            require(raw["states"].shape == (26, 2) and raw["observations"].shape == (26, 8) and bool(raw["complete"])
                and np.array_equal(raw["states"][:, 0], np.arange(26)), "Geometry source is not full original acquisition")
            owner = json.loads(str(raw["owner_json"]))
            require(owner["semantics"] == "executed_replanning" and owner["controller_hash"] == event["outcome"]["controller_hash"]
                and raw["query_ids"].tolist() == event["outcome"]["query_ids"], "Geometry source controller/query provenance differs")
            post, parts, cost, flags = transition_arithmetic(raw["states"][:-1], raw["actions"], config["task"])
            same(raw["states"][1:], post, "Geometry ancestor transition differs")
            same(raw["raw_costs"], cost, "Geometry ancestor cost differs")
            same(raw["cost_components"], np.stack([parts[k] for k in COMPONENTS], axis=-1), "Geometry ancestor components differ")
            same(raw["observations"], observations(raw["states"], 25), "Geometry source observation map differs")
            expected_observations.extend(raw["observations"][:12])
            expected_states.extend(raw["states"][:12])
            expected_row_refs.extend(dict(role=role, episode=name, source_raw=ref, source_state_index=t,
                source_query_id=str(raw["query_ids"][t]), owner_controller_hash=event["outcome"]["controller_hash"]) for t in range(12))
            reused_checks += 25
    geometry = result["geometry"]
    require(set(geometry["arms"]) == set(CHAINS) and geometry["source_action_labels"] is False
        and geometry["ordinary_calibration_episodes"] == 8 and geometry["teacher_prefix_rows_per_role"] == 96
        and geometry["actor_feature_forward_rows"] == 288, "Geometry measured support summary differs")
    inputs, feature = e.npz(base, geometry["inputs"]), e.npz(base, geometry["features"])
    verify_feature_arrays(feature, inputs, parent)
    same(inputs["observations"], expected_observations, "Geometry support observations differ from exact pre-action prefixes")
    same(inputs["states"], expected_states, "Geometry support states differ from exact pre-action prefixes")
    require(json.loads(str(inputs["row_references_json"])) == expected_row_refs
        and json.loads(str(feature["input_raw_json"])) == geometry["inputs"], "Geometry feature row/input references differ")
    feature_request, feature_complete = unique("geometry_features_requested"), unique("geometry_features_complete")
    require(feature_request["input_raw"] == geometry["inputs"] and feature_request["actor_hash"] == feature_complete["actor_hash"] == BOOTSTRAP_ACTOR
        and feature_request["rows"] == feature_complete["rows"] == 288 and feature_request["feature_dim"] == 128
        and feature_request["forward_calls"] == feature_complete["forward_calls"] == 1
        and feature_request["capture"] == "fc_mean_forward_pre_hook" and feature_request["internal_extra_forwards"] == 0
        and feature_complete["request_id"] == feature_request["id"] and feature_complete["raw"] == geometry["features"], "Feature before-call hook/owner declaration differs")
    local = [r for r in groups["neural_requested"] if r.get("purpose") == "geometry_single_feature_forward"]
    global_calls = [r for r in work if r["kind"] == "neural_requested" and r.get("test") == "geometry_single_feature_forward"]
    require(len(local) == len(global_calls) == 1, "Feature query is not one metered parent forward")
    global_calibration = [r for r in work if r["kind"] == "neural_requested" and r.get("test") == "geometry_parent_calibration"]
    require(len(global_calibration) == 200 and {r["actor_instance"] for r in global_calibration} == {global_calls[0]["actor_instance"]},
        "Feature query used a different runtime actor from parent calibration")
    for req, stream in ((local[0], rows), (global_calls[0], work)):
        done = [r for r in stream if r["kind"] == "neural_complete" and r["request_id"] == req["id"]]
        require(len(done) == 1 and req["rows"] == done[0]["rows"] == 288 and done[0]["returned"] is True
            and utc(feature_request) <= utc(req) <= utc(done[0]) <= utc(feature_complete), "Feature forward work was unresolved/outside request chronology")
    require(local[0]["scope"] == "geometry_single_feature_forward" and global_calls[0]["observation_shape"] == [288, 8]
        and utc(checkpoint_events["geometry_source_parent"]) < utc(feature_request)
        and max(utc(qdone[q]) for ep in calibration for q in ep["query_ids"]) <= utc(feature_request), "Feature inputs predate owner/calibration or shape differs")
    basis_start, basis_end = unique("geometry_basis_requested"), unique("geometry_basis_complete")
    basis_raw = e.npz(base, geometry["basis"])
    basis = make_basis()
    same(basis_raw["basis"], basis, "Geometry independent QR basis differs")
    require(int(basis_raw["seed"]) == basis_start["seed"] == 1701 and basis_start["shape"] == [4, 128]
        and basis_start["qr_calls"] == 1 and basis_end["request_id"] == basis_start["id"]
        and basis_end["raw"] == geometry["basis"] and utc(feature_complete) <= utc(basis_start) <= utc(basis_end), "Geometry basis request/source differs")
    arms, svd_ends = {}, []
    for chain in CHAINS:
        req, done = unique("geometry_svd_requested", chain=chain), unique("geometry_svd_complete", chain=chain)
        raw = e.npz(base, geometry["arms"][chain])
        indices = calibration_indices(chain)
        expected = geometry_arrays(feature["hidden"][indices], basis, chain)
        same(raw["weight_axes"], expected["weight_axes"], "Geometry calibrated weight axes differ")
        same(raw["bias_axes"], expected["bias_axes"], "Geometry calibrated bias compensation differs")
        exact(raw["feature_indices"], indices, "Geometry calibration role/duplicate support differs")
        same_tree(json.loads(str(raw["diagnostics_json"])), expected["diagnostics"], "Geometry SVD/cap diagnostics differ")
        same_tree(done["diagnostics"], expected["diagnostics"], "Geometry completed diagnostic differs")
        require(str(raw["feature_sha256"]) == geometry["features"]["sha256"] and str(raw["basis_sha256"]) == geometry["basis"]["sha256"]
            and req["feature_raw"] == geometry["features"] and req["basis_raw"] == geometry["basis"] and req["rows"] == 192
            and req["svd_calls"] == 1 and req["source_action_labels"] is False and req["source_return_weight"] == 0.
            and done["request_id"] == req["id"] and done["raw"] == geometry["arms"][chain]
            and utc(basis_end) <= utc(req) <= utc(done), "Geometry SVD inputs/intent/source differ")
        arms[chain] = expected
        svd_ends.append(done)
    def measured_score(panels, owner):
        ordinary = {}
        for mode, det in (("mean", True), ("stochastic", False)):
            values = panels[mode]
            ordinary[mode] = dict(mean_cost=math.fsum(ep["cost"] for ep in values)/8,
                captures=sum(ep["capture"] for ep in values), capture_flags=[ep["capture"] for ep in values],
                context_ids=[ep["context_id"] for ep in values], query_ids=[q for ep in values for q in ep["query_ids"]],
                controller_hash=controller_identity(dict(actor_hash=owner, family=0, deterministic=det)))
        return dict(actor_hash=owner, ordinary=ordinary,
            joint_normalized_cost=(ordinary["mean"]["mean_cost"]+ordinary["stochastic"]["mean_cost"])/50.,
            score_semantics="ordinary_only_equal_mean_of_two_SEPARATELY_EXECUTED_complete_controller_returns")
    require([entry["chain"] for entry in result["chains"]] == list(CHAINS), "Geometry chain roster differs")
    final_owners, step_reports, iteration_ends = {}, [], []
    for entry in result["chains"]:
        chain, center = entry["chain"], np.zeros(6)
        require(len(entry["steps"]) == 3, "Geometry search steps differ")
        for step, selection in enumerate(entry["steps"]):
            prefix = f"geometry_{chain}_s{step}"
            coords = candidate_coordinates(center, step)
            raw = e.npz(base, selection["proposals_raw"])
            same(raw["coordinates"], coords, "Geometry candidate bank/order differs")
            same(raw["center"], center, "Geometry current center differs")
            require(str(raw["geometry_sha256"]) == geometry["arms"][chain]["sha256"] and int(raw["step"]) == step,
                "Geometry proposal source differs")
            request, bank = unique("geometry_search_iteration_requested", chain=chain, step=step), unique("geometry_bank_ready", chain=chain, step=step)
            same(request["initial_coordinates"], center, "Geometry iteration center differs")
            require(selection["chain"] == chain and selection["step"] == step and request["id"] == selection["iteration_request_id"]
                and request["proposal_versions"] == request["candidate_scoring_calls"] == 13 and request["selector_calls"] == 1
                and bank["proposals_raw"] == selection["proposals_raw"] and bank["maximum_new_controls"] == bank["maximum_actor_rows"] == 5200
                and max(utc(r) for r in svd_ends) <= utc(request), "Geometry bank intent/caps/frozen geometry differs")
            owners, scores, first_times, last_times = [], [], [], []
            for i, coordinate in enumerate(coords):
                label = f"{prefix}_c{i:02d}"
                saved = checkpoints[label]
                owner = policy_identity(saved["actor"])
                owners.append(owner)
                verify_head_actor(parent, saved["actor"], arms[chain], coordinate)
                same(saved["coordinates"], coordinate, "Geometry saved absolute coordinates differ")
                require(saved["proposal_index"] == i and saved["step"] == step and saved["chain"] == chain
                    and saved["base_actor_hash"] == BOOTSTRAP_ACTOR and saved["geometry_raw"] == geometry["arms"][chain]
                    and saved["proposals_raw"] == selection["proposals_raw"] and saved["promotion_claim"] is False, "Geometry candidate checkpoint ancestry differs")
                version = unique("geometry_proposal_version_requested", chain=chain, step=step, candidate_index=i)
                version_done = unique("geometry_proposal_version_complete", request_id=version["id"])
                same(version["coordinates"], coordinate, "Geometry version request coordinate differs")
                require(version["iteration_request_id"] == request["id"] and version["parameter_constructions"] == 1
                    and version_done["checkpoint_label"] == label and version_done["actor_hash"] == owner
                    and utc(request) <= utc(version) <= utc(checkpoint_events[label]) <= utc(version_done) <= utc(bank), "Geometry bank not frozen before measured returns")
                evaluation = unique("geometry_candidate_evaluation_requested", chain=chain, step=step, candidate_index=i)
                scoring = unique("geometry_candidate_score_requested", chain=chain, step=step, candidate_index=i)
                returned = unique("geometry_candidate_return", chain=chain, step=step, candidate_index=i)
                require(evaluation["iteration_request_id"] == request["id"] and evaluation["actor_hash"] == owner
                    and evaluation["complete_episode_queries"] == 16 and evaluation["maximum_controls"] == evaluation["maximum_actor_rows"] == 400
                    and scoring["evaluation_request_id"] == evaluation["id"] and scoring["actor_hash"] == owner
                    and scoring["scoring_calls"] == 1 and scoring["owned_panels"] == 2
                    and returned["evaluation_request_id"] == evaluation["id"] and returned["scoring_request_id"] == scoring["id"]
                    and returned["checkpoint_label"] == label, "Geometry candidate before-call accounting links differ")
                panels = {}
                for mode, det in (("mean", True), ("stochastic", False)):
                    tag = f"{prefix}_c{i:02d}_{mode}"
                    values = panel(tag, 8, det, owner, 622000+100*step)
                    panels[mode] = values
                    event = unique("geometry_panel", panel_id=tag)
                    require(event["episode_names"] == [ep["episode"] for ep in values] and event["actor_hash"] == owner
                        and event["deterministic"] is det and event["raw_refs"] == [ep["raw_ref"] for ep in values]
                        and event["source_returns_used"] is False, "Geometry ordinary-only scored panel differs")
                    _summary_check(dict(outcomes=event["outcomes"], mean_cost=math.fsum(ep["cost"] for ep in values)/8,
                        capture_rate=sum(ep["capture"] for ep in values)/8), values)
                    rng = e.npz(base, event["sampling_rng"])
                    require(int(rng["seed"]) == 624000+100*step and str(rng["actor_hash"]) == owner
                        and bool(rng["deterministic"]) is det and rng["before"].dtype == rng["after"].dtype == np.dtype("uint8")
                        and rng["before"].shape == rng["after"].shape, "Geometry panel noise packet differs")
                    if det:
                        exact(rng["before"], rng["after"], "Deterministic geometry panel consumed noise")
                    first = min(utc(qreq[q]) for ep in values for q in ep["query_ids"])
                    last = max(utc(qdone[q]) for ep in values for q in ep["query_ids"])
                    require(utc(bank) <= utc(evaluation) <= first <= last <= utc(event) <= utc(scoring) <= utc(returned),
                        "Geometry measured score used future/unfrozen behavior")
                    first_times.append(first)
                    last_times.append(last)
                expected_score = measured_score(panels, owner)
                same_tree(returned["score"], expected_score, "Geometry measured complete ordinary score differs")
                same(returned["coordinates"], coordinate, "Geometry measured coordinates differ")
                scores.append(returned["score"])
            digest = numeric_digest(dict(coordinates=coords, owners=owners, geometry_sha256=geometry["arms"][chain]["sha256"]))
            require(bank["checkpoint_labels"] == [f"{prefix}_c{i:02d}" for i in range(13)] and bank["actor_hashes"] == owners
                and bank["proposal_digest"] == selection["proposal_digest"] == digest and utc(bank) <= min(first_times), "Geometry full candidate bank ownership/digest differs")
            require(selection["all_complete_scores"] == scores, "Geometry selector rewrote measured scores")
            decision = select_from_scores(scores)
            same_tree(selection["decision"], decision, "Geometry ordinary-only feasible selector differs")
            selector = unique("geometry_selector_requested", chain=chain, step=step)
            require(selector["iteration_request_id"] == request["id"] and selector["id"] == selection["selector_request_id"]
                and selector["selector_calls"] == 1 and selector["candidate_objective_evaluations"] == 13
                and selector["source_return_weight"] == 0. and selector["selector_input_sha256"] == selection["selector_input_sha256"] == numeric_digest(scores)
                and max(last_times) <= utc(selector), "Geometry selector saw different/unfinished returns")
            selected = coords[decision["selected_index"]]
            changed = not np.array_equal(center, selected)
            same(selection["selected_coordinates"], selected, "Geometry selected center differs")
            require(selection["center_changed"] == changed, "Geometry center-change summary differs")
            chosen_event = unique("geometry_selection", result=selection)
            complete = unique("geometry_search_iteration_complete", chain=chain, step=step)
            require(complete["request_id"] == request["id"] and complete["center_changed"] == changed
                and utc(selector) <= utc(chosen_event) <= utc(complete), "Geometry selection/iteration chronology differs")
            same(complete["final_coordinates"], selected, "Geometry completed coordinates differ")
            center = selected.copy()
            final_owners[chain] = owners[decision["selected_index"]]
            iteration_ends.append(complete)
            step_reports.append(dict(chain=chain, step=step, selected_index=decision["selected_index"],
                center_changed=changed, actor_hash=final_owners[chain], objectives=decision["objectives"], feasible_flags=decision["feasible_flags"]))
        require(entry["final_actor_hash"] == final_owners[chain], "Geometry final chain owner differs")
        same(entry["final_coordinates"], center, "Geometry final chain coordinates differ")
    expected_checkpoints = {"geometry_source_parent"}|{f"geometry_{chain}_s{step}_c{i:02d}" for chain in CHAINS for step in range(3) for i in range(13)}
    require(set(checkpoints) == expected_checkpoints, "Geometry checkpoint roster has extra/missing policies")
    roster = dict(parent=BOOTSTRAP_ACTOR, **final_owners)
    qual = {"mean": {}, "stochastic": {}}
    for mode, det in (("mean", True), ("stochastic", False)):
        for name, owner in roster.items():
            values = panel(f"geometry_qual_{name}_{mode}", 16, det, owner, 631000)
            qual[mode][name] = values
            _summary_check(result["qualification"]["mean_panels" if det else "stochastic_panels"][name], values)
            require(max(utc(r) for r in iteration_ends) <= min(utc(qreq[q]) for ep in values for q in ep["query_ids"]),
                "Geometry qualification preceded frozen final training actors")
    controls = ("fixed", "ordinary", "raw")
    control_assess = {name: assessment(qual["mean"], qual["stochastic"], name, ("parent",)) for name in controls}
    eligible = [name for name in controls if control_assess[name]["passed"]]
    best = "parent" if not eligible else min(eligible, key=lambda name: (sum(ep["cost"] for ep in qual["mean"][name])/16, controls.index(name)))
    long_assess = assessment(qual["mean"], qual["stochastic"], "long", ("parent", best))
    chosen = "long" if long_assess["passed"] else None
    expected_selection = dict(best_control=best, selected_long=chosen, control_parent_fallback=best == "parent",
        control_criteria=control_assess, long_criteria=long_assess)
    same_tree(result["qualification"]["selection"], expected_selection, "Geometry strongest-control qualification differs")
    qselect, qevent = unique("geometry_qualification_selection_requested"), unique("geometry_qualification")
    require(qselect["complete_roster"] == ["parent", *CHAINS] and qselect["selection_calls"] == 1
        and qselect["fresh_final_evidence_not_seen"] is True and qevent["fresh_final_evidence_not_seen"] is True
        and qevent["request_id"] == qselect["id"] and qevent["result"] == result["qualification"]
        and max(utc(qdone[q]) for p in qual.values() for values in p.values() for ep in values for q in ep["query_ids"]) <= utc(qselect) <= utc(qevent),
        "Geometry qualification selection used incomplete/future evidence")
    final_pass = False
    if chosen is None:
        require(result["passed"] is False and result["stage"] == "failed_state_geometry_qualification" and "final" not in result
            and not groups["geometry_final_assessment_requested"] and not groups["geometry_final_assessment_complete"], "Failed geometry qualification did not stop")
    else:
        final_roster = dict(parent=BOOTSTRAP_ACTOR, guided=final_owners["long"], best_control=roster[best])
        require(result["final"]["actor_hashes"] == final_roster, "Geometry fresh-final actor ownership changed")
        final = {"mean": {}, "stochastic": {}}
        for mode, det in (("mean", True), ("stochastic", False)):
            for name, owner in final_roster.items():
                values = panel(f"geometry_final_{name}_{mode}", 64, det, owner, 632000)
                final[mode][name] = values
                _summary_check(result["final"]["mean_panels" if det else "stochastic_panels"][name], values)
                require(utc(qevent) <= min(utc(qreq[q]) for ep in values for q in ep["query_ids"]), "Geometry final evidence preceded actor/control selection")
        criterion = assessment(final["mean"], final["stochastic"], "guided", ("parent", "best_control"))
        same_tree(result["final"]["criteria"], criterion, "Geometry final complete-return gate differs")
        req, done = unique("geometry_final_assessment_requested"), unique("geometry_final_assessment_complete")
        require(req["assessment_calls"] == 1 and done["request_id"] == req["id"] and done["criteria"] == result["final"]["criteria"]
            and max(utc(qdone[q]) for p in final.values() for values in p.values() for ep in values for q in ep["query_ids"]) <= utc(req) <= utc(done),
            "Geometry final assessment preceded complete actual returns")
        final_pass = criterion["passed"]
        require(result["passed"] == final_pass and result["stage"] == ("complete" if final_pass else "failed_fresh_state_geometry_improvement"),
            "Geometry scientific verdict differs")
    require(set(by_name) == used and set(qreq) == all_ids, "Geometry queries/episodes outside fixed panels")
    expected_events = {"geometry_basis_requested": 1, "geometry_basis_complete": 1, "geometry_features_requested": 1,
        "geometry_features_complete": 1, "geometry_svd_requested": 4, "geometry_svd_complete": 4,
        "geometry_search_iteration_requested": 12, "geometry_search_iteration_complete": 12,
        "geometry_proposal_version_requested": 156, "geometry_proposal_version_complete": 156,
        "geometry_candidate_evaluation_requested": 156, "geometry_candidate_return": 156,
        "geometry_candidate_score_requested": 156, "geometry_selector_requested": 12, "geometry_selection": 12,
        "geometry_panel": 312, "geometry_qualification_selection_requested": 1, "geometry_qualification": 1,
        "geometry_final_assessment_requested": int(chosen is not None), "geometry_final_assessment_complete": int(chosen is not None)}
    for kind, count in expected_events.items():
        require(len(groups[kind]) == count, "Geometry missing/extra numerical work event: "+kind)
    numerical = {name: len(groups["geometry_"+kind]) for name, kind in (
        ("basis_qr_requested", "basis_requested"), ("basis_qr_completed", "basis_complete"),
        ("feature_forwards_requested", "features_requested"), ("feature_forwards_completed", "features_complete"),
        ("geometry_svds_requested", "svd_requested"), ("geometry_svds_completed", "svd_complete"),
        ("search_iterations_requested", "search_iteration_requested"), ("search_iterations_completed", "search_iteration_complete"),
        ("proposal_versions_requested", "proposal_version_requested"), ("proposal_versions_constructed", "proposal_version_complete"),
        ("candidate_evaluations_requested", "candidate_evaluation_requested"), ("candidate_evaluations_completed", "candidate_return"),
        ("candidate_scoring_calls_requested", "candidate_score_requested"), ("candidate_scoring_calls_completed", "candidate_return"),
        ("selector_calls_requested", "selector_requested"), ("selector_calls_completed", "selection"),
        ("qualification_selections_requested", "qualification_selection_requested"), ("qualification_selections_completed", "qualification"),
        ("final_assessments_requested", "final_assessment_requested"), ("final_assessments_completed", "final_assessment_complete"))}
    require(result["numerical_search_work"] == numerical, "Geometry numerical work summary differs")
    actual_controls = 200+62400+4000+(9600 if chosen is not None else 0)
    require(sum(q["requested_controls"] for q in qreq.values()) == actual_controls
        and result["accounting"]["actor_rows"] == actual_controls+288
        and result["accounting"]["actor_calls"] == actual_controls+1
        and len(episodes) == 8+2496+160+(384 if chosen is not None else 0), "Geometry actual work differs from complete fixed schedule")
    cross = result["cost_crosswalk"]
    require(cross["new_assay"] == result["accounting"] and cross["early_ancestor"] == provenance["reused_early_ancestry"]
        and cross["v3_ancestor"] == provenance["reused_v3_ancestry"], "Geometry paid ancestry crosswalk differs")
    for key in ("requested_scalar_controls", "actor_rows", "adam_attempts"):
        require(cross["unique_lineage_"+key] == sum(row[key] for row in (result["accounting"], provenance["reused_early_ancestry"], provenance["reused_v3_ancestry"])),
            "Geometry paid ancestry was duplicated/omitted")
    return dict(complete=True, scientific_success=final_pass, selected_long=chosen, best_control=best,
        exact_parameter_versions_verified=156, search_steps=step_reports, numerical_search_work=numerical,
        complete_owned_training_episodes=2496, ordinary_calibration_episodes=8, ordinary_qualification_episodes=160,
        fresh_final_episodes=384 if chosen is not None else 0, actual_scalar_controls=actual_controls,
        actual_actor_rows=actual_controls+288, metered_parent_feature_forward_rows=288,
        analytic_reused_episode_transition_row_checks=reused_checks, reset_RNG_checks=len(episodes),
        analytic_geometry_work=dict(qr_calls=1, svd_calls=4, projected_calibration_rows=4*192,
            projection_scalar_products=4*192*4, projection_multiply_terms=4*192*4*128),
        frozen_trunk_and_private_rng=True, teacher_actions_and_returns_not_geometry_inputs=True,
        hidden_features_independently_recomputed=False, audit_actor_forward_rows=0, audit_physics_controls=0,
        audit_optimizer_steps=0, correlated_identical_candidate_repeats_are_not_independent_evidence=True,
        limits=["Hidden features and saved parent outputs are input/owner/checkpoint/source/hook/chronology bound; no neural layer or full actor was independently evaluated",
            "QR/SVD and statistical feature projections are audit arithmetic, separately counted from zero audit actor-forward rows",
            "Stochastic actions/noise draws are source/owner bound, not independently forwarded or regenerated",
            "Arithmetic PASS does not establish full GPS, physical Acrobot discovery, two beneficial feedback rounds or sample efficiency"])
