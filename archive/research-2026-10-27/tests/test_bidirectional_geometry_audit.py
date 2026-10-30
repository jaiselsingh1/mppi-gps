"""Synthetic independent geometry-audit fixtures, authored without execution.

Only NumPy and the pure audit module are imported. Checkpoints and measurements
are dictionaries, not actors; no neural layer, forward, engine or solver runs.
"""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

import src.gps.bidirectional.geometry_audit as geometry_audit
from src.gps.bidirectional.geometry_audit import (
    BOOTSTRAP_ACTOR, assessment, bind_feature_verification, calibration_indices, candidate_coordinates,
    declared_bound, geometry_arrays, make_basis, select_from_scores,
    verify_feature_arrays, verify_head_actor,
)


def support(scales=(2., 1., .5, .25), means=(0., 0., 0., 0.)):
    # Distinct orthogonal Fourier columns give independent known singular RMS.
    phase = np.arange(192)*2*np.pi/192
    h = np.zeros((192, 128))
    for i in range(4):
        h[:, i] = np.sqrt(2)*np.sin((i+1)*phase)*scales[i]+means[i]
    basis = np.zeros((4, 128))
    basis[:, :4] = np.eye(4)
    return h, basis


def actor(dtype=np.float32):
    values = {
        "fc1.weight": np.zeros((128, 10), dtype=dtype),
        "fc1.bias": np.zeros(128, dtype=dtype),
        "fc2.weight": np.zeros((128, 128), dtype=dtype),
        "fc2.bias": np.zeros(128, dtype=dtype),
        "fc_mean.weight": np.linspace(-.02, .02, 128, dtype=dtype)[None],
        "fc_mean.bias": np.array([.03], dtype=dtype),
        "log_std": np.array([-2.], dtype=dtype),
        "action_low": np.array([-1.], dtype=dtype),
        "action_high": np.array([1.], dtype=dtype),
        "action_scale": np.array([1.], dtype=dtype),
        "action_bias": np.array([0.], dtype=dtype),
    }
    return dict(format_version=1, policy_type="bidirectional-family-gaussian-v1",
        obs_dim=8, hidden_dim=128, num_families=2, action_low=[-1.], action_high=[1.],
        actor=values, sampling_rng_state=np.arange(32, dtype=np.uint8), policy_hash="synthetic-only")


def candidate(parent, geometry, coordinates):
    result = deepcopy(parent)
    c = np.asarray(coordinates, dtype=np.float64)
    delta = {"fc_mean.weight": c[1:5]@geometry["weight_axes"],
        "fc_mean.bias": .1*c[0]+c[1:5]@geometry["bias_axes"], "log_std": .5*c[5]}
    for name, change in delta.items():
        original = parent["actor"][name]
        result["actor"][name] = (original.astype(np.float64)+change).astype(original.dtype)
    return result


def feature_packet(parent):
    inputs = dict(observations=np.zeros((288, 8)), states=np.zeros((288, 2)),
        families=np.zeros(288, dtype=np.int64), row_references_json=np.array("[]"),
        actor_hash=np.array(BOOTSTRAP_ACTOR), teacher_action_labels=np.array(False))
    feature = dict(hidden=np.full((288, 128), .125, dtype=np.float32),
        observations=inputs["observations"].copy(), states=inputs["states"].copy(),
        parent_pre_squash_means=np.full((288, 1), .75, dtype=np.float32),
        parent_log_std=np.full((288, 1), -2., dtype=np.float32),
        sampling_rng_before=parent["sampling_rng_state"].copy(),
        sampling_rng_after=parent["sampling_rng_state"].copy(),
        input_raw_json=np.array(json.dumps({})), actor_hash=np.array(BOOTSTRAP_ACTOR))
    return inputs, feature


def scores():
    result = []
    for i in range(13):
        ordinary = {mode: dict(mean_cost=2., captures=6, capture_flags=[True]*6+[False]*2,
            context_ids=[f"ordinary:{j}" for j in range(8)]) for mode in ("mean", "stochastic")}
        result.append(dict(actor_hash=f"synthetic-{i}", ordinary=ordinary))
    return result


def outcomes(name, cost, captures):
    return [dict(context_id=f"ordinary:{i}", cost=cost, capture=bool(value)) for i, value in enumerate(captures)]


def test_work_bound_counts_calibration_feature_query_and_every_paid_candidate():
    bound = declared_bound()
    assert bound["training_actor_versions"] == 4*3*13 == 156
    assert bound["training_controls"] == 4*3*13*8*2*25 == 62400
    assert bound["calibration_controls"] == 8*25 == 200
    assert bound["qualification_controls"] == 5*16*2*25 == 4000
    assert bound["final_controls"] == 3*64*2*25 == 9600
    assert bound["total_controls"] == 200+62400+4000+9600 == 76200
    assert bound["actor_rows"] == bound["total_controls"]+bound["feature_forward_rows"] == 76488
    assert bound["model_controls"] == bound["adam_steps"] == bound["diagnostic_backward_passes"] == 0


def test_basis_uses_exact_private_seed_and_sign_canonical_qr():
    draw = np.random.Generator(np.random.PCG64(1701)).standard_normal((128, 4))
    q, r = np.linalg.qr(draw, mode="reduced")
    expected = (q*np.where(np.diag(r) < 0, -1., 1.)[None]).T
    np.testing.assert_array_equal(make_basis(), expected)
    np.testing.assert_allclose(expected@expected.T, np.eye(4), rtol=0, atol=1e-14)
    assert np.all(np.diag(expected@draw) > 0)


@pytest.mark.parametrize("mode,offset", [("fixed", 0), ("ordinary", 0), ("raw", 96), ("long", 192)])
def test_calibration_indices_are_exactly192_with_explicit_ordinary_duplicates(mode, offset):
    actual = calibration_indices(mode)
    np.testing.assert_array_equal(actual, np.r_[np.arange(96), np.arange(offset, offset+96)])
    assert actual.shape == (192,)


def test_centering_and_known_singular_rms_produce_orthogonal_point1_responses():
    h, basis = support(means=(.3, -.2, .1, -.05))
    geometry = geometry_arrays(h, basis, "long")
    response = h@geometry["weight_axes"].T+geometry["bias_axes"]
    np.testing.assert_allclose(response.mean(axis=0), np.zeros(4), rtol=0, atol=2e-15)
    np.testing.assert_allclose(response.T@response/192, .01*np.eye(4), rtol=0, atol=2e-15)
    np.testing.assert_allclose(geometry["diagnostics"]["singular_rms"], [2., 1., .5, .25], rtol=0, atol=2e-15)
    assert geometry["diagnostics"]["limiting_caps"] == ["target_rms_.1"]*4
    for vector in geometry["diagnostics"]["right_singular_vectors"]:
        assert vector[np.argmax(np.abs(vector))] >= 0.


def test_fixed_axes_ignore_svd_calibration_changes_and_have_zero_bias():
    h, basis = support()
    first, second = geometry_arrays(h, basis, "fixed"), geometry_arrays(h+100., basis, "fixed")
    np.testing.assert_array_equal(first["weight_axes"], .1*basis)
    np.testing.assert_array_equal(first["weight_axes"], second["weight_axes"])
    np.testing.assert_array_equal(first["bias_axes"], np.zeros(4))
    assert first["diagnostics"]["singular_vectors_used_for_axes"] is False


def test_below_cutoff_axes_stay_zero_without_replacement_or_rescaling():
    h, basis = support(scales=(1., .1, .0005, 0.))
    geometry = geometry_arrays(h, basis, "raw")
    assert geometry["diagnostics"]["active_axes"] == [True, True, False, False]
    np.testing.assert_array_equal(geometry["weight_axes"][2:], np.zeros((2, 128)))
    np.testing.assert_array_equal(geometry["bias_axes"][2:], np.zeros(2))
    assert geometry["diagnostics"]["active_parameter_axes"] == 2


def test_bias_cap_limits_achieved_rms_without_hiding_the_shortfall():
    h, basis = support(means=(100., 0., 0., 0.))
    geometry = geometry_arrays(h, basis, "long")
    assert geometry["diagnostics"]["limiting_caps"][0] == "bias_abs_.5"
    assert abs(geometry["bias_axes"][0]) == pytest.approx(.5)
    assert geometry["diagnostics"]["achieved_pre_squash_rms"][0] == pytest.approx(.01)
    assert geometry["diagnostics"]["achieved_pre_squash_rms"][0] < .1


def test_weight_cap_is_one_and_preserves_smaller_achieved_singular_rms():
    h, basis = support(scales=(.05, .025, .01, .002))
    geometry = geometry_arrays(h, basis, "ordinary")
    np.testing.assert_allclose(np.linalg.norm(geometry["weight_axes"], axis=1), np.ones(4), rtol=0, atol=1e-14)
    assert geometry["diagnostics"]["limiting_caps"] == ["weight_norm1"]*4
    np.testing.assert_allclose(geometry["diagnostics"]["achieved_pre_squash_rms"], [.05, .025, .01, .002], rtol=0, atol=1e-14)


@pytest.mark.parametrize("corruption", ["rows", "nonfinite", "basis", "mode"])
def test_geometry_rejects_undeclared_shape_nonfinite_basis_or_source(corruption):
    h, basis = support()
    mode = "ordinary"
    if corruption == "rows":
        h = h[:-1]
    elif corruption == "nonfinite":
        h[0, 0] = np.nan
    elif corruption == "basis":
        basis[0] *= 2
    else:
        mode = "other"
    with pytest.raises(ValueError):
        geometry_arrays(h, basis, mode)


def test_candidate_bank_keeps_center_first_and_inactive_axis_duplicate_slots():
    center = np.array([1., -1., 0., 0., 0., 0.])
    bank = candidate_coordinates(center, 2)
    assert bank.shape == (13, 6)
    np.testing.assert_array_equal(bank[0], center)
    np.testing.assert_array_equal(bank[1]-center, [-1., 0., 0., 0., 0., 0.])
    np.testing.assert_array_equal(bank[2]-center, [1., 0., 0., 0., 0., 0.])
    np.testing.assert_array_equal(bank[-1]-center, [0., 0., 0., 0., 0., 1.])
    assert np.all(np.abs(bank).sum(axis=1) <= 3)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_absolute_head_update_includes_geometry_bias_compensation_and_original_cast(dtype):
    parent = actor(dtype)
    h, basis = support(means=(.3, -.2, .1, -.05))
    geometry = geometry_arrays(h, basis, "long")
    coordinates = [1., .5, -.25, .25, -.25, .5]
    proposed = candidate(parent, geometry, coordinates)
    assert verify_head_actor(parent, proposed, geometry, coordinates) is True
    for name in parent["actor"]:
        if name not in ("fc_mean.bias", "fc_mean.weight", "log_std"):
            assert proposed["actor"][name].tobytes() == parent["actor"][name].tobytes()


@pytest.mark.parametrize("corruption", ["trunk", "head_ulp", "bias_compensation", "buffer_signed_zero", "rng_dtype"])
def test_proposals_reject_any_unowned_parameter_buffer_or_private_rng_change(corruption):
    parent = actor()
    h, basis = support(means=(.3, -.2, .1, -.05))
    geometry = geometry_arrays(h, basis, "long")
    coordinates = np.array([0., 1., 0., 0., 0., 0.])
    proposed = candidate(parent, geometry, coordinates)
    if corruption == "trunk":
        proposed["actor"]["fc2.weight"][0, 0] = .001
    elif corruption == "head_ulp":
        value = proposed["actor"]["fc_mean.weight"][0, 0]
        proposed["actor"]["fc_mean.weight"][0, 0] = np.nextafter(value, np.float32(np.inf))
    elif corruption == "bias_compensation":
        proposed["actor"]["fc_mean.bias"] = parent["actor"]["fc_mean.bias"].copy()
    elif corruption == "buffer_signed_zero":
        proposed["actor"]["action_bias"][0] = -0.
        assert np.array_equal(proposed["actor"]["action_bias"], parent["actor"]["action_bias"])
    else:
        proposed["sampling_rng_state"] = proposed["sampling_rng_state"].astype(np.int64)
        assert np.array_equal(proposed["sampling_rng_state"], parent["sampling_rng_state"])
    with pytest.raises(ValueError):
        verify_head_actor(parent, proposed, geometry, coordinates)


def test_feature_packet_validates_measurement_metadata_without_evaluating_a_network():
    parent = actor()
    inputs, feature = feature_packet(parent)
    assert verify_feature_arrays(feature, inputs, parent) is True
    np.testing.assert_array_equal(feature["sampling_rng_before"], feature["sampling_rng_after"])


@pytest.mark.parametrize("corruption", ["families", "input", "hidden_range", "hidden_dtype", "output_shape", "rng_dtype", "labels", "owner"])
def test_feature_packet_rejects_changed_support_measurement_contract_or_rng(corruption):
    parent = actor()
    inputs, feature = feature_packet(parent)
    if corruption == "families":
        inputs["families"][0] = 1
    elif corruption == "input":
        feature["observations"][0, 0] = .1
    elif corruption == "hidden_range":
        feature["hidden"][0, 0] = 1.1
    elif corruption == "hidden_dtype":
        feature["hidden"] = feature["hidden"].astype(np.float64)
    elif corruption == "output_shape":
        feature["parent_pre_squash_means"] = np.zeros((288, 2), dtype=np.float32)
    elif corruption == "rng_dtype":
        feature["sampling_rng_after"] = feature["sampling_rng_after"].astype(np.int64)
    elif corruption == "labels":
        inputs["teacher_action_labels"] = np.array(True)
    else:
        feature["actor_hash"] = np.array("foreign")
    with pytest.raises(ValueError):
        verify_feature_arrays(feature, inputs, parent)


@pytest.mark.parametrize("mode", ["mean", "stochastic"])
def test_selector_rejects_each_mode_cost_regression_before_joint_argmin(mode):
    values = scores()
    values[1]["ordinary"]["mean"]["mean_cost"] = 1.
    values[1]["ordinary"]["stochastic"]["mean_cost"] = 1.
    values[1]["ordinary"][mode]["mean_cost"] = 2.01
    result = select_from_scores(values)
    assert not result["feasible_flags"][1] and result["selected_index"] == 0


def test_selector_allows_paired_capture_tradeoffs_under_aggregate_nonregression():
    values = scores()
    for mode in ("mean", "stochastic"):
        values[2]["ordinary"][mode]["mean_cost"] = 1.9
        values[2]["ordinary"][mode]["capture_flags"] = [False]+[True]*6+[False]
    result = select_from_scores(values)
    assert result["selected_index"] == 2 and result["feasible_flags"][2]
    assert result["paired_capture_changes"][2] == {"mean": dict(lost=1, gained=1), "stochastic": dict(lost=1, gained=1)}
    assert result["objectives"][2] == pytest.approx(1.9/25)


def test_selector_keeps_stable_center_first_ties_and_checks_capture_counts():
    values = scores()
    assert select_from_scores(values)["selected_index"] == 0
    values[1]["ordinary"]["mean"]["captures"] = 5
    with pytest.raises(ValueError):
        select_from_scores(values)


def test_assessment_requires_both_modes_and_stronger_control_but_reports_capture_losses():
    baseline = [True]*15+[False]
    tradeoff = [False]+[True]*15
    mean = dict(parent=outcomes("parent", 2., baseline), control=outcomes("control", 1.7, baseline),
        long=outcomes("long", 1.8, tradeoff))
    noisy = deepcopy(mean)
    failed = assessment(mean, noisy, "long", ("parent", "control"))
    assert failed["passed"] is False
    assert failed["comparisons"]["mean"]["parent"]["paired_capture_losses"] == 1
    assert failed["comparisons"]["mean"]["parent"]["paired_capture_gains"] == 1
    for row in mean["long"]:
        row["cost"] = 1.6
    for row in noisy["long"]:
        row["cost"] = 1.6
    assert assessment(mean, noisy, "long", ("parent", "control"))["passed"] is True
    noisy["long"][0]["cost"] = 10.
    assert assessment(mean, noisy, "long", ("parent", "control"))["passed"] is False


def synthetic_receipt(tmp_path, monkeypatch):
    """Entirely synthetic bytes; no checkpoint or forward archive is decoded."""
    root, packet = tmp_path/"geometry-run", tmp_path/"feature-verification"
    root.mkdir()
    packet.mkdir()
    def sha(value):
        return hashlib.sha256(value).hexdigest()
    def put(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return sha(data)
    def document(path, value):
        return put(path, (json.dumps(value, sort_keys=True)+"\n").encode())
    verifier = b"# Synthetic source identity fixture; never executed.\n"
    bootstrap = b"synthetic checkpoint bytes, never deserialized\n"
    source_sha = put(packet/"verify.py", verifier)
    monkeypatch.setattr(geometry_audit, "FEATURE_VERIFICATION_SOURCE", source_sha)
    monkeypatch.setattr(geometry_audit, "BOOTSTRAP_SHA", sha(bootstrap))
    features = b"synthetic feature measurement bytes, never decoded\n"
    inputs = b"synthetic input measurement bytes, never decoded\n"
    input_sha = put(root/"evidence/assay/geometry/feature_inputs.npz", inputs)
    feature_sha = put(root/"evidence/assay/geometry/features.npz", features)
    put(root/"inputs/inputs/checkpoints/bootstrap.pt", bootstrap)
    config_sha = document(root/"config.json", {"synthetic": True})
    manifest_sha = document(root/"manifest.json", {"dependency_versions": {"torch": "synthetic-torch", "numpy": "synthetic-numpy"}})
    result = dict(manifest_sha256=manifest_sha, config_sha256=config_sha, geometry={
        "inputs": dict(file="geometry/feature_inputs.npz", sha256=input_sha, bytes=len(inputs)),
        "features": dict(file="geometry/features.npz", sha256=feature_sha, bytes=len(features))})
    document(root/"assay_result.json", result)
    names = ("assay_result.json", "manifest.json", "config.json", "inputs/inputs/checkpoints/bootstrap.pt",
        "evidence/assay/geometry/feature_inputs.npz", "evidence/assay/geometry/features.npz")
    bindings = {"synthetic-namespace/geometry-run/"+name: sha((root/name).read_bytes()) for name in names}
    manifest = dict(schema="independent-paid-feature-verification-source-v1", source_sha256=source_sha,
        input_sha256=bindings, max_actor_rows=288, max_optimizer_steps=0, max_physics_controls=0,
        max_scalar_controls=0, numerical_runtime=dict(torch="synthetic-torch", numpy="synthetic-numpy", threads=1))
    manifest_sha = document(packet/"manifest.json", manifest)
    ledger = [dict(id="work-000000000", kind="verification_requested", wall_seconds=.1,
        rows=288, actor_hash=BOOTSTRAP_ACTOR, source_sha256=source_sha),
        dict(id="work-000000001", kind="neural_requested", wall_seconds=.2, rows=288,
            actor_hash=BOOTSTRAP_ACTOR, scope="independent_functional_feature_replay", observation_shape=[288, 8],
            operations="twoLinearTanhplusmeanLinearandclampedlogstd", extra_forward_rows=0),
        dict(id="work-000000002", kind="neural_complete", wall_seconds=.3, rows=288,
            request_id="work-000000001", returned=True)]
    ledger_sha = put(packet/"work_ledger.jsonl", b"".join((json.dumps(row, sort_keys=True)+"\n").encode() for row in ledger))
    output_sha = put(packet/"verified_forward.npz", b"synthetic verification output, never decoded\n")
    receipt = dict(schema="independent-paid-feature-verification-v1", passed=True, inputs_unchanged=True,
        blocked_import_attempts=[], actor_rows_requested=288, actor_rows_completed=288,
        new_physics_controls=0, new_scalar_controls=0, optimizer_updates=0, backward_calls=0,
        source_sha256=source_sha, manifest_sha256=manifest_sha, work_ledger_sha256=ledger_sha,
        output_sha256=output_sha, wall_seconds=1., checks={name: dict(exactly_equal=True, max_abs_difference=0.)
            for name in ("hidden", "parent_pre_squash_means", "parent_log_std")})
    document(packet/"stage_complete.json", receipt)
    evidence = SimpleNamespace(root=root, hashes={})
    return packet, evidence, result, manifest, receipt, ledger, document


def test_paid_receipt_binds_exact_inputs_source_and_one_separate288row_attempt(tmp_path, monkeypatch):
    packet, evidence, result, _, _, _, _ = synthetic_receipt(tmp_path, monkeypatch)
    report = bind_feature_verification(packet, evidence, result)
    assert report["status"] == "PASS" and report["pending"] == report["errors"] == []
    assert report["actor_rows_requested"] == report["actor_rows_completed"] == 288
    assert report["actor_rows_unknown"] == report["audit_actor_forward_rows"] == 0
    assert report["hidden_features_independently_verified_by_separate_paid_packet"] is True
    assert len(report["input_sha256"]) == 11  # Five receipt files and exactly six producer inputs.
    assert len(evidence.hashes) == 11
    assert report["new_physics_controls"] == report["new_scalar_controls"] == report["optimizer_updates"] == 0


def test_missing_receipt_retains_observed_paid_rows_and_returns_pending(tmp_path, monkeypatch):
    packet, evidence, result, _, _, _, _ = synthetic_receipt(tmp_path, monkeypatch)
    (packet/"stage_complete.json").unlink()
    report = bind_feature_verification(packet, evidence, result)
    assert report["status"] == "PENDING" and report["pending"]
    assert report["actor_rows_requested"] == report["actor_rows_completed"] == 288


def test_interrupted_verification_keeps_requested288_unknown_without_completion(tmp_path, monkeypatch):
    packet, evidence, result, _, _, ledger, _ = synthetic_receipt(tmp_path, monkeypatch)
    (packet/"stage_complete.json").unlink()
    (packet/"work_ledger.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger[:2]))
    report = bind_feature_verification(packet, evidence, result)
    assert report["status"] == "PENDING"
    assert report["actor_rows_requested"] == report["actor_rows_unknown"] == 288
    assert report["actor_rows_completed"] == 0


@pytest.mark.parametrize("corruption", ["receipt_count", "input_hash", "mixed_namespace", "output_check", "source_pin"])
def test_paid_receipt_rejects_false_counts_changed_bindings_or_self_consistent_other_verifier(tmp_path, monkeypatch, corruption):
    packet, evidence, result, manifest, receipt, ledger, document = synthetic_receipt(tmp_path, monkeypatch)
    if corruption == "receipt_count":
        receipt["actor_rows_completed"] = 576
    elif corruption == "input_hash":
        key = next(name for name in manifest["input_sha256"] if name.endswith("geometry/features.npz"))
        manifest["input_sha256"][key] = "0"*64
    elif corruption == "mixed_namespace":
        key = next(name for name in manifest["input_sha256"] if name.endswith("geometry/features.npz"))
        manifest["input_sha256"]["other-namespace/"+key] = manifest["input_sha256"].pop(key)
    elif corruption == "output_check":
        receipt["checks"]["hidden"]["exactly_equal"] = False
    else:
        other = b"# Different self-attested verifier, never executed.\n"
        (packet/"verify.py").write_bytes(other)
        changed = hashlib.sha256(other).hexdigest()
        manifest["source_sha256"] = receipt["source_sha256"] = ledger[0]["source_sha256"] = changed
        (packet/"work_ledger.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger))
        receipt["work_ledger_sha256"] = hashlib.sha256((packet/"work_ledger.jsonl").read_bytes()).hexdigest()
    receipt["manifest_sha256"] = document(packet/"manifest.json", manifest)
    document(packet/"stage_complete.json", receipt)
    with pytest.raises(ValueError):
        bind_feature_verification(packet, evidence, result)


@pytest.mark.parametrize("corruption", ["resolution", "clock", "extra_forward"])
def test_paid_receipt_rejects_unbound_reversed_or_repeated_neural_work(tmp_path, monkeypatch, corruption):
    packet, evidence, result, _, receipt, ledger, document = synthetic_receipt(tmp_path, monkeypatch)
    if corruption == "resolution":
        ledger[2]["request_id"] = ledger[0]["id"]
    elif corruption == "clock":
        ledger[2]["wall_seconds"] = .15
    else:
        ledger.append(dict(ledger[1], id="work-000000003", wall_seconds=.4))
    (packet/"work_ledger.jsonl").write_text("".join(json.dumps(row)+"\n" for row in ledger))
    receipt["work_ledger_sha256"] = hashlib.sha256((packet/"work_ledger.jsonl").read_bytes()).hexdigest()
    document(packet/"stage_complete.json", receipt)
    with pytest.raises(ValueError):
        bind_feature_verification(packet, evidence, result)
