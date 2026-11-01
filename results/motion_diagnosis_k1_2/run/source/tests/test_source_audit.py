"""S1 saved-only proof fixtures; no simulator, network or optimizer arithmetic."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.gps.mppi_gps.audits import native
from src.gps.mppi_gps.audits.compatibility import recompute_agreement
from src.gps.mppi_gps.audits.source import (
    SourceAudit,
    array_binding,
    audit_fit,
    overlap_counts,
    prove_actual_initial,
    read_arrays,
    recompute_walking_screen,
    stage_ledger,
)

CONFIG = dict(walking_screen=dict(minimum_additional_successes=2, minimum_positive_pairs=2))


def panel(left, right, survival=None):
    result = {}
    for pair, (first, second) in enumerate(zip(left, right, strict=True)):
        result[f"mppi_s{pair}"] = dict(
            summary=dict(successes=first, survived=16, mean_cost=1000, mean_steps=500)
        )
        result[f"sac_reference_s{pair}"] = dict(
            summary=dict(
                successes=second,
                survived=16 if survival is None else survival[pair],
                mean_cost=1000,
                mean_steps=500,
            )
        )
    return result


def test_source_screen_requires_two_two_success_gains_and_no_count_regression():
    good = panel([0, 1, 2], [2, 3, 2])
    result = recompute_walking_screen(good, CONFIG)
    assert result["walking_screen"] is True
    assert [row["walking_gain"] for row in result["pairs"]] == [2, 2, 0]
    assert recompute_walking_screen(panel([0, 1, 2], [1, 3, 2]), CONFIG)["walking_screen"] is False
    assert recompute_walking_screen(panel([0, 1, 2], [2, 3, 1]), CONFIG)["walking_screen"] is False
    assert (
        recompute_walking_screen(panel([0, 1, 2], [2, 3, 2], [16, 16, 15]), CONFIG)[
            "walking_screen"
        ]
        is False
    )


def test_retention_or_cost_without_walking_cannot_pass_source_screen():
    values = panel([0, 0, 0], [0, 0, 0])
    for pair in range(3):
        values[f"sac_reference_s{pair}"]["summary"].update(mean_steps=1000, mean_cost=1)
        values[f"mppi_s{pair}"]["summary"].update(mean_steps=10, mean_cost=10000)
    assert recompute_walking_screen(values, CONFIG)["walking_screen"] is False


def test_effective_input_overlap_treats_signed_zero_as_same_but_not_labels():
    first = np.array([[0.0, 1.0], [2, 3]], dtype=np.float32)
    second = np.array([[-0.0, 1.0], [4, 5]], dtype=np.float32)
    assert overlap_counts(first, second) == dict(
        train_unique_rows=2, validation_unique_rows=2, cross_split_overlap=1
    )
    assert overlap_counts(first, second + 10)["cross_split_overlap"] == 0


def test_portable_array_source_bytes_and_dtype_tampering_are_detected(tmp_path):
    values = dict(observations=np.zeros((2, 17)), actions=np.ones((2, 6), dtype=np.float32))
    path = tmp_path / "data.npz"
    np.savez(path, **values)
    binding = dict(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        arrays={key: array_binding(value) for key, value in values.items()},
    )
    loaded = read_arrays(
        SourceAudit(tmp_path, tmp_path), tmp_path, "data.npz", dict(files={"data.npz": binding})
    )
    assert loaded["actions"].dtype == np.float32
    changed = deepcopy(binding)
    changed["arrays"]["actions"]["dtype"] = "<f8"
    with pytest.raises(AssertionError, match="dtype"):
        read_arrays(
            SourceAudit(tmp_path, tmp_path), tmp_path, "data.npz", dict(files={"data.npz": changed})
        )


def initial_fixture():
    state = {"network.0.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    private = torch.tensor([1, 2, 3], dtype=torch.uint8)
    rng = np.random.Generator(np.random.PCG64(3)).bit_generator.state
    adam = dict(
        state={},
        param_groups=[
            dict(
                params=list(range(6)),
                lr=0.0003,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
                foreach=False,
                fused=False,
            )
        ],
    )
    payload = dict(state_dict=state, adam=adam, rng=rng, private_rng_state=private)
    incoming = dict(
        policy_hash=native.tensor_hash(state), state_dict=state, private_rng_state=private
    )
    fingerprints = dict(
        policy_hash=incoming["policy_hash"],
        optimizer_fingerprint=native.fingerprint(adam),
        rng_fingerprint=native.fingerprint(rng),
        private_rng_fingerprint=native.fingerprint(private),
    )
    return payload, incoming, fingerprints, rng


def test_actual_initial_state_proves_tensors_empty_adam_and_both_rngs(tmp_path):
    payload, incoming, fingerprints, rng = initial_fixture()
    prove_actual_initial(SourceAudit(tmp_path, tmp_path), payload, incoming, fingerprints, rng)
    changed = deepcopy(payload)
    changed["state_dict"]["network.0.weight"][0, 0] += 1
    with pytest.raises(AssertionError, match="theta"):
        prove_actual_initial(SourceAudit(tmp_path, tmp_path), changed, incoming, fingerprints, rng)
    changed = deepcopy(payload)
    changed["adam"]["state"][0] = dict(step=0)
    with pytest.raises(AssertionError, match="empty"):
        prove_actual_initial(SourceAudit(tmp_path, tmp_path), changed, incoming, fingerprints, rng)
    changed = deepcopy(payload)
    changed["private_rng_state"][0] += 1
    with pytest.raises(AssertionError, match="initialization RNG"):
        prove_actual_initial(SourceAudit(tmp_path, tmp_path), changed, incoming, fingerprints, rng)


def test_source_ledger_recounts_learning_and_requires_closed_boundaries(tmp_path):
    work = dict(
        requested_backward_rows=256,
        completed_backward_rows=256,
        backward_rows=256,
        measurement_requests=1,
        snapshots=0,
        wall_seconds=0.4,
    )
    events = [
        dict(event="runtime_open"),
        dict(event="measurement_requested", measure_kind="backward_rows", amount=256),
        dict(
            event="measurement_complete",
            measure_kind="backward_rows",
            amount=256,
            request_id="event-000000001",
        ),
        dict(event="runtime_close", totals=work),
    ]

    def save():
        for index, row in enumerate(events):
            row.update(id=f"event-{index:09d}", kind=row["event"], wall_seconds=index * 0.1)
        (tmp_path / "ledger.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))

    with h5py.File(tmp_path / "raw.h5", "w"):
        pass
    save()
    assert (
        stage_ledger(SourceAudit(tmp_path, tmp_path), tmp_path, work)["counts"][
            "completed_backward_rows"
        ]
        == 256
    )
    wrong = dict(work, completed_backward_rows=512)
    with pytest.raises(AssertionError, match="census"):
        stage_ledger(SourceAudit(tmp_path, tmp_path), tmp_path, wrong)
    events.pop(2)
    save()
    with pytest.raises(AssertionError, match="Closed"):
        stage_ledger(SourceAudit(tmp_path, tmp_path), tmp_path, work)


def test_native_episode_raw_alias_maps_only_explicit_eval_phase(tmp_path):
    path = tmp_path / "evidence/eval/raw.h5"
    path.parent.mkdir(parents=True)
    values = np.arange(6, dtype=np.float64)
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    meta = dict(shape=[6], dtype=values.dtype.str, sha256=digest, encoding="array")
    ref = dict(file="raw.h5", group="arrays/test", tag="test", arrays=dict(values=meta))
    with h5py.File(path, "w") as stream:
        group = stream.create_group(ref["group"])
        group.attrs["tag"] = "test"
        array = group.create_dataset("values", data=values)
        array.attrs.update(numpy_dtype=meta["dtype"], encoding="array", sha256=digest)
    a = SourceAudit(tmp_path, tmp_path)
    assert np.array_equal(a.group(path, ref)["values"], values)
    assert np.array_equal(a.group(path.parent / "evidence/raw.h5", ref)["values"], values)
    assert len(a.hdfs) == 1
    for handle, _, _ in a.hdfs.values():
        handle.close()


def test_source_audit_cli_preserves_failed_receipt_and_import_boundary(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from src.gps.mppi_gps.audits.source import recompute_walking_screen
for name in sys.modules:
    assert not name.startswith(('mujoco','src.envs','src.gps.mppi_gps.policy','src.gps.mppi_gps.experiments','src.gps.mppi_gps.source_inputs','src.gps.mppi_gps.reporting'))
"""
    checked = subprocess.run(
        [sys.executable, "-B", "-c", code], cwd=root, capture_output=True, text=True
    )
    assert checked.returncode == 0, checked.stderr
    out = tmp_path / "audit"
    checked = subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/audits/student_source.py",
            "--run",
            str(tmp_path / "missing"),
            "--out",
            str(out),
            "--config-sha",
            "0" * 64,
            "--seal-sha",
            "0" * 64,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 1
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["passed"] is False
    assert (
        receipt["new_controls"] == receipt["new_neural_rows"] == receipt["new_optimizer_steps"] == 0
    )


def synthetic_fit_fixture(root):
    """Write tensor/curve receipts, never execute a network or optimizer."""
    raw_path = root / "evidence/fit/raw.h5"
    raw_path.parent.mkdir(parents=True)
    raw = h5py.File(raw_path, "w")

    def save_raw(tag, arrays):
        name = f"arrays/a{len(raw['arrays']) if 'arrays' in raw else 0:04d}"
        group = raw.create_group(name)
        group.attrs["tag"] = tag
        descriptors = {}
        for key, value in arrays.items():
            value = np.asarray(value)
            meta = array_binding(value) | dict(encoding="array")
            entry = group.create_dataset(key, data=value)
            entry.attrs.update(numpy_dtype=meta["dtype"], sha256=meta["sha256"], encoding="array")
            descriptors[key] = meta
        return dict(file="raw.h5", group=name, tag=tag, arrays=descriptors)

    def save_package(name, payload):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return dict(file=name, sha256=native.sha(path), bytes=path.stat().st_size)

    norm = dict(mean=np.zeros(17, np.float32), std=np.ones(17, np.float32))
    state = {
        "observation_mean": torch.from_numpy(norm["mean"]),
        "observation_std": torch.from_numpy(norm["std"]),
    }
    for layer, shape in ((0, (128, 17)), (2, (128, 128)), (4, (6, 128))):
        state[f"network.{layer}.weight"] = torch.zeros(shape)
        state[f"network.{layer}.bias"] = torch.zeros(shape[0])
    policy_hash = native.tensor_hash(state)
    cfg = dict(
        fit_arms=["mppi", "sac_reference"],
        learning=dict(batch_size=256, lr=0.0003),
        seed_pairs=[dict(policy_seed=pair + 1, fit_seed=100 + pair) for pair in range(3)],
    )
    data = {}
    for arm in cfg["fit_arms"]:
        for split, rows in (("train", 4000), ("validation", 2000)):
            data[arm + "_" + split] = dict(
                observations=np.zeros((rows, 17)),
                actions=np.zeros((rows, 6)),
                row_episode=np.repeat([arm + split + str(i) for i in range(rows // 1000)], 1000),
            )
    result = dict(points={}, fits=[], samplers={})
    private = torch.tensor([1, 2, 3], dtype=torch.uint8)
    config = dict(
        batch_size=256,
        lr=0.0003,
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="mean",
        anchor_refresh=0,
    )
    for pair, seeds in enumerate(cfg["seed_pairs"]):
        rng = np.random.Generator(np.random.PCG64(seeds["fit_seed"]))
        initial_rng = deepcopy(rng.bit_generator.state)
        indices = np.stack([rng.integers(0, 4000, 256) for _ in range(2000)])
        final_rng = deepcopy(rng.bit_generator.state)
        result["samplers"][str(pair)] = dict(
            seed=seeds["fit_seed"],
            initial_rng=initial_rng,
            final_rng=final_rng,
            minibatches=save_raw(f"sampler{pair}", dict(indices=indices)),
        )
        for arm in ("incoming", *cfg["fit_arms"]):
            name = f"{arm}_s{pair}"
            update = 0 if arm == "incoming" else 2000
            metadata = dict(
                pair=pair,
                arm=arm,
                updates=update,
                width=128,
                feature_mode="clipped",
                deployment_qualified=False,
            )
            optimizer = None
            if update:
                parameters = [key for key in state if key.startswith("network.")]
                group = dict(
                    params=list(range(6)),
                    lr=0.0003,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0,
                    foreach=False,
                    fused=False,
                )
                adam = dict(
                    param_groups=[group],
                    state={
                        i: dict(
                            step=torch.tensor(2000.0),
                            exp_avg=torch.zeros_like(state[key]),
                            exp_avg_sq=torch.zeros_like(state[key]),
                        )
                        for i, key in enumerate(parameters)
                    },
                )
                fit_data = dict(
                    rows=4000,
                    observations_sha256=native.fit_hash(data[arm + "_train"]["observations"]),
                    actions_sha256=native.fit_hash(data[arm + "_train"]["actions"]),
                )
                optimizer = dict(
                    schema="walker-adam-proximal-fit-v1",
                    policy_hash=policy_hash,
                    parent_policy_hash=policy_hash,
                    anchor_policy_hash=policy_hash,
                    anchor=deepcopy(state),
                    updates=2000,
                    fit_updates=2000,
                    rng=final_rng,
                    parameter_names=parameters,
                    fit_config=config,
                    fit_data=fit_data,
                    adam=adam,
                )
                initial = dict(
                    state_dict=deepcopy(state),
                    adam=dict(state={}, param_groups=[group]),
                    rng=initial_rng,
                    private_rng_state=private,
                )
                binding = save_package(f"learning/{name}_initial.pt", initial)
                binding.update(
                    policy_hash=policy_hash,
                    adam_fingerprint=native.fingerprint(initial["adam"]),
                    rng_fingerprint=native.fingerprint(initial_rng),
                    private_rng_fingerprint=native.fingerprint(private),
                    raw=save_raw(
                        "initial" + name,
                        {key: value.numpy() for key, value in state.items()}
                        | dict(private_rng_state=private.numpy()),
                    ),
                )
                fit = dict(
                    parent_policy_hash=policy_hash,
                    child_policy_hash=policy_hash,
                    fit_config=config,
                    fit_data=fit_data,
                    completed_updates=2000,
                    cumulative_updates=2000,
                    fit_updates=2000,
                    failed=False,
                    resume_fit=False,
                    optimizer_continued=False,
                    anchor_policy_hash=policy_hash,
                    parameter_count=19590,
                    proximal_divisor=19590,
                    anchor_refreshes=[],
                    work={
                        kind: dict(requested=amount, completed=amount, unknown_partial=0)
                        for kind, amount in dict(
                            neural_rows=512000, backward_rows=512000, optimizer_steps=2000
                        ).items()
                    },
                    learning_curve=[
                        dict(
                            update=i,
                            fit_update=i,
                            proximal_loss=0.0,
                            total_loss=0.0,
                            action_mse=0.0,
                            unclipped_gradient_norm=0.0,
                        )
                        for i in range(1, 2001)
                    ],
                )
                fit_path = root / f"learning/{name}.json"
                fit_path.write_text(json.dumps(fit))
                result["fits"].append(
                    dict(
                        point=name,
                        pair=pair,
                        arm=arm,
                        file=str(fit_path.relative_to(root)),
                        sha256=native.sha(fit_path),
                        initial=binding,
                        optimizer_fingerprint=native.fingerprint(optimizer),
                    )
                )
            payload = dict(
                schema="walker-tanh-policy-v1",
                state_dict=deepcopy(state),
                policy_hash=policy_hash,
                obs_dim=17,
                seed=seeds["policy_seed"],
                private_rng_state=private,
                optimizer_state=optimizer,
                metadata=metadata,
            )
            checkpoint = save_package(f"checkpoints/{name}.pt", payload) | dict(
                policy_hash=policy_hash
            )
            predictions = {
                key + "_predictions": np.zeros_like(values["actions"], dtype=np.float32)
                for key, values in data.items()
            }
            metrics = {
                key: dict(
                    mse=0.0,
                    per_trajectory={
                        str(owner): recompute_agreement(
                            predictions[key + "_predictions"][values["row_episode"] == owner],
                            values["actions"][values["row_episode"] == owner],
                            dict(early_controls=100, bin_controls=100, saturation_threshold=0.95),
                        )
                        for owner in np.unique(values["row_episode"])
                    },
                )
                for key, values in data.items()
            }
            result["points"][name] = dict(
                pair=pair,
                arm=arm,
                update=update,
                checkpoint=checkpoint,
                optimizer_fingerprint=native.fingerprint(optimizer),
                parameter_count=19590,
                metrics=metrics,
                predictions=save_raw(name, predictions),
            )
    raw.close()
    return cfg, result, data, norm


def test_full_saved_fit_contract_and_minibatch_or_prediction_tamper(tmp_path):
    cfg, result, data, norm = synthetic_fit_fixture(tmp_path)
    a = SourceAudit(tmp_path, tmp_path)
    checked = audit_fit(a, tmp_path, cfg, result, data, norm)
    assert len(checked["payloads"]) == 9 and checked["diagnostic_rows"] == 108000
    changed = deepcopy(result)
    changed["points"]["sac_reference_s1"]["metrics"]["mppi_validation"]["mse"] = 0.1
    with pytest.raises(AssertionError, match="MSE"):
        audit_fit(a, tmp_path, cfg, changed, data, norm)
    changed = deepcopy(result)
    changed["samplers"]["0"]["initial_rng"]["state"]["state"] += 1
    with pytest.raises(AssertionError, match="sampler endpoint"):
        audit_fit(a, tmp_path, cfg, changed, data, norm)
    for handle, _, _ in a.hdfs.values():
        handle.close()
