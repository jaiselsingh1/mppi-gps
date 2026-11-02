"""Independent saved-evidence W1 audit; no inference, physics or optimizer replay.

Torch only decodes trusted CPU tensor packages. Small sealed K1 inputs establish
lineage; historical physics arrays and training ledgers are not audited again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import native
from .common import assert_independent_imports
from .compatibility import recompute_agreement


class WidthAudit(native.Audit):
    """Saved-only native checks with width-aware checkpoint validation."""

    def prove(self, actual: Any, expected: Any, message: str) -> None:
        if isinstance(expected, dict):
            self.need(isinstance(actual, dict), message + " object")
            for key, value in expected.items():
                self.need(key in actual, message + " field " + key)
                self.prove(actual[key], value, message + "/" + key)
        elif isinstance(expected, (list, tuple)):
            self.need(isinstance(actual, (list, tuple)) and len(actual) == len(expected), message)
            for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
                self.prove(left, right, message + f"/{index}")
        elif isinstance(expected, (float, np.floating)):
            self.equal(actual, expected, message)
        else:
            self.need(actual == expected, message)


def point_name(pair: int, width: int, update: int) -> str:
    return f"s{pair}_w{width}_u{update}"


def contained(a: WidthAudit, run: Path, relative: str) -> Path:
    path = (run / relative).resolve()
    a.need(path.is_relative_to(run.resolve()), "Contained W1 evidence path")
    return path


def read_npz(a: WidthAudit, path: Path) -> dict[str, np.ndarray]:
    a.bind(path)
    with np.load(path, allow_pickle=False) as stream:
        return {key: np.array(stream[key], copy=True) for key in stream.files}


def array_binding(value: np.ndarray) -> dict[str, Any]:
    return dict(
        shape=list(value.shape),
        dtype=str(value.dtype),
        sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    )


def audit_inputs(a: WidthAudit, run: Path, cfg: dict) -> tuple[dict, dict, dict]:
    provenance = a.load(run / "inputs/provenance.json")
    inventory = a.load(run / "inputs/k1_inventory.json")
    k1_seal = a.load(run / "inputs/k1_input_manifest.json")
    a.need(provenance["schema"] == "walker-width-w1-executed-benchmark-v1", "Benchmark schema")
    a.bind(run / "inputs/k1_inventory.json", provenance["source"]["k1_inventory_sha256"])
    a.bind(run / "inputs/k1_input_manifest.json", provenance["source"]["k1_input_manifest_sha256"])
    a.need(
        k1_seal["files"]["inputs/inventory.json"] == native.sha(run / "inputs/k1_inventory.json"),
        "Original sealed K1 inventory owner",
    )
    source_root = Path(provenance["source"]["run"]).resolve()
    for relative, binding in provenance["source"]["files"].items():
        path = contained(a, source_root, relative)
        a.need(
            str(path) == binding["path"] and path.stat().st_size == binding["bytes"],
            "Exact selected source location and size",
        )
        a.bind(path, binding["sha256"])
        if relative in k1_seal["files"]:
            a.need(binding["sha256"] == k1_seal["files"][relative], "Original K1 input seal")
    source_dataset = read_npz(a, source_root / "inputs" / inventory["datasets"]["l1_training"])
    a.need(
        Counter(source_dataset["row_kind"][4000:4096].tolist())
        == dict(hypothetical_perturbed_resolve=64, executed_recovery=32),
        "Excluded 96 auxiliary rows",
    )
    expected_tags = dict(
        train=[f"discovery_{index:02d}" for index in range(4)],
        validation=[f"v2_r0_trial_raw_{index}" for index in range(2)],
    )
    data = {}
    for split, (start, stop) in dict(train=(0, 4000), validation=(4096, 6096)).items():
        binding = provenance["splits"][split]
        path = run / "inputs" / binding["file"]
        a.bind(path, binding["file_sha256"])
        values = read_npz(a, path)
        a.need(
            path.stat().st_size == binding["bytes"] and binding["rows"] == stop - start,
            "Exact split rows and storage",
        )
        a.prove(binding["indices"], dict(start=start, stop=stop), "Source row selection")
        a.need(
            set(values)
            == {"states", "observations", "actions", "row_episode", "row_step", "row_kind"},
            "Only executed benchmark fields",
        )
        for key, value in values.items():
            a.prove(
                binding["arrays"][key], array_binding(value), "Exact split original array bytes"
            )
            original = source_dataset["row_steps" if key == "row_step" else key][start:stop]
            a.need(
                np.array_equal(value, original) and value.dtype == original.dtype,
                "Unchanged source row values and order " + split + "/" + key,
            )
        for key, columns in (("states", 19), ("observations", 17), ("actions", 6)):
            a.need(
                values[key].shape == (stop - start, columns)
                and values[key].dtype == np.float64
                and np.isfinite(values[key]).all(),
                "Original finite float64 benchmark",
            )
        a.need(
            np.array_equal(values["observations"], native.obs(values["states"]))
            and np.all(np.abs(values["actions"]) <= 1),
            "Clipped pre-action features and executed bounds",
        )
        a.need(
            list(dict.fromkeys(values["row_episode"].tolist())) == expected_tags[split]
            and binding["teachers"] == expected_tags[split],
            "Fixed independent trajectory roster",
        )
        for tag in expected_tags[split]:
            mask = values["row_episode"] == tag
            a.need(
                mask.sum() == 1000 and np.array_equal(values["row_step"][mask], np.arange(1000)),
                "One ordered complete executed teacher",
            )
            teachers = [row for row in inventory["trajectories"] if row["id"] == tag]
            a.need(len(teachers) == 1, "Unique sealed teacher")
            teacher = teachers[0]
            a.prove(provenance["teachers"][tag], teacher, "Sealed teacher identity")
            a.prove(binding["outcomes"][tag], teacher["outcome"], "Complete fixed teacher outcome")
            a.need(
                all(teacher["outcome"][key] is True for key in ("complete", "success", "survived"))
                and teacher["controller"]["rho"] == 0
                and teacher["controller"]["effective_guidance_enabled"] is False,
                "Successful independent discovery remains the only target source",
            )
            saved = read_npz(a, source_root / "inputs" / teacher["file"])
            for key in ("states", "observations", "actions"):
                a.need(
                    np.array_equal(values[key][mask], saved[key]), "Executed teacher row lineage"
                )
            for key in ("observations", "actions"):
                a.need(
                    array_binding(values[key][mask])["sha256"]
                    == teacher["source"]["raw"]["arrays"][key]["sha256"],
                    "Original executed action/observation descriptor",
                )
        a.prove(
            binding["fit_data"],
            dict(
                observations_sha256=native.fit_hash(values["observations"]),
                actions_sha256=native.fit_hash(values["actions"]),
                rows=stop - start,
            ),
            "Fitting hash contract",
        )
        data[split] = values
    norm = read_npz(a, run / "inputs/normalization.npz")
    a.need(set(norm) == {"mean", "std"}, "Normalization buffer roster")
    expected_norm = dict(
        mean=data["train"]["observations"].mean(axis=0).astype(np.float32),
        std=np.maximum(data["train"]["observations"].std(axis=0), 0.05).astype(np.float32),
    )
    a.need(
        provenance["normalization"]["training_only"] is True, "Training-only normalization claim"
    )
    for key in norm:
        a.need(
            norm[key].dtype == np.float32 and np.array_equal(norm[key], expected_norm[key]),
            "Exact train-only normalization " + key,
        )
        a.prove(
            provenance["normalization"]["arrays"][key],
            array_binding(norm[key]),
            "Normalization array seal",
        )
    overlaps = {}
    for key in ("states", "observations", "actions", "normalized_inputs"):
        for dtype in (np.float32,) if key == "normalized_inputs" else (np.float64, np.float32):
            keys = []
            for split in ("train", "validation"):
                array = (
                    (data[split]["observations"].astype(np.float32) - norm["mean"]) / norm["std"]
                    if key == "normalized_inputs"
                    else data[split][key].astype(dtype)
                )
                value = np.ascontiguousarray(array).copy()
                value[value == 0] = 0
                keys.append({row.tobytes() for row in value})
            left, right = keys
            a.need(
                len(left) == 4000 and len(right) == 2000 and not left & right,
                "No duplicate or cross-split rows " + key + str(dtype),
            )
            overlaps[key + ":" + np.dtype(dtype).name] = dict(
                train_unique_rows=len(left),
                validation_unique_rows=len(right),
                cross_split_overlap=len(left & right),
            )
    a.prove(provenance["overlap_checks"], overlaps, "All observation/action overlap checks")
    seeds = [provenance["teachers"][tag]["seed"] for tags in expected_tags.values() for tag in tags]
    a.need(
        len(set(seeds)) == 6 and provenance["reset_seeds"] == sorted(seeds),
        "Distinct teacher reset ancestries",
    )
    return data, norm, provenance


def decode_checkpoint(a: WidthAudit, path: Path, width: int, normalization: dict) -> dict:
    a.bind(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    a.need(payload["schema"] == "walker-tanh-policy-v1", "Unchanged policy checkpoint schema")
    state = payload["state_dict"]
    shapes = {
        "observation_mean": (17,),
        "observation_std": (17,),
        "network.0.weight": (width, 17),
        "network.0.bias": (width,),
        "network.2.weight": (width, width),
        "network.2.bias": (width,),
        "network.4.weight": (6, width),
        "network.4.bias": (6,),
    }
    a.need(
        set(state) == set(shapes)
        and payload.get("obs_dim", 17) == 17
        and payload.get("hidden_dim", 128) == width,
        "Exact architecture and checkpoint width declaration",
    )
    a.need(
        ("hidden_dim" not in payload) if width == 128 else payload["hidden_dim"] == 256,
        "Default checkpoint key compatibility",
    )
    for name, shape in shapes.items():
        value = state[name]
        a.need(
            isinstance(value, torch.Tensor)
            and tuple(value.shape) == shape
            and value.dtype == torch.float32
            and bool(torch.isfinite(value).all()),
            "Finite width-specific tensor " + name,
        )
    a.need(
        native.tensor_hash(state) == payload["policy_hash"], "Independent named-tensor policy hash"
    )
    for key in ("mean", "std"):
        a.need(
            np.array_equal(state["observation_" + key].numpy(), normalization[key]),
            "Fixed training-only normalization buffers",
        )
    return payload


def audit_sampler(
    a: WidthAudit,
    stored: np.ndarray,
    recorded: dict,
    *,
    seed: int,
    rows: int,
    batch_size: int,
    updates: int,
    checkpoints: list[int],
) -> dict:
    a.need(
        stored.shape == (updates, batch_size) and stored.dtype == np.int64,
        "Exact shared minibatch array shape/dtype",
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    endpoints = {"0": dict(rng.bit_generator.state)}
    for update in range(1, updates + 1):
        expected = rng.integers(0, rows, size=batch_size)
        a.need(np.array_equal(stored[update - 1], expected), "Exact paired PCG64 minibatch")
        if update in checkpoints:
            endpoints[str(update)] = dict(rng.bit_generator.state)
    a.prove(recorded, endpoints, "Saved sampler endpoint states")
    return endpoints


def audit_prediction_metrics(a: WidthAudit, point: dict, data: dict, predictions: dict) -> int:
    rows = 0
    for split, values in data.items():
        prediction = predictions[split + "_predictions"]
        a.need(
            prediction.shape == values["actions"].shape
            and prediction.dtype == np.float32
            and np.isfinite(prediction).all()
            and np.all(np.abs(prediction) <= 1),
            "Raw finite bounded float32 predictions",
        )
        metric = point["metrics"][split]
        a.equal(
            metric["mse"],
            np.square(prediction.astype(np.float64) - values["actions"]).mean(),
            "Independent full split six-torque MSE",
        )
        tags = list(dict.fromkeys(values["row_episode"].tolist()))
        a.need(set(metric["per_trajectory"]) == set(tags), "Complete trajectory metric roster")
        for tag in tags:
            mask = values["row_episode"] == tag
            a.prove(
                metric["per_trajectory"][tag],
                recompute_agreement(
                    prediction[mask],
                    values["actions"][mask],
                    dict(early_controls=100, bin_controls=100, saturation_threshold=0.95),
                ),
                "Independent trajectory agreement",
            )
        rows += len(prediction)
    return rows


def recompute_contrast(points: dict, evaluations: dict, cfg: dict) -> dict:
    train, validation, task = [], [], []
    for pair in range(len(cfg["seed_pairs"])):
        names = [point_name(pair, width, cfg["updates"]) for width in (128, 256)]
        small, wide = [points[name]["metrics"] for name in names]
        train.append(1 - wide["train"]["mse"] / small["train"]["mse"])
        validation.append(1 - wide["validation"]["mse"] / small["validation"]["mse"])
        if evaluations:
            left, right = [evaluations[name]["summary"] for name in names]
            task.append(
                dict(
                    relative_cost_gain=1 - right["mean_cost"] / left["mean_cost"],
                    walking_gain=right["successes"] - left["successes"],
                    walking_nonregression=right["successes"] >= left["successes"],
                    survival_nonregression=right["survived"] >= left["survived"],
                    steps_nonregression=right["mean_steps"] >= left["mean_steps"],
                )
            )
    gates = cfg["gates"]
    fit_checks = dict(
        mean_train_gain=float(np.mean(train)) >= gates["minimum_train_mse_gain"],
        fitting_pairs=sum(value > 0 for value in train) >= gates["minimum_fitting_pairs"],
    )
    task_checks = (
        {}
        if not task
        else dict(
            mean_cost_gain=float(np.mean([row["relative_cost_gain"] for row in task]))
            >= gates["minimum_cost_gain"],
            cost_pairs=sum(row["relative_cost_gain"] > 0 for row in task)
            >= gates["minimum_cost_pairs"],
            walking_pairs=sum(row["walking_gain"] >= gates["minimum_walking_gain"] for row in task)
            >= gates["minimum_walking_pairs"],
            all_pair_walking_nonregression=all(row["walking_nonregression"] for row in task),
            all_pair_survival_nonregression=all(row["survival_nonregression"] for row in task),
            all_pair_steps_nonregression=all(row["steps_nonregression"] for row in task),
        )
    )
    return dict(
        relative_train_mse_gains=train,
        relative_validation_mse_gains=validation,
        fitting_checks=fit_checks,
        reconstruction_screen=all(fit_checks.values()),
        validation_guard=float(np.mean(validation)) >= 0,
        fitting_screen=all(fit_checks.values()) and float(np.mean(validation)) >= 0,
        task_pairs=task,
        task_checks=task_checks,
        task_screen=bool(task_checks) and all(task_checks.values()),
    )


def audit_optimizer(
    a: WidthAudit,
    payload: dict,
    initial: dict,
    previous: dict,
    point: dict,
    fit: dict,
    endpoints: dict,
    data: dict,
    cfg: dict,
) -> None:
    update, width = point["update"], point["width"]
    state = payload["optimizer_state"]
    a.need(
        isinstance(state, dict) and state["schema"] == "walker-adam-proximal-fit-v1",
        "Complete saved Adam/RNG/anchor state",
    )
    a.need(
        state["policy_hash"] == payload["policy_hash"] == fit["child_policy_hash"],
        "Exact checkpoint/fit/optimizer owner",
    )
    a.need(
        state["parent_policy_hash"] == fit["parent_policy_hash"] == previous["policy_hash"],
        "Exact learning continuation parent",
    )
    a.need(
        state["anchor_policy_hash"]
        == native.tensor_hash(state["anchor"])
        == initial["policy_hash"],
        "Original proximal block anchor remains fixed",
    )
    a.need(
        native.fingerprint(state["anchor"]) == native.fingerprint(initial["state_dict"]),
        "Full immutable anchor tensors",
    )
    a.need(
        state["updates"] == state["fit_updates"] == update
        and state["rng"] == endpoints[str(update)],
        "Exact cumulative updates and private minibatch RNG",
    )
    parameters = [name for name in payload["state_dict"] if name.startswith("network.")]
    a.need(state["parameter_names"] == parameters, "Optimizer parameter order")
    config = dict(
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        eta_theta=0.0,
        gradient_clip=0.5,
        prox_reduction="mean",
        anchor_refresh=0,
    )
    a.prove(state["fit_config"], config, "Unchanged sole-width fitting objective")
    a.prove(fit["fit_config"], config, "Fit record objective")
    expected_data = dict(
        rows=len(data["observations"]),
        observations_sha256=native.fit_hash(data["observations"]),
        actions_sha256=native.fit_hash(data["actions"]),
    )
    a.prove(state["fit_data"], expected_data, "Optimizer exact training matrix identity")
    a.prove(fit["fit_data"], expected_data, "Fit exact training matrix identity")
    adam = state["adam"]
    a.need(len(adam["param_groups"]) == 1, "Single fresh Adam parameter group")
    group = adam["param_groups"][0]
    a.need(
        group["params"] == list(range(6)) and set(adam["state"]) == set(range(6)),
        "Complete Adam tensor roster",
    )
    a.need(
        group["lr"] == cfg["lr"]
        and group["betas"] == (0.9, 0.999)
        and group["eps"] == 1e-8
        and group["weight_decay"] == 0
        and group["foreach"] is False
        and group["fused"] is False,
        "Unchanged Adam settings and arithmetic implementation",
    )
    for index, name in enumerate(parameters):
        saved = adam["state"][index]
        a.need(float(saved["step"]) == update, "Fresh Adam step counter")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = saved[key]
            a.need(
                tensor.shape == payload["state_dict"][name].shape
                and tensor.dtype == torch.float32
                and bool(torch.isfinite(tensor).all()),
                "Finite shape-matched Adam moments",
            )
    previous_update = (
        0 if previous["optimizer_state"] is None else previous["optimizer_state"]["updates"]
    )
    chunk = update - previous_update
    a.need(
        fit["completed_updates"] == chunk
        and fit["cumulative_updates"] == fit["fit_updates"] == update
        and not fit["failed"]
        and fit["resume_fit"] == (previous_update > 0)
        and fit["optimizer_continued"] == (previous_update > 0),
        "Fresh then exact continued fitting blocks",
    )
    a.need(
        fit["anchor_policy_hash"] == initial["policy_hash"]
        and fit["parameter_count"] == (19590 if width == 128 else 71942)
        and fit["proximal_divisor"] == fit["parameter_count"]
        and fit["anchor_refreshes"] == [],
        "No objective/anchor changes",
    )
    for kind, requested in dict(
        neural_rows=chunk * cfg["batch_size"],
        backward_rows=chunk * cfg["batch_size"],
        optimizer_steps=chunk,
    ).items():
        a.prove(
            fit["work"][kind],
            dict(requested=requested, completed=requested, unknown_partial=0),
            "Exact fitting work receipt",
        )
    curve = fit["learning_curve"]
    a.need(len(curve) == chunk, "Every update retains its minibatch learning curve")
    for index, row in enumerate(curve, previous_update + 1):
        a.need(
            row["update"] == row["fit_update"] == index
            and row["proximal_loss"] == 0
            and row["total_loss"] == row["action_mse"]
            and 0 <= row["action_mse"] <= 4
            and np.isfinite(row["unclipped_gradient_norm"])
            and row["unclipped_gradient_norm"] >= 0,
            "Finite indexed fixed-objective training curve",
        )


def audit_run(a: WidthAudit, run: Path) -> dict:
    cfg, seal, manifest = [
        a.load(run / name) for name in ("config.json", "input_manifest.json", "manifest.json")
    ]
    a.bind(run / "config.json", seal["config_sha256"])
    a.need(manifest["config_sha256"] == seal["config_sha256"], "Source/config seal agreement")
    a.bind(run / "manifest.json", seal["source_manifest_sha256"])
    for row in manifest["files"]:
        a.bind(contained(a, run / "source", row["file"]), row["sha256"])
    for relative, digest in seal["files"].items():
        a.bind(contained(a, run, relative), digest)
    a.need(
        cfg["schema"] == "walker-student-width-w1-v1"
        and cfg["widths"] == [128, 256]
        and cfg["updates"] == 10000
        and cfg["checkpoints"] == [0, 2000, 5000, 10000]
        and cfg["rows"] == 4000
        and cfg["validation_rows"] == 2000
        and cfg["batch_size"] == 256
        and cfg["lr"] == 0.0003
        and cfg["eta_theta"] == 0
        and cfg["prox_reduction"] == "mean"
        and cfg["anchor_refresh"] == 0,
        "Frozen one-mechanism W1 recipe",
    )
    a.need(
        cfg["device"] == "cpu"
        and cfg["torch_threads"] == 1
        and cfg["gradient_clip"] == 0.5
        and cfg["normalization_floor"] == 0.05
        and cfg["episode_steps"] == 1000,
        "Fixed numerical/execution settings",
    )
    a.prove(
        cfg["seed_pairs"],
        [dict(policy_seed=20261101001 + index, fit_seed=1103001000 + index) for index in range(3)],
        "Three fixed fresh paired seed streams",
    )
    for cohort, root, count in (
        ("curve", 1110000000, 4),
        ("primary", 1120000000, 16),
        ("confirmation", 1130000000, 16),
    ):
        a.need(
            cfg[cohort + "_seeds"] == list(range(root, root + count)), "Fixed disjoint reset panel"
        )
    data, normal, provenance = audit_inputs(a, run, cfg)
    stage = a.ledger(run)
    result = stage["result"]
    a.need(
        result["execution_complete"] is True
        and result["error"] is None
        and result["status"] == "complete",
        "Complete valid closed W1 execution",
    )
    a.need(
        result["config_sha256"] == seal["config_sha256"]
        and result["input_manifest_sha256"] == native.sha(run / "input_manifest.json"),
        "Worker launch identity",
    )
    expected_names = {
        point_name(pair, width, update)
        for pair in range(3)
        for width in (128, 256)
        for update in cfg["checkpoints"]
    }
    a.need(
        set(result["points"]) == expected_names and set(result["samplers"]) == {"0", "1", "2"},
        "All 24 checkpoints and three shared samplers; no selection",
    )
    fits = {row["point"]: row for row in result["fits"]}
    a.need(
        len(fits) == len(result["fits"]) == 18
        and set(fits) == {name for name in expected_names if not name.endswith("_u0")},
        "Exactly 18 continued fit blocks",
    )
    payloads, diagnostic_rows = {}, 0
    for pair, seeds in enumerate(cfg["seed_pairs"]):
        sampler = result["samplers"][str(pair)]
        a.need(sampler["seed"] == seeds["fit_seed"], "Bound sampler seed")
        batches = a.group(run / "evidence/raw.h5", sampler["minibatches"])["indices"]
        endpoints = audit_sampler(
            a,
            batches,
            sampler["rng_at_checkpoints"],
            seed=seeds["fit_seed"],
            rows=4000,
            batch_size=256,
            updates=10000,
            checkpoints=cfg["checkpoints"],
        )
        for width in (128, 256):
            initial = previous = None
            for update in cfg["checkpoints"]:
                name = point_name(pair, width, update)
                point = result["points"][name]
                a.prove(
                    point,
                    dict(pair=pair, width=width, update=update, **seeds),
                    "Point declared identity",
                )
                binding = point["checkpoint"]
                path = contained(a, run, binding["file"])
                a.bind(path, binding["sha256"])
                a.need(path.stat().st_size == binding["bytes"], "Checkpoint exact storage")
                payload = decode_checkpoint(a, path, width, normal)
                a.need(
                    payload["seed"] == seeds["policy_seed"]
                    and payload["policy_hash"] == binding["policy_hash"]
                    and native.fingerprint(payload["optimizer_state"])
                    == point["optimizer_fingerprint"],
                    "Checkpoint actor and optimizer identities",
                )
                a.need(
                    point["parameter_count"] == (19590 if width == 128 else 71942),
                    "Actual width parameter count",
                )
                a.prove(
                    payload["metadata"],
                    dict(
                        pair=pair,
                        width=width,
                        updates=update,
                        feature_mode="clipped",
                        fresh_initialization=True,
                        deployment_qualified=False,
                    ),
                    "Checkpoint metadata",
                )
                if update == 0:
                    a.need(
                        payload["optimizer_state"] is None, "Fresh Adam with no inherited moments"
                    )
                    initial = payload
                else:
                    record = fits[name]
                    a.bind(contained(a, run, record["file"]), record["sha256"])
                    fit = a.load(run / record["file"])
                    audit_optimizer(
                        a, payload, initial, previous, point, fit, endpoints, data["train"], cfg
                    )
                    a.need(
                        torch.equal(payload["private_rng_state"], initial["private_rng_state"]),
                        "Policy private initialization stream unchanged by learning",
                    )
                predictions = a.group(run / "evidence/raw.h5", point["predictions"])
                diagnostic_rows += audit_prediction_metrics(a, point, data, predictions)
                payloads[name] = payload
                previous = payload
    a.prove(
        result["teacher_outcomes"],
        {key: value["outcomes"] for key, value in provenance["splits"].items()},
        "Unchanged complete fixed teacher outcomes",
    )
    primary = recompute_contrast(result["points"], result["evaluations"]["primary"], cfg)
    a.prove(result["gates"]["primary"], primary, "Independently recomputed primary screens")
    final = {point_name(pair, width, 10000) for pair in range(3) for width in (128, 256)}
    trigger = primary["task_screen"] or any(
        result["evaluations"]["primary"][name]["summary"]["successes"]
        >= cfg["gates"]["competence_minimum_successes"]
        for name in final
    )
    a.need(
        result["gates"]["confirmation_triggered"] == trigger,
        "Prospective conditional confirmation gate",
    )
    confirmation = (
        recompute_contrast(result["points"], result["evaluations"]["confirmation"], cfg)
        if trigger
        else None
    )
    a.prove(result["gates"]["confirmation"], confirmation, "Conditional confirmation screen")
    a.need(
        result["gates"]["confirmed_task_benefit"]
        == bool(trigger and primary["task_screen"] and confirmation["task_screen"]),
        "Confirmed task benefit requires both complete panels",
    )
    reset_owners, seen, controls = {}, set(), 0
    for cohort in ("curve", "primary", "confirmation"):
        records = result["evaluations"][cohort]
        a.need(
            set(records)
            == (
                expected_names
                if cohort == "curve"
                else final
                if cohort == "primary" or trigger
                else set()
            ),
            "Fixed full evaluation roster " + cohort,
        )
        for name, record in records.items():
            a.need(
                record["policy_hash"] == payloads[name]["policy_hash"]
                and record["optimizer_fingerprint"]
                == native.fingerprint(payloads[name]["optimizer_state"]),
                "Evaluation fixed actor/Adam owner",
            )
            a.summary(stage, record["summary"])
            rows = record["episodes"]
            a.need(
                [row["seed"] for row in rows] == cfg[cohort + "_seeds"]
                and record["summary"]["tags"] == [row["tag"] for row in rows],
                "Declared paired evaluation reset order",
            )
            for row in rows:
                tag = row["tag"]
                a.need(tag not in seen, "Unique executed episode")
                seen.add(tag)
                event, arrays = a.episode(stage, tag)
                a.prove(event, row, "Result/ledger complete episode identity")
                reset = native.fingerprint(
                    {key: arrays[key][0] for key in ("states", "warmstarts", "integrations")}
                )
                a.need(
                    reset_owners.setdefault(row["seed"], reset) == reset,
                    "Exact full simulator/reset pairing",
                )
                for info in arrays["infos"]["rows"]:
                    a.prove(
                        info["controller"],
                        dict(
                            policy_hash=record["policy_hash"],
                            feature_mode="clipped",
                            width=result["points"][name]["width"],
                        ),
                        "Every action's declared standalone controller",
                    )
                for before, _ in stage["live"][tag]:
                    a.need(
                        before["context"]["point"] == name
                        and before["context"]["cohort"] == cohort,
                        "Every live query checkpoint/cohort owner",
                    )
                controls += row["steps"]
    a.need(seen == set(stage["episodes"]), "No unreported native episodes")
    for name in final:
        competence = bool(
            trigger
            and result["evaluations"]["primary"][name]["summary"]["successes"] >= 15
            and result["evaluations"]["confirmation"][name]["summary"]["successes"] >= 15
        )
        a.need(
            result["gates"]["competence"][name] == competence,
            "Competence uses same fixed policy in both panels",
        )
    counts = stage["counts"]
    fitting_rows = 6 * 10000 * 256
    a.need(
        diagnostic_rows == 144000
        and counts["completed_neural_rows"] == fitting_rows + diagnostic_rows + controls,
        "Exact training/diagnostic/standalone forward census",
    )
    a.need(
        counts["completed_backward_rows"] == fitting_rows
        and counts["completed_optimizer_steps"] == 60000,
        "Exactly six fixed finite learning budgets",
    )
    a.need(
        counts["completed_live_controls"] == controls
        and counts["requested_model_controls"] == 0
        and counts["completed_physics_substeps"] == controls * 4,
        "Only actual native live physics",
    )
    for key in (
        "failed_queries",
        "failed_measurements",
        "rejected_queries",
        "rejected_measurements",
        "unknown_partial_controls",
        "unknown_partial_neural_rows",
        "unknown_partial_backward_rows",
        "unknown_partial_optimizer_steps",
        "reset_errors",
        "restores",
    ):
        a.need(counts[key] == 0, "No failed/unknown/extra work " + key)
    for kind, cap in cfg["budgets"].items():
        if kind in ("wall_seconds", "storage_bytes"):
            continue
        key = kind if kind in ("requested_controls", "episodes") else "requested_" + kind
        a.need(counts[key] <= cap, "Predeclared finite budget " + kind)
    a.need(result["wall_seconds"] <= cfg["budgets"]["wall_seconds"], "Worker wall ceiling")
    a.need(
        sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
        <= cfg["budgets"]["storage_bytes"],
        "Closed-run storage ceiling",
    )
    return dict(
        points=len(payloads),
        fit_blocks=len(fits),
        episodes=len(seen),
        actual_controls=controls,
        diagnostic_forward_rows=diagnostic_rows,
        fitting_rows=fitting_rows,
        confirmation_triggered=trigger,
        gates=result["gates"],
        work=dict(counts),
        source_lineage="Small selected sealed K1 inputs; no historical raw/ledger replay",
        limitations=[
            "Saved checkpoint/ledger ownership and exact sampler continuation are verified without replaying optimizer arithmetic or evaluating a network.",
            "Three paired training seeds and shared reset panels do not establish a population effect or equal-compute comparison.",
        ],
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    a = WidthAudit(run, run / "source")
    findings, error = {}, None
    try:
        a.bind(Path(__file__))
        a.bind(Path(native.__file__))
        findings = audit_run(a, run)
        for path, binding in list(a.files.items()):
            a.need(a.bind(Path(path)) == binding, "Audited file unchanged after read")
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Audited HDF unchanged after read",
            )
        assert_independent_imports(a)
        a.need(
            not any(
                name.startswith(
                    ("src.gps.mppi_gps.compatibility_metrics", "src.gps.mppi_gps.width_inputs")
                )
                for name in sys.modules
            ),
            "No production metric/input imports",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        for handle, _, _ in a.hdfs.values():
            handle.close()
    receipt = dict(
        schema="walker-student-width-w1-independent-audit-v1",
        passed=error is None,
        error=error,
        checks=a.checks,
        bound_files=len(a.files),
        hashed_arrays=len(a.arrays),
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
        scope="Independent saved-array metrics, sealed benchmark lineage, width-specific checkpoint/Adam/anchor identities, exact PCG64 sampler continuation, native complete outcomes/reset pairing, gates and finite work census. No networks, simulator or optimizer arithmetic executed.",
    )
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", a.files),
        ("array_bindings.json", a.arrays),
    ):
        with (args.out / name).open("x") as stream:
            json.dump(native.plain(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(receipt))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
