"""Independent W2 saved-evidence audit; no networks, physics or optimizer replay."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import native, width
from .common import assert_independent_imports

UpdateBudgetAudit = width.WidthAudit


def point_name(pair: int, update: int) -> str:
    return f"s{pair}_u{update}"


def recompute_contrast(points: dict, evaluations: dict, cfg: dict) -> dict:
    """Use the predeclared 2k candidate against its 10k continuation baseline."""
    pairs = []
    for pair in range(len(cfg["seed_pairs"])):
        short, long = [point_name(pair, update) for update in (2000, 10000)]
        short_metrics, long_metrics = [points[name]["metrics"] for name in (short, long)]
        short_eval, long_eval = [evaluations[name]["summary"] for name in (short, long)]
        pairs.append(
            dict(
                pair=pair,
                relative_train_mse_gain=1
                - short_metrics["train"]["mse"] / long_metrics["train"]["mse"],
                relative_validation_mse_gain=1
                - short_metrics["validation"]["mse"] / long_metrics["validation"]["mse"],
                relative_controls_gain=short_eval["mean_steps"] / long_eval["mean_steps"] - 1,
                relative_cost_gain=1 - short_eval["mean_cost"] / long_eval["mean_cost"],
                controls_delta=short_eval["mean_steps"] - long_eval["mean_steps"],
                cost_delta=short_eval["mean_cost"] - long_eval["mean_cost"],
                walking_gain=short_eval["successes"] - long_eval["successes"],
                survival_gain=short_eval["survived"] - long_eval["survived"],
                walking_nonregression=short_eval["successes"] >= long_eval["successes"],
                survival_nonregression=short_eval["survived"] >= long_eval["survived"],
                controls_nonregression=short_eval["mean_steps"] >= long_eval["mean_steps"],
            )
        )
    validation_checks = dict(
        mean_gain=float(np.mean([row["relative_validation_mse_gain"] for row in pairs])) >= 0.10,
        positive_pairs=sum(row["relative_validation_mse_gain"] > 0 for row in pairs) >= 2,
    )
    common = dict(
        all_pair_walking_nonregression=all(row["walking_nonregression"] for row in pairs),
        all_pair_survival_nonregression=all(row["survival_nonregression"] for row in pairs),
    )
    retention_checks = dict(
        mean_controls_gain=float(np.mean([row["relative_controls_gain"] for row in pairs])) >= 0.10,
        positive_pairs=sum(row["relative_controls_gain"] > 0 for row in pairs) >= 2,
        **common,
    )
    cost_checks = dict(
        mean_cost_gain=float(np.mean([row["relative_cost_gain"] for row in pairs])) >= 0.05,
        positive_pairs=sum(row["relative_cost_gain"] > 0 for row in pairs) >= 2,
        **common,
        all_pair_controls_nonregression=all(row["controls_nonregression"] for row in pairs),
    )
    walking_checks = dict(gain_pairs=sum(row["walking_gain"] >= 2 for row in pairs) >= 2, **common)
    return dict(
        pairs=pairs,
        validation_checks=validation_checks,
        retention_checks=retention_checks,
        cost_checks=cost_checks,
        walking_checks=walking_checks,
        validation_screen=all(validation_checks.values()),
        retention_screen=all(retention_checks.values()),
        cost_screen=all(cost_checks.values()),
        walking_screen=all(walking_checks.values()),
    )


def audit_reused_inputs(a: UpdateBudgetAudit, run: Path) -> tuple[dict, dict, dict]:
    """Bind exact W1 copies and their original launch seal, without a raw re-audit."""
    provenance = a.load(run / "inputs/provenance.json")
    original = a.load(run / "inputs/w1_provenance.json")
    a.need(
        provenance["schema"] == original["schema"] == "walker-width-w1-executed-benchmark-v1"
        and provenance["reuse_schema"] == "walker-update-w2-reused-benchmark-v1",
        "W2 exact-reuse provenance schemas",
    )
    for key, value in original.items():
        if key != "schema":
            a.prove(provenance[key], value, "Unchanged W1 provenance " + key)
    source = provenance["w1_source"]
    expected_copies = {
        "train.npz": "inputs/train.npz",
        "validation.npz": "inputs/validation.npz",
        "normalization.npz": "inputs/normalization.npz",
        "task.json": "inputs/task.json",
        "w1_provenance.json": "inputs/provenance.json",
        "k1_inventory.json": "inputs/k1_inventory.json",
        "k1_input_manifest.json": "inputs/k1_input_manifest.json",
        "w1_config.json": "config.json",
        "w1_manifest.json": "manifest.json",
        "w1_input_manifest.json": "input_manifest.json",
        "w1_launch_receipt.json": "launch_receipt.json",
    }
    a.need(set(source["exact_copies"]) == set(expected_copies), "Complete fixed W1 copy roster")
    launch = a.load(run / "inputs/w1_launch_receipt.json")
    w1_seal = a.load(run / "inputs/w1_input_manifest.json")
    w1_config = a.load(run / "inputs/w1_config.json")
    a.need(w1_config["schema"] == "walker-student-width-w1-v1", "Original W1 protocol")
    for name, digest in (
        ("w1_config.json", launch["config_sha256"]),
        ("w1_manifest.json", launch["source_manifest_sha256"]),
        ("w1_input_manifest.json", launch["input_manifest_sha256"]),
    ):
        a.bind(run / "inputs" / name, digest)
    a.need(
        w1_seal["config_sha256"] == launch["config_sha256"]
        and w1_seal["source_manifest_sha256"] == launch["source_manifest_sha256"],
        "W1 launch/input/source seals agree",
    )
    for destination, relative in expected_copies.items():
        binding = source["exact_copies"][destination]
        a.need(binding["source"] == relative, "Exact W1 copy source name")
        original_path = width.contained(a, Path(source["run"]), relative)
        a.need(str(original_path) == binding["path"], "Explicit original W1 file location")
        for path in (original_path, run / "inputs" / destination):
            a.bind(path, binding["sha256"])
            a.need(path.stat().st_size == binding["bytes"], "Exact W1 source/copy byte count")
        a.prove(binding, source["files"][relative], "Original W1 source file binding")
        if relative.startswith("inputs/"):
            a.need(binding["sha256"] == w1_seal["files"][relative], "Original W1 input seal hash")
    data = {}
    for split, count, tags, kind in (
        ("train", 4000, [f"discovery_{index:02d}" for index in range(4)], "executed_discovery"),
        (
            "validation",
            2000,
            [f"v2_r0_trial_raw_{index}" for index in range(2)],
            "executed_added_independent",
        ),
    ):
        values = width.read_npz(a, run / "inputs" / f"{split}.npz")
        description = original["splits"][split]
        a.need(
            set(values)
            == {"states", "observations", "actions", "row_episode", "row_step", "row_kind"},
            "Fixed executed benchmark array roster",
        )
        for key, value in values.items():
            a.prove(
                description["arrays"][key],
                width.array_binding(value),
                "Unchanged original array bytes",
            )
        a.need(
            values["states"].shape == (count, 19)
            and values["observations"].shape == (count, 17)
            and values["actions"].shape == (count, 6)
            and np.array_equal(values["observations"], native.obs(values["states"]))
            and np.all(np.abs(values["actions"]) <= 1),
            "Aligned native states/features/executed torques",
        )
        for key in ("states", "observations", "actions"):
            a.need(
                values[key].dtype == np.float64 and np.isfinite(values[key]).all(),
                "Original finite float64 matrix",
            )
        a.need(
            np.array_equal(values["row_episode"], np.repeat(tags, 1000))
            and np.array_equal(values["row_step"], np.tile(np.arange(1000), len(tags)))
            and np.all(values["row_kind"] == kind),
            "Exact trajectory/step ownership, no auxiliary examples",
        )
        for tag in tags:
            teacher = original["teachers"][tag]
            a.need(
                all(teacher["outcome"][key] is True for key in ("complete", "success", "survived"))
                and teacher["controller"]["rho"] == 0,
                "Complete successful independent teachers",
            )
        data[split] = values
    normal = width.read_npz(a, run / "inputs/normalization.npz")
    for key, expected in dict(
        mean=data["train"]["observations"].mean(axis=0).astype(np.float32),
        std=np.maximum(data["train"]["observations"].std(axis=0), 0.05).astype(np.float32),
    ).items():
        a.need(
            normal[key].dtype == np.float32 and np.array_equal(normal[key], expected),
            "Unchanged train-only normalization",
        )
    a.prove(
        provenance["reuse_overlap_checks"],
        original["overlap_checks"],
        "Unchanged source overlap definitions/results",
    )
    for key in ("states", "observations", "actions", "normalized_inputs"):
        for dtype in (np.float32,) if key == "normalized_inputs" else (np.float64, np.float32):
            packed = []
            for split in ("train", "validation"):
                array = (
                    (data[split]["observations"].astype(np.float32) - normal["mean"])
                    / normal["std"]
                    if key == "normalized_inputs"
                    else data[split][key].astype(dtype)
                )
                array = np.ascontiguousarray(array).copy()
                array[array == 0] = 0
                packed.append({row.tobytes() for row in array})
            a.need(
                len(packed[0]) == 4000 and len(packed[1]) == 2000 and not packed[0] & packed[1],
                "Exact duplicate/cross-split overlap check",
            )
    return data, normal, provenance


def audit_reload(
    a: UpdateBudgetAudit, receipt: dict, payload: dict, binding: dict, pair: int
) -> None:
    expected = dict(
        policy_hash=payload["policy_hash"],
        optimizer_fingerprint=native.fingerprint(payload["optimizer_state"]),
        private_rng_fingerprint=native.fingerprint(payload["private_rng_state"]),
    )
    a.prove(
        receipt,
        dict(
            pair=pair,
            point=point_name(pair, 2000),
            file=binding["file"],
            checkpoint_sha256=binding["sha256"],
            before=expected,
            after=expected,
        ),
        "Actual 2k checkpoint load preserves policy/Adam/RNG owner",
    )


def audit_run(a: UpdateBudgetAudit, run: Path) -> dict:
    cfg, seal, manifest = [
        a.load(run / name) for name in ("config.json", "input_manifest.json", "manifest.json")
    ]
    a.bind(run / "config.json", seal["config_sha256"])
    a.bind(run / "manifest.json", seal["source_manifest_sha256"])
    a.need(manifest["config_sha256"] == seal["config_sha256"], "Source/config seal agreement")
    for row in manifest["files"]:
        a.bind(width.contained(a, run / "source", row["file"]), row["sha256"])
    for relative, digest in seal["files"].items():
        a.bind(width.contained(a, run, relative), digest)
    a.need(
        cfg["schema"] == "walker-student-update-w2-v1"
        and cfg["width"] == 128
        and cfg["updates"] == 10000
        and cfg["endpoints"] == [2000, 10000]
        and cfg["checkpoints"] == [0, 2000, 10000]
        and cfg["rows"] == 4000
        and cfg["validation_rows"] == 2000
        and cfg["batch_size"] == 256
        and cfg["lr"] == 0.0003
        and cfg["eta_theta"] == 0
        and cfg["prox_reduction"] == "mean"
        and cfg["anchor_refresh"] == 0,
        "Frozen sole update-budget contrast",
    )
    a.need(
        cfg["device"] == "cpu"
        and cfg["torch_threads"] == 1
        and cfg["gradient_clip"] == 0.5
        and cfg["normalization_floor"] == 0.05
        and cfg["episode_steps"] == 1000,
        "Unchanged numerical/execution settings",
    )
    a.prove(
        cfg["gates"],
        dict(
            minimum_validation_gain=0.1,
            minimum_retention_gain=0.1,
            minimum_cost_gain=0.05,
            minimum_positive_pairs=2,
            minimum_walking_gain=2,
            minimum_walking_pairs=2,
            competence_minimum_successes=15,
        ),
        "Exact predeclared separate thresholds",
    )
    a.prove(
        cfg["budgets"],
        dict(
            requested_controls=230000,
            neural_rows=8250000,
            backward_rows=7680000,
            optimizer_steps=30000,
            episodes=228,
            checkpoint_loads=3,
            model_controls=0,
            wall_seconds=3600,
            storage_bytes=17179869184,
        ),
        "Exact finite experiment ceilings",
    )
    a.prove(
        seal["planned_work"],
        dict(
            requested_controls=228000,
            neural_rows=7962000,
            backward_rows=7680000,
            optimizer_steps=30000,
            episodes=228,
            checkpoint_loads=3,
            model_controls=0,
            diagnostic_forward_rows=54000,
        ),
        "Full conditional roster reservation",
    )
    a.need(
        cfg["bootstrap_seed"] == 2026110299 and cfg["bootstrap_draws"] == 2000,
        "Frozen descriptive uncertainty recipe",
    )
    a.prove(
        cfg["seed_pairs"],
        [dict(policy_seed=20261102001 + pair, fit_seed=1203001000 + pair) for pair in range(3)],
        "Fresh paired initialization/sampler seeds",
    )
    for cohort, root, count in (
        ("curve", 1210000000, 4),
        ("primary", 1220000000, 16),
        ("confirmation", 1230000000, 16),
    ):
        a.need(
            cfg[cohort + "_seeds"] == list(range(root, root + count)),
            "Fixed new paired reset cohort",
        )
    data, normal, provenance = audit_reused_inputs(a, run)
    stage = a.ledger(run)
    result = stage["result"]
    a.need(
        result["execution_complete"] is True
        and result["error"] is None
        and result["status"] == "complete",
        "Valid closed W2 execution",
    )
    a.need(
        result["config_sha256"] == seal["config_sha256"]
        and result["input_manifest_sha256"] == native.sha(run / "input_manifest.json"),
        "Worker launch identity",
    )
    names = {point_name(pair, update) for pair in range(3) for update in (0, 2000, 10000)}
    endpoints = {point_name(pair, update) for pair in range(3) for update in (2000, 10000)}
    a.need(
        set(result["points"]) == names and set(result["samplers"]) == {"0", "1", "2"},
        "All nine points and three shared samplers",
    )
    fits = {row["point"]: row for row in result["fits"]}
    reloads = {row["pair"]: row for row in result["reloads"]}
    a.need(
        set(fits) == endpoints
        and len(result["fits"]) == 6
        and set(reloads) == {0, 1, 2}
        and len(result["reloads"]) == 3,
        "Exactly six fit blocks and three 2k loads",
    )
    payloads, diagnostic_rows = {}, 0
    for pair, seeds in enumerate(cfg["seed_pairs"]):
        sampler = result["samplers"][str(pair)]
        a.need(sampler["seed"] == seeds["fit_seed"], "Private minibatch seed owner")
        indices = a.group(run / "evidence/raw.h5", sampler["minibatches"])["indices"]
        rng_states = width.audit_sampler(
            a,
            indices,
            sampler["rng_at_checkpoints"],
            seed=seeds["fit_seed"],
            rows=4000,
            batch_size=256,
            updates=10000,
            checkpoints=[0, 2000, 10000],
        )
        initial = previous = None
        for update in (0, 2000, 10000):
            name = point_name(pair, update)
            point = result["points"][name]
            a.prove(point, dict(pair=pair, width=128, update=update, **seeds), "Point identity")
            binding = point["checkpoint"]
            path = width.contained(a, run, binding["file"])
            a.bind(path, binding["sha256"])
            a.need(path.stat().st_size == binding["bytes"], "Exact checkpoint byte count")
            payload = width.decode_checkpoint(a, path, 128, normal)
            a.need(
                payload["seed"] == seeds["policy_seed"]
                and payload["policy_hash"] == binding["policy_hash"]
                and native.fingerprint(payload["optimizer_state"]) == point["optimizer_fingerprint"]
                and point["parameter_count"] == 19590,
                "Checkpoint/optimizer identity and unchanged width",
            )
            a.prove(
                payload["metadata"],
                dict(
                    pair=pair,
                    width=128,
                    updates=update,
                    feature_mode="clipped",
                    fresh_initialization=True,
                    deployment_qualified=False,
                ),
                "Checkpoint role and fixed endpoint metadata",
            )
            if update == 0:
                a.need(payload["optimizer_state"] is None, "Fresh Adam initialization")
                initial = payload
            else:
                fit_binding = fits[name]
                a.bind(width.contained(a, run, fit_binding["file"]), fit_binding["sha256"])
                fit = a.load(run / fit_binding["file"])
                width.audit_optimizer(
                    a, payload, initial, previous, point, fit, rng_states, data["train"], cfg
                )
                a.need(
                    torch.equal(payload["private_rng_state"], initial["private_rng_state"]),
                    "Private initialization RNG unchanged",
                )
                if update == 2000:
                    audit_reload(a, reloads[pair], payload, binding, pair)
            predictions = a.group(run / "evidence/raw.h5", point["predictions"])
            diagnostic_rows += width.audit_prediction_metrics(a, point, data, predictions)
            payloads[name] = payload
            previous = payload
    a.prove(
        result["teacher_outcomes"],
        {key: value["outcomes"] for key, value in provenance["splits"].items()},
        "Unchanged complete recorded teacher outcomes",
    )
    gates = result["gates"]
    primary = recompute_contrast(result["points"], result["evaluations"]["primary"], cfg)
    a.prove(gates["primary"], primary, "Independent primary separate screens")
    trigger = any(primary[key + "_screen"] for key in ("retention", "cost", "walking")) or any(
        result["evaluations"]["primary"][name]["summary"]["successes"] >= 15 for name in endpoints
    )
    a.need(gates["confirmation_triggered"] == trigger, "Predeclared confirmation trigger")
    confirmation = (
        recompute_contrast(result["points"], result["evaluations"]["confirmation"], cfg)
        if trigger
        else None
    )
    a.prove(gates["confirmation"], confirmation, "Disjoint confirmation separate screens")
    for mechanism, field in (
        ("retention", "confirmed_retention_benefit"),
        ("cost", "confirmed_cost_benefit"),
        ("walking", "confirmed_walking_gain"),
    ):
        a.need(
            gates[field]
            == bool(
                trigger and primary[mechanism + "_screen"] and confirmation[mechanism + "_screen"]
            ),
            "Confirmation requires the same screen in both panels",
        )
    seen, reset_owners, controls = set(), {}, 0
    for cohort in ("curve", "primary", "confirmation"):
        records = result["evaluations"][cohort]
        expected = (
            names if cohort == "curve" else endpoints if cohort == "primary" or trigger else set()
        )
        a.need(set(records) == expected, "Complete predeclared checkpoint/cohort roster")
        for name, record in records.items():
            a.need(
                record["policy_hash"] == payloads[name]["policy_hash"]
                and record["optimizer_fingerprint"]
                == native.fingerprint(payloads[name]["optimizer_state"]),
                "Immutable evaluation policy/Adam owner",
            )
            a.summary(stage, record["summary"])
            rows = record["episodes"]
            a.need(
                [row["seed"] for row in rows] == cfg[cohort + "_seeds"]
                and record["summary"]["tags"] == [row["tag"] for row in rows],
                "Exact paired reset order",
            )
            for row in rows:
                tag = row["tag"]
                a.need(tag not in seen, "Unique actual episode")
                seen.add(tag)
                event, arrays = a.episode(stage, tag)
                a.prove(event, row, "Complete result/ledger episode ownership")
                reset = native.fingerprint(
                    {key: arrays[key][0] for key in ("states", "warmstarts", "integrations")}
                )
                a.need(
                    reset_owners.setdefault(row["seed"], reset) == reset,
                    "Paired full simulator reset context",
                )
                for info in arrays["infos"]["rows"]:
                    a.prove(
                        info["controller"],
                        dict(policy_hash=record["policy_hash"], feature_mode="clipped", width=128),
                        "Every standalone torque owner",
                    )
                for before, _ in stage["live"][tag]:
                    a.need(
                        before["context"]["point"] == name
                        and before["context"]["cohort"] == cohort,
                        "Live query checkpoint/cohort owner",
                    )
                controls += row["steps"]
    a.need(seen == set(stage["episodes"]), "Every native episode reported")
    for name in endpoints:
        a.need(
            gates["competence"][name]
            == bool(
                trigger
                and result["evaluations"]["primary"][name]["summary"]["successes"] >= 15
                and result["evaluations"]["confirmation"][name]["summary"]["successes"] >= 15
            ),
            "Same fixed policy qualifies on both panels",
        )
    counts = stage["counts"]
    a.need(
        diagnostic_rows == 54000
        and counts["completed_neural_rows"] == 7680000 + diagnostic_rows + controls,
        "Exact fitting/reconstruction/standalone forward census",
    )
    a.need(
        counts["completed_backward_rows"] == 7680000
        and counts["completed_optimizer_steps"] == 30000
        and counts["completed_checkpoint_loads"] == 3,
        "Three learning streams and three actual checkpoint loads",
    )
    load_events = [
        event
        for event in stage["events"]["measurement_requested"]
        if event["measure_kind"] == "checkpoint_loads"
    ]
    a.need(
        len(load_events) == 3 and all(row["amount"] == 1 for row in load_events),
        "Three separately metered serialization boundaries",
    )
    a.need(
        {row["context"]["point"] for row in load_events}
        == {point_name(pair, 2000) for pair in range(3)}
        and all(row["context"]["role"] == "update_budget_checkpoint_reload" for row in load_events),
        "Each actual load belongs to its paired 2k checkpoint",
    )
    a.need(
        counts["completed_live_controls"] == controls
        and counts["requested_model_controls"] == 0
        and counts["completed_physics_substeps"] == 4 * controls,
        "No new teacher/model physics",
    )
    for key in (
        "failed_queries",
        "failed_measurements",
        "rejected_queries",
        "rejected_measurements",
        "reset_errors",
        "unknown_partial_controls",
        "unknown_partial_neural_rows",
        "unknown_partial_backward_rows",
        "unknown_partial_optimizer_steps",
        "unknown_partial_checkpoint_loads",
        "restores",
    ):
        a.need(counts[key] == 0, "No failed/rejected/unknown/extra work " + key)
    for kind, cap in cfg["budgets"].items():
        if kind in ("wall_seconds", "storage_bytes"):
            continue
        key = kind if kind in ("requested_controls", "episodes") else "requested_" + kind
        a.need(counts[key] <= cap, "Finite predeclared budget " + kind)
    a.need(result["wall_seconds"] <= cfg["budgets"]["wall_seconds"], "Worker wall limit")
    a.need(
        sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
        <= cfg["budgets"]["storage_bytes"],
        "Closed evidence storage limit",
    )
    return dict(
        points=len(payloads),
        fit_blocks=6,
        reloaded_checkpoints=3,
        training_streams=3,
        episodes=len(seen),
        actual_controls=controls,
        diagnostic_forward_rows=diagnostic_rows,
        fitting_rows=7680000,
        confirmation_triggered=trigger,
        gates=gates,
        work=dict(counts),
        limitations=[
            "Saved checkpoint/ledger and sampler identities are verified without neural forwards, simulator replay or optimizer arithmetic.",
            "The two endpoints share one training/minibatch prefix per seed; only three fresh training streams are independent intervention replications.",
            "Retention benefit, cost benefit, walking gain and competence are separate claims; W2 cannot establish adaptive coupling.",
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
    a = UpdateBudgetAudit(run, run / "source")
    findings, error = {}, None
    try:
        for path in (Path(__file__), Path(native.__file__), Path(width.__file__)):
            a.bind(path)
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
                    ("src.gps.mppi_gps.compatibility_metrics", "src.gps.mppi_gps.update_inputs")
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
        schema="walker-student-update-w2-independent-audit-v1",
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
        scope="Independent saved predictions, exact W1 corpus reuse, checkpoint/Adam/RNG continuation including actual 2k reload receipts, complete native outcomes/reset pairing, separate frozen gates and total finite work. No networks, simulator or optimizer arithmetic executed.",
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
