"""Independent saved-only audit of the fixed v3.1 guided-data comparison.

This reconciles recorded demonstrations and standalone outcomes without policy
inference, fitting or physics. The immutable original remains in its capsule.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch

from . import common
from .common import assert_independent_imports, load_native_checker


def run(args: argparse.Namespace) -> int:
    study = args.run.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native_checker = load_native_checker(args.native_checker)
    audit = native_checker.Audit(study, study / "source")
    started = time.perf_counter()
    error = None
    report = {}
    try:
        audit.bind(Path(__file__))
        audit.bind(Path(common.__file__))
        audit.bind(Path(native_checker.__file__))
        manifest = audit.load(study / "manifest.json")
        config = audit.load(study / "config.json")
        audit.bind(study / "config.json", manifest["config_sha256"])
        audit.bind(study / "input.pt", manifest["input_sha256"])
        for row in manifest["files"]:
            audit.bind(study / "source" / row["file"], row["sha256"])
        seal = audit.load(study / "input_manifest.json")
        audit.need(
            seal["config_sha256"] == native_checker.sha(study / "config.json")
            and seal["source_manifest_sha256"] == native_checker.sha(study / "manifest.json"),
            "Preparation seal",
        )
        for name, sha in seal["files"].items():
            audit.bind(study / name, sha)
        audit.need(
            native_checker.sha(study / "config.json") == args.config_sha, "Prelaunch frozen config"
        )
        audit.need(
            native_checker.sha(study / "input_manifest.json") == args.seal_sha,
            "Prelaunch input seal",
        )
        stage = audit.ledger(study)
        result = stage["result"]
        audit.need(result["passed"] and result["error"] is None, "Valid physical/fit completion")
        audit.need(
            result["protocol_sha256"] == args.config_sha
            and result["input_manifest_sha256"] == args.seal_sha,
            "Worker bound prelaunch configuration",
        )
        audit.need(
            config["common_original_rows"] == 4064
            and config["common_recovery_rows"] == 32
            and (config["teacher_rows"] == 2000),
            "Fixed 4064+32+2000 roster",
        )
        audit.need(
            config["updates"] == 2000
            and config["checkpoints"] == [500, 1000, 2000]
            and (config["batch_size"] == 256),
            "Fixed update roster",
        )
        task = audit.load(study / "inputs/task.json")
        frozen_task = json.loads((study / "source/configs/mppi_gps_walker.json").read_text())
        audit.need(task["task"] == frozen_task["task"], "Unchanged native task")
        audit.need(
            task["episode_steps"] == 1000
            and task["task"]["target_velocity"] == 1.5
            and (task["task"]["frame_skip"] == 4),
            "Native horizon/target/clock",
        )
        provenance = audit.load(study / "inputs/provenance.json")
        datasets = {}
        for arm in ["independent", "guided"]:
            with np.load(study / f"inputs/{arm}.npz", allow_pickle=False) as f:
                datasets[arm] = {k: np.array(f[k]) for k in ["observations", "actions"]}
            for k, width in [("observations", 17), ("actions", 6)]:
                audit.need(
                    datasets[arm][k].shape == (6096, width)
                    and datasets[arm][k].dtype == np.float64,
                    "Actual 6096 float64 pairs",
                )
                audit.need(np.isfinite(datasets[arm][k]).all(), "Finite pairs")
            audit.need(np.all(np.abs(datasets[arm]["actions"]) <= 1), "Bounded recorded labels")
        for k in ["observations", "actions"]:
            audit.need(
                np.array_equal(datasets["independent"][k][:4096], datasets["guided"][k][:4096]),
                "Identical shared prefix",
            )
            for label, chunk in [
                ("original", datasets["independent"][k][:4064]),
                ("recovery", datasets["independent"][k][4064:4096]),
            ]:
                matching = [
                    v for path, v in provenance["datasets"][label].items() if path.endswith("/" + k)
                ]
                audit.need(len(matching) == 1, "Unique original/recovery pair owner")
                audit.desc(matching[0], chunk)
        teacher_rows = {}
        teacher_hdf = study / "inputs/teachers.h5"
        audit.bind(teacher_hdf)
        with h5py.File(teacher_hdf, "r") as f:
            for arm, tagarm in [("independent", "raw"), ("guided", "guided1")]:
                for i in range(2):
                    tag = f"v2_r0_trial_{tagarm}_{i}"
                    g = f[tag]
                    event = next(
                        (
                            x
                            for x in provenance["teacher_summaries"][arm]["events"]
                            if x["tag"] == tag
                        )
                    )
                    d = {}
                    for key, meta in event["raw"]["arrays"].items():
                        ds = g[key]
                        dtype = np.dtype(meta["dtype"])
                        v = np.asarray(ds[()])
                        v = (
                            np.char.decode(v, "utf-8").astype(dtype)
                            if dtype.kind == "U"
                            else v.astype(dtype, copy=False)
                        )
                        audit.need(
                            list(v.shape) == meta["shape"]
                            and hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
                            == meta["sha256"]
                            == ds.attrs["sha256"],
                            "Actual supplied teacher arrays",
                        )
                        d[key] = json.loads(str(v.item())) if meta["encoding"] == "json" else v
                    teacher_rows[tag] = d
                    audit.need(
                        event["complete"]
                        and event["success"]
                        and event["survived"]
                        and (event["steps"] == 1000),
                        "Actual successful teachers",
                    )
                    audit.need(
                        np.array_equal(d["observations"], native_checker.obs(d["states"][:-1])),
                        "Teacher preaction labels",
                    )
                    audit.equal(
                        d["costs"].sum() + d["terminal_adjustment"] + d["absorbing_cost"],
                        event["cost"],
                        "Teacher complete cost",
                    )
                    for k in ["observations", "actions"]:
                        audit.need(
                            np.array_equal(
                                datasets[arm][k][4096 + i * 1000 : 4096 + (i + 1) * 1000], d[k]
                            ),
                            "Actual full teacher suffix",
                        )
            for i in range(2):
                for k in ["states", "warmstarts", "integrations"]:
                    audit.need(
                        np.array_equal(
                            teacher_rows[f"v2_r0_trial_raw_{i}"][k][0],
                            teacher_rows[f"v2_r0_trial_guided1_{i}"][k][0],
                        ),
                        "Matched teacher full reset",
                    )
        parent = audit.checkpoint(study / "input.pt")
        parent_optimizer = parent["optimizer_state"]
        owner = parent["policy_hash"]
        audit.need(
            owner == config["parent_hash"] == result["policy_hashes"]["parent"],
            "Incoming policy owner",
        )
        audit.need(
            native_checker.fingerprint(parent_optimizer)
            == result["matching"]["incoming_optimizer_tree_sha256"]
            and parent_optimizer["rng"] == result["matching"]["incoming_pcg64"],
            "Exact saved incoming optimizer/RNG",
        )
        audit.need(
            parent_optimizer["fit_data"]
            == dict(
                rows=4064,
                observations_sha256=native_checker.fit_hash(
                    datasets["independent"]["observations"][:4064]
                ),
                actions_sha256=native_checker.fit_hash(datasets["independent"]["actions"][:4064]),
            ),
            "Parent original training input",
        )
        normal = {k: parent["state_dict"]["observation_" + k].numpy() for k in ["mean", "std"]}
        audit.need(
            native_checker.fingerprint(normal) == result["matching"]["normalization_sha256"],
            "Parent fixed normalization",
        )
        rng = np.random.Generator(np.random.PCG64(0))
        rng.bit_generator.state = deepcopy(parent_optimizer["rng"])
        indices = []
        rng_at = {}
        for t in range(1, 2001):
            indices.append(rng.integers(0, 6096, size=256))
            if t in [500, 1000, 2000]:
                rng_at[t] = deepcopy(rng.bit_generator.state)
        saved = audit.group(study / "evidence/raw.h5", result["matching"]["minibatch_indices"])
        audit.need(
            np.array_equal(saved["indices"], np.stack(indices))
            and result["matching"]["expected_terminal_pcg64"] == rng_at[2000],
            "Exact reconstructed common minibatches",
        )
        curves = {}
        owners = {"parent": owner}
        for arm in ["independent", "guided"]:
            previous_owner = owner
            curve = []
            refresh = []
            for checkpoint, block in [(500, 500), (1000, 500), (2000, 1000)]:
                checkpoint_data = audit.checkpoint(study / f"checkpoints/{arm}_{checkpoint}.pt")
                optimizer_state = checkpoint_data["optimizer_state"]
                fit_record = audit.load(study / f"learning/{arm}_{checkpoint}.json")
                name = f"{arm}_{checkpoint}"
                owners[name] = checkpoint_data["policy_hash"]
                audit.need(
                    checkpoint_data["policy_hash"]
                    == result["policy_hashes"][name]
                    == optimizer_state["policy_hash"]
                    == fit_record["child_policy_hash"],
                    "Child policy package owner",
                )
                audit.need(
                    optimizer_state["updates"] == 10000 + checkpoint
                    and optimizer_state["fit_updates"] == checkpoint
                    and (optimizer_state["rng"] == rng_at[checkpoint]),
                    "Checkpoint update/RNG position",
                )
                audit.need(
                    all(
                        (
                            float(v["step"]) == 10000 + checkpoint
                            for v in optimizer_state["adam"]["state"].values()
                        )
                    ),
                    "Adam per-parameter counters",
                )
                audit.need(
                    torch.equal(checkpoint_data["private_rng_state"], parent["private_rng_state"]),
                    "Unchanged private Torch generator",
                )
                for k in ["mean", "std"]:
                    audit.need(
                        torch.equal(
                            checkpoint_data["state_dict"]["observation_" + k],
                            parent["state_dict"]["observation_" + k],
                        ),
                        "Unchanged normalization buffers",
                    )
                audit.need(
                    optimizer_state["fit_config"]
                    == dict(
                        batch_size=256,
                        lr=0.0003,
                        eta_theta=0.01,
                        gradient_clip=0.5,
                        prox_reduction="mean",
                        anchor_refresh=250,
                    ),
                    "Frozen child objective",
                )
                audit.need(
                    optimizer_state["fit_data"]
                    == fit_record["fit_data"]
                    == dict(
                        rows=6096,
                        observations_sha256=native_checker.fit_hash(datasets[arm]["observations"]),
                        actions_sha256=native_checker.fit_hash(datasets[arm]["actions"]),
                    ),
                    "Actual fitting target binding",
                )
                audit.need(
                    fit_record["parent_policy_hash"] == previous_owner
                    and fit_record["resume_fit"] == (checkpoint > 500)
                    and fit_record["optimizer_continued"]
                    and (fit_record["completed_updates"] == block),
                    "Same-block checkpoint continuation",
                )
                audit.need(
                    fit_record["fit_updates"] == checkpoint
                    and fit_record["anchor_policy_hash"] == optimizer_state["anchor_policy_hash"],
                    "Active proximal anchor packaging",
                )
                audit.need(
                    native_checker.tensor_hash(optimizer_state["anchor"])
                    == optimizer_state["anchor_policy_hash"],
                    "Stored anchor tensor hash",
                )
                refresh.extend(fit_record["anchor_refreshes"])
                curve.extend(fit_record["learning_curve"])
                previous_owner = checkpoint_data["policy_hash"]
                for k, amount in [
                    ("neural_rows", 256 * block),
                    ("backward_rows", 256 * block),
                    ("optimizer_steps", block),
                ]:
                    audit.need(
                        fit_record["work"][k]
                        == dict(requested=amount, completed=amount, unknown_partial=0),
                        "Exact fit work budget",
                    )
            audit.need(
                [x["before_fit_update"] for x in refresh] == list(range(251, 2001, 250)),
                "Anchor chronology across split checkpoints",
            )
            audit.need(
                curve == result["learning_curves"][arm] and len(curve) == 2000,
                "Complete saved learning curve",
            )
            for t, point in enumerate(curve, 1):
                audit.need(
                    point["update"] == 10000 + t and point["fit_update"] == t,
                    "Continuous learning curve index",
                )
            curves[arm] = curve
        seen = Counter()
        audit.bind(study / "learning_curves.csv")
        with (study / "learning_curves.csv").open() as f:
            for row in csv.DictReader(f):
                point = curves[row["arm"]][seen[row["arm"]]]
                seen[row["arm"]] += 1
                audit.need(
                    all((float(row[k]) == point[k] for k in point)), "Curve CSV actual point"
                )
        audit.need(seen == dict(independent=2000, guided=2000), "Exactly two 2000-point fits")
        traces = {}
        for tag in stage["episodes"]:
            event, d = audit.episode(stage, tag)
            traces[tag] = d
            name = next(
                (
                    name
                    for name in owners
                    if all(
                        (
                            info["controller"].get("policy_hash") == owners[name]
                            for info in d["infos"]["rows"]
                        )
                    )
                )
            )
            for info in d["infos"]["rows"]:
                audit.need(
                    info["controller"] == dict(policy_hash=owners[name], feature_mode="state"),
                    "Standalone policy controller only",
                )
        for value in result["standalone_curves"].values():
            audit.summary(stage, value)
        for cohort, scores in result["cases"].items():
            for value in scores.values():
                audit.summary(stage, value)
            for i in range(16):
                reference = traces[scores["parent"]["tags"][i]]
                for arm in ["independent", "guided"]:
                    for k in ["states", "warmstarts", "integrations"]:
                        audit.need(
                            np.array_equal(reference[k][0], traces[scores[arm]["tags"][i]][k][0]),
                            "Fresh paired standalone full reset",
                        )
                    for baseline in ["parent", "independent"]:
                        audit.need(
                            stage["episodes"][scores[arm]["tags"][i]]["seed"]
                            == stage["episodes"][scores[baseline]["tags"][i]]["seed"]
                            == config[f"{cohort}_root"] + i,
                            "Fresh paired standalone seed",
                        )
            g = scores["guided"]
            comparisons = {}
            for baseline in ["parent", "independent"]:
                control = scores[baseline]
                gain = (control["mean_cost"] - g["mean_cost"]) / control["mean_cost"]
                checks = dict(
                    cost_gain=gain >= 0.05,
                    success_nonregression=g["successes"] >= control["successes"],
                    survival_nonregression=g["survived"] >= control["survived"],
                    mean_steps_nonregression=g["mean_steps"] >= control["mean_steps"],
                )
                comparisons[baseline] = dict(
                    passed=all(checks.values()),
                    checks=checks,
                    relative_complete_cost_gain=gain,
                    paired_cost_deltas_guided_minus_control=[
                        x - y for x, y in zip(g["costs"], control["costs"])
                    ],
                )
            audit.need(
                result[cohort + "_gate"]
                == dict(
                    schema="walker-finite-post-update-feedback-v3.1",
                    passed=all((x["passed"] for x in comparisons.values())),
                    controls=comparisons,
                    minimum_relative_cost_gain=0.05,
                    action_agreement_is_gate=False,
                ),
                "Independent complete-outcome gate calculation",
            )
        for i in range(4):
            reference = traces[result["standalone_curves"]["parent"]["tags"][i]]
            for value in result["standalone_curves"].values():
                tag = value["tags"][i]
                audit.need(
                    stage["episodes"][tag]["seed"] == config["curve_root"] + i, "Paired curve seed"
                )
                for k in ["states", "warmstarts", "integrations"]:
                    audit.need(
                        np.array_equal(reference[k][0], traces[tag][k][0]),
                        "Paired curve full reset",
                    )
        steps = sum((x["steps"] for x in stage["episodes"].values()))
        work = stage["counts"]
        audit.need(
            work["completed_controls"] == steps
            and work["completed_neural_rows"] == 1024000 + 14 * 6096 + steps,
            "Exact training/diagnostic/evaluation roster",
        )
        audit.need(
            work["completed_backward_rows"] == 1024000
            and work["completed_optimizer_steps"] == 4000
            and (work["completed_model_controls"] == 0),
            "Exact backward/optimizer/no-planner roster",
        )
        for k in ["controls", "neural_rows", "backward_rows", "optimizer_steps"]:
            audit.need(
                work["requested_" + k]
                <= config["budgets"]["requested_controls" if k == "controls" else k],
                "Finite configured budget",
            )
        if not result["primary_gate"]["passed"]:
            audit.need(
                "confirmation" not in result["cases"]
                and result["status"] == "primary_negative"
                and (not result["feedback"]["advance"])
                and (result["feedback"]["actor_hash"] == owner)
                and (result["feedback"]["guided_rho"] == 0),
                "Frozen negative stopping and parent retention",
            )
        audit.need(
            result["loop_closed"] is False and result["action_proxy_used_for_gate"] is False,
            "No loop or action-proxy claim",
        )
        for arm in ["independent", "guided"]:
            expected = dict(
                teacher_mean_cost=provenance["teacher_summaries"][arm]["mean_cost"],
                student_mean_cost=result["cases"]["primary"][arm]["mean_cost"],
                gap=result["cases"]["primary"][arm]["mean_cost"]
                - provenance["teacher_summaries"][arm]["mean_cost"],
                scope="Descriptive: saved teacher acquisition starts differ from fresh paired student evaluation starts.",
            )
            audit.need(
                result["teacher_student_gap"][arm] == expected,
                "Teacher/student descriptive cost gap",
            )
        report = dict(
            status=result["status"],
            cases=result["cases"],
            standalone_curves=result["standalone_curves"],
            primary_gate=result["primary_gate"],
            teacher_student_gap=result["teacher_student_gap"],
            policy_hashes=result["policy_hashes"],
            exact_work=dict(work),
            full_mse_recorded_not_recomputed=result["full_mse"],
        )
        for path, before in list(audit.files.items()):
            audit.need(audit.bind(Path(path)) == before, "Inputs unchanged after audit")
        for path, (_, size, mtime) in audit.hdfs.items():
            audit.need(
                path.stat().st_size == size and path.stat().st_mtime_ns == mtime,
                "Closed HDF unchanged",
            )
        assert_independent_imports(audit)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        for h, _, _ in audit.hdfs.values():
            h.close()
    receipt = dict(
        schema="walker-guided-data-independent-audit-v1",
        passed=error is None,
        error=error,
        checks=audit.checks,
        bound_files=len(audit.files),
        hashed_arrays=len(audit.arrays),
        new_controls=0,
        new_neural_rows=0,
        new_backward_rows=0,
        new_optimizer_steps=0,
        wall_seconds=time.perf_counter() - started,
        scope="Actual supplied and executed arrays, native complete-cost arithmetic, full state/warmstart/integration snapshots, native flags, paired starts, source/input seals, exact saved checkpoint/Adam/RNG/anchor packaging, common minibatch reconstruction, complete curves and ledger work. No inference, physics or optimizer arithmetic replay. Full-data MSE is retained and bound, not recomputed.",
    )
    for name, value in [
        ("receipt.json", receipt),
        ("findings.json", report),
        ("file_bindings.json", audit.files),
        ("array_bindings.json", audit.arrays),
    ]:
        with (out / name).open("x") as f:
            json.dump(native_checker.plain(value), f, indent=2, allow_nan=False)
            f.write("\n")
    print(json.dumps(receipt))
    return int(error is not None)


def main(argv: list[str] | None = None) -> int:
    """Audit an explicitly supplied closed v3.1 run into a fresh directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--native-checker",
        type=Path,
        help="Optional immutable external native checker; default is the maintained package implementation",
    )
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
