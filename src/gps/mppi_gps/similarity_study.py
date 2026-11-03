"""J1 study orchestration and create-only CLI; heavy execution imports stay lazy."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .proposal_evidence import read_array
from .provenance import default, sha, write

ROOT = Path(__file__).resolve().parents[3]
RECIPE = ROOT / "configs/studies/similarity_j1_0.json"


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def binding(path: Path, root: Path) -> dict:
    return dict(file=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path))


def append_metadata(path: Path, value: dict, meter: Any) -> None:
    """Append a durable completed record; final catalogs are created only once."""
    payload = (
        json.dumps(value, default=default, allow_nan=False, separators=(",", ":")) + "\n"
    ).encode()
    meter.reserve_external(len(payload))
    with path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def clean_exit(value: dict) -> bool:
    return (
        type(value.get("exit_code")) is int
        and value["exit_code"] == 0
        and value.get("leader_reaped") is True
        and value.get("unknown_partial_work_possible") is False
        and value.get("process_group_kill_requested") is False
        and value.get("stopping_reason") is None
        and type(value.get("pid")) is int
    )


def runtime_gate(config: dict) -> None:
    expected = config["runtime"]
    for name, actual in dict(
        host=platform.node(), python=sys.version, platform=platform.platform()
    ).items():
        if actual != expected[name]:
            raise ValueError("Pinned J1 runtime differs: " + name)
    for name, version in expected["distributions"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("Pinned J1 distribution differs: " + name)
    if sorted(os.sched_getaffinity(0)) != expected["cpu_affinity"]:
        raise ValueError("J1 CPU affinity differs")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_GL", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(name) != expected[name]:
            raise ValueError("J1 runtime environment differs: " + name)


def verify(run: Path) -> dict:
    from .lifecycle import verify as verify_source

    verify_source(run)
    config = load(run / "config.json")
    if (
        config.get("schema") != "walker-similarity-j1-execution-v1"
        or config.get("automatic_followup") is not False
    ):
        raise ValueError("Unknown J1 scope")
    if config["guidance_launch_authorized"] is not False:
        raise ValueError("J1 cannot dispatch guidance")
    if sha(run / "source" / config["protocol"]) != config["protocol_sha256"]:
        raise ValueError("Frozen J1 protocol changed")
    return config


def input_gate(run: Path) -> None:
    for name, value in load(run / "input_manifest.json")["files"].items():
        path = (run / name).resolve()
        if (
            not path.is_relative_to(run.resolve())
            or path.is_symlink()
            or path.stat().st_size != value["bytes"]
            or sha(path) != value["sha256"]
        ):
            raise ValueError("Frozen J1 input differs: " + name)


def freeze(recipe: Path, out: Path, source_commit: str) -> None:
    from .lifecycle import freeze as freeze_source

    config = load(recipe)
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("A published immutable source commit is required")
    config.update(
        source_commit=source_commit,
        recipe_sha256=sha(recipe),
        protocol_sha256=sha(ROOT / config["protocol"]),
    )
    freeze_source(out, config, "similarity-j1")


def prepare(run: Path) -> None:
    """Bind pinned small inputs and an offline rendering probe; no live physics or actor."""
    config = verify(run)
    runtime_gate(config)
    started = time.monotonic()
    prepared = run / "prepared"
    prepared.mkdir(exist_ok=False)
    reads = 0
    files = {}
    for name in ("reference", "normalization"):
        expected = config["inputs"][name]
        source = Path(expected["path"])
        data = source.read_bytes()
        reads += len(data)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("J1 pinned input changed: " + name)
        destination = prepared / ("reference.pt" if name == "reference" else "normalization.npz")
        destination.write_bytes(data)
        files[str(destination.relative_to(run))] = binding(destination, run)
    xml = run / "source" / config["inputs"]["xml"]["file"]
    if sha(xml) != config["inputs"]["xml"]["sha256"]:
        raise ValueError("J1 Walker XML changed")
    # Freshness is a declared role-separated seed contract, not a wall-clock seed.
    role_seeds: dict[str, set[int]] = {}
    for group in config["groups"]:
        for role in ("acquisition_seed", "planner_seed"):
            role_seeds.setdefault(role, set()).add(group[role])
        for learner in group["learners"]:
            for role in ("policy_seed", "fit_seed"):
                role_seeds.setdefault(role, set()).add(learner[role])
        role_seeds.setdefault("evaluation", set()).update(group["learners"][0]["evaluation_seeds"])
    seen = set()
    for role, seeds in role_seeds.items():
        if seen.intersection(seeds):
            raise ValueError("J1 seed roles overlap: " + role)
        seen.update(seeds)
    if len(config["groups"]) != 16 or {g["split"] for g in config["groups"]} != {
        "development",
        "protected",
    }:
        raise ValueError("J1 group roster differs")
    if shutil.disk_usage(run).free < config["limits"]["global_output_bytes"]:
        raise ValueError("J1 retained-output capacity unavailable")
    if (
        reads > config["limits"]["preparation_read_bytes"]
        or time.monotonic() - started > config["limits"]["preparation_wall_seconds"]
    ):
        raise ValueError("J1 preparation budget exhausted")
    from .similarity_atlas import renderer_preflight

    render = renderer_preflight(xml)
    write(prepared / "renderer.json", render)
    files["prepared/renderer.json"] = binding(prepared / "renderer.json", run)
    write(run / "input_manifest.json", dict(schema="walker-j1-inputs-v1", files=files))
    write(
        run / "preparation.json",
        dict(
            passed=True,
            read_bytes=reads,
            wall_seconds=time.monotonic() - started,
            seed_roles={k: sorted(v) for k, v in role_seeds.items()},
            new_neural_rows=0,
            new_physics=0,
            new_fitting=0,
        ),
    )


def trace_context(trace: dict, index: int = 0) -> dict:
    return dict(
        state=trace["states"][index],
        warmstart=trace["warmstarts"][index],
        integration=trace["integrations"][index],
        spec=np.array(int(trace["integration_spec"])),
        dimensions=np.array([9, 9, 6]),
    )


def metric_trace(trace: dict) -> dict:
    result = dict(trace)
    if len(result["observations"]) == len(result["actions"]) + 1:
        result["observations"] = result["observations"][:-1]
    result["done"] = np.asarray(result["done"], dtype=bool)
    return result


def scientific_worker(run: Path) -> dict:
    """One frozen development/final block; never train against a similarity score."""
    import torch

    from src.utils.config import MPPIConfig

    from .diagnostic_policies import LegacySACActor
    from .search import CoupledSelector, IndependentMPPI
    from .similarity_execution import JMeter, NativeRunner, fit_case, parent_prepare
    from .similarity_metrics import (
        calibration_suite,
        evaluate_predictions,
        fit_occupancy_normalizer,
        fit_prediction_models,
        fit_state_explainer,
        motion_metrics,
        predict_case_scores,
        score_case,
        score_definitions,
    )
    from .similarity_probe import decision_roster, run_decision, summarize_probe

    config = verify(run)
    runtime_gate(config)
    input_gate(run)
    if torch.get_num_threads() != 1:
        raise ValueError("J1 Torch thread count changed")
    torch.manual_seed(2026129600)
    worker = run / "worker"
    worker.mkdir(exist_ok=False)
    meter = JMeter(worker / "evidence", config)
    cases: list[dict] = []
    traces: list[dict] = []
    teachers: dict[str, dict] = {}
    parents: dict[str, dict] = {}
    result = dict(
        schema="walker-similarity-j1-result-v1",
        status="INCOMPLETE",
        automatic_followup=False,
        guidance_launched=False,
    )
    runner = None
    try:
        calibration = calibration_suite()
        write(worker / "calibration.json", calibration)
        if not calibration["passed"]:
            raise ValueError("J1 array/controller calibration failed")
        runner = NativeRunner(run, config, meter)
        reference = LegacySACActor.from_checkpoint(run / "prepared/reference.pt", meter=meter)
        with np.load(run / "prepared/normalization.npz", allow_pickle=False) as stream:
            normalization = {key: stream[key].copy() for key in stream.files}
        write(worker / "score_definitions.json", score_definitions())
        models: dict[str, Any] = {}

        def retain(outcome: dict) -> dict:
            traces.append(outcome)
            append_metadata(worker / "trace_catalog.jsonl", outcome, meter)
            return outcome

        def collect(split: str) -> list[dict]:
            pending = []
            for group in (g for g in config["groups"] if g["split"] == split):
                key = group["id"]
                teacher = retain(
                    runner.teacher(key + "_mppi", group["acquisition_seed"], group["planner_seed"])
                )
                teachers[key + "_mppi"] = teacher
                source = runner.load_trace(teacher)
                context = trace_context(source)
                ref = retain(
                    runner.trace(
                        key + "_reference",
                        lambda obs, t, current: reference.predict(obs, meter=meter),
                        seed=group["acquisition_seed"],
                    )
                )
                teachers[key + "_reference"] = ref
                reference_source = runner.load_trace(ref)
                # Same integer seed is insufficient: exact full initial context is required.
                for field in ("states", "warmstarts", "integrations"):
                    if not np.array_equal(source[field][0], reference_source[field][0]):
                        raise ValueError("J1 paired source contexts differ")
                for source_name, outcome in (("mppi", teacher), ("reference", ref)):
                    original = runner.load_trace(outcome)
                    replay = retain(
                        runner.trace(
                            key + "_" + source_name + "_replay",
                            lambda obs, t, current, tape=original["actions"]: tape[t],
                            initialsnapshot=trace_context(original),
                            replay_source=original,
                            horizon=1000,
                        )
                    )
                    outcome["replay"] = replay
                    if replay["controls"] != outcome["controls"]:
                        raise ValueError("J1 source replay length differs")
                for learner in group["learners"]:
                    parent = parent_prepare(
                        learner["policy_seed"],
                        learner["fit_seed"],
                        normalization,
                        worker / "parents" / learner["id"],
                        meter,
                    )
                    parents[learner["id"]] = parent
                    incoming = retain(
                        runner.trace(
                            learner["id"] + "_incoming",
                            lambda obs, t, current, actor=parent["actor"]: actor.predict(
                                obs, meter=meter
                            ),
                            initialsnapshot=context,
                        )
                    )
                    for source_name, outcome in (("mppi", teacher), ("reference", ref)):
                        case_id = learner["id"] + "_" + source_name
                        case = dict(
                            case_id=case_id,
                            group=key,
                            learner=learner["id"],
                            split=split,
                            source=source_name,
                            teacher=outcome,
                            incoming=incoming,
                            eligible=outcome["controls"] == 1000,
                            evaluation_seeds=learner["evaluation_seeds"],
                            incoming_bindings=parent["bindings"],
                        )
                        if not case["eligible"]:
                            case["ineligible_reason"] = (
                                "Predeclared source has fewer than1000executedrows; no replacement/padding"
                            )
                            cases.append(case)
                            continue
                        source_trace = metric_trace(runner.load_trace(outcome))
                        prediction = parent["actor"].predict(
                            source_trace["observations"], meter=meter
                        )
                        case["pre_prediction_batch_shape"] = [1000, 17]
                        case["pre_prediction_ref"] = meter.save_arrays(
                            case_id + "_before", predictions=prediction
                        )
                        pending.append(case)
                write(
                    worker / "source_groups" / (key + ".json"),
                    {k: teachers[k] for k in (key + "_mppi", key + "_reference")},
                )
                print(
                    json.dumps(dict(event="source_group_completed", split=split, group=key)),
                    flush=True,
                )
            return pending

        def score_and_seal(pending: list[dict], split: str) -> None:
            for case in pending:
                teacher = metric_trace(runner.load_trace(case["teacher"]))
                incoming = metric_trace(runner.load_trace(case["incoming"]))
                prediction = read_array(
                    meter.out, case["pre_prediction_ref"], "predictions", max_bytes=1024**2
                )
                score = score_case(
                    teacher,
                    incoming,
                    prediction,
                    models["explainer"],
                    models["occupancy_normalizer"],
                )
                case.update(score)
                if "prediction_models" in models:
                    case["predictions"] = predict_case_scores(
                        case["scores"], models["prediction_models"]
                    )
            destination = worker / "score_seals" / (split + ".json")
            destination.parent.mkdir(exist_ok=True)
            write(
                destination,
                dict(
                    schema="walker-j1-preupdate-seal-v1",
                    split=split,
                    cases=pending,
                    models_sha256=sha(
                        worker
                        / (
                            "score_models_prelearning.json"
                            if split == "development"
                            else "score_models.json"
                        )
                    ),
                    no_learning_in_split_yet=True,
                ),
            )
            event = meter.record(
                "j1_scores_sealed", split=split, sha256=sha(destination), cases=len(pending)
            )
            for case in pending:
                case["score_seal_file"] = str(destination.relative_to(run))
                case["score_seal_event"] = event

        def learn(pending: list[dict]) -> None:
            for case in pending:
                parent = parents[case["learner"]]
                teacher = metric_trace(runner.load_trace(case["teacher"]))
                torch.set_rng_state(parent["global_torch"].clone())
                event = meter.record(
                    "j1_fit_begin", case=case["case_id"], seal=case["score_seal_event"]
                )
                fitted = fit_case(
                    parent,
                    teacher["observations"],
                    teacher["actions"],
                    worker / "fits" / case["case_id"],
                    meter,
                )
                child = fitted["actor"]
                reproduction = retain(
                    runner.trace(
                        case["case_id"] + "_reproduction",
                        lambda obs, t, current: child.predict(obs, meter=meter),
                        initialsnapshot=trace_context(teacher),
                    )
                )
                motion = motion_metrics(teacher, metric_trace(runner.load_trace(reproduction)))
                evaluations = []
                for ordinal, seed in enumerate(case["evaluation_seeds"]):
                    evaluations.append(
                        retain(
                            runner.trace(
                                case["case_id"] + f"_eval_{ordinal:02d}",
                                lambda obs, t, current: child.predict(obs, meter=meter),
                                seed=seed,
                            )
                        )
                    )
                case.update(
                    reproduction=reproduction,
                    evaluations=evaluations,
                    post_motion=motion["error"],
                    motion_components=motion,
                    reproduced=motion["reproduced"],
                    walking_successes=sum(int(v["walking"]) for v in evaluations),
                    walking_trials=16,
                    parent_checkpoint=parent["bindings"]["checkpoint"],
                    child_checkpoint=fitted["checkpoint"],
                    tape_file=parent["bindings"]["tape"]["file"],
                    fit_receipt_file=fitted["receipt_file"],
                    fit_begin_event=event,
                )
                cases.append(case)
                append_metadata(worker / "case_outcomes.jsonl", case, meter)
                print(
                    json.dumps(
                        dict(
                            event="learning_case_completed",
                            case=case["case_id"],
                            split=case["split"],
                        )
                    ),
                    flush=True,
                )

        development = collect("development")
        if not any(case["source"] == "mppi" for case in development):
            raise ValueError(
                "No eligible development MPPI corpus; primary-source prerequisite failed"
            )
        dev_teacher_traces = [
            metric_trace(runner.load_trace(t))
            for k, t in teachers.items()
            if k.startswith("development") and t["controls"] == 1000
        ]
        if not dev_teacher_traces:
            raise ValueError("No complete development source corpus")
        models["explainer"] = fit_state_explainer(
            np.concatenate([t["observations"] for t in dev_teacher_traces]),
            np.concatenate([t["actions"] for t in dev_teacher_traces]),
            normalization,
            group_ids=[g["id"] for g in config["groups"] if g["split"] == "development"],
        )
        occupancy = [np.column_stack((t["observations"], t["actions"])) for t in dev_teacher_traces]
        for parent in sorted({c["learner"] for c in development}):
            incoming = next(c["incoming"] for c in development if c["learner"] == parent)
            trace = metric_trace(runner.load_trace(incoming))
            occupancy.append(np.column_stack((trace["observations"], trace["actions"])))
        models["occupancy_normalizer"] = fit_occupancy_normalizer(
            np.concatenate(occupancy),
            group_ids=[g["id"] for g in config["groups"] if g["split"] == "development"],
        )

        # Probe the new recorded controller, with every pass archived and no learner involved.
        contexts = []
        for anchor in config["probe"]["contexts"]:
            outcome = teachers[anchor["group"] + "_mppi"]
            if outcome["controls"] <= anchor["control"]:
                raise ValueError("A declared probe anchor is unavailable; no replacement")
            contexts.append(runner.planner_context(outcome, anchor["control"]))
        probe_results = []
        for request in decision_roster():
            planner = IndependentMPPI(runner.env, MPPIConfig(**config["mppi"]), 0)
            selector = CoupledSelector(**config["selector"])
            probe = run_decision(
                runner.env,
                planner,
                selector,
                contexts[request["context"]],
                request,
                meter=meter,
                restore_context=runner.restore,
                contact_reader=runner.contacts,
            )
            persisted = runner.persist_probe(probe)
            probe_results.append(persisted)
        write(worker / "probe_results.json", probe_results)
        probe_summary = summarize_probe(probe_results, normalization_std=normalization["std"])
        write(worker / "probe_summary.json", probe_summary)
        write(worker / "score_models_prelearning.json", models)
        score_and_seal(development, "development")
        learn(development)
        models["prediction_models"] = fit_prediction_models(cases, source="mppi")
        write(worker / "score_models.json", models)
        write(
            worker / "prediction_model_seal.json",
            dict(
                sha256=sha(worker / "score_models.json"),
                development_only=True,
                protected_learning_launched=False,
            ),
        )
        meter.record("j1_prediction_models_sealed", sha256=sha(worker / "score_models.json"))
        if (
            not models["prediction_models"]["development_roster_complete"]
            or models["prediction_models"]["motion_sd"] < config["prediction"]["minimum_motion_std"]
        ):
            result.update(
                status="CLOSED_DEVELOPMENT_VARIATION_INELIGIBLE",
                objective_passed=False,
                protected_test_revealed=False,
            )
        else:
            protected = collect("protected")
            score_and_seal(protected, "protected")
            learn(protected)
            comparison = evaluate_predictions(cases, models["prediction_models"], source="mppi")
            write(worker / "predictive_comparison.json", comparison)
            result.update(
                status="COMPLETED_J1_FIXED_VERSION",
                objective_passed=comparison["objective_passed"],
                protected_test_revealed=True,
            )
        result.update(
            cases=len(cases),
            eligible_cases=sum(c["eligible"] for c in cases),
            declared_source_groups=16,
            independent_groups=len({c["group"] for c in cases}),
            acquired_source_groups=len({k.rsplit("_", 1)[0] for k in teachers}),
            teacher_outcomes={
                k: {x: v[x] for x in ("controls", "walking", "survived", "cost")}
                for k, v in teachers.items()
            },
            scores_file="worker/cases.json",
        )
    except BaseException as error:
        result.update(
            status="STOPPED_PREREQUISITE_OR_INFRASTRUCTURE_FAILURE",
            objective_passed=False,
            error=dict(type=type(error).__name__, message=str(error)),
        )
    finally:
        try:
            if runner is not None:
                runner.close()
        except BaseException as error:
            result.update(
                status="STOPPED_EXECUTION_CLOSURE_FAILURE",
                error=dict(type=type(error).__name__, message=str(error)),
            )
        try:
            meter.close()
        except BaseException as error:
            result.update(
                status="STOPPED_EVIDENCE_CLOSURE_FAILURE",
                error=dict(type=type(error).__name__, message=str(error)),
            )
        result.update(
            costs=meter.accounting(),
            config_sha256=sha(run / "config.json"),
            source_manifest_sha256=sha(run / "manifest.json"),
            input_manifest_sha256=sha(run / "input_manifest.json"),
            worker_pid=os.getpid(),
            coordinator_pid=os.getppid(),
            no_guidance_experiment=True,
        )
        write(worker / "cases.json", cases)
        write(worker / "traces.json", traces)
        write(worker / "teachers.json", teachers)
        write(run / "worker_result.json", result)
    return result
