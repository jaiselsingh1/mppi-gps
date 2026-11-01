"""Independent saved-only audit of segmented S1.1 evaluation.

The checker reads closed Meter segments, checkpoint tensors and independent
mirror receipts. It imports neither production evaluation nor durability code
and performs no policy, optimizer, planner or simulator arithmetic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import native, source, width
from .common import assert_independent_imports


class EvaluationAudit(source.SourceAudit):
    """Resolve native episode primitives only for explicitly owned segments."""

    def __init__(self, run: Path):
        super().__init__(run, run / "source")
        self.segment_roots: set[Path] = set()

    def group(self, path, ref, keys=None):
        path = Path(path)
        if path.parent.name == "evidence" and path.parent.parent.resolve() in self.segment_roots:
            path = path.parent.parent / "raw.h5"
        return super().group(path, ref, keys)


def counter_sum(before: dict, work: dict) -> dict:
    """Meter rotation must not renew any scientific reservation."""
    keys = set(before) | set(work)
    return {key: before.get(key, 0) + work.get(key, 0) for key in keys}


def verify_files(a: EvaluationAudit, segment: Path, bindings: dict) -> None:
    a.need(set(bindings) == {"raw.h5", "ledger.jsonl"}, "Closed segment has both Meter files")
    a.need(
        {path.name for path in segment.iterdir()} == {"raw.h5", "ledger.jsonl", "receipt.json"},
        "Closed segment is exactly three immutable files",
    )
    for relative, expected in bindings.items():
        path = segment / relative
        a.need(path.is_file() and not path.is_symlink(), "Regular closed segment input")
        actual = a.bind(path, expected["sha256"])
        a.need(actual["bytes"] == expected["bytes"], "Exact closed segment size")


def verify_cumulative(a: EvaluationAudit, receipt: dict, before: dict) -> dict:
    a.prove(
        receipt["ownership"]["cumulative_before"], before, "Exact cumulative budget before episode"
    )
    expected = counter_sum(before, receipt["work"])
    a.prove(receipt["cumulative"], expected, "Closed segment advances one cumulative budget")
    a.need(set(receipt["cumulative"]) == set(expected), "Complete cumulative counter roster")
    a.need(
        all(
            type(value) is int and value >= 0
            for key, value in expected.items()
            if key != "wall_seconds"
        ),
        "Nonnegative integer cumulative counters",
    )
    a.need(
        np.isfinite(expected.get("wall_seconds", 0)) and expected.get("wall_seconds", 0) >= 0,
        "Nonnegative cumulative measured wall",
    )
    return expected


def verify_closed_segment(
    a: EvaluationAudit, segment: Path, expected_sha: str, before: dict
) -> tuple[dict, dict]:
    a.bind(segment / "receipt.json", expected_sha)
    receipt = a.load(segment / "receipt.json")
    a.need(
        set(receipt) == {"schema", "ownership", "episode", "work", "cumulative", "files"},
        "Explicit closed-episode receipt schema",
    )
    a.need(receipt["schema"] == "walker-durable-episode-v1", "Durable episode version")
    verify_files(a, segment, receipt["files"])
    stage = source.stage_ledger(a, segment, receipt["work"])
    a.need(
        len(stage["events"]["runtime_open"]) == len(stage["events"]["runtime_close"]) == 1,
        "Exactly one opened and closed Meter per segment",
    )
    close = stage["events"]["runtime_close"][0]
    a.need(set(receipt["work"]) == set(close["totals"]), "Complete closed Meter work roster")
    a.need(
        receipt["work"]["wall_seconds"] >= close["wall_seconds"]
        and receipt["work"]["wall_seconds"] >= close["totals"]["wall_seconds"],
        "Closed segment measured wall includes terminal boundary",
    )
    a.need(len(stage["episodes"]) == 1, "Exactly one complete episode per segment")
    a.prove(
        next(iter(stage["episodes"].values())),
        receipt["episode"],
        "Receipt is the actual native episode record",
    )
    a.segment_roots.add(segment.resolve())
    cumulative = verify_cumulative(a, receipt, before)
    for key in (
        "requested_model_controls",
        "requested_backward_rows",
        "requested_optimizer_steps",
        "requested_checkpoint_loads",
        "failed_queries",
        "failed_measurements",
        "rejected_queries",
        "rejected_measurements",
        "reset_errors",
        "unknown_partial_controls",
        "unknown_partial_neural_rows",
        "unknown_partial_checkpoint_loads",
        "restores",
    ):
        a.need(
            stage["counts"][key] == 0, "No forbidden, failed or unquantified episode work " + key
        )
    a.need(
        stage["counts"]["completed_neural_rows"] == stage["counts"]["completed_live_controls"]
        and stage["counts"]["episodes"] == stage["counts"]["resets"] == 1,
        "One inference per control and one reset per episode",
    )
    return receipt, stage | dict(cumulative=cumulative)


def prove_episode_owner(a: EvaluationAudit, ownership: dict, expected: dict) -> None:
    a.prove(ownership, expected, "Closed segment protocol/checkpoint/reset/owner identity")
    a.need(
        ownership["boot_id_after"] == ownership["boot_id"], "No host boot change within an episode"
    )


def prove_mirror_files(
    a: EvaluationAudit, worker_segment: Path, mirror_segment: Path, receipt_sha: str, receipt: dict
) -> None:
    a.need(
        worker_segment.resolve() != mirror_segment.resolve(),
        "Independent mirror is a distinct artifact path",
    )
    a.need(
        {path.name for path in mirror_segment.iterdir()}
        == {"raw.h5", "ledger.jsonl", "receipt.json"},
        "Mirror contains all and only closed segment files",
    )
    for relative, binding in receipt["files"].items():
        path = mirror_segment / relative
        a.need(path.is_file() and not path.is_symlink(), "Regular independent mirror file")
        actual = a.bind(path, binding["sha256"])
        a.need(actual["bytes"] == binding["bytes"], "Independent mirror exact size")
    a.bind(mirror_segment / "receipt.json", receipt_sha)


def verify_exchange(
    a: EvaluationAudit,
    run: Path,
    mirror: Path,
    descriptor: dict,
    receipt: dict,
    identifier: str,
    ordinal: int,
) -> dict:
    """Join independent file ownership without comparing clocks across hosts."""
    relative = descriptor["segment"]
    worker_segment = width.contained(a, run, relative)
    mirror_segment = width.contained(a, mirror, relative)
    prove_mirror_files(a, worker_segment, mirror_segment, descriptor["receipt_sha256"], receipt)
    documents = {}
    for kind in ("request", "ack"):
        binding = descriptor[kind]
        expected_path = (
            ("publications" if kind == "request" else "acks") + "/" + identifier + ".json"
        )
        a.need(binding["file"] == expected_path, "Canonical exchange path " + kind)
        path = width.contained(a, run, binding["file"])
        a.bind(path, binding["sha256"])
        documents[kind] = a.load(path)
    request, ack = documents["request"], documents["ack"]
    files = receipt["files"] | {"receipt.json": a.bind(worker_segment / "receipt.json")}
    a.prove(
        request,
        dict(
            schema="walker-s1.1-publication-v1",
            identifier=identifier,
            ordinal=ordinal,
            segment=relative,
            worker_boot_id=receipt["ownership"]["boot_id"],
            receipt_sha256=descriptor["receipt_sha256"],
            files=files,
        ),
        "Closed segment publication ownership",
    )
    a.need(set(request["files"]) == set(files), "All three publication files")
    a.prove(
        ack,
        dict(
            schema="walker-s1.1-ack-v1",
            identifier=identifier,
            request_sha256=descriptor["request"]["sha256"],
            receipt_sha256=descriptor["receipt_sha256"],
            worker_boot_id=receipt["ownership"]["boot_id"],
            mirror_commit_sha256=descriptor["mirror_commit_sha256"],
        ),
        "Independent acknowledgement ownership",
    )
    commit_path = mirror / "local_coordinator/commits" / (identifier + ".json")
    a.bind(commit_path, descriptor["mirror_commit_sha256"])
    commit = a.load(commit_path)
    a.prove(
        commit,
        dict(
            identifier=identifier,
            request_sha256=descriptor["request"]["sha256"],
            receipt_sha256=descriptor["receipt_sha256"],
            files=files,
        ),
        "Off-host mirror commit joins exact publication",
    )
    a.prove(
        commit,
        dict(
            schema="walker-s1.1-mirror-commit-v1",
            ordinal=ordinal,
            segment=relative,
            worker_boot_id=receipt["ownership"]["boot_id"],
            coordinator_instance=ack["coordinator_instance"],
            coordinator_host=ack["coordinator_host"],
            copied_bytes=sum(row["bytes"] for row in files.values()),
        ),
        "Distinct coordinator commit provenance and exact copied bytes",
    )
    a.need(
        bool(commit["coordinator_instance"])
        and bool(commit["coordinator_host"])
        and np.isfinite(commit["copy_wall_seconds"])
        and commit["copy_wall_seconds"] >= 0,
        "Finite independent copy cost and coordinator identity",
    )
    historical_directory = Path(commit["local_directory"])
    a.need(
        historical_directory.is_absolute()
        and historical_directory.parts[-len(Path(relative).parts) :] == Path(relative).parts,
        "Historical coordinator path owns this packet; explicit mirror relocation preserves its bytes",
    )
    a.bind(mirror / "local_coordinator/acks" / (identifier + ".json"), descriptor["ack"]["sha256"])
    a.need(0 <= descriptor["wait_seconds"] <= 60, "Finite mirror acknowledgement wait")
    return documents | dict(commit=commit)


def verify_coordinator(
    a: EvaluationAudit, run: Path, mirror: Path, descriptors: list[dict], launch: dict
) -> dict:
    coordinator = a.load(mirror / "coordinator_exit.json")
    a.need(
        coordinator["schema"] == "walker-s1.1-coordinator-exit-v1"
        and coordinator["execution_complete"] is True
        and coordinator["error"] is None
        and coordinator["worker_boot_id"] == launch["boot_id"],
        "Independent coordinator completed without restart or boot change",
    )
    a.need(
        coordinator["new_physics"]
        == coordinator["new_neural_rows"]
        == coordinator["new_optimizer_steps"]
        == 0
        and 0 <= coordinator["wall_seconds"] <= 3630,
        "Finite engineering-only coordinator work",
    )
    a.need(
        len(coordinator["records"]) == len(descriptors)
        and all(row["exit_code"] == 0 and row["wall_seconds"] >= 0 for row in coordinator["calls"]),
        "All publication transport calls closed successfully",
    )
    copied, copy_wall, instances = 0, 0.0, set()
    for descriptor, row in zip(descriptors, coordinator["records"], strict=True):
        identifier = Path(descriptor["segment"]).name
        receipt = a.load(run / descriptor["segment"] / "receipt.json")
        a.prove(
            row,
            dict(
                identifier=identifier,
                request_sha256=descriptor["request"]["sha256"],
                receipt_sha256=descriptor["receipt_sha256"],
                mirror_commit_sha256=descriptor["mirror_commit_sha256"],
                ack_sha256=descriptor["ack"]["sha256"],
                segment=descriptor["segment"],
                work=receipt["work"],
            ),
            "Complete ordered off-host coordinator closure",
        )
        commit = a.load(mirror / "local_coordinator/commits" / (identifier + ".json"))
        a.need(
            commit["coordinator_host"] != launch["host"],
            "Off-host coordinator identity differs from worker host",
        )
        instances.add(commit["coordinator_instance"])
        copied += commit["copied_bytes"]
        copy_wall += commit["copy_wall_seconds"]
    a.need(len(instances) == 1, "One coordinator instance without automatic retry")
    for name in (
        "launch_receipt.json",
        "worker_started.json",
        "worker_process.json",
        "result.json",
        "worker_exit.json",
        "owner_exit.json",
        "owner_ledger.jsonl",
    ):
        a.bind(mirror / "global" / name, native.sha(run / name))
    return dict(
        copied_closed_packet_bytes=copied,
        copy_wall_seconds=copy_wall,
        coordinator_wall_seconds=coordinator["wall_seconds"],
        storage_bytes=sum(path.stat().st_size for path in mirror.rglob("*") if path.is_file()),
    )


def verify_owner_journal(
    a: EvaluationAudit, run: Path, descriptors: list[dict], final_work: dict
) -> None:
    """One start/close/publication/ACK quartet precedes every next start."""
    path = run / "owner_ledger.jsonl"
    a.bind(path)
    rows = [json.loads(line) for line in path.open()]
    a.need(len(rows) == 4 * len(descriptors) + 1, "No missing, extra or failed owner transitions")
    previous_time, previous_ack = None, None
    for index, row in enumerate(rows):
        a.need(
            row["id"] == index and np.isfinite(row["monotonic"]), "Owner sequence and finite clock"
        )
        a.need(
            previous_time is None or row["monotonic"] >= previous_time, "Owner monotonic chronology"
        )
        previous_time = row["monotonic"]
    for index, descriptor in enumerate(descriptors):
        start, closed, requested, accepted = rows[4 * index : 4 * index + 4]
        identifier = Path(descriptor["segment"]).name
        ordinal = index - 1
        for row, event in zip(
            (start, closed, requested, accepted),
            ("segment_started", "segment_closed", "publication_requested", "ack_accepted"),
            strict=True,
        ):
            a.prove(
                row,
                dict(event=event, ordinal=ordinal, identifier=identifier),
                "Exact owner state machine",
            )
        a.need(
            start["previous_ack_sha256"] == previous_ack,
            "Next segment waits for previous off-host ACK",
        )
        a.need(
            closed["receipt_sha256"] == descriptor["receipt_sha256"], "Closed receipt owner journal"
        )
        receipt = a.load(run / descriptor["segment"] / "receipt.json")
        a.prove(closed["cumulative"], receipt["cumulative"], "Journal cumulative paid work")
        a.need(
            requested["request_sha256"] == descriptor["request"]["sha256"],
            "Publication journal SHA",
        )
        a.prove(accepted, descriptor, "Accepted ACK matches published result descriptor")
        previous_ack = descriptor["ack"]["sha256"]
    a.prove(
        rows[-1],
        dict(event="worker_finished", execution_complete=True, work=final_work),
        "Closed successful one-shot worker journal",
    )


def audit_inherited(a: EvaluationAudit, run: Path, cfg: dict, result: dict) -> tuple[dict, dict]:
    """Verify original source/corpora/closed fits; never charge them as new work."""
    root = run / "inputs/s1"
    pins = cfg["source_bindings"]
    for name, key in (
        ("config.json", "s1_config_sha256"),
        ("manifest.json", "s1_source_manifest_sha256"),
        ("input_manifest.json", "s1_input_manifest_sha256"),
        ("fit_result.json", "s1_fit_result_sha256"),
    ):
        a.bind(root / name, pins[key])
    old_cfg, manifest, seal, fit = [
        a.load(root / name)
        for name in ("config.json", "manifest.json", "input_manifest.json", "fit_result.json")
    ]
    a.need(
        manifest["config_sha256"] == seal["config_sha256"] == pins["s1_config_sha256"]
        and seal["source_manifest_sha256"] == pins["s1_source_manifest_sha256"],
        "Original S1 source/config/input ownership",
    )
    for row in manifest["files"]:
        a.bind(width.contained(a, root / "source", row["file"]), row["sha256"])
    for relative, binding in seal["files"].items():
        bound = a.bind(width.contained(a, root, relative), binding["sha256"])
        a.need(bound["bytes"] == binding["bytes"], "Exact original input size")
    history = run / "inputs/s1_interruption"
    a.bind(history / "original_run_manifest.json", pins["s1_original_run_manifest_sha256"])
    catalog = {row["file"]: row for row in a.load(history / "original_run_manifest.json")["files"]}
    for path in root.rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(root))
            a.need(relative in catalog, "Inherited file existed before fresh evaluation")
            record = a.bind(path, catalog[relative]["sha256"])
            a.need(record["bytes"] == catalog[relative]["bytes"], "Original inherited file bytes")
    a.need(
        fit["execution_complete"]
        and fit["error"] is None
        and fit["config_sha256"] == pins["s1_config_sha256"]
        and fit["input_manifest_sha256"] == pins["s1_input_manifest_sha256"],
        "Only the closed original six-fit phase is reused",
    )
    fit_exit = a.load(root / "fit_exit.json")
    a.need(
        fit_exit["exit_code"] == 0
        and fit_exit["timeout"] is False
        and fit_exit["fit_result_sha256"] == pins["s1_fit_result_sha256"],
        "Original fit external exit",
    )
    data, norm, inventory = source.audit_inputs(a, root, old_cfg)
    fit_stage = source.stage_ledger(a, root / "evidence/fit", fit["work"])
    details = source.audit_fit(a, root, old_cfg, fit, data, norm)
    a.need(
        fit_stage["counts"]["completed_neural_rows"] == 3180000
        and fit_stage["counts"]["completed_backward_rows"] == 3072000
        and fit_stage["counts"]["completed_optimizer_steps"] == 12000
        and fit_stage["counts"]["requested_controls"] == 0,
        "Closed original fit cost retained separately",
    )
    for name, binding in cfg["checkpoints"].items():
        point = fit["points"][name]
        a.prove(
            point["checkpoint"],
            {key: value for key, value in binding.items() if key != "optimizer_fingerprint"},
            "Exact unchanged inherited checkpoint",
        )
        a.need(
            point["optimizer_fingerprint"] == binding["optimizer_fingerprint"],
            "Exact unchanged inherited Adam",
        )
    points, fits = deepcopy(fit["points"]), deepcopy(fit["fits"])
    for point in points.values():
        point["checkpoint"]["file"] = "inputs/s1/" + point["checkpoint"]["file"]
    for row in fits:
        row["file"] = "inputs/s1/" + row["file"]
        row["initial"]["file"] = "inputs/s1/" + row["initial"]["file"]
    for key, value in dict(
        points=points, fits=fits, samplers=fit["samplers"], teacher_outcomes=fit["teacher_outcomes"]
    ).items():
        a.prove(result[key], value, "Inherited evidence metadata without new learning " + key)
    summary = a.load(history / "summary.json")
    receipt, full = [a.load(history / name) for name in ("receipt.json", "full_audit_receipt.json")]
    a.need(
        receipt["passed"] is False
        and full["passed"] is False
        and summary["execution_complete"] is False
        and summary["comparison_complete"] is False
        and receipt["summary_sha256"] == native.sha(history / "summary.json")
        and summary["config_sha256"] == pins["s1_config_sha256"]
        and summary["input_manifest_sha256"] == pins["s1_input_manifest_sha256"]
        and summary["fit_result_sha256"] == pins["s1_fit_result_sha256"]
        and summary["work"]["unknown_unrecorded_tail"]["status"] == "unresolved_not_zero",
        "Historical interrupted audits remain failed",
    )
    previous = {
        key: value - fit["work"].get(key, 0)
        for key, value in summary["work"]["known_durable_prefix"].items()
        if isinstance(value, (int, float)) and key != "wall_seconds_lower_bound"
    }
    previous.update(
        wall_seconds=None,
        snapshots=None,
        wall_seconds_lower_bound=summary["durable_prefix"]["last_record"]["wall_seconds"],
    )
    a.prove(
        result["inherited_work"],
        dict(
            original_fit=fit["work"],
            original_eval_known_prefix=previous,
            original_eval_unknown_tail=summary["work"]["unknown_unrecorded_tail"],
        ),
        "Original paid fit/prefix and unresolved tail remain separate",
    )
    a.prove(
        result["inherited_fit"],
        dict(
            file="inputs/s1/fit_result.json",
            sha256=pins["s1_fit_result_sha256"],
            source_bindings=pins,
        ),
        "Exact original closed fit binding",
    )
    return details["payloads"], inventory


def ordered_panel(a: EvaluationAudit, records: dict, order: list[str]) -> list[tuple[str, dict]]:
    """JSON object order is immaterial; segments and the journal own execution order."""
    a.need(
        len(records) == len(order) and set(records) == set(order),
        "Every actor runs both panels regardless of prior outcomes",
    )
    return [(name, records[name]) for name in order]


def audit_segments(
    a: EvaluationAudit,
    run: Path,
    mirror: Path,
    cfg: dict,
    result: dict,
    launch: dict,
    worker: dict,
    payloads: dict,
    inventory: dict,
) -> dict:
    order = ["reference"] + [
        name
        for pair in range(3)
        for name in (f"incoming_s{pair}", f"mppi_s{pair}", f"sac_reference_s{pair}")
    ]
    a.need(
        cfg["evaluation_order"] == order and len(result["segments"]) == 320,
        "Unconditional fixed ten-actor/two-panel roster",
    )
    owners = {}
    for name in order:
        payload = None if name == "reference" else payloads[name]
        owners[name] = dict(
            policy_hash=inventory["reference"]["policy_hash"]
            if payload is None
            else payload["policy_hash"],
            optimizer_fingerprint=native.fingerprint(
                None if payload is None else payload["optimizer_state"]
            ),
            private_rng_fingerprint=native.fingerprint(
                None if payload is None else payload["private_rng_state"]
            ),
        )
    a.prove(
        result["actors_initial"], owners, "Actual incoming CPU actor/Adam/private RNG identities"
    )
    a.need(
        result["actors_initial"] == result["actors_final"]
        and set(result["actors_initial"]) == set(order),
        "No actor/Adam/private RNG mutation during evaluation",
    )
    common = dict(
        boot_id=launch["boot_id"],
        boot_id_after=launch["boot_id"],
        owner_pid=worker["pid"],
        launch_receipt_sha256=native.sha(run / "launch_receipt.json"),
        worker_started_sha256=native.sha(run / "worker_started.json"),
        config_sha256=native.sha(run / "config.json"),
        input_manifest_sha256=native.sha(run / "input_manifest.json"),
        source_manifest_sha256=native.sha(run / "manifest.json"),
    )
    setup = result["setup"]
    a.need(setup["segment"] == "evidence/loads", "Canonical load segment")
    path = run / setup["segment"]
    a.bind(path / "receipt.json", setup["receipt_sha256"])
    receipt = a.load(path / "receipt.json")
    a.need(
        set(receipt) == {"schema", "ownership", "episode", "work", "cumulative", "files"}
        and receipt["schema"] == "walker-durable-loads-v1",
        "Closed setup receipt version",
    )
    verify_files(a, path, receipt["files"])
    stage = source.stage_ledger(a, path, receipt["work"])
    cumulative = verify_cumulative(a, receipt, {})
    prove_episode_owner(
        a,
        receipt["ownership"],
        common
        | dict(
            ordinal=-1,
            point="loads",
            panel=None,
            seed=None,
            tag="loads",
            checkpoint_sha256=None,
            policy_hash=None,
            optimizer_fingerprint=native.fingerprint(None),
            initial_context_fingerprint=None,
            cumulative_before={},
        ),
    )
    a.need(
        not stage["episodes"] and not stage["resets"] and not stage["live"],
        "No evaluation in checkpoint setup",
    )
    loads = [
        row
        for row in stage["events"]["measurement_requested"]
        if row["measure_kind"] == "checkpoint_loads"
    ]
    a.need(
        len(loads) == 10
        and [row["context"]["point"] for row in loads] == order
        and all(
            row["amount"] == 1 and row["context"]["role"] == "source_checkpoint_load"
            for row in loads
        ),
        "Exactly ten serialization loads in declared order",
    )
    for key in (
        "requested_neural_rows",
        "requested_backward_rows",
        "requested_optimizer_steps",
        "requested_controls",
        "failed_measurements",
        "rejected_measurements",
        "unknown_partial_checkpoint_loads",
    ):
        a.need(stage["counts"][key] == 0, "Setup contains only successful CPU loads " + key)
    verify_exchange(a, run, mirror, setup, receipt, "loads", -1)
    starts, seen, ordinal, controls, waits = {}, set(), 0, 0, setup["wait_seconds"]
    all_episodes = {}
    for panel in ("primary", "confirmation"):
        records = result["evaluations"][panel]
        for name, record in ordered_panel(a, records, order):
            a.prove(record, owners[name], "Native evaluation unchanged actor owner")
            a.need(
                [row["seed"] for row in record["episodes"]] == cfg[panel + "_seeds"],
                "Exact fresh reset roster",
            )
            cohort = {}
            for index, row in enumerate(record["episodes"]):
                tag = f"{panel}_{name}_{index:02d}"
                identifier = f"{ordinal:03d}_{tag}"
                descriptor = result["segments"][ordinal]
                a.prove(
                    descriptor,
                    dict(
                        ordinal=ordinal,
                        point=name,
                        panel=panel,
                        tag=tag,
                        segment="evidence/episodes/" + identifier,
                    ),
                    "One ordinal/actor/reset per segment",
                )
                a.need(
                    row["tag"] == tag
                    and tag not in seen
                    and row["segment"] == descriptor["segment"],
                    "Unique canonical native result owner",
                )
                seen.add(tag)
                before = cumulative
                receipt, episode_stage = verify_closed_segment(
                    a, run / descriptor["segment"], descriptor["receipt_sha256"], before
                )
                event, arrays = a.episode(episode_stage, tag)
                a.prove(
                    event,
                    {key: value for key, value in row.items() if key != "segment"},
                    "Recomputed native result matches complete evaluation row",
                )
                context = native.fingerprint(
                    dict(
                        state=arrays["states"][0],
                        warm=arrays["warmstarts"][0],
                        integration=arrays["integrations"][0],
                    )
                )
                a.need(
                    starts.setdefault(row["seed"], context) == context,
                    "All ten actors share exact physical/warm/integration reset",
                )
                binding = cfg["reference"] if name == "reference" else cfg["checkpoints"][name]
                own = common | dict(
                    ordinal=ordinal,
                    point=name,
                    panel=panel,
                    seed=row["seed"],
                    tag=tag,
                    checkpoint_sha256=binding["sha256"],
                    policy_hash=binding["policy_hash"],
                    optimizer_fingerprint=binding.get(
                        "optimizer_fingerprint", native.fingerprint(None)
                    ),
                    initial_context_fingerprint=context,
                    cumulative_before=before,
                )
                prove_episode_owner(a, receipt["ownership"], own)
                for info in arrays["infos"]["rows"]:
                    a.prove(
                        info["controller"],
                        dict(
                            policy_hash=owners[name]["policy_hash"], controller="standalone_torque"
                        ),
                        "Every saved action uses the fixed direct torque owner",
                    )
                for query, _ in episode_stage["live"][tag]:
                    a.prove(
                        query["context"],
                        dict(point=name, cohort=panel, role="source_standalone"),
                        "Every native query actor/panel context",
                    )
                predictions = episode_stage["events"]["measurement_requested"]
                a.need(
                    len(predictions) == row["steps"]
                    and all(
                        p["measure_kind"] == "neural_rows"
                        and p["amount"] == 1
                        and p["context"]["point"] == name
                        for p in predictions
                    ),
                    "Only one single-row standalone forward per native control",
                )
                verify_exchange(a, run, mirror, descriptor, receipt, identifier, ordinal)
                cumulative = episode_stage["cumulative"]
                controls += row["steps"]
                waits += descriptor["wait_seconds"]
                cohort[tag] = event
                all_episodes[tag] = event
                ordinal += 1
            a.need(
                record["summary"]["tags"] == list(cohort) and len(cohort) == 16,
                "Complete sixteen-reset cohort",
            )
            a.summary(dict(episodes=cohort), record["summary"])
        a.prove(
            result["gates"][panel],
            source.recompute_walking_screen(records, cfg),
            "Independent source walking/survival gate " + panel,
        )
    a.need(
        len(seen) == ordinal == 320 and len(starts) == 32,
        "Every predeclared fresh episode retained",
    )
    competence = {
        name: all(
            result["evaluations"][panel][name]["summary"]["successes"] >= 15
            for panel in ("primary", "confirmation")
        )
        for name in order
    }
    a.need(
        result["gates"]["confirmed_walking_benefit"]
        == all(result["gates"][panel]["walking_screen"] for panel in ("primary", "confirmation")),
        "Both panels independently confirm walking benefit",
    )
    a.prove(
        result["gates"]["competence"], competence, "Walking competence is separate from retention"
    )
    a.prove(result["work"], cumulative, "Disjoint closed setup and episodes counted exactly once")
    a.prove(
        result["phase_work"],
        dict(eval=cumulative),
        "No inherited fitting charged as new evaluation",
    )
    a.equal(result["mirror_wait_seconds"], waits, "Complete worker mirror waiting time")
    verify_owner_journal(a, run, [setup] + result["segments"], cumulative)
    a.need(
        cumulative["completed_live_controls"] == cumulative["completed_neural_rows"] == controls
        and cumulative["completed_physics_substeps"] == 4 * controls
        and cumulative["completed_checkpoint_loads"] == 10
        and cumulative["episodes"] == 320,
        "Independent complete new-work census",
    )
    for key in (
        "requested_backward_rows",
        "requested_optimizer_steps",
        "requested_model_controls",
        "failed_queries",
        "failed_measurements",
        "reset_errors",
        "restores",
    ):
        a.need(cumulative.get(key, 0) == 0, "No new learning/planning/replay/failure " + key)
    for counter, budget in dict(
        requested_live_controls="live_controls",
        requested_neural_rows="neural_rows",
        requested_physics_substeps="physics_substeps",
        requested_checkpoint_loads="checkpoint_loads",
        episodes="episodes",
    ).items():
        a.need(
            cumulative[counter] <= cfg["budgets"][budget],
            "Global reservations survive rotation " + budget,
        )
    a.need(
        cumulative["wall_seconds"] + waits <= result["worker_wall_seconds"] + 0.1
        and result["worker_wall_seconds"] <= cfg["budgets"]["wall_seconds"],
        "Global deadline includes off-host waits and all closed Meter time",
    )
    return dict(
        episodes=320,
        native_controls=controls,
        competence=competence,
        new_work=cumulative,
        mirror_wait_seconds=waits,
    )


def audit_run(
    a: EvaluationAudit, run: Path, mirror: Path, *, config_sha: str, seal_sha: str
) -> dict:
    a.bind(run / "config.json", config_sha)
    a.bind(run / "input_manifest.json", seal_sha)
    cfg, seal, manifest = [
        a.load(run / name) for name in ("config.json", "input_manifest.json", "manifest.json")
    ]
    a.need(
        cfg["schema"] == "walker-student-source-s1.1-v1"
        and manifest["config_sha256"] == seal["config_sha256"] == config_sha
        and seal["source_manifest_sha256"] == native.sha(run / "manifest.json"),
        "Independent frozen S1.1 configuration/input/source pins",
    )
    for row in manifest["files"]:
        a.bind(width.contained(a, run / "source", row["file"]), row["sha256"])
    for name, binding in seal["files"].items():
        actual = a.bind(width.contained(a, run, name), binding["sha256"])
        a.need(actual["bytes"] == binding["bytes"], "Complete portable input seal")
    proposal = a.load(run / "source/configs/studies/student_source_s1_1_proposal.json")
    for key, value in proposal.items():
        if key not in ("schema", "status", "stopping"):
            a.prove(cfg[key], value, "Preserved evaluation-only proposal science " + key)
    a.need(
        cfg["primary_seeds"] == list(range(1270000000, 1270000016))
        and cfg["confirmation_seeds"] == list(range(1280000000, 1280000016)),
        "Predeclared fresh panels with no outcome-based replacements",
    )
    for name in ("local_seed_audit.json", "remote_seed_audit.json"):
        proof = a.load(run / "inputs" / name)
        a.need(
            proof["passed"] is True
            and proof["seeds"] == cfg["primary_seeds"] + cfg["confirmation_seeds"],
            "Prelaunch fresh-reset audit " + name,
        )
        ledger_proof = proof["known_reset_ledgers"]
        a.need(
            ledger_proof["passed"] is True
            and ledger_proof["seeds"] == cfg["primary_seeds"] + cfg["confirmation_seeds"]
            and not ledger_proof["collisions"]
            and not ledger_proof["errors"],
            "Prelaunch known ledger seed census " + name,
        )
        a.need(
            all(
                type(row["bytes"]) is int
                and row["bytes"] >= 0
                and type(row["records"]) is int
                and row["records"] >= 0
                and len(row["sha256"]) == 64
                for row in ledger_proof["files"]
            ),
            "Explicit prelaunch known ledger identities " + name,
        )
    task = a.load(run / "inputs/s1/inputs/task.json")
    a.need(
        task == cfg["task"]
        and native.sha(run / "source/assets/walker2d.xml") == cfg["task"]["xml_sha256"]
        and native.sha(run / "inputs/s1/inputs/task.json") == cfg["source_bindings"]["task_sha256"],
        "Unchanged native task, XML and feature contract",
    )
    result = a.load(run / "result.json")
    a.need(
        result["schema"] == "walker-student-source-s1.1-result-v1"
        and result["config_sha256"] == config_sha
        and result["input_manifest_sha256"] == seal_sha,
        "Result ownership",
    )
    a.need(
        result["execution_complete"] is True
        and result["error"] is None
        and result["status"] == "evaluation_complete",
        "Complete one-shot comparison; incomplete execution cannot pass this audit",
    )
    launch, worker = [a.load(run / name) for name in ("launch_receipt.json", "worker_started.json")]
    a.prove(
        launch,
        dict(
            config_sha256=config_sha,
            input_manifest_sha256=seal_sha,
            source_manifest_sha256=native.sha(run / "manifest.json"),
        ),
        "External immutable launch seal",
    )
    a.prove(
        worker,
        dict(
            schema="walker-s1.1-worker-start-v1",
            config_sha256=config_sha,
            input_manifest_sha256=seal_sha,
            launch_receipt_sha256=native.sha(run / "launch_receipt.json"),
            boot_id=launch["boot_id"],
        ),
        "Actual native worker identity",
    )
    a.need(
        result["boot_id"] == launch["boot_id"] and type(worker["pid"]) is int and worker["pid"] > 0,
        "Single boot/PID worker identity",
    )
    a.prove(
        result["platform"],
        {key: cfg["platform"][key] for key in ("torch", "mujoco", "numpy", "scipy", "h5py")},
        "Pinned actual native CPU packages",
    )
    a.need(
        result["platform"]["python"] == cfg["expected_python"]
        and result["platform"]["platform"].startswith("Linux"),
        "Pinned Linux CPU interpreter",
    )
    a.prove(worker["platform"], result["platform"], "Worker platform receipt")
    payloads, inventory = audit_inherited(a, run, cfg, result)
    details = audit_segments(a, run, mirror, cfg, result, launch, worker, payloads, inventory)
    worker_exit, owner = [a.load(run / name) for name in ("worker_exit.json", "owner_exit.json")]
    a.need(
        worker_exit["exit_code"] == 0
        and worker_exit["timeout"] is False
        and worker_exit["error"] is None
        and worker_exit["result_sha256"] == native.sha(run / "result.json"),
        "External worker process closed successfully",
    )
    for stream in ("worker.stdout", "worker.stderr"):
        stream_binding = worker_exit["streams"][stream]
        bound = a.bind(run / stream, stream_binding["sha256"])
        a.need(bound["bytes"] == stream_binding["bytes"], "Closed external output stream bytes")
    a.need(
        owner["execution_complete"] is True and owner["error"] is None,
        "External one-shot owner completed",
    )
    a.prove(
        owner,
        dict(
            schema="walker-s1.1-owner-exit-v1",
            boot_id=launch["boot_id"],
            boot_id_after=launch["boot_id"],
            worker_exit_sha256=native.sha(run / "worker_exit.json"),
            owner=launch["owner"],
        ),
        "External owner remains on the same boot and binds exact worker exit",
    )
    a.prove(
        worker_exit,
        dict(
            schema="walker-s1.1-worker-exit-v1",
            boot_id=launch["boot_id"],
            boot_id_after=launch["boot_id"],
        ),
        "External worker exit boot identity",
    )
    process = a.load(run / "worker_process.json")
    lock = a.load(run / "launch.lock")
    a.need(
        process["pid"] == worker["pid"]
        and process["boot_id"] == launch["boot_id"]
        and isinstance(process["start_ticks"], str)
        and bool(process["start_ticks"]),
        "External worker process identity pins start ticks and boot",
    )
    a.prove(
        lock, launch["owner"] | dict(boot_id=launch["boot_id"]), "Create-only owner lock identity"
    )
    a.need(
        launch["retries_permitted"] is False
        and launch["host"] == a.load(run / "inputs/remote_seed_audit.json")["host"],
        "One authorized host and no retries",
    )
    a.equal(
        launch["deadline_monotonic"] - launch["started_monotonic"],
        cfg["budgets"]["wall_seconds"],
        "Single external cumulative deadline",
    )
    a.need(
        result["worker_wall_seconds"]
        <= worker_exit["wall_seconds"]
        <= owner["wall_seconds"]
        <= cfg["budgets"]["wall_seconds"],
        "External owner elapsed wall includes all worker and acknowledgement work",
    )
    engineering = verify_coordinator(a, run, mirror, [result["setup"]] + result["segments"], launch)
    size = sum(path.stat().st_size for path in run.rglob("*") if path.is_file())
    a.need(size <= cfg["budgets"]["storage_bytes"], "Closed worker evidence storage ceiling")
    return details | dict(
        comparison_complete=True,
        reused_fits=6,
        new_fits=0,
        new_reconstruction_rows=0,
        gates=result["gates"],
        inherited_work=result["inherited_work"],
        closed_run_bytes=size,
        off_host_engineering=engineering,
        limitations=[
            "Fresh evaluation reuses the same three paired S1 training realizations; no new training replications.",
            "Whole-source effects do not identify a source property, establish MPPI teachability or test adaptive coupling.",
            "Original S1 remains interrupted with a failed full-prefix audit and unresolved paid tail; its outcomes are not pooled with these panels.",
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-sha", required=True)
    parser.add_argument("--seal-sha", required=True)
    args = parser.parse_args(argv)
    run, mirror, out = args.run.resolve(), args.mirror.resolve(), args.out.resolve()
    if out.is_relative_to(run) or out.is_relative_to(mirror):
        parser.error("Audit output must be outside both immutable worker and mirror evidence")
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    a = EvaluationAudit(run)
    error, findings = None, {}
    try:
        for module in (
            Path(__file__),
            Path(native.__file__),
            Path(source.__file__),
            Path(width.__file__),
        ):
            a.bind(module)
        findings = audit_run(a, run, mirror, config_sha=args.config_sha, seal_sha=args.seal_sha)
        for path, before in list(a.files.items()):
            a.need(a.bind(Path(path)) == before, "Audited input unchanged after read")
        for path, (_, size, mtime) in a.hdfs.items():
            a.need(
                (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime),
                "Audited HDF unchanged",
            )
        assert_independent_imports(a)
        forbidden = (
            "src.gps.mppi_gps.durability",
            "src.gps.mppi_gps.evaluation_exchange",
            "src.gps.mppi_gps.source_evaluation_inputs",
            "src.gps.mppi_gps.source_evaluation_launch",
            "src.gps.mppi_gps.source_evaluation_coordinator",
            "src.gps.mppi_gps.diagnostic_policies",
            "src.gps.mppi_gps.reporting.source",
            "src.gps.mppi_gps.reporting.source_evaluation",
            "src.gps.mppi_gps.source_outcomes",
        )
        a.need(
            not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules
                for prefix in forbidden
            ),
            "No production evaluation, durability, metrics or reporting imports",
        )
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        for handle, _, _ in a.hdfs.values():
            handle.close()
    receipt = dict(
        schema="walker-student-source-s1.1-independent-audit-v1",
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
        scope="Independent frozen source/input/closed-fit provenance, unchanged actor continuation state, closed per-episode native arrays and outcomes, reset pairing, cumulative work, same-boot owner state machine and distinct off-host mirror commits. Saved-only CPU decoding; no physics, policy, optimizer or planner arithmetic.",
    )
    for name, value in (
        ("receipt.json", receipt),
        ("findings.json", findings),
        ("file_bindings.json", a.files),
        ("array_bindings.json", a.arrays),
    ):
        with (out / name).open("x") as stream:
            json.dump(native.plain(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(receipt))
    return int(error is not None)


if __name__ == "__main__":
    raise SystemExit(main())
