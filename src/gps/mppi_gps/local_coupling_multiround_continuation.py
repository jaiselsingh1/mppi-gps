"""L3.1 engineering continuation of the unchanged eight-round L3 experiment.

Completed science is imported by byte binding, never repeated. Two interrupted
episodes resume at their saved native boundary; one fully executed episode is
adopted only after a separate, physics-free numerical completion certificate.
The original failure, ledgers and source export remain immutable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import local_coupling_multiround as original
from .local_coupling_loop import bound, read
from .provenance import write
from .trajectory_pi_accounting import directory_bytes
from .trajectory_pi_native import binding

MODULE = "src.gps.mppi_gps.local_coupling_multiround_continuation"
ARMS = original.ARMS
RESUME_SCHEMA = "l3-1-certified-resume-inputs-1"


def add_counts(*parts: dict) -> dict:
    result = {}
    for part in parts:
        for key, value in part.items():
            if type(value) is not int or value < 0:
                raise ValueError("Nonnegative integer work counts required")
            result[key] = result.get(key, 0) + value
    return result


def pending_overhead(record: dict) -> dict:
    """Keep the uncertain forecast separate from the retained, known prefix."""
    counts = {}
    pending = record["ledger_counts"]["open_requests"]
    for request in pending:
        for unit, value in request["charges"].items():
            for prefix in ("requested_", "unknown_partial_"):
                key = prefix + unit
                counts[key] = counts.get(key, 0) + value
    if {
        k.removeprefix("unknown_partial_"): v
        for k, v in counts.items()
        if k.startswith("unknown_partial_")
    } != record["ledger_counts"]["unknown_partial_upper"]:
        raise ValueError("Interrupted forecast uncertainty does not match its requests")
    return dict(
        ledger=record["ledger"],
        job=record["job"],
        excluded_pending_requests=pending,
        counts=counts,
    )


def prefix_reservation(record: dict) -> dict:
    return {
        prefix + unit: value
        for unit, value in record["ledger_counts"]["completed"].items()
        for prefix in ("requested_", "completed_")
    }


def _verify_source(manifest: dict, root: Path) -> dict:
    source = read(bound(manifest))
    for row in source["files"]:
        actual = binding(root / row["path"])
        if any(actual[key] != row[key] for key in ("bytes", "sha256")):
            raise ValueError("Frozen source changed: " + row["path"])
    return source


def verify_launch(path: Path) -> dict:
    """Bind both exports, the original science recipe and native runtime."""
    launch = read(path)
    if (
        launch.get("schema") != "local-coupling-l3-launch-1"
        or launch.get("engineering", {}).get("schema") != "local-coupling-continuation-l3-1"
    ):
        raise ValueError("Frozen L3.1 continuation launch required")
    engineering = launch["engineering"]
    root = Path(engineering["source_directory"])
    if (
        Path(__file__).resolve()
        != root / "src/gps/mppi_gps/local_coupling_multiround_continuation.py"
    ):
        raise ValueError("Execute the immutable engineering export")
    parent = read(bound(engineering["parent_launch"]))
    for key in (
        "config",
        "prepared_manifest",
        "source_manifest",
        "environment",
        "source_directory",
    ):
        if launch[key] != parent[key]:
            raise ValueError("The original scientific launch binding changed: " + key)
    old_source = _verify_source(launch["source_manifest"], Path(launch["source_directory"]))
    new_source = _verify_source(engineering["source_manifest"], root)
    new_files = {row["path"]: row for row in new_source["files"]}
    for row in old_source["files"]:
        name = row["path"]
        if name.startswith("src/") and name.endswith(".py") and "/audits/" not in name:
            if new_files.get(name) != row:
                raise ValueError("Scientific implementation changed: " + name)
    bound(engineering["amendment"])
    inventory = read(bound(engineering["inventory"]))
    if Path(inventory["original_directory"]) != Path(parent["base_directory"]):
        raise ValueError("Recovery inventory refers to another experiment")
    config = read(bound(launch["config"]))
    if config["schema"] != "local-coupling-learning-l3-1" or not config["launch_ready"]:
        raise ValueError("The unchanged launch-ready L3 science recipe is required")
    prepared = read(bound(launch["prepared_manifest"]))
    if prepared["config_sha256"] != launch["config"]["sha256"]:
        raise ValueError("Original input seal/config mismatch")
    for spec in prepared["files"]:
        bound(spec)
    environment = read(bound(launch["environment"]))
    for spec in (
        environment["interpreter"],
        *environment["binary_files"],
        environment["model_xml"],
    ):
        bound(spec)
    if binding(Path(sys.executable)) != environment["interpreter"]:
        raise ValueError("Wrong bound interpreter")
    _antecedents(launch)
    return launch


def _antecedents(launch: dict) -> dict:
    value = launch["engineering"]["budget_antecedents"]
    if isinstance(value, dict) and {"path", "bytes", "sha256"} <= value.keys():
        value = read(bound(value))
    needed = (
        "stream_cpu_seconds_upper",
        "worker_cpu_seconds",
        "global_cpu_seconds",
        "worker_wall_seconds",
        "global_wall_seconds",
    )
    if any(key not in value for key in needed):
        raise ValueError("Explicit original resource antecedents are required")
    if (
        len(value["stream_cpu_seconds_upper"]) != 3
        or any(not np.isfinite(x) or x < 0 for x in value["stream_cpu_seconds_upper"])
        or any(not np.isfinite(value[key]) or value[key] < 0 for key in needed[1:])
    ):
        raise ValueError("Invalid frozen resource antecedents")
    return value


def _ledger_imports(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if any(row.get("event") != "closed_child_included" for row in rows):
        raise ValueError("Only the original closed-child import ledger can be inherited")
    return rows


def freeze_resume(launch: dict, certificate: dict) -> dict:
    """Create the full adoption/import plan before any continuation physics."""
    inventory = read(bound(launch["engineering"]["inventory"]))
    streams = []
    for entry in inventory["streams"]:
        snapshot = read(bound(entry["snapshot"]))
        if [block["round"] for block in snapshot["rounds"] if block["status"] == "COMPLETED"] != [
            1,
            2,
            3,
        ]:
            raise ValueError("Exactly three completed original round blocks required")
        ledger = Path(entry["snapshot"]["path"]).parent.parent / "study_ledger.jsonl"
        events = _ledger_imports(ledger)
        receipts = snapshot["accounting"]["child_receipts"]
        if [row["receipt"] for row in events] != receipts:
            raise ValueError("Original import ledger/snapshot receipt order differs")
        for event in events:
            receipt = read(bound(event["receipt"]))
            if receipt["counts"] != event["counts"]:
                raise ValueError("Original closed-child work binding differs")
        counts = add_counts(*(event["counts"] for event in events))
        if counts != entry["counts"] or counts != snapshot["accounting"]["counts"]:
            raise ValueError("Original closed work census differs")
        for row in entry["incoming_for_round4"]:
            bound(row["envelope"])
            bound(row["checkpoint"])
            if row["cumulative_updates"] != 10000:
                raise ValueError("Continue the exact 10,000-update round-three actors")
        streams.append(
            dict(
                stream=entry["stream"],
                snapshot=entry["snapshot"],
                ledger=binding(ledger),
                incoming_for_round4=entry["incoming_for_round4"],
                retained_after_latest_saved_admissions=entry[
                    "retained_after_latest_saved_admissions"
                ],
                closed_round4=entry["closed_round4"],
                interrupted_round4=entry["interrupted_round4"],
                infrastructure_overhead=[
                    pending_overhead(row) for row in entry["interrupted_round4"]
                ],
            )
        )
    result = dict(
        schema=RESUME_SCHEMA,
        status="FROZEN_BEFORE_CONTINUATION_PHYSICS",
        original_launch=launch["engineering"]["parent_launch"],
        config=launch["config"],
        inventory=launch["engineering"]["inventory"],
        amendment=launch["engineering"]["amendment"],
        engineering_source=launch["engineering"]["source_manifest"],
        numerical_completion=certificate["result"],
        numerical_completion_receipt=binding(Path(certificate["path"]) / "receipt.json"),
        budget_antecedents=_antecedents(launch),
        streams=streams,
        parent_streams=inventory["streams"],
        interrupted=[row for entry in inventory["streams"] for row in entry["interrupted_round4"]],
        adoptions=[
            dict(
                stream=0,
                round=4,
                arm="independent",
                slot=2,
                result=binding(Path(certificate["path"]) / "result.json"),
                adopted_manifest=certificate["result"]["adopted_manifest"],
                verification_receipt=binding(Path(certificate["path"]) / "receipt.json"),
            )
        ],
    )
    write(Path(launch["base_directory"]) / "resume_manifest.json", result)
    return result


class ResumeStudy(original.StreamStudy):
    """Reuse completed blocks and continue the same retained learning streams."""

    def __init__(self, launch: dict, stream: int, resume_manifest: dict):
        if (
            resume_manifest.get("schema") != RESUME_SCHEMA
            or resume_manifest["config"] != launch["config"]
        ):
            raise ValueError("Frozen L3.1 resume plan with original config required")
        super().__init__(launch, stream)
        entry = next(row for row in resume_manifest["streams"] if row["stream"] == stream)
        old = read(bound(entry["snapshot"]))
        self.original_root = Path(entry["snapshot"]["path"]).parent.parent
        self.resume_manifest = resume_manifest
        self.resume_entry = entry
        self.resource_antecedents = resume_manifest["budget_antecedents"]
        self.previous_cpu = self.resource_antecedents["stream_cpu_seconds_upper"][stream]
        self.previous_wall = self.resource_antecedents["worker_wall_seconds"]
        self.process_cpu_started = time.process_time()
        self.counts = dict(old["accounting"]["counts"])
        self.receipts = list(old["accounting"]["child_receipts"])
        self.extra_verification_receipts = []
        self.infrastructure_overhead = entry["infrastructure_overhead"]
        self.infrastructure_counts = add_counts(
            *(row["counts"] for row in self.infrastructure_overhead)
        )
        self.pending_prefixes = {
            (row["arm"], row["slot"]): row for row in entry["interrupted_round4"]
        }
        self.result = copy.deepcopy(old)
        self.result.pop("error", None)
        self.result.update(
            status="RUNNING",
            engineering=dict(
                comparison_version="MATCHED_GROUP_L3_1",
                original_snapshot=entry["snapshot"],
                resume_manifest=binding(Path(launch["base_directory"]) / "resume_manifest.json"),
                original_import_ledger=entry["ledger"],
            ),
        )
        self.result["engineering_continuation"] = dict(
            version="L3.1",
            parent_launch=launch["engineering"]["parent_launch"],
            parent_worker_result=launch["engineering"]["parent_worker_result"],
            amendment=launch["engineering"]["amendment"],
            inventory=launch["engineering"]["inventory"],
            engineering_source_manifest=launch["engineering"]["source_manifest"],
        )
        self.ledger.write(bound(entry["ledger"]).read_text())
        self.ledger.flush()
        os.fsync(self.ledger.fileno())
        self.incoming = {row["arm"]: row["envelope"] for row in entry["incoming_for_round4"]}
        completed = [row for row in old["rounds"] if row["status"] == "COMPLETED"]
        if len(completed) != 3 or self.incoming != {
            row["arm"]: row["envelope"] for row in completed[-1]["outgoing_streams"]
        }:
            raise ValueError("Round-four incoming actors differ from retained round-three outputs")
        for key, row in entry["retained_after_latest_saved_admissions"].items():
            arm, slot_text = key.split(":")
            slot = int(slot_text)
            self.retained_binding[(arm, slot)] = row["chosen"]
            self.retained[(arm, slot)] = read(bound(row["chosen"]))
            self.accepted_at[(arm, slot)] = row["last_accepted_round"]
        if stream == 0:
            spec = resume_manifest["numerical_completion_receipt"]
            self._import_receipt(spec, read(bound(spec)))
            self.extra_verification_receipts.append(spec)
            self.previous_cpu += read(bound(spec))["cpu_seconds"]
        self.check()

    def reserved_counts(self) -> dict:
        return add_counts(
            self.counts,
            self.infrastructure_counts,
            *(prefix_reservation(row) for row in self.pending_prefixes.values()),
        )

    def check(self):
        worker = self.config["budgets"]["worker"]
        if self.previous_wall + time.monotonic() - self.started > worker["wall_seconds"]:
            raise RuntimeError("Original plus continuation stream wall budget exhausted")
        if (
            self.previous_cpu + time.process_time() - self.process_cpu_started
            > worker["stream_cpu_seconds"]
        ):
            raise RuntimeError("Original plus continuation stream CPU budget exhausted")
        size = directory_bytes(self.original_root) + directory_bytes(self.root)
        if size + 1024**2 > worker["output_bytes"] // self.config["streams"]:
            raise RuntimeError("Original plus continuation stream output reservation exhausted")
        if any(key.startswith("unknown_partial_") and value for key, value in self.counts.items()):
            raise RuntimeError("New or imported logical work has unresolved execution")
        remaining = original.child_limits(self.config, self.reserved_counts())
        if any(value < 0 for value in remaining.values()):
            raise RuntimeError("Original plus continuation work reservation exhausted")

    def _import_receipt(self, spec: dict, receipt: dict):
        if any(row["path"] == spec["path"] for row in self.receipts):
            raise ValueError("Duplicate work import")
        if read(bound(spec)) != receipt:
            raise ValueError("Closed work receipt binding differs")
        self.receipts.append(spec)
        self.counts = add_counts(self.counts, receipt["counts"])
        self.ledger.write(
            json.dumps(
                dict(event="closed_child_included", receipt=spec, counts=receipt["counts"]),
                sort_keys=True,
            )
            + "\n"
        )
        self.ledger.flush()
        os.fsync(self.ledger.fileno())

    def job(self, function, job):
        self.check()
        limits = original.child_limits(self.config, self.reserved_counts())
        limits["wall_seconds"] = min(
            limits["wall_seconds"],
            self.config["budgets"]["worker"]["wall_seconds"]
            - self.previous_wall
            - (time.monotonic() - self.started),
        )
        job.update(config=self.launch["config"], limits=limits)
        value = function(job)
        prefix_key = (job["arm"], job["slot"]) if "prefix_job" in job else None
        if prefix_key is not None:
            self.pending_prefixes.pop(prefix_key)
        self._import_receipt(binding(Path(value["path"]) / "receipt.json"), value["receipt"])
        if value["result"]["status"] != "COMPLETED":
            raise RuntimeError("Invalid execution: " + str(value["result"]))
        self.check()
        return value["result"]

    def snapshot(self):
        self.result["accounting"] = dict(
            counts=add_counts(self.counts, self.infrastructure_counts),
            logical_counts=dict(self.counts),
            infrastructure_counts=self.infrastructure_counts,
            infrastructure_overhead=self.infrastructure_overhead,
            child_receipts=self.receipts,
            extra_verification_receipts=self.extra_verification_receipts,
            pending_known_prefix_reservations=[
                dict(arm=key[0], slot=key[1], counts=prefix_reservation(row))
                for key, row in self.pending_prefixes.items()
            ],
            worker_process_cpu_seconds=self.previous_cpu
            + time.process_time()
            - self.process_cpu_started,
            worker_wall_seconds=self.previous_wall + time.monotonic() - self.started,
            continuation_process_cpu_seconds=time.process_time() - self.process_cpu_started,
            continuation_wall_seconds=time.monotonic() - self.started,
            original_process_cpu_seconds_upper=self.previous_cpu,
        )
        path = self.root / "progress" / f"{self.index:06d}.json"
        write(path, self.result)
        self.index += 1
        temporary = self.root / "result.next"
        temporary.symlink_to(path.relative_to(self.root))
        os.replace(temporary, self.root / "result.json")

    def _prefix_job(self, record: dict, out: Path) -> dict:
        source_job = read(bound(record["job"]))
        old_dir = Path(record["directory"])
        reference = next(
            row["manifest"]
            for row in self.result["rounds"][0]["acquisitions"]
            if row["arm"] == record["arm"] and row["slot"] == record["slot"]
        )
        reference_trace = read(bound(reference))["trace"]
        full_banks = sorted(old_dir.glob("full_t*.npz"))
        if not full_banks:
            raise ValueError("Complete recorded prefix-bank evidence required")
        return dict(
            source_job,
            out=str(out),
            prefix_job=record["job"],
            prefix_trace=record["trace"],
            prefix_banks=record["banks"],
            prefix_ledger=record["ledger"],
            resume_context=record["resume_context"],
            reference_R1_trace=reference_trace,
            prefix_full_banks=[binding(path) for path in full_banks],
            resume_rng_state=record["resume_rng_state"],
        )

    def acquire(self, round_index, block):
        from .local_coupling_controller import run_episode
        from .local_coupling_multiround_measurements import admit_or_retain

        selected = {arm: [] for arm in ARMS}
        acquired = {(row["arm"], row["slot"]): row for row in block["acquisitions"]}
        admitted = {(row["arm"], row["slot"]): row for row in block["admissions"]}
        if set(acquired) != set(admitted):
            raise ValueError("Saved round has incomplete acquisition/admission joins")
        for arm in ARMS:
            checkpoint = binding(Path(self.incoming[arm]["path"]).parent / "policy.pt")
            alpha = next(row["alpha"] for row in self.config["arms"] if row["name"] == arm)
            for slot in range(8):
                key = arm, slot
                if key in acquired:
                    selected[arm].append(dict(slot=slot, trace=self.retained[key]["trace"]))
                    continue
                out = self.root / "acquisition" / f"r{round_index}" / arm / f"slot{slot}"
                if round_index == 4 and self.stream == 0 and key == ("independent", 2):
                    spec = self.resume_manifest["numerical_completion"]["adopted_manifest"]
                    episode = read(bound(spec))
                elif round_index == 4 and key in self.pending_prefixes:
                    from .local_coupling_prefix_resume import resume_episode

                    job = self._prefix_job(self.pending_prefixes[key], out)
                    episode = self.job(resume_episode, job)
                    spec = binding(out / "manifest.json")
                else:
                    episode = self.job(
                        run_episode,
                        dict(
                            out=str(out),
                            round=round_index,
                            stream=self.stream,
                            arm=arm,
                            slot=slot,
                            alpha=alpha,
                            seed=original.planner_seed(self.config, self.stream, slot),
                            checkpoint=checkpoint,
                            restarts=self.prepared["restarts"],
                            controls=300,
                            planner=self.config["controller"]["planner"],
                            task=self.config["controller"]["task"],
                            physics_threads=4,
                            replay_episode=slot == 0,
                        ),
                    )
                    spec = binding(out / "manifest.json")
                if episode.get("status") != "COMPLETED" or episode["checkpoint"] != checkpoint:
                    raise ValueError("Invalid or differently bound acquisition")
                previous, old = self.retained_binding[key], self.retained[key]
                decision = admit_or_retain(episode, old, self.anchors[slot])
                chosen, fresh = decision["selected"], decision["fresh_acquisition_admitted"]
                before = original.experience_fingerprint(old)
                after = original.experience_fingerprint(chosen)
                if fresh:
                    self.retained[key], self.retained_binding[key] = episode, spec
                    self.accepted_at[key] = round_index
                row = dict(
                    arm=arm,
                    slot=slot,
                    admitted=fresh,
                    fallback=not fresh,
                    previous=previous,
                    attempt=spec,
                    chosen=self.retained_binding[key],
                    last_accepted_round=self.accepted_at[key],
                    age=round_index - self.accepted_at[key],
                    previous_experience_fingerprint=before,
                    chosen_experience_fingerprint=after,
                    new_experience=bool(fresh and before != after),
                    new_rows=300 if fresh and before != after else 0,
                    checks=decision["decision"]["checks"],
                )
                block["acquisitions"].append(dict(arm=arm, slot=slot, manifest=spec))
                block["admissions"].append(row)
                selected[arm].append(dict(slot=slot, trace=chosen["trace"]))
                self.snapshot()
                print(
                    json.dumps(
                        dict(
                            stage="acquisition",
                            round=round_index,
                            stream=self.stream,
                            arm=arm,
                            slot=slot,
                            local_walking=episode["outcome"]["local_walking"],
                            controls=episode["outcome"]["controls"],
                            complete_cost=episode["outcome"]["complete_cost"],
                            admitted=fresh,
                            fallback=not fresh,
                            new_rows=row["new_rows"],
                        )
                    ),
                    flush=True,
                )
        return selected

    def _learn_round(self, r: int, block: dict, selected: dict):
        from .local_coupling_learning import fit_job
        from .local_coupling_multiround_measurements import build_retained_corpora

        corpus = build_retained_corpora(
            selected["independent"],
            selected["positive"],
            self.root / "corpus" / f"r{r}",
            config_binding=self.launch["config"],
        )
        block["corpus"] = binding(self.root / "corpus" / f"r{r}" / "manifest.json")
        outgoing = {}
        for arm, own, cross in (("independent", "a", "b"), ("positive", "b", "a")):
            out = self.root / "fits" / f"r{r}_{arm}"
            fit = self.job(
                fit_job,
                dict(
                    out=str(out),
                    incoming=self.incoming[arm],
                    corpus=corpus["files"][own],
                    round=r,
                    stream=self.stream,
                    arm=arm,
                    diagnostics=dict(
                        own=corpus["files"][own],
                        cross=corpus["files"][cross],
                        held=self.prepared["held"],
                        anchor=self.prepared["anchor_corpus"],
                    ),
                ),
            )
            block["fits"].append(dict(arm=arm, result=binding(out / "result.json")))
            outgoing[arm] = fit["outgoing"]
            block["outgoing_streams"].append(dict(arm=arm, envelope=fit["outgoing"]))
            checkpoint = binding(Path(fit["outgoing"]["path"]).parent / "policy.pt")
            first = self.evaluate(checkpoint, arm, "child", "primary", r, block["evaluations"])
            self.probe(first, checkpoint, arm, r, block["probes"])
        tapes = [
            np.load(bound(read(bound(row["result"]))["index_tape"]), allow_pickle=False)
            for row in block["fits"]
        ]
        try:
            if len(tapes) != 2 or not np.array_equal(tapes[0]["indices"], tapes[1]["indices"]):
                raise ValueError("Paired minibatch streams diverged")
        finally:
            for tape in tapes:
                tape.close()
        self.incoming = outgoing
        block["status"] = "COMPLETED"
        self.snapshot()
        print(
            json.dumps(
                dict(
                    stage="round_closed",
                    stream=self.stream,
                    round=r,
                    new_experience=sum(row["new_experience"] for row in block["admissions"]),
                    fallback=sum(row["fallback"] for row in block["admissions"]),
                    continuation="UNCONDITIONAL_NEXT_PLANNED_ROUND",
                )
            ),
            flush=True,
        )

    def run(self):
        try:
            self.snapshot()
            for r in range(4, self.config["rounds"] + 1):
                existing = [row for row in self.result["rounds"] if row["round"] == r]
                if existing:
                    block = existing[0]
                    if r != 4 or block["status"] != "RUNNING" or block["fits"]:
                        raise ValueError(
                            "Only the saved partial fourth acquisition block may resume"
                        )
                else:
                    block = dict(
                        round=r,
                        status="RUNNING",
                        acquisitions=[],
                        admissions=[],
                        fits=[],
                        evaluations=[],
                        probes=[],
                        outgoing_streams=[],
                    )
                    self.result["rounds"].append(block)
                self._learn_round(r, block, self.acquire(r, block))
            if self.pending_prefixes:
                raise ValueError("Interrupted native prefix has not been completed")
            for arm in ARMS:
                for kind, envelope, r in (
                    ("initial", self.initial[arm], 0),
                    ("final", self.incoming[arm], self.config["rounds"]),
                ):
                    self.evaluate(
                        binding(Path(envelope["path"]).parent / "policy.pt"),
                        arm,
                        kind,
                        "confirmation",
                        r,
                        self.result["final_evaluations"],
                    )
            self.result["status"] = "COMPLETED_ALL_ROUNDS"
        except BaseException as error:
            self.result.update(status="FAILED_STOPPED", error=repr(error))
        finally:
            self.snapshot()
            self.ledger.close()
        return self.result


def _certify(launch: dict) -> dict:
    from .local_coupling_numeric_validation import certify_existing

    inventory = read(bound(launch["engineering"]["inventory"]))
    stream = next(row for row in inventory["streams"] if row["stream"] == 0)
    candidates = [row for row in stream["closed_round4"] if row["status"] == "FAILED_STOPPED"]
    if len(candidates) != 1 or candidates[0]["slot"] != 2:
        raise ValueError("The exact original fully executed checker failure is required")
    failed = candidates[0]
    original_job = read(bound(failed["job"]))
    reserved = add_counts(
        stream["counts"],
        *(prefix_reservation(row) for row in stream["interrupted_round4"]),
        *(pending_overhead(row)["counts"] for row in stream["interrupted_round4"]),
    )
    limits = original.child_limits(read(bound(launch["config"])), reserved)
    limits["neural_rows"] = min(limits["neural_rows"], 24672)
    limits["wall_seconds"] = min(limits["wall_seconds"], 600)
    limits["output_bytes"] = 64 * 1024**2
    for unit in (
        "model_controls",
        "native_controls",
        "physics_substeps",
        "backward_rows",
        "optimizer_steps",
        "checkpoint_writes",
        "optimizer_constructions",
        "environment_constructions",
        "model_queries",
        "restores",
        "external_mj_forward_calls",
        "resets",
        "compiled_model_saves",
        "pi_passes",
    ):
        limits[unit] = 0
    limits["checkpoint_loads"] = limits["policy_constructions"] = 1
    result = certify_existing(
        dict(
            out=str(
                Path(launch["base_directory"]) / "verification/adoption_s0_r4_independent_slot2"
            ),
            original_manifest=failed["manifest"],
            checkpoint=original_job["checkpoint"],
            config=launch["config"],
            limits=limits,
        )
    )
    if result["result"]["status"] != "COMPLETED":
        raise RuntimeError(
            "Numerical completion failed; no continuation physics: " + str(result["result"])
        )
    return result


def coordinator(launch: dict) -> dict:
    """Certify, freeze the import plan, then launch only the three owned workers."""
    base = Path(launch["base_directory"])
    root = base / "worker"
    root.mkdir(exist_ok=False)
    write(root / "coordinator_identity.json", dict(pid=os.getpid()))
    config, antecedents = read(bound(launch["config"])), _antecedents(launch)
    processes, streams = [], []
    started, cpu_started = time.monotonic(), time.process_time()
    failure, sequence, reaped = None, 0, 0.0
    resume = None

    def snapshot():
        nonlocal sequence
        closed = [read(bound(row["result"])) for row in streams]
        result = dict(
            schema="local-coupling-multiround-l3-result-1",
            status="RUNNING",
            planned_rounds=config["rounds"],
            config=launch["config"],
            prepared_manifest=launch["prepared_manifest"],
            source_manifest=launch["source_manifest"],
            engineering=launch["engineering"],
            streams=streams,
            engineering_continuation=dict(
                version="L3.1",
                parent_launch=launch["engineering"]["parent_launch"],
                parent_worker_result=launch["engineering"]["parent_worker_result"],
                amendment=launch["engineering"]["amendment"],
                inventory=launch["engineering"]["inventory"],
                engineering_source_manifest=launch["engineering"]["source_manifest"],
            ),
            accounting=dict(
                counts=add_counts(*(row["accounting"]["counts"] for row in closed)),
                logical_counts=add_counts(*(row["accounting"]["logical_counts"] for row in closed)),
                infrastructure_counts=add_counts(
                    *(row["accounting"]["infrastructure_counts"] for row in closed)
                ),
                reaped_stream_cpu_seconds=reaped,
                coordinator_cpu_seconds=time.process_time() - cpu_started,
                wall_seconds=antecedents["worker_wall_seconds"] + time.monotonic() - started,
                original_worker_cpu_seconds=antecedents["worker_cpu_seconds"],
                original_global_cpu_seconds=antecedents["global_cpu_seconds"],
                continuation_worker_cpu_seconds=reaped + time.process_time() - cpu_started,
                resource_antecedents=antecedents,
            ),
        )
        if resume is not None:
            result["resume_manifest"] = binding(base / "resume_manifest.json")
        if failure:
            result.update(status="FAILED_STOPPED", error=failure)
        elif len(streams) == config["streams"]:
            from .local_coupling_multiround_metrics import summarize

            result.update(status="COMPLETED_ALL_ROUNDS", criteria=summarize(config, closed))
        path = root / "progress" / f"{sequence:06d}.json"
        write(path, result)
        sequence += 1
        temporary = root / "result.next"
        temporary.symlink_to(path.relative_to(root))
        os.replace(temporary, root / "result.json")
        return result

    try:
        certificate = _certify(launch)
        resume = freeze_resume(launch, certificate)
        for stream in range(config["streams"]):
            stdout = (root / f"s{stream}.stdout.log").open("x")
            stderr = (root / f"s{stream}.stderr.log").open("x")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    MODULE,
                    "--launch",
                    str(base / "launch.json"),
                    "--stream",
                    str(stream),
                ],
                cwd=launch["engineering"]["source_directory"],
                stdout=stdout,
                stderr=stderr,
            )
            processes.append(
                dict(stream=stream, process=process, stdout=stdout, stderr=stderr, reaped=False)
            )
        snapshot()
        while any(not row["reaped"] for row in processes):
            if (
                antecedents["worker_wall_seconds"] + time.monotonic() - started
                > config["budgets"]["worker"]["wall_seconds"]
            ):
                raise RuntimeError("Original plus continuation worker wall budget exhausted")
            for row in processes:
                if row["reaped"] or row["process"].poll() is None:
                    continue
                code = row["process"].wait()
                row["reaped"] = True
                usage = resource.getrusage(resource.RUSAGE_CHILDREN)
                reaped = usage.ru_utime + usage.ru_stime
                write(
                    root / "reaped" / f"s{row['stream']}.json",
                    dict(stream=row["stream"], exit_code=code, reaped_child_cpu_seconds=reaped),
                )
                path = root / "streams" / f"s{row['stream']}" / "result.json"
                if code != 0 or not path.exists() or read(path)["status"] != "COMPLETED_ALL_ROUNDS":
                    raise RuntimeError(f"Invalid continued stream s{row['stream']} code{code}")
                streams.append(dict(stream=row["stream"], result=binding(path)))
                snapshot()
            time.sleep(0.5)
    except BaseException as error:
        failure = repr(error)
        for row in processes:
            if row["process"].poll() is None:
                row["process"].terminate()
        for row in processes:
            try:
                row["process"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                row["process"].kill()
                row["process"].wait()
            path = root / "streams" / f"s{row['stream']}" / "result.json"
            if path.exists() and not any(old["stream"] == row["stream"] for old in streams):
                streams.append(dict(stream=row["stream"], result=binding(path)))
    finally:
        for row in processes:
            row["stdout"].close()
            row["stderr"].close()
    return snapshot()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--stream", type=int)
    args = parser.parse_args(argv)
    launch = verify_launch(args.launch)
    if args.stream is None:
        result = coordinator(launch)
    else:
        if args.stream not in range(3):
            raise ValueError("Frozen three-stream roster required")
        import torch

        from . import local_coupling_controller as controller
        from .local_coupling_numeric_validation import verify_native

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        controller.verify_native = verify_native
        resume = read(Path(launch["base_directory"]) / "resume_manifest.json")
        remaining = (
            read(bound(launch["config"]))["budgets"]["worker"]["stream_cpu_seconds"]
            - resume["budget_antecedents"]["stream_cpu_seconds_upper"][args.stream]
        )
        if args.stream == 0:
            remaining -= read(bound(resume["numerical_completion_receipt"]))["cpu_seconds"]
        if remaining <= 1:
            raise RuntimeError("No original CPU reservation remains")
        cap = int(remaining)
        resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 1))
        result = ResumeStudy(launch, args.stream, resume).run()
    return int(result["status"] != "COMPLETED_ALL_ROUNDS")


if __name__ == "__main__":
    raise SystemExit(main())
