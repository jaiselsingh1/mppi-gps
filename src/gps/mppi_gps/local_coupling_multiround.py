"""L3 fixed-length alternation; scientific outcomes never stop a planned round.

Each isolated process owns one retained paired learning stream. The coordinator
owns only process management and imports closed evidence; it does no science.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .local_coupling_loop import bound, read
from .provenance import fingerprint, write
from .trajectory_pi_accounting import directory_bytes
from .trajectory_pi_native import binding, load_trace

MODULE = "src.gps.mppi_gps.local_coupling_multiround"
ARMS = ("independent", "positive")


def planner_seed(config: dict, stream: int, slot: int) -> int:
    seeds = config["seeds"]
    if seeds["planner_round_stride"] != 0:
        raise ValueError("L3 fixes acquisition innovations across rounds")
    return seeds["planner_base"] + seeds["planner_stream_stride"] * stream + slot


def experience_fingerprint(episode: dict) -> str:
    raw = load_trace(bound(episode["trace"]))
    return fingerprint(dict(states=raw["states"][:-1], actions=raw["actions"]))


def child_limits(config: dict, counts: dict) -> dict:
    """Separate hard per-stream reservations fit the single shared work ceiling."""
    glob, worker = config["budgets"]["global"], config["budgets"]["worker"]
    result = dict(
        wall_seconds=worker["wall_seconds"],
        output_bytes=64 * 1024**2,
        memory_bytes=glob["memory_bytes"],
    )
    for unit in (
        "model_controls",
        "native_controls",
        "physics_substeps",
        "neural_rows",
        "backward_rows",
        "optimizer_steps",
    ):
        maximum = glob[unit] - (
            config["budgets"]["audit"]["neural_rows"] if unit == "neural_rows" else 0
        )
        result[unit] = maximum // config["streams"] - counts.get("requested_" + unit, 0)
    for unit in (
        "checkpoint_loads",
        "checkpoint_writes",
        "policy_constructions",
        "optimizer_constructions",
        "environment_constructions",
        "model_queries",
        "restores",
        "external_mj_forward_calls",
        "resets",
        "compiled_model_saves",
        "pi_passes",
    ):
        result[unit] = 1_000_000
    return result


def verify_launch(path: Path) -> dict:
    launch = read(path)
    if launch.get("schema") != "local-coupling-l3-launch-1":
        raise ValueError("Frozen L3 launch required")
    source, root = read(bound(launch["source_manifest"])), Path(launch["source_directory"])
    if Path(__file__).resolve() != root / "src/gps/mppi_gps/local_coupling_multiround.py":
        raise ValueError("Execute the immutable source export")
    for row in source["files"]:
        actual = binding(root / row["path"])
        if any(actual[k] != row[k] for k in ("bytes", "sha256")):
            raise ValueError("Frozen source changed: " + row["path"])
    config = read(bound(launch["config"]))
    if config["schema"] != "local-coupling-learning-l3-1" or not config["launch_ready"]:
        raise ValueError("Frozen launch-ready L3 recipe required")
    prepared = read(bound(launch["prepared_manifest"]))
    if prepared["config_sha256"] != launch["config"]["sha256"]:
        raise ValueError("Input seal/config mismatch")
    for spec in prepared["files"]:
        bound(spec)
    env = read(bound(launch["environment"]))
    for spec in (env["interpreter"], *env["binary_files"], env["model_xml"]):
        bound(spec)
    if binding(Path(sys.executable)) != env["interpreter"]:
        raise ValueError("Wrong bound interpreter")
    return launch


class StreamStudy:
    """One exact pair, all eight learning blocks, retained teacher safety only."""

    def __init__(self, launch: dict, stream: int):
        self.launch, self.stream = launch, stream
        self.base = Path(launch["base_directory"])
        self.root = self.base / "worker/streams" / f"s{stream}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.config, self.prepared = (
            read(bound(launch["config"])),
            read(bound(launch["prepared_manifest"])),
        )
        self.started, self.counts, self.receipts, self.index = time.monotonic(), {}, [], 0
        self.ledger = (self.root / "study_ledger.jsonl").open("x")
        initial = [x for x in self.prepared["initial_streams"] if x["stream"] == stream]
        self.incoming = {x["arm"]: x["envelope"] for x in initial}
        if set(self.incoming) != set(ARMS):
            raise ValueError("Complete retained pair required")
        self.initial = dict(self.incoming)
        self.anchors = {x["slot"]: read(bound(x["manifest"])) for x in self.prepared["anchors"]}
        self.retained = {(arm, slot): row for arm in ARMS for slot, row in self.anchors.items()}
        self.retained_binding = {
            (arm, x["slot"]): x["manifest"] for arm in ARMS for x in self.prepared["anchors"]
        }
        self.accepted_at = {(arm, slot): 0 for arm in ARMS for slot in range(8)}
        self.result = dict(
            schema="local-coupling-multiround-stream-1",
            status="RUNNING",
            stream=stream,
            planned_rounds=self.config["rounds"],
            config=launch["config"],
            initial_streams=[dict(arm=a, envelope=self.initial[a]) for a in ARMS],
            initial_evaluations=[],
            initial_probes=[],
            rounds=[],
            final_evaluations=[],
            accounting={},
        )

    def check(self):
        worker = self.config["budgets"]["worker"]
        if time.monotonic() - self.started > worker["wall_seconds"]:
            raise RuntimeError("Frozen stream wall budget exhausted")
        if time.process_time() > worker["stream_cpu_seconds"]:
            raise RuntimeError("Frozen stream CPU budget exhausted")
        if directory_bytes(self.root) + 1024**2 > worker["output_bytes"] // self.config["streams"]:
            raise RuntimeError("Frozen stream output reservation exhausted")
        for key, value in self.counts.items():
            if key.startswith("unknown_partial_") and value:
                raise RuntimeError("Unclosed paid scientific work")

    def job(self, function, job):
        self.check()
        job.update(config=self.launch["config"], limits=child_limits(self.config, self.counts))
        value = function(job)
        spec = binding(Path(value["path"]) / "receipt.json")
        if any(x["path"] == spec["path"] for x in self.receipts):
            raise ValueError("Duplicate work import")
        self.receipts.append(spec)
        for key, count in value["receipt"]["counts"].items():
            self.counts[key] = self.counts.get(key, 0) + count
        self.ledger.write(
            json.dumps(
                dict(
                    event="closed_child_included", receipt=spec, counts=value["receipt"]["counts"]
                ),
                sort_keys=True,
            )
            + "\n"
        )
        self.ledger.flush()
        os.fsync(self.ledger.fileno())
        if value["result"]["status"] != "COMPLETED":
            raise RuntimeError("Invalid execution: " + str(value["result"]))
        self.check()
        return value["result"]

    def snapshot(self):
        self.result["accounting"] = dict(
            counts=self.counts,
            child_receipts=self.receipts,
            worker_process_cpu_seconds=time.process_time(),
            worker_wall_seconds=time.monotonic() - self.started,
        )
        path = self.root / "progress" / f"{self.index:06d}.json"
        write(path, self.result)
        self.index += 1
        temp = self.root / "result.next"
        temp.symlink_to(path.relative_to(self.root))
        os.replace(temp, self.root / "result.json")

    def evaluate(self, checkpoint, arm, kind, panel, round_index, records):
        from .local_coupling_learning import evaluate_job

        first = None
        for i in range(self.config["seeds"]["panel_size"]):
            seed = self.config["seeds"][panel + "_base"] + i
            out = self.root / "evaluations" / f"r{round_index}_{arm}_{kind}" / panel / str(seed)
            self.job(
                evaluate_job,
                dict(
                    out=str(out),
                    checkpoint=checkpoint,
                    reset_seed=seed,
                    task=self.config["controller"]["task"],
                    physics_threads=4,
                    role=dict(
                        round=round_index, stream=self.stream, arm=arm, kind=kind, panel=panel
                    ),
                ),
            )
            row = dict(
                arm=arm,
                kind=kind,
                panel=panel,
                reset_seed=seed,
                result=binding(out / "result.json"),
            )
            records.append(row)
            if i == 0:
                first = row["result"]
        self.snapshot()
        return first

    def probe(self, evaluation, checkpoint, arm, round_index, records):
        from .local_coupling_multiround_measurements import probe_job

        seeds = self.config["seeds"]
        out = self.root / "probes" / f"r{round_index}_{arm}"
        self.job(
            probe_job,
            dict(
                out=str(out),
                evaluation=evaluation,
                checkpoint=checkpoint,
                restarts=self.prepared["restarts"],
                planner=self.config["controller"]["planner"],
                task=self.config["controller"]["task"],
                physics_threads=4,
                round=round_index,
                stream=self.stream,
                arm=arm,
                probe_seeds=[
                    seeds["probe_base"] + seeds["probe_stream_stride"] * self.stream + j
                    for j in range(3)
                ],
            ),
        )
        records.append(dict(arm=arm, result=binding(out / "result.json")))

    def acquire(self, round_index, block):
        from .local_coupling_controller import run_episode
        from .local_coupling_multiround_measurements import admit_or_retain

        selected = {arm: [] for arm in ARMS}
        for arm in ARMS:
            checkpoint = binding(Path(self.incoming[arm]["path"]).parent / "policy.pt")
            alpha = next(x["alpha"] for x in self.config["arms"] if x["name"] == arm)
            for slot in range(8):
                out = self.root / "acquisition" / f"r{round_index}" / arm / f"slot{slot}"
                episode = self.job(
                    run_episode,
                    dict(
                        out=str(out),
                        round=round_index,
                        stream=self.stream,
                        arm=arm,
                        slot=slot,
                        alpha=alpha,
                        seed=planner_seed(self.config, self.stream, slot),
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
                previous = self.retained_binding[(arm, slot)]
                old = self.retained[(arm, slot)]
                decision = admit_or_retain(episode, old, self.anchors[slot])
                chosen = decision["selected"]
                admitted = decision["fresh_acquisition_admitted"]
                before, after = experience_fingerprint(old), experience_fingerprint(chosen)
                if admitted:
                    self.retained[(arm, slot)] = episode
                    self.retained_binding[(arm, slot)] = spec
                    self.accepted_at[(arm, slot)] = round_index
                row = dict(
                    arm=arm,
                    slot=slot,
                    admitted=admitted,
                    fallback=not admitted,
                    previous=previous,
                    attempt=spec,
                    chosen=self.retained_binding[(arm, slot)],
                    last_accepted_round=self.accepted_at[(arm, slot)],
                    age=round_index - self.accepted_at[(arm, slot)],
                    previous_experience_fingerprint=before,
                    chosen_experience_fingerprint=after,
                    new_experience=bool(admitted and before != after),
                    new_rows=300 if admitted and before != after else 0,
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
                            admitted=admitted,
                            fallback=not admitted,
                            new_rows=row["new_rows"],
                        )
                    ),
                    flush=True,
                )
        return selected

    def run(self):
        from .local_coupling_learning import fit_job
        from .local_coupling_multiround_measurements import build_retained_corpora

        try:
            for arm in ARMS:
                cp = binding(Path(self.initial[arm]["path"]).parent / "policy.pt")
                first = self.evaluate(
                    cp, arm, "initial", "primary", 0, self.result["initial_evaluations"]
                )
                self.probe(first, cp, arm, 0, self.result["initial_probes"])
            for r in range(1, self.config["rounds"] + 1):
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
                selected = self.acquire(r, block)
                corpus = build_retained_corpora(
                    selected["independent"],
                    selected["positive"],
                    self.root / "corpus" / f"r{r}",
                    config_binding=self.launch["config"],
                )
                block["corpus"] = binding(self.root / "corpus" / f"r{r}" / "manifest.json")
                outgoing = {}
                for arm, data, cross in (("independent", "a", "b"), ("positive", "b", "a")):
                    out = self.root / "fits" / f"r{r}_{arm}"
                    fit = self.job(
                        fit_job,
                        dict(
                            out=str(out),
                            incoming=self.incoming[arm],
                            corpus=corpus["files"][data],
                            round=r,
                            stream=self.stream,
                            arm=arm,
                            diagnostics=dict(
                                own=corpus["files"][data],
                                cross=corpus["files"][cross],
                                held=self.prepared["held"],
                                anchor=self.prepared["anchor_corpus"],
                            ),
                        ),
                    )
                    block["fits"].append(dict(arm=arm, result=binding(out / "result.json")))
                    outgoing[arm] = fit["outgoing"]
                    block["outgoing_streams"].append(dict(arm=arm, envelope=fit["outgoing"]))
                    cp = binding(Path(fit["outgoing"]["path"]).parent / "policy.pt")
                    first = self.evaluate(cp, arm, "child", "primary", r, block["evaluations"])
                    self.probe(first, cp, arm, r, block["probes"])
                # Equal N, settings and exact incoming paired-PCG streams imply
                # identical minibatch-index tapes. Verify the operative evidence.
                tapes = [
                    np.load(bound(read(bound(x["result"]))["index_tape"]), allow_pickle=False)
                    for x in block["fits"]
                ]
                try:
                    if not np.array_equal(tapes[0]["indices"], tapes[1]["indices"]):
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
                            new_experience=sum(x["new_experience"] for x in block["admissions"]),
                            fallback=sum(x["fallback"] for x in block["admissions"]),
                            continuation="UNCONDITIONAL_NEXT_PLANNED_ROUND",
                        )
                    ),
                    flush=True,
                )
            for arm in ARMS:
                for kind, envelope, round_index in (
                    ("initial", self.initial[arm], 0),
                    ("final", self.incoming[arm], self.config["rounds"]),
                ):
                    self.evaluate(
                        binding(Path(envelope["path"]).parent / "policy.pt"),
                        arm,
                        kind,
                        "confirmation",
                        round_index,
                        self.result["final_evaluations"],
                    )
            self.result["status"] = "COMPLETED_ALL_ROUNDS"
        except BaseException as error:
            self.result.update(status="FAILED_STOPPED", error=repr(error))
        finally:
            self.snapshot()
            self.ledger.close()
        return self.result


def coordinator(launch: dict) -> dict:
    """Launch three bounded isolated streams; no outcome is an early stop."""
    base, source = Path(launch["base_directory"]), Path(launch["source_directory"])
    root = base / "worker"
    root.mkdir(exist_ok=False)
    config = read(bound(launch["config"]))
    write(root / "coordinator_identity.json", dict(pid=os.getpid()))
    processes = []
    started = time.monotonic()
    reaped = 0.0
    streams = []
    failure = None
    sequence = 0

    def snapshot():
        nonlocal sequence
        counts = {}
        for spec in streams:
            value = read(bound(spec["result"]))
            for key, n in value["accounting"]["counts"].items():
                counts[key] = counts.get(key, 0) + n
        result = dict(
            schema="local-coupling-multiround-l3-result-1",
            status="RUNNING",
            planned_rounds=config["rounds"],
            config=launch["config"],
            prepared_manifest=launch["prepared_manifest"],
            source_manifest=launch["source_manifest"],
            streams=streams,
            accounting=dict(
                counts=counts,
                reaped_stream_cpu_seconds=reaped,
                coordinator_cpu_seconds=time.process_time(),
                wall_seconds=time.monotonic() - started,
            ),
        )
        if failure:
            result.update(status="FAILED_STOPPED", error=failure)
        elif len(streams) == config["streams"]:
            from .local_coupling_multiround_metrics import summarize

            result["status"] = "COMPLETED_ALL_ROUNDS"
            result["criteria"] = summarize(config, [read(bound(x["result"])) for x in streams])
        target = root / "progress" / f"{sequence:06d}.json"
        write(target, result)
        sequence += 1
        temp = root / "result.next"
        temp.symlink_to(target.relative_to(root))
        os.replace(temp, root / "result.json")
        return result

    try:
        for s in range(config["streams"]):
            stdout = (root / f"s{s}.stdout.log").open("x")
            stderr = (root / f"s{s}.stderr.log").open("x")
            argv = [
                sys.executable,
                "-B",
                "-m",
                MODULE,
                "--launch",
                str(base / "launch.json"),
                "--stream",
                str(s),
            ]
            proc = subprocess.Popen(argv, cwd=source, stdout=stdout, stderr=stderr)
            processes.append(
                dict(stream=s, process=proc, stdout=stdout, stderr=stderr, reaped=False)
            )
        snapshot()
        while any(not x["reaped"] for x in processes):
            if time.monotonic() - started > config["budgets"]["worker"]["wall_seconds"]:
                raise RuntimeError("Shared worker wall exhausted")
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
                    raise RuntimeError(f"Invalid stream execution s{row['stream']} code{code}")
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
            if path.exists() and not any(x["stream"] == row["stream"] for x in streams):
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
        value = coordinator(launch)
    else:
        if args.stream not in range(3):
            raise ValueError("Frozen three-stream roster required")
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        cap = read(bound(launch["config"]))["budgets"]["worker"]["stream_cpu_seconds"]
        resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 1))
        value = StreamStudy(launch, args.stream).run()
    return int(value["status"] != "COMPLETED_ALL_ROUNDS")


if __name__ == "__main__":
    raise SystemExit(main())
