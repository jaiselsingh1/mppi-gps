"""Frozen fixed-alpha alternating full-trajectory PI study.

The coordinator owns chronology and admission, not the optimizer or learner.
It reuses the promoted FullPI worker, unchanged W2 fitting, immutable streams
and the common work ledger. Validation imports no simulator or neural library.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import fingerprint, write
from .trajectory_pi_accounting import Work, directory_bytes

SCHEMA = "trajectory-pi-p1-result-1"
CHECKS = {
    "full_context_replay",
    "pre_action_index",
    "full_sequence_no_shift",
    "score_formula",
    "checkpoint_continuation",
    "fall_accounting",
    "budget_enforcement",
    "zero_residual_reproduction",
}
CORE_UNITS = (
    "model_controls",
    "native_controls",
    "physics_substeps",
    "neural_rows",
    "backward_rows",
    "optimizer_steps",
)
IO_COUNTS: dict[str, int] = {}


def _io(name: str, amount: int) -> None:
    IO_COUNTS[name] = IO_COUNTS.get(name, 0) + amount


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while payload := stream.read(1024**2):
            _io("file_hash_read_bytes", len(payload))
            digest.update(payload)
    return digest.hexdigest()


def _need(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=_sha(path))


def _bound(value: dict) -> Path:
    _need(
        isinstance(value, dict) and {"path", "bytes", "sha256"} <= value.keys(),
        "Explicit file binding required",
    )
    path = Path(value["path"])
    _need(path.is_absolute() and path.is_file(), "Bound input unavailable: " + str(path))
    _need(
        type(value["bytes"]) is int
        and value["bytes"] == path.stat().st_size
        and _sha(path) == value["sha256"],
        "Bound input changed: " + str(path),
    )
    return path


def _json(path: Path) -> dict:
    payload = path.read_bytes()
    _io("metadata_read_bytes", len(payload))
    return json.loads(payload)


def _roster(root: Path, rows: list[dict]) -> None:
    _need(isinstance(rows, list) and bool(rows), "Nonempty file roster required")
    names = []
    for row in rows:
        name = row["path"]
        path = (root / name).resolve()
        _need(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and not {".", ".."} & set(Path(name).parts)
            and str(Path(name)) == name
            and path.is_relative_to(root.resolve()),
            "Unsafe file roster",
        )
        _need(name not in names, "Duplicate roster path")
        names.append(name)
        _bound(dict(row, path=str(path)))
    _need(names == sorted(names), "Canonical sorted file roster required")


def validate_launch(path: Path) -> tuple[dict, dict, dict, dict]:
    """Check exact source/input/prerequisite contracts before any paid task call."""
    launch = _json(path)
    _need(launch.get("schema") == "trajectory-pi-p1-launch-1", "Frozen launch schema required")
    _need(
        all(
            Path(launch[key]).is_absolute()
            for key in ("source_directory", "output_directory", "prepared_directory")
        ),
        "Explicit absolute study directories required",
    )
    files = {
        name: _bound(launch[name])
        for name in (
            "config",
            "source_manifest",
            "prepared_manifest",
            "preflight",
            "resource_observation",
        )
    }
    config, source, prepared = (
        _json(files[name]) for name in ("config", "source_manifest", "prepared_manifest")
    )
    _need(
        config.get("schema") == "trajectory-pi-p1-1" and config.get("launch_ready") is True,
        "Frozen launch-ready P1 configuration required",
    )
    source_root = Path(launch["source_directory"]).resolve()
    _need(source.get("schema") == "trajectory-pi-source-1", "Frozen source list schema required")
    canonical = json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()
    _need(hashlib.sha256(canonical).hexdigest() == source["roster_sha256"], "Source roster changed")
    _roster(source_root, source["files"])
    _need(
        "src/gps/mppi_gps/trajectory_pi_study.py" in {row["path"] for row in source["files"]},
        "Coordinator absent from source roster",
    )
    own = source_root / "src/gps/mppi_gps/trajectory_pi_study.py"
    _need(own.resolve() == Path(__file__).resolve(), "Execute the frozen coordinator source")
    data_root = Path(launch["prepared_directory"]).resolve()
    _need(
        prepared.get("schema") == "trajectory-pi-prepared-1"
        and prepared.get("config_sha256") == launch["config"]["sha256"]
        and prepared.get("normalization_inherited") is True,
        "Frozen prepared inputs differ",
    )
    _roster(data_root, prepared["files"])
    _need(
        {row["stream"] for row in prepared["parents"]} == set(config["streams"]),
        "Prepared parent roster differs",
    )
    _need(
        {row["slot"] for row in prepared["teachers"]}
        == {row["slot"] for row in config["teachers"]},
        "Prepared teacher roster differs",
    )
    listed = {row["path"] for row in prepared["files"]}
    parent_config = {row["stream"]: row for row in config["parents"]}
    for parent in prepared["parents"]:
        directory = parent["directory"]
        _need(
            directory == parent_config[parent["stream"]]["prepared_directory"]
            and all(
                str(Path(directory) / name) in listed
                for name in ("policy.pt", "global_rng.pt", "stream.json")
            ),
            "Prepared parent stream absent from bound roster",
        )
        _need(
            _sha(data_root / directory / "policy.pt")
            == parent_config[parent["stream"]]["checkpoint"]["sha256"],
            "Prepared policy differs from the recorded W2 parent",
        )
    teacher_config = {row["slot"]: row for row in config["teachers"]}
    _need(
        all(
            row["file_relative"] == teacher_config[row["slot"]]["prepared_path"]
            and row["file_relative"] in listed
            for row in prepared["teachers"]
        ),
        "Prepared teacher absent from bound roster",
    )
    seals = {
        name + "_sha256": launch[name]["sha256"]
        for name in ("config", "source_manifest", "prepared_manifest")
    }
    preflight, capacity = _json(files["preflight"]), _json(files["resource_observation"])
    for receipt in (preflight, capacity):
        _need(
            all(receipt.get(key) == value for key, value in seals.items()),
            "Prerequisite input/source seal differs",
        )
    _need(
        preflight.get("schema") == "trajectory-pi-preflight-1"
        and preflight.get("status") == "PASS",
        "Prerequisites did not pass",
    )
    rows = preflight.get("checks", [])
    _need(
        isinstance(rows, list)
        and len({row["name"] for row in rows}) == len(rows)
        and CHECKS <= {row["name"] for row in rows}
        and all(row["passed"] is True for row in rows),
        "Required prerequisite checks absent or failed",
    )
    _need(
        all(
            not key.startswith("unknown_partial_") or value == 0
            for key, value in preflight.get("work", {}).items()
        ),
        "Prerequisite contains unknown partial science work",
    )
    _need(
        capacity.get("schema") == "trajectory-pi-resource-observation-1"
        and capacity.get("passed") is True,
        "Capacity prerequisite failed",
    )
    _need(
        0 <= time.time() - capacity["observed_unix"] <= 300,
        "Capacity observation must be fresh at launch",
    )
    _need(
        capacity["available_logical_cpus"] >= 28
        and capacity["available_memory_bytes"] >= 32 * 1024**3
        and capacity["free_disk_bytes"] >= 80 * 1024**3,
        "Insufficient available capacity",
    )
    for key in ("python", "numpy", "scipy", "mujoco", "torch", "h5py"):
        _need(
            capacity["versions"][key] == config["resources"]["runtime"][key],
            "Capacity runtime mismatch: " + key,
        )
    return config, launch, prepared, preflight


def admit(candidate: dict, initial_cost: float) -> bool:
    outcome = candidate["result"]["outcome"]
    return bool(
        outcome["walking"] is True
        and outcome["full_survival"] is True
        and outcome["complete_cost"] <= 1.2 * initial_cost
    )


def summary(outcomes: list[dict]) -> dict:
    _need(bool(outcomes), "Nonempty evaluation panel required")
    return dict(
        episodes=len(outcomes),
        walking=sum(row["walking"] is True for row in outcomes),
        full_survival=sum(row["full_survival"] is True for row in outcomes),
        mean_cost=float(np.mean([row["complete_cost"] for row in outcomes])),
        mean_controls=float(np.mean([row["controls"] for row in outcomes])),
    )


def native_limits(config: dict, kind: str, alpha: float, iterations: int = 0) -> dict:
    horizon, count = config["horizon"], config["planner"]["K"]
    model = (iterations * count * horizon + horizon) if kind == "solve" else 0
    neural = (
        iterations * (count * horizon if alpha else horizon) + 2 * horizon
        if kind == "solve"
        else horizon
    )
    return dict(
        wall_seconds=3600,
        output_bytes=(512 if kind == "solve" else 16) * 1024**2,
        memory_bytes=config["resources"]["worker_ram_bytes"],
        model_controls=model,
        native_controls=horizon,
        physics_substeps=4 * (model + horizon),
        neural_rows=neural,
        pi_passes=iterations,
        model_queries=iterations + 1,
        environment_constructions=1,
        checkpoint_loads=1,
        numpy_constructions=1,
        restores=2 if kind == "solve" else 0,
        external_mj_forward_calls=2,
        resets=1 if kind == "actor" else 0,
    )


class StudyWork:
    """Small global reservation layer over the common native/fit ledger."""

    def __init__(self, out: Path, config: dict, preflight: dict, *, origin: tuple | None = None):
        limits = dict(
            wall_seconds=config["limits"]["execution_wall_seconds"],
            output_bytes=config["limits"]["execution_output_bytes"],
            memory_bytes=config["resources"]["coordinator_ram_bytes"],
            **{key: config["limits"][key] for key in CORE_UNITS},
        )
        limits.update(
            {
                key: 100000
                for key in (
                    "pi_passes",
                    "model_queries",
                    "environment_constructions",
                    "checkpoint_loads",
                    "numpy_constructions",
                    "restores",
                    "external_mj_forward_calls",
                    "resets",
                    "policy_constructions",
                    "optimizer_constructions",
                    "checkpoint_writes",
                )
            }
        )
        self.out, self.config = out, config
        self.work = Work(out / "coordinator", limits)
        self.pending: dict[str, dict] = {}
        self.pool = None
        self.started, self.cpu = origin or (time.monotonic(), time.process_time())
        self.work.started, self.work.cpu_started = self.started, self.cpu
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.child_cpu_start = usage.ru_utime + usage.ru_stime
        self.peak_tree_rss = 0
        for key, value in preflight.get("work", {}).items():
            if key.startswith(("requested_", "completed_", "unknown_partial_")):
                _need(type(value) is int and value >= 0, "Invalid prerequisite work count")
                self.work.counts[key] = value
        self.work.record("prerequisite_work_included", counts=dict(self.work.counts))

    def check(self, *, final: bool = False) -> None:
        self.work.check()
        _need(
            directory_bytes(self.out) + 1024**2 <= self.config["limits"]["execution_output_bytes"],
            "Global output budget exhausted",
        )
        for unit in CORE_UNITS:
            _need(
                self.work.counts.get("requested_" + unit, 0)
                + sum(row.get(unit, 0) for row in self.pending.values())
                <= self.config["limits"][unit],
                "Global requested/reserved budget exhausted: " + unit,
            )
        own_cpu = time.process_time() - self.cpu
        child = resource.getrusage(resource.RUSAGE_CHILDREN)
        worker_cpu = child.ru_utime + child.ru_stime - self.child_cpu_start
        rss = 0
        if Path("/proc").is_dir():
            ticks = os.sysconf("SC_CLK_TCK")
            for process in [] if self.pool is None else self.pool._pool:
                path = Path("/proc") / str(process.pid)
                try:
                    stats = (path / "stat").read_text().rsplit(")", 1)[1].split()
                    worker_cpu += (int(stats[11]) + int(stats[12])) / ticks
                    rss += int((path / "statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
                except FileNotFoundError:
                    pass
            rss += int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            self.peak_tree_rss = max(self.peak_tree_rss, rss)
            _need(
                rss <= self.config["resources"]["aggregate_ram_bytes"],
                "Aggregate RAM budget exhausted",
            )
        _need(
            worker_cpu <= self.config["limits"]["worker_cpu_seconds"]
            and own_cpu <= self.config["limits"]["coordinator_cpu_seconds"]
            and own_cpu + worker_cpu <= self.config["limits"]["cpu_seconds"],
            "CPU budget exhausted",
        )
        if final:
            self.final_cpu = dict(worker_cpu_seconds=worker_cpu, coordinator_cpu_seconds=own_cpu)

    def reserve(self, job: dict) -> None:
        token = job["out"]
        _need(token not in self.pending, "Duplicate pending worker")
        self.pending[token] = {
            key: value
            for key, value in job["limits"].items()
            if key not in ("wall_seconds", "output_bytes", "memory_bytes")
        }
        try:
            self.check()
        except BaseException:
            self.pending.pop(token)
            raise
        self.work.record("worker_reserved", path=token, reservation=self.pending[token])

    def complete(self, job: dict, receipt: dict) -> None:
        token = job["out"]
        reserved = self.pending[token]
        # Persist actual paid counts before checking any overrun or removing its
        # reservation. A failing worker cannot disappear from the accounting.
        for key, value in receipt["counts"].items():
            _need(type(value) is int and value >= 0, "Invalid worker work amount")
            self.work.counts[key] = self.work.counts.get(key, 0) + value
        self.work.record("worker_returned", path=token, receipt=receipt)
        self.pending.pop(token)
        _need(receipt["status"] == "COMPLETED", "Native worker failed; no retries")
        _need(
            all(
                receipt["counts"].get("requested_" + key, 0) <= value
                for key, value in reserved.items()
            ),
            "Worker exceeded its reservation",
        )
        self.check()

    def unknown_pending(self) -> None:
        for path, charges in list(self.pending.items()):
            receipt = Path(path) / "receipt.json"
            try:
                actual = _json(receipt) if receipt.is_file() else None
                if actual is not None:
                    _need(
                        isinstance(actual["counts"], dict)
                        and all(type(n) is int and n >= 0 for n in actual["counts"].values()),
                        "Invalid partial worker receipt",
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                self.work.record("unreadable_worker_receipt", path=path, error=repr(error))
                actual = None
            if actual is not None:
                for key, value in actual["counts"].items():
                    self.work.counts[key] = self.work.counts.get(key, 0) + value
                self.work.record("failed_worker_receipt_recovered", path=path, receipt=actual)
            else:
                for key, value in charges.items():
                    for prefix in ("requested_", "unknown_partial_"):
                        name = prefix + key
                        self.work.counts[name] = self.work.counts.get(name, 0) + value
                self.work.record("unknown_pending_worker", path=path, reserved=charges)
            self.pending.pop(path)

    def snapshot(self) -> dict:
        return dict(
            **self.work.receipt(),
            pending=dict(self.pending),
            aggregate_rss_peak_sampled_bytes=self.peak_tree_rss,
            final_cpu=getattr(self, "final_cpu", None),
            coordinator_logical_io=dict(IO_COUNTS),
            io_scope="Coordinator metadata/hash bytes and decoded arrays measured; checkpoint package sizes opened/written, not physical driver I/O",
            native_worker_accounting="Actual receipt counters; absent receipts charged full unknown reservation",
        )


def run_jobs(pool: Any, jobs: list[dict], ledger: StudyWork) -> list[dict]:
    from .trajectory_pi_native import run_job

    results = []
    for begin in range(0, len(jobs), ledger.config["resources"]["native_workers"]):
        batch = jobs[begin : begin + ledger.config["resources"]["native_workers"]]
        futures = []
        for job in batch:
            ledger.reserve(job)
            futures.append(pool.apply_async(run_job, (job,)))
        while not all(future.ready() for future in futures):
            ledger.check()
            time.sleep(0.1)
        for job, future in zip(batch, futures, strict=True):
            returned = future.get()
            ledger.complete(job, returned["receipt"])
            results.append(returned)
    return results


def _trace(path: Path) -> dict[str, np.ndarray]:
    _io("array_package_bytes_opened", path.stat().st_size)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            values = {key: source[key].copy() for key in source.files}
    else:
        import h5py

        with h5py.File(path, "r") as source:
            values = {key: source[key][:] for key in source}
    _io("decoded_array_bytes", sum(value.nbytes for value in values.values()))
    return values


def _training(bank: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    from .trajectory_pi_student import observations

    inputs, targets = [], []
    for item in bank:
        _bound(item["trace_binding"])
        raw = _trace(Path(item["trace"]))
        values = observations(raw["states"][:-1])
        _need(
            raw["states"].shape == (len(raw["actions"]) + 1, 19) and len(raw["actions"]) == 1000,
            "Complete N+1 walking teacher required",
        )
        recorded = (
            raw["observations"][:-1] if len(raw["observations"]) == 1001 else raw["observations"]
        )
        _need(
            np.array_equal(values, recorded),
            "Teacher labels are not native pre-action observations",
        )
        inputs.append(values)
        targets.append(raw["actions"])
    return np.concatenate(inputs), np.concatenate(targets)


def teacher_gap(bank: list[dict], actor: Any, meter: Any) -> dict:
    per_path = []
    for item in bank:
        inputs, targets = _training([item])
        prediction = actor.predict(inputs, meter=meter).astype(np.float64)
        D = float(0.5 * np.sum((targets.astype(np.float64) - prediction) ** 2))
        per_path.append(
            dict(
                slot=item["slot"],
                controls=len(targets),
                D=D,
                rms=float(math.sqrt(2 * D / len(targets))),
                trace_binding=item["trace_binding"],
            )
        )
    return dict(
        policy_hash=actor.policy_hash,
        paths=per_path,
        mean_D=float(np.mean([row["D"] for row in per_path])),
        rms=float(np.mean([row["rms"] for row in per_path])),
    )


class Operations:
    """The concrete file/data operations used by the small fixed round schedule."""

    def __init__(self, config: dict, launch: dict, prepared: dict, ledger: StudyWork, pool: Any):
        self.c, self.launch, self.ledger, self.pool = config, launch, ledger, pool
        self.out, self.data = Path(launch["output_directory"]), Path(launch["prepared_directory"])
        self.parents = {row["stream"]: self.data / row["directory"] for row in prepared["parents"]}
        self.current = {
            (stream, arm["name"]): path
            for stream, path in self.parents.items()
            for arm in config["arms"]
        }
        self.alpha = {arm["name"]: arm["alpha"] for arm in config["arms"]}
        self.initial_bank = []
        for row in sorted(prepared["teachers"], key=lambda item: item["slot"]):
            path = self.data / row["file_relative"]
            raw = _trace(path)
            outcome = dict(
                controls=len(raw["actions"]),
                walking=True,
                full_survival=True,
                complete_cost=float(np.sum(raw["costs"])),
                velocity_mae=float(np.mean(np.abs(raw["states"][1:, 10] - 1.5))),
            )
            _need(
                len(raw["actions"]) == 1000
                and math.isfinite(outcome["complete_cost"])
                and outcome["velocity_mae"] <= 0.3
                and np.all(
                    (raw["states"][1:, 2] > 0.8)
                    & (raw["states"][1:, 2] < 2)
                    & (np.abs(raw["states"][1:, 3]) < 1)
                ),
                "Prepared teacher no longer walks",
            )
            self.initial_bank.append(
                dict(
                    slot=row["slot"],
                    trace=str(path),
                    trace_binding=binding(path),
                    plan=str(path),
                    plan_binding=binding(path),
                    context=str(path),
                    context_binding=binding(path),
                    origin_round=0,
                    outcome=outcome,
                )
            )
        self.bank = {key: list(self.initial_bank) for key in self.current}

    def path(self, round_index: int) -> Path:
        return self.out / "rounds" / f"round_{round_index:03d}"

    def checkpoint(self, key: tuple[int, str]) -> Path:
        return self.current[key] / "policy.pt"

    def actor(self, checkpoint: Path) -> Any:
        from .trajectory_pi_student import NumpyPolicy

        result = self.ledger.work.call(
            "numpy_checkpoint",
            dict(checkpoint_loads=1, numpy_constructions=1),
            lambda: NumpyPolicy.from_checkpoint(checkpoint),
        )
        _io("checkpoint_package_bytes_opened", checkpoint.stat().st_size)
        return result

    def job(
        self, out: Path, checkpoint: Path, *, kind: str, alpha: float = 0.0, **extra: Any
    ) -> dict:
        iterations = extra.get("iterations", 0)
        return dict(
            kind=kind,
            out=str(out),
            task=self.c["task"],
            physics_threads=self.c["resources"]["native_threads_per_worker"],
            horizon=self.c["horizon"],
            checkpoint=str(checkpoint),
            checkpoint_binding=binding(checkpoint),
            alpha=alpha,
            limits=native_limits(self.c, kind, alpha, iterations),
            **extra,
        )

    def execute(self, jobs: list[dict]) -> list[dict]:
        return run_jobs(self.pool, jobs, self.ledger)

    def old_gaps(self) -> dict:
        return {
            key: teacher_gap(self.bank[key], self.actor(self.checkpoint(key)), self.ledger.work)
            for key in self.current
        }

    def acquire(self, round_index: int, old: dict) -> dict:
        jobs, tags = [], []
        seeds = self.c["seeds"]
        for key in self.current:
            stream, arm = key
            for item in self.bank[key]:
                slot = item["slot"]
                jobs.append(
                    self.job(
                        self.path(round_index) / f"s{stream}" / arm / "acquisition" / f"slot{slot}",
                        self.checkpoint(key),
                        kind="solve",
                        alpha=self.alpha[arm],
                        context_file=item["context"],
                        context_binding=item["context_binding"],
                        context_step=0,
                        initial_plan_file=item["plan"],
                        initial_plan_binding=item["plan_binding"],
                        planner=self.c["planner"],
                        planner_seed=seeds["planner_argument_base"]
                        + stream * seeds["stream_stride"]
                        + round_index * seeds["round_stride"]
                        + slot * seeds["slot_stride"],
                        iterations=self.c["optimizer"]["iterations"],
                        full_banks=self.c["optimizer"]["full_bank_iterations"],
                    )
                )
                tags.append((key, slot))
        results = self.execute(jobs)
        rows = {
            key: dict(teacher_before=old[key], admission=[], raw_candidates=[])
            for key in self.current
        }
        for (key, slot), returned in zip(tags, results, strict=True):
            initial = self.initial_bank[slot]
            accepted = admit(returned, initial["outcome"]["complete_cost"])
            rows[key]["admission"].append(
                dict(
                    slot=slot,
                    accepted=accepted,
                    outcome=returned["result"]["outcome"],
                    path=returned["path"],
                )
            )
            rows[key]["raw_candidates"].append(returned)
            if accepted:
                directory = Path(returned["path"])
                previous = self.bank[key][slot]
                self.bank[key][slot] = dict(
                    slot=slot,
                    trace=str(directory / "trace.h5"),
                    trace_binding=binding(directory / "trace.h5"),
                    plan=str(directory / "solution.npz"),
                    plan_binding=binding(directory / "solution.npz"),
                    context=previous["context"],
                    context_binding=previous["context_binding"],
                    origin_round=round_index,
                    outcome=returned["result"]["outcome"],
                )
        for key in self.current:
            paths = []
            for slot, item in enumerate(self.bank[key]):
                accepted = rows[key]["admission"][slot]["accepted"]
                D = (
                    rows[key]["raw_candidates"][slot]["result"]["selected_D"]
                    if accepted
                    else old[key]["paths"][slot]["D"]
                )
                paths.append(
                    dict(
                        slot=slot,
                        D=D,
                        controls=1000,
                        rms=math.sqrt(2 * D / 1000),
                        trace_binding=item["trace_binding"],
                    )
                )
            rows[key]["teacher_after_acquisition"] = dict(
                paths=paths,
                mean_D=float(np.mean([row["D"] for row in paths])),
                rms=float(np.mean([row["rms"] for row in paths])),
                policy_hash=old[key]["policy_hash"],
            )
            rows[key]["bank"] = list(self.bank[key])
        return rows

    def fit(self, round_index: int, rows: dict) -> None:
        import torch

        from .trajectory_pi_student import fit_round, load_stream, save_stream

        for key in self.current:
            stream, arm = key
            directory = self.path(round_index) / f"s{stream}" / arm / "fit"
            directory.mkdir(parents=True, exist_ok=False)
            incoming = self.current[key]
            actor, state, _ = self.ledger.work.call(
                "load_stream",
                dict(checkpoint_loads=2, policy_constructions=1),
                lambda: load_stream(incoming),
            )
            _io(
                "checkpoint_package_bytes_opened",
                sum((incoming / name).stat().st_size for name in ("policy.pt", "global_rng.pt")),
            )
            inputs, targets = _training(self.bank[key])
            data = dict(
                bank=self.bank[key],
                parent_stream_directory=str(incoming),
                parent_policy=binding(incoming / "policy.pt"),
                parent_envelope=binding(incoming / "stream.json"),
                incoming_optimizer_fingerprint=fingerprint(state),
            )
            write(directory / "data_bindings.json", data)
            indices, updates, fit_updates = [], [], []
            rng_file = (directory / "index_rng.jsonl").open("x")

            def initial(value: dict) -> None:
                def save() -> None:
                    with (directory / "initial_state.pt").open("xb") as target:
                        torch.save(value, target)
                        target.flush()
                        os.fsync(target.fileno())

                self.ledger.work.call("fit_initial_proof", dict(checkpoint_writes=1), save)
                _io("checkpoint_bytes_written", (directory / "initial_state.pt").stat().st_size)

            def tape(value: dict) -> None:
                indices.append(value["indices"].copy())
                updates.append(value["update"])
                fit_updates.append(value["fit_update"])
                copied = {
                    name: value[name]
                    for name in ("update", "fit_update", "rng_before", "rng_after")
                }
                copied["indices_fingerprint"] = fingerprint(value["indices"])
                rng_file.write(json.dumps(copied, sort_keys=True) + "\n")
                rng_file.flush()
                os.fsync(rng_file.fileno())

            try:
                child, continued, evidence = self.ledger.work.call(
                    "fit_block_constructors",
                    dict(policy_constructions=1, optimizer_constructions=1),
                    lambda: fit_round(
                        actor,
                        inputs,
                        targets,
                        updates=self.c["learning"]["updates"],
                        seed=self.c["seeds"]["fit_argument_base"] + stream,
                        optimizer_state=state,
                        resume_fit=False,
                        meter=self.ledger.work,
                        initial_observer=initial,
                        index_observer=tape,
                    ),
                )
            except BaseException as error:
                if hasattr(error, "fit_evidence"):
                    write(directory / "failed_fit.json", error.fit_evidence)
                raise
            finally:
                rng_file.close()
            with (directory / "index_tape.npz").open("xb") as target:
                np.savez_compressed(
                    target,
                    indices=np.asarray(indices, dtype=np.int64),
                    updates=np.asarray(updates),
                    fit_updates=np.asarray(fit_updates),
                )
            outgoing = directory.parent / "stream"
            saved = self.ledger.work.call(
                "save_stream",
                dict(checkpoint_writes=2),
                lambda: save_stream(
                    outgoing, child, continued, dict(round=round_index, stream=stream, arm=arm)
                ),
            )
            _io("checkpoint_bytes_written", sum(item["bytes"] for item in saved["files"].values()))
            reloaded, reloaded_state, _ = self.ledger.work.call(
                "reload_stream",
                dict(checkpoint_loads=2, policy_constructions=1),
                lambda: load_stream(outgoing),
            )
            _io(
                "checkpoint_package_bytes_opened",
                sum(item["bytes"] for item in saved["files"].values()),
            )
            _need(
                reloaded.hash() == child.hash()
                and fingerprint(reloaded_state) == fingerprint(continued),
                "Actual per-round actor/Adam/private-RNG reload differs",
            )
            evidence["study"] = dict(
                data_bindings=binding(directory / "data_bindings.json"),
                initial_state=binding(directory / "initial_state.pt"),
                index_tape=binding(directory / "index_tape.npz"),
                index_rng=binding(directory / "index_rng.jsonl"),
                child_stream_directory=str(outgoing),
                parent_policy_hash=actor.hash(),
                child_policy_hash=child.hash(),
                saved_optimizer_fingerprint=saved["optimizer_fingerprint"],
                reloaded_optimizer_fingerprint=fingerprint(reloaded_state),
            )
            write(directory / "fit.json", evidence)
            self.current[key] = outgoing
            rows[key]["fit"] = dict(
                path=str(directory),
                binding=binding(directory / "fit.json"),
                updates=evidence["completed_updates"],
            )
            rows[key]["teacher_after_fit"] = teacher_gap(
                self.bank[key], self.actor(self.checkpoint(key)), self.ledger.work
            )

    def evaluate(
        self, round_index: int, *, confirmation: bool = False, initial: bool = False
    ) -> dict:
        base = self.out / "final" if confirmation else self.path(round_index)
        panel = (
            self.c["evaluation"]["confirmation_seeds"]
            if confirmation
            else self.c["evaluation"]["panels"][round_index]["reset_seeds"]
        )
        jobs, tags = [], []
        if not initial:
            for key in self.current:
                stream, arm = key
                for index, seed in enumerate(panel):
                    jobs.append(
                        self.job(
                            base / "evaluation" / f"s{stream}" / arm / f"e{index:02d}",
                            self.checkpoint(key),
                            kind="actor",
                            reset_seed=seed,
                        )
                    )
                    tags.append((key, index))
        for stream, parent in self.parents.items():
            for index, seed in enumerate(panel):
                jobs.append(
                    self.job(
                        base / "evaluation/parents" / f"s{stream}" / f"e{index:02d}",
                        parent / "policy.pt",
                        kind="actor",
                        reset_seed=seed,
                    )
                )
                tags.append(((stream, "parent"), index))
        returned = self.execute(jobs)
        grouped: dict[tuple[int, str], list[dict]] = {}
        for (key, _), result in zip(tags, returned, strict=True):
            grouped.setdefault(key, []).append(result)
        if initial:
            for key in self.current:
                grouped[key] = grouped[(key[0], "parent")]
        import h5py

        contexts: dict[str, str] = {}
        reference = []
        for key, values in grouped.items():
            for index, item in enumerate(values):
                path = str(Path(item["path"]) / "trace.h5")
                if path not in contexts:
                    with h5py.File(path, "r") as raw:
                        context = {
                            name: raw[name][0]
                            for name in ("states", "warmstarts", "integrations", "observations")
                        }
                    _io("decoded_array_bytes", sum(value.nbytes for value in context.values()))
                    contexts[path] = fingerprint(context)
                if len(reference) <= index:
                    reference.append(contexts[path])
                _need(
                    contexts[path] == reference[index], "Fresh paired resets differ across policies"
                )
        return {
            key: dict(
                jobs=values,
                outcomes=[row["result"]["outcome"] for row in values],
                summary=summary([row["result"]["outcome"] for row in values]),
                paired_reset_context_fingerprints=list(reference),
            )
            for key, values in grouped.items()
        }

    def queries(self, round_index: int, evaluations: dict) -> dict:
        jobs, tags, result = [], [], {key: {} for key in self.current}
        seeds, query = self.c["seeds"], self.c["queries"]
        template = self.initial_bank[query["initial_template_slot"]]
        for key in self.current:
            stream, arm = key
            episode = evaluations[key]["jobs"][query["evaluation_episode"]]
            trace = Path(episode["path"]) / "trace.h5"
            values = _trace(trace)
            for index, anchor in enumerate(query["anchors"]):
                if len(values["actions"]) <= anchor:
                    result[key][str(anchor)] = dict(
                        present=False,
                        qualified=False,
                        reason="Student terminated before fixed pre-action anchor",
                        action_gap=None,
                    )
                    continue
                jobs.append(
                    self.job(
                        self.path(round_index) / "queries" / f"s{stream}" / arm / f"a{anchor:03d}",
                        self.checkpoint(key),
                        kind="solve",
                        alpha=self.alpha[arm],
                        context_file=str(trace),
                        context_binding=binding(trace),
                        context_step=anchor,
                        initial_plan_file=template["plan"],
                        initial_plan_binding=template["plan_binding"],
                        planner=self.c["planner"],
                        planner_seed=seeds["query_argument_base"]
                        + stream * seeds["stream_stride"]
                        + round_index * seeds["round_stride"]
                        + index * seeds["slot_stride"],
                        iterations=query["iterations"],
                        full_banks=query["full_bank_iterations"],
                    )
                )
                tags.append((key, anchor, episode, values["actions"][anchor].copy()))
        for (key, anchor, episode, action), returned in zip(tags, self.execute(jobs), strict=True):
            with np.load(Path(returned["path"]) / "solution.npz", allow_pickle=False) as raw:
                gap = float(
                    np.linalg.norm(raw["actions"][0].astype(np.float64) - action.astype(np.float64))
                )
            outcome = returned["result"]["outcome"]
            result[key][str(anchor)] = dict(
                present=True,
                qualified=outcome["walking"] is True,
                action_gap=gap,
                query_outcome=outcome,
                path=returned["path"],
                actor_trace=episode["result"]["trace"],
                policy_cost_gap=(
                    episode["result"]["outcome"]["complete_cost"] - outcome["complete_cost"]
                )
                if anchor == 0
                else None,
                cost_pairing="same full1000 initial context"
                if anchor == 0
                else "unpaired horizons",
            )
        return result

    def close_round(self, round_index: int, rows: dict, evaluations: dict, queries: dict) -> dict:
        complete = dict(schema="trajectory-pi-round-1", round=round_index, streams={}, parents={})
        for key in self.current:
            stream, arm = key
            row = rows[key]
            row["student"] = evaluations[key]
            row["queries"] = queries[key]
            complete["streams"].setdefault(str(stream), {})[arm] = row
            write(self.path(round_index) / f"s{stream}" / arm / "measurements.json", row)
        for stream in self.parents:
            complete["parents"][str(stream)] = evaluations[(stream, "parent")]
        write(self.path(round_index) / "complete.json", complete)
        write(
            self.path(round_index) / "state.json",
            dict(
                schema="trajectory-pi-state-1",
                round=round_index,
                seals={
                    key: self.launch[key]["sha256"]
                    for key in ("config", "source_manifest", "prepared_manifest", "preflight")
                },
                streams=[
                    dict(stream=key[0], arm=key[1], directory=str(path), bank=self.bank[key])
                    for key, path in self.current.items()
                ],
                accounting=self.ledger.snapshot(),
            ),
        )
        return complete


def execute_schedule(config: dict, operations: Any) -> tuple[list[dict], dict]:
    """One fixed chronology; the synthetic lifecycle fixture supplies only operations."""
    first = operations.old_gaps()
    rows = {
        key: dict(
            teacher_before=value,
            teacher_after_acquisition=value,
            teacher_after_fit=value,
            admission=[],
            raw_candidates=[],
            bank=list(operations.bank[key]),
            fit=None,
        )
        for key, value in first.items()
    }
    evaluation = operations.evaluate(0, initial=True)
    query = operations.queries(0, evaluation)
    history = [operations.close_round(0, rows, evaluation, query)]
    for round_index in range(1, config["rounds"] + 1):
        rows = operations.acquire(round_index, operations.old_gaps())
        operations.fit(round_index, rows)
        evaluation = operations.evaluate(round_index)
        query = operations.queries(round_index, evaluation)
        history.append(operations.close_round(round_index, rows, evaluation, query))
    return history, operations.evaluate(config["rounds"], confirmation=True)


def _window(history: list[dict], rounds: list[int], stream: int, arm: str) -> list[dict]:
    by_round = {row["round"]: row for row in history}
    return [by_round[index]["streams"][str(stream)][arm] for index in rounds]


def _contraction(
    early: list[float],
    late: list[float],
    control: list[float],
    minimum: float,
    maximum_ratio: float,
) -> dict:
    before, after, independent = map(float, (np.mean(early), np.mean(late), np.mean(control)))
    resolved = before > 1e-12 and independent > 1e-12
    return dict(
        resolved=resolved,
        early=before,
        late=after,
        independent_late=independent,
        contraction=1 - after / before if resolved else None,
        late_ratio=after / independent if resolved else None,
        passed=bool(
            resolved and after <= (1 - minimum) * before and after <= maximum_ratio * independent
        ),
    )


def walking_contrast(child: dict, independent: dict, parent: dict, threshold: int) -> dict:
    controls = (independent, parent)
    return dict(
        passed=all(child["walking"] - other["walking"] >= threshold for other in controls),
        no_regression=all(
            child["walking"] >= other["walking"]
            and child["full_survival"] >= other["full_survival"]
            for other in controls
        ),
        walking_vs_independent=child["walking"] - independent["walking"],
        walking_vs_parent=child["walking"] - parent["walking"],
        survival_vs_independent=child["full_survival"] - independent["full_survival"],
        survival_vs_parent=child["full_survival"] - parent["full_survival"],
    )


def criteria(history: list[dict], confirmation: dict, config: dict) -> dict:
    gates = config["criteria"]
    per_stream, joint = {}, []
    all_nonregression = True
    for stream in config["streams"]:
        early = _window(history, gates["early_rounds"], stream, "coupled")
        late = _window(history, gates["late_rounds"], stream, "coupled")
        independent = _window(history, gates["late_rounds"], stream, "independent")
        teacher_gate = gates["teacher_gap"]
        teacher = _contraction(
            [row["teacher_after_fit"]["rms"] for row in early],
            [row["teacher_after_fit"]["rms"] for row in late],
            [row["teacher_after_fit"]["rms"] for row in independent],
            teacher_gate["minimum_relative_contraction"],
            teacher_gate["maximum_late_ratio_to_independent"],
        )
        student = {}
        for anchor in gates["student_gap"]["anchors"]:
            key = str(anchor)
            all_windows = [
                *_window(history, gates["early_rounds"], stream, "independent"),
                *early,
                *late,
                *independent,
            ]
            qualified = all(
                row["queries"][key]["present"] and row["queries"][key]["qualified"]
                for row in all_windows
            )
            student[key] = (
                _contraction(
                    [row["queries"][key]["action_gap"] for row in early],
                    [row["queries"][key]["action_gap"] for row in late],
                    [row["queries"][key]["action_gap"] for row in independent],
                    gates["student_gap"]["minimum_relative_contraction"],
                    gates["student_gap"]["maximum_late_ratio_to_independent"],
                )
                if qualified
                else dict(
                    passed=False,
                    resolved=False,
                    reason="Fixed query missing or not walking-qualified",
                )
            )
        original = history[0]["streams"][str(stream)]["coupled"]["bank"]
        original_cost = float(np.mean([item["outcome"]["complete_cost"] for item in original]))
        late_cost = float(
            np.mean([item["outcome"]["complete_cost"] for row in late for item in row["bank"]])
        )
        retained = all(
            item["outcome"]["walking"] and item["outcome"]["full_survival"]
            for row in history
            for arm in ("independent", "coupled")
            for item in row["streams"][str(stream)][arm]["bank"]
        )
        raw_walk = [
            sum(row["raw_candidates"][slot]["result"]["outcome"]["walking"] for row in late)
            for slot in range(len(original))
        ]
        competent = (
            retained
            and late_cost
            <= gates["teacher_competence"]["maximum_late_aggregate_cost_ratio_to_initial"]
            * original_cost
            and all(
                count
                >= gates["teacher_competence"][
                    "raw_best_minimum_walking_rounds_per_path_in_late_window"
                ]
                for count in raw_walk
            )
        )
        final = history[-1]
        primary = walking_contrast(
            final["streams"][str(stream)]["coupled"]["student"]["summary"],
            final["streams"][str(stream)]["independent"]["student"]["summary"],
            final["parents"][str(stream)]["summary"],
            gates["walking"]["minimum_additional_successes"],
        )
        confirm = walking_contrast(
            confirmation[(stream, "coupled")]["summary"],
            confirmation[(stream, "independent")]["summary"],
            confirmation[(stream, "parent")]["summary"],
            gates["walking"]["minimum_additional_successes"],
        )
        all_nonregression &= primary["no_regression"] and confirm["no_regression"] and retained
        passed = (
            teacher["passed"]
            and all(row["passed"] for row in student.values())
            and competent
            and primary["passed"]
            and confirm["passed"]
        )
        if passed:
            joint.append(stream)
        per_stream[str(stream)] = dict(
            teacher_gap=teacher,
            student_gap=student,
            teacher_competence=dict(
                passed=competent,
                retained_nonregression=retained,
                late_cost_ratio=late_cost / original_cost,
                raw_walking_rounds_per_path=raw_walk,
            ),
            primary=primary,
            confirmation=confirm,
            joint_passed=passed,
            individual_competence=(
                final["streams"][str(stream)]["coupled"]["student"]["summary"]["walking"] >= 15
                and confirmation[(stream, "coupled")]["summary"]["walking"] >= 15
            ),
        )
    confirmed = len(joint) >= gates["minimum_joint_streams"] and all_nonregression
    walking_streams = [
        s
        for s, row in per_stream.items()
        if row["primary"]["passed"]
        and row["confirmation"]["passed"]
        and row["teacher_competence"]["passed"]
    ]
    gap_streams = [
        s
        for s, row in per_stream.items()
        if row["teacher_gap"]["passed"]
        and all(item["passed"] for item in row["student_gap"].values())
        and row["teacher_competence"]["passed"]
    ]
    unresolved = any(
        not row["teacher_gap"]["resolved"]
        or any(not item["resolved"] for item in row["student_gap"].values())
        for row in per_stream.values()
    )
    verdict = (
        "HELPED_FINITE_ALTERNATING_LOOP"
        if confirmed
        else "PARTIAL_WALKING_GAIN_WITH_UNRESOLVED_GAP"
        if len(walking_streams) >= 2 and all_nonregression and unresolved
        else "PARTIAL_WALKING_GAIN_GAP_NOT_CLOSED"
        if len(walking_streams) >= 2 and all_nonregression
        else "GAP_ONLY_IMPROVEMENT"
        if len(gap_streams) >= 2 and all_nonregression
        else "NO_CONFIRMED_HELP"
    )
    return dict(
        verdict=verdict,
        passed=confirmed,
        joint_streams=joint,
        per_stream=per_stream,
        all_stream_nonregression=all_nonregression,
        full_project_success=False,
        broader_requirements_not_tested=gates["broader_requirements_not_tested"],
    )


def _worker_environment() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"


def run_loop(launch_path: str | Path) -> dict:
    origin = time.monotonic(), time.process_time()
    IO_COUNTS.clear()
    config, launch, prepared, preflight = validate_launch(Path(launch_path))
    out = Path(launch["output_directory"])
    _need(not out.exists(), "Study output is create-only; paid incomplete work is never retried")
    # Imports follow all source/input/preflight/capacity checks.
    import importlib.metadata
    from multiprocessing import get_context

    import torch

    versions = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=torch.__version__,
        **{name: importlib.metadata.version(name) for name in ("scipy", "mujoco", "h5py")},
    )
    _need(
        all(versions[key] == config["resources"]["runtime"][key] for key in versions),
        "Actual study runtime differs from the frozen CPU environment",
    )
    _worker_environment()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _need(not torch.cuda.is_initialized(), "The frozen experiment permits no GPU execution")
    out.mkdir(parents=True)
    write(out / "launch.json", launch)
    write(out / "config.json", config)
    ledger = StudyWork(out, config, preflight, origin=origin)
    pool, result = (
        None,
        dict(
            schema=SCHEMA,
            status="RUNNING",
            full_project_success=False,
            launch=binding(Path(launch_path)),
            **{
                name + "_sha256": launch[name]["sha256"]
                for name in ("config", "source_manifest", "prepared_manifest", "preflight")
            },
        ),
    )
    try:
        pool = get_context("spawn").Pool(
            config["resources"]["native_workers"], initializer=_worker_environment
        )
        ledger.pool = pool
        operations = Operations(config, launch, prepared, ledger, pool)
        history, final = execute_schedule(config, operations)
        final_serialized = {
            str(stream): {arm: final[(stream, arm)] for arm in ("independent", "coupled", "parent")}
            for stream in config["streams"]
        }
        write(out / "final_panels.json", final_serialized)
        result.update(
            status="COMPLETED_20_ROUNDS",
            history=history,
            confirmation=final_serialized,
            criteria=criteria(history, final, config),
        )
    except BaseException as error:
        result.update(status="FAILED_STOPPED", error=repr(error))
        if pool is not None:
            pool.terminate()
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        try:
            ledger.unknown_pending()
        except BaseException as error:
            result.update(status="FAILED_STOPPED", accounting_recovery_error=repr(error))
        try:
            ledger.check(final=True)
        except BaseException as error:
            result.update(status="FAILED_STOPPED", closure_error=repr(error))
        result["accounting"] = ledger.snapshot()
        if "history" not in result:
            result["history"] = [
                _json(path) for path in sorted((out / "rounds").glob("round_*/complete.json"))
            ]
            result["incomplete_rounds"] = [
                str(path)
                for path in sorted((out / "rounds").glob("round_*"))
                if not (path / "complete.json").exists()
            ]
        try:
            ledger.work.close(
                "COMPLETED" if result["status"] == "COMPLETED_20_ROUNDS" else "FAILED"
            )
        except BaseException as error:
            result.update(status="FAILED_STOPPED", receipt_error=repr(error))
            if not ledger.work.closed:
                ledger.work.close("FAILED")
        write(out / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args(argv)
    return int(run_loop(args.launch)["status"] != "COMPLETED_20_ROUNDS")


if __name__ == "__main__":
    raise SystemExit(main())
