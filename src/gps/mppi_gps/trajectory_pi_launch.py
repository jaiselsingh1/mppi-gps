"""Create-only external supervision of the frozen P1 lifecycle on the CPU host.

This process never simulates or fits. Each scientific subprocess receives its
own process group; deadlines and aggregate resource guards apply from outside.
An externally killed prefix remains incomplete, with no fabricated paid work.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .provenance import sha, write

GIB = 1024**3


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def tree_usage(pid: int, proc: Path = Path("/proc")) -> dict:
    """Own/live-child CPU plus their reaped descendants, counted once.

    cutime/cstime belong to children already waited by that process, whereas
    recursively visited child processes have not yet been waited by it. Reading
    only task/*/children avoids inspecting unrelated processes or commands.
    """
    ticks, page = os.sysconf("SC_CLK_TCK"), os.sysconf("SC_PAGE_SIZE")
    seen, pending, cpu, rss = set(), [pid], 0.0, 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        directory = proc / str(current)
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            cpu += sum(int(fields[index]) for index in (11, 12, 13, 14)) / ticks
            rss += int((directory / "statm").read_text().split()[1]) * page
            for children in directory.glob("task/*/children"):
                pending.extend(map(int, children.read_text().split()))
        except FileNotFoundError:
            continue
    return dict(cpu_seconds=cpu, rss_bytes=rss, observed_pids=sorted(seen))


def _idle() -> tuple[int, int]:
    counters = list(map(int, Path("/proc/stat").read_text().splitlines()[0].split()[1:]))
    # guest/guest_nice are already included in user/nice; do not count twice.
    return sum(counters[:8]), counters[3] + counters[4]


def observe_capacity(base: Path, versions: bool = False) -> dict:
    if not Path("/proc").is_dir():
        raise RuntimeError("Frozen supervision requires Linux /proc accounting")
    runtime = {}
    if versions:
        import torch

        if torch.cuda.is_initialized():
            raise RuntimeError("CPU supervision must not initialize CUDA")
        runtime = dict(
            python=platform.python_version(),
            torch=torch.__version__,
            **{
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "mujoco", "h5py")
            },
        )
    before = _idle()
    time.sleep(0.25)
    after = _idle()
    cores = len(os.sched_getaffinity(0))
    available = cores * (after[1] - before[1]) / max(1, after[0] - before[0])
    memory = {
        line.split(":")[0]: line.split(":")[1].strip().split()
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    result = dict(
        schema="trajectory-pi-resource-observation-1",
        observed_unix=time.time(),
        available_logical_cpus=available,
        allowed_logical_cpus=cores,
        idle_observation_seconds=0.25,
        available_memory_bytes=int(memory["MemAvailable"][0]) * 1024,
        free_disk_bytes=shutil.disk_usage(base).free,
    )
    if versions:
        result["versions"] = runtime
    return result


def capacity_passed(observed: dict, config: dict, *, main: bool) -> bool:
    return bool(
        observed.get("available_logical_cpus", 0) >= (28 if main else 8)
        and observed.get("available_memory_bytes", 0) >= (32 if main else 8) * GIB
        and observed.get("free_disk_bytes", 0) >= 80 * GIB
        and (
            not main
            or all(
                observed.get("versions", {}).get(key) == config["resources"]["runtime"][key]
                for key in ("python", "numpy", "scipy", "mujoco", "torch", "h5py")
            )
        )
    )


def _kill_group(process: subprocess.Popen) -> None:
    for sent in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sent)
        except ProcessLookupError:
            pass
        if sent == signal.SIGTERM:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    process.wait()


def run_phase(
    phase: str,
    argv: list[str],
    base: Path,
    source: Path,
    timeout: float,
    guard: Callable[[], dict],
    *,
    poll_seconds: float = 2,
) -> dict:
    """One attempt with raw stdout/stderr and a durable external receipt."""
    directory = base / "supervisor"
    started, own_cpu = time.monotonic(), time.process_time()
    waited = resource.getrusage(resource.RUSAGE_CHILDREN)
    waited_before = waited.ru_utime + waited.ru_stime
    sample = getattr(guard, "sample", None)
    initial = final = sample() if sample is not None else dict(cpu_seconds=0.0, rss_bytes=0)
    receipt = dict(
        schema="trajectory-pi-phase-1",
        phase=phase,
        argv=argv,
        pid=None,
        exit_code=None,
        limit_error=None,
    )
    process = None
    with (
        (directory / f"{phase}.stdout.log").open("xb") as stdout,
        (directory / f"{phase}.stderr.log").open("xb") as stderr,
    ):
        try:
            initial = final = guard()
            if timeout <= 0:
                raise TimeoutError("No total wall budget remains for " + phase)
            temporary = directory / "tmp"
            temporary.mkdir(exist_ok=True)
            environment = dict(os.environ)
            environment.update(
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                MPLBACKEND="Agg",
                MPLCONFIGDIR=str(directory / "mpl"),
                TMPDIR=str(temporary),
                TMP=str(temporary),
                TEMP=str(temporary),
            )
            process = subprocess.Popen(
                argv,
                cwd=source,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                env=environment,
            )
            receipt["pid"] = process.pid
            while process.poll() is None:
                final = guard()
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f"{phase} external wall limit {timeout:g}s")
                time.sleep(min(poll_seconds, max(0.001, timeout - (time.monotonic() - started))))
            receipt["exit_code"] = process.wait()
            # A dead coordinator must not leave its still-running native pool
            # behind. The session/group belongs exclusively to this attempt.
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                receipt["remaining_group_terminated"] = False
            else:
                receipt["remaining_group_terminated"] = True
                _kill_group(process)
            final = guard()
        except BaseException as error:
            receipt["limit_error"] = repr(error)
            if process is not None:
                _kill_group(process)
                receipt["exit_code"] = process.returncode
        finally:
            if sample is not None:
                final = sample()
            waited = resource.getrusage(resource.RUSAGE_CHILDREN)
            receipt.update(
                wall_seconds=time.monotonic() - started,
                process_tree_cpu_seconds=max(0.0, final["cpu_seconds"] - initial["cpu_seconds"]),
                supervisor_cpu_seconds=time.process_time() - own_cpu,
                waited_children_cpu_seconds=waited.ru_utime + waited.ru_stime - waited_before,
                cpu_scope="Own process tree and waited descendants; RUSAGE is a cross-check, not added twice. Externally orphaned/killed descendants may have unknown final increments.",
                peak_sampled_tree_rss_bytes=final.get("peak_rss_bytes", final.get("rss_bytes")),
                stdout=str(directory / f"{phase}.stdout.log"),
                stderr=str(directory / f"{phase}.stderr.log"),
            )
            write(directory / f"{phase}.json", receipt)
    return receipt


class Guard:
    def __init__(self, base: Path, config: dict, *, antecedents: list[dict] | None = None):
        self.base, self.config = base, config
        self.started, self.own_cpu = time.monotonic(), time.process_time()
        self.antecedents = antecedents or []
        self.started -= sum(row["wall_seconds"] for row in self.antecedents)
        self.antecedent_cpu = sum(row["cpu_seconds"] for row in self.antecedents)
        self.initial_cpu = tree_usage(os.getpid())["cpu_seconds"]
        self.peak = 0
        self.maximum_cpu = 0.0
        self.cpu_ceiling = config["limits"]["cpu_seconds"]
        self.output_ceiling = config["limits"]["output_bytes"]

    def sample(self) -> dict:
        usage = tree_usage(os.getpid())
        self.maximum_cpu = max(self.maximum_cpu, usage["cpu_seconds"] - self.initial_cpu)
        usage["cpu_seconds"] = self.maximum_cpu + self.antecedent_cpu
        usage["antecedents"] = self.antecedents
        self.peak = max(self.peak, usage["rss_bytes"])
        usage["peak_rss_bytes"] = self.peak
        return usage

    def __call__(self) -> dict:
        usage = self.sample()
        limits = self.config["limits"]
        checks = (
            (time.monotonic() - self.started <= limits["pool_wall_seconds"], "total wall"),
            (usage["cpu_seconds"] <= self.cpu_ceiling, "total CPU/closure reservation"),
            (
                usage["rss_bytes"] <= self.config["resources"]["aggregate_ram_bytes"],
                "aggregate RAM",
            ),
            (
                sum(path.stat().st_size for path in self.base.rglob("*") if path.is_file())
                + 1024**2
                <= self.output_ceiling,
                "total output/closure reservation",
            ),
        )
        for passed, name in checks:
            if not passed:
                raise RuntimeError(name + " budget exhausted")
        return usage


def antecedent_usage(prepared: Path) -> list[dict]:
    """Import closed active-phase resource receipts; never charge nested times twice."""
    manifest = read(prepared / "manifest.json", {})
    if manifest.get("input_mode") != "worker-native":
        return []
    from .trajectory_pi_replay_validation import bound

    proof = read(bound(manifest["validation_manifest"]))
    rows = []
    for phase, seal in (
        ("r0", proof["origin_replay"]["external_summary"]),
        ("rv", proof["external_summary"]),
    ):
        value = read(bound(seal))
        cpu = (
            sum(value[key]["process_tree_cpu_seconds"] for key in ("replay", "audit"))
            if phase == "r0"
            else value["cpu_seconds"]
        )
        wall = value["wall_seconds"]
        if not (0 <= cpu < float("inf") and 0 <= wall <= 600 and value["no_retries"] is True):
            raise ValueError("Closed finite antecedent resource receipt required")
        rows.append(
            dict(
                phase=phase,
                receipt=seal,
                wall_seconds=wall,
                cpu_seconds=cpu,
                scope="Paid active phases; source preparation/transfer engineering is separate",
            )
        )
    return rows


def supervise(
    base: Path, *, config_relative: str = "configs/studies/trajectory_pi_p1.json"
) -> dict:
    base = base.resolve()
    source, prepared = base / "source", base / "prepared"
    relative = Path(config_relative)
    config_path = (source / relative).resolve()
    if (
        relative.is_absolute()
        or str(relative) != config_relative
        or ".." in relative.parts
        or not config_path.is_relative_to(source)
    ):
        raise ValueError("Contained normalized source-relative configuration required")
    source_manifest = read(base / "source_manifest.json", {})
    declared = {row["path"]: row for row in source_manifest.get("files", [])}
    if config_relative not in declared or {
        key: binding(config_path)[key] for key in ("bytes", "sha256")
    } != {key: declared[config_relative][key] for key in ("bytes", "sha256")}:
        raise ValueError("Scientific configuration absent from frozen source roster")
    config = read(config_path)
    if config["schema"] != "trajectory-pi-p1-1" or config["launch_ready"] is not True:
        raise ValueError("Committed launch-ready configuration required")
    if (
        not source.is_dir()
        or not (prepared / "manifest.json").is_file()
        or not (base / "source_manifest.json").is_file()
    ):
        raise ValueError("Transferred frozen source and prepared manifest required")
    (base / "supervisor").mkdir(exist_ok=False)
    for name in (
        "preflight",
        "worker",
        "audit",
        "report",
        "launch.json",
        "resource_observation.json",
        "supervisor.json",
    ):
        if (base / name).exists():
            raise FileExistsError("No supervision retries: " + str(base / name))
    guard, phases = Guard(base, config, antecedents=antecedent_usage(prepared)), []
    seals = {
        key + "_sha256": sha(path)
        for key, path in dict(
            config=config_path,
            source_manifest=base / "source_manifest.json",
            prepared_manifest=prepared / "manifest.json",
        ).items()
    }
    command = [sys.executable, "scripts/train_mppi_gps.py", "study", "trajectory-pi"]
    started_study, error, stage = False, None, "initial_capacity"

    def journal(event: str, **fields: Any) -> None:
        with (base / "supervisor/journal.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(event=event, elapsed_seconds=time.monotonic() - guard.started, **fields),
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def phase(name: str, args: list[str], seconds: int) -> dict:
        remaining = config["limits"]["pool_wall_seconds"] - (time.monotonic() - guard.started)
        reserved_closure = config["limits"]["audit_wall_seconds"] + 300 if name == "study" else 0
        guard.cpu_ceiling = config["limits"]["cpu_seconds"] - reserved_closure
        guard.output_ceiling = config["limits"][
            "execution_output_bytes" if name in {"preflight", "study"} else "output_bytes"
        ]
        remaining -= reserved_closure
        journal(
            "phase_start",
            phase=name,
            reserved_closure_wall_seconds=reserved_closure,
            reserved_closure_cpu_seconds=reserved_closure,
            external_wall_seconds=min(seconds, remaining),
        )
        value = run_phase(name, command + args, base, source, min(seconds, remaining), guard)
        phases.append(value)
        journal("phase_exit", phase=name, receipt=binding(base / "supervisor" / f"{name}.json"))
        return value

    try:
        initial = observe_capacity(base)
        write(base / "supervisor/initial_capacity.json", initial)
        if not capacity_passed(initial, config, main=False):
            raise RuntimeError("P0 capacity gate failed; no physics attempted")
        stage = "preflight"
        receipt = phase(
            "preflight",
            [
                "preflight",
                "--config",
                str(config_path),
                "--prepared",
                str(prepared),
                "--source-manifest",
                str(base / "source_manifest.json"),
                "--out",
                str(base / "preflight"),
            ],
            600,
        )
        preflight = read(base / "preflight/preflight.json", {})
        if receipt["exit_code"] != 0 or receipt["limit_error"] or preflight.get("status") != "PASS":
            raise RuntimeError("P0 failed or lacked a closed PASS; no main attempt")
        stage = "main_capacity"
        observation = observe_capacity(base, versions=True)
        observation.update(passed=capacity_passed(observation, config, main=True), **seals)
        write(base / "resource_observation.json", observation)
        if not observation["passed"]:
            raise RuntimeError("Fresh main capacity/runtime gate failed")
        stage = "study"
        launch = _launch(base, source, prepared, config_path)
        write(base / "launch.json", launch)
        started_study = True
        receipt = phase(
            "study",
            ["run", "--launch", str(base / "launch.json")],
            config["limits"]["execution_wall_seconds"],
        )
        if (
            receipt["exit_code"] != 0
            or receipt["limit_error"]
            or read(base / "worker/result.json", {}).get("status") != "COMPLETED_20_ROUNDS"
        ):
            raise RuntimeError("Study stopped; preserved output will be audited without retry")
    except BaseException as failure:
        error = repr(failure)
        journal("scientific_work_stopped", stage=stage, error=error)
    _close_stop(base, source, prepared, config_path, seals, stage, error, started_study, phases)
    stopped = read(base / "worker/result.json").get("status") != "COMPLETED_20_ROUNDS"
    closure_errors = []
    for name, args, limit in (
        (
            "audit",
            [
                "audit",
                "--run",
                str(base / "worker"),
                "--out",
                str(base / "audit"),
                *(["--stopped"] if stopped else []),
            ],
            config["limits"]["audit_wall_seconds"],
        ),
        ("report", ["report", "--run", str(base / "worker"), "--out", str(base / "report")], 300),
    ):
        try:
            value = phase(name, args, limit)
            if value["exit_code"] != 0 or value["limit_error"]:
                closure_errors.append(dict(phase=name, receipt=value))
        except BaseException as failure:
            closure_errors.append(dict(phase=name, error=repr(failure)))
            journal("closure_stopped", phase=name, error=repr(failure))
    result = dict(
        schema="trajectory-pi-supervisor-1",
        status="CLOSED" if not error and not closure_errors else "STOPPED_PRESERVED",
        error=error,
        closure_errors=closure_errors,
        phases=phases,
        accounting=guard.sample(),
        no_retries=True,
        engineering=dict(
            supervisor_cpu_seconds=time.process_time() - guard.own_cpu,
            wall_seconds=time.monotonic() - guard.started,
        ),
        transfer=dict(
            status="PREEXISTING_TRANSFER",
            measured_by_supervisor=False,
            receipt=read(base / "transfer_receipt.json"),
        ),
        science_repeated=0,
    )
    write(base / "supervisor.json", result)
    return result


def _launch(base: Path, source: Path, prepared: Path, config: Path) -> dict:
    return dict(
        schema="trajectory-pi-p1-launch-1",
        source_directory=str(source),
        prepared_directory=str(prepared),
        output_directory=str(base / "worker"),
        **{
            name: binding(path)
            for name, path in dict(
                config=config,
                source_manifest=base / "source_manifest.json",
                prepared_manifest=prepared / "manifest.json",
                preflight=base / "preflight/preflight.json",
                resource_observation=base / "resource_observation.json",
            ).items()
        },
    )


def _close_stop(
    base: Path,
    source: Path,
    prepared: Path,
    config: Path,
    seals: dict,
    stage: str,
    error: str | None,
    started: bool,
    phases: list[dict],
) -> None:
    stop = dict(
        schema="trajectory-pi-external-stop-1",
        stage=stage,
        reason=error,
        phase_receipt=phases[-1] if phases else None,
        accounting_scope="External receipts and preserved raw prefix; missing internal closure remains unknown",
        new_research_calls_after_stop=0,
    )
    if not (base / "preflight/preflight.json").is_file():
        write(
            base / "preflight/preflight.json",
            dict(
                schema="trajectory-pi-preflight-1",
                status="EXTERNAL_STOPPED" if phases else "NOT_ATTEMPTED",
                checks=[],
                work={},
                external_stop=stop,
                accounting_complete=False,
                **seals,
            ),
        )
    stop["preflight_status"] = read(base / "preflight/preflight.json")["status"]
    if not (base / "resource_observation.json").is_file():
        write(
            base / "resource_observation.json",
            dict(
                schema="trajectory-pi-resource-observation-1",
                passed=False,
                observed_unix=time.time(),
                external_stop=stop,
                **seals,
            ),
        )
    if not (base / "launch.json").is_file():
        write(base / "launch.json", _launch(base, source, prepared, config))
    worker = base / "worker"
    worker.mkdir(exist_ok=True)
    for name, value in (("launch.json", read(base / "launch.json")), ("config.json", read(config))):
        if not (worker / name).is_file():
            write(worker / name, value)
    if not (worker / "result.json").is_file():
        history = (
            [read(path) for path in sorted((worker / "rounds").glob("round_*/complete.json"))]
            if started
            else []
        )
        write(
            worker / "result.json",
            dict(
                schema="trajectory-pi-p1-result-1",
                status="FAILED_STOPPED" if started else "PRELAUNCH_STOPPED",
                error=error,
                history=history,
                full_project_success=False,
                external_stop=stop,
            ),
        )
    if error:
        write(worker / "external_supervisor.json", stop)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config-relative", default="configs/studies/trajectory_pi_p1.json")
    args = parser.parse_args(argv)
    return int(supervise(args.root, config_relative=args.config_relative)["status"] != "CLOSED")


if __name__ == "__main__":
    raise SystemExit(main())
