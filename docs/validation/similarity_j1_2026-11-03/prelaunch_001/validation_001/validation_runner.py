"""Create-only engineering validation; owns and bounds each child process group."""

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
OUT = Path(__file__).resolve().parent
LIMIT = 512 * 1024**2


def save(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_roster():
    paths = set()
    for base in ("src", "scripts", "tests"):
        paths.update((ROOT / base).rglob("*.py"))
    paths.update(ROOT / name for name in ("pyproject.toml", "uv.lock", "configs/studies/similarity_j1_0.json", "docs/similarity_j1_protocol.md"))
    return {str(p.relative_to(ROOT)): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths) if p.is_file()}


def output_bytes():
    return sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())


def run_stage(name, command, timeout):
    started = time.monotonic()
    environment = os.environ.copy()
    delta = dict(CUDA_VISIBLE_DEVICES="", MUJOCO_GL="disable", MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(OUT / "mpl"), XDG_CACHE_HOME=str(OUT / "cache"))
    environment.update(delta)
    print(json.dumps(dict(event="validation_stage_start", stage=name, command=command)), flush=True)
    killed = False
    with (OUT / (name + ".stdout.txt")).open("xb") as stdout, (OUT / (name + ".stderr.txt")).open("xb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, start_new_session=True)
        stopping = None
        while True:
            pid, status, usage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                break
            elapsed = time.monotonic() - started
            if stopping is None and (elapsed > timeout or output_bytes() > LIMIT):
                stopping = "wall_limit" if elapsed > timeout else "output_limit"
                os.killpg(child.pid, signal.SIGTERM)
                terminate_at = time.monotonic()
            if stopping is not None and not killed and time.monotonic() - terminate_at > 5:
                os.killpg(child.pid, signal.SIGKILL)
                killed = True
            time.sleep(0.1)
    receipt = dict(schema="walker-j1-engineering-stage-v1", stage=name, command=command, environment_overrides=delta, pid=child.pid, parent_pid=os.getpid(), exit_code=child.returncode, child_reaped=True, killed=killed, stopping_reason=stopping, wall_seconds=time.monotonic()-started, wall_limit_seconds=timeout, user_seconds=usage.ru_utime, system_seconds=usage.ru_stime, peak_rss_bytes=int(usage.ru_maxrss if sys.platform == "darwin" else usage.ru_maxrss*1024), minor_page_faults=usage.ru_minflt, major_page_faults=usage.ru_majflt, block_input_operations=usage.ru_inblock, block_output_operations=usage.ru_oublock, voluntary_context_switches=usage.ru_nvcsw, involuntary_context_switches=usage.ru_nivcsw, output_bytes_after_stage=output_bytes(), new_research_experiments=0, engineering_model_and_physics_calls="not individually instrumented; existing tests only")
    save(name + ".receipt.json", receipt)
    print(json.dumps(dict(event="validation_stage_end", stage=name, exit_code=child.returncode, wall_seconds=receipt["wall_seconds"])), flush=True)
    return receipt


before = source_roster()
save("source_before.json", before)
save("runtime.json", dict(python=sys.version, executable=sys.executable, platform=platform.platform(), pid=os.getpid(), new_research_experiments=0, clean_export_pending_source_commit=True))
stages = [
    ("pytest", [sys.executable, "-m", "pytest", "-q"], 1800),
    ("ruff", [sys.executable, "-m", "ruff", "check", "."], 300),
    ("format", [sys.executable, "-m", "ruff", "format", "--check", "."], 300),
    ("repository", [sys.executable, "scripts/check_repository.py"], 300),
]
receipts = [run_stage(*stage) for stage in stages]
after = source_roster()
save("source_after.json", after)
save("summary.json", dict(schema="walker-j1-engineering-validation-v1", all_commands_passed=all(r["exit_code"] == 0 and r["stopping_reason"] is None for r in receipts), source_changes_during_validation=[name for name in sorted(set(before)|set(after)) if before.get(name) != after.get(name)], stages=receipts, new_research_experiments=0, clean_export_pending_source_commit=True, output_ceiling_bytes=LIMIT, output_bytes=output_bytes()))
