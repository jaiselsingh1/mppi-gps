"""Owned local J2 processes; source and admission receipts precede execution."""

from pathlib import Path
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
REPO = BASE.parents[1]
SOURCE = BASE / "source_002"
RUN = REPO / "runs/2026-11-03_jitter_diagnosis_j2_001"
COMMIT = "edda434a2771194163aa5be883b83f4f94d3cd56"
PYTHON = REPO / ".venv/bin/python"
CONFIG = SOURCE / "configs/studies/jitter_diagnosis_j2_0.json"


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def retained_bytes():
    return sum(p.stat().st_size for root in (BASE, RUN) if root.is_dir()
               for p in root.rglob("*") if p.is_file())


def signal_owned(process, value):
    try:
        os.killpg(process.pid, value)
    except ProcessLookupError:
        pass


phase = sys.argv[1]
assert phase in {"worker", "audit", "check"}
ci = json.loads((BASE / "prelaunch_ci_001/status.json").read_text())
assert ci["conclusion"] == "success" and ci["headSha"] == COMMIT
source_manifest = json.loads((SOURCE / "source_manifest.json").read_text())
assert source_manifest["git_commit"] == COMMIT
assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == source_manifest["config_sha256"]
if phase == "check":
    assert not RUN.exists()
    print(json.dumps(dict(status="PASSED_LAUNCHER_IMPORT_AND_ADMISSION", source_commit=COMMIT,
                         worker_root_exists=False, new_scientific_work=0)))
    raise SystemExit(0)
out = BASE / ("operator_worker_001" if phase == "worker" else "operator_audit_001")
out.mkdir(exist_ok=False)
if phase == "worker":
    assert not RUN.exists()
    command = [str(PYTHON), "-B", "-m", "src.gps.mppi_gps.jitter_diagnosis",
               "run", "--config", str(CONFIG), "--prepared", str(BASE / "prepared_001"),
               "--out", str(RUN)]
    hard_seconds, soft_seconds = 3600, 3540
else:
    worker_exit = json.loads((BASE / "operator_worker_001/exit.json").read_text())
    assert worker_exit["exit_code"] == 0
    command = [str(PYTHON), "-B", "-m", "src.gps.mppi_gps.audits.jitter_diagnosis",
               "--run", str(RUN), "--out", str(BASE / "audit_001")]
    hard_seconds, soft_seconds = 900, 840
environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
environment.pop("PYTHONPATH", None)
started = time.monotonic()
before = resource.getrusage(resource.RUSAGE_CHILDREN)
self_before = resource.getrusage(resource.RUSAGE_SELF)
interruption = None
observed_peak = retained_bytes()
with (out / "stdout.txt").open("xb") as stdout, (out / "stderr.txt").open("xb") as stderr:
    process = subprocess.Popen(command, cwd=SOURCE, env=environment, stdout=stdout,
                               stderr=stderr, start_new_session=True)
    save(out / "launch.json", dict(command=command, cwd=str(SOURCE), pid=process.pid,
         process_group=process.pid, operator_pid=os.getpid(), source_commit=COMMIT,
         soft_seconds=soft_seconds, hard_seconds=hard_seconds,
         cleanup_seconds_reserved=hard_seconds-soft_seconds, phase=phase))
    while process.poll() is None:
        elapsed = time.monotonic() - started
        log_bytes = sum((out / name).stat().st_size for name in ("stdout.txt", "stderr.txt"))
        storage = retained_bytes()
        observed_peak = max(observed_peak, storage)
        reason = ("wall_allocation" if elapsed >= soft_seconds else
                  "study_storage" if storage > 4*1024**3 else
                  "operator_log_storage" if log_bytes > 16*1024**2 else None)
        if reason and interruption is None:
            interruption = dict(reason=reason, elapsed_seconds=elapsed, signal="SIGINT")
            signal_owned(process, signal.SIGINT)
        if elapsed >= hard_seconds:
            signal_owned(process, signal.SIGKILL)
            interruption = dict(reason="hard_wall_limit", elapsed_seconds=elapsed,
                                signal="SIGKILL", prior=interruption)
        time.sleep(1)
    exit_code = process.wait()
after = resource.getrusage(resource.RUSAGE_CHILDREN)
self_after = resource.getrusage(resource.RUSAGE_SELF)
save(out / "exit.json", dict(exit_code=exit_code, phase=phase,
     wall_seconds=time.monotonic()-started, child_user_seconds=after.ru_utime-before.ru_utime,
     child_system_seconds=after.ru_stime-before.ru_stime, child_max_rss_native=after.ru_maxrss,
     child_max_rss_unit="bytes on declared Darwin host; inherited-peak semantics retained",
     interruption=interruption, retained_study_bytes=retained_bytes(),
     observed_retained_peak_bytes=observed_peak, storage_peak_scope="one-second samples, not filesystem peak",
     operator_user_seconds=self_after.ru_utime-self_before.ru_utime,
     operator_system_seconds=self_after.ru_stime-self_before.ru_stime,
     operator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(json.dumps(json.loads((out / "exit.json").read_text()), indent=2))
raise SystemExit(exit_code)
