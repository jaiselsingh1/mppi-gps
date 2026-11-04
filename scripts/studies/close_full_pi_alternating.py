"""Closed-evidence audit/report orchestration; never launches scientific workers."""

import datetime
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[2]
art = root / "artifacts/full_pi_alternating_2026-11-03"
run = root / "runs/2026-11-03_full_pi_alternating_bounded_001"
py = "/Users/jaisel/Documents/mppi-gps/.venv/bin/python"
coordinator = json.loads((run / "coordinator_result.json").read_text())
assert (run / "seal_receipt.json").exists(), "Wait for closed raw seal"
assert coordinator["complete"] and not coordinator["stopped"], (
    "Do not reinterpret a stopped study as complete"
)
env = os.environ.copy()
env.update(
    {
        k: "1"
        for k in (
            "PYTHONDONTWRITEBYTECODE",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
)
env["MPLCONFIGDIR"] = str(art / "matplotlib_cache")
steps = [
    (
        "audit_001",
        run / "source",
        "full_pi_alternating_audit",
        ["--run", str(run), "--out", str(art / "audit_001")],
    ),
    (
        "join_audit_001",
        root,
        "full_pi_alternating_join_audit",
        [
            "--run",
            str(run),
            "--out",
            str(art / "join_audit_001"),
            "--prior",
            str(art / "audit_001/audit.json"),
        ],
    ),
    (
        "report_001",
        run / "source",
        "full_pi_alternating_report",
        [
            "--run",
            str(run),
            "--out",
            str(art / "report_001"),
            "--audit",
            str(art / "audit_001/audit.json"),
        ],
    ),
    (
        "trace_report_001",
        root,
        "full_pi_alternating_trace_report",
        ["--run", str(run), "--out", str(art / "trace_report_001")],
    ),
]
for label, cwd, module, args in steps:
    source = cwd / ("src/gps/mppi_gps/" + module + ".py")
    command = [py, "-u", "-m", "src.gps.mppi_gps." + module, *args]
    receipt = art / (label + "_process_receipt.json")
    assert not receipt.exists() and not (art / label).exists(), "Create-only, no silent retry"
    begin = time.monotonic()
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (
        (art / (label + "_stdout.txt")).open("x") as stdout,
        (art / (label + "_stderr.txt")).open("x") as stderr,
    ):
        child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
        try:
            code = child.wait(timeout=7200)
        except subprocess.TimeoutExpired:
            child.terminate()
            code = child.wait(timeout=30)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(
        label=label,
        command=command,
        cwd=str(cwd),
        started_utc=utc,
        wall_seconds=time.monotonic() - begin,
        child_user_seconds=after.ru_utime - before.ru_utime,
        child_system_seconds=after.ru_stime - before.ru_stime,
        maxrss_cumulative_children_bytes=after.ru_maxrss,
        exit_code=code,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        new_training_by_design=0,
    )
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if code:
        print("Preserved failed output; no automatic repair or retry", flush=True)
        break
