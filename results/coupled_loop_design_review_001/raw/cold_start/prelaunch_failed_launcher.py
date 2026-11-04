"""One metered saved-only analysis retry; retain its resource receipt on failure."""

import json
import resource
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    with path.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def main():
    root = Path(__file__).resolve().parent
    resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    started, cpu_started = time.monotonic(), time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    status, error, returncode = "FAILED_ANALYSIS_ONLY", None, None
    try:
        with (root / "rerun.stdout.log").open("x") as stdout, (root / "rerun.stderr.log").open("x") as stderr:
            child = subprocess.run([
                sys.executable, str(root / "review.py"), "--config", str(root.parent / "config.json"),
                "--cpu-cap", "25",
            ], stdout=stdout, stderr=stderr, check=False)
        returncode = child.returncode
        if returncode == 0:
            status = "COMPLETED_SAVED_ONLY_REVIEW"
    except BaseException as exception:
        error = repr(exception)
    finally:
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
        parent_cpu = time.process_time() - cpu_started
        write(root / "rerun_resource_receipt.json", dict(
            schema="saved-review-resource-receipt-1", status=status, returncode=returncode, error=error,
            child_process_cpu_seconds=child_cpu, launcher_process_cpu_seconds=parent_cpu,
            total_measured_cpu_seconds=child_cpu+parent_cpu,
            wall_seconds=time.monotonic()-started,
            child_cpu_hard_cap_seconds=25, command_and_launcher_cpu_reserve_seconds=5,
            original_failed_process_actual_cpu_seconds=None,
            original_failed_process_cpu_upper_seconds=120,
            cumulative_cpu_upper_seconds=120+child_cpu+5,
            revised_total_cold_start_cpu_ceiling_seconds=150,
            physics_substeps=0, model_queries=0, neural_forwards=0, policy_updates=0,
        ))
    return int(returncode != 0)


if __name__ == "__main__":
    raise SystemExit(main())
