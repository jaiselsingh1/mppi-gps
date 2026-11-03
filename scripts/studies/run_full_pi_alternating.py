"""Launch exactly three frozen local workers; stop peers on a failed exit."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def dump(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def binding(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024**2), b""):
            h.update(b)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--commit", required=True)
    a = p.parse_args()
    run = a.run.resolve()
    source = run / "source"
    config = json.loads((run / "config.json").read_text())
    expected = json.loads((run / "source_manifest.json").read_text())
    for name, b in expected.items():
        if binding(source / name) != b:
            raise ValueError("Frozen source changed: " + name)
    if any((run / f"stream_{i}").exists() for i in range(3)):
        raise FileExistsError("Workers are create-only; no automatic resume or retry")
    env = os.environ.copy()
    env.update(
        {
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        }
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    commands = [
        [
            sys.executable,
            "-m",
            "src.gps.mppi_gps.full_pi_alternating",
            "run",
            "--run",
            str(run),
            "--stream",
            str(i),
        ]
        for i in range(3)
    ]
    dump(
        run / "launch_receipt.json",
        dict(
            commit=a.commit,
            interpreter=sys.executable,
            source_manifest=binding(run / "source_manifest.json"),
            config=binding(run / "config.json"),
            commands=commands,
            concurrency=3,
            native_threads=4,
            torch_threads=1,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            deadline_seconds=config["total_limits"]["execution_wall_seconds"],
            gpu=False,
        ),
    )
    started = time.monotonic()
    processes = []
    files = []
    stopped = False
    try:
        for i, command in enumerate(commands):
            stdout = (run / f"worker_{i}_stdout.txt").open("x")
            stderr = (run / f"worker_{i}_stderr.txt").open("x")
            files += [stdout, stderr]
            processes.append(
                subprocess.Popen(command, cwd=source, env=env, stdout=stdout, stderr=stderr)
            )
        while any(q.poll() is None for q in processes):
            codes = [q.poll() for q in processes]
            if (
                any(c is not None and c != 0 for c in codes)
                or time.monotonic() - started > config["total_limits"]["execution_wall_seconds"]
            ):
                stopped = True
                for q in processes:
                    if q.poll() is None:
                        q.terminate()
                for q in processes:
                    try:
                        q.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        q.kill()
                        q.wait()
                break
            time.sleep(1)
    except BaseException:
        stopped = True
        for q in processes:
            if q.poll() is None:
                q.terminate()
        for q in processes:
            q.wait()
        raise
    finally:
        for f in files:
            f.close()
        for name, b in expected.items():
            if binding(source / name) != b:
                stopped = True
        codes = [q.poll() for q in processes]
        dump(
            run / "coordinator_result.json",
            dict(
                exit_codes=codes,
                stopped=stopped,
                complete=len(codes) == 3 and all(c == 0 for c in codes),
                wall_seconds=time.monotonic() - started,
            ),
        )
        manifest = {}
        read_bytes = 0
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.name not in ("files_manifest.json", "seal_receipt.json"):
                b = binding(path)
                manifest[str(path.relative_to(run))] = b
                read_bytes += b["bytes"]
        dump(run / "files_manifest.json", manifest)
        dump(
            run / "seal_receipt.json",
            dict(
                files=len(manifest),
                hash_read_bytes=read_bytes,
                manifest=binding(run / "files_manifest.json"),
                exclusions=["files_manifest.json", "seal_receipt.json"],
            ),
        )
    print(
        json.dumps(
            dict(exit_codes=codes, stopped=stopped, wall_seconds=time.monotonic() - started)
        ),
        flush=True,
    )
    return 0 if not stopped and all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
