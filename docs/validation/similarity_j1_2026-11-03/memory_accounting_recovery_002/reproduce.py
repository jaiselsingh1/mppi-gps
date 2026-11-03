"""One bounded Linux-only publication memory reproduction; no research libraries."""

from __future__ import annotations
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    import argparse
    import gzip

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("This controlled fork/exec reproduction requires Linux")
    base = Path(__file__).resolve().parent
    protocol = json.loads((base / "protocol.json").read_text())
    for name, binding in protocol["sources"].items():
        if sha(base / name) != binding["sha256"]:
            raise ValueError("Source changed:" + name)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out / "synthetic_capsule"
    (source / "run").mkdir(parents=True)
    original = b'{"learning_curve":[{"update":1,"action_mse":0.2}]}\n'
    encoded = gzip.compress(original, mtime=0)
    (source / "run/curve.json.gz").write_bytes(encoded)
    new(
        source / "capsule.json",
        dict(
            mode="metadata",
            files={
                "run/curve.json.gz": dict(
                    encoding="gzip_exact_original_bytes",
                    original_bytes=len(original),
                    original_sha256=hashlib.sha256(original).hexdigest(),
                )
            },
        ),
    )
    new(
        source / "files_manifest.json",
        dict(
            files={
                str(path.relative_to(source)): dict(
                    bytes=path.stat().st_size, sha256=sha(path)
                )
                for path in source.rglob("*")
                if path.is_file()
            }
        ),
    )
    manifest = sha(source / "files_manifest.json")
    new(source / "collection_exit.json", dict(passed=True, manifest_sha256=manifest))
    closing = sha(source / "collection_exit.json")
    commands = []
    for name in ("before.py", "after.py"):
        target = out / name.removesuffix(".py")
        code = (
            "import runpy,sys; sys.argv="
            + repr(
                [
                    str(base / name),
                    "--capsule",
                    str(source),
                    "--out",
                    str(target),
                    "--manifest-sha256",
                    manifest,
                    "--exit-sha256",
                    closing,
                ]
            )
            + "; runpy.run_path("
            + repr(str(base / name))
            + ",run_name='__main__')"
        )
        commands.append([sys.executable, "-I", "-c", code])
    pressure = """import json,resource,subprocess,sys
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
allocation=bytearray(580*1024**2)
for index in range(0,len(allocation),4096):allocation[index]=1
memory={}
for line in open('/proc/self/status'):
 key,_,value=line.partition(':')
 if key in ('VmHWM','VmRSS'):memory[key]=int(value.split()[0])*1024
children=[]
for command in json.loads(sys.argv[1]):
 process=subprocess.run(command,capture_output=True,text=True,timeout=20)
 children.append(dict(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
print(json.dumps(dict(parent_current_exec_peak_rss_bytes=max(memory.values()),parent_resource_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,children=children)))
"""
    command = [sys.executable, "-I", "-c", pressure, json.dumps(commands)]
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (
        (out / "stdout.txt").open("xb") as stdout,
        (out / "stderr.txt").open("xb") as stderr,
    ):
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, start_new_session=True
        )
        reason = None
        try:
            rc = process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            reason = "wall_limit"
            os.killpg(process.pid, signal.SIGKILL)
            rc = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = dict(
        pid=process.pid,
        exit_code=rc,
        leader_reaped=True,
        stopping_reason=reason,
        wall_seconds=time.monotonic() - started,
        user_seconds=usage.ru_utime - usage_before.ru_utime,
        system_seconds=usage.ru_stime - usage_before.ru_stime,
        protocol_sha256=sha(base / "protocol.json"),
        new_scientific_work=0,
    )
    new(out / "process_exit.json", receipt)
    if rc != 0 or reason:
        raise ValueError("Controlled memory parent did not close cleanly")
    measured = json.loads((out / "stdout.txt").read_text())
    before = json.loads((out / "before/materialization_receipt.json").read_text())
    after = json.loads((out / "after/materialization_receipt.json").read_text())
    checks = dict(
        parent_above512_and_below768MiB=512 * 1024**2
        < measured["parent_current_exec_peak_rss_bytes"]
        < 768 * 1024**2,
        old_reproduced=measured["children"][0]["returncode"] == 1
        and before["passed"] is False
        and before["source_read_bytes"] == 0
        and before["peak_rss_bytes"] > 512 * 1024**2,
        new_passed=measured["children"][1]["returncode"] == 0
        and after["passed"] is True,
        new_current_exec_under512MiB=0 < after["peak_rss_bytes"] < 512 * 1024**2,
        new_inherited_accounting_retained=after["resource_peak_rss_bytes"]
        > 512 * 1024**2,
        decoded_exact=(out / "after/run/curve.json").read_bytes() == original,
        no_scientific_work=before["new_scientific_work"]
        == after["new_scientific_work"]
        == 0,
    )
    new(
        out / "result.json",
        dict(
            passed=all(checks.values()),
            checks=checks,
            measurements=measured,
            before=before,
            after=after,
            process=receipt,
            bytes=sum(path.stat().st_size for path in out.rglob("*") if path.is_file()),
        ),
    )
    print(json.dumps(dict(passed=all(checks.values()), checks=checks, process=receipt)))
    return int(not all(checks.values()))


if __name__ == "__main__":
    raise SystemExit(main())
