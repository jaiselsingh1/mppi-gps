from pathlib import Path
import json, shutil, subprocess, tempfile, time, os, hashlib

ROOT = Path("/Users/jaisel/Documents/mppi-gps")
ART = ROOT / "artifacts/2026-10-31_student_source_s1_001"
OUT = ART / "publication_validation_001"
OUT.mkdir(exist_ok=False)
export = Path(tempfile.mkdtemp(prefix="walker-s1-final-export-"))
paths = (
    subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    )
    .decode()
    .split("\0")
)
files = sorted(set(p for p in paths if p))
for name in files:
    source = ROOT / name
    target = export / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
for name in [".git", "runs", "artifacts", ".venv"]:
    assert not (export / name).exists()
cap = json.loads((ROOT / "results/student_source_s1/capsule.json").read_text())
assert all(
    (export / "results/student_source_s1" / name).is_file() for name in cap["files"]
)
env = dict(
    os.environ,
    PYTHONDONTWRITEBYTECODE="1",
    MPLBACKEND="Agg",
    MPLCONFIGDIR=str(OUT / "matplotlib_cache"),
)
env.pop("PYTHONPATH", None)
command = [
    str(ROOT / ".venv/bin/python"),
    "-B",
    "scripts/validate_export.py",
    "--export",
    str(export),
]
started = time.perf_counter()
with (OUT / "stdout.txt").open("xb") as out, (OUT / "stderr.txt").open("xb") as err:
    process = subprocess.run(command, cwd=export, env=env, stdout=out, stderr=err)
receipt = dict(
    command=command,
    cwd=str(export),
    files=len(files),
    exit_code=process.returncode,
    wall_seconds=time.perf_counter() - started,
    no_ignored_evidence=True,
)
with (OUT / "receipt.json").open("x") as f:
    json.dump(receipt, f, indent=2)
    f.write("\n")
print(json.dumps(receipt, indent=2))
print((OUT / "stdout.txt").read_text()[-5000:])
print((OUT / "stderr.txt").read_text()[-2000:])
raise SystemExit(process.returncode)
