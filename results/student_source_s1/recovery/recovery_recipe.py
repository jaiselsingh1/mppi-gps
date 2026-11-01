"""Append only missing EOF bytes to a DERIVED copy; never accept unverified arrays."""

from pathlib import Path
import hashlib, json, re, shutil, time, h5py

r = Path("/home/jaisel/mppi-gps-workspace/runs/2026-10-31_student_source_s1_001")
a = Path("/home/jaisel/mppi-gps-workspace/artifacts") / r.name
out = a / "recovery_001"
out.mkdir(exist_ok=False)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


source = r / "evidence/eval/raw.h5"
before = sha(source)
size = source.stat().st_size
manifest = json.loads((a / "original_interrupted_run_file_manifest.json").read_text())
binding = next(x for x in manifest["files"] if x["file"] == "evidence/eval/raw.h5")
assert before == binding["sha256"] and size == binding["bytes"]
try:
    with h5py.File(source, "r"):
        pass
except OSError as e:
    original_error = str(e)
else:
    raise RuntimeError("Original unexpectedly opens; no recovery authorized")
match = re.search(r"eof = (\d+),.*stored_eof = (\d+)", original_error)
assert match and int(match[1]) == size
expected = int(match[2])
append = expected - size
assert append == 14352 and 0 < append <= 65536
copy = out / "eval_recovered_container.h5"
shutil.copy2(source, copy)
assert sha(copy) == before
with copy.open("ab") as f:
    f.write(bytes(append))
    f.flush()
    __import__("os").fsync(f.fileno())
record = dict(
    schema="walker-s1-derived-container-recovery-v1",
    original_file=str(source),
    original_sha256=before,
    original_bytes=size,
    original_error=original_error,
    recovered_file=str(copy),
    recovered_sha256=sha(copy),
    recovered_bytes=copy.stat().st_size,
    appended_zero_bytes=append,
    method="Exact byte copy followed by append of14352zero bytes to match HDF storedEOF; this is NOT a repair of scientific data. Only datasets whose saved dtype/shape/SHA256 match the durable ledger may be accepted.",
    new_inference=0,
    new_physics=0,
    new_training=0,
)
try:
    with h5py.File(copy, "r") as f:
        record.update(readable=True, array_groups=len(f["arrays"]))
except Exception as e:
    record.update(readable=False, recovery_open_error=type(e).__name__ + ": " + str(e))
assert sha(source) == before and source.stat().st_size == size
record["original_unchanged"] = True
with (out / "recovery_receipt.json").open("x") as f:
    json.dump(record, f, indent=2)
    f.write("\n")
shutil.copy2(__file__, out / "recovery_recipe.py")
print(json.dumps(record, indent=2))
