import gzip
import hashlib
import json
import pathlib
import shutil

root = pathlib.Path(__file__).resolve().parents[2]
run = root / "runs/2026-11-03_full_pi_alternating_bounded_001"
art = root / "artifacts/full_pi_alternating_2026-11-03"
out = root / "results/full_pi_alternating_001"
out.mkdir(parents=True, exist_ok=False)
reads = 0
hash_reads = 0
origins = {}


def bind(p):
    global hash_reads
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
            hash_reads += len(b)
    return {"bytes": p.stat().st_size, "sha256": h.hexdigest()}


def copy(p, name):
    global reads
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, target)
    reads += p.stat().st_size
    origins[name] = {"source": str(p), "original": bind(p), "encoding": "exact"}


def gz(p, name):
    global reads
    data = p.read_bytes()
    reads += len(data)
    target = out / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as g:
            g.write(data)
    origins[name] = {"source": str(p), "original": bind(p), "encoding": "gzip lossless"}


for name in (
    "config.json",
    "freeze.json",
    "launch_receipt.json",
    "coordinator_result.json",
    "seal_receipt.json",
    "source_manifest.json",
    "input_manifest.json",
):
    copy(run / name, "run/" + name)
copy(run / "files_manifest.json", "external/raw_file_manifest.json")
for s in range(3):
    for n in ("result.json", "work.json"):
        copy(run / f"stream_{s}" / n, f"run/stream_{s}/" + n)
    for n in (f"worker_{s}_stdout.txt", f"worker_{s}_stderr.txt"):
        copy(run / n, "run/" + n)
    gz(run / f"stream_{s}/ledger.jsonl", f"run/stream_{s}/ledger.jsonl.gz")
    for p in sorted((run / f"stream_{s}").glob("r*/result.json")):
        copy(p, f"rounds/s{s}/" + p.parent.name + ".json")
    for p in sorted((run / f"stream_{s}").glob("r*/fit.json")):
        gz(p, f"fits/s{s}/" + p.parent.name + ".json.gz")
    for p in sorted((run / f"stream_{s}").glob("r*/*/iterations.json")):
        gz(p, f"searches/s{s}/" + p.parent.parent.name + "/" + p.parent.name + ".json.gz")
    for p in sorted((run / f"stream_{s}").glob("r*/search_*/selection.json")):
        d = json.loads(p.read_text())
        reads += p.stat().st_size
        # No physical initial context or input arrays are copied into the public capsule.
        selected = {
            k: d[k]
            for k in (
                "selected",
                "raw_best",
                "alpha",
                "passes",
                "raw_best_rejected",
                "selected_change_rms",
                "objective_relative_gain",
            )
        }
        name = f"selections/s{s}/" + p.parent.parent.name + "/" + p.parent.name + ".json"
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(selected, indent=2) + "\n")
        origins[name] = {
            "source": str(p),
            "original": bind(p),
            "encoding": "declared scalar projection; original context stays external",
        }
for dirname in ("audit_001", "join_audit_001", "report_001", "trace_report_001"):
    for p in sorted((art / dirname).rglob("*")):
        if p.is_file() and p.suffix in (".json", ".md", ".png", ".jsonl"):
            name = dirname + "/" + str(p.relative_to(art / dirname))
            gz(p, name + ".gz") if p.suffix == ".jsonl" else copy(p, name)
for pattern in ("*_stdout.txt", "*_stderr.txt"):
    for p in sorted(art.glob(pattern)):
        copy(p, "accounting/" + p.name)
for p in sorted(art.glob("*receipt*.json*")):
    gz(p, "accounting/" + p.name + ".gz") if p.suffix == ".jsonl" else copy(
        p, "accounting/" + p.name
    )
for p in sorted(art.glob("*census*.json")):
    copy(p, "accounting/" + p.name)
for name in (
    "engineering_start.json",
    "first_solve_observation.json",
    "control_first_solve_observation.json",
    "checkpoint_join_observation.json",
    "first_missing_anchor_observation.json",
    "round2_observation.json",
    "round3_query_coverage_observation.json",
):
    if (art / name).exists():
        copy(art / name, "accounting/" + name)
for p in sorted((art / "query_context_check_001").rglob("*")):
    if p.is_file() and p.suffix in (".json", ".jsonl"):
        name = "query_context_check_001/" + str(p.relative_to(art / "query_context_check_001"))
        gz(p, name + ".gz") if p.suffix == ".jsonl" else copy(p, name)
(out / "origins.json").write_text(json.dumps(origins, indent=2) + "\n")
external = {
    "full_run": str(run),
    "availability": "local-only; no remote raw backup asserted",
    "raw_inventory": "raw_file_manifest.json",
    "source_commit": json.loads((run / "freeze.json").read_text())["commit"],
    "missing_hypothetical_banks": "Only the frozen sparse full banks are preserved. Every evaluated nominal and per-candidate score/weight/length is retained; other full banks are not raw preserved evidence.",
}
(out / "external/availability.json").write_text(json.dumps(external, indent=2) + "\n")
(out / "collection_receipt.json").write_text(
    json.dumps(
        {
            "logical_copy_and_decode_read_bytes": reads,
            "origin_hash_read_bytes_before_output_seal": hash_reads,
            "new_physics": 0,
            "new_inference": 0,
            "new_fitting": 0,
            "source_run": str(run),
            "raw_arrays_and_checkpoints_exported": False,
            "scope": "Application-level bytes counted in collector, not physical disk I/O; final output sealing and OS/library reads are separate.",
        },
        indent=2,
    )
    + "\n"
)
manifest = {str(p.relative_to(out)): bind(p) for p in out.rglob("*") if p.is_file()}
(out / "files_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(
    json.dumps(
        {
            "files": len(manifest),
            "bytes": sum(v["bytes"] for v in manifest.values()),
            "out": str(out),
        }
    )
)
