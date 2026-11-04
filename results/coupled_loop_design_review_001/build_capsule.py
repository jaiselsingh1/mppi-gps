"""Package existing review outputs and render saved CSVs, without research work."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

resource.setrlimit(resource.RLIMIT_CPU, (55, 55))
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
started = time.monotonic()
root = Path(__file__).resolve().parent
repo = root.parents[1]
source = repo / "runs/2026-11-04_loop_design_review_001"
final = source / "final_snapshot"
copies = []


def binding(path):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
            count += len(block)
    return dict(path=str(path.resolve()), bytes=count, sha256=digest.hexdigest())


def preserve(path, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    original = binding(path)
    if path.name == "per_bank.csv":
        destination = destination.with_suffix(".csv.gz")
        with path.open("rb") as incoming, destination.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=raw, mtime=0) as compressed:
                shutil.copyfileobj(incoming, compressed)
        decoded = hashlib.sha256()
        decoded_bytes = 0
        with gzip.open(destination, "rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                decoded.update(block)
                decoded_bytes += len(block)
        assert decoded_bytes == original["bytes"] and decoded.hexdigest() == original["sha256"]
        copies.append(dict(original=original, capsule=binding(destination), encoding="gzip", decoded_sha256=decoded.hexdigest(), decoded_bytes=decoded_bytes))
    else:
        shutil.copyfile(path, destination)
        retained = binding(destination)
        assert all(retained[key] == original[key] for key in ("bytes", "sha256"))
        copies.append(dict(original=original, capsule=retained, encoding="identity"))


# Preserve the earlier partial review, including its packaging failure, as well
# as every final metric/config/source/receipt. No source path is altered.
for path in sorted(source.rglob("*")):
    if path.is_file():
        preserve(path, root / "raw" / path.relative_to(source))

sidecars = {
    "per_bank.csv": final / "banks_and_fit/per_bank.csv",
    "per_cell.csv": final / "banks_and_fit/per_cell.csv",
    "fit_mse.csv": final / "banks_and_fit/fit_mse.csv",
    "earlier_slot_costs.csv": final / "banks_and_fit/earlier_slot_costs.csv",
    "admissions.csv": final / "contexts_admission/admissions.csv",
    "corpus_novelty.csv": final / "contexts_admission/corpus_novelty.csv",
    "fall_support.csv": final / "contexts_admission/fall_support.csv",
    "spectra.csv": final / "spectrum_conflict/metrics.csv",
    "cold_start.csv": final / "cold_start/episodes.csv",
    "probe_initialization.csv": final / "cold_start/measurement_probes.csv",
}
for name, path in sidecars.items():
    if name == "per_bank.csv":
        shutil.copyfile(root / "raw/final_snapshot/banks_and_fit/per_bank.csv.gz", root / "per_bank.csv.gz")
    else:
        shutil.copyfile(path, root / name)

admissions = list(csv.DictReader((root / "admissions.csv").open()))
novelty = list(csv.DictReader((root / "corpus_novelty.csv").open()))
table = [
    "# Complete acquisition and retention table",
    "",
    "All eight rounds, three streams, two arms. Each row describes eight attempted episodes and the 2,400 rows actually supplied to fitting. ‘Changed’ means newly admitted source arrays; ‘identical’ means admitted arrays equal the previous source. Fallback retains the prior competent source. Source new rows compare exact recorded input–target pairs with the previous episode. FP32 new rows compare actual learner arrays with all ancestral supplied corpora; neither counts independent tasks. [Raw episode decisions](admissions.csv), [corpus novelty](corpus_novelty.csv).",
    "",
    "| Arm | Stream | Round | Changed | Identical | Fallback | Source new rows vs previous | FP32 new rows vs ancestors | Unique corpus rows |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for arm in ("independent", "positive"):
    for stream in range(3):
        for round_index in range(1, 9):
            rows = [row for row in admissions if row["arm"] == arm and int(row["stream"]) == stream and int(row["round"]) == round_index]
            assert len(rows) == 8
            counts = {key: sum(row["classification"] == key for row in rows) for key in ("admitted_changed", "admitted_identical", "fallback")}
            corpus = next(row for row in novelty if row["arm"] == arm and int(row["stream"]) == stream and int(row["round"]) == round_index and row["representation"] == "stored_raw")
            learner = next(row for row in novelty if row["arm"] == arm and int(row["stream"]) == stream and int(row["round"]) == round_index and row["representation"] != "stored_raw")
            new = sum(int(row["exact_new_unique_input_target_rows"]) for row in rows)
            table.append(f"| {'Coupled' if arm == 'positive' else 'Static refitting'} | {stream} | {round_index} | {counts['admitted_changed']} | {counts['admitted_identical']} | {counts['fallback']} | {new} | {learner['new_unique_rows_vs_ancestral_union']} | {corpus['unique_input_target_rows']} |")
(root / "admission_table.md").write_text("\n".join(table) + "\n")

render_started = time.monotonic()
child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
subprocess.run([sys.executable, "-B", str(root / "render_review.py")], check=True, env=os.environ.copy())
child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
render_cpu = (child_after.ru_utime + child_after.ru_stime) - (child_before.ru_utime + child_before.ru_stime)

manifest = dict(
    schema="saved-loop-design-review-capsule-1",
    scope="Exact saved review outputs; source arrays/checkpoints and experiment evidence remain external. No new audit or experiment.",
    original_review_root=str(source),
    final_config=binding(final / "config.json"),
    final_external_receipt=binding(final / "external_receipt.json"),
    preserved_sources=copies,
    final_per_bank_decoded=binding(final / "banks_and_fit/per_bank.csv"),
    original_audit_failure_preserved=True,
    capsule_files=[binding(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name not in ("manifest.json", "publication_receipt.json")],
)
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
cpu = time.process_time() + render_cpu
receipt = dict(
    schema="saved-review-publication-receipt-1", status="COMPLETED",
    cpu_seconds=cpu, process_cpu_seconds=time.process_time(), render_child_cpu_seconds=render_cpu,
    wall_seconds=time.monotonic() - started,
    render_wall_seconds=time.monotonic() - render_started,
    command_and_prose_edit_cpu_reserve_seconds=5,
    charged_cpu_upper_seconds=cpu + 5,
    frozen_publication_cpu_limit_seconds=60,
    new_physics=0, new_model_queries=0, new_neural_forwards=0, new_policy_updates=0,
    new_experiment_audits=0,
    preserved_file_count=len(copies),
    output_bytes=sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
    manifest=binding(root / "manifest.json"),
)
assert receipt["charged_cpu_upper_seconds"] <= 60
(root / "publication_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
capsule = dict(
    schema="saved-loop-design-review-capsule-checksums-1",
    files={str(path.relative_to(root)): {key: value for key, value in binding(path).items() if key != "path"} for path in sorted(root.rglob("*")) if path.is_file() and path.name != "capsule.json"},
)
(root / "capsule.json").write_text(json.dumps(capsule, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
