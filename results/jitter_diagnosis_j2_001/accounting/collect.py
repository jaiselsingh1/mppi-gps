"""Collect exact compact J2 metadata; never copy raw arrays or checkpoints to Git."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
RUN = REPO / "runs/2026-11-03_jitter_diagnosis_j2_001"
OUT = REPO / "results/jitter_diagnosis_j2_001"
OUT.mkdir(exist_ok=False)
started = time.monotonic()
read_bytes = 0
origins = {}


def digest_file(path):
    global read_bytes
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            read_bytes += len(block)
            digest.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return dict(bytes=after.st_size, sha256=digest.hexdigest())


def copy(name, source, compressed=False):
    global read_bytes
    assert source.suffix not in {".pt", ".npz", ".h5"}
    before = source.stat()
    assert before.st_size <= 32*1024**2
    data = source.read_bytes()
    read_bytes += len(data)
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    payload = gzip.compress(data, compresslevel=9, mtime=0) if compressed else data
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    origins[name] = dict(source=str(source.resolve()), original_bytes=len(data),
                         original_sha256=hashlib.sha256(data).hexdigest(),
                         encoding="gzip" if compressed else "exact")


for name in ("config.json", "result.json", "work.json", "inputs/manifest.json", "inputs/explainers.json"):
    copy("run/" + name, RUN / name)
for case in sorted((RUN / "cases").iterdir()):
    copy(f"run/cases/{case.name}/summary.json", case / "summary.json")
    copy(f"run/cases/{case.name}/fit.json.gz", case / "fit.json", compressed=True)
copy("audit/audit_result.json", BASE / "audit_001/audit_result.json")
for path in sorted((BASE / "report_001").iterdir()):
    copy("report/" + path.name, path)
for path in sorted((BASE / "within_clean_posthoc_001").iterdir()):
    copy("posthoc/" + path.name, path)
for name in ("freeze_receipt.json", "freeze_receipt_002.json", "prelaunch_source_correction_002.json", "run_owned_phase.py"):
    copy("accounting/" + name, BASE / name)
for folder in ("operator_worker_001", "operator_audit_001", "prelaunch_ci_001", "unused_source_ci_001", "input_inventory_001", "seed_scan_001", "source_preflight_001", "source_preflight_failure_001", "launcher_import_failure_001"):
    for path in sorted((BASE / folder).iterdir()):
        if path.is_file():
            copy(f"accounting/{folder}/{path.name}", path)
for version in ("source_001", "source_002"):
    copy(f"source/{version}_manifest.json", BASE / version / "source_manifest.json")
for name in ("jitter_diagnosis.py", "audits/jitter_diagnosis.py", "reporting/jitter_diagnosis.py"):
    copy("source/" + name, BASE / "source_002/src/gps/mppi_gps" / name)
copy("source/protocol.md", BASE / "source_002/docs/jitter_diagnosis_j2_protocol.md")

worker = json.loads((RUN / "result.json").read_text())
audit = json.loads((BASE / "audit_001/audit_result.json").read_text())
report = json.loads((BASE / "report_001/receipt.json").read_text())
cost = dict(scope="Incremental J2 worker/audit/report; inherited source acquisition and reference training totals unavailable.",
            worker=worker["work"],
            worker_process=json.loads((BASE / "operator_worker_001/exit.json").read_text()),
            audit_process=json.loads((BASE / "operator_audit_001/exit.json").read_text()),
            audit=dict(read_bytes=audit["read_bytes"], decoded_array_bytes=audit["decoded_array_bytes"],
                       neural_rows=audit["new_neural_rows"], network_calls=audit["functional_network_calls"],
                       checkpoint_loads=audit["checkpoint_loads"], new_physics=0, new_gradients=0, new_updates=0),
            report=report,
            preparation=json.loads((BASE / "prepared_001/manifest.json").read_text()),
            engineering_known={"focused_final_tests": {"passed":67,"reported_pytest_seconds":16.35},
                               "full_prelaunch_ci": {"passed":1767,"skipped":3,"xfailed":1,"commands":57},
                               "failures": ["system-Python import before source preflight", "launcher filename shadowed stdlib operator before process creation"],
                               "unused_first_export": "a089d5d; no scientific execution"},
            unknown=["complete engineering labor/CPU", "physical disk I/O", "energy", "money", "inherited acquisition and reference-training totals"],
            limits=json.loads((RUN / "config.json").read_text())["budget"])
(OUT / "accounting/complete_costs.json").write_text(json.dumps(cost, indent=2)+"\n")

external = []
for path in sorted(RUN.rglob("*")):
    if path.is_file():
        relative = str(path.relative_to(RUN))
        compact = "run/" + relative
        compressed = compact + ".gz"
        external.append(dict(file=relative, local_path=str(path.resolve()), availability="local_only",
                             compact_copy=compact if compact in origins else compressed if compressed in origins else None,
                             **digest_file(path)))
(OUT / "external_evidence.json").write_text(json.dumps(dict(worker_root=str(RUN.resolve()), files=external,
    raw_arrays_and_checkpoints_in_git=False, remote_backup_verified=False), indent=2)+"\n")

context = REPO / ".claude/worktrees/sac-continuation"
context_paths = ["scripts/diagnostics/stability_terminal.py", "scripts/diagnostics/takeover_terminal.py",
                 "scripts/diagnostics/sac_continuation.py",
                 "runs/2026-11-03_stability_calibration_walker_001/summary.json",
                 "runs/2026-11-03_stability_calibration_faller_001/summary.json"]
for name in context_paths:
    copy("context/" + name, context / name)
copy("context/user_context.txt", Path("/Users/jaisel/.codex/attachments/1b05e036-3a19-4719-993b-669d8dbfe916/Pasted text.txt"))
(OUT / "context/provenance.json").write_text(json.dumps(dict(
    status="PROVISIONAL_OBSERVED_CONTEXT_ONLY", source_worktree=str(context),
    source_commit=subprocess.check_output(["git","-C",str(context),"rev-parse","HEAD"],text=True).strip(),
    original_launch_binding=False, raw_calibration_vectors_available=False,
    independently_verified_correlations=False,
    limitations=["Ordinal double-argsort rank helper mishandles ties", "No full checkpoint/context/cost census preserved by those scripts", "State-level takeover calibration is different from across-student walking diagnosis", "No novelty or useful-learning claim admitted"],
    takeover_output_directories_observed_empty=True,
),indent=2)+"\n")

(OUT / "origins.json").write_text(json.dumps(origins, indent=2)+"\n")
receipt = dict(read_bytes=read_bytes, wall_seconds=time.monotonic()-started, new_scientific_work=0,
               collector_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               actual_cases=len(list((RUN / "cases").glob("*/summary.json"))),
               source_commit="edda434a2771194163aa5be883b83f4f94d3cd56")
(OUT / "collection_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
shutil.copyfile(__file__, OUT / "accounting/collect.py")
assert sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()) <= 64*1024**2
print(json.dumps(receipt,indent=2))
