"""Create-only publication of saved L3.1 output; no audit or science execution."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import resource
import shutil
import statistics
import time
from pathlib import Path

START_CPU, START_WALL = time.process_time(), time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU, (55, 60))
ROOT = Path(__file__).resolve().parents[2]
CAPSULE = Path(__file__).resolve().parent
BASE = ROOT / "runs/2026-11-04_local_coupling_multiround_l3_1_001"
OLD = ROOT / "runs/2026-11-04_local_coupling_multiround_l3_001"
OUT = CAPSULE / "completion_001"
DOC = ROOT / "docs/local_coupling_multiround_l3_1_result.md"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path.read_bytes()))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def json_read(path):
    return json.loads(path.read_text())


original_bindings = [binding(path) for path in CAPSULE.rglob("*") if path.is_file()]
OUT.mkdir(exist_ok=False)
copied = []


def copy_exact(source, name, compressed=False):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        raw = source.read_bytes()
        with target.open("xb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0, compresslevel=9) as zipped:
                zipped.write(raw)
        assert gzip.decompress(target.read_bytes()) == raw
    else:
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        assert target.read_bytes() == source.read_bytes()
    copied.append(dict(source=binding(source), capsule=str(target.relative_to(CAPSULE)), stored=binding(target), compression="gzip" if compressed else None, decoded_sha256=digest(source.read_bytes()), decoded_bytes=source.stat().st_size))


worker = json_read(BASE / "worker/result.json")
supervisor = json_read(BASE / "supervisor_result.json")
audit = json_read(BASE / "audit/final/findings.json")
assert worker["status"] == "COMPLETED_ALL_ROUNDS" and worker["planned_rounds"] == 8
assert supervisor["status"] == "CLOSED_UNCERTIFIED" and audit["passed"] is False
assert "Probe manifest teacher torque" in audit["error"]

for name in ("external_worker.json", "supervisor_result.json", "launch.json", "resume_manifest.json"):
    copy_exact(BASE / name, name, compressed=name == "resume_manifest.json") if name != "resume_manifest.json" else copy_exact(BASE / name, name + ".gz", True)
copy_exact(BASE / "worker/result.json", "worker_result.json")
for path in sorted((BASE / "audit/final").iterdir()):
    if path.is_file():
        large = path.stat().st_size > 256 * 1024
        copy_exact(path, "audit/" + path.name + (".gz" if large else ""), large)
note_path = ROOT / "runs/2026-11-04_loop_design_review_001/probe_representation_note.json"
note = json_read(note_path)
assert note["max_teacher_numeric_error"] == note["max_student_numeric_error"] == 0
assert note["observed_anchors"] == 157 and note["new_audit_run"] is False
copy_exact(note_path, "audit/probe_representation_note.json")
copy_exact(OLD / "source/configs/studies/local_coupling_multiround_l3_0.json", "science_config.json")
copy_exact(OLD / "environment.json", "original_environment.json")
for path in sorted((BASE / "report").iterdir()):
    if not path.is_file():
        continue
    large = path.stat().st_size > 512 * 1024 or path.suffix == ".md"
    copy_exact(path, "report/" + path.name + (".gz" if large else ""), large)

# Recursively preserve source bindings from closed result records. Large arrays
# and checkpoints stay at their original local paths; their saved hashes are
# references, not a claim that this publication reran the independent audit.
raw_bindings = {}
seen_records = set()


def collect(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value) and len(value) == 3:
            raw_bindings[value["path"]] = value
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)


def read_record(spec):
    path = Path(spec["path"])
    assert binding(path) == spec
    data = json_read(path)
    collect(data)
    seen_records.add(str(path))
    return data


collect(worker)
all_streams, endpoint_rows, evidence_rows = [], [], []
for stream_entry in sorted(worker["streams"], key=lambda value: value["stream"]):
    stream = stream_entry["stream"]
    snapshot = read_record(stream_entry["result"])
    assert snapshot["status"] == "COMPLETED_ALL_ROUNDS"
    complete = [value for value in snapshot["rounds"] if value["status"] == "COMPLETED"]
    assert [value["round"] for value in complete] == list(range(1, 9))
    copy_exact(Path(stream_entry["result"]["path"]), f"streams/s{stream}_result.json.gz", True)
    all_streams.append(snapshot)
    for round_record in complete:
        for acquisition in round_record["acquisitions"]:
            read_record(acquisition["manifest"])
        for fit in round_record["fits"]:
            read_record(fit["result"])
        for probe in round_record["probes"]:
            read_record(probe["result"])
        for envelope in round_record["outgoing_streams"]:
            record = read_record(envelope["envelope"])
            for name, item in record["files"].items():
                raw_bindings[str(Path(envelope["envelope"]["path"]).parent / name)] = dict(path=str(Path(envelope["envelope"]["path"]).parent / name), **item)
    final_entries = [value for value in complete[-1]["evaluations"] if value["panel"] == "primary" and value["kind"] == "child"]
    final_entries += [value for value in snapshot["final_evaluations"] if value["panel"] == "confirmation" and value["kind"] == "final"]
    for arm in ("independent", "positive"):
        for panel in ("primary", "confirmation"):
            entries = [value for value in final_entries if value["arm"] == arm and value["panel"] == panel]
            assert len(entries) == 16 and len({value["reset_seed"] for value in entries}) == 16
            outcomes = []
            for entry in entries:
                result = read_record(entry["result"])
                assert result["status"] == "COMPLETED"
                assert result["outcome"]["requested_controls"] == 1000
                outcomes.append(result["outcome"])
                evidence_rows.append(dict(stream=stream, arm=arm, arm_label="alpha0 static refitting" if arm == "independent" else "alpha0.12 coupled", panel=panel, reset_seed=entry["reset_seed"], **result["outcome"], result_binding=entry["result"], trace_binding=result["trace"], checkpoint_binding=result["checkpoint"]))
            endpoint_rows.append(dict(stream=stream, arm=arm, arm_label="alpha0 static refitting" if arm == "independent" else "alpha0.12 coupled", panel=panel, episodes=16, walking=sum(value["walking"] for value in outcomes), full_survival=sum(value["full_survival"] for value in outcomes), controls=sum(value["controls"] for value in outcomes), mean_controls=statistics.mean(value["controls"] for value in outcomes), mean_complete_cost=statistics.mean(value["complete_cost"] for value in outcomes)))
    # Preserve every completed evaluation binding, not merely final favorable
    # examples. This reads only compact metadata; no checkpoint is decoded.
    for entry in snapshot["initial_evaluations"] + snapshot["final_evaluations"] + [entry for round_record in complete for entry in round_record["evaluations"]]:
        if entry["result"]["path"] not in seen_records:
            read_record(entry["result"])
    for entry in snapshot["initial_probes"]:
        read_record(entry["result"])

assert len(evidence_rows) == 192
assert sum(row["walking"] for row in evidence_rows) == 0
assert sum(row["full_survival"] for row in evidence_rows if row["arm"] == "independent") == 0
assert sum(row["full_survival"] for row in evidence_rows if row["arm"] == "positive") == 2
write(OUT / "final_outcomes_verified_from_saved_results.json", dict(schema="l3-final-saved-outcomes-1", scope="Publication aggregation of closed saved results; not an audit rerun", panels=endpoint_rows, episodes=evidence_rows))

teacher_rows = list(csv.DictReader((BASE / "report/teacher_outcomes.csv").open()))
activity = list(csv.DictReader((BASE / "report/acquisition_activity.csv").open()))
teacher_summary = []
for arm in ("independent", "positive"):
    for role in sorted({value["role"] for value in teacher_rows}):
        rows = [value for value in teacher_rows if value["arm"] == arm and value["role"] == role]
        teacher_summary.append(dict(arm=arm, role=role, horizon=300, episodes=len(rows), local_walking=sum(value["local_walking"] == "True" for value in rows), full_survival=sum(value["full_survival"] == "True" for value in rows), mean_complete_cost=statistics.mean(float(value["complete_cost"]) for value in rows)))
write(OUT / "local_teacher_summary.json", dict(scope="300-control local outcomes; repeated fixed contexts/seeds are not independent task replications", outcomes=teacher_summary, admission_activity=activity))

root_sizes = []
for directory in (OLD, BASE):
    files = [path for path in directory.rglob("*") if path.is_file()]
    root_sizes.append(dict(path=str(directory), availability="LOCAL_ONLY", files=len(files), logical_bytes=sum(path.stat().st_size for path in files), includes="Frozen source, arrays, checkpoints, ledgers, validation failures and engineering evidence; file symlinks count their targets, directory symlinks are not traversed."))
raw_path = OUT / "external_raw_bindings.json.gz"
raw_data = json.dumps(dict(schema="l3-external-evidence-references-1", availability="LOCAL_ONLY", roots=root_sizes, note="Saved result/source bindings; publication does not rehash all large raw inputs or certify their scientific validity.", files=sorted(raw_bindings.values(), key=lambda value: value["path"])), indent=2).encode() + b"\n"
with raw_path.open("xb") as stream:
    with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        zipped.write(raw_data)
assert gzip.decompress(raw_path.read_bytes()) == raw_data

costs = dict(schema="l3-closed-saved-accounting-1", audit_status="FAILED_NO_RERUN", worker_accounting=worker["accounting"], supervisor_combined_cpu_seconds=supervisor["combined_process_tree_cpu_seconds"], supervisor_combined_active_pool_wall_seconds=supervisor["combined_active_pool_wall_seconds"], preparation_cpu_wall_reservation=supervisor["preparation_cpu_wall_reservation"], final_audit=supervisor["independent_audit"], original_validation_audit_neural_rows=52, original_audit_and_report_in_supervisor_antecedents=supervisor["budget_antecedents"], root_sizes=root_sizes, scopes="Worker reservations include acquisition, verification, fitting and standalone evaluations across both roots; logical+uncertain infrastructure counts are explicit. Supervisor CPU includes nested worker/audit/report processes and is not added to those processes again. The original52 engineering-verification forwards and final audit21827 forwards are separate from scientific worker neural-row counters. Publication/review costs are separate engineering overhead.")
write(OUT / "complete_saved_accounting.json", costs)

report_text = """# L3.1: eight-round local MPPI–policy loop

**All eight planned rounds finished; standalone walking remained zero. The final independent audit failed, so the result is complete but uncertified.**

The frozen comparison used standard receding-horizon MPPI (H48/K256), fixed reconstruction coupling α=0.12 versus **α=0 static refitting**, three continued L2 streams, eight fixed student-derived restart contexts, 300-control acquisition episodes, 2,400 retained rows and 2,000 Adam updates per round. Each updated actor participated in the next acquisition; zero walking never cancelled a planned round. A fixed competence rule admitted local demonstrations or reused each slot's previous safe episode.

Coupling produced different experience while the retained local teachers remained competent. Across192 attempted acquisitions per arm, local walking occurred in170 coupled versus160 α=0 episodes, and full local survival in183 versus176. These are300-control outcomes from repeated contexts/seeds, not1,000-control standalone successes. Admission/fallback and actual activity are reported separately. The α=0 control repeatedly refits stationary MPPI experience under fixed planner seeds.

Every final stream/arm/panel had **0/16 walking**: 0/192 final standalone episodes overall. Final full-survival counts are:

| Stream | α=0 primary / confirmation | α=0.12 primary / confirmation |
|---|---|---|
| 0 | 0/16 / 0/16 | 0/16 / 1/16 |
| 1 | 0/16 / 0/16 | 0/16 / 0/16 |
| 2 | 0/16 / 0/16 | 1/16 / 0/16 |

Coupled policies generally lasted longer, but this is retention without walking. Current-teacher fitting error fell; fixed held-trajectory error worsened in all six actors. Coupled student-state gap ratios (final/initial) were1.95,0.50,1.67 across streams; the α=0 ratios were1.42,1.54,2.20. Only one coupled stream contracted this measured gap. The queries use independent MPPI initialized with a zero plan at restored student contexts; they are supporting measurements, not evidence of competent recovery.

Thus updated policies changed later acquisition, but useful feedback into standalone walking was not demonstrated. There is no convergence or superiority claim. Initial actors/Adam states were already arm-specific L2 descendants; eight restarts share one source episode. Equal-compute low-pass cloning/DAgger and second-body controls remain absent.

The original L3.0 failure and L3.1 engineering amendment remain preserved. The final audit stopped after250,589 checks at `Probe manifest teacher torque`: a typed list-versus-array comparison. A saved-only inspection found zero numerical discrepancy at157 anchors, but did not rerun the audit or certify later checks. No tolerance was relaxed and no failure was relabelled as passing.

[Exact results, all curves, joint angles, failed audit, costs and source hashes](../results/local_coupling_multiround_l3_1_001/completion_001/README.md). The next necessary step is a separately bounded audit-only completion; any scientific follow-up should test one failure-linked hypothesis, preserving this negative trajectory.
"""
with DOC.open("x") as stream:
    stream.write(report_text)

readme = """# Completed L3.1 evidence — audit failed

All24 stream-rounds completed. Every final16-reset panel had zero walking; two
coupled episodes survived1,000 controls without meeting walking. The α=0 arm is
**static refitting**, not fresh discovery. See the [one-page result](../../../docs/local_coupling_multiround_l3_1_result.md).

- [All192 final episode results and bindings](final_outcomes_verified_from_saved_results.json)
- [300-control local teacher outcomes](local_teacher_summary.json)
- [Complete saved accounting](complete_saved_accounting.json)
- [Failed independent audit](audit/findings.json), [receipt](audit/receipt.json), [representation explanation only](audit/probe_representation_note.json)
- [Standalone curves](report/standalone_trajectory.png), [teacher curves](report/teacher_trajectory.png), [fitting curves](report/fitting_trajectory.png), [reconstruction](report/reconstruction_trajectory.png), [student-state gap](report/student_state_gap_trajectory.png)
- [Per-episode outcomes](report/standalone_outcomes.csv), [per-panel summaries](report/standalone_summary.csv), [teacher outcomes](report/teacher_outcomes.csv), [every admission/fallback](report/admission_fallback.csv), [activity](report/acquisition_activity.csv), [reconstruction rows](report/reconstruction.csv), [gap endpoints](report/student_gap_endpoints.csv)
- [Every fitting update, gzip](report/fitting_curves.csv.gz), [declared criteria](report/declared_final_criteria.json)
- [Complete report source bindings, gzip](report/continuation_manifest.json.gz), [external raw-array/checkpoint bindings, gzip](external_raw_bindings.json.gz)

Every copied artifact is byte-exact to its saved source. Gzip entries include
the original decoded SHA256 and byte length in `publication_manifest.json`.
Original rendered Markdown is preserved compressed under `report/`; this README
provides working navigation to the compact published artifacts. All24 joint-angle
figures are preserved as `report/joint_angles_r{1..8}_s{0..2}.png`; the original
report marks these unpaired illustrations, including failures.

Large arrays, ledgers, checkpoints and frozen source remain local-only under
`runs/2026-11-04_local_coupling_multiround_l3_001` and
`runs/2026-11-04_local_coupling_multiround_l3_1_001` in the worktree declared in
the external binding index. The earlier capsule files and failures are unchanged.
This publication aggregates saved metadata, verifies copied bytes, and runs no
physics, neural inference, fitting, evaluation or independent audit.
"""
with (OUT / "README.md").open("x") as stream:
    stream.write(readme)
for spec in original_bindings:
    assert binding(Path(spec["path"])) == spec
write(OUT / "publication_manifest.json", dict(schema="l3-completed-capsule-publication-1", source=binding(Path(__file__)), raw_base=str(BASE), existing_capsule_bindings_unchanged=original_bindings, copied_artifacts=copied, derived_artifacts=[binding(path) for path in sorted(OUT.rglob("*")) if path.is_file() and not any(str(path.relative_to(CAPSULE)) == item["capsule"] for item in copied)], document=binding(DOC), external_raw_decoded_sha256=digest(raw_data), external_raw_decoded_bytes=len(raw_data), scientific_status="COMPLETED_ALL_ROUNDS", audit_status="FAILED_UNCERTIFIED", no_audit_rerun=True))
receipt = dict(cpu_seconds=time.process_time() - START_CPU, wall_seconds=time.monotonic() - START_WALL, cpu_limit_seconds=60, output_bytes=sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()), files=sum(path.is_file() for path in OUT.rglob("*")), checkpoint_deserializations=0, physics=0, model_queries=0, neural_forwards=0, policy_updates=0, audit_reruns=0, note="Publication execution only; short preliminary shell/metadata inspection calls are separate engineering overhead.")
write(OUT / "publication_accounting.json", receipt)
print(json.dumps(dict(output=str(OUT), endpoints=endpoint_rows, costs=receipt)))
