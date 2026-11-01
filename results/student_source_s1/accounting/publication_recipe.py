"""Publish exact original S1 metadata plus explicitly derived interruption evidence."""

from pathlib import Path
import hashlib, json, shutil, time

ROOT = Path("/Users/jaisel/Documents/mppi-gps")
RUN = ROOT / "runs/2026-10-31_student_source_s1_001"
ART = ROOT / "artifacts" / RUN.name
AUDIT = ART / "audit_interruption_001"
OUT = ROOT / "results/student_source_s1"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    started = time.perf_counter()
    summary = json.loads((AUDIT / "interruption_summary.json").read_text())
    receipt = json.loads((AUDIT / "receipt.json").read_text())
    full = json.loads((ART / "local_full_run_file_manifest.json").read_text())
    assert (
        not receipt["passed"]
        and summary["audit_complete"]
        and summary["integrity_status"] == "incomplete_retained_evidence"
        and not summary["execution_complete"]
    )
    assert not (RUN / "result.json").exists() and not (RUN / "owner_exit.json").exists()
    assert (ART / "local_full_run_file_manifest.json").read_bytes() == (
        ART / "remote_full_run_file_manifest.json"
    ).read_bytes()
    full_audit = json.loads((ART / "full_audit_001/receipt.json").read_text())
    assert not full_audit["passed"]
    accounting = dict(
        schema="walker-student-source-s1-interrupted-accounting-v1",
        closed_fit_work=json.loads((RUN / "fit_result.json").read_text())["work"],
        interruption_summary_work=summary.get("work", {}),
        interruption_accounting=summary.get("accounting", {}),
        original_full_audit=full_audit,
        scoped_prefix_audit=receipt,
        report=json.loads((ART / "report_001/receipt.json").read_text()),
        prelaunch_validation=json.loads(
            (ART / "engineering/prelaunch_clean_export/final_summary.json").read_text()
        ),
        prelaunch_ci=json.loads(
            (ART / "engineering/prelaunch_ci_final/stdout.txt").read_text()
        ),
        inherited_evidence=dict(
            direct_lineage=[
                "2026-10-31_transfer_diagnosis_d1d2_001",
                "2026-10-31_action_compatibility_k1_001",
                "2026-10-31_student_update_w2_001",
            ],
            selected_teacher_episodes=12,
            training_rows_per_source=4000,
            held_rows_per_source=2000,
            scope="Historical acquisition, reference training and earlier fits stay charged in their own ledgers/capsules. Shared ancestors are referenced, not summed twice; no lifetime efficiency claim.",
        ),
        preservation=dict(
            original_run_files=full["file_count"],
            original_run_bytes_per_mirror=full["total_bytes"],
            initial_export_file_bytes=49074278,
            closed_fit_transfer_file_bytes=39794613,
            interrupted_original_additional_transfer_file_bytes=1978250256,
            derived_container_bytes_per_mirror=1761505518,
            derived_copy_appended_zero_bytes=14352,
            scope="Original and derived containers are separately SHA-bound. Dataset values are accepted only after independent saved hashes verify. The raw original remains truncated. Audit/report/validation/Git copies are separate preservation overhead.",
        ),
        engineering=dict(
            scope="Synthetic tests, source/input validation, audits, recovery copying, plotting, metadata inspection, hashing, transfers, CI and notes are separate engineering overhead. No real S1 fits or native controls were retried.",
            known_prelaunch_tests="455 passed,1 skipped,1 xfailed; final refreshed audit/launch12 passed.",
            global_synthetic_neural_physics_counts=None,
            receipts="engineering/ retains commands, stdout, stderr, durations and failures, including SSH timeout and original full-audit refusal. One shell-glob transfer invocation failed before starting any command and was corrected.",
        ),
        unknowns=[
            "Work after the last durable evaluation record and before reboot cannot be certified as an exact full census.",
            "Owner/CPU exit receipts and a final worker result are absent.",
            "Host reboot cause is not established.",
            "Global synthetic fixture arithmetic, early engineering/reasoning time, peak hardware resources, energy, billing and CI/cloud retention footprint are unmetered.",
        ],
        no_efficiency_claim=True,
    )
    write(ART / "engineering_accounting.json", accounting)
    OUT.mkdir(exist_ok=False)
    bindings = {}

    def copy(p, name):
        q = OUT / name
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        assert sha(p) == sha(q)
        bindings[name] = dict(
            source=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            sha256=sha(q),
            bytes=q.stat().st_size,
        )

    for name in [
        "config.json",
        "manifest.json",
        "input_manifest.json",
        "fit_result.json",
        "prelaunch_receipt.json",
        "launch_receipt.json",
        "fit_start.json",
        "fit_exit.json",
        "eval_start.json",
        "owner_started.json",
        "fit.stdout",
        "fit.stderr",
        "eval.stdout",
        "eval.stderr",
        "owner.stdout",
        "owner.stderr",
    ]:
        copy(RUN / name, "run/" + name)
    for name in ["inventory.json", "task.json", "seed_audit.json"]:
        copy(RUN / "inputs" / name, "run/inputs/" + name)
    copy(AUDIT / "interruption_summary.json", "run/interruption_summary.json")
    copy(AUDIT / "receipt.json", "run/receipt.json")
    copy(ART / "ledger_claims_v2.json", "run/ledger_claims.json")
    copy(ART / "local_full_run_file_manifest.json", "run/full_run_file_manifest.json")
    for p in sorted((ART / "report_001").iterdir()):
        if p.is_file():
            copy(p, "report/" + p.name)
    for folder, destination in [
        (AUDIT, "audit"),
        (ART / "full_audit_001", "full_audit"),
    ]:
        for name in ["receipt.json", "findings.json", "file_bindings.json"]:
            if (folder / name).is_file():
                copy(folder / name, destination + "/" + name)
    for name in ["source.py", "native.py", "width.py", "common.py", "compatibility.py"]:
        copy(RUN / "source/src/gps/mppi_gps/audits" / name, "audit_source/" + name)
    copy(
        ROOT / "src/gps/mppi_gps/audits/source_interruption.py",
        "audit_source/source_interruption.py",
    )
    copy(ROOT / "src/gps/mppi_gps/reporting/source.py", "report_source/source.py")
    copy(
        ART / "local_full_run_file_manifest.json",
        "external/full_run_file_manifest.json",
    )
    copy(ART / "mirror_receipt.json", "external/mirror_receipt.json")
    for p in sorted((ART / "recovery_001").iterdir()):
        if p.is_file() and p.suffix != ".h5":
            copy(p, "recovery/" + p.name)
    copy(
        ART / "engineering/interruption_observation_001/stdout.txt",
        "interruption/observation_001.json",
    )
    copy(
        ART / "engineering/interruption_observation_002/stdout.txt",
        "interruption/observation_002.json",
    )
    copy(ART / "engineering_accounting.json", "accounting/engineering_accounting.json")
    copy(
        ART / "analysis_source_manifest.json",
        "accounting/analysis_source_manifest.json",
    )
    for p in sorted((ART / "engineering").rglob("*")):
        if (
            p.is_file()
            and "matplotlib_cache" not in p.parts
            and p.suffix in {".json", ".txt"}
        ):
            copy(p, "accounting/engineering/" + str(p.relative_to(ART / "engineering")))
    copy(Path(__file__), "accounting/publication_recipe.py")
    copy(Path("/tmp/manifest_transfer_run.py"), "accounting/manifest_recipe.py")
    capsule = dict(
        schema="walker-result-capsule-v1",
        study="student_source_s1",
        run_id=RUN.name,
        status="interrupted_host_reboot",
        integrity_status="incomplete_retained_evidence",
        report_requires_failed_audit_display=True,
        scope="Exact original S1 metadata, closed fits and explicitly derived/audited surviving native prefix. The original worker never wrote result.json or exit receipts. This capsule does not claim a completed S1 comparison.",
        files=bindings,
        bulk_evidence=dict(
            local_run=str(RUN),
            remote_host="robopil5090",
            remote_run="/home/jaisel/mppi-gps-workspace/runs/" + RUN.name,
            local_artifacts=str(ART),
            full_run_files=full["file_count"],
            full_run_bytes=full["total_bytes"],
            original_raw_readable=False,
            recovery="Separately preserved zero-extended container; accepted scientific arrays require saved SHA256 verification.",
            omitted=[
                "evidence/fit/raw.h5",
                "evidence/fit/ledger.jsonl",
                "evidence/eval/raw.h5",
                "evidence/eval/ledger.jsonl",
                "source/",
                "inputs/*.npz",
                "inputs/reference.pt",
                "inputs/teachers/",
                "checkpoints/",
                "learning/",
                "recovery_001/eval_recovered_container.h5",
                "audit_interruption_001/array_bindings.json",
            ],
        ),
        implementation_commit="8182926f18f7981dd4f15d13eee000497ba302d6",
        operation_scope="Capsule copies and portable reports add no inference, fitting, native physics or MPPI. Experimental work is separately labeled as closed fitting plus the known evaluation prefix, with an unresolved tail.",
        new_physics=0,
        new_inference=0,
        new_training=0,
        experimental_work=summary.get("work", {}),
        capsule_copy_wall_seconds=time.perf_counter() - started,
    )
    write(OUT / "capsule.json", capsule)
    (OUT / "README.md").write_text("""# S1 interrupted source-comparison capsule

S1 completed its six fixed fits, then the host rebooted during confirmation. The original full auditor refuses completion. The separate postmortem also fails full-prefix integrity: 187 completed episodes verify, eight completion records lack required raw groups, one attempt is censored and 124 roles are absent from the durable prefix. No audit was weakened or relabeled as passing. [Interpretation](../../docs/student_source_s1_results.md) · [frozen protocol](../../docs/student_source_s1_execution_protocol.md) · [forensic report](report/report.md) · [failed prefix audit](audit/receipt.json) · [original full-audit refusal](full_audit/receipt.json) · [proposed S1.1 completion, unlaunched](../../docs/student_source_s1_1_protocol.md).

All original surviving files, including the truncated HDF, are preserved locally and on robopil5090: 328 files / 2,067,119,147 bytes. [Full hashes](external/full_run_file_manifest.json) and [mirror proof](external/mirror_receipt.json) bind both copies. `result.json`, `eval_exit.json` and `owner_exit.json` remain absent. The derived `run/interruption_summary.json` is explicitly postmortem evidence, not a fabricated worker result.

The HDF container lost 14,352 trailing bytes. A separate exact copy was zero-extended only to its stored EOF; directory enumeration remains damaged. Direct saved paths can be read, and only dtype/shape/hash-verified arrays are accepted. [Recovery receipt](recovery/recovery_receipt.json) and exact recipe document this operation. No original file was repaired or overwritten, and no post-fall or missing state was invented.

The capsule retains all reconstruction splits, source-bound bins covering every fit update, figures, known native outcomes, partial/missing roles, and work accounting with an unresolved interruption tail. Raw arrays, checkpoints, original full curves and detailed audit registries remain outside Git. Parent studies' local-only availability is unchanged.

```sh
python scripts/train_mppi_gps.py inspect --run results/student_source_s1 --verify
python scripts/diagnostics/report_student_source.py --run results/student_source_s1 --display-failed-audit --out /tmp/walker-s1-report-FRESH
```

The explicit forensic flag displays failed-audit evidence; it does not accept it as a valid complete comparison. Verified native outcomes form a nonrepresentative subset; ledger declarations remain separately labeled unverified. Gates, native source rankings and bootstrap claims are withheld. These commands work without ignored evidence directories and add no model inference or simulation. There was no research retry, resume or automatic follow-up. S1.1 requires separate review and authorization; completed fits need not be repeated.
""")
    print(
        json.dumps(
            dict(
                capsule=str(OUT),
                bound_files=len(bindings),
                bound_bytes=sum(v["bytes"] for v in bindings.values()),
            ),
            indent=2,
        )
    )
