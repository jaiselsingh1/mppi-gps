"""Create-only portable S1.1 publication; run only after final execution/audit.

This helper reads metadata and hashes/copies bytes. It imports no research code,
performs no inference/physics, and never changes original runs or old capsules.
Every prerequisite is checked before creating the destination. A failed copy
leaves its partial new destination for inspection; it is never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

FORBIDDEN_SUFFIXES = {".h5", ".hdf5", ".pt", ".pth", ".npy", ".npz", ".jsonl", ".mp4", ".pkl"}
RESULT_SCHEMA = "walker-student-source-s1.1-result-v1"
AUDIT_SCHEMA = "walker-student-source-s1.1-independent-audit-v1"
REPORT_SCHEMA = "walker-student-source-s1.1-report-v1"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(path.read_text())


def binding(path):
    need(path.is_file() and not path.is_symlink(), "Expected regular immutable file: " + str(path))
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    need(
        (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "File changed during publication read: " + str(path),
    )
    return dict(sha256=digest.hexdigest(), bytes=after.st_size)


def same(path, expected):
    actual = binding(path)
    need(actual["sha256"] == expected["sha256"], "SHA binding differs: " + str(path))
    if "bytes" in expected:
        need(actual["bytes"] == expected["bytes"], "Size binding differs: " + str(path))
    return actual


def contained(root, name):
    relative = Path(name)
    need(
        not relative.is_absolute() and ".." not in relative.parts,
        "Nonrelative evidence path: " + str(name),
    )
    path = (root / relative).resolve()
    need(path.is_relative_to(root), "Evidence escapes input root: " + str(name))
    return path


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def checked_catalog(path, run_id):
    catalog = load(path)
    need(
        catalog["schema"] == "walker-full-run-file-manifest-v1",
        "Unexpected full-run catalog schema",
    )
    need(catalog["run_id"] == run_id, "Full-run catalog run identity differs")
    rows = catalog["files"]
    names = [row["file"] for row in rows]
    need(
        len(set(names)) == len(rows) == catalog["file_count"],
        "Duplicate or incomplete full-run file census",
    )
    need(
        sum(row["bytes"] for row in rows) == catalog["total_bytes"],
        "Full-run catalog bytes do not sum",
    )
    for name in names:
        need(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Unsafe full-run catalog path",
        )
    return catalog, {row["file"]: row for row in rows}


def prepare(args):
    """Check scientific closure and prepare only exact-copy/derived-metadata plans."""
    root, run, mirror = args.root.resolve(), args.run.resolve(), args.mirror.resolve()
    audit, report, out = args.audit.resolve(), args.report.resolve(), args.out.resolve()
    need(not out.exists(), "Capsule destination already exists: " + str(out))
    need(out.parent == root / "results", "Capsule must be a new direct child of ROOT/results")
    need(out.name == "student_source_s1_1", "This helper publishes only the new S1.1 capsule")
    need(
        run != mirror and not run.is_relative_to(mirror) and not mirror.is_relative_to(run),
        "Worker and off-host mirror must be distinct",
    )
    need(
        re.fullmatch(r"[0-9a-f]{40}", args.implementation_commit) is not None,
        "Expected full implementation commit SHA",
    )
    config, result, seal = [
        load(run / name) for name in ("config.json", "result.json", "input_manifest.json")
    ]
    need(
        args.remote_host == config["platform"]["host"] and Path(args.remote_run).is_absolute(),
        "Worker remote host/run must match the frozen host and an explicit absolute path",
    )
    need(config["schema"] == "walker-student-source-s1.1-v1", "Not the frozen S1.1 recipe")
    need(
        result["schema"] == RESULT_SCHEMA
        and result["execution_complete"] is True
        and result["error"] is None
        and result["status"] == "evaluation_complete",
        "A complete, error-free S1.1 result is required",
    )
    pins = {
        name: binding(run / name) for name in ("config.json", "result.json", "input_manifest.json")
    }
    need(
        seal["config_sha256"] == pins["config.json"]["sha256"],
        "Input seal does not own the configuration",
    )
    need(
        result["config_sha256"] == pins["config.json"]["sha256"]
        and result["input_manifest_sha256"] == pins["input_manifest.json"]["sha256"],
        "Result provenance pins differ",
    )
    inherited = result["inherited_fit"]
    need(
        inherited["sha256"] == config["source_bindings"]["s1_fit_result_sha256"],
        "Inherited closed fitting artifact is not the pinned S1 fit",
    )
    same(contained(run, inherited["file"]), inherited)
    need(
        len(result["points"]) == 9 and len(result["fits"]) == 6 and len(result["segments"]) == 320,
        "Incomplete inherited actors/curves or new episode roster",
    )
    for panel in ("primary", "confirmation"):
        need(
            set(result["evaluations"][panel]) == set(config["evaluation_order"]),
            "Incomplete all-actor panel",
        )
        for record in result["evaluations"][panel].values():
            episodes = record["episodes"]
            need(
                len(episodes) == 16
                and [row["seed"] for row in episodes] == config[panel + "_seeds"]
                and all(row["complete"] is True for row in episodes),
                "Native rows are incomplete or not the frozen fresh panel",
            )
    for name in ("owner_exit.json", "worker_exit.json"):
        exit_record = load(run / name)
        need(exit_record.get("error") is None, "Owner/worker exit records an error")
        if name == "owner_exit.json":
            need(exit_record["execution_complete"] is True, "Owner did not close complete")
        else:
            need(
                exit_record["exit_code"] == 0
                and exit_record["timeout"] is False
                and exit_record["result_sha256"] == pins["result.json"]["sha256"],
                "Worker did not close successfully on this exact result",
            )
    audit_receipt, findings = load(audit / "receipt.json"), load(audit / "findings.json")
    need(
        audit_receipt["schema"] == AUDIT_SCHEMA
        and audit_receipt["passed"] is True
        and audit_receipt["error"] is None,
        "Publication requires a passing full independent S1.1 audit",
    )
    need(
        findings["comparison_complete"] is True
        and findings["new_fits"] == findings["new_reconstruction_rows"] == 0
        and findings["gates"] == result["gates"],
        "Audit findings do not close this exact evaluation-only comparison",
    )
    audit_files = load(audit / "file_bindings.json")
    for name in ("config.json", "result.json", "input_manifest.json"):
        need(
            audit_files.get(str(run / name)) == pins[name],
            "Audit did not bind this exact publication input: " + name,
        )
    report_receipt = load(report / "receipt.json")
    need(
        report_receipt["schema"] == REPORT_SCHEMA
        and report_receipt["result_sha256"] == pins["result.json"]["sha256"]
        and report_receipt["config_sha256"] == pins["config.json"]["sha256"]
        and report_receipt["input_manifest_sha256"] == pins["input_manifest.json"]["sha256"],
        "Report receipt belongs to a different result/configuration/seal",
    )
    need(
        report_receipt.get("failed_audit_display") is False
        and report_receipt["render_scope"] == "full_saved_curves",
        "A canonical complete full-saved-curves report is required",
    )
    for name, expected in report_receipt["outputs"].items():
        same(contained(report, name), expected)
    curves = load(report / "learning_bins_manifest.json")
    need(
        curves["curve_origin"] == "inherited_s1"
        and curves["inherited_fit"] == inherited
        and curves["binned_updates"] == 12000
        and len(curves["source_curves"]) == 6,
        "Portable curves do not cover the six unchanged inherited streams",
    )
    with (report / "learning_bins.csv").open(newline="") as stream:
        bins = list(csv.DictReader(stream))
    need(
        len(bins) == 120 and sum(int(row["updates"]) for row in bins) == 12000,
        "Portable fixed 100-update curve bins are incomplete",
    )
    need(
        binding(args.full_manifest) == binding(args.remote_manifest),
        "Local and remote final full-run catalogs differ",
    )
    catalog, catalog_files = checked_catalog(args.full_manifest, run.name)
    actual_files = {str(path.relative_to(run)) for path in run.rglob("*") if path.is_file()}
    need(
        actual_files == set(catalog_files),
        "Local full-run file roster changed after its final catalog",
    )
    need(
        all((run / name).stat().st_size == row["bytes"] for name, row in catalog_files.items()),
        "Local full-run sizes changed after the final catalog",
    )
    for name in ("config.json", "result.json", "input_manifest.json"):
        need(
            {key: catalog_files[name][key] for key in ("sha256", "bytes")} == pins[name],
            "Full-run catalog does not own published core bytes",
        )
    coordinator = load(mirror / "coordinator_exit.json")
    need(
        coordinator["execution_complete"] is True
        and coordinator["error"] is None
        and len(coordinator["records"]) == 321,
        "Off-host coordinator did not commit all 321 packets",
    )
    plan, derived = {}, {}

    def add(path, name):
        need(
            Path(name).suffix.lower() not in FORBIDDEN_SUFFIXES,
            "Bulk data cannot enter this capsule: " + name,
        )
        need(name not in plan and name not in derived, "Duplicate publication destination: " + name)
        path = path.resolve()
        actual = binding(path)
        if path.is_relative_to(run):
            original = catalog_files.get(str(path.relative_to(run)))
            need(
                original is not None
                and actual == {key: original[key] for key in ("sha256", "bytes")},
                "Copied worker metadata differs from its final catalog: " + str(path),
            )
        if path.is_relative_to(mirror):
            need(
                audit_files.get(str(path)) == actual,
                "Copied mirror metadata differs from the independent audit: " + str(path),
            )
        plan[name] = (path, actual)

    def make(value, name):
        need(name not in plan and name not in derived, "Duplicate derived destination: " + name)
        derived[name] = value if isinstance(value, bytes) else encode(value)

    for name in (
        "config.json",
        "result.json",
        "input_manifest.json",
        "manifest.json",
        "launch.lock",
        "owner_invocation.json",
        "prelaunch_publication.json",
        "launch_receipt.json",
        "worker_started.json",
        "worker_process.json",
        "worker_exit.json",
        "owner_exit.json",
        "worker.stdout",
        "worker.stderr",
    ):
        add(run / name, "run/" + name)
    add(contained(run, inherited["file"]), "run/" + inherited["file"])
    for name in ("inventory.json", "local_seed_audit.json", "remote_seed_audit.json"):
        add(run / "inputs" / name, "run/inputs/" + name)
    for name in ("config.json", "manifest.json", "input_manifest.json", "fit_exit.json"):
        add(run / "inputs/s1" / name, "run/inputs/s1/" + name)
    for name in (
        "summary.json",
        "receipt.json",
        "full_audit_receipt.json",
        "original_run_manifest.json",
    ):
        add(run / "inputs/s1_interruption" / name, "run/inputs/s1_interruption/" + name)
    descriptors = [result["setup"], *result["segments"]]
    packet_index = []
    for ordinal, descriptor in enumerate(descriptors, start=-1):
        relative = descriptor["segment"]
        identifier = "loads" if ordinal == -1 else Path(relative).name
        receipt_path = contained(run, relative + "/receipt.json")
        same(receipt_path, dict(sha256=descriptor["receipt_sha256"]))
        add(receipt_path, "run/" + relative + "/receipt.json")
        for kind in ("request", "ack"):
            same(contained(run, descriptor[kind]["file"]), descriptor[kind])
            add(contained(run, descriptor[kind]["file"]), "run/" + descriptor[kind]["file"])
        commit_path = mirror / "local_coordinator/commits" / (identifier + ".json")
        same(commit_path, dict(sha256=descriptor["mirror_commit_sha256"]))
        add(commit_path, "mirror/local_coordinator/commits/" + identifier + ".json")
        packet_index.append(
            dict(
                ordinal=ordinal,
                identifier=identifier,
                segment=relative,
                receipt_sha256=descriptor["receipt_sha256"],
                request=descriptor["request"],
                ack=descriptor["ack"],
                mirror_commit_sha256=descriptor["mirror_commit_sha256"],
            )
        )
    add(mirror / "coordinator_exit.json", "mirror/coordinator_exit.json")
    make(
        dict(
            schema="walker-s1.1-portable-commit-index-v1",
            packets=packet_index,
            scope="Exact packet receipts, publication requests, worker ACKs and Mac commit files are copied; raw segment bytes remain external and were independently audited.",
        ),
        "mirror/commit_index.json",
    )
    for path in sorted(report.iterdir()):
        if path.is_file() and path.name != "receipt.json":
            need(path.name in report_receipt["outputs"], "Unbound report artifact: " + path.name)
            add(path, "report/" + path.name)
    add(report / "receipt.json", "report/receipt.json")
    for name in ("receipt.json", "findings.json"):
        add(audit / name, "audit/" + name)
    # Large raw-array registries are deliberately referenced, not silently omitted.
    registries = []
    for name in ("file_bindings.json", "array_bindings.json"):
        path = audit / name
        registries.append(
            dict(
                kind=name.removesuffix(".json"),
                location=str(path),
                available_at_publication=True,
                **binding(path),
            )
        )
    audit_history = []
    for earlier in args.prior_audit:
        earlier = earlier.resolve()
        receipt = load(earlier / "receipt.json")
        need(
            earlier != audit and receipt["schema"] == AUDIT_SCHEMA and receipt["passed"] is False,
            "Prior audit inputs must be distinct preserved failed S1.1 audit attempts",
        )
        for name in ("receipt.json", "findings.json"):
            add(earlier / name, "audit_history/" + earlier.name + "/" + name)
        earlier_files = load(earlier / "file_bindings.json")
        for name in ("source_evaluation.py", "native.py", "source.py", "width.py"):
            candidates = [
                Path(path)
                for path in earlier_files
                if path.endswith("/audits/" + name)
                and not Path(path).is_relative_to(run)
                and not Path(path).is_relative_to(mirror)
            ]
            need(len(candidates) == 1, "Cannot identify preserved failed audit module")
            same(candidates[0], earlier_files[str(candidates[0])])
            add(candidates[0], "audit_history/" + earlier.name + "/source/" + name)
        earlier_registries = [
            dict(location=str(earlier / name), **binding(earlier / name))
            for name in ("file_bindings.json", "array_bindings.json")
        ]
        audit_history.append(
            dict(
                directory=str(earlier),
                capsule_receipt="audit_history/" + earlier.name + "/receipt.json",
                registries=earlier_registries,
                scope="Preserved failed saved-only checker attempt; never relabeled as passing",
            )
        )
    add(args.full_manifest, "external/full_run_file_manifest.json")
    add(args.remote_manifest, "external/remote_full_run_file_manifest.json")
    if args.mirror_receipt:
        add(args.mirror_receipt, "external/mirror_receipt.json")
    make(
        dict(
            schema="walker-s1.1-external-registry-index-v1",
            audit=registries,
            prior_failed_audits=audit_history,
            full_run_manifest="external/full_run_file_manifest.json",
            remote_catalog_bytes_equal=True,
            remote_catalog_scope="Caller-supplied local and worker-side final catalogs are exactly equal. This publisher performs no network query; their transfer provenance remains in engineering evidence.",
            original_raw_scope="Complete independently audited new S1.1 segments; original S1 remains interrupted and its unknown tail is not erased.",
        ),
        "external/registry_index.json",
    )
    add(args.accounting, "accounting/engineering_accounting.json")
    if args.engineering:
        engineering = args.engineering.resolve()
        for path in sorted(engineering.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".json", ".txt"}
                and "matplotlib_cache" not in path.parts
            ):
                need(
                    path.stat().st_size <= 4 * 1024**2,
                    "Engineering file needs explicit external indexing instead of bulk copying: "
                    + str(path),
                )
                add(path, "accounting/engineering/" + str(path.relative_to(engineering)))
    for name in ("source_evaluation.py", "native.py", "source.py", "width.py"):
        candidates = [
            Path(path)
            for path in audit_files
            if path.endswith("/audits/" + name)
            and not Path(path).is_relative_to(run)
            and not Path(path).is_relative_to(mirror)
        ]
        need(
            len(candidates) == 1,
            "Cannot identify exact independently executed audit module: " + name,
        )
        same(candidates[0], audit_files[str(candidates[0])])
        add(candidates[0], "audit_source/" + name)
    reporter_source = (
        args.report_source or root / "src/gps/mppi_gps/reporting/source.py"
    ).resolve()
    same(reporter_source, dict(sha256=report_receipt["source_sha256"]))
    add(reporter_source, "report_source/source.py")
    add(Path(__file__), "accounting/publication_recipe.py")
    readme = """# S1.1 fresh evaluation capsule

This new evaluation reuses the exact original S1 fitting artifacts and evaluates all ten fixed actors on two fresh all-actor reset panels. All 320 episodes and 321 closed setup/episode packets completed, with independent Mac mirror acknowledgements and a passing [saved-only audit](audit/receipt.json). Source benefit and provisional competence are separate decisions shown in the [report](report/report.md) and [gates](report/gates.json). These are the same three original training realizations, not new training replications. [Execution protocol](../../docs/student_source_s1_1_execution_protocol.md) · [interpretation](../../docs/student_source_s1_1_results.md).

All fitting metrics and six fixed 100-update loss-bin streams are inherited. New work is native CPU evaluation only: no fitting, reconstruction inference, GPU work, teacher acquisition or MPPI search. [Worker costs](report/cost_accounting.json) and [engineering costs](accounting/engineering_accounting.json) preserve separate inherited-fit, old known-prefix, old unknown-tail and new-evaluation scopes. The [original S1 capsule](../student_source_s1/README.md) remains interrupted with a failed retained-evidence audit. Its native attempts are not pooled here, and fresh testing cannot retroactively complete its screen or constitute a learning gain.

The capsule copies exact configuration/result/input/source manifests, the original closed fit result, native outcome rows, reconstruction summaries, portable curves, figures, audit findings, all packet receipts/publications/ACKs and Mac commits. Bulk raw HDF, ledgers, checkpoints, full loss JSONs and detailed audit registries remain outside Git. [Full worker files](external/full_run_file_manifest.json), [external registry locations and hashes](external/registry_index.json) and [packet index](mirror/commit_index.json) make availability explicit. A capsule rerender uses saved values and existing bins; it adds no scientific calls.

```sh
python scripts/train_mppi_gps.py inspect --run results/student_source_s1_1 --verify
python scripts/diagnostics/report_student_source_s1_1.py --run results/student_source_s1_1 --out /tmp/walker-s1-1-report-FRESH
```

Whole-source differences do not identify multimodality, prove MPPI teaching competence, test adaptive coupling or close the MPPI–policy loop. No retry or follow-up is authorized by publication.
"""
    make(readme.encode(), "README.md")
    total = sum(meta["bytes"] for _, meta in plan.values()) + sum(
        len(data) for data in derived.values()
    )
    need(
        total <= args.max_capsule_bytes,
        "Planned portable files exceed explicit capsule byte ceiling",
    )
    return dict(
        root=root,
        run=run,
        mirror=mirror,
        audit=audit,
        report=report,
        out=out,
        plan=plan,
        derived=derived,
        result=result,
        catalog=catalog,
        planned_bytes=total,
        packet_count=len(descriptors),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Users/jaisel/Documents/mppi-gps"))
    for name in (
        "run",
        "mirror",
        "audit",
        "report",
        "full-manifest",
        "remote-manifest",
        "accounting",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/jaisel/Documents/mppi-gps/results/student_source_s1_1"),
    )
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--engineering", type=Path)
    parser.add_argument("--prior-audit", type=Path, action="append", default=[])
    parser.add_argument("--mirror-receipt", type=Path)
    parser.add_argument("--report-source", type=Path)
    parser.add_argument("--remote-host", default="robopil5090")
    parser.add_argument("--remote-run", required=True)
    parser.add_argument("--max-capsule-bytes", type=int, default=32 * 1024**2)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    prepared = prepare(args)
    out, files = prepared["out"], {}
    out.mkdir(exist_ok=False)
    for name, (source, expected) in prepared["plan"].items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        same(source, expected)
        with source.open("rb") as original, target.open("xb") as copied:
            shutil.copyfileobj(original, copied)
        same(target, expected)
        same(source, expected)
        files[name] = dict(source=str(source), **expected)
    for name, content in prepared["derived"].items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
        files[name] = dict(
            source="Explicit publication metadata; no scientific recomputation", **binding(target)
        )
    result, catalog = prepared["result"], prepared["catalog"]
    capsule = dict(
        schema="walker-result-capsule-v1",
        study="student_source_s1_1",
        run_id=prepared["run"].name,
        status=result["status"],
        integrity_status="independent_complete_segmented_audit_passed",
        scope="Fresh evaluation only of the same original S1 fitting realizations. Exact native outcomes, metadata and portable inherited loss bins; no old S1 native pooling or loop claim.",
        implementation_commit=args.implementation_commit,
        files=files,
        bulk_evidence=dict(
            local_run=str(prepared["run"]),
            local_mirror=str(prepared["mirror"]),
            local_audit=str(prepared["audit"]),
            remote_host=args.remote_host,
            remote_run=args.remote_run,
            full_run_files=catalog["file_count"],
            full_run_bytes=catalog["total_bytes"],
            acknowledged_closed_packets=prepared["packet_count"],
            local_remote_catalog_bytes_equal=True,
            detailed_registries="external/registry_index.json",
            omitted=[
                "evidence/*/raw.h5",
                "evidence/*/ledger.jsonl",
                "evidence/episodes/*/raw.h5",
                "evidence/episodes/*/ledger.jsonl",
                "owner_ledger.jsonl",
                "source/",
                "inputs/s1/checkpoints/",
                "inputs/s1/learning/",
                "inputs/s1/evidence/fit/",
                "inputs/s1/inputs/",
                "audit/array_bindings.json",
                "audit/file_bindings.json",
            ],
        ),
        inherited_work=result["inherited_work"],
        experimental_work=result["work"],
        no_efficiency_claim=True,
        operation_scope="Byte copying/hashing and explicit publication metadata only; no inference, training, physics, MPPI or retries.",
        new_physics=0,
        new_inference=0,
        new_training=0,
        capsule_copy_wall_seconds=time.perf_counter() - started,
    )
    with (out / "capsule.json").open("xb") as stream:
        stream.write(encode(capsule))
    print(
        json.dumps(
            dict(
                capsule=str(out),
                bound_files=len(files),
                bound_bytes=sum(row["bytes"] for row in files.values()),
                acknowledged_closed_packets=prepared["packet_count"],
                new_scientific_calls=0,
                wall_seconds=time.perf_counter() - started,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
