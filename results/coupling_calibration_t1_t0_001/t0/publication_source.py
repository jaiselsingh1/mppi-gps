"""Publish completed T0 saved evidence; no controller, inference or fitting."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import resource
import tempfile
import time
from pathlib import Path

START_CPU, START_WALL = time.process_time(), time.monotonic()
resource.setrlimit(resource.RLIMIT_CPU, (35, 40))
os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT / "runs/2026-11-04_coupling_calibration_t1_t0_001"
REPORT = RAW / "t0_report_001"
DOC = ROOT / "docs/coupling_controller_t0_result.md"


def binding(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


copied = []


def copy(source, name, compressed=False):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = source.read_bytes()
    with target.open("xb") as stream:
        if compressed:
            with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as archive:
                archive.write(raw)
        else:
            stream.write(raw)
    assert (gzip.decompress(target.read_bytes()) if compressed else target.read_bytes()) == raw
    copied.append(dict(source=binding(source), capsule=binding(target), gzip=compressed, decoded_sha256=hashlib.sha256(raw).hexdigest(), decoded_bytes=len(raw)))


supervisor = json.loads((RAW / "t0/result.json").read_text())
report = json.loads((REPORT / "result.json").read_text())
verification = json.loads((RAW / "verification/result.json").read_text())
assert supervisor["status"] == "COMPLETED" and len(supervisor["records"]) == 20
assert report["status"] == "COMPLETED" and len(report["episodes"]) == 16 and len(report["pairs"]) == 24
assert report["decision"]["effective"] and report["decision"]["destabilizing"]
assert verification["status"] == "PASSED_SAVED_ONLY_CROSS_CHECK"
assert verification["decision"]["effective"] and verification["decision"]["destabilizing"]
for name in ("launch.json", "environment.json", "source_manifest.json"):
    copy(RAW / name, name)
copy(RAW / "t0/result.json", "supervisor_result.json")
for path in sorted(REPORT.iterdir()):
    if path.is_file():
        compressed = path.stat().st_size > 256 * 1024
        copy(path, "report/" + path.name + (".gz" if compressed else ""), compressed)
for path in sorted((RAW / "verification").iterdir()):
    if path.is_file():
        copy(path, "verification/" + path.name)
copy(ROOT / "src/gps/mppi_gps/reporting/coupling_controller_check.py", "source/saved_reporter.py")
copy(ROOT / "results/sac_diagnostics_review_001/source/scripts/diagnostics/action_spectrum.py", "source/preserved_action_spectrum.py")
launch = json.loads((RAW / "launch.json").read_text())
copy(Path(launch["config"]["path"]), "protocol.json")
copy(Path(launch["recipe"]["path"]), "launch_recipe.json")
copy(Path(launch["calibration"]["path"]), "calibration_result.json")
copy(Path(launch["inventory"]["path"]), "input_inventory.json")
for index in range(4):
    copy(RAW / f"prepared/context_{index:03d}.npz", f"inputs/context_{index:03d}.npz")

raw_references = {}
all_manifests, all_receipts = [], []


def collect(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            raw_references[value["path"]] = value
        for child in value.values():
            collect(child)
    elif isinstance(value, list):
        for child in value:
            collect(child)


collect(report)
collect(launch)
for index, entry in enumerate(supervisor["records"]):
    assert binding(Path(entry["manifest"]["path"])) == entry["manifest"]
    manifest_path = Path(entry["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    all_manifests.append((entry, manifest))
    collect(manifest)
    receipt = json.loads((manifest_path.parent / "receipt.json").read_text())
    assert receipt["status"] == "COMPLETED"
    all_receipts.append(receipt)
    target = f"jobs/{index:02d}_{manifest_path.parent.name}"
    copy(Path(entry["job"]["path"]), target + "/job.json")
    copy(manifest_path, target + "/manifest.json")
    copy(manifest_path.parent / "receipt.json", target + "/receipt.json")

bank_rows = list(csv.DictReader((REPORT / "candidate_banks.csv").open()))
scope_rows = []
for entry, manifest in all_manifests:
    if entry["kind"] == "controller":
        rows = [row for row in bank_rows if int(row["index"]) == entry["index"] and row["mode"] == entry["mode"] and float(row["alpha"]) == entry["alpha"]]
        first = rows[0]
        scope_rows.append(dict(kind="controller", index=entry["index"], mode=entry["mode"], alpha=entry["alpha"], banks=len(rows), first_pass_candidate_survival_fraction=float(first["survival_fraction"]), all_banks_equal_weight_mean_candidate_survival_fraction=float(np.mean([float(row["survival_fraction"]) for row in rows])), first_pass_fallen_weight=float(first["fallen_weight"]), all_banks_equal_weight_mean_fallen_weight=float(np.mean([float(row["fallen_weight"]) for row in rows])), first_pass_ESS=float(first["ess"]), all_banks_equal_weight_mean_ESS=float(np.mean([float(row["ess"]) for row in rows]))))
    else:
        with h5py.File(manifest["bank_stats"]["path"], "r") as source:
            rows = []
            for group in source.values():
                weights, survival = group["weights"][:], group["full_survival"][:]
                rows.append(dict(survival=float(survival.mean()), fallen=float(weights[~survival].sum()), ess=float(1 / np.sum(weights**2))))
        scope_rows.append(dict(kind="shadow_preparation_only", index=entry["index"], mode="predecessor_zero_query", alpha=0., banks=len(rows), first_pass_candidate_survival_fraction=rows[0]["survival"], all_banks_equal_weight_mean_candidate_survival_fraction=float(np.mean([row["survival"] for row in rows])), first_pass_fallen_weight=rows[0]["fallen"], all_banks_equal_weight_mean_fallen_weight=float(np.mean([row["fallen"] for row in rows])), first_pass_ESS=rows[0]["ess"], all_banks_equal_weight_mean_ESS=float(np.mean([row["ess"] for row in rows]))))
with (OUT / "candidate_scope_summary.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(scope_rows[0]))
    writer.writeheader()
    writer.writerows(scope_rows)

alpha_pairs = [row for row in report["pairs"] if row["contrast"] == "alpha_same_initialization"]
first_range = (min(row["first_action_rms"] for row in alpha_pairs), max(row["first_action_rms"] for row in alpha_pairs))
prefix_range = (min(row["common_prefix_action_rms"] for row in alpha_pairs), max(row["common_prefix_action_rms"] for row in alpha_pairs))
episodes = report["episodes"]
summary_pairs = []
for index in range(4):
    for mode in ("zero", "shadow_carried"):
        values = [row for row in episodes if row["index"] == index and row["mode"] == mode]
        independent = next(row for row in values if row["alpha"] == 0)
        coupled = next(row for row in values if row["alpha"] > 0)
        pair = next(row for row in alpha_pairs if row["index"] == index and row["left_mode"] == mode)
        summary_pairs.append(dict(index=index, mode=mode, alpha0_cost=independent["complete_cost"], coupled_cost=coupled["complete_cost"], alpha0_controls=independent["controls"], coupled_controls=coupled["controls"], alpha0_local_walking=independent["local_walking"], coupled_local_walking=coupled["local_walking"], alpha0_survival=independent["full_survival"], coupled_survival=coupled["full_survival"], first_action_rms=pair["first_action_rms"], common_prefix_action_rms=pair["common_prefix_action_rms"], alpha0_policy_gap=independent["mean_teacher_policy_action_mse"], coupled_policy_gap=coupled["mean_teacher_policy_action_mse"]))
assert all(row["coupled_policy_gap"] < row["alpha0_policy_gap"] for row in summary_pairs)
with (OUT / "eight_alpha_pairs.csv").open("x", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(summary_pairs[0]))
    writer.writeheader()
    writer.writerows(summary_pairs)

plot_cpu = time.process_time()
with tempfile.TemporaryDirectory(prefix="t0_figure_") as cache:
    os.environ["MPLCONFIGDIR"] = cache
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(8)
    labels = [f"{row['index']}/{'zero' if row['mode'] == 'zero' else 'shadow'}" for row in summary_pairs]
    axes[0].bar(x - .18, [row["alpha0_cost"] for row in summary_pairs], .36, label="alpha0")
    axes[0].bar(x + .18, [row["coupled_cost"] for row in summary_pairs], .36, label="coupled")
    axes[0].set(ylabel="Complete local cost", title="One coupled shadow episode falls")
    axes[1].bar(x - .18, [row["first_action_rms"] for row in summary_pairs], .36, label="first action")
    axes[1].bar(x + .18, [row["common_prefix_action_rms"] for row in summary_pairs], .36, label="common prefix")
    axes[1].axhline(.01, color="black", lw=1, ls="--")
    axes[1].set(ylabel="Matched executed-action RMS", title="Effective intervention in all8 pairs")
    for alpha, offset, label in ((0., -.18, "alpha0"), (31.734901633936317, .18, "coupled")):
        values = [next(row["all_banks_equal_weight_mean_fallen_weight"] for row in scope_rows if row["kind"] == "controller" and row["index"] == pair["index"] and row["mode"] == pair["mode"] and row["alpha"] == alpha) for pair in summary_pairs]
        axes[2].bar(x + offset, values, .36, label=label)
    axes[2].set(ylabel="Mean per-bank fallen weight", title="Supporting; visited states differ")
    for axis in axes:
        axis.set_xticks(x, labels, rotation=60, ha="right")
        axis.legend(frameon=False, fontsize=8)
        axis.grid(axis="y", alpha=.2)
    figure.tight_layout()
    figure.savefig(OUT / "controller_contrast.png", dpi=140)
    plt.close(figure)
plot_cpu = time.process_time() - plot_cpu
assert plot_cpu <= 10

table = "\n".join(f"| {row['index']} / {'zero' if row['mode'] == 'zero' else 'shadow'} | {row['alpha0_cost']:.3f}→{row['coupled_cost']:.3f} | {row['alpha0_controls']}→{row['coupled_controls']} | {int(row['alpha0_local_walking'])}→{int(row['coupled_local_walking'])} | {row['common_prefix_action_rms']:.4f} |" for row in summary_pairs)
text = f"""# T0: effective coupling, with a local competence regression

**The calibrated score changes the executed controller, but fails the frozen safety-of-performance criterion. Complete this block and stop: no T3 or fitting.** All16 episodes and four shadow preparations finished; the saved-only independent cross-check passed.

We compared α=0 against the frozen α=31.734901633936317 in standard H48/K256 MPPI, using the same final L3 s0 coupled actor, four fixed current-student contexts, paired innovations and zero/shadow initial nominals. Each episode allowed300 controls.

| Restart / initialization | Complete cost, α0→coupled | Controls | Local walking | Common-prefix action RMS |
|---|---|---|---|---:|
{table}

Coupling is effective in **8/8** matched comparisons: common-prefix action RMS spans {prefix_range[0]:.6f}–{prefix_range[1]:.6f}. First-action RMS spans {first_range[0]:.6f}–{first_range[1]:.6f}, exceeding .01 in7/8. All24 condition pairs are preserved without post-fall padding. Innovation/RNG fingerprints match across all four conditions' common bank prefixes; same-initialization alpha pairs have byte-identical first-pass sampled actions, states and native costs. Later candidate clouds can differ after their nominal updates.

The decisive regression is **restart1/shadow**: α0 walked all300 controls at cost149.470615; coupling fell after225 at complete cost652.606164 (native277.606164, including one20-unit fall terminal, plus375 missing-control tail). Its mean executed policy-gap MSE nevertheless decreased, .257445→.213238. All eight coupled episodes have lower gap MSE, so agreement improvement does not establish useful teaching. Cold-start coupling raises task cost in4/4 cases. Shadow restart2 improves local walking while restart1 loses it: equal marginal walking counts7/8 hide that paired regression. Full survival drops8/8→7/8.

These are **300-control local teacher outcomes**, not1,000-control standalone-policy competence or learning gains. Shadow preparations paid extra preceding-state MPPI queries; the actual student action, not the teacher recommendation, connects to the target state. Case0 overlaps T1's calibration episode/state; cases1–3 are outside its bank roster, not a confirmation panel.

[Candidate summaries](../results/coupling_calibration_t1_t0_001/t0/candidate_scope_summary.csv) separate first-pass survival fraction from the equal-bank mean over subsequent controller banks, and sample fractions from fallen weight; shadow-preparation banks are separate. No50% candidate-survival gate is used. Spectra/10-neighbor R² are descriptive only: separate episodes, fixed W2 normalization,5Hz bands, burn0, shortened Welch windows and whole-restart-held-out controls differ from K1; negative/missing values remain visible. They do not prove learnability or admit data.

[All outcomes, failure traces' bindings,24 pairs, independent verification, figure and cost appendix](../results/coupling_calibration_t1_t0_001/t0/README.md). No new architecture, score tuning, teacher search, neural inference, fitting or L3-audit closure was added during reporting.
"""
with DOC.open("x") as stream:
    stream.write(text)

counts = {}
for receipt in all_receipts:
    for key, value in receipt["counts"].items():
        counts[key] = counts.get(key, 0) + value
raw_size = sum(path.stat().st_size for path in (RAW / "t0").rglob("*") if path.is_file())
write(OUT / "cost_appendix.json", dict(native_supervisor=supervisor, sum_closed_job_work=counts, raw_T0_bytes=raw_size, source_export=launch["source"], source_export_manifest=launch["source_manifest"], source_commit=launch["source_commit"], report_work=report["work"], verification_resource_receipt=json.loads((RAW / "verification/resource_receipt.json").read_text()), accounting_scope="Native supervisor sums20 child CPU receipts, including four paid shadow preparations and one-control source transition verifications. Parent supervisor CPU, report, independent check and publication are separate; no nested receipts are added twice."))
write(OUT / "external_raw_bindings.json", dict(availability="LOCAL_ONLY", root=str(RAW), raw_T0_bytes=raw_size, source_export=launch["source"], files=sorted(raw_references.values(), key=lambda row: row["path"]), note="Large H5 traces, candidate banks, arrays and checkpoints remain at bound local paths; the capsule preserves compact metadata and saved hashes."))
readme = """# T0 completed controller contrast — stop after regression

[One-page result](../../../docs/coupling_controller_t0_result.md) · [Figure](controller_contrast.png).

All16 controller episodes and four shadow preparations completed. Coupling changed
actions in8/8 pairs but caused one walking/survival regression; no T3 or fitting.

- [All16 outcomes](report/episodes.csv), [eight alpha pairs](eight_alpha_pairs.csv), [all24 condition pairs](report/pair_differences.csv)
- [Pairing proofs](report/innovation_pairing.csv), [first-pass cloud identity](report/first_cloud_identity.csv), [report result](report/result.json)
- [All candidate-bank metrics, gzip](report/candidate_banks.csv.gz), [first-bank/all-bank scope summary](candidate_scope_summary.csv)
- [Per-episode spectra](report/spectra.csv), [held-trajectory neighbor R²](report/neighbour_r2.csv), [spectral arrays](report/spectral_arrays.npz)
- [Independent saved-only verification](verification/result.json), [verification source](verification/verification_source.py), [external resource receipt](verification/resource_receipt.json)
- [Exact protocol](protocol.json), [launch](launch.json), [frozen source manifest](source_manifest.json), [environment](environment.json), [calibration result](calibration_result.json)
- [Closed native accounting](cost_appendix.json), [external raw bindings](external_raw_bindings.json), [input inventory](input_inventory.json)

Every job's exact manifest, job and receipt is retained under `jobs/`, including
the failed225-control physical episode (a closed scientific outcome). Large
H5/full-bank arrays and actor checkpoints stay outside Git at explicit local
paths/hashes. Gzip preserves decoded SHA256 in the publication manifest.
The independent check covers these saved T0 equations/bindings; it does not
repair or close the earlier L3 audit. No reporting step runs physics, neural
inference, fitting, controller evaluations or a T3 prototype.
"""
with (OUT / "README.md").open("x") as stream:
    stream.write(readme)
write(OUT / "publication_manifest.json", dict(schema="coupling-controller-t0-publication-1", source=binding(Path(__file__)), source_report=binding(REPORT / "result.json"), copied_exact=copied, document=binding(DOC), derived_artifacts=[binding(OUT / name) for name in ("candidate_scope_summary.csv", "eight_alpha_pairs.csv", "controller_contrast.png", "cost_appendix.json", "external_raw_bindings.json", "README.md")]))
receipt = dict(cpu_seconds=time.process_time() - START_CPU, wall_seconds=time.monotonic() - START_WALL, cpu_limit_seconds=40, figure_cpu_seconds=plot_cpu, output_bytes=sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()), physics=0, neural_forwards=0, policy_updates=0, T3_executions=0, L3_audit_reruns=0)
write(OUT / "publication_accounting.json", receipt)
print(json.dumps(dict(output=str(OUT), first_action_RMS=first_range, common_prefix_RMS=prefix_range, first_exceeds_01=sum(row["first_action_rms"] > .01 for row in alpha_pairs), costs=receipt, document_words=len(text.split()))))
