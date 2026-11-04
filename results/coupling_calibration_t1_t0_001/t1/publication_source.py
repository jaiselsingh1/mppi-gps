"""Create-only publication of saved T1 scale calibration; no controller calls."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

START_CPU, START_WALL = time.process_time(), time.monotonic()
START_CHILD_CPU = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
resource.setrlimit(resource.RLIMIT_CPU, (40, 45))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RAW = ROOT / "runs/2026-11-04_coupling_calibration_t1_t0_001"
CONFIG = ROOT / "configs/studies/coupling_calibration_t1_t0_1_0.json"
DOC = ROOT / "docs/coupling_calibration_t1_result.md"
FROZEN_COMMIT = "dc91b92"


def binding(path):
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def copy_exact(source, name):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source.read_bytes()
    with target.open("xb") as stream:
        stream.write(data)
    assert target.read_bytes() == data
    return dict(source=binding(source), capsule=binding(target))


result = json.loads((RAW / "t1/result.json").read_text())
assert result["status"] == "COMPLETED" and binding(CONFIG) == result["config"]
assert result["selected_alpha"] == result["alphas"]["competitive"]["0.3"]
rows = list(csv.DictReader((RAW / "t1/per_bank.csv").open()))
assert len(rows) == 36
selected = [row for row in rows if row["basis"] == "competitive" and row["target"] == "0.3"]
assert len(selected) == 6
copied = []
for name in ("config_binding.json", "result.json", "per_bank.csv", "measurements.npz"):
    copied.append(copy_exact(RAW / "t1" / name, name))
copied.append(copy_exact(CONFIG, "frozen_config.json"))
copied.append(copy_exact(RAW / "prepared/inventory.json", "input_inventory.json"))
copied.append(copy_exact(RAW / "prepared/receipt.json", "input_preparation_receipt.json"))
for path in ("src/gps/mppi_gps/coupling_calibration.py", "src/gps/mppi_gps/bank_review.py"):
    target = OUT / "source" / Path(path).name
    target.parent.mkdir(parents=True, exist_ok=True)
    data = subprocess.check_output(["git", "show", FROZEN_COMMIT + ":" + path], cwd=ROOT)
    with target.open("xb") as stream:
        stream.write(data)
    copied.append(dict(source_commit=FROZEN_COMMIT, source_path=path, capsule=binding(target)))
frozen_config = subprocess.check_output(["git", "show", FROZEN_COMMIT + ":configs/studies/coupling_calibration_t1_t0_1_0.json"], cwd=ROOT)
assert frozen_config == CONFIG.read_bytes()

category_rows = []
with np.load(RAW / "t1/measurements.npz", allow_pickle=False) as measurements:
    for bank in range(6):
        c = measurements[f"b{bank}_C"]
        residual = measurements[f"b{bank}_D"]
        lengths = measurements[f"b{bank}_lengths"]
        survival = measurements[f"b{bank}_full_survival"]
        valid = np.isfinite(c)
        category_rows.append(dict(bank=bank, context=int(selected[bank]["context"]), group=selected[bank]["group"], candidates=len(c), finite_cost_candidates=int(valid.sum()), nonfinite_cost_candidates=int((~valid).sum()), zero_length_candidates=int((lengths == 0).sum()), finite_zero_residual_candidates=int((valid & (residual == 0)).sum()), censored_N_lt48_candidates=int((lengths < 48).sum()), last_control_fall_N_eq48_candidates=int(((~survival) & (lengths == 48)).sum()), full_survival_candidates=int(survival.sum()), minimum_executed_length=int(lengths.min()), maximum_executed_length=int(lengths.max())))
        selected[bank]["remaining_nonfallen_mass"] = 1 - float(selected[bank]["fallen_mass"])
        selected[bank]["baseline_mass_outside_competitive32"] = 1 - float(selected[bank]["competitive_baseline_mass"])
for name, values in (("selected_counterfactual.csv", selected), ("candidate_categories.csv", category_rows)):
    with (OUT / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)

alpha_rows = "\n".join(f"| {target} | {result['alphas']['all'][str(target)]:.17g} | {result['alphas']['competitive'][str(target)]:.17g} |" for target in (0.1, 0.3, 1.0))
selected_rows = "\n".join(
    f"| {row['context']}/p{int(row['group'])} | {row['baseline_top']}→{row['top']} | {float(row['total_variation']):.4g} | {float(row['baseline_ess']):.4f}→{float(row['ess']):.4f} | {float(row['baseline_fallen_mass']):.3g}→{float(row['fallen_mass']):.3g} | {float(row['censored_mass']):.3g} / {float(row['last_control_fall_mass']):.3g} | {row['remaining_nonfallen_mass']:.6f} | {float(row['first_action_rms']):.5g} |"
    for row in selected
)
zero_n = sum(row["zero_length_candidates"] for row in category_rows)
zero_r = sum(row["finite_zero_residual_candidates"] for row in category_rows)
last_step = sum(row["last_control_fall_N_eq48_candidates"] for row in category_rows)
report = f"""# T1: calibrate the mean-prefix reconstruction scale

**The frozen choice is α={result['selected_alpha']:.17g}. It changes some weighted actions, but leaves every top candidate unchanged. This is saved-bank calibration, not executed teacher or learning evidence.**

We used six banks (K256/H48) from three student states at controls 0, 50 and100 of the **same first primary evaluation**, with two same-state refinements each. All bind the final L3 s0 coupled actor. They are six banks, not six independent states; p1 starts from p0's refined nominal without executing a control. Temperature stays3.

For each valid candidate, R=D/N is half-summed torque error divided by its actual executable-prefix length. For each basis, α=q/median_bank[std(R)/std(C)] (population std; equal bank weighting). The competitive basis is the **32 lowest finite task costs**, selected without R, then treated uniformly for spread calibration:

| Target q | α, all finite candidates | α, competitive32 |
|---|---:|---:|
{alpha_rows}

The protocol selected competitive q=.3 **before** execution or fitting. It gives a median achieved spread ratio .3, but individual banks range .219–.521. This data-dependent scale heuristic does not guarantee a strong intervention: competitive32 contains at least99.99963% of baseline softmin mass while ESS is near1. Unweighted spread over32 candidates differs from variation among candidates carrying appreciable weight.

At the selected α, the counterfactual score is C+αR:

| State/pass | Top index | TV | ESS, α0→selected | Fallen mass, α0→selected | Censored / last-step fall mass | Nonfallen remainder | First-action RMS |
|---|---|---:|---|---|---|---:|---:|
{selected_rows}

Top changes: **0/6**. First-action RMS exceeds .01 in3/6 banks, all second refinements; this is not the future executed-controller criterion. Fallen weight increases in3/6 banks, including .00484→.01358 at state100/p0. No teacher-competence or learning benefit follows from these reweightings, and none of these proxies admits or rejects demonstrations.

Zero-length candidates: {zero_n}; finite zero-R candidates: {zero_r}. Last-control falls: {last_step} candidates across banks. A fall at N=48 contributes fallen mass but is **not censored** (N<48); these are reported separately. In `measurements.npz`, the historical key `b*_D` stores normalized R, not the original summed D.

Proceed with exactly **31.734901633936317** for the already-declared T0 comparison because it is the preregistered competitive-.3 choice, not because it produced the largest observed change. T0 must determine actual action intervention and executed survival/task outcomes. No T0 was run for this report. [Exact outputs, all36 rows, source bindings and costs](../results/coupling_calibration_t1_t0_001/t1/README.md).
"""
with DOC.open("x") as stream:
    stream.write(report)
readme = """# T1 saved-bank calibration

[One-page interpretation](../../../docs/coupling_calibration_t1_result.md).
Frozen competitive-.3 choice: **31.734901633936317**. No executed-controller,
standalone walking or learning outcome is measured here.

- [Exact frozen config](frozen_config.json), [result](result.json), [all36 rows](per_bank.csv), [saved weights and measurements](measurements.npz)
- [Selected coefficient, six banks](selected_counterfactual.csv), [zeros/censoring/last-step falls](candidate_categories.csv)
- [Input inventory and source episode/control bindings](input_inventory.json), [input preparation cost](input_preparation_receipt.json)
- [Frozen calibration implementation](source/coupling_calibration.py), [frozen bank update](source/bank_review.py)

All copied outputs preserve their original bytes. Large full candidate banks,
H5 records and checkpoints stay at the explicit local paths and SHA256 bindings
inside the config/inventory. Six banks share three physical states from the first
primary evaluation of one final actor. This publication performs no controller
execution, policy forward, fitting or T0 rollout.

`measurements.npz` uses `b*_D` for the normalized mean-prefix residual R=D/N.
Top32 membership is determined by task cost alone; calibration uses an unweighted
population standard deviation within that subset. Near-unit ESS means an imposed
spread ratio does not ensure a large change to weighted actions. Candidate
proxies are not teacher-competence, demonstration-admission or learning gates.
"""
with (OUT / "README.md").open("x") as stream:
    stream.write(readme)
write(OUT / "publication_manifest.json", dict(schema="coupling-calibration-t1-publication-1", frozen_commit=FROZEN_COMMIT, source=binding(Path(__file__)), copied_exact=copied, derived=[binding(OUT / name) for name in ("selected_counterfactual.csv", "candidate_categories.csv", "README.md")], document=binding(DOC), no_T0_execution=True))
child_cpu = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime - START_CHILD_CPU
receipt = dict(cpu_seconds=time.process_time() - START_CPU, child_cpu_seconds=child_cpu, aggregate_cpu_seconds=time.process_time() - START_CPU + child_cpu, wall_seconds=time.monotonic() - START_WALL, cpu_limit_seconds=45, original_T1_work=result["work"], physics=0, neural_forwards=0, policy_updates=0, T0_executions=0, output_bytes=sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file()), source=binding(Path(__file__)))
write(OUT / "publication_accounting.json", receipt)
print(json.dumps(dict(output=str(OUT), selected_alpha=result["selected_alpha"], zero_length=zero_n, zero_R=zero_r, last_step_falls=last_step, costs=receipt)))
