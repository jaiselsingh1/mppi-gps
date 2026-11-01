# D1/D2 exact result capsule

The diagnostic completed once and passed its independent audit. It changes no learner or planner. [Main interpretation](../../docs/transfer_diagnosis_d1d2_results.md) · [frozen protocol](../../docs/transfer_diagnosis_d1d2_protocol.md) · [full report](report/report.md) · [audit receipt](audit/receipt.json) · [next S1 proposal, unlaunched](../../docs/student_source_s1_protocol.md).

All 20 teacher replays had zero recorded numerical error before any substitution. All 240 branches completed their stopping logic: 21,938 native controls, 87,752 substeps; zero new inference, gradients, Adam or MPPI queries. D1 completed **69,701** saved-support queries. Audit: **1,061,961 assertions**, 531 bound files and 289,363 arrays; assertion counts are not task replications.

Support is descriptive. All recorded first-100 portions of the 2k students are inside the calibrated support threshold despite walking failure. D2 establishes single-action effects only under the saved open-loop teacher continuation; its ordering is not a validated teaching score or closed-loop competence ranking.

## What is tracked

Original configuration/results/source and input manifests, launch/exit receipts, nine PNG/PDF figure families, summary and consequence CSVs, independent audit summaries, full external file hashes, cost accounting, and a deterministic 2.74 MB derived-curve cache. The cache contains eight derived D1 plotting fields; it contains no full simulator states, actions, checkpoints or native D2 arrays. All selected original copies are byte-bound by `capsule.json`.

## Full external evidence

The complete run is **544 files / 898,176,991 bytes** locally at `/Users/jaisel/Documents/mppi-gps/runs/2026-10-31_transfer_diagnosis_d1d2_001` and on `robopil5090` at `/home/jaisel/mppi-gps-workspace/runs/2026-10-31_transfer_diagnosis_d1d2_001`. Both full manifests are byte-identical. [File hashes](external/full_run_file_manifest.json) · [mirror proof](external/mirror_receipt.json).

Full reports, every raw-row CSV and the 87 MB audit-array registry are under the corresponding local `artifacts/2026-10-31_transfer_diagnosis_d1d2_001/` directory; the independent original audit is also retained in the remote workspace's artifacts directory. The [artifact manifest](external/artifact_file_manifest.json) binds all full report/audit outputs; the report receipt also lists exact omitted-table hashes. Temporal/contributor tables can be regenerated from the tracked derived cache. Native arrays, ledgers, checkpoints and frozen source stay outside Git. Original discovery's missing local terminal/context fields remain explicitly missing and are not used for D2.

## Clean-checkout commands

```sh
python scripts/train_mppi_gps.py inspect --run results/transfer_diagnosis_d1d2 --verify
python scripts/diagnostics/report_transfer_diagnosis.py --run results/transfer_diagnosis_d1d2 --out /tmp/walker-transfer-report-fresh-001
```

Portable rendering verifies metadata/cache hashes and reproduces the nine figure families and summary/temporal tables without the ignored evidence directories. Full raw-row CSVs and a full independent scientific audit require the external run. [Research and engineering accounting](accounting/engineering_accounting.json) separates those operations and records unavailable overhead rather than assigning it zero.
