# H5.0 round one: audited teacher-competence stop

Independent MPPI walked **12/12** acquisition episodes; selector-only and proposal-guided teachers each walked **11/12**. Each guided arm had one early fall. The frozen protocol finished all 36 acquisitions, then stopped before any child fit, standalone panel or paired-gap episode. Student usefulness remains **unmeasured**. No adaptive loop or second round occurred.

| Arm | Walking | Full survival | Controls | Mean complete cost |
|---|---:|---:|---:|---:|
| Independent |12/12|12/12|12,000|364.580|
| Selector-only |11/12|11/12|11,201|695.370|
| Proposal guidance |11/12|11/12|11,199|692.137|

The independent full audit passed: **1,085,983,492 prediction rows, 103,308 calls and 6 checkpoint loads**, plus source/input, cost, planner-isolation, RNG, outcome and accounting checks. It performed no new physics or training. Three parent warm-ups account for all 6,000 Adam updates; all 6,000 curve points are preserved. The child comparison was not executed.

Read the [research report](../../docs/teacher_proposal_h5_0_round1_results.md), [saved-data tables and curves](report/report.md), [accounting](accounting/accounting.json), [audit receipt](audit/result/receipt.json) and [external catalog](external_catalog.json). The separately recommended [C1.0 protocol](../../docs/student_consequence_c1_protocol.md) is for review only, unimplemented and unlaunched.

## Exact compact evidence

Original configuration, source/input manifests, launch/exit metadata, worker outcomes, parent learning JSON and audit receipts are copied unchanged. `diagnostics.json` binds 84 scalar events and nine mechanism summaries covering 34,400 selections to the complete original ledger hash. The report consumes only saved JSON; it opens no raw arrays/checkpoints and performs no inference or simulation. `capsule.json` binds every copied file. Later publication/validation/Obsidian receipts are preserved in the [closure record](../../docs/validation/teacher_proposal_h5_0_round1_2026-11-01/closure/README.md), rather than rewriting this capsule.

```sh
python scripts/train_mppi_gps.py inspect --run results/teacher_proposal_h5_0_round1 --verify --json
python scripts/diagnostics/report_teacher_proposal.py \
  --run results/teacher_proposal_h5_0_round1 \
  --saved-diagnostics results/teacher_proposal_h5_0_round1/diagnostics.json \
  --out /tmp/walker-h5-report-review-001
```

The report output path must be new. Clean-export validation passed 1,015 tests with one documented skip and one expected failure, and 37 existing CLI/report checks. The closure record adds the actual round-one capsule checks. Frozen scientific source is `ff67495dae33ee43952677d0c0165a43f8d469b0`; the later saved-only reporter is separately bound by its `de1ad2c…` hash. It did not change the worker.

## External evidence and availability

The full run remains on `robopil5090`:

```text
/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_r1_001
```

The three closed raw files total **206,888,154,706 bytes**; the array payload alone is 204,794,440,824 bytes. All six checkpoints, NPZ inputs and the full frozen source remain external with per-file sizes and SHA-256 values. [Audit work files](accounting/external_audit_outputs_catalog.json) also remain external. No raw arrays, checkpoints, complete ledgers, SQLite indexes, journals or videos are copied into Git.

There is **no complete local raw mirror**. The local same-name `runs/` directory is only the original 539-file/10,622,281-byte source/input export. The compact packet/report download was 1,066,760 bytes in one attempt. Full raw hashes are reused from the closed audit; the collector additionally rehashed the original ledger during its single extraction pass. Earlier runs, preparation, approved backup and historical archive bytes remain unchanged.
