# D1/D2 implementation and publication validation

The scientific worker ran once from published frozen commit
`95f0a55e490204f9ca898b6f8891c56e95e7a824`. Its config, input seal, source,
snapshots, predictions and results remain unchanged. This is diagnostic execution,
not new policy training or loop closure.

## Before launch

- Clean source export without `runs/` or `artifacts/`: **414 passed, one documented
  skip and one expected stock-Gymnasium failure**; Ruff, formatting, CLI and link/
  historical-manifest checks passed.
- [Locked CPU CI](https://github.com/jaiselsingh1/mppi-gps/actions/runs/41259416102)
  passed on the published commit before execution.
- The remote inspector verified 230 source files and all input/config records.
  Preparation imported no policy or simulator and performed no real support-query
  analysis, inference, training or physics before sealing.

[Exact prelaunch receipts](../../../results/transfer_diagnosis_d1d2/accounting/prelaunch/clean_export.json).

## Scientific closure

All 20 baseline continuations passed before substitutions, with maximum numerical
error zero. The fixed 240 branches used 21,938 native controls and 87,752 substeps,
with no failed, rejected or unknown physical work. The frozen independent audit
passed 1,061,961 assertions in 38.328 seconds. It read/recomputed saved evidence,
including 69,701 support queries, without simulator, network or optimizer calls.

[Audit receipt](../../../results/transfer_diagnosis_d1d2/audit/receipt.json) ·
[findings](../../../results/transfer_diagnosis_d1d2/audit/findings.json) ·
[full mirror proof](../../../results/transfer_diagnosis_d1d2/external/mirror_receipt.json).

## Portable publication

The post-launch active reporting update adds a deterministic, hash-bound derived-
curve cache. The metadata checker also accepts sized input bindings alongside the
legacy plain-hash contract. Neither change edits or re-executes the frozen worker.

A second clean export passed **417 tests, one skip and one expected failure**, Ruff,
CLI, seven capsule inspections, documentation/archive checks and actual portable
report rendering. All nine PNGs match the full report bytes; temporal, contributor
and frequency CSVs also match. The capsule includes no full native HDF, training
NPZ, checkpoint, ledger or full-row CSV payload. The source-derived plotting cache
contains eight derived fields only. Full native/registry/row-table evidence remains
available with complete external hashes.

[Final clean-export receipt](../../../results/transfer_diagnosis_d1d2/accounting/publication_validation/clean_export.json) ·
[pytest output](../../../results/transfer_diagnosis_d1d2/accounting/publication_validation/pytest.stdout.txt) ·
[portable render check](../../../results/transfer_diagnosis_d1d2/accounting/portable_render_validation.json) ·
[publication scope](../../../results/transfer_diagnosis_d1d2/accounting/publication_validation.json).

Publication receipts and navigation metadata receive a final separate hash/link/
inspection check after their addition. The final push-triggered CI validates the
published tree; its result is checked on GitHub rather than assumed in a receipt
written before that commit exists. Scientific, engineering, copying, audit and
rendering work are separated in the [cost accounting](../../../results/transfer_diagnosis_d1d2/accounting/engineering_accounting.json);
unmetered overhead is unavailable, not zero. No efficiency claim is made.

## Next decision

The [S1 protocol](../../student_source_s1_protocol.md) and proposal JSON are
unlaunched and unregistered. Saved-only source checks found no raw, float32 or
normalized-input cross-split collisions in the specified data. This is input
feasibility, not a learning result. No S1 worker, fitting, native evaluation,
importance weighting, OT objective or acquisition followed D1/D2 automatically.
