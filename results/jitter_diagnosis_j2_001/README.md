# J2.0: audited clean-control stop

The [maintained result](../../docs/jitter_diagnosis_j2_results.md) explains the
closed qualification failure and the separate exploratory six-student finding.
The [frozen protocol](../../docs/jitter_diagnosis_j2_protocol.md) remains unchanged.

Six clean-data children walked **5, 15, 32, 30, 30 and 31 /32**. The frozen clean
gate failed, so none of the other 72 fits ran and the primary comparison is
missing. Three children meet the two-panel competence convention. The reference
walked 32/32. The independent audit passed without repeating physics or fitting.

- [Worker result](run/result.json), [configuration](run/config.json) and
  [actual work](run/work.json).
- [Independent audit](audit/audit_result.json), [case table](report/cases.csv),
  [all 60,000 curve points](report/learning_curves.csv.gz) and
  [saved-only report](report/report.md).
- [Complete costs](accounting/complete_costs.json), actual
  [worker exit](accounting/operator_worker_001/exit.json) and
  [audit exit](accounting/operator_audit_001/exit.json).
- [Exploratory within-clean analysis](posthoc/summary.json),
  [paired reference gaps](posthoc/paired_gap.json) and [figure](posthoc/within_clean.png).
- [External evidence](external_evidence.json) lists local raw arrays,
  checkpoints, tapes and full contexts. They are outside Git; no remote backup
  is asserted.

The original verbose fitting JSON is preserved as lossless `fit.json.gz` files.
The reporter verifies the decoded original SHA against the audit before using
them. A clean checkout can replay the report without ignored evidence:

```sh
python -m src.gps.mppi_gps.reporting.jitter_diagnosis \
  --worker results/jitter_diagnosis_j2_001/run \
  --audit results/jitter_diagnosis_j2_001/audit/audit_result.json \
  --out /tmp/j2-report-fresh
```

`context/` preserves explicitly provisional parallel-session summaries/source;
their correlations are not independently verified. `source/` distinguishes the
unused first export from the executed `edda434` export. Both engineering import
failures are retained under `accounting/`; neither created a study worker.

The [J2.1 proposal](../../docs/jitter_diagnosis_j2_1_proposal.md) is review-only.
No additional learning, stability acquisition or MPPI run was launched.
