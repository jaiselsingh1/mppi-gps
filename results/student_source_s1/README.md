# S1 interrupted source-comparison capsule

S1 completed its six fixed fits, then the host rebooted during confirmation. The original full auditor refuses completion. The separate postmortem also fails full-prefix integrity: 187 completed episodes verify, eight completion records lack required raw groups, one attempt is censored and 124 roles are absent from the durable prefix. No audit was weakened or relabeled as passing. [Interpretation](../../docs/student_source_s1_results.md) · [frozen protocol](../../docs/student_source_s1_execution_protocol.md) · [forensic report](report/report.md) · [failed prefix audit](audit/receipt.json) · [original full-audit refusal](full_audit/receipt.json) · [proposed S1.1 completion, unlaunched](../../docs/student_source_s1_1_protocol.md).

All original surviving files, including the truncated HDF, are preserved locally and on robopil5090: 328 files / 2,067,119,147 bytes. [Full hashes](external/full_run_file_manifest.json) and [mirror proof](external/mirror_receipt.json) bind both copies. `result.json`, `eval_exit.json` and `owner_exit.json` remain absent. The derived `run/interruption_summary.json` is explicitly postmortem evidence, not a fabricated worker result.

The HDF container lost 14,352 trailing bytes. A separate exact copy was zero-extended only to its stored EOF; directory enumeration remains damaged. Direct saved paths can be read, and only dtype/shape/hash-verified arrays are accepted. [Recovery receipt](recovery/recovery_receipt.json) and exact recipe document this operation. No original file was repaired or overwritten, and no post-fall or missing state was invented.

The capsule retains all reconstruction splits, source-bound bins covering every fit update, figures, known native outcomes, partial/missing roles, and work accounting with an unresolved interruption tail. Raw arrays, checkpoints, original full curves and detailed audit registries remain outside Git. Parent studies' local-only availability is unchanged.

```sh
python scripts/train_mppi_gps.py inspect --run results/student_source_s1 --verify
python scripts/diagnostics/report_student_source.py --run results/student_source_s1 --display-failed-audit --out /tmp/walker-s1-report-FRESH
```

The explicit forensic flag displays failed-audit evidence; it does not accept it as a valid complete comparison. Verified native outcomes form a nonrepresentative subset; ledger declarations remain separately labeled unverified. Gates, native source rankings and bootstrap claims are withheld. These commands work without ignored evidence directories and add no model inference or simulation. There was no research retry, resume or automatic follow-up. S1.1 requires separate review and authorization; completed fits need not be repeated.
