# S1.1 fresh evaluation capsule

This new evaluation reuses the exact original S1 fitting artifacts and evaluates all ten fixed actors on two fresh all-actor reset panels. All 320 episodes and 321 closed setup/episode packets completed, with independent Mac mirror acknowledgements and a passing [saved-only audit](audit/receipt.json). Source benefit and provisional competence are separate decisions shown in the [report](report/report.md) and [gates](report/gates.json). These are the same three original training realizations, not new training replications. [Execution protocol](../../docs/student_source_s1_1_execution_protocol.md) · [interpretation](../../docs/student_source_s1_1_results.md).

All fitting metrics and six fixed 100-update loss-bin streams are inherited. New work is native CPU evaluation only: no fitting, reconstruction inference, GPU work, teacher acquisition or MPPI search. [Worker costs](report/cost_accounting.json) and [engineering costs](accounting/engineering_accounting.json) preserve separate inherited-fit, old known-prefix, old unknown-tail and new-evaluation scopes. The [original S1 capsule](../student_source_s1/README.md) remains interrupted with a failed retained-evidence audit. Its native attempts are not pooled here, and fresh testing cannot retroactively complete its screen or constitute a learning gain.

The capsule copies exact configuration/result/input/source manifests, the original closed fit result, native outcome rows, reconstruction summaries, portable curves, figures, audit findings, all packet receipts/publications/ACKs and Mac commits. Bulk raw HDF, ledgers, checkpoints, full loss JSONs and detailed audit registries remain outside Git. [Full worker files](external/full_run_file_manifest.json), [external registry locations and hashes](external/registry_index.json) and [packet index](mirror/commit_index.json) make availability explicit. A capsule rerender uses saved values and existing bins; it adds no scientific calls.

```sh
python scripts/train_mppi_gps.py inspect --run results/student_source_s1_1 --verify
python scripts/diagnostics/report_student_source_s1_1.py --run results/student_source_s1_1 --out /tmp/walker-s1-1-report-FRESH
```

Whole-source differences do not identify multimodality, prove MPPI teaching competence, test adaptive coupling or close the MPPI–policy loop. No retry or follow-up is authorized by publication.
