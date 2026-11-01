# K1.1 temporal agreement review

Complete saved-only analysis of all 34 K1 trajectories, 30,070 controls and 304 bins. Some near-equal full scores conceal opposing local differences. This does not establish phase, recurrence, consequential errors, teachability or useful coupling.

- [Result and causal verdict](../../docs/trajectory_review_k1_1_results.md); [frozen protocol](../../docs/trajectory_review_k1_1_protocol.md).
- [Figures and seven complete tables](report/report.md); [frozen result](run/result.json).
- [Passed independent audit](audit_002/receipt.json); [original failed audit](audit_001/receipt.json).
- [Complete accounting](accounting/accounting.json), [prelaunch publication](publication/prelaunch_publication.json), [corrected tools source](analysis_source_002/source_manifest.json).
- [Next learning hypothesis, unlaunched](../../docs/student_visitation_s1_2_hypothesis.md).

This capsule contains the entire new run, including exact nine saved inputs and frozen source. No raw arrays/checkpoints/videos were generated or consumed by this analysis. Existing K1 bulk evidence remains [external](../../docs/evidence_availability.md); original capsules and archives are unchanged. The original checker/report and the separate corrected versions are retained. Do not execute a worker inside this closed evidence directory.

Portable read-only inspection and plotting:

```sh
python -B scripts/train_mppi_gps.py inspect --run results/trajectory_review_k1_1 --verify --json
python -B scripts/diagnostics/report_trajectory_review.py --run results/trajectory_review_k1_1 --out /absolute/fresh/external/report
```

A new analysis must be prepared at a fresh path using the frozen recipe and explicitly located original inputs; see the protocol. The auditor recomputes saved metadata only and cannot certify unsaved motions or replay networks. Later publication-validation receipts live separately so this capsule remains immutable.
