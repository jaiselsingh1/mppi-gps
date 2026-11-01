# K1.2 saved-motion diagnosis

One source-frozen diagnosis of all 34 K1 trajectories, 30,070 controls and 561 cross-trajectory pairs. No new inference, physics, fitting or acquisition ran. The independent audit passed 380,112 checks. Motion and short-window similarity do not establish gait phase, teachability, a transfer cause or useful coupling.

- [Result and causal verdict](../../docs/motion_diagnosis_k1_2_results.md); [frozen protocol](../../docs/motion_diagnosis_k1_2_protocol.md).
- [Figures and complete tables](report/report.md); [full actual-motion atlas](report/full_motion_atlas.pdf); [derived summary atlas](report/derived_curve_atlas.pdf).
- [Frozen result](run/result.json), [independent audit](audit_001/receipt.json) and [accounting](accounting/accounting.json).
- [All-page visual review](visual_qa/inspection_001.json) and [ordered contact sheets](visual_qa/contact_sheets_001/receipt.json).
- [Next H5.0 hypothesis](../../docs/teacher_proposal_h5_0_hypothesis.md), untested and not authorized for launch.

The entire new run is copied here except `evidence/motion.npz`: 6,064,785 bytes, SHA-256 `55337bee1245059c96e208865fa04730364be0267b8489c42558073b16aff783`. The [full run catalog](external/full_run_file_manifest.json) binds that local-only raw archive and all 373 closed-run files. Original K1 raw HDF/NPZ/checkpoints remain external and local-only; no legacy raw arrays were copied to Git. Four discovery endpoints remain missing. Available terminal states do not create terminal actions.

Clean-checkout reporting renders saved block means/min/max and copies the exact bound original full-motion atlas and fixed-pair figures. It explicitly discloses this provenance; it does not reconstruct raw motion from block summaries. All 68 page PNGs remain external with hashes in the visual-review receipts; only compact contact sheets are included here.

```sh
python -B scripts/train_mppi_gps.py inspect --run results/motion_diagnosis_k1_2 --verify --json
python -B scripts/diagnostics/report_motion_diagnosis.py --run results/motion_diagnosis_k1_2 --out /absolute/fresh/external/report
```

The two failed preflight path attempts, independent metadata-check timing limitation, failed deadline test, corrected regression checks, source publication, report marker, rendering and QA costs are preserved. Do not execute workers inside this immutable capsule. Later publication validation stays separate.
