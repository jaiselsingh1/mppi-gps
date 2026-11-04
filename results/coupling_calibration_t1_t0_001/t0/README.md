# T0 completed controller contrast — stop after regression

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
