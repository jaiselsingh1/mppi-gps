# Completed coupled-loop design review

[One-page conclusions](../../docs/coupled_loop_design_review.md). All eight rounds completed in all three paired streams. Final standalone walking was **0/192**; full survival was **2/96 coupled and 0/96 static**. The review evaluates saved evidence and counterfactual weights; it does not execute another experiment or certify the original experiment audit.

The final original audit remains **CLOSED_UNCERTIFIED** after a typed JSON-list/NumPy-array teacher-action comparison failed. [Saved representation inspection](raw/probe_representation_note.json) shows zero numerical teacher/student action discrepancy at all 157 observed anchors. It explains that comparison only: unexecuted audit checks remain uncertified. No auditor, tolerance, original failure or experiment input was changed.

| Evidence | Contents |
|---|---|
| [Admission table](admission_table.md) | All 48 arm/stream/round cells: changed, identical, fallback and learner-visible cumulative novelty |
| [Admissions](admissions.csv), [corpus novelty](corpus_novelty.csv) | All 384 episodes and exact raw/float32 row counts |
| [Fall support](fall_support.csv) | 380 queries: terminal and last pre-action boundaries for 190 final falls; missing fall states remain absent for two survivors |
| [Every bank](per_bank.csv.gz), [cell summaries](per_cell.csv) | 112,856 banks; score strength, candidate survival and prefix-normalized counterfactuals |
| [Fit error](fit_mse.csv), [bands and neighbours](spectra.csv) | Both arms, all three streams and eight rounds; reconstruction/held errors, power and variance shares, restart-trajectory-held-out and temporal-exclusion R² |
| [Earlier slot costs](earlier_slot_costs.csv) | Avoided-fall terminal/tail accounting, separate from the new loop |
| [Initialization](cold_start.csv), [probes](probe_initialization.csv) | Zero initial plans and first/second refinement differences; no same-random cold/warm causal contrast |
| [Figure](fit_and_predictability.png), [render source](render_review.py) | Descriptive three-stream means from the exact CSVs |

`raw/` preserves every earlier partial-review output, including its final metadata serialization failure, and the complete final review under `raw/final_snapshot/`. Original script kernels, adapters, configurations, receipts and input bindings are included. The spectrum adapter changes only declared analysis settings: 300/150 Welch segments/overlap (200/100 burn-in sensitivity), fixed W2 normalization and 5 Hz cutoff, finite-neighbour safeguards, and no SAC inference. The eight restart trajectories all originate from one original student rollout; held-out restart trajectories are not eight independent task sources. Repeated corpora are explicitly cached, not additional replications.

[Manifest](manifest.json) maps exact local source files to capsule payloads, including **decoded bytes and SHA-256** for compressed bank CSVs. [Capsule checksums](capsule.json) cover every published payload. Gzip preserves exact decoded CSV bytes. `python render_review.py` redraws the figure using saved CSVs only. Preserved analysis scripts use their original absolute input bindings and require fresh output locations for intentional replay; publication did not replay them.

Original banks, traces, checkpoints and frozen worker source remain outside Git at:

- `/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/runs/2026-11-04_local_coupling_multiround_l3_001`
- `/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/runs/2026-11-04_local_coupling_multiround_l3_1_001`
- Review source: `/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/runs/2026-11-04_loop_design_review_001`

These are local evidence locations, not remotely available artifacts. Exact per-file paths, sizes and hashes are preserved in the copied input/provenance manifests.

The [cost appendix](accounting.json) distinguishes measured CPU from conservative charges for failed/unmetered work. The final saved-only extension cost **61.137571 measured child CPU seconds**. Initial partial work and failures remain charged; publication has its own [receipt](publication_receipt.json). Every review/publication phase performs **zero new physics, model queries, neural forwards, fitting updates or experiment audits**. No redesign pilot was launched.
