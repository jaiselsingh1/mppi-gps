# First Walker CIO–MPPI pilot: negative at policy transfer

The cleaned implementation and first bounded run are complete. **Independent discovery succeeded; standalone policy transfer failed.** Guidance remained zero and neither feedback round ran. The run ended at the declared gate without changing settings or enlarging the budget.

| Measured stage | Walking success | Mean complete cost | Velocity MAE |
|---|---:|---:|---:|
| Independent discovery |4/4|401.95|0.154m/s|
| Fitted policy qualification |0/16|4558.12|1.532m/s|
| Same policy, held starts |0/16|4581.19|1.579m/s|

Actor episodes used only the network. All held episodes fell after 211–340 controls (mean 264.3), versus the required 1,000. Complete costs include the absorbing failure charge. Qualification and held points evaluate **the same checkpoint** on distinct cohorts; they are not successive training improvements.

The declared 2,000 Adam updates reduced mean action MSE only modestly: first 100 updates 0.2152, last 100 updates 0.2008. This is a bootstrap fitting/standalone-transfer failure. Representation, coverage and optimization causes have not been isolated. Policy-guided search, executed control improvement and next-child benefit remain **untested**, rather than established failures of qualified guidance.

## Inspect the evidence

- [Curves](../runs/2026-10-29_walker_cio_mppi_001/curves_reviewed.png), [training records](../runs/2026-10-29_walker_cio_mppi_001/learning/bootstrap.json), [evaluation table](../runs/2026-10-29_walker_cio_mppi_001/curve_reviewed.csv).
- [Independent MPPI video](../runs/2026-10-29_walker_cio_mppi_001/independent_mppi.mp4) and [standalone policy video, ending in a fall](../runs/2026-10-29_walker_cio_mppi_001/standalone_actor.mp4). Videos render recorded states and add zero controls.
- [Trained checkpoint](../runs/2026-10-29_walker_cio_mppi_001/checkpoints/final.pt), with explicit [failed deployment assessment](../runs/2026-10-29_walker_cio_mppi_001/checkpoints/final.assessment.json).
- [Frozen configuration](../runs/2026-10-29_walker_cio_mppi_001/config.json), [producer source manifest](../runs/2026-10-29_walker_cio_mppi_001/remote_manifest.json), [full result](../runs/2026-10-29_walker_cio_mppi_001/result.json) and [raw ledger](../runs/2026-10-29_walker_cio_mppi_001/evidence/ledger.jsonl).
- [Independent saved-outcome audit](../runs/2026-10-29_walker_cio_mppi_001/outcome_audit_001/receipt.json): passed 1,530,599 checks across 36 complete episodes. It verifies outcomes/costs/trace hashes/work counts; it does not replay dynamics, policy computation or guidance causality.

## Costs and verification

The pilot charged 51,176,593 controls: 51,164,160 modeled and 12,433 executed; 204,706,372 physics substeps; 24,693 queries; 520,433 neural rows; 512,000 backward rows; 2,000 optimizer steps. Query failures, rejected queries and unknown partial controls were zero. These are successful software calls, not successful robot outcomes.

External pilot wall time was 1,164.25seconds (19.40minutes) on robopil5090 using 16 native CPU simulation threads and CPU PyTorch. Additional two CPU-thread preflights charged 1,609,856 controls; all 3,795 saved array pairs matched exactly between 4 and 16 threads. The [resource report](../runs/2026-10-29_walker_cio_mppi_001/resource_report.json) separates these and captured validation costs. Early uncaptured development-test overhead prevents an all-development efficiency claim.

The full retained suite passed 114 tests, with 2 skips and 1 expected stock-Gymnasium physics mismatch; [validation receipt](../artifacts/2026-10-29_cio_reset_full_validation_001/receipt.json). [Independent metadata/preservation review](../artifacts/2026-10-29_cio_reset_independent_metadata_audit_002/review.md) verified the source/configuration and archive bindings. Real promotion/revocation was not exercised because no policy qualified; its software paths passed contract tests.

The 26.12GB raw candidate/trace HDF5 file remains intact at the exact remote path in the [evidence locator](../runs/2026-10-29_walker_cio_mppi_001/remote_evidence.json). Source, ledgers, checkpoints, reports and videos are mirrored locally. The original frozen plots and failed rendering receipts remain alongside the corrected postprocessing outputs; [postprocessing record](../runs/2026-10-29_walker_cio_mppi_001/postprocessing.json) confirms unchanged scientific function ASTs.

Historical experiments are unchanged; see the [history index](history.md). No ending-behavior, beneficial feedback or efficiency result is claimed.
