# Historical evidence

Earlier raw runs/checkpoints remain under runs/, audits under artifacts/, and preserved source under archive/. Paths are unchanged to retain evidence links and source bindings.

- [Pre-reset source manifest](../archive/pre_cio_reset-2026-10-29/manifest.json):110 exact files. [Active retirement list](../archive/pre_cio_reset-2026-10-29/retirement.json).
- [Previous results](../archive/pre_cio_reset-2026-10-29/docs/results.md): physical Point Mass baseline and retired scalar diagnostics.
- [Non-MuJoCo retirement](../archive/non_mujoco-2026-10-28/ARCHIVE.md).
- Earlier research snapshots remain in archive/research-2026-10-27 and archive/research-2026-10-28.

Useful Walker physics, tuned MPPI, replay, weighting and alignment utilities were retained. Experimental orchestration and unused workflows were removed only after preservation. No previous result is relabeled.

## Before v2

[Exact v1 source/configuration/docs snapshot](../archive/cio_walker_v1-2026-10-29/manifest.json) preserves52 files before the fitting/recovery/early-coupling changes. Its [result report](../archive/cio_walker_v1-2026-10-29/docs/results.md) and all original raw runs remain unchanged.

## Closed v2

- [Zero-physics fitting study](../runs/2026-10-29_cio_fit_v2_001/result.json): six fixed arms, exact legacy reproduction, 60,000 optimizer steps.
- [Frozen physical study](../runs/2026-10-29_cio_feedback_v2_001/result.json): standalone policies failed; both teacher-data trials failed admission; no confirmation or child comparison ran.
- [Outcome audit](../artifacts/2026-10-29_v2_outcome_audit_001/receipt_001/receipt.json): 150 episodes and 32 separately accounted short recovery branches.
- [Raw archive locator and whole-file hash](../runs/2026-10-29_cio_feedback_v2_001/exports/remote_evidence.json): full model evidence retained on the canonical remote host; executed traces are mirrored locally.
- [Separately charged saved-plan diagnostic](../artifacts/2026-10-29_omitted_baseline_saved_plan_diagnostic_001/receipt_001/result.json): an omitted verified baseline could change 517 recorded decisions under the augmented objective. This is an optimization finding with no new controller executions.

The later recovery correction and outcome-based teaching requirements apply to future versions, as described in the [algorithm](algorithm.md). They do not alter the recorded v2 filters or reinterpret its rejected data as admitted.
