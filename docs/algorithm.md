# Frozen v2 algorithm

This describes the completed v2 experiment, including its predeclared filters. It is retained to explain the recorded result. The user-corrected requirements for the next version are listed at the end; the frozen source and outcomes are unchanged.

The task, native physics, K256/H48 proposals, temperature3, bounded controls and Butterworth noise remain unchanged. Episodes last1,000 controls at0.008seconds/control; success requires survival and velocity MAE≤0.3m/s at a1.5m/s target. Complete comparison cost includes native costs, the fall terminal once, and5 per unexecuted remaining control after a fall.

## 1. Saved-data diagnostic

Use the exact4,064 v1 targets. Six fixed arms receive matched minibatch streams and2k→10k updates: original sum proximity to initialization; no anchor; mean-over-parameters proximity refreshed every100 or250 updates; and unanchored23-input padded/history models with identical initialization. Inputs for the latter differ only in whether the last six entries contain the previous executed action.

Measure full-dataset MSE separately from minibatch loss. Require original2k checkpoint hash reproduction. This stage imports no simulator and spends zero controls. All neural rows, backwards and60,000 Adam steps are charged. Bfree10k is fixed beforehand as the primary physical-study parent, regardless of diagnostic ranking.

## 2. Closed-loop diagnosis and corrective coverage

Evaluate all six10k actors and original/free2k actors on16 paired fresh ordinary starts. History deployment uses the student's own preceding action. Measure complete cost, survival horizon and task success separately.

Probe selected parent rollout states with two recorded nominal contexts and matched RNG repeats; the local search outputs are not certified successful continuations. Whitened nearest-neighbor distances use a leave-one-trajectory-out reference and remain descriptive.

Run32 independent recovery branches from saved teacher snapshots/private planner state: four anchors per teacher, two repeats, two noisy executed impulses (σ0.15), then32 replanned controls. Impulse controls are excluded from training. A branch qualifies only locally: no fall, endpoint normalized distance≤max(0.1,0.9×post-impulse distance), and last-eight cost≤matched nominal+0.2. Preserve every failed attempt. The accepted corrective set is shared between RAW and GUIDE children.

## 3. Provisional coupling and student feedback

No standalone-success prerequisite blocks experimental teacher coupling. Keep all256 independent proposals and task-only private nominal/RNG updates. Policy agreement changes only the copied-bank selection and actual executed actions. Both selectors share action-sequence proximity0.01 and verify means.

One distinction from the pasted suggestion is deliberate: the guided weighted mean never replaces MPPI's private warm start. Guided execution can change the physical state visited next; at that state the raw planner still uses its own task-scored nominal and RNG. This preserves proposal independence at an identically restored physical and planner state.

Calibrate two fixed weights from64 existing verified plans:5% and20% times median local task-cost/agreement ratio, clipped[1e-4,10]. Calibration is not qualification. Use two paired trial starts, then two disjoint confirmation starts for the lowest-agreement eligible candidate; never retry another weight on confirmation outcomes.

Teacher-data eligibility requires all guided teachers to succeed, survival/success nonregression, mean complete task cost within5% of raw, at least5% lower actual executed action disagreement, and changed actions above1e-6. The5% task-cost allowance is permission to test teaching data, not a claim of improved control. Strict control improvement separately requires at least0.1% lower complete cost without task regression.

Train RAW/GUIDE children from identical parent/Adam/RNG states and equal2,000-update budgets. Use mean parameter proximity0.01 with250-step anchor refresh. Both receive the same original/corrective data plus their actual teacher data. Select raw fallback for a guided slot violating the per-slot quality bound.

Compare parent and both children on16 fresh paired starts. Advance GUIDE only when complete cost beats both by0.1%, success/survival do not regress, and mean survival horizon does not regress. Partial progress can advance the research loop; it never qualifies deployment.

Attempt at most two rounds. The second fixes the first selected weight and requalifies it with the changed policy. Revoke the old authority on version change/failure. Final standalone qualification still requires15/16 fresh held successes, and a full-loop claim additionally requires the held actor to be the retained guided lineage.

## Costs and limits

Fit and feedback stages share350million new requested controls and150million neural rows; feedback subtracts the already-paid fitting work. Report the imported v1 expenditure separately. Every model/live/verification/recovery/post-hoc/evaluation call and learning operation is recorded. The task and budget stay fixed after outcomes. One development seed and small paired cohorts test the mechanism, not general superiority or sample efficiency.

## Requirements for the next version

Recovery means restoring successful task performance after a disturbance. A different successful continuation is valid. Qualify it using complete executed task outcomes; remove return-to-original-trajectory distance as a data-admission requirement. The 32 short v2 branches do not establish recovery by this definition.

Separate teacher task qualification from teaching usefulness. Complete teacher outcomes establish whether the continuation solves the task. Matched child-policy executions establish whether its experience improves the standalone policy. Action agreement and trajectory distance are diagnostic measurements; neither proves learnability. In particular, failure of v2's 5% action-agreement requirement did not test whether the higher-weight teacher's data could improve a child.

These requirements have not been retroactively applied to v2. No new physical study using them has run yet.
