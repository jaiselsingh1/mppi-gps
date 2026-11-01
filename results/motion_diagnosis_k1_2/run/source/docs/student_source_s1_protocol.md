# S1 proposal: can the current student learn a different competent behavior?

**Unlaunched proposal for review.** This document and its [proposal configuration](../configs/studies/student_source_s1_proposal.json) authorize no training or evaluation. No S1 worker or public study-dispatch entry is added. The user selected diagnosis before choosing a learning change; the completed [D1/D2 diagnostic](transfer_diagnosis_d1d2_protocol.md) therefore motivates a decision for review, not an automatic follow-up.

## Why this contrast next

D1/D2 run `2026-10-31_transfer_diagnosis_d1d2_001` completed with an independent audit. Its LOTO support threshold was 0.8601479856552903. The held independent teachers had 0.45% of rows outside that threshold; the three W2-2k cohorts' first 100 controls remained inside despite subsequent failure. The competent SAC reference visited states outside the MPPI support threshold. Support distance is descriptive and does not certify competence or identify a coverage cause.

All 20 unmodified D2 continuations passed the frozen numerical replay gate. Single-action substitutions caused different downstream costs under the same saved teacher suffix, but did not consistently rank whole-policy retention: stream 2’s 10k endpoint had larger action MSE and lower mean 100-control cost increase than its 2k endpoint. The competent reference's substituted actions produced 10/20 falls under that suffix despite its successful own rollouts. This is not a reference recovery test or evidence that its own behavior is unlearnable.

Those results do not justify turning action error, support distance, clock time or D2 cost into a training weight. The next narrower question is whether the **same 128-tanh student and fixed learning procedure can acquire one other demonstrated competent behavior**. A reference-data clone versus an MPPI-data clone is still untested: K1 did no learning; W1/W2 fitted MPPI experience; the earlier 980-row corrective experiment also used MPPI labels after an actor prefix. S1 is a whole-corpus source contrast, not a proof that MPPI noise, multimodality or diversity caused failure.

## Two corpora, one student and fixed update

| Arm | Training episodes / rows | Held episodes / rows |
|---|---|---|
| MPPI | Exact W2 `discovery_00..03`, 4,000 rows | Exact W2 `v2_r0_trial_raw_0..1`, 2,000 rows |
| Reference behavior | K1 `k1_primary_reference_00..03`, 4,000 rows | K1 `k1_primary_reference_04..05`, 2,000 rows |

Reference reset seeds are 2026103100–03 for training and 2026103104–05 for held experience. Each selected source episode executed all 1,000 controls successfully. The first-four/next-two rule selects complete episodes by ID, not by a new fit result. These corpora have already been examined in K1 and D1; the held episodes are development holdouts from S1 gradients, not newly blinded evidence. MPPI and reference resets, states, gaits, action statistics and controller histories differ. The source contrast changes all these properties jointly; inference is conditional on the chosen corpora.

Original discovery’s missing local terminal/warmstart/integration arrays remain missing. S1 uses its already audited state–action pairs and reconstructs no absent context.

Targets are original **executed** torques at the original pre-action observations. Do not replace MPPI labels with reference predictions at MPPI states. Preserve source dtypes and raw-array/file hashes: MPPI targets were stored in float64, reference torques in float32; both use the current float32 fitting conversion and float64 diagnostic residual arithmetic. Check every feature against `concat(state[2:10],clip(state[10:19],−10,10))`, validate bounds, episode/control order and split membership before fitting. Audit state/observation overlap in original, float32 and normalized coordinates; reject cross-split input leakage without deleting or replacing rows. A saved-only feasibility check found zero cross-split effective-input collisions in raw, float32 and common-normalized float32 coordinates, both within the reference corpus and across the combined source corpora; revalidate after authorization. Report repeated action values as label statistics: target equality alone is not input leakage or a reason to reject data. No hypothetical, corrective, recovery or random-action labels are added.

Each of three fresh paired initializations uses the existing 17→128 tanh→128 tanh→6 tanh student. Within a pair, both source arms start from identical actual parameter bytes, normalization buffers and private generator state, with zero Adam. The common normalization is the **exact W2 buffer set**, fitted only on original MPPI training observations. It is not refitted for reference data or held examples. This preserves the current input map; failure could still reflect conditioning or finite-budget limitations and cannot prove class incapacity.

Fit each arm once for exactly 2,000 updates, batch 256, lr 0.0003, gradient clipping 0.5, zero parameter proximity. Use identical saved PCG64 minibatch indices within each pair, with replacement over 4,000 rows, and preserve source row order. Save all incoming and final checkpoints with optimizer/RNG ownership. There is no 10k continuation or selected endpoint. All six fits finish before simulator construction.

For either recorded corpus \(D\), the unchanged learning objective is
\[
L_D(\theta)=\frac{1}{6|D|}\sum_{(o,a)\in D}\|\pi_\theta(o)-a\|_2^2,
\qquad \theta_D^{(s)}=\operatorname{Adam}_{2000}(\theta_0^{(s)},D).
\]
Execution applies \(\pi_{\theta_D^{(s)}}(o_t)\) directly. There is no trajectory-optimization feedback in S1: it measures the result of one fixed learning update under two experience sources.

The proposed fit backend is the verified 32 GiB RTX 5090 on `robopil5090`, using `/home/jaisel/mppi-gps-venvs/cuda/bin/python`, Torch 2.10.0+cu128, float32 with TF32 disabled and deterministic algorithms required; set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing Torch. Adam remains nonfused/nonforeach, with one Torch CPU thread. Both source arms are new fits on this same backend. Historical CPU W2 policies are context, not matched controls. Standalone execution and the unchanged task/XML use the pinned D1/D2 native CPU runtime at `/home/jaisel/mppi-gps/.venv/bin/python`. Preserve platform receipts and stop if the deterministic runtime contract cannot be met; no outcome-driven CPU/GPU switch is allowed.

## Outcomes and decisions

Use 16 fresh paired primary resets and 16 disjoint confirmation resets from the proposal configuration. **Both panels run for all ten fixed actors:** six final children, three incoming untrained parents and the existing frozen reference actor. A favorable seed or primary result does not select the confirmation roster. Evaluate 1,000 controls / 8 seconds per attempt, direct policy torque without prefix, fallback or replanning. Record initial full snapshots to verify pairing. Retain all failed episodes.

Walking is the existing full native criterion: survive all 1,000 controls and satisfy velocity MAE ≤ 0.3 m/s. Survival, controls, complete cost with its terminal/absorbing-tail decomposition and velocity error remain separate outcomes. Longer survival without walking is retention improvement. Report complete historical source-teacher outcomes and fresh reference/parent/child outcomes; historical-teacher/fresh-child gaps are descriptive and unpaired. Different successful trajectories are valid.

The primary source-benefit screen requires the reference-data child to gain at least 2 walking successes in at least 2 of 3 pairs, with no walking-count or full-survival-count regression in any pair. Apply the same screen separately to both panels; only a pass on both is a confirmed walking benefit. An individual actor is provisionally competent at ≥15/16 walking on **each** panel. Report all seeds; one competent seed does not establish robust fitting across seeds. The live reference is a competence control under the new execution context; complete its fixed roster even if it fails, and disclose any loss of its qualification.

Each of the nine student/incoming actors predicts all four fixed splits once: both arms’ 4,000 training and 2,000 held rows, totaling 108,000 diagnostic forward rows. Retain within-source and cross-source errors, per-trajectory/actuator errors, saturation and the already computed minibatch-loss curves. The reference reuses existing K1 predictions and logged torques. Low MSE is not the advancement gate. Do not select a source, normalization, seed or checkpoint from held errors.

Use the three paired training seeds as intervention replication units, with a descriptive crossed seed/reset bootstrap (2,000 draws). Frames and episodes are not independent training runs, the source corpora are fixed, and all-zero walking counts do not prove identical population success rates.

## Finite budget, closure and limits

| Work | Proposed ceiling |
|---|---:|
| Adam updates / backward rows | 12,000 / 3,072,000 |
| Training / diagnostic / evaluation forward rows | 3,072,000 / 108,000 / 320,000 |
| Total forward rows | 3,500,000 |
| Native controls / physics substeps | 320,000 / 1,280,000 |
| Complete attempt roster / checkpoint reloads | 320 episodes / 10 |
| New teacher acquisition / MPPI model controls | 0 / 0 |
| Worker wall time / storage | 3,600 s / 16 GiB |

The forward reservation includes every incoming/child reconstruction panel and the full all-ten-actor evaluation roster; no extra curve evaluations are hidden outside it. Accounting separates inherited evidence from new fitting, evaluation and engineering work. Preserve exact source/configuration/input seals, full minibatch tapes, Adam/RNG checkpoints, predictions, native snapshots, ledger, raw curves, failed/partial attempts and an independent saved-only audit. Existing runs and capsule bytes remain immutable. The proposed reset seeds passed the current 230-configuration local census; revalidate freshness and finish a new source/input/platform seal only after separate authorization.

Stop on any source, alignment, interface, normalization, deterministic-runtime, numeric, ownership, work, wall or storage error. Preserve partial work. Do not retry, replace a corpus, extend fitting, change the feature map or add weighting, an OT loss, a critic or another environment.

If reference-data cloning succeeds while MPPI cloning fails, that is a positive learning control and a whole-source effect. It does not identify the causal property of that source or authorize MPPI guidance toward the reference. If both fail, retain the result and formulate one explicitly versioned hypothesis about remaining fitting, conditioning, target or execution obstacles. Retention-only gains remain partial. Even a competent reference-data clone does **not** establish that the MPPI–policy loop works: a later, separately approved MPPI acquisition–learning mechanism and a second useful round would still be required. No follow-up launches automatically.
