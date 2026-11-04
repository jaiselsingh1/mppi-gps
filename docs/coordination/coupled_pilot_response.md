# Coupled-pilot handoff: evidence and execution boundaries

The [received brief](coupled_pilot_brief.md) is now shared on this branch. Its [exact original](sources/coupled_pilot_brief_sac_2026-11-04.md) remains preserved. This task reads saved evidence and establishes ownership; it launches no experiment or audit.

**Mechanism:** stronger reconstruction coupling might suppress unpredictable actions while preserving useful motion. **Falsifier:** competent, changed experience fails to improve standalone walking against matched controls. **Cheapest discriminator:** saved-bank calibration, then short paired controller execution before fitting. **Decision:** ineffective or destabilizing coupling stops this mechanism; competent, different experience warrants matched learning. Spectra cannot establish teachability.

## Evidence corrections

- SAC walked from **534/560 sampled successful-MPPI states**, with one checkpoint and no independent audit. This establishes substantial reference continuation compatibility, not that state distribution is irrelevant to student failure or that SAC recovers from current-student deviations.
- The original spectrum's **68.58% is 1 minus low-pass variance share**. Direct high-pass residual/full variance is **64.56%**; low and high components are correlated. Neighbour R² was **.070 full / .382 low / −.075 high**. This is poor prediction by that estimator, not an identified irreducible or state-independent component. Our loop's 300-control episodes, fixed W2 normalization and whole-trajectory exclusion differ from K1's 900-control analysis window and pooled normalization. **69% versus 60% is not a matched comparison.**
- The conflict assay found no extra between-walk disagreement under its particular neighbourhood metric. It did not rule out every latent-intent or gait-family conflict.
- The drift story remains a hypothesis. The saved 72-state curvature assay put **54.81%** of residual energy in three nominally flat directions (geometric chance **50%**); its recorded conclusion was **“it does not”**. The ridge check was **“mixed”**: **68.1%** negative held-out probes, fitted median **−18.78** versus held-out **−3.66**. Preserve these contrary findings. Single-action continuation costs varied substantially and ranked actions unreliably; that does not establish flat one-step Q.
- Our eight rounds produced **0/192 walking**, despite changed coupled experience. The small spread ratio **.0318** motivates testing strength; it does not erase that negative result. Rising held error is consistent with overfitting, forgetting or changed targets; it does not by itself validate a particular early-stopping rule. [Completed review](../coupled_loop_design_review.md).

## Ownership and protocol changes before any launch

The `sac-continuation` owner keeps mirrored-sampling and source-drift investigations. Saved mirrored variants already produced **zero student walking**, with consistency **.019–.052**, under a different exploratory recipe. Reuse those findings before repeating them. This branch owns coupling calibration and any subsequently authorized corrective-data comparison.

Calibration should bind one incoming actor and an explicit bank roster before fitting. The proposed **0.3** spread ratio is a heuristic, not evidence of useful strength; **0.1 and 1** are saved-weight sensitivity checks. Retire **5.76D/N** as the proposed strength test: it only preserves the previous weak full-horizon scale. Report effects among competitive candidates as well as all-candidate spread. **D/N changes prefix-length dependence; it does not remove all fall or censoring bias.**

The warm-start contrast needs the same complete context, actor and innovations, with a pre-existing carried plan versus zero. Candidate survival alone is insufficient. Subsequent labels must come from current-student states, with DAgger aggregation and explicit treatment of failed teacher continuations. Admit a learning test on effective intervention and executed teacher competence, not a desired spectral or R² direction. Freeze held-data separation, stopping/checkpoint selection, continuation, actual update counts and end-to-end compute allocation before using early stopping. Preserve **three short rounds**, **at least 32 unused paired resets**, walking separate from survival, and a declared coupling-discard criterion.

The original failed audit remains unchanged. Closing it requires a bounded, versioned audit-only recovery preserving the failure and tolerances; this handoff has not rerun it. The [source bindings and saved contradictory results](sources/README.md) accompany these corrections.

## Coordination

Each branch has its own log; entries do not synchronize automatically. Read the other worktree's file for uncommitted progress, or its branch version after publication. Write only on the owned branch. No direct message to another session was sent.
