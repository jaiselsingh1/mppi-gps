# Coupled-value pipeline: failure diagnosis and bounded repair

This review uses the corrected `_002` study's saved data. The walking study is
still running with its original frozen source and configuration. Local fixes
below are for subsequent studies only. No new simulator controls or neural
training were used for the diagnosis; simple offline regressors were fitted.

## Findings supported by measurements

### A small positive alpha can remove the local baseline

The original planner assigns unsupported endpoints infinite score whenever
alpha is positive. Its population support gate does not require support for
the actual minimum-local-cost plan. Consequently, even a constant value model
can change the chosen action solely by excluding the baseline.

An executable proof uses real MPPI weighting with symmetric deterministic
perturbations: alpha zero selects the near-zero weighted mean, while alpha
from 1e-12 through .1 selects -.4. Four of five healthy endpoints are supported;
the locally best mean is not. This establishes the code behavior, not its
frequency or causal contribution in Walker.

**Prepared local correction:** if the alpha-zero winning plan lacks endpoint
support, abstain for the entire decision and use exactly the local scores and
candidate set. Do not substitute a zero continuation value. The diagnostic
also records the actual baseline candidate and its support. This repairs the
demonstrated exclusion problem; it does not guarantee smooth closed-loop
behavior for every alpha or certify safety. The extra guided mean and finite
candidate argmin can introduce other discontinuities.

[Proof and frozen-source reproduction](../runs/2026-10-25_coupled_value_walker_seed0_002/supervision/failure_analysis_20261026/proof.json).

### The neural value generalizes poorly on this saved split

The cycle14 model has19,073 parameters and137 training labels from21 collection
episodes. It is refitted from fresh weights for600 steps. The exact frozen
model's normalized training RMSE is .1000 versus .3173 on47 held labels.
Earlier fits show even larger gaps: cycle3 .0030/.2670 and cycle4 .0028/.1650.
These gaps establish poor generalization; they do not uniquely separate model
capacity, input noise and distribution shift.

Fixed alternatives were declared before their outcomes and evaluated on the
identical cycle14 split, with train-only scaling and no holdout tuning:

| Predictor | Held normalized RMSE | Selected-endpoint RMSE |
|---|---:|---:|
| Frozen neural model | .3173 | .5758 |
| Training normalized mean | .3417 | .3680 |
| Regularized linear model, penalty10 | .2538 | .3546 |
| Five-neighbor outcome memory | .2953 | .3452 |
| Pooled training-only linear model | .2466 | .4135 |
| Pooled training-only five-neighbor memory | .2995 | .2829 |

Every tested alternative still fails the .20 endpoint threshold. A smaller
model is a credible improvement in broad prediction, not a demonstrated
control fix. Pooling is not uniformly beneficial on selected states.

Compatible labels from the memory arm were already being generated but not
used in the neural fit. Pooling excludes all held ancestry groups from BOTH
arms and removes exact repeated measurements; it does not create independent
evidence from duplicated starts.

### Selected endpoint predictions are often optimistic

Across the four saved models with endpoint audits, nine of ten valid supported
predictions underestimate the measured continuation cost. The worst cycle14
endpoint predicts395.03 versus2836.52 measured, with952 controls remaining.
These are sparse, correlated, single-seed outcomes. They establish observed
optimistic errors, not statistically established expected bias or proof that
optimization exploited the predictor.

Absolute RMSE alone also cannot measure action ordering: a common additive
error can cancel between candidates. The current audit lacks measured
alternative endpoints at the same decision, so it cannot estimate ranking
accuracy or selected-action regret.

### Training does not deliberately acquire the states exposing those errors

Most training queries sample sparse points from deployed collection episodes,
uniformly with replacement. Deployed alpha has remained zero. The final audit
instead measures endpoints selected using the newly fitted value model.
Keeping audit outcomes outside fitting is correct, but there is no separate
training stream that systematically queries those candidate-selected regions.
The rejected proposal therefore need not generate data that repairs its error.

### Alpha does not measure the actual strength or reliability of guidance

An alpha of .1 multiplies the whole long-horizon tail, while MPPI's softmin
temperature is3. At an early episode endpoint, the .5758 normalized error
illustration contributes about168.5 score units at alpha .1. This is an error
propagation calculation, not a recorded within-bank ranking error. A shared
offset could cancel; unequal errors can dominate local candidate differences.

Alpha also downweights the known absorbing tail of already-fallen candidates.
That matches the declared partial-tail objective, so it is not an accidental
implementation bug. It does mean that alpha conflates confidence in a learned
estimate with how much future cost matters. A future confidence mechanism
should not pretend known failure consequences are uncertain.

## Unresolved causes that should be measured before another long run

- The137 training rows contain120 distinct observed feature/time inputs;
  sixteen repeated-input groups have different returns. Their empirical
  within-input RMSE floor is .0810 on this finite sample. Private planner
  randomness and omitted solver warmstart may both contribute. This is not a
  population or irreducible-noise estimate.
- The oracle restores solver warmstart, but the predictor omits it. Repeating
  continuations with fixed physical state and varied seeds/warmstart would
  distinguish these effects; those new queries have not been run.
- Audits replay a selected plan and recompute its prediction. The logs do not
  retain the original batch endpoint/value and guidance-active flag for that
  decision. Batch-versus-sequential discrepancies remain unmeasured.
- Reference values describe a fixed canonical-restart bootstrap controller,
  not the evolving guided controller. Good reference prediction alone does
  not establish good deployed behavior or discovery of unseen successes.
- Different archive-refresh rules confound cross-arm comparisons, even when
  both deployed alphas are zero. Same-archive contrasts remain necessary.

## Recommended next loop

Retain independent MPPI proposals and measured complete reference returns.
Use the existing query budget more deliberately before increasing it:

1. Preserve the exact local baseline as fallback. Require supported baseline
   and candidate endpoints before a value-based replacement is considered.
2. Collect a separate training stream of **paired candidate and baseline
   endpoints from the same planning decision**. Use matched continuation seeds
   and reserve some repeats to estimate target dispersion. Preserve all
   current audits as untouched historical evidence.
3. Pool compatible training labels with shared-ancestry grouping and duplicate
   accounting. Keep fresh qualification ancestry entirely outside fitting.
   Do not relabel every suffix timestep as a fresh-restart target: its retained
   planner memory describes a different controller state.
4. Start with a small regularized predictor and compare against outcome memory
   on fixed grouped development examples. Report training/holdout error,
   signed errors, pairwise return-difference error and measured ranking regret.
   A neural model should earn replacement of the simpler predictor.
5. Qualify differences in measured future cost, rather than absolute-value
   accuracy alone. Check the noise floor before choosing tolerances. Absolute
   calibration remains informative, but cannot certify action rankings.
6. Set influence using observed errors relative to local score differences and
   softmin temperature; retain full known failure accounting. Any weight/KL
   budget would bound only the declared proposal weights, not safety or
   episode performance. Keep fresh full-controller tests for deployment.
7. Use identical archive update rules for value-on/value-off controllers and
   compare them with the same archive. Charge acquisition, repeats, rejected
   work and qualification. Walking alone would not prove sample efficiency.

This is a repair specification, not a launched or validated new training
algorithm. More epochs, larger alpha or a looser endpoint threshold do not
address the measured data and decision problems.

## Local implementation and validation

- `src/gps/coupled_planner.py`: implemented whole-decision fallback when the
  actual local winner is unsupported; recorded baseline candidate diagnostics.
- `src/gps/global_value.py`: added training error, held signed bias and a
  train-only normalized-mean baseline to future fit metrics. Fitting and
  admission thresholds remain unchanged.
- **63 focused tests passed**, including real-MPPI symmetric-proposal fallback
  regressions and a held-label leakage check for the prediction baseline.
- The remote study's15 archived/runtime source hashes still match. No patch
  was copied to it, and no new study or additional evaluation panel was started.

[Offline prediction report](../runs/2026-10-25_coupled_value_walker_seed0_002/supervision/failure_analysis_20261026/predictive_diagnosis.md)
and [strict numeric evidence](../runs/2026-10-25_coupled_value_walker_seed0_002/supervision/failure_analysis_20261026/predictive_diagnosis.json)
include the exact group split, artifact hashes, frozen-source verification,
individual endpoint residuals, and limitations.
