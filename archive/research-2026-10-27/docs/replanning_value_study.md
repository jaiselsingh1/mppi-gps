# Controller-consistent replanning value: bounded Walker pilot

The user authorized implementation, testing and a fresh comparison against
memory, preserving the already running paired-endpoint study unchanged. This
is a separate actor-free mechanism test. It is not a complete GPS system or
evidence for exploration/sample efficiency merely because walking improves.

## Target and control relationship

Freeze an actual controller C_k for an epoch: its common archive, independent
MPPI settings, active scorer, influence and retained nominal update rule.
V^{C_k} predicts the complete measured finite cost of that controller from a
state BEFORE its next planning call. Input includes17 physical features,9 solver
warmstart coordinates, the entire48x6 raw retained nominal, initial-plan flag
and remaining/1000 (316 coordinates). PRNG state is saved for replay; the
value averages the declared future private planner randomness. This stochastic
modeling convention is not a certificate of an exact Markov representation.

For each MPPI-bank action a_i, physically execute exactly ONE scratch control
from the same state/warmstart. Preserve the resulting solver warmstart and the
SAME post-search raw nominal for every candidate. Restore the live simulator
after scratch work. Known first-action falls have the exact finite absorbing
return; a healthy final control has zero tail. Other scores are

    Q_i = c_i + gamma * V^{C_k}(next physical/solver/retained-plan state).

The baseline action b minimizes the independently verified H-step local score
B. Influence uses the explicit empirical interpolation

    S_i = (1-alpha)*(B_i-B_b) + alpha*(Q_i-Q_b).

For alpha<1 this blends different horizon objectives; it is not an exact
policy-value or a policy-improvement theorem. At alpha1 it ranks full Q.
It has no MPPI-temperature-based value correction cap. Unsupported baseline
causes whole-decision fallback; unsupported alternatives retain nonnegative
local gaps and cannot strictly displace baseline. Alpha0 retains the original
paired-bank planner's actions/RNG/nominal bitwise. No actor supplies proposals,
no learned weighted mean is introduced, and selected hypothetical tails are
never installed as the retained nominal.

## Frozen protocol

-24cycles, seed0, Walker2D,1000controls=8seconds, gamma.999, K32/H48,
  eight archive candidates,2048 training-memory entries, six CPU workers.
- Four bootstrap memory-controller episodes, then16 initial development
  evaluations per arm. Both arms use the exact same archive and independent
  MPPI bank/verification rule. Value-off avoids unnecessary one-step forks;
  the guided arm's extra scratch physics and prediction work are counted.
- At each epoch START refresh the common archive unconditionally from all
  previously collected TRAIN trajectories. If retained guidance is active,
  recheck it against memory on six fresh matched retention resets before
  freezing the new C_k. Revoke to alpha0 if it fails. Archive refresh is never
  accepted/rejected asymmetrically between arms.
- Collect eight ordinary-root C_k episodes. Compute complete suffix MC costs
  backward, keeping every eighth decision context. Falling-transition cost
  occurs once and the entire remaining finite absorbing cost is charged.
  Healthy finite cutoffs have zero tail. There are no Bellman targets.
- Select eight distinct root-balanced cached decisions. Half use the current
  scorer's shadow selection when available; half select a uniform distinct
  first action. With no scorer all are uniform. Equal shadow choices remain
  explicit zero comparisons. One candidate and exact baseline first action
  each receive two matched fresh future streams, followed by the ACTUAL C_k
  retained-state replanning controller to completion. Equal actions reuse
  one execution without fabricating another independent observation.
- Only post-forced-action contexts belong to V^{C_k}; the forced action's
  predecision state is not mislabeled as on-controller value. All branches
  retain their ordinary-root ancestry. Every queried/rejected branch is charged.
- Fit three64x64 neural networks,300 updates, from this epoch's C_k records
  ONLY. Do not pool old-controller labels as if current. Newest25% whole roots,
  at least two, are held out. Loss balances roots then episodes; ensemble
  resamples whole roots. Matched post-first-action tail differences have unit
  loss weight. All scalers/support use training roots only. Positive output is
  uncapped softplus, normalized by the finite5*discount-sum as a numerical
  scale rather than a return bound. Dense rows are correlated observations,
  not additional independent experience.
- Propose influence alpha+.25, capped1. Fresh calibration runs two separate
  C_k ordinary roots and SHADOW-ranks their cached banks using the exact
  proposed scorer/blend. Select at most four actual changed supported decisions,
  balanced across roots. Measure first-action+C_k returns with two matched
  streams per side. Require at least two nontrivial supported comparisons,
  at least two noise-resolved comparisons spanning two roots, normalized
  difference RMSE<=.20, resolved sign accuracy>=.5 and mean measured cost
  difference<0. Noise-resolved means |mean difference|>its two-repeat SE;
  this is an empirical filter, not a confidence certificate.
- Only a calibrated proposal receives a further six fresh full-controller
  matched qualification resets. It must beat BOTH same-archive memory and
  the current incumbent by>1 mean cost, win at least half the resets and
  lose no survivors. Otherwise retain the already rechecked incumbent or
  remain memory-only. Qualification data never train or refresh archives.
- The deployed controller packet is flat: one archive, one scorer, alpha and
  its own version. A promoted C_{k+1} contains a scorer targeting C_k; these
  identifiers deliberately differ. No recursively nested predecessor model.
- Evaluate both deployed arms every four cycles on the fixed16 development
  resets, and finally on64 fresh common resets. Last epoch's training traces
  are stored for memory but do not change its frozen qualified archive before
  final evaluation. Development and final panels remain separate.

## Reset roles and evidence

Seed0 blocks: bootstrap46m; training47m; incumbent retention48m; calibration49m;
qualification50m; repeated development52m; final53m. Final resets are
53000000–53000063. Reserved5000000–5000199 and other active studies' resets
are never used. Paired branch future streams occupy separate seed blocks from
collection, calibration and full-controller qualification.

Store complete per-cycle controller versions, archive digests, physical and
solver traces, raw pre/post nominals, first-action banks, MC targets, fitting
root groups, paired differences, abstentions, promotions/revocations and work.
Both deployed arms' archive digests must agree on every panel. Report actual
action changes separately from value-score exposure. Source hashes/config
freeze before launch; new source-changing failures require a separate reviewed
trial, not a patch to the live checkout.

Checkpoint recovery preserves uncommitted evidence and reruns only uncommitted
epochs, with interrupted unknown work explicitly a lower bound. Durable final
evaluations are recovered rather than rerun. Resume refuses a live predecessor,
config change, runtime-source change or archived-source change.

Joint observed walking gate:>=90% eight-second survival AND<=.30m/s mean
speed MAE. Report finite cost, alive time and work including failed acquisition,
scratch verification, fitting queries, retention, calibration, qualification
and evaluation. Prediction rows and planning calls are not simulated controls.
One paired training seed; reset intervals do not establish training-seed
robustness. Retain negative results without extending24cycles or retuning.

Entry points: scripts.training.train_replanning_value and
scripts.diagnostics.compare_replanning_value. The still-running paired-endpoint
study, its source/configuration/checkpoints and monitor remain untouched.
