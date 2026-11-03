# J2.1 review proposal: rank fresh students within a fixed label condition

**Review only. This is not a frozen executable protocol or authorization to
launch.** Source/input bindings, implementation, perturbation calibration,
complete accounting and statistical review remain prerequisites. No new
experiment or data query accompanies this proposal.

## Why the question changes

The [audited J2.0 result](jitter_diagnosis_j2_results.md) stopped at its frozen
clean-child gate. The reference walked 32/32; six clean children walked
`[5, 15, 32, 30, 30, 31]`, or 143/192. Three children met the separate 15/16 in
each panel convention. All six fits and outcomes remain in the
[capsule](../results/jitter_diagnosis_j2_001/README.md). The other 72 fits did
not run, and the frozen primary comparison remains unexecuted.

A post hoc comparison among those six clean children gave utility 0.7826908981
for on-policy disagreement and -0.7247137946 for held-clean action MSE, where
utility is negative Spearman correlation with walking count. All 48 probes ran
100 controls; one fell on its final control. Thus different probe lengths do
not explain this particular six-case ordering. Every child walked at least
once, so any-walking AUC is undefined here. These are development observations,
not confirmation of the frozen claim or of the proposal below.

J2.0 asked whether a score ranks **13 label conditions within each of six
learner streams**. The new question asks whether it ranks **ten fresh learner
streams within each of four fixed label conditions**. These are different
estimands. Weak clean fits are valid labels for the new diagnosis question;
requiring all students to walk well would remove the variation being studied.
The clean-child qualification gate is therefore absent from this proposed new
version. J2.0's gate and stopping result are preserved unchanged.

## Proposed roster, learner and measurements

Use exactly four existing conditions: clean, MPPI-derived jitter at signal
share 0.9, white jitter at 0.9 and white jitter at 0.75. Use their original
bound label arrays, the same 9,000 clean SAC training observations, 7,000 held
clean rows, corpus-derived normalization and width128 torque-policy interface.
No new MPPI discovery, teacher data, architecture, critic or loss is proposed.

Use ten new learner streams, with proposed policy/minibatch seeds
`3700003000…3700003009`. Within each stream, clone identical incoming tensors,
empty Adam and private/global RNG state into all four branches and reuse one
`[10000,256]` minibatch tape. Each fit uses exactly 10,000 Adam updates, batch
256, learning rate `3e-4`, betas `(0.9,0.999)`, epsilon `1e-8`, gradient clip
`0.5`, ordinary action MSE and no proximal term or anchor refresh. No branch
continues training from another condition. Preserve each actual endpoint and
use that exact checkpoint for all measurements, without refitting.

The proposed fresh evaluation resets are `3700100000…3700100031`; the disjoint
eight 100-control standing probes are `3700200000…3700200007`. These roles and
the remaining `3700…` namespace require a complete historical collision scan
before freezing. None is allocated or executed by this document. The old 39
students and all six J2.0 children remain development-only, never confirmation.

Retain J2.0's source/replay and reference-alone prerequisites: complete bound
contexts; 63 held-reference anchors replayed for 25 controls; byte-exact initial
restoration; continuation contexts and costs at `rtol=0, atol=1e-10`; and at
least 15/16 walking in each of the reference's two fresh panels before fitting.
Stop on source, task, reference, replay, matching, numerical or budget failure.
Report every clean student's outcome without requiring a minimum walking rate;
do not replace or exclude a weak stream.

The primary diagnostic remains the fixed 100-control standing-start on-policy
MSE against SAC on the student's actual pre-action observations, pooled over
the eight probes and six torques. Stop each probe on a native fall; save its
length and gaps without an artificial suffix. The sole comparator remains
action MSE on all 7,000 held clean SAC rows. Preserve mean probe duration, fall
count and their within-condition associations with walking as secondary controls.
The 25-control prefix is secondary and uses these same probes.

For every child, retain all 32 standalone episodes and report walking, full
survival, observed duration/censoring, native terminal cost, absorbing-tail
cost, complete cost and reconstruction separately. Walking uses the unchanged
full-horizon speed criterion. Full survivors are censored at eight seconds.
Keep the native fall charge once and five per missing control; never fabricate
post-fall states. The two 16-reset panels support the existing individual
15/16-each convention, separately from the diagnostic claim.

## One proposed primary test

For each condition \(c\), compute average-tie Spearman correlations across all
ten fresh streams. With walking count \(W_{sc}\) and errors \(E_{sc}\), define

\[
u_c^{\rm on}=-\rho_s(E^{\rm on}_{sc},W_{sc}),\qquad
u_c^{\rm off}=-\rho_s(E^{\rm off}_{sc},W_{sc}),\qquad
d_c=u_c^{\rm on}-u_c^{\rm off}.
\]

Use signed utilities, not absolute correlations. The proposed admission rule
requires mean on-policy utility across the four conditions at least 0.7, mean
advantage at least 0.15, at least three of four condition advantages
nonnegative, and a one-sided 95% lower whole-seed bootstrap bound on mean
advantage strictly above zero. All original condition correlations must be
defined. A constant original outcome or score makes the primary comparison
ineligible; retain that result without dropping a condition or adding seeds.

For uncertainty, draw 5,000 resamples of ten whole learner-seed blocks with
replacement, applying the same sampled seed IDs to all four conditions and both
scores. Do not resample conditions, frames or reset episodes as independent
students. Recompute ties and all four correlations in every draw. If any draw's
required correlation is undefined, assign that draw's aggregate advantage
**-2**, the least possible utility difference, and report the total and
per-condition undefined-draw counts. Do not discard or redraw it. The proposed
lower endpoint is the empirical fifth percentile using the `inverted_cdf`
quantile convention. A proposed private PCG64 seed is `3700400000`, subject to
the same namespace review.

This conservative treatment and the finite ten-block percentile procedure
require statistical review before freezing. They are a proposed conditional
decision rule, not a guarantee of exact coverage or a randomized causal test.
The ten streams, shared clean data, fixed noise profiles and shared reset panels
define the inference scope. No generalization across new corpora, gaits or
environments follows. Secondary AUC, pooled correlations or a favorable subset
cannot rescue the primary result. No score or gate will be selected using the
fresh outcomes.

## Exploratory stability column, separate from the primary claim

The parallel session's calibration used policy continuations from MPPI states
and retained only summary correlations, including a tie-handling limitation.
This proposed column has a different starting distribution and an explicit
identity control. It is ungraded and cannot change primary admission.

At each of the same eight probes' healthy initial **full simulator contexts**,
reuse the actual nominal first 25 controls and their `N+1` states. No second
nominal policy rollout is needed. Restore the identical full context and replay
its literal nominal actions for at most 25 controls as a zero-nudge identity
check, including the original terminal event. Require byte-exact restoration
and the existing `rtol=0, atol=1e-10` state, warmstart, integration and native-cost
continuation contract. No relaxed replay tolerance is proposed.

Create four fixed Gaussian directions from proposed PCG64 seed `3700300000`,
shared across all streams, conditions and probes. In the 17 physical coordinates
comprising height, seven angles and nine velocities, use
\(\delta_k=0.05\,\operatorname{diag}(\sigma)\xi_k\), with
\(\xi_k\sim\mathcal N(0,I)\). Do not perturb time or root-x. Compute the fixed
population standard deviations \(\sigma\) from the 9,000 action-aligned clean
SAC **physical states**, not their velocity-clipped observations. Bind the
array, ordering and finite positive scales before launch. This geometry is
reference-calibrated; it is not teacher-free calibration. Online perturbed
actions come only from the evaluated child's fixed actor, through its unchanged
clipped-observation interface.

Restore the same context for each direction, change only the declared physical
coordinates, and run that child for at most 25 controls. Preserve all other
serialized context fields under a declared, calibrated restoration procedure.
The zero-delta version of that same procedure must pass the identity check;
otherwise stop rather than attributing restoration differences to stability.
Stop each branch at its first native fall, including an initially unhealthy
perturbation, and retain its duration and fall flag. Do not continue physics
past termination.

For nominal and perturbed states, compare height and un-clipped velocities
directly, wrap angle differences into \([-\pi,\pi)\), and divide each physical
coordinate difference by its fixed \(\sigma\). For four branches let

\[
D_t=\sqrt{\frac1{4\cdot17}\sum_{k=1}^4
 \left\|\operatorname{diag}(\sigma)^{-1}
 \operatorname{diff}(x_t^k,x_t^0)\right\|_2^2},\qquad
S=\log(D_{25}/D_0).
\]

If any nominal or perturbed branch falls, record that probe's score as missing
with all branch falls and durations separately. There is no replacement penalty
of five and no made-up final state. Preserve raw \(D_0,D_{25}\); require a positive
finite initial spread. Exact zero final spread represents mathematical
\(-\infty\); its storage convention must be fixed during calibration, without
substituting an arbitrary favorable finite score. A case-level exploratory mean
requires all eight probe scores to be defined; otherwise leave it missing and
report availability. Do not average only a favorable subset.

This is a finite-time directional RMS growth assay from standing starts over
0.2 seconds. It is not a Lyapunov exponent, a novelty claim, pre-update
teachability or proof of recovery. Contact-regime changes and the difference
between physical measurement geometry and clipped policy inputs must be shown.
Any association with walking remains exploratory, with missingness visible.

## Proposed finite budget and prerequisites

| Work | Full-roster maximum |
| --- | ---: |
| Fits / Adam updates | 40 / 400,000 |
| Training rows in each forward/backward stream | 102,400,000 |
| Standalone controls, `40×32×1000` | 1,280,000 |
| Existing diagnostic probes, `40×8×100` | 32,000 |
| Four perturbations, `40×8×4×25` | 32,000 |
| Zero-nudge literal-action checks, `40×8×25` | 8,000 |
| Reference qualification / recorded replay | 32,000 / 1,575 |
| Total native controls / physics substeps | **1,385,575 / 5,542,300** |

Proposed hard ceilings are one worker-hour, 15 audit-minutes and 4 GiB of
study output, with 1,400,000 native controls, 5,600,000 physics substeps,
110,000,000 worker neural-forward rows and 102,400,000 backward rows.
The audit has at most 5,000,000 neural rows and repeats no fitting, gradients
or physics. All source and ancillary work, including direction preparation,
copies, restores, failures, I/O, publication and engineering, requires a complete
nonoverlapping census and actual exit receipts before the protocol can freeze.

Necessary implementation checks include matched actor/Adam/RNG/tape continuation;
exact endpoint reuse; ties and constant conditions; whole-seed bootstrap
indexing and undefined draws; zero-nudge identity; known synthetic contraction
and expansion; angle wrapping and un-clipped physical velocity geometry;
fall/missingness rules; source/capacity/budget stops; and saved-only independent
recomputation of outcomes and scores. Do not use live result-dependent tuning to
calibrate perturbation amplitude or score thresholds.

No execution configuration or launch command is supplied here. This proposal
must receive source/input, statistical and perturbation review before a
separately frozen version can run. A positive post-fit diagnosis would still
leave pre-update learning prediction and useful MPPI coupling untested.
