# J2.0: fresh-seed confirmation of post-fit walking diagnosis

**Frozen before the first native control or fresh learner update.** Launch
requires the published source export to pass its clean-export validation and
the source/input checks below. The configuration is
[jitter_diagnosis_j2_0.json](../configs/studies/jitter_diagnosis_j2_0.json).
This version tests one post-fit diagnosis claim. It does not test pre-update
teachability, a learning mechanism or an MPPI–policy loop.

## Claim and fixed comparison

On six fresh learner streams, does the trained policy's action disagreement
with the existing SAC reference on its own briefly visited states diagnose
standalone walking better than held-reference-state action MSE? Both scores use
the exact saved actor whose walking is evaluated. There is no actor refit to
reproduce a scalar error, no selection of probe length and no selection among
scores on the fresh results.

The old 39 students and their shared sources, seeds and resets are development
evidence only. Their exploratory AUC and [tie-corrected review](similarity_j1_review.md)
motivate this comparison; they do not count as confirmation. In particular,
one weak learner seed appearing under several conditions is not several
independent confirmations. The old records contain 39 separate fits grouped by
three repeated seed IDs, rather than one failed fit counted three times.
Within those seed groups, offline/on-policy utilities were respectively
0.8120256746/0.8120256746, 0.4629100499/0.4629100499 and
0.8426021685/0.8707933192. Their behavior motivates fresh grouped confirmation,
not a causal claim about a single bad seed or a confirmed algorithm advantage.

## Inputs and 13 fixed conditions

Reuse the nine saved SAC rollouts 0–8 for 9,000 training observations/actions;
hold out rollouts 9–15 for 7,000 clean action-MSE rows. Use the same native
17-observation, six-torque interface and existing SAC checkpoint. The exact
arrays, row ordering, reference tensors, XML, labels and source files require
SHA-256 and byte-count bindings before freezing. The input inventory binds the
K1 raw HDF5 (`2b73fc8e…`, 1,239,388,136 bytes), reference checkpoint
(`174e9886…`, 3,524,729 bytes) and original MPPI traces (`b5a71258…`,
44,935,839 bytes); full hashes and local paths are explicit in the
configuration. Execute from a create-only local source export on macOS arm64,
Python 3.11.15, CPU PyTorch 2.10.0, MuJoCo 3.6.0 and NumPy 2.4.3, with one
Torch thread. The prepared manifest SHA-256 is
`5d59723d610e56f2f8fd69f6fb7ab2eb90ed9dd0e3b1ada636e5bb5616d5e615`;
it binds the corpus, reference checkpoint, clean normalization, all 13 label
arrays and the clean-data affine map. Its full bindings and source-file seals
are in the configuration. The create-only source export must match those seals
before execution; the inventory alone is not launch admission.

There are 13 conditions: clean labels once, then MPPI-derived, white and slow
jitter at nominal signal shares 0.9, 0.75, 0.5 and 0.25. Preserve the earlier
label-generation arithmetic and order: the same nine complete default-MPPI
control sequences are pooled-centered/standardized per torque; white noise is
Gaussian; slow noise uses the original fourth-order zero-phase 2 Hz filter at
125 Hz and pooled scaling. The private label RNG is PCG64 seed 20261102, with
the original generation order. Prefer bound saved label arrays; reconstructing
them must close exact source/array checks before fitting.

For each non-clean condition, labels are

\[
 y_{j,t}=\operatorname{clip}_{[-1,1]}
 \left(u^{\rm SAC}_t+
 \sqrt{\operatorname{Var}(u^{\rm SAC})\frac{1-f_j}{f_j}}\,\epsilon_{j,t}\right).
\]

The same condition arrays are used for all six streams. Signal share specifies
nominal pre-clipping variance, not action amplitude. Preserve clipping rates.
These label perturbations are not newly executed physical teacher trajectories.
No MPPI query, teacher reacquisition, critic, architecture intervention or video
is included. Before fitting, replay 63 held-reference anchors: each of seven
held rollouts at controls 100, 200, …, 900 for 25 recorded actions. Require
byte-exact snapshot restoration and all replay contexts and native costs to
match at `rtol=0, atol=1e-10`. Missing full contexts or any mismatch stops the
version without source replacement or tolerance changes.

## Matching and training

Use learner stream seeds `3600003000` through `3600003005`. Each initializes one
width128 tanh actor, empty Adam and private/global RNG package. Compute one
normalization from the 9,000 clean training observations: float32 corpus mean
and standard deviation with floor `1e-6`. All conditions share it.

Within each stream, deep-copy the complete incoming package for every condition
and reuse the identical `[10000,256]` minibatch-index tape. Each child receives
all 9,000 unique rows and exactly 10,000 updates: Adam learning rate `3e-4`,
betas `(0.9,0.999)`, epsilon `1e-8`, gradient clip `0.5`, ordinary torque MSE,
proximal weight zero and no anchor refresh. Preserve numerical operation order,
normalization, parent tensors, checkpoint keys, Adam state and RNG/tape joins.
No condition receives warm-started tensors from another condition. Save each
actual endpoint checkpoint before executing it; probe and evaluation load that
same hash, never a new fit.

Before any fit, execute the reference alone on the same 32 fresh standalone
resets. Treat the first and last 16 IDs as separately reported panels and
require at least 15/16 walking in each. A reference or replay qualification
failure stops before fitting. This reference control does not label
student-visited states or establish recovery there.

Then fit the six clean children and execute their 32-reset standalone panels.
Require at least **90% pooled walking** (`≥173/192`) and at least **24/32 in
every stream**. If any condition fails, close the version before the other 72
fits. Keep all six children and results; do not discard, replace, add seeds or
relax the gate. A failed qualification is a positive-control failure, not a
similarity-score result. If it passes, complete the remaining 12 conditions in
each stream with their original matched packages/tapes.

## Disjoint execution panels and measurements

For every actual child, use the same 32 fresh standalone reset IDs
`3600100000…3600100031`, up to 1,000 controls each. Walking retains the existing
full-horizon survival and mean absolute forward-speed error `≤0.3 m/s` against
`1.5 m/s` criterion. Report walking counts, full survival, observed controls,
native cost, absorbing-tail cost and complete cost separately. Apply the native
fall terminal once and five per missing control; do not fabricate post-fall
states. Full survivors are right-censored at eight seconds, not observed falls.

The primary diagnostic uses eight separate standing-start reset IDs
`3600200000…3600200007`, each at most 100 controls. At each actually visited
pre-action observation, save the child's action and the frozen SAC actor's
action; execute only the child's action. Stop a probe on a native fall. Pool
the actual rows across all eight probes and average squared error over rows
and six torques. Preserve each probe's length, termination, observations and
action gaps. This fixed observed-prefix score has no fabricated suffix; early
falls and unequal probe lengths remain explicit limitations. Do not substitute
25-control, mid-stride or selected-probe variants as the primary score. A
secondary 25-control score uses only the first 25 actual rows of these same
probes and causes no additional execution.

The sole primary comparator is action MSE on all 7,000 held clean SAC rows.
Training reconstruction and complete learning curves are fitting measurements,
reported separately from walking. Scores and walking are joined by the exact
endpoint checkpoint hash and stream/condition IDs.

Before-update J1 measurements are stored as **ungraded exploratory records**
where their definitions are applicable. A/B compare bound labels with the
incoming actor's predictions and aggregate the nine trajectory-level means
equally. B uses the fixed five-control causal box within each trajectory.
C uses the existing J1 affine-explainer formula, fitted on the fixed old clean
9,000-row reference corpus with its actual clean normalization before any fresh
learner fit; save coefficients in the input packet as a development-only
transform. Its per-trajectory values are also averaged equally. D is null and
inapplicable because perturbed labels are not physically executed target motion.
E is null and inapplicable because this study acquires no fresh incoming-policy
occupancy. Do not substitute conditional teacher-state pairs or fabricate
rollouts. None of these records enters score selection, the primary test or
acquisition.

## Primary analysis and admission

Each fresh stream supplies 13 condition-level points: walking count out of 32,
primary on-policy error and held-row comparator error. Compute tie-corrected
Spearman correlations using average ranks. Define utilities
`u_on = -rho(on_policy_error, walking_count)` and
`u_off = -rho(held_action_mse, walking_count)`; do not use absolute correlation.
The six paired advantages are `d_s = u_on - u_off`.

The single claim passes only if all these conditions hold:

- All six stream correlations are defined. A constant outcome or score is
  ineligible; preserve it and fail admission without selecting replacement
  conditions or seeds.
- Mean on-policy utility is at least `0.7`.
- Mean paired advantage is at least `0.15`, and at least five of six advantages
  are nonnegative.
- The one-sided exact paired sign-flip test has `p≤0.05`. Enumerate all `2^6=64`
  sign assignments, use mean signed advantage as the statistic and count
  assignments at least as large as the observed mean, including ties and the
  observed assignment. Divide by 64; no Monte Carlo or pseudocount is needed.

The exact sign-flip calculation is a conditional finite decision rule under
independent seed-wise differences and a sign-symmetry null. It is not a
randomized causal-treatment test. The six learner initializations are the paired
units, conditional on fixed clean inputs, noise profiles and reset panels.
Conditions and reset episodes are not independent learner replications. This
finite confirmation does not establish generality across new corpora, gaits,
references or environments. The old within-seed results show why the stricter
fresh confirmation may fail; the 0.15 advantage gate remains fixed after seeing
new results.

Secondary reports include per-stream and pooled AUC for any walking (`≥1/32`
versus zero), walking-count plots, survival, cost, clipping and reconstruction.
Any-walking AUC is not competence. These analyses cannot rescue a failed primary
test. Mean actual probe duration and probe fall count are secondary controls
computed from the same traces, with no extra execution. They expose whether
disagreement mainly tracks already observed short-probe failure. Individual
competence requires at least 15/16 walking on each of the two declared disjoint
16-reset panels. No learning or guidance mechanism is admitted by a successful
post-fit diagnostic.

## Finite work, checks and stopping

The full roster is at most 78 actual fits, 780,000 Adam updates and 199,680,000
training rows in each forward/backward stream. The two panels imply at most
`78*(32*1000 + 8*100) = 2,558,400` native controls. Add at most 32,000 reference
qualification controls and 1,575 replay controls, for **2,591,975 controls and
10,367,900 physics substeps**. The hard caps are 2,600,000 native controls,
10,400,000 physics substeps and 205,000,000 worker neural-forward rows. The
backward-row cap is exactly 199,680,000. Six clean qualification fits/panels are
part of the main roster, not duplicated on passage.

Hard stage limits are **one worker-hour**, **15 audit-minutes** and **4 GiB of
study output** across named local/remote study roots. The complete primitive
census, I/O reservations, input bindings and owner/exit receipts must close in
the final executable configuration before launch. Count engineering,
transfers, qualification failures, audit reads/inference and publication work
separately; do not treat elapsed Git chronology as labor accounting.
The worker output ceiling is 3 GiB, reserving 1 GiB for the combined prepared
inputs, source, audit and compact reporting roots within the total 4 GiB.
Existing input acquisition and reference training are inherited costs, whose
complete historical totals are unavailable. This run cannot establish total
MPPI-versus-RL efficiency. Logical reads and hashes are measured separately from
physical disk I/O, which is unknown.

The frozen ancillary maxima are 78 fits, 84 policy constructions, 78 optimizer
constructions, one reference actor/checkpoint load, one environment, 3,152
resets, 63 restores, 6,367 external forwards, 2,595,253 serialized snapshot reads
and 3,432,962 neural calls. Count 15,360,000 preallocated index values separately
from 199,680,000 actual fit index values. The full worker-forward maximum is
203,634,800 rows, below its 205,000,000-row ceiling. These are full-roster upper
bounds; all paid prerequisites and stopped prefixes remain in actual accounting.
The audit has separate 900-second, 64-GiB read, 4-GiB peak-RSS, 16-MiB output and
5,000,000-forward-row ceilings, with zero physics, gradients or updates.

Before source freeze, test tied/constant ranks, exact sign enumeration, the
qualification boundary, matched parent/tape/Adam/RNG joins, exact endpoint
reuse, disjoint panels, fall/censor/tail accounting, inapplicable score records,
budget stops and missing bindings with small synthetic fixtures. Keep simulator
imports lazy. Check historical seeds for collisions and all source hashes.
The independent audit recomputes saved outcomes, gaps, rank statistics and
accounting and verifies checkpoint/provenance joins without repeating fitting
or physics. Its separate neural-forward ceiling is 5,000,000 rows within the
15-minute audit allocation.

Stop on any binding, matching, numerical, qualification, execution, ownership,
audit or budget failure; preserve paid work and actual process exits. No seed
replacement, tolerance relaxation, additional condition, teacher acquisition or
automatic follow-up is allowed. The final report must distinguish a failed
qualification, a failed diagnostic, an ineligible test and a passing finite
post-fit diagnosis. It must not call any of them a closed learning loop.
