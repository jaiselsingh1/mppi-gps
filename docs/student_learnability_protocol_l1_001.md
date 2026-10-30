# L1.001: fresh clipped versus full-velocity student training

Frozen design before new fitting, inference or physics. This is the student
learnability track, not a coupling experiment and not H3.2's inherited-state
continuation. All existing runs, configurations and evidence remain unchanged.

The hypothesis is that the full-velocity representation with its declared
fresh-training pipeline improves competent walking relative to the clipped
pipeline at a fixed budget. Same physical rows/actions do not make this a pure
information-content intervention: map-specific normalization changes input
geometry. The experiment removes inherited weights/Adam as an explanation for
larger full-input magnitudes, but does not isolate all consequences of the
representation/training pipeline.

Use exactly the 6,096 underlying physical states and action targets of v3.1's
independent-data arm, in its frozen order: 4,000 actual discovery controls, 64
hypothetical perturbed-state re-solves, 32 accepted actual recovery controls,
and 2,000 actual independent controls. Every state must bind to saved physical
arrays. Each re-solve binds its own bank digest and exact initial state, not its
nominal anchor. The recovery mask excludes both impulse controls and reproduces
the admitted 32 rows. Native clipped features must reproduce the supplied
float64 observation matrix byte-for-byte; action targets must also match
exactly. Missing provenance stops preparation without new simulation or row
substitution. Preserve the hypothetical/executed distinction.

Feature maps use the 19-entry saved physical state: positions `state[2:10]`
and nine velocities `state[10:19]`. The clipped arm clips those velocities to
[-10,10]; the full arm retains them. Both have 17 inputs and the same
17→128 tanh→128 tanh→6 tanh bounded torque architecture. Each normalization is
computed only from its own full training feature matrix, with float64 means
and standard deviations, the same 0.05 standard-deviation floor, then float32
storage. Seal both arrays before fitting.

Initialize both fresh networks with the same private parameter generator seed
20261029. Verify identical trainable parameter bytes and private generator
state, separately from the legitimately different normalization buffers and
policy hashes. Start both Adam states empty, without inherited moments or
counters. Use PCG64 seed 953000000 and preserve the exact common 10,000×256
minibatch indices. Both arms receive exactly 10,000 updates, batch 256, learning
rate 0.0003, action MSE, gradient norm clip 0.5, zero parameter penalty and no
anchor refresh. Checkpoints at 2,000, 5,000 and 10,000 updates resume one block.
No early stopping or checkpoint selection follows physical outcomes.

The unchanged original Bfree 10k clipped-input parent is a reference control,
not the initializer of either fresh arm. Native MuJoCo dynamics, task, clipped
native observations and complete 1,000-control horizon are unchanged. Standalone
controllers execute only direct torques. The full-input controller derives
declared features from its actual physical state; do not change native
observations or accidentally feed clipped observations to that child. Save and
verify actual declared controller features against episode pre-action states.
There is no teacher prefix, fallback, planner, critic or new acquisition.

All seven fixed policies (original parent plus both arms' three checkpoints)
receive the same four descriptive curve starts at 953110000+i, i=0..3. Final
10k policies and parent receive the same 16 primary starts at 953100000+i,
i=0..15. If **either** final fresh arm has at least 15/16 walking successes,
evaluate all three final policies on 16 further confirmation starts at
953120000+i. Otherwise stop. Check that these 36 exact seeds have not occurred
in canonical historical run ledgers before sealing. Match full reset states,
warm starts and integration state as well as seeds.

Success requires full survival through control 1,000 and forward-velocity MAE
at most 0.3 m/s against the existing +1.5 m/s target. Complete native task cost
charges the fall terminal once and 5 per unexecuted control after falling.
Survival while stationary is reported separately and is not walking. All
outcomes permit different successful paths; there is no path-distance filter.

Competence is declared per fresh arm only if it has at least 15/16 walking
successes on both primary and confirmation cohorts. Full-feature benefit is a
different predicate: full has at least 15 successes, success/survival/mean
controls no worse than clipped, and at least 5% lower mean complete cost than
clipped, on both cohorts. A clipped-only competence result is still confirmed
under the either-arm trigger. Report every paired outcome and all failed
predicates, regardless of these gates; neither gate establishes population-wide
competence or input information as the unique cause.

Save every minibatch learning point. Full-data MSE is measured once for the
original parent on clipped data, once for each fresh initializer on its own
feature map, and once per child checkpoint on its own map: nine passes totaling
54,864 forward rows. It is diagnostic and does not choose data or policies.
The maximum schedule is 124 standalone episodes/124,000 controls, 5,298,864
neural forward rows (5,120,000 fitting +54,864 diagnostics +at most124,000
standalone actions), 5,120,000 backward rows and 20,000 Adam steps. Hard finite
caps are 250,000 requested controls and 6,000,000 forward rows; no unused
allowance authorizes additional episodes, labels, diagnostics or retries.
Model/planner controls must remain zero. Abort and retain partial work on
budget exhaustion, provenance/source mismatch or runtime failure.

Prepare a create-only source/configuration/checkpoint snapshot and a supplied
input hash manifest before worker launch. The saved-state preparation is
read-only and pays zero experimental controls/inference/learning. All new
controls, forward/backward rows and optimizer steps pass through Meter, and
full actual episode snapshots and declared feature arrays remain preserved.
Source preparation, tests, audits, plotting, transfers and reasoning are
separate overhead. Stop after the fixed roster and conditional confirmation;
do not change normalization, initialization, feature map, update count or
evaluation seeds after outcomes. A negative result is retained, not retuned.
