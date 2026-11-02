# H4.1 prospective acquisition protocol, round 1

This protocol implements specification revision 3 of
`docs/adaptive_coupling_h4_1.md`. It is a development-informed test of one
outcome-triggered weight reduction, not proof that a feedback schedule beats a
lower fixed weight. The prepare command creates a new source/configuration/
checkpoint/input snapshot; all those hashes must verify before a worker starts.
No prior run, configuration, dataset, checkpoint or ledger is modified.

**Prelaunch local-platform revision:** automatic approval review rejected
transferring the new H4 code/docs to the remote host; no bytes or experimental
queries were sent. The previous remote-platform protocol and rejection are
preserved under `artifacts/2026-10-30_adaptive_coupling_h4_1_001`. Run this
experiment locally instead, using the same native Walker model, task, sources,
checkpoint and history already present here. Freeze the exact macOSarm64,
Python3.11.15 and installed dependency versions in configuration; worker refuses
an environment mismatch. MuJoCo3.6.0, NumPy2.4.3, SciPy1.17.1, Torch2.10.0 and
h5py3.16.0 are the local versions. The historical trigger/checkpoint came from
Linuxx86/Python3.11.16/TorchCPU; all prospective rho arms share the same local
platform. This isolates their within-run comparison, not cross-host bitwise
reproduction or a cross-host effect. Inputs remain byte-identical; no feature,
weight, task, threshold, seed or budget is changed by this safer execution path.
Bind both the local historical-ledger seed audit and the root's read-only
canonical-host seed check before launch.

The incoming experimental learner is the actual v3.1 guided +2k child, policy
hash `7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916`.
Every arm uses its same complete Adam, PCG64, normalization, private generator
and proximal metadata, with the exact 6,096-row guided history that trained it.
The old matched 0-vs-3 full-survival signal triggers half weight; current action
agreement, data MSE and cost do not enter this rule. This checkpoint is not a
qualified deployment policy and its lack of walking does not block this test.

The six fixed teacher attempts are two independent episodes at rho=0, two
fixed-high episodes at rho=0.1199341983507337, and two backoff episodes at
rho=0.05996709917536685. Each uses reset `952000000+i` and private planner seed
`952100000+i`, i=0,1. Preparation checks proposed seeds against all prior saved
run ledgers and configurations, retaining their hashes. A collision or changing
ledger stops preparation. All arms retain the same native Walker task and
MPPI configuration, short horizon, independent producer, existing selector,
and documented baseline-mean omission. No policy initializes or updates the
private MPPI nominal or proposal RNG. The positive-weight selectors use only
the updated incoming actor; every actual decision's actor/rho/enabled state is
verified. Full before/after planner snapshots, candidate banks and native query
inputs/outputs are retained by the existing Meter and teacher interfaces.

Acquire all six attempts through 1,000 controls or native termination before
applying the task gate. Eligibility requires all six complete task successes
(full survival and velocity MAE<=0.3m/s), and each positive-weight pair's mean
complete cost <=1.05 times independent. A failed required teacher gate stops
child fitting, retaining the complete six-attempt roster and the unchanged
incoming updated policy's16 primary and two acquisition-start evaluations. The
latter provide standalone outcomes and paired incoming-parent/teacher gaps
even if no child can be fitted. They are a subset of the already-budgeted
parent cohort, not extra work beyond the maximum roster. No action-agreement
test, reference-distance test, per-slot substitution, extra reset or weight
retry is allowed. Software/ownership/numerical-revocation errors stop at the
error boundary and retain partial work rather than continuing invalid science.

If admitted, each of three datasets has 6,096 identical retained rows plus
2,000 own actually executed teacher pairs, in episode/time order: 8,096 rows.
Record full float32 target hashes and preserve NPZ datasets and HDF references.
Fit three children from exact clones of the incoming learner for 2,000 Adam
steps each, batch256, learning rate0.0003, mean-parameter proximity0.01,
anchor refresh250 and gradient norm clip0.5. Continue the incoming PCG64 stream;
save its full2,000x256 sampled-index array and verify identical terminal states.
Fixed checkpoints500/1,000/2,000 resume the same block. Full-data MSE is measured
once for parent on each dataset and once per own child checkpoint:12passes,
97,152 forward rows. No checkpoint selection or standalone learning-curve
cohort is added. All fitting minibatch curves are retained.

Parent plus three final children execute only the torque policy at every
control on16 fresh paired primary resets `952200000+i`. They also execute the
two acquisition resets as an explicitly training-start paired teacher-student
gap diagnostic, regardless of primary result. Every cohort verifies full
initial state/warmstart/integration/observation equality across policies.
Success, survival, controls, velocity error, native termination and complete
cost remain separate. Complete cost uses the existing native terminal once
and5 per unexecuted control after a fall.

Round2 reserves acquisition954000000, planner954100000, primary954200000 and
confirmation954300000 (offset+2,000,000). The earlier unlaunched+1,000,000
proposal would overlap the parallel L1 study's953xxseeds; this prelaunch
correction changes no weight, outcome gate or paid work. Round1 preparation
checks both prospective seed rosters, and round2 must check again when its
actual incoming child and history are frozen.

The backoff child passes only with at least one walking success and strictly
more than fixed-high; success count no lower than independent or parent;
survival and mean controls no lower than all three controls; at least5% lower
mean complete cost than both new-data children; and no cost regression against
parent. All criteria apply jointly. Standing cannot pass. A passing primary
result admits the same four policies on16 disjoint paired confirmation resets
`952300000+i`, with identical criteria. Otherwise stop. Every paired per-start
delta is retained. A confirmed benefit permits a separately frozen second
round using the backoff child and8,096-row history. Because passing requires
success/survival nonregression against independent, the next rule necessarily
holds the lower weight. Round2 would test continued updated-policy learning
with that chosen weight, not a second adjustment or schedule superiority.

The complete worst-case roster pays74,569,728 model controls,142,000 live
controls and51,311,152 neural forward rows, including guidance, fitting,
declared data diagnostics, primary, conditional confirmation and paired-gap
diagnostics. The caps are80,000,000 controls,60,000,000 forward rows,
1,536,000 backward rows and6,000 Adam steps. Extra allowance is not permission
for extra science. Budget exhaustion, validation failure or incomplete work
retains all evidence and stops. Source validation, tests, preparation, audits,
exports and transfer timing are separate overhead; no efficiency claim.

After closure, the export command computes and pins the full local raw HDF
and ledger hashes, then mirrors compact complete episodes, training data,
sample indices and full bank proofs at fixed and first divergent decisions.
The large full query corpus and all planner snapshots remain in the preserved
local run; compact exports never replace it. The worker is create-only and cannot
be relaunched into an existing evidence directory. All failed or negative
outcomes remain preserved.
