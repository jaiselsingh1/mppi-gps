# L2.0 — local MPPI coupling, fixed learning, and conditional feedback

**Frozen recipe; `launch_ready=true` requires sealed source/inputs/runtime and the native prerequisite.** The user authorizes this bounded study after its
implementation, tests, input/runtime/source seals and independent review pass.
The [machine-readable recipe](../configs/studies/local_coupling_learning_l2_0.json)
is frozen before native work. No experimental result is implied by this protocol.
P1.0/P1.1, the six-round loop, A/B/C and F1.0 remain preserved under their own
versions; their failures and stop rules are not rewritten.

Test whether a fixed torque reconstruction penalty changes **competent local
MPPI behavior**, yields more useful demonstrations after a fixed learning
update, and can support a second acquisition–learning round with the updated
actor. Current action proximity is neither an admission test for teachability
nor the outcome. Standalone held-start walking is the learning outcome.

## Inputs and controller

Bind W2 s0's exact update-2,000 checkpoint (SHA-256
`f8014db94bc5f32910cfaf12d0ab38a462bbf950f2567580729d02929e6cc03d`),
its Adam/private Torch/PCG state and frozen normalization. Copy the already-bound
P1 `initial/stream0` global envelope exactly; it does not recover W2's unrecorded
historical global RNG. Checkpoint bytes, actor tensors and optimizer fingerprints
are separate hash contracts. Machine locations and full SHA-256s are explicit in
the configuration and copied-input manifest.

Restore all eight existing complete contexts at controls
`1,17,34,51,67,84,101,118` of the saved student episode. These are eight contexts
from **one episode**, not eight independent task replications. Preserve full
state, solver warmstart, integration arrays, state specification and model
identity. A/B/C's 14/256 candidate-survival failure remains in the record; this
version deliberately has **no candidate-survival gate** and tests actual
receding-horizon execution instead.

Run all 24 first-round teacher episodes: eight contexts × fixed
`alpha ∈ {0, .12, 3.4744413290249314}`, up to 300 actual controls each. The larger
weight is the historical calibration, not a fresh search. Use the existing
`MPPI.plan_step`, `H=48`, `K=256`, `lambda=3`, sigma `.3`, unchanged order-2
23.4375-Hz low-pass covariance at 125 Hz and bounded actions. Start each episode
with a zero plan. Two refinement passes precede the first action and one each
later action; carry and shift the plan exactly once per actual decision. Do not
add an ESS target, zero-noise incumbent, survival filtering, policy centering,
extra forecast or native candidate selection.

For the current fixed actor and actual simulator state, use

\[
 J_\alpha(U;x)=C_{48}(U;x)+\alpha D_\theta(U;x),\qquad
 D_\theta=\tfrac12\sum_{t\in\text{executable prefix}}
 \|u_t-\pi_\theta(\phi(x_t))\|^2.
\]

The feature map stays `concat(state[2:10], clip(state[10:19],−10,10))`.
Scoring, fitting and direct execution retain the bounded **Torch** actor and
inherited float32 normalization; do not deploy P1's NumPy action adapter.
Physical float64 torques are compared with bounded float32 predictions promoted
to float64; fitted targets retain the existing float32 conversion. Preserve the
actual policy calling conventions and quantify any batching parity discrepancy.

Use pre-action observations, include the action causing the first fall, and do
not infer on the unused post-fall suffix. Native task cost charges the fall
terminal 20 once and absorbing 5 for every missing horizon control. An invalid
executable prefix has infinite score; a finite fallen candidate remains a
normal candidate. Alpha zero optimizes task cost alone even when diagnostic D is
computed. Standard MPPI executes the first weighted-mean torque and replans.
**Falls end only their own episode; complete the remaining roster.**

## Pre-fit selection and equal data

After all 24 episodes, choose a positive arm without inspecting any fit:

- Both the independent and selected positive teachers must achieve at least
  **6/8 local walking episodes**: all 300 controls, no fall and velocity MAE
  at most `.3 m/s` about `1.5 m/s`.
- The positive arm must have no fewer local walking or full-survival episodes
  than independent and mean complete cost at most `1.10` times independent.
- Its pooled action RMS difference from independent must be at least `.01`,
  using all actual common `(restart, control, action-dimension)` rows; every
  restart contributes at least one row. This establishes actual influence,
  not teachability.

Prefer `.12` if eligible; otherwise choose the fixed larger weight if eligible.
Neither eligible closes the study after all 24 controllers, without a fit or
weight retry. Save unused-arm outcomes and costs. Report each episode's observed
velocity MAE, pooled MAE and prefix length separately; never invent missing-tail
velocities. Three-hundred-control walking is a **local** teacher outcome, not
1,000-control competence.

Join independent/selected-positive data by `(restart_id, control_index)` and keep
**every common executed prefix row**, in restart/control order. Each of eight
restarts must contribute; equal actual `N≤2,400` is recorded after acquisition.
Do not select complete survivors, low-error states, spectra or favorable rows.
The rule is frozen now; the resulting corpus bytes and row joins are sealed
before fitting. Failed prefixes remain available outside the paired corpus.

## First learning round and confirmation

Create three **minibatch-stream forks**, not three new actors: clone the parent's
stored PCG64 and apply `jumped(fork+1)` once. Materialize/hash each fork before
making its two arm copies. Within a contrast, actor, normalization, Adam,
private Torch/global RNG, sampled indices and update count match. Across forks
only the declared minibatch PCG stream changes.

Each arm receives exactly 2,000 updates on its new equal-sized corpus, batch 256,
Adam `lr=3e-4`, `foreach=False/fused=False`, gradient clip `.5`,
`eta_theta=0`, `prox_reduction=mean`, `anchor_refresh=0`, CPU/Torch/BLAS one
thread. Start a new corpus block while continuing Adam/PCG and cumulative
counters. No parent refit, normalization recalculation, filtering, weighting,
architecture change, endpoint selection or extended fit is allowed. Save every
loss, sampled index/state transition and actual incoming/outgoing checkpoint;
write/reload and verify operative continuation. Final cumulative updates are
4,000 in round one and, conditionally, 6,000 in round two.

Evaluate six children and the single shared parent on fresh paired 16-reset
primary **and** 16-reset confirmation panels (224 episodes). Direct policy only,
maximum 1,000 controls; walking requires complete survival and velocity MAE
`≤.3`. Report raw/complete cost, survival, controls and own/cross/held-trajectory
reconstruction separately. The unchanged 2,000-row W2 validation corpus is
measurement only. Both panels run for each fitted round even when benefit fails.

R1 passes only if the positive child adds at least two walking successes over
**both** independent child and parent in the **same at least two forks on both
panels**, with no walking/full-survival regression against either comparator
in any fork. Individual competence separately requires at least 15/16 walking
on both panels. More survival without walking is retention improvement.

## Conditional second round

Only confirmed R1 benefit plus a passing provenance/accounting audit permits R2.
Keep the selected alpha fixed. Each of the three forks uses its own actual two
R1 actors for two arms × eight same-context controllers: **48 episodes**, again
up to 300 controls each. Use the **same** planner seeds as R1; alpha-zero repeats
are expected under a policy-free score and are charged actual work. Updated
actor hashes must be bound into positive-arm scoring. No additional PCG jump or
learning reseed occurs. Rebuild a new-round-only equal-prefix corpus per fork,
then continue each arm's exact Adam/private/global state for 2,000 updates.

Complete all 48 controllers and all six fixed fits when common prefixes remain
positive, even if teachers deteriorate. Teacher deterioration prevents the
useful-feedback verdict; it is not a proxy stop or reason to hide fitted results.
Evaluate all six incoming R1 actors and six children on fresh paired 16+16
panels (384 episodes). These incoming evaluations supply correctly paired
parent-relative gains; old-panel counts are not substituted.

For the **same at least two forks on both panels**, require:

\[
 (w^+_2-w^+_1)-(w^0_2-w^0_1)\ge2,\quad
 w^+_2-w^+_1\ge2,\quad w^+_2-w^0_2\ge2.
\]

There may be no positive-child walking/full-survival regression against its
own incoming parent or independent child in any fork. Useful feedback also
requires actual R2-positive versus R1-positive action RMS `≥.01` on all common
prefixes, and teacher local-walking/full-survival nonregression with complete
cost `≤1.10` both R1 same-alpha and R2 independent. Teacher nonregression is
required in every fork; the actual-influence and learning criteria must pass
jointly in the same at least two. A saturated incoming parent can make the
additional-success criterion unattainable; do not relax it after observing this.

These are finite two-round feedback-associated results. There is no frozen-old-
actor R2 acquisition control, so this roster does not isolate the causal benefit
of actor feedback from additional fitting. A smaller gap alone does not pass.

## Frozen work, evidence and stop rules

A full episode uses `301×256×48 = 3,698,688` forecast controls. The maximum
scientific roster is:

| Work | Maximum |
| --- | ---: |
| Acquisition episodes / forecast banks | 72 / 21,672 |
| Model controls | 266,305,536 |
| Acquisition / evaluation native controls | 21,600 / 608,000 |
| Total native controls / physics substeps | 629,600 / 1,067,740,544 |
| Fits / Adam updates / backward rows | 12 / 24,000 / 6,144,000 |
| Candidate forward rows before prefix masking | 266,305,536 |
| Evaluation episodes | 608 |
| Full saved candidate banks | ≤288 |

Hard global ceilings, including preparation, failures and audit, are 270 million
model controls, 660,000 native controls, 1,082,640,000 physics substeps, 285 million
forward rows, 6,146,048 backward rows and 24,008 Adam updates (eight synthetic
continuation updates are already paid in the component fixture). No additional scientific roster is
permitted by unused capacity. Reserve **23,400 aggregate CPU seconds**, split
preparation/tests 1,800, worker 18,000, audit 3,600; and **19,800 active pool-wall
seconds**, split 1,800/14,400/3,600. Use one worker, four native physics threads,
one Torch/BLAS thread, no GPU, 4 GiB RAM and at most **8 GiB output** (6 worker,
2 audit). The audit total includes saved reporting/rendering/publication,
with **300 CPU and 300 wall seconds reserved** for that closure. Intermediate
R1 audit is limited to **1,650 CPU/wall seconds and three million forward
rows**. Final audit receives `3300 − actual R1 audit CPU/wall` seconds and
`8 million − actual R1 completed forward rows`, rather than a new full audit
allocation. Portable live process-tree CPU/RSS guards and phase RLIMIT/deadlines
enforce worker+audit+report totals of **21,600 CPU / 18,000 active wall seconds**;
the conservative 1,800-second preparation reservation is added once to reach
the declared global totals. A tiny synthetic audit Torch-parity check may use
at most 32 preparation forward rows, with actual counts recorded before freeze;
it authorizes no W2 fit or physics and no repeat of the eight-update fixture.
The historical saved bank measured `.520609 CPU s/.137374 wall s`;
linear extrapolation is 11,282.638 CPU seconds before all other work. It was an
early-falling bank, not a healthy-trajectory benchmark; the finite margin is a
reservation, and exhaustion preserves a partial result rather than extending it.

Worker native verification replays the first-restart episode of each R1 alpha
and each R2 fork/arm (at most nine × 300 controls), plus candidates 0/255 from
each episode's first bank through first fall or 48 controls (at most
72 × 2 × 48). The maximum **9,612 native controls** is included in the global
ceiling. Preserve raw H5 states/warmstarts/integration/actions/costs, actor
predictions and C/D components; compare restoration byte-exactly and replay
states/warmstarts/integration with `rtol=0, atol=1e-10`, native costs with
`rtol=0, atol=1e-9`. For candidate reconstruction, compare every corresponding
bank/native action prediction component with absolute tolerance `1e-6`, retaining
all raw predictions/errors. A native call has `L≤48` rows whereas a bank may
predict up to 12,288 rows. With torques and predictions bounded by one,
`|.5(u−p)^2−.5(u−q)^2|≤2|p−q|`; therefore compare aggregate candidate D using
`rtol=0, atol=1e-10 + 2*(6L)*1e-6`, with **actual** executable prefix length L.
The `1e-10` term covers float64 reduction accumulation. This propagates the
per-action numerical bound instead of assuming identical batch grouping.
Independent source full-bank D retains `1e-6` tolerance under matching grouping.
The action function, fitting and physics tolerances are unchanged, as are all
historical P1 contracts. A mismatch stops without relaxation.
Formal prerequisites add only eight native controls, four model controls and
48 substeps. Reuse the **already-executed** student-component fixture:
`A2→A2` and `B2→B2`, two exact paired 2+2 continuations costing eight total Adam
updates and 2,048 backward rows. Bind that fixture's source/test receipt; do not
run a second formal eight-update fixture or describe this as 2+2 versus a
separate four-update fit. Actual R1-final→R2-initial Adam/private-PCG/private-
Torch/global equality and paired sampled indices supply the study continuation
proof. Existing P1 split-versus-whole tests remain separate historical evidence.
Combined
maximum model/native/physics counts are 266,305,540 / 639,220 / 1,067,779,040.
Independent audit has no physics; these paid native checks belong to the worker.
Its total neural-row allowance is eight million, inside the global
285-million cap. Full native episode replay compares saved warmstarts and
integration arrays. Candidate replay preserves those arrays and verifies its
complete initial context plus forecast poststates/C/D; the batch forecast does
not expose post-warmstart/integration counterparts, so no such comparison is
claimed.

Save all-bank C/D/J, weights, prefix lengths, nominal updates and RNG transitions.
Save full candidate banks only at `(control,pass)=(0,0),(0,1),(149,0),(299,0)` if
reached; absent post-fall banks stay absent. Preserve complete native N+1 traces,
all three R1 arms, row/corpus seals, checkpoints, fitting/evaluation curves,
joint-angle/velocity plots, source/runtime receipts and requested/completed/
unknown-partial ledgers. Independently recompute every saved physical outcome,
row join, alpha choice, continuation, criterion and accounting total. Regenerate
saved noise/weighted updates without physics; verify C/D/J on the preserved full
banks. Declare sparse candidate-state audit coverage. Audit has no new physics,
fitting, evaluation or MPPI queries.

Before launch, verify full-context restoration, pre-action indexing, score and
fall accounting, standard plan shift, paired RNG, the saved paired-continuation fixture,
all-prefix joins, criteria, budget rejection and source/runtime/input hashes.
Source tests and runtime binding failures stop before research execution. During
execution, software/numerical/provenance failure or budget exhaustion stops the
study and retains paid evidence; physical falls do not cancel its roster. A
negative/unconfirmed R1 ends before R2. Stop after R2/audit without a third round,
weight retry or automatic follow-up.

H4.1 already tested an actual feedback-driven weight change and regressed; do
not relabel it as a non-coupling comparison. L2 differs in direct carried-plan
H48 MPPI, fixed positive selection, W2 s0/eta-zero continuation and new-round-only
matched corpora. It does not establish superiority to equal-compute plain
cloning, low-pass cloning or DAgger, the paper's convergence/ablation pattern,
a second body, or asymptotic convergence. Historical six-round, R0/P1.1 and A/B/C
costs remain separately visible; unused allowances do not establish equal
expenditure. Preserve every scientific failure and recommend one next version
only after audit; launch none automatically.

The controller private seed roster deliberately retains A/B/C `4510000000 + restart`,
so the new version completes the original controller comparison rather than changing
its sampling realization. Its first bank is a real controller refinement, not an
extra calibration or another diagnostic. The already calibrated coefficient is fixed.
