# P1.0 — fixed reconstruction in a finite full-trajectory PI loop

**Scientific comparison fixed; execution remains conditional.** The
[configuration](../configs/studies/trajectory_pi_p1.json) has `launch_ready=true`
so its exact bytes can bind preparation and prerequisites. This flag alone
authorizes no execution: source, inputs, all eight prerequisite checks and fresh
capacity must pass their launch seals. This protocol authorizes no automatic
follow-up.

The question is whether repeated optimization of complete control sequences with
fixed policy reconstruction can reduce a finite teacher–student gap while
preserving teacher competence and improving standalone walking. The contrast is
alpha0 versus fixed alpha.12. Recurrence remains stopped; no architecture,
feature, learned-noise, critic, task-cost or execution-controller intervention is
introduced.

## Existing evidence and optimizer reuse

Read the [formulation](full_trajectory_pi_similarity.md), the
[prototype review and input inventory](validation/full_pi_loop_2026-11-03/prototype_review.md)
and the [independent design review](validation/full_pi_loop_2026-11-03/design_review.md).
The read-only SAC prototype already performs whole-sequence PI. Its available
MPC-initialized results retained incoming controls, so walking there establishes
retention, not newly discovered behavior or useful reconstruction coupling.
The older CIO-inspired H48 loops are not full-sequence PI controls.

Reuse the pinned `FullPI` noise subclass and `MPPI._plan_iteration`, with its
exact noise model and weighted update. The configuration binds the inspected
prototype bytes; canonical library placement adds evidence capture and the
declared reconstruction callback without importing another experiment script.
The no-shift, alpha0 path must pass an equivalence test against the original
prototype before execution.

## Cohort and unchanged learner

Use the three recorded W2 update2000 checkpoints `s0_u2000.pt` through
`s2_u2000.pt`, including Adam and private minibatch PCG state. Their exact paths,
sizes and hashes are in the configuration. Copy their checkpoint bytes unchanged;
do not refit or select replacement parents. Each stream begins with identical
incoming policy and optimizer state in both arms. A declared fresh global RNG
envelope is paired between arms; legacy checkpoints do not establish historical
global-RNG continuity, which must not be claimed.

The policy is the existing17→128tanh→128tanh→6tanh torque network. Features are
the17 native pre-action observations: physical columns2:10, then velocities
10:19 clipped to[-10,10]. Inherit each parent's exact FP32 normalization buffers.
Their W1/W2 provenance is training-only FP64 population statistics, std floor.05,
then FP32; do not recalibrate on new teachers or evaluations.

Keep the recorded W2 learner settings: CPU, one Torch thread, batch256,
learning rate3e-4, eta_theta0, proximal reduction `mean`, anchor_refresh0 and
gradient clipping.5. Each round adds100 Adam updates;20 rounds add2000 updates
per arm/stream, continuing legacy2000 to total4000. Retain actual Adam/private
PCG continuation across new corpora, with checkpoint save/reload after every
round. The fitting seed argument is recorded but never reseeds an inherited
private stream. No learning update is paid to create or verify a parent.

Use two existing complete-context H4 independent teacher episodes preserved as
K1 inputs. Both contain1000 controls plus separately saved terminal context.
These same initial sequences and complete contexts initialize all six arms.
The round corpus is the two current retained complete paths:2000 rows. This
differs from W2's historical4000-row corpus, is common to both arms, and is not a
W2 reproduction. Evaluations, query trajectories and frozen-parent baselines
never enter fitting.

## Objective, precision and failure accounting

For a complete physical control sequence U of horizon T=1000:

\[
x_{t+1}=f(x_t,u_t),\qquad
D_\theta(U)=\frac12\sum_{t=0}^{999}
 \|u_t-\pi_\theta(\phi(x_t(U)))\|^2,
\qquad J_\theta(U)=C(U)+\alpha D_\theta(U).
\]

This is a torque-policy analogue of the clean CIO reconstruction term. It is not
the paper's velocity/contact-force policy, optimized memory trajectory, soft
dynamics execution controller or full-trajectory Gauss–Newton optimizer.
Sensory-feedback/hidden-noise interventions and the paper's noise/joint ablation
suite are outside this first contrast. Ordinary MSE fitting and a finite Adam
block do not establish exact coordinate minimization of a joint objective.

The action function has tanh bounds in fitting, PI scoring and standalone
execution. Use the same deterministic NumPy export in planning and physical
policy execution; its batch/grouping invariance is exact and its parity with
Torch is checked at the frozen prerequisite maximum absolute action tolerance
`1e-6`. Native replay uses its separate `rtol=0, atol=1e-10` contract. Promote its FP32 output
to float64 for native execution. Compute D against the **actual float64 U**;
rounding U first can falsely report zero residual. The existing learner retains
its FP32 target conversion, a declared numerical approximation.

Batched native states are post-action. Build pre-action states by prepending the
exact initial state and dropping the final post-action row. Keep the raw sum in
the optimization objective; report per-control RMS and coordinate MSE separately.
Dividing the score by horizon while retaining the old temperature would change
the search and is forbidden.

For incomplete candidates, extend D only over the executable prefix, including
the fall-causing action. Preserve missing observations after termination. Native
terminal cost20 is charged once, followed by absorbing5 per missing control.
Bounded six torques make per-control D at most12; alpha.12 can omit at most1.44
per missing control, below the absorbing charge5. This does not prove a fallen
candidate cannot have a smaller total task score; it prevents an explanation
based solely on avoiding a larger omitted agreement charge. Report all falls.

Alpha.12 exactly corresponds to the historical weak `.06*sum(error²)` torque
coefficient, rather than transferring the paper's alpha10 across incompatible
units. It is fixed for every coupled acquisition and query; there is no adaptive
weight rule or outcome-based retuning.

Preserve the existing optimizer's rejection of invalid hypothetical candidates
using infinite task scores and zero weights. Mask unusable post-fall rows before
feature extraction or policy inference. Invalid live execution, policy/input/
checkpoint values, all-invalid population, failed restoration or failed
prerequisite stop the study. Do not transform missing D into zero: the independent
arm computes current-nominal D for diagnostics, while unmeasured population D is
an absent population dataset with `population_D_measured=false`; it never
enters its weights. Nominal-only diagnostics have their own availability flag.

## Full-sequence solve and alternating order

Use K256/H1000, sigma.3 and the existing order2 low-pass covariance with23.4375 Hz
cutoff at125 Hz. Use100 refinements, scale `.1**(k/99)`, adaptive temperature for
ESS.05 and zero-noise current sample0. This is the prototype's update with a
declared100-step budget instead of300; it does not establish that100 settles.
`_plan_iteration` is called directly, so `initial_iterations=2` adds no passes.
There is no horizon shift or physical execution inside a solve.

The original complete simulator context, policy tensors/normalization and alpha
are fixed throughout each solve. Start with that source's previously retained
U; round1 uses its original H4 controls. Select the lowest **current total J**
among previously scored current nominal sequences, including the incoming U.
Strict-improvement ties retain the earlier nominal, matching the prototype.
Do not select using `_last_costs`, which contains C only, or reuse a score from an
older policy. Evaluate the final weighted mean for diagnostics; it is excluded
from the prototype's eligible previously scored nominal set.

The common output admission gate requires full1000-control survival, speed
MAE<=.3 around1.5 m/s and complete C<=1.2 times that source's frozen initial cost.
There is no action-agreement decrease gate. If the best raw proposal fails,
preserve it and retain the prior incumbent. Retained-bank competence is protected
by this rule and is not independent evidence of successful optimization.

For rounds1–20:

1. Recompute D on the old bank with incoming policy theta(r−1).
2. Optimize both complete paths with that fixed policy and arm alpha, preserving
   all within-solve evidence. Apply the common admission rule. Measure D after
   optimization with the same incoming policy.
3. Fit100 updates on the retained2000 rows, continuing Adam/PCG, and save/reload
   the actual child checkpoint. Recompute D on this bank with theta(r).
4. Execute the fresh16-reset panel and frozen-parent baseline, then the declared
   student-state queries using the current post-fit policy. Save actual joint
   angles and costs. Close the round's evidence before proceeding.

There are no extra initial fits or early stops for one poor fitting update,
walking result or flat inner curve. Report current/best objective progress,
proposal displacement, ESS, maximum weight, teacher admission and stalling.
Shrinking noise or a flat best-ever incumbent is not proof of convergence.

## Student-visited queries and fresh evaluations

At every round0–20, use available controls0 and100 of the first fresh standalone
episode in each stream/arm. Restore the actual complete native context and run
a separate full-H1000 PI query with25 refinements, the same noise/ESS formula,
that arm's alpha and current policy. Initialize every query from original slot0
U from its **beginning**, never a current bank or clock-matched suffix. This is
a lower-budget approximate study operator, not MPC or an equally settled expert.

Report the first query action versus the recorded actor action, plus the query's
own cost, walking, survival and optimization progress. Direct policy pull can
mechanically reduce this gap. No recovery or transfer claim follows without
competent query behavior and standalone improvement. Control0 is a reset-state
measurement, independent of preceding learner actions; control100 probes an
actual learner-induced state. These are conditioned diagnostics, not additional
independent task replications.

Unavailable control100 after a fall stays missing, with no dummy query or zero
gap. Strong student-gap closure requires **both anchors** present and their
queries walking-qualified in both arms throughout all early1–5 and late16–20
windows. Otherwise the stream's student-gap criterion is UNRESOLVED. Report
coverage and qualification in every round. Multiple competent behavior families
can leave nonzero action differences, so unresolved gap closure does not by
itself prove that a walking gain is useless.

Control0 also gives a true full-H1000 query/actor cost pair from the identical
context. Do not cost-pair a control100 query of1000 steps with the saved actor's
900-control suffix. At100 report action discrepancy and the query's own outcomes.
Do not call differences between two fitting-start teachers and16 fresh-reset
actor averages paired performance gaps.

Use16 fresh reset seeds in each round0–20, shared across arms and streams. Their
formula is `2821000000 +1000*round +episode`; no such reset enters fitting. At
round0 evaluate each frozen parent once per stream and share its result across
arms. On every round1–20 panel and the final16-reset confirmation panel,
reevaluate the unchanged parent once per stream for paired parent-child gains.
Confirmation seeds are `2831000000 +episode`. Confirmation pays no learning.

## Paired seeds and source/input freeze

The configuration lists all paired seed arguments explicitly and binds their
canonical roster hash. Acquisition arguments are
`2801000000 +1000000*stream +1000*round +10*slot`.
Query arguments are `2811000000 +1000000*stream +1000*round +10*anchor_index`.
`FullPI` adds its unchanged20,000 offset to obtain the actual PCG seed.
Global-envelope seeds are `2841000000 +stream`; physical prerequisites use
2851000000 and synthetic continuation2861000000. Fitting arguments
`2871000000 +stream` are recorded but ignored by inherited private continuation.

Before launch, seal the actual source export, prototype provenance, all parent
weights/Adam/private-RNG/global envelopes, both initial control sequences,
complete initial/terminal contexts, immutable normalization, exact task and
planner dictionaries, all reset/optimization roles and the prepared-input
manifest. Complete contexts use state/warmstart/integration bytes, dimensions
9/9/6 and integration width91; resolve the `mjSTATE_INTEGRATION` enum from the
pinned simulator rather than mistaking91 for its enum value. The outer launch
binds source/configuration/protocol/prepared manifests without a circular
configuration self-hash. Other worktrees remain read-only.

## Finite work and resource envelope

| Scheduled work | Full-length maximum |
| --- | ---: |
| Acquisition |240 solves ×100 passes |
| Student-state queries including round0 |252 solves ×25 passes |
| PI refinement passes / model controls |30,300 /7,756,800,000 |
| Final-mean single-sequence checks |492,000 model controls |
| Input validation / prerequisite model allowance |2,000 /2,000,000 |
| Planned model maximum including those allowances |7,759,294,000 |
| Fitting |120 blocks;12,000 updates;3,072,000 backward rows |
| Primary/confirmation actor and frozen-parent episodes |3,072;at most3,072,000 native controls |
| Selected teacher/query live replays |240,000 /252,000 native controls |
| Native maximum before input/prerequisite checks |3,564,000 |

Global ceilings are8B model controls,4M native controls and32,016,000,000 physics
substeps, including four substeps per model and live control. The study's12,000
updates plus eight allowed synthetic prerequisite updates give a global12,008
update cap. The backward cap includes3,072,000 study rows plus2,048 conservative
synthetic rows. The neural cap is4B forward rows; the coupled population scoring
alone can request3.8784B before prefix masking. Zero-arm nominal diagnostics,
all actors, fitting and other joins are additional paid neural work.

Allocate24 execution pool-wall hours plus two audit hours,792 complete CPU-hours
(768 workers +24 coordinator), zero GPU use and64 GiB output (56 execution +8
audit). Use four native workers with eight rollout threads each, BLAS1 and
Torch1, plus the coordinator. This permits up to33 runnable CPU roles; record
actual usage and require fresh capacity instead of claiming a strict32-thread
total. RAM caps are32 GiB actual process-tree RSS,6 GiB per worker and4 GiB for
the coordinator. Independent audit working RAM is32 GiB.

The prototype's approximately five-minute300-step source-log timing is a
measurement to bind and qualify, not independently metered CPU cost. The smaller
100/25 budgets and multiple alternating rounds determine this allocation; extra
policy scoring must be measured separately. Equal optimizer/update settings do
not establish equal complete compute. Record source preparation, validation,
transfers, checkpoint I/O, environment/policy constructions and physical-call
failures separately, including unknown partial work. Physical disk traffic,
energy and monetary cost are unknown unless actually measured.

The separately frozen prerequisite run is limited to600 seconds,2M model
controls,2,500 native controls, eight synthetic updates and512 MiB output,
inside the overall allowances. It verifies complete-context restoration,
pre-action indexing, full no-shift refinement, alpha0 prototype equivalence,
score and action-function parity, exact checkpoint continuation, fall accounting,
zero-residual deterministic trajectory reproduction and reservation enforcement.
Any failed prerequisite stops launch; preserve it without relaxing its gate.

## Evidence, audit and decision rules

Keep every nominal's controls/states, RNG prestate, per-candidate C/D availability/
J, weights, first-fall length, task components and proposal. Save full acquisition
population banks only at fixed indices0,49,99:720 banks are approximately36.864 GB
uncompressed state/action data. Query populations are not fully stored; all their
nominals/statistics/RNG are. Save every selected teacher/query and actor's valid
N+1 trace, warmstarts, integration and terminal context, costs, actions, policy
bindings and actual six joint-angle traces. Preserve available contact data
without adding a task component. Count temporary files in the cap. Raw arrays,
ledgers and checkpoints remain outside Git; publish compact curves/manifests.

The independent saved-only audit is capped at two hours,32 GiB RAM,8 GiB output
and140M NumPy head rows. It performs zero physics, MPPI, gradients or updates.
Recompute C/D/J on all nominals, native C on all720 saved acquisition banks and
D on the360 coupled banks; zero-arm unmeasured population D remains explicitly
unmeasured. Check all selected/evaluation outcomes, fitting-row/checkpoint/RNG
joins, source/input bindings and requested/completed/unknown accounting. This
coverage is not exhaustive equivalence of all hypothetical physics banks.

Use early rounds1–5 and late16–20. Teacher reconstruction is the equal-path mean
of each full trajectory's `sqrt(2D/T)`, measured post-fit. A stream's gap screen
requires at least20% early-to-late coupled contraction and late coupled gap at
most90% of independent. The same contraction/advantage is required separately
for both qualified query anchors. A denominator<=1e-12 or missing/unqualified
anchor makes that percentage screen UNRESOLVED.

Retained teacher walking must not regress in any stream; late aggregate teacher
cost must not exceed1.1 times initial aggregate cost. In the late five rounds,
the raw best proposal must walk in at least four rounds for each source path in
at least two streams. Report this alongside the admission-protected incumbents.

On both final primary and confirmation panels, require at least two additional
coupled walking successes out of16 versus **both** the independent child and
same-stream frozen parent, in at least two streams. No stream may regress in
walking or full survival versus either comparator. At least the **same two stream
IDs** must satisfy teacher-gap, both query-gap, raw-teacher competence and walking
gains for full finite-hypothesis confirmation. Report every component and all
three stream effects; resets/frames are not independent training replications.

At least15/16 walking on both panels labels an individual competent policy.
Walking gain without qualified gap closure is a partial finite benefit; gap-only,
retention-only, fitting-only and null results have separate verdicts. A shrinking
gap with worse teacher behavior is not successful transfer. Frozen-parent
reevaluation is not compute-matched plain cloning. Equal-compute cloning/DAgger,
the paper's matched noise/joint ablations and a second body are untested here;
`full_project_success` remains false even if this finite score contrast helps.

Stop for source, input, context, shape, live numerical, continuation, evidence,
resource or budget failure; preserve all paid failures. Otherwise complete the
fixed roster, with no performance-based futility stop, seed replacement, parent
refit, teacher reacquisition, adaptive alpha, extra refinement, or automatic
follow-up. Publish the audited finite result and one next discriminating
experiment; do not launch that next experiment automatically.

A confirmed walking gain with a resolved failure to close the gap is
`PARTIAL_WALKING_GAIN_GAP_NOT_CLOSED`; unavailable or unqualified gap evidence
uses `PARTIAL_WALKING_GAIN_WITH_UNRESOLVED_GAP`. Retention remains a descriptive
survival/control result, with no invented walking success.

## Supported commands and safe execution

Use the existing public dispatcher. Help and inspection load no simulator or
checkpoint. Preparation copies existing evidence; it does not fit parents.

```sh
python scripts/train_mppi_gps.py study trajectory-pi --help
python scripts/train_mppi_gps.py study trajectory-pi prepare --config configs/studies/trajectory_pi_p1.json --out /fresh/base/prepared
python scripts/train_mppi_gps.py study trajectory-pi freeze --out /fresh/base/source
python -B -m src.gps.mppi_gps.trajectory_pi_launch --root /fresh/base
python scripts/train_mppi_gps.py study trajectory-pi inspect --run /fresh/base/worker --verify
python scripts/train_mppi_gps.py inspect --run /fresh/base/worker --json
```

Run the supervisor from the immutable source export, using the pinned CPU
interpreter on robopil5090. Transfer preparation and its manifest unchanged from
the explicitly recorded Mac inputs; do not rewrite historical source locations.
The supervisor seals capacity and launch metadata, runs P0 before the comparison,
and closes with saved-only audit and rendering. It never retries failed science
or starts a new research hypothesis. All output paths are create-only.

Library map: `trajectory_pi_inputs` freezes inputs; `trajectory_pi_preflight`
implements P0; `trajectory_pi_optimizer` promotes the exact prototype update;
`trajectory_pi_student` bridges the existing learner/checkpoint stream;
`trajectory_pi_native` records native jobs; `trajectory_pi_accounting` reserves
work; `trajectory_pi_study` schedules the fixed rounds; `trajectory_pi_launch`
supervises external limits; `audits.trajectory_pi` recomputes saved evidence
independently; `reporting.trajectory_pi` renders saved values. The original
prototype source remains immutable under `archive/prototype_sources/`.
