# Independent design review: bounded full-trajectory PI loop

Review of the [committed formulation](../../full_trajectory_pi_similarity.md),
[paper/source evidence](../../cio_optimizer_gap_review.md), the read-only SAC
prototype `scripts/diagnostics/full_pi.py`, current `MPPI`/rollout helpers and
the root agent's proposed finite roster. This review ran no physics, inference,
fitting or experiment. It is a design review, not the frozen execution protocol.

## Assessment

The proposed contrast tests one useful question: does adding fixed whole-path
reconstruction to the existing full-PI trajectory update improve a finite
alternating loop? It retains the same feedforward actor and the same search in
both arms. It does not test whether this distance is novel, prove asymptotic
convergence, reproduce every CIO ingredient, or establish the project's broader
equal-compute cloning/DAgger and generalization requirements.

The historical H48 fresh/refine/always loops do not answer this question. The
new SAC prototype already implements the relevant no-shift update and should be
reused. Its current comments overstate what the older negative evidence proves
about MPPI-MPC and action copyability; retain those as source comments rather
than adopting them as established causes. Saved prototype results and their
source/launch bindings must be reviewed before launch.

## Score and the fixed learner

For a complete sequence, use the requested raw sum:

\[
D_\theta(U)=\tfrac12\sum_{t=0}^{999}
 \|u_t-\pi_\theta(\phi(x_t(U)))\|^2,
\qquad J_\theta(U)=C(U)+\alpha D_\theta(U).
\]

The proposed fixed alpha `.12` equals an existing `.06 * sum(error²)` torque
agreement coefficient. That provenance and action units should be cited in the
protocol. It is not the paper's alpha10 in velocity/force units. Do not adapt
alpha after observing fitting, evaluation or search results.

The root specifies three existing W2 update2000 parents, with their exact Adam
and private minibatch RNG states;17 clipped native observations, width128 tanh
architecture, frozen parent normalization, learning rate3e-4, batch256,
eta_theta0 and the W2 fitting convention. These exact values must be checked
against the bound W2 metadata, not inferred from defaults: `fit_policy` defaults
to eta_theta.01 and summed parameter proximity. Preserve both arms' incoming
policy/Adam/RNG bytes and all checkpoint ownership/hash contracts. New corpora
start new fitting blocks while continuing Adam and private sampling state.

Main `Policy` already uses tanh at its output. Export exactly that bounded
function, including normalization and float32 inference, for PI and standalone
execution. Do not add execution-only clipping to a differently fitted output.
Compute the optimizer's residual against the **actual physical float64 candidate
controls**, promoting predicted float32 actions for subtraction. Casting U to
float32 first could create a false zero residual while native controls differ.
Keep the learner's established float32 target conversion and disclose this small
numerical approximation rather than changing the learner.

The existing fitting MSE averages over rows and six coordinates. Its minimizer
on a fixed unweighted corpus agrees with this clean residual objective, but a
finite Adam block is not an exact minimization. Alpha0 intentionally still fits
the policy although its trajectory objective contains no theta term. Describe
the experiment as a torque-policy analogue, not literal joint-objective block
descent. No new noise-target, memory or execution-controller intervention is
needed in this first contrast.

## Reusing the full-PI update correctly

Reuse `FullPI._sample_noise` and the existing `_plan_iteration` update: K256,
H1000, order2 low-pass sigma.3, geometrical scale1→.1, zero-noise current sample,
and adaptive temperature targeting ESS.05. The root proposes100 refinements per
acquisition rather than the prototype's300. This is a declared budget change,
with the same optimizer and schedule formula, not evidence that100 steps settle.
Do not call `plan_step`, execute a control or shift U during a solve. Calling
`_plan_iteration` directly means there is **no initial two-pass bonus** from
`cfg.initial_iterations`; count the actual pass calls.

Hold theta, alpha and the complete initial context fixed within the solve.
Cache neither a policy score nor a best-score ordering across policy updates.
The prototype's `_last_costs[0]` is native C only. The coupled arm must select
the best previously evaluated current nominal by **current J**, while recording
C and D separately. Keep the prototype's distinction between sampled candidates,
the newly weighted-mean proposal, and previously evaluated nominal sequences.
The final unscored mean is not an evaluated incumbent.

Averaging valid bounded controls preserves bounds, not walking or objective
descent. Preserve rejected/failed proposals and the reason for admission. The
common output admission gate may require walking and cost no more than1.2 times
the same path's frozen initial cost. It must add no independent action-agreement
decrease gate. A stationary retained bank is not successful adaptive acquisition.

## Falling candidates and zero-residual reproduction

Pair each control with its pre-action state; batched MuJoCo returns post-action
states. Preserve the action that causes the first fall, exclude the subsequent
prefix from training and policy scoring, and keep post-fall observations missing.
Native terminal cost is charged once, then five per missing control.

The root's proposed extension scores D on the executable prefix for incomplete
candidates and reports that convention explicitly. Complete trajectories retain
the exact requested D. This does not fabricate missing states. Bounded six
torques give maximum per-control D of12; alpha.12 can omit at most1.44 per missing
control, less than the native absorbing charge5. Thus dropping unavailable D
cannot **by itself** save more agreement penalty than the tail charge. This is
not a general theorem that a fallen trajectory cannot have a lower task score.
Raw falls and final teacher-competence gates remain essential.

Do not silently introduce an infinite-survival mask inside the PI weights.
At H1000 it could leave only the zero-noise incumbent feasible and create a new
search mechanism. Use the same declared fall extension and output admission
rule in both arms. Report actual ESS, unreachable ESS targets, maximum weight,
proposal displacement and nontrivial accepted changes.

The zero-residual prerequisite should verify deterministic execution, not assume
network expressivity. One native fixture can use a state-keyed controller
returning exact saved physical actions. Reject conflicting labels for identical
observation keys, restore the complete integration/warmstart context, and check
the full N+1 trajectory. Add a precision fixture that would fail if actions were
rounded before score calculation. These checks do not show that the real actor
can learn the table or recover from perturbations.

## Student-visited query context

The latest proposal uses anchors0 and100 in the first fresh standalone episode
per stream/arm/round, including round0. Missing anchors after a fall stay missing.
At each available anchor, restore its actual complete context and run a separate
H1000 full-PI solve for25 refinements, initialized with the **same frozen initial
bank template from its beginning**, not an acquisition suffix chosen at equal
clock time. Use the study arm's fixed alpha and its current post-fit policy.
There is no horizon shifting or intervening execution inside this diagnostic.

Compare the selected query's first action with the recorded actor action. Also
report the query's own task cost, walking, survival, score progress and admission
status. This25-step diagnostic is a lower-budget approximation to the study's
100-step acquisition operator, not an equally settled expert. Its direct policy
pull can reduce disagreement mechanically; only competent query outcomes and
standalone improvement make that reduction useful. Separate raw, qualified and
missing query counts. A zero-qualified or changing-coverage trend is unresolved.

Anchor0 permits a genuine teacher-query/actor cost comparison over the same
initial context and H1000. At anchor100 the saved actor suffix has only900
controls. Report action disagreement and query outcomes there, or explicitly
budget a new H1000 actor continuation before calling its cost difference paired.
Do not compare acquisition teachers' two fitting starts with the16 fresh reset
panel and label their aggregate cost difference paired. These are diagnostics
conditioned on two source trajectories, not extra independent task replications.

## Proposed finite census and success interpretation

The root's latest roster is three paired parent streams, two arms,20 rounds,
two complete initial paths and100 fitting updates per round. The new2000-row
corpus differs from W2's historical4000 rows; that is explicit and common to
both arms. Each stream continues from2000 to4000 total legacy-plus-new updates.

| Work | Scheduled full-length count |
| --- | ---: |
| Acquisition solves |240 ×100 PI passes |
| Student-visited solves, including round0 |252 ×25 PI passes |
| PI refinement passes |30,300 |
| Model controls before validation/audit |7,756,800,000 |
| New Adam updates / backward rows |12,000 /3,072,000 |
| Primary actor panels |2,016 episodes, at most2,016,000 live controls |
| Final confirmation |96 episodes, at most96,000 live controls |
| Selected teacher live replay, if every solve is replayed |240,000 live controls |

An8-billion model-control ceiling has243.2 million slack for declared validation
and auditing, not discretionary extra experiments. Physics must include four
substeps for both model **and** live controls;8B model plus2.7M native would mean
32,010,800,000 substeps. Account separately for any final query replay, anchor100
actor continuation and engineering checks. Recompute the final roster before
freezing. The proposed24 execution+2 audit hours,800 CPU-hours, zero GPU and64 GiB
output are allocation candidates requiring measured full-PI throughput and fresh
capacity. Four8-thread native workers use32 active native threads; coordinator
overhead must be measured rather than hidden behind a strict32-thread claim.

Use fixed20 rounds with no single-update futility stop. Source, restore,
continuation, numerical, evidence or budget failure stops and preserves work.
Assess early rounds1–5 against late16–20, with reconstruction measured in a
consistent block phase. The proposed20% coupled contraction and late coupled
gap at most90% of control in at least two streams remain secondary to walking.
Teacher deterioration or declining qualified query coverage cannot support a
convergence claim. Raw selected-teacher competence must be assessed in addition
to the retained bank, whose competence is protected by admission.

The standing walking-gain rule—at least two additional16-panel successes in at
least two streams, no walking/full-survival regression in any stream, repeated
on confirmation—is a sensible finite transfer criterion. At least15/16 walking
on both panels is a separate individual-competence label. Retention-only, gap-only,
or fitting-only improvements remain partial results. Equal refinement/update
budgets are not equal complete compute: the positive arm's candidate policy
evaluations are an added cost, which needs its own rows, time and memory census.

## Evidence and independent audit

Do not attempt to store every full hypothetical bank: that would be roughly
1.2 TB of states/actions for this acquisition roster. Keep every iteration's U,
private RNG prestate, per-candidate C/D/J, weights, first-fall lengths and cost
sufficient statistics. Keep full state/action banks at frozen iterations0,49,99:
720 acquisition banks are about36.9 GB uncompressed. Query bank coverage, all
selected trajectories/full snapshots, all evaluation traces, figures, ledgers,
checkpoints and temporary-file margins must fit the declared storage cap.

An independent audit should recompute all selected/evaluation outcomes and
fitting-row/checkpoint joins; recompute objective components on all saved full
banks; verify native and neural request/completion/unknown ledgers; and check
source/config/input hashes. If a bounded exact replay subset is used, charge it
and declare its coverage. Sparse-bank checks are not exhaustive physics
equivalence. Audit a result's complete census before publishing a causal verdict.
No alpha sweep, architecture change, critic, automatic follow-up or second-body
launch follows from this first finite experiment.
