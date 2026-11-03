# Bounded repeated full-trajectory PI / torque-policy comparison

Version 1, 2026-11-03. This protocol is frozen by the implementation commit
recorded in the launch receipt, before scientific execution. It extends the
[committed formulation](full_trajectory_pi_similarity.md); that derivation and
its finite-horizon sensitivity limitations remain applicable. No new score is
introduced. This is a torque-policy adaptation, not a CIO reproduction.

## Question and scope

Does fixed nonzero agreement improve six repeated search–fit rounds compared
with alpha=0 under the same full-PI optimizer and continued learner? Six rounds
give five opportunities for a fitted actor to change later acquisition. Three
paired W2 training streams give a bounded cross-stream check. Two initial paths
provide 2,000 rows per round without acquiring a new initialization corpus.
Six rounds do not establish asymptotic convergence, broad prediction,
exploration efficiency or equal-compute superiority to cloning/DAgger.

Both arms execute all six planned rounds. There is no walking-improvement,
discrepancy-reduction, optimization-progress or fit-quality continuation gate.
Intermediate failures of standalone walking remain measured outcomes. A
validity, teacher-competence, numerical, evidence or budget failure stops the
study and preserves every paid prefix. No alternate score, recipe, coefficient,
architecture, start, seed or follow-up is selected within this experiment.

## Exact optimizer reconciliation

The preserved prototype is
`archive/prototype_sources/full_pi_2026-11-03/full_pi.py`, SHA256
`350cdce35c43585eb7edcda11bc2fd9a65c542b0b336783362715703929ffea2`.
Its statements that history explains uncopyability are hypotheses, not adopted
causal conclusions. The partial prototype evidence review is preserved beside
this protocol; it reports retention of initial tapes, not successful discovery.
Another worktree's active, uncommitted preparation is not changed or launched.

| Property | Prototype | This comparison, both arms |
|---|---|---|
| State advancement | Fixed start; no shift or intervening execution | Identical |
| Horizon and sample count | H1000, K256 | Identical |
| Noise | Existing MPPI order-2 low-pass, sigma .3, cutoff23.4375Hz at125Hz | Identical |
| Annealing | `.1**(k/(passes-1))` | Identical formula, restarted within every solve |
| Current nominal | Draw all K noise tapes, then overwrite sample0 noise with zero | Identical draw consumption and overwrite |
| Temperature | Existing ESS-target .05 routine, including unreachable-target behavior | Identical |
| Update | Existing bounded-effective-perturbation `_plan_iteration` | Inherited without rewriting its update |
| Initialization passes | Calling `_plan_iteration` bypasses the two-pass MPC initial bonus | Identical; exactly the declared number of passes |
| Refinements | 300 | 100 acquisition, 25 diagnostic query; declared finite-budget reductions, not settlement claims |
| Candidate weights | Native C | C+alpha D; alpha is the only between-arm objective change |
| Returned candidate | Best previously scored current nominal by C | Best admissible previously scored current nominal by current C+alpha D; alpha0 ranks by C |
| Output safeguard | None beyond best nominal | Common output-only walking and C<=1.2 original-C guard; does not mask samples or alter nominal updates |
| Final mean | Unscored final update is not eligible | Identical; retain its controls explicitly as unscored |
| Best sampled rollout | Not selected directly | Identical; no elite/best-sample acceptance |
| Monotone acceptance | None | Identical; do not recenter on the best incumbent during the solve |
| Private planner RNG | New solve seed | Paired slot-specific streams continue across outer rounds; same fixed draw count in both arms |
| Retained warm start | Per independent prototype case | Each arm's retained complete path initializes its next round; diagnostic queries always use the original full template |
| Initial conditions | Sample-bank states/warmstarts, missing full contexts | Bound H4 independent paths with complete N+1 integration/warmstart contexts; native/batch parity checks |
| Roster | Three starts ×two optimizer seeds, zero/MPC initializations | Two existing competent starts ×three inherited learner streams ×two arms ×six rounds; refinement only |
| Similarity/fitting | No policy or fitting | Same bounded maintained Policy in scoring, fitting and execution; repeated existing MSE Adam fitting |
| Evidence | Selected controls and scalar curves | Every nominal, RNG prestate, C/D/weights/lengths/update; sparse full banks; all actual trajectories/checkpoints/tapes/ledgers |

Do not import the earlier chat proposal's hard feasibility mask, constant
temperature/noise, best-sample choice, mean acceptance or early stalling stop.
These would change the prototype optimizer. The only safeguard added here is
the same output admission rule for both arms. Retain raw best-nominal outcomes
as well as admitted outcomes, so fallback cannot look like successful search.

## Objective, falls, learning and continuation

For complete paths D=.5 sum over controls of squared six-torque residuals on
that path's own pre-action states. Units are normalized actuator controls;
the six motors have gear100. Keep physical controls in float64 when subtracting
float32 policy outputs. The existing learner converts targets to float32; this
small numerical approximation is disclosed rather than changing the learner.

Use fixed alpha .12 versus0. This equals the historical weak matching scale
`.06*sum(error**2)` and is selected by provenance, not observed new outcomes.
It is not the paper's velocity/force coefficient. There is no calibration or
schedule. Each inner objective is frozen while the policy is fixed; recompute
all scores under the updated policy next round.

Incomplete sampled trajectories use D on the executable prefix, including the
first fall-causing action. No post-fall observation contributes to D or fitting.
Native cost charges terminal20 once and absorbing5 per missing control. Maximum
omitted agreement is1.44 per missing control at alpha.12, below absorbing5;
this alone does not prove a fallen candidate cannot score better. Both arms use
the same extension, with no survival mask in the PI weights. Output admission
requires full walking and cost<=1.2 times that slot's original cost, never a
discrepancy decrease. Select the best admitted evaluated nominal; sample0 in
pass0 supplies the carried competent tape. If no admitted tape exists, stop.

Parents are exact W2 s0/s1/s2 update2000 packages, including Adam and private
PCG states. Their policy seeds are20261102001–003. Preserve their existing
17 clipped observations,128/128 tanh architecture, normalization and bounded
tanh output. Use existing CPU `Policy.predict` throughout; no NumPy export or
backend replacement. Training uses batch256, learning rate3e-4, Adam(.9,.999),
epsilon1e-8, zero weight decay, gradient clip.5, zero proximal weight,250
updates per round. No noisy targets, hidden noise or new architecture.

Each arm fits only its current two retained trajectories (2,000 rows), then
continues from that child in the next round. Adam moments, step counters,
private minibatch RNG, policy RNG and newly bound global RNG streams continue.
New corpora begin new fitting blocks with the existing fit_policy convention;
they do not reset Adam. Every stream receives1,500 new updates, ending at3,500
total historical-plus-new updates. Identical row-index draws stay paired between
arms because corpus size and draw counts are fixed. Save actual minibatch tapes.

For positive alpha, MSE has the same fixed-data minimizer as the parameter
block of C+alpha D. Finite clipped-gradient Adam is approximate. Alpha0 still
fits its policy despite its search objective having no parameter term. This
control is repeated uncoupled acquisition/cloning, not an assertion that F0
depends on theta. No measured loss descent implies walking or convergence.

## Frozen roster and measurements

The machine-readable recipe and exact input bindings accompany this file.
Acquire from the two K1 H4 independent paths00/01, preserving their full initial
contexts and native tapes. Verify initial byte restoration and native/batch
replay at rtol0, atol1e-10 before accepting either source. Initial native parity
is an implementation-validity check; host choice may be fixed after that check
but before scientific launch. No outcome-based host or initializer selection.

Acquisition planner seeds are4110000000+100*stream+slot, continued separately
per arm/slot. Seed new global CPU/NumPy/Python streams4100000000+stream;
historical W2 global RNG is unavailable and is not claimed to be reconstructed.
Inherited private training states remain exact. Processing order alternates by
round. No Torch inter-op or intra-op parallelism beyond one; native workers4
threads each, three concurrent stream processes, CPU only.

At round0 and every round1–6, execute16 fixed fresh primary resets
4130000000–015 per actor. Reuse these reset identities over rounds, without
feeding measurements into search/fitting or selecting a checkpoint. At round6
only, execute16 disjoint confirmation resets4130000100–115. Always evaluate the
predefined endpoint, including disappointing endpoints.

Every round retains teacher C, walking/survival, raw and admitted optimizer
progress, source/action changes, pre/post fitting reconstruction, all learning
curve points, standalone C/walking/survival/duration, and joint angles versus
actual elapsed time. Save N+1 states/integrations/warmstarts, actual controls,
costs, terminal flags and missing post-fall regions. Plot all streams/arms/rounds;
do not pick a favorable episode after seeing outcomes.

Student-visited disagreement uses anchors0 and100 in the first primary reset
episode, if an action exists there. Restore that anchor's full context. Query
a **common task-only** H1000 full-PI teacher for25 passes with the original
slot0 full action template from its beginning, not a same-clock suffix. Query
seed4120000000+10000*stream+100*round+anchor, freshly initialized each query.
This teacher never uses the student's agreement term. Compare its selected
first physical action with the actor's recorded action. Report raw MSE, query
C, walking, survival, progress, available/missing anchors and competent-query
counts separately. Its lower budget is an approximate query, not an optimal or
equally settled expert. Query D arrays are marked unmeasured placeholders and
must never be interpreted as zero actor disagreement. Query best selection is
task-only over evaluated nominals; it is not forced to succeed.

At anchor0, actor and query have the same start and full horizon, permitting a
paired complete-cost gap. At anchor100, report action disagreement and query
outcomes only; no900-versus1000 cost comparison. Falling before an anchor is
missing evidence, not zero error. The metric is post-fit diagnosis, not a
pre-update teachability predictor.

## Decisions and stopping

Run all six rounds without an intermediate benefit or futility gate. Primary
finite walking benefit requires final coupled walking at least2/16 above
control in at least two streams, no stream regressing walking or full survival,
on both primary and confirmation panels. Individual competence remains>=15/16
walking on both panels. Report gains relative to round0 as well as between arms.

Secondary progression: compare rounds1–2 with5–6. Report reconstruction
contraction, qualified-query disagreement/gaps and walking curves per stream.
A20% coupled contraction and late coupled value<=90% of control in at least two
streams is a secondary pattern, meaningful only with stable query qualification
and competent teachers. Fitting-, retention- or gap-only changes do not pass the
walking criterion. No significance/general population claim from three streams.

Meaningful search progress means a retained tape changes RMS>=1e-4 and lowers
the same incoming-policy objective by>=1% in that solve. Report its frequency,
including zero. Flat best-ever cost, annealing and ESS diagnostics are not
stationarity. If search repeatedly returns its initialization, identify an
optimizer limitation and lack of effective acquisition manipulation; do not
change the intervention or attribute the failure uniquely to the score.

Stop only for source/hash, replay, continuation, nonfinite executable state,
teacher-guard, evidence-write, resource or deadline failure. Any worker failure
stops its peers; preserve original exits and partial outputs. No automatic retry.

## Complete finite allocation and audit

Worst-case acquisition:72 solves ×100 passes; diagnostic:84 solves ×25 passes.
Both use256 samples ×1000 controls. Total2,380,800,000 optimization controls,
plus6,000 initializer batch controls. At most846,000 native controls cover six
initializer replays,72 teachers and768 actor episodes. New fitting is36 fits,
9,000 Adam updates,2,304,000 backward rows. Diagnostic anchors lost to falls
reduce actual work. Charge D inference in **both** acquisition arms (alpha0
discards it from weights); query search uses no policy inference. Different
prefix lengths can still change neural row counts; report actuals.

Scientific ceilings across three workers:2.4B model controls,900k native,
9,603,600,000 physics substeps,1.89B neural-forward rows,2,304,000 backward rows,
9,000 Adam updates,12h elapsed and450 CPU-hours, zero GPU. Each worker has6GiB
RSS,10GiB output,150 CPU-hours,12h wall and its declared per-kind share. The
32GiB scientific retained-storage ceiling includes inputs/source/coordinator;
no second raw mirror is silently added. Keep every iteration's nominal states,
controls, scores, weights, lengths, RNG and output; full banks at0,50,99 for
acquisition and0,12,24 for queries. Other hypothetical banks are not preserved;
their regeneration is not called raw evidence. Preserve all selected live traces.

Engineering allowance:2h,100k model controls,10k native controls,100 Adam steps;
audit allowance2h,20k model/native controls combined,100M forward rows, zero
training. Keep preparation, implementation checks, failed attempts, copy/hash
bytes, worker/audit/report resources and known/unknown engineering overhead
separate. Early uninstrumented editing/tool CPU/wall is explicitly unknown;
new inference/physics/fitting before engineering instrumentation was zero.
Money, energy and physical disk IO are not inferred from process time.

Run independent audit in a separate process with separately written reductions.
Check all source/input/file bindings, trace outcomes, observation joins,
normalization invariance, checkpoint ownership and Adam/private RNG continuation,
paired tapes, every nominal score and retained selection, all stored bank C/D
reductions and weighting/update identities. A fixed replay subset covers stream0
rounds1/6 both arms slot0 (4,000 controls), plus100 controls of final stream0
actor execution in each arm. Audit physics is explicitly sampled; it does not
certify all unpreserved banks. Publish findings even if incomplete or negative.

The ultimate standard remains held-out walking, equal-total-compute cloning and
DAgger comparisons, a successful repeated-round/joint-versus-static pattern,
separate noise ablation where claimed, and a second body with unchanged recipe.
This finite contrast establishes at most the measured repeated-loop behavior
under this source, optimizer and allocation. It authorizes no follow-up.

## Execution location and numerical audit contracts

The remote upload was rejected by automatic approval review before any transfer
because destination-specific payload export was not explicitly authorized.
Execution is local on the validated native macOS runtime, three workers with
four native threads each. Both arms share this allocation. No remote export or
workaround occurs. This change was declared before the protocol commit.

Audit scalar/native reductions use atol1e-8, rtol1e-12; exact restored and replayed
physical states use atol1e-10, rtol0. Existing float32 Torch inference may vary
slightly with GEMM batch grouping: action parity atol2e-6 and nominal D aggregate
atol5e-4, rtol1e-6 are frozen in the independent reader. These tolerances concern
numerical verification, not walking, selection, or benefit thresholds.
