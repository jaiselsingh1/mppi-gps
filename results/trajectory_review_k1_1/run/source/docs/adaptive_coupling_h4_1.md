# H4.1: outcome-triggered agreement-weight backoff

**Execution update:** H4.1 has now run and closed negative. The actual weight
change influenced trajectories with an updated actor, all six teachers walked,
and all three fixed-update children regressed against the incoming policy.
[Complete results, raw evidence and accounting](walker_experiments_l1_h4_results.md).
No confirmation or second round ran. The following is the preserved
pre-execution formulation; [its previous active version (external evidence)](evidence_availability.md#external-ff792f76dda0)
and the frozen run source retain the original status and hypotheses.

**Historical pre-execution status: formulation specified; decision rule implemented separately; no new
acquisition or learning experiment launched.** V3.1 completed a fixed-demonstration comparison. It did not test
this mechanism, a changing agreement weight, or an updated-policy acquisition.
The independent child's survival benefit and the guided child's loss of it
are material outcomes, not merely a missed cost margin.

Specification revision 2 clarifies that passing the first prospective test
necessarily produces a hold decision next. The first decision-rule snapshot
is preserved in `artifacts/2026-10-30_adaptive_coupling_formulation_001/source`;
no experiment was launched under either wording.
Revision 3 reserves a disjoint second-round seed offset of 2,000,000 after a
prelaunch review found the earlier 1,000,000 offset overlaps L1's seed ranges.
This is a seed-plan correction before physical launch or outcomes, not a
post-result adjustment. The previous specification is preserved in
`artifacts/2026-10-30_adaptive_coupling_formulation_002/source`.

## Separate scientific questions

The [student learnability track](student_learnability_track_v1.md) asks whether
this torque student can learn full-horizon, target-velocity walking. H3.2 is
an input-representation diagnostic in that track. Keeping inherited weights,
normalization and Adam while unclipping velocities changes input magnitudes
and functional initialization; it does not isolate information loss.

H4.1 asks whether one explicit change to the coupling weight improves the
next acquisition–learning outcome. Its task, student feature map, architecture,
targets, optimizer and execution controller remain fixed. It does not make a
feature adjustment part of the coupling intervention. Competent incoming
walking is not assumed, nor is policy superiority to MPPI. The weak-student
case is precisely where a coupling rule can cause harm.

## One untested hypothesis and its rationale

**H4.1:** if, after equal fixed learning updates, positive-weight teacher data
lose task successes or full-episode survival relative to independent teacher
data, reducing the positive agreement weight by half will improve subsequent
standalone task execution relative to holding that weight fixed, without
sacrificing teacher task completion.

The proposed explanation is that a large penalty toward an unsuccessful
student can suppress independently useful controls. Backoff permits more task
search influence when the measured learning response is damaging. The saved
data establish the damage to survival under one comparison, not that excessive
weight caused it or that halving will help. Half is a predeclared perturbation,
not a fitted optimum. Different task-successful trajectories remain admissible.

This is a deliberately limited safety/backoff hypothesis. It has no automatic
increase rule and does not claim to solve co-adaptation or guarantee
convergence. If reducing coupling only recovers the zero-weight control, the
result is harm mitigation, not evidence that positive coupling is useful.

## Variables and exact update

Let \(\ell_k=(\theta_k,z_k)\) contain the policy weights and the entire learning
state: Adam moments and counters, PCG64 stream, normalization, and proximal
block metadata. Let \(D_k\) be retained, provenance-bound teaching history.
The existing teacher executes \(T_\rho(\theta_k)\) using the native independent
MPPI producer and its separate selector. At each control it ranks candidate
action tapes using

\[
 Q_\rho(U;x)=C_H(U;x)+\frac{\eta_u}{2}\|U-U_{\rm ref}\|^2
 +\frac{\rho}{2}\sum_{t\in\text{healthy prefix}}
       \|u_t-\pi_{\theta_k}(h(x_t))\|^2.
\]

The selector verifies synthesized means with native simulation and executes
the first torque. Its producer receives no policy, coupling penalty or selected
guided nominal. The baseline-mean omission remains the same in every new arm;
repairing it would be a separate intervention.

For two same-start, same-planner-seed acquisition alternatives \(a\in\{0,\rho_k\}\),
their actual pairs give matched responses

\[
 \ell_{k+1}^{a}=\mathcal A_{2000}(\ell_k,D_k\cup D_k^a).
\]

Both responses begin from identical \(\ell_k\), use equal row counts, identical
sampled indices and fixed existing Adam settings. Define \(s_k^a\) and
\(v_k^a\) as counts of full task successes and full survivals on the same fresh
16-start standalone cohort. Success still means all 1,000 controls and velocity
MAE at most 0.3 m/s. The update is exactly

\[
 b_k=\mathbf1\{s_k^{\rho_k}<s_k^0\ \lor\ v_k^{\rho_k}<v_k^0\},
 \qquad \rho_{k+1}=\rho_k\,2^{-b_k}.
\]

This tests one rule: halve on loss of success or survival; otherwise hold.
Current action disagreement, fitting MSE, lower teacher cost and smaller
teacher–student action distance never trigger an increase or establish
teachability. Better survival by an independent child can trigger caution even
if that child stands still. That is a reason to reduce influence, not a claim
that standing is the objective or that independent data taught walking.

There is a fixed maximum of two prospective acquisition–learning rounds; no
unbounded halvings or tuning follow failure. Numerical underflow, invalid
ownership or missing paired outcomes stops the worker rather than silently
setting a new weight. A zero-weight control is always explicitly available.

The target being improved is the next standalone policy's **complete walking
task outcome**, with native complete cost, survival and target-velocity success
reported separately. The trigger is a hypothesized useful feedback signal, not
the target itself. Its validity is tested prospectively against a held weight.

## First causal test: hold the incoming updated policy fixed

V3.1 supplies one development-informed trigger:

\[
 (s^{\rho_h},v^{\rho_h})=(0,0),\quad(s^0,v^0)=(0,3),\quad
 \rho_h=0.1199341983507337.
\]

Therefore the rule proposes
\(\rho_{\rm backoff}=0.05996709917536685\). This arithmetic is not experimental
validation. The choice of rule is informed by these results; only fresh
prospective outcomes can test its prediction.

Fork all next-acquisition arms from the **same actual updated guided child**,
policy hash `7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916`,
including its complete saved Adam/PCG64/normalization state. Do not compare an
adaptive child lineage with a different fixed-weight incoming policy.

| Prospective acquisition | Incoming learner | Weight | What it controls |
|---|---|---:|---|
| Outcome-triggered backoff | Same guided +2k child | 0.05996709917536685 | Proposed adjustment |
| Hold-weight control | Same guided +2k child | 0.1199341983507337 | Causal effect of changing the weight |
| Independent control | Same guided +2k child | 0 | Whether positive coupling adds value |

Each acquisition uses the same two new reset seeds and paired private planner
seeds, runs all 1,000 controls or native termination, and preserves full
simulator/planner snapshots and model-query evidence. Full teacher success
and an aggregate cost at most 5% above the matched independent pair are
constraints; no action-agreement or path-return test is applied. If any required
teacher fails those constraints, retain all outcomes and stop the matched
student comparison. Do not replace its data with independent rows or shorten
other arms to obtain a favorable comparison.

Each child receives the same 6,096-row guided history that trained its incoming
parent, plus its own 2,000 actual newly executed teacher pairs: **8,096 rows per
child**. Train all three children from the same incoming learner for exactly
2,000 updates, batch 256, learning rate 0.0003, gradient clip 0.5, mean-parameter
proximity 0.01 and anchor refresh 250, with fixed original clipped inputs and
normalization. Record initial/final full-data MSE diagnostically. All standalone
controllers execute only the torque network, with no prefix or MPPI fallback.

Fresh paired evaluation includes the unchanged incoming updated policy and
all three children. Evaluate 16 primary starts; reserve a disjoint 16-start
confirmation for a passing primary result. Pair full integration resets, not
merely integer seeds. The proposed roots, to be checked against all existing
ledgers before launch, are acquisition `952000000`, planner `952100000`, primary
`952200000` and confirmation `952300000`, with `root+i` for each member. A seed
collision stops preparation of this version rather than silently changing it.
Verify at every guided decision that the actor hash is the updated child's,
the supplied weight is the declared arm's, and the independent producer's
private updates remain policy-free. Retain selected controls and the first
divergent paired decision. A numerical weight change with identical executed
teachers and training inputs is absence of trajectory influence, not evidence
of a useful mechanism. The rule utility alone proves none of these execution
properties.

## Evidence that would count as benefit

The first prospective adjustment helps the task only if, on primary and then
confirmation starts, its child:

1. Has **more walking successes than the hold-weight child**, at least one
   walking success, and no fewer successes than the independent child or parent.
2. Has no fewer full survivals and no lower mean executed controls than either
   new-data control or the unchanged incoming policy.
3. Lowers mean complete cost by at least 5% versus both new-data controls, with
   no cost regression against the unchanged incoming policy.

These are finite development criteria, not guarantees or population tests.
All paired deltas, failed criteria and native termination causes are reported.
Zero walking successes cannot be described as learning competent walking,
regardless of survival or cost. Beating held positive weight but not zero
weight shows mitigation of harmful coupling, not useful positive coupling.

No prerequisite walking gate prevents acquisition from this weak updated
student. Conversely, a failed prospective child comparison stops H4.1; there
is no larger fit budget, alternate backoff factor, changed observation map or
weight retry within the hypothesis.

## Second round and limits on the adaptive claim

If the first prospective adjustment passes and confirms, retain that child,
its exact optimizer/RNG and its 8,096-row history. Use the matched backoff-child
versus independent-child response from that round to apply the same update
again. Before the second acquisition, seal the resulting checkpoint, weight,
history, input hashes and full configuration in a new create-only snapshot.
No new hyperparameter choice is allowed. Append two new teacher episodes per
arm (10,096 rows per child), keep the same update budget, and use disjoint roots
offset by 2,000,000 from the first prospective round: acquisition `954000000`,
planner `954100000`, primary `954200000` and confirmation `954300000`.

Passing round one requires success and survival no worse than the independent
child, so **the next backoff trigger is necessarily zero**. The second weight
is therefore the same lower weight, 0.05996709917536685. This is a mathematical
consequence of the criterion, not merely a possible outcome. Do not relax the
criterion to manufacture another weight change.

Fork the next arms again from that same incoming retained child. Use the same
three-weight roster as round one: the retained lower weight, the **original
fixed-high** 0.1199341983507337, and zero. This replaces a redundant hold-current
arm, rather than adding another arm or spending extra acquisition/fit work.
The same first-round success and nonregression criteria apply against those
controls and the new incoming parent. This tests continued learning under the
chosen lower weight with a genuinely updated actor, and whether the reduction
remains useful relative to the original coefficient. It cannot establish a
second weight-adjustment effect. Both within-round comparisons use the same
incoming learner; these are repeated local interventions, not separate
multi-round fixed/adaptive learner lineages.

Two confirmed useful acquisition–learning rounds would support repeated
feedback progress, not asymptotic convergence. To claim that the **feedback
schedule** outperforms simply choosing a lower fixed weight or a preprogrammed
halving schedule, a later preregistered schedule ablation must produce and test
different weight decisions on the same paid information. The first prospective
comparison alone can establish only the effect of this feedback-selected
adjustment at this learner state. Even a useful adjustment does not establish
that the trigger is generally optimal.

Measure complete teacher and standalone outcomes and the gap on the same
acquisition starts as a training-start diagnostic, separately from the fresh
primary/confirmation cohorts. Any fresh-cohort teacher–student gap requires
teachers evaluated at those same starts and separately budgeted; otherwise label
it cross-cohort and descriptive. Do not claim convergence from that mixed gap.

## Finite budget and pre-launch requirements

This is a formulation, **not yet a sealed physical experiment**. Before launch,
freeze a runnable worker, exact source/model/dependency versions, checkpoint
and history hashes, all seeds, complete gates and stopping rule. Validate the
teacher/query roster against the existing Meter before any acquisition.

For each of at most two prospective rounds, allocate at most **80 million
requested controls, 60 million neural forward rows, 1,536,000 backward rows
and 6,000 Adam steps**. The control cap covers six full teacher episodes, the
four-policy primary and conditional confirmation cohorts, and four-policy
evaluation on the two acquisition starts for a paired gap diagnostic.
Guidance inference over candidate banks, full-data diagnostics, all failed
teacher attempts, resets/restores, tests, audits and transfer overhead must
be accounted separately as appropriate. The cap does not authorize extra
queries, weights or seeds. The combined ceiling is 160 million controls,
120 million forward rows, 3,072,000 backward rows and 12,000 Adam steps.
Repeated identical roles share computation and accounting rather than charging
fictitious work. Hard Meter exhaustion retains partial evidence and stops.

The original fixed-comparison results and all frozen configurations remain
unchanged. This proposal retains the torque-policy departure from CIO's
state-to-state learning and execution optimization. Prior work already includes
adaptive GPS step sizes and learner-response teaching; the
[mechanism review](adaptive_coupling_review_v1.md) explains that changing a
torque-agreement penalty is not equivalent to the MDGPS KL-radius update.
No novelty or convergence theorem is claimed.
