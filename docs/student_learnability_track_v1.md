# Student learnability track L1 — scope and proposed diagnostic

**Execution update:** the separately frozen fresh-training L1.001 variant has
now run and closed negative: both final arms scored 0/16 walking success and
full survival. [Complete results](student_learnability_l1_results.md).
The inherited-state H3.2 intervention below remains unlaunched. This note's
proposal text is retained as historical design context; source snapshots and
the [previous active version](../artifacts/2026-10-30_experiments_execution_001/previous_docs_before_results/docs/student_learnability_track_v1.md)
preserve its pre-execution wording.

This track asks whether the existing torque student can learn competent Walker
locomotion from successful teacher experience. It is separate from testing an
adaptive MPPI–policy coupling mechanism. This note launches no experiment,
changes no controller, and supersedes no frozen run or configuration.

The [completed v3.1 comparison](loop_v3_results.md) tested two recorded
demonstration sets under one inherited learner and fitting procedure. It found
zero walking successes for all three evaluated policies. The independent-data
child survived three of sixteen complete episodes, with near-zero or slightly
backward mean velocity. The guided child had slightly lower complete cost but
lost that survival benefit. These results establish neither competent student
walking nor a test of changing the agreement weight over acquisition rounds.

## What H3.2 would actually test

The existing [unlaunched H3.2 draft](loop_v3_next_hypothesis_v3_2.md) changes the
student's train-and-deploy feature map from clipped to full physical velocities
while continuing the same clip-trained weights, normalization and Adam state.
That is a specific **operational continuation intervention**:

\[
 (h_c,\theta_{\rm inherited},n_{\rm old},z_{\rm Adam})
 \longrightarrow
 (h_f,\theta_{\rm inherited},n_{\rm old},z_{\rm Adam}),
\]

where only the feature map is changed in code. Nevertheless, the treatment
receives larger normalized inputs and can extrapolate beyond the range on which
its inherited network and optimizer history were trained. Equal parameter and
Adam tensors do not create equal functional starting conditions after changing
the input representation.

A positive H3.2 result would show that this particular continuation intervention
improved execution at its fixed budget. A negative result could reflect the
inherited network's input sensitivity, inherited optimizer moments, insufficient
updates, unhelpful extra information, or other obstacles. Neither result alone
isolates information loss as the cause of the historical failure. Retaining old
normalization avoids silently changing two settings, but does not remove the
input-magnitude and extrapolation issue.

The measured clipping frequency establishes that the student discards physical
velocity distinctions. It does not establish that those distinctions are needed
for a successful policy, that labels conflict at otherwise identical inputs, or
that unclipping will help. This is an input-representation diagnostic, not a new
MPPI–GPS formulation. No agreement-weight update follows from it.

## A cleaner future comparison if the question is learning from teacher data

For a separately versioned learnability study, the recommended two-arm design
is **fresh training with clipped versus full velocity features**, using the
same saved physical rows and torque targets. This removes the particular
confound of adapting a clip-trained network and Adam history to larger inputs.
It does not turn a representation experiment into an information-only proof.

Both arms would retain the existing 17→128→128→6 tanh architecture, bounded
torque output, direct torque execution, MuJoCo task and native success/cost
definitions. Reconstruct both feature matrices from the exact same full-state
provenance; preserve the 64 hypothetical re-solve rows as such, rather than
calling every row an executed demonstration. The 6,096 underlying rows and
action targets of v3.1's independent-data arm are an available fixed dataset.
No new acquisition, critic, target relabeling, memory or environment is needed.

Initialize both networks from the same private random parameter stream and
start both Adam optimizers with zero moments and identical counters. Use the
same minibatch-index stream, row order, architecture, optimization settings and
finite update budget. Fit each feature map's normalization from its training
matrix only, with the same predeclared normalization rule and scale floor;
freeze both normalization arrays before fitting. Evaluation data must not
determine them. Parameter equality at initialization and comparable
standardization should be documented separately from equality of the initial
policy functions, which is not guaranteed.

This comparison estimates the effect of the **representation and its declared
training pipeline** under fresh training. Feature-dependent normalization
controls first-moment/scale handling and removes inherited Adam effects, but
changes input geometry and need not match tails or all hidden activations.
Claims that information content alone caused any performance difference would
remain unjustified. Do not combine this fresh-training design with H3.2's
inherited-state intervention under one hypothesis or reuse H3.2's budget as if
the experiments were identical.

Before launch, a new protocol must choose and seal the exact update budget,
learning checkpoints, common initialization, normalization arrays, dataset and
full-state hashes, paired evaluation seeds, primary criterion, confirmation
cohort, query/forward/backward/Adam caps and stopping rule. Neither this note nor
the old H3.2 draft is that executable freeze. Preserve a failed preparation or
negative fit rather than changing representations, normalization or training
length after seeing evaluations.

## Outcomes and relation to the coupling track

Competent walking requires completing all 1,000 controls **and** velocity MAE
at most 0.3 m/s. Upright stationary behavior is a distinct outcome. Report task
success, full survival, executed controls, velocity error, complete cost and
native termination reasons separately, along with fitting curves and all
failed episodes. Reduced MSE, reduced cost or an extra survival without target
tracking is not sufficient to call the student competent.

A future protocol must distinguish an incremental-improvement gate from a
competence criterion. For example, a predeclared 15/16 walking-success threshold
on both a primary and a disjoint confirmation cohort would support competence
on those finite task cohorts; it would not establish population-wide success.
The exact threshold and budget must be set before queries. Do not label a
cost-only continuation win as closing the learnability question.

Evidence from this track may motivate choosing a fixed student configuration
for a later coupling study. That configuration must then remain unchanged
between adaptive-weight and fixed-weight controls. If representation and weight
schedule change together, the resulting performance difference cannot be
attributed to weight adaptation.

Conversely, a bounded coupling experiment can study a weak student without
assuming it already knows better than MPPI. It must explicitly measure what a
changing weight does to subsequent successful teachers and fixed-update
standalone policies. This learnability diagnostic supplies no adaptive-weight
rule, no executed second acquisition round and no evidence of convergence.
