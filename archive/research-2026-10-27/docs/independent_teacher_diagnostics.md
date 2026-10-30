# Independent teacher and learner continuation diagnostics

Protocol recorded 2026-10-24 after publication checkpoint `384ad7f` and before
executing these diagnostics. The user authorized testing the research points
after preserving the completed experiment. These are bounded development
diagnostics, not a new long training study. The original two75-iteration runs
and reserved final resets5000000–5000199 remain unchanged.

## Questions and proposed loop

The proposed teacher uses only the task model/cost, its own receding-horizon
plan and private random samples. No learner nominal, policy tail, critic or
policy-matching cost enters MPPI. The learner may consume actual teacher
transitions alongside its own experience. A reward-based learner must evaluate
its own actions and continuation; a teacher's successful trajectory is not a
return earned by the learner. We first test these distinctions, without
updating any policy or fitting a new critic.

The completed Anchor/Selective policies are contextual references. They use a
policy-conditioned teacher and supervised action fitting. They are not matched
controls for an independent-teacher, long-return learning method.

## Probe1: independence, synthesis and horizon rankings

Use the frozen raw CPU MPPI task/controller, overriding only `K=32,H=48`:
temperature3, noise.3, initial two refinements, then one refinement and its own
shifted plan per decision. Collect three independent-teacher episodes on new
development resets8900000–8900002, at most300 controls each. Retain full
simulator states, solver warm starts, planner plans, initial flags and private
RNG state before decisions0,75,150,225 when reached. Missing anchors after a
fall remain missing; do not replace episodes or select visually good states.

At every saved anchor:

1. Restore the exact query repeatedly, while loading/evaluating the initial
   and final Selective policy only outside the teacher query. Require exact
   equality of actions and sampled cost vectors. The teacher collection/query
   API receives no policy. Also retain the executed source/config hashes.
2. Score the incoming nominal of the **final refinement pass**, its lowest-cost
   sampled plan and the synthesized weighted mean under the same48-control
   simulator objective. Count mean plans worse than their nominal and best
   sample, with cost differences. Do not substitute the initial zero plan for
   the second pass's nominal. This examines finite synthesis, not long-return
   optimality.
3. Separately force one first action (raw MPPI versus the continuation actor's
   action), then follow either the initial or final Selective actor
   deterministically for a total of1000 controls. Record costs at48 and1000,
   survival, time alive and speed MAE. A fall pays the actual transition cost
   once and5 per remaining control; healthy administrative cutoffs pay no
   failure suffix. Report action-ranking reversals and ties at both horizons.

The full-plan score in step2 and first-action counterfactual in step3 are
different experiments. A short/long ranking reversal demonstrates dependence
on horizon for that fixed continuation, not globally optimal-action discovery
or a causal guarantee of long-run improvement. These few correlated states
are descriptive mechanism checks, not independent performance trials.

## Probe2: frozen critic versus learner-continuation returns

Use the existing Step19 seed1 actor-tail checkpoint at200000 controls,
`runs/2026-10-23_step19_ctx_seed1/checkpoints/step_0200000.pt`. Freeze actor,
both critics and entropy coefficient, retaining its recorded gamma.995 and
absorbing-failure semantics. This checkpoint was trained with a coupled
teacher; it cannot establish whether a future independent mixture succeeds.

Select at most eight available Probe1 anchors using integer-truncated evenly
spaced indices from0 throughN-1 (`numpy.linspace`), before any critic
continuation results, retaining source indices. With eleven collected anchors,
these rows are0,1,2,4,5,7,8,10. At each, query the same
independent MPPI action and this SAC actor's deterministic first action.
For each first action, perform eight paired stochastic-actor continuations
from identical state/solver warm start, for at most2500 controls. Independent
paired RNG seeds are recorded. No training or outcome-based state selection.

Compare frozen `min(Q1,Q2)` against Monte Carlo soft returns. The first fixed
action contributes its real reward with **no entropy bonus**. Later actor
actions contribute `reward-alpha*log_pi`, discounted by gamma. A physical fall
already includes its analytic absorbing reward tail exactly once; there is no
critic bootstrap or entropy after the fall. A healthy cutoff does not become
a termination. Record Monte Carlo variability, bias, predicted/realized action
rankings, and the residual discount gamma^2500 (about3.6e-6). Surviving-cutoff
Monte Carlo returns remain truncated estimates, not certified exact infinite
values or a tail bound; do not bootstrap the same critic as ground truth.

Large value/rank errors would show an actual reliability problem for this
frozen critic on these queries. Low teacher-action value under a weak learner
may instead be correct. Agreement on this small panel would not certify
coverage, useful actor gradients or successful independent-mixture training.

## Subsequent learning comparison

Only after recording these results, specify the actual learner update and
support remedy as needed. The larger [draft](independent_mppi_return_study.md)
separates plain reward learning, the same learner plus independent MPPI data,
and a random-data source control. Match gross executed controls, updates,
reward/gamma, model-query budget accounting and paired training seeds. Measure
standalone-policy learning curves and actor-originated useful exploration;
teacher coverage or one successful visual rollout is not improved learner
exploration. No sample-efficiency claim follows from the diagnostics here.
