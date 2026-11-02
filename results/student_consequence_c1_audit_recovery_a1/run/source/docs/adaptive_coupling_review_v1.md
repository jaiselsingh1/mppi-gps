# Review of an outcome-triggered coupling backoff, H4.1

Status: formulation review only, 2026-10-30. No mechanism has been implemented
or launched for this review. No existing run, frozen configuration, checkpoint
or ledger was changed. The prospective experiment needs its own frozen input,
source, seed and budget bindings. This note complements the main H4.1
specification; it does not turn the closed v3.1 comparison into an adaptive
experiment.

## What the existing result can and cannot motivate

V3.1 fitted two fixed datasets. Its agreement coefficient did not change, and
neither fitted policy subsequently affected acquisition. The guided child lost
all three full survivals that the independent-data child achieved, despite a
2.92% complete-cost advantage. Both scored zero walking successes. This is a
loss of a measured stability benefit, not just a cost-margin near miss.

That observation motivates testing whether reducing policy-agreement pressure
helps the next data acquisition. It does not establish that a smaller
coefficient teaches better, or that the coefficient is the cause of the failed
transfer. Remaining upright while standing is not the desired task. Observation
clipping and inherited input scaling remain a separate student diagnostic, not
part of this coupling intervention.

## One mechanism, not a family of alternatives

Recommend testing **outcome-triggered backoff only**: halve the agreement weight
when a fixed-update guided child loses either walking success or full survival
against its matched independent-data child; otherwise hold it. Do not add an
increase rule, an action-agreement trigger, an MSE trigger, a critic or a new
trajectory optimizer in the same test.

Let \(z_r\) include the complete incoming policy, Adam, minibatch RNG,
normalization and proximal state. Let \(T_\rho(\theta_r)\) denote actual
executed trajectories from the existing independent MPPI bank and coupled
selector at weight \(\rho\). After equal-sized acquisition additions and the
same fixed \(M=2000\) update block,

\[
 z^+_{r,\rho}=\mathcal A_M(z_r,D_r\cup D(T_\rho(\theta_r))),
 \qquad z^+_{r,0}=\mathcal A_M(z_r,D_r\cup D(T_0)).
\]

On the same fresh standalone starts, let \(S_{r,q}\) count complete walking
successes and \(V_{r,q}\) count full survivals. The exact proposed feedback is

\[
 b_r=\mathbb 1\{S_{r,\rho_r}<S_{r,0}\ \lor\
                  V_{r,\rho_r}<V_{r,0}\},\qquad
 \rho_{r+1}=2^{-b_r}\rho_r.
\]

The student observations, torque targets, supervised loss and standalone
execution controller do not change. Complete cost, mean controls, velocity
error and native failure mode remain measured outcomes. They do not enter this
first weight-update trigger. A policy that improves walking success but loses
some full survivals also triggers backoff under this deliberately conservative
OR rule; this is a design choice, not a theorem about the best tradeoff.

For the observed v3.1 signal, \(b=1\), so the predicted next weight is exactly
\(0.1199341983507337/2=0.05996709917536685\). The rule and factor two are
chosen after seeing v3.1. That is legitimate hypothesis development, but v3.1
cannot validate this rule retrospectively. Its numerical prediction must be
tested on new acquisition and standalone outcomes.

## Why it could help, and why it could fail

The hypothesized causal chain is: weaker agreement pressure changes the
selected teacher plans; their executed state/action distribution changes;
fixed-budget fitting supplies more useful feedback to the next student; the
next standalone student retains stability while improving forward locomotion.
The independent-data survival result motivates the stability part of this
prediction. It does not support an assumption that policy proximity measures
teachability.

Halving \(\rho\) reduces the influence of the current student's torque map in
the existing candidate score

\[
 C_H(U)+\frac{\eta_u}{2}\|U-U_{\rm ref}\|^2
 +\frac{\rho}{2}\sum_{t\in\mathrm{healthy\ prefix}}
      \|u_t-\pi_\theta(h(x_t))\|^2.
\]

It does not change the private policy-independent MPPI proposal distribution or
its nominal/RNG update. The updated actor still enters future guided scores.
Thus this is an adaptive outer coefficient for the existing trajectory
selection objective, not a newly derived inner MPPI optimizer. Its protected
zero-weight control preserves independent discovery at every round.

The hypothesis fails if weaker pressure does not materially change executed
teacher plans, damages the teacher's task performance, fails to improve the
next student, or merely makes it stand longer without learning locomotion.
There is no monotonicity guarantee in \(\rho\); a half weight can be worse than
both the higher weight and zero.

## The required prospective causal comparison

For the first intervention, fork every arm from the **same actual v3.1 guided
2k child**, policy hash
`7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916`,
with its complete saved Adam/PCG64/normalization state and the same retained
6,096-row guided history. This checkpoint is an experimental incoming learner;
it is not selected or qualified as a deployment policy. Its lack of walking
competence does not block the acquisition test.

| Prospective arm | Actor used in acquisition | Agreement coefficient | Child training |
|---|---|---:|---|
| Triggered backoff | Same updated guided 2k actor | 0.05996709917536685 | Same incoming learning state, retained history and 2k update budget |
| Fixed high control | Same updated guided 2k actor | 0.1199341983507337 | Same incoming learning state, retained history and 2k update budget |
| Independent control | Protected independent MPPI; actor cannot enter bank production or selection | 0 | Same incoming learning state, retained history and 2k update budget |

Acquire two full episodes per arm at the same new reset states and planner
seeds. Preserve the actual independent nominal and RNG before/after every
decision, full simulator snapshots, candidate banks, executed controls and
complete outcomes. Matching planner seeds pairs randomness, not future states
after actions diverge. No reference-path recovery criterion or per-slot data
substitution is allowed.

With 2,000 actual labels per arm, every supplied dataset has 8,096 rows. Each
child starts from an exact clone of the same incoming learning state, with the
same sampled row indices, fixed original normalization, torque targets and
2,000 Adam steps. Parent plus all three final children receive the same fresh
16 standalone starts. A separate fresh confirmation cohort must be frozen in
advance, with a fixed conditional execution rule. Neither new evaluation
cohort determines the already observed first backoff trigger.

Both nonzero arms pay the same acquisition and fitting schedule. This minimal
rule needs no online shadow hyperparameter search: its feedback comes from
the matched independent-data learning control. If a later implementation adds
shadow acquisitions or fits, the fixed-weight control must receive the same
paid computation, with those results barred from changing its coefficient.
Otherwise an adaptive win can be explained by an extra search/training budget.

Retain complete teacher success and the existing aggregate 5% teacher-cost
guard as acquisition constraints. A teacher that violates a frozen constraint
is a retained negative outcome and cannot be silently replaced with
independent rows. Failure handling, whether to stop before fitting, and any
partial work reservation must be frozen before launch. Do not impose a
standalone-competence prerequisite that prevents the updated actor from
participating in this very test.

## What a positive result would mean

Freeze the improvement and nonregression thresholds before launch, and report
all task-success, full-survival, complete-cost and mean-control differences
regardless of the gate. Complete-cost reduction cannot compensate for lost
full survivals in the same way it nearly hid that loss in v3.1.

A confirmed next-student improvement over fixed high would support this one
backoff intervention at this incoming learner. If backoff does not outperform
the zero-weight control, the result supports reducing coupling harm; it does
not establish useful coupling. Zero task successes in all arms still leaves
competent teacher-to-student walking transfer unresolved. Independent standing
survival must not be reported as walking progress without its velocity and
task-success results.

If the mechanism improves the next standalone policy, run another separately
frozen acquisition–learning comparison using that newly fitted actor. Recompute
the exact trigger from the new matched child outcomes before the next
acquisition. Preserve a fixed-weight comparator and a zero-weight discovery
control at matched budgets. Measure teacher–student performance gaps at paired
task contexts; do not claim contraction from unrelated teacher/student cohorts.

**A first half-versus-high win does not show that feedback adaptation is better
than choosing a smaller fixed weight.** At the first intervention the feedback
has already selected the half weight, so a fixed-half arm is numerically the
same controller. Likewise, if every observed trigger is one, an always-halving
open-loop schedule reproduces the coefficient sequence. A later claim about
the response rule itself requires prospective feedback that produces a
different hold/backoff choice from a predeclared static or open-loop schedule,
and an equal-budget comparison against that schedule. Even two improved
acquisition–learning rounds are evidence of an operating local loop, not a
general convergence theorem.

## Finite cost envelope to verify before any launch

For the existing native \(K=256,H=48\) planner, 1,000-control full episodes and
two initial refinement passes, each teacher pays at most
\((1000+1)KH\) native sampled controls, plus 1,000 separately verified native
means, 1,000 baseline selector means and (for nonzero guidance) 1,000 guided
selector means of length \(H\). This gives 12,396,288 model controls per
independent episode and 12,444,288 per guided episode. The six-episode roster
therefore pays at most **74,569,728 model controls**, plus 6,000 teacher live
controls. The implemented query schedule must be checked against these bounds
before sealing; additional diagnostics must be counted separately.

Three 2k fits pay **1,536,000 forward rows, 1,536,000 backward rows and 6,000
Adam steps**. The four-policy primary cohort and a conditional four-policy
confirmation cohort pay at most another 128,000 live controls and action
forward rows. With the current selector's maximum healthy-prefix inference,
guidance is approximately 49.54 million further forward rows; the exact
full-data diagnostic roster must also be declared. An 80-million-control /
55-million-forward-row envelope can cover this specific roster without extra
physics diagnostics or curve-evaluation cohorts, but is not a launch protocol.
The final source-derived budget must be finite, metered and frozen. Imported
v3.1 work is charged once; source review, testing, transfers and auditing remain
separate overhead.

## Closest primary sources and limits

- **MDGPS, §§4.2 and B.3:** the global step-size rule uses estimated cost after
  the supervised projection to adjust later local optimization. This is close
  precedent for learner-outcome feedback, but its KL-radius adaptation is not
  algebraically equivalent to halving a torque-MSE coefficient. Its guarantees
  do not transfer to this backoff rule or our MPPI bank.
  [Montgomery and Levine, 2016](https://arxiv.org/pdf/1607.04614).
- **Earlier constrained GPS, Algorithm 1 and §4:** policy/trajectory agreement
  multipliers are updated using measured KL violations. That establishes
  changing coupling penalties as prior art, but its residual-based increase
  rule is not our post-update task-outcome backoff. We should not call this
  experiment a reproduction of that dual update.
  [Levine and Abbeel, 2014](https://papers.nips.cc/paper_files/paper/2014/file/c7c9344b5a3c0533e29fa69ce807cf08-Paper.pdf).
- **Behaviour Distillation, Eqs. 3–4 and §4.2:** the outer objective evaluates
  task return after supervised learning on optimized data. Our data remain
  actual executed MPPI experience, and no synthetic-data or ES component is
  needed for this test. Post-update teaching is not a novelty claim.
  [Lupu et al., ICLR 2024](https://arxiv.org/pdf/2406.15042).

The relevant primary-source sections above were read through PDF text for this
review. No new proof, full literature novelty search, physics replay or
experimental validation was performed. This is a falsifiable control law and
a prospective causal design, with competent student walking still a separate
open prerequisite for a successful final loop.
