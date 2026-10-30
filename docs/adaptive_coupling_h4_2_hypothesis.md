# H4.2: parent-relative learner-response selection — UNTESTED

This is a next hypothesis formulated after the retained negative H4.1 result.
It is not implemented, launched or a claim that the Walker loop now works.
H4.1's configuration, gates and stopping rule remain unchanged.

H4.1's weight change genuinely changed optimized trajectories, and every
teacher walked. Yet all three fixed-update children regressed against the
unchanged incoming policy. The success/survival-only rule sees ties of zero
and proposes holding the weight. Thus it supplies no signal of this shared
negative transfer. This is an observed limitation of that feedback signal,
not evidence for a particular neural failure cause or for further halving.

**H4.2 hypothesis:** selecting an agreement weight using the child's complete
post-update outcome relative to both the unchanged parent and competing data
arms predicts useful subsequent acquisition better than the H4.1 relative
success/survival trigger. The hypothesis may fail because no weight in the
available teacher family teaches this student at the fixed update budget.
Improving the feedback statistic alone cannot establish walking competence.

Here \(\rho\) denotes the original reference weight 0.1199341983507337, not an
automatically accepted continuation of H4.1's rejected child.
For a fixed incoming learner state \(\ell\) and history \(D\), a prospective
finite test could acquire \(q\in\{0,\rho/2,\rho,2\rho\}\), retain complete teacher
outcomes, and fit exactly matched responses

\[
 \ell_q^+=\mathcal A_M(\ell,D\cup D_q).
\]

The new observable is the parent-relative vector

\[
 A(q)=\left(S_q-S_p,\;V_q-V_p,\;\overline T_q-\overline T_p,\;
            \frac{\overline J_p-\overline J_q}{\overline J_p}\right),
\]

where \(S\) is walking-success count, \(V\) full survival, \(T\) executed controls
and \(J\) complete cost on identical fresh standalone starts. Select by complete
cost among candidates satisfying predeclared teacher constraints and child
nonregression against the parent in all four entries; resolve ties by the
predeclared task ordering and coefficient rule. If none is feasible, retain
the parent and report that no useful teaching alternative was found. Do not
force a coefficient change or call the least harmful child an improvement.

With H4.1's three observed candidates, that parent-nonregression feasible set
is empty. Their existing results therefore do not validate H4.2, identify an
unobserved useful weight, or justify deploying a new child. The positive
prediction requires new prospective evidence and could again be negative.

A selected weight would have to be used in **another actual acquisition** with
the retained updated actor, against a fixed-weight control and independent
search, at equal acquisition/learning budgets. Merely ranking the current
three demonstrations would repeat the limited scope of v3.1. A later adaptive
schedule claim additionally needs a fixed or predetermined schedule control
whose decisions differ from the response rule.

The state observations, torque targets, architecture, MSE learning block and
direct execution controller stay fixed for this proposed coupling test. There
is no critic, synthetic data or new environment. The internal MPPI bank remains
policy-independent; the selected coefficient changes only its existing
policy-distance selection term. Student learnability remains a separate open
track, and changing its features, learning loss or execution controller would
require another hypothesis and control design.

Post-update teaching and policy-projection feedback have prior art; this is
not a novelty claim. See [Behaviour Distillation](https://arxiv.org/pdf/2406.15042)
and [mirror-descent GPS](https://arxiv.org/pdf/1607.04614), with the distinctions
in the [primary-source review](loop_v3_prior_art.md). No benefit is assumed.
The exact candidate roster, update budget, tie rule, thresholds, seeds, total
query/forward/backward/Adam caps and executable input bindings must be frozen
before any H4.2 query. No H4.2 experimental work has been paid.
