# Walker v3: finite teacher selection through the learner's response

**Scope correction after v3.1:** this is the fixed-demonstration comparison's
historical specification, not a tested adaptive-coupling formulation. The
agreement weight stayed fixed and the proposed continuation below never ran.
The survival regression is material independently of the cost margin. Current
work separates [student learnability](student_learnability_track_v1.md) from
[H4.1 changing-weight backoff](adaptive_coupling_h4_1.md). The previous document
version is preserved in the [scope-correction receipt](../artifacts/2026-10-30_research_scope_correction_001/preservation.json).

This document specifies the formulation before a new acquisition or feedback
mechanism is implemented. The first experiment closes the historical missing
comparison using recorded experience. It does not rerun or alter v2. Its
configuration and input hashes are frozen separately before fitting.

## Established facts and the question

Independent MPPI discovers successful Walker behavior. The historical high
coupling produced two successful 1,000-control executions at +1.76% complete
cost relative to their matched independent executions. Neither high-coupling
episode trained a child: a 5% action-agreement-improvement gate rejected them.
Those facts establish neither better nor worse teaching. The separate
actor-prefix corrective-data experiment trained different children on different
acquisition data and found no standalone benefit.

**H3.1 (untested at specification):** with the retained Bfree 10k learner and
exactly 2,000 additional Adam updates, the two historical successful
high-coupling teachers teach a better standalone torque policy than the paired
independent teachers, with equal data size and incoming learner state. This is
a local, finite hypothesis. It does not assert that current action proximity
measures teachability, that the student knows more than MPPI, or that post-update
teaching objectives are novel or beneficial in general.

## Variables, objective, targets and execution

The physical state is the complete MuJoCo integration state \(x\), including
solver warm start. The student input \(o=h(x)\in\mathbb R^{17}\) is the existing
Walker observation, including its clipped velocities. The torque policy is
\(u=\pi_\theta(o)\in[-1,1]^6\), with two 128-unit tanh hidden layers. The learner
state \(z\) includes weights, Adam moments and step counters, PCG64 minibatch
state, normalization, and proximal-block state. A teacher episode contains
actual pre-action observations, executed controls, physical snapshots and
complete task outcomes. Model proposals are not silently made training labels.

Let \(T_\rho(\theta;x_0,\xi)\) denote the existing receding-horizon teacher.
At each control the protected, policy-independent MPPI produces a bank from its
own nominal and RNG. For an action tape \(U\) in that bank the existing selector
uses

\[
 Q_\rho(U)=C_H(U;x)+\tfrac{\eta_u}{2}\|U-U_{\rm ref}\|^2
 +\tfrac{\rho}{2}\sum_{t\in\text{healthy prefix}}
       \|u_t-\pi_\theta(h(x_t))\|^2.
\]

The selector also simulates its synthesized mean before executing the first
control. The frozen historical selector has its documented baseline-mean
omission; this experiment uses its actual recorded experience without repairing
the controller or introducing another intervention. \(C_H\) is the existing
native finite-horizon task cost, not the student objective below.

Write \(D_\rho\) for the actual successful teacher pairs. Define

\[
 (\theta_\rho^+,z_\rho^+)
 =\mathcal A_M(\theta,z;D_{\rm common}\cup D_\rho),\qquad M=2000,
\]

where \(\mathcal A_M\) is the existing deterministic Adam learning block:
uniform sampling with replacement, batch 256, learning rate \(3\times10^{-4}\),
gradient norm clip 0.5, action MSE plus mean-parameter proximal penalty
\(\tfrac{0.01}{2P}\|\theta-\bar\theta\|^2\), refreshed every 250 updates.
All alternatives start with identical \((\theta,z)\). Continuations between
reported checkpoints resume the same block; they do not restart it.

The finite outer problem is

\[
 \rho^*\in\arg\min_{\rho\in\{0,\rho_{\rm historical\ high}\}}
    \widehat J_{\rm standalone}(\theta_\rho^+;S),
\]

subject to complete teacher task success and mean teacher cost no greater than
1.05 times paired independent cost. This is a constrained finite search over
two executed trajectory-generating alternatives, not differentiation through
MuJoCo, a continuous bilevel solver, or a new inner MPPI algorithm. The only
learning targets remain recorded teacher torques at their own observations.
Action agreement is reported as a diagnostic and never admits or rejects data.
The present experiment tests the predeclared guided alternative against an
independent-data control. Its advancement rule below is deliberately
asymmetric: an independent-child win is recorded but retains the parent. Thus
the implementation evaluates the two outer objective values; it is not a
generic optimizer that always deploys their argmin.

Standalone execution uses only \(\pi_\theta(h(x))\) at every control. There is
no MPPI fallback, teacher prefix, hidden planner input or policy modification.
The complete 1,000-control cost charges the native terminal cost once and the
existing absorbing cost of 5 per unexecuted control after a fall. Success means
survival through control 1,000 and velocity MAE at most 0.3 m/s. Different
successful trajectories are allowed; no path-distance test enters any gate.

## Exact feedback into subsequent trajectory optimization

The first comparison uses the existing two teacher pairs at matched starts and
planner seeds, \(D_{\rm common}\) of 4,064 original rows plus the same 32
accepted recovery rows in both arms, and 2,000 alternative rows: 6,096 per arm.
The unchanged parent is a third standalone control. A fresh 16-start paired
cohort measures the outer objective. Separate four-start curves are descriptive.

Advance only if the guided child has at least a 5% lower mean complete cost
than **both** the parent and independent child, without regression in success
count, survival count or mean executed controls against either. If this passes,
repeat the same comparison on a disjoint 16-start confirmation cohort. Report
every paired difference; these development gates are not population guarantees.

Upon confirmation, feedback is the explicit tuple
\((\theta,z,\rho)\leftarrow(\theta_{\rho^*}^+,z_{\rho^*}^+,\rho^*)\).
The next guided acquisition therefore scores future candidate trajectories
against the **updated** actor, at the selected coupling. The next independent
control still runs with \(\rho=0\) and its own private planner nominal/RNG.
At the same new paired acquisition starts, acquire two full episodes for each
alternative, check task success and the 5% cost constraint, and fit matched
children from the same incoming updated learner state with the same 2,000-step
budget. Append identical retained history to both datasets. Use new paired
selection and confirmation starts. Freeze that second round's exact inputs,
seeds, budget and configuration before any acquisition; no weight search or
retuning follows first-round outcomes. Two successful rounds plus a narrowing
teacher–student cost gap are required before claiming an operating loop.

If the first guided child does not improve, retain the result and stop H3.1.
There is no extra acquisition, larger training budget or replacement mechanism
inside this experiment. A next hypothesis must receive a distinct version and
pre-launch protocol. Parent retention is explicit; failed fitted children and
their optimizer states remain preserved.

The difference from v2 is the outer teacher-selection and advancement rule:
measured performance **after** the fixed learning block replaces the untested
pre-learning agreement proxy. The bank producer and inner reweighting formula
are unchanged. This does not claim to move the protected MPPI proposal
distribution using a meta-gradient. Independent discovery remains available
even if no student improves. The full formulation remains falsifiable if this
limited family contains no useful teaching alternative.

## Relation to CIO–GPS

This is a torque-policy adaptation, not a reproduction of Mordatch et al. 2015.
Their state-to-state policy produces desired pose/memory velocities and contact
forces and is executed through a one-step dynamics optimization. They optimize
full trajectories with soft physics/contact variables and Hessian-derived
feedback targets. Here hard native forward dynamics, short MPPI horizons,
bounded torque regression, empirical perturbation examples and a direct torque
execution controller replace those ingredients. Failed torque action fitting
does not establish that their algorithm fails.

Post-update teaching and policy-projection feedback have prior art, including
mirror-descent GPS, iterative machine teaching, dataset distillation, and RL
dataset distillation. See `loop_v3_prior_art.md` for primary sources and precise
scope. H3.1 concerns this particular saved experience, learner, update and task.

## Accounting and scope

The first comparison adds no teacher simulation. Imported teacher acquisition
work remains separately attributed to v1/v2. Count all new training forwards,
backward rows, Adam steps, full-data diagnostics, curve evaluations, primary
evaluations and conditional confirmation. Preserve requested/completed/unknown
work, failures, source/config hashes, full snapshots, checkpoints, ledgers and
per-episode native outcomes. Report teacher outcomes, student outcomes and
teacher–student gaps on their actual respective cohorts; different-cohort gaps
are descriptive, not paired causal effects. No efficiency or population claim
is permitted from this finite development test.
