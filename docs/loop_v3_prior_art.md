# Walker loop v3: prior art and a restricted learner-response formulation

**Scope clarification:** v3.1 subsequently completed only the recorded-data
comparison. Neither its weight nor its acquisition policy was updated. The
conditional continuation discussed here was a proposal, not an adaptive
experiment. [H4.1](adaptive_coupling_h4_1.md) now specifies an explicit weight
update; [the mechanism review](adaptive_coupling_review_v1.md) distinguishes its
prior art and the controls needed to attribute a benefit to changing the weight.

This note was written on 2026-10-30 before the missing guided-data child comparison was available. It is research context and a proposed interpretation of that experiment, not its result or an experiment launch authorization. Historical v1/v2 runs and criteria are unchanged. The [source-read ledger (external evidence)](evidence_availability.md#external-9a2b8bcc82ad) records the primary sources and read scope. This is a focused search, not an exhaustive novelty review.

## What the closest primary sources establish

| Primary source | Established mechanism | Consequence for this investigation |
|---|---|---|
| Mordatch et al., [Interactive Control of Diverse Complex Characters with Neural Networks](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf), 2015, §§2–3, 7 | Joint trajectory/network optimization uses a quadratic reconstruction penalty and separate proximal trajectory/parameter penalties. Sensory perturbations have feedback-corrected action targets. | Alternating optimization and teacher adaptation are established. The present implementation makes consequential representation, optimizer, and execution changes; torque fitting failures do not reproduce and falsify this paper. |
| Levine et al., [End-to-End Training of Deep Visuomotor Policies](https://jmlr.org/papers/volume17/15-522/15-522.pdf), 2016, §§3–4 | GPS alternates local trajectory-controller improvement and global supervised fitting, using BADMM agreement terms. Local controllers initially train independently until basic task competence. | Independent discovery followed by co-adaptation is established. Protected continuing MPPI proposals are a design restriction, not proof of convergence or a new general learning principle. |
| Montgomery and Levine, [Guided Policy Search as Approximate Mirror Descent](https://arxiv.org/pdf/1607.04614), 2016, §§3–4.2 | Supervised fitting is an approximate projection. In the nonlinear case the projection need not preserve trajectory distributions exactly. One step-size adaptation rule uses global-policy cost after fitting to account for failed projection as well as model error. | Feedback from what the student executes to subsequent trajectory optimization already exists. Its simplified guarantees do not transfer to our finite MPPI candidate reweighting and torque-Adam procedure. |
| Liu et al., [Iterative Machine Teaching](https://proceedings.mlr.press/v70/liu17b/liu17b.pdf), 2017, §§3–4 | The teacher models the learner's parameters, representation, optimizer and learning rate. For one gradient step, example choice trades gradient magnitude against alignment with a target parameter displacement. | Small current prediction error alone does not characterize useful teaching even in simpler linear settings. The paper's target-model assumptions and teaching guarantees do not establish nonlinear Walker control improvement. |
| Wang et al., [Dataset Distillation](https://arxiv.org/pdf/1811.10959), 2018, §3 | Data and learning rates are optimized through one or more training updates to reduce subsequent real-data loss; fixed initialization is an explicit case. | Evaluating data by performance after a fixed learning update is established. These synthetic supervised examples need not be physically realizable trajectories. |
| Lupu et al., [Behaviour Distillation](https://arxiv.org/pdf/2406.15042), ICLR 2024, §4.1–4.2, Eqs. 3–4 | The bilevel objective maximizes policy return after supervised training on a learned state-action dataset. HaDES uses evolution strategies outside and supervised fitting inside, with fixed or randomized policy initialization. | This is especially close prior art for the proposed learning-response objective, including blackbox outer optimization. Here data remain actual MPPI experience; adding ES, synthetic labels or a new architecture is unnecessary for the missing comparison. |
| Wilhelm and Ventura, [Distilling Reinforcement Learning into Single-Batch Datasets](https://arxiv.org/pdf/2508.09283), 2025 preprint, §3.1, Algorithm 1, §6 | A single supervised gradient update defines a policy; PPO meta-learning optimizes the synthetic dataset through that update. It includes continuous MuJoCo tasks. | Post-learning task performance is not a guarantee of practical success. Table 4 reports Walker2D mean reward 331.7 ± 117 versus PPO 3537 ± 432 and notes possible nonconvergence. These are another task and metric; they do not predict our result. |

DAgger addresses a related but different concern: predictions change the states subsequently observed, so collecting expert labels under a learner's induced distribution can address covariate shift under its assumptions. It does not optimize a teacher trajectory by a child's fixed-update task cost. A short actor-prefix corrective acquisition is also not automatically a full DAgger experiment. [Ross, Gordon and Bagnell, 2011](https://proceedings.mlr.press/v15/ross11a.html).

## The CIO representation and deployment departure

The 2015 paper defines sensory input as \(s_t=[q_t,r_t,\dot q_{t-1},f_{t-1}]\) and action as \(a_t=[\dot q_t,\dot r_t,f_t]\). Its execution solves a one-step next-state optimization to match desired velocities/contact forces while satisfying its dynamics formulation; the paper allows soft dynamics and residual root forces. The Walker student here maps a fixed 17-dimensional observation to six bounded torques, which MuJoCo executes directly every 0.008 s. It has neither that state-to-state target nor that downstream optimizer, recurrent memory, or Hessian-derived feedback targets. A correct comparison therefore labels our negative outcome as a failure of this torque-policy adaptation at its measured budget, not a failed CIO–GPS reproduction. [Mordatch et al., §§2, 3.2, 7](https://papers.nips.cc/paper_files/paper/2015/file/2612aa892d962d6f8056b195ca6e550d-Paper.pdf).

## A finite formulation for the missing comparison

These equations describe our proposed restricted outer objective, not a new general algorithm or a mathematical convergence result. The exact finite row counts, update budget, seeds, margins and stopping rule belong in the experiment's frozen protocol.

Let \(x_t\) denote the full simulator state, \(w_t\) the simulator warm start/integration context, and \(p_t\) the independent MPPI nominal/RNG state. The student observes \(o_t=h(x_t)\), with the current 17-input normalization, and executes \(u_t=\pi_\theta(o_t)\in[-1,1]^6\). Its learning state \(z\) includes weights, Adam moments/step, minibatch RNG, normalization, proximal anchor/block position and stored fit settings. Equality of incoming weights alone is insufficient matching.

For each acquisition alternative \(a\in\{0,\alpha_{\rm high}\}\), define actual teacher trajectories \(\mathcal T_a\) and their executed pre-action pairs

\[
D_a=\{(h(x_t),u_t):u_t\text{ was executed by teacher }a\}.
\]

Use equal-sized additions and shared base data \(D_{\rm base}\). Preserve complete trajectories and all failures. Qualify teacher task quality by full task outcomes and the frozen cost guard, never by requiring reduced action disagreement or return to a reference path. Model-query candidates are paid search work; they are not executed teacher labels.

Let \(U_B(z,D;\omega)\) be exactly \(B\) existing supervised Adam updates with fixed minibatch stream \(\omega\), fixed proximal settings and unchanged torque targets. The inner loss remains

\[
L_D(\theta)=\frac{1}{6|D|}\sum_{(o,u)\in D}\|\pi_\theta(o)-u\|_2^2
+\frac{\eta_\theta}{2P}\|\theta-\bar\theta\|_2^2,
\]

with \(P\) parameter entries and the anchor refresh schedule frozen separately. The implementation evaluates its minibatch version, clips the total gradient norm at 0.5, and applies Adam; the update is not ideal loss minimization. The learner response is \(z_a^+=U_B(z,D_{\rm base}\cup D_a;\omega)\). A matched two-arm comparison estimates

\[
\Delta_B=\frac1M\sum_{j=1}^{M}
\big[C(\pi_{z^+_{\alpha_{\rm high}}};\xi_j)
-C(\pi_{z^+_0};\xi_j)\big],
\]

where \(\xi_j\) are fresh paired reset contexts and \(C\) is complete executed task cost, including the fall terminal and missing-controls penalty. Survival, success, horizon, velocity error and the signed teacher–student cost gap are reported separately. Parent evaluation distinguishes a relative child win from an actual standalone improvement. Finite budgets make \(\Delta_B\) specific to this incoming learner, data and update; it does not measure intrinsic data teachability.

A restricted outer optimization is

\[
a^*\in\arg\min_{a\in A_{\rm task}}\widehat C_{\rm select}(\pi_{U_B(z,D_{\rm base}\cup D_a;\omega)}),
\]

subject to frozen nonregression and improvement rules against both parent and independent-data child. Selection starts, confirmation starts and any final held starts must have distinct roles. A fresh second acquisition–learning round is required before a beneficial-loop claim.

## Exact feedback and its limits

The current inner selector remains

\[
S_a(U)=C_H(U)+\frac{\eta_u}{2}\|U-U_{\rm incoming}\|^2
+\frac{a}{2}\sum_{t\in\text{healthy prefix}}\|u_t-\pi_\theta(h(x_t))\|^2.
\]

It reweights the copied final MPPI bank and verifies synthesized candidates. Its private MPPI producer still advances using only task cost. The outer learner-response experiment changes **which acquisition setting and policy version receive subsequent use**: only if the guided child's actual standalone performance passes the frozen gate does the next acquisition set the actor to that child and retain the selected guidance coefficient. Otherwise it stops that hypothesis, preserving the parent and independent acquisition. \(a=0\) stays available in every round; a policy can earn no authority by merely disagreeing less.

This is a correction to v2's teacher-data admission and a finite outer search over existing teachers. It does **not** replace the inner policy-distance objective with a differentiable post-update value, optimize arbitrary trajectories in a bilevel solver, or establish a novel method. The old workflow already intended child evaluation after admission; the missing experiment tests the data that its proxy prematurely excluded. A negative result is informative and does not justify relabeling the proxy as valid.

## One explicitly untested next hypothesis, contingent on a negative result

**H-next-observation-v1:** clipped velocity features discard control-relevant distinctions that impede this torque student's fixed-budget execution. The existing observation audit measured clipping in 2,359/4,000 pre-action teacher states while MPPI used full velocities. That observation motivates the hypothesis; it does not identify the cause of falling.

A separate future test can compare the current clipped velocity features with full recorded velocities in the same 17-dimensional representation, retaining the physical task, independent planner, torque targets, architecture and budget. Both training and deployment must apply the declared feature map, and normalization/initial learner matching must be specified before queries. It needs no critic, new environment or additional network. This would test an observation intervention, not establish the MPPI–policy loop; it must be versioned and frozen separately. Do not mix it into the guided-data comparison or infer that unclipping will help.
