# H5.0 — guide proposal optimization, then measure useful learning

**Round one is complete and independently audited; the learning contrast was not reached.** The [result](teacher_proposal_h5_0_round1_results.md) records independent teachers walking 12/12 and selector-only/proposal-guided teachers each walking 11/12. The fixed teacher-competence stop prevented all child fits, standalone panels and paired gaps. Its [capsule](../results/teacher_proposal_h5_0_round1/README.md) preserves the evidence and costs. The [launch record](validation/teacher_proposal_h5_0_round1_2026-11-01/launch/README.md), [one-round decision](teacher_proposal_h5_0_round1_launch.md), [authorized recipe](../configs/studies/teacher_proposal_h5_0_round1.json), [protocol](teacher_proposal_h5_0_execution_protocol.md) and [metric review](teacher_proposal_h5_0_metric_review.md) preserve the original variables, controls, budgets and stopping rules below. The useful-learning prerequisite remains unmet; no adaptive loop or second round is claimed. Original preparation and launch snapshots remain unchanged. The launched source passed [987-test prelaunch validation](validation/teacher_proposal_h5_0_round1_2026-11-01/prelaunch/README.md), with one skip and one expected failure, and 540 preparation checks; later current-code validation passes 1,015 tests with the same skip/expected failure.

## Why this choice, rather than defaulting to S1.2

K1.2 finds closer local state/action returns in the reference than MPPI under one fixed physical-state probe, alongside visibly more repeated reference motion after startup. Both sources walk. The more recurrent reference demonstrations nevertheless produced mostly failing students in S1.1; no student was competent and the cross-stream source-benefit gate failed. Recurrence, smoothing or spectral matching therefore has no established claim to fix transfer here. The diagnostic distance is not proposed as a teaching objective.

[S1.2](student_visitation_s1_2_hypothesis.md) remains a valid experiment on visitation while fixing the reference label function. It does not adapt MPPI. K1.2 contains no student-visited/recovery roster that makes S1.2 uniquely discriminating, and the earlier 980-row corrective MPPI experiment already reduced local fitting error without walking. This does not refute coverage; it prevents treating another coverage experiment as the automatic next choice.

The research question is whether competent search can change toward what the student can actually learn. [H4.1](adaptive_coupling_h4_1.md) already changed an agreement coefficient and acquired walking teachers, but all children regressed. Another scalar sweep over the same bank is weakly supported. **H5.0 asked whether using agreement inside proposal/nominal optimization supplies more useful fixed-update data than applying the same agreement only to a completed independent bank.** The acquisition test failed the teacher condition, leaving that learning benefit unmeasured. This is an implementation-level hypothesis within established guided-policy-search ideas, not a novelty or convergence claim; see the [primary-source review](loop_v3_prior_art.md).

## Variables, objective and exact change

Keep the current native Walker task, 128-unit two-hidden-layer tanh actor, 17-dimensional clipped observations, fixed W2 normalization, six bounded torque targets and direct standalone execution. No critic, new representation, recurrence penalty, transport loss, state-to-state execution controller or new environment is introduced.

Let \(\ell_r\) be the complete learner state: actor tensors, normalization, Adam moments/step, private minibatch RNG, Torch RNG and anchor state. Let \(\theta_r\) denote its actor. At each actual state \(x_t\), keep a **protected actor-free producer** with its own nominal and private RNG. Add a separate, budget-matched second proposal stream. Its variables are a control horizon \(U\), simulated states satisfying native dynamics \(x_{h+1}=f(x_h,u_h)\), and its own nominal/RNG. It never writes the protected producer's nominal, caches, RNG or initial-plan flag.

For the guided second stream only, score every optimization pass by

\[
Q_r(U;x_t)=J_H(U;x_t)+\frac{\alpha}{2}\sum_{h=0}^{H-1}m_h
 \|u_h-\pi_{\theta_r}(o(x_h))\|_2^2.
\]

The mask is the current finite supported healthy prefix, including the fall-causing action and excluding later rows. Construct **pre-action** states from the initial state plus saved post-action rollout states; the existing low-level hook receives post-action arrays, so simply scoring those would be a one-step error. No action-agreement reduction, state-support rule or normalization changes between arms.

Use the existing MPPI perturbation update, now with \(Q_r\) in that second stream:

\[
w_k\propto\exp[-(Q_k-\min_jQ_j)/\tau],\qquad
U\leftarrow\operatorname{clip}(U+\sum_kw_k\epsilon_k).
\]

This changes its next optimization pass and later shifted proposal nominal. The protected stream remains task-only. Keep **\(\alpha=0.1199341983507337\)**, the historical H4.1 fixed coefficient, fixed throughout this proposed contrast; keep softmin temperature **\(\tau=3\)** distinct from it. The final selector retains plan proximity **\(\eta_u=0.01\)** and the same along-candidate agreement. Plan proximity is not newly inserted into the guided branch's inner objective. No coefficient schedule or learner-output penalty is assumed to measure teachability.

The current `CoupledSelector` already sums agreement at every candidate's own successive states, and selection already changes the next executed state. H5.0's substantive change is **where agreement enters the search update**. Its acquisition test used the existing `MPPI.plan_step` prior/coupling hooks for an isolated guided stream; the protected `IndependentMPPI.propose` received no actor. This is not a whole-episode trajectory solver or a reproduction of CIO's desired state-to-state controller/analytic feedback targets. It remains receding-horizon torque control.

## Contemporary controls and competence protection

Use three arms with the same two proposal-stream work envelopes, matched random innovations and physical reset contexts:

| Arm | Protected stream | Second stream | Final choice |
|---|---|---|---|
| I: independent | Task-only | Task-only | Native task/proximity rule |
| S: selector-only | Task-only | Task-only | Current along-candidate agreement |
| P: proposal guidance | Task-only | \(Q_r\) at each nominal update | Same final agreement rule as S |

Two streams in every arm prevent P from receiving an uncharged extra search bank. All arms reserve identical native rollout/verification limits and common actor-forward limits. Measurements on immutable control banks may be used to match/charge observer work, but can never feed back into the protected producer or I's decision. Count actual supported neural rows separately; equal ceilings are not a claim of equal realized compute or efficiency. Do not describe the two-stream common-control design as a byte-identical historical H4.1 reproduction.

Apply the same declared native-cost feasibility rule to final candidates/verified means in all arms, relative to the separately verified protected candidate; proposed bound \(J_H(U)\le1.05J_H(U_{\rm protected})+10^{-12}\). An infeasible synthesized mean falls back to the protected plan, never an unsupported policy action. This is a common teacher constraint, not the varying mechanism. It cannot guarantee whole-episode walking. Record all fallback/failed/unsupported queries and the protected independent outcome.

Teacher admission additionally requires full native walking on every complete acquisition episode and aggregate complete cost within 5% of matched independent acquisition. Preserve every episode and label; do not reject a successful teacher merely because agreement or R fails to improve. Fit every complete successful arm under the fixed budget even if its admission cost screen fails, so a cost-screen failure is not confused with untested teaching value. If any arm lacks the fixed row count because a teacher falls, finish the bounded teacher roster, preserve the negative competence result and stop the learning contrast without replacements or shortened fits. No failed teacher is silently promoted.

## Fixed-update learning contrast

Use three fresh paired streams. Build each common incoming parent **once**, then fork its complete state across I/S/P; do not independently refit the parent for each arm. A proposed construction is the current 128-unit learner trained for 2,000 updates on the unchanged 4,000-row W1 corpus, with fixed W2 normalization. Save that parent checkpoint and continue its exact Adam/RNG stream. New policy/fit/planner/reset seeds and all input hashes must be selected and sealed before any new work.

Acquire four 1,000-control teacher episodes per arm/stream, giving **4,000 unique new rows per child**. Fit each child for exactly **2,000 continuation updates**, batch 256, action MSE, Adam learning rate 0.0003, gradient clip 0.5, current zero proximal coefficient and no anchor refresh. The minibatch-index tape is identical within a contrast; never reset inherited Adam/RNG or refit normalization. Report the parent warm-up budget separately from the fixed child update budget. This matches the current learner rather than claiming that H4.1's different fitting recipe was reproduced.

Primary evaluation is direct standalone torque execution on 16 fresh paired starts, followed by 16 disjoint confirmation starts, for all parents and children. Report reconstruction, held-trajectory error, walking, full survival, executed controls and complete cost separately. Different successful trajectories are allowed; no motion-reference or recurrence match is required. Native fall terminal cost is charged once and the existing absorbing tail is retained.

Use the established walking-gain convention against both the unchanged parent and selector-only control: at least two additional successes in at least two streams, with no walking/full-survival regression in any stream, repeated on confirmation. Require nonregression against independent search's matched child too; if independently acquired data win, retain that fact. Individual competence remains at least 15/16 walking on both panels. Retention alone is partial. No cost or action-fit improvement substitutes for the walking criterion.

Teacher acquisition outcomes and student primary outcomes use different starts and must not be subtracted as paired gaps. Reserve two extra fresh gap contexts per stream/arm, disjoint from acquisition and the 32 primary/confirmation starts; execute the retained acquisition controller and its child from identical full resets there. Report these small-panel paired gaps separately from primary walking. A two-context gap is not a population guarantee.

## Finite work envelope

The proposed first-round roster has 3 common parent fits,9 child fits,36 acquisition episodes and 18 additional teacher gap episodes. With 12 parents/children on 32 evaluation starts, plus 18 child gap episodes, live-control maxima are 54,000 planner-assisted controls plus 402,000 direct-policy controls. Teacher failures reduce realized work and invalidate the fixed-data contrast as described; they do not authorize replacements.

Parent warm-up plus child fitting totals **24,000 Adam updates / 6,144,000 backward rows**; fixed child work is 18,000 updates. Preserve every checkpoint/loss curve and exact optimizer/RNG continuation. Native model/actor work must be enumerated from the two-stream implementation, including initial refinement, each sampled rollout, both mean verifications, guarded selection and all failed attempts. Use the existing K=256, H=48 and initial_iterations=2 schedule as the proposed starting configuration, with common filter/noise settings; do not silently reduce K to fund the guided branch.

The [execution protocol](teacher_proposal_h5_0_execution_protocol.md) derives 1,338,799,104 model controls and 456,000 live controls, including all means, observers, diagnostics and gap contexts. Worker plus full independent prediction verification reserves 4,005,571,104 neural rows. Each arm remains below its two-billion model-control and combined neural-row ceilings. One 16-core worker receives six wall-hours and its independent audit two hours: at most128 reserved core-hours, rather than eight CPU core-hours. RAM is16 GiB for the worker and4 GiB for the auditor. The complete uncompressed payload/index/ledger/external-file census fits512 GiB, conditional on its typed write and pre-call reservations; engineering validation must verify those boundaries before launch review. No compression saving or artificially matched observer work supports an efficiency claim. Preprocessing, failures, transfers and unknown engineering/billing overhead remain separate.

## Exact prospective feedback, and stopping

In round 1 the guide is the frozen incoming parent. A child that fails the frozen walking/nonregression/teacher screens is retained as evidence and **does not replace the parent**. Do not halve or sweep \(\alpha\), replace seeds, choose a favorable checkpoint or add an intervention after a negative/ambiguous result. Close and version one next hypothesis.

Only a confirmed useful child permits preparing a separate second acquisition–learning protocol. Its accepted actor changes the policy field in \(Q_r\), with \(\alpha\) still fixed. The incoming learner in every second-round contrast is the same retained child, including complete Adam/RNG/normalization/anchor state. A fixed-initial-guide control uses that same learner for fitting but the original parent actor for teacher optimization; this distinguishes updated-policy feedback from merely continuing learning. Independent and selector-only controls remain available under equal budgets. Fresh second-round starts must be disjoint from admission panels.

That second round requires its own review/freeze and actual new acquisition. Two useful rounds would support finite loop evidence, not asymptotic convergence. **Current status: round one closed at the audited teacher-competence stop.** Three parent warm-ups and 36 acquisition episodes ran; child learning, standalone panels and paired gaps did not. Fixed-update usefulness and any learner-to-search feedback remain unmeasured. The [round-one authorization](teacher_proposal_h5_0_round1_launch.md) keeps the useful-learning prerequisite unmet and the second round unauthorized. Its [launch snapshot](validation/teacher_proposal_h5_0_round1_2026-11-01/launch/README.md) remains unchanged; the [result](teacher_proposal_h5_0_round1_results.md) supplies closure and availability. The next [C1.0 learner protocol](student_consequence_c1_protocol.md) is for review only, unimplemented and unlaunched.
