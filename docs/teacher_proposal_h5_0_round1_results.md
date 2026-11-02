# H5.0r1: teacher competence failed before the learning contrast

**Completed and independently audited.** Independent search walked in all 12 acquisition episodes. Selector-only guidance and proposal guidance each walked in 11 of 12, with an early fall in a different parent/reset context. The frozen stopping rule ended the study before every child fit. The proposal mechanism received an acquisition test; its effect on learning after the fixed update remains **unmeasured**.

| Acquisition arm | Walking | Full survival | Executed controls | Mean complete cost |
|---|---:|---:|---:|---:|
| I: independent | 12/12 | 12/12 | 12,000 | 364.580 |
| S: selector-only | 11/12 | 11/12 | 11,201 | 695.370 |
| P: proposal guidance | 11/12 | 11/12 | 11,199 | 692.137 |

Costs include the executed prefix, the native fall terminal once, and five per missing control after a fall. Missing observations stay missing. Each episode has equal weight in these means; the three shared-parent streams remain the replication units. Walking requires all 1,000 controls and mean absolute velocity error at most 0.3 against target 1.5. Longer retention alone cannot pass. Different successful motions were allowed.

The [capsule](../results/teacher_proposal_h5_0_round1/README.md) contains exact configurations, outcomes, curve points, report tables, manifests and audit receipts. Start with the [saved-data report](../results/teacher_proposal_h5_0_round1/report/report.md), [complete accounting](../results/teacher_proposal_h5_0_round1/accounting/accounting.json) and [external checksum catalog](../results/teacher_proposal_h5_0_round1/external_catalog.json). The [protocol](teacher_proposal_h5_0_execution_protocol.md) and [one-round authorization](teacher_proposal_h5_0_round1_launch.md) were published before launch. The useful-learning prerequisite remains unmet; the ordering exception authorized no second round.

## What changed in trajectory optimization

Three width-128 torque policies were each warmed up once for 2,000 updates on the unchanged 4,000-row W1 corpus. Native 17-component clipped observations, W1 normalization, targets, Adam settings and private RNG ownership were unchanged. Each parent stayed fixed during acquisition. The intended next 2,000-update child blocks, with matched 4,000-row corpora and incoming Adam/RNG continuation, did not run.

At each actual state, every arm had a protected actor-free MPPI stream and a separately owned second stream. For P, the second stream optimized

\[
Q(U;x)=J_H(U;x)+\frac{\alpha}{2}D(U;\theta),\qquad
D=\sum_{h=0}^{H-1}m_h\|u_h-\pi_\theta(o(x_h))\|_2^2,
\]

subject to native simulated dynamics. Here the actor includes its frozen normalization; observations precede their candidate actions. The finite supported executable prefix includes the fall-causing action. D is an unweighted horizon sum over six actions. The unchanged coefficient was **alpha = 0.1199341983507337**, softmin temperature 3, and final plan-proximity coefficient 0.01.

| Arm | Protected producer | Second producer | Final selector |
|---|---|---|---|
| I | Task only | Task only | Native task cost and plan proximity |
| S | Task only | Task only | Same rule plus alpha D / 2 |
| P | Task only | Q during each nominal update | Same selector as S |

P therefore changed proposal optimization and its shifted future nominal, beyond reweighting an already completed bank. I and S performed corresponding agreement observer work without feeding it into their task-only producer updates. All arms paid for two streams and four mean verifications per ordinary decision. Realized supported neural rows differ; equal ceilings do not establish equal efficiency.

The final native feasibility guard was

\[
J_H(U)\leq1.05J_H(U_{\rm protected})+10^{-12}.
\]

An infeasible synthesized mean fell back to the protected plan. The audit verified private producer ownership, sampled updates, masks, scores, guards and selection bindings for all 34,400 decisions. The protected producer retained independent search at each visited state. That availability and the finite-horizon guard did **not** guarantee whole-episode competence.

## Complete acquisition outcomes and stopping

| Stream | Arm | Walking / full survival | Controls | Mean complete cost | Change vs matched I | Walking gate | Cost within 5% of I |
|---|---|---:|---:|---:|---:|---|---|
| 0 | I | 4/4 each | 4,000 | 359.729 | — | Pass | Pass |
| 0 | S | 4/4 each | 4,000 | 375.784 | +4.463% | Pass | Pass |
| 0 | P | 4/4 each | 4,000 | 361.132 | +0.390% | Pass | Pass |
| 1 | I | 4/4 each | 4,000 | 374.293 | — | Pass | Pass |
| 1 | S | 3/4 each | 3,201 | 1,333.172 | +256.184% | Fail | Fail |
| 1 | P | 4/4 each | 4,000 | 377.166 | +0.768% | Pass | Pass |
| 2 | I | 4/4 each | 4,000 | 359.718 | — | Pass | Pass |
| 2 | S | 4/4 each | 4,000 | 377.153 | +4.847% | Pass | Pass |
| 2 | P | 3/4 each | 3,199 | 1,338.112 | +271.989% | Fail | Fail |

`teacher_s1_selector_only_00` fell after 201 controls (1.608 seconds): prefix cost 235.693, including the native terminal, plus 3,995 absorbing cost. `teacher_s2_proposal_guided_00` fell after 199 controls (1.592 seconds): prefix 226.047 plus 4,005 absorbing cost. The requested episode lasts eight seconds. Both failures and all other episodes were retained; no post-fall states or replacement rows were created.

The S1 and P2 corpora had 3,201 and 3,199 rows, respectively, against the required 4,000 successful executed rows. As frozen, acquisition finished all 36 slots and stopped before **any** child fit. No successful subset was selected for a smaller contrast. A cost-screen failure alone would not have prevented fitting a complete walking roster; successful data were not rejected through an agreement proxy.

The status is `teacher_competence_negative_no_child_fits`: `execution_complete=true`, `comparison_complete=false`. Only the three parent warm-ups ran. No new child trained or executed; no standalone parent/child primary or confirmation panel and no paired-gap episode ran. Those outcomes are unmeasured, not zero successes. The later 18 gap teachers and the full 54-teacher admission condition were not reached.

## Parent fitting measurements

| Stream | W1 reconstruction MSE, initial → 2k | Held-trajectory MSE, initial → 2k |
|---|---:|---:|
| 0 | 0.304238 → 0.167896 | 0.295831 → 0.198999 |
| 1 | 0.305332 → 0.168076 | 0.297634 → 0.197693 |
| 2 | 0.283772 → 0.169905 | 0.275892 → 0.196344 |

These are full saved-data measurements, averaging rows and six action components. They describe incoming-parent fitting, not new-data child learning or standalone walking. All [6,000 update points](../results/teacher_proposal_h5_0_round1/report/learning_curves.csv), [curves](../results/teacher_proposal_h5_0_round1/report/learning_curves.png) and [vector figure](../results/teacher_proposal_h5_0_round1/report/learning_curves.pdf) are retained without smoothing or checkpoint selection.

The [36 teacher-state agreement measurements](../results/teacher_proposal_h5_0_round1/report/teacher_action_agreement.csv) remain conditional agreement: mean over executed time of the sum of six squared errors against native teacher actions. Their reduction and target dtype differ from fitting MSE; dividing their scalar by six is not an exact substitute for the fitting metric. The [mechanism table](../results/teacher_proposal_h5_0_round1/report/mechanism_summary.csv) preserves all selected-origin and verification/fallback counts. Neither quantity is a teachability gate.

Trajectory-equal observed-prefix agreement means were I **1.218530**, S **1.250294** and P **1.202265**. The time-weighted S/P means were 1.219535 and 1.192818; episode lengths differ because of the falls. These compare each parent on its arm's own visited states, rather than matching states across arms. The smaller P average does not rescue its competence failure or establish teaching value. Every matched I–S and I–P episode had different action, observation and state array digests (24 pairs). This verifies changed recorded experience, not that any difference is useful to a student.

## Audit, cost and availability

The [frozen independent audit](../results/teacher_proposal_h5_0_round1/audit/result/receipt.json) passed. It verified **1,085,983,492 non-training prediction rows in 103,308 calls**, six checkpoint loads, 17,768,832 saved native trajectory costs, 103,236 agreement checks and 34,400 selections. It recreated 5,077,794,816 planner standard-normal draws and 1,536,000 minibatch indices. It checked complete outcomes, snapshots, input/source ownership and accounting. It added **zero native controls, gradients or Adam updates**; training and physics were not rerun.

| Paid phase / quantity | Actual |
|---|---:|
| Teacher model controls | 852,903,936 |
| Live controls | 34,400 |
| Physics substeps | 3,411,753,344 |
| Worker forward rows | 1,087,519,492 |
| Audit forward rows | 1,085,983,492 |
| Parent forward / backward training rows, each | 1,536,000 |
| Parent Adam updates | 6,000 |
| Child fits / updates | 0 / 0 |
| Native worker wall time | 8,684.200 seconds |
| Independent audit wall time | 2,355.448 seconds |
| Closed raw arrays + index + ledger | 206,888,154,706 bytes |
| Compact result download | 1,066,760 bytes; one attempt |

Worker forward rows include 1,085,913,092 guidance rows, 34,400 posthoc teacher rows, 36,000 parent diagnostic rows and 1,536,000 training rows. Model rollouts compute their full horizon even after a modeled fall; live episodes stop on termination. There were 36 resets and no restores; the two explicit reset-forward calls per reset are separate from integration substeps. Rollout-internal forward work is not separately instrumented.

The array payload alone is **204,794,440,824 bytes** (190.73 GiB); all three raw files total 192.68 GiB. The frozen worker's earlier storage snapshot is 1,624 bytes smaller than the closed catalog, attributable by source/metadata ordering to the final `runtime_close` ledger record. Both values are preserved. The worker watchdog observed 684,109,824 bytes RSS; the auditor recorded peak RSS 423,645,184 bytes. Allocated 16-core wall-time bounds are not measured CPU core-hours. Collector/report/transfer work, logical read passes, tests and unknown engineering/billing/energy overhead are separated in the accounting and [closure receipts](validation/teacher_proposal_h5_0_round1_2026-11-01/closure/README.md); nested timers are not added.

Scientific source was frozen at `ff67495dae33ee43952677d0c0165a43f8d469b0`. The saved-only reporter is a separately bound later implementation (`de1ad2c…`); it never changed that worker. The canonical auditor and its frozen hash are preserved. Current code passed a clean export with **1,015 tests, one skip and one expected failure**, 37 existing CLI/report checks and Ruff; final capsule checks and publication receipts are linked in closure.

Full raw evidence, all six checkpoints and source/input trees remain on `robopil5090` at `/home/jaisel/mppi-gps-workspace/runs/2026-11-01_teacher_proposal_h5_0_r1_001`. The external catalog supplies per-file sizes, SHA-256 values and audit scope. Audit work files have a [separate catalog](../results/teacher_proposal_h5_0_round1/accounting/external_audit_outputs_catalog.json). There is **no complete local raw mirror**: the local `runs/2026-11-01_teacher_proposal_h5_0_r1_001` is the original 539-file input/source export. Historical runs, preparation and approved backup remain unchanged.

## Causal verdict and next protocol

| Status | Conclusion |
|---|---|
| Established | Independent acquisition walked throughout; each guided arm lost one episode. The fixed teacher-competence condition failed. |
| Supported within this assay | Putting the unchanged D inside proposal optimization did not meet this protocol's competence requirement. The local cost guard did not ensure whole-episode walking. |
| Unresolved | Whether these new demonstrations improve a child after 2k updates; what caused the individual falls; representation, coverage, consequential errors and trajectory structure as transfer obstacles. |
| Next recommendation | Review [C1.0 cost-effect-weighted fitting](student_consequence_c1_protocol.md): one loss intervention on all 12 successful independent sources, against uniform and stratum-shuffled weights, with fixed 2k updates and fresh walking confirmation. |

The two guided failures occurred in different contexts; this finite roster does not establish a universal S/P ranking. There was no newly admitted child and no learning feedback into later acquisition, so no adaptive loop or convergence result. The retained torque policy directly executes actions; it lacks the paper's state-to-state targets/controller. This is **not a failed reproduction of CIO–GPS**. The earlier [H4.1 feedback-driven negative result](adaptive_coupling_h4_1.md) remains unchanged.

C1.0 tests a learning hypothesis drawn from action consequences, rather than another feature adjustment or similarity ranking. Fixed-tape cost effects are not expert Q, recovery value or proven teaching importance. Its sources and finite census are concrete; recorder/continuation validation, final seed/capacity checks and an executable freeze remain prerequisites. **C1.0 is not implemented or launched. No follow-up or second round runs automatically.**
