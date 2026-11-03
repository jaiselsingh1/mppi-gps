# S1.2 V: a useful reference at some student-visited states

The frozen reference-usefulness screen **passed**, with substantial limits. On the 72 noninitial student-state anchors, the SAC reference met the walking criterion in **27/72** continuations and achieved full survival in **46/72**, versus **0/72** for the retained parents. This establishes a useful closed-loop reference intervention on part of this fixed population. It does not establish that reference labels will train a better student.

The [evidence capsule](../results/student_visitation_s1_2_v/README.md), [independent numerical audit](../results/student_visitation_s1_2_v/audit/result/audit.json), [closing admission](../results/student_visitation_s1_2_v/run/admission.json) and [all-outcome report and curves](../results/student_visitation_s1_2_v/report/report.md) preserve the result. Scientific execution used source commit `70075397bfb2966dff97e325c4104915f03c9581` under the [reviewed formulation](student_visitation_s1_2_protocol_v2.md) and [V1.1 execution protocol](student_visitation_s1_2_v_execution_1_1.md).

## What was executed

Three frozen width128 parents supplied 24 existing episodes across eight shared reset seeds. Four anchors per episode produced 96 paired starts: 24 initial anchors and 72 anchors after at least one parent action. At each identical restored full simulator snapshot, either the unchanged parent or the fixed SAC reference controlled the next **1,000 controls**, terminating at an actual fall. Reference control remained closed loop throughout; this was not a single-action substitution.

Before reference inference, all 96 unchanged-parent continuations passed the restoration/replay gate, preserving recorded falls. Actions/restoration obeyed their byte-exact contracts; returned states, warmstarts, integration arrays and costs obeyed `rtol=0, atol=1e-10`. Thirty-two matched reference own-reset controls then qualified the reference before all 96 takeovers ran. Actors, clipped 17-dimensional observations, normalization and task remained fixed. No fitting, teacher acquisition or MPPI optimization ran.

For a 1,000-control horizon, full survival requires all 1,000 actual controls without a fall. Walking additionally requires

\[
\frac1H\sum_{t=1}^{H}|v_{x,t}-1.5|\le0.3.
\]

Complete cost was

\[
C_H=\sum_{t=1}^{n}c_t^{\rm native}+5(H-n)\mathbf1_{\rm fall},
\]

with the native fall terminal charged once. Saved MAE for a fallen trace covers only its returned prefix. Post-fall observations and velocities remain missing.

## Outcomes and the frozen decision

Each table cell contains 12 **noninitial** anchors. All parent walking and full-survival counts were zero.

| Panel | Stream | Reference walking | Reference full survival | Walking gain | Meets ≥2 gain |
| --- | --- | --- | --- | --- | --- |
| Primary | 0 | 1/12 | 8/12 | +1 | No |
| Primary | 1 | 7/12 | 8/12 | +7 | Yes |
| Primary | 2 | 4/12 | 7/12 | +4 | Yes |
| Confirmation | 0 | 2/12 | 7/12 | +2 | Yes |
| Confirmation | 1 | 7/12 | 8/12 | +7 | Yes |
| Confirmation | 2 | 6/12 | 8/12 | +6 | Yes |

The same qualifying streams were **1 and 2**, with no walking or survival regression in any stream/panel. Stream 0's primary gain of one did not qualify. The resulting pass retains this heterogeneity.

Of the 72 reference continuations, **19 achieved full survival but failed the frozen walking criterion**, classified as retention-only under this protocol; **26 fell**. Full-window classification includes recovery transients and permits movement within a trace that misses the threshold. The secondary initial-anchor controls were reference **24/24** walking/survival versus parent **0/24**. Own-reset controls were reference **32/32**, including **16/16 on each panel**. These 56 reference successes did not enter the noninitial gain decision.

## Costs and independent verification

For the 72 noninitial pairs, mean complete cost decreased **4892.86 → 2239.25**. Most of this change came from the absorbing tail, **4601.04 → 1792.01**. Mean native prefix cost increased **291.82 → 447.24**; these prefixes had very different executed lengths, so their native sums are not equal-duration performance comparisons. Walking and survival remain separate outcomes.

The worker executed **113,621 controls**, **454,484 native physics substeps** and **113,621 neural calls/rows**. These include 11,426 parent-replay controls, 32,000 own-reset controls and 70,195 reference-takeover controls. Accounting also records 192 restores, 32 resets, 256 external `mj_forward` calls and 227,466 snapshot reads.

The enclosing worker invocation took **1150.247 seconds**; the audit invocation took **249.860 seconds**. The audit independently replayed **113,621 neural calls**, with four checkpoint loads and zero new physics or fitting. It read **623,662,563 bytes** in its core; the separately bounded source/metadata prelude was at most **87,250,204 bytes**. Actual worker/audit processes closed with clean, reaped exits and no unknown partial work. Nested timings are not additive.

Worker Linux I/O reported **44,001,697,792 `write_bytes`**, distinct from **249,890,054 retained evidence bytes**. Storage-layer I/O, logical reads and retained output are separate costs. The [complete accounting](../results/student_visitation_s1_2_v/accounting/complete_costs.json) includes audit I/O, preparation failures, source transfers, publication and engineering limitations. Worker/audit limits remained 7,200/3,600 seconds; V retained its eight-GiB output envelope.

## Failure history and interpretation

The original V1 preparation under source `f1e9c6e` failed before science: it selected 108 parent rows instead of the declared 96, including twelve historical source-context controls. V1.1 corrected only that join in a new frozen namespace, carrying its costs and preserving the failed exits. Report001 subsequently failed because reading a file changed its access time. [Publication renderer002](../results/student_visitation_s1_2_v/publication/report_version_002.json), source hash `61fa3198…`, corrected that identity check using saved evidence. No scientific execution or audit was repeated for rendering. Existing C1 failures remain recorded.

The 96 anchors are nested within **24 episodes from eight shared reset seeds**, not independent task replications. This assay does not establish generalization across complete student episodes, useful labels at every visited state, architecture-matched teachability, gait alignment, adaptive MPPI coupling or a closed learning loop.

The next step is to **review and freeze L**, comparing equal-sized reference-labelled student-visited and reference-visited datasets: 4,000 training and 2,000 held-trajectory rows, fixed width128 architecture, and 2,000 updates. Preserve exact parent tensors, normalization, Adam/RNG continuation, paired minibatch streams and fresh paired evaluation resets. Bind every selected row, label and checkpoint before fitting; retain all three streams, including stream 0. This contrast changes the constructed data distribution, including visitation frequencies; it does not isolate coverage alone. No learning or follow-up launches automatically.
