# Executed Walker studies: L1 and H4.1

**Both studies completed as valid negative experiments.** L1 did not establish
competent walking from either fresh input pipeline. H4.1 actually applied the
feedback-selected weight change in new optimization with an updated actor,
trained all three matched children, and found no standalone benefit. The
teacher–student gaps widened. No confirmation, second round, retuning or
additional mechanism was launched after the frozen stopping rules triggered.

The [full local validation](../artifacts/2026-10-30_experiments_execution_001/full_validation_local_platform.json)
passed **187 tests**, with two skips and one expected failure. Both independent
saved-evidence audits passed. Existing runs, frozen configurations, checkpoints
and ledgers remain preserved.

## Student learnability: L1.001

| Fresh paired primary evaluation | Walking success | Full survival | Mean controls / 1,000 | Complete cost |
|---|---:|---:|---:|---:|
| Historical Bfree reference | 0/16 | 0/16 | 220.4375 | 4,334.903417 |
| Fresh clipped-input student | 0/16 | 0/16 | 173.5625 | 4,547.869041 |
| Fresh full-velocity student | 0/16 | 0/16 | 155.0000 | 4,761.772611 |

The fresh arms shared the same 6,096 physical rows, targets, parameter bytes,
zero initial Adam, PCG64 minibatches and 10,000-update budget. Each used its own
training-only normalization under the same rule. This avoids adapting an
inherited clipped-input network and Adam to larger inputs, while still testing
a representation/training pipeline rather than information content alone.

Final own-data MSE was almost identical: 0.12706687 clipped and 0.12707419 full.
The clipped 5k curve checkpoint survived 3/4 episodes, but all three stood or
drifted backward; none walked. Its later fixed 10k endpoint lost survival.
No checkpoint was selected from those curves. Full velocity inputs did not
resolve this student's execution problem at this budget, and the general
learnability question remains open.

The [detailed L1 report](student_learnability_l1_results.md) includes complete
curves, physical provenance, deployment-feature verification, outcomes,
normalization limits, all 178 hash-verified run files and reproduction details.
This is a student diagnostic, not an adaptive-coupling result. L1 ran on the
canonical Linux host; its input intervention was not introduced into H4.1.

## Coupling mechanism: H4.1

The [specified rule](adaptive_coupling_h4_1.md) halves the agreement weight if
the guided child loses task success or full survival against its matched
independent-data child after fixed learning updates; otherwise it holds. The
old 0-versus-3 survival result triggered
**0.1199341983507337 → 0.05996709917536685**. That was hypothesis development;
the new prospective acquisition and learning below tested the adjustment.

All arms used the same actual updated guided child,
`7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916`,
with identical Adam, PCG64, normalization and 6,096-row history. The actor was
fixed during acquisition. The independent producer retained its task-only
nominal/RNG update. Nonzero weights changed its existing candidate-selection
penalty; they did not modify its proposal generator or repair the documented
baseline-mean omission. The [prelaunch protocol](adaptive_coupling_protocol_h4_1_001.md)
froze all controls, budgets, gates and stopping rules.

| Actual complete teacher pair | Agreement weight | Walking / survival | Mean complete cost | Velocity MAE |
|---|---:|---:|---:|---:|
| Independent MPPI | 0 | 2/2 / 2/2 | 382.491768 | 0.148406 |
| Fixed high | 0.1199341983507337 | 2/2 / 2/2 | 371.574504 | 0.143955 |
| Feedback-selected backoff | 0.05996709917536685 | 2/2 / 2/2 | 379.479145 | 0.147370 |

**All six teachers walked and both positive weights passed the cost guard.**
The backoff pair cost 0.79% less than independent but 2.13% more than fixed high.
There was no action-agreement admission filter and no per-slot substitution.
Different successful trajectories were allowed.

The weight manipulation had a verified causal effect. Backoff versus fixed
high changed 999/1,000 actions in both pairs. The first difference was control
1, with action-vector distances 0.06043357 and 0.00305475. At that decision,
the complete physical snapshot, private nominal/RNG, proposal/state/cost bank
and policy agreement values matched byte-for-byte; only the coefficient
differed. The resulting changes were not caused by different random starts,
proposal tapes or an old actor accidentally entering the experiment.

Every successful pair then trained a child on 8,096 rows: the same 6,096-row
history plus 2,000 own executed pairs. All three fits started from the exact
same learner/optimizer state and used identical sampled indices and exactly
2,000 Adam updates, batch 256, learning rate 0.0003, mean-parameter proximity
0.01 and anchor refresh 250. Checkpoints 500/1,000/2,000 resumed the same block.
Targets and direct torque execution were unchanged. Standalone policies had
no teacher prefix, planner fallback or access to private planner state.

| Fresh paired standalone primary | Walking success | Full survival | Mean controls / 1,000 | Complete cost | Velocity MAE |
|---|---:|---:|---:|---:|---:|
| Unchanged incoming updated policy | 0/16 | 0/16 | 235.3125 | 4,289.545694 | 0.830369 |
| Independent-data child | 0/16 | 0/16 | 125.8125 | 4,850.449783 | 1.839527 |
| Fixed-high-data child | 0/16 | 0/16 | 169.2500 | 4,957.605909 | 2.204889 |
| Backoff-data child | 0/16 | 0/16 | 155.7500 | 4,843.232106 | 1.869204 |

Backoff had 2.307% lower cost than fixed high, but averaged **13.5 fewer
controls**. Its cost was 12.9078% worse than the unchanged parent, and it had
higher cost and shorter execution on **every one of the 16 paired starts**.
All three fits regressed against the parent. This is substantive negative
transfer, not merely a narrow miss of a cost threshold. No child learned the
walking task.

![Changed-weight teachers, child outcomes and learning curves](../artifacts/2026-10-30_adaptive_coupling_h4_1_report_002/learning_and_outcomes.png)

| Own supplied-data MSE | Before fitting | 500 updates | 1,000 updates | 2,000 updates |
|---|---:|---:|---:|---:|
| Independent | 0.157492 | 0.147417 | 0.142918 | 0.137743 |
| Fixed high | 0.153180 | 0.143760 | 0.140033 | 0.135419 |
| Backoff | 0.154937 | 0.143589 | 0.139293 | 0.134442 |

Reconstruction improved by 12.54%, 11.59% and 13.23% respectively, while
standalone execution worsened. Neither lower MSE nor a successful teacher was
a guarantee of useful fixed-budget teaching.

## Paired teacher–student gaps

These gaps use the **same two actual acquisition resets**, verified by full
initial simulator/integration/warmstart/observation bytes. They are explicitly
training-start diagnostics, not held generalization estimates. Each column
subtracts that arm's teacher cost from the corresponding standalone cost.

| Acquisition arm | Incoming-parent gap | Child gap after learning | Gap widening |
|---|---:|---:|---:|
| Independent | 3,923.723110 | 4,530.563035 | +606.839925 |
| Fixed high | 3,934.640374 | 4,580.487122 | +645.846748 |
| Backoff | 3,926.735733 | 4,300.330057 | +373.594324 |

The gap widened in every arm, including at the very starts whose new teacher
experience trained the children. No contraction or beneficial loop is
established. The H4.1 count-only rule would propose holding the lower weight
because all children tie at zero success and survival. That **unexecuted
proposal** has no signal of their shared regression against the parent. The
separate frozen advancement criterion correctly rejected continuation.

## Execution platform, tests and evidence

Automatic approval review rejected sending the H4 worker payload to the
canonical remote host. The rejection and original remote protocol are retained.
H4 was instead frozen and executed locally on macOS arm64, Python 3.11.15,
MuJoCo 3.6.0, NumPy 2.4.3, SciPy 1.17.1, Torch 2.10.0 and h5py 3.16.0. Its
three arms share that platform. The model, source algorithm, task, inputs,
weights, learning settings and criteria were unchanged; the platform revision
preceded launch. Historical feedback and the input checkpoint came from
Linux/Python 3.11.16/Torch CPU. No cross-host bitwise reproduction or efficiency
comparison is claimed. No private-code upload workaround was used.

The [independent H4 audit](../artifacts/2026-10-30_experiments_execution_audit_001/h4_receipt_001/report_v2.md)
passed **772,685 checks**, binding 128 files and 1,457 array fingerprints.
It inspected all 12,000 private before/after planner snapshots, 24 selected
full banks, actual teacher and policy episodes, exact datasets, nine child
checkpoint/Adam/RNG packages, sampled indices, curves, paired resets, native
costs and all ledger work. The full original local HDF and ledger hashes match
the [explicit local locator](../runs/2026-10-30_adaptive_coupling_h4_1_r1_001/exports/full_local_evidence_locator.json).
The [supplemental proofs](../artifacts/2026-10-30_experiments_execution_audit_001/h4_receipt_001/supplemental_native_detail.json)
verify same-bank weight causality and paired gap widening. Audit time was
41.845734 seconds plus 0.618001 supplementary seconds, with zero experimental
work. Neural/Adam arithmetic and native dynamics were not replayed; not every
model trajectory array was semantically revalidated.

The full H4 raw HDF is **38,618,575,691 bytes**, SHA-256
`5805a0ea194b0cf2228c4a98830f6156887856f6eb46d1790e76942712c7a48c`.
The full 278,187,145-byte ledger has SHA-256
`c229397f0e3118da76a3db3ba1e287f8e1353fe26c975ae1e1e7b443e04b5f2d`.
Both remain locally preserved; the compact export is supplemental and never
substitutes for the full raw path. Its historical `remote_evidence.json`
filename/scope was left unchanged in frozen source; the explicit local locator
corrects the location without rewriting that source or the original export.

H4 configuration SHA-256:
`409bfb60d5b5563a5665c4105e4e211a7f8af4b21977b32ba828612a10c45113`.
Input-seal SHA-256:
`0833f5291bdfccbeb55c6be85478aab095c583e29634d3b33823a1f6603c1c2b`.

## Complete new work accounting

| New stage | Model controls | Live controls | Forward rows | Backward rows | Adam steps | Internal seconds |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 0 | 17,245 | 5,192,109 | 5,120,000 | 20,000 | 328.194760 |
| H4.1 | 74,569,728 | 18,405 | 45,735,401 | 1,536,000 | 6,000 | 908.526899 |
| New total | 74,569,728 | 35,650 | 50,927,510 | 6,656,000 | 26,000 | 1,236.721658 |

All **74,605,378 requested controls completed**, with zero failed/rejected
queries or measurements and zero unknown work. There were 154 completed or
natively terminated episodes. Summed external launch time was 1,238.719260
seconds. L1's caps were 250,000 controls/6 million forward rows; H4's were
80 million controls/60 million forward rows. Exact scheduled Adam and backward
budgets were completed. Unused confirmation and second-round allowances were
not spent.

H4 forward work reconciles as 44,083,844 guided-bank rows +6,000 teacher
post-hoc rows +1,536,000 training rows +97,152 full-data diagnostics +12,405
standalone actions. L1's separate breakdown is in its detailed report. Source
preparation, tests, failed staging reviews, saved-data audits, two retained
figure revisions, exports, transfer and engineering are separate overhead.
Not all reasoning/tool overhead is timed; no efficiency claim is made.

The [complete cost receipt](../artifacts/2026-10-30_experiments_execution_001/cost_accounting.json)
charges inherited v1/v2, corrective diagnosis and v3.1 work once. Relevant
cumulative work is 229,187,422 requested controls (229,187,421 completed),
104,337,452 forward rows, 24,576,000 backward rows and 96,000 Adam steps. The
one inherited reserved-but-unexecuted control belongs to the old failed
diagnostic harness; this investigation added no such failure. Unrelated
archived research is not represented as a complete project-efficiency census.

## Causal verdict and next version

The weight change affected optimization, executed trajectories and teaching
data. It did not improve this student's next standalone execution. Independent
discovery remained successful; all three successful data alternatives produced
negative fixed-update transfer. The mechanism's failure therefore cannot be
explained by an absent weight manipulation, untrained treatment child, failed
teacher, unequal incoming learner or unmatched update budget.

These experiments do not identify a unique cause among objective/feedback
sensitivity, coverage, action-target ambiguity, optimizer behavior or execution
representation. They do not prove that every adaptive coefficient is useless,
that torque-policy walking is impossible, or that CIO–GPS fails. Our direct
torque regression and execution remain different from the paper's state-to-state
formulation and execution optimizer. [Primary-source comparison](loop_v3_prior_art.md).

The explicitly versioned next coupling hypothesis is
[H4.2: parent-relative learner-response selection](adaptive_coupling_h4_2_hypothesis.md),
**untested and unlaunched**. It is motivated by the observed count-feedback
blind spot, not an assumption that another weight will help. With the current
three candidates, parent-nonregression rejects all three; a better feedback
metric alone does not produce a walking student. The student learnability
question remains separate and open. No new feature tweak is presented as a
coupling formulation, and no further experimental work followed these negatives.

## Runnable implementation and raw evidence

- [L1 worker](../scripts/diagnostics/student_learnability.py), [protocol](student_learnability_protocol_l1_001.md), [closed result](../runs/2026-10-30_student_learnability_l1_001/result.json), [all fitting curves](../runs/2026-10-30_student_learnability_l1_001/learning_curves.csv), [full raw](../runs/2026-10-30_student_learnability_l1_001/evidence/raw.h5).
- [H4 worker](../scripts/diagnostics/adaptive_coupling.py), [protocol](adaptive_coupling_protocol_h4_1_001.md), [closed result](../runs/2026-10-30_adaptive_coupling_h4_1_r1_001/result.json), [all fitting curves](../runs/2026-10-30_adaptive_coupling_h4_1_r1_001/learning_curves.csv), [full raw](../runs/2026-10-30_adaptive_coupling_h4_1_r1_001/evidence/raw.h5), [ledger](../runs/2026-10-30_adaptive_coupling_h4_1_r1_001/evidence/ledger.jsonl).
- [H4 plotted data](../artifacts/2026-10-30_adaptive_coupling_h4_1_report_002/plotted_data.json), [per-start data](../artifacts/2026-10-30_adaptive_coupling_h4_1_report_002/standalone_episodes.csv), [PDF figure](../artifacts/2026-10-30_adaptive_coupling_h4_1_report_002/learning_and_outcomes.pdf), [artifact reproduction guide](../artifacts/2026-10-30_adaptive_coupling_h4_1_001/README.md).

Workers refuse existing evidence directories. Exact replays require copying
only sealed source/configuration/input files into a fresh directory and using
the original pinned platform; copying old result/evidence directories is not
a replay. New scientific replications need separately frozen fresh seeds and
inputs: their prepare commands deliberately reject already-used study seeds.
