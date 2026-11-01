# Current Walker research scope

[D1/D2](transfer_diagnosis_d1d2_results.md) is closed and independently audited:
all final W2 traces plus K1 source data, then 20 exactly matched teacher continuations
and 220 single-action substitutions. It performed no learning, new inference or
MPPI search. Support does not certify competence; consequences under the fixed
open-loop continuation do not consistently rank whole-policy retention. The
[S1 source-learning comparison](student_source_s1_results.md) completed six fixed
fits but was interrupted by a host reboot during confirmation. Its original
bytes and incomplete roster are preserved. [S1.1](student_source_s1_1_protocol.md)
proposes fresh evaluation of the existing actors after durability work; it is
unlaunched. No new learning or coupling mechanism is selected from the incomplete
comparison. S1's original proposal and protocol remain preserved.

[W2](student_update_w2_results.md) is closed: fixed 2k checkpoints improve held-trajectory error, retention and complete cost relative to their exactly continued 10k counterparts. Retention/cost benefits repeat on the disjoint confirmation panel, while all endpoints have zero walking and full-survival successes. It is a student update-budget result, not adaptive coupling or competent control. No follow-up was launched.

[W1](student_width_w1_results.md) is closed: widening improves fitted reconstruction but worsens held-trajectory transfer and standalone control across three paired seeds. It tests a student-width intervention, not adaptive coupling. The recommended [W2 contrast](student_update_w2_results.md) subsequently ran under separate explicit authorization.

The [K1 compatibility diagnostic](action_compatibility_k1_results.md) is now closed and independently audited. It compares MPPI/random/qualified-reference data against untrained/cloned/reference policies, with no new learning. The SAC reference passes 16/16 primary and 16/16 confirmation, while clones agree better with MPPI but fail standalone. This tests action agreement and conditional competence, not prospective teachability or a new coupling mechanism. Sparse cross-arm input collisions do not establish contradictory supervision in the actual teacher-training corpora. Its recommended width contrast subsequently ran as W1 under a separate frozen protocol and user authorization.

The [L1 and H4.1 studies are closed](walker_experiments_l1_h4_results.md), with
complete raw evidence, independent audits, learning curves and costs. Both are
valid negative results. V3.1 remains a fixed-demonstration/fitting comparison;
its lost survival benefit was substantive, and that earlier study did not
change a weight or acquire with an updated policy. H4.1 has now done both.

| Track | Question | Current status |
|---|---|---|
| [Student learnability L1](student_learnability_track_v1.md) | Can this torque student learn competent walking from successful teacher data? | Fresh clipped/full-velocity 10k training both scored 0/16 walking and survival; nearly identical final MSE. The broader question remains open. |
| [Coupling H4.1](adaptive_coupling_h4_1.md) | Does an outcome-triggered reduction of agreement pressure improve the next acquisition–learning outcome? | Actual changed-weight acquisition executed; all six teachers walked, all three matched children trained, all regressed against their incoming parent. |
| [Next coupling hypothesis H4.2](adaptive_coupling_h4_2_hypothesis.md) | Does parent-relative learner-response utility improve teaching-weight selection over relative count feedback? | Untested and unlaunched; all three current candidates violate parent nonregression. A better metric alone does not establish walking. |

The H4.1 rule halves the agreement weight if the fixed-update guided child loses
task success **or** full survival against its matched independent-data child;
otherwise it holds the weight. V3.1 therefore proposed
`0.1199341983507337 -> 0.05996709917536685`. This is a development-informed
prediction, not retrospective evidence that backoff helps.

The first prospective causal test forked half-weight, fixed-high and zero-weight
acquisition from the **same updated guided child and complete learning state**.
It kept features, normalization, targets, data sizes and update budgets fixed.
The weight change causally changed selected controls at an identical bank and
physical/planner state. Teacher–student cost gaps nevertheless widened after
learning. The frozen outcome criteria failed, so no confirmation or second
round ran. Standing survival or lower MSE cannot substitute for walking. The
count-only rule would hold on zero/zero ties despite shared regression; that
next proposal was not executed. No new component or environment was added.

A half-versus-high win would establish the effect of one weight reduction at
this learner state. Demonstrating that feedback adaptation beats a lower fixed
weight or a predetermined decay schedule requires subsequent decisions that
differ from those schedules and a matched comparison. Two finite useful rounds
would not prove asymptotic convergence.

The [primary-source mechanism review](adaptive_coupling_review_v1.md) identifies
existing adaptive GPS and learner-response teaching precedents. The [rule
utility](../src/gps/mppi_gps/coupling_backoff.py) only computes a proposed weight;
unit tests are not an empirical test of teacher or student performance. The
subsequent actual experiments are documented in the closed results. H3.2's
inherited-state intervention remains unlaunched; the executed L1 fresh-training
variant is a separate student diagnostic. The [previous scope version (external evidence)](evidence_availability.md#external-425f89902caf)
and previous reports remain preserved with their original
hashes in the [scope-correction archive (external evidence)](evidence_availability.md#external-ab1aecb6df5f).
