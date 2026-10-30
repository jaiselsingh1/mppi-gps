# Frozen-value planner mechanism diagnostic

The user authorized three executable diagnostics and a concrete recommendation:
measured continuation utility, prediction/ranking fidelity, and transfer to
actual receding-horizon execution. Both active walking studies remain unchanged.
This is a separate simulator diagnostic, not another training run.

## Inputs and scope, fixed before new measurements

- Coupled MLP frozen after22 committed cycles (`value_0021.pkl`) and paired
  ridge after2 committed cycles (`value_0001.pkl`). Each has its own unchanged
  reference archive and frozen incumbent archive. Inputs were copied and hashed
  before execution; no live checkpoint is loaded by workers.
- Use the same independent **paired-bank** planning implementation for both
  cohorts. This isolates model/target mechanisms on fixed verified banks. It
  does **not** reproduce the original coupled controller's positive-alpha
  extra-weighted-mean or support-filtering behavior.
- Four fresh roots per cohort:42000000–42000003 and42000100–42000103. No
  existing training, development or reserved final reset is used. Capture one
  decision after128 memory-only controls. Early falls remain reported with no
  replacement or easier-state selection. K32/H48,1000-control finite horizon.
- Freeze three distinct plans before continuation outcomes: local winner,
  actual bounded model choice if different (otherwise an explicitly exploratory
  unbounded model alternative), and a seeded uniform distinct alternative.
  Record fewer candidates if the bank lacks three distinct finite plans.

## Diagnostic1: does the target distinguish useful alternatives?

Execute each H-step prefix from identical physical state and solver warmstart.
After each healthy endpoint, restart that cohort's fixed reference once and
execute its entire remaining episode. Retain full known absorbing costs for
falls and zero future at healthy finite cutoff.

Two matched discovery seeds select a within-bank measured-return winner.
Four **disjoint** matched confirmation seeds estimate that selected plan's
advantage over the local winner. Confirmation outcomes cannot change selection.
This is a finite-query measured oracle, not an exact expected-value oracle.
Report all plans and repeats, including ties and failures, rather than only the
selected result. Hindsight confirmation minima are not unbiased improvements.

## Diagnostic2: does the learned model preserve the useful ranking?

Compare original batch-prefix-plus-predicted-tail scores with complete measured
confirmation returns, relative to the same local baseline. Report per-root
sign/rank errors, normalized difference RMSE, support and score-cap constraints.
Also recompute predictions at physically replayed endpoints and record batch
versus replay state discrepancies. Do not call an exploratory nonbaseline plan
the model's actual deployed choice.

At alpha.1, lambda3 and a two-temperature correction cap, a candidate can gain
at most.6 score units. Distinguish absence of useful predictions from predictions
that cannot affect selection because of support/cap restrictions.

## Diagnostic3: does the ranking survive replanning?

For every selected plan, execute only its first action, then memory-only MPC
until finite termination. Clone the raw planner's POST-bank nominal and first-
decision flag. Future randomness is a fresh stream paired between candidates;
it is deliberately disjoint from endpoint discovery/confirmation streams.

An additional predeclared mechanism contrast hands off the shifted selected
plan once, instead of the shifted raw weighted mean, for the local and discovery-
selected oracle plans. Subsequent nominal updates remain the production rule.
This tests a possible loss of coordinated continuation; it does not deploy a
fix or establish repeated-handoff benefits.

Finally, test the actual bounded model first action followed by the frozen
alpha.1 model-guided controller. This is explicitly **unqualified diagnostic
execution**, outside both training studies, not an earned promotion.

## Budget, integrity and interpretation

At most8 captured decisions,24 plan prefixes,144 reference suffixes,96 ordinary
first-action continuations,64 single-handoff continuations and32 frozen-guided
continuations. Falls and duplicate baseline/oracle selections reduce actual
work. Record executed controls separately from internal sampled/memory/mean
verification controls, including failed and rejected work. All traces and source
hashes are preserved. Six low-priority CPU workers; no GPU or actor training.

Whole-root comparisons are the statistical units. Four roots per model and
four confirmation seeds are a bounded mechanism test, not a population
reliability certificate or training-seed replication. The ridge has far less
training data than the MLP, and their reference archives differ; this is not a
controlled architecture comparison. An unresolved effect remains unresolved.

Entry points: `scripts.diagnostics.run_planner_mechanism` and
`scripts.diagnostics.report_planner_mechanism`. Integration smokes use different
roots,64 controls and no scientific performance claim.
