# Ending actor comparison: complete outcomes help; the base-value surrogate fails

Run: `runs/2026-10-26_ending_actor_comparison_001`. One frozen configuration,
three paired seeds, two actor arms, 24 updates each. No outcome-driven retuning,
additional training or replacement runs. Existing runs were not modified.

The predeclared overall ending criterion failed for **both arms**. Complete
candidate-return learning nevertheless produced substantial actor-only ending
behavior: 90 of 96 fresh episodes succeeded. The critic-assisted arm produced
zero of 96. This supports a limited claim that complete outcomes can train
earlier preparation without imitating MPPI. It does not establish reliable
learning across the three seeds, a benefit from this critic surrogate, general
exploration efficiency, or an alternating GPS loop.

## What was actually learned

The learned global object is an action policy,
`u=tanh(theta0+theta1*y+theta2*clock/25)`. Each policy parameter search changes
all its future actions. Both arms start from the same zero parameters and use
the same perturbations and bounded optimizer. The critic-free arm scores those
changes using complete executions of the changed actor. The assisted arm
scores three changed actions plus a value of the frozen unchanged base actor.
Every actual update still requires complete candidate-policy improvement on
training and separate validation contexts. MPPI actions are never labels.

MPPI's memory is its own shifted selected action plan and private RNG. It has
no actor or critic input. Its independent short and long searches supply
reached states for the common curriculum. This is state-distribution transfer,
not behavioral cloning or planner-action aggregation. No learned policy
changes subsequent MPPI search in this experiment.

## Qualification and discovered experience

Across the 36 qualification resets, executed short-horizon MPPI achieved its
local lane criterion 36/36 and the desired ending 0/36. Long-horizon independent
MPPI achieved 36/36 endings. Mean full costs were approximately 20.161 versus
1.867. Mean selected short-window cost gaps to the optimistic local bound were
.00377–.00380 per control, below the frozen .04 threshold. The ending penalty
is included in every controller's model but lies beyond every controllable
short window. The representable actor witness succeeded 36/36 from original
roots and 36/36 from actual clock-3 short-MPPI states. All qualification gates
passed before actor training.

The subsequent common acquisition contains 24 complete MPPI episodes: the 12
long-horizon episodes succeed and the 12 short-horizon episodes fail. Useful
successful experience therefore actually exists in the reset bank. Discovery
depends on bounded longer independent search with the constant negative
proposal and retained feasible suffixes. It is not discovery by the short
planner alone or evidence invented by a critic.

This continuous-state lane task deliberately makes a locally rewarded choice
conflict with a delayed ending. Its local criterion is an attractive subtask,
not a necessary skill that a successful actor must preserve. Consequently the
result does not establish that a global policy preserves good walking while
improving stopping or settling. The simplification makes the credit and
continuation questions directly inspectable.

## Actor-only outcomes and paid-cost comparison

Final evaluation uses 32 fresh ordinary resets per seed, with no planner,
curriculum reset, value function or reference takeover during execution.

| Scoring arm | Seed 0 endings | Seed 1 endings | Seed 2 endings | Mean complete cost | Ending criterion |
|---|---:|---:|---:|---:|---|
| Complete candidate actor | 32/32 | 26/32 | 32/32 | 3.227 | Failed: every seed needed >=90% |
| Prefix + frozen-base value | 0/32 | 0/32 | 0/32 | 20.228 | Failed |

Full-return learning accepted 35/72 updates; critic-assisted learning accepted
14/72. Accepted updates alone do not establish ending behavior. Initial full
candidate score ranges were 9.56–13.80 with mean antithetic differences
5.13–7.87, so there was a measurable parameter-search signal. Assisted initial
score ranges were only .13–.28; its surrogate suppressed most ending credit.

The frozen common-budget comparison uses 515,788 paid controls per seed/arm.
The latest complete-return checkpoint below that budget is update 12 at
503,212 controls; assisted update 24 costs 515,788. Development ending rates
are 100% in all three full-return seeds and 0% in all assisted seeds. The
assisted-minus-full mean cost difference is +17.463. Its paired-seed bootstrap
95% interval is [17.323,17.667]; success difference is -1.0 in all seeds.
These are only three training seeds; root episodes are not independent
training replicates. The critic-help criterion fails decisively in this task.

![Paid-cost development learning curves](../runs/2026-10-26_ending_actor_comparison_001/learning_curves.png)

The later final failure is retained. In seed 1, the last full-return update
improved training mean cost 2.082074→2.060444 and validation 2.096279→2.069969,
with 4/4 successes on each panel. But its largest training initial y=.376196
finished only .000479 inside the good-ending threshold. All six final failures
start at y=.378499–.399605 and miss the gate threshold by .000219–.006628,
then pay the measured penalty of 20. This is insufficient finite acceptance
coverage and poor margin within the SAME reset distribution, not a simulator
fault or environmental distribution shift. The monitored development rate
also fell from 100% at update 20 to 87.5% at update 24. Frozen rules did not
select or roll back actors using development evaluation outcomes. No actor is
qualified for deployment under the overall criterion. We did not execute the
previous actor on the final roots, so its counterfactual final rate is unknown.

## Prediction is not the changed actor's continuation

All 72 MC value models are archived, including models whose proposals were
rejected. Each value predicts complete measured suffix cost of its named
frozen BASE actor. MPPI outcomes are never relabeled as base-actor outcomes.
Mean held-out base-value MAE is .00904 before the gate and .00547 over all
suffix rows. Known phase/lane features and restart coverage make this a small,
task-aware prediction problem; these errors do not establish useful actor
credit.

On 18 fixed diagnostic panels (72 nonzero antithetic pair comparisons), C agrees
with exact hybrid B on the sign of pair differences 98.6% of the time. B agrees
with complete candidate A only 72.2%; C agrees with A 70.8%. Pooled C−B RMSE is
1.527 and B−A RMSE 3.482. The large errors occur especially at initialization;
later small errors also reflect actors remaining in the bad-ending region.

A concrete eligible candidate in seed 0, update 0 has parameter norm .347,
inside the .6 update radius. On the two diagnostic roots:

| Measured quantity | Root 1 | Root 2 |
|---|---:|---:|
| A: complete candidate actor | 2.1031 | 2.0924 |
| B: candidate prefix, exact zero-base continuation | 20.9285 | 20.9194 |
| C: candidate prefix, predicted zero-base continuation | 20.9106 | 20.9076 |

The critic is close to the continuation it was trained to predict, while that
continuation has a different ending from the full changed actor. Raw A traces
are `queries/q0008379.npz` and `queries/q0008380.npz`; the diagnostic record
links the corresponding prefix, B tails and frozen model. This is direct
evidence against treating a frozen-base value as the complete changed-policy
return. It is not evidence that every assisted failure has this single cause:
the fixed diagnostic also measures genuine approximation error.

At 36 actual proposed changes on held-out original roots, the cost RMSEs are
only .0345 for C−B and .0805 for B−A, yet C gets the sign of full improvement
right only 38.9% of the time. Predicted improvements number 19 versus nine
actual improvements. Small absolute losses near tiny improvements remain
unsafe ranking evidence. Complete-rollout acceptance prevented many such
proposals from becoming actor updates. These oracle diagnostics were collected
after each decision and never used in fitting or subsequent training.

## Accounting and independent checks

| Work | Main controls/checks | Including independent audit |
|---|---:|---:|
| Development | 218 | 436 |
| Qualification and capacity/reached-state checks | 511,416 | 1,022,832 |
| Shared independent acquisition | 169,908 | 339,816 |
| Candidate actor scoring | 122,112 | 244,224 |
| Complete-policy acceptance, including rejected proposals | 74,448 | 148,896 |
| Critic target and held-out continuation collection | 19,872 | 39,744 |
| Exact-tail and proposed-change diagnostics | 9,000 | 18,000 |
| Development and final actor evaluation | 21,600 | 43,200 |
| **Unique actual total** | **928,574** | **1,857,148** |

Standalone cost includes the full shared seed's work for each arm, rather than
charging half to make either arm look cheaper: 556,604 controls per full-return
seed and 517,388 per assisted seed, each below the 1,000,000 cap. Unique actual
work above counts shared data once. There are 25,681 raw ledger entries,
19,560 physics-query artifacts, 1,152 bound artifacts, 72 fitted value models
and 11,376 value predictions. The audit repeats the 11,376 predictions and
charges every physics/bound verification slot separately. One intentional
terminal-planner query rejection in a development test consumed zero physics;
there were no failed experiment queries or additional training campaigns.

Main wall time is 30.705 seconds including collection, fitting and artifact
serialization. Fitting itself is .147 seconds (already included). Independent
audit wall time is 17.102 seconds; report body/plotting .407 seconds, excluding
Python/library startup and initial font-cache creation. Pytest-reported
development times were .07s core, .23s learning, .34s driver, plus a zero-physics
14-test recheck taking .27s. All 22 distinct checks pass; development physics
and bound calls are included in the table.

The independent auditor imports no training task/controller/scorer. It
recomputes equations, component costs, ending/termination flags, planner RNG
banks and retained memory, actor actions, model identities and predictions,
MC suffix labels and root exclusions, perturbations, ES proposals, both scores,
complete-return acceptance and A/B/C algebra. All 586,037 checks pass with no
failed checks. The seven original development bound checks lack separate raw
outputs; their count/test assertions are retained, their audit cost is charged,
and this archival gap is explicitly reported. The 211 development transitions
and all experiment physics/bounds have raw evidence.

## What survived, what did not

* **Task qualification and representation:** passed on the declared panel.
  Independent longer search actually discovers successful endings; actor
  expressiveness is witnessed by execution.
* **Coverage transfer:** successful acquired states can be used as a shared
  reverse-time reset curriculum without copying actions. This assumes exact
  simulator restoration, a sufficient successful region, and transfer over
  the narrow known reset support. Short search alone has no success guarantee.
* **Prediction:** accurate base-continuation targets and rankings are possible,
  but they do not describe a whole changed policy.
* **Continuation/optimization:** prefix-plus-base values can remove the return
  difference that makes a globally useful policy change attractive. Actual
  complete-return scoring learns much better endings in this bounded study.
* **Executed control and generalization:** the complete-return arm makes real
  long-horizon improvements but fails the all-seed robustness gate. The
  assisted arm fails the ending in every seed. Neither earns the overall
  deployment criterion.
* **Exploration efficiency and GPS:** not established. Successful coverage was
  supplied by longer independent search; there is no learned global-to-local
  planning feedback. No claim of efficiency against strong long-horizon
  search follows from this two-arm actor comparison.

ES policy search, Monte Carlo policy evaluation, reverse reset curricula and
learned-value MPC are established prior art. This experiment adds a small
executable separation of complete actor credit, exact hybrid continuation and
value approximation, with explicit acceptance and query accounting; it makes
no new-algorithm claim. Primary references and the distinction from conventional
GPS are in `docs/ending_actor_prior_art.md`. Frozen source, configuration,
protocol, dependency versions and hashes are in the run's `source/`,
`config.json` and `manifest.json`; raw trajectories/queries, all critics,
`ledger.jsonl`, `records.jsonl`, `audit.json` and plots are retained alongside.
