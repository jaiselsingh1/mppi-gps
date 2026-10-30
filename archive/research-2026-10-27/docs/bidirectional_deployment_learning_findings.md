# Deployment learning: findings and limits

The new learning formulation did not close a beneficial bidirectional loop.
It improved one changed actor in a finite qualification, but the complete
repaired study lost ending reliability. Both feedback rounds ran; neither
promoted a guided teacher or improved the next standalone policy.

This is development on the cheap Ending environment, starting from one already
competent pure-return actor. It is not a fresh-discovery, Acrobot, Walker,
three-seed or efficiency result. Older runs remain unchanged.

It also omits an explicit MPPI-informed agreement constraint on policy
learning. Teacher states provide contexts, while endpoint constraints come
from the task. This is narrower than the jointly adapted teacher/student
relationship in CIO-based GPS; controller qualification does not replace that
coupling. See the [coupling limitation](bidirectional_deployment_learning.md#the-missing-cio-style-coupling).
The negative results apply to this candidate, not the broader research vision.

## Why finishing was insufficient

The original 64-start raw diagnosis found 64/64 captures for both the actor
and independent search plus trajectory memory. Costs were 2.29131 and 1.77755.
The actor reached the ending region earlier (mean control boundary 7.56 versus
12) and committed farther into it (mean final y=-0.34576 versus -0.10364).
Its greed running cost was 1.27387 versus search's 0.66766; its lower effort
did not compensate. Learning capture had not learned efficient preparation
and timing. This diagnoses recorded behavior, not the cause of every update.

Original evidence: `runs/2026-10-27_bidirectional_actor_gap_diagnosis_001`.

## What changed in learning

The [algorithm description](bidirectional_deployment_learning.md) gives the
update and continuation contracts. Teacher experience supplies acquired
starting states; the actor executes its own complete continuations. There are
no teacher action labels or learned critic. Known transition derivatives link
its early actions to later costs and ending position. Actual complete outcomes
test every changed actor. Independent MPPI proposals and its memory are protected.

The development sequence retained every version:

| Run | Learning change | Outcome |
| --- | --- | --- |
| deployment_adjoint_001 | Full-network running-cost gradient and relaxed ending barrier | 0/192 accepted updates; both matched feedback rounds completed negative. |
| deployment_adjoint_002 | Separate cost descent and ending halfspaces, Euclidean parameter bound | 0/192 valid descent directions; no accepted update; both rounds negative. |
| deployment_adjoint_003 | Frozen backbone, mean-head learning in a closed-loop action metric | First update accepted; next feature consistency check failed. Operational interruption; no full-loop verdict. |
| deployment_adjoint_004 | Same metric method, repaired scalar feature-forward layout | 192 valid numerical directions, 43 accepted updates; fresh ending qualification failed. Both rounds and untouched final evaluation completed negative. |

The scalar-layout repair retained the 2e-7 consistency tolerance. Before the
full repaired run, the interrupted run's first accepted actor was checked at
the changed parameters: analytic derivatives agreed with paid autograd, then
16 fresh paired starts captured for both actors. Mean cost improved 2.29568
to 2.17697, with 16 gains and no losses. Its arithmetic-only qualification audit
passed. This actor was selected chronologically, not by choosing the best
post-hoc checkpoint. It was a qualification diagnostic, not the final actor
of the repaired run or evidence of better MPC/next-student learning.

## Complete repaired study

Initial fresh qualification captured 8/16 for the new actor versus 16/16 for
the source parent. The algorithm retained its original parent. Round-zero
children captured 6/16; round-one children captured 10/16. In each round, RAW
and guided acquisition produced identical child policies. No actor handover
executed or teacher was promoted. The parent and private independent search
were not weakened to manufacture a benefit.

On the untouched 64 paired final starts:

| Controller | Captures | Complete mean cost |
| --- | ---: | ---: |
| Retained standalone parent | 64/64 | 2.279455 |
| Independent-data standalone child | 27/64 | 12.977002 |
| Guided-data standalone child | 27/64 | 12.977002 |
| Independent search plus memory | 62/64 | 2.351424 |
| Guided controller, using independent fallback | 62/64 | 2.351424 |

The independent reference's two failures are retained in its mean. On this
fresh panel the parent is cheaper than independent search overall; the
earlier statement that the actor loses complete cost is not universal across
panels. Search's successful conditional cost was 1.76435, but conditioning away
its failures cannot replace its measured complete cost or select a controller.
Failed teacher qualification prohibited switching guidance on using final
evaluation outcomes.

## What the negative result reveals

The new child reduced greed cost from the parent's 1.26202 to 0.98127 and
effort from 0.01744 to 0.01135. It moved much closer to the desired boundary:
final y ranged from -0.14500 to -0.05817, versus the parent's -0.44506 to
-0.24459. But 37 endings missed the y<=-0.1 condition, by 0.000161 to 0.041830.
Those failures alone contributed 11.5625 mean cost. More efficient running
behavior did not preserve the ending across initial contexts.

[All recorded final endpoints and the labeled boundary zoom](../runs/2026-10-27_bidirectional_deployment_margins_report_001/ending_margins.png)
show this dependence on initial context. The original CIO-based work also
teaches feedback gains. Here the actor's own derivatives were used for credit;
the teacher's corrective feedback structure was not transferred. That is a
specific missing mechanism to investigate, not a demonstrated fix.

All 43 accepted updates passed their actual four-start ordinary capture
checks and acquired worst-violation tests. The eight-start development panels
and fresh qualification subsequently failed. This is observed generalization
failure of the sampled acceptance constraints. It does not establish that the
actor architecture lacks capacity, that all analytic policy gradients fail,
or that acquired-state experience is useless.

The measured action RMS of accepted metric steps stayed below their declared
first-order metric bound; a simple claim that action trust was exceeded is
unsupported. Nevertheless, endpoint first-order prediction error reached
0.02430 on tested trajectories, larger than the 0.001 surrogate margin.
Actual changed-controller tests protected the tested contexts; they did not
guarantee ending reliability elsewhere. The present experiment combines a
changed objective, restricted trainable parameters and a metric, so it cannot
identify one ingredient as the unique cause.

Failure separation:

- Discovery: useful cheap-task experience existed in the paid ancestry;
  physical MPPI coverage remains unresolved.
- Target ownership: gradients and outcomes belong to the exact executing
  actor, without copied planner actions or relabeled updated-actor returns.
- Numerical execution: run003 stopped on a feature-layout mismatch; it is
  separate from completed scientific negatives.
- Optimization: barrier/Euclidean proposals made no accepted progress; the
  metric method made measured local progress.
- Generalization: local update acceptance did not preserve fresh ending success.
- Trust/control: failed qualification retained the parent and independent
  fallback. The independent reference also had two final control failures.
- Feedback learning: no improved teacher, no different useful acquisition
  data and no guided-child advantage.

## Evidence, reproduction and costs

Runnable stages and frozen caps are in
[protocol 3](bidirectional_deployment_adjoint_protocol_dev3.md).
Run directories contain copied source/configuration, checkpoints, executed
episodes, model candidates, component costs and query/work ledgers.
`deployment_learning_diagnosis_001` and `deployment_endpoints_001` are post-hoc
saved-array diagnoses, with zero new controls or neural forwards.

The four studies, including the interrupted run, charged 7,691,051 scalar
controls: 7,024,475 model and 666,576 executed. They used 1,300,336 neural rows,
704 Adam calls, 385 manual parameter updates and 1,856 autograd-grad calls.
Summed study body wall time was 2,129.259 seconds; concurrent wall scopes
overlap. Tests, qualification, numerical replay, diagnoses, plotting and audits
are additional paid work, recorded separately in the additive cost receipt.
Prior discovery/learning ancestry is retained and charged once in the scoped
lineage; historical unmetered development prevents an all-cost efficiency claim.

All three completed studies passed independent arithmetic audits. Run001 uses
additive audit002: its original full raw checks passed, while the first adapter
failed on absent legacy metadata. The repaired adapter inherited those bound
raw checks and verified the missing update evidence. The first repair wrapper's
absolute/relative-path error remains recorded. Run002 and run004 passed audit001.
Run003's audit has no arithmetic errors but remains PENDING because the study
stopped operationally. No scientific evidence was rerun to repair a verifier.
These receipts distinguish evidence validity from the negative scientific verdict.
They reconstruct raw transitions/costs and numeric update arithmetic, but do
not independently repeat every neural forward or optimizer operation.
The metric runs contain an inherited incorrect backward_calls=1 per-gradient
intent field. Global before-call hooks establish actual analytic versus
autograd work, and the verifier counts this metadata defect explicitly.
Future live code corrects that field; frozen study source is unchanged.

[Learning curves, fresh comparisons, charges and recorded trace animation](../runs/2026-10-27_bidirectional_deployment_report_001/receipt.json)
and the [additive cost receipt](../runs/2026-10-27_bidirectional_deployment_costs_001/costs.json)
bind the corresponding raw inputs. The endpoint CSV is in
`runs/2026-10-27_bidirectional_deployment_endpoints_001/actual_endpoints.csv`.

Gradient-through-model learning and constrained policy optimization have
established precedents, including
[SVG](https://proceedings.neurips.cc/paper/2015/hash/148510031349642de5ca0c544f31b2ef-Abstract.html)
and [CPO](https://proceedings.mlr.press/v70/achiam17a.html). This finite projection
is not CPO and supplies no global guarantee. Novelty of the combined loop is
unestablished. Its proposed contribution remains the empirical relationship
between protected discovery, reusable learning and better subsequent learning;
this development study did not establish that relationship.
