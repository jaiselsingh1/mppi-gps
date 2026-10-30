# Action-metric constrained actor learning, development revision 2

This is a new prospective development run. Preserve both earlier adjoint runs,
all failed/repaired tests, numerical probes and existing negatives. The raw
actor-gap diagnosis and fixed task objective are unchanged. No controller or
reward is weakened to obtain an advantage.

The dev0 weighted penalty traded lower ordinary cost for worse acquired ending
violation and was rejected. The first completed dev1 Euclidean direction
restored acquired endings but predicted ordinary cost INCREASE+.006365, so it
correctly became zero. A saved numerical SVD found the gradient largely aligned
with ending-state changes; the component orthogonal to eight ending gradients
had norm.00257 versus5.776 total. This is local parameter geometry, not a proof
of global policy capacity or an explanation of all learning failures.

## New formulation and qualified primitive

Keep the same bounded128x128 policy and fixed learned backbone, and train its
129mean-head parameters. Using its OWN complete actual trajectories, re-evaluate
the80 active pre-action states in one paid neural batch. Save both hidden-layer
features, means and observations. Analytically differentiate the neural head,
backbone's state response and qualified Ending dynamics through the entire
remaining actor continuation. This gives ordinary running-cost gradient g,
eight ending gradients A, and the80-row closed-loop CONTROL-sequence Jacobian J.
Teacher actions are not labels; only the named actor's measured actions anchor
the local linearization. Log standard deviation and backbone remain fixed.

The80-row replay001 matched the independent earlier autograd calculation's
cost and ending mean-head derivatives within2.84e-7 and3.82e-7 respectively.
It costs80NN rows,80analytic policy and80physical Jacobian rows, one linear
solve and512numerical halfspace checks; zero controls/optimizer updates. Its
initial metric direction met the ending halfspaces but exceeded the action
metric radius slightly (.050177>.05), so it was INVALID and never executed.
A separate cached-geometry correction adds the missing ball projection, costs
zero new NN/controls, and preserves the qualified neural derivative function
exactly (AST bound). The corrected direction has metric norm.05, numerical
constraint residual<=1e-7, and predicted cost derivative-.117882. These are
numerical qualification facts, not changed-actor outcomes.

Use F=J'J/80+1e-5*I as a local squared-action-change metric. Solve once for
F^-1*g and F^-1*A'. Start in direction -F^-1*g normalized to metric radius.05.
For64cycles project onto the same eight ending halfspaces as dev1, then onto
the metric ball. Ordinary upper bounds are max(slack,0); acquired bounds are
slack when feasible or -min(violation,.01) otherwise; slack=-.101-y_final.
Reject the direction if numerical feasibility, metric norm or strict cost
descent fails. This finite alternating projection is not an exact-QP guarantee.
One solve,512halfspace checks and64ball checks are recorded per attempt.

Try fractions1,.5,.25,.125 with ACTUAL complete changed-actor executions.
The unchanged acceptance rule requires ordinary mean cost improvement>=1e-6,
all ordinary captures with nonregression, and no increase in worst acquired
ending violation. Fresh16-root qualification and64-root final evaluation remain
separate. All rejected trials remain charged; no loss or directional prediction
can promote a policy. The physical objective retains its original binary
ending penalty, effort, greed and detour costs. Only the training update method
changes. Derivative/metric methods are established primitives, not a novelty
claim for this experiment.

The additional assumption is that the already learned backbone supplies enough
state/time features for a useful head update. The metric is local and its
generalization outside measured states is unproven. Exact differentiated
Ending transitions do not qualify a native Acrobot or Walker derivative.

## Complete loop and fixed expenditure

Everything else matches
[dev0](bidirectional_deployment_adjoint_protocol_dev0.md): exact paid pure-MC
parent and independent discovery corpus; common initial Adam reset; external
32-update MC curriculum control;64initial and32per-child new updates; two
feedback rounds; independent H4K32+8smooth+8memory MPPI and verified mean; strong
complete-memory RAW;50prefix/controller forecasts; version-bound actor-only
support; actual prefix4-then-frozen-actor21 handovers; promotion against RAW
AND actor alone; matched RAW/guided-data children from the exact same parent;
RAW-only memory append; revocation;16fresh qualification and64final roots.
Both full rounds and children execute even if a link fails. The objective/data
weighting, mean-head restriction and metric preconditioning form a combined
learning intervention; their individual causal contributions are not isolated.

New namespaces:6100000baseline,6110000training,6200000qualification,
6300000/6500000feedback,6800000--6800063final. These actual reset/evaluation
rosters are disjoint from prior experiments. Fixed config is
`configs/bidirectional_deployment_adjoint_dev2.json`. Historical barrier and
Euclidean config fields are inactive; `action_metric` defines this update.
Minimum ordinary teacher/student improvement remains.001, capture>=90% and
capture nonregression; full success still needs repair, both teacher and
next-student links, and final confirmation. Never alter those after outcomes.

Caps:3,222,784scalar controls (3,000,816model;221,968actual),464,960NN rows,
128Adam/128training backward/32diagnostic autograd calls for the MC control,
192manual metric updates,192linear solves,98,304halfspace checks,12,288ball
checks,15,360analytic policy and15,360physical Jacobian rows. Count all four
fractions rejected at all steps; unused work cannot fund retries or more steps.
Pure metric fixtures041 and cached numerical correction have zero new
controls/NN. All earlier test costs and80-row replay remain paid.

Freeze source/config/protocol/inputs before changed-actor executions. Audit raw
arithmetic, target/controller ownership, saved neural features/analytic
derivatives, numerical projection, actual acceptance and exact next-student
comparisons. A source-bound gradient audit is not independent neural replay.
One cheap-task parent cannot establish physical competence, novelty, exploration
or all-cost efficiency. Original results and budgets remain unchanged.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_deployment_adjoint_loop \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --parent-run runs/2026-10-27_bidirectional_return_curriculum_003 \
  --config configs/bidirectional_deployment_adjoint_dev2.json \
  --out runs/2026-10-27_bidirectional_deployment_adjoint_003
```
