# Independent search and global learning: frozen mechanism pilot

This is a new experiment. It neither consumes nor changes historical Walker
runs. The replanning-value repair motivates one diagnostic; it is not assumed
to solve discovery, learning, or control. Run once with the checked-in JSON and
retain the outcome, including failure. No horizon, model, seed, criterion, or
budget changes after launch. A source-changing failure requires a separate
named trial; its spent queries remain evidence.

## Question and scope

Can independent local model search acquire successful experience, can a
global function learn usable long-term information from complete outcomes,
and does that information improve *executed* receding-horizon control beyond
strong search with memory and a learned-value MPC control after costs?

The global component is a cost-to-go function, not an action policy. This
actor-free pilot is a mechanism test related to GPS. It establishes no
standalone-policy transfer or completed GPS result. No behavior cloning,
SAC, actor initialization, novelty reward, or particular neural critic is
required. RBF ridge regression is a deliberately small implementation choice.

## Transitions, objective, information available

The sole main experiment is a deterministic, exact-model continuous
MountainCar variant. Position p is in [-1.2,.6], velocity v in [-.07,.07],
action a in [-1,1]. One control computes

    v' = clip(v + .0015*a - .0025*cos(3*p), -.07, .07)
    p' = clip(p + v', -1.2, .6)

At the left boundary negative velocity becomes zero. Reaching p>=.45 is
absorbing, with zero subsequent cost. A transition starting below the goal
costs 1+.02*a²; the transition reaching the goal is charged once. There is no
success bonus, terminal failure penalty, or learned reward. The objective is
the undiscounted sum for 180 controls. Every reset samples p uniformly in
[-.6,-.4], v=0, using its recorded private seed. Goals and dynamics are fixed;
this is within-task transfer to fresh resets, not new-task generalization.

All controllers know the same exact simulator and fixed heuristic
80*max(.45-p,0). That heuristic is a planning preference, never a return
label or an asserted lower bound. There are no contacts or numerical solver
states. Search restarts independently at each decision, so its only persistent
memory is the immutable trajectory archive and optional frozen value. Model
inputs are [p/1.8,v/.07,remaining/180]. Controller, archive, model and target
version hashes are attached to each dataset; epochs never pool different
continuation labels as if they described the same controller. Future planner
randomness is integrated out, not fitted as a PRNG-state feature.

## Independent discovery and memory

At each decision retain 64 independent bounded plans: half independent
four-control blocks, one quarter sixteen-control blocks, and one quarter
constant controls. Values are uniform in [-1,1]; proposal zero is the exact
zero plan. The private proposal seed is fixed by the episode stream and step.
No actor, value, archive or trust weight supplies these controls. The
independent plans are scored with measured running costs and the known
heuristic; MPPI's exponential weights at temperature 3 generate a mean plan,
which is separately re-simulated. This is a multiscale sampling MPPI variant
with a verified discrete safeguard, not a claim to reproduce the unbounded
Gaussian importance-sampling derivation.

Eight additional archive fragments are proposed: four nearest rows in
state/time features and four lowest measured suffix costs among the nearest
64, avoiding duplicates. Fragments are replayed from the CURRENT state;
short fragments repeat their last action as proposal padding. The archive
contains successful and failed complete trajectories, with at most 4096
rows balanced across roots. Its nearest-neighbor terminal cost is used within
feature distance .25; otherwise use the known heuristic. At the finite cutoff
the tail is zero. Baseline selects the minimum score among independent plans,
the verified MPPI mean and memory fragments. This is a strong memory control,
not bare short-horizon MPPI or an unverified stored-action lookup.

Discovery is the separate, value-free acquisition controller: exactly the
same independent mechanism at H=100, executing ONE selected action then
replanning, for a complete episode. Ordinary control uses H=16. A long-horizon
query that predicts success does not count as acquired successful experience;
an actual closed-loop acquisition trajectory must reach the goal. Failed
acquisitions remain in memory and the ledger. These temporal proposals plus
the longer acquisition horizon supply the only route to initially unseen
successful behavior. The value learner cannot create such evidence.

## Smallest loop and comparisons

Three independent training seeds, three cycles, three arms. One independent
bootstrap acquisition per seed is executed once and shared at initialization;
charge it fully to every arm's counterfactual standalone cost, once to the
actual experiment. All learned influence begins absent.

1. **Memory:** use the entire six-million-control training allowance each
   cycle for independent H=100 acquisition. Refresh memory after every
   complete acquisition. Stop before another complete episode's conservative
   upper bound would exceed the allowance. Do not turn leftover allowance into
   fictitious controls. This reinvestment is the compute/experience control.
2. **MC:** acquire one independent H=100 episode, refresh memory, and if an
   old scorer is active, recheck it on four fresh paired roots against the
   new-memory baseline. Failure or insufficient budget revokes it. Freeze the
   surviving full controller C_k: archive, scorer, influence and H=16 search.
   Collect six complete C_k episodes; first four roots train, last two are
   held out. In each of the first two training roots, take the middle cached
   bank (banks every 12 controls), choose a uniform independent action, execute
   it and the memory baseline action, then run the actual C_k continuation to
   completion with matched fresh future randomness. Forced predecision states
   are not labeled V^C_k. These branches provide measured action coverage.
3. **POLO-style value MPC:** same acquisition, archive refresh, retention and
   ordinary collection. Fit three optimized model H-step backup sweeps on
   training-root banks; no actor or proposal regularizer. Sweep zero uses the
   known heuristic and subsequent sweeps use the previous fitted value:

       y(z) = min_i [ sum_{j=0}^{H-1} c_ij + V_previous(z_iH) ].

   These targets use already charged simulator queries, not additional free
   rollouts. They are approximate optimality targets, not measured returns of
   C_k. Clamp to the known finite cost range. No unused MC training branches
   are executed in this arm. It receives the same heldout diagnostics and
   complete-controller qualification. This is a controlled POLO-style
   comparator, not a full reproduction of POLO's ensembles/priors/optimism.

MC targets are exact backward sums of actual complete C_k trajectories:

    G_t = c_t + ... + c_(179),     V(z_t) approximates E[G_t | C_k].

No bootstrap or optimized hypothetical tail enters these MC labels. New
models fit this epoch's target controller only. There are up to 64 farthest
training-feature RBF centers, width .25, ridge .01, a linear feature term and
intercept; roots receive equal total loss weight. Predictions are clipped to
[0,1.02*remaining], an exact cost range. Support requires distance <=.25 to
a training input. Dense suffix rows and repeated branches are correlated;
independent root counts are reported separately.

MC changes planning by physically scratch-executing each UNIQUE bank first
action from the current state, then scoring

    Q_i = c(z,a_i) + V^C_k(z'_i, remaining-1).

The deployed controller executes only the chosen first action and replans.
There is no retained nominal to hand off. A promoted model evaluates its
predecessor C_k, while the new repeated reranker is a different controller;
therefore empirical full-controller qualification is necessary. No policy
improvement theorem or controller-equivalence assumption is asserted.
The POLO-style arm scores the H-step running sum plus learned endpoint value.
For both, unsupported baseline means whole-decision fallback; unsupported
alternatives cannot displace it. Ties retain baseline. All independent plans
remain present even with a harmful model.

## Trust, promotion, revocation, deployment

Trust is binary absent/active. On the two whole heldout roots, log MC return
RMSE, nearest-memory RMSE, horizon-only RMSE and support. Shadow-score cached
banks; per root select the changed supported decision with largest predicted
improvement, earliest step breaks ties. Measure each selected first action
and its memory baseline followed by actual C_k on two paired future streams.
Selection precedes those measurements; rejected selections are retained.

MC requires return RMSE/180<=.20, two nontrivial comparisons on two roots,
paired difference RMSE/180<=.10, >=.75 correct improvement signs (thus both
with two pairs), and measured mean difference <-1. The POLO-style model's
discrepancy from V^C_k is diagnostic; optimality targets need not equal V^C_k,
so this prediction filter does not veto its qualification.

Next run FOUR fresh complete paired qualification roots against BOTH the
same-archive memory baseline and incumbent, reusing identical-controller
executions once. Require >=2 mean cost improvement, >=3/4 wins, upper 80%
paired-root bootstrap interval below zero and no fewer successful roots.
Only then promote to full influence. Retention uses the same empirical rule
against memory after every archive refresh. These small development panels
are empirical screening rules, not guarantees. A model frozen with one
archive is never silently deployed with a different archive.

Training, branching, shadow scratch work, rejected fits, retention and
qualification share each arm's six-million-control/cycle cap. Reserve a
conservative complete-query upper bound before optional branch/qualification
work; budget insufficiency rejects rather than extends the run. MC's four
training-root ordinary episodes and training branches may enter NEXT cycle's
archive. Heldout, qualification and evaluation experience never trains or
refreshes archives. Final deployment uses exactly the last qualified packet;
there is no actor, distillation or further update.

## Frozen panels, acceptance, and failure attribution

Eight matched development roots at initialization and after each cycle; 24
fresh final roots per training seed and arm. Development resets repeat across
cycles and do not support independent final inference. Seed namespace starts
at 60 million, separate from Walker studies; exact role formula is in runner.
No environment outcome is used to select settings or extend three cycles.

Success is a conjunction, not a favorable mean in one panel:

- **Discovery:** at least one nontrivial successful executed independent
  acquisition is recorded. Report success counts/time to first acquired success
  per seed and spent queries, including failed attempts.
- **Prediction:** MC heldout calibration passes; final diagnostics must also
  show strictly lower beyond-H RMSE than a remaining-time-only predictor.
  Nearest-memory errors show whether smoothing adds anything to retrieval.
- **Decision:** promoted guidance actually changes executed first actions.
  Scores or hypothetical plan improvements alone do not count.
- **Control:** final MC mean cost improves BOTH controls by >=5%, with negative
  cost differences in ALL THREE training seeds and an upper 95% hierarchical
  paired bootstrap interval below zero. Report success rates too. With three
  training seeds this remains a small pilot, not broad robustness evidence.
- **Efficiency/exploration:** additionally outperform both strong controls
  after charging every acquisition, learning query, qualification, failure,
  evaluation and preflight; require no larger charged control count for MC.
  Plot actual-cost curves. Shared acquisitions cannot establish a discovery
  advantage. No isolated walking/success/control gain is an exploration claim.

Record discovery failures (no useful acquired outcomes), prediction failures
(heldout return/difference error), distribution failures (unsupported candidate
states), optimization failures (selected predictions disagree with measured
branches), and control failures (full executed qualification/final regression).
Support is an empirical interpolation test, not a calibrated uncertainty bound.
Absence of eligible changed decisions is reported as inactivity, not safety.

## Cost and raw evidence

Every modeled control slot is counted, including absorbing/padded slots.
Independent bank, memory replay, MPPI mean verification, one-step scratch and
executed controls are disjoint categories. Forced branches execute the first
action again; equal actions reuse one measured outcome with an explicit reuse
flag. We continue absorbing episodes to the finite cutoff and count those
calls too. This deliberately conservative accounting avoids free termination
padding. Prediction rows and fitting updates are separate counters. Failed
queries retain spent/attempted slots. No GPU, external data or learned model
rollouts substitute for counted exact simulator transitions.

`queries.jsonl` records every plan, execution, scratch query, fitted backup,
selection and branch result, including seeds, controller/archive hashes,
memory row ids, verified mean plans and independent proposal digests.
Per-episode NPZs contain actual states/actions/costs/complete returns, all bank
first actions/endpoints/stage costs/local scores/selections. The raw ledger,
source snapshot and seeds reconstruct full independent proposal trajectories;
all archive fragments refer to saved measured episodes. Model packets, exact
targets/root groups and heldout predictions are saved separately. Config,
source hashes, Python/NumPy/SciPy/platform, wall time and final integrity checks
are archived. Report actual experiment cost with shared bootstrap counted once
and standalone per-arm cost with it charged to each arm. Charge the 212-control
preflight once globally and once to each standalone arm as a common setup cost.
Unit test calls are development checks, not main-run samples; preflight has its
own exact raw ledger. Keep all existing run directories unchanged.

## Minimal executable falsifier and prior art

Before main training, run a four-control continuous integrator with deferred
quadratic terminal cost. Independent full-plan acquisition is re-executed;
fresh complete zero-controller suffixes fit a quadratic global value. Heldout
prediction, changed independent-bank decisions and actually re-executed
improvement must all pass. A corrupted value must fail promotion, and injected
post-promotion corruption must be revoked. Any failed sentinel writes negative
evidence and prevents the main launch. This exact small problem validates
causal plumbing only; success does not validate MountainCar generalization.

Independent sampling MPPI is established by
[Williams et al.](https://arxiv.org/abs/1707.02342). Local planning with global
value learning and coordinated exploration, including MPPI, is already the
explicit premise of [POLO](https://arxiv.org/abs/1811.01848). Trajectory terminal
sets and measured suffix costs are established in
[Learning MPC](https://arxiv.org/abs/1609.01387), whose guarantees this nearest
neighbor implementation does not inherit. [LOOP](https://proceedings.mlr.press/v164/sikchi22a.html)
is related learned-value lookahead control; its actor regularization is not
imposed here. [GPS approximate mirror descent](https://arxiv.org/abs/1607.04614)
includes fitting a global action policy, absent here. The broad thesis is not
a novelty claim. The possible contribution is a small, auditable causal and
cost-accounted experiment with controller-specific evidence, earned/revocable
influence, and strong controls. A negative outcome is a valid delivered result.

Run from repository root:

```sh
.venv/bin/python -m pytest -q tests/test_independent_core.py tests/test_independent_preflight.py tests/test_independent_learning.py tests/test_independent_driver_review.py
.venv/bin/python -m scripts.training.run_independent_search --out-dir runs/NEW_UNIQUE_NAME
.venv/bin/python -m scripts.diagnostics.report_independent_search --out-dir runs/NEW_UNIQUE_NAME
```
