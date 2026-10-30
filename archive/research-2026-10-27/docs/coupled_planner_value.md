# Coupled independent MPPI and grounded global continuation value

This pilot tests the user's planner/value formulation first. Its deployed
controller is receding-horizon MPPI with optional learned endpoint guidance.
There is no standalone actor, action-cloning loss, SAC actor gradient or claim
that a neural policy has learned to walk. Existing experiments remain intact.

The concrete source, reset lists, model budgets and stopping rule are frozen in
the new run's protocol. The intended bounded pilot is at most 24 iterations,
with a smaller configuration permitted only before the launch freeze following
the implementation benchmark. This document explains the mechanism and its
interpretation; the recorded protocol determines the actual configuration.

## 1. Freeze the meaning of every value target

Define one continuation oracle for the entire pilot:

\[
C_{\mathrm{ref}} = \text{raw independent MPPI plus frozen bootstrap archive }A_0.
\]

Each oracle query starts from the saved full physical simulator state and
solver warmstart, with a fresh zero nominal plan, initial refinement and a
private seeded random stream. It then retains its own shifted plan for its
complete remaining rollout. It never receives the learned value, an actor
nominal or later archive additions. Thus

\[
V^{C_{\mathrm{ref}},\mathrm{restart}}_n(z)
=\mathbb E\!\left[\sum_{t=0}^{n-1}\gamma^t c_t
\mid z,\ C_{\mathrm{ref}}\text{ freshly initialized}\right].
\]

This is a fixed, restarted-controller continuation value. It is neither a
standalone policy value nor the value of the changing deployed guided planner.
Its fixed identity permits accumulated targets from different training
iterations without silently mixing continuation controllers. The run must
preserve the frozen oracle configuration, initial archive and query seeds.

Each target is the raw finite cost of an actually executed complete oracle
query. A physical fall incurs its real falling-transition charge once and the
declared finite absorbing remainder; a healthy episode cutoff has zero tail.
An interior state of that query is **not** another restart-value label: its
recorded suffix used retained planning memory. Only the query's original start
receives its measured root-return label. Additional starts require independent
fresh queries, and their work is charged.

## 2. Direct Monte Carlo learning

For measured query starts \((z_i,n_i)\) and returns \(G_i\), fit

\[
\min_\phi \sum_{i\in\mathcal D_{\rm train}}
\left[\widehat v_\phi(h(z_i),n_i/N)
-G_i/s(n_i)\right]^2,
\qquad
s(n)=5\sum_{t=0}^{n-1}\gamma^t.
\]

The scale \(s\) is a numerical normalizer, **not a physical cost bound**. The
small two-layer MLP predicts normalized cost through a sigmoid with a cap
derived only from the training targets. This prevents numerical runaway of
the network output; it does not prove accuracy, conservatism or robustness.
Holdout targets must not determine the scaler, feature standardization or cap.
There are no Bellman backups or moving target networks.

Split whole queries into deterministic training and latest-group holdout
partitions. Report raw-unit holdout RMSE, normalized MAE/RMSE and support.
Standardize features using training rows only. Candidate endpoints abstain if
their nearest training query is too far in standardized RMS distance or their
remaining horizons differ too much. These distances are heuristic coverage
indicators; they do not certify the value. In particular, holdout support and
error do not measure the worst endpoint selected by optimization.

Full states and warmstarts are restored by the oracle, while the deployed
regressor uses the declared 17 physical features and remaining time. Omitted
solver state and stochastic planner outcomes can create irreducible label
variation or aliasing. A single query is one realized target, not an exact
expected value. More accurate training cannot remove these limitations.

## 3. The reciprocal collect–fit–qualify loop

Start with a small fixed bootstrap collection, which supplies \(A_0\) and
ordinary initial experience without a successful demonstration requirement.
Maintain two arms from that common initialization:

- **Memory control:** independent raw MPPI and an evolving trajectory archive.
- **Guided arm:** the same independent proposals and archive mechanism, plus
  a qualified learned endpoint score.

At each iteration:

1. Execute complete ordinary-reset episodes under each arm's currently
   deployed controller. Save actual actions, costs, full states and warmstarts.
   The guided arm's accepted value influence therefore changes the states
   available to subsequent acquisition.
2. Select a fixed count of reachable query starts from those actual episodes
   by the recorded acquisition rule. The rule must not use goal-state labels
   or knowledge of critical preparation states. Execute complete, fresh
   \(C_{\rm ref}\) continuations from those starts and accumulate root targets.
   Both arms receive the same query-count protocol. Both may retain complete
   measured query trajectories in their respective archives. Any cross-arm
   sharing must be explicit in the run protocol.
3. Refit the guided arm's direct-return predictor from its accumulated query
   starts. Keep all query groups intact and retain holdout measurements.
4. Build one proposed deployment version using the newly frozen model and
   current archive. Propose \(\alpha_{\rm proposed}=\min(.5,
   \alpha_{\rm incumbent}+.1)\); the incumbent's influence increases only
   after successful promotion. Raw MPPI proposal sampling remains independent
   of the learned component. Candidate scores combine measured horizon cost
   and supported endpoint estimates. Unsupported candidates abstain; raw
   planning remains the fallback. Refreshing the archive revokes prior trust
   until the changed controller qualifies again.
5. On six fresh common qualification resets, compare complete closed-loop
   returns of the proposal against both the memory control using the **same
   newly updated archive** and the prior incumbent. Use paired random seeds.
   Independently audit four selected candidate endpoints from the proposal's
   qualification trajectories: restore each endpoint and execute a fresh
   complete \(C_{\rm ref}\) continuation. These audit labels and traces never
   enter fitting or either archive. Require at least two supported audited
   endpoints and normalized audit RMSE at most .2, as well as the recorded
   full-episode development gate. Retain the incumbent if promotion fails.
   The proposed controller and model stay frozen throughout qualification;
   duplicate inactive-baseline panels are reused rather than rerun.
6. Continue collection under the accepted version. Perform periodic standalone
   **controller** evaluation and one locked final evaluation. This controller
   includes MPPI; the word standalone does not denote an actor-only policy.

The intended scale before benchmarking is four bootstrap episodes, two root
collection episodes per arm per iteration, sixteen continuation starts per arm
per iteration, \(K=32\) or \(64\), \(H=48\), six qualification resets, evaluations
every four iterations and 64 fresh common final resets. Eight workers keep
the bounded trial practical. Exact settings and reset IDs must be frozen before
launch; never use reserved final resets 5000000–5000199.

## 4. Strong memory is part of the question

An archived action sequence is an additional candidate, never a teleported
solution. Select relevant archived starts, resimulate their proposed actions
from the **current** physical state and warmstart, and score the resulting
trajectory. Include raw independent candidates and the verified weighted-mean
plan. The memory control receives real query traces, not just the initial
archive. Replaying a sequence that works deserves credit; a learned component
must beat this control rather than exploit its absence.

The MPPI weighted action update is not guaranteed to outperform its best
sample. Any final weighted-mean sequence must be resimulated and subjected to
the same support checks before use. Preserving a baseline candidate likewise
does not prove long-horizon nondegradation.

## 5. What trust means here

Before a model can guide a proposal, require the declared minimum training and
holdout query counts, finite calibrated errors and sufficient held-out support.
The implementation's normalized RMSE gate and nearest-neighbor radius are
empirical development rules. They are not a statistical certificate.

The independent evidence for promotion is realized paired full-episode cost
and survival, plus direct fresh-oracle checks of selected candidate endpoints,
not the model's own score, training loss, age or agreement with another
network. Selected-endpoint auditing addresses the optimizer's tendency to
choose flattering value errors; four checks do not bound all such errors.
Six adaptive development resets per proposal cannot establish
reliability on all starts. Fresh qualification blocks limit reuse, while the
locked final resets remain outside training and qualification. Report rejected
proposals and abstention rather than quietly increasing influence until one
works. Local horizon-cost nondegradation must not veto every costly preparation
action; promotion uses the declared full finite-return objective.

No theoretical policy-improvement guarantee is asserted. The endpoint oracle
restarts once after an imagined open-loop plan, whereas actual deployment
executes a first action and repeatedly replans with retained memory. The score
and deployed continuation therefore differ. Qualification measures the actual
controller but does not algebraically eliminate that discrepancy. Increasing
\(\alpha\) cannot create competence in a fixed weak continuation oracle; its
usefulness itself is a hypothesis to test.

## 6. Accounting and falsification

Record separately executed collection controls, candidate rollout controls,
memory resimulation, weighted-mean verification, full continuation queries,
qualification, selected-endpoint auditing and evaluation. Count failed and
rejected work too. Raw sample count and memory-candidate budget are identical
between deployment arms, but the guided weighted-mean verification can require
an extra \(H\) simulated controls. Charge that work explicitly; actual model
calls are **not** exactly budget matched. Report additional qualification and
audit overhead separately. This pilot is **not** an end-to-end sample efficiency
proof, even if guided execution improves.

The primary question is whether accumulated, grounded long-horizon predictions
improve the actual independent planner beyond a strong evolving memory
control, and whether that improvement changes subsequent useful acquisition.
Report final finite cost, survival, time alive and speed MAE on common fresh
resets, plus acquisition outcomes and the complete ledger. A conventional
walking gate is at least 90% survival and speed MAE at most .30 m/s; satisfying
it in one training seed is not training-seed robustness.

The mechanism fails this pilot if guidance never qualifies, predicts well only
on irrelevant starts, does not improve final controller behavior over memory,
or consumes enough qualification/query work to erase any proposed efficiency
gain. Retain those results without expanding the frozen budget. Walking
success alone does not establish exploration gains, novelty or sample
efficiency. No standalone actor result can be inferred from this trial.
