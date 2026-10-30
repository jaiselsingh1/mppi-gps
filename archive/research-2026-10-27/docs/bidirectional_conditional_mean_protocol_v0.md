# Conditional mean-controller feedback: prospective development test

This is a new, finite mechanism test on the cheap Ending environment. It does not
change the negative verdict of geometry001, which required an advantage for both
the deterministic mean and stochastic controller. This test evaluates the actual
deterministic mean controller intended for deployment. Standard deviation and the
actor's private sampling stream remain unchanged. No stochastic-controller,
physical Acrobot, three-seed comparison, novelty or efficiency result follows.

## Question and raw starting point

Can a competent action policy improve executed search beyond a strong full-episode
memory control, and can the states acquired by that improved search help train a
better next standalone actor, twice?

The common parent is the already selected ordinary-data actor from independently
audited geometry001, not its long-data actor. Parent policy hash is
`0c6d18b9e164de08de998a75546a179bc614e0969c053981d487f67e1bb79ca8`;
checkpoint SHA256 is
`4d89b881a18075921c8d4560e7624f52cfd5fbdc0a6c0bd06acd4077c801a223`.
The original audit SHA256 is
`9f486c1f1e00889c680a2b4acc8e28ad042ef7e8d7ab1a218e816b8b248a9882`.
Inputs include the eight actual successful long-independent-search episodes
already acquired in the v5 ancestry. Copy and hash the raw traces, checkpoint,
audit, parent provenance and ledgers before execution. Reused costs stay charged.

The task remains gate12, delay12, bounded actions, horizon25, ending y<=-.1,
effort.02, greed.2, detour.2 for five steps, move.2 and capture-failure20. There
is no reward or reset change. Fresh roots use namespace701000--731063, distinct
from earlier Ending panels. The exact root roster is in the frozen JSON.

## Independent control and policy influence

The private independent bank retains H4, K32 Gaussian, eight smooth and eight
fragment-memory candidates, plus separately simulated MPPI mean verification.
Neither actor nor selector may mutate its nominal, memory or RNG. At each actual
state, RAW also retrieves the nearest eight stored full-episode suffixes at that
exact elapsed time and resimulates them through the true terminal. It tests four
deduplicated first actions (private champion, minimum, median, maximum), pays for
their next states, retrieves a nearest suffix from those POST-action states and
simulates the remaining controls. RAW selects by complete native task cost and
has no neural access. Thus success cannot be attributed just to comparing a
learned full continuation against a deliberately forgetful short-horizon control.

GUIDE preserves that whole RAW result. In tested support it first pays for the
current actor's complete model suffix and requires its ending to succeed. This
is an eligibility guard, not actual outcome evidence. It may then additionally
evaluate a fixed pool of first actions followed by the frozen deterministic actor
through the true terminal. These are owned Q^pi counterfactuals, not measured
returns of future MPPI replanning. RAW wins exact cost ties. Only the selected
first action executes; planning repeats at the next state. Fresh complete actual
replanning episodes, not hypothetical tail improvements, qualify deployment.

Every policy version starts with zero influence. Sixteen newly executed own
suffixes at t0 and t24 from eight actual current-parent trajectories can establish
trial eligibility. Their total is208 controls, not16 full episodes. Observation
radius3, unit scaling and time interval0--24 are fixed. This support rule assumes
local transfer of competence; it does not prove behavior on all intermediate or
perturbed states. Actual full-controller qualification is still required.

On16 fresh paired ordinary roots GUIDE must improve mean complete cost by>=.01
against BOTH RAW and actor alone, improve at least75% of pairs by>=.01, maintain
aggregate capture count and capture>=90%. Only then promote the tested setting.
A separate16-root panel executes the actual promoted controller and must pass
the same comparisons. No second promotion relabels the confirmation controller.
Failure revokes influence and stops this run. A changed actor always needs new
suffix tests and both panels. A changed memory is an explicit controller change.

## Learning and two feedback rounds

After initial qualification, acquire eight complete RAW, GUIDE and parent-actor
episodes on paired fresh roots. The same teacher improvement criteria must hold.
Store all actual traces, including failures, separately from candidate plans.

For matched child learning, teacher information consists ONLY of pre-gate
states0--11. No teacher actions are labels, no acquired-state returns enter the
objective, and there is no critic. One paid288-row parent forward obtains hidden
features from96 ordinary,96 RAW and96 GUIDE state rows. Ordinary rows reuse eight
already executed parent confirmation episodes. Both children use192 rows:
ordinary+RAW or ordinary+GUIDE. The same fixed four random head directions
(PCG seed1701) are transformed using each empirical geometry. Frozen trunk,
bounded head axes and bias control coherent policy perturbations; neither student
averages planner actions.

Each child runs three identical finite whole-policy search steps: center and
plus/minus each of five mean-head coordinates,11 candidates, eight complete
ordinary-start episodes per candidate. Coordinates are absolute from the common
parent, L1<=3; standard deviation is fixed. Select by the proposed actor's OWN
actual complete deterministic return with aggregate capture nonregression versus
the freshly evaluated center. Complete outcomes assign credit to the entire
coherent parameter proposal; no return is relabeled as another actor's return.
Both arms share resets, update work, basis and parent. Teacher states affect the
proposal geometry, not the return target. This is critic-free return optimization,
not a claim of policy-gradient or imitation learning.

On a fresh16-root actor-only panel, the GUIDE-data child must beat both its parent
and RAW-data child by mean complete cost>=.001, with capture nonregression and
capture>=90%. If it passes, test its own suffixes and then new controller
qualification/confirmation. Only newly acquired successful RAW episodes may enter
the shared independent memory. The old guiding actor is ALSO tested under this
same updated memory, so a better archive alone cannot explain an accepted new
teacher. The new GUIDE must additionally beat OLDGUIDE by the teacher criteria.

Repeat exactly twice. Only after both beneficial rounds, execute the untouched
64-root final panel: RAW, OLDGUIDE, GUIDE and the last parent, RAW-data child and
GUIDE-data child actors. Assess once; final data never choose an update. Require
the same teacher and actor advantages. The result remains a conditional one-task,
one-lineage mechanism result even if every gate passes.

Forced revocation, stale actor/task and unsupported-state checks replay a saved
immutable RAW bank decision using pure arithmetic. They add no model/NN/live
calls and establish fallback selection only, not new executed degradation tests.
Do not spend reserved controls on unlisted probes.

## Predeclared expenditure and stopping

Maximum RAW model work per root is8506:49*sum(min(4,R)) bank controls plus
12*sum(R) full-memory controls for R1--25. GUIDE adds at most6*sum(R)=1950,
giving10456 model controls and<=1850 neural rows. Every role adds25 live controls.
Independent memory deduplication and unsupported guidance can reduce actual work.

The complete two-round schedule is bounded by4,725,104 scalar controls:
4,675,680 model and49,424 actual controls;600,400 neural rows;132 learner proposal
versions;12 matched search iterations; one basis QR; four geometry SVDs; zero
Adam steps or backward passes. Separate model/live/NN caps are enforced before
calls, and all failed/rejected calls remain in the ledgers. Previously paid parent
discovery, geometry learning, verification and audits remain charged separately.
Record body and launch wall times, hardware, saved evidence size, candidate
actions/scores and complete raw executed traces. No efficiency claim is planned.

Stop at the FIRST failed scientific gate, numerical/query error, missing evidence,
changed source or exhausted cap. There are no retries, retuned gains, changed
criteria or recycled reserves within this run. A negative may motivate another
explicitly versioned hypothesis; it is never overwritten. Independent raw-evidence
audit is required before a final scientific verdict. The frozen source/config,
not this prose alone, specifies executable details.

Command, after source review and accountable pure tests:

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_ending_conditional_feedback \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --config configs/bidirectional_ending_conditional_mean_v0.json \
  --out runs/2026-10-27_bidirectional_ending_conditional_001
```
