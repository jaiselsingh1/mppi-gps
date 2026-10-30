# Frozen ending actor comparison, version 1

This is one bounded comparison of two global action-policy learners. It is not
an alternating GPS result, a general exploration benchmark, or a claim of a new
RL algorithm. All prior runs remain unchanged. The independent-search value
pilot and replanning-value repair are separate diagnostics.

## Task and qualification, before actor training

State is `(clock,y)`, action is continuous `u in [-1,1]`, and `y in [-1,1]`
is continuous. Starting clock is zero and y is uniform on [.2,.4]. For clocks
0 through 11, `y'=clip(y+.2u,-1,1)`. After that y is fixed. Clock increases to
25 and then is absorbing. This is a stylized irreversible lane choice, not a
locomotion or docking benchmark. Local competence means reaching y>=.8 at the
gate. Ending success means the opposite lane, y<=-.1 at clock 25. Thus local
competence is an explicitly tempting subtask, not a necessary component of the
desired ending. The study tests foresight that overrides a locally rewarded
choice; it does not establish preservation of a useful locomotion skill.

Before the gate, cost is `.02u²+.1(1-y')`. A successful lane pays a .2 detour
on the first five post-gate transitions. A bad ending pays 20 on the transition
from clock 24 to 25. Every controller sees exactly this same known model and
objective. There is no terminal heuristic, velocity, hidden task progress,
latent solver state or stochastic transition. The clock is part of the state.

The short planner horizon is four controls. Because delay 12 exceeds that
horizon, the bad-ending penalty is absent from every controllable short-window
objective. This is a structural horizon limitation, not a withheld reward.
For each pre-gate window an analytic lower bound sets each reachable y to its
maximum and drops all other nonnegative costs. The independent +1 plan has at
most .02 per-control gap to that bound from the positive qualification states.
Measured selected-plan gaps check the actual search quality. Gaps are evaluated
only before the gate, never at unavoidable late penalties.

For each of three seeds, 12 predeclared roots must satisfy ALL numerical gates:
short MPPI local rate>=.9, short ending rate<=.1, mean local bound gap/control
<=.04; long independent MPPI ending rate>=.9; executable actor capacity witness
ending rate>=.9 from original roots AND clock-3 states actually reached by the
short planner; short-minus-long mean complete cost>=5. The bounded reference
uses the identical planner with horizon 25. The actor witness has parameters
`[atanh(-.9),0,0]` and is used only for qualification, never initialization,
labels or selection. If any gate fails, stop the entire study and report it.
No retuning, additional seeds, budget extension or replacement task is allowed.

## Independent acquisition and planner memory

MPPI uses 32 independently generated Gaussian proposals around its own retained
nominal plan, sigma .8 and temperature .2. Six proposal slots are its nominal,
zero, +1, -1, +.5 and -.5 plans. It also evaluates the weighted mean, then
executes the first action of the lowest measured cost plan among these 33
plans. It retains that selected plan shifted by one control. This is a declared
verified-plan safeguard on MPPI, shared by both horizons. The actor and critic
never enter its proposals, scores or memory. Its private PCG64 state, nominal
before/after, every proposal, costs and selection are saved. No trajectory
retrieval supplies actor actions.

After qualification, acquire four short and four long episodes per seed. First
three of each horizon form training reset pools; last one is held out. Both
actor arms get the identical immutable acquisition data. Experience discovery
here comes from independent search including a bounded longer-horizon reference.
The short planner alone is not claimed to discover a successful ending.

## Two matched actor learners

The standalone actor is `tanh(theta0+theta1*y+theta2*clock/25)`, initialized at
zero in both arms. There are 24 updates and eight common antithetic Gaussian
parameter directions per update (sigma .3). Direction seeds, actor family,
fixed-scale ES gradient (scale 20), learning rate .4, update radius .6, training
starts, validation starts and acceptance criteria are identical across arms.
Candidates outside the radius can be measured but cannot be promoted. Propose
the bounded ES step and the best eligible measured perturbation. Both are
accepted only if their ACTUAL complete candidate-actor mean cost improves by
more than 1e-6 on BOTH training and held-out validation contexts, with no decrease
in ending-success rate on either set. Choose the eligible proposal with lowest
validation cost. Rejected proposals, not just accepted actors, remain recorded.
This outcome verification also provides the operational trust gate: scores can
propose, but only measured full-policy outcomes authorize an actor update.

`full_return` scores every perturbed actor by its own complete remaining
execution. No predicted, expert, nearest-neighbor or base-actor tail ranks it.
`critic_prefix` scores its first three actions plus a frozen estimate of the
CURRENT BASE ACTOR's complete remaining cost. It is a hybrid surrogate, not an
estimate of the candidate actor's complete cost. An unsupported endpoint uses
an explicitly charged exact frozen-base tail, never an exact candidate tail.

Each critic is newly fitted once per base actor. Targets are measured full
base-actor continuations from four shared curriculum starts plus four training
probe contexts. Their reverse sums supply undiscounted finite-horizon MC labels;
there is no TD, optimal-value target or planner-action regression. Features
include scaled clock/y, known gate/bad-lane indicators and 48 farthest-point
RBF centers, width .25, ridge .01, root-balanced weights. These task-aware
features are declared priors. The support radius is .35. All models are saved
before held-out checking or actor acceptance, including models whose proposals
are rejected. Four disjoint held-out contexts test complete base-value targets.
Report pre-gate errors separately from easy post-gate prediction.

For updates 0–7 the four shared training starts are two fresh ordinary roots and
two acquired states at clock 6 (one short, one long); updates 8–15 use clock 3;
updates 16–23 use four ordinary roots. Validation uses fresh roots and held-out
acquisition episodes with the same schedule. Critic probes alternate clocks 6
and 3, independent of outcomes. This fixed reverse-time curriculum requires
simulator state restoration and discovered reset coverage. Reset ancestry is
preserved. Curriculum success cannot replace original-root actor evaluation.

## Distinguish approximation from continuation mismatch

Every fourth update on two fresh diagnostic roots, inspect four candidates:
antithetic pair zero and the pair with largest predicted pair difference among
the remaining pairs. Record C and the frozen model BEFORE querying A and B.
A is full candidate execution; B is candidate prefix followed by exact base
execution; C is prefix plus predicted base value (or the declared fallback).
C-B is approximation error; B-A is continuation mismatch. Report absolute and
paired-difference errors and gradient alignment. A/B diagnostic data are never
added to critic fitting, parameter scores, acceptance or future resets. A
near-zero base can make B bad while A succeeds: an accurate value need not fix
this surrogate bias. Both learning arms still accept with complete rollouts.
At these same checkpoints, also measure B/C for both actual proposed updates
on the two held-out ordinary validation roots, reuse their already measured A,
and compare to the base actor. These post-decision diagnostic results cannot
change acceptance. This checks the critic at proposed changes rather than only
at arbitrary suffix rows or randomly selected perturbations.

## Budget, evidence and verdict

Seeds are [0,1,2]; each seed/arm has at most 1,000,000 paid controls, including
the entire seed's shared qualification/acquisition and shared development work.
Every main transition or bound check reserves one additional audit check. An
update begins only with a conservative reserve of 16,000 further main controls;
budget exhaustion stops work. Actual unique study cost and the standalone cost
of each arm (which each includes the shared data) are both reported. Fit calls,
prediction rows and wall time are additional reported work, not converted to
physics controls. Failed/rejected queries, mean-plan verification, full-policy
acceptance, diagnostics, evaluations and audit slots are all charged. No padding
is free. Core development tests consumed 211 physics +7 arithmetic controls;
learning and driver tests consume zero physics. All development raw records are
retained. The original seven development bound steps retained their count and
test assertions but not separate raw bound outputs; this small archival gap
must remain explicit. They are charged again in the audit. One terminal-plan
query was deliberately rejected before physics in a boundary test (zero controls).
Ledger and NPZs preserve complete states, actions, cost components,
ending/termination flags, cutoff, remaining clock, actor hashes and ancestry.

Evaluate actor alone initially/every four updates on 16 fixed development roots;
evaluate the final actor on 32 previously unused roots per seed/arm. A learned
ending requires >=.9 final success in EACH seed and mean cost at least five
below qualified short MPPI. Primary paired cost comparison uses the last
available development checkpoint at the largest common paid budget available
to every seed/arm, without interpolating unknown outcomes. Critic help requires
both >=.1 success advantage and >=1 cost improvement at that point, with a
paired-seed bootstrap 95% interval excluding zero in the beneficial direction;
32-root final results are also reported. Three seeds only support a small
mechanism test; root repetitions do not become independent training seeds.

Save source/configuration hashes before qualification. A separate auditor
recomputes physics, returns, provenance, score/acceptance algebra and costs from
raw evidence. It does not share the training scorer. Plots use charged controls
and distinguish discovery, coverage/representation, prediction, continuation,
optimization and final control failures. Retain a negative result. Prior art
and the missing global-to-local GPS coupling are documented separately.

Runnable commands (a fresh directory is mandatory):

```sh
.venv/bin/python scripts/training/run_ending_actor_comparison.py --run-dir runs/NEW_NAME
.venv/bin/python scripts/diagnostics/audit_ending_actor.py --run-dir runs/NEW_NAME
.venv/bin/python scripts/diagnostics/report_ending_actor.py --run-dir runs/NEW_NAME
```
