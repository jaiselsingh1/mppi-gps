# Credit test stack: can a learner supply the ending beyond MPPI's horizon?

Preregistered 2026-10-26, before any measurement on reserved or training seeds.
Code: `src/envs/delayed_fork.py`, `src/envs/mountain_car.py`, `src/credit/`,
`scripts/credit/run_credit_stack.py`; frozen settings `configs/credit_testbed.json`;
tests `tests/test_credit_testbed.py`.

## Question

The claim under test: a global learner can learn the part of a task's cost that
lies beyond MPPI's planning horizon (the "ending") from complete outcomes,
while MPPI stays independent and no action is copied, and this closes the
gap between MPPI alone and MPPI with the exact value. The test stack separates
the links the claim needs:

- **(a)** do MPPI's own options at a state lead to different complete outcomes?
- **(b)** does an estimator rank those options like the ground truth?
- **(c)** does choosing by it improve the executed closed loop?
- and how many complete outcomes each estimator needs at each delay.

## Independence of MPPI

MPPI is the unchanged `src/mppi/mppi.py` (a test checks identity with the
repository class). A learner may act only by:

- **select:** MPPI plans exactly as alone; the learner scores MPPI's own options
  (its K = 64 sampled sequences and its weighted-mean plan). The first action of
  the best-scored option is executed. MPPI keeps its own carried plan and random
  stream (tested).
- **terminal:** the learner's value is added to MPPI's own cost at the horizon
  end, replacing MPPI-alone's terminal cost.

Nothing from a learner enters MPPI's initialization, nominal or sampling. No
learner regresses onto MPPI's actions.

## Tasks and ground truth

**Delayed fork** (`DelayedFork`, the delayed-consequence toy). State (t, y), with y
on a lattice of spacing .05 in [-1, 1]. Before the gate at t = 30, the action
u ∈ [-1, 1] moves y by .25u per step, at cost .1u² + .1(1 - y'). The second term
is a greedy pull toward lane A (y ≥ 0). At the gate y freezes and its sign is
the lane:

- Lane B pays a visible detour of .5 per step for 10 steps.
- Lane A pays a penalty of 20 at step 30 + L.

The delay past the horizon is d = L - (H - 2): the number of steps by which the
penalty falls after the last costed step of the last pre-gate decision's horizon.
Delays are d ∈ {-10 (visible control), 1, 20, 60, 140} with H = 20, giving
episodes of 39 to 189 steps.

Exogenous cost noise σ_r ∈ {0, 2} per step comes from a recorded per-episode
stream. MPPI's model does not include it, and it does not affect states. The
exact optimal value is backward induction over lattice moves with continuous
actions (the cheapest action reaching a cell is the point of its interval
nearest 0). It is exact: the DP policy attains V* to 1e-15 from every tested
start, and no sampled action sequence beats it.

**MountainCar** (`MountainCar`): Gymnasium MountainCarContinuous-v0 dynamics
(tested step by step), with engine power ∈ {.004 (visible control), .0015,
.001, .0007}. Each step before the goal costs 1 + .1u², and episodes are capped
at 500 steps. MPPI alone uses the admissible terminal heuristic
(0.45 - p)₊ / 0.07, a lower bound on the remaining time. The value is solved by
value iteration on a 301² grid with 21 actions and bilinear interpolation
(grid-converged rather than exact). Its accuracy is reported by simulating its
greedy policy from the reserved middle start. Development found the greedy
policy within 1.5% of the grid value at every power.

**Ground truth for an option** (commit-H): Q*_H = model running cost of the
option over H steps + V*(its horizon end). Counterfactual ground truth executes
the option for H steps from the recorded decision, while MPPI keeps planning
underneath with its recorded snapshot. The given continuation controller then
continues under the recorded planner stream and noise. A branch that executes
MPPI's own action reproduces the recorded suffix bitwise (tested).

MPPI: K = 64, λ = .3, σ = 1, fork H = 20, MountainCar H = 30.

## Data, estimators, deployment

A **unit** is one closed-loop episode of the data controller from a training
start, plus 3 branches. Each branch is at a decision drawn uniformly from the
episode and commits MPPI's sample whose horizon end is farthest from the executed
option's (a value-free diversity rule). Every episode and every branch is one
**complete outcome**.

- Fork training starts: y0 ∈ {.1, .2, .3, .4, .5}.
- MountainCar training starts: uniform over all non-goal states (exploring
  starts, an explicit coverage assumption).
- Round 1 uses MPPI alone as the data controller.

Estimators share one tile coding plus an unpenalized intercept, so unvisited
states fall back to the data mean, never to an optimistic zero:

- fork: 8 tilings of 5-step × .25 tiles;
- MountainCar: 8 tilings of 16 × 16 tiles.

Ridge is 1. The estimators differ only in their supervision:

| Estimator | Learns from | Scores an option by |
|---|---|---|
| MC | every-visit regression of realized cost-to-go | model cost + V(horizon end) |
| TD(.9) (GAE critic target), TD(0) | LSTD(λ), per-step costs + its own bootstrap | model cost + V(horizon end) |
| Paired option scorer | only outcome differences between the executed continuation and a branch from the same recorded state, shared randomness | model cost + relative V(horizon end) |
| Return decomposition (RRD-style) | only each complete episode's total; per-step credits fitted to sum to it | summed in-horizon credit, no terminal value |

In Markov tasks, RUDDER-style decomposition with a state-based return
predictor reduces to MC value shaping. So the decomposition arm uses totals
only, which is where that family differs. Value estimators are deployed in
both modes; return decomposition in select mode only.

**Gap reference.** "MPPI with the exact value" means MPPI with the exact
terminal value (exact_term), in both modes. In development, select mode with
the exact value was weak: fork lane B in 3 of 12 episodes. V* prices the
optimal continuation, a last-moment drop that MPPI's own samples rarely contain.
It is reported as a condition, not used as the reference. Gap share =
(J_alone - J_c) / (J_alone - J_exact_term), where J is the mean noise-free cost
of 40 reserved episodes paired across controllers by start, planner seed and
noise stream. It is computed only where J_alone - J_exact_term ≥ 5 (fork) or
≥ 50 (MountainCar).

Reserved starts: fork y0 ∈ {.15, .25, .35, .45}, never used in training;
MountainCar p ~ U[-.6, -.4], v = 0. Seed blocks: reserved planner 9e8 + i,
reserved noise 9.1e8 + i, training 1e8 + 1e5·seed + 1e3·round + unit.

## Measurements, predictions and falsifiers

**V0 — testbed validity** (reserved, 40 episodes per configuration).

- *Fork d = -10:* MPPI alone takes lane B in ≥ 90% of episodes, and
  J_alone - J_exact_term < 1.
- *Fork d ≥ 1:* MPPI alone takes lane B in ≤ 10%; exact_term in ≥ 90%; the gap
  is ≥ 5.
- *MountainCar .004:* MPPI alone succeeds in ≥ 90%, with J_alone ≤ 1.25·J_exact_term.
- *MountainCar .0015/.001/.0007:* exact_term succeeds in ≥ 90%;
  J_alone - J_exact_term ≥ 125 (a quarter of the cap); MPPI alone succeeds in ≤ 50%.
- *Falsifier:* any violation removes that configuration from the
  interpretation of (b) and (c); its results are still reported.

**R0 — representation.** Projecting V* onto the features (least squares) and
deploying it as a terminal value closes ≥ 85% of the gap at every
past-horizon configuration. Otherwise estimator failures there are confounded
with representation.

**M-a — do MPPI's own options lead to different complete outcomes?** At MPPI
alone's reserved decisions:

- **a1 (exact).** No option differs post-gate (exactly 0 headroom). At least
  30% of window decisions (gate - H ≤ t < gate, whose horizon end reaches the
  gate) have exact headroom Q*_H(mean) - min Q*_H ≥ 5.
- **a2 (counterfactual, MPPI alone continues).** At t = 5 (horizon end 5 steps
  before the gate), MPPI alone returns every branch to lane A: |Δ outcome vs
  mean option| < 1 in ≥ 90% of branches. At t ∈ {11, 15, 19}, the exact-best
  option's counterfactual outcome is ≥ 5 lower than the mean option's in
  ≥ 60% of decisions. At t = 35, all differences are 0.
- **a2 (MountainCar, .001 and .0007).** MPPI alone never reaches the goal, so
  |Δ| < 10 in ≥ 90% of branches, while the median exact headroom is ≥ 5.
  Options differ only under a continuation that would use them.
- **a3 (machinery).** Fork counterfactual differences are identical at σ_r = 0
  and 2: the noise moves no state and is shared. Under exact_term's own
  continuation at its own decisions, counterfactual differences track Q*_H
  differences (Pearson ≥ .9 fork, ≥ .8 MountainCar).
- *Falsifier:* window headroom ≥ 5 at < 30% of decisions would mean MPPI's
  options rarely contain the better ending, so neither selection nor branching
  could teach it.

**M-b — decision-level regret and rank correlation against exact Q*_H.**
Metrics are on the bank of MPPI-alone and exact_term reserved decisions with
differing Q*_H:

- normalized regret Σ(Q*_H(pick) - min Q*_H) / Σ(Q*_H(mean) - min Q*_H), where
  1 means no better than MPPI's own choice;
- mean Spearman ρ between scores and Q*_H.

N_b = smallest outcome count at which the median over data seeds (5 fork,
3 MountainCar) has normalized regret ≤ .2 at that and every larger budget.

- **b1.** Fork, σ_r = 2: MC's N_b does not decrease with d over {1, 20, 60, 140},
  and N_b(d = 140) ≥ 4·N_b(d = 1) (or is beyond the budget). At σ_r = 0,
  N_b(140) ≤ 4·N_b(1), which is coverage dilution only, since branches land in
  the window less often in longer episodes.
- **b2.** Fork, σ_r = 2, d ∈ {60, 140}: TD(.9)'s N_b ≤ MC's.
- **b3.** Paired scorer: N_b at σ_r = 2 ≤ 2× its N_b at σ_r = 0 for every d,
  because shared randomness cancels the noise. At σ_r = 2, d ≥ 60, its N_b ≤ MC's.
- **b4.** Return decomposition: median normalized regret > .5 at 1024 outcomes
  for d ≥ 60. Totals cannot say when credit is due, so it is spread over the
  delay and diluted inside the horizon.
- **b5.** MountainCar round 1, .001 and .0007: every estimator's normalized
  regret is ≥ .4 at 512 outcomes. MPPI alone's outcomes carry no ending from
  the bottom states.

**M-c — closed-loop gain.**

- **c1.** Across all valid round-1 cells (configuration × estimator × N × seed),
  Spearman(bank normalized regret, select-mode gap share) ≤ -.6, so decision-level
  metrics predict closed-loop gains.
- **c2.** Fork terminal mode at 1024 outcomes: MC and TD(.9) median gap share
  ≥ .8 at every d ≥ 1 with σ_r = 0, and at d ∈ {1, 20} with σ_r = 2.
- **c3.** Fork select mode, σ_r = 0, 1024 outcomes: MC's gap share ≥ exact_sel's
  + .2 at every d ≥ 1. MPPI alone's value is pessimistic about waiting, which
  suits a chooser limited to MPPI's samples.
- **c4.** MountainCar round 1 at 512 outcomes: every estimator's median gap
  share ≤ .3 (both modes) at .001 and .0007. One improvement step on MPPI
  alone's value cannot bridge these delays.
- **c5.** MountainCar approximate policy iteration: each estimator's own
  controller generates the next round, 64 units per round, 4 rounds. Terminal
  MC or TD(.9) reaches median gap share ≥ .8 within 4 rounds at .0015 and
  .001. Rounds needed do not decrease with delay.

**Sample efficiency (the mechanism).** N_c = smallest outcome count at which
the median gap share is ≥ .8 at that and every larger budget, per estimator,
mode and configuration. MountainCar budgets include API rounds as cumulative
outcomes. The report tabulates N_b and N_c against delay and noise, with
every episode and branch counted.

## Pass criteria for moving to Walker

1. V0 and R0 hold at ≥ 3 of 4 past-horizon delays on each task.
2. On each task, some learned estimator reaches N_c within budget at every valid
   past-horizon delay (fork ≤ 1024 outcomes in round 1; MountainCar ≤ 1024
   cumulative).
3. c1 holds.
4. a3 holds.

If all pass, the Walker stage is preregistered separately. It will claim
results only on reserved start states and on a second task. Otherwise the
report names the failing link and Walker does not start.

## Development history (declared)

All on dev seeds disjoint from the blocks above.

- MPPI's λ = .3, σ = 1 were chosen from three fork starts: λ = .1, σ = .5 failed
  even at the visible delay.
- MountainCar sparse cost failed at the visible power because samples never
  reached the goal, so the heuristic terminal cost was added.
- Power .0005 was dropped because exact_term missed the goal within 500 steps.
- σ = 2 and K = 128 did not improve exact_term.
- Features were chosen by the R0 projection on 12 dev episodes.
- A tiny driver smoke run (2 configurations, ≤ 4 units) checked crashes and
  timing; its metrics were not inspected.

No estimator was fitted to data before this preregistration.

## Not claimed

These are mechanism tests on small tasks. They are not a standalone-policy
result, a Walker result, or an exploration claim. MountainCar's exploring
starts are an explicit coverage assumption, absent on Walker.
