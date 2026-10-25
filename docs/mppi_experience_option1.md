# Option 1: independent MPPI experience and learner-owned outcomes

This is one seed-0 capability trial of the user's original idea: independent
MPPI supplies executed experience and useful starting states; a long-term
learner critic helps propose changes to a standalone policy. Actual complete
learner rollouts decide whether those changes are accepted. There is no
action-cloning loss, SAC actor gradient, entropy objective or teacher rescue.

Entry point: [`train_mppi_experience.py`](../scripts/training/train_mppi_experience.py).
Study: `runs/2026-10-25_option1_mppi_experience_walking_seed0_001/`.
Server checkout: `/home/jaisel/mppi-gps-option1-walking-20261025-seed0-001`.
Status: **complete,200/200; final policy fails walking**:0/100 eight-second
completions,0.89248s mean duration,1.237366m/s speed MAE. See the
[completed result](experiment_evidence/2026-10-25_option1_mppi_experience_walking_seed0/README.md).
Launched2026-10-25 16:21:34 America/New_York, driver PID74540,
immutable commit `fed056ae45440d119e4104154b993626512d9ca3`.
The launch record and generated `protocol.json` identify the exact executed
configuration and source hashes. No further training budget is implied by this result.

## Experience and finite learner value

Collect eight independent teacher episodes once: six training and two held-out
episodes. MPPI uses K32/H48, two first-decision refinements, lambda 3 and the
existing filtered sigma 0.3 noise. It receives physical state, solver warmstart,
configuration and a private seed. It retains its own shifted plan between
decisions; learner actions, critic values and policy tails do not enter it.

Fit twin positive cost networks to actual one-step teacher transitions together
with current-learner transitions. A transition records raw cost `c`, physical
failure and remaining controls `n`, including that transition. For a healthy
transition with `n>1`, the detached target is

`c + gamma * max(Q1_target(s_next, pi_base(s_next), n−1), Q2_target(...))`.

The next action always comes from a frozen copy of the current learner,
including for teacher transitions. Whole teacher returns and recorded next
teacher actions are not learner-return targets. For a fall, use
`c + gamma * 5 * sum(gamma^j, j=0..n−2)`; `c` already includes the falling
transition's terminal charge once. At `n=1` the future is exactly zero.
Healthy finite-episode ends also have zero future. The continuing adapter's
infinite failure tail is unused.

The critic inputs are 17 observations, six actions and remaining/N, with two
128-unit tanh layers and a scaled softplus output. Fit 1,000 updates per search
iteration, batch 128, learning rate 0.0003, target Polyak 0.005. Sample teacher and
policy rows 50:50 when both exist. Training/holdout reset IDs are disjoint.
The reference scale is `5*sum(gamma^j, j=0..N−1)`, not a cost upper bound.
Twin maximum is a bias heuristic, not a guarantee. Log held-out Bellman error
and separate calibration against current-learner complete rollout returns.

## Search, ownership and schedule

N=1,000 controls (eight seconds), gamma 0.999. The reactive actor has 17 inputs,
two 64-unit tanh layers and six bounded outputs. Each iteration uses four fresh
ordinary training resets and four fresh verification resets. Positive-handover
stages additionally restore the six training/two held-out teacher snapshots.
Already fallen teacher episodes remain in averages with their finite absorbing
cost; they do not invent healthy starts or contribute replay transitions.

The fixed handover schedule is **900, 500, 0**, with **40, 40, 120 search
iterations**, respectively: 200 in total. Stage changes are budget changes,
not claims of mastery. At handover `b`, the learner owns every control from
`b` through N. Saved guide states are simulator interventions; they do not
establish that the learner can reach those states itself.

Generate 32 antithetic parameter pairs per iteration, sigma 0.025. Score each
candidate's own first min(32, remaining) actions followed by the frozen
**base-policy** critic tail. This surrogate is not the candidate's observed
complete return. Cost-normalized ES uses learning rate 0.03 and parameter-step
norm at most 0.25. Verify both the synthesized ES policy and the best sampled
whole policy through actual complete learner suffixes on training and
verification panels. Accept only strict improvement on both panels; among
qualifying choices take the lower verification cost.

For `b>0`, the accepted objective is the equal-weight mean of root and guide
costs, each divided by its finite remaining absorbing-cost scale (floored at 1).
Early mixture improvement may worsen ordinary-root performance: this is
deliberate, to allow coordinated preparations instead of imposing a root
nondegradation barrier. Record both components. At **b=0**, starts and
acceptance are root-only. The critic still guides short-prefix proposals;
complete candidate-policy returns remain the acceptance test. Archived teacher
transitions still contribute to critic fitting at b=0. No MPPI or expert-value
continuation executes after any learner handover.

## Evaluation, recovery and interpretation

Evaluate initially, every 20 iterations and finally on the same 100 development
resets **12900000–12900099**, policy alone for 1,000 controls. Report finite
discounted cost, survival count, time alive and speed MAE. The observed walking
gate is at least 90/100 eight-second completions and MAE≤0.30 m/s. This reused
development panel is not a reserved final qualification panel or evidence of
training-seed robustness. Critic loss, accepted mixture updates and stage
advancement cannot substitute for standalone results.

Workers exchange serialized policy bytes and NumPy results, avoiding shared
Torch tensor payloads. Each committed update preserves policy, critic/targets,
optimizer and ledger before appending its metrics row. Resume requires matching
configuration/source hashes and restores the last committed update; unfinished
work may repeat. Recovery events preserve prior status, and evaluation reports
reconcile evaluation counts. Across interruptions, unknown in-flight work makes
the committed control ledger a **lower bound**. Keep teacher planning, teacher
execution, learner execution and standalone evaluations separately visible.

This trial does not compare against a matched root-only control, establish an
exploration/sample-efficiency advantage, or resolve critic support and guide
distribution shift. The interrupted earlier overnight study is preserved as
an interrupted experiment, not a new baseline. Keep existing studies unchanged
and report this trial's actual standalone outcome before drawing conclusions.
