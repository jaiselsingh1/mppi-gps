# Matched-continuation scoring: v2 development protocol

Status: declared before running the seed controls or v2 scoring. The audited
S2/S3 v1 implementation is preserved in commit `fdf58ed`; its original
experiment bundles are read-only. The broader v1 accepted-state study is
paused. This is a development experiment, not active gating or qualification.

## Why change the comparison?

The post-hoc plan-versus-realized analysis finds that MPPI's realized
48-control cost is below its retained open-loop plan cost in 86.1% of the
1,495 usable windows. These comprise 1,448 fully observed windows and 47
windows completed with the prescribed absorbing cost after a fall. Only
1/176 accepted strong-policy trajectories beats realized MPPI cost. Windows
overlap within two episodes; they are not independent qualification trials.

V1 scores a full policy trajectory against an MPPI plan, but its intervention
executes one policy action followed by replanning MPPI. V2 changes the
continuation being scored, not the candidate first action, task or planner.
The analysis motivates this change; it does not exonerate policy actions or
prove the new prediction errors will cancel.

## A. First-action seed sensitivity control

Use exactly the four earlier development decisions:

| Reference | Policy label identifying the saved decision | Decision |
|---|---|---:|
| Success, reset 30 / planner 20030 | Intermediate | 96 |
| Success, reset 30 / planner 20030 | Strong | 10 |
| Failure, reset 31 / planner 20031 | Intermediate | 126 |
| Failure, reset 31 / planner 20031 | Strong | 0 |

At each decision, produce five alternative MPPI first actions using proposal
seeds **31000, 31001, 31002, 31003, 31004**. Restore the saved **pre-call**
nominal and refinement flag; change only the proposal planner RNG. Discard
the proposal planner's resulting nominal and random state.

For the raw action, an identical-action control, and each alternative action:
restore the **original post-call** MPPI snapshot, execute the supplied action
without another planning call, then continue with raw MPPI. Both nominal and
future random stream are identical at the branch boundary. The simulator
warmstart follows the actual reached state in each branch. Run 250 controls
including the first action, stopping at a fall. No policy or recurrent gate
acts in these seed-control continuations.

Required checks: the raw branch matches its saved continuation exactly; the
identical-action control is bit-identical; restoring the original proposal
RNG reproduces the original first action. Save all five seed outcomes, not
only large/small changes. Report first-action L2 distance and differences in
fixed-duration task cost at 48 and 250 controls, completion and speed MAE.
Pad only a fall with the frozen absorbing cost (5 per remaining control),
never a time limit. This is a sensitivity benchmark, not a zero-centered or
action-magnitude-matched statistical null. Five seeds do not support precise
tail-probability estimates.

## B. Fixed v2 scoring rule

For each of the **same two first actions**, compute:

`score(a) = immediate task cost + task cost of 47 successor-plan actions`

1. Start from the same current state and solver warmstart and take `a` in an
   isolated simulator.
2. Clone the same original post-call MPPI nominal/RNG/refinement flag for
   both alternatives. From the respective reached state and reached solver
   warmstart, make one ordinary MPPI planning call with frozen H=48, K=256
   and all other original settings.
3. Explicitly score the **first 47 actions of that optimized plan** using
   the frozen rollout cost. Do not substitute minimum/mean sample costs or
   add 48 successor actions (that would change the total horizon to 49).
4. Keep tau=0 and the no-additional-predicted-fall guard. Invalid candidates
   reject; invalid baselines stop. If the first action falls, its live cost
   already includes the terminal penalty: add only 47 absorbing costs and
   do not plan after termination. Subsequent falls follow the existing
   rollout scorer's terminal/absorbing rules.

Neither scratch branch changes the live planner or consumes its randomness.
The future live controller still keeps its original shifted nominal; it does
not inherit either probe's optimized plan. V2 needs up to two extra planner
calls per comparison plus explicit rescoring. It predicts one future replan
followed by an open-loop tail, **not** 48 future replanning steps.

## C. Evaluate predictions without changing the rule

After A is recorded, score the four known policy actions, all twenty seed
actions and four identical-action controls with the fixed v2 implementation.
The known policy continuations are reused by hash, not rerun or overwritten.
For each action report predicted cost difference, acceptance/fall flags,
actual 48-control difference (horizon-matched), and actual 250-control
difference (persistence). Report sign agreement, false acceptance/rejection,
and raw numeric differences; separate policy and seed-control summaries.
Count a predicted cost tie as a tie, not a correct positive/negative sign.
The fall guard remains a distinct rejection reason. Report both v1 and v2
predictions for the four policy cases without selecting favorable examples.

Required implementation tests: identical-action score equality; live/global
RNG isolation; exact baseline successor replay; actual reached-state solver
warmstarts; H rather than H+1 cost accounting; immediate/later fall penalties
once; invalid baseline/candidate distinctions; and deterministic repetition.

The four policy outcomes were already inspected: this is calibration and
debugging, not held-out validation. Do not tune thresholds/horizon/weights to
make these cases agree. Freeze the code and report before preselecting the
broader v2 counterfactual sample; sample from eligible states independently
of v1 acceptance. New branch outcomes must remain uninspected until the
selection is recorded. No additional policy training is part of A–C.

## Files and evidence

- `scripts/diagnostics/compare_walker_plan_realized.py`: tracked reproducer
  of the post-hoc analysis; original scratch evidence remains untouched.
- `scripts/diagnostics/walker_action_controls.py`: seeded single-action
  controls, using the already tested continuation machinery.
- `src/mppi/gate_v2.py`: isolated, matched-continuation scoring only.
- `scripts/diagnostics/walker_gate_v2_diagnostic.py`: v2 predictions compared
  with saved policy and seed-control continuations.
- Planned outputs: `runs/2026-10-12_s3_action_controls_001/` and
  `runs/2026-10-12_s3_gate_v2_calibration_001/`, with fresh manifests, source
  snapshots, configuration/input hashes and per-case reports.

No evidence supports using v2 as an improved teacher yet. S4's fixed margins
and untouched qualification data are unaffected; S5 remains downstream.
