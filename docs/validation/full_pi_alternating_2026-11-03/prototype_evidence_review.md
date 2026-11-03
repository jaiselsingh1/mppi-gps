# Full-length PI prototype: evidence and reuse review

Read-only review for the alternating reconstruction experiment. All referenced
files and the exact inspected prototype/log text are bound in
[input_inventory.json](input_inventory.json). No physics, model inference,
fitting or new experiment was performed for this review. Neither reference
branch was edited.

## Evidence status at this observation

The prototype is the **untracked**
`/Users/jaisel/Documents/mppi-gps/.claude/worktrees/sac-continuation/scripts/diagnostics/full_pi.py`
at branch HEAD `d563712`. Its inspected SHA256 is
`350cdce35c43585eb7edcda11bc2fd9a65c542b0b336783362715703929ffea2`.

Local `runs/2026-11-03_full_pi_mpc_001` contains only `start_0.npz`, not a final
summary or within-solve curves. A relevant saved scratchpad log was found at
`/private/tmp/claude-501/-Users-jaisel--claude-spares-spare-9NRwAu/d1be111d-3d91-4afd-aa53-f75f917509e0/scratchpad/full_pi_mpc.log`.
The adjacent `fullpi_launch.txt` says `5090 launched`. These are not a complete
launch receipt: interpreter, thread count, executable source binding, remote
output location, exit status and resource accounting are absent. Read-only
checks of the two known Walker `runs` locations on robopil5090 found no matching
full-PI directories. No process commands or private process inventory were read.

The log grew from two to three cases during review. Its preserved partial rows
are:

| Start / printed planner index | Cumulative seconds | Selected cost / MPC cost | Walks | Selected agreement with MPC | Final-50 current-cost endpoint change |
| --- | ---: | ---: | --- | ---: | ---: |
| 0 / 0 | 287 | 401.28863328981333 / same | true | 1.0 | .00801419 |
| 0 / 1 | 583 | 401.28863328981333 / same | true | 1.0 | .01545149 |
| 1 / 0 | 881 | 378.30951482694223 / same | true | 1.0 | .02151977 |

Both saved `start_0.npz` solutions were independently compared to its `mpc`
array: they are exactly equal, with maximum absolute difference zero. These
cases establish **retention of the incoming walking sequence**, not new walking
discovery or successful improvement. The other cases, final consistency statistic
and preset overall readings are not yet available locally. The final-50 statistic
is only an endpoint difference of the current cost, not a stationarity proof.

## Optimizer that should be reused

The prototype has an actual full-sequence optimizer, rather than repeated MPC:

- `FullPI(MPPI)` keeps the existing noise process, multiplies its draw by a scale,
  and overwrites sample zero's noise with zero. The private generator still draws
  all K samples before that overwrite.
- `optimize` fixes start/warmstart and U shape `(1000,6)`, calls
  `_plan_iteration(start, None, None, warm)` 300 times and never shifts the
  horizon or executes a physical control between refinements. Geometric noise
  scale runs from 1 to .1. `initial_iterations=2` is bypassed here; there are
  exactly 300 private refinement passes, not 301.
- Config is the existing K256, sigma .3, order2 low-pass 23.4375 Hz at125 Hz,
  H1000, with temperature chosen for ESS5% each pass.
- The printed planner index is 0 or1. Actual private reset seed is
  `20000 + 100*start_seed + index`.
- The scored current U is sample zero. The returned U is the best historically
  scored **nominal**, not the best of all K candidates and not the unscored final
  weighted update. A mean update may roll out worse than its samples.

This is task-only: no policy is passed, and `_last_costs[0]` is native C only.
A coupled reuse must add the reconstruction callback to this same whole-sequence
path and retain by **C+alpha D**, re-evaluated under the current fixed policy.
Otherwise adding D to weights while retaining by C would test a different rule
than the declared objective. Preserve current/best population scores and
weighted-nominal progress as separate diagnostics; do not silently change the
selection algorithm because these partial nominal solutions stayed unchanged.

The prototype's annealing, ESS rule and incumbent sample are already part of
its optimizer recipe. They must be shared by alpha0 and nonzero-alpha arms.
They mean this prototype is not an isolated full-horizon versus old-MPC test.
MPC initialization tests refinement/retention; the separate zero initialization
addresses discovery.

## Context, indexing and failure contracts

`sample_bank.py` stores pre-action states `(N,19)`, warmstarts `(N,9)` and exact
float64 executed actions. Sample prefixes are reduced to float16, sample costs/
weights and plans to float32. The bank contains **no complete integration state
or terminal N+1 context**. Its summary records seeds0–4 and6–9 walking, seed5
falling after624 controls, and exact agreement with the earlier default arm.
This is useful recorded behavior, but its context format does not by itself
satisfy the requested complete-context prerequisite.

Existing K1 inputs provide stronger sources without acquiring another teacher:
`h4_r1_independent_00.npz` and `_01.npz` have full float64 pre-action states,
actions, warmstarts, integrations `(1000,91)`, native costs and separately stored
terminal state/warmstart/integration. Their source rows can form N+1 traces by
appending the terminal context. The v2 raw teacher files also have these fields.
Use explicit immutable bindings and native restore/replay checks before accepting
any initializer; no such replay was performed in this read-only review.

Native batch rollouts return **post-action** states. Reconstruction needs
`pre_action_states(start, post_states)`. The source Walker cost includes the
fall-causing control, charges native terminal20 once, then absorbing5 per missing
control. Post-fall model states are computed but ignored by the task score;
standalone traces must stop and retain missing post-fall observations as missing.
A coupled D must not silently gain an advantage from a shorter executed prefix.
Its failure/feasibility treatment must be declared identically for both arms.

The existing weak matching coefficient `.06*sum(error^2)` corresponds to
`alpha=.12` under the proposed `.5*sum(error^2)` D convention. This is an available
historical scale, not evidence that it is useful. With six bounded torques,
per-control D is at most12. Alpha.12 contributes at most1.44 per valid control;
this bound may inform fall accounting without selecting a weight after outcomes.

## Cost evidence and bounds

Each normal solve requests `300*256*1000 = 76,800,000` model controls. The
six-case invocation additionally evaluates one MPC sequence per start and one
selected sequence per case: **460,809,000 model controls**, or1,843,236,000
requested physics substeps at frame-skip4. Both initializations would require two
such invocations. Native `env.step` controls are zero in this prototype; its
outcome forecasts also use `batch_rollout`.

These are **source-derived counts**, not an independently metered paid-work
ledger. The first three logged cases imply approximately287–298 wall seconds
per solve under the unidentified execution allocation, about.96–.99 seconds per
full pass. CPU-hours, GPU work, concurrent load and thread count cannot be
recovered from the saved log. The old sample-bank summary reports417.25 seconds
for its own different short-horizon collection; do not substitute that for
full-length PI timing.

At positive alpha, a straightforward all-row callback adds256,000 policy rows
per pass, before any masking optimization. Both model and policy work must be
reserved and recorded separately. Equal K/iterations/fitting settings do not
make complete compute equal to alpha0. Full sampled state banks are about38.9 MB
per pass in float64, before actions/predictions; the new storage design must
state which raw quantities are retained and which audits they support.

The next experiment should therefore reuse this optimizer, freeze complete
contexts and a matched alpha contrast, meter its actual work, and judge the
alternating loop. These partial task-only results neither test D nor establish
that whole-sequence optimization solved transfer.
