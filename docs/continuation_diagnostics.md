# Continuation diagnostics: fixed protocol, 2026-10-12

This protocol is written before either new experiment. Starting revision:
`5a05a32`. The two experiments run sequentially: record A completely before
starting B. No new policy training, active gate, qualification, or tuning.
Results go in `docs/experiment_log.md`, not in this fixed protocol.

## Question and limits

Does changing the planner's initial action sequence affect the outcome of
the same first-action intervention? Separately, does the sign of an
intervention's effect persist over different future planner random streams?

V2 compares matched continuation models but still predicts one replan plus
an open-loop tail. Its four correct policy directions are calibration, not
proof that prediction bias disappeared. The twenty seed interventions share
four selected states and a fixed incumbent action; they are not independent
trials or a zero-centered null. A fixed-stream branch measures a conditional
effect exactly; it need not estimate the effect averaged over future streams.

## Frozen inputs

Use all four existing cases, without selection based on the new outcomes:

| Case | Shadow episode | Policy | Decision (zero-based) |
|---|---|---|---:|
| success_intermediate_96 | success | intermediate | 96 |
| success_strong_10 | success | strong | 10 |
| failure_intermediate_126 | failure | intermediate | 126 |
| failure_strong_0 | failure | strong | 0 |

Inputs are the audited shadow bundles under
`runs/2026-10-11_s3_shadow_001/` and all four case bundles under
`runs/2026-10-12_s3_action_controls_001/`. Proposal actions and their discarded
post-call plans are already saved for seeds 31000, 31001, 31002, 31003, 31004.
Do not regenerate or choose replacement proposals. Verify input hashes and
the current frozen runtime sources before simulation. Record input manifest
hashes, exact source copies, package versions and this protocol in each new
run bundle before its first continuation. Preserve all previous evidence.

MPPI remains H=48, K=256 with the frozen cost, sampling, refinement, physics
and timing. No policy parameters, scoring thresholds, fall guards or costs
change. Each continuation includes the supplied first action followed by raw
MPPI, for 250 controls (2 seconds) or until a fall.

## A. Planner starting-plan ablation

For each of the twenty saved alternative actions, compare:

1. The archived continuation retaining the original shifted nominal `U0`.
2. A new continuation using the proposal's own shifted nominal `Uj`.

Copy **only `U`** into the original post-call snapshot. Preserve the original
post-call RNG and `initial_plan=False`. Do not restore the proposal snapshot
wholesale. Both branches start at the same simulator state and solver
warmstart; the solver warmstart after the identical first action must also
be identical. Subsequent simulator warmstarts evolve naturally.

Run one original raw continuation per case as an exact replay guard. Check
the proposal's first action and shifted nominal against its saved optimized
plan. Check that the new and archived alternative branches have the same
first physical transition/cost and the same future RNG sequence through
their shared executed prefix. Stop and report a failed audit on mismatch.

Primary outcome: matched-U minus original-U fixed-duration task cost over
48 controls, for each identical action. Negative means changing the nominal
helped. Secondary: the same difference over 250 controls, falls, and each
arm's difference from the original raw MPPI continuation. Report every pair,
counts and min/median/max by state; pooled descriptives are not independent
statistical evidence. Four raw replay guards plus twenty new arms =24 runs.

This does not isolate action-plan compatibility from a generically better
nominal (there is no crossed original-action/proposal-nominal arm). A benefit
does not establish that a policy's plan should be handed over. No benefit
does not establish that substitution is inherently costly. Do not rescore
v2 under a different nominal or reuse its original-U scores as predictions
for the new matched-U execution.

Output: `runs/2026-10-12_s3_nominal_ablation_001/<case>/`.

## B. Paired future-randomness sensitivity

After A is completed and recorded, run B with **original-U execution for all
arms regardless of A's outcome**. This convention is chosen now to diagnose
the outcome stability of the existing v2 execution model, not to select the
best convention after seeing A. A possible matched-U gate is a separate
future experiment; there is no policy-plan handoff here.

At every fixed state use future seeds
`41000, 41001, 41002, 41003, 41004, 41005, 41006, 41007`.
Replace only the original post-call RNG with the state of NumPy's
`default_rng(seed)`. Preserve `U0` and the post-call refinement flag. First
actions are fixed before these draws and never regenerated using them.

For each state/seed run raw MPPI, the original policy first action, and all
five saved alternate MPPI first actions. Reuse that state's raw continuation
for its six paired comparisons. Within a pair the same future random stream
is used; across future seeds it changes. The same seed list is also used
across states, so do not treat pooled rows as independent trials.

Four states x eight streams x seven arms =224 continuations, with no optional
stopping or removal of unfavorable actions. Tests must cover same-action
identity, snapshot isolation, common RNG prefixes including unequal stopping
times, and cost accounting after falls. Production checks must verify paired
initial snapshots and future RNG prefixes. A failed check stops that run.

For each fixed state/action report all eight paired cost differences at 48
controls (primary) and 250 controls (persistence), their mean, sample standard
deviation, minimum/maximum and positive/negative/tie counts. Also report each
arm's falls and paired additional/prevented falls. Report raw-only variation
separately. Speed MAE remains an executed-prefix descriptive measure and is
not a comparable fixed-duration survival metric.

Secondary descriptive analysis may compare the already fixed original v2
prediction and acceptance with the eight-stream mean and sign stability.
Do not recompute scores using these outcome streams or optimize decisions
against them. They are separate from the original scoring stream. This is
not held-out-state validation: the states and original scores are known.

Output: `runs/2026-10-12_s3_future_seed_check_001/`.

## Accounting, decision and reporting

Both horizons include the first action. Apply the existing live first-fall
cost once; pad remaining controls after a fall with the frozen absorbing
cost of 5 per control. Never pad a non-fall truncation as failure. Use exact
zero as the cost-difference tie, with no post-hoc epsilon or outlier removal.

Record results and audits for A before B starts, and all results for B before
proposing another variant. No significance test, population confidence
interval, non-inferiority claim or gate promotion follows from these four
states/eight streams. Report failure distributions, not only mean costs.
Do not claim either diagnostic uniquely identifies the source of score
error. Keep the broader active-gate study paused. The next decision is a
written interpretation of which hypothesis remains supported and what one
additional test would discriminate it; no new optimization begins here.

Implementation tests may use small synthetic configurations. Production
sources/protocol are snapshotted and hashed; fixes after a production start
must be documented, never silently overwrite a completed evidence bundle.
