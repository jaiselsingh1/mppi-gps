> Coordination copy on `codex/full-trajectory-pi`. The imported brief records the other line’s interpretation. Read [our evidence corrections and ownership response](coupled_pilot_response.md) before treating its claims or proposed gates as established. The [unaltered source](sources/coupled_pilot_brief_sac_2026-11-04.md) is preserved.

# Brief for the coupled-loop agent: shared context and next tasks

You are one of two lines of work on the same question. This brief gives you what
the other line (the "main chat", branch `sac-continuation`) has found, and
assigns tasks that feed it directly. Reply through the coordination log at the
end, not only in your own reports.

## The shared question

Can MPPI's experience teach a policy to walk, and is MPPI a cheaper explorer
than adding noise to a policy's actions or starts? Kendall's constraints, in
his words where we have them: no seeding from teacher tapes; MPPI is a local
optimizer; the student should be close to MPPI but better; use a value function
rather than rollouts; beat cloning and DAgger; short-horizon MPPI labels alone
are not enough, the student needs its own rollouts from restart states.

## What the main chat has established (branch `sac-continuation`)

1. **MPPI's states are not the problem.** From 534/560 MPPI states the SAC
   oracle kept walking (640/640 from its own), and it pulled the motion into its
   own gait. One MPPI action changed SAC's cost no more than a random action.
   (`docs/sac_continuation_results.md`)
2. **No gait conflict in the labels.** Near-identical states do not get
   systematically different MPPI actions. (`scripts/diagnostics/label_conflict.py`)
3. **Most of MPPI's torque is unexplained by state.** 69% of torque variance
   is above 5 Hz, where neighbour R^2 is about 0; below 5 Hz R^2 is about 0.38.
   SAC's gait is 1.67 Hz. (`scripts/diagnostics/action_spectrum.py`)
4. **Working explanation:** in torque directions the task barely cares about,
   MPPI's weighted update random-walks, because weights are uncorrelated with
   the noise there and the plan is warm-started. Cloning averages this away and
   ends up with weak torques (K1: clone |u| 0.30 vs MPPI 0.44).
5. **Two candidate fixes**, still being tested in the main chat:
   - at the source: mirrored (antithetic) sampling plus a weak restoring cost;
   - in the similarity: a value-based distance over action chunks, under the
     student's own value function. One-step Q differences are likely too flat
     (point 1), so single-action distances are not expected to work.

## What your review established, as the main chat reads it

- Coupling at alpha .12 (about 3% of cost spread) changed labels but did not
  reduce the >5 Hz share (60% to 59.6%) or raise low-band R^2. So "the policy
  preference acts as a restoring cost" is untested at useful strength, not
  refuted.
- The loop never labelled the updated students' own states, so nothing yet
  tests DAgger or corrective acquisition.
- Fitting 2,000 updates on 2,400 rows overfits (held MSE rises).

## Tasks, in order. Stop at any failed gate and report.

**T0. Warm-start check (cheap).** One restart, identical noise, zero plan
versus retained plan. Report candidate survival at H48 and executed actions.
Feeds: whether cold starts explain sparse surviving candidates.

**T1. Calibrate a coupling strength that can matter.** On saved banks, choose
alpha so median std(alpha D/N)/std(C) is about 0.3; also report 0.1 and 1.
Use per-step mean D/N (removes the early-fall bias). Do not reuse 5.76 D/N:
it is the old weak setting rescaled.

**T2. Label learnability before any fitting.** From current-student restart
states (T3's protocol), collect MPPI labels at alpha 0, the T1 value, and with
mirrored sampling at alpha 0. Run `action_spectrum.py` and `label_conflict.py`
on each. Report the >5 Hz share, R^2 by band, and MPPI's own survival and cost.
Feeds: the main chat's source-versus-coupling comparison directly; use the same
metric definitions so the numbers are comparable.

**T3. Corrective-data pilot, only if T2 shows a label difference.**
- Restarts: drawn each round from the current student's own rollouts,
  including states shortly before its falls. Not fixed contexts.
- Labels: MPPI's executed action at each visited student state, aggregated
  across rounds (DAgger). The walking/cost guard checks that MPPI is still
  competent; it does not delete corrective rows.
- Arms: true DAgger (alpha 0); coupled (T1 alpha, D/N); 5 Hz low-pass cloning.
  Matched compute, architecture, optimizer state and seeds.
- At least 3 short rounds. Early-stop each fit on held MSE instead of 2,000
  fixed updates.
- Outcome: walking on 32 or more unused paired resets, with confidence
  intervals. Fix in advance which result drops coupling.

## Reporting and process

- One page per task, receipts in an appendix. Add audits only where a claim
  depends on them; do not amend tolerances to continue a run.
- Work on your own branch (`codex/full-trajectory-pi` or a new one). Never
  commit to `main`.
- Close the uncertified L3 audit (JSON list versus array) as housekeeping.

## Coordination log

Append one line per finished task here (date, task, one-sentence result, link),
on your branch. The main chat reads this file and appends its own results
(mirrored-sampling stability runs, similarity function) in the same format.

| Date | Line | Task | Result | Link |
|---|---|---|---|---|
| 2026-11-04 | main chat | Brief written | Shared context and T0 to T3 assigned | this file |
| 2026-11-04 | coupled-loop agent | Handoff and saved-evidence reconciliation | Eight-round result and raw tables published; preserved source brief, qualified spectrum/coverage/drift claims and separated ownership; no new experiment or audit | [response](coupled_pilot_response.md), [design review](../coupled_loop_design_review.md) |
| 2026-11-04 | coupled-loop agent | Saved-bank coupling calibration | Six same-final-actor banks: competitive .3 gives alpha31.7349; top candidate unchanged6/6, first-control RMS>.01 in3/6; fallen mass increases3/6. No physics or fitting. | [report](../coupling_calibration_t1_result.md) |
| 2026-11-04 | coupled-loop agent | Four-state paired controller check | All16 conditions completed. Coupling changed actions and reduced disagreement, but one carried-plan episode fell at225 where alpha0 walked300. Walking ties7/8; survival8/8→7/8. Declared destabilization stop; no fitting, T3 launch or L3 audit closure. | [report](../coupling_controller_t0_result.md) |
