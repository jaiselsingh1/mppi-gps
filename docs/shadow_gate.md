# S3: task-first gate, shadow check

S2 produced three frozen policies: untrained (0 controls), mostly standing
(200k), and forward walking (800k). S3 asks whether policy suggestions can pass
an explicit task test at states actually visited by the unchanged MPPI teacher.
No new training is needed. This stage runs locally on CPU.

## The one new mechanism

For each decision, MPPI first produces its usual optimized 48-action plan.
Each frozen policy separately predicts 48 actions by simulating its own
closed-loop behavior from the **same current state and solver warmstart**.
The policy sees each state before selecting that state's action; prediction
stops on the first fall.

Both actual action sequences are then scored together using the unchanged
Walker task cost, including its absorbing cost after a fall. Do not compare
the policy against the cheapest MPPI sample or average sample cost: neither
is necessarily the plan MPPI executes.

Accept a suggestion only when:

- Its trajectory is finite and its task cost is no greater than MPPI's
  actual plan cost. Slack is fixed at zero.
- It does not predict a fall when the MPPI plan predicts no fall.

An invalid candidate rejects. An invalid baseline stops the run rather than
claiming that fallback is safe. Both plans may predict falls; acceptance in
that case is only a cost comparison, **not a claim of safety**.

In this shadow stage, acceptance is a recorded hypothetical decision.
**Every live action still comes from raw MPPI.** No policy cost is added to
MPPI, no nominal plan is re-centered, and no candidate changes planner RNG.
The isolated scorer lives in `src/mppi/gate.py`; the separate runner is
`scripts/runners/run_walker_shadow.py`. The original runner is untouched.

## First development check, declared before running

Use the two existing S1 episodes, not fresh qualification seeds:

| Reset / planner seed | Reference | Purpose |
|---|---|---|
| 30 / 20030 | 1,000-control success | Check suggestions during ordinary walking. |
| 31 / 20031 | 542-control fall | Check suggestions around the previously examined failure. |

All three policies observe the same raw trajectory. Run continuously from
reset; do not restore saved states between decisions, which could mask an
unwanted side effect. Required result: exact equality of states, warmstarts,
actions, costs, contacts, forces, optimized plans, and incoming planner
nominals/RNG/refinement flags against S1. Any mismatch fails the command.

Record candidate plans, cost comparisons, predicted falls, invalid-candidate
reasons, hypothetical acceptance, action difference, and wall time. A low
acceptance rate is a result, not a reason to retune thresholds. These two
episodes describe development behavior; neither acceptance nor the absence
of a mismatch establishes better closed-loop behavior.

```sh
.venv/bin/python -m scripts.runners.run_walker_shadow \
  --reference runs/2026-10-11_walker_instrumentation/replay_reset30_planner20030/traces.npz \
  --out-dir runs/2026-10-11_s3_shadow_001/success

.venv/bin/python -m scripts.runners.run_walker_shadow \
  --reference runs/2026-10-11_walker_instrumentation/replay_reset31_planner20031/traces.npz \
  --out-dir runs/2026-10-11_s3_shadow_001/failure
```

The runner verifies the selected checkpoint hashes, task and physics hashes,
saves its source snapshot and manifest, and writes status every 100 controls.
`report.json` gives parity and aggregate diagnostics; each policy directory
contains `decisions.json` and `candidates.npz`. `traces.npz` also saves
post-planning snapshots explicitly for the next counterfactual check.
Use fresh output directories; existing evidence is never overwritten.

## What follows, without combining experiments

After shadow correctness passes, use paired **single-intervention** branches:
from the same saved decision, one branch takes the accepted policy action,
the other takes MPPI's action, and both subsequently use raw MPPI. Restore
the post-planning nominal and RNG in both branches, without planning again
before their first action. Compare their continuation for 250 controls,
stopping each at its first fall. This tests whether a favorable short-horizon
score translates into a better continuation; it does not activate recurrent
gating or retrain either controller.

For this first diagnostic, choose the **earliest accepted decision for each
policy in each of the two reference episodes**, plus the earliest accepted
decision per policy inside the failure episode's previously identified
489–501 window. Deduplicate selections and omit a category if no acceptance
exists. This selection rule is set before examining continuation outcomes;
do not replace an unhelpful branch with a more favorable one. Report every
selected branch, including failures, over 250 controls (2 seconds). This is
a small mechanistic diagnostic, not an episode-level statistical test.

Only after those checks should an active full-episode development pilot
inform S4. The fixed S4 margins remain 2 percentage points of completion and
0.02 m/s speed MAE, both with two-sided 95% intervals. Shadow acceptance is
not paired episode discordance and must not be used as its sample-size proxy.
No non-inferiority result or GPS improvement has been established yet.

## Current result

The two reference episodes pass exact shadow parity. The first four
preselected single-action comparisons also pass raw-continuation replay,
but all four have higher actual two-second task cost after intervention;
three have worse speed tracking. Both arms survive in all four cases.
This is a useful warning about predicted plan score versus the continuation
that actually executes, not a population-level result. See the
[S3 experiment entry](experiment_log.md#2026-10-11--s3-shadow-only-task-first-gate)
for counts, numbers and limitations.

The broader S3 study (about 50 sampled accepted states per checkpoint where
available) remains outstanding. Do not infer that the gate is ready as a
better teacher or that S4/S5 has been run. A single pair can be reproduced
using the saved, audited bundle:

```sh
.venv/bin/python -m scripts.diagnostics.walker_gate_counterfactual \
  --shadow-dir runs/2026-10-11_s3_shadow_001/success \
  --policy-label intermediate --decision-step 96 --steps 250 \
  --out-dir runs/example_fresh_counterfactual
```
