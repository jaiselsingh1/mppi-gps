# Current algorithm

One native Walker task, one MPPI implementation, one action policy and one alternating runner. The old outcome-only learner and specialized ending task are archived.

## Task and accounting

Use the existing Walker XML and 1.5m/s target-velocity costs unchanged, at 0.008 seconds per control. Episodes last 1,000 controls. Success means surviving with velocity MAE at most 0.3m/s. Comparison cost includes executed native costs, the native fall terminal once, and absorbing failure rate 5 for each unexecuted control after a fall. Absorbing cost is separate from executed cost.

Every model batch, live control, verification, re-solve, actor rollout and evaluation passes through the recorder. Batched post-fall simulation is charged. Neural/backward rows, optimizer steps, failures, hardware and wall time are recorded. Hypothetical plans and actual traces are separate.

## Independent search and coupled selection

Native MPPI uses K256/H48, temperature3, noise standard deviation0.3 and existing Butterworth noise, with two first-decision refinement passes. It receives no policy.

Copy its final bank and add the independently verified native mean. Use pre-action observations. Both baseline and guided selectors add the same action-sequence proximal penalty, weight0.01, relative to the entering native nominal. Thus alpha=0 exactly reproduces baseline selection. The private nominal still updates from task costs only.

Guided selection adds alpha/2 times the squared differences between actions and the frozen policy, with a single proposed alpha1. Verify synthesized means before selection. Numerical guidance failure falls back; budget/evidence failure stops and remains recorded.

## Policy fitting

The deterministic network maps17 native observations through two128-unit tanh layers to six bounded torques. Normalize using initial discovery data with standard-deviation floor0.05, fixed thereafter.

Each block uses Adam3e-4, batch256,2,000 updates, gradient clipping0.5 and parameter proximity0.01 to the incoming policy. Checkpoint weights, normalization, Adam, original fitting anchor and private minibatch RNG.

Use successful complete teachers as executed action targets. Preserve failures. At eight states per successful episode, perturb physical height/angles/velocities twice and re-solve. Distinguish these hypothetical labels. Guided teachers use the qualified coupled objective when supported. This approximates feedback supervision, not analytic feedback gains.

## Qualification and feedback

Every version begins with zero influence. Require15/16 fresh standalone successes before provisional guidance trials. Bind support to the policy hash, task and a normalized neighborhood of successful actor states. Unsupported states disable guidance.

Run two paired complete off/on episodes and two disjoint confirmation pairs. Require survival and success nonregression, at least0.1% lower mean complete cost and action changes above1e-6. Confirmation qualifies only that parent/controller; its traces may later teach children evaluated on separate roots.

Fit matched raw-data and guided-data children from identical policy/Adam states, update budgets and minibatch seeds. Retain raw teacher slots when guided execution worsens task cost or fails. Compare both children and parent alone on fresh paired starts. A beneficial round requires the guided child to qualify and improve cost against both without success regression. Evaluate the final selected actor on16 held starts.

Changed versions and failed/numerical qualification revoke influence. A failed gate remains negative and does not fabricate rounds or retune parameters. The one-seed pilot attempts up to two rounds, capped at350million requested controls and150million neural rows. It establishes no ending-task or efficiency claim.
