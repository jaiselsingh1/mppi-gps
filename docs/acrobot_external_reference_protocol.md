# Acrobot identical-model external references, v1.0

Frozen question: do established trajectory optimization and control methods solve
the **unchanged** dm_control physical Acrobot under its ±2 Nm elbow limit? No
policy fitting, MPPI retuning, task substitution or learning-loop launch belongs
to this investigation. Previous failures remain immutable. Work is on
`codex/full-trajectory-pi`, never the other research branches.

## Binding and interpretation

The model is `assets/acrobot.xml`, SHA256
`324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe`,
with its three existing common includes, native MuJoCo 3.6.0, RK4 at .01 s,
joint damping .05 and normalized motor range [-1,1], gear 2. All new native
references execute on robopil5090. Frozen create-only source exports and isolated
dependency environments live below
`/home/jaisel/mppi-gps-codex-acrobot/external_refs_2026-11-05/`.
Every stage records source/model hashes, installed versions, attempts including
failures, wall/CPU work and output sizes before drawing a scientific conclusion.

Official dm_control uses 100 Hz actions, uniformly random joint angles, zero
velocities and its dense tip-distance reward for ten seconds. Our earlier 50 Hz,
near-hanging starts/final-two-second height condition are different experimental
conditions on the same physical model. Both metrics will be identified rather
than substituted for each other. A solver success flag, an energy crossing or a
sampled capture region is not task success or a certificate.

## Stage budgets and pass/kill rules

1. **Saved evidence and official-source review.** No new physics. Model metadata
   review at most 15 CPU seconds/16 MiB; the separately frozen complete retained
   bank scan at most 180 CPU seconds/300 wall seconds/512 MiB working memory and
   16 MiB outputs. Process every retained scalar candidate and every retained
   trajectory, disclose arrays never recorded. A missing array remains missing;
   it cannot be reconstructed by silently rerunning physics.
2. **CasADi/IPOPT reference.** Isolated dependency setup at most ten wall minutes
   and 2 GiB. At most nine coarse Hermite–Simpson solves (three fixed initial
   shapes at initial durations 6, 8, 10 seconds), 120 intervals, variable duration
   [3,12] s, 600 iterations and 60 solver CPU seconds each. Objective is duration
   plus .0001 times torque effort. Bounds are the real ±2 Nm, not an unbounded
   virtual input. Terminal upright angles and velocities are equality constraints.
   Retain every solve, even infeasible ones. Select the feasible lowest-duration
   candidate before native execution; failure of all attempts stops this route.
   At most two exact-RK4 refinements at 100 Hz, 1000 iterations and 180 CPU seconds
   each, first feasibility/minimum effort and then the original dense task-cost
   integral at that fixed duration. Native verification includes at most 32
   one-step parity checks, two selected trajectories plus two-second zero-input
   suffixes and at most 32 complete standard-start trajectories if a reference
   controller exists; hard maximum 40,000 native steps. The coarse candidate is
   not called optimal: nonconvex IPOPT supplies local solutions and upper bounds,
   not a global minimum-time/lower-bound proof. Coarse constraint tolerance 1e-6;
   exact-RK4 constraints 1e-8; native one-step parity atol1e-9, rtol0; final native
   two-second height >=3.8 m, final angles <=.02 rad and speed <=.05 rad/s.
3. **MJPC.** Pinned official source, separate preparation protocol and adapter.
   Build at most 20 wall minutes/four CPU threads/2 GiB; native work at most
   20 wall minutes/four CPU threads/10,000,000 physics steps/2 GiB. Stock iLQG's
   `mjd_transitionFD` does not support RK4. Preserve its unsupported result, then
   use an explicitly identified finite-difference derivative adapter around the
   unchanged native RK4 step. Derivatives must pass a separate central-difference
   check before running the adapted planner. Never switch the XML to Euler.
   Sampling and adapted iLQG retain official planner/cost definitions. Failed
   episodes continue the valid roster; source/physics/budget failures stop it.
4. **Drake-derived control and certificate.** Separate preparation protocol;
   dependency setup at most ten wall minutes/2 GiB; numerical/SOS work at most
   20 wall minutes/two CPU threads/1 GiB and 100,000 native steps. The official
   controller's private default plant and ±20 Nm saturation cannot count as
   identical-model ground truth. Bind its explicit port to our constants and
   ±2 Nm, disclose balance-gain/cost changes. No duplicated four-gain historical
   PFL sweep. A Taylor-polynomial SOS result is not a certificate for the exact
   trigonometric/sampled system without a verified remainder/domain argument.
   Identify continuous versus sampled guarantees. Solver numerical status alone
   does not certify a region of attraction.
5. **Differential diagnosis, conditional.** No swaps before a reference passes
   native replay on identical physics. Reuse saved counterfactual banks for cost
   and weighting first. Freeze any additional one-factor native comparison in a
   separately numbered addendum, with a finite budget, before executing it. A
   failing external route is preserved and does not authorize a new sweep.

Total output ceiling is 8 GiB, setup/build at most 40 wall minutes, numerical
reference work at most 65 wall minutes; reference routes may run in parallel on
the worker without disturbing the user's jobs. Counters include setup,
derivative/parity work, unsuccessful attempts and verification. No resume under
changed limits. Any nonfinite physics, invalid binding, irrecoverable tool failure
or exhausted cap ends that route and retains its partial evidence.

## Closure

### Required standard-start external check, addendum v2.0

The five-context MJPC roster completed without a swing-up reference; it remains
unchanged. It does not answer the user's separate request for the official
dm_control metric on standard starts. Before any such controls execute, freeze
eight private `RandomState` seeds 2026110500 through 2026110507, the two uniform
joint angles in [-pi,pi), zero velocity and their full native initial contexts.
Run both existing MJPC planners on all eight for ten seconds at **100 Hz**,
holding each action for one native RK4 step and replanning every four steps.
Costs, model, horizon, noise, optimization effort and the validated derivative
adapter remain fixed. This is the upstream action rate; the earlier 50 Hz
results remain separately identified. There is no new MPPI experiment or tuning.

Reserve at most 120 additional build wall seconds (within the original remaining
build allowance), **16,000,000 additional native steps, 300 wall seconds and
2 GiB output**. The measured prior roster cost was 9,024,006 steps and about
15 seconds; the new roster is expected to use approximately 14.44 million steps.
This reservation uses the unused parent 65-minute numerical allowance; total
study storage still cannot exceed 8 GiB. Reuse unchanged derivative/model checks,
bind their earlier receipts, and stop on invalid binding, nonfinite execution,
evidence corruption or exhausted resources. Ordinary task failures complete the
roster. Report official reward mean and sum, first reach and final hold
separately. Eight starts are a small external check, not a reproduction of a
published aggregate score. No differential swaps or learning follow unless a
working reference is established and their own bounded protocol is frozen.

Publish raw saved-candidate coverage, EOM/linearization, native reference traces,
dm_control metric definitions, any actual certificates with verifier, failures
and receipts. Distinguish feasibility from optimality and a controller trajectory
from a certified funnel. If identical-model references do not yet work, the
diagnosis remains unresolved: do not blame the task or claim one particular
MPPI ingredient caused the failure.
