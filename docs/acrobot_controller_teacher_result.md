# Acrobot: teacher prerequisite failed

**The selected MPPI controller balanced from 0/10 fresh hanging starts. None of
all sixteen complete episodes reached the 3.8 m upright threshold.** The specified
9/10 teacher gate therefore failed. No bank was admitted for learning, no policy
was fitted and none of the nine alternating-loop runs was launched. The similarity
mechanism remains untested on Acrobot.

The [frozen protocol](acrobot_controller_loop.md) reused the preserved model and
stock MPPI. Each episode ran 500 controls at 50 Hz: ten seconds. Success required
height ≥3.8 m at all final 100 control boundaries; velocities were reported
separately. The dense cost was squared tip distance to `(0,0,4)` plus
`0.01‖qdot‖² + 0.01‖u‖²`, with no terminal shaping. Zero angles are upright;
`(π,0)` is hanging.

| Development recipe | Balance | Mean native cost |
|---|---:|---:|
| 3 s white noise, σ=0.6 | 0/2 | 4277.369 |
| 6 s, 2 Hz smooth noise, σ=0.6 | 0/2 | 4298.374 |
| 6 s, 2 Hz smooth noise, σ=1.0 | 0/2 | 4315.160 |

The predeclared rule selected the first recipe by cost after all three tied on
success. Its ten disjoint qualification starts had mean cost **4084.572**
(range 3717.098–4406.062) and maximum sampled height **2.107 m**.
[Every trace](../results/acrobot_controller_teacher_001/figure.png) shows height,
raw joint angles and torques. A [median-cost failure still](../results/acrobot_controller_teacher_001/qualification_median_failure_at5s.png)
illustrates the motion; the [animation and raw arrays](../results/acrobot_controller_teacher_001/external_evidence.json)
remain outside Git. The illustration uses saved states and no new simulation.

**Interpretation.** These recipes failed to discover a qualifying swing-up.
Ability to stabilize from near upright remains untested. The small development
panel confounds horizon and noise shape, so it cannot identify a unique cause or
rank all three recipes' reliability. Full candidate state trajectories were not
saved, preventing a retrospective separation of inadequate exploration from
selection of poor updates. This is not evidence against the unexecuted learning loop.

**Decision.** Stop at the unchanged teacher gate. Recommend one small probe: the
same selected planner and cost from near-upright starts, separating balance
control from swing-up discovery. It is not launched. Architecture, feedback rules
and score changes remain deferred.

The independent saved-only review passed **136,918 checks**, recomputing executed
costs and Gibbs weights and checking seed/row bindings, plan shifts, executed
action bindings and work counts. Maximum per-control cost discrepancy was `3.55e-15`; native
replay/batch checks kept their original tolerances. This verifies saved outcomes,
not unrecorded candidate physics.

[Evidence and source bindings](../results/acrobot_controller_teacher_001/README.md)
include every failure. [Accounting](../results/acrobot_controller_teacher_001/complete_accounting.json):
8,000 executed controls, 36 replay/control-equivalent evaluations, 385,536,000
forecast controls and 771,088,072 worker physics substeps; local contract checks
added 48. Worker CPU was **51.32 minutes**, wall time **26m40s**, peak RSS
**216.82 MiB**; zero network forwards or optimizer updates. The 3h5m total CPU
reservation and 512 MiB output ceiling were respected. Other jobs and branches
were untouched.
