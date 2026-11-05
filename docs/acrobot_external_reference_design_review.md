# Independent design review: identical-model Acrobot references

Saved-source/evidence review, 2026-11-05. No physics, fitting, build or optimization
was performed. Numeric audits of the saved collocation constraints and local
certificate were not repeated. Source hashes and the bounded review receipt are
under `runs/2026-11-05_acrobot_external_reference_001/design_review_001/`.

**Verdict:** the investigation narrows some implementation errors but has not yet
established working swing-up ground truth or isolated a breaking MPPI change.
One substantive defect was found in the new iLQG compatibility adapter. Preserve
its outputs as attempted-reference evidence; do not treat its negative result
as a faithful negative result for stock MJPC iLQG.

## Finding requiring correction before further iLQG execution

`reference_rk4_transitionFD` always perturbs a control forward by epsilon. At
normalized `u=+1`, the perturbed actuator is clipped, so this can estimate the
outward control derivative as zero. At `u=-1`, the forward perturbation is inward.
This differs from MuJoCo's required range-aware finite-difference convention:
when a forward perturbation would exceed the control range, its implementation
uses an admissible backward perturbation. The frozen 16-case validation uses
`u=.7*sin(.6*sample)`, so none tests either actuator limit. Agreement with centered
differences on those interior controls does not validate the boundary behavior.
[MuJoCo 3.6 derivative implementation, control-limit handling](https://github.com/google-deepmind/mujoco/blob/3.6.0/src/engine/engine_derivative_fd.c#L312-L351).

This affects the **adapted iLQG reference**, especially saturated trajectories;
it does not explain the original MPPI failure and does not affect predictive
sampling. A repair needs a separately frozen boundary-aware implementation and
a validation roster including both limits, with the old failure retained. It
must not be silently repaired underneath already recorded results. The owner
and root were notified before the planned standard-start benchmark ran.

Sensor timing is not an additional finding: native RK4 suppresses sensor
recomputation during its intermediate stages, and the adapter differentiates the
same pre-step sensor map exposed by trajectory rollout. The final direct forward
pass supplies the terminal sensor map. No one-step sensor shift is inferred.
[Native RK4 implementation](https://github.com/google-deepmind/mujoco/blob/3.6.0/src/engine/engine_forward.c#L993-L1034).

A requested second source-only recheck reached the same conclusion. For the
first five residuals `(4-z, -x, v0, v1, u)`, the expected pre-action sensor
Jacobians are

\[
C=\begin{bmatrix}
\sin q_0+\sin(q_0+q_1)&\sin(q_0+q_1)&0&0\\
-\cos q_0-\cos(q_0+q_1)&-\cos(q_0+q_1)&0&0\\
0&0&1&0\\0&0&0&1\\0&0&0&0
\end{bmatrix},\quad D=(0,0,0,0,1)^T.
\]

The sensor vector's extra three `tip_pos` entries have the ordinary tip-position
geometry Jacobian and zero control derivative. Exact expected matrices for all
40 frozen v2.1 boundary cases are saved in
`design_review_001/sensor_timing_recheck_001/analytic_sensor_jacobians.json`.
This recheck downloaded the exact 3.6.0 source and performed no native steps.
It does not add an unperformed numeric comparison to the recorded gate.

## What the other reference attempts establish

- **Collocation:** the nine local initializations produced no feasible candidate;
  the best recorded constraint violation was `0.2058975614`, versus `1e-6`.
  Independent saved arithmetic checked the actual equations/constraints. This
  does not establish existence, a minimum swing-up time, minimum task cost or
  infeasibility. The bounded initializations, 120-interval mesh, duration range
  [3,12] seconds and local solver remain unresolved search limitations. No
  differential swap can use these guesses as working ground truth.
- **Drake port:** plant parameters, torque bound, balance gain and handoff all
  differ from the stock example. In particular its stock balancing threshold is
  **1e3**, whereas the adapted controller switches only inside the new certified
  level **1.6401953694e-12**. Those numerical levels use different value matrices,
  but the change still makes this a different hybrid design. Its **0/2** swing-up
  result cannot isolate the torque limit or show that energy shaping with a
  practical handoff fails on this model. [Official threshold definition](https://github.com/RobotLocomotion/drake/blob/v1.51.1/examples/acrobot/spong_controller_params.h#L57-L62).
- **Certificate:** the analytic trigonometric remainder argument plus an exact
  rational positive quadratic Gram matrix supports a tiny nominal continuous
  feedback region, radius `6.103515625e-6`. It is not a nonlinear SOS search for a
  useful large region, a swing-up certificate, or a sampled/RK4 guarantee. The
  existing report makes these distinctions correctly. A CSDP success flag alone
  would have been insufficient; the exact arithmetic/remainder argument matters.
- **Standard metric:** eight frozen random-angle starts at 100 Hz can provide a
  small dm_control-distribution check using its reward. They cannot reproduce a
  published aggregate score. Because those starts differ from the five-context
  50 Hz roster, their difference is not an isolated causal control-rate effect.

## Decision

Do not launch learning or claim a single MPPI defect from these results. Repair
the identified reference-adapter contract under a new frozen version, then use
an actually executing same-model reference before one-factor swaps. Treat the
certificate as a mathematical positive control, not the practical terminal
funnel. A TVLQR trajectory becomes a certified funnel only with verified boundary
decrease and terminal containment under its real actuator and execution rules.
The task has not been shown infeasible; the requested external ground truth
remains incomplete.
