# A transient into a locked attractor

The common structure is a finite controller that brings each start into the
basin of one shared, stable terminal behavior. The terminal behavior can be an
upright equilibrium or a repeating gait. This factors the teacher's execution;
it does not by itself establish that the resulting controller can be cloned or
that a particular policy-similarity score improves learning.

## Variables and locked controller

Let `c` denote the complete native simulator context and let `z=ψ(c)` contain the
physical coordinates used for stabilization, excluding the advancing clock.
The one-control map `F` includes the declared frame skip and actuator limits.
Context restoration retains integration state, warmstarts and other native
fields; equal joint coordinates alone are not a complete replay binding.

An attractor candidate is a reference
`A={(z*_j,u*_j): j=0,…,P−1}` satisfying

\[
F(z_j^*,u_j^*)=z_{(j+1)\bmod P}^*.
\]

For a fixed point, `P=1`. A walking gait is a **relative periodic orbit**:
this equality holds after quotienting forward translation, with a forward
advance stored per period. Time itself never returns to its initial value.
Rollouts propose attractors; a small return residual is evidence for a candidate,
not a stability certificate. For fixed points, a low-velocity endpoint must also
satisfy the equilibrium equation; a momentarily stationary turning point does not.

With fixed physical scales `S=diag(s)`, define
`e=S⁻¹ Δ(z,z*_j)`. Angular differences use a declared local wrapped chart;
linear coordinates and velocities use ordinary subtraction. Linearize the same
native one-control map at each reference to obtain scaled `A_j,B_j`. LQR minimizes
`Σ(eᵀQ_j e + δuᵀR_j δu)`, giving

\[
K_j=(R_j+B_j^\top P_{j+1}B_j)^{-1}B_j^\top P_{j+1}A_j,
\qquad
u=\operatorname{clip}_{\mathcal U}(u_j^*-K_j e).
\]

A periodic Riccati solution has `P_P=P_0`; a fixed point uses the discrete
algebraic Riccati equation. Check the closed-loop linear multipliers on the
stabilized coordinates. Actuator saturation and nonlinear dynamics still require
native validation. A local linear stability check is not a global basin proof.

## Transient, basin and join

For start `i`, the transient consists of its finite start-up and continuation
controllers. A shared carrier can provide the same start-up law from different
starts; their realized states and actions need not be identical. The assembled
controller is

\[
u_t=\begin{cases}
\kappa_i^{\rm tr}(c_t,t),&t<\tau_i,\\
\kappa_A(c_t,(j_i+t-\tau_i)\bmod P),&t\ge\tau_i.
\end{cases}
\]

The true phase-dependent basin `B_j` contains contexts whose locked-controller
execution remains valid and approaches the attractor. In an experiment use a
**separately frozen, validated entry set** `B̂_j`, with its finite test horizon,
capture/contraction tolerances, state ranges and failures reported. A sampled
grid describes tested captures; its bounding box is not automatically an entire
captured four-dimensional region. A Lyapunov ellipsoid `eᵀP_j e≤ρ` is a certified
inner basin only if nonlinear decrease/invariance and actuator conditions are
established throughout it, rather than checked at a few points.

The general join rule is the earliest eligible transient boundary

\[
\tau_i=\min\{t:\exists j,\ c_t\in\widehat B_j\}.
\]

For an orbit, select an admissible phase at that boundary and then advance it
once per control. For a fixed point there is only phase zero. State distance can
propose join attempts, but a close pass with excessive velocity may lie outside
the basin. If entry is established by a counterfactual locked suffix, preserve
the exact switched context, suffix outcome and cost, and count that verification.
Report failed joins and no-join fallbacks separately from completed joins.

## Mapping the current Walker machinery

Source is immutable commit
[`c13d14c1e4df84af7cb3ffe143723db07cb5b961`](https://github.com/jaiselsingh1/mppi-gps/tree/c13d14c1e4df84af7cb3ffe143723db07cb5b961).
Line numbers below refer to that commit, not a later working file.

| General piece | Walker implementation | Fixed-point replacement |
|---|---|---|
| Find a recurring terminal behavior | [`Section` / `best_stride`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/stride_cycle.py#L40): scaled section-return difference, excluding time/forward position; select the most repeatable stride after start-up | Find a stationary candidate and enforce its equilibrium equation; no section crossing or stride selection |
| Stabilize and lock it | [`periodic_gains`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/stride_cycle.py#L99), `close_cycle`, [`locked`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/controller_loop.py#L1103) | Discrete DARE at the equilibrium, bounded native execution and a measured capture region; no repeated-stride closure or forward advance |
| Bring starts to it | Shared carrier start-up, then carrier/own tracking continuation in [`cycle_round`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/controller_loop.py#L1230) | A swing-up transient into the frozen upright controller's entry set |
| Choose an entry and switch | [`cycle_walk`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/controller_loop.py#L1127): phase search, earliest near passes and validated locked suffix | Earliest verified fixed-point-basin entry, followed by phase-zero LQR |
| Assemble or rebuild the structure | [`build_cycle`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/controller_loop.py#L1867) and [`--adopt`](https://github.com/jaiselsingh1/mppi-gps/blob/c13d14c1e4df84af7cb3ffe143723db07cb5b961/scripts/diagnostics/controller_loop.py#L2163) | Rebuild policy-derived transients and revalidate their joins; an unchanged upright task/model retains the same equilibrium rather than inventing a new terminal pose |

The concrete code approximates the ideal formulation. `periodic_gains` performs
finite-horizon Riccati recursion over four stride copies and caps gain singular
values at three; it does not solve an exact periodic Riccati boundary condition.
`locked` runs 30 strides twice, recomputing gains between passes, and accepts
mean last-five return residual ≤0.1 scaled units plus task admissibility. Thus
“locked” means the code's numerical validation, not a theorem of exact periodicity.

`cycle_walk` first tries local distance minima within 0.5 scaled RMS units in
time order. It then tries closest fallback places within 1.0, up to ten attempts
total, with the declared spacing. Each candidate suffix must satisfy whole-walk
admissibility and cost. A no-join continuation can be retained and is explicitly
logged with `join=None`; it is not an attractor join.

Adoption requires an accepted deployed policy with score at least `1/budget`.
For budget 1.5 this is `2/3`; the score uses mean MPPI cost over mean policy cost
on 32 fixed starts. `own_walks` additionally checks health and individual cost
limits on every training start; it does not itself call walking admissibility.
The rebuilt locked stride and assembled walks have their own subsequent task
checks. This changes the transient and terminal behavior family when policy
rollouts propose an acceptable new gait, rather than only refitting stale labels.

## Acrobot instantiation and evidence boundary

On the [unchanged model](../assets/acrobot.xml), use physical state
`z=(q0,q1,v0,v1)`, upright `z*=(0,0,0,0)` modulo independent full turns, and
`u*=0`. The native state also contains time. The control period is 0.02 s,
`u∈[-1,1]`, and actuator gear is two. All four physical coordinates matter;
there is no Walker-style free forward-position coordinate. Scales are declared
in radians and radians/second, with `Q,R` frozen before execution. Walker's
gain cap in per-walk standard-deviation units should not be copied implicitly.

For feedback, use `wrap(q0), wrap(q1), v0, v1` in a local upright chart. Preserve
raw native angles in evidence. This is an error-coordinate choice for the
locked controller, not an unannounced change to a future learner's observations.
The known task equilibrium supplies the first candidate because no current
teacher rollout contains a successful catch. Its LQR test first measures the
exact four near-upright contexts that the prior MPPI controller failed, then the
declared angle/velocity panel without silently enlarging it. Their four capture
outcomes are reported separately: a failure outside the selected local region
does not invalidate a region that passes all its separately frozen confirmations.

Swing-up success in the new structure is **verified basin entry**, not a
two-second hold produced by MPPI alone. The switched LQR suffix validates that
the entry definition actually predicts capture; failure after an admitted join
challenges the basin definition. Historical height-hold failures stay unchanged.
The temperature check and swing-up retry require separately frozen execution
protocols; this formulation authorizes no policy fitting or learning loop.

The [previous audited Acrobot result](acrobot_energy_discovery_result.md) remains
0/4 near-upright holds and 0/6 energy-recipe successes. The Walker six-seed result
and mean gap 0.0093 are reported in `c13d14c`'s committed docstring, lines 523–534,
which also reports mean held-out mid-walk recovery of 13.5/51. This source review
confirms the stated implementation and recorded claim; it is not an independent
audit of those seeded raw runs or a claim of broad recovery/generalization.
