# Transient into an attractor: Acrobot result

**The local upright controller is established, but the transient still misses
its entry region.** LQR captured 128/128 fresh points in the selected empirical
region; adaptive MPPI held from 0/4 historical starts and entered that region from
0/2 hanging starts. The frozen gate stopped before fresh qualification or learning.

## General structure

An episode uses a transient controller until the first valid entry into a locked
attractor's basin, then switches to its stabilizing controller. A periodic orbit
has a phase-indexed reference and LQR gains; a fixed point is period one, with no
phase search or clock advancement. Policy adoption rebuilds those pieces from a
competent policy's own rollouts. This organizes the existing machinery; it does
not establish learnability or convergence.

The [formulation and exact Walker code map](transient_attractor_formulation.md)
cover `best_stride`/`Section`, `locked`/`periodic_gains`, `cycle_walk` and
`build_cycle` at `c13d14c`. Walker's orbit is relative to forward translation,
and its numerical locking/periodic-gain approximation is not an exact theorem.
Its six-seed result is recorded as the source's claim, not independently audited
by this Acrobot study. Here the upright was specified by the task and checked as
a native equilibrium; it was not discovered in a successful policy rollout.

## Locked fixed point and measured basin

The [frozen design](acrobot_attractor_protocol.md) uses the same model and complete
historical contexts. For `e=[wrap(q0),wrap(q1),v0,v1]`, the controller is
`u=clip(-Ke,-1,1)`, with native gear two and

```
K = [-202.1607273, -52.4055434, -74.2106332, -22.1887262]
entry region: e'Pe <= 0.11064310893419543
```

P and every exact coefficient are in the [final lock](../results/acrobot_attractor_001/run/recovered-lqr/lqr.json).
The native Jacobian matched analytic RK4 within `1.25e-12`; the local closed-loop
spectral radius was **0.94282**. No gain cap was applied; motor torque remains bounded.
LQR captured **1/4** historical starts. The other three saturated **99.6–100%**
of controls. The largest contiguous successful calibration shell was four times
the unsaturated level: 24/24 captures there, versus 18/24 at the next shell.
Its independent boundary/interior panels each captured 64/64.

| Isolated coordinate | Captured tested starts | Next tested failures |
|---|---|---|
| Shoulder angle | ±0.01 rad | ±0.025 rad |
| Elbow angle | ±0.01, ±0.025 rad | ±0.05 rad |
| Shoulder speed | No tested nonzero point | Smallest was ±0.05 rad/s |
| Elbow speed | ±0.05, ±0.1 rad/s | ±0.25 rad/s |

The ellipsoid's coordinate projections are ±`[.1682,.3042]` rad and
±`[.7541,2.1131]` rad/s. **They are not a captured box:** only 5/81 joint-grid
points captured. These are finite empirical tests, not proof of nonlinear
invariance or of every state between tested points. [Basin and failure plots](../results/acrobot_attractor_001/lqr_figures/basin_projections.png)
show the coupling between angles and velocities.

## Weighting and transient outcomes

Adaptive temperature attained ESS32 in all **2,008** near-upright optimization
passes, with λ **34.61–156.86**. Holds stayed **0/4**, with mixed task-cost changes.
The intervention worked numerically but was insufficient for balance.

| Hanging start | Old → ESS32 peak height | Old → ESS32 energy cost | Closest new V/rho | Join |
|---|---:|---:|---:|---:|
| 0 | 3.504 → 3.836 m | 4774.14 → 4360.33 | 64.33 | No |
| 1 | 3.652 → 3.902 m | 5032.91 → 4384.74 | 68.38 | No |

All six initial candidate banks matched their historical action hashes exactly;
subsequent banks follow different plans/states. Both transients improved height
and energy cost but never reached the independently selected entry set. Neither
triggered an LQR suffix. The original recordings also had no basin entry. These
two-seed observations are not a reliability or learning claim.

[Every paired trace](../results/acrobot_attractor_001/controller_comparison.png),
[raw measurements and hashes](../results/acrobot_attractor_001/README.md) and
[complete accounting](acrobot_attractor_accounting.md) are preserved. Independent
saved review passed **79,612 checks**, including the original serialization
failure and its zero-physics finalization recovery. No scientific tolerance,
controller coefficient, seed or objective was changed after launch.

**Next discriminating test, not launched:** one short paired candidate-bank assay
at the saved closest-approach contexts, recording entry into this fixed region
for every candidate. This separates missing reachable proposals from poor cost
ranking before choosing another transient objective or spending on a full run.
The learning loop remains untested and the teacher gate remains unmet.
