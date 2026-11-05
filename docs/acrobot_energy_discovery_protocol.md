# Near-upright control and energy-shaped Acrobot discovery

Version 1.1. This is a teacher-only probe. The previous failed prerequisite,
recordings, native costs and audit remain unchanged. Work stays on
`codex/full-trajectory-pi`; other branches and running jobs remain untouched.

1. **Mechanism:** the distance objective can reward folded motion before a useful
   swing-up is found. Regulating mechanical energy gives pumping an immediate
   objective; the earlier proximity-weighted pose/velocity terms handle capture.
2. **Contradiction:** reaching the upright energy while staying folded or failing
   the final height hold would show that this energy recipe does not resolve discovery.
3. **Cheapest contrast:** four unchanged-controller starts near upright, in parallel
   with two paired hanging starts for each of three fixed energy recipes. No learning.
4. **Decision:** if no energy recipe passes both starts, stop and report a teacher
   problem. A 2/2 result is exploratory; it does not replace fresh 9/10 qualification
   or authorize bank admission, fitting or an alternating loop.

## Reused evidence and declared changes

The [preserved catch source](../archive/acrobot_energy_catches-2026-11-05/README.md)
reports three short catches, about half a second each. Its raw runs are not tracked.
The unchanged model has upright `(0,0)` and hanging `(π,0)`. Its earlier normalizer
used folded `(π,π)`: the gap was 29.43 J. The true hanging-to-upright gap is 39.24 J,
with upright energy 58.86 J and hanging energy 19.62 J. Matching the old weight 10
quadratic coefficient therefore requires `160/9` with the correct denominator.

The new term penalizes the **full signed energy error squared**, including excess,
and turns down near the target using the earlier proximity reward. These changes
are explicit; the old deficit-only source is not rewritten. Energy episodes run
600 controls, twelve seconds, because the earlier catches needed 8–10 seconds of
pumping. The success criterion stays height ≥3.8 m at all final 100 control boundaries.
There is no extra speed, reconstruction or spectral admission condition.

## Frozen objective

Let `d` be tip distance to `(0,0,4)`, `z` tip height, `x` horizontal tip position,
and `v=(v0,v1)` the two joint velocities. Define

```
E = kinetic + gravitational potential
error = (E - 58.86) / 39.24
r = exp(log(0.1) * (max(d - 0.2, 0) / 2)^2)
g = r^2
pose = 1 - clip(z/4, 0, 1) + 0.05*x^2
scaled_speed = (v0/2.5)^2 + (v1/5)^2
excess = sum((max(abs(v)-8, 0)/8)^2)
c = pose + wE*(1-g)*error^2 + 0.15*g*scaled_speed + 0.05*excess
terminal = 100*pose + 25*g*scaled_speed
```

The pose, target-weighted velocity, excess-speed and terminal terms are reused
from the earlier catches; no new spin penalty is added. The energy term dominates
far from the target and is zero inside its 0.2 m proximity region. Pose incentive
remains present outside that region. Unlike the first study's squared-distance
cost, this teacher objective need not equal the success criterion. Its native
running sum and final terminal cost are reported separately, alongside the
original dense cost as a common secondary measurement.

## Frozen roster

The [configuration](../configs/studies/acrobot_energy_discovery_v1_1.json) contains
all full initial states, private seeds, exact planner dictionaries, model hashes,
limits and constants. Gaussian position/velocity perturbations have standard
deviation 0.05; initial time is zero. No seed, variant or input is selected by outcome.
If several energy variants pass 2/2, the first in the declared roster is the
candidate for a future qualification protocol; costs from different objectives
are not used to choose a winner.

| Probe | Starts | Controls | Planner | Physics threads |
|---|---:|---:|---|---:|
| Near upright, unchanged dense cost | 4 | 500 | H150/K256, λ3, σ0.6 white | 2 |
| Energy, matched historical coefficient | 2 | 600 | H300/K512, λ0.15, σ0.18553280373949055 white; wE=160/9 | 4 |
| Same energy coefficient, longer horizon | 2 | 600 | H400, otherwise identical | 4 |
| Doubled energy coefficient | 2 | 600 | H400, wE=320/9, otherwise identical | 4 |

At 50 Hz, H300/H400 mean six/eight seconds. All planners use three initial passes
and the existing receding-horizon MPPI implementation. Energy variants share the
two frozen starts and private planner seeds. Different horizons do not create
identical candidate banks. The near-upright diagnostic applies to the original
controller, not every energy recipe. The energy probe also changes planner settings
and episode duration relative to the first study, so it is not an isolated causal
A/B test of energy cost alone.

## Validation, limits and records

Before controller launch, validate analytic energy against native potential and
kinetic energy at 32 fixed states, including upright, hanging and a folded
configuration with upright total energy. Check model masses, COMs, inertias,
actuator scaling and exact initial contexts. Run the existing twelve-control
record/replay/batch comparison for each of the three distinct cost profiles.
Native replay remains `rtol=0, atol=1e-10`; complete-context restoration is byte-exact.
Include terminal cost when comparing an energy batch objective with sequential
running costs. Preserve and stop on any failed prerequisite.

Two workers execute in parallel after this shared preflight: two native threads
for near-upright and four for energy, with one math thread each. CPU allocations
are 1,200 seconds and 9,600 seconds respectively. Prerequisites/tests have 120 CPU
seconds; saved audit/reporting 180; engineering/publication 60. The shared ceiling
is **11,160 CPU seconds (3h6m)**, **2 GiB new output including mirrors and transfers**,
and **1,510,411,416 physics substeps**, including preflight. No neural forwards,
gradients or optimizer updates are allowed. Stop for invalid physics/bindings,
unrecoverable infrastructure or exhausted allocations. Ordinary task failures do
not cancel remaining valid comparisons; finish both seeds of every variant.

Preserve N+1 full contexts, pre-action observations, executed controls, running
costs, all planner costs/weights/nominals and per-candidate energy/height summaries.
Keep two candidate traces per bank (minimum cost, maximum height) and exact bank
hashes; do not imply that every candidate trajectory was retained. Record potential
versus kinetic energy, proximity gate, final speed, complete hold duration and time
spent near target energy while below 3 m. The latter uses |normalized energy error|
≤0.05 and is explanatory, never an admission rule.

Independently recompute saved outcomes, objective components, action/plan bindings
and accounting without new physics. For cross-platform cost summation and grouped
components, declare the FP64 bound `1e-10 + 64*eps64*max(1,abs(C))` before launch and
report absolute/scaled residuals. This differs from the unchanged strict native
trajectory-replay contract. Save every failure and all costs. Publish all-trial
height and energy figures; no fit, new score, architecture change or follow-up
experiment is launched by this protocol.
