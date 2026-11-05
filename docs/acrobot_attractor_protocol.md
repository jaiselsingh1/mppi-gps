# Acrobot transient into a locked fixed point — version 1.0

## Mechanism, contradiction, contrast and decision

1. A useful teacher needs a stabilizing attractor, not just a high tip or matched
   total energy. MPPI supplies the transient; an upright LQR controller supplies
   capture. Separately, near-single-sample MPPI weights may make its update brittle.
2. Failed local LQR capture contradicts the proposed entry region. Raising effective
   sample size without better executed holds contradicts temperature as a sufficient
   balance fix. Failure to enter the region leaves swing-up discovery unresolved.
3. First test one LQR design and its finite basin; then change only temperature on
   the four historical near-upright starts. Finally test two MPPI-to-LQR transients.
4. Preserve each outcome. Only a confirmed local entry region permits MPPI stages;
   only two successful development joins/captures permit fresh ten-start teacher
   qualification. No learning or alternating loop is part of this protocol.

The [general formulation](transient_attractor_formulation.md) maps the actual
`c13d14c` Walker implementation. Its relative orbit and numerical locking are not
promoted to an exact periodic-controller theorem. Previous Acrobot results,
recordings, costs, failures and audits remain unchanged.

## Frozen inputs and LQR controller

The [configuration](../configs/studies/acrobot_attractor_v1_0.json) binds the same
model, native runtime, 50 Hz control period, all historical full contexts, baseline
H5 hashes and private seeds. Work stays on `codex/full-trajectory-pi` and uses a
create-only source export on `robopil5090`. Other branches and jobs are untouched.

The upright is `q=(0,0), v=0, u=0`. Exclude simulator time and wrap angular error
modulo `2π`, leaving raw observations unchanged. Solve the discrete DARE for the
native one-control map with the dense cost's exact quadratic:

```
Q = [[4,2,0,0], [2,1,0,0], [0,0,.01,0], [0,0,0,.01]],  R = .01
u = clip(-K [wrap(q0), wrap(q1), v0, v1], -1, 1)
```

No gain cap or Q/R sweep is introduced. Gear two is included once in the native
input map. First preserve 37 one-control measurements: equilibrium; central
differences at `1e-6` and `5e-7`; sixteen private held-out perturbations of scale
`1e-5`. Native Jacobians must match the independently derived RK4 mechanical map
within `1e-7`, held-out predictions within `1e-7`, and equilibrium within `1e-12`.
Require positive P, Riccati relative residual ≤`1e-8` and closed-loop radius <1.
Replay twelve recorded LQR actions from the full first historical context with
byte-exact restoration and `rtol=0, atol=1e-10` state/context/cost agreement.

## Finite basin measurement and entry rule

Every LQR trace lasts 500 controls. Height success remains the final 100 POST
boundaries at least 3.8 m. **Fixed-point capture** additionally requires all those
boundaries to have absolute wrapped angles ≤.02 rad and speeds ≤.05 rad/s. Report
both outcomes separately; the latter defines the proposed basin, not a changed
height criterion for the teacher.

There are 234 calibration starts: the four historical starts, equilibrium,
24 Lyapunov-coordinate directions at each of `.25,1,4,16` times the unsaturated
input level, 52 single-coordinate probes, and an 81-point joint grid with angles
`{-.05,0,.05}` and speeds `{-.25,0,.25}`. The larger unneeded default shells are
not executed. The config fixes the marginal magnitudes and direction construction.

Choose the largest contiguous shell level whose 24 probes all capture. Confirm it
on 64 fresh boundary directions and 64 independent interior states; freeze their
three PCG64 seeds before execution. Any confirmation failure stops the route;
there is no post-confirmation shrink/retry. A nonempty region with 128/128 captures
is the **local-attractor prerequisite**. The historical 0–4 capture count is reported
separately: starts outside that region do not invalidate a confirmed local lock.

Use `e'Pe ≤ rho` as the operational entry set. This is finite empirical validation,
not a nonlinear invariant-set proof. Report P/rho, all failed starts, coordinate
projections and marginal captures; never label their bounding box jointly safe.
Every actual swing-up join receives a native LQR continuation to check this
generalization beyond the sampled basin probes.

## Temperature contrast, then transient acquisition

Reuse the stock `lam_target_neff_fraction` implementation. The saved 20-bank check
found λ32 around 61–122, versus historical λ3; it selects **ESS32**, not a new
optimizer. For the four near-upright starts use K256/H150, σ.6, three initial
passes and every other historical setting unchanged, with fraction `.125`.
Reuse recorded λ3 outcomes as the control. Verify the first same-seed candidate
bank against each historical bank before executing a control. Save actual adaptive
temperatures, cost spreads, weights, actions, height holds, clipping and costs.
Require reachable ESS32 within `1e-6`; equal-cost/tied unreachable cases are reported
explicitly. Holding remains an actual outcome, not implied by passing ESS.

After this diagnostic completes and the local lock passes, retry only the two
already selected hanging starts, using the prior six-second energy recipe
H300/K512, weight160/9, λ.15 base, σ.18553280373949055, with fraction `.0625`.
At the first executed boundary inside the frozen entry set, stop MPPI and execute
200 LQR controls in the **same complete context**. No new MPPI calls are paid after
joining. The transient has at most 600 controls. A missing join is a task failure;
finish both starts. Report first entry, LQR outcome and prefix/suffix costs separately.
MPPI-only near-upright holding is a diagnostic, not an additional admission gate:
the composed teacher delegates stabilization to LQR.

Both development starts must join and pass the unchanged final-two-second height
test under LQR before ten disjoint, pre-frozen qualification starts are executed.
Qualification requires at least nine joins with successful continuations. Reaching
the basin is the acquisition success; the continuation checks that the advertised
lock actually captures. No criterion requires MPPI alone to hold for two seconds.

## Budgets, stops and records

| Part | CPU cap | Native/control-equivalent cap | Forecast controls | Physics substeps | Output cap |
|---|---:|---:|---:|---:|---:|
| LQR, all basin probes and validation | 600 s | 181,049 | 0 | 362,098 | 128 MiB |
| Adaptive near-upright diagnostic | 1,500 s | 2,000 | 77,107,200 | 154,218,400 | 128 MiB |
| Two development joins | 2,600 s | 1,600 | 184,934,400 | 369,872,000 | 128 MiB |
| Conditional ten-start qualification | 11,000 s | 8,000 | 924,672,000 | 1,849,360,000 | 384 MiB |

Shared maximum **16,000 CPU seconds**, **2 GiB total new output including mirrors**,
and **2,373,812,498 physics substeps**. Support allocations are 30 CPU seconds saved
temperature analysis, 150 tests/saved review and 120 engineering/publication.
Native stages are sequential; limits and unlaunched conditional work remain explicit.
The cheap LQR stage runs before spending on MPPI. Stop for failed validity checks,
binding changes, invalid physics, unrecoverable infrastructure or exhausted limits.
Ordinary task failures complete their stage roster and remain evidence.

Record N+1 complete contexts, PRE observations, deployed clipped/raw torques,
costs, all planner weights/costs/nominals, gain/linearization data, join bindings and
all work counts. Independently recompute the saved measurements and accounting;
no audit repeats physics. Publish all-trial motion/height/velocity figures and
the teacher-gate verdict, with failed or conditional-unexecuted work retained.
