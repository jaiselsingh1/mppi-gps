# MuJoCo MPC on the identical Acrobot

Status: corrected bounded reference runs completed; independent saved-evidence audit pending.

Official MJPC has an Acrobot task and both requested planners. Its stock patch removes dm_control's RK4 attribute, leaving Euler. Its iLQG calls MuJoCo's transition derivative routine, which rejects RK4. Unmodified iLQG is therefore incompatible with our integration recipe; this is not evidence that Acrobot cannot be controlled. [Official model patch](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/tasks/acrobot/acrobot.xml.patch), [derivative caller](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/planners/model_derivatives.cc), [MuJoCo restriction](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mjd-transitionfd).

The reference retains RK4 and the existing worker MuJoCo 3.6.0 library. First it preserves one original derivative failure. A separately identified adapter then replaces only the unsupported derivative call with finite differences of actual RK4 steps. Downloaded official source bytes, Acrobot residual and planner algorithms remain unchanged. Describe the result as **MJPC with a validated RK4 derivative adapter**, not unmodified stock iLQG.

## Exact source and differences

Source commit ff572a21e7c2bf9fda62e1862a758da7e9a8719b was downloaded 2026-11-05. Its local preparation occupied approximately 7.5 MiB; this is not a measured build estimate. Per-file hashes and exact input bindings are saved in runs/2026-11-05_acrobot_external_reference_001/mjpc_preparation_001/frozen_inputs.json.

| Property | Official MJPC Acrobot | Identical-model reference |
|---|---|---|
| Construction | patched dm_control model | project XML plus user sensors/custom numerics only |
| Elbow torque | normalized [-1,1], gear 2 | same ±2 N·m |
| Integration | Euler, 0.01 s | RK4, 0.01 s |
| Planning | 2 s horizon, 0.01 s prediction interval | same |
| Execution | headless example acts every native step, replans every four | 50 Hz held control; replans every four native steps |
| Sampling | 10 candidates, 10 cubic spline points, exploration SD .05 | same |
| iLQG | 10 line-search rollouts, forward FD 1e-6, action limits | same plus RK4 derivative adapter |
| Starts | home [3.142,0], zero velocity | exact saved hanging context and all four historical near-upright contexts |

The residual is vertical/horizontal tip error, joint velocities, normalized control. Quadratic norms include one half, so the literal MJPC cost is:

\[
\ell=25[(4-z_{\rm tip})^2+x_{\rm tip}^2]
 +0.5(v_0^2+v_1^2)+0.025u^2.
\]

The project dense cost uses distance squared plus .01 times velocity/control squares and is reported separately. MJPC averages its 201-state costs, including a terminal residual with repeated final action. Raw costs cannot be compared across these objectives as common units. [Residual](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/tasks/acrobot/acrobot.cc), [weights](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/tasks/acrobot/task.xml), [norm](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/norm.cc), [trajectory](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/trajectory.cc).

Predictive Sampling picks the lowest-cost sampled spline rather than PI's weighted average. It has no temperature or effective sample size. It creates internal absl::BitGen instances without public seed control. Preserve actual candidate arrays and do not claim paired seeded randomness. Candidate zero preserves the nominal policy. [Sampling implementation](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/planners/sampling/planner.cc).

## Frozen run

The [configuration](../configs/acrobot_mjpc_reference_v1.json), [driver](../scripts/diagnostics/acrobot_mjpc_reference/main.cc), [supervisor](../scripts/diagnostics/acrobot_mjpc_reference/run.py) and [minimal build](../scripts/diagnostics/acrobot_mjpc_reference/CMakeLists.txt) define:

1. Create-only isolated build: **1,200 wall seconds, four CPU cores, 2 GiB output**. Compile unchanged official core/task/two-planner files against the existing MuJoCo wheel SDK. Fetch pinned Abseil only; no package/environment change. C++20, CMake, compiler, threads and the existing MuJoCo SDK are required. This omits unrelated GUI/gRPC/Menagerie build targets. [Official build](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/CMakeLists.txt), [headless precedent](https://github.com/google-deepmind/mujoco_mpc/blob/ff572a21e7c2bf9fda62e1862a758da7e9a8719b/mjpc/testspeed.cc).
2. Preserve one original unsupported-RK4 invocation. Validate adapted A/B/C/D at 16 declared deterministic held-out states against separate native central probes, epsilon 1e-5. Adapter remains forward epsilon 1e-6; maximum absolute block error ≤1e-4. Failure stops without changing tolerance.
3. Both planners, five starts, 10 seconds each: 500 executed controls, 1,000 native steps, 250 planning refinements per episode. Native stage including validation: **1,200 wall seconds, four CPU cores, 10,000,000 counted step equivalents, 2 GiB new output**. Static call census is approximately nine million; no measured timing is claimed yet.
4. Stop on source/model/runtime mismatch, failed derivatives, nonfinite execution, failed evidence writes or resource exhaustion. Ordinary task failures complete the roster. Count all rollout and derivative steps.
5. Save every exposed post-optimization Sampling trajectory and final iLQG line-search trajectory: states, controls, residuals, times, costs, failure flags. Earlier iLQG nominal-phase arrays are overwritten internally and are not falsely claimed retained. Save all 501 execution states and per-control work counts.
6. Report peak height, first crossing, final two-second minimum, strict final-height >3.8 m success and both costs separately. This limited roster is not the standard dm_control random-start benchmark, minimum-time optimum or SOS certificate. No tuning or learning follows automatically.

Protocol/source bindings must be committed before build. Worker output is create-only under /home/jaisel/mppi-gps-codex-acrobot/2026-11-05_external_reference_v1_0_001/mjpc/. A failed build is a compatibility/infrastructure result, not a failed teacher.

## Version 1.1 compatibility repair

The first isolated build failed after 27.318854 wall seconds with zero physics: current MJPC uses the old eight-argument mj_ray call in its Ground helper; MuJoCo 3.6 requires a ninth normal-output argument. The original build logs and receipt remain unchanged. Version 1.1 supplies an inline compatibility overload passing a null normal pointer, without modifying any downloaded source or Acrobot dynamics. Ground is unused by Acrobot. The fresh rebuild has at most 1,172 wall seconds, so total builds remain below the original 1,200-second ceiling. The native budget, inputs, planner recipe, costs, derivative gate and roster are unchanged.

Version 1.1 then compiled the entire official core, but the final driver failed: the installed 3.6 derivative signature uses mjtByte, whereas newer web documentation uses mjtBool. Version 1.2 corrects this external declaration to the exact SDK type. The failed build and its 31.426095 wall seconds remain preserved, with zero physics. Remaining build budget is 1,141 seconds; all scientific inputs and rules remain unchanged.

Version 1.2 built successfully in 32.780863 seconds. Its first prerequisite stopped before any physics because the external driver incorrectly assumed an old version-number encoding (360); the installed SDK and library both report 3006000. Version 1.3 corrects this binding implementation to require exact version string 3.6.0 and header/library integer equality. This does not relax a physical gate. Its fresh rebuild has 1,108 wall seconds; remaining native wall budget is 1,199 seconds after preserving the 0.274334-second, zero-step failed prerequisite.

## Version 1.3 completed result

All ten episodes completed. Sampling held none of the four historical near-upright starts; iLQG held one. Neither swung up from the preserved hanging start: peaks were 2.75784 m (Sampling) and 2.20380 m (iLQG). The derivative adapter passed all 16 held-out checks, with worst A/B/C/D errors 3.6542e-7 / 2.2204e-10 / 9.9487e-7 / 3.3307e-11. Native execution cost 15.295992 wall seconds and exactly 9,024,006 counted steps. Thus the chosen external recipe is not yet a working swing-up reference on the identical runtime. It establishes neither infeasibility nor a unique failure cause.

**Independent-review validity finding:** the initial adapter always nudged control forward, including at the +1 actuator limit, where clipping can produce a zero B column. Official MuJoCo chooses a feasible inward difference at this boundary. The 16-state gate used only interior actions and missed this defect. The saved version 1.3 iLQG outcomes are therefore outcomes of the flawed adapter, not a valid rejection of the requested MJPC reference. Sampling is unaffected. All original outputs remain unchanged. Version 2 was held before build or physics when this finding arrived. [Official range-aware finite differences](https://github.com/google-deepmind/mujoco/blob/3.6.0/src/engine/engine_derivative_fd.c).

## Version 2: required standard-start metric

The historical 50 Hz roster does not answer the user's separate request for the standard dm_control metric on standard starts. The parent authorized this separately frozen completion stage, not a resumption or tuning of version 1.3.

- Eight exact reset seeds 2026110500 through 2026110507, each using RandomState(seed).uniform(-pi,pi,2), zero velocities and full zero ancillary context. Both planners use the same reset contexts.
- Control is now the standard 100 Hz, held for one 0.01-second RK4 step; replanning remains 25 Hz, every four native steps. This changes control timing and roster together to the external standard and is not a causal one-variable comparison with version 1.3.
- Model, residual cost, horizon, spline/noise recipe, planner algorithms and validated derivative adapter remain unchanged. The source/model/derivative validation is reused without repeating validation physics.
- Limits: fresh build at most 120 wall seconds within the original remaining build allowance; native stage at most 300 wall seconds, four cores, 16,000,000 counted steps and 2 GiB output. Static census is 14,438,000 steps. No tuning or follow-up follows failures.
- Reward is computed after each of 1,000 native steps using the official dense swingup tolerance: target radius .2 m, margin 1 m, Gaussian sigmoid and value .1 at the margin. Report reward mean/sum and final two-second height hold separately. Eight starts provide a small standard-distribution check, not a full published benchmark.

Official metric/reset source is pinned dm_control f1f91e0309e5df81bf7a3ee83849ebcee86779d8. Exact rewards.py SHA256 is 009f02389fd2af84b2955324707638e79a0e18b6a540dd7c254abf5096e157f3. [Acrobot task](https://github.com/google-deepmind/dm_control/blob/f1f91e0309e5df81bf7a3ee83849ebcee86779d8/dm_control/suite/acrobot.py), [tolerance implementation](https://github.com/google-deepmind/dm_control/blob/f1f91e0309e5df81bf7a3ee83849ebcee86779d8/dm_control/utils/rewards.py), [frozen version 2 recipe](../configs/acrobot_mjpc_reference_v2.json), [exact bindings](../configs/acrobot_mjpc_reference_bindings_v2.json).

## Version 2.1: restore the derivative contract before execution

Version 2 was never built or run. Independent source review found the control-boundary defect described above. The corrected adapter now follows official MuJoCo 3.6 forward/backward feasibility, including centered behavior, without changing any model, cost, horizon, noise or optimizer setting. Forty exact frozen cases (eight states, each at controls -1, -1+5e-7, 0, 1-5e-7, 1) test forward and centered derivatives against independent central or feasible three-point one-sided probes. All A/B/C/D matrices and complete native probe inputs/outputs are retained. The first five C/D rows are also checked against analytic pre-action residual Jacobians, closing a possible shared sensor-timing error; all gates retain tolerance 1e-4. Official RK4 skips sensor recomputation at internal stages, so after-step sensor arrays correctly preserve the pre-action map.

The parent extended the explicit native cap to 21,000,000 steps, 300 wall seconds, four cores and 2 GiB to include the 16 standard-start episodes plus five corrected historical iLQG episodes. Previously valid Sampling episodes are reused, not repeated. The latter isolate the derivative correction on the exact old inputs/control timing. Planned work is 20,957,830 steps, including 1,080 gate probes; the measured previous 15.296 seconds for 9.024 million steps supports the finite wall cap. Fresh build remains at most 120 seconds within the original remaining build allowance. Both header and native library now have exact SHA256 bindings checked before build. Existing flawed iLQG data are labelled INVALID_EXTERNAL_REFERENCE_CONTRACT and kept. No further tuning or follow-up is authorized.

## Corrected version 2.1 result

All 40 frozen cases passed in both differentiation modes, including the analytic pre-action C/D check. All 21 planned episodes completed within 34.957279 native wall seconds and exactly 20,957,830 counted integration steps; build took 32.510074 seconds. On eight standard-distribution resets at 100 Hz, Sampling held 0/8 and had mean dense swingup reward 0.0429756942; corrected iLQG held 3/8 and had mean reward 0.2821176154. The latter held seeds 2026110500, 2026110504 and 2026110507. The corrected historical iLQG comparison still held only 1/4 near-upright starts, and its hanging-start peak was 2.158531 m.

This establishes that the validated external planner can stabilize some starts on the identical torque-limited model. It does not provide a reliable hanging-start swing-up teacher, a minimum-time/cost optimum or the claimed full published benchmark. The original derivative defect remains in the causal record. Independent audit of all saved measurements and probes is pending; no further physics or tuning was launched.

Raw output is retained on the worker in preparation/execution_v2_1. The local compact archive is 4,249,600 bytes, SHA256 ac00122539ea0ac567708e716b7871fa69830572b5bf6a2bf156520b88f0581d. It includes all execution traces, matrices, probe inputs/outputs, receipts and raw-array checksums; the 21 candidate banks total 1,015,140,000 bytes and remain outside Git on the worker. The native counter includes all mj_step calls, not a separate count of direct mj_forward calls; wall time includes the complete process. Per-process CPU time was not measured and must not be invented.
