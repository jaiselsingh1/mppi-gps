# MuJoCo MPC on the identical Acrobot

Status: source reviewed and preparation frozen; no native reference result yet.

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
