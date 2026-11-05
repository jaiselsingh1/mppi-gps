# Identical-model Drake reference and funnels — v1.0

Prepared 2026-11-05 on `codex/full-trajectory-pi`. This records an externally sourced reference, not a claim that our failed Acrobot teacher is correct. No learning loop is authorized here.

## What the external reference actually establishes

Our XML is upright-zero, has one-metre capsule links, centre inertias `0.0955703125 / 0.09530365719649561`, joint damping `0.05 / 0.05`, and physical elbow torque `tau=2u`, `|tau|<=2`. Drake's analytical `AcrobotPlant` can represent these parameters exactly. Coordinates map as `theta1=q0+pi, theta2=q1`, with unchanged velocities. Vertical translation and constant potential-energy offset do not change dynamics. [Official plant source](https://github.com/RobotLocomotion/drake/blob/1610676d38df7854153cd04e1ba32e286fd617b1/examples/acrobot/acrobot_plant.cc).

The stock Drake example is **not an identical-model working reference**: defaults have `l2=2, lc2=1, Ic1=.083, Ic2=.33, b1=b2=.1`. More seriously, `AcrobotSpongController` privately creates its own default plant and LQR, and clips at **20 Nm**, ten times our limit. Changing the external plant's parameters does not change that internal controller. Its official gains are `ke=5,kp=50,kd=5`, with balance `Q=diag(10,10,1,1), R=1` on physical torque. The source explicitly cautions that gains working for defaults need not work for other parameters. [Controller source](https://github.com/RobotLocomotion/drake/blob/1610676d38df7854153cd04e1ba32e286fd617b1/examples/acrobot/spong_controller.h), [gain defaults](https://github.com/RobotLocomotion/drake/blob/1610676d38df7854153cd04e1ba32e286fd617b1/examples/acrobot/spong_controller_params.h).

The source-derived port therefore uses our exact model, computes a new CARE balance gain with those declared Q/R, retains the official energy/PFL equations and gains, clips physical torque at 2 Nm, and switches to balance only inside its computed local invariant level. It is an **adapted external method**, not an already validated reference or a proof that energy shaping solves this torque limit. The energy term is `-ke*(E-Eupright)*v2`; PFL contributes `(M_inv[0,1]*bias[0] -kp*q1-kd*v2)/M_inv[1,1]+bias[1]`. Their sum and saturation do not inherit the energy-only monotonicity argument. [Underactuated Acrobot derivation](https://underactuated.mit.edu/acrobot.html#section6).

## Frozen preparation and native check

Implementation: `src/gps/mppi_gps/acrobot_drake_reference.py`. Reference source download is pinned to Drake commit `1610676d38df7854153cd04e1ba32e286fd617b1`; seven files, 62,908 bytes, in the preparation evidence directory. Installation pins **drake==1.58.0** in a separate Python 3.11 environment. The installed-wheel version and source binding are recorded independently; a different wheel must not be substituted silently.

1. Install only: at most 600 wall seconds and 2 GiB environment/download storage. No existing environment mutation. Stop on unavailable package/solver.
2. Preparation: one CARE solve and one quadratic CSDP feasibility solve; at most 1,200 wall seconds, two CPU cores, 1 GiB preparation output. Compare the parameterized analytical plant and the bound native EOM/linearization before any controller execution. Source export and output directory are create-only. No gain search.
3. Native controller check, coordinated with the root's identical-model verifier: four preserved failed near-upright contexts with balance alone and the two preserved discovery starts with hybrid energy/balance; 10 seconds each, 500 controls/1,000 physics steps each. This roster is at most 6,000 physics steps, within the separately reserved **100,000-substep hard ceiling**. Use the existing complete-context restoration and RK4 `.01`/two-substep control path. Preserve every state, normalized action, unclipped demand, energy and switch time. Do not retry or tune after failure.
4. Pass/kill: bindings, finite physics, physical torque bound and evidence integrity must pass; stop on any failure. Report the fixed roster's final-two-second height>=3.8m holds separately from entry into the invariant level. A positive reference requires both discovery starts to join and hold; otherwise report this adapted controller failed. No negative result launches a new controller automatically. Controller success does not prove time optimality.

The parent protocol must record exact context hashes, source export identity and native-runtime versions before step 3. The installation and preparation bounds above do not authorize extra native tests.

## What the SOS claim can and cannot be

`RegionOfAttraction` is designed for polynomial dynamics, or rational polynomial dynamics in implicit form. Its own documentation warns that numerical coefficient/cone checks are **not a rigorous inner-approximation certificate**. Raw sine/cosine dynamics and saturated wrapped-angle feedback cannot simply be handed to a polynomial example. [Drake API](https://drake.mit.edu/pydrake/pydrake.systems.analysis.html#pydrake.systems.analysis.RegionOfAttraction).

The exact local proof implemented here avoids silently changing the controller. Work within `||x||<=r<pi`, where wrapped-angle feedback is exactly `tau=-Kx`, and enforce `||K||_1*r<=1.8 Nm`, leaving saturation inactive. Let `P` be the saved symmetric CARE matrix and `Qcl=-(Acl'P+P Acl)`. Write exact nonlinear acceleration as `Acl_bottom*x + e(x)`. With `a=m2*l1*c2`, `M0=M(0)`, `d=2a`, `mlo=det(M0)/trace(M0)`, and `L=M0^-1*(linear force)`, the elementary bounds `|sin(z)-z|<=|z|^3/6`, `|cos(z)-1|<=z^2/2` give

`||e(x)|| <= B(r)||x||^3`,

`B(r) = [g*(m1*c1+m2*l1+8*m2*c2)/6 + 2a + d*sum(abs(L))] / [mlo-d*r^2]`.

For clarity, the Euclidean triangle bound is `||gravity_remainder|| <= g*[shoulder*|q0|^3 + sqrt(2)*end*|q0+q1|^3]/6 <= g*(shoulder+4*end)*||x||^3/6`; the implementation deliberately uses the larger `shoulder+8*end` constant.

Thus `Vdot <= -x'Qcl*x + gamma*||x||^2`, where `gamma=2*sum(abs(P[:,2:]))*B(r)*r^2`. Halve `r` from 0.1, at most 40 times, until `gamma<=qlower/2`, where `qlower=1-||CARE_residual||infinity>0`; require positive denominator. Solve the quadratic SOS/PSD problem for Gram `Qcl-gamma*I`. Independently check its **exact rational LDL decomposition**, rather than relying on CSDP's tolerance. Binary64 model/gain/value coefficients are converted to exact rational values; save every coefficient, diagonal pivot, remainder bound and solver residual. `rho=r^2/||P^-1||infinity` ensures `{x:V<=rho}` stays within the ball, so the bound proves local invariance and convergence for that nominal continuous EOM. It does not maximize the region, certify energy-shaped swing-up, or certify a globally optimal controller.

Most critically, this is a **continuous-feedback** statement. Native deployment samples every .02 seconds and numerically integrates held actions. Successful native tests supplement the certificate; they do not extend its mathematical claim to sample-and-hold/RK4. A sampled-data formal certificate remains unresolved. Trigonometric lifting (`s_i,c_i`, circle equalities), implicit manipulator constraints, and actuator-region partitions are valid routes to larger exact-model SOS programs, but raw linear angles are not polynomial functions of their lifts; Taylor models need explicit remainder bounds. [Official Lyapunov treatment](https://underactuated.mit.edu/lyapunov.html#section5).

The frequently cited torque-limited Acrobot SOS demonstration used an identified hardware model, **5 Nm** limit, degree-three Taylor dynamics, 200 Hz sensing, quadratic Lyapunov functions and a cubic balancing controller. It is evidence that the methodology can work, not an existing certificate for our XML/2 Nm/50 Hz controller. It explicitly handles saturation by separate lower/unsaturated/upper derivative inequalities, or conservatively confines the funnel to unsaturated input. [Majumdar, Ahmadi and Tedrake 2013, Sections IV and VI](https://groups.csail.mit.edu/robotics-center/public_papers/majumdar13.pdf).

## Funnels unify transient and attractor

A stabilized nominal `xbar(t),ubar(t)` with TVLQR is a **candidate** funnel. A verified funnel additionally supplies `F(t)={x:V(t,x)<=rho(t)}` and a proof that `Vdot-rhodot<0` on its boundary, including input constraints and the actual controller. Terminal containment in another verified funnel is the join condition. A nearest geometric pass alone is not that condition. [LQR-Trees 2010](https://groups.csail.mit.edu/robotics-center/public_papers/Tedrake10.pdf), [official composition rules](https://underactuated.mit.edu/feedback_motion_planning.html).

The upright fixed point is a stationary terminal funnel (`V,rho` independent of time); a limit cycle adds a phase-dependent terminal tube and contraction transverse to phase, with periodic consistency. The transient is another time-varying funnel feeding that terminal object. Attractors do not add a distinct control principle; they supply reusable infinite-duration terminal behavior. Existing Walker `best_stride/Section`, `locked/periodic_gains`, `cycle_walk` near-pass join, and `build_cycle` map to nominal attractor selection, local stabilization, empirical joining, and composition. They are LQR-stabilized trajectories with empirical checks, **not certified LQR-Trees funnels** until these invariance/containment conditions are established. Acrobot replaces stride finding and phase locking with upright equilibrium and constant gain; it does not replace funnel verification with an ellipsoid plot.

## Dependency-only amendment v1.1 (before preparation)

The frozen v1.0 install attempt failed because Drake 1.58.0 has no wheel compatible with the existing worker's Python3.11; it ran zero CARE/SDP solves and zero physics. This is an environment compatibility failure, not a controller result. v1.1 pins **Drake1.51.1** in a distinct create-only environment. Its official v1.51.1 `acrobot_plant.cc`, parameter definitions, `spong_controller.h` and controller-gain definitions are byte-identical to the four corresponding retained HEAD sources. No dynamics, gain, limit, target, join rule or certificate formula changes. The exact hashes/diffs are retained in `v1_51_1_source_comparison.json`. `configs/acrobot_drake_reference_v1_1.json` binds the unchanged finite budget and historical context configuration. The original failed installation is retained as `install_failure_v1_0.json`.

### API-only repair v1.1.1

v1.1 setup succeeded in10.309wall seconds (634,821,725bytes), then preparation stopped before CARE/SDP because the Python binding omits `set_l2`. That failed attempt used0.19wall seconds,0.20CPU seconds and119,520KiB peakRSS. v1.1.1 assigns the same1.0m value through documented `AcrobotParamsIndices.kL2=3` using BasicVector; all model/controller/certificate values remain unchanged. Introspection verified all remaining parameter setters, vector round-trip, CSDP availability and program construction without solving. The repaired preparation also checks its Drake upright Jacobian against the already-validated analytic same-model Jacobian at1e-10 before CARE. Failed log and APIreceipt remain preserved.

### Nonlinear same-model prerequisite

Before the frozen six native episodes, `acrobot_drake_parity` evaluates the parameterized Drake plant's continuous EOM on exactly the32already-paid collocation/native parity states and physical torques (`collocation_v1_0/native/parity.json`). Compare to this implementation's mechanical EOM at `rtol=0,atol=1e-9`; preserve every derivative and input hash. This is32zero-time Drake EOM evaluations,32analytical EOM evaluations, zero new integration/physics and no repeated CARE/SDP. Failure prevents native controller execution. The prior native one-RK4-step checks remain unchanged and are reused.

## Executed result (v1.1.1)

Both equation checks passed: upright A/B maximum error **1.42e-14**, and all32 reused nonlinear parity states passed at maximum derivative error **2.84e-14**. Thus the parameterized Drake model agrees with the analytical EOM already checked against native MuJoCo; the stock example's model/torque differences were not carried into this test.

One CARE solve and one CSDP solve produced `K_physical=[-315.9780984,-80.23087395,-116.1332567,-34.52941223]`. CSDP's maximum coefficient residual was **8.73e-11**, minimum Gram eigenvalue **0.85018**. Independent saved-fraction verification passed **22/22** checks. The provable local level is extremely conservative: `r=6.103515625e-6`, `rho=1.6401953694e-12`, torque upper bound **0.00333784Nm**. This is a valid small continuous-feedback positive control, not evidence of a useful recovery basin or a sampled-data certificate.

| Frozen start | Controller | Height hold, final2s | Peak height (m) | Native dense cost | Torque clipped |
|---|---|---:|---:|---:|---:|
| 3022000000 | Balance | Fail | 3.9850 | 4877.3526 | 100.0% |
| 3022000001 | Balance | Pass | 4.0000 | 0.7694 | 0.2% |
| 3022000002 | Balance | Fail | 3.9957 | 4548.2313 | 99.8% |
| 3022000003 | Balance | Fail | 3.9983 | 3216.7459 | 99.4% |
| 3022000100 | Energy/PFL + balance | Fail | 2.0696 | 5488.9055 | 99.6% |
| 3022000101 | Energy/PFL + balance | Fail | 2.0432 | 5432.6995 | 99.8% |

The adapted official-gain reference therefore **does not supply a working swing-up ground truth under2Nm**: neither discovery episode entered the local certified region, and both failed the unchanged height hold. One of four balance starts succeeded; the other three spent nearly their entire episodes saturated. This establishes persistent actuator saturation in the failed reference, not that the task is infeasible, that no better gain/controller exists, or that the same-model collocation optimum is known. No gain tuning, new starts, physics reruns or learning followed this negative result.

Raw evidence: `runs/2026-11-05_acrobot_external_reference_001/drake_preparation_001/` contains all equations, preparation, native HDF5 traces, outcomes, failed setup/API attempts, source copies, hashes and cost receipts. `drake_certificate_audit_001/` holds the independent fraction audit. Native measurements are being reviewed separately from the continuous proof.

Cost appendix: setup10.309wall seconds and634,821,725bytes; successful preparation0.26process-wall seconds/0.38CPU seconds/143,080KiB peakRSS; nonlinear parity0.19wall/0.24CPU/117,656KiB; six native episodes0.49process-wall/0.54CPU/103,912KiB. Native execution used **3,000 controls/6,000 physics substeps**, versus the6,000-substep frozen allowance. The failed API attempt consumed0.19wall/0.20CPU seconds, zero solves/physics. The earlier unavailable-wheel attempt's exact process timing was not retained and is marked missing. Retained study output is approximately1.15MB, excluding the isolated environment. Full inventory/accounting is `drake_preparation_001/accounting.json`; no claim of zero engineering or transfer overhead.

## Source-faithfulness correction v2.0 — frozen before execution

Independent design review identified a scientific mismatch in v1.1.1: its hybrid used the tiny certified terminal level (`rho=1.64e-12`) as a balance switch, whereas the official source switches at **V<1000**. Thus the old0/2 swing-up result is for a strict-join adaptation, not a faithful official hybrid with the real torque cap. Preserve that result, all traces, proof, costs and audit failures unchanged. This correction is not gain tuning or a new parameter sweep.

v2.0 restores precisely the official threshold1000 and strict inequality. Reuse the saved same-model K/P (no CARE/SDP repeat), official energy/PFL gains, physical2Nm cap, two historical discovery contexts, native runtime,50Hz rate and10-second duration. `configs/acrobot_drake_reference_v2_0.json` binds the old preparation/config hashes and a create-only output. Bound: exactly2episodes, at most1,000controls/2,000physics substeps,60wall seconds and16MiB output. No new starts, optimization, fitting or additional controller variant. Stop for invalid bindings/physics/evidence or exhausted bounds. Height>=3.8m for the final2s remains the success test; all failures remain outcomes. Save every joint angle, action, raw torque, cost, balance mode and handoff. Report clipping counts and switch entries/exits, and compare the first changed actions to the saved strict-join traces.

Switching outside the certified tiny terminal set has **no global Lyapunov guarantee**. The existing SOS proof is unchanged and does not certify the restored threshold or sampled-data hybrid. The earlier native audit's one-ULP cross-host local-value discrepancy remains recorded; no tolerance is changed by this source correction. No follow-up launches automatically.

### v2.0 result: official handoff restored

The corrected hybrid completed both fixed10-second episodes: **0/2 height holds**, with peaks **2.06955/2.04318m** and dense native costs **5488.90546/5444.98412**. Clipping occurred on **498/500** and **499/500** controls. The first episode never reached V<1000 (minimum2784.21), so its states and actions are byte-identical to the preserved strict-join run. The second used balance on controls194,195,196,215,287,379: four entries and four exits. Although its first handoff was194, saturation made the deployed actions identical until287; six controls changed, with maximum absolute normalized-action difference2.0. Cost increased12.28466 and joint-angle trace differences reached0.08349/0.29883rad. Its closest V-to-switch margin was14.47, far above the prior cross-host rounding discrepancy.

This corrects the external-reference claim: the official-source hybrid, adapted to the identical model and2Nm, still failed these two swing-up starts. The small certified region did suppress handoffs in the earlier version, but restoring them did not produce swing-up here. This is a failure of these fixed source-derived gains/hybrid at the real limit; it is not evidence that Acrobot or all energy-based controllers are infeasible. No further variants were run.

Frozen source3b11514, configSHA35c778b81248638237f9da9393e49c87ce3cfbfacc430193206be6640469ef57. New execution used exactly **1,000controls/2,000physics substeps**,0.30process-wall/0.31CPU seconds and103,552KiB peakRSS; no CARE/SDP repeat. Raw traces include every joint angle, control, unclipped torque, energy and handoff. The saved comparison and cost receipt are `drake_preparation_001/drake_native_v2_0/comparison_to_strict_join.json`. Original native audit failure is retained and has not been relabelled as an unqualified pass.
