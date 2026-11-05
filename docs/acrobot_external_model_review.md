# Acrobot external model and saved-bank review

The physical XML matches DeepMind Control's Acrobot. That does **not** establish that our controller implements its task: our action period, starts, optimization cost and success test differ. The evidence reviewed so far validates geometry, energy accounting and the upright local Jacobian; it does not validate global MPPI search, establish that swing-up is impossible, or provide a nonlinear region-of-attraction certificate.

## Source identity and task differences

This review pins `google-deepmind/dm_control` at [`f1f91e0309e5df81bf7a3ee83849ebcee86779d8`](https://github.com/google-deepmind/dm_control/tree/f1f91e0309e5df81bf7a3ee83849ebcee86779d8). Exact downloaded source, SHA256 manifest and the structural XML comparison are preserved in `runs/2026-11-05_acrobot_external_reference_001/model_review_001/`. No simulator was imported or run for this review.

After removing our added `framepos` sensor `tip_pos`, parsed XML tags and attributes are identical to the [official Acrobot XML](https://github.com/google-deepmind/dm_control/blob/f1f91e0309e5df81bf7a3ee83849ebcee86779d8/dm_control/suite/acrobot.xml). The included `visual.xml`, `skybox.xml` and `materials.xml` are byte-identical. The added sensor changes observation availability, not the force model. Our XML SHA256 is `324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe`; byte identity of the complete XML is neither asserted nor required for this structural comparison.

| Item | DeepMind Control default | Our failed teacher runs |
|---|---|---|
| Native model | Two 1 m capsule links, 1 kg each; shoulder unactuated, elbow motor | Same physical XML |
| Motor | normalized `u` in [-1,1], gear 2 | Same: elbow torque `2u`, at most ±2 N m |
| Physics | RK4, 0.01 s, constraints disabled | Same |
| Action period | One native step, 0.01 s (100 Hz) | `frame_skip=2`: 0.02 s (50 Hz) |
| Starts | Shoulder and elbow independently uniform [-π,π); reset default velocities | Frozen hanging-neighbourhood roots or four selected near-upright contexts |
| Episode | 10 s by default | 10 s near-upright; 12 s energy discovery |
| Dense reward/cost | `tolerance(distance, bounds=(0,0.2), margin=1)` | Squared tip distance plus 0.01 velocity/control penalties, or separately frozen energy objective |
| Success | Suite supplies per-step reward; no 2 s hold gate | Every final 2 s post-control height ≥3.8 m; later LQR-entry gate |

Default action period follows [Environment's one-substep default](https://github.com/google-deepmind/dm_control/blob/f1f91e0309e5df81bf7a3ee83849ebcee86779d8/dm_control/rl/control.py#L58-L71). Starts, reward and time limit follow [the official domain implementation](https://github.com/google-deepmind/dm_control/blob/f1f91e0309e5df81bf7a3ee83849ebcee86779d8/dm_control/suite/acrobot.py). Our energy objective also uses a proximity margin of **2**, not the official dense reward's margin **1**. Neither our height gate nor our energy cost is the `acrobot-swingup` metric. A future external comparison must report the native suite reward on its standard starts as a separate measurement rather than relabel the historical gate.

## Exact equations in our coordinates

Let `q=(q0,q1)` be shoulder and relative elbow angles, and `v=qdot`. Both links extend along +z at `q=(0,0)` (upright); `(π,0)` is the hanging configuration. Time is the first element of the saved five-element state and is excluded from the mechanical state. The tip is

\[
x=\sin q_0+\sin(q_0+q_1),\quad z=2+\cos q_0+\cos(q_0+q_1).
\]

The parameter packet preserved in `runs/2026-11-05_acrobot_energy_discovery_001/preflight/energy_validation.json` contains masses 1 kg, COM offsets 0.5 m, upper-link length 1 m, principal hinge-axis inertias `I1=0.0955703125`, `I2=0.09530365719649561` kg m², damping 0.05 N m s/rad on each joint, zero armature and `g=9.81` m/s². These are the existing native-model parameters, not a fresh upstream compilation in this read-only review. Source identity plus the existing parameter/energy checks supports the mechanical correspondence; it does not substitute for the requested independently executing references.

\[
M(q)\ddot q+c(q,v)+0.05v+\nabla V(q)=\begin{bmatrix}0\\2u\end{bmatrix},
\quad
M=\begin{bmatrix}
1.6908739696964956+\cos q_1&0.3453036571964956+0.5\cos q_1\\
0.3453036571964956+0.5\cos q_1&0.3453036571964956
\end{bmatrix},
\]
\[
c=\begin{bmatrix}-0.5\sin q_1(2v_0v_1+v_1^2)\\0.5\sin q_1v_0^2\end{bmatrix},\quad
V=9.81[4+1.5\cos q_0+0.5\cos(q_0+q_1)],
\]
\[
\nabla V=-9.81\begin{bmatrix}1.5\sin q_0+0.5\sin(q_0+q_1)\\0.5\sin(q_0+q_1)\end{bmatrix}.
\]

The hanging-to-upright energy difference is **39.24 J** (`58.86−19.62`). The saved 32-state energy comparison reports maximum error `1.1368683772161603e-13` J, against its frozen `1e-10` tolerance. This checks the energy calculation on those states, not global closed-loop correctness.

The upright continuous linearization, with input **normalized** `u`, is

\[
A=\begin{bmatrix}0&0&1&0\\0&0&0&1\\
12.247304804&-11.426622567&-0.080441479&0.196920812\\
-15.776524308&42.177270764&0.196920812&-0.626862410\end{bmatrix},\quad
B=\begin{bmatrix}0\\0\\-7.876832494\\25.074496414\end{bmatrix}.
\]

The existing implementation differentiates two held-input RK4 steps, rather than using an exact matrix exponential for the 0.02 s action map. In `runs/2026-11-05_acrobot_attractor_001/recovered-lqr/lqr.json`, analytic-vs-native finite-difference discrepancies are at most `1.2421175199506251e-12`; held-out local prediction error is `6.776673541980874e-15`. This is evidence against a sign, input-gear or upright-Jacobian error in the tested implementation.

## What the torque limit does and does not show

At a motionless configuration, static balance requires **both**

\[
1.5\sin q_0+0.5\sin(q_0+q_1)=0,
\quad \tau_\mathrm{elbow}=-4.905\sin(q_0+q_1),\quad |\tau|\le2.
\]

Thus the elbow condition alone requires `|sin(q0+q1)|≤0.407747`, corresponding to a local absolute lower-link angle ≤`0.41998548` rad. The unactuated shoulder equation must also hold. This is a necessary **static** condition, not a dynamic capture limit. At exact upright the equilibrium torque is zero. Dynamic swing-up and capture can traverse states that cannot be held statically.

The saved discrete LQR requests far larger torques at three of the four historical near-upright roots:

| Historical root | Unclipped LQR torque (N m) | Native available torque (N m) | Elbow static gravity term (N m) |
|---|---:|---:|---:|
| 3022000000 | 40.0913 | +2 | −0.7717 |
| 3022000001 | −3.6947 | −2 | +0.1483 |
| 3022000002 | −29.5976 | −2 | +0.1803 |
| 3022000003 | 24.8652 | +2 | +0.0459 |

All four initial LQR requests saturate. These are demands of **that unconstrained LQR**, not the minimum torque needed by any controller. The small empirical capture region is consistent with saturation but does not show that a torque-aware reference cannot capture these states. The empirical `rho=0.11064310893419543` is not an SOS certificate or a proof of nonlinear invariance; its coordinate projections are not a capture box.

There is also a simple continuous-mechanics time lower bound, independent of our
planner. Energy obeys
`Edot=tau*v1-.05*(v0^2+v1^2) <= 2*abs(v1)-.05*v1^2 <= 20 J/s`.
Exact hanging rest to exact upright rest requires 39.24 J, so its time is at
least **1.962 s**. This loose necessary bound supplies neither a feasible
trajectory nor an optimum; actual acceleration, unactuated-shoulder dynamics
and losses make it non-tight. It is a statement about the nominal continuous
equations, not a formal energy bound for every numerical RK4 trajectory. Since
no collocation attempt passed feasibility, this investigation currently has no
valid optimal-time upper bound or cost optimum to pair with it.

## What the saved candidate evidence actually contains

Raw sources are preserved unchanged. No missing states are synthesized in this review.

- Every completed root episode records executed actions and costs, pre-action observations, and the `N+1` simulator states/warmstarts/integration contexts. The earlier interrupted near-upright file also retains its partial third episode and incomplete final bank.
- The energy-discovery bank callback saves **every candidate's scalar cost, weight, aggregate cost components, terminal energy data, peak height, final-window minimum height and height qualification**. It saves full predicted state/action traces for **two selected roles** per bank: minimum task cost and maximum height. Those roles can select the same candidate.
- It also saves the bank's full-action-array SHA256, shape/dtype, planner RNG-after state and full pre-context. It does **not** save all candidate state/action arrays.
- The later ESS32 near-upright and swing-development recordings save **every candidate's cost and weight**, nominal before/after, action-array SHA256/shape/dtype, RNG-after state, temperature and ESS. They do **not** save per-candidate predicted states or per-candidate height/energy components.
- For the declared white-noise recipes, complete candidate actions can be reconstructed without physics from the frozen private PCG64 stream, Gaussian sampling recipe, saved nominal and clipping rule, then verified against each saved SHA256. Reconstructing unretained candidate states would require new native execution, which this review does not perform.

Consequently, the user's request to inspect every candidate's **states** at failure cannot be fully satisfied from preserved arrays. Scalar/weight and action-reconstruction checks can be exhaustive; unrecorded predicted states remain missing. The explicit file/bank roster and bounded saved-only scan are recorded separately under `saved_scan_001`.

## Interpretation and next discriminant

The simple "squared distance settles at a two-metre pose" account needs care.
At `(q0,q1)=(pi,pi)` the tip is at 2 m, torque zero is an equilibrium and the
pose cost is four. Its pose Hessian is `[[0,4],[4,6]]`, with eigenvalues **-2,8**:
it is a stage-cost **saddle**, not a demonstrated local minimum of the finite-
horizon control objective. At the other folded equilibrium `(0,pi)`, the cross
terms change sign and the eigenvalues remain -2,8. That does not guarantee
MPPI can exploit the descending direction through underactuated dynamics.

Saved-only inspection of every last-100 post-action state from all ten original
dense qualification episodes gives **42/1,000** points within .05 rad of both
angles of either stationary folded pose. Episode-median speed norms range
**.6183–2.4756 rad/s**. Failure commonly occupies the folded neighbourhood but
does not generally settle motionless at its exact equilibrium. This contradicts
the literal stationary-pose account; it does not rule out a cost/horizon/search
interaction. Protocol, all points and episode-level results are preserved under
`folded_pose_review_001`; there is no new inference, physics or fitting.

No model substitution, incorrect torque-unit conversion or wrong upright Jacobian has been found in these checks. That narrows particular implementation errors; it does **not** establish overall planner correctness. The difference between the official task and our gate, low-temperature weight collapse, short-horizon search, initialization and cost design remain distinct explanations. Adaptive weighting already removed ESS≈1 yet failed to establish holds/capture; weight collapse alone is therefore insufficient as the complete explanation.

An independently executed, torque-limited reference on this exact mechanical model is needed before attributing the failure to the task. If it works, a one-piece-at-a-time comparison can identify a breaking change. Until then, there is no working reference relative to which a single MPPI defect has been isolated, no certified funnel, and no basis for launching the learning loop.

## Exhaustive saved-bank measurements

The create-only saved-scan protocols and all review failures are preserved under `runs/2026-11-05_acrobot_external_reference_001/saved_scan_001/`. Final receipt: `result_r4.json`; detailed rows: `banks_r2.jsonl.gz` plus `banks_r4_tail.jsonl.gz`. The compressed prefix is a verified lossless preservation of the original emitted bytes (`lossless_preservation.json`). All measurements below are arithmetic on existing arrays; no new physics or fitting was performed.

| Retained source | Complete banks | Candidate costs/weights | Exact regenerated action hashes | Predicted state/action trace slots retained | Median ESS |
|---|---:|---:|---:|---:|---:|
| Original dense teacher | 8,032 | 2,056,192 | No historical full-action hash | 0 | 1.000230 |
| Old near-upright, including partial third episode | 1,436 | 367,616 | 1,436 / 1,436 | 2,870 | 1.001852 |
| Old near-upright completion | 1,004 | 257,024 | 1,004 / 1,004 | 2,008 | 1.001537 |
| Energy variants | 3,612 | 1,849,344 | 3,225 / 3,612 | 7,224 | 1.000000 |
| ESS32 near-upright | 2,008 | 514,048 | 2,008 / 2,008 | 0 | 32.000000 |
| ESS32 swing development | 1,204 | 616,448 | 1,093 / 1,204 | 0 | 32.000000 |
| **Total** | **17,296** | **5,660,672** | **8,766 / 9,264** | **12,102** | — |

Trace slots count the two saved roles separately and can refer to the same candidate. No bank retains all candidate predicted states. The interrupted near-upright bank contains a partially written selected-role record; it is preserved as incomplete rather than filled in. The original and completion near-upright trials overlap in one partial historical episode, so this is an **evidence census**, not a count of independent trials.

Every complete bank's saved weights were recomputed from its candidate costs and saved temperature; the maximum absolute residual across all banks is `2.220446049250313e-16` (below `1e-10`). No cost/weight formula discrepancy was found in the retained measurements. In the old near-upright and energy banks, the saved final-window height qualifications are **zero for every candidate**, and consequently no weight sits on such qualified candidates. This refers to each candidate's final 100 predicted control boundaries; it is not an executed full-episode success test.

Strict action binding is **unresolved for 498 banks** (387 energy, 111 later swing banks). The first such bank is `/trials/variant_0/trial_0/planning/control_0005/pass_000` in the energy file. Its PCG64 RNG-after state matches exactly; its two retained selected action traces also match exactly (maximum difference 0). Nevertheless, the reconstructed full-bank SHA256 differs from the recorded hash. It remains unbound under the exact contract. The saved selected arrays cannot locate differences in unretained actions, and this review does not infer a unique numerical cause or blame physics.

The original dense-teacher file has no full-bank action hashes. Reconstructing its nominal updates from the current Mac runtime gives 2,008 banks with residual above `1e-10`, associated with the filtered-noise recipes; the first is approximately `1.91e-8`. The MPPI source file is byte-identical to the frozen source (`0368614152155a63cf41919fcbe5a8c1d2caee79f3144e106c806f3922cfec99`). This limits cross-runtime action reconstruction; it is not evidence that the recorded native controller used the wrong update. No tolerance was relaxed.

The first saved-scan attempts stopped on incomplete-record handling, an unhandled temporal-noise recipe, the output cap, and a NumPy-boolean serialization error. Those errors changed no historical evidence. Recovery preserved the completed prefix, completed only the 148 previously unpreserved tail measurements, and retained each failed status. Tail finalization used 0.366146 CPU seconds. Earlier attempts did not finalize CPU measurements before their errors, so their precise cumulative CPU use is unavailable; no exact total is claimed. The review's reporting limitations are explicit in `result_r4.json`, including the transient boolean-check log that was not finalized before the output-cap stop. Approximately 5 MiB of saved-scan output (including a figure-data CSV) remains, within its frozen 16 MiB storage ceiling.

## Existing classical-controller evidence, retained rather than repeated

The earlier raw PFL study is `/Users/jaisel/Documents/mppi-gps/runs/2026-10-27_bidirectional_native_pfl_001/`. Its own frozen four gain triples `(1,1,.1)`, `(1,1,1)`, `(4,2,.1)`, `(4,2,1)` were each run on two ordinary roots for 800 controls: **6,400 native controls total**. None achieved local/capture admission; no controller was selected. Native instantaneous checks of mass, bias, passive force, actuation, elimination, clipping and acceleration passed. The independent saved-only audit reports arithmetic `PASS`, scientific verdict `FAIL`, zero new physics. Source paths and hashes are recorded in this review's `preserved_pfl_review.json`.

That result already rules out repeating this particular four-gain PFL probe as if it were new. It is not a certified Drake controller, not an SOS certificate, and not a demonstration that no successful energy-shaping controller exists on this model.

## Independent collocation-output review

The saved-only audit in `runs/2026-11-05_acrobot_external_reference_001/collocation_audit_001/` checks the owner's `ed3b954` frozen source export and nine failed CasADi/IPOPT attempts. **All 13 exported source hashes match.** Independent NumPy equations, without importing CasADi or MuJoCo, reproduce each saved Hermite–Simpson constraint vector, including time, start/end and physical ±2 N m bounds, with maximum difference **1.1368683772161603e-13**. No EOM, torque-unit or constraint-assembly defect was found in these saved checks.

| Initial duration/shape | IPOPT outcome | Iterations | Debug duration (s) | Largest actual HS defect |
|---|---|---:|---:|---:|
| 6 / 0 | Infeasible problem | 414 | 6.9690 | 0.509187 |
| 6 / 1 | Iteration limit | 600 | 12.0834 | 95.813641 |
| 6 / 2 | Iteration limit | 600 | 11.8619 | 105.051427 |
| 8 / 0 | Infeasible problem | 189 | 12.0000 | 0.841898 |
| 8 / 1 | Iteration limit | 600 | 13.1867 | 12.734122 |
| 8 / 2 | Infeasible problem | 575 | 8.4261 | 0.205898 |
| 10 / 0 | Iteration limit | 600 | 12.0000 | 0.972515 |
| 10 / 1 | Infeasible problem | 356 | 8.5090 | 0.733719 |
| 10 / 2 | Infeasible problem | 332 | 12.0000 | 1.257227 |

These are **failed debug iterates**, not executable trajectories; some also violate the duration or torque bounds. Every actual HS defect exceeds the frozen `1e-6` feasibility threshold. The reported infeasibility flags agree with independent recomputation. The logs and saved solver statistics agree on **4,266 IPOPT iterations**, 10,988 objective calls, 10,991 constraint calls, 3,733 objective-gradient calls, 4,156 Hessian calls and 4,404 constraint-Jacobian calls. Saved attempt totals are 108.910924 CPU seconds and 108.942611 wall seconds; the enclosing process receipt records 109.01 user +0.24 system seconds. No fine solve was launched after all coarse solves failed, as specified.

The owner's separately generated native receipt contains 32 local parity steps and no executable reference. The independent review itself used **0.015176 CPU seconds, zero solver calls and zero physics controls**, and retained approximately 3.4 MB including copied inputs. This supports correctness of the tested saved equations/constraint accounting, not a working reference, an optimal time/cost or a task-infeasibility conclusion. Initial guesses, finite mesh and local optimization failure remain unresolved; one breaking MPPI change cannot yet be isolated relative to a reference that has not been obtained.

## Independent review of the local certificate

The owner's Drake preparation produced one CARE solution and one CSDP feasibility result. This independent audit binds its actual source to `5e56ead` and verifies **22/22 exact-fraction identities and inequalities**, with no new CARE, SDP or physics calls. The first review attempt stopped because the source changed during the owner's installed-API repair; that failed binding and the explicitly rebound review are preserved in `drake_certificate_audit_001/`.

The certificate is mathematically sufficient for the **exact nominal continuous dynamics with continuously recomputed local linear feedback**, inside a very small ellipsoid: radius **6.103515625e-6** in the chosen four-coordinate Euclidean norm, level **1.6401953694227502e-12**, and conservative physical-torque upper bound **0.0033378396072 N m**, below ±2. The exact symmetric Gram/LDL identity is positive, `Qcl_lower=0.9999996618663798` and `gamma=0.14981879869321005`. The saved CSDP result has coefficient residual `8.731149137020111e-11` and minimum Gram eigenvalue `0.8501812013836677`. The detailed independent derivation is `drake_certificate_audit_001/source_review.md`; the arithmetic receipt is `result.json`.

This proof combines analytic sine/cosine remainder bounds with an exact quadratic SOS Gram. It is neither a global trigonometric SOS search nor a useful demonstrated swing-up/recovery region. **It does not certify the 50 Hz held-input execution, RK4 map, global energy-shaping controller, or entry into the ellipsoid.** Its tiny size cannot explain or admit the historical near-upright roots. Independent verification used **0.001340 CPU seconds**. No sampled basin is relabelled as a certified one.

## Independent validation of every retained selected trace

The `selected_trace_audit_001/` receipt independently reconstructs geometry, mechanical energy, running/terminal objectives and recorded geometric aggregates from **all 12,102 complete selected state/action trace slots**, containing **3,380,500 post-action states**. It streams one bank at a time and does not call the simulator or the existing objective implementation. Every complete selected candidate cost agrees with its saved candidate index within **1.8189894035458565e-12**; maximum terminal-cost error is **2.842170943040401e-14**, component energy-sum error **7.275957614183426e-12**, and peak-height error **8.881784197001252e-16**. No tolerance was changed. The only unavailable pair is the already interrupted bank `/trials/variant_0/trial_2/planning/control_0429/pass_000`, missing both selected states and actions.

This checks the actual retained states rather than only counting them. It still cannot inspect unrecorded candidate states. Geometry is independently derived from the saved angles and compared with saved geometric aggregates; no new native tip positions are generated. Cost/geometry checks used **3.624731 CPU seconds, 7.154335 wall seconds**, zero physics and zero fitting.

## Independent review of the adapted Drake controller's saved execution

All **3,000** deployed normalized actions and physical torque requests were recomputed from the saved **pre-action** states using an independent implementation. Maximum raw-torque discrepancy was **4.433786671143025e-12 N m**, action discrepancy **6.838973831690964e-14**, and post-action native-cost discrepancy **7.105427357601002e-15**. All six exact initial full contexts, 501-state episode lengths, 0.02 s intervals and the **6,000 native integration substeps** reconcile with the frozen roster and receipt. Balance-only clipping counts were **500, 1, 499, 497 out of 500**; swing-up clipping counts were **498, 499**. Balance held on **1/4** starts; the hybrid joined/held on **0/2**, with peak heights **2.0695511731 m** and **2.0431767141 m**.

The saved native audit remains **qualified**, not an unconditional pass: recomputing the diagnostic `x'Px` on the Mac differs by up to `2.3283064365386963e-10` with a vector contraction, and `1.1641532182693481e-10` using the source's scalar operation order, above its frozen absolute `1e-10` check. Both failed checks are preserved in `drake_native_audit_001/result.json` and `result_r1.json`; the stopping rule was applied and the tolerance was not relaxed. All switch classifications and reported outcomes matched. This does not identify a physics defect or invalidate the separately exact-rational continuous proof; it leaves this diagnostic's numerical reproduction unconfirmed at that exact absolute threshold. No native replay or further recovery was launched by this reviewer.

## Independent review of the saved MJPC reference

**Later contract finding:** the v1.3 adapter used an outward control difference at the +1 limit. Its interior-only derivative gate could not detect this. The arithmetic checks below remain reproducible, but do **not** make that iLQG result a valid external-reference control. Preserve it as an invalid-contract attempt; the predictive-sampling result is unaffected. The separately frozen v2.1 range-aware adapter and boundary/probe validation address this defect.

The bounded MJPC v1.3 review streams every persisted candidate on the worker: **25,000 complete binary records, 483,400,000 bytes**, across both planners and all five historical contexts. It also checks all ten 501-row execution traces, **302 official source-file hashes**, ten frozen model/context files, source/model structural identity, all header schedules, candidate state-to-residual geometry, official costs/mean returns, action bounds, execution cost indexing and the 12-job receipt. All **175,445 recorded checks pass**. Maximum candidate residual error is `7.993605777301127e-15`, cost error `1.1368683772161603e-12`, and mean-return error `5.684341886080801e-13`. No saved final candidate is marked failed.

The 16 derivative validation cases have maximum A/B/C/D errors `3.65418806325124e-7`, `2.220446049250313e-10`, `9.948708523388472e-7`, `3.330680176105716e-11`, respectively, below the frozen `1e-4` gate. This checks the adapted RK4 derivative interface against the independently executed centered probes already paid for by the owner. The reviewer executes no probes.

| MJPC planner | Hanging peak height (m) | Hanging post-action project cost | Historical near-upright holds |
|---|---:|---:|---:|
| Predictive sampling | 2.757840 | 3,811.427894 | 0 / 4 |
| iLQG | 2.203802 | 2,812.345404 | 1 / 4 |

Costs use the existing project's post-action convention `sum c(x[t+1],u[t])`, independently reconstructed from saved pre-action CSVs. The CSV's same-row dense diagnostic and the MJPC framework's own pre-action cost remain separate; they are not mislabeled as this post-action sum. These historical-start results are not the standard random-start benchmark.

All per-control counters reconcile exactly: predictive sampling **2,505,000 steps**, iLQG **6,518,750**, plus held-out derivative validation **256**, totaling **9,024,006 native integration steps**. Predictive sampling uses `5*(250*10*200+1000)`; iLQG uses `5*(250*5211+1000)`. Each iLQG planning call includes 4,000 nominal/line-search rollout steps and 1,211 derivative steps. The source's duplicated penultimate derivative index is included rather than omitted from accounting. The unsupported stock RK4 derivative call performs zero integration steps. The completed native stage reports **15.295992 wall seconds**, within its frozen budget.

The audit uses **1.750844 CPU seconds**, zero new physics and zero solver calls. Exact arrays remain on the worker; their sizes and SHA256 hashes are in `mjpc_saved_audit_001/result.json`. A hash-verified compact copy of CSVs, bindings, receipt and the `16f2ea3` driver source is retained locally. Only the final candidates exposed by the upstream API are inspected: internally overwritten nominal/line-search candidates are not falsely claimed preserved. Official predictive sampling has no bound seed interface here, so these are not paired-randomness comparisons.

The canonical dense reward's defaults are additionally pinned from [DeepMind's reward implementation](https://github.com/google-deepmind/dm_control/blob/f1f91e0309e5df81bf7a3ee83849ebcee86779d8/dm_control/utils/rewards.py): Gaussian, margin value 0.1. The exact source operation is `exp(-0.5*(d*sqrt(-2*log(0.1)))**2)` outside the target radius, with `d=max(distance−0.2,0)` for Acrobot's margin 1; values inside the target radius are exactly 1. The preserved source SHA256 is `009f02389fd2af84b2955324707638e79a0e18b6a540dd7c254abf5096e157f3`.

## Corrected MJPC v2.1: independent saved-probe and execution review

The separately frozen `820f20b` adapter is now checked at the actual actuator boundaries. The saved-only review under `mjpc_v2_1_saved_audit_001/` reconstructs **all 80 forward/centered A/B/C/D sets** and their independent reference matrices directly from **1,080 recorded native probes**, preserving complete before/after simulator contexts. Every reconstructed matrix entry agrees **exactly** with its saved matrix. All perturbed controls remain inside **[-1,1]**; full pre-contexts differ only by their declared perturbation, and each recorded step advances 0.01 s.

The independent pre-action sensor formula matches every probe's eight stored sensor outputs within **8.881784197001252e-16**. Comparison against the design review's separately preserved analytic C/D matrices gives maximum C error **9.937166442208767e-7**, D error **2.8755664516211255e-11**, within the frozen `1e-4` gate. The separate finite-difference reference comparison has A/B/C/D maxima **3.9787062534185624e-7**, **2.775557284007135e-10**, **9.937384248814851e-7**, **4.4408920985006255e-11**. At the two exact control limits, the corrected B blocks are nonzero (largest entries range **0.14465797–0.24066662**), rather than the clipped outward derivative.

The review streams **all 52,500 persisted candidate records (1,015,140,000 bytes)** and checks all 21 completed episodes: 16 standard-start 100 Hz traces and five corrected historical iLQG 50 Hz traces. It verifies **302 official source hashes, 24 frozen input files, both SDK hashes and the frozen driver source**, candidate headers/geometry/residuals/costs/returns, execution costs, canonical reward and step counters. The work census reconciles exactly:

- Boundary validation: **1,080** native steps.
- Eight standard predictive-sampling episodes: **4,008,000**.
- Eight standard iLQG episodes: **10,430,000**.
- Five corrected historical iLQG episodes: **6,518,750**.
- Total: **20,957,830**, within 21,000,000; native wall time **34.957279 s**, within 300 s.

| Corrected reference | Final-height holds | Mean canonical dense reward |
|---|---:|---:|
| Predictive sampling, eight standard starts | 0 / 8 | 0.04297569423 |
| iLQG, eight standard starts | 3 / 8 | 0.28211761537 |
| iLQG, four historical near-upright starts | 1 / 4 | Historical roster, reported separately |
| iLQG, historical hanging start | 0 / 1; peak 2.158531 m | Historical roster, reported separately |

These eight starts are a small check using the canonical reward/reset distribution; they do not reproduce a published aggregate result. The corrected adapter's negative hanging-start result is now interpretable under its tested contract. It still supplies no successful hanging-to-upright reference or a minimum time/cost, and does not isolate one breaking change in MPPI.

**Reporting and reproducibility qualifications remain preserved.** The initial independent audit reports 17 failed comparisons, without changing its gates. Fourteen are byte comparisons against fresh Linux regeneration of seven archived random-angle starts per planner; the maximum angular difference is **4.440892098500626e-16**. All frozen context-to-execution bindings pass exactly. The owner's same-origin Mac regeneration matches all 16 archived angle values byte-for-byte; its Python/NumPy/platform metadata is retained. Cross-platform regeneration byte identity is not claimed, and no unique numerical cause is assigned.

The other three failures compared the core task's last-N POST-only window against MJPC's frozen inclusive recorded-boundary window `time>=8−1e−10`. This was a **review-contract mismatch**, not an incorrect owner summary. Both definitions and the failed comparison are retained:

| Trace | Frozen MJPC minimum, including 8.00 s | Core last-N POST-only minimum | Hold classification |
|---|---:|---:|---|
| Standard iLQG 0 | 3.999999824513126 | 3.999999840460405 | Pass under both |
| Standard iLQG 4 | 3.999842443520444 | 3.999843089258780 | Pass under both |
| Historical iLQG 2 | 3.999994950635270 | 3.999995452500547 | Pass under both |

No physical tolerance or outcome criterion was amended. All walking/holding classifications, canonical reward and post-action cost comparisons agree. `contract_resolution.json` preserves the distinction, rather than replacing the original failed audit status. Main streaming review: **4.346298 CPU s / 4.347102 wall s**; compact clarification **0.017898 CPU s**, analytic/window cross-check **0.011754 CPU s**; zero new physics or planner calls. Large banks remain on the worker with SHA256s; approximately 6 MB of compact inputs/receipts are retained locally.

## Adapted Drake handoff v2: independent saved comparison

The `3b11514` handoff comparison keeps the plant, ±2 N m bound, K/P and swing-up gains fixed and changes the empirical switching level to 1,000. This follows the source's threshold convention; it does **not** make this parameter-adapted controller identical to Drake's stock plant/controller or enlarge the proven continuous certificate.

The saved-only `drake_v2_saved_audit_001/` review checks both historical swing episodes, **1,000 deployed actions, 2,000 native substeps**, costs, clipping, initial contexts and the comparison against the strict certified-level handoff. Swing 0 is byte-identical in actions/states and never enters balance. Swing 1 uses balance at controls **194,195,196,215,287,379**, with **four entries and four exits**; its first changed deployed action is **287**, six actions change overall, and maximum action difference is 2. The closest recorded switching value is **14.4703533** from the threshold. Its cost rises by **12.2846632**; both starts still fail the hold test.

All those measurements reproduce. The sole retained audit qualification is the previously identified scalar `x'Px` cross-runtime discrepancy of **1.1641532182693481e-10**, just above the unchanged `1e-10` check. Torque/action/cost checks pass; the smallest switching margin is far larger than that discrepancy, and switch classifications agree. The failed diagnostic gate is preserved, with no tolerance relaxation, replay or further recovery. Review used **0.006580 CPU s**, zero new physics and zero solves.

## Saved-prefix exact-RK4 initialization route: final independent review

The bounded route tried two **already executed** MPPI prefixes as initial guesses for the existing offline optimizer. The `collocation_warmstart_audit_001/` review passes **78 saved-arithmetic checks**, including all **19 remote source/prefix hashes**, authorization of manifest `de4fcfa92debe6d88bee7d4fcbdf4b0ed10dcd682a08e9d3c0a25299bd59f1fe`, and exact full initial contexts. It independently confirms the earliest minimum over all 601 recorded states selects controls **569 and 508**, giving fixed durations **11.38 s and 10.16 s**. Both fine guesses preserve every original native endpoint byte-for-byte and have maximum nominal RK4 defect **3.552713678800501e-15**. Their initial terminal-equality errors, **2.5973447 and 6.2967897**, remain present rather than being hidden by replacing the endpoint.

The first saved debug candidate satisfies its start and target endpoint equalities exactly and stays strictly within the real torque bound: maximum **1.999935421610028 N m**. However, its largest RK4 dynamics defect is **0.0022246092933997375**, far above **1e-8**. Independent reconstruction of the entire constraint vector differs from the saved vector by at most **3.552713678800501e-15**. Thus the optimizer has not produced an executable swing-up trajectory, despite satisfying the endpoints. Its effort value **12.8659410 N² m² s** is a failed-iterate objective, not a feasible optimal-control cost.

IPOPT stopped the first attempt after **805 iterations** for its CPU limit: **180.114441 s** inside IPOPT, **181.324880 s** for the full attempt. The second attempt has a preserved initial guess and a final printed iteration index **911**, but no final candidate, result or feasibility verdict. The supervisor then stopped with **SIGKILL / return code −9** at the shared hard CPU limit. Actual child accounting is **359.973184 user +0.079010 system =360.052194 CPU s**; the **0.052194 s** measured excess is reported rather than rounded down. Wall time was **360.118870 s**, retained output **22,420,176 bytes**, and no native replay occurred. No interrupted iterate was recovered or inferred.

This closes the route as **resource-exhausted without a feasible reference**, not as evidence that swing-up is impossible. The independent audit used **0.023303 local +0.010906 remote CPU s**, re-evaluated **13,168 nominal EOM rows solely for saved-constraint arithmetic**, and performed **zero native steps and zero optimizer calls**. Both failed/unfinished solves, the original preparation versions and all accounting remain preserved; no follow-up was launched.
