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

## What the saved candidate evidence actually contains

Raw sources are preserved unchanged. No missing states are synthesized in this review.

- Every completed root episode records executed actions and costs, pre-action observations, and the `N+1` simulator states/warmstarts/integration contexts. The earlier interrupted near-upright file also retains its partial third episode and incomplete final bank.
- The energy-discovery bank callback saves **every candidate's scalar cost, weight, aggregate cost components, terminal energy data, peak height, final-window minimum height and height qualification**. It saves full predicted state/action traces for **two selected roles** per bank: minimum task cost and maximum height. Those roles can select the same candidate.
- It also saves the bank's full-action-array SHA256, shape/dtype, planner RNG-after state and full pre-context. It does **not** save all candidate state/action arrays.
- The later ESS32 near-upright and swing-development recordings save **every candidate's cost and weight**, nominal before/after, action-array SHA256/shape/dtype, RNG-after state, temperature and ESS. They do **not** save per-candidate predicted states or per-candidate height/energy components.
- For the declared white-noise recipes, complete candidate actions can be reconstructed without physics from the frozen private PCG64 stream, Gaussian sampling recipe, saved nominal and clipping rule, then verified against each saved SHA256. Reconstructing unretained candidate states would require new native execution, which this review does not perform.

Consequently, the user's request to inspect every candidate's **states** at failure cannot be fully satisfied from preserved arrays. Scalar/weight and action-reconstruction checks can be exhaustive; unrecorded predicted states remain missing. The explicit file/bank roster and bounded saved-only scan are recorded separately under `saved_scan_001`.

## Interpretation and next discriminant

No model substitution, incorrect torque-unit conversion or wrong upright Jacobian has been found in these checks. That narrows particular implementation errors; it does **not** establish overall planner correctness. The difference between the official task and our gate, low-temperature weight collapse, short-horizon search, initialization and cost design remain distinct explanations. Adaptive weighting already removed ESS≈1 yet failed to establish holds/capture; weight collapse alone is therefore insufficient as the complete explanation.

An independently executed, torque-limited reference on this exact mechanical model is needed before attributing the failure to the task. If it works, a one-piece-at-a-time comparison can identify a breaking change. Until then, there is no working reference relative to which a single MPPI defect has been isolated, no certified funnel, and no basis for launching the learning loop.
