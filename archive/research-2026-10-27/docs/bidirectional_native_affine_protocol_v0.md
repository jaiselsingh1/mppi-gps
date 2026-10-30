# Separate native affine-feedback reference test

This finite development diagnostic follows the closed, independently audited negative native reference002. Four bounded trajectory-optimization attempts produced guessed endings with zero capture/local hinge residual, but their scaled native dynamics defects remained .004897–.008447 rather than closing to 1e-6. All eight actual open-loop replays failed local approach and capture. Those guesses are not successful experience. The new question is whether a fixed feedback law can make those exact references executable. This is not a correction to their old scores or a new MPPI/policy result.

Pin the exact reference002 manifest, result, arithmetic audit, four candidate arrays and critical native sources. Preserve .01-second native RK4 physics, frame skip one, 800 controls, starts near hanging and native elbow bounds. The original local condition remains tip distance≤.5 by control600. Capture remains distance≤.2 and joint-speed norm≤.5 throughout the final100 controls. Task costs and the true 1,000 capture-failure component remain unchanged. The correction-design quadratic below is a controller construction surrogate, never a replacement task return.

At all800 nodes of each of the four saved guesses, pay fresh native one-control probes: four symmetric state pairs, a bound-aware action pair and the exact unperturbed transition. Keep clock and solver-warm-start intervention explicit. This is 11×800=8,800 rows per guess, split into8,000+800 queries without changing the executor's limit. The last saved Jacobian from the old solver was not generally at the chosen best guess, so it cannot substitute for these new derivatives.

Let d=xbar_next−F_native(xbar,ubar), giving the local error model e_next=Ae+Bδu−d. Use Q=diag(10,10,1,1), R=.1 and the same terminal Q. For V_next(e)=e'Pe+2p'e+constant, compute H=R+B'PB, K=H⁻¹B'PA, k=H⁻¹B'(Pd−p), P_new=Q+A'PA−A'PBK and p_new=(A−BK)'(p−Pd). Execute u=clip(ubar+k−Ke,−1,1), wrapping only the relative angle errors into [-π,π). Save all exact A/B/F/d/P/p/K/k/H arrays. Invertibility, finite values and numerical failure are checked and recorded. This affine correction has no guarantee when the reference is far away or the actuator saturates; actual outcomes decide.

First repeat the48-control native heterogeneous/scalar/warm-start qualification. Then measure one complete feedback episode on the original root for each of the four fixed references. A nominal must meet both original local and ending conditions before its fixed8 fresh selection roots are used. Roots194000–194031 are allocated in eight-root blocks by fixed reference index. Require all8 to qualify a reference. If several qualify, select the lowest mean complete native task cost, with fixed index as tie break. Execute the single selected controller on32 untouched roots195000–195031. Final pass requires at least29/32 meeting both original conditions. No final root can choose a controller or gain.

The hard maximum is **89,648 native physics controls**:48 qualification +35,200 fresh linearization +3,200 nominal +25,600 selection +25,600 final. The initial prospective86,448 estimate omitted the3,200 unperturbed rows needed for exact defects; it was corrected before configuration freezing or queries. There are no new actor forwards, Adam updates or nonlinear-solver iterations. Riccati work and actual clipping/feedback evaluations are recorded separately. Earlier reference001's352,848 and reference002's1,417,648 controls remain paid ancestry, without being duplicated as new queries.

Any failed gate closes this version with its actual costs; no retries, gain tuning, solver refinements, new roots or reserve reallocation. Keep failed traces and both ledgers. Independent saved-evidence audit checks native trajectories, ownership, exact fresh stencils, derivative/defect arithmetic, affine recursion, actual actions, all gates and the shared budget. It performs no extra physics or inference. Report execution and audit wall time separately.

Even a pass would establish only a finite feedback-reference feasibility witness. It would not establish MPPI discovery coverage, successful global-policy compression, beneficial learning feedback, sample efficiency, or transfer. Guessed references and these diagnostic trajectories remain excluded from teacher supervision. Affine LQ tracking and trajectory optimization are established tools; the larger protected bidirectional learning loop remains the research object.

```sh
.venv/bin/python -m scripts.diagnostics.run_affine_native_reference \
  --parent-run runs/2026-10-27_bidirectional_native_reference_002 \
  --config configs/bidirectional_native_affine_v0.json \
  --out runs/2026-10-27_bidirectional_native_affine_001
```
