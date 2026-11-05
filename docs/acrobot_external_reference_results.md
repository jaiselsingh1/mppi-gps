# Acrobot: external references, defects and remaining gap

**The teacher problem remains unresolved; the body has not been shown infeasible.**
The physical XML matches dm_control. Independent equations agree with native RK4
and parameterized Drake. We found and corrected a defect in our *new reference
adapter*, but that repair did not fix hanging swing-up. No learning or coupled
loop was run, and no original MPPI cost, sampler or controller was changed.

| Check | Saved result | Interpretation |
|---|---|---|
| Exact model | Physical XML identical after removing our extra sensor; common includes byte-identical; elbow torque ±2 Nm | Correct body and torque units; our historical 50 Hz/start/cost/hold settings differ from the standard task |
| Equations | 32 native one-step checks: max error **8.88e-16**; 32 Drake nonlinear checks: **2.84e-14** | Evidence against the tested sign/inertia/damping/gear bindings being wrong |
| Historical raw candidates | **17,296 banks / 5,660,672 candidates**; all weight formulas agree; **12,102** retained trajectory slots independently recomputed | No weighting-arithmetic defect found. Full states were never retained for every candidate; **498** full-action hashes remain unresolved |
| Offline optimization | Nine smooth-guess attempts: no feasible solution, best defect **.2059**. Recorded-prefix attempt: **.002225**, not **1e-8**; second interrupted by shared CPU cap | No verified hanging swing-up, minimum-time upper bound or cost optimum. No invalid candidate was replayed |
| MJPC, standard starts | Sampling **0/8** holds, mean dm_control reward **.04298**; corrected iLQG **3/8**, reward **.28212** | Two captures begin below 3.8 m; one already begins above it. Eight starts are not a published benchmark reproduction |
| MJPC, preserved starts | Corrected iLQG **1/4** near-upright holds; hanging peak **2.159 m**, no hold | Boundary repair changes actions/costs but does not resolve teacher reliability |
| Drake-derived reference | Balance **1/4**; published-handoff swing-up **0/2**; clipping **498/500**, **499/500** controls | Identical-model, ±2 Nm port is not a working hanging reference |
| Local certificate | Exact rational SOS/remainder verification **22/22**; radius **6.10e-6**, rho **1.64e-12** | A tiny ideal continuous-feedback region; not a practical capture region or a sampled/native funnel certificate |

The iLQG adapter originally perturbed outward at maximum torque, so actuator
clipping could give a false zero input derivative. The interior-only gate missed
this. The corrected implementation follows MuJoCo's feasible-direction rule;
**40 boundary cases in both derivative modes**, with saved probes and analytic
sensor Jacobians, passed. Old iLQG outcomes remain labelled
`INVALID_EXTERNAL_REFERENCE_CONTRACT`. Matched historical reruns still hold
1/4 near starts and fail hanging, so this defect is not the explanation for our
original MPPI failure. Sampling was unaffected.

The first Drake hybrid also replaced the published balance threshold 1,000 with
the microscopic certified level. Restoring the published threshold was tested
separately: one trajectory stayed byte-identical; the other changed six deployed
actions and cost **+12.28**, without capture. Both results remain. Stock Drake's
own demonstration uses a different plant and **±20 Nm** and cannot count as our
reference.

**No single breaking change in the original MPPI has been isolated.** We have
partial external captures, not the reliable hanging reference needed for the
requested one-piece-at-a-time diagnosis. The recorded-prefix optimization also
changes transcription, starts, terminal winding and fixed duration relative to
the earlier coarse solves; its smaller residual cannot be attributed solely to
initialization. The smallest next test is one explicitly versioned feasibility
continuation from the saved .002225 primal iterate, at the same11.38s duration
and ±2 Nm bounds, followed by native replay only if it meets the unchanged
constraint gate. It needs a newly frozen solver budget and has **not** been
launched. This closes the missing reference before swapping original planner
pieces or launching learning. A new similarity score or policy architecture is
not justified by these results.

Existing Walker LQR trajectories are **candidate funnels**. A verified funnel
needs boundary decrease, actuator validity and terminal containment; a near pass
is insufficient. A fixed point gives a stationary terminal set; a periodic orbit
gives a phase-dependent terminal tube. Neither makes the empirical joins a
certificate. See the [funnel formulation and exact proof scope](acrobot_drake_sos_reference_review.md).

Independent reviews confirmed the recorded costs, rewards, actions, matrices and
work counts. Qualifications remain explicit: one Drake diagnostic has an
unresolved one-ULP strict-tolerance failure; Linux seed regeneration differs by
at most **4.44e-16** from the frozen Mac-generated vectors, while actual input
bindings match exactly. Inclusive and post-action final-window minima differ in
three comparisons; every hold classification agrees. No tolerance was relaxed.

[Evidence and runnable commands](../results/acrobot_external_reference_001/README.md)
· [equations and raw-bank review](acrobot_external_model_review.md)
· [MJPC details](acrobot_mjpc_reference_review.md)
· [independent design review](acrobot_external_reference_design_review.md)
· [accounting and audit appendix](acrobot_external_reference_accounting.md).
