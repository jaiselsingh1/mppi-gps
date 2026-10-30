# H3.2: velocity-feature intervention — UNTESTED / NOT LAUNCHED

**Classification correction:** H3.2 is an input-representation diagnostic in
the [student learnability track](student_learnability_track_v1.md), not a new
MPPI–GPS coupling formulation. Its inherited-state design below changes the
functional input distribution, including magnitudes, while retaining old
normalization and Adam. It cannot isolate information content alone. The
learnability note describes a separately versioned fresh-training alternative;
neither design has launched. [H4.1](adaptive_coupling_h4_1.md) is the separate
weight-adaptation hypothesis. The original H3.2 draft is preserved in the
[scope-correction archive](../artifacts/2026-10-30_research_scope_correction_001/previous_docs/docs/loop_v3_next_hypothesis_v3_2.md).

This is a conditional next hypothesis for use **only if H3.1 fails**. Writing this document does not launch or implement H3.2. It remains unlaunched because H3.1's predeclared stopping rule closes that finite experiment, and the next test must be separately frozen. No new fitting, inference, simulator query, acquisition or evaluation was performed to prepare it. H3.1's source, inputs, thresholds and result remain unchanged. A fresh create-only protocol, source snapshot, input manifest and their hashes must be frozen and verified before any H3.2 launch. Failure to validate the required saved-state provenance stops preparation; no new data or simulator reconstruction replaces missing provenance within this hypothesis.

## Hypothesis and causal scope

**H3.2 (untested):** retaining full physical velocity values in the existing 17-dimensional student feature vector, at both training and deployment, improves the next fixed-budget standalone torque policy relative to otherwise matched continued fitting with the current ±10 velocity clipping.

The existing audit measured at least one clipped velocity in 2,359/4,000 original pre-action teacher states; MPPI used full physical velocities. This establishes information loss under the student feature map. It does not establish control-relevant aliasing, identify the cause of reconstruction error or falling, or show that removing clipping will help. This intervention tests that specific explanation for this inherited learner and data. It is not a test of the entire CIO–GPS algorithm, population-wide observability, or a new environment.

## The single intervention

Use the same physical Walker task, MuJoCo backend, reset distribution, complete 1,000-control cost and success rule as H3.1. No planner or critic is added. The existing torque policy remains 17→128 tanh→128 tanh→6 tanh, with direct bounded-torque execution every control. No teacher prefix, MPPI fallback, action repair, new memory or target relabeling is allowed.

For a recorded physical state \(x=(t,q,\dot q)\), define

\[
h_c(x)=[q_{1:9},\operatorname{clip}(\dot q_{0:9},-10,10)],
\qquad h_f(x)=[q_{1:9},\dot q_{0:9}].
\]

These vectors each have 17 entries: the same eight position features and nine velocity features. The implementation must bind the slices to the existing state schema; for its 19-entry recorded state they are `state[2:10]` and `state[10:19]`. Physical state, simulator integration, task costs and reset mechanics remain unchanged.

The control child trains and deploys using \(h_c\). The treatment child trains and deploys using \(h_f\). This train-and-deploy feature-map change is the **only** intervention. Do not train on full velocities and deploy on clipped observations, or change the environment's native observation semantics for the control. The treatment feature vector must be constructed from the actual current full physical state already provided by the controller interface.

Both children retain the exact original observation mean and standard deviation stored in the incoming Bfree checkpoint. No new normalization is fitted. Consequently the treatment can present normalized values outside the parent's clipped-feature range; that extrapolation is part of this intervention. Fixing the normalization prevents an additional normalization intervention, but does not make this a pure test of information content independently of input magnitudes or inherited Adam moments. Those limits must accompany the verdict.

## Matched learner and saved data

Both children begin from the exact retained Bfree 10k parent used by H3.1, policy hash `c107f02196502b2e2fe8aed52cea8691e5204c27ef413adfb5fcbd2a41b1bcf0`. They start from identical weights, complete Adam moments/step counters, PCG64 minibatch state, original normalization and stored learning metadata. They start one new proximal block with the same parent anchor. Do not initialize either child from an H3.1 fitted child. Preserve the unchanged **clipped-input parent** as the third standalone control.

Use the same 6,096 underlying physical states and action targets as the H3.1 **independent-data arm**, in precisely its frozen order:

| Rows | Existing source | Required ownership |
|---|---|---|
| 4,000 | Four original successful independent teacher episodes | Actual pre-action states and executed torques, preserving episode/time order |
| 64 | Original perturbed-state re-solves | Separately identified hypothetical re-solve states and selected first torques; never relabel these as actual executed teacher controls |
| 32 | Accepted v2 short-recovery dataset | Actual pre-action states and applied torques selected by the original branch training mask; exclude the two impulse controls |
| 2,000 | `v2_r0_trial_raw_0/1` | Actual pre-action states and executed torques from the two complete recorded independent teachers |

The treatment changes only the features derived from those states. Both children use identical row indices and bitwise-equal action targets. The clipped reconstruction must reproduce the H3.1 independent-arm observation bytes, allowing only the predeclared training float32 conversion; the complete original float64 source and conversion hashes must also be retained. Hash both feature matrices, action arrays, row identities, incoming checkpoint and complete optimizer tree. Equal row counts alone are insufficient.

**Preparation must resolve every row's full-state provenance before fitting.** For the 64 re-solves, bind each bootstrap feedback record to its exact `bank_digest` and saved selector `initial_state`, preserving the perturbed velocity before clipping. Do not substitute its nominal teacher anchor state, infer velocity from the clipped observation, or rerun perturbations. Verify that the saved selected first action and clipped pre-action observation reproduce that bootstrap row. For the 32 recovery rows, bind the accepted dataset to the exact saved branch, its original `training_mask`, and corresponding pre-action entries of the full `states`/`warmstarts`/`integrations` arrays; verify observation/action order against the admitted rows. The 4,000 original and 2,000 added executed rows must similarly bind to complete saved episode arrays. Preserve all original provenance classifications and the original acceptance decision; H3.2 adds no recovery or outcome filtering.

If the exact full state for any of the 6,096 rows cannot be verified, mark H3.2 preparation incomplete and stop. A reduced dataset, imputed velocity, substituted re-solve or new acquisition would be another hypothesis and needs a separately versioned protocol.

## Fixed learning and evaluation schedule

Each child receives exactly 2,000 additional Adam updates, batch size 256, learning rate 0.0003, action-component MSE, total gradient-norm clip 0.5, parameter-mean proximity weight 0.01 and anchor refresh every 250 updates. Use the existing fitting implementation. Restore the same incoming PCG64 state in both arms and save the full 2,000×256 sampled index array. The configured seed is merely a required fit argument when restoring the saved RNG; it must not replace that state. Freeze its value before launch.

Checkpoints at updates 500, 1,000 and 2,000 resume the same fitting block, including Adam, RNG, active anchor and block counters. They are descriptive checkpoints; only the fixed 2,000-update endpoints enter advancement. Preserve each fitted checkpoint and optimizer state even on failure.

Use the following fresh paired evaluation roots, with episode \(i\) assigned seed `root+i`:

| Cohort | Root | Fixed roster |
|---|---:|---|
| Learning curves | 951110000 | Same four starts for parent and both children at all three checkpoints: seven policies, 28 complete episodes |
| Primary | 951100000 | Same 16 starts for the unchanged clipped parent and both 2,000-update children: 48 complete episodes |
| Conditional confirmation | 951120000 | Same 16 further starts for those three policies, only if the primary gate passes: 48 complete episodes |

Before sealing the protocol, verify these exact seeds do not occur in prior evaluation ledgers. A collision blocks this version's preparation; do not silently change roots. Each cohort must also bind full initial simulator states and task configuration, not only integer seeds. Preserve every actual state, action, cost, termination reason and snapshot through completion or native termination. Complete cost includes the existing fall terminal once and 5 per unexecuted control after a fall. Survival and velocity MAE≤0.3 m/s define the existing task success. Different successful paths are valid.

Before simulation, compute full-data MSE for the unchanged parent and all six fixed child checkpoints, using **each policy's own declared feature map** on the same 6,096 targets. This is seven full passes, 42,672 forward rows. Do not compare raw clipped-input versus full-input losses as proof of control quality or choose a checkpoint using those losses. Record minibatch learning curves and complete standalone performance curves separately.

The maximum fixed schedule is 124,000 live controls, zero model/planner controls, 1,190,672 neural rows (1,024,000 fitting + 42,672 full-data diagnostics + at most 124,000 standalone rows), 1,024,000 backward rows and 4,000 Adam steps. The prelaunch metered caps may be 124,000 live controls and 1,400,000 neural rows to match H3.1's finite ceiling; the excess neural allowance is not permission for extra measurements or retries. Abort on budget exhaustion or runtime/provenance failure and retain all partial work. Preparation, testing, audits, plotting and transfer overhead are separately timed and reported.

## Gate, stopping rule and subsequent interpretation

The full-velocity treatment passes the primary gate only if its final mean complete cost is at least 5% lower than **both** the unchanged clipped parent and clipped-data child, with success count, full-survival count and mean executed controls each nondecreasing against both. Report per-start cost and controls differences. This allows incremental improvement without requiring a previously competent standalone policy. Use the same rule on the disjoint confirmation cohort, without parameter or checkpoint changes.

If primary fails, stop after the prescribed curves and primary cohort. If confirmation fails, stop. Do not add seeds, data, updates, normalization changes, feature clipping thresholds or alternate checkpoints. Retain the original parent as the active policy on either failure, and use the measured evidence to formulate a new explicit version if needed. A confirmed result supports only this combined train-and-deploy feature intervention on this learner/data/budget. It does not isolate information loss from scale/extrapolation effects, prove that clipping uniquely caused the historical failures, or establish standalone walking unless the actual task success criterion is met.

The teacher outcomes remain those of the fixed independent data; report them as inherited evidence with zero new teacher acquisition work. Report the descriptive teacher–student complete-cost gaps and clearly state that saved teacher starts and fresh student starts differ. This H3.2 test makes no MPPI feedback change and cannot close the MPPI–policy loop. If the intervention improves the next standalone policy, any subsequent acquisition–learning round using that feature map requires its own frozen protocol before execution, and a second successful round before a loop claim.

## Required prelaunch freeze

The future create-only launch directory must contain the exact H3.1 failed-result hash establishing the condition for H3.2, this hypothesis's version/hash, complete source/configuration manifest, finite schedule and caps, task/model hashes, checkpoint and complete incoming learner fingerprints, all source ledger/HDF references, the validated 6,096-row full-state identity map, both derived feature arrays and action hashes, normalization hash, declared feature map per checkpoint, seed-disjointness audit and command. Preparation failure must be retained in its own create-only receipt. This conditional hypothesis note does not substitute for that frozen executable protocol.
