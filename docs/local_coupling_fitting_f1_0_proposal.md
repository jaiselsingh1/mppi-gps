# F1.0 — conditional fitting of locally coupled MPPI data

**FOR REVIEW; `launch_ready=false`. No execution authorized.**

**Current status: NOT QUALIFIED.** A/B/C stopped at calibration support:
14/256 valid candidates, below the required 128. C was not reached; no
qualifying bound teacher-label sets were provided. F1 requires fresh review
before any launch, not an automatic follow-up.

Test whether the owner's pre-update coupling signal predicts improvement after
a fixed learning update. A/B/C uses eight student-visited restarts, up to 300
controls each, standard MPPI `H=48/K=256`: A=`α=0`, B=`α=.12`, C=`α*`, requiring
qualified calibration before B. **C versus A is the primary contrast**;
B remains diagnostic. No weight or corpus is selected using fitting results.

## Conditional admission and inputs

First independently review the saved owner-predeclared B/C patterns, held-restart
support and teacher competence, using the owner's original criteria. Bind all
failures and costs. Missing or ambiguous qualification blocks freezing F1;
it does not authorize acquisition, replacement restarts or relaxed gates.
Action MSE is reported, **never an admission gate** or proof of teachability.
Three-hundred-control survival/velocity outcomes do not certify 1,000-control
walking. [W2](student_update_w2_results.md) already separates fitting and
retention gains from walking.

Join A/C by `(restart_id, control_index)` and retain **all common executed
pre-action rows**, across all eight restarts, in fixed order. Each restart must
contribute; actual `0<N≤2,400` is frozen after inspection, without resampling or
an invented minimum. Review adequacy explicitly before launch. State paths and
labels differ between arms; preserve failed prefixes and missing post-fall rows.
Report distinct physical/effective-input counts separately from source-event N.
Owner held restarts included here become fitting data, not F1 held-out tests.

Unresolved bindings: owner source/configuration/audit/ledger; actual alpha* and
calibration; contexts, carried plans/random streams, eight restart/row rosters;
qualification and N; parent ownership/envelope and historical 2,000-row held-data
hashes; runtime/XML/source; capacity and seed freshness. These are placeholders,
not frozen prerequisites.

## Learner and matched continuation

Keep W2 s0's recorded 2k actor, `17→128 tanh→128 tanh→6 tanh`, normalization and Adam.
The A/B/C `parent0.pt` aliases original W2 s0 update2000, file SHA-256
`f8014db94bc5f32910cfaf12d0ab38a462bbf950f2567580729d02929e6cc03d`.
Features remain `concat(state[2:10],clip(state[10:19],−10,10))`; retain existing
float32 normalization buffers and `.05` standard-deviation floor, without
re-estimation. Preserve the checkpoint's private Torch-generator state and
declared global envelope; do not claim recovery of unrecorded historical globals.

Create three **minibatch-stream forks**, not new policy initializations: clone
the incoming PCG64 and apply `jumped(s+1)`, `s=0,1,2`. Materialize/hash each
fork before creating its A/C copies. Within each contrast, actor, normalization,
Adam, private/global RNG and minibatch indices match exactly. Across forks,
the PCG state deliberately changes: this is not unchanged historical PCG
continuation. Supplying a new fit-seed argument alone does not override stored RNG.

Fit each arm once for exactly 2,000 updates, batch 256 with replacement:

\[
 L(\theta;B)=\frac1{6|B|}\sum_{i\in B}\|\pi_\theta(o_i)-u_i\|^2.
\]

Use existing float32 fitting, Adam `lr=3e-4`, default moments,
`foreach=False/fused=False`, gradient clip `.5`, `eta_theta=0`,
`prox_reduction=mean`, `anchor_refresh=0`, one CPU/Torch/BLAS thread. Start a new corpus
block retaining Adam and forked PCG. Save actual incoming proofs, all losses,
indices/RNG transitions and final cumulative-update 4,000 checkpoints; reload
and verify continuation exactly. No refits, early stopping or endpoint selection.

Hold the bound A/B/C action function fixed: bounded **Torch `Policy.predict`**
for scoring and execution, retaining its calling conventions, observations,
normalization, clipping and float64 torque conversion. Do not replace deployment
with P1's canonical NumPy map. An optional NumPy reference may audit saved
predictions/parity `≤1e-6` on 32 observations per actor; record batching/grouping
differences. It does not control the simulator.

## Outcomes and stopping

Evaluate six children and one identical shared parent on **both** fresh paired
16-reset primary and 16-reset confirmation panels. Proposed ranges:
`2941000000..2941000015`, `2942000000..2942000015`, subject to pre-freeze collision
checks. Reuse the parent's outcomes across identical fork copies.

Direct actor only, maximum 1,000 controls; walking requires full survival and
velocity MAE `≤.3 m/s` around 1.5. Allow different successful paths. Retain raw
costs, fall terminal 20 once and absorbing 5 per missing control; fabricate no states.

Confirm walking gain only with ≥2 additional successes over **both** matched A
child and parent in the same ≥2 forks on both panels, and no walking/full-survival
regression in any fork. Individual competence requires ≥15/16 walking on both.
This remains conditional on one incoming actor. Report survival, controls,
costs, own/cross/held reconstruction and fitting curves separately. Retention
without walking is partial. Different teacher/student horizons and starts
prevent a paired full-task cost-gap claim.

Stop on failed qualification, provenance, continuation, runtime or budget gates,
or after the fixed fits/panels. Falls are outcomes, not retries. Independently
audit, preserve negatives and recommend one versioned next step; launch none.

## Accounting appendix

For actual N and held size 2,000:

| New F1 work | Roster / hard ceiling |
| --- | ---: |
| Fits / Adam steps | 6 / 12,000 |
| Fitting forward / backward rows | 3,072,000 each |
| Seven-actor own/cross/held predictions | `7(2N+2,000)≤47,600` |
| Parity forward rows | 448 |
| Evaluation attempts / controls / substeps | 224 / 224,000 / 896,000 |
| Worker forward rows | ≤3,344,048; cap 3,400,000 |
| Audit forward rows | cap 300,000, including ≤7,200 A/B/C proxy checks; no physics/gradients/updates |
| New teacher queries / model controls / PI passes | 0 / 0 / 0 |
| Worker/audit/total wall; total CPU | 7,200/3,600/10,800 s; 10,800 s |
| GPU / RAM / output | 0 / 4 GiB / 8 GiB; reserve 2 GiB audit |

Charge constructors, serialization, resets, failed/unknown calls, supervisor,
review, transfer and rendering; reserve closure budgets. Import prior A/B/C
calibration/acquisition once, including unused B: up to 7,200 native controls,
every forecast bank/refinement, agreement forward, safety action and reset.
Compute model controls from actual bank sizes×horizons, not native steps×48.
Keep inherited W2/discovery costs and unavailable physical I/O/money explicit.

Equal data/updates are **not equal acquisition compute**. Before superiority
claims, separately freeze raw cloning, low-pass-target cloning and DAgger under
matched measured hardware/resource budgets. Charge filtering and physical
validation; filtering torques does not certify a teacher or behavioral alignment.
Charge DAgger learner visits, every expert query, safety/fallback and failures.
Report total and consistently amortized reused-data costs, physics, NN rows,
updates, unique labels, CPU/GPU/wall and storage. Require matched expenditure or
cost/outcome dominance; unused allowance is not equal spending. These controls
and any alternating-loop/generalization claims remain outside F1 authorization.
