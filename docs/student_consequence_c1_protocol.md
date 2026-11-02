# C1.0: cost-effect-weighted fitting — recommended next hypothesis for review

**Recommended as the single next hypothesis for review; unimplemented and not authorized to launch.** The completed H5.0 R1 scientific audit and 12-source outcome eligibility now pass. Exact inherited metadata is bound in the [source manifest](proposals/student_consequence_c1_sources.json). C1 selected-byte extraction, reduction/replay proof, fresh-seed checks, capacity and engineering validation remain pending. This is not a runnable frozen recipe. No automatic follow-up, parent refit or second round is permitted.

The question is whether a fixed, bounded native-cost-effect weight improves standalone walking after a fixed learning update, relative to uniform fitting and the same weights shuffled within declared strata. Its prediction is that allocating fit loss to errors with larger measured downstream cost helps execution when those effects remain relevant as the actor changes; the null is that this association adds no confirmed walking benefit. Only the fitting loss changes. This is a learner diagnostic, not a new MPPI–GPS formulation or a coupling test. The existing direct six-torque interface stays fixed; no CIO–GPS reproduction or failure of reproduction is claimed.

## Evidence and eligibility

H5.0 R1 completed and passed its full independent scientific audit. Independent teachers walked in 12/12 episodes; each guided arm walked in 11/12, with falls in different paired cases. The frozen teacher gate prevented every child fit and standalone panel. This is a teacher-competence negative result; it did not test the proposed student update. All 12 independent sources qualify for this next diagnostic's source-outcome gate.

The bound roster is four audited 1,000-control independent episodes per stream, 4,000 unique rows per stream. Preserve all 12 qualified sources and their full-audit chain. Use every source row in fixed episode/control order. Bind the completed R1 result/accounting/audit; source worker `ff67495dae33ee43952677d0c0165a43f8d469b0`; full N+1 state, warmstart and integration context; observations, actions, costs and terminal context; saved prediction batches; task/runtime; and each retained `checkpoints/s{s}_parent.pt` with its policy, normalization, full Adam, private PCG64, private Torch, global CPU Torch sidecar and anchor ownership. Actual full-file hashes, source array descriptors, exact episode/agreement joins, three retained-parent checkpoint/actor hashes and six incoming W1/held metric events are in the source manifest. These are metadata-level bindings inherited from the completed audit; no C1 array extraction or numerical assay has yet run.

No guided replacement, favorable subset or reacquisition is allowed. Old W1 discovery episodes lack eligible full simulator contexts. W1 supplies the existing normalization/historical diagnostic data and inherited fitting state, not substitute assay trajectories. The three retained H5 parent streams are the replications; overlapping windows are not independent trials.

## Fixed-prefix action assay

Future C1 preparation must first resolve each source native integration-spec integer and terminal returned-observation binding. The compact packet exposes N+1 state/warmstart/integration boundaries (integration width 91), but only N pre-action observations. Bind the missing semantic fields from the audited external native context/query evidence within the frozen preparation limits; otherwise stop/version. Do not infer a terminal observation or relax the replay gate. Then replay all 12 complete teacher episodes before any substitution. Require byte-exact restoration and all actual states, warmstarts, integration vectors, observations and native costs to match their recorded N+1/N sequences with rtol = 0, atol = 1e-10. Any missing binding or mismatch stops; no tolerance relaxation or source replacement.

For every source control t = 0 through 999, set h = min(100, 1000 - t). Restore that exact source context, execute its saved incoming-parent action once, then the next h - 1 recorded teacher controls. Reuse the exact already audited 1,000-row source-episode prediction batches; no new assay policy forward or MPPI query is needed. Bind their actual float32 outputs and float32 observation conversion. Do not silently replace them with single-row predictions: batched saved actions and single-row deployment have distinct numerical bindings.

\[
\Delta_i=C_{h_i}(x_i;\hat u_i^{\pi_0},u^T_{i+1:i+h_i-1})
          -C_{h_i}(x_i;u^T_{i:i+h_i-1}).
\]

Derive baseline windows from the once-validated full replay. Follow D2's reduction: sum each actual native-cost window using `np.sum(..., dtype=np.float64)`; native cost already includes its terminal charge. After a normal fall, stop physics and add five per missing control only inside the requested h. Save raw cost, terminal metadata and absorbing adjustment separately. Infrastructure errors retain unknown partial work and cannot produce a usable delta. Never create post-fall observations.

This at-most-0.8-second fixed tape is not feedback Q, recovery, intrinsic importance, teachability or whole-policy competence. It measures the consequence of this particular incoming error under a foreign open-loop suffix. D2's ranking limitations and the earlier corrective-data failure remain reasons for caution.

## One weighting rule and two controls

Use the existing native fall-terminal scale of 20 cost units:

\[
r_i=\min(9,\max(0,\Delta_i)/20),\qquad
\tilde w_i=1+r_i,\qquad
w_i=\tilde w_i/\operatorname{mean}_{j=1}^{4000}\tilde w_j.
\]

Compute deltas and normalization once in float64, then freeze/hash the actual float32 weights. Nonpositive effects have raw weight 1; effects 20, 100 and at least 180 give 2, 6 and 10. No error denominator, retuning, minibatch renormalization, resampling or adaptive reweighting is allowed.

Each stream has uniform, cost-effect and one fixed stratum-shuffled-weight child:

\[
\mathcal L_w(\theta)=\frac{1}{6|B|}\sum_{i\in B}w_i
\|\pi_\theta(o_i)-u_i^T\|_2^2.
\]

The null control preserves each episode's weight multiset within nine 100-control blocks covering t = 0 through 899 and four final remaining-horizon bands: 1–10, 11–25, 26–50, 51–100. Within each temporal stratum, stably sort by the saved native-target parent sum-six action error, ties by control index; split into four near-equal rank groups; permute once per group in fixed order. This gives 52 groups per episode and 624 private-PCG64 permutations, returning 12,000 entries. Seeds 2800000000, 2800000001 and 2800000002 remain provisional until freshness review. The null tests association beyond these **coarse** time/horizon/error strata; it does not control every phase or finer error pattern.

Finish the complete assay, then stop before every fit if any actual float32 treatment vector is constant or its single shuffled vector equals it. Do not redraw or retune. That is absence of the planned contrast, not evidence that the corpus is unteachable; uniform teaching would remain unmeasured.

## Matched learning and direct execution

Train nine children, exactly 2,000 updates each, with no parent refits. Within a stream, every arm starts from identical parent tensors, W1 normalization, complete Adam/private-PCG/private-Torch continuation and the same minibatch indices. Set the changed-data block's anchor to that incoming actor, continuing Adam/RNG ownership. Retain width 128, two tanh hidden layers, native clipped 17-component observations, six tanh torque outputs, original normalization with std floor 0.05, uniform sampling with replacement, batch 256, Adam learning rate 0.0003, gradient clip 0.5, eta_theta = 0 and no anchor refresh. Targets are the same executed teacher controls; assay branch states/actions are not training examples. Deploy each torque policy directly, without MPPI or a recovery controller.

Multiply elementwise squared residuals by `w[:, None]` **before the existing global mean**. Preserve the old default/unweighted path and checkpoint keys. Before launch, a meaningful all-one-versus-old-path synthetic regression must verify byte-identical parameter/buffer, Adam, anchor and RNG payloads; unchanged checkpoint keys/schema and scientific continuation fields; and the complete 2,000-update moments/steps and minibatch tapes. Bind each complete checkpoint file separately, including its explicit provenance and wall-time metadata. Record weighted loss, ordinary action MSE and gradient norm from the same paid forward. This regression and recorder engineering are prerequisites, not claimed passes.

Scalar weights apply to all six torques although the measured consequence comes from one direction, and remain stale as the student changes. This diagnostic does not fix coverage, hidden planner context or conflicting targets. Cost-sensitive imitation is established; no novelty or benefit is assumed. [AggreVaTe](https://arxiv.org/html/1406.5979v1) uses expert cost-to-go and learner-induced states, whereas [DAgger](https://proceedings.mlr.press/v15/ross11a.html) and [DART](https://proceedings.mlr.press/v78/laskey17a.html) address distribution/recovery issues this assay leaves open.

## Outcomes, confirmation and interpretation

Evaluate the three parents and nine children on 16 fresh common primary resets and 16 disjoint common confirmation resets, unconditionally. Provisional panels are 2810000000 through 2810000015 and 2820000000 through 2820000015. Final unused-seed checks are mandatory before freeze; do not choose replacements after seeing outcomes.

Confirm weighting only if the weighted child gains at least two walking successes over **each** of its parent, uniform child and shuffled child in the same at least two of three streams; no stream may regress walking or full survival against any comparator. Repeat on confirmation. Individual competence requires at least 15/16 walking on both panels. Longer survival without walking is retention only. Reconstruction, held error and lower cost cannot pass this gate. Different successful motions are acceptable.

If a uniform child learns walking without a confirmed weighted gain, report that as a separate useful teaching result; the weighting claim still fails or remains inconclusive. Apply the same declared walking-gain convention to the parent–uniform comparison when calling a gain confirmed. Do not collapse failure of this weighting rule into failure of learning from these sources. A parent–uniform gain does not isolate the new corpus from the extra updates; there is no old-corpus continuation arm in this single-mechanism contrast. No positive outcome authorizes automatic coupling.

Separately evaluate all four actors per stream on its four source teacher reset contexts: 48 student episodes paired with the 12 already paid teacher baseline replays. These are **in-sample initial contexts**, not the 384 fresh-panel episodes. Report teacher/student complete cost, walking, survival, controls and correctly paired gaps there. No new teacher acquisition is performed.

Use 90,000 new full-data prediction rows across the nine children: own corpus 4,000, historical W1 4,000 and held corpus 2,000 per child. Own-corpus metrics use the four source-episode-aligned 1,000-row batches for both incoming parents and children. Reuse the 12,000 audited incoming source predictions and 18,000 incoming historical rows. Derive fitting MSE/weighted MSE from saved predictions and **float32 training targets**, with float64 residual reductions; the original native-target sum-six agreement scalar is a different metric. Historical W1/held metrics retain their existing 1,024-row batch boundaries. Exact actor/input/batch reuse proof must pass before freezing this reduced census; otherwise stop/version, not silently buy replacement forwards.

## Finite work, evidence and closure

The [source manifest](proposals/student_consequence_c1_sources.json), [machine-readable census](proposals/student_consequence_c1_census.json) and [storage contract](proposals/student_consequence_c1_storage.md) are part of this review protocol. Limits are ceilings, not completion forecasts:

| New work | Maximum |
|---|---:|
| Teacher full replays / substitution branches | 12 / 12,000 |
| Baseline / substitution native controls | 12,000 / 1,140,600 |
| Fresh / source-context student episodes | 384 / 48 |
| Total native controls / physics substeps | 1,584,600 / 6,338,400 |
| MPPI/model queries / parent refits | 0 / 0 |
| Child fits / Adam updates | 9 / 18,000 |
| Fit forward and backward rows, each | 4,608,000 |
| New metric rows / calls | 90,000 / 90 |
| Worker forward rows / calls | 5,130,000 / 450,090 |
| Audit forward rows / calls | 522,000 / 432,090 |
| Combined forward rows | 5,652,000 |

Worker: six hours, 16 GiB RSS, 16 pinned CPU cores. Audit: two hours, 4 GiB RSS, no physics or fit reruns. New output: worker at most 8 GiB, all study-owned output at most 16 GiB. Preparation, engineering tests, filesystem IO, failed/unknown work, audit, transfers and inherited R1 work are accounted separately. Compact transfers are limited to two attempts of at most 128 MiB each.

Use existing `walker-array-stream-v1` named arrays, one group per completed/gracefully failed branch, and one bounded standard NumPy active-prefix spool for durability. No new evidence format is required. Preserve all actual N+1/N prefixes, partial arrays, paid intents/completions and unknown partials. The conditional worker byte bound is 5,760,318,360, subject to the storage contract's guards and fixtures. Raw arrays/checkpoints remain external; publish compact scalars, learning curves, hashes and independent audit.

Any provenance, restoration, numerical, ownership, deadline or capacity failure stops immediately. No retry, replacement, new intervention or second round. Complete the fixed roster when no such stop occurs, audit every saved measurement/binding/accounting entry, report walking/survival/error/cost separately, then produce a candid verdict. A positive learning result permits preparing a separately frozen coupling protocol for review; it does not launch one. **Review/adoption and the remaining C1 prelaunch prerequisites are pending; no research is launched.**
