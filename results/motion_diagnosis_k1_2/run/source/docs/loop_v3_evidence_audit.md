# Saved evidence audit for the Walker loop v3

The missing experiment is an actual child trained on the successful higher-weight guided experience. V2 stopped before that child existed. Its action-agreement rejection and the later corrective-data result do not establish whether these particular guided trajectories teach better, nor do failed torque fits constitute a failed reproduction of CIO–GPS.

This audit reads the actual saved arrays, checkpoint tensors and complete ledgers. It adds **zero controls, neural rows, backward rows or optimizer steps**. The [remote receipt (external evidence)](evidence_availability.md#external-82d8651d5158) passes 252,812 explicitly counted checks; the [local receipt (external evidence)](evidence_availability.md#external-c70766d4fb6f) passes 70. Counts describe this audit's assertions, not independent physical replications. The [checkpoint receipt (external evidence)](evidence_availability.md#external-f9b95125ac28) separately binds original and retained learning state.

## Actual teachers and matched starts

All four episodes completed 1,000 controls and satisfy the recorded native success and survival tests. Complete cost includes the saved terminal/absorbing terms, which are zero for these episodes. Velocity error uses the frozen 1.5 m/s target.

| Recorded episode | Reset seed | Complete cost | Velocity MAE | Rows with a clipped physical velocity |
|---|---:|---:|---:|---:|
| `v2_r0_trial_raw_0` | 943000000 | 338.551202 | 0.127976 | 487/1000 |
| `v2_r0_trial_raw_1` | 943000001 | 421.167134 | 0.160913 | 548/1000 |
| `v2_r0_trial_guided1_0` | 943000000 | 389.267373 | 0.148698 | 597/1000 |
| `v2_r0_trial_guided1_1` | 943000001 | 383.792195 | 0.151674 | 526/1000 |

The higher weight is exactly `0.1199341983507337`. Its mean complete cost is 386.529784 versus independent 379.859168 (+1.756%). The saved aggregate agreement reduction is approximately 0.0655%, below the old 5% proxy gate. Independent and guided each scored 2/2 success/survival. These are two fixed starts, not a population-level teacher result.

Both pairs have byte-identical initial physical state, 9-value warmstart, **full 91-value integration state**, 17-value observation, private 48×6 nominal plan, initial-plan flag and complete PCG64 state. The audit reads every executed trajectory's 1,001 full simulator snapshots and verifies every pre-action training observation against its corresponding physical state and native velocity clipping. Actions are finite, bounded and actually executed. The paired trajectories differ at 995/1000 and 1000/1000 action rows, and 995/1000 and 999/1000 observation rows. This paired-path difference is different from the saved count of 1,283 within-decision selections changed relative to that decision's independent comparator.

The pairs' cost changes differ: the first guided teacher costs about 14.98% more and the second about 8.88% less. The old v2 code would replace a guided slot exceeding its paired independent cost by 5% if it reached data construction. Testing all 2,000 actual successful high-weight rows is therefore an explicit comparison of recorded experience, not a replay of that unexecuted fallback rule.

The exact HDF groups are:

| Role | HDF group |
|---|---|
| Independent, pair 0 | `arrays/a000072040_episode_v2_r0_trial_raw_0` |
| Independent, pair 1 | `arrays/a000081044_episode_v2_r0_trial_raw_1` |
| High guided, pair 0 | `arrays/a000105421_episode_v2_r0_trial_guided1_0` |
| High guided, pair 1 | `arrays/a000116425_episode_v2_r0_trial_guided1_1` |
| Common accepted short recovery | `arrays/a000063019_accepted_short_recovery_data` |

## Training pairs and incoming learning state

The frozen v1 bootstrap loader reads and validates all **4,064** original observation/action rows, their actual pre-action alignment, four successful 1,000-control discovery episodes, and the 64 separately recorded perturbed re-solves. Re-solve labels bind their own selected plan and observation rather than adjacent buffer rows. Its source/configuration bindings and dataset hashes are retained in the remote receipt's `v1_bootstrap.provenance`.

V2's common child prefix was those 4,064 rows plus **32** accepted short-recovery rows. The recovery data remain the old result, not a full-task recovery certification. Adding 2,000 recorded independent or high-guided rows gives two equally sized **6,096-row** datasets with a shared 4,096-row prefix. No unsuccessful low-weight trajectory, actor-generated prefix label or relabelled model candidate is needed for this comparison. Keeping the prefix also avoids silently substituting the later corrective experiment's 5,044-row construction.

The incoming parent is Bfree at 10,000 updates, policy hash `c107f02196502b2e2fe8aed52cea8691e5204c27ef413adfb5fcbd2a41b1bcf0`. The audit independently recomputes that hash from the saved tensors. These three checkpoint files have identical weights, normalization buffers, private Torch generator state, complete Adam state, PCG64 state, update counts, anchor state, fit-data binding, seed and training flag:

- Original fit: `/home/jaisel/mppi-gps-workspace/runs/2026-10-29_cio_fit_v2_001/fits/Bfree/checkpoints/u010000.pt`.
- Retained parent: `/home/jaisel/mppi-gps-workspace/runs/2026-10-29_cio_feedback_v2_001/checkpoints/final.pt`.
- Prior diagnosis input: `/home/jaisel/mppi-gps-workspace/history/artifacts/2026-10-29_policy_instability_diagnosis_001/inputs/checkpoint.pt`.

The original fit and retained checkpoint files have different package hashes because metadata differ; their learning contents match. The full optimizer-tree fingerprint is `b626010269473394f18b47ca70e4b17b5d86e6fdf8979bf11a06fd10c9907218`. All six Adam parameter states have step 10,000. The saved PCG64 state is preserved in the receipt. The checkpoint's fixed float32 observation mean/std equal the original discovery normalization exactly; its fit-data fingerprints match the validated original 4,064 rows. The parent was trained with no parameter penalty. V2's child fitting settings were fixed separately: 2,000 Adam updates, batches of 256, learning rate 0.0003, parameter-mean proximity weight 0.01, refreshed every 250 updates. Whichever settings a new protocol chooses must be identical across arms and bound before launch.

## Planner snapshots, raw evidence and ledgers

The canonical feedback raw HDF is `/home/jaisel/mppi-gps-workspace/runs/2026-10-29_cio_feedback_v2_001/evidence/raw.h5` on `robopil5090`, 41,223,973,666 bytes. The preserved [locator (external evidence)](evidence_availability.md#external-efa52e1357d6) records SHA-256 `b29fdde72df664cfe8df452f1cf3a663e616405bd036e557d73248988715387d`. This new audit records size/mtime and reads actual selected contents; it does not recompute the entire 41 GB hash or physically replay the model corpus. The local compact HDF is an executed-trace export, not a replacement for the full model evidence.

The complete feedback ledger is 325,461,370 bytes with freshly verified SHA-256 `43593c4aad640ae789f160f7ea6186f4631cf4237a8765ac5e8dd6f004dfc147`. All 219,673 event IDs are monotonic, ending in a closed runtime record. Work recounted from the full ledger equals 78,778,992 model controls, 36,405 live controls and 35,207,541 neural rows. No backward rows or optimizer steps were paid in that physical stage.

The audit inspects **all 8,000 private before/after planner snapshots** for the four relevant teachers, verifies plan shape/finiteness and PCG64 ownership, and verifies all 3,996 consecutive private-state transitions. Every one of the 4,000 actual steps has a before snapshot, after snapshot, decision, live query input and live result. At fixed controls 0, 500 and 999 of each episode it also reads and hashes all stored candidate/selection arrays and full live input/output arrays. Selected actions, candidate initial state/warmstart, private reference plans and query simulator snapshots match the executed trace exactly. This checks stored ownership and alignment; it does not prove optimizer arithmetic, complete-bank global optimality or deterministic physics by a new replay.

All frozen source files and configuration hashes of the original fit and feedback runs were rechecked. Existing runs/configurations were opened read-only and remain unchanged.

## What the prior corrective test establishes

The [prior corrective study (external evidence)](evidence_availability.md#external-4d6345f75034) used one actor-prefix-20 acquisition followed by independent MPPI completion and 980 corrective rows. Its two 5,044-row datasets shared the 4,064 discovery rows, and compared ordinary rows at controls 20–999 with corrective rows at the same clocks. Those children and the parent all scored 0/16 success/survival; mean controls were 173.9375, 165.75 and 255.125 respectively. Corrective fitting reduced its local in-sample error without improving walking. That intervention concerns corrective occupancy after an actor prefix. It neither trained nor tested the two guided teachers above.

Established observations are successful independent discovery, successful high-weight guided trajectories, a large teacher–student performance gap, lower saved-data fitting error without walking, exact saved-parent state packaging, and the negative corrective-data result. Clipping, covariate shift, inconsistent torque targets, hidden planner memory and optimizer effects remain possible explanations, not isolated causes. Similarity to present outputs is not a measured guarantee of post-update execution, and different successful paths should remain admissible.

## Audit lineage and cost

The original `audit.py`, `local_receipt.json` and `remote_receipt.json` remain preserved. Their descriptive `velocity_mae` field mistakenly used 1.0 m/s, and their normalization summary used an empty name filter. The corrected `audit_v2.py` and `*_receipt_v2.json` use the frozen 1.5 m/s target and explicit buffer names. Array hashes, matching checks and work recounts were unchanged. Neither version ran dynamics, network inference or learning.

The repeat audits are charged as read-only analysis overhead, with their measured internal wall times and hashes in the [audit manifest (external evidence)](evidence_availability.md#external-f110c2fb80d6). Those times exclude exploratory shell inspection, transfers and tool overhead; they are not an experimental-efficiency measurement. The new fit/evaluation experiment must separately account for every paid query, training row and optimizer update.

## Closed new comparison audit

After the separately frozen v3.1 comparison closed, the [independent outcome audit](../results/guided_data_v3_1/audit/report.md) passed **513,430 checks**, with zero new experimental work. It verifies exact supplied rows and incoming learning state, six checkpoints, matched minibatches and proximal continuation, all 76 native episodes/full simulator histories, fresh paired starts, native cost and success arithmetic, the frozen gate and complete ledger accounting.

The guided-data child does not pass the predeclared improvement rule: its mean cost improves 0.9136% against the parent and 2.9226% against the independent-data child, below 5%, while survival and mean controls regress against the independent child. All three policies score 0/16 walking success. The independent child survives 3/16 full episodes while approximately standing or drifting backward, with velocity errors over 1.5 m/s; these are neither walking successes nor falls. Confirmation and another acquisition correctly do not run. The exact native termination and cost components are retained in [survival detail](../results/guided_data_v3_1/audit/survival_detail.json).

This newly tested data comparison supports a finite negative result, not “no effect”: the learning alternatives change survival, tracking and complete cost differently. The audit's 2.527128064 internal seconds and separate read-only extraction overhead remain distinct from the experiment's 20,029 live controls, 1,129,373 neural rows, 1,024,000 backward rows and 4,000 optimizer steps. No new model controls are paid.
