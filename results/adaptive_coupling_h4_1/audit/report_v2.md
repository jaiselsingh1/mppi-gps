# Independent H4.1 saved-evidence audit

The first prospective feedback-selected weight reduction completed as a valid experiment and failed its fixed standalone task criterion. All six teachers walked successfully, all three matched children were actually trained, and no confirmation or second round ran.

The core local audit passed **772,685 checks**, bound **128 files** and verified **1,457 array payloads** in **41.845734 seconds**. It added zero controls, neural forward rows, backward rows or Adam steps. The supplemental exact-reset, same-bank causal and local-locator checks also passed in **0.618001 seconds**, with zero experimental work.

## Measured outcomes

| Arm | Teacher success / survival | Teacher cost | Fresh student success / survival | Mean student controls | Student cost |
|---|---:|---:|---:|---:|---:|
| Incoming updated parent | — | — | 0/16; 0/16 | 235.3125 | 4289.545694 |
| Independent acquisition | 2/2; 2/2 | 382.491768 | 0/16; 0/16 | 125.8125 | 4850.449783 |
| Fixed high weight | 2/2; 2/2 | 371.574504 | 0/16; 0/16 | 169.2500 | 4957.605909 |
| Half-weight backoff | 2/2; 2/2 | 379.479145 | 0/16; 0/16 | 155.7500 | 4843.232106 |

No student survived a complete primary episode. The backoff child had worse complete cost and fewer executed controls than its incoming parent on **every one of the 16 paired primary starts**. A modest cost advantage over the other new-data children does not establish a beneficial learning update, task success or convergence.

Recorded own-dataset action MSE decreased by 12.5398% for independent, 11.5949% for fixed-high and 13.2275% for backoff. Those values were bound to the source records, not recomputed by network inference. All three updates worsened actual standalone execution relative to the entering policy.

## The weight change really affected trajectory optimization

The incoming actor was the actual updated v3.1 guided child, hash `7bf162f6e8c9bb63718e44f8392838503528a6669e7553088a07f13be6036916`, rather than the old Bfree parent. Its complete Adam/RNG/normalization state and 6,096-row teaching history were common to all three new acquisition/fit arms. The backoff coefficient was exactly half the fixed-high coefficient.

Backoff and fixed-high executed different actions at 999 of 1,000 controls in each teacher pair. Their first changed control was 1. At that decision, the supplemental check reads the actual original full HDF and proves byte-identical full simulator snapshots, warmstarts, private planner nominal/RNG, proposal actions, model states/costs, baseline scores, student observations and agreement values. The same actor was used, and the only objective difference was the agreement coefficient, with exactly a factor-two influence ratio. Selected first-action distances were 0.0604335730 and 0.00305474782 in the two pairs. Thus the trajectory influence is an actual controlled weight effect, not a difference in random tapes or incoming state.

All 12,000 private before/after planner snapshots were checked for continuity and paired RNG ownership. Every teacher decision binds the actual updated actor and declared coefficient, while the native producer's policy-coupling flags remain disabled. At 24 fixed/first-divergent complete selection banks, the audit verifies actual pre-action feature conversion, score components, executed controls, and that the private nominal advances its own native mean rather than the selected guided tape.

## Paired gap evidence

Teacher-to-student gaps use the two acquisition contexts as a training-start diagnostic, not fresh held generalization. The supplemental check confirms actual full initial state, warmstart, integration and observation bytes are identical between each teacher and both its incoming parent and its own fitted child. Equal seed numbers alone were not used as proof.

| Arm | Before: parent minus teacher cost | After: child minus teacher cost | Gap increase |
|---|---:|---:|---:|
| Independent | 3923.723110 | 4530.563035 | +606.839925 |
| Fixed high | 3934.640374 | 4580.487122 | +645.846748 |
| Backoff | 3926.735733 | 4300.330057 | +373.594324 |

Both individual gaps widen in all three arms. There is no measured contraction. On the new primary responses every child has zero success and survival; their count ties make the implemented rule propose **hold**, even though all updates regress relative to the parent. This is a blind spot of the count-loss trigger. It is only a saved next-weight proposal: no subsequent acquisition was executed.

Primary native terminations were 16 height failures for the parent; 15 angle and one height failure for independent; 15 angle and one height failure for fixed-high; and 13 angle plus three height failures for backoff. These identify native failure boundaries, not unique neural causes.

## Accounting, storage and limits

The exact new experimental census is 74,569,728 model controls plus 18,405 live controls, totaling 74,588,133 controls and 298,352,532 native physics substeps. Neural rows reconcile as 44,083,844 guidance rows + 6,000 teacher agreement diagnostics + 1,536,000 training forwards + 97,152 full-data diagnostics + 12,405 standalone actions = 45,735,401. Backward work is 1,536,000 rows and 6,000 Adam steps. All 78 complete episodes and every declared failure/rejection/unknown counter reconcile; those counters are zero. Conditional confirmation was budgeted and unspent.

The whole original **local** raw HDF is 38,618,575,691 bytes, SHA-256 `5805a0ea194b0cf2228c4a98830f6156887856f6eb46d1790e76942712c7a48c`; the 278,187,145-byte full ledger hashes to `c229397f0e3118da76a3db3ba1e287f8e1353fe26c975ae1e1e7b443e04b5f2d`. The main audit independently hashed the original full local file before and after its checks. The supplemental check binds those independently computed hashes to `exports/full_local_evidence_locator.json`. The legacy export filename `remote_evidence.json` does not describe the actual storage location; the explicit local locator corrects that wording without modifying frozen source. A compact complete-episode/proof export is supplementary and was not substituted for the original raw file in this audit.

All source/configuration/input seals, actual 8,096-row datasets, full child checkpoint/Adam/RNG/anchor packages, restored shared minibatch stream, 6,000 learning points, CSV records, policy ownership, complete native outcomes, paired resets and frozen task/benefit gates passed. No source or input hash changed after launch.

The new arms all used the pinned macOS arm64 native MuJoCo 3.6 platform. The inherited trigger/checkpoint came from Linux x86. Within-round arms are matched; no cross-host bitwise inference/physics, cross-host causal comparison or efficiency claim follows.

This saved-only audit imports no project policy or simulator and executes no network inference, gradients, Adam arithmetic or dynamics replay. It checks full file byte identity, all executed episode arrays, all private planner records, live-query descriptor alignment and 24 selected actual model banks. It does not independently semantically replay every nonselected model-query array, prove planner optimality, or validate all neural forward/optimizer arithmetic. The shared baseline-mean omission remains unchanged; no repaired controller is claimed.

The prospective result rejects this backoff intervention's frozen benefit criterion at this incoming learner and finite budget. It does not establish that all adaptive coupling fails, that a lower fixed coefficient is inherently better, or that feedback scheduling has been compared against open-loop decay. Failed torque-policy transfer also remains distinct from a CIO–GPS reproduction.
