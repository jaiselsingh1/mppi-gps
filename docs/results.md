# Closed Walker v2 results

Raw `runs/` and `artifacts/` links refer to [separately preserved evidence](evidence_availability.md). Summary figures are included in Git.

**The retained policy did not qualify for standalone walking:** final held evaluation scored **0/16 successes and 0/16 full-episode survivals**, with mean **230.8125 of 1,000 controls** and complete cost **4,297.570654**. The physical run closed without a software error. No feedback child was fitted or compared.

The [closed result (external evidence)](evidence_availability.md#external-bee83edb9a94), [frozen configuration (external evidence)](evidence_availability.md#external-c1d0f77c2170), [source manifest (external evidence)](evidence_availability.md#external-f30e1fbbe50b) and [retained checkpoint (external evidence)](evidence_availability.md#external-377f6b888e28) are preserved. The retained actor is the predeclared Bfree 10k policy.

![Closed v2 summary](figures/walker_v2_summary.png)

## Why the teacher-data gate failed

The baseline MPPI teachers completed both walking episodes. Two fixed coupling weights were then tested on the same paired starts, without a standalone-success prerequisite.

| Executed teacher | Weight | Success / survival | Mean complete cost | Step-weighted policy disagreement | Changed decisions |
|---|---:|---:|---:|---:|---:|
| Independent MPPI | 0 | 2/2 | 379.859168 | 1.598099 | 0 |
| Lower-weight guided | 0.02998355 | 1/2 | 2,287.898334 | 1.975384 | 619 |
| Higher-weight guided | 0.11993420 | 2/2 | 386.529784 | 1.597052 | 1,283 |

The lower weight lost one walking episode and failed the cost and disagreement checks. The higher weight walked successfully on both starts. Its **+1.76% mean cost passed the frozen 5% cost guard**, and every teacher check passed except agreement: the measured reduction was only **0.0655%**, below the required **5%**.

That proxy alone rejected the higher-weight data. Both trials were therefore ineligible, and execution stopped at teacher_data_trial_gate_failed. There was no confirmation, feedback child fit or next-child comparison. **Whether these teacher data would improve a child's actual standalone performance was not tested.** Zero completed feedback rounds must not be described as a failed child comparison.

## Fitting and standalone transfer

The offline comparison used the same 4,064 labels, with zero simulated controls and no MuJoCo import. Alegacy at 2k reproduced the original v1 checkpoint hash exactly.

| Fitting arm | Full-data MSE at 2k | At 10k | Mean controls at 10k | Success / survival |
|---|---:|---:|---:|---:|
| Alegacy | 0.200917 | 0.200544 | 253.500 | 0/16 |
| Bfree | 0.168650 | 0.103482 | 227.625 | 0/16 |
| Cmean100 | 0.168651 | 0.103482 | 233.375 | 0/16 |
| Cmean250 | 0.168651 | 0.103483 | 230.250 | 0/16 |
| Dpadded | 0.173500 | 0.109939 | 133.125 | 0/16 |
| Ehistory | 0.115094 | 0.072790 | 130.500 | 0/16 |

Alegacy uses a summed weight anchor; Bfree removes it. Cmean100/250 use parameter-mean proximity with refreshed anchors and behave nearly identically to Bfree at these settings. Dpadded and Ehistory share their 23-input architecture and initialization; only Ehistory receives previous executed actions. History helped saved-data reconstruction against that capacity control, without improving survival. These errors do not establish irreducible target noise or identify the cause of falling.

The additional 2k cohorts also scored 0/16 success and survival: Alegacy averaged 266.5625 controls, Bfree 175.625. All eight fixed cohorts fell before completing the task. The [full-task plot (external evidence)](evidence_availability.md#external-fb380d9fb0b2) shows the gap to 1,000 controls; its [progress snapshot (external evidence)](evidence_availability.md#external-f35ec3cce708) preserves earlier ledger/log prefixes.

[Full fitting result (external evidence)](evidence_availability.md#external-32b31b029fbc), [fitting curves (external evidence)](evidence_availability.md#external-4d6e24048e02), [full-data CSV (external evidence)](evidence_availability.md#external-876ff9c78e0e) and [zero-physics ledger (external evidence)](evidence_availability.md#external-8df8aa619885) remain available. The [offline audit (external evidence)](evidence_availability.md#external-4515d5226b38) verifies work, ownership, 18 full-data error records and legacy reproduction without inference or optimizer replay. Its first manifest-alias failure remains preserved.

## Recovery, coverage and observations

All **32 short recovery branches stayed upright**; 28 passed the last-eight-control cost check. Only one passed the return-to-saved-trajectory distance check, admitting 32 corrective label rows. This restrictive local filter neither means that 31 branches fell nor certifies full-task recovery after a disturbance.

In the frozen clipped-observation metric, **9.006%** of Bfree diagnostic states lay outside the leave-one-trajectory-out reference threshold. Adding the 32 accepted rows changed none of their nearest-neighbor distances. This describes coverage under that representation, not causal coverage or recoverability; see [saved coverage (external evidence)](evidence_availability.md#external-8c6995fca9f0).

The [saved observation profile (external evidence)](evidence_availability.md#external-99beed81d06b) checks the four original successful teachers. At least one physical velocity was clipped by the native observation's ±10 limit in **2,359/4,000 pre-action states (58.975%)**. Recorded observations match that clipping exactly; MPPI uses full physical velocities. This measured information loss does not prove that clipping caused reconstruction error or falling.

For the next formulation, recovery should mean **completing the task after a disturbance**, rather than returning to one saved trajectory. Preserve complete executed outcomes and establish data usefulness through a matched child's actual standalone improvement. Action agreement should be diagnostic rather than substitute for that outcome. These are future requirements; no v2 criterion, controller or result has been changed.

## Auxiliary saved-plan diagnostic

A [post-hoc diagnostic (external evidence)](evidence_availability.md#external-43aafd29b1c2) scored 1,123 separately verified baseline means omitted from guided candidate comparison. Including that available comparator would change the first action at **517 of 3,215 recorded guided decisions**. Many changes are small: median action-change norm 0.00031034, 95th percentile 0.00599224, maximum 0.31197106.

This is a saved-path optimization omission, not an established cause of the measured control failures. No corrected controller trajectory was executed, so no repaired-controller outcome is claimed. The frozen selector and results remain unchanged. The diagnostic paid **53,687 additional neural rows and 13.638 seconds**, with zero physics, backward work or optimizer steps.

## Evidence and cost

The [saved-outcome audit (external evidence)](evidence_availability.md#external-6e4eff2dad9b) passed **4,092,016 checks** across 150 complete episodes, hashing 475,550 datasets and checking 652,373 dataset descriptors. It added no simulation, inference, gradients or optimization. Model-trajectory contents and checkpoint/network correctness remain outside its stated scope. The [summary-plot receipt (external evidence)](evidence_availability.md#external-37a8006b61fb) also records zero experimental work.

| New v2 stage | Controls | Neural rows | Backward rows | Adam steps | Internal seconds |
|---|---:|---:|---:|---:|---:|
| Six-arm fitting | 0 | 15,433,152 | 15,360,000 | 60,000 | 630.415 |
| Physical comparison | 78,815,397 | 35,207,541 | 0 | 0 | 2,209.008 |
| Auxiliary diagnostic | 0 | 53,687 | 0 | 0 | 13.638 |
| New v2 total | 78,815,397 | 50,694,380 | 15,360,000 | 60,000 | 2,853.061 |

The physical stage comprises 78,778,992 model controls and 36,405 live controls, with no recorded query/measurement failure, rejection or unknown work. The [cost-accounting receipt (external evidence)](evidence_availability.md#external-7e68fd2dc7b3) keeps imported **v1 work separate: 51,176,593 controls, 520,433 neural rows and 1,161.884 internal seconds**. Counting v1 once gives 129,991,990 controls, 51,214,813 neural rows and 4,014.945 summed internal stage seconds. These sums exclude validation, auditing, rendering, transfer and other development overhead; no efficiency claim is made.

Physical validation passed 145 tests, with two skips and one expected stock-Gymnasium mismatch: [receipt (external evidence)](evidence_availability.md#external-4548dd946002). Pure fitting validation preceded new physics: [receipt (external evidence)](evidence_availability.md#external-ffbfe79ea2a3). Earlier captured validation and incomplete initial capture limits remain in the [independent metadata review (external evidence)](evidence_availability.md#external-6fb0b2dc7820) and original receipts.

The compact [executed traces (external evidence)](evidence_availability.md#external-dc912a2eb1f1) are mirrored locally. The [raw-evidence locator (external evidence)](evidence_availability.md#external-efa52e1357d6) binds the full remote HDF/ledger and compact export. Four videos render saved states with no new controls or neural rows:

- [Independent MPPI walking (external evidence)](evidence_availability.md#external-f2d703bf00df).
- [Higher-weight guided MPPI walking (external evidence)](evidence_availability.md#external-dadfa86d7791).
- [Lower-weight guided MPPI falling (external evidence)](evidence_availability.md#external-5c6a5eeb1433).
- [Retained standalone policy falling (external evidence)](evidence_availability.md#external-c61567cfda65).

V1 remains negative at its initial transfer gate: MPPI 4/4, policy 0/16 qualification and 0/16 held, zero feedback rounds. Its [preserved report](../archive/cio_walker_v1-2026-10-29/docs/results.md) and [original result (external evidence)](evidence_availability.md#external-b4c716a5cf2b) are unchanged. No population-wide, beyond-horizon ending or beneficial-feedback result is claimed.

## Closed failure diagnosis and one corrective-data intervention

The [focused diagnosis and reproduction guide (external evidence)](evidence_availability.md#external-4d6345f75034) reproduced the retained actor's 181-control / 1.448-second height failure in four native cases, with no differing state/action bytes. Independent MPPI completed the same reset; its first 16 successful controls did not rescue the original actor, which fell at control 54. One frozen actor-prefix-20 corrective acquisition then completed all 1,000 controls and admitted 980 actual teacher rows by the full native task outcome, without a distance filter.

The matched 2k-update children shared the original 4,064 rows, parent weights/Adam/PCG64/normalization and update budget. On 16 fresh paired starts, **parent, ordinary-data child and corrective-data child all scored 0/16 success and survival**; mean controls were 255.125, 173.9375 and 165.75. Treatment reduced local in-sample corrective MSE by approximately 63%, and both fits removed the original long straight-joint stall, without improving walking. The unique neural cause remains unresolved; no active policy patch or additional trial is supported by this test. The [saved-evidence audit (external evidence)](evidence_availability.md#external-7fc18c7ea01a) passed 383,389 checks without inference or dynamics replay; the guide records all new work and preserved failures. Earlier v2 results above remain unchanged.

Versioned figures: [focused diagnosis](figures/policy_diagnosis.png) and [matched learning curves](figures/policy_learning_curves.png).
