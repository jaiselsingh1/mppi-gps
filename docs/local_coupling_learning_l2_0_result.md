# L2.0 — competent local coupling, retention gains, no walking gain

**The end-to-end comparison completed and passed independent audit. Coupling changed competent local MPPI experience, but the matched learning update did not produce standalone walking. The conditional feedback round was therefore not launched.**

The [frozen protocol](local_coupling_learning_l2_0_protocol.md) used standard H48/K256 receding-horizon MPPI, `walker2d_best.json`, the bound W2 s0 actor and all eight restart contexts. All **24 controller episodes, six 2,000-update fits and 224 standalone evaluations** completed. Physical falls ended episodes, not the roster. The old candidate-survival failure, six-round result and held H1000 experiment remain unchanged.

| Agreement α | Local walking /8 | Local survival /8 | Mean complete cost |
|---|---:|---:|---:|
| 0 | 7 | 7 | 249.74 |
| .12 | 7 | 8 | 184.93 |
| 3.474441329 | 0 | 6 | 415.03 |

These are **300-control (2.4-second) local** outcomes, not full-task teacher evaluations. The weak weight avoided independent MPPI's fall at one restart; its last restart survived but missed the velocity criterion. Its 25.95% mean-cost reduction largely reflects the avoided fall and absorbing tail. The larger weight deteriorated both survival and tracking.

The frozen pre-fit rule selected **α=.12**. Its first executed action differed at the identical contexts by pooled RMS **.04523**, above `.01`. Whole-rollout action RMS was `.62026`; that includes subsequent state divergence. Initial candidate banks matched exactly. Later banks retained paired innovations while plans and states diverged. No reconstruction, spectral or candidate-survival gate admitted the data.

Both arms trained on **2,298 common-prefix rows**, including the failed prefix, with matched incoming tensors, normalization, Adam/RNG and minibatch indices in three prescribed stream forks. Every parent and child scored **0/16 walking on both fresh panels**: the primary difference in walking gain was **zero in every fork**. Coupled children executed more controls in all forks. One gained **4/16 primary and 2/16 confirmation full-survival episodes**, versus zero for its independent child and parent. This is retention improvement; it is not competent walking.

Own-data action MSE fell from `.2022` to about `.1095` for independent data and `.1771` to `.1043–.1049` for coupled data. Held-trajectory MSE worsened from `.1993` in every fit. Thus neither better fit nor more survival established useful walking transfer. **R2 was deliberately not triggered**, rather than left unfinished.

The [evidence capsule](../results/local_coupling_learning_l2_001/README.md) contains outcomes, complete learning curves, joint-angle plots and the passing audit; [accounting and scope](local_coupling_learning_l2_0_accounting.md) are separate. The audit checked all 6,834 banks, independently recomputed 93 preserved full banks and all 224 executed evaluation outcomes. Eight contexts share one source episode; three forks share one actor. No convergence, second-body or superiority claim follows. Equal-compute low-pass cloning and DAgger remain required comparisons.

**One next hypothesis:** new-round-only local fitting loses useful coverage of the original teacher distribution. A separately frozen, equal-size replay-mixture control against new-only fitting could test this while keeping coupling and architecture fixed, with walking as the outcome. Increased held error supports investigating that tradeoff, not claiming it caused failure: the parent never walked, and prior corrective-data failures remain relevant. No follow-up was launched.
