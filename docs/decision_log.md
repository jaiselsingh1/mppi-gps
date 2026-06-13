# Decision log: idea -> experiment -> evidence -> decision

Chronological chain of every design decision, the experiment that forced it,
and what would falsify it. Companion docs: merge_coupling.md (mechanism),
acrobot_controller.md (controller tuning), literature_review.md (prior-art
verification of each mechanism).

| # | Idea / hypothesis | Experiment | Evidence | Decision |
|---|---|---|---|---|
| 1 | Policy couples to MPPI through the score (track/filter, scheduled trust) | inherited design, discussion | conflates trust with a tuned weight; later runs showed score-coupling starves or distorts | rejected as primary; kept as ablation arm |
| 2 | Coupling = post-update projection with rollout certificate ("optimizer as trust region") | acrobot exp_merge2/3 | works mechanically; certificate never harmed the task | adopted; the core construct (apparently novel — lit review #1) |
| 3 | Merged plan may re-center MPPI's warm start | exp_merge2 iter1 | collection collapse (later attributed partly to hard seeds; compounding risk verified by suboptimal-MPC theory) | execution-only merge; proposal stays pure |
| 4 | KL-gate the blend by policy agreement | exp_merge3/4 (acrobot) | gate starves the loop: 3-8% acceptance, BC stuck at 0.29 | replaced by certificate-only beta line search |
| 5 | MPPI temperature tuned for control is right for GPS | BC plateau test at 3 lrs + lambda sweep | plateau 0.22 at ANY lr; lam=0.15 gives n_eff~112, same task cost, half roughness | lam=0.15+; finding: control-optimal temperature is pathological for distillation |
| 6 | Acrobot is a viable distillation benchmark | energy shaping, frame_skip, 12s budget (controller solved 8/10 from 2/10) | residual BC floor 0.10 = intrinsic swing-up multimodality | controller findings kept; benchmark switched to Walker2d (user direction) |
| 7 | Walker gait is clonable | walker_gps1 iter0 | BC 0.044, policy walks on all eval seeds after 1 iter | confirmed; walker = main benchmark |
| 8 | Consistency coupling alone closes the loop | gps1 (delta=1%), gps3 (delta=5%) | all consistency metrics improve; survival flat ~200/400 | falsified: labels are not the bottleneck, visitation is |
| 9 | Policy never learns recovery because collection never drifts | failure autopsy (10 episodes) | overspeed runaway 5/10 (no data above 2.5 m/s), backward-pitch 2/10; saturation 0.000 | recovery data is the deficit; dagger episodes + exec noise queued |
| 10 | Consolidated fix: dagger 0.5 + obs-norm + OU noise | gps5 | survival 127->327 over 5 iters, dip at end | mechanism class works; attribution unclear (3 changes at once — flagged) |
| 11 | Per-iter stats recompute caused the gps5 dip; freeze them (GPS/PopArt practice) | gps6 (single change) | flatlined 127->169: adaptivity was an ACTIVE ingredient, not just a hazard | falsified in one direction; -> PopArt-compensated adaptive stats |
| 12 | Compensated-adaptive stats reproduce the climb | gps7 (single change) | 182->141->172: no jump; gps5's spike partly run-to-run variance | variance acknowledged; stop re-rolling, use instrumentation |
| 13 | Recovery labels are unfittable (PLATO drift-gap) | label-residual bins on gps7 (new per-step instrumentation) | (policy-driven, beta=0) bin: residual p90=0.89 — worst by far; ratchet test negative (merge closed-loop benign) | measured root cause -> tempered labels |
| 14 | Two-ring certificate: tight budget for execution (5%), looser for labels (15%) | gps8 (single change vs gps7) | unfittable-label fraction 40%->25%, tempered bin fits 0.34 vs 0.53; survival 118->290 monotone | adopted; current best recipe |
| 15 | Every mechanism checked against primary literature, adversarially | 5 verification agents (see literature_review.md) | merge construct apparently novel; OU noise is our invention; TD-MPC uses 5% mixing + terminal value; PLATO labels are non-adapted | corrections queued as ablations (below) |

## Inverted-experiment thread (Opus continuation, post-Fable)

| # | Idea / hypothesis | Experiment | Evidence | Decision |
|---|---|---|---|---|
| 16 | BC distillation collapses walker's L/R gait into a peg-leg shuffle | phase analysis of gps8 policy | thigh R/L corr +0.48, left thigh std 0.03 rad | PARTIALLY WRONG (corrected #18): train independent policy via RL |
| 17 | Warm-start GPS from an RL policy; certificate shares control | gps10 (mismatched-objective TD3) | certificate distrusted it (acc 0.14); BC destroyed it 1000->123 | falsified by objective mismatch; S-step trust region (anchor) added |
| 18 | The +corr was degenerate (peg-leg) | matched TD3 phase check | TD3 (1000/1000, RL-trained) ALSO has corr +0.43 | CORRECTION: +corr is walker's natural bounding gait; BC's real defect was amplitude collapse (one leg std 0.03), not phase |
| 19 | Objective mismatch alone caused gps10's collapse | gps11: matched TD3 warm-start, single change (matched objective), NO anchor | iter0: SURVIVAL 1000/1000 preserved (vs gps10 123), acc 0.55, mixshare 0.17 | CONFIRMED: mismatch was the whole story; matched competent policy is preserved |
| 20 | The loop *smooths* a competent-but-jittery policy (Fig-3 claim, strongest form) | smoothness before/after 1 GPS iter on matched policy | 0.875 -> 0.415 (2.1x smoother) at 1000/1000 survival, vx 1.34->1.38 | PROMISING: certified policy regularization works; trend pending over more iters |

| 21 | C4 trust ladder with matched policy + mismatch rung | acc/beta/mixshare/cost vs degradation | competent: acc 0.64, cost 0.282 < MPPI 0.337; all degraded: acc <=0.37, cost 0.37-0.43 > MPPI | 3 findings below |

**C4 findings (decision #21):**
- *Earned handover (PASS):* competent policy earns acc 0.64 / mixshare 0.16 and makes MPPI cheaper (0.282 < 0.337); every degraded policy earns <=0.37. Trust tracks quality.
- *Trust = cost-compatibility, not quality:* random-init (passive, ~0 actions) gets MORE acceptance (0.37) than confidently-wrong param-noise (0.09). The certificate rewards do-no-harm, not competence — a refinement of what "trust" means.
- *Mixing is unprotected (important):* bad policies raised cost/step ABOVE pure MPPI (0.43 vs 0.34) because the merge guards execution but 25% of samples still come from the policy; a bad prior wastes that budget and degrades the planner. Direct evidence for TD-MPC's 5% mixing over our 25%, OR gating mixing on acceptance. The P2 certificate claim should be measured on merge_cost_gap (execution), not total cost/step (which includes the unprotected mixing).

## Open ablations owed before any publication claim

1. dagger vs exec-noise attribution (gps5 turned both on).
2. Label budget sweep (0 / 0.15 / larger) — isolates decision #14 cleanly.
3. Mix fraction 25% vs 5% (TD-MPC's operating point).
4. White anisotropic residual-matched noise vs our OU invention (DART-faithful).
5. track/filter (score-coupling) baselines on walker for the paper's comparison table.
6. C4 trust-monotonicity test, full degradation ladder (partial results: coupled
   beats pure MPPI 0.272 vs 0.337 cost/step at the checkpoint rung).

## On the efficiency-vs-RL claim (paper framing — be precise)

The loop consumes ~13k environment steps per run (8 iters x 4 episodes x 400
steps) vs ~1-3M for model-free RL (SAC/PPO) to walk — but we use the true
simulator inside MPPI, so the honest comparison axis is NOT model-free sample
efficiency. The defensible claims: (a) vs model-based RL with planning
(TD-MPC class): no value-function training, no latent model, competitive
wall-clock; (b) vs plain BC-of-MPC at matched budget: the coupling mechanisms
are what close the gap from "smooth but falls" to "walks" (the gps1->gps8
series IS this comparison); (c) the certificate gives a property RL baselines
lack: collection-time task performance is never degraded by the learner.
Positioning closest to PLATO/MDGPS with a novel trust mechanism, evaluated on
distillation efficiency and closed-loop stability.
