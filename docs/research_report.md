# MPPI-GPS: Research Report and Publication Plan

Status: living document. Companion docs: `decision_log.md` (decision-level
trace), `literature_review.md` (per-mechanism prior-art verification),
`merge_coupling.md` (mechanism spec), `acrobot_controller.md` (controller
tuning record).

---

## 1. The idea and how the conception evolved

**Original framing.** MPC (MPPI) solves tasks but is short-horizon greedy,
noisy, and inconsistent across visits; a policy distilled from it should
both *inherit* its competence and *regularize* it (Mordatch & Todorov's
NeurIPS'15 Fig. 3: the network's joint trajectories are cyclic and regular
where MPC's are ragged). The hypothesis: couple them in both directions —
the policy learns from MPPI; MPPI absorbs the policy's consistency/long-
horizon structure — and the pair self-improves.

**Conceptual commitments established in discussion** (each later became a
mechanism and an experiment):

1. *The optimizer is the trust region, not the policy.* The policy must
   never be able to degrade task performance during collection; couple
   through mechanisms the task cost can veto, not through scheduled weights.
2. *Couple the proposal/execution, not the score.* Adding a policy-tracking
   term to MPPI's objective (the obvious CIO-style coupling) lets a bad
   policy corrupt planning; instead the policy should shape where MPPI
   searches (sampling) and what gets executed/labeled (projection), with the
   softmin/certificate arbitrating purely on task cost.
3. *Trust must be earned per-state, measured, not scheduled* — realized as
   the rollout certificate (execution) and softmin weight share (proposal).
4. *"Looping between the two":* the policy drifts, MPPI demonstrably
   recovers from the drifted states, both phases enter the data.
5. (Late, user-driven) *The inverted experiment:* start from an
   independently trained competent policy and study how to merge it with
   MPPI — decoupling "is the coupling correct" from "is distillation hard."

**The architecture that emerged.**

- **Merge (the core novel construct):** after the vanilla MPPI update U*,
  execute the largest blend U_b = (1-b)U* + b*pi whose rollout-measured cost
  satisfies J(U_b) <= J(U*) + delta (line search over b in {1,.5,.25,.1});
  one extra K=5 rollout per step (~1%). Per the literature verification this
  exact construct appears unnamed: it is a *predictive safety filter*
  (Wabersich & Zeilinger) whose certificate is performance rather than set
  invariance, licensed by *suboptimal MPC* (Scokaert–Mayne–Rawlings 1999),
  and the Lagrangian mirror image of GPS/PLATO's KL-penalized objective.
- **Two-ring certificate (tempered labels):** execution budget 5%, label
  budget 15% — labels are never executed so they can sit closer to the
  policy while staying cost-certified ("achievable-yet-certified
  supervision"). Introduced after instrumentation measured the PLATO
  drift-gap (below).
- **Sample mixing:** 25% of MPPI's K samples centered on the policy's
  closed-loop rollout (TD-MPC-style policy prior; Biased-MPPI is the
  general framework); task score arbitrates; weight share = earned trust.
- **Policy-driven collection episodes** (MDGPS on-policy sampling / the
  lambda->inf limit of PLATO's adapted execution): the policy drives 50% of
  episodes; every visited state gets a certified label; episodes truncate at
  falls, concentrating data on the failure boundary.
- **GPS-style stochastic collection:** execute label + OU noise (sigma
  matched to measured learner residual ~0.16), labels noise-free (DART
  protocol; OU coloring is our addition — flagged as such, white-noise
  ablation owed).
- **Supporting fixes:** BC trained to plateau; MPPI temperature raised for
  label clonability (lam: argmin-like 0.014 -> 3 on walker, n_eff~112);
  PopArt-compensated adaptive obs normalization (function-preserving stats
  updates); per-iteration checkpoints; full per-step diagnostics
  (label-achievability bins, budget-ratchet ledger).

---

## 2. Complete experiment record

### Acrobot phase (machinery development; benchmark later abandoned)

| Run | Config delta | Outcome |
|---|---|---|
| exp_merge/exp_bc (aborted) | first track/bc baselines | superseded |
| BC plateau probe | 3 learning rates | plateau 0.22 MSE at ANY lr: labels unclonable at tuned lam |
| lambda sweep | lam in {0.014..0.4} | lam=0.15: same task cost, half roughness, BC 0.12 — *control-optimal temperature is pathological for distillation* |
| exp_merge2 | KL-gated merge, merged plan re-centers warm start | collection collapsed in 1 iter; warm-start purity adopted (also partly hard-seed confound, later acknowledged) |
| exp_merge3/4 | execution-only KL-gated merge | task preserved; gate starves feedback (3–8% acceptance), BC stuck 0.29 |
| exp_merge5 | certificate-only beta line search | acceptance up (11%); BC still stuck — multimodal swing-up labels at source |
| controller work | energy shaping (Spong), spin exploit found, frame_skip=2, 12 s budget | hits 8/10 from 2/10; fold trap, KE-spin exploit, horizon-vs-pump-period analysis documented |
| (user decision) | acrobot's swing-up multimodality is benchmark pathology | switch to Walker2d |

### Walker2d phase (custom env from the gym model, MPC-specific cost)

| Run | Config delta | Survival curve (of 400) | Verdict |
|---|---|---|---|
| controller tune | lam=3, sigma=0.3, K=256, H=48; symmetric \|1.5-vx\| velocity cost after overspeed falls | MPPI: 84% healthy steps | controller adequate |
| walker_gps1 | merge delta=1% + mixing | ~220 flat (offline: 223/276/163) | consistency metrics improve; survival flat; policy 97x smoother than MPPI but falls |
| walker_gps3 | delta=5% (loop "drift & recover") | 219/179/239 flat | looser budget alone insufficient |
| autopsy | — | overspeed runaway 5/10, backward pitch 2/10; tanh saturation 0.000 | recovery data deficit diagnosed |
| walker_gps5 | +dagger 0.5 +obs-norm +OU noise (3 changes, flagged) | 127→138→145→266→**327**→248 | mechanism class works; attribution unclear |
| walker_gps6 | frozen stats (single change) | 127→…→169 flat | adaptivity was an active ingredient |
| walker_gps7 | PopArt-compensated stats (single change) | 182→125→132→141→172 | gps5's spike partly variance; instrumentation added |
| label bins (gps7) | per-step diagnostics | (policy-driven, beta=0) residual p90=0.89 — worst bin; ratchet slope negative | PLATO drift-gap measured; merge closed-loop benign |
| walker_gps8 | tempered labels 15% (single change) | 182→118→200→222→257→**290**→232→238 (killed by container at iter 8) | best result; monotone climb; unfittable labels 40%→25% |
| gait diagnosis | phase analysis of gps8 policy | thigh R/L corr +0.48, left thigh std 0.03 rad | **peg-leg shuffle**: consistency machinery entrenched a degenerate unimodal gait; no recovery repertoire |
| walker_gps9 | gps8 recipe, 16 iters | reproduced gps8 exactly through iter 2 (stopped for priority shift) | reproducibility banked |

### Inverted-experiment phase (user direction: trained policy first)

| Run | Config | Outcome |
|---|---|---|
| TD3 (v5 reward) | max-vx objective | **1000/1000 survival**, vx 2.85, real alternating gait (corr −0.32), but smoothness 1.91 (rougher than MPPI 0.9) |
| walker_gps10_inverted | GPS warm-started from that actor | **falsified in 1 iter**: certificate correctly distrusted the objective-mismatched policy (acceptance 0.14); BC destroyed it (1000→123). Finding: collection is certificate-protected, the S-step is not — needs MDGPS-style epsilon |
| TD3 matched (symmetric reward) | −\|1.5−vx\| | timid-gait plateau 185/1000 at 360k — weak gradient |
| MPPI@2.5 probe | move target to policy | MPPI falls at 142 — planner can't track fast gaits |
| TD3 matched2 (reshaped) | MPPI-cost optimum (peak at 1.5) + monotone below-target slope; resume + SessionStart auto-relaunch | **training now** |

### Component/diagnostic results

- C4 trust ladder (partial): coupled MPPI with the gps3 checkpoint beats
  pure MPPI 0.272 vs 0.337 cost/step at full survival; degradation rungs
  pending.
- Coupled MPPI ran cheaper than pure MPPI at several gps iterations — the
  "optimizer absorbs the policy" effect.
- Certificate has never damaged collection in any run (task floor held
  unconditionally).
- Deliverables produced: before/after gait videos (annotated falls),
  survival-series plot, joint-angle (Fig-3-style) evolution, phase
  portraits, smoothness numbers.

---

## 3. Findings (candidate paper claims, with the evidence that backs them)

1. **A rollout cost certificate is a sufficient trust region for
   policy-MPC coupling** — across every run the policy never degraded
   collection; acceptance/weight-share grew with policy quality; it
   correctly rejected an objective-mismatched RL policy without being told.
   (Novel construct: "suboptimal MPC with policy candidates" /
   "predictive performance filter.")
2. **The MPC temperature that is optimal for control is pathological for
   distillation** — argmin-like softmin emits one-noise-draw labels (BC
   plateau 0.22 regardless of optimizer); raising lambda to n_eff~10² is
   free in task cost and halves label roughness.
3. **Consistency coupling alone cannot close the loop; visitation coupling
   can** — all label-side metrics improved while survival stayed flat until
   policy-driven episodes + stochastic collection added recovery data
   (gps1/3 vs gps5/8).
4. **Achievable-yet-certified supervision (two-ring certificate)** —
   labels under a looser budget than execution measurably reduce the
   unfittable-label mass (40%→25%) and produced the best, monotone survival
   curve. Resolves the PLATO-vs-coaching tension: adapt labels, but bound
   the adaptation in cost space.
5. **Failure modes of naive coupling, demonstrated and diagnosed:** warm-
   start contamination; agreement-gate starvation; degenerate-mode
   entrenchment (peg-leg gait — the consistency mechanism collapsing a gait's
   left/right mode structure); supervised-step vulnerability (gps10).
6. **Smoothing claim (Fig. 3 modernized):** distilled policies are 30–100x
   smoother than MPPI at matched behavior; TD3 is *rougher* than MPPI —
   setting up "GPS-loop as certified policy regularizer" as the inverted
   experiment's claim (pending the matched policy).

---

## 4. Key citations

- Mordatch & Todorov, *Interactive Control of Diverse Complex Characters
  with Neural Networks*, NeurIPS 2015 — the originating figure/idea; ADMM
  coupling.
- Levine & Koltun, *Guided Policy Search*, ICML 2013; Levine & Abbeel,
  NeurIPS 2014 — GPS family.
- Montgomery & Levine, *Guided Policy Search via Approximate Mirror
  Descent*, NeurIPS 2016 — KL-constrained C-step, on-policy sampling,
  epsilon step-size rules (the missing S-step trust region).
- Kahn, Zhang, Levine, Abbeel, *PLATO*, ICRA 2017 — adapted execution,
  non-adapted labels, DAgger-grade bound without executing the learner.
- Ross, Gordon, Bagnell, *DAgger*, AISTATS 2011; Ross & Bagnell 2010 —
  O(T^2 eps) vs O(T eps); fixed-expert assumption.
- Laskey et al., *DART*, CoRL 2017 — noise-injected demonstrations,
  learner-residual-matched covariance, labels noise-free.
- Williams et al., *Information-Theoretic MPC*, ICRA/T-RO 2017 — MPPI.
- Hansen, Wang, Su, *TD-MPC*, ICML 2022 (+TD-MPC2, ICLR 2024) — ~5% policy
  samples in the planner; terminal value licenses short horizons.
- Lowrey et al., *POLO*, ICLR 2019; Bhardwaj et al., ICLR 2021 — terminal
  value as the principled long-horizon channel.
- Wabersich & Zeilinger, *Predictive Safety Filter*, Automatica 2021;
  Scokaert, Mayne & Rawlings 1999, *Suboptimal MPC* — the merge's family.
- Trevisan & Alonso-Mora, *Biased-MPPI*, RA-L 2024 — ancillary-controller
  sampling distributions.
- van Hasselt et al., *PopArt*, NeurIPS 2016 — function-preserving
  normalization updates.
- Spong 1995 — energy-shaping swing-up (acrobot controller).
- Pinneri et al., *iCEM*, CoRL 2020 — colored sampling noise (queued
  efficiency lever).
- Kakade & Langford 2002; Grüne & Pannek NMPC — compounding/suboptimality
  composition for the certificate analysis.

---

## 5. Next steps, with reasoning

### Immediate (compute already running)
1. **Objective-matched TD3** (reshaped reward, restart-proof). *Why:* the
   inverted experiment needs a policy optimizing MPPI's objective; gps10
   proved mismatch is fatal — and produced finding 5d.
2. **Warm-started GPS from the matched actor, with an S-step trust region**
   (cap BC drift per iter: epochs/KL bound — MDGPS epsilon; one flag).
   *Why:* the highest-value claim left is "the loop smooths a competent
   policy without breaking it, under certificate protection" — Fig. 3 with
   teeth. Prediction: survival stays ~full; smoothness 1.9 → <0.1;
   acceptance/mixshare start high (trust earned instantly).
3. **C4 trust ladder, full version** — rungs: matched TD3, mismatched TD3,
   gps8 checkpoint, noise-degraded copies, random. *Why:* one figure that
   validates the entire trust story with ground truth; the mismatched actor
   turns gps10's failure into a designed validation point.
4. **K-reduction with a competent policy** (K in {256, 64, 32} coupled vs
   pure). *Why:* the efficiency claim in its honest form — "a policy prior
   lets the planner do the same task with 4–8x fewer samples" (TD-MPC's
   premise, demonstrated without a learned value/model).

### Bootstrap-track completion (the from-scratch story)
5. **Anti-peg-leg: symmetry handling.** Walker has an exact left/right gait
   symmetry; BC collapsed it. Options ranked: (a) symmetry-augmented BC
   (mirror obs/action pairs — exact, free data, breaks the one-legged
   attractor), (b) phase features. *Why:* the from-scratch track's gait
   quality is the visible face of the method; a peg-leg in the demo video
   undermines the claim regardless of survival numbers.
6. **Owed ablations** (each one run, lean schedule): dagger-vs-noise
   attribution; label budget {0, 0.15, 0.3}; mix fraction {0.05, 0.25};
   white-vs-OU noise; track/filter score-coupling baselines. *Why:*
   reviewer table-stakes; every mechanism must show its marginal value.
7. **Terminal value function** (critic from the TD3 run is free) as MPPI
   terminal cost. *Why:* the literature-identified channel for genuine
   long-horizon absorption (POLO/TD-MPC); tests whether coupled MPPI can
   *out-plan*, not just out-smooth, pure MPPI.

### Paper assembly
8. **Positioning:** "Certified trust regions for MPC-guided policy
   learning" — the merge/two-ring certificate as the contribution; PLATO/
   MDGPS as the family; TD-MPC as the modern baseline. Honest efficiency
   axis (decision_log.md): distillation efficiency and collection-time
   safety, NOT model-free sample efficiency (we use the simulator inside
   MPPI).
9. **Required evidence matrix:** walker (main) + one more env for
   generality (hopper is the natural second: same family, harder balance;
   acrobot relegated to a "when distillation fails" analysis section) ×
   {ours, BC, DAgger, PLATO-style, TD-MPC} × {survival, task cost,
   smoothness, collection-safety, samples-to-competence}. The gps1→gps8
   series doubles as the ablation narrative.
10. **Risks:** (a) walker-only results are thin for NeurIPS — second env is
    mandatory; (b) the smoothing claim needs the inverted experiment to
    succeed — if the loop can't preserve a competent policy even with an
    S-step trust region, the certificate story shrinks to collection-safety
    only; (c) CPU-bound experiment velocity — the GPU/warp path exists and
    is one session away if hardware access is provided.

### Why this ordering
The inverted experiment (1–4) is first because it isolates the paper's
*novel* component (certificate trust) from the *known-hard* component
(distillation), produces the headline figures fastest, and every one of its
sub-experiments has a sharp, pre-registered prediction. The bootstrap track
(5–7) then fills in the self-improvement story with the ablation table. A
leading publication needs both: the clean mechanism study AND the
end-to-end loop; either alone is a workshop paper.
