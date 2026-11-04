# T0: effective coupling, with a local competence regression

**The calibrated score changes the executed controller, but fails the frozen safety-of-performance criterion. Complete this block and stop: no T3 or fitting.** All16 episodes and four shadow preparations finished; the saved-only independent cross-check passed.

We compared α=0 against the frozen α=31.734901633936317 in standard H48/K256 MPPI, using the same final L3 s0 coupled actor, four fixed current-student contexts, paired innovations and zero/shadow initial nominals. Each episode allowed300 controls.

| Restart / initialization | Complete cost, α0→coupled | Controls | Local walking | Common-prefix action RMS |
|---|---|---|---|---:|
| 0 / zero | 158.985→182.303 | 300→300 | 1→1 | 0.6383 |
| 0 / shadow | 183.987→198.230 | 300→300 | 1→1 | 0.6083 |
| 1 / zero | 165.867→184.501 | 300→300 | 1→1 | 0.6174 |
| 1 / shadow | 149.471→652.606 | 300→225 | 1→0 | 0.6362 |
| 2 / zero | 159.834→180.838 | 300→300 | 1→1 | 0.6121 |
| 2 / shadow | 225.859→191.120 | 300→300 | 0→1 | 0.6163 |
| 3 / zero | 113.186→149.573 | 300→300 | 1→1 | 0.6216 |
| 3 / shadow | 129.629→139.607 | 300→300 | 1→1 | 0.6086 |

Coupling is effective in **8/8** matched comparisons: common-prefix action RMS spans 0.608304–0.638255. First-action RMS spans 0.005861–0.286462, exceeding .01 in7/8. All24 condition pairs are preserved without post-fall padding. Innovation/RNG fingerprints match across all four conditions' common bank prefixes; same-initialization alpha pairs have byte-identical first-pass sampled actions, states and native costs. Later candidate clouds can differ after their nominal updates.

The decisive regression is **restart1/shadow**: α0 walked all300 controls at cost149.470615; coupling fell after225 at complete cost652.606164 (native277.606164, including one20-unit fall terminal, plus375 missing-control tail). Its mean executed policy-gap MSE nevertheless decreased, .257445→.213238. All eight coupled episodes have lower gap MSE, so agreement improvement does not establish useful teaching. Cold-start coupling raises task cost in4/4 cases. Shadow restart2 improves local walking while restart1 loses it: equal marginal walking counts7/8 hide that paired regression. Full survival drops8/8→7/8.

These are **300-control local teacher outcomes**, not1,000-control standalone-policy competence or learning gains. Shadow preparations paid extra preceding-state MPPI queries; the actual student action, not the teacher recommendation, connects to the target state. Case0 overlaps T1's calibration episode/state; cases1–3 are outside its bank roster, not a confirmation panel.

[Candidate summaries](../results/coupling_calibration_t1_t0_001/t0/candidate_scope_summary.csv) separate first-pass survival fraction from the equal-bank mean over subsequent controller banks, and sample fractions from fallen weight; shadow-preparation banks are separate. No50% candidate-survival gate is used. Spectra/10-neighbor R² are descriptive only: separate episodes, fixed W2 normalization,5Hz bands, burn0, shortened Welch windows and whole-restart-held-out controls differ from K1; negative/missing values remain visible. They do not prove learnability or admit data.

[All outcomes, failure traces' bindings,24 pairs, independent verification, figure and cost appendix](../results/coupling_calibration_t1_t0_001/t0/README.md). No new architecture, score tuning, teacher search, neural inference, fitting or L3-audit closure was added during reporting.
