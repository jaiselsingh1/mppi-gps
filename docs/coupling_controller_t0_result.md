# T0: effective coupling, with a local competence regression

**The calibrated score changes the executed controller and fails the frozen teacher competence criterion. The block is complete; no T3 or fitting will launch.** All 16 episodes and four shadow preparations finished; the saved-only independent cross-check passed.

We compared α=0 against the frozen α=31.734901633936317 in standard H48/K256 MPPI, using the same final L3 s0 coupled actor, four fixed current-student contexts, paired innovations and zero/shadow initial nominals. Each episode allowed 300 controls.

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

Coupling is effective in **8/8** matched comparisons: common-prefix action RMS spans 0.608304–0.638255. First-action RMS spans 0.005861–0.286462, exceeding .01 in 7/8. All 24 condition pairs are preserved without post-fall padding. Innovation/RNG fingerprints match across all four conditions' common bank prefixes; alpha pairs with the same initialization have byte-identical first-pass sampled actions, states and native costs. Later candidate clouds can differ after their nominal updates.

The decisive regression is **restart 1/shadow**: α0 walked all 300 controls at cost 149.470615; coupling fell after 225 at complete cost 652.606164 (native 277.606164, including one 20-unit fall terminal, plus 375 missing-control tail). At the shared initial state, teacher–policy MSE falls 0.174791→0.111632, before paths diverge; first-action RMS is 0.286462. Mean gap MSE on their own visited states also falls .257445→.213238, but those means use different state distributions. All eight coupled episodes have lower visited-state gap MSE; agreement improvement does not establish useful teaching. Cold-start coupling raises task cost in 4/4 cases. Shadow restart 2 improves local walking while restart 1 loses it: equal marginal walking counts 7/8 hide that paired regression. Full survival drops 8/8→7/8.

These are **300-control local teacher outcomes**, not 1,000-control standalone-policy competence or learning gains. Shadow preparations paid extra preceding-state MPPI queries; the actual student action, not the teacher recommendation, connects to the target state. Case 0 overlaps T1's calibration episode/state; cases 1–3 are outside its bank roster, not a confirmation panel.

[Candidate summaries](../results/coupling_calibration_t1_t0_001/t0/candidate_scope_summary.csv) separate first-pass survival fraction from the mean over subsequent controller banks, and sample fractions from fallen weight; shadow-preparation banks are separate. No candidate-survival gate is used. Spectra/10-neighbor R² are descriptive only: separate episodes, fixed W2 normalization, 5 Hz bands, burn 0, shortened Welch windows and whole-restart-held-out controls differ from K1; negative/missing values remain visible. They do not prove learnability or admit data.

[All outcomes, failure traces' bindings,24 pairs, independent verification, figure and cost appendix](../results/coupling_calibration_t1_t0_001/t0/README.md). No new architecture, score tuning, teacher search, neural inference, fitting or L3-audit closure was added during reporting.
