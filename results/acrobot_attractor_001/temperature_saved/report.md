# Saved near-upright MPPI temperature calibration

This is arithmetic on 20 predeclared banks from original near-upright case 0,
controls 0, 25, …, 475, refinement pass 0. No candidate rollout, controller
execution, policy forward or fitting was performed. Configuration d14cb419… and
original H5 96408611… are bound in the frozen input roster.

| Quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Original ESS at λ = 3 | 1.0000 | 1.0058 | 2.1569 |
| Candidate cost standard deviation | 103.893 | 122.536 | 188.645 |
| Temperature giving ESS = 32 | 60.723 | 101.194 | 121.738 |

The existing `MPPIConfig.lam_target_neff_fraction=0.125` gives an ESS target of
32 with K=256. Its stock helper matches the independently bisected weights
exactly; maximum achieved-ESS error was 1.07e-13. It uses a spread-scaled bracket
[1e-8, 1e8] times the bank's range and has no fixed absolute λ clipping. No new
temperature implementation is needed in the controller.

Positive temperatures preserve candidate ranking, so the top candidate did not
change. The intervention broadens weight on other candidates and changes the
weighted update; this saved-bank result does **not** predict executed actions,
balance, or subsequent candidate distributions. Those require the separately
frozen native controller contrast. None of the full previous probe's recorded
banks contained a candidate meeting the height hold, so broadening weights is
not by itself a competence guarantee.

The bank table and figure preserve all selected results, not a favorable subset.
Complete measured child CPU was 9.574739 s (including font-cache work), wall
10.307121 s, within the 30 CPU-second/8 MiB allocation. A harmless font-cache
warning was retained. All new physics, fitting and neural counters are zero.
