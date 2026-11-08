# 2026-11-03 to 11-05: Codex follow-up branches

All four branched from old main (`archive/main-20261107`) and add docs under `docs/` plus
evidence under `results/`. Their ignored `runs/` and `artifacts/` (about 37 GB) are in
`/Users/jaisel/Documents/workbench/mppi-gps-archive-20261107/codex/<branch>/`.

## Whole-trajectory PI with a similarity term — tag `archive/codex/full-pi-alternating-bounded`

Write-up: `docs/full_pi_alternating_results.md`, `docs/full_trajectory_pi_similarity.md`.
Six alternating rounds, coupling α = 0.12 vs 0, three paired learners. Neither arm walked on
held-out starts. **All 72 trajectory solves returned their unchanged input**, so both arms were
identical every round: an optimiser limitation, not evidence about the similarity idea.

## Why full PI failed — tag `archive/codex/local-mppi-coupling-diagnosis`

Write-up: `docs/local_coupling_result.md`. In the first pass only the supplied trajectory
survived the full horizon (1 of 256 samples); with effective sample size 12.8, 72% of the
weight went to samples that fell, so the averaged plans fell. The native calibration
prerequisite failed; nothing was fitted.

## CIO vs our loops — tag `archive/codex/mppi-cio-gap`

Write-up: `docs/cio_optimizer_gap_review.md`, `docs/proposals/cio_memory_h1/`. Read-only review:
earlier loops did not reproduce every CIO ingredient (the noisy reconstruction term, block
alternation with proximal penalties), and the critic-based role swap is not the same algorithm.
A recurrent-memory proposal (H1) was stopped before launch.

## Acrobot and local coupling — tag `archive/codex/full-trajectory-pi`

Write-ups: `README.md`, `docs/acrobot_*_results.md`, `docs/local_coupling_learning_l2_0_result.md`,
`docs/local_coupling_multiround_l3_1_result.md`.
- Acrobot: the teacher never qualified. Controller-loop port 0/10 balanced; near-upright holds
  0/4; energy swing-ups 0/6. A local LQR captures 128/128 near upright, but MPPI swing-ups made
  0/2 basin entries. The model matches dm_control and independent equations (error ~1e-15).
- Walker, coupling inside real H48 MPPI: α = 31.7 cut disagreement but made a teacher fall.
  L2.0 (α = 0.12) kept MPPI walking and improved its survival and cost, but children walked
  0/16. An eight-round run (L3.1) ended with 0/192 standalone walking; its final audit failed.
