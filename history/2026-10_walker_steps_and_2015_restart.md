# October 2026: Walker Steps 1–38, then the 2015-formulation restart

**Code:** tag `archive/fresh/mppi-baseline` (2026-10-04 clean baseline); tag `archive/ciogps`
(Steps 34–38 and the restart, 2026-10-27 to 10-30; never pushed).
**Write-ups in `archive/ciogps`:** `docs/established_results.md` (what held, with line references),
`docs/experiment_log.md` (every step), `docs/mppi_gps.md`, `docs/ciogps.md`,
`docs/acrobot_consensus_coupling.md`. Uncommitted edits from 2026-10-29 are in
`/Users/jaisel/Documents/workbench/mppi-gps-archive-20261107/ciogps/`.

## Starting point (2026-10-04)

Clean CPU MPPI baseline after archiving the June work. The frozen Walker teacher
(`configs/walker2d_best.json`) walked 19/20 eight-second episodes, mean speed 1.461 m/s, speed
error 0.148 m/s.

## Steps 1–38: what held (from `established_results.md`)

Task: Walker2d at 1.5 m/s, 8 ms controls, 1,000 controls per episode. Pass mark: survival ≥ 90%
and speed error ≤ 0.30 m/s.

- **The teacher is solved.** Unmodified MPPI K256/H48 (λ 3, σ 0.3, low-pass noise) passes:
  59/64 fresh starts. K32 survives but tracks speed poorly. Horizons ≤ 16 never walk.
  One teacher episode costs 15–21 min on one core and about 12.3M simulated controls.
- **MPPI's actions are mostly set by its carried plan, not the state.** A state-only predictor
  explains 8% (K32) to 17% (K256) of the executed action. Re-planning from scratch every
  control does not walk.
- **Cloning MPPI stalls.** Students fall after about 1.4 s; fit loss stays near 0.28 (6 seeds).
  DAgger with a weak K32 expert: 0/150.
- **Policy-seeded MPPI-GPS walks in most seeds:** 204/300 pooled (dev panel, caveats in
  `code_audit.md`). Anchor and Selective variants 96/100 and 95/100 on fresh starts (seed 0 only).
- **RL walks quickly.** SAC 30/30 at 800k steps; PPO passes in 250–400k executed controls.
  Starting PPO from MPPI's states did not speed it up. A simulation-efficiency claim against RL
  on Walker is out of reach.
- **A matching term (α/2)‖u − π(s)‖² in MPPI's cost** moves MPPI about a fifth of the way toward
  a fixed policy, never makes its actions state-determined, and at α = 10 stops it walking (1/20).
- **MPPI's states lie mostly outside where a walking policy can continue** (31% survive 2 s).
- The same controller scores differently on different machines (robopil 19/50, robopil5090
  15/50): compare arms on one host.
- Correction: Fig. 3 of Mordatch et al. 2015 shows a bird holding position, not a gait.

## The restart: the 2015 joint optimisation with MPPI as trajectory optimiser (10-28 to 10-30)

- Implemented the NeurIPS 2015 joint optimisation (matching term in MPPI's cost, noisy targets),
  checked its fixed points on an LQ toy.
- Stage 0 (one network fit): the matching term pulled a fresh MPPI run only 9–17% toward the
  network. The first full run and the refinement pilot accepted no candidate trajectory; a third
  to a half fell.
- Pilot without the acceptance test: after 6 rounds the coupled MPPI stayed as far from the
  network as before (gap −0.7%).
- Roles swapped (policy global through its critic, MPPI local at the policy's states):
  pilot 1's trust gate trapped the loop; pilot 1b's pull toward MPPI kept the policy behind TD3;
  pilot 1c (pull filtered by the critic) stood safely 100/100 but barely moved.
- Preregistered but not run: a standing diagnosis and an Acrobot consensus-coupling study.
