# June 2026: the first MPPI-GPS loop (Acrobot, then Walker2d)

**Code:** tag `archive/gps/experiments` (= remote branch `gps/experiments`, 2026-06-13). Later
June work: tag `archive/gps-experiments-20260801` (June 13–14) and tag
`archive/crosspolicy-wip-20260801` (June 15–23).
**Write-ups in that tree:** `docs/research_report.md` (full record), `docs/decision_log.md`,
`docs/merge_coupling.md`, `docs/literature_review.md`, `docs/acrobot_controller.md`.

## Idea

Couple MPPI and a policy both ways: the policy learns from MPPI, and MPPI uses the policy.
The coupling was a cost **certificate**: after MPPI's update, execute the largest blend of MPPI's
plan and the policy whose rollout cost stays within a budget of MPPI's own (line search over the
blend weight). Also: 25% of MPPI's samples centred on the policy's rollout, the policy driving
half the collection episodes, noise added to executed actions while labels stay noise-free.

## Acrobot (machinery development, then dropped)

- Behaviour cloning plateaued at 0.22 MSE at any learning rate: at the control-tuned temperature
  MPPI's labels could not be cloned.
- Raising MPPI's temperature (λ 0.014 → 0.15) kept the task cost, halved label roughness, BC 0.12.
- Gated merges starved the policy of feedback (3–11% acceptance); swing-up labels stayed
  multimodal. The user judged Acrobot swing-up a bad benchmark and moved to Walker2d.

## Walker2d (MPPI K256, H48, λ 3, σ 0.3; 84% healthy steps)

| Run | Change | Survival (of 400 steps) |
|---|---|---|
| gps1, gps3 | merge with 1% / 5% budget | flat, about 220 |
| gps5 | + DAgger 0.5, observation normalisation, OU noise (3 changes) | peak 327 |
| gps7 | normalisation statistics updated without changing the function | 125–182 |
| gps8 | labels under a looser budget than execution (15% vs 5%) | best: monotone climb to 290 |

The gps8 policy walked with one leg barely moving (amplitude collapse).

## The inverted experiment (start from a competent RL policy)

- TD3 trained on a reward reshaped to match MPPI's objective: 1000/1000 survival at 1.34 m/s.
- gps11, the loop warm-started from it: survival stayed 1000/1000 and the policy got 3.1× smoother
  (0.875 → 0.280) over 3 iterations. An earlier TD3 trained on a different reward (gps10) was
  rejected by the certificate and destroyed by the supervised step (1000 → 123).

## What it concluded

1. The cost certificate never let the policy degrade collection.
2. MPPI's best temperature for control is a bad one for distillation.
3. Matching labels alone did not close the loop; visiting the policy's own states did.
4. Distilled policies were 30–100× smoother than MPPI; TD3 was rougher than MPPI.

## June 13–August (tags only)

`archive/gps-experiments-20260801`: a guarded research harness, Walker TD3 gait verifiers, a policy
trust matrix and adaptive mixing gated by policy quality. `archive/crosspolicy-wip-20260801`:
Walker2d env, DDPG/TD3 for Walker, cost-function changes and a SAC cross-policy work in progress.
Pre-October code and Walker teacher evidence are also in
`/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004`.
