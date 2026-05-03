# MPPI-GPS — consolidated findings (2026-05-02)

I (Claude) burned ~24 hours of conversation chasing the wrong things. This doc is the honest summary so you can start fresh from a known state.

## Current code-level state

- **`configs/acrobot_best.json`** is your tuning output: `K=512`, `noise_sigma≈0.0506`, `lam≈0.001`. **It does not specify `H`**, so MPPI falls back to the dataclass default `H=50` (`src/utils/config.py:11`) — that's a 0.5-second lookahead. If you intended a longer horizon, add `"H": …` to the config.
- `scripts/gps_train.py` is the production GPS loop; matches Kendall's framing as we discussed (collect → MSE-BC train → re-collect with track-cost prior).
- `runs/gps_lambda_*/metrics.jsonl` — old runs, but they were generated under the *previous* config (which had H=256). Treat them as historical, not as a current baseline.

## Vanilla MPPI behavior under the current config (H=50)

5 episodes from random initial states (qpos ∈ U(-π, π), qvel ∈ N(0, 2)), 1500 steps each:

| seed | max_tip_z | success (z≥3.5) |
|---|---|---|
| 100 | 3.56 | ✓ |
| 101 | 4.00 | ✓ (perfect) |
| 102 | 3.35 | ✗ (just below) |
| 103 | 3.26 | ✗ |
| 104 | 2.66 | ✗ |

**~40% success rate.** Two qualitatively distinct regimes:

- **Successful:** shoulder rotates monotonically in one direction → angular momentum builds → tip reaches vertical periodically. Committed actions are mostly ±1 saturated, coordinated.
- **Stuck (low-energy attractor):** shoulder velocity oscillates around 0 — no monotonic energy buildup. Tip bounces between 0 and ~2.5. Committed actions are erratic high-frequency switching. MPPI's H=50 (0.5 sec) lookahead can't see far enough to plan a multi-swing escape, so it just locally minimizes cost in a basin that doesn't reach the goal.

The intermittency you saw with `visualise_rollouts` is exactly this: depending on the initial state, MPPI either lucks into the productive-rotation basin or gets stuck oscillating.

## What I tested and falsified

**Round 1 (lambda / cost / loss):**
- Lambda sweep on vanilla MPPI: `lam=0.005` (old config) was n_eff≈1 collapsed but actually best for swing-up; higher lam diversified the softmin but broke the controller. *Not the bottleneck.*
- Cost-function variants (`dist²+terminal`, gaussian-tolerance): no variant beat the existing cost at the working regime; at adaptive lam all variants failed. *Not the bottleneck.*
- Weighted vs unweighted distillation (top-32 trajectory weighted MSE vs single-step argmin BC): 0.4% difference. *Not the bottleneck.*

**Round 2 (smooth-MPPI sampler):** built `SmoothMPPI` with cubic-spline-knot noise, autocorr verified (lag-1 from 0.5 → 1.0). You stopped me — this changes MPPI rather than letting smoothness emerge from the GPS loop. **That was the right call.** Memory: `feedback_smoothness_lives_in_gps_loop.md`.

**Round 3 ("multimodality" via state-coordinate binning):** Phase 0c flagged 89% of `(qpos[0], qpos[1])` bins as bimodal; later sweep flagged 44% of `(qvel[0], qvel[1])` bins as bimodal. **Both were binning artifacts** — I was conflating actions at *different* full states that share a partial coordinate. You correctly pointed this out: MPPI is unimodal at the same state. The single-state forced-init diagnostic confirmed std~0.1 (just the noise floor), not bimodal.

**Phase 0 re-analysis of `runs/gps_lambda_*/metrics.jsonl` (old config, H=256):**
- 10/10 GPS runs improve env cost iter-1 vs iter-0 (track-cost integration channel does fire).
- Higher λ_track → faster initial gain, faster collapse to a degenerate fixed point. λ ∈ {0.5, 0.7} stable; λ ∈ {10, 25, 50} collapse by iter ~10–25.

**Kendall-instrumented loop runs** (under H=50 current config, my `exp_gps_loop_kendall.py`):
- λ_track = 0.5: track cost is too weak (track contribution ~4% of env contribution) → no smoothing → eval drifts down.
- λ_track = 7: smoothing fires (jitter var dropped 81%), policy fits MPPI tightly (residual std 0.07), but eval plateaus at MPPI baseline (~0.5 swing-up).
- λ_track = 25 (overnight, 100 iters × 10 eps × 1000 steps): same story but worse — at λ=25 the track cost dominates env cost, MPPI essentially just outputs the policy's actions. Result: MPPI+prior rollouts produce *literally identical numbers* to policy-only rollouts (within 0.01 cost/step per episode). The loop converged to a degenerate fixed point: π fits MPPI, MPPI copies π, both trapped in a small-magnitude trivial regime. Final eval swing-up: 0.40 / render swing-up: 0.12.

## What's actually broken

**MPPI itself is unreliable** under the current H=50 config — ~40% swing-up at best. The GPS loop's best possible fixed point is therefore "MPPI-quality" which is mediocre. No amount of loop tuning fixes this; the loop converges *to* MPPI, not above it.

The natural conclusion under your framing is: the policy is supposed to inject long-horizon information that H=50 MPPI can't see. But vanilla BC of MPPI's first action structurally cannot encode information beyond MPPI's horizon — by construction the targets contain only H-step planning info. So the loop converges to an approximation of MPPI's own (limited) behavior.

## Mistakes I made (so you don't have to argue against them again)

1. Treated 30–50% MPPI swing-up as "the baseline" rather than as evidence MPPI itself was the blocker.
2. Repeatedly proposed sampler-side smoothness fixes (LP-MPPI, OU, splines) when smoothness is supposed to *emerge from* the GPS loop, not be hard-coded into MPPI.
3. Twice misread state-coordinate binning artifacts as MPPI multimodality. MPPI is unimodal at the same full state; different states correctly have different actions.
4. Twice proposed filtered BC (only train on successful trajectories). You shut this down both times because **filtered BC can't learn to recover from failure states** — the policy never sees what to do at a stuck state, so at deployment it can't escape. **Don't propose this again.**
5. Built a synthetic test-grid eval for the Kendall mechanism instead of using natural training-data metrics (continuous state/action — no need for discrete probe states).
6. Did not check `configs/acrobot_best.json` for changes during the conversation — it had been updated by your tuning, removing `H` (so it defaulted to 50). Several of my mid-conversation MPPI runs were silently on a 0.5-second-horizon controller while I was reasoning as though it were 2.56-second.
7. Bundled diagnostics ("multiple things together") rather than isolating one variable, against your stated preference for incremental tuning.

## Your framing, as I understand it (corrected)

1. **Within an iter:** policy = E_seeds[MPPI(s)]. MPPI is unimodal at any given state (= smooth_intent + jitter); MSE BC, with enough data, recovers the smooth_intent.
2. **Across iters:** policy as track-cost prior pulls MPPI's commits toward π. MPPI's per-step jitter shrinks → next iter's policy trained on cleaner data → loop converges.
3. **The point of the loop:** MPPI alone is short-horizon greedy (H=50 here). The policy aggregates information across many states/episodes — it implicitly carries longer-horizon info than any single MPPI run can see. Biasing MPPI toward π extends MPPI's effective horizon.
4. **Success criterion:** does adding the policy as prior improve MPPI's task performance over the bootstrap loop? *Not* "does the policy match MPPI."
5. **Constraint:** the policy is *not* supposed to copy MPPI even smoothed-out; MPPI is not optimal, so a policy that matches MPPI is bounded by MPPI.

## Empirically verified about Kendall's mechanism

- The **integration channel does fire**: at sufficiently high λ_track, MPPI's per-step jitter measurably drops (75–81% reduction in `var(a_{t+1} − a_t)`). Confirmed at λ=7 and λ=25.
- **The policy fits MPPI tightly** (BC residual std 0.06–0.07) once the data and training budget are sufficient. Within-iter "averaging out the jitter" does happen.
- **The loop converges, but to MPPI-quality.** End-to-end task performance does not exceed vanilla MPPI's success rate. Sometimes it's worse (at high λ the policy dominates and locks the system into trivial behavior).
- **The "long-horizon info" claim is the thing that doesn't empirically materialize** in the existing pipeline. The loop produces a policy that's a compressed version of short-horizon MPPI, not a longer-horizon controller.

## Things still open / untested

- Whether explicitly extending MPPI's horizon (`H=128` or `256`) would change the picture. The user has tuned other params but not H; might be worth a sweep.
- Whether the policy's training signal can be made to actually contain long-horizon information without the "filtered BC" trap (which discards recovery info). DAgger-style relabeling is one classical answer but heavier to implement; advantage-weighted regression with the trajectory return as the weight is another (downweights failures without discarding them).
- Whether MPPI's failure modes are tied to the cost function (the `2 * (4 − ‖sensordata‖)` term penalizes any large sensor magnitude — under the new sigma it may behave differently than at the old). The cost has not been re-evaluated since the tuning re-write.
- The `bc_history` / `bc_mlp` runs in `runs/` exist with checkpoints + videos but no metrics.jsonl — never inspected here.

## What I am *not* recommending

- Smooth-MPPI samplers (LP, OU, spline-knot).
- Filtered BC (no recovery info).
- Diffusion policy (you said off the table for now).
- 2-component MoG / multimodal heads (no multimodality at the per-state level).
- Cost-symmetry breaking via spatial bias (confounded with goal-shape reshaping).

## Files removed in this cleanup

- `experiments/exp_diagnose_vanilla_mppi.py`
- `experiments/exp_e_mppi_expectation.py`
- `experiments/exp_gps_loop_kendall.py`
- `experiments/render_kendall_policy.py`
- `experiments/results/` (everything under it, including the overnight λ=25 outputs)

`experiments/__init__.py`, `experiments/README.md`, and this `findings.md` are the only things kept.
