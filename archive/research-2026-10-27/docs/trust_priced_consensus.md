# Trust-priced consensus (TPC) MPPI-GPS: design and test stack

Proposal, 2026-10-26. Nothing here is implemented or run yet.

## Goal

Model-based search finds behavior, a global policy learns a reusable version of
it, and the improving policy earns influence over later search. The product is
a policy that runs alone. Walker first, harder environments after.

## What the evidence fixes

1. **Students have only walked from policy-centred labels.**
   - MDGPS, Steps 10–12: 204/300 pooled over six seeds at 75 iterations.
   - Anchor 96/100 and Selective 95/100, seed 0.
   - The weak point is take-off, not final quality. Selective seed 0 had zero dev survival through iteration 29, reached .6 at 39, and first passed the gate at 59. MDGPS seeds took off anywhere from iteration 31 to 73.
2. **Independent MPPI does the task, but its actions can't be learned from the state.**
   - K256 walks 59/64 fresh resets (Step 32).
   - Yet the state explains only .167 of its action variance (Step 33). The rest follows the plan MPPI carries over, which a policy can't see. Picking among branches raises nothing that matters (.134 against .117 for a random branch).
3. **Help measured on cost that the teacher completes doesn't transfer.** The identity J = A_L + G_L in `assistance_transfer_mechanism.md` shows this. Trust has to be measured on what the policy does alone.
4. **Plan scores and learned critics have misled every Walker gate.**
   - gate_v2: 7 of 8 accepted policy suggestions made cost worse.
   - Replanning-value scorer: Spearman .13.
   - Exact deterministic rollouts are reliable, and cheap next to MPPI.
5. **Long-horizon credit works as a terminal value, not as selection.** A terminal value learned from complete outcomes closes .87–.98 of the beyond-horizon gap on the toy tasks. Selection among MPPI's options stops at .43–.78.

Together: labels must come from MPPI centred on the policy, so they can be
learned. Independent MPPI stays in the loop, so the robot visits competent
states from iteration 0. Trust comes from exact rollouts of the policy alone.
Any policy influence on independent MPPI is earned and can be revoked.

## The algorithm

### Components

- **MPPI_I:** the tuned K256 H48 MPPI (`configs/walker2d_best.json`), with its own carried plan and its own samples. It is never seeded from the policy.
- **π:** the deterministic policy, which is the product.
- **E:** Selective's policy-centred E-step, reused unchanged: K256, H32, policy tail of 100, n_eff/K .2, weights tanh(Δ/τ_s)·exp(−Δa²/2σ²). It produces labels. It never drives the robot.
- **τ(s):** the trust field. It is the measured chance that π, acting alone from s, does the task.

### The "magic force"

The magic force is MPPI's residual over the policy, δ(s) = a_exec(s) − π(s).
Trust decides which side gives way, so the force acts in both directions:

| Where π is | Who drives | How π learns | How MPPI_I treats π |
|---|---|---|---|
| Untrusted | MPPI_I; δ can be large | Labels from E at states MPPI_I reaches, including mid-gait states π can't yet reach alone | Ignores it |
| Trusted | π; δ = 0 | Absorbs only small, verified improvements from E | Leans on it, through earned channels c1 to c3 below |

As τ approaches 1 on the states the task visits, δ approaches 0 and the policy
runs alone.

This is CIO turned around:

| CIO | TPC |
|---|---|
| Physics weight | Trust price |
| Contact force at a distance (the non-physical help) | Planner residual δ |
| Phase 3 (fully physical) | Standalone policy |

CIO moves through its phases on a fixed schedule, one way. TPC withdraws help
per state, on measured evidence, and gives it back if competence drops.

### One iteration

1. **Trust labels.**
   - At every 10th executed state from the last iteration, run π alone for W = 250 controls (2 s, five MPPI horizons). The rollout is deterministic and starts from the exact MuJoCo state and warm start.
   - τ* = 1 if the policy survives and its speed MAE is at most .30, the gate's threshold. Otherwise τ* = 0.
2. **Trust model.**
   - τ̂(s) is the Wilson lower bound (z = 1) of τ* over the 20 nearest probed states, in standardized observation space. With z = 1, τ̂ ≥ .8 needs 18 of 20 neighbours trusted.
   - Only probes of the current π count, so trust lapses unless the policy keeps earning it.
   - τ̂ = 0 when s is farther from the probes than their 95th-percentile nearest-neighbour distance: states the policy hasn't shown it can handle are untrusted.
3. **Collection.**
   - 10 episodes of 400 controls from training resets.
   - MPPI_I plans at every control, so its plan stays its own and stays warm.
   - π drives once τ̂ ≥ .8 and keeps driving until τ̂ < .6. Otherwise MPPI_I drives.
   - Controls π drives get N(0, .1²) noise, as in Selective. Controls MPPI_I drives get none.
4. **Labels.**
   - Label every 2nd executed state.
   - Also label states along the step-1 probes, up to their first failure. These are π's own mistakes, which the assisted runs would otherwise hide.
   - Cap the total at Selective's per-iteration count, so the two get the same label budget.
   - Every label comes from E, with Selective's weights.
5. **Fit.** Selective's M-step, unchanged: 2-iteration replay, batch 256, learning rate 3e-4, 2,000 updates.
6. **Accounting.**
   - Executed controls.
   - MPPI_I samples × H × passes.
   - E samples × (H + tail).
   - Probe controls and evaluation controls.
   - Wall time.

Deployment is π alone.

### Earned channels from π into MPPI_I

Each channel is used only after a test shows it helps. All are priced by τ̂,
so they are zero wherever π hasn't earned trust:

- **c1, matching cost.** `PolicyMatching` with α(s) = α₀·τ̂(s). This is the `general.md` rule: "the policy term weight in the cost should be proportional to the policies ability to do the task".
- **c3, handover terminal.**
  - For samples ending at s_H with τ̂(s_H) > 0, add τ̂(s_H)·(V^π_W(s_H) − V̄).
  - V^π_W is computed exactly, by rolling π out from s_H, for the 16 lowest-cost samples only.
  - MPPI_I then prefers to hand the robot over in states where π has shown it can continue. At trusted states this is Bertsekas's rollout step.
- **c2, closed-loop selection.** π chooses which of MPPI_I's equally good segments runs, so later carried plans follow π. This is option 1 in `walking_failure_diagnosis.md` §5.

**Why influence tracks competence here, unlike MDGPS's step rule.** E keeps
MDGPS's logic: labels stay close enough to π for π to follow them. Trust
controls something else: who drives, and how much MPPI_I leans on π. These are
two separate knobs.

### What is new

| Piece | Closest prior work | Difference |
|---|---|---|
| Labels centred on the policy | MDGPS; PI²-GPS; Selective in this repo | None; reused |
| Expert drives where the novice is untrusted | SafeDAgger; EnsembleDAgger; RPI/MAPS (ICLR 2024) | Trust comes from exact rollouts of the policy alone, not a classifier or value bounds. It is revoked each iteration. |
| Policy influence on a sampling planner | TD-MPC2, LOOP, Residual-MPPI, POLO (fixed weights); MPQ(λ) (schedule); Hori et al. 2026 (weights only decrease) | Each channel's weight is earned from measured standalone competence and can be revoked. The MPPI runs on an exact simulator and keeps its own plan. |
| Handover terminal | Bertsekas rollout; POLO | The policy's continuation counts only where it has shown it can finish |
| Withdrawing help | CIO (fixed three phases); Yu–Turk–Liu 2018 and HoST 2025 (helper forces removed one way on competence) | Help is the planner's residual, withdrawn per state on evidence and returned when competence drops |

## Test stack

### Stage 1: no training (robopil5090, about 1.5 h)

The policies are Selective seed 0 checkpoints from iterations 9, 19, 29 and 74.
Iterations 9, 19 and 29 come before take-off; 74 walks 10/10.

**1a. Trust map.**
- Probe each checkpoint (W = 250) from 400 states on 16 independent K256 episodes and from its own episodes. Fit τ̂ on half the episodes and test it on the other half.
- Predictions:
  - Held-out AUC of τ̂ is at least .80 at every checkpoint.
  - The trusted share of independent-MPPI states rises with the checkpoint: 9 ≤ .10, 19 .10–.30, 29 .30–.60, 74 ≥ .90.
- Rule: if fewer than 5% of states are trusted at iteration 29 under MAE ≤ .30, Stage 2 uses survival alone.

**1b. Can earned influence make MPPI_I's own actions learnable?**
- Follows Step 33, with 24 episodes on fresh resets.
- Arms, for the iteration-29 and iteration-74 policies:
  - c1 with α₀ ∈ {0.3, 3}
  - c3
  - c1 (α₀ = 3) + c3
  - plain K256 as a shared baseline (R² .167 in Step 33)
- Measures:
  - Held-out state-only R² of the executed actions (Step 33's k-NN estimator).
  - Survival and MAE.
  - |δ|.
  - Trusted share along MPPI_I's own path.
  - Compute.
- Decision rules:
  - If some arm reaches R² ≥ .30 with the iteration-29 policy while walking ≥ 20/24 with MAE ≤ .25, MPPI's own actions can serve as labels. Stage 2 then adds an arm that learns from MPPI_I's executed actions, with no seeding anywhere.
  - Otherwise labels come only from E.
  - A channel joins Stage 2's TPC arm if it raises the trusted share along MPPI_I's path by at least .10 without costing walking (at most 2 more falls of 24 than plain K256).

### Stage 2: bounded loop (robopil5090, about 3 h wall)

- **Arms:** Selective and TPC (plus the MPPI-action arm if 1b passes). Three seeds each, run from scratch on the same machine, 35 iterations, with the same label and fit budgets per iteration.
- **Primary endpoint:** dev survival and MAE at iterations 20 and 35.
- **Secondary endpoints:**
  - The first iteration at which 5 of 10 training episodes complete.
  - Executed controls.
  - Total simulated controls.
  - Wall time.
  - |δ| and trusted share over training.
- **Prediction:** TPC reaches dev survival ≥ .6 by iteration 20 in at least 2 of 3 seeds. Selective seed 0 had 0 at iteration 20 and .6 at 39.
- **Futility:** stop the TPC arms at iteration 20 if their pooled dev mean steps do not exceed Selective's, paired by seed.

### Stage 3: the claim

- Run the winner to the gate (at most 75 iterations), three seeds.
- Compare against Selective, MDGPS, SAC (the existing 1M-control ladder) and PPO (new, same machine, three seeds):
  - Primary: executed controls to the gate.
  - Always reported alongside: total simulated controls and same-machine wall time.
- Evaluate once on the reserved resets 5,000,000–5,000,199.
- Show the Fig. 3 joint plots and side-by-side videos.

### Stage 4: harder environments

After Walker. Candidates are Walker or Hopper with gaps and obstacles, where a
step has to be prepared beyond MPPI's horizon, and Humanoid. Long-horizon
credit enters as a terminal value learned from complete outcomes, earned and
revocable as in the credit test stack.

## Code

New:
- **`src/gps/trust.py`:**
  - Deterministic probes, reusing `expert_continuation.rollout_policy_prefix`.
  - τ* labels.
  - `TrustField`: Wilson lower bound over k-NN, with an unfamiliarity cutoff from `cKDTree`, as in `own_trajectories_check.unfamiliarity`.
- **`src/gps/handover.py`:**
  - c3 as a `prior_cost`, reusing `WalkerPolicyTail` on the top-m samples.
  - c1 as a `PolicyMatching` whose α is priced by τ̂.

Changed:
- **`scripts/training/train_policy_selective.py`:**
  - `--cfg.execution {policy, assisted}`, plus labels on probe states and the accounting.
  - The default stays `policy`, so Selective reruns unchanged.
  - MPPI_I is loaded through `src/utils/config.load("walker2d")`, with K = 256 asserted. Several helpers silently fall back to K32.
- **`scripts/diagnostics/trust_channels.py`:** Stage 1, reusing `label_floor_knn.knn_errors` for R².

Tests:
- Wilson bound, expiry, and unfamiliar states mapping to 0.
- Hysteresis in who drives.
- c1 and c3 add exactly zero when τ̂ = 0, and MPPI_I's plan and samples are untouched. This uses the monkeypatch pattern in `tests/test_mppi_stream.py`.

## Risks, and what would falsify the idea

- **The policy never earns trust.** τ̂ stays near 0, TPC reduces to Selective's labels on MPPI's states, and take-off doesn't improve. Stage 2's futility rule catches this.
- **Labels from E at states π can't reach yet may not help until it can.** This is the same question as above, answered by Stage 2.
- **Over-trust.** If τ̂ is too generous, π drives into falls. Measured in Stage 1a (AUC) and by falls during collection.
- **Compute.** MPPI_I running at every control costs about 1.7× Selective per iteration. TPC must halve the iterations to the gate to break even on total simulation. Planning with K64 while π drives is the first saving, if it is needed.
