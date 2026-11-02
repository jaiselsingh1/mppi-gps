# SAC continuation map: can the oracle carry on from MPPI's walks?

**Status: frozen 2026-11-02, before the full run.** Only the pipeline smoke test had run when this was frozen, and it used SAC's own rollouts posing as teachers, so no MPPI result had been seen.

## Question

The idea from the 2026-11-02 field note: an RL policy like the SAC oracle locks onto the first gait that walks, while MPPI finds several. If that is right, some of MPPI's walks (or some moments in them) should lie inside the oracle's region of attraction and others outside it.

This check asks, for every saved MPPI walk and every 25 controls along it:

1. If the oracle takes over at this state, does it keep walking?
2. Does MPPI's own action at this state help or hurt the oracle's continuation, compared with the oracle's own action?
3. When the oracle does keep walking, does it pull the motion into its own gait?

Question 2 is the first measurement of the "continuation compatibility" similarity, `A(s,u) = Q^π(s,u) − V^π(s)`: two actions count as similar when the policy can take one and carry on as itself. Here π is the oracle, so the numbers describe the oracle's gait, not a student's.

## Inputs (all saved, read-only)

From `runs/2026-10-31_action_compatibility_k1_001/`:

| Input | Path | Notes |
|---|---|---|
| 14 walking MPPI trajectories | `inputs/teachers/*.npz` | 1,000 controls each; full 19-value physics state, executed actions. 10 of 14 also store MuJoCo integration state and solver warm start; the 4 `discovery_*` do not |
| SAC oracle | `inputs/checkpoints/reference.pt` | Actor via `LegacySACActor`; the checkpoint also keeps its twin Q-networks (`agent.q1`, `agent.q2`; raw 17 obs + 6 torques → 256 ReLU → 256 ReLU → 1) |
| 16 SAC walking rollouts | `evidence/stage_b/raw.h5`, `episode_k1_primary_reference_*` | Control arm and the reference for "the oracle's gait" |

Task: the native Walker target-velocity task (`NATIVE_TASK` in `diagnostic_policies.py`), 0.008 s per control.

## What runs

Start states: every MPPI trajectory at controls t = 0, 25, …, 975 (40 per trajectory, 560 in all). From each start, four arms, each T = 500 controls (4 s) long:

| Arm | First actions | Then |
|---|---|---|
| `sac` | none | oracle for 500 controls. This is V(s_t) |
| `mppi1` | MPPI's executed u_t | oracle for 499. This is Q(s_t, u_t) |
| `rand1` | one uniform random torque (fixed seed) | oracle for 499. Shows how much a single action can matter |
| `mppi25` | MPPI's executed u_t … u_t+24, replayed open loop | oracle for 475. Shows whether following MPPI for 0.2 s more helps or hurts |

Control: the `sac` arm from the same 40 grid points on each of the 16 SAC rollouts (640 starts). The oracle should keep walking from its own states. If it doesn't, the restore or replay plumbing is wrong.

States are restored exactly (integration state plus warm start) where saved. The 4 discovery trajectories restore from the 19-value physics state only, so the solver warm start comes from one `mj_forward`. The fidelity checks below measure the difference.

Costs use the native task's per-control cost. A fall adds the native terminal cost (20) once, then 5 for each remaining control up to T, the same rule MPPI's rollout scoring uses. Every arm is therefore scored over exactly T controls, and lower is better.

All starts of an arm advance in lockstep. The oracle's `tanh(mean)` is evaluated one row at a time in K1's batch-one order, so the oracle replays its own saved rollouts exactly. (A batched forward pass drifted about 1e-5 in state within 25 controls in the smoke test.)

## Measures

Per start state (MPPI trajectory i, control t):

- **Continues:** the `sac` arm survives all 500 controls. **Continues walking:** it survives and has velocity mean absolute error ≤ 0.3 m/s around 1.5 over its last 400 controls (the first 100 are a 0.8 s handover window). This is K1's walking rule moved onto the continuation.
- **One-action advantage** `A1 = J(mppi1) − J(sac)` and **chunk advantage** `A25 = J(mppi25) − J(sac)`, in task-cost units over T. Negative means MPPI's action made the oracle's continuation better. Report the full distributions, the fraction negative and the fraction where survival flips. `rand1 − sac` gives the scale for A1.
- **Critic advantage** (no physics): `min(q1,q2)(o_t, u_t) − min(q1,q2)(o_t, π(o_t))` at all 14,000 MPPI rows. It is in SAC's reward units (reward = −cost, γ = 0.995, soft Q), so it is compared with −A1 only by rank (Spearman) at the 560 grid states. The critic was trained on the oracle's own states, so the Monte Carlo A1 is the primary measure and the critic is secondary.
- **Gait distance:** K1.2's frozen 25-control window distance R (`motion_diagnosis.segment_distances`: SI units, root-x removed, wrapped angles) to the nearest window of the 16 SAC rollouts, using windows that start at control 100 or later. It is computed twice: for MPPI's own window at t (how far this moment of MPPI's walk is from the oracle's gait), and for the last 25 controls of the `sac` continuation (whether the oracle pulled the motion into its own gait). The control arm excludes its own rollout when finding the nearest window. R is dominated by velocity (K1.2 caveat).

## Fidelity checks (reported, not used to filter)

- Control arm: for the first 25 controls of each `sac` start on a SAC rollout, the maximum state deviation from the saved rollout. Restore and actor order both match the original, so this should be exactly 0.
- `mppi25` prefix: the state reached after replaying 25 MPPI actions against the saved s_t+25, and the replayed prefix costs against the saved costs where they exist. Each is reported separately for exact-restore and discovery trajectories.

## How to read the outcome

These are descriptive readings, not pass/fail gates.

- **Split:** the oracle continues some MPPI walks or phases and not others, and failures concentrate where R to the oracle's gait is large. This supports "the oracle owns one gait". It also makes `A(s,u)` under a value function a usable similarity, so the next step is to measure it under a student's own value function.
- **Covers all:** the oracle continues from ≳90% of MPPI states. MPPI's gaits sit inside the oracle's basin, so having several gaits is probably not what blocks cloning. That points back to fitting and torque attenuation (K1: clone |u| ≈ 0.30 against MPPI 0.44).
- **Covers none:** the oracle falls from ≳90% of MPPI states. MPPI's states are off the oracle's support. A single oracle's value then says little about MPPI's data, and a student would need its own rollouts started from MPPI states (Kendall's restart idea) to learn a value function there.

A1 near zero with `rand1 − sac` also near zero means one action is too small to matter at most states. In that case A25 is the informative column.

## What this does not show

- It covers one SAC checkpoint and seed, so it describes that policy's basin, not "RL policies" in general.
- It trains no student and does not measure whether learning from compatible MPPI actions produces a better student. That is the follow-up if the split reading holds.
- It makes no change to MPPI and adds no new MPPI queries. The executed actions are replayed open loop.

## Cost (estimate)

Each run covers 560 × 4 + 640 = 2,880 rollouts of ≤500 controls: at most 1.44 M controls, 5.8 M physics substeps and 1.44 M actor rows, plus 14,000 × 2 critic evaluations. This should take a few minutes on one CPU. Outputs are under `runs/<date>_sac_continuation_<n>/` (git-ignored): `results.npz`, `summary.json` and `continuation_map.png`.

## Run

```bash
python scripts/diagnostics/sac_continuation.py --smoke            # plumbing only: SAC rollouts as stand-in teachers
python scripts/diagnostics/sac_continuation.py --out runs/2026-11-02_sac_continuation_001
```
