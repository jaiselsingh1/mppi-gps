# Why the replanning-value Walker controller does not walk (Step 31)

Status: Step 31 diagnosis complete; Step 32 validation running. Preregistered in
[experiment_log.md, Steps 31–32](experiment_log.md). This page collects the evidence,
the alignment check the user asked for on 2026-10-26, and the repair decision.

## 1. Does the pipeline still test the idea in `general.md`?

The user's notes (`Obsidian/labspace/MPPI GPS/general.md`) and Kendall's quoted
constraints set the argument. The replanning-value pipeline
([design](replanning_value_study.md)) is compared with each part below.

| Idea, quoted from `general.md` | What the replanning-value pipeline does | Aligned? |
|---|---|---|
| "MPPI -> local policy/ controller; Policy -> Global Context" | The global object is V^{C_k}, the value of the MPPI+memory controller itself. No policy exists. | Partly. A global learner exists, but it is the planner's own value. |
| "By making mppi more like the policy, it's supposed to absorb some of the policy's long term info ... rather than being short horizon greedy" | The value may only rerank MPPI's own first actions. It never earned influence: alpha was 0 in every development and final panel. | No effect in practice. |
| "Fundamentally the trajectories still need to do the task." "we should assume it's always pretty good / can do the task" | The teacher is K32 MPPI plus memory. It survives 108/192 of its own collection episodes; survivors' speed MAE is .420, none at or below .30. | No. The premise fails before learning starts. |
| "mppi is just a training time thing to efficiently get out policies you can use for 'inference'" | MPPI is the deployed controller, about 2,000 simulated controls per decision. No policy is trained. | No. This is the central departure. |
| "A policy is meant to compress the infinite space of trajectories that do a task ... constrain the trajectories to a learnable set" | There is no policy and no learnable set. | No. |
| "MPPI GPS, more explorative + sample efficient" | The measured product is planner walking. No exploration or sample-efficiency comparison. | Not tested. |
| Kendall: no seeding from the policy; MPPI is local; "close to the policy but better"; trajectories must (a) solve the task and (b) be learnable | No seeding (kept). (a) fails, as above. (b) does not apply, because nothing learns actions. | Partly. |

Verdict: the replanning-value pipeline is a value-guided MPC mechanism test (the
POLO family). Its own design calls it "actor-free" and "not a completed GPS
system". Measured against `general.md`, it defeats the main argument in two ways:

1. Its product is an inference-time planner. The notes treat MPPI as the
   training-time tool and the policy as the product.
2. Its teacher does not do the task. The notes assume it does, and Kendall's
   (a) requires it.

A third point runs through the whole independent-MPPI line. Every attempt that
kept the policy out of MPPI used the K32/H48 teacher: behavior cloning (Steps
15–16), learnable-set selection (Steps 20–21), and the coupled, paired-endpoint
and replanning-value pilots. At K32 with the temperature tuned for K256, MPPI's
weighted mean is nearly one noisy sample (median n_eff 1.6–2.1 of 32). The only
standalone policies that passed the walking gate, Anchor 96/100 and Selective
95/100 (2026-10-24), searched with K256. They also seeded MPPI with the policy's
nominal and tail, which Kendall's constraint now excludes. Teacher capacity and
seeding were never separated, so the evidence does not show that independent
MPPI cannot teach a policy. It shows that the K32 teacher could not.

## 2. Evidence that needed no new simulation

Study directory: `runs/2026-10-26_replanning_value_walker_seed0_001`.

- **Nothing learned ever acted.** Every development and final panel ran at
  alpha 0 with 0% learned scoring (`evaluations/*.json`). The 40/64 final
  result belongs to the alpha-0 memory controller C_23. Links 4 and 5 can only
  explain why learning did not repair it.
- **What C_23 executes.** The first action of the H-step-cheapest of 32 noisy
  samples, 8 resimulated memory fragments and MPPI's weighted mean
  (`src/gps/paired_planner.py`, `plan_step`). MPPI keeps its own mean as the
  carried plan whichever candidate ran. Final panel: memory 34.1%, sample 24.8%,
  raw mean 41.1% of executed controls (`evaluations/final.json`).
- **Speed tracking fails even without falls.** Across the 192 alpha-0
  collection episodes (`collection/iteration_*.pkl`), survivors' MAE is .420:
  .612 in the first 2 s, .356 afterwards at a mean 1.635 m/s. 82 of 84 falls
  are height collapses after the torso pitches forward and speed rises to
  1.75–1.9 m/s.
- **The scorer ranks at chance on fresh decisions.** Each cycle's model was
  checked on its own fresh calibration pairs (`calibration/iteration_*.json`):
  88 pairs, measured with two matched future streams each. On the 44 pairs whose
  two-repeat mean exceeds its standard error, the scorer's choice was better
  20 times and worse 24 (sign agreement 45%). Over all 88, Spearman .13
  (p = .25). Predicted differences have SD 109 against a measured SD of 644. In
  27% of single repeats one side fell and the other survived.
- **Isolated wins did not survive self-guidance.** The two candidates that
  passed pair calibration (cycles 6 and 20) changed 12–85% of actions per
  qualification episode (`qualification/iteration_0005.json`, `_0019.json`).
  Cycle 6: 0/6 survived against memory's 2/6. Cycle 20: 2/6 against 3/6. In
  cycle 20 the four failing episodes changed 66–85% of actions, the two
  survivors 39–46%.
- **Missing evidence and how it was recovered.** Per-control traces were saved
  for one final episode only, and for no development or qualification
  episodes. robopil5090 reproduces recorded episodes bitwise, so all 64 final
  episodes were regenerated there and matched their recorded public records
  exactly (Step 31 D0). This Mac diverges from the recorded actions after 24
  controls, so every comparison runs on robopil5090.

## 3. Matched closed loop on the final resets (Step 31 D1)

Same 64 resets (53000000–53000063) and planner seeds as the recorded final
panel, all on robopil5090. The deployed row is the exact regeneration (D0).
Early falls are falls before 250 controls (2 s).

| Arm | Survived | Time alive | MAE | Cost | Early falls | Gate |
|---|---:|---:|---:|---:|---:|---|
| Deployed C_23 (K32 + memory, argmin) | 40/64 | 5.959 s | .486 | 1286.4 | 15 | fail |
| Memory removed (same argmin rule) | 52/64 | 7.138 s | .387 | 870.2 | 6 | fail |
| Raw K32 MPPI (executes its mean) | 47/64 | 6.873 s | .387 | 923.5 | 7 | fail |
| Raw K256 MPPI, tuned file unchanged | 61/64 | 7.793 s | .151 | 338.3 | 1 | **pass** |
| SAC 800k actor alone (positive control) | 64/64 | 8.000 s | .197 | 358.8 | 0 | **pass** |

Paired reset-bootstrap 95% intervals:

| Contrast | Survival | MAE | Reading |
|---|---|---|---|
| Memory removed − deployed | +.188 [.047, .328]; 18 vs 6 discordant | −.099 [−.125, −.074] | Memory fragments cause falls and slow tracking (link 3 supported). |
| Raw K32 − memory removed | −.078 [−.203, .047] | +.001 | The executed-action rule makes no clear difference. |
| Raw K256 − raw K32 | +.219 [.094, .344]; 17 vs 3 | −.236 [−.257, −.216] | K32 search lacks the gait (link 2 supported). |
| SAC − deployed | +.375 [.25, .50]; 24 vs 0 | −.289 | The objective and dynamics support walking on these starts (link 1 refuted). |

Compute per executed control: deployed 1,971 simulated control slots, raw
K256 12,302 (6.2 times). The K256 planner walks; the K32 planner, with or
without memory, does not.

## 3b. Exact contexts, 0.77 s before a fall (Step 31 D2)

Eight recorded falls (five before 250 controls, three later) and eight matched
survivors. Each context is the exact recorded decision boundary: state,
warmstart, carried plan and planner randomness. All 16 recomputed decisions
matched the recorded action bitwise. Every continuation runs to the end of the
episode with fresh future streams.

| Contexts | Deployed C_23 | Raw K32 | Raw K256 | SAC |
|---|---:|---:|---:|---:|
| Early, failure | 22/40 | 15/20 | 15/20 | 5/5 |
| Early, healthy match | 21/40 | 17/20 | 19/20 | 5/5 |
| Late, failure | 8/24 | 5/12 | 4/12 | 1/3 |
| Late, healthy match | 19/24 | 11/12 | 11/12 | 2/3 |

- **Early falls are a start-up hazard, not settled 0.77 s ahead.** From the same
  early states, the deployed controller survives about half the time whichever
  way the recorded episode ended. Memory-free MPPI survives 80% (K32) or 85%
  (K256) of the time, and SAC always.
- **Some late falls are settled earlier.** Two of three late-failure contexts
  were beyond both K256 and SAC; those episodes had already drifted from the
  survivors' states.
- **No better first action could be confirmed.** Every distinct first action
  in the deployed bank (40–41 per context) was run twice, then the best was
  run against the baseline on six more streams. No context, failure or
  healthy, showed a confirmed improvement, and the two-stream winners reversed
  at three failure contexts. A perfect first-action scorer would have nothing
  demonstrable to pick.
- **The value can't rank the bank.** The cycle-23 value's predicted differences
  correlate with the measured ones at Spearman −.32 to +.12 (mean about −.03).

Verdict on the five links:

| Link | Verdict | Evidence |
|---|---|---|
| 1. Objective or control semantics | Refuted | SAC 64/64 and K256 61/64 on the same starts; SAC recovers from every early context. |
| 2. Planner proposals lack gait or recovery | **Supported** | K32 misses the speed target even without memory (MAE .387); K256 meets it (.151). |
| 3. Memory retrieval or refresh | **Supported** | Removing memory adds 12 survivors; same-state survival 54% with memory against 80% without. |
| 4. Scorer misranks better available actions | Not a cause | No confirmable better first action; the value ranks at chance on fresh decisions and at these contexts. |
| 5. Self-guidance undoes isolated wins | Explains why learning could not repair walking | Calibrated candidates lost in closed loop while changing 35–64% of actions. |

## 4. Proposed restructure of the learning (for the user's decision)

Nothing below has been implemented or run. It is a proposal that follows from
sections 1–3.

What the evidence allows:

- A teacher that does the task exists with no policy input: unmodified MPPI at
  its tuned K256 (D1: 61/64, MAE .151; fresh-reset check in Step 32).
- Reranking K32 first actions with a learned value has no usable signal. The
  scorer ranked at chance on fresh decisions, first-action effects are swamped
  by MPPI's own future randomness, and isolated wins reversed under
  self-guidance.
- Steps 22–24 found that K32 labels are chosen by the carried plan (the state
  explains 8–10% of label variance). Nobody has measured this at K256.

The loop, with the line of `general.md` or Kendall each step answers:

1. **Teacher, training time only.** Unmodified K256 MPPI with its own carried
   plan and noise. Nothing from the policy enters its initialization, plans,
   sampling or cost. ("mppi is just a training time thing"; Kendall: no seeding.)
2. **Data that does the task.** From each start, run N independent teacher
   continuations for one segment. Discard any that fall or leave the speed band.
   ("Fundamentally the trajectories still need to do the task"; Kendall's (a).)
3. **The policy picks its learnable set.** Among the survivors, keep the
   continuation whose actions are closest to the current policy's, and execute
   it during collection, so the next segment starts where it ended. ("roll mppi
   N times from a given state and take the top K trajs that are close to the
   policy rather than the policy cost"; Kendall: "close to the policy but
   better" and (b).)
4. **Policy update.** Regression on the selected state-action pairs. The policy
   is the product, judged alone on the walking gate and the Fig. 3 joint plots.
   ("A policy is meant to compress the infinite space of trajectories that do a
   task.")
5. **Long-term context.** The policy acts only through step 3. As it improves,
   its choice among MPPI's own task-solving options carries its gait cycle into
   the data. That is how MPPI would "absorb some of the policy's long term info"
   here without seeding. This step is an interpretation, untested.

This is Step 20's selection idea with the two changes Step 31 motivates: a
teacher that does the task, and task-first filtering. It drops the memory
archive (harmful at K32) and the first-action value reranker (no signal). It
does not by itself show better exploration or sample efficiency.

Cheapest falsification before any training run: label learnability at K256.
From teacher states, run four independent continuations per state. Measure the
held-out label variance a state-only regressor explains, for K32 and K256
teachers, and whether choosing the continuation closest to a regressor fitted on
the other half raises it. If K256 labels with selection stay as plan-chosen as
K32's, the loop fails before it costs a training budget.

Decisions this needs from the user:

- Does executing the closest-to-policy segment (step 3) fit Kendall's
  constraint? It never touches MPPI's search, but the policy chooses which of
  MPPI's own solutions runs.
- Training-time compute: K256 costs about 6.2 times K32 per executed control.
