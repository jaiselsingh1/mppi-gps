# Experiment map and POLO comparison — 2026-10-25

This inventory groups seeds and parameter ablations under their actual algorithm.
A bug-fixed rerun is not a new research formulation. It includes completed,
stopped and interrupted studies, validation tests, and proposals that never ran.
The complete chronological record remains [experiment_log.md](experiment_log.md).

## Three different things we have called learning

1. **Action-policy learning:** train a network that outputs actions, then test
   that network alone. MPPI is training-time assistance or a teacher.
2. **Value-guided planning:** train a model that scores future outcomes; MPPI
   still selects actions during deployment. No standalone actor is trained.
3. **Acquisition learning:** train a model to choose which reached state receives
   an expensive MPPI search. The model does not itself control the robot or
   necessarily modify the planner's objective.

The current two walking jobs are in category 2. The separate Walker acquisition
assay was category 3. The successful Anchor/Selective policies were category 1.

## Walking studies and baselines

Unless explicitly marked otherwise, survival here means completing an 8-second,
1000-control episode. MAE is forward-speed absolute error in m/s, including
failed episodes. Results use each study's own panel: differing reset sets,
training budgets, teachers, objectives and seed counts prevent treating this
table as a controlled cross-algorithm ranking. The observed walking gate is
at least 90% survival **and** MAE at most .30. One successful training seed
does not establish robustness across training seeds.

| Family | What learns and how MPPI enters | Deployment | Observed outcome |
|---|---|---|---|
| Raw MPPI teacher / memory baselines | No trained actor or value. MPPI searches locally and normally carries its shifted plan between controls. Later memory baselines also resimulate archived action sequences as candidates. | MPPI | Historical common panel: K32/H48 **39/50**, MAE **.387**; K256/H48 **44/50**, MAE **.162**. These are planner outcomes, not learned-policy results. |
| Ordinary SAC baseline, Oct 11 | Actor and twin critics learn from ordinary interaction. | Actor alone | Completed 1M-control run. Preselected 800k checkpoint: **30/30**, MAE **.19755**; later common panel **50/50**, MAE **.198**. Final 1M checkpoint: 30/30, MAE .23449. One training seed. |
| Early policy-centered MPPI-GPS / MDGPS-inspired, Steps 5–12 | Policy rollouts supply MPPI's nominal; local search supplies first-action regression targets. Tail variants explicitly execute the current policy beyond the local horizon. No learned critic. | Actor alone | Early 25-iteration K/tail/adaptive-step variants mostly **0/150** completions; adaptive variant 1/150. Longer K256+tail, 75 iterations over six seeds: **204/300**, MAE **.326**, mean duration **6.63s**. Seed counts: 34,35,12,25,49,49 out of 50. |
| DAgger, Step 11 | Raw MPPI labels states visited by the learner; supervised action regression. | Actor alone | 75 iterations, three seeds: **0/150**, MAE **.662**, **1.13s**. This used a weaker K32/H32 teacher, so it is not a teacher-capability or total-compute-matched GPS control. |
| Old predictive-check ablation, Step 14 | Same older GPS family; replace worse mean-plan labels with the current policy action. | Actor alone | 75 iterations, six seeds: **55/300**, MAE **.473**, **3.81s**, versus older GPS 204/300. Completed negative ablation. |
| Independent-teacher behavior cloning, Steps 15–16 | MPPI alone generates demonstrations; student fits recorded first actions. A second version smooths the regression labels. Student does not seed the teacher. | Actor alone | Both stopped for futility: roughly 32–34 iterations across six seeds; latest development duration about **1.3–1.4s**. No final evaluation panel. Better label fit did not produce walking. |
| SAC with an MPPI replay stream, Step 19 | SAC learns from mixed actor/MPPI transitions; an actual current-actor rollout supplies MPPI's tail. | Actor alone | Six 200k-control runs: **87/300**, MAE **.883**, **4.58s**; no seed passed the joint gate. Planned matched SAC-only/raw-stream controls did not run, so no causal efficiency conclusion. |
| Teacher-trajectory selection, Steps 20–21 | Independent MPPI creates trajectories; either the learner or a fixed strong SAC policy selects easier-to-fit trajectories, followed by action regression. | Actor alone | Both variants user-stopped around 16–18 iterations; latest development duration about **1.2–1.4s**, without completions. No final panel. |
| New Anchor vs Selective 75, Oct 24 | Policy nominal plus explicit policy tail. Anchor retains old-action targets when search is worse; Selective fits strict improvements with frozen weights. Deterministic tanh actor; no learned critic. | Actor alone | Both completed **75/75**. Fresh 100-reset panel: **Anchor 96/100**, **7.893s**, MAE **.22384**; **Selective 95/100**, **7.806s**, MAE **.20290**. Both passed the observed gate in this one paired training seed. |
| Four-arm split-proposal / learner-value pilot, Oct 25 | Anchor, Selective, half-policy/half-independent proposals plus discrete fallback, and that split plus Vπ. First three use actual T100 policy tails. Critic fits only learner rollouts using TD and return anchors. All still use action regression. | Actor alone | All four completed **15/15**, all **0/100**. Duration/MAE: Anchor **1.786s/.635**, Selective **1.720s/.717**, Split **1.770s/.706**, Critic **1.542s/1.209**. Shorter study and changed settings; not a contradiction of the 75-iteration success. |
| Independent-expert withdrawal | Parameter-space search changes learner actions over an increasing prefix; independent MPPI finishes the episode. Critic estimates **expert suffix cost**. Actual assisted returns verify proposals. No cloning. | Intended actor alone after withdrawal | **Interrupted after 22 updates**, 12 accepted, at prefix 128; never reached 400/full 1000. Recovered latest actor: **0/100**, **.57864s**, MAE **1.70504**. Assisted acceptance did not establish standalone improvement. |
| Option1: independent MPPI experience + learner-owned critic | MPPI supplies transitions and guide starts. Twin Qπ targets always use the current learner's next action. Parameter-space search proposes actor updates, verified by complete learner-only execution. No teacher rescue or action regression. | Actor alone | Completed **200/200**, only **3 accepted**. Final **0/100**, **.89248s**, MAE **1.237366**. Critic MC RMSE worsened from 1189.87 to **2,247,149.50**. No matched control isolates why learning failed. |
| Coupled planner/value, `_001` and corrected `_002` | MLP predicts measured complete suffix costs of a **fixed raw-MPPI + bootstrap-memory reference**. It supplies a terminal score and an extra weighted-mean proposal, subject to empirical audits and full-episode qualification. | MPPI + memory, optionally value | `_001` failed on actuator bounds after **2/24**; preserved. `_002` is the same formulation with bounded mean verification, not another algorithm. It is currently **20/24**, with deployed guidance still **zero**. |
| Paired-endpoint repair | Same fixed-reference target, now a regularized linear value fit to returns and candidate–baseline differences. Query both endpoints of decision-relevant plan pairs, even while guidance is rejected. Identical shared archive, bounded relative scoring, fresh paired calibration. | MPPI + shared memory, optionally value | New study running, **1/24** completed; cycle 2 beginning. Initial controllers identical: **12/16**, **6.977s**, MAE **.43035**. These are baseline results before learning. |

The newer Anchor/Selective pair is not an isolated rerun of the older predictive
check: architecture, search-noise schedule and replay weighting also differ.
Likewise, the four-arm critic pilot changes more than one mechanism. These
comparisons identify outcomes, not a unique cause for every difference.

Evidence for completed recent studies:

- [Anchor/Selective 75 final record](experiment_evidence/2026-10-24_policy_selective_75_seed0/README.md)
- [Four-arm pilot](experiment_evidence/2026-10-25_decoupled_value_15_seed0/README.md)
- [Interrupted withdrawal and recovered actor](experiment_evidence/2026-10-25_expert_withdrawal_walking_seed0/README.md)
- [Option1 final record](experiment_evidence/2026-10-25_option1_mppi_experience_walking_seed0/README.md)
- [Coupled design](coupled_planner_value.md), [failure diagnosis](coupled_value_failure_analysis.md), [paired repair](paired_endpoint_study.md)

## What each critic actually means

| Formulation | Whose future is predicted? | What the prediction changes |
|---|---|---|
| SAC | Current actor's action value | Actor gradient update |
| Four-arm split-critic | Current learner's Vπ | MPPI label scoring and regression admission |
| Expert withdrawal | Independent expert's suffix | Learner-prefix parameter-search score |
| Option1 | Current learner's Qπ, even for teacher transitions | Whole-policy proposal score, followed by actual learner-return verification |
| Coupled and paired-endpoint | One fixed restarted reference planner with its initial archive | Deployed MPPI plan selection |
| Learned acquisition assays | Expected outcome of requesting another MPPI continuation | Which reached state gets a query |

An expert's successful future is not automatically a future the learner can
execute. The exact [assistance-transfer counterexample](assistance_transfer_mechanism.md)
demonstrated assisted cost improving while standalone cost worsened, without
critic error. It motivated separating these target meanings.

## Separate acquisition experiments

These were bounded side experiments; none is another active full walking run.

| Experiment | What it tested | Outcome |
|---|---|---|
| Finite fragment/credit toy | Value-independent acquisition discovers experience; measured continuation values propagate credit; tabular policies improve. | Guided planner **238/256** versus guidance-off **0/256**, but complete-trajectory memory **256/256** at lower cost. All 16 final table policies succeed. Qualification consumed **86.2%** of main work. This validates a toy credit mechanism, not superior acquisition or neural MPPI. |
| Learned room/gate request selector | Ridge/UCB selects where to request independent MPPI, against nearest-neighbor and uniform selectors. | Learned median **8,589** controls versus NN **9,244**; paired ratio **.906**, failing the declared 20% improvement criterion. A direct-action memory control used only **672** controls. MPPI was unnecessary for that task. |
| Continuous slalom v1/v2 preflights | Check that the continuous task actually needs expensive MPPI before testing learned acquisition. | V1 MPPI/PD **4/8 vs4/8**; v2 **2/8 vs7/8**. Both failed suitability. No primary acquisition learner was trained there. |
| Walker reached-state query selector | Learned linear predictor allocates **2-second** independent MPPI continuations among already reached snapshots, versus NN and uniform. | Learned mean cost **322.842**, productive **47/64**; NN **293.736**, productive **52/64**. Learned wins **2/8** state banks, median paired cost **14.5% worse**. Negative result; no terminal value enters MPPI, and no 8-second actor is trained. |
| Later exact frontier/fragment formulation | Proposed witnessed graph values and return-and-expand acquisition. | Derivation/specification only. Companion assay was syntax-parsed, not executed. Its first-goal acquisition reduces to search plus memory. |

Side-study sources: [finite toy](../../workbench/mppi_learning_mechanism/acquisition_report.md),
[room/gate selector](../../workbench/mppi_learning_mechanism/learned_acquisition_20261025/results/report.md),
[slalom v1](../../workbench/mppi_learning_mechanism/continuous_acquisition_20261025/preflight_results/report.md),
[slalom v2](../../workbench/mppi_learning_mechanism/continuous_acquisition_v2_20261025/preflight_results/report.md),
[Walker selector](../../workbench/mppi_learning_mechanism/walker_acquisition_20261025/acquisition_results/report.md).

## Diagnostics and validations, rather than new trained algorithms

- **S1 replay/fall autopsy; S3 shadow gates and v2; plan-handoff and future-RNG
  checks:** used frozen policies at saved MPPI states. They exposed scoring
  versus receding-horizon continuation mismatch and random-stream sensitivity.
  They did not train a new policy or qualify an active improved controller.
- **Steps 1–4,13,18 and 22–30:** LQR/exact checks, headroom and tail rescoring,
  label-regression/hidden-history tests, resetting planner memory, and policy
  matching interventions. These are mechanism tests, not additional gait
  training families. Retained-plan raw MPPI's 39/50 became 0/50 when reset every
  control in one diagnostic; planner memory matters to the teacher tested.
- **Expert-withdrawal and paired-pipeline smokes:** exercised integrations,
  finite fall costs, withdrawal, verification and resume. Tiny-horizon results
  validate implementation paths, not walking.
- **Research telemetry, early-stop switches and controller ladder:** tools
  implemented for later studies, not a trained controller themselves.
- **Not run:** old standalone sigma-schedule Step 17; broader S3 active gate;
  reverse-handover bounded mechanism assay; slalom primary acquisition studies;
  missing matched SAC-stream controls. Their proposals are not results.

## Relationship to POLO

POLO deploys MPC with a learned terminal value. It trains using planner-optimized
N-step bootstrapped value targets; randomized-prior value ensembles supply
optimistic trajectory exploration. It requires no separately trained actor.
See [the original paper, §§2.4–2.5](https://arxiv.org/pdf/1811.01848).

Our coupled and paired pilots share that planner/value architecture. Their
complete measured **fixed-reference-controller** returns and conservative
qualification have different semantics. They do not implement POLO's optimistic
ensemble exploration. No matched POLO reproduction or benchmark has run.
Paired acquisition repairs labels and comparisons; it has not demonstrated a
new exploration principle or sample-efficiency advantage. The older actor
distillation and query-allocation studies address different learning roles.

## Live snapshot

Checked **2026-10-25 23:43 EDT / 2026-10-26 03:43 UTC**; both drivers alive.
The archived [read-only status and evaluations](../runs/experiment_inventory_20261025/live_status.json)
preserve this snapshot separately from each frozen study.

- **Corrected coupled-value `_002`: 20/24 committed**, cycle 21 in qualification.
  Latest 16-reset development panel: designated guided
  incumbent **8/16**,5.5335s,MAE **.52558**,finite cost1381.806; memory
  **10/16**,5.494s,MAE **.48518**,cost1412.714. Deployed alpha **0**; neither
  passes the joint gate. Cross-arm archive-refresh selection differs, so these
  differences cannot be credited to learned guidance.
- **Paired-endpoint `_001`: 1/24 committed**, cycle 2 beginning. Initial 16-reset
  baseline **12/16** for both, MAE **.43035**,cost944.180,alpha0. First cycle
  had insufficient training data to fit a model. No learned improvement result yet.

Each active study ends after 24 cycles and has its own untouched 64-reset fresh
final panel. Current development metrics are not final results. Existing
completion monitors remain active; neither training source was changed for
this inventory.
