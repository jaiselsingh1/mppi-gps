# Evaluation validity and Walker differential audit

Audit date: 2026-10-23. Baseline source: `01eb594ccba3780f29179f14db3caf9755c5d6dc`, identical locally and on robopil5090 when inspected. This audit began with runner code, raw JSON/NPZ artifacts, installed official Gymnasium code, and git history. `objective.md` and `theory.md` were not used as evidence. Experiment-log decision wording was consulted only after raw-result reconstruction, to distinguish actual recorded decisions from inferences about experiment use.

## Findings by severity

| Severity | Finding | Evidence and consequence | Guard/fix |
|---|---|---|---|
| P1 | Resets 70000–70049 are development data, not a final holdout. | Seven raw evaluation campaigns, 73 policy/checkpoint evaluations and seven planner evaluations reuse the same 50 resets. Follow-up choices include tail/sample count, training duration, adaptation, averaging and horizon. | Reserve fresh range below. No historical result was relabeled as confirmatory. |
| P1 | The frozen Walker is not stock Walker2d-v5, either in dynamics or objective. | Stock v5 right-foot friction is 1.9; local is 0.9. Identical states/actions produce different contact trajectories and sometimes different termination times. The target-speed objective and failure tail also differ from Gymnasium reward. | Differential characterization tests. Preserve existing task and label it explicitly in all claims; stock benchmark claims require a separate standard-task run. |
| P1 | Pooled episode Wilson intervals omit training-seed uncertainty. | The 204/300 result is six trained policies evaluated on a shared set of 50 resets, not 300 independent draws of the algorithm. Per-seed completion ranges 0.24–0.98. | Independent training-seed analysis and crossed training-seed/reset bootstrap in final protocol. Existing summaries are descriptive. |
| P2, fixed | Acrobot search and validation defaults overlap. | Default search 0–4 was entirely included in validation 0–9; explicit overlap was accepted. | Two failing regression tests were recorded before edits. First change sets validation to 100–109; second rejects explicitly overlapping sets. Both pass. Historical selected configurations remain development results. |
| P2 | Provenance is weaker for the Step-5 comparison runner than the standalone evaluator. | `walker_step5_compare.py` stores checkpoint path and results but no checkpoint hash, task hash, source hash, explicit per-policy reset list or package versions. Reset order is inferred from code and corroborated by raw planner rows/report. | Retained raw snapshots/hashes for this audit; final manifest must bind hashes, environment and every reset. No claim that missing provenance was recovered. |
| P2 | Seed-domain allocation is not universal across runners. | Current Walker training/evaluation ranges are disjoint for recorded settings; older demo runners use global RNG, and the old MDGPS formula permits overlapping 10,000-seed blocks for sufficiently long runs. | Seed ledger below; parent audit owns MDGPS guard. Demo results are not independent seeded benchmark evidence. |

## Primary evidence and completeness

`evaluation/server_raw_json.tgz` is a read-only snapshot of JSON/JSONL run files from robopil5090; `evaluation/server_snapshot/` contains the expanded files and `snapshot_manifest.json` hashes each. `server_environment.txt` records Python 3.11.16, NumPy 2.4.3, MuJoCo 3.6.0, Gymnasium 1.2.3 and the remote git hash. The snapshot is not transactional: Step-14 training was running while files were copied. Its incomplete metrics are not final results. No GPU task was launched for this audit.

The earlier Step-4–8 artifacts are not present in the inspected local/server run trees. The tracked log says they reside on another host, `robopil`; those raw results were not independently verified here. Their historical decisions below are explicitly identified as tracked-history evidence, not raw-artifact verification. Deleted/untracked artifacts, external commands and undocumented human choices cannot be exhaustively reconstructed. Thus this is an exhaustive ledger of *observable* use in the inspected sources, not a claim to know every unrecorded decision.

Recompute raw ledgers without running any environment:

```sh
.venv/bin/python docs/audit_evidence/evaluation/analyze_evaluation.py
```

## Every observed use of 70000–70049

The machine-readable `evaluation/reset_reuse_ledger.json` lists every observed campaign, exact checkpoint source, arm, reset range, completion count, mean error and survival time. There are 3,650 policy episode evaluations and 350 planner episode evaluations, many repeated observations of identical policies and resets. They are not 4,000 independent observations.

| Campaign / historical decision | Use of the 50 resets | Evidence status |
|---|---|---|
| Step 5 | Compare initial MPPI-GPS, DAgger, raw planner and SAC; retain MPPI-GPS machinery and choose policy-tail follow-up. Reset 70000 also used for videos. | Tracked experiment-log lines 750–832 and git history; raw Step-5 episodes unavailable here. |
| Step 6 | Test own-policy tail; add time-alive as exploratory descriptor; drop tail from next comparison and increase sample count. Reset 70000 videos. | Tracked history, lines 834–915; raw unavailable. |
| Step 7 | Test K=256 without tail; choose K=256 plus 100-control tail next. Reset 70000 videos. | Tracked history, lines 917–993; raw unavailable. |
| Step 8 | Evaluate combined K=256/tail and keep it as working configuration; motivate adaptive step. | Tracked history, lines 995–1078; raw unavailable. |
| `2026-10-22_5090_crosscheck_heldout` | Check transfer of raw K32/H32 and SAC800k evaluation to new server. | Raw: planner 15/50, SAC 50/50. This is another look at the same resets, not new test evidence. |
| `2026-10-22_step9_heldout_001` | Adaptive step versus same-machine fixed-step rerun (3 seeds each), plus SAC. Recorded decision keeps adaptive step for further runs. | Raw 7 policy evaluations; adaptive completion 0,1,0 versus fixed 0,0,0. Decision commit `5a4a77d`; this does not establish robust superiority. |
| `2026-10-22_step10_heldout_001` | 75 versus 25 iterations, GPU versus prior CPU arms, adaptive/fixed alternatives. Keep 75-iteration method and request equal-duration DAgger. Video reset 70000. | Raw 13 policy evaluations; iteration74 completion 34,35,12 versus iteration24 2,0,0. Decision `08a02d5`. |
| `2026-10-22_step11_heldout_001` | DAgger 75 versus 25 iterations and MPPI-GPS; use outcome to support method over DAgger. | Raw 10 policy evaluations; DAgger 0/150 at both durations. Decision `64198b2`. This does not independently isolate the tail, proposal or label rule. |
| `2026-10-22_step12_heldout_001` | Add training seeds 3–5 after first three results; repeat earlier arms; add SAC150k/175k/200k post hoc; keep method, investigate take-off. | Raw 16 policy evaluations; six-seed completion 204/300. Extra-seed decision `909da1d`; post-hoc SAC addition `4bf7ec1`; final interpretation `2442cc0`. |
| `2026-10-23_step13ab_k32_h48_001` | Compare last iterate, action average of iterations65–74, exploratory weight average, and H48/K32 planner. Keep last-iterate base after averages fail to improve pooled outcomes. | Raw 19 policy evaluations; last204/300, action169/300, weight173/300; planner39/50. Decision `6f1f88c`. |
| `2026-10-23_step13b_k256_h48_001` | Increase planner horizon/sample budget; revise interpretation of whether learned policy outperforms capable MPC. | Raw 7 policy evaluations; stronger planner44/50 versus policy204/300. Same recorded decision. |
| Step 14 planned rule | Keep predictive label check only if completion exceeds the 204/300 base and ≥4/6 training seeds improve; select downstream base using these resets. | Explicit planned decision in tracked lines1784–1801, commit `3bf1c1f`. No completed Step-14 held-out artifacts were present at snapshot time. |
| Step 17 planned rule | Choose check/no-check base from Step14, then retain sigma schedule if speed error improves without reducing completion on the same resets. | Explicit plan, tracked lines1943–1982, commit `f0b59b3`. No completed run observed. The schedule's k0=25 is motivated by *training* take-off timing, not directly by these reset outcomes. |

Step13c label/noise diagnostics use training-policy states and are not counted as additional 70000–70049 evaluations. LQR oracle results likewise are not counted. Separate hardware/numerical checks do not make the repeatedly inspected reset set fresh again. Preregistering each next experiment before running it does not undo adaptive reuse of the same test set.

## Seeds and splits across runners

All eight nonempty modules in `scripts/runners` were inspected, plus both training runners, all three tuning runners, and seed references in every benchmark/diagnostic module. `evaluation/seed_source_hits.txt` is the source-line inventory; `training_seed_ledger.json` reconstructs reset domains and executed-control totals for all 25 observed MDGPS/DAgger training configurations. Individual config paths and recorded iteration counts are retained, including incomplete runs.

| Entry point | RNG / split behavior | Audit conclusion |
|---|---|---|
| `run_walker2d.py` | Reset `seed+episode`; planner `planner_seed+episode`, default planner10000; records both and traces; independent planner Generator. | Reproducible domains within caller-provided range; no global final-reservation enforcement. |
| `eval_policy_walker2d.py` | Default 50000–50029 classification; deterministic actor; raw reset IDs and task/source/checkpoint hashes in CLI output. | Classification intentionally selects earliest weak/intermediate/strong checkpoints, so these are development resets. |
| `run_walker_shadow.py` | Replay exact reset/planner seeds from a selected reference episode; recovered state and solver warmstart. | Selected diagnostic cases, not random independent evaluation. |
| `run_ant_maze.py` | Global NumPy reset `seed+episode`, then planner consumes same global stream because no private planner seed. | Repeatable with identical call order; environment/reset and planner randomness are coupled. |
| `run_humanoid.py` | Global NumPy seed0 fixed; planner uses global stream. | One fixed demo seed, no independent-seed claim. |
| `run_acrobot.py` | No seed argument; global reset and planner stream. | Demonstration only; process RNG state is not recorded. |
| `run_cheetah.py` | No seed argument; global reset and planner stream. | Demonstration only. |
| `run_point_mass.py` | No seed argument; same global RNG across five episodes. | Demonstration only. |
| `train_walker2d_sac.py` | Torch/action RNG `seed`; reset RNG `seed+1`; replay RNG `seed+2`; training reset draw in0–39999; monitor60000+; classification50000–50029. | Domains disjoint in recorded run. Sampling training reset seeds with replacement is intentional; same reset may occur multiple times. Only one substantive SAC training seed supports current comparisons. |
| `train_walker_mdgps.py` | Reset `100000+10000*training_seed+episodes_per_iter*iteration+episode`; planner reset+500000; execution reset+900000; dev81000–81009; private fit RNG and Torch seeded. | Recorded75×10 runs occupy100000–150749 across seeds0–5 without final/dev overlap. Formula needed a block-overflow guard for larger settings (parent audit). Resume behavior is audited separately by parent. |
| `tune_acrobot.py` | Search0–4; old validation0–9, seeded TPE sampler. | Fixed to disjoint100–109 and reject overlap. Validation still selects among top trials and therefore is model-selection data, not a final test set. |
| `tune_cheetah.py` | Reuses global seeds0,1 across trials; Optuna sampler not explicitly seeded; no separate validation. | Search performance only; no independent final generalization evidence. |
| `tune_humanoid.py` | Trial repeats global seeds0..n−1 for each configuration; no final set. | Configuration sweep only. |
| `walker_headroom.py` | Dev resets80000+ and planner90000+; chooses K/H cell based on completion nearest40% (within20–60%). | Deliberately chosen weak-planner operating point. The H32 result cannot support an unrestricted superiority-over-MPC claim. |
| `walker_step5_compare.py` | Same70000+ for every policy/checkpoint, planner95000+; accepts arbitrary CLI ranges. | Repeated development comparison; arm pairing is list-order pairing, not verified training-seed matching. Step12's new seeds3–5 versus DAgger0–2 cannot be called matched-training-seed pairs. |
| Other diagnostics/benchmarks | LQR seeds0,1; GPU check collected training-policy states, planner seeds10000+i; Warp parity80000+; counterfactual/shadow tests branch from selected existing states and use explicit future noise seeds. | Mechanistic/debug evidence; reuse and selection are acceptable only under that interpretation. Full source-line inventory is retained. |

The label-fitting replay pool has no independent supervised validation split. Its fit loss is an optimization diagnostic; it cannot establish generalization, especially because consecutive states and adjacent replay iterations are correlated. Evaluation is separate at the reset level, but the reused70000 set has become model-selection data.

## Walker versus official Gymnasium

Official reference is the installed `gymnasium.envs.mujoco.walker2d_v5.Walker2dEnv` from Gymnasium1.2.3, MuJoCo3.6.0. Source and XML hashes appear in `evaluation/walker_differential.json`. The actual default is `walker2d_v5.xml`, **not** `walker2d.xml`. The latter matches the local XML except its final newline, but comparing it would miss the v5 friction correction.

Three action sequences were independently replayed from identical initial qpos/qvel/time/solver warmstart. Stock-v5 divergence begins at controls4,9,12; maximum observation errors before either termination are1.333,7.743,2.400. Termination differs on the second sequence. When official Gymnasium is explicitly instantiated with the local XML, all observations/qpos/qvel match exactly (maximum0.0) through termination on all three sequences (19,34,30 controls). This separates the model-file change from a stepping implementation error.

Other differences:

- Reset noise has the same uniform ±0.005 support, but legacy `RandomState` versus Gymnasium `Generator` yields different states for the same integer seed. Paired tests therefore copy the physical state, not merely the seed.
- Gymnasium forward reward uses interval displacement divided by0.008s; local `cost_style="gymnasium"` uses terminal qvel[0]. Even with identical dynamics and local terminal penalty disabled, reward discrepancies reach0.411 in these trajectories. That mode is an endpoint-velocity surrogate, not an exact Gymnasium reward implementation.
- The frozen paper task uses `2|1.5−vx| + 0.5vz² + 0.1angle² + .001||a||² + 5·unhealthy`, plus20 on the falling transition. The test recomputes this directly from official-reference states and actions. `WalkerRL` additionally adds the analytic training-reward failure tail, −0.995·5/(1−0.995)=−995. Standard Walker2d-v5 has neither that task nor that tail.
- Both use0.002s physics timestep, frame skip4,17-dimensional observations with velocity clipping±10, and the same strict healthy height/angle ranges. Custom evaluation calls1000 controls “eight-second completion”; it is not stock benchmark return.

`tests/test_walker_gymnasium_differential.py`: five passing tests and one **strict expected failure** for stock-v5 equivalence, on both the local machine and robopil5090 CPU environment. The expected failure is a documented incompatibility, not a bug fix or a claim of parity. `walker_differential_tests.txt` and `server_walker_differential_tests.txt` retain the results. The first temporary-server invocation had an asset-path setup failure, corrected by resolving the actual imported local asset; its log is retained separately. No task, asset, reset distribution or recorded result was changed.

## Fresh final reset reservation and statistical design

Reserve **5000000–5000199 inclusive**,200 resets, solely for final claims. No environment was reset to any member during this audit. **This reservation is not a finding that200 resets suffice for population-level power.** Reservation metadata is `evaluation/fresh_reset_reservation.json`: no collisions in420 current source/config/raw-JSON files,485 local NPZ files' seed arrays, reachable source/config git patches, or all observed training schedules. Scan errors:0. The absence claim is limited to those inspected records; unavailable robopil artifacts and undocumented external actions remain unknown. A population-level design must be resolved before this panel is opened; the current reservation stays untouched pending that decision.

Before accessing this set, freeze source and task hashes, algorithms/hyperparameters, training and planner budgets, all final checkpoint hashes, metrics, comparison family and analysis. If800 training seeds are used, choose `training_seed=1000+100*i` for `i=0..799`: reset blocks start10100000 and end809100749 under the current75×10 schedule, separate from the final range and below2³². The1,000,000 spacing between blocks also prevents cross-run planner/execution RNG collisions: contiguous training seeds40 apart would otherwise share a private Generator seed because planner and execution offsets differ by400000. Recorded training seeds0–5 do not have that overlap. Evaluate all prespecified arms exactly once on the same200 resets. Preserve an append-only per-episode manifest; a failed job may resume only missing episodes, retaining completed ones. Do not choose variants, checkpoints or paper scope after seeing final outcomes. An unsuccessful final comparison is reported, not repaired by reusing these resets.

Primary metric should be completion, with speed error and time alive reported together because survival can be achieved by standing still. Per-episode speed error averages only executed steps, so it changes with censoring/falls and must not be interpreted alone. For the claim of comparable performance with cheaper inference, the primary completion contrast should be **noninferiority to the stronger frozen K256/H48 planner**, with a prespecified5-percentage-point margin, accompanied by a separately measured inference-cost advantage. A favorable comparison against the deliberately weak H32 reference does not establish this claim. Keep learned-baseline contrasts exploratory until they are separately powered.

Power calculations in `evaluation/power_analysis.json` use the **training-seed difference** as the independent observation: `D_i = mean_reset(score(candidate_i) − score(reference_i))`. These calculations are **conditional on the fixed common reset panel and its prescribed planner randomness**. Their estimand is the expected training-seed difference on that particular panel, not the expectation over the deployment reset population. For noninferiority, test H0:E[D|panel]≤−.05 with one-sided alpha.025. If a *new, frozen method* is expected to match the planner on that panel (true conditional difference0), the planning noncentrality is `sqrt(n)*.05/sigma_D`. This yields80% conditional power at258 seeds using the optimistic pilot SD.2851,401 with25% SD inflation, and787 using the bounded fixed-panel SD.5.

**800 independent training seeds gives approximate conditional power.8065 for that fixed-panel noninferiority contrast;800×200 is not a population-power guarantee.** This is a rounded conditional count, not an approved design for a claim over unseen resets. Current pilot completion.68 versus planner.88 has a gap estimate−.20, far outside the−.05 margin: collecting more seeds cannot convert this existing method into a noninferior one. Improve the method on development data or narrow the claim; do not substitute a weak comparator merely to obtain a feasible power calculation.

For a population claim, let `D_ij` be the paired outcome difference for training seed i and reset j. Under an independent crossed sampling design, a centered random-effects decomposition gives

`Var(mean(D)) = sigma_train²/n + sigma_reset²/m + sigma_interaction²/(n*m)`.

The interaction component includes independent cell noise. Any additional shared planner randomness must be included as a modeled factor or in the sampled reset-case definition. Holding the reference policy/planner fixed removes its training variation, **not** its reset-population variation. The `sigma_reset²/m` term does not vanish as more policies are trained. For illustration, with reset-component SD.30 and m=200, even infinitely many training seeds give only about.654 normal-approximation power for a.05 margin at true difference0 and one-sided alpha.025. This is a hypothetical sensitivity example, not an estimate from the reused data. Achieving80% would require total variance≤.0003185; with m=200, reset-component SD alone must be below about.2524 even if every other variance term vanishes.

**No population-level training/reset count is established yet.** After algorithm selection, obtain an independent paired development pilot crossed over training seeds and reset cases, estimate all variance components with uncertainty, and jointly choose n and m for the prespecified test before accessing any final resets. The reused70000–70049 data are not that independent pilot. If the required reset count exceeds200, expand and collision-check the unopened reservation before final evaluation; do not add cases after inspecting final outcomes. A crossed bootstrap at analysis time handles uncertainty but does not retroactively supply missing power.

For sensitivity only, the following table gives80% power for an absolute **+15pp superiority effect**, applicable to a lower-performing comparator such as H32 at.30. It is not the primary plan: +15pp is impossible against a comparator at.88 under the current pilot ceiling.

| Variance assumption | One primary contrast, alpha0.05 | Three confirmatory completion contrasts, alpha0.05/3 |
|---|---:|---:|
| Fixed reference; observed SD0.2851 |31 training seeds |41 |
| Fixed reference; pilot SD inflated25% |47 |62 |
| Fixed reference; bounded worst-case SD0.5 |90 |120 |
| Two learned methods; bounded worst-case SD of paired difference1.0 |351 |469 |

These are fixed-panel conditional normal/t planning calculations, not distribution-free or population-power guarantees. A learned-vs-learned superiority claim (e.g. trust versus fixed ESS) needs a separate independent development pilot of **paired training-seed differences and crossed reset effects**. The bounded-difference counts351/469 apply only to the corresponding fixed-panel single/three+15pp contrasts; they do not remove the reset variance floor. The single-arm pilot SD cannot be substituted for a learned-pair SD. Three claims are not automatically three completion tests; deployment-cost measurements and mathematical claims are different hypotheses. Freeze the actual statistical family and adjust alpha before the final reset set is accessed. No final set was evaluated during this audit.

The pilot is small and selected: its six completion rates are[.68,.70,.24,.50,.98,.98], SD.2851. Its naive normal/chi-square upper-SD estimate.5956 even exceeds the known.5 bound for a bounded fixed-reference outcome, illustrating uncertainty rather than supporting a precise sample-size estimate. Runtime/budget feasibility may require narrowing the claim instead of pretending six seeds provide high power.

Final uncertainty should include training randomness and common-reset sampling: resample training seeds and shared reset columns, preserving all paired arms, or use an explicitly specified hierarchical model. For context only, reanalyzing the existing development result gives a seed-level t interval[.381,.979] and a crossed bootstrap interval[.453,.887], versus the narrow pooled-episode interval cited for204/300. These are descriptive sensitivity analyses after adaptive reuse, not final significance tests (`clustered_uncertainty.json`).

## Claims that survive this sub-audit

The raw artifacts support a **development observation on the custom task**: the six saved75-iteration policies complete204/300 episode evaluations on50 reused resets; a frozen K32/H32 planner completes15/50; K256/H48 completes44/50; the single SAC800k policy completes50/50. The learned policy avoids a planner call at deployment. Generalization superiority, stock Walker2d-v5 performance, superiority over adequately budgeted MPC, and seed-robust SAC/PPO superiority are not established by this evidence. Fresh evaluation must wait until the broader code/method audit is resolved and the final methods are frozen.
