# Experiment log

This log starts with no new policy-learning experiments. Earlier experiments remain in the [archive](artifact_layout.md#archive); they are not carried forward as fresh results or restored checkpoints.

## Frozen starting-point provenance

The cleanup starts from repository commit `176eb0a2b190f5c6ccae54d076385b0075e795a1` (`adding walker2d stats`). The active checkout retains standard CPU MPPI, environment runners, CPU tuning, and simulator tests. Use the actual checked-out commit as the baseline for each new experiment.

Walker controller/task behavior and their JSON files are retained. The frozen inputs have these SHA-256 fingerprints:

| Input | SHA-256 |
|---|---|
| `configs/walker2d_best.json` | `16435775c846c8037cc17be218cdf1255fa9abb971174ff04f4df363449a3ad6` |
| `configs/walker2d_task.json` | `546f6902405daaddb37fb9c458d77fb5159234d1218ec73abf9a625ec9f8f27a` |
| `assets/walker2d.xml` | `4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590` |

The historical Walker qualification completed **19/20** nominal eight-second episodes, with mean speed **1.461 m/s** and episode-averaged speed MAE **0.148 m/s**. It used resets 30–39 paired with planner seeds 10030–10039 and 20030–20039, 1000 controls per episode, and the protocol recorded as chosen on 2026-09-05. The original [protocol](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/runs/walker2d_teacher/initial2_protocol.md) and [raw-trace-checked summary](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/runs/walker2d_teacher/initial2_qualification_summary.json) are the authoritative provenance record, including original source/config hashes, package versions, and artifact hashes.

That result belongs to the recorded historical source. Cleanup edits can change source hashes even when controller behavior is preserved. Fresh interface tests and short seeded smoke/parity checks do not constitute another twenty-episode qualification. No qualified controller-performance claim is made for the other retained environments. Detailed historical interpretation is in [Walker teacher notes](walker_teacher.md).

## New experiment protocol

1. Work on `main`, one question at a time. Record the exact starting commit before changing code or configs.
2. Allocate a unique ignored directory such as `runs/<date>_<question>_<run-id>/`. Keep configs, reports, traces, diagnostics, and checkpoints together.
3. Write the question, comparison/baseline, environment and task objective, config paths and hashes, source fingerprints, dependency versions, and reset/planner seeds before running. Separate development from untouched qualification seeds.
4. Fix the acceptance criteria and primary metrics before inspecting results. Change one factor at a time when the comparison requires attribution.
5. Record complete, failed, interrupted, and rejected runs. Preserve every report and raw trace used to support a decision. State whether a check concerns simulator correctness, interface compatibility, teacher control quality, or standalone policy performance.
6. Update status, results, interpretation, and the keep/reject/revise decision. Link artifact paths and code/config revisions. A partial run or a proposed method is not a validated result.
7. Keep new behavior off by default so frozen controllers stay unchanged, and commit only after the relevant regression checks pass. Record the commit that produced each result. Revert rejected code in a new commit and keep its record here.

## Experiments

### S1: Walker teacher instrumentation and exact replay

```text
Experiment ID / date: S1 / 2026-10-11
Question: Can the frozen Walker teacher's decisions and its recorded qualification failure be reproduced exactly, mid-episode included, and what did the planner predict before that fall?
Branch: exp/walker-instrumentation (fast-forwarded into main as f271ece; branch deleted)
Baseline commit: 02f4c2bbf2ae56d09d5ddbe58d05f86fc96dcbea
Method and comparison: Instrumentation only; no controller, task, or simulation change. MPPI gains last_plan (pre-shift plan), snapshot() and restore(); MuJoCoEnv gains set_warmstart(); Walker traces add pre_plan_nominals, plans, planner_rng_states and initial_plan_flags. Compared against main (parity) and against archived qualification traces (replay). The fall autopsy is descriptive.
Task/objective: configs/walker2d_task.json (frozen target-velocity task).
Code revision and source fingerprints (SHA-256, as committed):
  src/mppi/mppi.py                          f404aae1490ebb4f318fd4494b9d9727220321cb97e5ebf8298f0270ea3c2503
  src/envs/mujoco_env.py                    7e98983197a4ffe853923c60c698656317ee4e540f50428e5ebc07716936722a
  src/envs/walker2d.py (unchanged; equals the qualification source)
                                            477c3d2d8735171825392c9fe011941bc26e386f2113b4dd54df8a8c1e03ebca
  scripts/runners/run_walker2d.py           5d3e2e6a5d4d79b07f30cb27325e57cb82b7732553a7b0132b3096f36715db85
  src/utils/trace_replay.py                 469594a7255e5df789614949efcbe14b16bfc40030fbb77d2253a94f952c819d
  src/utils/rollout_data.py                 c6f698644a33a1177aaff87436b412803cd3434b12f1c6f04558ca6459ae03f5
  src/utils/stats.py                        b7f98f63f2964efe0f4765077e6fd29a8a063faddde555903900ba507abba431
  scripts/diagnostics/replay_walker.py      d065b6d91bfea147ebccd31a011ba8c0ec5c45fca4a9e041dc4ea9e786bc48b0
  scripts/diagnostics/walker_fall_autopsy.py 0c04810cc08af5d6aceb5061e9c32ae832aebd9fa4a735cf938e5409c846d336
Revision note: Parity and replay ran from the uncommitted tree before three later edits. None of the edits changes simulation, planning or trace arrays.
  - run_walker2d.py: the trace-layout "replay" text, a metrics-JSON string, was corrected for multi-episode indexing. The replay metrics.json files still carry the earlier wording.
  - replay_walker.py: now exits 1 on a mismatch after writing its report.
  - walker_fall_autopsy.py: no longer attributes short-horizon causes, and renames two summary fields. It was re-run; rows.jsonl is byte-identical to the first run, and the earlier summary is kept as superseded_summary_before_label_fix.json.
Configs and SHA-256 fingerprints:
  configs/walker2d_best.json  16435775c846c8037cc17be218cdf1255fa9abb971174ff04f4df363449a3ad6
  configs/walker2d_task.json  546f6902405daaddb37fb9c458d77fb5159234d1218ec73abf9a625ec9f8f27a
  assets/walker2d.xml         4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590
Environment/package versions: Python 3.11.15, mujoco 3.6.0, numpy 2.4.3, scipy 1.17.1 (same as the archived qualification).
Development seeds / qualification seeds: Parity used resets 0-1 with planner seeds 10000-10001 (100 steps). Replay used reset 31 / planner 20031 and reset 30 / planner 20030 from the archived initial2 qualification (already-used seeds; replaying them is not a new qualification). Autopsy probes use seeds 900000 + step.
Criteria and primary metrics chosen before running: (1) existing trace fields byte-identical to main for the same seeds; (2) archived qualification episodes reproduced exactly (states, warm-starts, actions, costs, contacts, forces); (3) restoring a recorded decision reproduces its continuation exactly; (4) autopsy descriptive, classification allowed to be mixed or unresolved.
Run directory / artifact links: runs/2026-10-11_walker_instrumentation/ (ignored by Git; backed up as described below). Every file's SHA-256 is in runs/2026-10-11_walker_instrumentation/MANIFEST.sha256 (manifest SHA-256 e240aca4395bbc02dcbc0ae9b84543eecec1b9aacfbf01276545c2d27e1024bb). Key files:
  parity_main_vs_branch/parity_report.json   6d7b5f218b4c565bf47c1eb6a05c8a200e72de7a1589aac0ec1dc1438e8ef4cb
  parity_main_vs_branch/main_traces.npz      37b0e8a1546e657807a5e47876fedbcd38da673aae3274850f06e95ad7bcde99
  parity_main_vs_branch/branch_traces.npz    a1dd5888097eac97704f7dda386c191ab256c95f66dea0ca8b530d45a8b379b4
  replay_reset31_planner20031/replay_report.json 8802c6928e045a3130ea7ce415f9012dfa90a1775f3c4bd840d7fa56488e98f2
  replay_reset31_planner20031/traces.npz         a64faf0576a5df7bfaad3468ddebaf6e7dd9f8a5120bf069c68a204a6d5267a3
  replay_reset30_planner20030/replay_report.json 33a3db8946553cd3d39223016970b7947286542e3cfeb2e6efc48c417634be7d
  replay_reset30_planner20030/traces.npz         d324ad1f87f625584a94adf6cd278fac226e9cdd4f779bd365b154c20ccbf0b6
  fall_autopsy_reset31/rows.jsonl                5222196e56958d965466820f40158876718c4155a028ecd32a3a2075634fc7c0
  fall_autopsy_reset31/summary.json              dab7bb5dc36cf85a9a56b7b88bf23f387e488e7b7189aa2de0e87c5a143375f7
  Reference trace (read only): archive runs/walker2d_teacher/initial2_qualification_b.npz
                                                 7c153af134f555267f3b58aad4bf3398b20a1b0e4fde469475945b776a26b09f
Status: complete
Results:
  (1) Parity: all ten existing trace fields are identical to main, which was run from an exported copy of 02f4c2b. Four fields were added.
  (2) Replay: reset 31 / planner 20031 reproduces the archived fall at 542/542 steps exactly. Reset 30 / planner 20030 reproduces 1000/1000 steps exactly.
  (3) Restore: decisions 0/271/522 (reset 31) and 0/500/980 (reset 30) replay 20 steps exactly. Unit tests cover arbitrary restore points, and a tampered reference makes the checker exit 1.
  (4) Autopsy of the step-542 fall (falling action index 541; window 461-541): classification "mixed".
      - The executed plan first predicts a fall within H=48 at step 489, 52 decisions (0.42 s) before the falling action. This is a fall within H, not necessarily the later executed fall.
      - Averaging: at steps 489, 494 and 498-500 the plan predicts a fall within H while the best sample does not. n_eff is about 1-7 over steps 474-499.
      - Objective: at step 501 the lowest-cost sample predicts a fall although surviving samples exist.
      - From step 502 onward, neither tested sampler (K=256, or the K=1024 probe) found a trajectory surviving H=48. That does not establish that the fall was unavoidable, nor why.
      - Control: during ordinary walking (reset 30, every 50th decision), 44% of default H=48 samples predict a fall within H. 98% of H=96 probe samples predict a fall, and 67% of H=96 probe plans do. The H=96 probe therefore does not discriminate and is not used as evidence.
Decision: keep (instrumentation); committed as the S1 checkpoint f271ece, now on main.
Interpretation and limitations: One fall, heuristic evidence. "Survives" means within the probe horizon only. Whether the surviving 48-step options at steps 489-501 would have avoided the fall beyond the horizon is untested; a paired counterfactual branch from those decisions (S3 machinery) would test it. Short-horizon causes are not attributed.
Evidence backup: /Users/jaisel/Documents/workbench/mppi-gps-evidence/2026-10-11_walker_instrumentation.tar.gz
  (SHA-256 b6a3510f313a68e3610a750345eab424522d4a3d4d7a03167e2ed254f76cec7d); all 18 files verified against the manifest after extraction.
```

### S2a: task-matched RL interface and learning plumbing

- Starting commit: `eb32e383b5370b8d710049a11caf4baee04fc28b`, on `main`.
- Question: does SAC train against the same local physics, observations and task
  costs, with correct absorbing-failure and time-limit accounting?
- Protocol fixed before running: [S2 ladder](rl_ladder.md). No MPPI/source/task/XML
  changes, no gate, no observation normalization or reward-scaling experiment.
- Reuse: audited archived `scripts/ddpg_mppi_gps/train_walker2d_gym_sac_baseline.py`
  learning architecture, target and optimizer schedule; no archived policy or Gym
  environment restored. Gamma intentionally changes from 0.99 to 0.995. Native
  local Walker costs plus the analytic absorbing tail replace the old Gym reward.
  Archived source SHA-256:
  `78033ddf60ea5e35c3867e40c3686e88d3771c1aceb2e177f353034e17ab51a3`.
- Acceptance: exact seeded local simulator/observation/action/warmstart parity;
  analytic tail and explicit absorbing return agree; fall penalty once; a fall
  on the final allowed step remains terminal; timeout uses final pre-reset
  observation in the actual learner target; deterministic evaluation preserves
  training RNG; all regression tests pass. A short fresh run must produce finite
  updates and safely loadable policy checkpoints before full training.
- Planned smoke artifacts: `runs/2026-10-11_s2_sac_smoke_001/`, seed 0,
  1,000 controls, 100 warm-up controls, default 256-wide learner/batch, checkpoints
  every 500, two 1,000-step monitoring episodes per checkpoint. This is a plumbing
  test; the shortened warm-up is NOT a controller-quality experiment.
- Status: passed. **180 tests pass** (100 S1/existing + 80 new); frozen Walker
  source, XML and both task/controller configs retain the S1 hashes. No edits to
  the frozen MPPI controller or runner. Full source/args/dependency fingerprints
  are preserved with each run, including exact source copies.
- Smoke `_001` completed with 745 finite updates and checkpoints at 0/500/1000.
  `_002` repeated the same computation after adding raw evaluation traces and
  fuller provenance; episode results match `_001` (timings excluded). Both runs
  remain preserved. Latest `_002` includes NPZ traces for all monitoring reports.
- Automatic ladder classification was also exercised on `_002`: all three
  checkpoints are weak (0/30 completions each). This verifies the path, **not**
  learning quality; the 1,000-step smoke has shortened exploration warm-up and
  is not the training ladder. Reports are under `_002/classification/`.

### S2b: first task-matched SAC checkpoint ladder

- Starting commit: same as S2a; additive source hashes identify uncommitted work.
- Question: what weak/intermediate/strong policies does the fixed learner
  actually produce? See the predeclared development categories in
  [the protocol](rl_ladder.md).
- Artifacts: `runs/2026-10-11_s2_sac_ladder_seed0_001/`.
- Fixed: 1M controls, 5k random warm-up, gamma .995, archived SAC hyperparameters,
  seed 0, CPU/one Torch thread; save step 0 and every 25k. Monitoring resets
  60000–60004; classification resets 50000–50029, disjoint from training resets
  0–39999. One training seed is developmental only.
- Status: **training and classification complete; checkpoint review passed**.
  1,000,000 controls, 995,000 learner updates, 41 saved/evaluated checkpoints;
  total training + classification wall time 2,268.29 s (37.80 minutes). Command:
  `.venv/bin/python -m scripts.training.train_walker2d_sac --run-dir runs/2026-10-11_s2_sac_ladder_seed0_001 --classify-at-end`.
  Final state is in that directory's `status.json`; evaluations and raw traces
  were saved every 25k. Every checkpoint was evaluated on the 30 classification
  seeds and indexed in `classification/ladder.json`. No active/
  shadow gate or student experiment may be inferred from this entry.
- Early monitoring (five development seeds, **not qualification**): step 0
  completed 0/5, mean speed MAE 2.1473 m/s; step 25k completed 0/5, MAE
  2.0087 m/s; step 50k completed 3/5, MAE 1.3220 m/s. Those early policies were
  not strong walkers. These interim results are retained, superseded by the
  full classification below. Learning settings remained fixed through 1M.
- Preview: `preview_025k/walker.mp4` (640x480, 31.25 fps, tracking camera), with
  report and raw trace for reset 60000; it reproduces the 170-step monitoring
  failure. First rendering attempt was blocked by sandbox CoreGraphics before
  producing outputs; rerunning with approved graphics access succeeded. Video
  timing/encoding and extracted frames were checked. This preview is not a new
  qualification episode or a change to the learner.
- Full checkpoint review artifacts: `review_001/review.json` (machine-readable
  audit), `review_001/review_ladder.py` (reproducer),
  `review_001/frozen_selection.json` (selected original paths/hashes), and
  `review_001/learning_curve.png`. All 41 checkpoint hashes, all classification
  trace hashes, source fingerprints, and 1,230 classification episodes were
  checked against saved evidence. Completion and speed-error statistics match
  raw post-action state traces; costs/rewards and state/action offsets agree.
- Earliest checkpoint in each **predeclared development category**:

  | Category | Training step | Full episodes | Mean vx | Speed MAE | Interpretation |
  |---|---:|---:|---:|---:|---|
  | Weak | 0 | 0/30 | -0.635 m/s | 2.135 m/s | Untrained; falls after 23–24 controls (0.184–0.192 s). |
  | Intermediate | 200,000 | 29/30 | -0.178 m/s | 1.679 m/s | Retreats and settles almost stationary, not intermediate walking quality. |
  | Strong | 800,000 | 30/30 | 1.465 m/s | 0.19755 m/s | Sustained forward travel near the 1.5 m/s target. |
  | Final, comparison only | 1,000,000 | 30/30 | 1.314 m/s | 0.23449 m/s | Worse tracking than 800k; does not replace the selected strong policy. |

- Selection is unchanged: weak SHA-256
  `8b9ff1ec731741cea861d06d87dfde64e99ffd0ff75923869029dcd2b8a72d67`;
  intermediate `bacb711efa57969c9431dc66349e5d44c7cda8fe2eea39fb307980694869c3a8`;
  strong `bb62968017dfef46d2aa85545286eeefb0d90587d32bfd888d8ab9de4e077773`.
- Across the 41 checkpoints: 13 weak, 27 intermediate, one strong. Learning is
  non-monotonic. The 800k mean speed MAE is only 0.00245 m/s below the 0.20
  development threshold; it is not a broad robustness or safety result.
  Intermediate's single failure is reset 50021 at control 166. Its 29 survivors
  average about 0.00049 m/s after 2 s: surviving without doing the task is a
  useful separate candidate behavior for the gate test.
- Review videos: `review_001/{weak,intermediate,strong,final}/walker.mp4`, all
  fixed reset 50000 with the tracking camera. Rendered replays match the original
  classification episode **exactly** for states, solver warmstarts, actions,
  contacts, task costs and adjusted training rewards. These are repeat renders,
  not extra qualification trials. Sampled frames and video encoding/timing were
  inspected. The strong policy's raw contact onsets are asymmetric; these
  geometric events are not sustained force-bearing steps or a natural-gait
  certificate, and saved traces do not contain force-based support diagnostics.
- Decision: retain the original 0/200k/800k selection for **S3 shadow-only**
  development. First verify identical executed MPPI actions/states/planner RNG,
  pre-action policy inputs, actual-plan scoring and invalid/fall rejection. Do
  not activate the gate or retune the teacher during that check.
- Limits: one training seed, deterministic policy, nominal small reset noise,
  8-second episodes, development-selection data only. No fresh paired MPPI
  comparison, no demonstrated non-inferiority, and no coupled GPS result yet.
  No additional training was run during review. Future training belongs on
  RoboPIL after explicit CUDA support and a separate timing/correctness check;
  the present source and policy evidence remain CPU results.
- Before S4: shadow action acceptance cannot set sample count. Use a separate
  active full-episode development pilot or conservative joint outcome assumptions,
  and assess probability of passing both fixed NI criteria. The previously
  proposed shadow-acceptance sizing is superseded; margins remain unchanged.

## 2026-10-11 — S3 shadow-only task-first gate

- Question: can the three frozen S2 policies be scored at actual MPPI states
  without changing the baseline trajectory, with pre-action alignment and
  explicit scoring of the actual optimized plan?
- Starting commit: `eb32e383b5370b8d710049a11caf4baee04fc28b`. Additive working
  tree changes; no frozen planner, environment, task/config, XML or S2 source
  changed. No new training, no active gate, no threshold sweep.
- Protocol declared before the runs: [S3 shadow gate](shadow_gate.md). Fixed
  tau=0; reject an added predicted fall. Invalid candidates reject; invalid
  baselines stop. All three policies observe the same raw-MPPI trajectory.
- Development inputs: existing S1 success reset30/planner20030 (1,000
  controls) and failure reset31/planner20031 (542 controls). The selected
  checkpoint hashes remain the 0/200k/800k hashes recorded in S2 above.
- Primary correctness criterion: all ten baseline trace fields, including
  planner snapshots, match the continuously replayed reference exactly.
  Secondary diagnostics: predicted costs/falls, rejection reasons, acceptance
  counts and action differences. Acceptance is not controller improvement.
- Outputs: `runs/2026-10-11_s3_shadow_001/{success,failure}/`, with source
  snapshots, manifests, candidate trajectories, decision records and parity
  reports. Status: shadow correctness and first four paired continuations
  complete; the broader S3 counterfactual study is not complete.
- Shadow results: continuous success (1,000 controls) and failure (542
  controls) both match all ten S1 baseline fields exactly. Gate accepts on
  success: weak 0/1,000, intermediate 17/1,000, strong 142/1,000; on failure:
  weak 0/542, intermediate 2/542, strong 40/542. No invalid-candidate or
  sequential/open-loop mismatch rejection occurred. Weak's every suggestion
  was rejected, but intermediate occasionally passes: the rule tests the
  current predicted trajectory, not the policy's development label.
- No policy suggestion passes in the preidentified failure window 489–501.
  This is not evidence that the fall was unavoidable, or that a recurrent
  gate could not have changed the trajectory earlier.
- Runtime: success 62.07 s (36.28 s inside MPPI), failure 33.53 s (19.74 s
  inside MPPI), local CPU. All three policy scorers combined add about 23–24
  ms per decision; these unoptimized audits are not a real-time controller.
- The success run's source snapshot precedes an input-archiving addition.
  Its exact original config/task/selection bytes were subsequently copied
  into `inputs/`, verified against the already recorded hashes, and the
  artifact manifest refreshed. Its executed source snapshot is unchanged.
- Before running continuations, the fixed earliest-acceptance selection rule
  yielded four pairs, recorded in `branch_selection.json`: success
  intermediate@96 and strong@10; failure intermediate@126 and strong@0.
  No weak or failure-window pair qualifies. Each branch gets 250 controls,
  including its one initial action; both then use raw MPPI.
- Correctness verification: **278 tests pass** (180 prior, 57 gate, 19 shadow
  runner, 22 counterfactual). Independent evidence audit checked all 4,626
  policy decision records, all ten trace fields byte-for-byte, artifact/input/
  checkpoint/source hashes, 44 independently rescored trajectories, and nine
  post-call replay restorations. Frozen controller/environment/RL sources,
  task/config and XML remain unchanged.
- All four selected 250-control paired continuations completed without a
  fall. Every raw branch exactly reproduces its recorded MPPI continuation.
  Positive changes below mean worse after the one policy action:

  | Source / checkpoint / decision | Predicted plan cost change | Actual 2-second task cost change | Speed MAE change |
  |---|---:|---:|---:|
  | Success / intermediate / 96 | -7.25595 | +23.78590 | +0.021630 m/s |
  | Success / strong / 10 | -12.74166 | +2.18861 | -0.006569 m/s |
  | Failure / intermediate / 126 | -1.61132 | +14.70375 | +0.022070 m/s |
  | Failure / strong / 0 | -2.45399 | +10.39104 | +0.013134 m/s |

- Reports/traces: `branches/{success_intermediate_96,success_strong_10,
  failure_intermediate_126,failure_strong_0}/`. Machine-readable aggregate:
  `branch_summary.json`, including source report hashes and first-48-control
  executed costs. All four accepted suggestions have worse 2-second task
  cost; three have worse speed MAE. These four deliberately selected
  development points are not a representative sample, confidence interval,
  or formal rejection of the episode-level non-inferiority hypothesis.
- SHA-256 anchors: success shadow report
  `36b6293e3c89b02ed0ed1d978bb5d1d4196daeca635c3cc839e5f9a503f94c2b`;
  failure shadow report
  `f91678ca856bec3b8150c11a834274ec3571979a1fae7be4c13c07c00a4ee82f`;
  branch summary
  `1e531e41203d400fd27a96d6f3da86ecf16527e1df3147a81b7e5c5b183f6803`.
- Mechanistic warning: the gate compares two proposed full 48-action plans,
  but the intervention executes one action and then replans with MPPI. Those
  are different continuations. At success@96, the retained open-loop plan
  predicts a fall, yet raw receding-horizon MPPI survives and costs only
  30.53 over the next 48 controls, versus its retained plan's predicted
  70.44. This illustrates the mismatch; it does not uniquely attribute the
  regressions to switching, horizon length, or a need for a learned value.
- The failure-source branches end before the original step-542 fall. No
  fall-prevention or recovery conclusion follows. Margins, objective,
  candidate generation and checkpoints have not been changed after seeing
  these results.
- Decision: keep the tested instrumentation and fixed gate as research
  machinery, **not a validated improved teacher**. Next is the plan's broader
  preselected accepted-state counterfactual study (about 50 per checkpoint
  where acceptances exist). No recurrent coupling, S4 qualification or
  distillation is implied; no additional training is needed for this step.

## 2026-10-12 — Post-hoc plan-versus-realized check and v2 protocol

- Audited S2/S3 v1 preserved as local commit `fdf58ed` after all 278 tests
  passed; nothing pushed. This supersedes the preceding entry's decision to
  expand v1 immediately. The broader v1 counterfactual study is paused.
- User's original post-hoc analysis is preserved at
  `runs/2026-10-11_s3_plan_vs_realized_check/{check.py,result.json}`, outside
  the audited S3 bundle. Reproduced from saved data with no new simulation:
  1,495 usable windows; realized MPPI cheaper in 86.0869565%; median
  realized/plan ratio 0.7229641; only 1/176 accepted strong-policy full
  trajectories cheaper than realized MPPI over the same horizon.
- Accounting clarification: 1,448 fully observed windows plus 47 filled
  with the prescribed absorbing failure cost, not 1,495 fully observed
  windows. Fully observed only: 86.1878453%, median ratio 0.7133887, still
  1/176. Non-falling incomplete windows at truncation are excluded. Windows
  overlap across only two development episodes; no independent-trial or
  causal policy-benefit claim follows.
- Interpretation: the scorer does not predict the continuation actually
  executed (one candidate action followed by MPPI replanning). V2 will
  evaluate matched continuations; cancellation of residual errors is a
  hypothesis, not a guarantee. Original four negative results are retained.
- Before new simulations: [fixed protocol](gate_v2.md) declares five proposal
  seeds (31000–31004) at each of the four original decisions, original
  post-call continuation snapshots/common random streams, 250 controls,
  exact same-action controls, and cost comparisons at 48 and 250 controls.
- V2 scoring is fixed at one immediate cost plus 47 optimized successor-plan
  actions, with unchanged H=48 optimization, objective, candidate first
  actions, tau=0, fall guard, checkpoints and failure accounting. All scratch
  controller states are discarded. No new training or active gate.
- Protocol committed before the simulations as `fb1a6c9`; its file SHA-256
  is `4e4061c381630515e21a346d996f032d398e8936546d70e18a5bba038865feb6`.
  Implementation and its tests committed as `281270c` before inspecting v2
  calibration scores. V2 scorer SHA-256:
  `11413dc5a6d84555929f9db0ee5b43de1cd03a697957afb4ff02dc06182f5f15`.
- Status: A–C complete. No weights, policies, horizon, threshold, planner
  settings or proposal seeds changed after outcomes. Frozen S1 runtime
  files, S2 learning implementation and prior evidence remain untouched.
- Tracked post-hoc reproducer:
  `scripts/diagnostics/compare_walker_plan_realized.py`; its independent fresh
  report is `runs/2026-10-12_s3_plan_vs_realized_reproduction_001/report.json`,
  SHA-256 `652bbd994424febabba5da49d4aab3da69738117999005fcc54821fb5a687505`.
  It reproduces every original statistic, separates observed/padded windows,
  verifies evidence hashes, and contains per-window rows. Original scratch
  script/report and audited S3 bundle were not modified.
- Seed controls: all four cases complete under
  `runs/2026-10-12_s3_action_controls_001/`. All four raw branches, identity
  controls and original proposal-RNG recoveries are exact. Twenty alternate
  first-action continuations retain the original post-call nominal and future
  random stream. Case artifacts include source copies, proposal actions and
  discarded proposal snapshots, input hashes and full branch traces.
- Seed-control cost deltas (candidate minus raw, positive means worse):

  | Case | Δ48 min / median / max | Δ250 min / median / max | Worse at 250 |
  |---|---|---|---:|
  | Success / intermediate / 96 | −7.181 / −1.857 / +6.399 | −11.308 / −2.128 / +18.579 | 2/5 |
  | Success / strong / 10 | −1.079 / +1.559 / +5.308 | −24.106 / −9.583 / +15.429 | 1/5 |
  | Failure / intermediate / 126 | +8.292 / +8.946 / +10.507 | +6.204 / +14.030 / +25.393 | 5/5 |
  | Failure / strong / 0 | −2.632 / +2.110 / +10.968 | +5.577 / +15.683 / +441.135 | 5/5 |

- One alternate action causes a fall: failure/strong/0, proposal seed 31001,
  control 177 (one-based, including initial action). Its Δ48 is only +0.400,
  but Δ250 is +441.135 including the frozen absorbing suffix. Across the
  20 seeds, 15 worsen cost at 48 controls and 13 at 250. Seed action L2
  min/median/max is 0.268/0.494/1.287; policy L2 is 1.683–2.301. These are not
  magnitude-matched perturbations and not a zero-centered statistical null.
- Aggregate seed report: `runs/2026-10-12_s3_action_controls_001/summary.json`,
  SHA-256 `e3069c3a458f7fe01c0e4477506624bc7c17e6f7aea4ff44e7c0d6eb3b8554cb`.
- V2 calibration: `runs/2026-10-12_s3_gate_v2_calibration_001/report.json`,
  SHA-256 `f4c8739ade8e15ef5a13d4e6bf961853ace6b125c441c45f4d403ee71d9da6a5`.
  Every one of the 56 H=48/K=256 baseline/candidate probes exactly matches
  its saved first transition and next full optimized MPPI plan/snapshot,
  including between the single-thread scratch scorer and original pool.

  | Policy case | V2 predicted Δ | Actual Δ48 | Actual Δ250 | Accepted |
  |---|---:|---:|---:|---|
  | Success / intermediate / 96 | +7.862 | +4.105 | +23.786 | No |
  | Success / strong / 10 | +34.220 | +4.469 | +2.189 | No |
  | Failure / intermediate / 126 | +0.588 | +4.607 | +14.704 | No |
  | Failure / strong / 0 | −11.534 | −12.384 | +10.391 | Yes |

- V2 gets 4/4 known policy cost directions right at 48 controls, 3/4 at 250.
  The one acceptance genuinely improves the matched horizon before becoming
  worse later. On the 20 seed controls, however, only 9/20 directions agree
  at either horizon. At 48 controls, eight accepted actions include seven
  false acceptances; four beneficial actions are rejected. Four identity
  comparisons produce exact cost ties and unchanged outcomes.
- Independent audits verified 258 file hashes across calibration/control/
  prior policy bundles, all 28 comparison rows recomputed from NPZ, and all
  production probe states/warmstarts/actions/full plans/RNGs. No accounting
  or implementation discrepancy was found. Full test suite: 413 tests pass
  (including a post-run output-path guard; see below).
- Post-run hardening adds only a guard against writing a controls output
  directory inside audited shadow evidence (including symlinks). The archived
  control-run sources predate that guard; scoring, proposal generation,
  continuation and metric code are unchanged. Original evidence stays intact.
- Decision after calibration, not a preregistered statistical rejection:
  preserve v2 as tested research machinery, but do not promote it to a
  reliable teacher. The broader study and active gate remain unstarted;
  matched continuation removes one asymmetry but does not reliably rank the
  seed interventions. These known-state calibration results cannot establish
  performance on fresh states or identify a unique remedy. No new training,
  S4 qualification, student distillation, commits to remotes or pushes occurred.

## 2026-10-12 — nominal handoff and future-randomness diagnostics (preregistered)

- Starting commit: `f6c6875`. Fixed protocol:
  [continuation diagnostics](continuation_diagnostics.md), SHA-256
  `204cc756b215919407b283be35f4baf740f6f73b5d5d03727fc303c5131e6dae`.
- Status: planned; no new continuation outcomes inspected at registration.
- A: twenty existing alternate MPPI actions, replacing only their original
  post-call nominal with their saved proposal's shifted nominal. Original
  future RNG/refinement and simulator state/warmstart stay unchanged. Four
  raw replay guards; primary same-action cost difference at 48 controls,
  persistence/falls at 250. Compare with hashed original-U continuations.
- B starts after A is recorded. It deliberately retains original-U execution
  regardless of A's result, diagnosing the existing v2 execution model rather
  than selecting a new convention post-hoc. Four states, eight future seeds
  (41000–41007), raw plus each policy action plus five seed actions per state:
  224 continuations. First actions are fixed; only future RNG changes.
- Both use frozen task/controller inputs and costs with absorbing fall
  accounting. No rescoring, new training, active gate, or non-inferiority
  claim. Reports retain all pairs and distinguish per-state results from
  pooled descriptive counts. No significance/acceptance threshold is added.
- Planned outputs: `runs/2026-10-12_s3_nominal_ablation_001/` and
  `runs/2026-10-12_s3_future_seed_check_001/`. Source/protocol snapshots,
  input/output hashes and branch traces are preserved in these new bundles.

Add one entry per question using the fields below:

```text
Experiment ID / date:
Question:
Baseline commit:
Result commit(s):
Method and comparison:
Task/objective:
Code revision and source fingerprints:
Configs and SHA-256 fingerprints:
Environment/package versions:
Development seeds / qualification seeds:
Criteria and primary metrics chosen before running:
Run directory / artifact links:
Status: planned | running | complete | partial | failed
Results:
Decision: keep | reject | revise
Interpretation and limitations:
```
