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

### A result — starting-plan handoff (recorded before B)

- Protocol commit `47ddab6`; implementation commit `aa82e49`. All 24 declared
  continuations complete under `runs/2026-10-12_s3_nominal_ablation_001/`.
  Full regression suite before execution: 479 tests pass. No protocol,
  controller, cost, proposal, horizon or scoring parameter changed.
- All four raw replays are exact across every saved trace field. All twenty
  nominal interventions preserve their archived first physical transition
  and common future RNG prefix. No proposal RNG was handed over.
- Independent audit before B verified 148 artifact hashes, all input/source
  copies, all four full raw replays and all twenty action/nominal/RNG pairs;
  costs, falls and differences were recomputed from NPZ without discrepancy.
- Same action, matched starting plan minus original starting plan (negative
  cost difference means the plan handoff helped):

  | Case | Improved / 5 at 48 | Median Δ48 | Improved / 5 at 250 | Median Δ250 |
  |---|---:|---:|---:|---:|
  | Success / intermediate / 96 | 3 | −3.308 | 4 | −11.145 |
  | Success / strong / 10 | 4 | −0.744 | 3 | −2.430 |
  | Failure / intermediate / 126 | 4 | −5.227 | 2 | +0.501 |
  | Failure / strong / 0 | 2 | +1.477 | 1 | +10.205 |

- Pooled descriptive counts: 13/20 improve at 48 controls (median −3.0105,
  range −12.3023 to +13.4072); 10/20 improve at 250 (median −0.3405,
  range −413.9837 to +172.5844). These twenty comparisons share four states.
- The matched-plan arms have no falls, preventing the archived fall for
  failure/strong/0, seed 31001. That arm's Δ250 improves by 413.9837 but
  remains 27.1518 worse than raw MPPI. Conversely success/strong/10,
  seed 31004, worsens by 172.5844 without falling within the 250 controls.
  Survival alone therefore does not explain all large cost differences.
- Against raw MPPI, matched-plan substitutions still worsen cost in 12/20
  cases at each horizon (original-plan substitutions: 15/20 and 13/20).
- Interpretation: planner initialization can materially affect a substituted
  action's continuation, including whether a fall occurs, but matching the
  proposal plan is not a consistent remedy here. This does not establish
  action-plan compatibility as the unique mechanism or justify policy-plan
  handoff. No gate rescoring or promotion follows.
- Decision: retain these results; proceed to B with the original starting
  plan exactly as preregistered, regardless of A's mixed outcome.
- Report SHA-256 fingerprints, case order as in the table:
  - `success_intermediate_96/report.json`:
    `9cb49e2a7d81379f34de25ff4a08edd11c7aada63ea0955f719acfca9ab5cc4e`
  - `success_strong_10/report.json`:
    `3cd45fade702c068c2a04fc50f01e518fb2b5be8a1315284cc1d6f6fbbd57d4a`
  - `failure_intermediate_126/report.json`:
    `762f0b922e202da2c44f4fb4105d572280e84c291d0110182c5d07b9a3e40a35`
  - `failure_strong_0/report.json`:
    `4a7ff489df905503fdca1bb9479b1af0863d69ecbaf07952f87d0084c61a8dd5`

### B result — paired future random streams

- Status: complete. Launched only after A was audited and its results
  committed as `eff21b4`. The protocol stayed at `47ddab6`, SHA-256
  `204cc756b215919407b283be35f4baf740f6f73b5d5d03727fc303c5131e6dae`;
  diagnostic implementation stayed at `aa82e49` throughout both experiments.
- Run: `runs/2026-10-12_s3_future_seed_check_001/`. All 224 continuations and
  192 paired comparisons complete, using the original post-call nominal
  in every arm and all eight predeclared future seeds. Local CPU runtime
  summed over the four cases was 2,409.00 seconds (40.15 minutes). No new
  policy training, gate activation, rescoring or parameter changes occurred.
- Independent audit verified the root manifest's exact 263-file coverage,
  all four case bundles and their original inputs, all 224 first transitions,
  all 192 common-RNG pairs, and both horizon costs, falls and summaries
  recomputed from NPZ. No provenance, execution or accounting discrepancy.
- Final regression: 479 tests pass. Frozen S1 planner, Walker, physics/cost
  helpers, runner, configs and XML remain byte-identical to `f271ece`.
- For each fixed policy action, cost differences are **policy first action
  then MPPI minus raw MPPI**, paired within each future seed. Positive is
  worse. The ± values below are sample standard deviations of eight paired
  differences, **not confidence intervals**. These are not standalone-policy
  evaluations. Success/failure labels identify the archived source episode.

  | Saved decision | Mean Δ48 ± SD | Lower cost / 8 | Mean Δ250 ± SD | Lower cost / 8 | Policy / raw falls at 250 |
  |---|---:|---:|---:|---:|---:|
  | Success / intermediate / 96 | +7.632 ± 5.816 | 1 | +11.958 ± 11.030 | 1 | 0 / 0 |
  | Success / strong / 10 | +3.231 ± 3.755 | 2 | +49.266 ± 124.256 | 2 | 1 / 0 |
  | Failure / intermediate / 126 | +1.722 ± 4.257 | 4 | −31.890 ± 94.486 | 6 | 0 / 1 |
  | Failure / strong / 0 | −2.954 ± 9.564 | 4 | −45.440 ± 125.158 | 6 | 0 / 1 |

- Holding the first action and nominal fixed, the effect changes sign across
  future streams for 22/24 interventions at 48 controls and 23/24 at 250.
  This includes all four policy actions at both horizons. There are no
  exact-zero paired differences; no post-hoc near-zero tolerance was used.
- Raw MPPI itself falls in 2/32 continuations; the policy-action arms fall in
  1/32 and seed-action arms in 2/160. These are descriptive counts over four
  selected states, not independent episode samples or reliability estimates.
  Every fall occurs after the 48-control scoring window:
  - success/strong/10, future 41004: policy falls at control 195;
  - failure/intermediate/126, future 41002: raw falls at control 217;
  - failure/strong/0, future 41001: seed 31002 falls at control 201;
  - failure/strong/0, future 41002: seed 31000 falls at control 211;
  - failure/strong/0, future 41003: raw falls at control 194.
- The large negative two-second policy means in the two failure-source cases
  are heavily influenced by preventing one raw fall each. The positive
  success/strong mean is likewise influenced by one added policy-arm fall.
  Do not replace these distributions with a single favorable mean or discard
  these events as outliers.
- Raw-only two-second cost SD across future seeds, in table order:
  5.335, 13.050, 89.742 and 121.957. Raw 48-control cost SD is much smaller:
  3.254, 2.938, 2.574 and 2.755. The reports retain all absolute ranges,
  paired ranges, sign counts and per-stream observations.

#### Secondary comparison with the already fixed v2 scores

This was allowed by the preregistration. Scores were **not** recomputed with
the new outcome streams. It compares an original single-score prediction
against an eight-stream mean effect, not v2 recalibration under each stream.

- At 48 controls, original score directions agree with the new mean in 4/4
  policy cases but only 7/20 seed-action cases (11/24 overall). Seven of the
  eight originally accepted seed actions have positive repeated mean cost
  differences; one is only +0.0188, so the count is not a claim of statistically
  established harm for each action. The one accepted policy action has a
  negative mean but helps in only four of eight short-horizon repetitions.
- At 250 controls, agreement with the repeated mean is 3/4 policy and 14/20
  seed actions. This is persistence information, not a horizon-matched
  scoring validation. The original single-stream realized effect agrees in
  sign with the new mean for only 7/24 actions at this longer horizon.
- Interpretation: conditional intervention outcomes are highly sensitive to
  future planning randomness, and a single continuation was insufficient to
  characterize these actions. Averaging outcome labels does not make the
  original single-score ranking reliable at its own horizon. This does not
  establish that scorer randomness is the sole cause: continuation-model
  error and finite eight-stream estimation error remain unresolved.
- Decision: do not activate or retune v2, hand over policy plans, or claim a
  safe/qualified GPS teacher from this study. Keep the frozen teacher and
  all evidence intact. No broader qualification or fresh-state study begins.
- Next proposed *single* diagnostic (not implemented/run): repeat only the
  v2 score computation over a separately preregistered set of planner seeds,
  keeping the candidate actions, original nominal, cost and horizon fixed.
  Compare mean predicted differences and their variability with these
  separately sampled outcome means. This would test whether scorer variance
  explains the ranking failure before changing the acceptance rule or moving
  policy influence inside the optimizer. It remains known-state development
  analysis and would not by itself establish fresh-state reliability.
- Root report SHA-256:
  `d369e1e6b49646dcc9c142bb940b6fd0ecf88fbe5ba5690a96cad72362397d99`.
  Root artifact-manifest SHA-256:
  `b82a1099ccd5b162735c4d70aa7e90b3e53d98b82469d1c1d9ebcd7d34482a29`.
- Case report SHA-256 fingerprints, in table order:
  - `520b3e51e83e6b4a41261187cf9e4f793b54d59001218138864fa26218b98d17`
  - `894bde3299064c9702e3969539b4d8ca448583d6f9eb25210fc321da0486aa4b`
  - `a6dd523340a79e38b91af183d7d8b13b024934f79e243b252ba881e5eb86eece`
  - `30ad98fe7e9552ce80d862191c1a828eaf19c752e28efda1afbd68e6fa9403d4`

Commands used (these output directories are now immutable; use a new run ID
for any explicitly planned replication):

```sh
for case_name in success_intermediate_96 success_strong_10 failure_intermediate_126 failure_strong_0; do
  .venv/bin/python -m scripts.diagnostics.walker_nominal_ablation \
    --controls-dir "runs/2026-10-12_s3_action_controls_001/$case_name" \
    --out-dir "runs/2026-10-12_s3_nominal_ablation_001/$case_name" \
    --preregistration docs/continuation_diagnostics.md || exit
done

.venv/bin/python -m scripts.diagnostics.walker_future_seed_check \
  --out-dir runs/2026-10-12_s3_future_seed_check_001 \
  --preregistration docs/continuation_diagnostics.md
```

## 2026-10-21 — MPPI-GPS as mirror descent (B): objective, audit, LQR oracle

Plan: the rewritten 2026-10-21 prompt (one method, one closed-form check, one
task, one comparison). Starting commit `806719a`. New behavior is off by
default: `MPPIConfig.lam_target_neff_fraction=None` leaves the frozen teacher
byte-identical (all 479 prior tests pass unchanged; also on a clean clone on
robopil, RTX 5080 host, `uv sync --locked`).

### Step 1 — objective before code

- [objective.md](objective.md) writes (A) what the code optimizes, (B) MPPI-GPS
  as mirror descent and (C) a CIO policy in one notation, from the primary
  sources' equations (Williams 2018 §III, MDGPS §§2–4, PI²-GPS §§III–IV,
  CIO 2012 §4, Mordatch & Todorov 2014, Mordatch et al. 2015, Amos §2.2).
- Verdict on (A): confirmed. `main` trains no policy in the loop. The June
  loop is regression-based imitation of a full-strength MPPI (DAgger/PLATO
  family). The only policy-centred attempt (`5a667e5`) had no trust region
  (λ = 0.014) and was judged as a controller.
- The three facts (improvement inequality, MPPI = −(Σ/λ)∇F_λ by Stein's lemma,
  plug-in KL ≤ log(K/n_eff) with the REPS dual derivative ε − KL) were checked
  numerically: Stein step = closed-form q* mean to 3e-16; dual derivative
  matches to 1e-6.
- (B) is PI²-GPS with global-policy sampling plus a receding horizon; the
  narrow claim to test is the receding-horizon E-step (with or without a
  policy tail).

### Step 2 — audit of what blocks (B)

Tests: [test_audit_b.py](../tests/test_audit_b.py) (5 pass, 2 strict xfails
documenting defects). Numbers: `runs/2026-10-21_audit_b_001/report.json`,
from `scripts/diagnostics/audit_b_measurements.py`.

| Item | Finding | Evidence |
|---|---|---|
| `plan_step(nominal=)` | Discards MPPI's own warm start. Correct for (B), whose prior is the policy rollout; it is why the April run could not also keep MPPI's plan. | passing test |
| Fixed λ | Median n_eff under frozen configs: Walker 7.7/256 (KL step ≤ 3.5 nats), Acrobot **1.00**/512 (pure argmin, ≤ 6.2 nats), point mass 2.4/256. | measured, 150/50/50 decisions |
| Label noise | Walker first-action std across 16 planner seeds at 5 fixed states: frozen λ 0.091; n_eff/K = 0.05: 0.072; n_eff/K = 0.2: **0.037** (λ ≈ 11). | measured |
| Raw-angle observations | Acrobot obs not 2π-periodic. Walker unaffected (angles bounded by the healthy range). | strict xfail |
| No cost-to-go | Walker charges nothing for a healthy end state; plans see 0.384 s. | passing test; S3's −7.3 vs +23.8 |
| Unpaired seeds | Runners other than `run_walker2d` build MPPI without a seed, so planner noise shares numpy's global RNG with resets. New scripts seed every planner. | strict xfail |
| Warp path | No test exercises `use_warp=True`; not used here. | grep |

Fix kept (off by default): `lam_target_neff_fraction` bisects λ per
iteration so n_eff = ρK (`src/utils/math.py`), bounding the step by log(1/ρ).

### Step 3 — closed-form oracle (LQR point mass)

- Question: does (B) with a linear policy converge to the LQR gain?
- Setup: `src/envs/lqr_point_mass.py` (joint limits removed; exact
  linear step, tested to 1e-10; terminal weight P − Q makes −Kx the H-step
  optimum for every H, tested in the simulator's own scoring). Policy
  u = −Gx from G = 0; per iteration 6 episodes × 25 on-policy states, one
  policy-centred E-step per state (H=16, σ=0.3), least-squares M-step; 30
  iterations; seeds 0, 1. Run: `runs/2026-10-21_lqr_oracle_001/`
  (`scripts/diagnostics/lqr_oracle.py`, 240 s local CPU).
- Result (relative gain error ‖G−K‖/‖K‖, mean over 2 seeds; floor = median of
  iterations 20–30):

  | K | n_eff/K | error at it 10 | floor | first iter < 0.10 |
  |---:|---:|---:|---:|---|
  | 64 | 0.05 / 0.2 / 0.5 | 0.17 / 0.13 / 0.15 | 0.17 / 0.10 / 0.08 | 9–16 |
  | 256 | 0.05 / 0.2 / 0.5 | 0.08 / 0.03 / 0.10 | 0.07 / **0.046** / 0.040 | 5–11 |
  | 1024 | 0.05 / 0.2 / 0.5 | 0.04 / 0.03 / 0.10 | 0.05 / 0.032 / **0.019** | 5–11 |
  | 256, frozen λ=0.01 (n_eff/K 0.004) | — | 0.34 | **0.23 ± 0.04** | 22 / never |
  | 256, 0.2, started at K | — | 0.04 | 0.046 | 0 |

- Fact 1 held in every E-step (minimum sample improvement 0.40 > 0).
  Started at the optimum, (B) stays at the same 4.6% floor, so the floor is
  label noise, not bias; it shrinks with K and with n_eff/K. Larger n_eff/K
  converges more slowly to a lower floor: the trust-region trade-off. The
  mean plan beats the policy plan early on and stops doing so once
  converged, which is the signal MDGPS's step rule uses to shrink the step.
- Decision: **keep**. The closed-form check passes; proceed to Walker.

### Step 4 — Walker headroom (predictions written before running)

- Question: where does raw MPPI on Walker leave room for a policy (20–60%
  eight-second completion), and does a policy's simulated cost-to-go as
  MPPI's terminal cost use that room?
- Code: `scripts/diagnostics/walker_headroom.py`, `src/gps/policy_tail.py`.
  Frozen Walker task and planner config except K and H. Development resets
  80000+, planner seeds 90000+ (disjoint from SAC's 0–39999/50000+/60000+ and
  from the held-out set 70000–70049 reserved for Step 5). 10 episodes per
  cell, 1,000 controls. Runs on robopil (24 cores, one MuJoCo thread per
  worker).
- Grid: H ∈ {8, 12, 16, 24, 32, 48} × K ∈ {32, 64, 128, 256}, frozen λ = 3.
- Selection rule (fixed now): the cell with completion closest to 40% within
  [20%, 60%]; ties go to the smaller K·H. If no cell is in the band, take the
  nearest.
- Tail test at that cell: four arms, all with the same n_eff-targeted
  temperature (the raw cell's median n_eff/K, so that only the tail's
  information differs): no tail; tail from SAC weak (0k), intermediate (200k)
  and strong (800k); 100 tail controls (0.8 s); 10 episodes each.
- Predictions:
  1. Completion rises with H and with K. Cells with H ≥ 32 and K ≥ 128
     complete ≥ 80%; at K = 256, H ≤ 12 completes < 40%.
  2. Strong tail: completion ≥ no-tail + 30 points and ≥ 80% (rollout's cost
     improvement over a policy that completes 30/30 alone).
  3. Intermediate tail: completion ≥ no-tail + 20 points; speed MAE not
     better than no-tail (it survives by standing still).
  4. Weak tail: within ±20 points of no-tail (it falls from every state, so
     its tail adds a near-constant cost).
- Status: grid running. A first launch (`..._grid_001`) was stopped after 3
  minutes and kept as `runs/2026-10-22_walker_headroom_grid_001_aborted_blas_threads`:
  each worker spawned 48 BLAS threads (load 31 on 24 cores), which distorts
  the per-decision timings. The relaunch (`..._grid_002`) sets
  `OMP/OPENBLAS/MKL_NUM_THREADS=1`; nothing else changed.
- Order change: to start training sooner, the tail test runs alongside Step 5
  training instead of before it (it only informs an optional later B+tail
  arm). The selection rule is now code (`walker_headroom.py select`; ties
  compare rounded rates, since rates are multiples of 0.1) and is applied
  automatically by `scripts/training/launch_step5.sh`.

- Grid result (`runs/2026-10-22_walker_headroom_grid_002/report.json`,
  240 episodes, 1,057 s on robopil). Completions out of 10 eight-second
  episodes:

  | H \ K | 32 | 64 | 128 | 256 |
  |---:|---:|---:|---:|---:|
  | 8 | 0 | 0 | 0 | 0 |
  | 12 | 0 | 0 | 0 | 0 |
  | 16 | 0 | 0 | 0 | 0 |
  | 24 | 1 | 0 | 0 | 1 |
  | 32 | **5** | 7 | 5 | 5 |
  | 48 | 10 | 10 | 8 | 10 |

  Below H = 32, every cell falls after 1.1–1.7 s on average. At H = 48, speed
  MAE improves with K (0.358 → 0.139 m/s) though completion does not. With
  λ fixed at 3, median n_eff/K falls from 0.9 (H = 8) to 0.13 (H = 32) and
  0.03–0.05 (H = 48): the same temperature is a much larger KL step at long
  horizons.
- Prediction 1: **partly wrong.** Completion rises with H but is flat in K;
  H ≥ 32 with K ≥ 128 is only 5/10 at H = 32; K = 256 with H ≤ 12 is 0/10 as
  predicted.
- Selected cell (rule applied by `launch_step5.sh`, `selection.json`):
  **K = 32, H = 32**, 5/10 (tied with K = 128 and 256; smaller K·H wins);
  median n_eff/K 0.134, used by every tail-test arm. Step 5 and the tail test
  launched automatically at 00:02 on 2026-10-22.

- Tail test result (`runs/2026-10-22_walker_headroom_tail_001/report.json`,
  40 episodes on the grid's dev seeds, 2,244 s on 3 workers alongside Step 5).
  K = 32, H = 32, every arm with λ solved per decision for n_eff/K = 0.134,
  100 tail controls:

  | Arm | Completed / 10 | Speed MAE | Fisher p vs no tail | Plan s/decision |
  |---|---:|---:|---:|---:|
  | No tail | 2 | 0.388 | — | 0.048 |
  | Weak SAC (0k) tail | **9** | 0.479 | 0.0055 | 0.257 |
  | Intermediate (200k) tail | **10** | 0.472 | 0.0007 | 0.199 |
  | Strong (800k) tail | **10** | **0.287** | 0.0007 | 0.202 |

- Predictions 2 and 3: **held** (+80 points each; the intermediate tail's MAE,
  0.472, is not better than no tail's 0.388). Prediction 4: **wrong** (+70
  points). A policy that falls in 0.2 s still ranks horizon-end states by how
  long they stay healthy, which is a balance cost-to-go the 0.256 s horizon
  lacks.
- Caveats: the matched-temperature no-tail arm completed 2/10 where frozen
  λ = 3 completed 5/10 on the same seeds. On raw compute, a longer horizon is
  the cheaper fix at this cell (H = 48, K = 32: 10/10 at 0.084 s/decision
  versus ~0.2 s with a tail). Only the strong tail beats it on speed MAE
  (0.287 vs 0.358). The tail matters for MPPI-GPS because it is how the
  policy's long-horizon behavior enters the E-step.

### Step 5 — main experiment (predictions written before running)

- Cell: the Step 4 selection, written by the launcher to the grid run's
  `selection.json` and recorded here with the results.
- Arms, training seeds 0, 1, 2 each, `scripts/training/train_walker_mdgps.py`:
  - **(B) MPPI-GPS.** Random-init MLP (SacActor shape, 17→256→256→6, tanh).
    One policy-centred MPPI step per labeled state (every 2nd executed
    state), n_eff/K = 0.2. Executed action = policy + N(0, 0.1²). Replay of
    the last 2 iterations; 2,000 Adam steps (lr 3e-4, batch 256) per
    iteration; 25 iterations × 10 episodes × 400 controls. Training episodes
    are 3.2 s to bound compute; every evaluation uses 1,000 controls.
  - **DAgger from raw MPPI.** Same policy, fit and collection. Labels come
    from a raw MPPI expert at the same cell (frozen λ = 3, own warm start,
    plans at every state); iteration 0 executes the expert.
  - **References:** raw MPPI at the cell (50 held-out episodes, planner seeds
    95000+) and the SAC 800k checkpoint.
  - **Not run: the June best config.** `origin/gps/experiments` optimizes a
    different Walker task (`walker2d.py`, XML, no task config, white noise),
    so a fair comparison needs a port of that loop to main's task, not a
    re-run.
- Seeds: collection resets 100000 + 10000·seed + …; dev evaluation
  81000–81009, monitoring only; held-out resets 70000–70049 used once, on the
  last iteration's policies (no checkpoint selection).
- Primary metric: policy-alone eight-second completion on held-out resets,
  pooled over training seeds (150 episodes), Wilson 95%; paired Newcombe
  differences per seed against raw MPPI and SAC 800k. Secondary: speed MAE
  (paired bootstrap). Logged per iteration: executed cost per step, predicted
  vs actual improvement, n_eff/K, λ.
- Predictions:
  1. Fact 1 holds in every E-step (minimum sample improvement ≥ 0).
  2. (B) pooled completion ≥ 50%.
  3. (B) completion ≥ DAgger completion, because DAgger imitates a
     20–60% expert whose labels carry frozen-λ jitter (n_eff/K ≈ 0.03).
  4. Neither learned policy reaches SAC 800k's speed MAE (0.20) in 25
     iterations.
  5. (B) vs raw MPPI at the same cell: no directional prediction (both
     optimize the same H-step objective); report the two-sided interval.
- Status: complete (launched 00:02, 2026-10-22).
- Result. Training: 25 iterations for all six runs in ~38 minutes wall, in
  parallel. Held-out evaluation: `runs/2026-10-22_step5_heldout_001/report.json`,
  102 s. Resets 70000–70049; policies alone, last iteration:

  | Arm | Completed | Wilson 95% | Speed MAE | Completion vs raw MPPI, per seed (Newcombe 95%) |
  |---|---:|---|---:|---|
  | Raw MPPI, K = 32, H = 32 | 19/50 (38%) | 26–52% | 0.387 | — |
  | (B) MPPI-GPS, seeds 0/1/2 | 0, 1, 0 of 50 (0.7%) | 0.1–3.7% | 0.927 | −0.38 [−0.52, −0.24], −0.36 [−0.50, −0.21], −0.38 [−0.52, −0.24] |
  | DAgger from raw MPPI, seeds 0/1/2 | 0, 0, 0 of 50 | 0–2.5% | 0.792 | −0.38 [−0.52, −0.24] each |
  | SAC 800k | 50/50 | 93–100% | 0.198 | +0.62 [+0.46, +0.74] |

- Training curves (`runs/2026-10-22_step5_*/metrics.jsonl`):
  - (B): mean collection episode length rose from 16–22 controls
    (≈0.15 s) at iteration 0 to 144–175 (≈1.3 s) at iteration 24, still
    rising; executed cost per step fell from 6.3–7.5 to 2.2–2.3. Dev
    completion was 0/10 at every evaluation except one 1/10 (seed 1,
    iteration 14).
  - (B), theory checks: fact 1 held in every E-step (minimum sample
    improvement 0.076). The M-step realized ~40–60% of the predicted
    improvement (mean predicted 9–12, actual 4–6 per probe state).
  - DAgger: iteration 0 executes the expert (186–245 of 400 controls);
    afterwards the policy hovers at 108–225 controls with no trend.
- Videos (`runs/2026-10-22_step5_videos_001/`, held-out reset 70000).
  (B) keeps the torso upright, folds its knees and sinks below the 0.8 m
  height limit, at iteration 9 and at 24. DAgger pitches forward at
  ~1 m/s and falls.
- Predictions: 1 **held**. 2 **wrong** (0.7%). 3 **undecided** (1/150 vs
  0/150). 4 **held**. 5: (B) is 36–38 points below raw MPPI in every seed.
- Reproduction: a clean clone at `d414a85` on robopil (`uv sync --locked`, no
  local edits) reproduced all 350 held-out episodes exactly, and (B) seed 0
  iterations 0–1 bit-for-bit, including the CUDA fit loss.
- Interpretation: the E-step does what the theory says (fact 1; positive
  actual improvement every iteration), and the policy keeps improving. But an
  H = 32 objective cannot see the failure it converges into: sinking is free
  until height crosses 0.8 m. Without a cost-to-go, (B)'s fixed point is a
  receding-horizon policy with that blind spot, and 25 iterations do not
  reach raw MPPI's 38%. Step 4 shows what supplies the cost-to-go: any
  policy's simulated continuation moves the planner from 2/10 to 9–10/10.
- Not run: the June best config (task mismatch, above). robopil5090 rejects
  key login, so everything ran on robopil (24 cores, RTX 5080).
- Decision: **revise.** Keep (B) and its machinery. The next single change is
  the current policy's own simulated tail inside the E-step (policy
  iteration, the "MPPI absorbs the policy's long-horizon information" idea),
  at the same cell, seeds and budget, compared against this run.


### Step 6 — MPPI-GPS with the policy's own tail (predictions written before running)

- Question: does adding the current policy's simulated cost-to-go to the
  E-step (policy iteration with the policy as rollout base) turn Step 5's
  failing MPPI-GPS into a completing policy?
- One change from Step 5 (B): `--cfg.tail-steps 100`. Every E-step cost —
  samples, policy nominal and improved plan — is H = 32 controls plus 100
  controls (0.8 s) of the current policy's closed-loop continuation, with
  absorbing-fall accounting (`src/gps/policy_tail.py`, now routed through
  `e_step(..., terminal=)` so all three use one objective). The improvement
  check scores the new policy on its own H + tail cost.
- Same as Step 5: K = 32, H = 32, n_eff/K = 0.2, training seeds 0–2 with
  identical collection, planner and noise seeds, 25 iterations × 10 episodes
  × 400 controls, label every 2nd state, exec noise 0.1, replay 2,
  2,000 Adam steps. Same data budget; planner compute per E-step is higher.
- Command (robopil, BLAS threads 1): for seed in 0 1 2:
  `train_walker_mdgps --out-dir runs/2026-10-22_step6_mdgps_tail_seed$seed
  --cfg.labeler mdgps --cfg.seed $seed --cfg.K 32 --cfg.H 32
  --cfg.tail-steps 100 --cfg.iterations 25 --cfg.episodes-per-iter 10
  --cfg.steps 400 --cfg.workers 6`. Worker count changes speed only:
  episodes are seeded per job.
- Comparison: held-out resets 70000–70049 with `walker_step5_compare.py`,
  against Step 5's (B), DAgger, raw MPPI (K = 32, H = 32) and SAC 800k, plus
  per-seed paired differences (B+tail seed i vs B seed i) via `--pairs`.
- Predictions:
  1. Fact 1 holds in every E-step, tail included.
  2. (B+tail) pooled completion exceeds (B)'s 1/150, with the paired
     per-seed interval excluding 0 in at least 2 of 3 seeds.
  3. (B+tail) pooled completion ≥ raw MPPI's 38% (less certain).
  4. (B+tail) speed MAE stays worse than SAC 800k's 0.20; no directional
     prediction against (B)'s MAE (it may learn to stand before it walks).
  5. Planner time per labeled state is ~4× (B)'s.
- Status: complete. Launched 01:01 on 2026-10-22 at `7c30f3a`; 25 iterations
  per seed in 347–888 s wall (6 workers each).
- Held-out result (`runs/2026-10-22_step6_heldout_001/report.json`, resets
  70000–70049, policies alone, last iteration; arms re-evaluated in the same
  run, identical to Step 5 where shared). Time alive is an exploratory
  descriptor added after Step 5, not preregistered:

  | Arm | Completed | Wilson 95% | Speed MAE | Mean time alive |
  |---|---:|---|---:|---:|
  | Raw MPPI, K = 32, H = 32 | 19/50 | 26–52% | 0.387 | 3.78 s |
  | (B) + policy tail, seeds 0/1/2 | 0/150 | 0–2.5% | 1.115 | 1.36 s (1.58, 1.47, 1.02) |
  | (B), Step 5 | 1/150 | 0.1–3.7% | 0.927 | 1.34 s (1.27, 1.50, 1.24) |
  | DAgger, Step 5 | 0/150 | 0–2.5% | 0.792 | 1.30 s |
  | SAC 800k | 50/50 | 93–100% | 0.198 | 8.00 s |

- Paired by reset seed, (B)+tail minus (B), seeds 0/1/2:
  - Completion: +0.00 [−0.07, +0.07], −0.02 [−0.10, +0.05],
    +0.00 [−0.07, +0.07].
  - Speed MAE: +0.151 [+0.013, +0.294], +0.031 [−0.054, +0.122],
    +0.384 [+0.240, +0.529].
  - Time alive (exploratory): +0.31 s [+0.07, +0.63],
    −0.03 s [−0.37, +0.21], −0.22 s [−0.35, −0.09].
  - Against raw MPPI, each tail seed is −0.38 [−0.52, −0.24].
- Training: dev time alive at iteration 24 was 1.9, 1.5 and 0.9 s (seed 0
  completed 1/10 dev episodes at iterations 19 and 24). Collection episode
  length was still rising for seeds 0 and 1.
- Theory checks: fact 1 held in every E-step (minimum sample improvement
  0.066). Predicted vs actual improvement was noisy, with both signs within
  single iterations.
- Compute: ≈0.31 CPU-s per labeled state against ≈0.078 without the tail.
- Videos (`runs/2026-10-22_step6_videos_001/`, reset 70000). Seed 0 no
  longer crouches and sinks: it steps, then pitches forward at ~1.2 s, like
  DAgger. Seed 2 takes a splayed stride and collapses at ~1.1 s.
- Predictions: 1 **held**. 2 **wrong**: no seed's paired completion interval
  excludes 0. 3 **wrong**. 4 **held**. 5 **held** (~4×).
- Interpretation: the tail that lifts the *planner* from 2/10 to 9–10/10
  (Step 4) does not carry over to the *learned policy* in this budget; it
  changes the failure mode, not the time to failure. Every learned arm —
  (B), (B)+tail, DAgger — stalls at 1.3–1.4 s whatever its labels optimize,
  so the binding constraint is the learning loop, not the E-step objective.
  The most suspect part of that loop is label quality: the Step 4 rule broke
  ties toward K = 32, so each label averages n_eff ≈ 6 samples. The LQR
  oracle showed K sets the label-noise floor (10% at K = 64 vs 4.6% at
  K = 256, n_eff/K = 0.2), and raw MPPI's headroom at H = 32 does not depend
  on K (5–7/10 for K = 32–256).
- Decision: **revise.** Drop the tail from the next comparison (no effect on
  completion or time alive). Change one factor against Step 5's (B): E-step
  K = 256 at H = 32 (n_eff ≈ 51 per label at n_eff/K = 0.2), same seeds and
  data budget. The grid makes it a fair cell: raw MPPI at K = 256, H = 32 also
  completes 5/10.

### Step 7 — MPPI-GPS with K = 256 E-steps (predictions written before running)

- Question: is label quality what stalls the learned policy near 1.3 s?
- One change from Step 5 (B): E-step sample count K = 32 → 256 at H = 32.
  With n_eff/K = 0.2, each label averages n_eff ≈ 51 samples instead of ≈ 6.
  No tail. Everything else is Step 5's: n_eff/K = 0.2, training seeds 0–2 with
  identical collection, planner and noise seeds and policy initialization,
  25 iterations × 10 episodes × 400 controls, label every 2nd state, exec
  noise 0.1, replay 2, 2,000 Adam steps. The cell is fair: raw MPPI at
  K = 256, H = 32 also completed 5/10 in the grid.
- Command (robopil, BLAS threads 1): for seed in 0 1 2:
  `train_walker_mdgps --out-dir runs/2026-10-22_step7_mdgps_k256_seed$seed
  --cfg.labeler mdgps --cfg.seed $seed --cfg.K 256 --cfg.H 32
  --cfg.iterations 25 --cfg.episodes-per-iter 10 --cfg.steps 400
  --cfg.workers 7`.
- Comparison: held-out resets 70000–70049 with `walker_step5_compare.py`
  against every earlier arm, plus per-seed pairs (K = 256 seed i vs Step 5's
  (B) seed i) via `--pairs`.
  - Primary: completion (Wilson; paired Newcombe).
  - Secondary, now preregistered: mean time alive (paired bootstrap),
    defined in the Step 6 comparison.
- Predictions:
  1. Fact 1 holds in every E-step.
  2. Time alive exceeds Step 5's (B) in at least 2 of 3 seeds (paired
     interval above 0).
  3. Pooled completion stays below raw MPPI's 38%: 25 KL-bounded iterations
     are a small budget even with clean labels.
  4. The M-step realizes a larger share of the predicted improvement than
     at K = 32 (mean actual/predicted there: 4.2–5.7 of 9.1–12.1).
  5. CPU per labeled state is ~6–8× Step 5's.
- Status: complete. Launched 01:24 on 2026-10-22 at `4066756`; 25 iterations
  per seed in 767–1,394 s wall (7 workers each).
- Held-out result (`runs/2026-10-22_step7_heldout_001/report.json`; arms
  shared with Steps 5–6 re-evaluated identically):

  | Arm | Completed | Speed MAE | Time alive (seeds 0/1/2) |
  |---|---:|---:|---:|
  | Raw MPPI, K = 32, H = 32 | 19/50 | 0.387 | 3.78 s |
  | (B), K = 256 | 0/150 | **0.755** | **0.99 s** (0.77, 0.89, 1.30) |
  | (B), K = 32 (Step 5) | 1/150 | 0.927 | 1.34 s (1.27, 1.50, 1.24) |
  | (B) + tail, K = 32 (Step 6) | 0/150 | 1.115 | 1.36 s |
  | DAgger (Step 5) | 0/150 | 0.792 | 1.30 s |
  | SAC 800k | 50/50 | 0.198 | 8.00 s |

- Paired by reset seed, K = 256 minus K = 32, seeds 0/1/2:
  - Completion: +0.00 [−0.07, +0.07], −0.02 [−0.10, +0.05],
    +0.00 [−0.07, +0.07].
  - Speed MAE: −0.312 [−0.396, −0.242], −0.310 [−0.375, −0.256],
    +0.107 [−0.008, +0.221].
  - Time alive: −0.50 s [−0.57, −0.44], −0.61 s [−0.92, −0.41],
    +0.05 s [−0.09, +0.19].
- Training: seeds 0 and 1 reached 1.5–1.8 s dev time alive by iterations
  4–9, then declined to 0.8–0.9 s while dev speed MAE fell from 1.2–2.3 to
  0.58–0.59. Seed 2 was still rising at iteration 24 (1.2 s).
- Theory and fit checks: fact 1 held in every E-step (minimum 0.074). The
  M-step realized 91%, 60% and 58% of predicted improvement (K = 32: 39%,
  63%, 36%). Fit loss was ~10× lower.
- Compute: 0.57–0.71 CPU-s per labeled state (7–9× Step 5).
- Videos (`runs/2026-10-22_step7_videos_001/`, reset 70000). Seed 0 at
  iteration 24 leans into the target speed and dives forward, falling at
  0.74 s at ~1 m/s. At iteration 4 it stepped backward upright and fell at
  1.2 s.
- Predictions: 1 **held**. 2 **wrong**: time alive is *shorter* in 2 of 3
  seeds, both intervals excluding 0. 3 **held**. 4 **held** in 2 of 3 seeds.
  5 **held**, slightly above (7–9×).
- Interpretation: label noise was not what held (B) near 1.3 s. With clean
  labels, (B) optimizes its 32-step objective better — speed tracking
  improves by 0.31 m/s in two seeds — and that objective, with no
  cost-to-go, rewards diving toward the target speed. Steps 6 and 7 each
  changed one factor: the tail with noisy labels gave nothing, and clean
  labels without the tail optimized the wrong objective. The untested cell is
  both together. The tail's fall costs add large variance to sample scores,
  and at K = 32 each label averaged n_eff ≈ 6 samples, likely too few for the
  cost-to-go to move the labels.
- Decision: **revise.** Next single change: K = 256 with the 100-control
  tail, same seeds and data budget, completing the 2 × 2 (K × tail). Its
  cost is roughly 4× this run per label (the tail is K × 100 extra controls).

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
