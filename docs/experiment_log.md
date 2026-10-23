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
8. Futility rule (adopted 2026-10-23, at the user's request). Learned arms are watched for learning progress as well as health.
   - From iteration 29, and every 5 iterations after, compare the mean dev time alive over the arm's last three dev evaluations with its first three.
   - Stop the arm if that rise is under 0.3 s and no seed has completed more than 2 of 10 dev episodes. Record it as stopped for futility.
   - Checked against earlier arms at iteration 29: MPPI-GPS (Steps 10/12) rose +0.33 s and reached 68% held out; DAgger (Step 11) rose +0.03 s and finished at 0%.

## Experiments

### S1: Walker teacher instrumentation and exact replay

```text
Experiment ID / date: S1 / 2026-10-11
Question: Can the frozen Walker teacher's decisions and its recorded qualification failure be reproduced exactly, mid-episode included, and what did the planner predict before that fall?
Branch: exp/walker-instrumentation (fast-forwarded into main as 9786193; branch deleted)
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
Decision: keep (instrumentation); committed as the S1 checkpoint 9786193, now on main.
Interpretation and limitations: One fall, heuristic evidence. "Survives" means within the probe horizon only. Whether the surviving 48-step options at steps 489-501 would have avoided the fall beyond the horizon is untested; a paired counterfactual branch from those decisions (S3 machinery) would test it. Short-horizon causes are not attributed.
Evidence backup: /Users/jaisel/Documents/workbench/mppi-gps-evidence/2026-10-11_walker_instrumentation.tar.gz
  (SHA-256 b6a3510f313a68e3610a750345eab424522d4a3d4d7a03167e2ed254f76cec7d); all 18 files verified against the manifest after extraction.
```

### S2a: task-matched RL interface and learning plumbing

- Starting commit: `86a29d7bafc5af8e3c22235a1e7269b02535f078`, on `main`.
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
- Starting commit: `86a29d7bafc5af8e3c22235a1e7269b02535f078`. Additive working
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

- Audited S2/S3 v1 preserved as local commit `b7d897e` after all 278 tests
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
- Protocol committed before the simulations as `71b18a2`; its file SHA-256
  is `4e4061c381630515e21a346d996f032d398e8936546d70e18a5bba038865feb6`.
  Implementation and its tests committed as `6ed8ec8` before inspecting v2
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

- Starting commit: `5a05a32`. Fixed protocol:
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

- Protocol commit `3fcbcfa`; implementation commit `60cfef5`. All 24 declared
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
  committed as `8805eb2`. The protocol stayed at `3fcbcfa`, SHA-256
  `204cc756b215919407b283be35f4baf740f6f73b5d5d03727fc303c5131e6dae`;
  diagnostic implementation stayed at `60cfef5` throughout both experiments.
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
  helpers, runner, configs and XML remain byte-identical to `9786193`.
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
task, one comparison). Starting commit `d128d46`. New behavior is off by
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
- Reproduction: a clean clone at `3f23193` on robopil (`uv sync --locked`, no
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
- Status: complete. Launched 01:01 on 2026-10-22 at `78e0296`; 25 iterations
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
- Status: complete. Launched 01:24 on 2026-10-22 at `502f62d`; 25 iterations
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

### Step 8 — MPPI-GPS with K = 256 and the policy tail (predictions written before running)

- Question: with labels clean enough (n_eff ≈ 51) to carry it, does the
  current policy's simulated cost-to-go turn MPPI-GPS into a policy that
  survives? This completes the K ∈ {32, 256} × tail ∈ {no, yes} design.
- Settings: Step 7's K = 256 at H = 32 plus Step 6's 100-control tail
  (`--cfg.tail-steps 100`). Otherwise Step 5's settings: n_eff/K = 0.2,
  training seeds 0–2 with identical collection, planner and noise seeds and
  policy initialization, 25 iterations × 10 episodes × 400 controls, label
  every 2nd state, exec noise 0.1, replay 2, 2,000 Adam steps.
- Command (robopil, BLAS threads 1): for seed in 0 1 2:
  `train_walker_mdgps --out-dir runs/2026-10-22_step8_mdgps_k256_tail_seed$seed
  --cfg.labeler mdgps --cfg.seed $seed --cfg.K 256 --cfg.H 32
  --cfg.tail-steps 100 --cfg.iterations 25 --cfg.episodes-per-iter 10
  --cfg.steps 400 --cfg.workers 7`.
- Comparison: held-out resets 70000–70049 against every earlier arm, plus
  per-seed pairs against Step 7 (K = 256, no tail; isolates the tail) and
  Step 6 (K = 32, tail; isolates K).
  - Primary: completion (Wilson; paired Newcombe).
  - Secondary: mean time alive (paired bootstrap).
- Predictions:
  1. Fact 1 holds in every E-step.
  2. Time alive exceeds Step 7's in at least 2 of 3 seeds (paired interval
     above 0): the tail charges the dive that clean labels learned.
  3. Time alive exceeds Step 6's in at least 2 of 3 seeds: clean labels
     carry the cost-to-go that noisy ones did not.
  4. Pooled completion is above 0/150 but below raw MPPI's 38% (least
     certain).
  5. CPU per labeled state is ~3–5× Step 7's.
- Status: complete. Launched 01:52 on 2026-10-22 at `b4f700f`; 25 iterations
  in 7,012 s, 6,071 s and 773 s wall (seeds 0/1/2, 7 workers each).
- Held-out result (`runs/2026-10-22_step8_heldout_001/report.json` on
  robopil; arms shared with Steps 5–7 re-evaluated identically). The full
  K × tail design, held-out completion, speed MAE and mean time alive (seeds
  0/1/2):

  | | No tail | Policy tail (100 controls) |
  |---|---|---|
  | K = 32 | 1/150, 0.927, 1.34 s (Step 5) | 0/150, 1.115, 1.36 s (Step 6) |
  | K = 256 | 0/150, 0.755, 0.99 s (Step 7) | 0/150, 1.118, **1.45 s** (1.99, 1.46, 0.89) |

  References: raw MPPI (K = 32, H = 32) 19/50, 0.387, 3.78 s; SAC 800k
  50/50, 0.198, 8.00 s. Step 8 per-seed MAE: 0.65, 0.69, 2.01.
- Paired by reset seed, K = 256 + tail minus K = 256, seeds 0/1/2:
  - Time alive: +1.22 s [+1.11, +1.36], +0.57 s [+0.48, +0.67],
    −0.40 s [−0.56, −0.25].
  - Speed MAE: +0.070 [+0.051, +0.090], +0.118 [+0.094, +0.142],
    +0.901 [+0.786, +1.008].
- Paired by reset seed, K = 256 + tail minus K = 32 + tail, seeds 0/1/2:
  - Time alive: +0.42 s [+0.09, +0.70], −0.01 s [−0.13, +0.10],
    −0.13 s [−0.28, +0.04].
  - Speed MAE: −0.392 [−0.511, −0.286], −0.223 [−0.300, −0.161],
    +0.624 [+0.482, +0.764].
- Completion differences are 0 everywhere (intervals ±0.07).
- Training: seeds 0 and 1 reached dev time alive 1.9 and 1.7 s at
  iteration 24, with collection episodes still lengthening (seed 0: 21 → 241
  controls). Seed 2 stayed near 25-control episodes for ~15 iterations in
  both K = 256 runs and ended at 0.9 s.
- Theory checks: fact 1 held in every E-step (minimum 0.072). In seeds 0 and
  1 the realized improvement averaged *above* the predicted one (23.0 vs
  17.1; 17.3 vs 14.0). By MDGPS's step rule, the n_eff/K = 0.2 trust region
  is tighter than needed there.
- Compute: 2.17–2.25 CPU-s per labeled state for seeds 0/1 (3.2–3.9× Step 7).
- Video: not watched. robopil became unreachable (SSH timeout) right after
  the held-out run; the run directories are only on robopil.
- Predictions: 1 **held**. 2 **held** (seeds 0 and 1, intervals excluding 0).
  3 **wrong** (only seed 0). 4 **wrong** (0/150). 5 **held**.
- Interpretation: the factors interact. The tail lengthens survival only
  with clean labels (K = 256: +1.22 and +0.57 s in two seeds; K = 32: no
  effect). Clean labels sharpen speed tracking only without the tail. Seeds 0
  and 1 here are the first learned policies that survive longer *and* still
  track speed (MAE 0.65–0.69). No learned policy completes an episode in 25
  iterations, and the best one (1.99 s) is still far from raw MPPI at the
  same cell (3.78 s, 38%).
- Decision: **keep** K = 256 + tail as the working configuration. The two
  good seeds were still improving and realized more than the predicted gain,
  so the next single change is MDGPS's adaptive step size (objective.md,
  "Data and step size"). It replaces the fixed n_eff/K = 0.2, with the same
  seeds and budget. Restore access to robopil first.

### Moving to robopil5090 (2026-10-22)

- robopil (`robopil-5080-alpha`, 128.59.22.12) stopped answering SSH after
  Step 8's held-out run. Steps 4–8 artifacts are only there.
- robopil5090 (`robopil5090-Lambda-Vector`, Ryzen 9 9950X, 16 cores / 32
  threads, RTX 5090) is shared by nine users. Its disk had 8.6 GB free, too
  little for CUDA PyTorch, so it uses `scripts/setup_cpu_env.sh`: the lock
  minus torch/triton/nvidia-*, plus torch 2.10.0 CPU (2.7 GB venv; mujoco
  3.6.0 and numpy 2.4.3 as before). 492 tests pass.
- Cross-machine check (`runs/2026-10-22_5090_crosscheck_*` on the 5090):

  | Check | robopil (5080) | robopil5090 |
  |---|---|---|
  | Held-out raw MPPI, K = 32, H = 32 | 19/50, MAE 0.38720, 3.78 s | 15/50, MAE 0.39197, 3.30 s |
  | Held-out SAC 800k | 50/50, MAE 0.197769 | 50/50, MAE 0.197846 |
  | Step 5 (B) seed 0, iteration 0 | 21.0 steps, 107 labels, λ 4.7131752518 | 21.0 steps, 107 labels, λ 4.7131745355 |

- Results agree to ~7 significant digits per step but are not bit-identical
  across CPUs, and MPPI's chaotic episodes turn that into different outcomes.
  15/50 lies inside 19/50's Wilson interval (26–52%), but held-out
  comparisons are valid only when every arm is evaluated on the same machine.
  Training can run anywhere.

### Step 9 — MDGPS adaptive step size, with a same-machine control (predictions written before running)

- Question: does MDGPS's adaptive step size learn faster than the fixed
  n_eff/K = 0.2 trust region at K = 256 with the tail? In Step 8, realized
  improvement exceeded prediction in the two seeds that learned.
- Arms, both on robopil5090 (CPU-only environment), training seeds 0–2,
  25 iterations × 10 episodes × 400 controls, 4 workers each, `nice 19`:
  - **Step 9 (adaptive):** Step 8's settings plus `--cfg.adapt-step
    --cfg.probe-probability 0.1`. Starts at n_eff/K = 0.2;
    eps = log(1/ρ) is rescaled after each iteration by
    `mdgps_step_multiplier` (reference clip [0.1, 5]) and kept within
    ρ ∈ [0.05, 0.5]. Probes are drawn from the same per-state random number
    as before, so collection is unchanged by the probe rate.
  - **Step 8r (control):** Step 8's exact settings re-run on this machine.
    It is the same-machine reference, and it shows whether Step 8's result
    reproduces on another CPU.
- Comparison: held-out resets 70000–70049, every arm evaluated on
  robopil5090. Arms: Step 9, Step 8r, raw MPPI (K = 32, H = 32), SAC 800k,
  and Steps 5–8 if their checkpoints come back from robopil. Per-seed pairs
  Step 9 vs Step 8r. Primary: completion. Secondary: mean time alive.
- Predictions:
  1. Fact 1 holds in every E-step of both arms.
  2. The step grows: the median ρ used over iterations 5–24 is below 0.2 in
     all three Step 9 seeds.
  3. Step 9's time alive exceeds Step 8r's in at least 2 of 3 seeds (paired
     interval above 0).
  4. Step 9's pooled completion is above 0/150 (least certain).
  5. Step 8r reproduces Step 8 qualitatively: pooled time alive within
     ±0.5 s of 1.45 s and at most 2/150 completions.
- Status: complete. Launched 13:40 on 2026-10-22 at `2face5e`; the last run
  (8r seed 0, relaunched 14:31) finished ~17:40. Wall times: 0.7–3.6 h per
  run, iterations slowed to 6–15 min on the shared CPU.
- Held-out result (`runs/2026-10-22_step9_heldout_001/report.json` on
  robopil5090, CPU-only environment, the same as the cross-check; raw MPPI
  reproduces its 15/50 exactly):

  | Arm | Completed | Speed MAE | Time alive (seeds 0/1/2) |
  |---|---:|---:|---:|
  | Raw MPPI, K = 32, H = 32 | 15/50 | 0.392 | 3.30 s |
  | **Step 9, adaptive step** | 1/150 | 0.903 | **1.85 s** (1.61, 2.58, 1.38) |
  | Step 8r, fixed n_eff/K = 0.2 | 0/150 | 0.821 | 1.51 s (1.77, 1.75, 1.03) |
  | SAC 800k | 50/50 | 0.198 | 8.00 s |

- Paired by reset seed, Step 9 minus Step 8r, seeds 0/1/2:
  - Time alive: −0.16 s [−0.21, −0.11], **+0.83 s [+0.50, +1.21]**,
    **+0.35 s [+0.23, +0.45]**.
  - Speed MAE: +0.093 [+0.069, +0.115], −0.112 [−0.136, −0.088],
    +0.265 [+0.116, +0.413].
  - Completion: all within ±0.07. Against raw MPPI, each Step 9 seed is
    −0.28 to −0.30 (intervals exclude 0).
- Training: minimum sample improvement 0.058–0.163 in every run. Step 9's
  median ρ over iterations 5–24 was 0.050, 0.050 and 0.081. The rule sat at
  the 0.05 floor in 14, 16 and 11 of 25 iterations and returned a
  multiplier > 1 in 16–18 of 25. Dev: Step 9 seed 1 peaked at 2.9 s
  (iteration 19); Step 8r seed 1 completed 1/10 dev episodes at iteration 9,
  not kept later.
- Predictions: 1 **held**. 2 **held**. 3 **held** (seeds 1 and 2, intervals
  above 0; seed 0 worse). 4 **held, barely** (1/150). 5 **held** (Step 8r
  1.51 s vs Step 8's 1.45 s; 0 completions).
- Interpretation: MDGPS's step rule mostly wants larger KL steps than
  n_eff/K = 0.2 allows. Taking them buys 0.35–0.83 s of survival in two of
  three seeds, but speed tracking is mixed and completion stays near zero.
  The rule ends up pinned at the n_eff floor, so the trust region is now
  limited by label noise (n_eff ≈ 13 at K = 256), not by KL. All learned
  arms remain far below raw MPPI at the same cell (3.30 s, 15/50).
- Decision: **keep** the adaptive step as the default for further runs. The
  open question is the iteration budget (Step 10, running).


- Incident: Step 8r seed 0 crashed at iteration 7 (14:28). `torch.save`
  failed with "unexpected pos" and left a 0-byte `policy_007.pt`: the
  shared disk briefly ran out of space. No other run was affected. The
  crashed run is kept as `runs/2026-10-22_step8r_fixed_seed0_crashed_disk_full/`.
  Seed 0 was relaunched from iteration 0 at 14:31 with identical settings,
  after 32 GB was freed.
- Free space came from moving `~/Documents/percept-id-archive-20260921`
  (32 GB of sol1–sol8 robot recordings) to the Mac. All 404,271 files were
  verified (404,236 against its `sha256.jsonl`, 35 by hashing on both
  machines) before the user deleted it on the 5090.
- GPU environment: the full lock now installs separately at
  `~/mppi-gps-venvs/cuda` (`UV_PROJECT_ENVIRONMENT=... uv sync --locked`,
  8.6 GB). It has torch 2.10.0+cu128 on the RTX 5090 and warp 1.12.1 with
  mujoco_warp; 492 tests pass. Step 9/8r keep the CPU-only `.venv` they
  started with, so both arms share one environment.

### MuJoCo Warp parity and GPU E-step estimate (audit A1, robopil5090)

- Question: can MPPI-GPS E-steps run on the RTX 5090 through MuJoCo Warp
  without changing the labels, and how much faster would they be?
- Two bugs found first, fixed in `5969c44` (CUDA-only test
  `tests/test_warp_rollout.py`):
  1. The Warp path crashed on Walker. The model has no sensors and Warp
     cannot slice the zero-width sensor buffer.
  2. Each Warp call kept the previous call's solver warm start and time.
     Rollouts now start from zeros or the warm start passed in, as on CPU.
- Parity (`scripts/benchmarks/warp_parity.py`,
  `runs/2026-10-22_warp_parity_002/report.json`). 12 states from a SAC 800k
  episode, 256 policy-centred samples × 32 controls (the E-step shape).
  Warp (float32) vs CPU (float64), medians:
  - State difference: 4.7e-5 after one control, 3.1e-6 at the horizon end.
    Cost difference 3.6e-5 relative; Spearman 0.999999; fall agreement 100%.
  - E-step label (n_eff/K = 0.2): GPU vs CPU differs by 0.0002 (max 0.0016),
    against 0.13 between two independent CPU sample draws.
  - The CPU warm start has no effect on Walker here (1e-15).
  - Two identical Warp calls are **not** bit-identical: costs differ by up
    to 0.08 in 256 × 32 rollouts, likely order-dependent floating-point
    accumulation in GPU kernels. Negligible for labels, but GPU runs cannot
    be replayed bit-for-bit like CPU runs.
- Speed (physics steps per second; the CPU was shared with 24 training
  workers, consistent with the 5080's uncontended 0.465 s per K = 256,
  H = 32 decision):

  | Rollouts | CPU, 1 thread | GPU, 256 worlds | 1,024 | 4,096 | 16,384 |
  |---|---:|---:|---:|---:|---:|
  | Open loop, CUDA graph | 65k | 0.18M | 0.68M | 2.2M | 4.4M |
  | Closed-loop 100-control policy tail, no graph | — | 37k | 0.59M | — | 2.2M |

  The closed-loop tail takes ~2.8 s per 100 controls at any world count:
  it is launch-bound (no CUDA graph around the physics steps).
- E-step estimate (K = 256, H = 32, 100-control tail): Step 8 measured
  ~2.2 CPU-s per label. Batched 64 states at a time (16,384 worlds), the
  GPU needs 0.47 s for the open loop plus 2.96 s for the tail, ~0.054 s
  per label: ~40× one CPU thread, ~2× the whole 16-core CPU. Capturing the
  tail's physics in a CUDA graph should cut that to an estimated ~0.014 s
  per label (~8× the whole CPU). Batching works because (B) executes the
  policy, not the labels, so an iteration's labels can all be computed
  after its episodes.
- Decision: **keep** the fixed Warp path. Running MPPI-GPS on the GPU needs
  (1) a batched E-step over all of an iteration's states, (2) the tail
  under a CUDA graph, and (3) acceptance of non-bitwise GPU runs. Not done
  yet; the Step 9/8r runs stay on CPU.

### Batched GPU E-step: verification and measured speed (robopil5090)

- Question: does the batched GPU E-step (`src/gps/gpu_estep.py`, `89e765b`)
  give the CPU E-step's labels, and how much faster is it?
- Design: all of an iteration's labelled states at once, 64 or 128 states ×
  K = 256 samples as GPU worlds. Policy nominals, open-loop samples and the
  100-control closed-loop policy tail all run on MuJoCo Warp, with each
  control step's 4 physics steps captured as one CUDA graph and the policy in
  torch between launches. Costs, temperature and label follow `e_step`
  exactly. The CUDA-only test `tests/test_gpu_estep.py` checks labels, λ and
  costs against `e_step` with identical noise (passes on the 5090).
- Measured (`scripts/benchmarks/gpu_estep_check.py`,
  `runs/2026-10-22_gpu_estep_check_001/`): 1,190 states collected like
  training from Step 9 adaptive seed 1's `policy_014.pt`. The CPU labelled
  the first 48 with identical noise.
  - Speed: CPU 1.98 s per label (one thread, machine shared with 24 training
    workers). GPU 0.049 s per label at 16,384 worlds and **0.040 s at 32,768**:
    49× one thread. That is ~2–2.5× all 32 CPU threads, and it frees the
    CPU. A Step 8-sized iteration (~1,200 labels) takes ~48 s on the GPU,
    against ~340 s for a 7-worker CPU run.
  - Agreement: GPU vs CPU labels differ by median 0.072 (max 0.134); λ, plan
    and nominal costs by 6–8%. The same GPU computation at batch 64 vs 128
    differs by up to 0.18.
- Why (`scripts/benchmarks/estep_chaos_check.py`,
  `runs/2026-10-22_estep_chaos_001/`, 24 states, identical noise). Label
  differences, median (max):

  | Comparison | Label difference |
  |---|---:|
  | CPU, root x-velocity nudged by 1e-6 | 0.068 (0.113) |
  | Two identical GPU runs | 0.063 (0.126) |
  | GPU vs CPU | 0.077 (0.112) |
  | Two noise seeds (sampling noise) | 0.133 (0.264) |

  No contact or constraint overflow: buffers are 48 contacts and 64
  constraint rows per world, and no warnings were logged.
- Interpretation: with a 100-control closed-loop tail, the E-step label is
  chaotic. A 1e-6 change to the state moves it by about half the sampling
  noise, on CPU alone, so the GPU–CPU gap is this physical sensitivity, not
  GPU error. Without the tail, labels agree to 0.0002 (Warp parity entry).
  This also matters for Steps 6–9: raising K reduces sampling noise but not
  this chaotic component of every tail-augmented label.
- Decision: **keep.** The GPU E-step is statistically equivalent to the CPU
  one and ~7× faster per run than the 7-worker CPU setup. It is not yet wired
  into `train_walker_mdgps.py`; doing so means collecting episodes with the
  policy on CPU, labelling all states on the GPU, then fitting.

### GPU E-step in training: one-iteration check (robopil5090)

- `--cfg.gpu-estep` (`2b2e0aa`): workers only collect, using the same random
  draws as the CPU path. All of an iteration's states are then labelled by
  `WalkerGPUEStep` with the CPU path's per-episode noise streams.
  `tests/test_train_collect.py` checks that collect-only mode visits the CPU
  path's states. The CPU path is unchanged: Step 5's first iteration
  reproduces (21.0 steps, 107 labels, λ to 8 digits).
- Check: one iteration in each mode with K = 256, H = 32, the 100-control
  tail, n_eff/K = 0.2, seed 0, 10 × 400 controls, 8 collection workers,
  starting from Step 9 adaptive seed 1's `policy_014.pt`
  (`runs/2026-10-22_gpu_iter_check_{gpu,cpu}/`):

  | | GPU E-step | CPU E-step |
  |---|---:|---:|
  | Labelled states (identical in both) | 1,093 | 1,093 |
  | Iteration time | **48 s** | 417 s |
  | of which collect / label / fit | 0.4 / 44 / 1.8 s | 413 (incl. labels) / — / 2.0 s |
  | Fit loss | 0.00016 | 0.00015 |
  | Probe predicted / actual improvement | 15.5 / 18.3 | 18.2 / 19.4 |

- Labels: GPU vs CPU median 0.068 (p95 0.13, max 0.20), the same as the
  CPU's own 1e-6-nudge sensitivity (0.068). Median λ differs by 6%, nominal
  and plan costs by 5–7%; n_eff/K is 0.200 in both.
- Speed: 8.7× an 8-worker CPU iteration on the shared machine. The CPU path
  spent ~3.0 CPU-s per label; even all 32 threads (~24 effective) would need
  ~140 s for this iteration, so the GPU is ~3× the whole CPU and leaves it
  free.
- Decision: **keep.** New MPPI-GPS runs on robopil5090 should use
  `--cfg.gpu-estep` with the CUDA environment (`~/mppi-gps-venvs/cuda`).
  GPU runs are not bitwise reproducible. Step 9/8r stay on the CPU path they
  started on.

### Step 10 — 75 iterations of K = 256 + tail on the GPU (predictions written before running)

- Question: does MPPI-GPS keep improving past 25 iterations, and does it
  eventually complete episodes? In Step 8 both learning seeds were still
  lengthening their episodes at iteration 24.
- Settings: Step 8's configuration (fixed n_eff/K = 0.2, K = 256, H = 32,
  100-control tail, 10 × 400 controls per iteration, label every 2nd state,
  exec noise 0.1, replay 2, 2,000 Adam steps, probe probability 0.02), with
  **75 iterations** and the GPU E-step (`--cfg.gpu-estep
  --cfg.gpu-states-per-batch 128`, CUDA environment). Training seeds 0–2,
  4 collection workers each, all three sharing the RTX 5090 alongside the
  CPU-path Step 9/8r runs.
- Command: for seed in 0 1 2: `~/mppi-gps-venvs/cuda/bin/python -m
  scripts.training.train_walker_mdgps --out-dir
  runs/2026-10-22_step10_gpu75_seed$seed --cfg.labeler mdgps --cfg.seed
  $seed --cfg.K 256 --cfg.H 32 --cfg.tail-steps 100 --cfg.iterations 75
  --cfg.episodes-per-iter 10 --cfg.steps 400 --cfg.gpu-estep
  --cfg.gpu-states-per-batch 128 --cfg.workers 4`.
- Evaluation: dev resets every 5 iterations (monitoring). At the end,
  held-out resets 70000–70049 on robopil5090 for the iteration-74 and
  iteration-24 policies, alongside the same-machine arms (Step 8r, Step 9,
  raw MPPI K = 32 H = 32, SAC 800k). GPU runs are not bitwise reproducible,
  so seeds are not paired with the CPU runs.
- Predictions:
  1. Fact 1 holds in every E-step.
  2. Dev time alive at iteration 74 exceeds iteration 24's in at least 2 of
     3 seeds.
  3. At least one seed completes ≥ 1/10 dev episodes at some evaluation from
     iteration 50 on (less certain).
  4. Held-out pooled time alive of the iteration-74 policies exceeds 1.45 s
     (Step 8's 25-iteration result), and pooled completion exceeds 0/150
     (least certain).
  5. No directional prediction against raw MPPI at K = 32, H = 32.
- Status: complete. Launched 15:52 on 2026-10-22 at `8ed3a71`; 2.1–2.5 h
  per seed with three sharing the GPU (81–143 s labelling per iteration for
  1,400–1,900 labels). No errors.
- Held-out result (`runs/2026-10-22_step10_heldout_001/report.json`,
  robopil5090, same CPU-only evaluation environment as Step 9; shared arms
  reproduce exactly):

  | Arm | Completed | Wilson 95% | Speed MAE | Time alive (seeds 0/1/2) |
  |---|---:|---|---:|---:|
  | **(B), K = 256 + tail, 75 iterations (GPU)** | **81/150 (54%)** | 46–62% | 0.393 | **5.86 s** (5.98, 6.86, 4.74) |
  | Same runs, iteration 24 | 2/150 | 0.4–4.7% | 0.967 | 1.49 s |
  | Step 8r, 25 iterations (CPU) | 0/150 | 0–2.5% | 0.821 | 1.51 s |
  | Step 9 adaptive, 25 iterations (CPU) | 1/150 | 0.1–3.7% | 0.903 | 1.85 s |
  | Raw MPPI, K = 32, H = 32 | 15/50 (30%) | 19–44% | 0.392 | 3.30 s |
  | SAC 800k | 50/50 | 93–100% | 0.198 | 8.00 s |

  Per seed at iteration 74: 34/50 (MAE 0.36), 35/50 (0.28), 12/50 (0.53).
- Paired by reset seed, seeds 0/1/2:
  - **Against raw MPPI**, completion: **+0.38 [+0.19, +0.53]**,
    **+0.40 [+0.20, +0.56]**, −0.06 [−0.23, +0.11]. Speed MAE: −0.027
    [−0.079, +0.022], −0.110 [−0.152, −0.069], +0.140 [+0.080, +0.206].
  - Against the same runs at iteration 24: completion +0.64, +0.70, +0.24 and
    time alive +4.3, +5.4, +3.4 s (all intervals exclude 0).
  - Against SAC 800k: completion −0.32, −0.30, −0.76 (all exclude 0).
- Training curves (dev, 10 resets): seed 1 climbed steadily from iteration
  29 to 90% completion at iteration 74 (MAE 0.26), and the last three
  iterations completed every training episode. Seed 0 jumped at iterations
  54–59 and oscillated at 30–50%. Seed 2 started rising only at 69–74 and
  was still rising. Minimum sample improvement 0.023–0.057.
- Video (`runs/2026-10-22_step10_videos_001/`, reset 70000): seed 1 walks the
  full 8 s, 11.5 m at 1.43 m/s, with an alternating gait and a forward-leaning
  torso. Seed 2 walks the same way for 5.4 s and then falls forward.
- Predictions: 1 **held**. 2 **held** in all three seeds. 3 **held** (seed 1
  from iteration 49). 4 **held** (5.86 s, 81/150). 5: two of three seeds
  beat raw MPPI on completion.
- Interpretation: the 25-iteration budget was what held Steps 5–9 at ~1.5 s.
  With 75 iterations, MPPI-GPS (K = 256, policy tail, fixed trust region)
  produces policies that, run alone, complete more held-out episodes than
  the H = 32 planner (68–70% vs 30% in two seeds). Raw MPPI's completion at
  H = 32 does not improve with K (grid: 5–7/10 for K = 32–256), so a larger
  sample budget alone would not close the gap. This is the `general.md`
  claim: the policy carries information beyond MPPI's horizon. Caveats:
  - three seeds, one lagging;
  - still well below SAC 800k;
  - the E-step spends far more simulation per decision than raw MPPI, so
    this is not a sample-efficiency result;
  - no equal-budget control yet: DAgger and the adaptive step were only run
    for 25 iterations.
- Decision: **keep.** Before any claim, the next control is DAgger at the
  same 75-iteration budget.

### Step 11 — DAgger at the 75-iteration budget (predictions written before running)

- Question: does DAgger also reach Step 10's level when given the same 75
  iterations, or is the gain specific to MPPI-GPS?
- Settings: Step 5's DAgger (`--cfg.labeler dagger`): a raw MPPI expert at
  K = 32, H = 32 (frozen λ = 3, its own warm start, two optimizer passes at
  each episode's first decision) labels every visited state; iteration 0
  executes the expert, later iterations the policy + N(0, 0.1²). Same
  collection and fit as Step 10: 10 × 400 controls, replay 2, 2,000 Adam
  steps, 75 iterations, training seeds 0–2. CPU path in the CPU-only
  environment on robopil5090, 6 workers per seed.
- Command: for seed in 0 1 2: `.venv/bin/python -m
  scripts.training.train_walker_mdgps --out-dir
  runs/2026-10-22_step11_dagger75_seed$seed --cfg.labeler dagger --cfg.seed
  $seed --cfg.K 32 --cfg.H 32 --cfg.iterations 75 --cfg.episodes-per-iter 10
  --cfg.steps 400 --cfg.workers 6`.
- Confound, stated in advance: this matches the iteration and data budget,
  not the teacher. MPPI-GPS's E-step uses K = 256 plus the policy's
  100-control tail; this expert uses K = 32 and no tail. Raw MPPI's H = 32
  completion barely depends on K (grid: 5–7/10 for K = 32–256), so the tail
  is the main difference. A DAgger loss means "not just more iterations",
  not "the KL-bounded policy-centred step is what matters"; separating those
  needs an expert with the student's own tail.
- Comparison: held-out resets 70000–70049 on robopil5090 in the CPU-only
  evaluation environment. DAgger iterations 74 and 24 against Step 10's
  iteration-74 MPPI-GPS policies (per-seed pairs by reset seed), raw MPPI
  (K = 32, H = 32) and SAC 800k. Primary: completion. Secondary: time alive.
- Predictions:
  1. DAgger's pooled held-out completion at iteration 74 is at most its
     expert's (raw MPPI, 15/50 = 30% on this machine).
  2. Step 10's MPPI-GPS beats DAgger on held-out completion pooled, and in
     at least 2 of 3 seed pairs with the paired interval above 0.
  3. DAgger's pooled held-out time alive at iteration 74 is below raw
     MPPI's 3.30 s.
- Status: complete. Launched 21:27 on 2026-10-22 at `be130da`; 75 iterations
  took ~35 min per seed (18–30 s per iteration). No errors.
- Held-out result (`runs/2026-10-22_step11_heldout_001/report.json`,
  robopil5090, CPU-only evaluation environment; shared arms reproduce Step
  10's numbers exactly):

  | Arm | Completed | Speed MAE | Time alive (seeds 0/1/2) |
  |---|---:|---:|---:|
  | **MPPI-GPS, 75 iterations (Step 10, seeds 0–2)** | **81/150** | 0.393 | **5.86 s** (5.98, 6.86, 4.74) |
  | DAgger, 75 iterations | 0/150 | 0.662 | 1.13 s (1.07, 1.19, 1.13) |
  | DAgger, iteration 24 | 0/150 | 0.753 | 1.25 s (1.36, 1.16, 1.22) |
  | Raw MPPI, K = 32, H = 32 (DAgger's expert) | 15/50 | 0.392 | 3.30 s |
  | SAC 800k | 50/50 | 0.198 | 8.00 s |

- Paired by reset seed, seeds 0/1/2:
  - **MPPI-GPS minus DAgger at 75 iterations**, completion: **+0.68
    [+0.52, +0.79], +0.70 [+0.55, +0.81], +0.24 [+0.12, +0.37]**. Time alive
    +4.9, +5.7, +3.6 s; speed MAE −0.28, −0.39, −0.14 (all intervals exclude 0).
  - DAgger iteration 74 minus iteration 24: completion +0.00 each; time alive
    −0.29 s [−0.38, −0.19], +0.03 s [−0.11, +0.14], −0.09 s [−0.21, +0.01];
    speed MAE −0.13, −0.07, −0.08.
  - DAgger against its own expert: −0.30 each, intervals excluding 0.
- Training: DAgger's fit loss stayed at 0.10–0.13 throughout. MPPI-GPS's
  was 0.0005–0.0008, two orders of magnitude lower. Its labels come from the
  frozen-λ expert at n_eff/K ≈ 0.13–0.16, about 5 effective samples of 32,
  the label noise the Step 2 audit measured. Its collection episodes stayed
  at 1.0–1.4 s.
- Predictions: 1 **held** (0/150 ≤ 30%). 2 **held** (pooled and all three
  pairs). 3 **held** (1.13 s < 3.30 s).
- Interpretation: given the same 75 iterations, DAgger does not improve at
  all, so Step 10's gain is not just a longer training budget. Its student
  never approaches its expert (0% vs 30%) and stays at the ~1.1–1.3 s
  plateau seen since Step 5. As preregistered, this does not isolate which
  part of MPPI-GPS matters. Against this expert, MPPI-GPS's E-step differs
  in the n_eff-targeted temperature (cleaner labels), K = 256, the
  policy-centred KL-bounded step, and the policy's 100-control tail.
- Decision: **keep.** MPPI-GPS at 75 iterations beats both the equal-budget
  DAgger control and the planner it is derived from. Next ablation: an
  expert with the student's own tail (DAgger + tail), or MPPI-GPS without
  the tail at 75 iterations, to isolate the tail.

### Step 12 — three more seeds of Step 10 (predictions written before running)

- Question: does Step 10's result hold beyond its first three seeds, one of
  which lagged?
- Settings: Step 10's exact configuration and command with training seeds
  3, 4 and 5 (`runs/2026-10-22_step10_gpu75_seed{3,4,5}`): K = 256, H = 32,
  100-control tail, fixed n_eff/K = 0.2, 75 iterations, GPU E-step (CUDA
  environment), 4 collection workers each. They run alongside Step 11's
  CPU DAgger runs.
- Comparison: the same held-out evaluation as Step 11, which runs once both
  finish. Seeds 3–5 enter as their own arm (`step10b_it74`); six-seed
  results pool them with Step 10's seeds 0–2.
- Predictions:
  1. Fact 1 holds in every E-step.
  2. Seeds 3–5 pooled held-out completion at iteration 74 exceeds raw
     MPPI's 15/50 (30%).
  3. Across all six seeds, at least 3 beat raw MPPI on completion with the
     paired interval above 0.
  4. The six-seed pooled completion has a Wilson 95% lower bound above 30%.
- Exploratory arms, **not preregistered**. Added on 2026-10-23 before the
  held-out run, after Steps 10–11, to compare with SAC at matched
  environment steps.
  - Arms: SAC checkpoints at 150k, 175k and 200k training controls from the
    S2b ladder (`runs/2026-10-11_s2_sac_ladder_seed0_001/checkpoints/step_0150000.pt`,
    `step_0175000.pt`, `step_0200000.pt`; SHA-256 `abe4e6f0…`, `093c39dc…`,
    `bacb711e…`, matching the ladder's `checkpoints.jsonl`), copied from the
    Mac to robopil5090.
  - Why these: Step 10's runs executed 133k–197k training controls each, and
    simulated 2.3–3.3 billion more in E-steps.
  - Every MPPI-GPS seed is paired against each by reset seed
    (`walker_step5_compare.py --references`). No predictions.
  - S2b classified the 200k policy as surviving while nearly stationary
    (29/30, speed MAE 1.68 m/s), so completion is reported beside speed MAE.
- Status: complete. Seeds 3–5 ran 21:35 on 2026-10-22 to 00:00–00:32 on
  2026-10-23 (GPU E-step, 4 collection workers each, alongside Step 11);
  75 iterations each, no errors. The held-out comparison ran at `2cc03ac` in
  111 s, started automatically when seed 4 finished
  (`runs/2026-10-22_step12_heldout_001/report.json`, robopil5090, CPU-only
  evaluation environment). Step 10's seeds 0–2, raw MPPI and SAC 800k
  reproduce their earlier numbers exactly.
- Held-out result:

  | Arm | Completed | Wilson 95% | Speed MAE | Time alive (per seed) |
  |---|---:|---|---:|---:|
  | **MPPI-GPS, 75 iterations, seeds 3–5** | **123/150 (82%)** | 75–87% | 0.258 | **7.41 s** (6.38, 7.89, 7.97) |
  | MPPI-GPS, 75 iterations, seeds 0–2 (Step 10) | 81/150 (54%) | 46–62% | 0.393 | 5.86 s (5.98, 6.86, 4.74) |
  | **Six seeds pooled** | **204/300 (68%)** | 62.5–73.0% | 0.326 | 6.63 s |
  | Seeds 3–5 at iteration 24 | 0/150 | 0–2.5% | 0.644 | 1.82 s |
  | DAgger, 75 iterations (Step 11) | 0/150 | 0–2.5% | 0.662 | 1.13 s |
  | Raw MPPI, K = 32, H = 32 | 15/50 (30%) | 19–44% | 0.392 | 3.30 s |
  | SAC 800k | 50/50 | 93–100% | 0.198 | 8.00 s |

  Per seed at iteration 74: 25/50 (MAE 0.33), 49/50 (0.23), 49/50 (0.22).
- Paired by reset seed, seeds 3/4/5:
  - **Against raw MPPI**, completion: +0.20 [+0.01, +0.37],
    **+0.68 [+0.52, +0.79]**, **+0.68 [+0.52, +0.79]**. Speed MAE −0.06,
    −0.16, −0.17 (all intervals exclude 0).
  - Against SAC 800k, completion: −0.50 [−0.63, −0.35],
    −0.02 [−0.10, +0.05], −0.02 [−0.10, +0.05]. Speed MAE +0.13, +0.03,
    +0.02 (all intervals exclude 0).
  - Against the same runs at iteration 24: completion +0.50, +0.98, +0.98;
    time alive +4.9, +5.8, +6.1 s (all intervals exclude 0).
  - Against DAgger seeds 0/1/2 (exploratory; pairs different training
    seeds): completion +0.50, +0.98, +0.98.
- Exploratory SAC arms (not preregistered), one checkpoint each:

  | SAC checkpoint | Completed | Speed MAE | Time alive |
  |---|---:|---:|---:|
  | 150k | 15/50 | 1.298 | 6.03 s |
  | 175k | 16/50 | 1.374 | 6.25 s |
  | 200k | 47/50 | 1.699 | 7.59 s |

  - The six MPPI-GPS runs executed 163k, 197k, 133k, 163k, 232k and 220k
    training controls (seeds 0–5). They also simulated 2.3–3.9 billion
    controls in E-steps, ~3,000–5,000× SAC 800k's total.
  - At 150k–200k steps SAC does not walk yet; the 200k policy survives by
    standing nearly still.
  - Every MPPI-GPS seed tracks speed better than all three checkpoints (MAE
    differences −0.77 to −1.48 m/s, all intervals exclude 0).
  - On completion, seeds 0, 1, 4 and 5 beat SAC 150k and 175k (intervals
    above 0); seeds 2 and 3 do not differ. Against SAC 200k, seeds 4 and 5
    tie (+0.04 [−0.05, +0.14]) and the other four are lower.
- Training: take-off (the first iteration with at least 5/10 training
  episodes reaching 400 controls) came at iterations 70, 31 and 46 for
  seeds 3–5, against 57, 40 and 73 for seeds 0–2. Held-out completion
  follows that order. Dev at iteration 74: 5/10, 10/10, 10/10. Fit loss
  0.0007–0.0009.
- Theory checks:
  - Minimum sample improvement was 0.041–0.088. This check holds by
    construction: weights that decrease with cost always put the weighted
    mean cost below the plain mean (Chebyshev's sum inequality), so
    Prediction 1 here and in Steps 3–10 could not fail.
  - The informative signal is on the plans. In seeds 4 and 5, median
    predicted and actual improvement on probe states fell from 9.5–15
    (iterations 0–24) to 1.2–1.5 (50–74). Predicted improvement was negative
    in 6–7 of the last 25 iterations for seeds 3–5: there, the averaged plan
    was worse than the policy's own plan.
  - Early on, actual improvement exceeded the prediction in seeds 0, 1 and 3
    (medians 11.5–12.9 vs 3.5–8.8).
- Predictions: 1 **held**, but vacuously (above). 2 **held** (82%). 3
  **held** (5 of 6 seeds: 0, 1, 3, 4, 5; seed 3's lower bound is +0.01). 4
  **held** (lower bound 62.5%).
- Interpretation:
  - Step 10's result holds with three more seeds: 68% pooled completion
    (62.5–73%). Seeds 4 and 5 are within 2 points of SAC 800k on completion,
    with 0.02–0.03 m/s more speed error.
  - The spread across seeds tracks take-off time (iteration 31 to 73), so at
    a fixed 75-iteration budget final performance is mostly set by how long a
    seed takes to start walking.
  - The fixed n_eff/K = 0.2 step gains more than predicted early and little,
    sometimes less than nothing, late: the pattern MDGPS's step rule exists for.
  - At matched environment steps SAC does not walk yet, but MPPI-GPS uses
    thousands of times more simulation.
- Decision: **keep.** MPPI-GPS at 75 iterations beats raw MPPI at K = 32,
  H = 32 in 5 of 6 seeds and DAgger in every seed, and two seeds approach
  SAC 800k. The next question is take-off time.

### Step 13 — three checks that need no training (predictions written before running)

- Question: do three checks from the 2026-10-23 theory review hold up on the
  existing runs?
  - (a) Does averaging the last iterates help? Stochastic mirror descent's
    guarantee is for the averaged iterate, not the last one.
  - (b) Does an H = 48 planner beat the MPPI-GPS policies? That is the regime
    `general.md` assumes, where MPC "can do the task".
  - (c) How often is the E-step's averaged plan worse than the policy's own
    plan, and is that systematic or sampling noise?
- Fixed: Step 10/12's six training runs (seeds 0–5); no training. Held-out
  resets 70000–70049 on robopil5090. 13a and 13b use Step 12's CPU-only
  evaluation environment; 13c uses the CUDA environment.

**13a — averaged iterates.**
- Arms per seed: the last iterate (`policy_074.pt`, as in Step 12) and the
  action average of iterations 65–74, which acts with the mean of the ten
  policies' actions (the mean of the mixture policy). Exploratory: the
  weight average of the same ten checkpoints, as one network.
- Comparison: paired by reset seed against the last iterate, per seed and
  pooled. Primary: completion. Secondary: speed MAE and time alive. Run with
  13b's K = 32 cell in one `walker_step5_compare.py` call.
- The averaging guarantee holds for convex problems; here it is a heuristic.
- Predictions:
  1. For the four seeds that took off by iteration 57 (0, 1, 4, 5), the
     action average completes at least as many episodes as the last iterate
     in at least 3 of 4. Pooled over those four, its completion is higher and
     its speed MAE lower.
  2. For the two late seeds (2 and 3, take-off at 73 and 70), the action
     average completes fewer episodes than the last iterate, because it mixes
     in policies from before take-off.
  3. Weight average: no prediction.

**13b — raw MPPI at H = 48.**
- Arms: raw MPPI at K = 32, H = 48 and at K = 256, H = 48 (frozen λ = 3,
  own warm start, planner seeds 95000+, 1,000 controls). Compared with the
  six MPPI-GPS last iterates and SAC 800k, paired by reset seed. Step 4's
  dev grid (robopil, 10 episodes each): 10/10 at both cells, speed MAE 0.358
  and 0.139.
- Predictions:
  1. Both H = 48 cells complete at least 80% of held-out episodes.
  2. Both beat the six MPPI-GPS policies' pooled 68% on completion. Against
     the K = 32, H = 48 cell, at most seeds 4 and 5 are not below it (paired
     interval).
  3. K = 256, H = 48 has a lower speed MAE than SAC 800k (0.198) and than
     every MPPI-GPS seed.
- Seconds per decision are recorded for both cells.

**13c — plan versus policy on E-step labels.**
- Design: for each seed's iteration-24 and iteration-74 checkpoints, collect
  states as training does (10 episodes × 400 controls, policy + N(0, 0.1²),
  every 2nd state, collection seed 123000). Label them twice with the batched
  GPU E-step at the training settings (K = 256, H = 32, 100-control tail,
  n_eff/K = 0.2), using independent noise draws.
- Per state, gain = policy plan cost − improved plan cost on the E-step
  objective: the predicted improvement, at every state instead of 2% probes.
- Report: the fraction of states with gain < 0 (the improved plan is worse
  than the policy's own), the median gain, sign agreement between the two
  draws, and label distance from the policy's action on states with gain < 0
  vs > 0.
- Predictions:
  1. At iteration 74, at least 25% of states have gain < 0 in every seed.
  2. That fraction is higher at iteration 74 than at iteration 24 in every
     seed.
  3. At iteration 74 the sign of the gain repeats across the two draws on
     fewer than 75% of states (mostly sampling noise); at iteration 24, on
     more than 75%.

- What each result changes:
  - If (a) helps, later steps report averaged iterates beside last ones.
  - If (b)'s planners beat MPPI-GPS, they join the headline comparison, and
    the claim becomes amortization (a fast policy) rather than "beats the
    planner".
  - If (c) shows ≥ 25% worse plans at convergence, the next training change
    is gating labels on plan cost or shrinking the step late.
- Status: complete. Ran 00:48–01:30 on 2026-10-23 at `763ea5b` on
  robopil5090. 13a ran with 13b's K = 32 cell (862 s); 13b's K = 256 cell
  took 2,513 s; 13c took ~22 min on the GPU.

**13a result** (`runs/2026-10-23_step13ab_k32_h48_001/report.json`):

| Policy | Completed, seeds 0–5 | Pooled | Speed MAE | Time alive |
|---|---|---:|---:|---:|
| Last iterate (iteration 74) | 34, 35, 12, 25, 49, 49 | 204/300 | 0.326 | 6.63 s |
| Action average, iterations 65–74 | 24, 38, 0, 8, 50, 49 | 169/300 | 0.340 | 6.04 s |
| Weight average, iterations 65–74 (exploratory) | 25, 36, 1, 11, 50, 50 | 173/300 | 0.339 | 6.18 s |

- Action average minus last iterate, completion (seeds 0–5): −0.20
  [−0.38, −0.00], +0.06 [−0.12, +0.23], −0.24 [−0.37, −0.12], −0.34
  [−0.51, −0.14], +0.02 [−0.05, +0.10], +0.00 [−0.09, +0.09].
- Predictions: 1 **wrong**. The average matched or beat the last iterate in
  3 of the 4 early seeds (1, 4, 5), but pooled over them it completed fewer
  episodes (161 vs 167 of 200) with a higher speed MAE (0.282 vs 0.273);
  seed 0 lost 0.20. 2 **held** (seeds 2 and 3 lost 0.24 and 0.34).
- Interpretation: the last ten iterates are not noise around a fixed point.
  Averaging is neutral where the policy has stopped changing (seeds 4 and 5)
  and costs completion wherever it was still improving. The
  averaged-iterate guarantee assumes convexity and does not transfer here.

**13b result** (`runs/2026-10-23_step13ab_k32_h48_001`,
`runs/2026-10-23_step13b_k256_h48_001`):

| Controller | Completed | Wilson 95% | Speed MAE | Time alive | Time per decision |
|---|---:|---|---:|---:|---:|
| Raw MPPI, K = 32, H = 32 (Step 12) | 15/50 | 19–44% | 0.392 | 3.30 s | — |
| Raw MPPI, K = 32, H = 48 | 39/50 (78%) | 65–87% | 0.387 | 6.94 s | 0.135 s |
| Raw MPPI, K = 256, H = 48 | 44/50 (88%) | 76–94% | **0.162** | 7.47 s | 0.887 s |
| MPPI-GPS policies, six seeds (Step 12) | 204/300 (68%) | 62.5–73% | 0.326 | 6.63 s | ~40 µs |
| SAC 800k | 50/50 | 93–100% | 0.198 | 8.00 s | ~40 µs |

- MPPI-GPS minus raw MPPI, completion per seed (0–5):
  - Against K = 32, H = 48: −0.10 [−0.27, +0.08], −0.08 [−0.24, +0.08],
    −0.54 [−0.67, −0.35], −0.28 [−0.45, −0.09], **+0.20 [+0.07, +0.33]**,
    **+0.20 [+0.07, +0.33]**. Speed MAE −0.02, −0.10, +0.15, −0.06, −0.16,
    −0.17 (all but seed 0's intervals exclude 0).
  - Against K = 256, H = 48: −0.20, −0.18, −0.64, −0.38 (intervals exclude
    0), +0.10 [−0.01, +0.22], +0.10 [−0.01, +0.22]. Speed MAE +0.06 to
    +0.37 (all intervals exclude 0).
- SAC 800k minus raw MPPI at K = 256, H = 48: completion +0.12
  [+0.02, +0.24], speed MAE +0.04 [+0.02, +0.05]. The planner tracks speed
  better than SAC.
- Predictions: 1 **wrong** (K = 32, H = 48 completes 78%; K = 256, 88%). 2
  **wrong** in part: both cells beat the pooled 68%, but against K = 32,
  H = 48 seeds 0 and 1 are not below it, and seeds 4 and 5 are above it. 3
  **held** (0.162 < 0.198, and below every MPPI-GPS seed).
- Interpretation: at H = 48 the planner does the task, as `general.md`
  assumes, and at K = 256 it tracks speed better than SAC 800k, at
  0.14–0.89 s per decision against a policy's ~40 µs. The best two MPPI-GPS
  seeds match or beat both H = 48 planners on completion but not K = 256's
  speed tracking; the other four lose to both. "Beats the planner it learns
  from" holds only against H = 32 planners.

**13c result** (`runs/2026-10-23_step13c_label_check_001/report.json`;
751–2,000 states per checkpoint; fractions from the first draw, the second
within 0.03):

| Seed | Gain < 0, it 24 | Gain < 0, it 74 | Sign agreement, it 24 / 74 | Median gain, it 24 / 74 |
|---|---:|---:|---:|---:|
| 0 | 38% | 42% | 0.70 / 0.65 | 1.65 / 2.65 |
| 1 | 44% | 43% | 0.69 / 0.72 | 0.97 / 1.27 |
| 2 | 23% | 38% | 0.74 / 0.66 | 23.4 / 5.50 |
| 3 | 38% | 42% | 0.75 / 0.70 | 1.87 / 2.33 |
| 4 | 34% | 46% | 0.76 / 0.69 | 5.05 / 0.46 |
| 5 | 33% | 53% | 0.71 / 0.71 | 5.13 / −0.31 |

- Label distance from the policy's action: median 0.096–0.120, the same
  whether the gain is negative or positive. Weights that carry no
  information about the noise (n_eff/K = 0.2; per-dimension noise std
  0.30, checked) give 0.094. So a label is mostly sampling noise: its
  systematic part is roughly 0.03 (iteration 74) to 0.07 (iteration 24) in
  action norm, against 0.094 of noise, if the two are independent.
- Predictions: 1 **held** (38–53% at iteration 74). 2 **wrong**: higher at
  iteration 74 in 5 of 6 seeds, not seed 1 (44% → 43%). 3 **wrong** in part:
  at iteration 74 the sign repeats on 65–72% of states (held), but at
  iteration 24 only seed 4 exceeds 75%.
- Interpretation: at every stage, a third to a half of labels point to plans
  worse than the policy's own on the E-step's own objective, and which
  states those are changes between noise draws. Per label, the E-step's
  signal is weaker than its noise; learning works because the regression
  averages over thousands of states, which fits the slow, seed-dependent
  take-off. This supports two ideas in the notes: check the plan you would
  execute (`Trust coupling methods.md`), and let the planner's task
  performance, not a fixed KL step, decide how far a label moves.

- Decision: keep Steps 10/12's method as the working base. Report both
  H = 48 planners beside SAC from now on. Drop averaged iterates. Next,
  formalize the notes' trust coupling (`docs/theory.md`) and add its pieces
  to the working method one at a time.

### Step 14 — the predictive check (predictions written before running)

- Question: does keeping a label only when its improved plan costs no more
  than the policy's own plan make MPPI-GPS learn faster or better? This is
  `docs/theory.md` Proposition 4, the predictive filter of
  `Trust coupling methods.md` with the policy's own plan as the baseline.
- One change from the working method (Steps 10/12):
  `--cfg.predictive-check`.
  - At each labelled state, compare two costs on the E-step objective (32
    controls plus the 100-control policy tail): the improved mean plan's
    and the policy's own plan's.
  - If the improved plan costs more, the label becomes the policy's own
    action.
  - Both costs are already computed, so there is no extra simulation.
  - Every iteration logs `plan_worse_fraction`, the replaced share.
- Fixed, as in Steps 10/12:
  - K = 256, H = 32, 100-control tail, n_eff/K = 0.2;
  - 10 × 400 controls per iteration, every 2nd state labelled, exec noise
    0.1;
  - replay 2, 2,000 Adam steps, probe probability 0.02, 75 iterations;
  - GPU E-step (128 states per batch), 4 collection workers;
  - training seeds 0–5, all six sharing robopil5090's GPU.
- Command: for seed in 0–5: `~/mppi-gps-venvs/cuda/bin/python -m
  scripts.training.train_walker_mdgps --out-dir
  runs/2026-10-23_step14_check_seed$seed --cfg.labeler mdgps --cfg.seed
  $seed --cfg.K 256 --cfg.H 32 --cfg.tail-steps 100 --cfg.iterations 75
  --cfg.episodes-per-iter 10 --cfg.steps 400 --cfg.gpu-estep
  --cfg.gpu-states-per-batch 128 --cfg.workers 4 --cfg.predictive-check`.
- Comparison:
  - Iteration-74 policies on held-out resets 70000–70049, on robopil5090 in
    the CPU-only evaluation environment.
  - Against Steps 10/12 seed by seed, paired by reset seed. GPU training is
    not bitwise reproducible, so seed i of each arm is a separate draw.
  - Also against raw MPPI at H = 48 (K = 32 and 256), SAC 800k and DAgger at
    75 iterations.
  - Primary: six-seed pooled completion. Secondary: speed MAE, time alive,
    and take-off (the first iteration with at least 5/10 training episodes
    reaching 400 controls).
- Predictions:
  1. The median replaced fraction over iterations is 30–50% in every seed.
  2. Take-off comes earlier than the base's in at least 4 of 6 seeds (base:
     57, 40, 73, 70, 31, 46).
  3. Six-seed pooled completion exceeds the base's 204/300.
  4. Pooled speed MAE is no worse than the base's 0.326.
- Decision: keep the check if pooled completion beats the base's and at
  least 4 of 6 seeds improve on their base seed. Step 15 then adds the
  planner fallback to it; otherwise Step 15 builds on the base.
- **Deviation** (recorded 2026-10-23, before resuming):
  - **What stopped the runs.** robopil5090 restarted at 03:46. The previous
    boot's journal ends at 03:45:00 with no kernel error, out-of-memory kill
    or GPU fault, which fits a hard reset or power loss. All six runs
    stopped after iterations 31, 35, 47, 39, 34 and 28 (seeds 0–5), each
    with that iteration's policy saved.
  - **Why it went unnoticed.** The GPU sat idle until 12:02. The watcher's
    process count matched its own command line, so it never saw the runs
    die; it has been replaced.
  - **How they resume.** From those policies with `--cfg.resume`. The next
    iteration index, and so every per-iteration collection, planner and
    noise seed, is the one an unbroken run would use.
    `tests/test_resume.py` checks that a resumed iteration collects exactly
    what an unbroken one does.
  - **What differs.** The first resumed iteration fits on its own labels
    only, because the previous iteration's replay was in memory. The fit's
    minibatch order also restarts. GPU runs are not bitwise reproducible
    anyway, so this is treated as a statistically equivalent continuation.
- Status: complete. Training resumed at 12:20 and finished at 15:00 on
  2026-10-23. The held-out comparison ran in 284 s
  (`runs/2026-10-23_step14_heldout_001/report.json`, robopil5090, CPU-only
  evaluation environment). The base's six seeds, SAC 800k and DAgger
  reproduce their earlier numbers exactly.
- Held-out result:

  | Arm | Completed | Wilson 95% | Speed MAE | Time alive | Per seed (0–5) |
  |---|---:|---|---:|---:|---|
  | Predictive check (Step 14) | 55/300 (18%) | 14–23% | 0.473 | 3.81 s | 0, 15, 0, 3, 0, 37 |
  | Working method (Steps 10/12) | 204/300 (68%) | 62.5–73% | 0.326 | 6.63 s | 34, 35, 12, 25, 49, 49 |
  | Raw MPPI, K = 32, H = 48 | 39/50 (78%) | 65–87% | 0.387 | 6.94 s | — |
  | SAC 800k | 50/50 | 93–100% | 0.198 | 8.00 s | — |

- Step 14 minus the base, paired by reset seed, seeds 0–5:
  - completion −0.68, −0.40, −0.24, −0.44, −0.98, −0.24;
  - speed MAE +0.10 to +0.28;
  - time alive −0.9 to −5.8 s.

  Every interval excludes 0.
- Training: seeds 1, 3 and 5 took off at iterations 56, 70 and 60; seeds 0,
  2 and 4 never did (base: 57, 40, 73, 70, 31, 46). The check replaced a
  median of 32–36% of labels per seed, ranging from 0 to 49% across
  iterations.
- Predictions: 1 **held** (32–36%). 2 **wrong** (take-off earlier in 0 of 6
  seeds). 3 **wrong** (55/300). 4 **wrong** (0.473).
- Decision: **reject.** The check is dropped: pooled completion fell, and no
  seed improved.
- Interpretation:
  - The LQR oracle predicted this (Step 14b, D). Judging each noisy label on
    its own throws away the improvement that only shows up across many
    labels. The policy then keeps its own action at a third of its states,
    barely moves, and stays where it cannot walk.
  - It also fits the criticism received on 2026-10-23: MPPI searching
    locally around the states a poor policy reaches cannot find good states.
    The check made that search more conservative still.

### Step 14b — LQR oracle checks of `docs/theory.md` (predictions written before running)

- Question: do Consequence 2 (the noise floor), Proposition 2 (the trust
  step) and Proposition 4 (the predictive check) hold where the answer is
  known exactly?
- Setup: Step 3's LQR point mass.
  - Linear policy u = −Gx; 6 episodes × 25 on-policy states per iteration;
    one policy-centred E-step per state.
  - H = 16, white noise, least-squares M-step, seeds 0 and 1.
  - Error = ‖G − K‖/‖K‖. Runs on the local CPU (Mac).
  - Script: `scripts/diagnostics/lqr_trust_check.py`. Every arm uses one
    E-step implementation, `estep_label`, which reproduces `e_step`'s label
    for the fixed-ρ rule (`tests/test_trust.py`). Noise streams differ from
    Step 3's.
- **A, noise floor.**
  - Start at the optimum; fixed σ ∈ {0.15, 0.3, 0.6} × n_eff/K ∈ {0.05,
    0.2, 0.5} at K = 256, plus K ∈ {64, 1024} at σ = 0.3, ρ = 0.2; 10
    iterations.
  - Floor = median error over iterations 1–10.
  - A1: fitting log floor = a·log σ + b·log n_eff + c over the 22 runs gives
    a ∈ [0.8, 1.2] and b ∈ [−0.65, −0.35].
- **B, σ schedule.**
  - K = 256, ρ = 0.2, 40 iterations, from G = 0 and from the optimum.
  - Fixed σ = 0.3 against σ_k = 0.3·(1 + k/5)^(−1/2).
  - B1: over iterations 31–40, the schedule's median error is at most half
    the fixed arm's, for both starts and both seeds.
  - B2: from G = 0, the schedule first goes below 0.10 at most 2 iterations
    after the fixed arm.
- **C, trust step.**
  - From G = 0, K = 256, σ = 0.3, 30 iterations.
  - Arms: fixed ρ = 0.05; fixed ρ = 0.2; and the trust step. The trust step
    uses the temperature at which the samples' weighted mean cost meets the
    exact LQR controller's noisy expected cost (same noise draws), clipped
    between the ρ = 0.05 and ρ = 0.2 steps.
  - C1: the trust arm first goes below 0.10 at most 1 iteration after fixed
    ρ = 0.05, in both seeds.
  - C2: its floor (median error over iterations 20–30) is within 0.01 of
    fixed ρ = 0.2's and below fixed ρ = 0.05's, in both seeds.
  - C3: its median n_eff/K is ≤ 0.08 over the first 3 iterations and
    ≥ 0.15 over iterations 20–30.
  - For reference, Step 3 had fixed ρ = 0.05 below 0.10 at iteration 5 with
    a floor of 0.065–0.079, and ρ = 0.2 at iterations 7–8 with a floor of
    0.045–0.046.
- **D, predictive check.**
  - Fixed ρ = 0.2 with the check, from G = 0, 30 iterations; the control is
    C's fixed ρ = 0.2 arm.
  - D1: its floor is below the control's, in both seeds.
  - D2: the replaced fraction is ≥ 0.8 from iteration 20 on.
- Status: complete. Ran on 2026-10-23 at `a5bf873` on the local CPU in 214 s
  (`runs/2026-10-23_step14b_lqr_trust_001/report.json`).
- **A result.** Floor = median error over iterations 1–10 from the optimum,
  seeds 0/1:

  | σ \ n_eff/K | 0.05 | 0.2 | 0.5 |
  |---|---|---|---|
  | 0.15 | 0.035, 0.040 | 0.023, 0.020 | 0.021, 0.014 |
  | 0.3 | 0.065, 0.078 | 0.043, 0.039 | 0.042, 0.025 |
  | 0.6 | 0.162, 0.151 | 0.106, 0.084 | 0.096, 0.096 |

  - At σ = 0.3, ρ = 0.2, varying K: 64 gives 0.099, 0.101; 256 gives 0.043,
    0.039; 1024 gives 0.033, 0.025.
  - Fit: floor ∝ σ^1.12 · n_eff^−0.35.
  - A1 **wrong**, narrowly: a = 1.12 is inside [0.8, 1.2], but b = −0.348
    falls just outside [−0.65, −0.35].
  - More samples at a fixed n_eff/K help about as 1/√n_eff from K = 64 to
    256, and less after. A larger n_eff/K helps little: it also shrinks each
    step, so the iterate's noise ball does not shrink with the label noise.
- **B result.** Median error over iterations 31–40, seeds 0/1:
  - fixed σ: 0.040 and 0.060;
  - shrinking σ: 0.014 and 0.023.

  Both starts end at identical late errors: the iteration forgets its start.
  From G = 0, both arms first go below 0.10 at iterations 7 and 8. B1
  **held** (ratios 0.35 and 0.38). B2 **held** (no delay).
- **C result** (seeds 0/1):

  | Arm | First below 0.10 | Floor (it 20–30) | n_eff/K, first 3 its / late |
  |---|---|---|---|
  | Fixed ρ = 0.05 | 5, 5 | 0.074, 0.094 | 0.05 / 0.05 |
  | Fixed ρ = 0.2 | 7, 8 | 0.044, 0.051 | 0.2 / 0.2 |
  | Trust step | 7, 8 | 0.044, 0.051 | 0.2 / 0.2 |

  - C1 **wrong**; C2 **held**, trivially; C3 **wrong**. The trust step never
    left the floor and reproduced fixed ρ = 0.2 to four digits.
  - Why: the budget compared the reweighted policy samples' mean cost with
    the planner's unweighted noisy mean. Sampling noise dominates the spread
    of costs here, so a tiny reweighting removes more noise cost than the
    gap between the policy's plan and the LQR plan. The budget was met by
    steps smaller than the floor.
  - So (T), as formalized, measures competence in expectation under the
    policy's own noise; when noise dominates, even G = 0 looks almost
    competent. The notes' Kendall rule compares plans, not noisy
    expectations. The next version budgets on the improved plan itself: the
    smallest step whose plan costs no more than the planner's plan.
- **D result** (seeds 0/1):
  - first below 0.10 at iterations 11 and 13 (control: 7 and 8);
  - floor 0.074 and 0.066 (control: 0.044 and 0.051);
  - replaced fraction ≥ 0.93 from iteration 20.

  D1 **wrong** (the floor is higher). D2 **held**. Near the optimum almost
  every improved plan is worse than the policy's own (Step 3 saw no plan
  improve at convergence). The check then replaces nearly every label with
  the policy's action, and the gain freezes near 0.07. Without the check,
  the regression averages 150 noisy labels per iteration whose shared
  systematic part still points at the optimum; per-state acceptance throws
  that shared signal away.
- Interpretation:
  - Consequence 2 holds: the floor is a constant-step noise ball that scales
    with σ, and shrinking σ removes most of it with no delay. Of the terms
    tested here, this is the one the oracle supports.
  - The trust budget as written in (T) is too weak when noise dominates; it
    needs a plan-level version.
  - The predictive check slows LQR learning and raises its floor. For Step
    14 (running) this predicts the check will not help, unless Walker's
    wrong-direction labels do harm that LQR's do not.
- Decision:
  - Revise `docs/theory.md`: Consequence 2's exponent, a plan-level budget
    for Prop. 2, and the check's failure mode for Prop. 4.
  - Step 14 runs to completion under its preregistered rule. If it does not
    beat the base, the σ schedule (Step 17) moves up as the best-supported
    next term.

### Step 17 — a σ schedule (predictions written before running)

- Question: does shrinking the E-step's sampling noise over iterations lower
  MPPI-GPS's noise floor on Walker? If so, its policies should track speed
  more precisely without losing completion. On the LQR oracle this term cut
  the late error to 0.35–0.38 with no slower start (Step 14b, B).
- One change from the version Step 14's decision rule keeps: either the
  working method of Steps 10/12, or that method plus the predictive check.
  - The change is `--cfg.sigma-decay-iterations 25`. The E-step noise std at
    iteration k becomes σ_k = 0.3·(1 + k/25)^(−1/2): 0.30 at iteration 0,
    0.21 at 25, 0.17 at 50 and 0.15 at 74.
  - Execution noise (0.1), n_eff/K = 0.2 and everything else stay as in
    Step 14's command.
- Why k0 = 25 rather than the oracle's 5:
  - The base's median take-off is iteration 51 (Step 12). With k0 = 25, σ
    is still 0.17 there, more than half its starting value, so exploration
    through take-off is mostly kept.
  - By iteration 74 σ is halved, which by the oracle's σ^1.12 scaling should
    roughly halve the noise floor.
  - k0 = 5 would cut σ to 0.11 by iteration 30, before most seeds take off.
- Every iteration logs `noise_sigma` and `label_shift_median`, the median
  ‖label − policy action‖.
- Seeds 0–5 and 75 iterations, with the GPU E-step, all six sharing
  robopil5090's GPU, launched after Step 14 is recorded. Output goes to
  `runs/2026-10-23_step17_sigma_seed$seed`.
- Comparison: iteration-74 policies on held-out resets 70000–70049, against
  the version Step 14 keeps, seed by seed. The base (if different), raw MPPI
  at H = 48 (K = 32 and 256) and SAC 800k are evaluated alongside.
- Predictions:
  1. If built on the working method: `label_shift_median` over iterations
     60–74 is at most 0.07 in every seed. Step 13c measured 0.096–0.120 at
     iteration 74 for the working method. If built on the predictive check,
     this is reported without a prediction, since replaced labels have zero
     shift.
  2. Pooled speed MAE is lower than the kept version's.
  3. Pooled completion is not lower than the kept version's.
  4. Median take-off is no later than the kept version's plus 5 iterations.
- Decision: keep the schedule if predictions 2 and 3 both hold.
- Status: dropped on 2026-10-23 without running. It only refines a loop in
  which MPPI searches locally around the states a poor policy reaches, which
  the criticism of that day and Step 14's result set aside.

### Step 15 — MPPI generates the data (predictions written before running)

- Question: if MPPI, a planner that does the task, generates every state and
  the policy only learns from its actions, does the student learn to walk?
  - This answers the criticism of 2026-10-23: MPPI is local, so searching
    around the states a poor policy reaches cannot find good states.
  - Steps 5–14 all labelled the policy's own states, DAgger included.
- A change of direction, not one term: the policy-seeded loop of Steps 5–14
  is set aside. The policy plays no part in planning or collection: no
  seeding, no tail, no coupling.
- This is also the frozen-teacher cloning control that reviewers will expect
  (`docs/related_work.md`, required baselines for C1). VAMPS found plain
  cloning about as good as penalty coupling.
- Teacher:
  - Raw MPPI at K = 32, H = 48, with its own warm start and the frozen
    temperature λ = 3; the rest of `configs/walker2d_best.json` is unchanged.
    This is Step 13b's planner: 39/50 held-out completions, 0.135 s per
    decision.
  - It runs every training episode, with N(0, 0.1²) added to each action it
    executes so the data include small recoveries (as in DART).
  - Every visited state is labelled with MPPI's own action
    (`--cfg.labeler teacher`; `tests/test_teacher_mode.py`).
- Student:
  - The same MLP and fit as before: 2,000 Adam steps per iteration, batch
    256, lr 3e-4.
  - Replay keeps every iteration's data (`--cfg.replay-iterations 75`). The
    teacher does not depend on the policy, so old labels do not go stale.
- Budget:
  - 75 iterations × 10 episodes × ≤ 400 controls per seed, training seeds
    0–5.
  - Runs on robopil5090's CPU with 5 workers per seed, fitting on the GPU;
    about 2.5 h.
  - Each decision simulates 32 × 48 controls, so a run simulates at most
    about 0.46 billion: under a fifth of an MPPI-GPS run.
- Command: for seed in 0–5: `~/mppi-gps-venvs/cuda/bin/python -m
  scripts.training.train_walker_mdgps --out-dir
  runs/2026-10-23_step15_teacher_seed$seed --cfg.labeler teacher --cfg.seed
  $seed --cfg.K 32 --cfg.H 48 --cfg.iterations 75 --cfg.episodes-per-iter 10
  --cfg.steps 400 --cfg.replay-iterations 75 --cfg.workers 5`.
- Comparison:
  - The iteration-74 student alone, on held-out resets 70000–70049 (robopil5090,
    CPU-only evaluation environment).
  - Against its teacher (raw MPPI at K = 32, H = 48, re-evaluated), the
    working MPPI-GPS policies (Steps 10/12), DAgger (Step 11) and SAC 800k.
  - Primary: six-seed pooled completion. Secondary: speed MAE and time alive.
  - Each arm also reports completion per training seed. Pooled episode
    intervals omit training-seed variation (`docs/code_audit.md`, A7).
  - Also reported: the teacher's own completion in training, under the
    action noise.
  - Also reported, a compute ledger: executed transitions, the teacher's
    simulated transitions (decisions × K × H, plus the second iteration at
    each episode's first decision), policy forward passes and wall time.
  - These resets have guided Steps 5–14 (`docs/code_audit.md`, A6), so this
    is a development decision. Final claims wait for the reserved, untouched
    resets 5,000,000–5,000,199.
- Predictions:
  1. The teacher completes at least 60% of its noisy training episodes (78%
     held out without noise).
  2. The student's pooled completion is above DAgger's 0/150. States from a
     planner that does the task fix what labels at the student's own states
     could not.
  3. The student's pooled completion is below its teacher's 78%, and below
     MPPI-GPS's 204/300 (less certain). At H = 48 the frozen temperature
     averages only 1–2 effective samples per action (Step 4), so labels are
     jittery, and a cloned policy compounds its own errors.
  4. No directional prediction on speed MAE.
- Decision:
  - If the student's pooled completion reaches MPPI-GPS's 204/300,
    MPPI-generated data becomes the working base. The next step then adds
    the policy back only where it is competent.
  - If not, the next step targets the cause before any coupling: cleaner
    labels from the same teacher, or more noise for coverage.
- Status: **stopped for futility** at 16:30 on 2026-10-23, after 32–33
  iterations. It ran from 15:36 at `77b1c54` on robopil5090. The stop applies
  protocol rule 8, adopted at the user's request during the run.
  Checkpoints and metrics are kept in `runs/2026-10-23_step15_teacher_seed*`.
- Evidence at the stop (dev resets, 10 per seed):

  | Arm | Mean dev time alive at iterations 4, 9, 14, 19, 24, 29 | Rise, last three minus first three |
  |---|---|---|
  | Step 15 student | 1.6, 1.3, 1.2, 1.1, 1.7, 1.4 s | +0.03 s |
  | MPPI-GPS (Steps 10/12) | 1.2, 1.3, 1.6, 1.4, 1.6, 2.1 s | +0.33 s |
  | DAgger (Step 11) | 1.5, 1.1, 1.2, 1.2, 1.4, 1.3 s | +0.03 s |

  - No student seed completed a dev episode, except 2/10 for seed 2 at
    iteration 24.
  - The teacher completed 79–84% of its noisy training episodes (238–253 of
    300 per seed by iteration 30).
  - The student's fit loss stayed near 0.28, against about 0.0003 for
    MPPI-GPS and 0.10–0.13 for DAgger.
- Predictions: 1 **held** (79–84%). 2–4 were not tested: there was no
  held-out evaluation.
- Interpretation:
  - The teacher does the task from its own states, so the states are not
    the problem here. The labels are.
  - With its frozen temperature at H = 48, MPPI's executed action averages
    one or two effective samples, so it jitters by roughly the sampling
    noise from state to state.
  - The network can fit only the average of that jitter, and so far the
    average does not walk. More data from the same teacher would not change
    that average.
- Decision: **revise.** Keep MPPI generating the states and make the labels
  cleaner, the step's own preregistered fallback. One way is to label with a
  smoother weighted average of MPPI's samples than the near-best single
  sample it executes.

### Step 16 — cleaner labels from the same teacher (predictions written before running)

- Question: Step 15's student did not learn because the teacher's executed
  actions jitter too much to fit (fit loss near 0.28). If each label instead
  averages the same samples more smoothly, does the student learn to walk?
- One change from Step 15: `--cfg.label-neff-fraction 0.2`.
  - MPPI (K = 32, H = 48, frozen λ = 3, own warm start) still plans and
    executes exactly as in Step 15, with N(0, 0.1²) on the action it
    executes. The visited states are therefore generated the same way.
  - Each label is the first action of the same 32 samples re-weighted at the
    temperature that gives n_eff/K = 0.2 (about 6.4 effective samples), not
    the near-best action MPPI executes.
  - Every iteration logs `label_vs_executed_median`, the median distance
    between label and executed action. `tests/test_teacher_mode.py` checks
    that the change leaves execution untouched.
- Fixed, as in Step 15: 75 iterations, 10 × 400 controls, replay of every
  iteration, 2,000 Adam steps, seeds 0–5, 5 workers each on robopil5090's
  CPU, fit on the GPU.
- Command: Step 15's, with `--out-dir runs/2026-10-23_step16_smooth_seed$seed`
  and `--cfg.label-neff-fraction 0.2` added.
- Monitoring: rule 8 runs every 5 minutes in the watcher
  (`scripts/diagnostics/futility_check.py`). The arm stops if the rule fires.
- Comparison, if the arm completes: as preregistered for Step 15. That means
  the teacher re-evaluated, MPPI-GPS, DAgger and SAC 800k; per-seed rates; a
  compute ledger; development resets only.
- Predictions:
  1. The student's fit loss at iteration 29 is at most 0.10 in every seed,
     about a third of Step 15's. Label noise variance falls roughly with
     1/n_eff, from about 1.5 to 6.4 effective samples.
  2. The arm passes rule 8 at iteration 29: its mean dev time alive rises by
     at least 0.3 s, or a seed completes more than 2 of 10 dev episodes.
  3. If it completes: pooled held-out completion above DAgger's 0/150 and
     below its teacher's 78%. No directional prediction against MPPI-GPS's
     204/300.
- Decision:
  - If it completes and reaches MPPI-GPS's 204/300, MPPI-generated data with
    smoothed labels becomes the working base.
  - If it completes below that, the next term targets coverage or label
    quality further, for example more samples for labelling.
  - If rule 8 stops it, smoothing alone is not enough, and the next step
    changes the teacher itself (K, H or the action noise).
- Status: **stopped for futility** at 17:35 on 2026-10-23, after 33–34
  iterations. It ran from 16:38 at `0d2f067` on robopil5090; the watcher
  applied rule 8 at iteration 29. Checkpoints and metrics are kept in
  `runs/2026-10-23_step16_smooth_seed*`.
- Evidence at the stop (dev resets, 10 per seed):

  | Arm | Mean dev time alive at iterations 4, 9, 14, 19, 24, 29 | Rise, last three minus first three |
  |---|---|---|
  | Step 16 student | 1.7, 1.4, 1.4, 1.6, 1.4, 1.3 s | −0.05 s |
  | Step 15 student | 1.6, 1.3, 1.2, 1.1, 1.7, 1.4 s | +0.03 s |

  - One dev completion in total: seed 2, 1/10 at iteration 19.
  - Fit loss at iteration 29: 0.254–0.257 in every seed, against about 0.28
    in Step 15.
  - The smoothing was active: the median label sat 0.25–0.28 from the
    executed action. MPPI's own weights kept a median n_eff/K of 0.05
    (about 1.7 effective samples), as Step 15 assumed.
  - The teacher completed 80–84% of its noisy training episodes (269–277 of
    330–340 per seed).
- Predictions: 1 **failed** (fit loss 0.25, not ≤ 0.10). 2 **failed** (rule
  8 fired). 3 was not tested.
- Interpretation:
  - Smoothing moved the labels as intended but cut the fit loss only from
    about 0.28 to 0.26, not to a third. Most of what the student cannot fit is
    therefore not the sampling jitter that Step 15 blamed.
  - The likeliest remaining cause is that MPPI's action depends on its
    carried-over plan, which the student does not see, so the labels are not
    a function of the state. This was not tested directly.
  - Regressing onto MPPI's actions at MPPI's states has now failed twice,
    with the student's time alive flat both times.
- Decision: **reject.** Smoothing alone does not make MPPI's labels learnable.
  The preregistered fallback (changing the teacher's K, H or action noise) is
  on hold. On 2026-10-23 the user set the next question as how task cost
  drives policy learning, with nothing from the policy entering MPPI: no
  initialization, plans or sampling proposals, including as an ablation.

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
