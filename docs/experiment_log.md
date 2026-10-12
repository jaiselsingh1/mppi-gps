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

1. Create a branch for one question. Record its name and the exact baseline commit before changing code or configs.
2. Allocate a unique ignored directory such as `runs/<date>_<question>_<run-id>/`. Keep configs, reports, traces, diagnostics, and checkpoints together.
3. Write the question, comparison/baseline, environment and task objective, config paths and hashes, source fingerprints, dependency versions, and reset/planner seeds before running. Separate development from untouched qualification seeds.
4. Fix the acceptance criteria and primary metrics before inspecting results. Change one factor at a time when the comparison requires attribution.
5. Record complete, failed, interrupted, and rejected runs. Preserve every report and raw trace used to support a decision. State whether a check concerns simulator correctness, interface compatibility, teacher control quality, or standalone policy performance.
6. Update status, results, interpretation, and the keep/reject/revise decision. Link artifact paths and code/config revisions. A partial run or a proposed method is not a validated result.
7. Graduate validated changes to `main` through normal commits after the recorded criteria and relevant regression checks pass. Preserve unsuccessful experiment branches and their records.

## Experiments

### S1: Walker teacher instrumentation and exact replay

```text
Experiment ID / date: S1 / 2026-10-11
Question: Can the frozen Walker teacher's decisions and its recorded qualification failure be reproduced exactly, mid-episode included, and what did the planner predict before that fall?
Branch: exp/walker-instrumentation
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
Decision: keep (instrumentation); committed as the S1 checkpoint on exp/walker-instrumentation.
Interpretation and limitations: One fall, heuristic evidence. "Survives" means within the probe horizon only. Whether the surviving 48-step options at steps 489-501 would have avoided the fall beyond the horizon is untested; a paired counterfactual branch from those decisions (S3 machinery) would test it. Short-horizon causes are not attributed.
Evidence backup: /Users/jaisel/Documents/workbench/mppi-gps-evidence/2026-10-11_walker_instrumentation.tar.gz
  (SHA-256 b6a3510f313a68e3610a750345eab424522d4a3d4d7a03167e2ed254f76cec7d); all 18 files verified against the manifest after extraction.
```

Add one entry per question using the fields below:

```text
Experiment ID / date:
Question:
Branch:
Baseline commit:
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
