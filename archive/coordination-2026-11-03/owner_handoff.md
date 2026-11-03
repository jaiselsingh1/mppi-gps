# Walker sac-continuation handoff (Claude session) — 2026-11-03 12:45 EST

Acknowledging the Codex coordination request. Starting now I make **no source edits, launch nothing new, and do no refactors or deletions**. Cleanup and archiving are yours.

## Ownership and active work

- **Branch and worktree:** branch `sac-continuation`, worktree `/Users/jaisel/Documents/mppi-gps/.claude/worktrees/sac-continuation`.
  - Committed and pushed: `18482df` (protocol) and `bd4ba79` (SAC continuation results).
  - All 34 untracked files in `scripts/diagnostics/` are mine. They are exploratory diagnostics, never audited.
- **Still running:** `scripts/diagnostics/coupled_loop.py`, PID 34584, started about 12:31 EST. This run was already paid for and is bounded: 2 arms × 3 rounds × 6 MPPI episodes. I'm letting it finish (about 15–25 more minutes).
  - It writes only to `runs/2026-11-03_coupled_loop_001/` (finished so far: `torques_round{1,2}.npz`; `summary.json` comes at the end).
  - Its log is in my scratchpad: `/private/tmp/claude-501/-Users-jaisel--claude-spares-spare-9NRwAu/d1be111d-3d91-4afd-aa53-f75f917509e0/scratchpad/coupled_loop.log`.
  - It reads the sample bank and imports the scripts listed under "Scripts I still need". Please don't move them until it exits.

## Newest results

All are exploratory, not frozen protocols. Each script's design and readings are in its docstring, written before the run.

- **Ridge check** (`ridge_check.py`, `runs/2026-11-03_ridge_check_001`): held-out rollouts along the directions `flat_directions` fitted. Preset reading: **mixed**.
  - The dip is real at 68% of states.
  - Median curvature is −3.7, against −18.8 fitted. That's about 0.2 cost units over ±0.3 torque, against λ = 3.
  - So "a ridge at every state" (from `flat_directions`) was overstated and is retracted.
- **Side-picking MPPI** (`side_picking_mppi.py`, `runs/2026-11-03_side_picking_001`): split each update's samples in two, then average only the heavier half, the half nearer the falling student, or the half nearer the walking SAC clone.
  - Consistency −4% to 0% (baseline 3%); cost +25 to 32%; students 0/96.
  - Negative, as predicted beforehand.
- **Sample bank** (`sample_bank.py`, `runs/2026-11-03_sample_bank_001`, 284 MB): unchanged default MPPI, seeds 0–9. For every control it stores each sample's first 8 controls, the costs and weights, the plans and the warm starts.
  - It repeats the default arm of `runs/2026-11-02_noise_cutoff_001` exactly; 9/10 walk.
- **Policy picks** (`policy_picks.py`, `runs/2026-11-03_policy_picks_001`): offline labels made from the bank. MPPI's samples are importance-reweighted as if drawn around the policy's action (h 0.2/0.3, λ 3/10), over 3 relabel → refit rounds.
  - Label consistency 21–33%, against 4% for MPPI's own actions.
  - Labels beat the policy's own action in rollouts at 59–69% of checked states, but are worse than MPPI's own action.
  - Students 0/64; takeover 75 → 78–81 of 100. Preset readings: both false.
  - Post hoc (`posthoc_ceiling.json`): a perfectly state-determined label scores only **55%** on these MPPI states. The "better" part (label minus policy) has R² −15%, and the policy sits 0.85 from MPPI's sampling centre (MPPI's own update moves it 0.26).
- **Coupled loop** (running): MPPI with torque matching at weight 0.3 toward the current policy, iterated 3 rounds. One arm trains on MPPI's own torques; the other trains on policy-picked labels.
  - Arm "MPPI's torques", rounds 1–2: MPPI walks 6/6 at −4% cost. Coverage (policy-to-sampling-centre distance) 0.95, not reduced. Consistency 8–10%; "better" part −7 to −8%; students 0/64; takeover 77.

## Retained raw outputs

All are git-ignored and local in the worktree's `runs/`.

- **Mine:**
  - `runs/2026-11-02_*`: action_spectrum, coupled_mppi_{walker,faller}, error_consequences, feedback_vs_action, jitter_tolerance, kendall_ideas_{a,b}, label_conflict, match_only_mppi, motion_consistency, motion_coupled_{walker,faller}, mppi_compute, mppi_consistency, noise_cutoff (119 MB), on_policy_disagreement, plan_keeping_{a,b}, sac_continuation (83 MB), smooth_teacher_students, student_failure, student_positive_control, student_takeover, update_noise_{a,b}.
  - `runs/2026-11-03_*`: confidence_w{1000,5000}, coupled_loop (running), flat_directions, mirrored_{a,b}, policy_picks, ridge_check, sample_bank (284 MB), side_picking, stability_calibration_{walker,faller}, stability_terminal_{walker,faller}, takeover_terminal_{walker,faller} (plus `posthoc_survivors.json`).
- **Not mine:** `runs/2026-11-01_*` and `runs/2026-11-03_jitter_diagnosis_j2_001`.
- **Hard-coded paths in the scripts:** `runs/2026-11-02_noise_cutoff_001` (`SOURCE`), `runs/2026-11-02_smooth_teacher_students_001` (`--previous`), `runs/2026-11-03_sample_bank_001` (`--bank`). Please keep these paths, or update them together with the sources.
- **Only in my scratchpad, which can be cleaned up at any time — archive them if you want them kept:**
  - `picks_ceiling.py`, the post hoc that wrote `posthoc_ceiling.json`;
  - run logs (`*.log`).
  - The `flat_directions` |λ| recheck was run inline, so no script exists for it.

## Missing controls and audits

- None of these runs is audited, capsuled or frozen in `docs/`. The protocols live only in docstrings.
- The samples are small: 6 MPPI episodes per cell and 2–3 student seeds.
- **Consistency metric:** the 10-nearest-state R² has a ceiling of about 55% on MPPI's states (it's sparse there), while SAC's 99% is measured on SAC's periodic states. Earlier consistency comparisons across teachers should carry that caveat.
- **Missing control for policy picks:** a relabel toward a non-learner smooth prior, to separate "any smooth prior raises consistency" from the policy's own role.
- **Missing controls for the coupled loop:** a weight sweep (only 0.3 is run), and an uncoupled iterated arm on fresh MPPI runs. The policy_picks rounds are only a partial stand-in, because they use a fixed bank.
- **Known doc error:** `docs/sac_continuation_results.md` (pushed) cites S1.1 as "even oracle clones fail". S1.1's failure was undertraining: 10k updates walk 94/96. It needs correcting.

## Scripts I still need

Please keep import paths and module names:

- **Shared modules:**
  - `action_spectrum` (neighbour_index, state_dependence, FIRST)
  - `noise_cutoff` (observation)
  - `smooth_teacher_students` (ARMS, SOURCE, EVAL_SEEDS, RECIPE)
  - `student_failure` (fit_students)
  - `student_positive_control` (fit_s1, standing)
  - `student_takeover`
  - `coupled_mppi` (CONFIG, SEEDS, STEPS, WALK_MAE, matching_cost, describe)
  - `mirrored_mppi` (run_episode, summarize)
  - `takeover_terminal`
  - `stability_terminal` (takeover_from)
  - `flat_directions` (quadratic_fit, PROBES)
  - `sample_bank` (RecordingMPPI)
  - `policy_picks` (fit, relabel, load_bank, better)
  - `update_noise`, `match_only_mppi`, `jitter_tolerance`
  - the committed `sac_continuation`
- **Active line:** `sample_bank`, `policy_picks`, `coupled_loop`, `ridge_check`.
- **Leaf scripts I don't need to run again** (archive as they are): label_conflict, error_consequences, feedback_vs_action, on_policy_disagreement, mppi_consistency, mppi_compute, motion_consistency, motion_coupled_mppi, kendall_ideas, plan_keeping_mppi, side_picking_mppi, and the `plot_*` scripts.

Results are also summarized in Obsidian `MPPI GPS/Research ideas.md` (items 5–7). I noted the J2 status as you described it: stopped on frozen clean qualification (143/192 walking), with the six-case post-fit correlation exploratory only.
