# Experiment history

Main holds only the current method. Everything tried before it is logged here, one file per line
of work. The old code and full write-ups are kept at `archive/*` git tags; raw run data stays
outside git (see the end of this page).

| When | Line of work | Outcome in one line | Log | Code |
|---|---|---|---|---|
| 2026-06 | First MPPI-GPS loop with a cost certificate (Acrobot, then Walker2d) | Survival flat until on-policy visits were added; warm-started from a matched TD3 it kept 1000/1000 and got 3× smoother | [2026-06_gps_experiments.md](2026-06_gps_experiments.md) | `archive/gps/experiments` |
| 2026-10-04 → 10-30 | Walker Steps 1–38, then the 2015 joint-optimisation restart | MPPI teacher solved; cloning it stalls; policy-seeded GPS walked 204/300; the 2015 loop never accepted a trajectory | [2026-10_walker_steps_and_2015_restart.md](2026-10_walker_steps_and_2015_restart.md) | `archive/fresh/mppi-baseline`, `archive/ciogps` |
| 2026-10-29 → 11-03 | Preregistered Walker studies (CIO v1/v2, L1, H4, K1, W1/W2, S1, C1, J1, J2, …) | No MPPI-taught student walked; SAC's data cloned easily | [2026-10-29_11-03_walker_studies.md](2026-10-29_11-03_walker_studies.md) | `archive/main-20261107` |
| 2026-11-02 → 11-06 | Probes, controller loop, MPPI loop with a similarity term | Probes did not make MPPI's actions learnable; controller loop walked 99/100 starts; MPPI loop 3/5 seeds vs 0/4 without similarity | [2026-11-02_06_controller_and_mppi_loop.md](2026-11-02_06_controller_and_mppi_loop.md) | `archive/sac-continuation`, `archive/mppi-loop` |
| 2026-11-03 → 11-05 | Codex follow-ups: whole-trajectory PI, local coupling, CIO gap review, Acrobot | Full PI never changed a trajectory; Acrobot teacher never qualified; coupled children 0/16 | [2026-11-03_05_codex_branches.md](2026-11-03_05_codex_branches.md) | `archive/codex/*` |

Still in progress on its own branch: `cio-gps-replication` (from-scratch CIO-GPS replication on
swimmer and cartpole, started 2026-11-07).

## What held across all of it

- Unmodified MPPI (K256, H48) walks Walker2d. Copying its actions with a state-only network does
  not: MPPI's action depends mostly on the plan it carries, not on the state.
- A student copies a teacher whose action the state sets (cloning SAC's own data walked 94/96).
- Pulling MPPI toward the policy through a term in its cost moves it only a little, never made its
  actions learnable, and makes it fall when strong.
- Whether a student walks is predicted by its error on the states it reaches itself, not by
  offline error on the teacher's states.
- Loops that walked: June's warm-started loop, October's policy-seeded MPPI-GPS (204/300), and
  November's controller and MPPI loops.
- RL (SAC, PPO) walks Walker in about 0.5M executed controls; one MPPI episode simulates about
  12M. Beating RL on simulation count on Walker is out of reach.
- The same controller scores differently on different machines; compare arms on one host.

## Opening old code and results

```bash
git tag -l 'archive/*'                                           # every archived line
git show archive/main-20261107:docs/research_tracks.md           # read one file
git ls-tree -r --name-only archive/ciogps docs                   # list a tag's files
git worktree add ../mppi-gps-old archive/sac-continuation        # check one out to browse or run
```

Remove a browsing worktree afterwards with `git worktree remove ../mppi-gps-old`.

## Where the run data is

- `runs/` and `artifacts/` in the main checkout (ignored by git): runs named
  `runs/<date>_<name>/`, October and November 2026. The Walker sample bank the current loop needs
  is `runs/2026-11-03_sample_bank_001`.
- `/Users/jaisel/Documents/workbench/mppi-gps-archive-20261107/`: the 2026-11-07 cleanup
  (uncommitted files from removed worktrees, Codex runs; see its `MANIFEST.txt`).
- `/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/`: pre-October code and the
  Walker teacher qualification evidence.
- robopil5090: Claude's runs in `~/mppi-gps-claude-sac/runs/`; Step 37 in `~/mppi-gps-step37`.
