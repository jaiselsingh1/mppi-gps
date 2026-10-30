# History and archive recovery

The 27 October cleanup preserved all 792 original source/documentation files (11,418,693 bytes), including 337 untracked source files and the pre-existing modification to `docs/experiment_log.md`. Eight additional untracked artifacts remain in place. Of the source files, 98 were selected for the active tree; 694 were removed from the active tree only after their archived SHA-256 checks passed. The root README was then rewritten. New documentation, tests and the canonical command are separate additions.

The complete original checkout is `archive/research-2026-10-27/`. Its `metadata/worktree.json` records every original relative path, size and SHA-256, the original Git state, and exclusions. `metadata/original_git_diff.patch` retains the worktree changes. The cleanup receipt and original-to-archive path map are under `runs/2026-10-27_repository_cleanup_001/selection.json`.

## Access

The `legacy` command starts a fresh Python process with its working directory and `PYTHONPATH` set to the archive. Thus `src.*` imports resolve within the historical checkout. It cannot accidentally combine a historical runner with the new active learner.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps legacy \
  scripts.training.train_bidirectional_gps --help
```

The archive's `runs` and `artifacts` symlinks refer to the original directories. Any historical run must use fresh output paths and retain its declared configuration. This recovery facility is not permission to overwrite frozen results. Some old receipts contain absolute live-source paths; those source files now resolve through the recorded original-to-archive map, not through rewritten receipts. Frozen run-local source packets remain unchanged. A source snapshot preserves bytes and dependencies; it does not prove every old experiment is replayable on current hardware.

An earlier archive at `/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004` is also unchanged. Its original documentation is retained inside the new archive as `docs/artifact_layout.md`.

## What the previous work established

The complete bidirectional thesis has not passed. The cheap Ending task supplied successful independent-search experience and policies that could finish. Some policy-assisted executed controllers improved on short-horizon search. The strongest search-plus-memory controls often remained better than actor continuations, and guided data did not reliably improve the next standalone policy. Acrobot discovery/ending qualification and the full physical loop remain unresolved; there is no all-cost efficiency claim.

The latest deployment formulation retained a parent with 64/64 Ending successes and cost 2.279455. Its final child succeeded 27/64 with cost 12.977002. Independent search plus memory and the guided fallback both succeeded 62/64 with cost 2.351424; zero handovers meant no demonstrated feedback advantage. This is a retained negative result, not the new joint algorithm.

For detailed evidence, start with the archived `docs/bidirectional_deployment_learning_findings.md`, `docs/bidirectional_algorithm_state.md`, `docs/bidirectional_causal_verdict_2026-10-27.md` and `docs/experiment_log.md`. The original reports include costs, failed audits, corrections and limits. Their raw run directories remain in place.

The earlier approach used MPPI-acquired states, actor-owned complete returns and later outcome qualification, but did not explicitly constrain the actor update with verified MPPI behavior. The new goal addresses that missing joint optimization structure. It does not assume this structural change alone will repair the measured failures.
