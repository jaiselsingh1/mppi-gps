# Artifact layout

Source, configs, tests, and documentation stay in Git. Generated outputs stay in ignored directories:

- `runs/<unique-run-id>/`: reports, raw traces, plots, videos, and run-specific checkpoints.
- `run_mp4/<unique-run-id>/`: optional rendered outputs when a runner uses this convention.
- `data/<unique-dataset-id>/`: generated demonstrations or other datasets.
- `logs/<unique-run-id>/`: tuning databases and runtime logs.
- `checkpoints/<unique-run-id>/`: standalone checkpoints when they cannot live with the run report.

Prefer keeping a run's related artifacts together under `runs/`. Choose a new directory before each invocation; do not overwrite a previous run or reuse historical qualification filenames. No archived policies or checkpoints are restored into the clean checkout.

## Archive

The pre-cleanup repository and its generated runs are preserved at:

[/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004)

Repository files live directly at that archive root. This includes the old policy/GPS/training and GPU experiments. Historical Walker evidence is under [runs/walker2d_teacher](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/runs/walker2d_teacher), including the qualification protocol, reports, raw NPZ traces, analyses, and videos.

Previously archived artifact collections are preserved under [previous_archives](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/previous_archives): `2026-09-03/`, `2026-09-10/`, and `autoresearch-20260619-2030/`.

The [recovery guide](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/metadata/RECOVERY.md) explains preserved Git history and how to resume an old implementation in a separate copy. The [archive manifest](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/metadata/archive-manifest.json) records the source, exclusions, inventories, and verification results. Run the read-only [verification script](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/metadata/verify_archive.py) before recovery. These paths refer to the local archive, which is not published to GitHub.

The cleanup's [validation record](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/metadata/cleanup_validation/validation.json) keeps the test result, exact Walker trace comparison, retained-source checks, and publication commit separately from historical controller qualification. Its neighboring files contain both raw regression runs and their checksums.

Treat archive contents as historical evidence: inspect or copy a needed item into a newly named experiment directory rather than modifying the archived original. Preserve its original provenance and identify any restored input explicitly.

## Experiment records

Use the [experiment log](experiment_log.md) for questions, baseline commits, configs and fingerprints, seed pairs, criteria chosen before running, status, results, decisions, and artifact links. Keep metrics and large binary outputs in the ignored run directory. Failed, rejected, and partial runs remain part of the record.
