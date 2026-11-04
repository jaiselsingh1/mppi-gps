# L2.0 — local coupling and matched learning, audited result

**Closed: competent different local experience, retention improvement, no walking gain.** All 24 controllers, six fits and 224 standalone evaluations completed; the confirmed-walking criterion did not trigger round two. No further experiment was launched.

Read the [one-page scientific answer](../../docs/local_coupling_learning_l2_0_result.md), [frozen protocol](../../docs/local_coupling_learning_l2_0_protocol.md) and [accounting appendix](../../docs/local_coupling_learning_l2_0_accounting.md).

- [Exact worker result](worker/result.json), [pre-fit selection](worker/selection.json), [standalone summary](publication/standalone_summary.json).
- [Passing independent audit](audit/final/receipt.json), [findings and audit scope](audit/final/findings.json), [checked file bindings](audit/final/file_bindings.json), [additional independent saved review](publication/independent_review.json).
- [Learning curves and all figures](report/report.md): all 12,000 update losses, own/cross/held errors, 224 episode outcomes, all teacher starts and unpaired joint-angle illustrations including failures. The original generated report's generic feedback wording is clarified by the one-page answer: R2 was deliberately not triggered, not an unfinished execution.
- [Candidate survival, fallen-weight mass, ESS, temperature and paired initial banks](publication/controller_diagnostics.json), [exact cost census](publication/accounting.json), [external run file catalogue](publication/raw_file_manifest.json).
- [Publication validation](publication/validation/receipt.json): static lint, repository hashes/documentation and both read-only inspectors passed.
- [Launch](frozen/launch.json), [source seal](frozen/source_manifest.json), [input and continuation seal](frozen/prepared_manifest.json), [native runtime seal](frozen/environment.json), [supervisor](supervisor_result.json).

The [prelaunch capsule manifest](frozen/capsule_prelaunch.json) and [original navigation bytes](frozen/README_prelaunch.md) preserve the earlier publication. They describe that historical capsule layout; `capsule.json` defines the current files. Frozen source/configuration/input/runtime bytes remain unchanged. The [sandbox launcher failure](engineering/sandbox_launch_001/attempt.json) occurred before any worker; the identical sealed source then ran with process-observation access, with no scientific retry.

Full evidence is local only at `/Users/jaisel/Documents/workbench/mppi-full-trajectory-pi/runs/2026-11-03_local_coupling_learning_l2_001`. Raw checkpoints, arrays, complete ledgers and the immutable source export remain external. The catalogue binds 9,437 closed-study files /875,142,974 logical bytes before publication derivatives; symlinks are recorded separately. It does not assert a remote raw backup. Exact compact result/configuration/job/receipt metadata, curves, figures and audit outputs are tracked here; `capsule.json` binds their bytes and source locations.

The `run/` directory contains exact compatibility copies of configuration/result/source/input/launch metadata for the established generic inspector. Canonical lifecycle records retain their original layout above.

Safe metadata commands from the repository root:

```sh
uv run python scripts/train_mppi_gps.py study local-coupling-loop --help
uv run python scripts/train_mppi_gps.py inspect --run results/local_coupling_learning_l2_001 --verify
uv run python scripts/check_repository.py
```

On the original host, the study's read-only inspector is `study local-coupling-loop inspect --base /absolute/path/to/the/closed/run`. Saved report replay uses a fresh output directory via `gps.mppi_gps.reporting.local_coupling_learning.render(base, out)` with the external run available. It performs no physics or policy fitting; do not reuse a frozen output directory. A compact capsule alone is not a complete raw run for numerical audit/replay.
