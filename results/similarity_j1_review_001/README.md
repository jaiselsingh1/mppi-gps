# J1 saved-outcome review, version 001

This post-hoc supplement decomposes all 32 J1 development cases using saved
metadata only. The original scores, eligibility gate, outcome records and
unopened protected test remain unchanged. The [maintained review](../../docs/similarity_j1_review.md)
states the interpretation and limits.

- [Per-case outcomes](cases.csv): walking, standalone duration, same-start
  duration, original motion score, overlap-only motion error and original scores.
- [Summary](summary.json): exact decomposition and descriptive tied-rank
  correlations; there is no protected prediction test.
- [Figure](motion_and_duration.png): original versus overlap-only motion error
  against duration, with both sources shown separately.
- [Receipt](receipt.json): input/source hashes, measured reads and wall time,
  finite limits, and zero new simulation, inference, fitting or updates.
- [Capsule](capsule.json): exact file inventory.
- [Duration extension](duration_extension/episodes.csv): all 512 standalone and
  32 reproduction episodes, including observed seconds, fall events, right
  censoring, walking and separate cost terms. Its [case table](duration_extension/cases.csv)
  also exposes the four capped overlap-component means; its separate
  [receipt](duration_extension/receipt.json) binds the extended implementation.

The overlap-only measure uses only the saved error over actually shared time.
It conditions on unequal surviving prefixes and remains sensitive to timing;
it does not measure walking competence. No missing physical state is fabricated.
The first decomposition, its source and receipt remain intact. The extension
reads the same saved inputs and adds no scientific work.

Recompute into a fresh directory outside the immutable original capsule:

```sh
.venv/bin/python -m src.gps.mppi_gps.reporting.similarity_diagnostic \
  --capsule results/similarity_j1 --out /tmp/j1-review-fresh
```
