# P1.0 full-trajectory PI — stopped before comparison

**PRELAUNCH_STOPPED; similarity mechanism untested.** The first complete-context
teacher replay failed its frozen numerical tolerance. Both source and replay
still walked. No PI solve, policy forward, fit, fresh standalone evaluation or
alternating round ran. No retry or follow-up launched.

- [Result and causal limits](../../docs/trajectory_pi_p1_result.md),
  [frozen protocol](../../docs/trajectory_pi_p1_protocol.md),
  [proposed next replay diagnostic](../../docs/trajectory_pi_replay_r0_1_proposal.md).
- [Original failed prerequisite](preflight/preflight.json),
  [worker stop](run/result.json), [independent audit](audit/findings.json)
  and [audit receipt](audit/receipt.json).
- [Saved source/replay curves and figures](analysis/replay/replay_report.json),
  [joint angles](analysis/replay/joint_angles.png),
  [source review](analysis/replay_source_review.md)
  and [closure review](analysis/stop_audit_review/review.md).
- [Work census](accounting/work_summary.json),
  [external raw evidence](external_evidence.json),
  [original source manifest](frozen/source_manifest.json)
  and [prepared input manifest](frozen/prepared_manifest.json).

`PASS_STOPPED` certifies preserved stopping/accounting, not successful source
replay or learning. The empty learning/progress outputs are original evidence of
an unlaunched comparison. Raw arrays, checkpoints and ledgers remain outside Git.
The executed immutable source is commit `211891d4de2bce7703a3faf586fe324e40fb481b`.

```sh
python scripts/train_mppi_gps.py inspect --run results/trajectory_pi_p1_001 --verify
python scripts/train_mppi_gps.py study trajectory-pi report --run results/trajectory_pi_p1_001/run --out /fresh/report
python scripts/train_mppi_gps.py study trajectory-pi replay-report --help
```

These are metadata, saved-report and help commands; they launch no experiment.
