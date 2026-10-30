# Credit test stack evidence (2026-10-26)

Copied from `runs/2026-10-26_credit_testbed_001` (ignored by git), produced by
`scripts/credit/run_credit_stack.py` (setup, sweep) and
`scripts/credit/report_credit_stack.py` (report). Protocol and predictions:
[credit_testbed.md](../../credit_testbed.md); verdicts:
[credit_testbed_findings.md](../../credit_testbed_findings.md).

- `manifest_setup.json`, `manifest_sweep.json`: frozen configuration, source
  hashes and git head at each phase's launch.
- `report.md`: every table (testbed validity, representation, option outcomes,
  per-budget decision and closed-loop metrics, outcomes needed, raw prediction
  ledger).
- `summary.json`: the same numbers, machine-readable, with per-seed values.
- `fork_terminal_share.png`, `fork_select_share.png`, `car_share.png`: gap share
  against complete outcomes.
- `posthoc_first_action.py` / `_headroom.json` and `posthoc_paired_ridge.py` /
  `.json`: two analyses added after the results, not preregistered.

The MountainCar policy-iteration phase (`api`) was not run.
