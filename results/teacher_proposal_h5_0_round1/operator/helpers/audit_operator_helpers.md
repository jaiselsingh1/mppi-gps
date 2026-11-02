# H5.0 normal teacher-stop audit operator

These standalone operator helpers do not modify the frozen worker or auditor. They have not launched anything during preparation. Run them only on `robopil5090`, using `/home/jaisel/mppi-gps/.venv/bin/python`, after the research coordinator has written both its completed normal-negative result and successful outer exit.

The bootstrap requires the **observed SHA-256 of the final `worker_result.json`**. It checks that binding, fixed config/source/input/auditor owners, all36 completed teacher episodes, three parent fits/six checkpoint points, no child or evaluation/gap work, and normal worker exit without unknown partial work. A mismatch stops. The committed scientific source remains `ff67495dae33ee43952677d0c0165a43f8d469b0`; the useful-learning prerequisite is still false and no second round is authorized.

After copying the reviewed helpers to the remote artifacts directory, the exact launch shape is:

```sh
/home/jaisel/mppi-gps/.venv/bin/python -B \
  /home/jaisel/mppi-gps-workspace/artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/audit_bootstrap.py \
  --expected-worker-result-sha256 REPLACE_WITH_OBSERVED_64_HEX_SHA256
```

The literal placeholder fails validation. Supply the real observed hash, not an anticipated result. The helper creates only `audit_operator_001`, then detaches the **public frozen audit CLI** with `--mode round --run RUN --out .../audit_round_001`. It does not call hidden `--worker`. The frozen coordinator owns nonce/PID/file authentication and the **7,200-second /4GiB** watchdog. It reserves at most1,999,713,552 saved prediction rows,564,270 calls,15 checkpoint loads,3TiB reads and4GiB output; no new physics or fitting. The teacher-stop prefix will consume less than the full roster. Both audit and operator output directories are create-only, with no retry or replacement allowance.

Monitor the one coordinator with bounded read-only snapshots:

```sh
/home/jaisel/mppi-gps/.venv/bin/python -B \
  /home/jaisel/mppi-gps-workspace/artifacts/2026-11-01_teacher_proposal_h5_0_r1_001/audit_monitor.py \
  --samples 160 --interval-seconds 45
```

The monitor verifies boot ID, process birth ticks and audit-owner PID/result binding. It reads only bounded metadata and journal/stdout tails, never SQLite contents or raw arrays. The whole series caps reads at64MiB, stdout at16MiB, address space at512MiB and elapsed time at7,500 seconds. Missing progress is unavailable, not zero; journal rows describe paid prediction progress, not audit success. Only a successful outer `exit.json` **and** completed successful scientific receipt establish closure. A missing coordinator without closure preserves an incomplete prefix and never authorizes retry. Capture monitor stdout in a new operator artifact outside the immutable run/audit directories.

`audit_operator_validation_001/receipt.json` records five synthetic operator checks, eight synthetic collector checks and Ruff results. No SSH, audit, policy inference, physics or fitting ran in those checks. Ordinary imports/filesystem metadata work and editing overhead are not fully metered.

The reviewed collector remains byte-identical (`8f92b5b4...ff48b`) and handles the normal teacher-competence stop without assuming15 checkpoints, child policies, evaluations or paired gaps. It retains actual201-/199-control prefixes, twelve full-data metrics and all36 teacher episode/posthoc pairs. Its new synthetic negative-closure case checks six legitimate checkpoint aliases and all36 variable-length teacher episodes. Execute collection only after successful audit closure; use a separate forensic workflow if the audit fails.
