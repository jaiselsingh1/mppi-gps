# Linux report-fixture isolation

The first [published CI run](https://github.com/jaiselsingh1/mppi-gps/actions/runs/41280422071) at `d95e68c` passed 813 tests but failed the synthetic raw/capsule report fixture. The captured child exception hid its stderr and report receipt. The original [decoded job log](commands/043_ci_failed/decoded_job_logs.txt) is retained; the child error itself was not observed.

Three bounded Linux reproductions used the same frozen reporter and three synthetic trajectories, without loading original motion arrays or models:

| Parent and launch | Result | Report peak RSS |
|---|---|---:|
| Light parent, direct renderer | Pass | 152,883,200 bytes |
| Parent with 600 MiB allocated, direct renderer | Budget stop before plotting | 675,811,328 bytes |
| Same pressure, small intermediary spawning renderer | Pass | 152,055,808 bytes |

[Exact receipts, hashes, source and accounting](receipt.json) preserve all three attempts. The failure occurs at the first decoded-array budget check, rather than from large plot output. Linux [preserves the previous address-space high-water RSS at exec](https://github.com/torvalds/linux/blob/v6.8/fs/exec.c#L987-L990). This reproduces and supports the test-isolation diagnosis; it does not reconstruct the missing original CI child receipt.

Only the test helper changes: a small intermediary launches the renderer, and a failed child exposes stdout, stderr and its saved receipt. An inner 110-second timeout kills/reaps a hung renderer before the outer 120-second timeout. The report's 512 MiB limit, production modules, recipes and all frozen science remain unchanged. Thirteen focused report tests pass, as do Ruff checks on 202 active files; final full-suite CI follows the fix's push.

The three synthetic fixture constructions account for 162 pair comparisons and 3,438 aligned state pairs; they are engineering work, separate from K1.2's original 720,786 comparisons. Enclosing captured command walls total 20.456299 seconds through command 050 and include the remote/report subprocesses. The earlier failed CI test stage took 149.15 seconds; setup/cleanup, global fixture work and uninstrumented editing are not reported as zero. Later commit/push/CI receipts remain append-only outside Git at the existing engineering location.
