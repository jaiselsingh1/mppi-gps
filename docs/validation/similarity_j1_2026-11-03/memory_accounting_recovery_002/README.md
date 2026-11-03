# Bounded Linux publication-memory reproduction

This is one engineering reproduction, using only Python's standard library and
a private tiny saved-data capsule. No actual study evidence, policy, fitting,
simulator or scientific work is opened or repeated.

Deploy the manifest-listed files unchanged into a fresh remote artifact
directory, then invoke once with the pinned CPU interpreter:

```text
CPU_PYTHON -I ENGINEERING/reproduce.py --out ENGINEERING/result
```

The result directory is create-only. Preserve actual outer exit/stdout/stderr;
use a 60-second external wall envelope, 1 GiB parent RSS and 16 MiB output cap.
The owned pressure parent has a 45-second timeout and 768 MiB address-space cap,
allocates and touches exactly 580 MiB, and runs each child for at most 20 seconds.
Both children retain the real 512 MiB AS/RSS materialization ceiling. The old and
new sources are pinned by protocol hashes and see identical private bytes.

Required outcome: the preserved old source fails before reading payload bytes
because Linux's inherited `ru_maxrss` exceeds 512 MiB; the corrected source
passes with its actual current-exec `VmHWM`/`VmRSS` below 512 MiB and preserves the
larger inherited `ru_maxrss` as separate accounting. Its decoded original must
match exactly. All materialization receipts, process exits, memory measurements,
stdout/stderr and source/protocol bindings remain in the result.

Any mismatch stops this version. No retry, full CI rerun or scientific follow-up
is launched by the fixture. The original failed CI and frozen scientific source
remain unchanged.
