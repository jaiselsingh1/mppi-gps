# H5.0 round-one launch snapshot

The single authorized round was launched from frozen source commit `ff67495dae33ee43952677d0c0165a43f8d469b0` on `robopil5090`. The [bootstrap receipt](bootstrap_receipt.json) records coordinator PID **166160**, its process birth identity and the exact frozen-source command. The coordinator owns the **21,600-second** worker watchdog. The bootstrap did not bypass it or invoke the worker directly. No retry or second round is authorized.

- [Exact bootstrap source](launch_bootstrap.py) and [launch-command receipt](launch_command_receipt.json).
- [Create-only initial export](initial_export_receipt.json): **539 files /10,622,281 bytes**, all hashes matched before launch. The archive was **4,107,708 bytes**. [Transfer-command receipt](export_command_receipt.json) and [freeze-command receipt](freeze_command_receipt.json) preserve enclosing engineering costs.
- [Preparation audit summary](preparation_audit_summary.json): **540 metadata checks**, no array decodes, inference, physics or updates in that preparation phase. [Audit-command receipt](preparation_audit_command_receipt.json). This is not the scientific outcome audit.
- [Independent prelaunch review](prelaunch_independent_review.json) verified all **523 source files**, all11 sealed inputs/metadata, unchanged scientific settings and the original unlaunched preparation. The [987-test prelaunch validation](../prelaunch/README.md) remains the validation of the launched source.
- [Launch snapshot manifest](manifest.json) binds every copied payload file in this directory.

This is an immutable **launch snapshot**, not a result or current process-health claim. The bounded first round is in progress; teacher outcomes, standalone outcomes, paired gaps, scientific audit and learning-benefit verdict are pending. The unmet useful-learning prerequisite remains false under the explicit one-round ordering exception. The original preparation and its false launch flag remain unchanged.

The local `runs/2026-11-01_teacher_proposal_h5_0_r1_001` directory contains only the initial source/config/input export. New scientific checkpoints, learning evidence and raw array/index/ledger streams are being written on the remote host only. The initial539-file verification does not describe the current growing remote run. No final size or checksum is claimed for those growing streams. Full raw evidence remains outside Git; later closure and independent audit will supply completed-file bindings.

Launch/transfer command timers include their inner work and must not be added to nested timers. The preparation audit's zero scientific-work counters describe preparation only, not the running study. This record does not inspect scientific arrays or repeat any paid operation.
