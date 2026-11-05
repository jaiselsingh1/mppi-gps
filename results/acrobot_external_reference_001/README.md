# Acrobot external-reference evidence

The identical-model reference checks are complete under their frozen limits;
**a reliable hanging swing-up reference remains missing**. See the
[scientific report](../../docs/acrobot_external_reference_results.md),
[accounting/audit appendix](../../docs/acrobot_external_reference_accounting.md),
[model equations and retained-state review](../../docs/acrobot_external_model_review.md)
and [independent design review](../../docs/acrobot_external_reference_design_review.md).

## Reading the evidence

- `model_review_001/`: exact official-source identity and XML comparison.
- `saved_scan_001/`: all17,296 historical banks' measurements, unresolved action
  hashes, preserved failed review attempts and verified compressed records.
- `selected_trace_audit_001/`: independent geometry, energy and cost checks on
  every retained full predicted trace slot.
- `collocation_v1_0/`: nine failed CasADi/IPOPT attempts, original solver logs,
  debug iterates, native32-case equation parity and exact work receipts.
- `collocation_warmstart_v2_0/`: frozen recorded prefixes, two attempted exact-RK4
  solves and the CPU-cap stop. The second attempt has no final saved candidate.
- `drake_preparation_001/`: same-model parameterization, exact local proof,
  setup failures, native controls and the separate published-handoff check.
  `motion.csv` files are exact plotting extracts, not replacements for complete
  simulator contexts retained in the indexed HDF5 files.
- `mjpc_preparation_001/`: source/SDK bindings, compatibility failures, exposed
  execution traces, outcome summaries, boundary-gate receipts and accounting.
  The old iLQG contract failure remains labelled; corrected results are v2.1.
- `*_audit_001/` and `design_review_001/`: independent saved-output reviews,
  their original failures and explicit contract clarifications.
- `folded_pose_review_001/`: all1,000 late states of the ten original dense
  qualification failures and the narrow stationary-pose hypothesis review.
- `figures/`: PNG/PDF figures challenging weighting, feasibility, static-pose,
  saturation and derivative-repair explanations.

Large arrays, external builds and checkpoints remain outside Git. The
[per-file external index](external_evidence_index.json) gives byte sizes,
SHA256 checksums, locations and availability; `capsule.json` binds every tracked
payload. Official source repositories are pinned in the tracked configurations
and compact preparation manifests. All old evidence remains in its original
locations; failed science and engineering work were retained.

## Supported commands and source

From a clean checkout, re-render the figures without importing a simulator:

```sh
python -m src.gps.mppi_gps.reporting.acrobot_external_figures \
  --run results/acrobot_external_reference_001 \
  --output /tmp/acrobot-external-figures-fresh
python scripts/check_repository.py
```

The output directory is create-only. These commands read tracked evidence and
need no ignored raw directories. The numerical workflows require the explicitly
indexed external inputs and a new frozen output directory:

- [CasADi protocol](../../docs/acrobot_external_reference_protocol.md) and
  [optimizer](../../src/gps/mppi_gps/references/acrobot_collocation.py).
- [Recorded-prefix protocol](../../docs/acrobot_collocation_warmstart_protocol.md)
  and [driver](../../src/gps/mppi_gps/references/acrobot_warmstart.py), which calls
  the same optimizer.
- [Drake/SOS protocol](../../docs/acrobot_drake_sos_reference_review.md),
  [controller/proof](../../src/gps/mppi_gps/acrobot_drake_reference.py),
  [native verification](../../src/gps/mppi_gps/acrobot_drake_native.py) and
  [zero-integration parity](../../src/gps/mppi_gps/acrobot_drake_parity.py).
- [MJPC protocol](../../docs/acrobot_mjpc_reference_review.md),
  [build/supervisor](../../scripts/diagnostics/acrobot_mjpc_reference/run.py),
  [RK4 adapter](../../scripts/diagnostics/acrobot_mjpc_reference/main.cc) and
  [saved-trace summary](../../scripts/diagnostics/acrobot_mjpc_reference/summarize.py).

Core MPPI, simulator interfaces and scientific recipes were not retuned.
No policy fitting or coupled-loop experiment belongs to this packet.
