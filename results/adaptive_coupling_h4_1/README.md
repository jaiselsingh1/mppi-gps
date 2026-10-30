# Adaptive coupling H4.1: exact result capsule

Run: `2026-10-30_adaptive_coupling_h4_1_r1_001`. These are exact original small evidence files copied without changing their scientific values or bytes. [capsule.json](capsule.json) records the source path, byte count and SHA-256 for every copied file. The original immutable evidence remains untouched.

- [Frozen configuration](run/config.json), [closed result](run/result.json), [source manifest](run/manifest.json), [input seal](run/input_manifest.json), and [launch environment](run/launch_receipt.json).
- [Every fitting point](run/learning_curves.csv), [figure](figures/learning_and_outcomes.png), and [PDF figure](figures/learning_and_outcomes.pdf).
- [Audit receipt](audit/receipt.json), [findings](audit/findings.json), [original auditor](audit_source/audit.py), and [original native checker](audit_source/native_checker.py). Original audit source is retained byte-for-byte for provenance; active readable audit tools live separately.
- [Stage accounting](report/cost_accounting.json).

The source and input manifests refer to the full original run, including files deliberately excluded from this small capsule. They are not claims that all original files are included here. Full raw HDFs, query ledgers, checkpoints, datasets, videos and frozen source trees remain external. Their original paths in copied JSON are provenance, not portable defaults.

The full H4 raw evidence is **local-only**; no remote backup is claimed. See the [explicit local locator](external/full_local_evidence_locator.json).
