# K1.1 publication validation

The saved-only scientific result and its immutable capsule were committed as `6ae18cd`. The successful clean Git export of `a2ccbc8` contains no ignored runs/artifacts and passes **707 tests, one skip and the existing expected failure**, all 26 CLI/inspection/report commands, 2,650 capsule bindings, 162 historical files and 911 maintained links. Ruff checks 191 active files. No scientific worker reran.

The export verifies and renders the new capsule as well as all older studies. It also checks regenerated output bytes against their own report receipts, so legitimate new PDF timestamp metadata is not confused with original frozen PDF bytes. The complete source/input/worker closure was independently verified by [audit002](../../../results/trajectory_review_k1_1/audit_002/receipt.json).

- [Validation census and exact file hashes](receipt.json).
- [Successful export command/elapsed receipt](commands/025_final_export_validation/receipt.json), [stdout](commands/025_final_export_validation/stdout.txt), [stderr](commands/025_final_export_validation/stderr.txt).
- [Previous failed export](commands/022_results_export_validation/receipt.json): all704 tests and all26 commands passed; the final containment check miscompared macOS `/var` and `/private/var` aliases. Every output size/hash matched. The canonical-base correction has alias, escaped-symlink and changed-byte regression checks.
- [Final fixture consistency checks](commands/026_export_fixture_consistency/receipt.json): three focused checks pass after using a seven-byte expected payload in the same-size tamper test. This only changes a test literal.
- [52 maintained-report numeric checks](checks/report_number_verification.json). The first formatting checker stopped at a Unicode minus; both checker versions and failed/passed commands are retained, without changed scientific values.
- [Obsidian update receipt](checks/obsidian_update.json): three existing notes updated, four-note structure retained, exact backups outside the vault, original user scratch text and Research Thoughts preserved. No new notes or result tables were added.

The [scientific capsule accounting](../../../results/trajectory_review_k1_1/accounting/accounting.json) separates inherited K1 counters, new zero-model/physics arithmetic, failed/passed audits and both reports. This later bundle retains publication/export/check overhead, including failures. Command wall times include their subprocesses and must not be added to those internal times. Global synthetic-fixture arithmetic/physics counts and uninstrumented editing/tool billing are unknown, not zero. No efficiency claim is made.

This snapshot precedes its own commit/push. Later append-only publication, remote-verification and CI receipts remain at the explicitly located `artifacts/2026-11-01_trajectory_review_k1_1_001/engineering` outside Git. They do not change the frozen run, capsule, or scientific conclusion.
