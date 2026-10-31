# W1 validation and publication history

The scientific worker was frozen at `55da848` and ran once. Its result, source, configuration, input bindings and thresholds remain unchanged. The [independent audit](../../../results/student_width_w1/audit/receipt.json) passed without neural, optimizer or simulator replay.

- Before launch: **339 passed, one skipped, one expected failure**, with Ruff lint/format passing.
- Final clean export: **340 passed, one skipped, one expected failure** (23.67 s). The additional check covers compact W1 metadata inspection. All 12 validation commands passed: regressions, capsule/archive/link verification, all five capsule inspections and all five saved-result reports. The export contains no `.git`, `runs`, `artifacts` or `.venv`.
- Current Ruff checks and formatting pass on 118 active files. The frozen run's source/input seals and all previous capsule/archive copies also verify.
- Notes review: 18 research notes plus three backup indexes, 168 wiki references, eight heading anchors and 15 local evidence links; all 20 dated backups match their indexed hashes and sizes. No broken links or unbalanced fences.

Linux CI first exposed a test portability mistake: the original Mac serialized checkpoint ZIP SHA differed across PyTorch builds although the named tensor/RNG hashes matched. The regression now checks the original file SHA on its bound arm64 Mac build and verifies implicit/explicit default-128 bytes, semantic hashes, old keys and continuation on all builds. A subsequent intermediate push failed formatting of that conditional; final formatting and export passed. These engineering failures are retained, not presented as scientific failures or omitted retries. No experiment was rerun.

[Prelaunch pytest](prefreeze_full_pytest.txt), [environment](prelaunch_environment.json), [portability correction](ci_portability_fix.json), [intermediate CI formatting failure](intermediate_ci_failed_log.txt), [final export output](final_clean_export.txt), [export receipt](final_clean_export_receipt.json), [final validation/overhead](final_validation.json), and [capsule copy/hash overhead](capsule_copy_hash_receipt.json).

[Full research and earlier engineering costs](../../../results/student_width_w1/accounting/engineering_accounting.json) and [all local evidence hashes](../../../results/student_width_w1/external/full_run_file_manifest.json) remain frozen. Local/CI test-fixture arithmetic is separate engineering work and not globally metered; billing/aggregate CPU time are unavailable. Later Git publication and CI are recorded separately. No efficiency advantage is claimed.
