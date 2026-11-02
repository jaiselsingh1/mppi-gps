# W2 closure validation

The W2 worker was frozen at `5c3241e` and ran once. All scientific source, recipe, task/XML, inputs and results remain unchanged after launch. The [saved-only audit](../../../results/student_update_w2/audit/receipt.json) passed without new neural, optimizer or simulator arithmetic.

- Before launch: **375 passed, one skipped, one expected failure** (29.32 s), with Ruff lint/format passing. [Preregistration Linux CPU CI](https://github.com/jaiselsingh1/mppi-gps/actions/runs/41237160062) passed.
- Final clean export: **375 passed, one skipped, one expected failure** (29.19 s), plus all 14 commands: capsule/archive/link checks, six capsule inspections and six saved-report renders. The export contains no `.git`, `.venv`, `runs` or `artifacts`.
- Ruff checks and formatting pass on 129 active files. Frozen source/input seals, previous historical manifests and all capsule copies verify.
- Read-only notes review covered 19 research notes plus four backup indexes, 189 wiki references, eight heading anchors and 21 file-URI links. All 27 dated backup hashes/sizes match; fences are balanced and targets resolve. The final stale W1 recommendation wording was corrected to link the closed W2 result.

[Prelaunch tests](prefreeze_full_pytest.txt), [environment](prelaunch_environment.json), [final export output](final_clean_export.txt), [export receipt](final_clean_export_receipt.json), [final validation and overhead](final_validation.json), [copy/hash receipt](capsule_copy_hash_receipt.json), and [full experimental/engineering accounting](../../../results/student_update_w2/accounting/engineering_accounting.json).

The only post-run code change extends clean-export coverage to the new sixth capsule/report; scientific worker/auditor/report code remains identical to the frozen source. Full local/CI regression fixtures are engineering work, separately scoped and not globally row/control-metered. Billing and aggregate CPU time are unavailable; later Git publication and CI are retained separately. No efficiency advantage is claimed.
