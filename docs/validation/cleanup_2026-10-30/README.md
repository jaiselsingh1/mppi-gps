# Repository cleanup validation

The preservation checkpoint is `d630fb4`; the behavior-preserving refactor is `e9408d0`. No research experiments were launched during cleanup. [validation.json](validation.json) records the checks and links each independent audit receipt to its original frozen findings.

- The pre-refactor suite passed 187 tests, with two skips and one expected failure. The refactor passed **234 tests**, with one skip and the same expected failure: 42 new regression cases plus five Hypothesis cases enabled by the locked test dependencies. [Raw pytest output](pytest.txt).
- All 91 exact capsule copies retain their original bytes. All 1,149 initially tracked archive files remain byte-identical; the 162 individually sealed historical source files and 399 completed-study source/input bindings verify.
- The canonical [v3.1](v3_audit_receipt.json), [L1](l1_audit_receipt.json) and [H4.1](h4_audit_receipt.json) saved-only audits reproduce each original `findings.json` exactly. Those checks read existing snapshots, arrays, checkpoints and ledgers without running inference, physics or optimizer arithmetic.
- Complete scientific recipes match pre-refactor golden hashes. Existing tests retain Adam/RNG/proximal-anchor continuation and independent planner checks. New fixtures distinguish raw/file/fitting/typed hash contracts, source sealing, read-only inspection, CLI routing and fitting without simulator imports.
- Production environment, planner and utility method bodies retain the same parsed operations after excluding imports, annotations and docstrings. Core policy and runtime operations are unchanged. The NumPy-array annotations now name `np.ndarray` instead of importing the JAX-specific `jaxtyping.Array`, removing an undeclared clean-install dependency.
- Basic report rendering takes explicit saved metadata and optional lineage inputs. A missing external lineage is reported as unavailable; a clean checkout no longer requires a dated ignored artifact to render a figure.

Ruff checks and formatting apply only to active Python source. PyTorch stays at 2.10.0; Linux CI selects its CPU wheel and resolves the checked-in lock. Frozen sources, capsules and archives are excluded from formatting.

The [export validator](../../../scripts/validate_export.py) runs the complete test suite, capsule/source-history/link checks, all three inspectors and all three saved report renders in a clean Git export without ignored evidence. Rendering uses saved records; small simulator and fitting operations occur only inside unit tests. Original report figures remain unchanged.

Validation wall times are retained in the raw test log and audit receipts. They are engineering overhead, separate from the preserved scientific accounting. No total engineering efficiency or new research performance claim is made. H4 full raw evidence remains local-only, as recorded in [evidence availability](../../evidence_availability.md#h41-local-evidence).

The [first clean-export test log](first_export_pytest.txt) is retained: a local Git exclusion for `report/` omitted 16 capsule files from the first push. Explicit capsule inclusion restores those exact files, and inspector verification now fails when a bound copy is missing. Immutable evidence also has Git line-ending/whitespace exemptions so original CSV bytes survive checkouts.

The corrected [clean-export run](clean_export.txt) passes all eight commands: 234 tests, three hash-verifying inspectors, repository checks and three PNG/PDF report renders. Rendered-output hashes are retained in `validation.json`; fresh renders were visually inspected and were kept outside the immutable result capsules.

The [first Linux CPU CI run](https://github.com/jaiselsingh1/mppi-gps/actions/runs/41138270318) passed locked installation, CPU PyTorch and Ruff, then failed because EGL was unavailable during collection. CPU CI now uses `MUJOCO_GL=disable`: tests do not create MuJoCo renderers, and all physics/replay checks remain enabled. The [same headless suite passes locally](headless_pytest.txt), with 234 passes, one skip and one expected failure.
