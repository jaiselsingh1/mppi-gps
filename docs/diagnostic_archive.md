# Retired diagnostics: restoration and supported tools

The owner paused source edits and new launches. After PID 34584 exited, cleanup checked all 36 source bindings and the committed archive before retiring the 34 untracked entry points. Exact sources and output paths remain preserved. This maintenance change launched no experiment; the cleanup receipt records the evidence boundary.

## Small active surface

- `src/gps/mppi_gps/experiments/sac_continuation.py` contains the documented historical continuation workflow. `scripts/diagnostics/sac_continuation.py` remains a thin compatible entry point and forwards explicit helper imports. Help imports no simulator or Torch. The old ordinal-rank Spearman helper is deliberately retained and labelled legacy; new statistical inference must use a tie-aware implementation.
- `src/gps/mppi_gps/sample_capture.py` retains the original `RecordingMPPI` class and `PREFIX=8`. It has no acquisition command. Its samples are final-pass, eight-control float16 prefixes, with float32 costs, weights and selected plan: the archive is not a complete trajectory/simulator certificate.
- Main's existing pure `sample_bank` and `research_review` implementations should be copied by the coordinating owner; this stage does not duplicate them. Current native runner/replay refactors are separately owned.

The only suggested additional public dispatcher mapping is `sac-continuation` to `src.gps.mppi_gps.experiments.sac_continuation`, preserving the documented legacy command. The helper class requires no CLI. Do not activate sample acquisition, bank relabel/refit, ridge or coupled-loop commands during cleanup; future scientific execution needs its own reviewed protocol.

## Archived source boundary

`cleanup_plan.json` lists all 34 untracked exploratory entry points for removal from active `scripts/diagnostics/`. It also seals their tracked script dependencies: the original SAC continuation script and native Walker runner. Their 36-file script closure is 308,620 bytes. The coordinating archive is `archive/coordination-2026-11-03`: `untracked/scripts/diagnostics/` contains the 34 files, and `tracked/sac-continuation/` contains the two original tracked entry points. Original sources, docstrings, limitations and result bytes stay unchanged.

The historical source commit is `bd4ba791944a41025aeea4db6b42e714bef5b1e3`; the protocol commit is `18482df`. Retain an accessible history reference or bundle before any eventual branch removal. Other source dependencies are the tracked package modules at that commit, including `policy`, `diagnostic_policies`, `motion_diagnosis`, the native Walker environment, planner and replay/data helpers. The removed scripts import one another; restoring only `policy_picks.py` or `coupled_loop.py` is insufficient.

## Restore for investigation without changing active source

1. Verify the coordinating archive manifest and source inventory. Create a fresh detached checkout at `bd4ba791944a41025aeea4db6b42e714bef5b1e3`; keep the historical commit reachable.
2. Copy the complete archived `untracked/scripts/diagnostics/` tree into the same relative path in that checkout. Refuse to overwrite existing files. Verify the copied 34 files against the archive seals and the two tracked script dependencies against `tracked/sac-continuation/`. Do not run archived scripts directly from their archive directory: their imports and relative input paths expect the original repository layout.
3. Keep original data immutable. The worktree's `runs` location currently resolves to the shared main-checkout evidence. Scripts hard-code K1 inputs, `runs/2026-11-02_noise_cutoff_001`, `runs/2026-11-02_smooth_teacher_students_001` and `runs/2026-11-03_sample_bank_001`; preserve those bindings or record an explicit relocation map. Large NPZ banks and checkpoints remain outside Git.
4. The old scripts' `--quick`, `--smoke` and default output options are not new scientific authorization. Use a separately frozen protocol and a new create-only output directory before any intentional execution. Preserve actual exit status and full cost accounting.

There is no restoration-time physics, fitting or inference step in these instructions.

## Validation before applying

`validation.json` records the staged checks: all original function numerical AST statements agree after only import placement, local enum bindings, docstrings and the argparse `argv` adapter; the entire capture-class AST agrees; pure legacy helpers agree on synthetic arrays; the wrapper's help and capture import load no Torch, MuJoCo or Warp. Ruff checks use the existing repository configuration through stdin because its include patterns intentionally ignore files outside the repository.

Validation covers wrapper help, import laziness, preserved numerical AST/hash checks, pure array tests, archive/source seals and the existing test suite. Run clean-export help/documentation/metadata checks. Do not run the historical continuation workflow, acquire a new bank, refit a policy, or execute any physics as a refactor test.
