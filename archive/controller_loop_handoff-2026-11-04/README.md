# Controller-loop handoff snapshot

Preserved on 2026-11-04 for the Acrobot port. This directory is an immutable
research-source reference, not an active implementation or an experiment result.
Do not format or silently update its payloads.

The two entry points are `controller_loop.py` and `plot_controller_loop.py`.
Their complete recursive `scripts.diagnostics` import closure contains 28 Python
files, captured with `git show` from `sac-continuation` commit
`aa54790d1f5598888b1deb2557b3d0db4f9b9179`. The closure includes imports inside
functions and traverses cycles once. Non-diagnostic modules and third-party
packages are listed in `manifest.json`; they are not part of this source-only
snapshot. Running these files requires an explicitly bound repository/runtime.

`manifest.json` records the commit, original paths, SHA-256 checksums, byte sizes,
import edges and provenance class for all 31 payloads (365,139 bytes).

- `git/scripts/diagnostics/`: exact committed diagnostic sources.
- `reference/untracked/controller_loop_acrobot_instructions.md`: exact untracked
  handoff document from the other worktree. Its prose describes requested work;
  it is not a verified scientific result or a replacement for user instructions.
- `reference/modified/src/envs/walker2d.py`: exact live modified Walker source,
  retained only to preserve the other branch's uncommitted work. It is **not**
  copied over the active environment.
- `reference/git/src/envs/walker2d.py`: the committed Walker baseline beside that
  modified reference. The live change adds a configurable model path.

Neither the other worktree nor its branch was edited. No physics, fitting,
evaluation, teacher acquisition or experiment was executed to make this snapshot.

## Inherited limitations relevant to the Acrobot port

These are source-review findings. They do not invalidate or repair saved results.

1. **Fresh Adam per fit.** `slope_fit.fit_rows` constructs Adam and a seeded
   minibatch generator each call. The loop carries policy parameters between
   rounds but does not preserve an Adam/RNG training stream. Saved policies are
   weight dictionaries, not resumable complete worker checkpoints.
2. **The `none` arm is a no-pull controller control.** Setting its objective
   pull to zero leaves policy-dependent proposal directions, Jacobian-based gain
   mixing and slope fitting active. It is not plain cloning or DAgger.
3. **Proposed versus retained policy gaps.** A refit's `gap_after_fit` is computed
   before the task-score acceptance decision. When a refit is rejected, the
   deployed policy rolls back but those stored gap fields still describe the
   proposed policy. Acrobot reporting must distinguish the two explicitly.
4. **Raw angles and scale.** The slope code assumes each observation is one
   state coordinate. Acrobot's unwrapped angles can represent the same upright
   pose on different branches, creating a spurious full-turn tracking distance.
   Record winding/branch and verify shared-startup behavior; silently wrapping
   observations changes that slope contract. Near-stationary coordinates also
   require checking the per-trajectory standard-deviation scales and finite gains.
5. **Physical time.** The inherited carry-on horizon is 125 controls, spacing
   100 controls, competence horizon 250 controls, and noise settings use Walker's
   125 Hz configuration. A different Acrobot control rate changes their physical
   meaning. Freeze explicit task-specific timings rather than claim unchanged
   one-/two-second measurements accidentally.
6. **Regression bytes.** `summary.json` contains elapsed wall time. Whole-file
   identity across two executions cannot be assumed. Preserve complete files
   and compare scientific fields, arrays and checkpoint tensors exactly, with
   any metadata exception stated. Do not rewrite historical outputs to obtain
   matching checksums.
7. **Capped acceptance score.** Equal score at the cap can accept a policy with
   higher native task cost. Record cost and actual task success independently;
   score monotonicity alone is not competence or convergence.

The Acrobot instructions' claims about Walker action noise and state
predictability remain estimator-specific interpretations. This snapshot does not
establish a state-independent torque component or certify the nine-run comparison.
