# Acrobot teacher prerequisite 1.0

**Failed teacher qualification: 0/10 fresh starts balanced.** All six development
and ten qualification episodes completed. None reached the 3.8 m height threshold.
No policy fitting or alternating loop ran.

- [One-page result](../../docs/acrobot_controller_teacher_result.md), [protocol and commands](../../docs/acrobot_controller_loop.md).
- [Episode measurements](episodes.csv), [planner passes](planner_passes.csv), [all-trial figure](figure.png).
- [Illustrative median-cost failure still](qualification_median_failure_at5s.png): saved-state kinematics only; all outcomes remain in the figure. The animation stays outside Git.
- [Saved audit summary](audit_summary.json), [compute accounting](complete_accounting.json), [storage accounting](storage_accounting_002.json), [exact raw and animation locations](external_evidence.json).
- [Original handoff snapshot](../../archive/controller_loop_handoff-2026-11-04/README.md).

The recorded replay, costs, seed bindings, optimization weights, executed actions
and work census passed 136,918 saved checks. Full candidate trajectories were not
recorded. The saved-only review reran no physics, inference or fitting.

The raw arrays, trial journal, full audit and animation remain outside Git, with availability
and checksums above. This compact capsule does not replace them. Frozen worker
source `8e566ba` and reviewer source `d5e074c` are distinct. Every copied payload
is byte-sealed. Original failures and untracked source snapshots remain preserved.

Publication 2 corrects formatting and adds storage accounting. The exact first
publication, including its manifest and animation, is retained externally; no
scientific payload, measurement or threshold changed.
