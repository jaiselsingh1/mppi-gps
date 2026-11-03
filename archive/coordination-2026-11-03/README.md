# Pre-cleanup Walker source preservation

The manifest seals 83 payload files (542,978 bytes): all 34 initially untracked SAC diagnostics, tracked dependencies and main entry points, inventories, the owner acknowledgement, and scratchpad sources/logs. The initial coupled-loop log was copied while that already-paid run was active; it is explicitly a partial snapshot. This README is maintained navigation, outside the seal.

No numerical source or failed result is discarded. Main was `0ce214c026b2ba884e1b1410e4e6b705e126d180`; SAC was `bd4ba791944a41025aeea4db6b42e714bef5b1e3`. Both histories and worktrees remain. The owner paused source changes and new launches and assigned cleanup to this investigation.

For historical reconstruction, use an isolated checkout of the recorded SAC commit, then overlay `untracked/scripts/diagnostics/` onto that checkout’s matching directory. Tracked dependencies are included separately for inspection. Do not run those historical entry points as a new experiment without reviewing their incomplete gates and obtaining a separately authorized protocol. Their docstrings are retrospective source evidence, not cryptographically proven pre-registration.

The [coordination map](../../docs/walker_coordination_review.md) identifies supported tools, unresolved claims and the next unlaunched research decision. Raw arrays and continuing run output stay in the original ignored `runs/` locations.
