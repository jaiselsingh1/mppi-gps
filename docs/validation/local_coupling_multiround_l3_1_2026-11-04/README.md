# L3.1 prelaunch validation

The combined relevant suite passed 142 tests; a subsequent three-test CLI/report subset passed, including one added report-hash regression (143 distinct passing tests). Ruff and git diff whitespace checks passed. Tests cover matched numerical grouping, strict physics/cost predicates, adoption without rewriting the failed receipt, exact interrupted prefix/plan/RNG continuation, conservative pending-work charges, reused rounds, all-eight-round continuation with zero walking, same-arm optimizer continuation and metadata-only CLI behavior.

The initial combined run had one stale error-message assertion; the invalid cross-arm stream was correctly rejected. That failed check is retained in initial_assertion_failure.txt. Its expected message was updated to the already-frozen learner message; the learner source was not changed.

No study physics, acquisition, standalone evaluation or parent refit was performed. The software suite includes isolated synthetic policies (eight synthetic Adam updates and 2,048 backward rows per fit-fixture invocation; three such suite/subset invocations during root validation). Engineering CPU is covered by the original single 1,800-second preparation reservation. Formal command CPU/wall measurements appear in validation.json. Frozen scientific exports, original evidence and archives were excluded from formatting.
