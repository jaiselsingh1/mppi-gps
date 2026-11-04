# Preserved handoff evidence

The [original brief](coupled_pilot_brief_sac_2026-11-04.md) is byte-exact to its untracked `sac-continuation` source on 2026-11-04. Its [binding](coupled_pilot_brief_sac_2026-11-04.json) records branch HEAD, original paths, sizes and SHA-256 hashes. The [coordination copy](../coupled_pilot_brief.md) adds our branch's log entry; the [response](../coupled_pilot_response.md) qualifies its interpretations.

The following exact saved summaries and current diagnostic sources were copied read-only. Their [bindings](saved_claim_bindings.json) preserve hashes and source paths. Neither copying nor inspecting these files reran their code or certified their original runs.

| Saved source | Observation and scope |
|---|---|
| [Action spectrum](action_spectrum_summary.json) | Full/low/high variances 1.400206/.439884/.903937; low-share .314157. Thus 1−low-share = 68.58%, while high/full = 64.56%. These correlated-band definitions differ. 10-neighbour R² .06999/.38202/−.07541. |
| [Label conflict](label_conflict_summary.json) | A distance-binned neighbourhood diagnostic, not a test covering all exact simulator states or hidden planner intents. |
| [Flat directions](flat_directions_summary.json), [current source](flat_directions.py) | 72 states; pooled share .548141 in three of six directions; geometric chance .5. Recorded reading: “it does not”. |
| [Ridge check](ridge_check_summary.json), [current source](ridge_check.py) | 72 states; 68.06% negative held-out; median fitted −18.7776 versus held-out −3.6582. Recorded reading: “mixed”. |
| [Mirrored variants A](mirrored_a_summary.json) | Six teacher episodes per cell: mirrored survival/walking 4/6, mirrored+effort 5/6; student walking rates zero. Unchanged baseline 5/6. |
| [Mirrored variants B](mirrored_b_summary.json) | Mirrored+policy teacher survival/walking 5/6, effort-only 6/6; student walking rates zero. Unchanged baseline repeats A and must not be counted as a fresh replication. |

The spectrum and conflict scripts match the existing immutable [spectrum source](../../../archive/coordination-2026-11-03/untracked/scripts/diagnostics/action_spectrum.py) and [conflict source](../../../archive/coordination-2026-11-03/untracked/scripts/diagnostics/label_conflict.py) hashes. Their command-line mains have simulator/policy dependencies and were not executed for this handoff. Their original 1,000-control corpus, burn-in, Welch window and fitted normalization differ from the shorter-loop analysis.

The flat-direction and ridge scripts reference an external raw tape:

`/Users/jaisel/Documents/mppi-gps/.claude/worktrees/sac-continuation/runs/2026-11-02_noise_cutoff_001/cutoff_23.4375hz/traces.npz`

44,935,839 bytes; SHA-256 `b5a712587a43f70cb42bf8374b5195ae9431bb6c7e013b94b996cce8c4a33369`. This tape remains outside Git. Current source copies are not asserted to be frozen launch-source identities. The original summaries preserve their own measured runtime fields; those costs are historical, not new work in this handoff.

No physics, neural inference, fitting, evaluations or experiment audit were run. The other branch and its untracked sources were not modified.
