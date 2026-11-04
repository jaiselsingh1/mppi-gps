# Completed L3.1 evidence — audit failed

All24 stream-rounds completed. Every final16-reset panel had zero walking; two
coupled episodes survived1,000 controls without meeting walking. The α=0 arm is
**static refitting**, not fresh discovery. See the [one-page result](../../../docs/local_coupling_multiround_l3_1_result.md).

- [All192 final episode results and bindings](final_outcomes_verified_from_saved_results.json)
- [300-control local teacher outcomes](local_teacher_summary.json)
- [Complete saved accounting](complete_saved_accounting.json)
- [Failed independent audit](audit/findings.json), [receipt](audit/receipt.json), [representation explanation only](audit/probe_representation_note.json)
- [Standalone curves](report/standalone_trajectory.png), [teacher curves](report/teacher_trajectory.png), [fitting curves](report/fitting_trajectory.png), [reconstruction](report/reconstruction_trajectory.png), [student-state gap](report/student_state_gap_trajectory.png)
- [Per-episode outcomes](report/standalone_outcomes.csv), [per-panel summaries](report/standalone_summary.csv), [teacher outcomes](report/teacher_outcomes.csv), [every admission/fallback](report/admission_fallback.csv), [activity](report/acquisition_activity.csv), [reconstruction rows](report/reconstruction.csv), [gap endpoints](report/student_gap_endpoints.csv)
- [Every fitting update, gzip](report/fitting_curves.csv.gz), [declared criteria](report/declared_final_criteria.json)
- [Complete report source bindings, gzip](report/continuation_manifest.json.gz), [external raw-array/checkpoint bindings, gzip](external_raw_bindings.json.gz)

Every copied artifact is byte-exact to its saved source. Gzip entries include
the original decoded SHA256 and byte length in `publication_manifest.json`.
Original rendered Markdown is preserved compressed under `report/`; this README
provides working navigation to the compact published artifacts. All24 joint-angle
figures are preserved as `report/joint_angles_r{1..8}_s{0..2}.png`; the original
report marks these unpaired illustrations, including failures.

Large arrays, ledgers, checkpoints and frozen source remain local-only under
`runs/2026-11-04_local_coupling_multiround_l3_001` and
`runs/2026-11-04_local_coupling_multiround_l3_1_001` in the worktree declared in
the external binding index. The earlier capsule files and failures are unchanged.
This publication aggregates saved metadata, verifies copied bytes, and runs no
physics, neural inference, fitting, evaluation or independent audit.
