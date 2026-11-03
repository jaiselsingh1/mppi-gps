# Retrospective SAC diagnostic evidence review

These files preserve the three previously unpublished diagnostic summaries and figures, plus the currently observed source worktree snapshot. They were collected after execution. **This capsule is not a launch-frozen protocol or an independent audit of the original training and physics.** Missing checkpoints, optimizer/RNG packages, minibatch tapes, executed student traces and original work census remain missing.

- [Preservation and availability inventory](preservation.json): exact source paths, hashes, original Git state and missing evidence.
- [Positive-control summary](runs/2026-11-02_student_positive_control_001/summary.json), [jitter summary](runs/2026-11-02_jitter_tolerance_001/summary.json), [on-policy summary](runs/2026-11-02_on_policy_disagreement_001/summary.json): original bytes.
- [Corrected saved-summary analysis](reanalysis/summary.json) and [execution receipt](reanalysis/receipt.json): average-tie Spearman and explicitly defined exploratory AUC. Original statistics remain recorded separately.
- [Original SAC continuation report](original_docs/sac_continuation_results.md) and [protocol](original_docs/sac_continuation_protocol.md): historical bytes. The separate original [SAC capsule](../sac_continuation_001/capsule.json) is unchanged.

The on-policy AUC labels a fitted student positive if it had any walking episode among 32 resets. It does not classify competence, validate a protected prediction, or establish before-update teachability. Its original workflow refitted actors and checked one held-error scalar; exact checkpoint equality was not established. Jitter tests fitted fresh actors rather than fine-tuning competent parents.

The canonical saved-only command is:

```bash
python -m src.gps.mppi_gps.reporting.sac_diagnostics_review --capsule results/sac_diagnostics_review_001 --output /tmp/sac-review-fresh
```

The output directory must be fresh. The command reads three sealed summaries and executes no model, fitting or simulator operation. [The implementation used for this review](reanalysis/implementation.py) and [its input capsule manifest](reanalysis/input_capsule.json) are preserved. All 24 originally untracked diagnostic scripts plus their committed `sac_continuation.py` dependency are under `source/`; they are historical source, not newly endorsed execution entry points.
