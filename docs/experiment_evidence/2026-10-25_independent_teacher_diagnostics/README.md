# Independent-teacher and frozen-critic diagnostics

Two bounded diagnostics completed without policy or critic updates. They preserve the adverse cases and remain descriptive; no new learner design, training run, exploration claim, or sample-efficiency conclusion follows from them. The completed 75-iteration study is unchanged.

## Independent raw MPPI probe

The preregistered runner used clean revision `e63f9e1024b9a834fef0ceccb6858d7586f01403`. Raw MPPI retained its own shifted plan and used K=32, H=48, lambda=3, sigma=0.3, two initial passes, and the existing 23.4375 Hz low-pass noise at 125 Hz. It used neither the student's nominal nor its continuation cost. Three teacher episodes used reset seeds 8900000–8900002 and planner seeds 8910000–8910002, capped at 300 controls. Anchors were fixed at controls 0, 75, 150, and 225. One teacher fell at control 195; its missing control-225 anchor was retained as missing, leaving 11 correlated anchors. The other two teacher episodes reached the administrative cutoff healthy.

| Check | Observed result |
| --- | --- |
| Exact replay after initial/final actor loading and external rollouts | 11/11 anchors exact; all checked action/sample-cost/sample-plan/mean-plan/nominal/weight differences zero |
| Weighted mean plan costs more than incoming nominal over H=48 | 3/11 |
| Weighted mean plan costs more than best final-pass sample over H=48 | 3/11 |
| Short/long first-action rank reversals with initial continuation | 1/11 |
| Short/long first-action rank reversals with final Selective continuation | 2/11 |
| Eight-second survival, initial actor first / raw MPPI first, then initial actor | 0/11 / 0/11 |
| Eight-second survival, final Selective first / raw MPPI first, then final Selective | 6/11 / 5/11 |

Every observed rank reversal favored the raw teacher action **less over 48 controls but more over 1,000 controls**, with the named actor continuing after the forced first action. Thus these reversals do not demonstrate a greedy teacher action that looks better short-term but causes worse long-term outcomes. Initial-continuation reversal: anchor 7; final-Selective reversals: anchors 0 and 5. Entire open-loop plan scores and forced-first-action/actor-continuation scores are different quantities. Costs include one terminal cost at a fall and an absorbing cost of 5 per missing control. The 11 anchors come from three teacher trajectories; the survival fractions are not independent-reset reliability estimates. Wall time was 67.63 s.

## Frozen historical critic continuation probe

The separate runner was preregistered in revision `f2215d4`. It froze the historical seed-1 SAC actor/twin critics at training step 200,000, trained with a 25% MPPI stream and actor tails; it did not train a critic in a new independent-teacher replay loop. Eight input rows `[0,1,2,4,5,7,8,10]` were selected uniformly before critic/action/continuation outcomes from the recorded independent-teacher snapshots. Each state compared its recorded raw MPPI first action with the historical actor's mean first action, followed by that **historical actor's stochastic continuation**, not the final Selective network.

There were eight paired stochastic replicates per state/action, at most 2,500 controls each; gamma=0.995, alpha=0.08825156092643738, seed base=9420000. The target was `r0 + sum(t>=1) gamma^t * (r_t - alpha*log pi(a_t|s_t))`, with no entropy at the fixed first action. The environment's analytic absorbing failure reward was charged once; no postfall entropy or bootstrap was added. Healthy truncations have no critic tail.

Differences are historical actor mean action minus recorded independent-MPPI action; higher soft return is better. The intervals below are descriptive normal approximations for Monte Carlo uncertainty (`mean ± 1.96 paired SE`), not population confidence guarantees.

| Input row | Twin-min Q difference | Paired MC mean difference | Paired MC SE | Approximate 95% MC interval |
| --- | ---: | ---: | ---: | ---: |
| 0 | +2.229 | +23.559 | 16.882 | [-9.530, +56.648] |
| 1 | -0.536 | -124.026 | 67.837 | [-256.987, +8.935] |
| 2 | +4.272 | +32.402 | 43.615 | [-53.083, +117.888] |
| 4 | +3.397 | +19.463 | 26.633 | [-32.738, +71.664] |
| 5 | +7.874 | +34.478 | 148.234 | [-256.062, +325.017] |
| 7 | +2.948 | +13.860 | 38.006 | [-60.631, +88.351] |
| 8 | +5.369 | +9.589 | 26.280 | [-41.920, +61.098] |
| 10 | +0.804 | -35.210 | 50.168 | [-133.540, +63.119] |

The signs of the critic and finite-MC mean differences agree at 7/8 states. **None of the eight paired MC ranking differences excludes zero in this normal approximation.** Sign agreement alone therefore does not establish good ranking accuracy, and the disagreement does not establish a resolved critic error. Eight repeats leave substantial calibration uncertainty.

| First-action branch | Mean min-Q minus MC mean | RMSE against finite MC means | Healthy 2,500-control cutoffs |
| --- | ---: | ---: | ---: |
| independent_mppi | +30.434 | 91.835 | 15/64 |
| actor_mean | +36.964 | 71.761 | 21/64 |

These bias/RMSE figures are point estimates against noisy finite MC means; they do not isolate critic error from MC sampling error. Healthy cutoffs occurred in both branches. The largest residual discount weight was `3.611618461290886e-6`; it is a weight, not a bound on omitted soft return. The run executed 173,349 continuation controls, made zero new teacher searches, finished in 31.84 s, and reports `source_unchanged=true`. These correlated teacher-state probes do not establish cross-seed calibration, out-of-distribution reliability, or a learner-design decision.

## Provenance and preserved local evidence

Raw artifact checksums and both executed-source inventories were verified against local files. Paired MC differences/standard errors and bias/RMSE summaries above were independently recomputed from the saved replicate rows. The shared task SHA-256 is `546f6902405daaddb37fb9c458d77fb5159234d1218ec73abf9a625ec9f8f27a`; Walker XML SHA-256 is `4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590`. Server versions were Python 3.11.16, MuJoCo 3.6.0, NumPy 2.4.3, and Torch 2.10.0.

| Input / executed source | SHA-256 |
| --- | --- |
| Initial tanh-policy checkpoint | `b05c727765f0e2fed0032b633a048e53abab3d1d0130fea743ee0da0932cee5d` |
| Final Selective checkpoint | `49e72d0e2db46a7202d517c9093f6412f75c70152846c83688f2740e1823d648` |
| Historical SAC step-200000 checkpoint | `68a10f86600d6797614dc7e8461dbde8985ca7658333928bce559e11b434a2e3` |
| `scripts/diagnostics/independent_teacher_probe.py` | `97dc3dac6b8466df614a9a0c92f62b0c8c4858939b0fb7d4f1243eb824233848` |
| `scripts/diagnostics/critic_continuation_probe.py` | `f4bb299dc4149d63535c2d0bb62062959ae302b3e995285c8532064046c12bc8` |
| Raw snapshots NPZ | `f8e24a9564b4b50fcdbeefd55a88e4e9460fd3a047ea39a005eda7d98fb025d9` |
| Raw protocol | `b4a5914d34fd0b9de12b0adf7d41c9ccb8aef8d898d158bc9bbebd88b2aab648` |
| Raw report | `b1422127cf8097bb71b1b128caa097b6bb55e19c7a6860cb01af6629186e5883` |
| Critic protocol | `223914e74da7f01e93f77fc23309e3151f32d1a27f97c2fb267996cbd4f35984` |
| Critic comparison | `4a45884c70a9d743fbc41a274d9c6c12197271906232247e48362aaef69d049b` |

Generated JSON/NPZ artifacts remain local and ignored under the [artifact policy](../../artifact_layout.md). Complete source/configuration inventories and all branch outcomes are preserved in these local files:

- Raw probe: [protocol](../../../runs/2026-10-25_independent_teacher_probe_001/protocol.json), [report](../../../runs/2026-10-25_independent_teacher_probe_001/report.json), [snapshot metadata](../../../runs/2026-10-25_independent_teacher_probe_001/snapshots.json), [snapshots](../../../runs/2026-10-25_independent_teacher_probe_001/snapshots.npz), and [query/branch traces](../../../runs/2026-10-25_independent_teacher_probe_001/query_and_branch_traces.npz).
- Frozen critic: [protocol](../../../runs/2026-10-25_critic_continuation_probe_001/protocol.json), [comparison](../../../runs/2026-10-25_critic_continuation_probe_001/comparison.json), [replicate cases](../../../runs/2026-10-25_critic_continuation_probe_001/cases.json), and [completion status](../../../runs/2026-10-25_critic_continuation_probe_001/status.json).

The learner design remains undecided; no subsequent training study was launched.
