# Literature evidence ledger

Audit date: 2026-10-23. This directory records primary-source retrieval, reading scope, search protocol, code pins and unresolved access gaps. Downloaded papers were cached outside the repository at `/tmp/mppi-literature/`; source URLs and SHA256 hashes allow retrieval without adding all papers to the git tree. The inspected papers and tracked files from the three pinned official code repositories are also preserved in a private server archive; [material_archive.json](material_archive.json) records its path, contents and verified hash. Initial download manifests preserve failed attempts; `source_read_ledger.json` records the final successful sources and actual reading status.

## Reading standard

“Full text read” means extracted main text, equations, references and included appendices were read. It does **not** mean all plotted data were digitized, every theorem independently proved, every experiment replicated, or visual supplements watched. Numerical claims in the review use explicit prose/tables; plot-only comparisons are labeled accordingly. The final TD-M(PC)² paper was read in addition to the older preprint, not assumed identical. No separate appendix was present in the supplied eight-page VAMPS or SGPS PDFs. MDGPS's five-page supplement was retrieved from the official linked ZIP and read separately.

The hand-CIO 2012 PDF was retrieved but **not read fully**; it is a backward-reference lead, not a basis for the matrix. CoVO-MPC was subsequently read in full (14-page proceedings version plus 32-page arXiv version including Appendices A–C), as was the eight-page PL-MPC preprint. Their code has not been audited. The independent [Mordatch2015 crosscheck](../mordatch2015_review.md) records a teammate's full paper and five-review/rebuttal read, with its own source manifest.

## Queries actually executed

Queries were grouped into tool batches, with adaptive follow-ups. Search results were used for discovery; accepted evidence was then read at its primary location.

1. `"VAMPS" "2026" reinforcement learning`
2. `"SGPS" "guided policy" 2026`
3. `"PO-MPC" reinforcement learning`
4. `"TD-M(PC)"`
5. `"ProxPI" reinforcement learning`
6. `"BMPC" reinforcement learning policy 2026`
7. `"Jaisel" "CIO"`
8. `"Jaisel" "guided policy search"`
9. `"Mirror Descent Guided Policy Search" paper`
10. `"Path Integral Guided Policy Search" Chebotar`
11. `"Mordatch" "Todorov" "2014" neural network`
12. `"Jaisel Singh" research CIO optimization`
13. `Mordatch Todorov 2014 combining benefits function approximation trajectory optimization pdf`
14. `"CIO" "policy" "Singh" optimization`
15. `"POLO" "policy" "Lowrey" 2019`
16. `"Maximum a Posteriori Policy Optimisation" 2018 arxiv`
17. `"POLO" Lowrey learning online planning offline`
18. `"CIO" "contact" optimization Mordatch`
19. `"Guided Policy Search via Approximate Mirror Descent" arxiv`
20. `"TD-MPC2" arxiv 2024`
21. `"VAMPS" "Kabouri" github`
22. `"Sampling-Guided Policy Search" github`
23. `"A KL-regularization framework" github`
24. `"Bootstrapped Model Predictive Control" OpenReview`
25. `"VAMPS" "Kabouri" code github openreview`
26. `"Accelerating Visual Policy Learning" code github openreview`
27. `"Bootstrapped Model Predictive Control" reviewer weaknesses`
28. `"A KL-regularization" "OpenReview" review`
29. `"TD-M(PC)" "OpenReview"`
30. `"Synthesis of Complex Behaviors" "Contact-Invariant" pdf`
31. `"Guided Policy Search" "a00e5eb" supplemental`
32. `"Guided Policy Search via Approximate Mirror Descent" supplementary appendix`
33. `"ProxPI" code github`
34. `"TD-M(PC)" OpenReview reviews`
35. `"COVO-MPC" official paper code`
36. `"Guided Policy Search via Approximate Mirror Descent" "VAMPS"`
37. `"Path Integral Guided Policy Search" "MPO"`
38. `"Plan Online, Learn Offline" "Bootstrapped"`
39. `"Combining the benefits of function approximation" "SGPS"`
40. `site:openreview.net/forum "TD-M(PC)" "Improving Temporal"`
41. `site:openreview.net "ph7BXPU08c"`
42. `site:openreview.net/forum "Bootstrapped Model Predictive Control"`
43. `site:openreview.net/forum "A KL-regularization Framework for Learning to Plan"`
44. `"CoVO-MPC" "PMLR"`
45. `"PL-MPC" "pl-mpc-humanoid" code`

Identity searches (7–8,12,14) were stopped after the user clarified these were not their publications. Query30's guessed wording was corrected using the primary paper to **Discovery of Complex Behaviors through Contact-Invariant Optimization**. Acronym exclusions included the VAMPS mathematics benchmark, partially observable driving PO-MPC, Branch MPC/BMPC, and the unrelated `proxpi` Python package. D.VA code was not treated as SGPS code. Secondary review sites were not used as reviewer evidence.

## Official code read ledger

| Repo | Commit | Files inspected |
|---|---|---|
| `wertyuilife2/bmpc` | `d5dabc6cb58d463829571548c47711dff074f5af` | Full `bmpc/bmpc.py`, `bmpc/trainer/online_trainer.py`, `bmpc/config.yaml`; relevant `common/math.py` Gaussian KL and `common/world_model.py` actor; README/result filenames. |
| `alvaro-serra/pompc` | `e0e1efea2f1e33ea98568ace960641d19b36a770` | Full `pompc/pompc_jax.py`, `pompc/config.yaml`; `world_model.py:279–354` density and sampling; targeted training/reanalysis/evaluation/seed paths in `pompc/train.py`. Independent agent ran official `paper_results/data_load_plots.py` on released per-seed data. |
| `DarthUtopian/tdmpc_square_public` | `d1c2632c36effd2f7b661bfe5f822a3db8054d40` | Full `tdmpc_square/tdmpc_square/tdmpc_square.py`, matching config, root README; targeted `trainer/online_trainer.py` planner/actor/value evaluation and replay fields. Bundled TD-MPC2 baseline located to avoid mistaking it for the method. |

These are available-code comparators, not a claim that VAMPS code was read. Version differences matter: PO-MPC default planner temperature1.0 versus paper-table .5, default six iterations versus high-dimensional overrides, and default policy min log-std−2 require exact run configuration before replication. The executed density check in `external_pompc_density_check.py` records a failing comparison against an independent Normal-density formula. It executes the original upstream method body with deterministic arithmetic stand-ins, not a separately retyped implementation, and leaves upstream code untouched. It does not validate JAX gradients or GPU execution.

## Reviews and access failures

| Official endpoint | Observed outcome | Status |
|---|---|---|
| `https://openreview.net/forum?id=i7jAYFYDcM` (BMPC) | Browser verification challenge | Reviews/response not read |
| `https://openreview.net/forum?id=gj8Dw4hhDA` (PO-MPC ICLR submission) | HTTP429/verification challenge | Reviews/response not read |
| `https://openreview.net/forum?id=zO8vzSGgTn` (PO-MPC ICML record) | Browser verification challenge | Reviews/response not read |
| `https://openreview.net/forum?id=ph7BXPU08c` (TD-M(PC)² submission) | Browser verification challenge; submission title confirmed from indexed official PDF, discussion inaccessible | Reviews/response not read |
| API2 `notes?forum=...&limit=1000`, for BMPC, PO-MPC and TD-MPC2 | HTTP403 | No comments retrieved |
| `https://proceedings.neurips.cc/paper/2016/file/a00e5eb0973d24649a4a920fc53d9564-Reviews.html` | Accessible official NeurIPS reviews | Read; historical context only |

No author-linked VAMPS, SGPS or ProxPI implementation was located in the inspected primary links or targeted searches. This is a bounded search result, not proof that none exists anywhere. No OpenReview review text for those three was found. The closest-competitor implementation/review requirement and fresh training reproduction therefore remain unresolved.

A final direct-browser fallback was attempted for the BMPC forum through `mcp__cua_repl`: `cua.getBrowser({url: ...})` returned **“No browser is available”**, and `cua.listBrowsers()` returned `[]`. No page loaded and no review text was obtained. [The exact attempt record](browser_access_attempt.json) distinguishes this unavailable browser surface from the earlier HTTP/verification failures. No challenge bypass was attempted.

PL-MPC's advertised project page returned an access error in the web tool; the paper says code/data will be released. This audit does not claim to have inspected them. CoVO-MPC's official page links code, but the additional review here covered papers and an independent mathematical check, not code. `covo_norm_counterexample.py` checks the final Theorem1/arXiv AppendixA.1 Euclidean-norm assertion for noncommuting positive-definite matrices. It demonstrates norm expansion with cost decrease, without refuting the commuting covariance design or claiming to reproduce a benchmark.

## Reproduction boundary

See `../reproduction/pompc_published_number.json`: official released-run reanalysis matches PO-MPC Table1 H1hand-walk λ9, IQM934 [544,936]. It does not constitute retraining or VAMPS replication. Source-file hashes and an independent density-discrepancy check are separate evidence, and neither implies benchmark result correctness.
