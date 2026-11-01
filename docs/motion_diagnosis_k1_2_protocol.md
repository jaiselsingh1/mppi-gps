# K1.2 — actual saved motion and one ordered-segment probe

**Authorized diagnostic, prelaunch protocol, 2026-11-01.** The user explicitly authorizes preparing, freezing and executing this saved-motion diagnosis, its implementation/tests/audit/report/publication, then a next-protocol recommendation. No new policy inference, simulator integration, fitting, acquisition, checkpoint selection or automatic S1.2/coupling experiment is authorized. Commit and push the validated implementation and protocol before numerical motion decoding. Execute one create-only frozen worker.

## Questions and inherited evidence

[K1.1](trajectory_review_k1_1_results.md) established cancellation of **signed differences between nonnegative squared-error summaries**. It read no motion arrays, so equal episode means do not establish equal motion, repeated gait, changing coordination or matching subsequences. K1.2 inspects the actual sequences and asks:

1. What joint/body configuration, velocity and action patterns occur over each complete recorded trace? Are local ordered patterns repeated or changing?
2. Which local physical subsequences match within/across trajectories, including the already identified near-equal-score later-independent pair? Do matched physical segments have similar executed controls or different conditional reference errors?
3. What remains unresolved for useful policy learning and feedback that changes competent teacher optimization?

This is descriptive diagnosis on previously collected data, not a blind confirmation test or a proof of the original failure cause. No desired similarity ordering, recurrence threshold or positive-pattern gate is specified. Different successful gaits and nonperiodic behavior remain valid. A short repeated trace cannot establish a stable limit cycle; a neural policy need not have one attractor. There is no FFT, time warping, phase fit, transport objective or metric sweep. Similar local sequences cannot certify stability, recovery or teachability.

## Exact roster, sources and missing context

The [recipe](../configs/studies/motion_diagnosis.json) and [input catalog](../configs/studies/motion_inputs_k1_2.json) bind **all 34 K1 trajectories in original order**: 14 MPPI, 16 qualified-reference and four executed-random trajectories. There are **30,070 pre-action controls**. Native state19 is time + qpos9 + qvel9. Keep real unclipped velocities; clipped policy observations cannot reconstruct them. Action order is the six named motors in the frozen Walker XML. Reference predictions are the existing column8 of the nine-policy K1 matrix; verify each whole saved matrix before selecting the column. No network or checkpoint is loaded.

Sixteen exact original files total **1,301,839,980 bytes**: fourteen sealed teacher NPZs plus K1 stage-A/B HDF files. Their whole-file SHA-256 values and every selected array shape/dtype/C-byte SHA are declared before decoding. The authoritative earlier file catalog and K1 capsule/config/result/input seal are also frozen as metadata. File SHA, C-contiguous array-byte SHA and policy hash remain distinct contracts.

The worker decodes **160 selected arrays / 16,331,200 bytes**: physical states, fourteen row-index arrays, ten separate terminals and 34 sets of predictions/actions/observations. Four discovery episodes have N pre-action states only. Ten later MPPI episodes have N plus separately stored terminal states. All twenty K1 reference/random episodes have N+1 states. Preserve **30 available endpoints and four missing endpoints**. No warmstart/integration/contacts/body-position reconstruction is required or claimed. Do not synthesize missing endpoints or post-fall observations.

Verify the physical-state projection `concat(state[:,2:10], clip(state[:,10:19],−10,10))` byte-exactly against saved native observations; match executed actions and prediction hashes, physical row indices and original K1/K1.1 error bins. Time increments and terminal increments must equal 0.008 seconds within absolute 1e−10, relative0. Array identities/ownership are exact; existing error reductions reconcile at absolute/relative 1e−12. Derived segment recomputation uses absolute/relative 1e−10. No tolerance may be relaxed after launch.

No student executions are added. The primary question concerns the actual 34 motions already compared in K1; adding S1.1 traces would enlarge this diagnostic without being required to inspect them. Conditional predictions on these recorded states remain distinct from separately executed student/reference behavior. K1.2 cannot establish learner-induced distribution shift from this roster.

## One fixed probe

For physical states \(x=(t,x_{\rm root},z,q_1,\ldots,q_7,v_1,\ldots,v_9)\), define

\[
d^2(x,y)=\frac1{17}\left[
 (\Delta z/1\mathrm{m})^2+
 \sum_{j=1}^7(\operatorname{wrap}_{[-\pi,\pi)}\Delta q_j/1\mathrm{rad})^2+
 \sum_{j=1}^9(\Delta v_j/s_j)^2\right],
\]

where the two slide-velocity scales are 1 m/s and the seven angular-velocity scales are 1 rad/s. All trajectories share these fixed conventions; no scale is fitted. Drop clock and absolute root-x only. Forward velocity stays present, so forward translation does not itself prevent recurrence and standing is not equated with walking. Preserve unwrapped native angles in plots; wrap only relative hinge differences in this distance. Report height/angle/velocity contributions to expose coordinate dominance without retuning it.

For an ordered segment beginning at control \(a\) in trajectory \(i\), and \(b\) in \(j\):

\[
R_{ij}(a,b)=\sqrt{\frac1H\sum_{\ell=0}^{H-1}
d^2(x^i_{a+\ell},x^j_{b+\ell})}.
\]

Use **25 controls (0.2 s), stride25**, on the grid 0,25,… where a complete window exists. This is a local pattern probe, not an assumed gait period. Within-trajectory pairs require \(|a-b|\ge100\) controls (0.8 s), excluding self, overlap and nearby smoothness. Across trajectories evaluate every window pair. Use original roster/start order; ties resolve lexicographically `(R, first_start, second_start)`.

For N<25, retain one actual-length prefix. Pair it with the corresponding prefix of the other's eligible segment, \(H=\min(H_i,H_j)\). These **H<25 comparisons remain separate** from full25 minima, images and summaries. The N26 random trace contributes one full25 window; its final control is still in the motion curves. All four short traces remain visible and cannot be labeled nonrecurrent merely because they have no eligible within pair.

The frozen census is **1,204 probe segments, 561 cross-trajectory records, 19,980 within and 700,806 cross-window evaluations: 720,786 total pair evaluations / 17,982,374 aligned state pairs**. Evaluate only eligible pairs, in batches ≤1,536, and journal requested/completed work before/after each batch. Store complete within matrices and all cross-pair minima, their actual starts/horizons, squared-distance contributions, matched executed-action MSE and each segment's own saved-reference error. The action/error quantities do not select the physical match. No global teachability ranking follows.

Rigid start-grid matching permits a declared time offset between segments; there is no time rescaling, interpolation, phase label, leg reflection or optimized alignment. Nearest matches among many windows are descriptive selected minima, not unbiased similarity estimates or independent task replications. Shared reset ancestry or a common initial posture can dominate a minimum; retain starts and the first100 view rather than discarding startup windows after observing results. Two source episodes remain two episodes. Static poses can recur too; task outcomes are reported separately.

## Motion evidence, controls and reporting

Inspect **every recorded control** in a 34-page full-motion atlas: root displacement, height/pitch, six joint angles, nine velocities, six executed controls and per-control conditional reference errors. Include the first100 controls separately, terminal markers only where available, and existing error profiles. This uses actual generalized joint/body coordinates; no articulated pose is invented and no simulator forward pass/render/integration is performed.

Publish compact 25-control mean/min/max curves and their actual widths for all coordinates/actions/errors, recurrence maps, full/partial cross matrices, all561 pair records and a predeclared prominent view of the later-independent pair. Full raw NPZs stay outside Git. A clean capsule can regenerate derived-curve views and copy the hash-bound original full atlas; it must explicitly distinguish that mode from full-raw re-rendering. Short failed episodes remain short, and missing later time stays missing.

Synthetic regression controls cover self identity, root-x translation invariance, 2π hinge-offset invariance, order changes with the same marginal states, lag/nonoverlap masks, partial/full separation, deterministic ties, row/source joins, N/N+1 endpoints, finite limits and paid failed-work retention. Reference-self conditional zeros are an inherited plumbing control, not evidence of generalizable learning. No real motion permutation or additional diagnostic objective is introduced.

## Finite budgets, validity and stopping

| Stage | Frozen ceiling |
|---|---:|
| Scientific worker, external owner deadline | 600 seconds, one CPU thread |
| Worker observed peak RSS | 512 MiB |
| Original-file hash reads | 2 GiB |
| Logical source file-object decoder reads, including headers/compressed bytes | 256 MiB |
| Decoded selected arrays | 32 MiB; exact successful census above |
| Complete new run including source, inputs and raw export | 512 MiB |
| Independent audit | 600 s / 512 MiB; 2 GiB hash, 256 MiB decoder, 64 MiB decoded ceilings |
| Report | 600 s / 512 MiB; 64 MiB compact-raw read and 128 MiB output ceilings |
| Combined original-source reads including recorded preflight | 8 GiB |
| New inference, physics, gradients/Adam, MPPI, GPU, checkpoint loads | 0 each |

Pin Python3.11.15/NumPy2.4.3/h5py3.16.0 on CPU. Whole-file hashing precedes numerical decoding; selected-array hashes precede reductions. Preserve original files and check their identity/size/timestamps around the read. Decoder counters measure returned logical bytes, not physical disk/cache traffic. RSS is an observed guard, not OS isolation. The owner kills/reaps its process group at the deadline. There is no retry/resume or trajectory/source replacement.

Stop on any source/hash/shape/dtype/order/native-observation/time/error-reconciliation/finite/budget/ownership mismatch. Preserve every failed attempt, raw/partial export and requested/completed journal. Incomplete CPU/IO work is unknown within its recorded request bounds, not completed zero work. No missing context is repaired with new integration.

Scientific success means complete readable evidence and an independent audit that reopens original saved sources and recomputes all arrays/joins/curves/evaluations/minima/costs without importing production analysis, policy or simulator code. Null, inconsistent and unhelpful patterns are valid outcomes. Separate original acquisition/fitting/qualification, preflight metadata reads and failed path lookups, worker reads/arithmetic, audits, both scientific and engineering failures, report/transfer/storage costs and unknown overhead. Do not sum nested wall times twice or claim an efficiency advantage.

## Commands and closure

```sh
python -B scripts/train_mppi_gps.py study motion-diagnosis prepare --source-root /Users/jaisel/Documents/mppi-gps --out /absolute/fresh/run
python -B /absolute/fresh/run/source/scripts/train_mppi_gps.py study motion-diagnosis run --out /absolute/fresh/run
python -B scripts/audits/motion_diagnosis.py --run RUN --out NEW_EXTERNAL_AUDIT --config-sha CONFIG_SHA --seal-sha SEAL_SHA
python -B scripts/diagnostics/report_motion_diagnosis.py --run RUN_OR_CAPSULE --out NEW_EXTERNAL_REPORT
python -B scripts/train_mppi_gps.py inspect --run RUN_OR_CAPSULE --verify
```

Audit/report destinations must be outside closed run/capsule evidence. Existing runs, K1/K1.1 metrics/configurations, failed/successful audits and archives remain immutable. Use normal `origin/main` pushes and the established jaisel author/committer identity; verify remote HEAD. Keep the four-note Obsidian structure and add only a short conclusion/link plus the research intent, with exact backups.

Close with established observations, explanations supported within this assay, unresolved causes and **one justified next protocol recommendation**. S1.2 isolates visitation under a fixed reference label function; it does not adapt the teacher and must not be the default merely because drafted. If not the most discriminating choice, prepare a separately versioned specific teacher-adaptation proposal. Keep competent search softly influenced by the learner central, followed by a fixed update, useful standalone outcomes and updated-policy feedback into another actual optimization. Account for existing along-candidate selection and H4.1's negative children. Do not relabel candidate reweighting as a new solver or assume a weak student should veto competent discovery. No follow-up training, acquisition or coupling launches automatically.
