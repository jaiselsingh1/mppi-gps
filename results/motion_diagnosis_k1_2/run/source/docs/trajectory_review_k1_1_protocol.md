# K1.1 — temporal resolution of fixed-reference action agreement

**Prelaunch protocol, 2026-11-01.** This is a bounded analysis of saved K1 measurements. It changes neither the learner nor the planner. The checked-in [recipe](../configs/studies/trajectory_review.json) is authoritative for exact input hashes, runtime, values and ceilings. Commit and push this implementation and protocol before preparing the create-only run. No new experiment follows automatically.

## Question and scope

K1 already establishes the qualified reference's action-MSE ordering: self < MPPI < random. The [fixed-reference review](action_compatibility_k1_results.md#fixed-reference-trajectory-spread-review) retains each trajectory's score. The next question is narrower: **do close whole-trajectory scores conceal different error profiles over the recorded control sequence?** K1.1 decomposes the existing score; it does not introduce a new similarity metric, choose a desired ranking, test teachability, or identify causally important time steps.

All data have already been viewed. This is descriptive numerical diagnosis, not a blind confirmation test. Individual trajectory identity, ancestry, source, original order and outcomes remain visible. No bootstrap, hypothesis test, inferred independent frames, gait-phase alignment, time warping, selected favorable episodes or per-trajectory color normalization is permitted.

Kendall's clarification is retained: the general question concerns behavior unfolding over time; repeated gait is only a Walker-specific possible source of error. Nonperiodic solutions and different successful gaits remain valid. This scalar-bin audit cannot identify repeated motion, transitions between patterns, phase relationships or matching state–action subsequences. Inspect actual saved motions/sequences before interpreting those possibilities. Spectral analysis is a possible later diagnostic, not a selected metric or objective. Short repetition does not establish a stable limit cycle, missing repetition does not establish unlearnability, and a neural policy need not have one attractor. Conditional action agreement on recorded states is distinct from comparing separately executed behaviors; similar spectra would not certify coordination, recovery, competence or teachability.

## Fixed inputs and controls

Copy nine exact files from an explicitly supplied repository/evidence root: K1 capsule, config, result, trajectory-policy matrix, source-control curves, actuator errors, saturation, group means, and the existing fixed-reference spread JSON. Their total is **8,328,586 bytes**; all byte lengths and SHA-256 values are in the recipe. These files contain saved scalar summaries, descriptors and bins. Do not open their raw-array/checkpoint descriptors or import a model, simulator or planner.

The actor is the same deterministic SAC reference, policy hash `e1ec300c39d6880f893995012fa7c5a0789f9ef0b931280454310b57e62c9cba`, checkpoint SHA-256 `174e988618a478d41b846cf428a63b91fef5a0a0f94ae0b36637708856ef363f`. It is a qualified walking reference, not a proven optimal policy. Native clipped 17-dimensional observations and six normalized torque outputs are inherited unchanged. No fresh predictions are requested.

Retain **34 trajectories, 30,070 observed control rows, 304 bins and 204 whole-trajectory actuator errors**:

- Four original-discovery and two each of later independent, later guided, H4 independent, H4 fixed-high and H4 backoff MPPI trajectories: 14 successful 1,000-control trajectories.
- Sixteen qualified-reference 1,000-control trajectories. Deterministic self-comparison is a zero-error plumbing control, not evidence that another student can learn the behavior.
- Four executed random-action trajectories of lengths 8, 26, 22 and 14 controls. Keep their partial first bins; all later observations are missing. These are not duration-matched controls for an episode sum.

Also retain the original nine-policy matrix as provenance; only the fixed reference's column is analyzed here. Check every fixed-reference CSV join against the original result and spread record. Reconcile historical source-curve aggregation separately from individual trajectories. The source list remains eight groups. A pooled 14-MPPI summary is explicitly a descriptive mixture, not 14 independent task replications. Two-trajectory source groups are only two trajectories.

## Variables and reductions

For recorded pre-action states and executed actions, the unchanged K1 score is

\[
E_i=\frac{1}{6N_i}\sum_{t=0}^{N_i-1}\sum_{j=1}^{6}
  \left[u_{itj}-\pi_{\rm ref,j}(o(x_{it}))\right]^2.
\]

The saved 100-control bin means are \(q_{ib}\). Use every observed row, including the fall-causing action. There is no added support mask, post-fall prediction or padding. Report \(D_i=6N_iE_i\) as an algebraic reduction of saved \(E_i\), not a re-created raw summation or duration-independent score. For MPPI, all \(N_i=1000\), so \(D_i=6000E_i\). That equivalence does not hold across unequal random durations.

At each observed bin end \(n\), report the row-weighted cumulative mean

\[
\bar E_i(n)=\frac{\sum_{b:\,\mathrm{stop}(b)\le n}n_{ib}q_{ib}}{n}.
\]

Do not interpolate within a bin. Bin standard deviation describes variation of bin means, not per-control noise. Time is absolute control index; one control is 0.008 seconds. No conclusion about causal temporal importance follows from a higher bin error or earlier clock time.

For a balanced source group with ten equal bins, \(E_i=B^{-1}\sum_bq_{ib}\), \(B=10\). Using population moments over the finite recorded roster (`ddof=0`), calculate

\[
V_{\rm full}=\operatorname{Var}_i(E_i),\qquad
V_{\rm bin}=B^{-1}\sum_b\operatorname{Var}_i(q_{ib}),
\]
\[
V_{\rm diag}=B^{-2}\sum_b\operatorname{Var}_i(q_{ib}),\qquad
V_{\rm off}=B^{-2}\sum_{b\ne c}\operatorname{Cov}_i(q_{ib},q_{ic}),
\]
\[
V_{\rm full}=V_{\rm diag}+V_{\rm off}.
\]

This identity distinguishes ordinary averaging from covariance between bin positions. A negative off-diagonal term indicates cancellation of opposite local differences in this finite dataset. It does not establish phase differences, chaotic sensitivity or learning difficulty. For each balanced two-trajectory group, additionally show first-minus-second full and signed bin differences, retaining the original two IDs. Do not sort them by error or introduce an acceptance threshold or cancellation ratio. Random partial trajectories are ineligible for this balanced decomposition. The self control has zero variance; undefined error shares remain null.

Retain all six saved per-actuator MSEs and their shares of total squared error. Reconcile their mean with \(E_i\). Show saved marginal RMS, absolute magnitude and saturation (`|u| >= 0.95`) separately for recorded actions and predictions. These marginal summaries cannot establish whether saturation caused or coincided with a prediction error; signed residuals and conditional associations are unavailable in these inputs.

## Controller, targets and optimization feedback

There is **no execution controller, learning target or feedback into trajectory optimization in K1.1**. The direct width128 torque policy, normalization, Adam/RNG states, targets and all previous gates remain unchanged. This is not a new MPPI–GPS mechanism.

The [implementation map](code_map.md) and [methods](methods.md) describe the current agreement term: it evaluates the actor at each candidate's own pre-action states and adds a masked sum over a finite horizon to the native candidate cost before selection. The independent candidate producer does not receive the actor. Guidance changes selection and subsequent executed states; it does not insert the policy term inside the independent producer's refinement, optimize a full episode jointly, or establish that a similar trajectory is teachable. The torque policy also lacks CIO–GPS's desired state-to-state controller/analytic feedback targets. Failed action fitting is not a failed CIO–GPS reproduction.

## Frozen budget, success and stopping rule

One create-only run, no retries or resume:

| Item | Ceiling |
|---|---:|
| Analysis owner deadline, including input/source verification | 300 seconds |
| Observed worker peak RSS | 512 MiB |
| Nine copied inputs | 12 MiB |
| Complete frozen run storage, including source and inputs | 128 MiB |
| Trajectories / bins | 34 / 304 |
| New model forwards, physics controls, optimizer updates, MPPI controls | 0 each |
| Raw-array reads, checkpoint loads, GPU operations | 0 each |

Pin Python 3.11.15 and NumPy 2.4.3 on CPU. The owner enforces an external deadline and terminates/reaps the process group on timeout. RSS is an observed high-water guard, not a claim of OS memory isolation. Record failed/partial arithmetic as incomplete, never as completed zero work. Freeze a source export, exact recipe and input seal; execute only that source. Owner/worker locks prevent reuse. Any missing binding, changed source, roster/bin discrepancy, nonfinite value, failed numerical identity or exhausted ceiling stops this version. Preserve paid work and failure receipts; do not relax tolerances (`rtol=atol=1e-12` for scalar identities).

Scientific completion means every input and join reconciles, all observations and missing regions are represented, and an independent saved-data audit passes. Null or inconsistent patterns are valid results. A visually compelling pattern is not an admission criterion. Preserve seven CSV tables and four figure pairs: individual ordered bins, cumulative profiles, actuator/marginal summaries, and variance/signed differences. Use a common bin-error color scale and gray missing regions. Report worker wall/RSS/storage separately from audit, plotting, source/input transfers and engineering overhead. Inherited K1 acquisition, inference and qualification remain inherited costs; do not present them as free new evidence or pool them with new work.

## Reproduction and closure

Use fresh paths (shell variables here are examples, not scientific parameters):

```sh
.venv/bin/python -B scripts/train_mppi_gps.py study trajectory-review prepare \
  --source-root /Users/jaisel/Documents/mppi-gps \
  --out runs/2026-11-01_trajectory_review_k1_1_001
.venv/bin/python -B runs/2026-11-01_trajectory_review_k1_1_001/source/scripts/train_mppi_gps.py \
  study trajectory-review run --out /Users/jaisel/Documents/mppi-gps/runs/2026-11-01_trajectory_review_k1_1_001
.venv/bin/python -B scripts/audits/trajectory_review.py --run RUN \
  --out NEW_AUDIT_DIRECTORY --config-sha CONFIG_SHA --seal-sha INPUT_MANIFEST_SHA
.venv/bin/python -B scripts/diagnostics/report_trajectory_review.py \
  --run RUN_OR_CAPSULE --out NEW_REPORT_DIRECTORY
.venv/bin/python -B scripts/train_mppi_gps.py inspect --run RUN_OR_CAPSULE --verify
```

Audit and report directories must be outside the closed run. The independent auditor recomputes all reductions/joins and checks exact input/source/launch/exit/storage bindings without importing the production analysis. Frozen inputs and result metadata make the capsule independently usable without ignored raw evidence. Preserve original K1 and S1.1 capsule bytes. S1.1's minority reference-data walking result remains below its frozen confirmation gate; original S1 remains interrupted and unaudited at the native-data level.

Close with established observations, supported explanations, unresolved causes and one recommended next versioned hypothesis with its discriminating control. Distinguish coverage, sampling frequency, consequential errors and trajectory structure. Any coverage proposal must address the previous corrective-data failure. This analysis cannot select a new loss merely because it produces the expected ranking. Do not automatically launch a weighting rule, new teacher, recovery acquisition, transport objective, learning run or coupling round. A later learning protocol still needs fresh paired streams, exact continuation, fixed budgets, standalone walking/survival and cost endpoints, and separate review.
