# S1.2 v2: fixed reference labels, different visited-state distributions

**Current status, 2026-11-02:** the user authorized proceeding with V. Its [first preparation attempt](validation/student_visitation_s1_2_v_2026-11-02/README.md) stopped at the remote connection gate. The user then authorized retrying, and the [second read-only connection/runtime/input check](validation/student_visitation_s1_2_v_2026-11-02/preflight_002/remote_stdout.json) passed. The [V1 execution contract](student_visitation_s1_2_v_execution_1.md) freezes the implementation and phase allocation before scientific launch. No S1.2 worker has launched. L remains separately review-gated.

**Review-proposal scope; this document is not an execution command.** This versions the preserved [S1.2 hypothesis](student_visitation_s1_2_hypothesis.md), rather than editing it. The [machine-readable review recipe](../configs/studies/student_visitation_s1_2_review_v2.json) pins existing inputs, exact episode/anchor rules, proposed budgets and hard launch blockers. It remains the preserved nonexecutable review configuration; the separate preparation record binds the V-only authorization. C1 audit recovery has its own authorization and budget.

The first proposed executable block is **V: reference behavior at saved student states**. Even a passing V result only permits review of **L: the matched learning comparison**. No learning, architecture change or further round follows automatically. Before either block launches, create a new source/input/runtime seal, bind every required input and independent checker, and pass its provenance, continuation, replay, seed, capacity and durable-accounting prerequisites. Do not use the narrative or a `user_authorized_launch:false` recipe as an execution command.

## Question and evidence

The hypothesis is that, with a fixed reference labeling function, supervised learning on states reached by the incoming student improves its next standalone policy more than equal-size reference-visited data. It tests a state-distribution intervention, not a priority-sampling rule, loss weighting, sequence objective or architecture change.

**Behavioral alignment** is a separate, broader teaching proposal: use MPPI's flexibility to find task-successful trajectories aligned with the policy's developing behavior family, while preserving teacher competence and independent discovery. Kendall's “higher/lower frequency” framing refers to that behavioral alignment; it does not prescribe spectral filtering. For Walker, these families may be gaits, without requiring a particular gait, periodicity or reference path. Present action agreement alone still does not establish teachability.

**Corrective coverage** is the question addressed here: can the fixed reference act usefully after student deviations, and, conditionally, does its supervision on student-visited states improve learning? S1.2 does not shape MPPI's trajectories toward student-compatible behavior, so it does not directly test the broader alignment proposal. Neither question's answer is assumed to settle the other.

The negative [C1 record](student_consequence_c1_results.md) motivates this question without establishing its answer: consequence-weighted reconstruction improved, while all parents and children failed walking/full survival and held-trajectory errors worsened. S1.1's reference-data children achieved low reference-state error but only one stream produced occasional walking; the reference itself walked 32/32. The [earlier 980-row corrective MPPI experiment](loop_v3_evidence_audit.md#what-the-prior-corrective-test-establishes) reduced local fitting error without walking. Coverage is therefore not an established remedy. Unlike that corrective-data experiment and S1's whole-source comparison, this contrast holds the labeling function fixed across both arms.

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) establishes an imitation-learning approach to learner-induced observation distributions. This one fixed-update contrast is not iterative DAgger and inherits no automatic guarantee. [AggreVaTe](https://arxiv.org/abs/1406.5979) uses cost information; no cost-sensitive target or weighting is added here. [DART](https://proceedings.mlr.press/v78/laskey17a.html) studies noise-perturbed demonstrations and recovery; no demonstration noise is added here. None establishes benefit for these Walker inputs.

## Fixed actors and existing inputs

Use the **three retained incoming C1 parents**, not any selected C1 child. Their complete packages contain 2,000 cumulative incoming updates. Reusing these parents avoids selecting a winner from C1's outcomes; it does not create fresh independent training replications. The three original training streams remain the replication units.

| Stream | Incoming checkpoint SHA-256 | Actor hash |
|---|---|---|
| 0 | `f76ff22c816f0554517355721b2ad1c92e89cf3e58b5751eb0b79211a22594c2` | `8d01dd29e1324770d188a7fa87c9f4961974e7bbe98b380acec4f46d5bd8ce50` |
| 1 | `dfff0bf7b76f775bebef892a0c058df65faec6d68078d2e7039886b80e18dfa5` | `93ed85d1ec60e98325bcabaf116774f1655cccc6f49da60d94793b537f98724d` |
| 2 | `c7c42a8365d936df18eed06f15572ebcc420b6688ad3a53ad1b863d161a9a55f` | `1cde667a9fb4d287ac8b60dcebeedab3b4c517d37169e0b24af20f1bd7236639` |

The recipe binds the exact C1 evaluation JSON, actor registry, prerequisites and prepared manifest, including normalization, typed optimizer/private-PCG/private-Torch/global-Torch fingerprints and the three original global CPU RNG sidecars. Whole-file SHA, actor hash, optimizer fingerprint and array-byte hash are distinct contracts. Full raw C1 evidence remains on `robopil5090` at `/home/jaisel/mppi-gps-workspace/runs/2026-11-02_student_consequence_c1_001`. C1’s separate [A2 compensating admission](../results/student_consequence_c1_certificate_closure_a2/run/admission.json) is now bound: SHA-256 `13e0985a1004cbbbe55788f06d6b97eccfee3375b5d4de47f2de580924011538`, source commit `a8998d2d2aad508175be2960230c38d0709bb39d`. The [separate recovery capsule](../results/student_consequence_c1_certificate_closure_a2/README.md), its exact protocol/source manifest/result and all three clean worker/coordinator/finalizer exits are bound in the review recipe. A2 retains the original numerical checks and A1’s point-in-time integrity/accounting certification; it does not repeat that payload audit. Both original failed launcher histories and the unknown original C1 child exit remain preserved. **This resolves only the C1 recovery prerequisite.** The user has since authorized proceeding with S1.2 V, subject to all remaining source/runtime/seed/capacity/accounting gates and a separate execution freeze. Its first preparation stopped at the connection gate as recorded above; even a passing V outcome would stop for review before L.

The fixed labeler is the already qualified deterministic SAC actor, whole checkpoint SHA-256 `174e988618a478d41b846cf428a63b91fef5a0a0f94ae0b36637708856ef363f`, actor hash `e1ec300c39d6880f893995012fa7c5a0789f9ef0b931280454310b57e62c9cba`. The explicit remote input location is `/home/jaisel/mppi-gps-workspace/runs/2026-10-31_student_source_s1_1_001/inputs/s1/inputs/reference.pt`; current availability and exact bytes must be verified. Missing/mismatched bytes stop preparation, rather than selecting or training a replacement reference.

Use the existing `LegacySACActor` adapter: frozen 17→256 ReLU→256 ReLU actor, original float32 batch-one deterministic tanh-mean arithmetic and its own identity normalization. Construct no critic or SAC learner. Feed the unchanged raw clipped native17 observation to this adapter; do not give it the student's W1-normalized coordinates. All student architectures stay 17→128 tanh→128 tanh→6 tanh, with the exact incoming W1 normalization (normalization NPZ SHA-256 `652fb44b1796ecf685c631ed86d1c9c9909580fffed29bbb0d339f140ae4ae73`). The reference/student architecture difference is inherited and is not an intervention in this comparison.

Architecture-matched reference cloning remains a separate proposed control requiring its own reviewed protocol. It is outside S1.2; the architectures above remain fixed within this comparison.

Use C1's native CPU runtime and task/XML: Python 3.11.16, Torch 2.10.0+cpu, MuJoCo 3.6.0, NumPy 2.4.3, SciPy 1.17.1, CPU affinity 0–15, one Torch/OMP/BLAS thread, `MUJOCO_GL=disable`, CUDA disabled, frame skip 4 and 0.008 seconds/control. XML SHA-256 is `4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590`. Preserve all task-cost values from the bound C1 configuration. No backend, feature, width, normalization or optimizer switch is permitted after outcomes are seen.

## V: test the reference under its own feedback at student states

### A fixed nonselective roster

Reuse every saved pre-action state from each incoming parent's 32 C1 fresh-panel traces. The prospective training episode order is C1 `primary_00..15`, then `confirmation_00..03` (20 episodes); held episodes are `confirmation_04..15` (12 episodes). These are known development traces, not newly blinded holdouts.

For the V primary panel, choose each stream's C1 primary episodes 00–03. For V confirmation, choose its C1 confirmation episodes 04–07. For each of those four episodes with N actual controls, freeze four pre-action anchors at

\[
t\in\{0,\lfloor(N-1)/3\rfloor,\lfloor2(N-1)/3\rfloor,N-1\}.
\]

This gives 16 anchors per stream per panel, **96 anchors total**. All exact controls and trace groups are listed in the review recipe. Include the last pre-action state, even when its original action caused the fall. Exclude no anchor because of support distance, action error, predicted reference agreement or eventual recovery. Never use a post-fall terminal observation as a new healthy training state. The 96 anchors are correlated samples from eight source episodes per stream, not 96 independent task replications.

**The primary V outcome concerns the 72 noninitial anchors**, at ordinals 1, 2 and 3: twelve per stream per panel, each with actual control index t>0, after the incoming student has acted. Verify every declared noninitial index is positive before launch; any zero/nonpositive index stops preparation rather than dropping or replacing it. The 24 ordinal-0/t=0 anchors remain secondary reset/restore controls, fully executed and reported. Their walking and full-survival results enter neither the usefulness gain nor its nonregression gate. Otherwise ordinary reset walking could pass the recovery screen while every noninitial takeover failed. This is an action-history condition, not a newly selected state-distance threshold.

At every anchor restore the same full state, warmstart and native integration context, then compare two complete closed-loop continuations:

\[
x_{k+1}=f(x_k,\pi_R(\phi(x_k)))\quad\text{versus}\quad
x_{k+1}=f(x_k,\pi_s(\phi(x_k))),\qquad k=0,\ldots,999.
\]

Use 1,000 controls from the anchor, independent of its original clock time; stop actual physics on a fall. The reference controls **every** action after handover. This differs from D2/C1's one substituted action followed by a foreign open-loop teacher suffix. It measures the reference's feedback behavior at the selected learner states, not an intrinsic value of its first action, teachability or recovery from every training row.

### Validity prelude and positive control

For each of the 32 reset seeds, the canonical initial context is the saved C1 `s0_parent` context at control 0. **Before any reference inference or physics**, statically compare all three saved parents’ control-0 contexts for that seed to this canonical context: full state, warmstart, native integration, observation, exact structure/dimensions/integration specification and the existing byte/numerical contract (`rtol=0, atol=1e-10` for numerical comparisons). Any disagreement stops/versions preparation, without replacement contexts, additional actors or reference rollouts. This is a saved-array comparison only: no new network, physics or cases. It establishes the required common starting-context binding for one reference corpus shared across all three students.

Before any reference takeover, execute all 96 unchanged-parent continuations as the paired control and numerical replay prelude. Byte-exact restoration is mandatory. Their actual original prefixes must match saved actions, states, observations, warmstarts, integration arrays, native costs and fall boundary with `rtol=0, atol=1e-10`; original float32 action bindings must also match. Include complete N+1 terminal context. Any missing context or replay mismatch stops V immediately, preserving the paid prefix. Do not relax tolerance or substitute an episode.

Then run the reference once on all 32 original C1 reset seeds (2810000000–15 and 2820000000–15), saving full actual prefixes and contexts. This is one shared own-reset reference control, not three independently acquired teachers. Each reference reset’s complete initial state, warmstart, integration and observation must match that seed’s canonical saved C1 `s0_parent` control-0 context under the existing numerical contract; the all-three-parent static comparison must already have passed. The same integer seed alone is insufficient. A mismatch stops V. Both 16-reset panels run before the positive-control decision. Require at least **15/16 walking on each**. Failure stops before reference takeovers, without replacement seeds or a new reference. These are known reset contexts, so this is current-runtime qualification for this block, not a new broad generalization claim.

After both preludes pass, run all 96 reference takeovers, in fixed panel/stream/episode/anchor order, even if early outcomes are disappointing. Report each anchor and all four anchor-position groups, with full-window velocity error, walking, full survival, executed controls, native/raw/terminal/absorbing/complete cost and paired differences. Height, pitch, velocity and clipping curves use actual prefixes; post-fall states remain missing.

Walking uses the existing definition: **1,000 actual controls, no fall, mean absolute x-velocity error from 1.5 m/s ≤0.3 over the complete window**. A shorter successful-looking tail, longer retention or lower cost is not walking competence. A successful takeover from an in-sample anchor is not fresh-reset competence.

Apply the declared reference-usefulness screen **only to the twelve noninitial anchors per stream per panel** (ordinals 1–3 and actual t>0): at least two more walking successes than the matched parent in the **same at least two of three streams**, with no walking/full-survival regression in any stream on this same noninitial population, repeated on V confirmation with those same qualifying streams. Ordinal-0/t=0 outcomes cannot contribute to either admission condition. In particular, reference walking on all four t=0 anchors and failing all twelve noninitial takeovers in each stream/panel **must fail** this screen, even if the separate own-reset positive control passes. It is a finite screen for some useful reference behavior, not certification that all selected labels are corrective. Even if it passes, **stop for result review before any labels for fitting or any learning launch**. If it fails, preserve all outcomes and stop/version; do not turn successful anchors into a filtered data source. Near-fall states may be unrecoverable by either controller, so failure rejects the declared recovery-label assumption on this roster, not coverage as a general idea.

## L: the conditional fixed-label learning comparison

### Exact state selection and labels

Only after separate review of V, prepare two corpora per stream with the same frozen reference labeling function \(a_R(x)=\pi_R(\phi(x))\):

\[
D^{(s)}_{SV}=\{(\phi(x),a_R(x)):x\in\mathcal R_{4000}(S^{(s)}_{train})\},\qquad
D_{RV}=\{(\phi(x),a_R(x)):x\in\mathcal R_{4000}(R_{train})\}.
\]

Here \(S^{(s)}\) is the saved incoming-parent visitation and \(R\) is the 32 reference own-reset rollouts from V, starting at those same reset contexts. \(D_{RV}\) is one common corpus reused across the three optimization streams. Reference trajectory failures, if any pass the overall positive-control threshold, remain in the source universe and are never removed.

The shared deterministic selector \(\mathcal R_n\) first allocates per-episode integer quotas: repeatedly pass through the fixed episode list, adding one row to an episode while its quota is below its actual length, until quotas sum to n. For length N and quota q>1, select controls `floor(k*(N−1)/(q−1))`, k=0..q−1; q=1 selects control 0. Emit the final row list by selected-rank then episode order. This covers each complete actual prefix, preserves both endpoints when q>1, and avoids selecting only the first 200 controls of long reference rollouts. The same selector applies to both arms, without outcome/error/recovery scoring. Exact SV quotas and canonical selected-row-table hashes are in the recipe. Source row counts before selection are:

| Incoming stream | 20 training-episode rows available | 12 held-episode rows available |
|---|---:|---:|
| 0 | 4,184 | 2,311 |
| 1 | 4,477 | 2,691 |
| 2 | 5,583 | 3,170 |

Select **4,000 training and 2,000 held rows per arm per stream**, with disjoint source trajectories. Retain the entire original universe, unused rows, source frequencies, fall-causing-action prestates and all failed attempts. Unique means distinct recorded episode/control IDs, not a claim that effective network inputs never coincide. Report exact input collisions in raw17, float32 and frozen-normalized float32 coordinates. Cross-split effective-input collisions stop/version rather than deleting rows; within-training repetition and cross-arm matched initial states are disclosed, not filtered. Insufficient actual rows in either reference split stop/version, without new rollouts or post-fall padding.

Reuse the reference's own actually executed actions for its selected rows after exact adapter/input/arithmetic proof. Query the same reference once per selected student train/held state: **18,000 new float32 labels total**. Preserve the existing feature arithmetic `concat(state[2:10], clip(state[10:19],−10,10))`, float32 training conversion and float64 diagnostic residual reductions. Native execution promotes the actual float32 actor output to float64 before both physics and native cost, matching C1. Do not use the incoming student's executed actions as supervision. Freeze full row tables, source-array/label hashes, feature conversion and actual datasets **before fitting**. These numerical intermediate outputs do not exist yet; the review recipe is deliberately blocked until they are sealed. Label preparation itself needs explicit authorization and a bounded source seal. No label-quality filter or training on takeover rollout suffixes is added.

The held trajectories are untouched by gradients, but their source states have been inspected during development and some host V anchors; call them development held trajectories, not a blinded test. Fresh standalone panels below are disjoint from every source/reset/anchor context.

### Matched update and execution

Let the complete incoming learner state be

\[
\ell_s=(\theta_s,\nu,\mathrm{Adam}_s,z_s^{PCG},z_s^{Torch},z^{global},a_s).
\]

Fork each \(\ell_s\) exactly twice, including tensors, normalization \(\nu\), all Adam moments/steps/options, private PCG64/Torch and global CPU Torch state, and proximal-anchor ownership. Use the same saved minibatch-index tape \(\iota_s\) in the two arms:

\[
\ell^+_{s,j}=\mathcal A_{2000}(\ell_s,D_j;\iota_s),\qquad
L_D(\theta)=\frac1{6|B|}\sum_{i\in B}\|\pi_\theta(o_i)-a_R(x_i)\|^2.
\]

Keep ordinary uniform row sampling with replacement, batch 256, Adam learning rate 0.0003, betas (0.9,0.999), epsilon 1e-8, no weight decay/AMSGrad/foreach/fused Adam, gradient clip 0.5, parameter proximity zero and no anchor refresh. Continue the inherited optimizer and RNG; do not reset Adam or reseed a fresh private stream. Set the changed-data block's anchor to the incoming actor with existing continuation semantics. Endpoints have 2,000 new updates / 4,000 cumulative updates. Preserve actual sampled indices before each forward and saved paid minibatch predictions/losses for independent checking. There are exactly six fits, no parent refit, extra endpoint, early stopping, selected seed or architecture intervention.

The two children and retained parent execute their existing six-torque policies directly. There is no reference prefix/fallback, recovery controller or MPPI. Independent control-space search remains available elsewhere in the project but is unchanged and unqueried here. This torque formulation remains distinct from CIO's state-to-state controller/targets; no CIO–GPS reproduction or adaptive loop is claimed.

### Endpoints and confirmation

Evaluate three unchanged parents, six fixed endpoints and the one frozen reference on **16 fresh paired primary plus 16 disjoint confirmation resets: 320 episodes**, at most 1,000 controls each. Proposed primary seeds are 2910000000–15; confirmation seeds are 2920000000–15. Freshness is not yet verified: check all local/remote metadata before freezing, and stop/version on collision rather than silently substituting seeds. Evaluate both panels for all ten actors unconditionally unless an infrastructure gate stops the paid roster.

The primary causal contrast is SV child versus RV child within each stream. Confirm useful visitation learning only if SV gains at least two walking successes over **both** its incoming parent and its matched RV child in the same at least two streams, with no walking/full-survival regression against either comparator in any stream, repeated on confirmation with those same qualifying streams. Individual competence remains ≥15/16 walking on both panels. Retention-only gains, lower complete cost, held error or reconstruction cannot pass the walking gate.

For each of the three streams, measure **that stream’s own parent and its two children** on that stream’s SV train/held sets and the common RV train/held sets: `3 streams × 3 actors per stream × (4,000+2,000+4,000+2,000) = 108,000` new prediction rows, in 108 calls at fixed 1,024-row boundaries. Do not evaluate all nine actors on every stream’s SV sets. All 108,000 metric rows are newly measured, including the incoming parents; saved C1 batch-one actions are not silently reused as 1,024-batch metric predictions. No extra historical W1/held metric forwards are planned. Preserve all 12,000 paid learning-curve points without smoothing or checkpoint selection. Report per-source and per-trajectory reconstruction, held error, all walking/survival counts, actual controls and velocity error. Pair each fresh child/reference cost gap to the same reset; the shared reference is one control, not three teacher replications. Complete cost uses native costs including the fall terminal once, plus five per unexecuted control; keep raw/native/terminal/absorbing parts separate and invent no post-fall states.

## Finite budget proposed for review

The following are future ceilings, not completed work or permission. V and L have independent create-only ownership and source/input freezes; unused V resources cannot silently buy more L work. The JSON carries exact counts and pending seals. A completed implementation must prove its helper-call, compact-record and storage bounds before either freeze; if it cannot meet them, revise the protocol before launch rather than exceed the reservation.

| New work | V maximum | Conditional L maximum | Combined maximum |
|---|---:|---:|---:|
| Native controls | 224,000 | 320,000 | 544,000 |
| Physics substeps | 896,000 | 1,280,000 | 2,176,000 |
| Own-reset reference / paired anchor branches | 32 / 192 | 0 | 32 / 192 |
| Fresh standalone episodes | 0 | 320 | 320 |
| Static student-state reference labels | 0 | 18,000 | 18,000 |
| Fit forward rows / backward rows, each | 0 | 3,072,000 | 3,072,000 |
| Adam updates | 0 | 12,000 | 12,000 |
| All worker forward rows | 224,000 | 3,518,000 | 3,742,000 |
| Worker forward calls | 224,000 | 350,108 | 574,108 |
| MPPI queries / parent refits / critics | 0 | 0 | 0 |

The bound original parent prefixes imply **11,426 expected unchanged-parent replay controls**. V conservatively reserves up to 1,000 per parent branch as well as 1,000 per reference branch and own-reset episode. Report actual work; unused headroom cannot buy additional cases.

The bounded V worker is at most **two hours**, conditional L at most **four hours**, combined worker six hours; peak worker RSS 8 GiB, combined worker output 8 GiB. Independent audits together get two hours, 4 GiB RSS, 32 GiB source reads and 4 GiB output: at most 670,000 nontraining forward rows / 562,108 calls, twenty checkpoint loads, zero backward/Adam/physics. Audit saved fitting residuals from already paid predictions, never rerun optimization; gradient norms remain hash-bound worker measurements.

Preparation is at most ten minutes, 512 MiB RSS/reads/output. Saved-only reports get thirty minutes, 2 GiB RSS and 512 MiB output; engineering records 512 MiB. All new study-owned output, copied source exports, transfers and retained temporary files must fit **16 GiB**, with 2.5 GiB allocation/headroom. Pin two native environments / 34 MjData objects across V and L, not simultaneous extra simulator pools. The full helper ceiling is 352 resets, 192 restores, 896 external `mj_forward` calls, 1,088,544 snapshot reads, twenty checkpoint loads and twenty-four actor constructions. Preserve per-phase actual requested/completed/returned-unpersisted/unknown work; an unknown count is never zero-filled.

Keep raw arrays, checkpoints, minibatch prediction/index tapes, native index/journals and videos outside Git. Use the existing native trace/spool format; at most 2,000 array groups, six million compact journal records of at most 512 bytes, and 73,728,000 bytes for paid fitting-prediction tapes. Reserve three expected index tapes / 1,536,000 entries and six actual tapes / 3,072,000 entries. Two bounded source/result archive transfers per separately reviewed block (four total), each ≤128 MiB, are included in global storage and separately charged. Compact interim V results must be reviewable before L; no large raw mirror is implied.

On any missing binding, restoration/replay, continuation, deterministic-runtime, numerical, seed, capacity, ownership, durable recording, deadline or work/output failure: stop, retain the complete paid prefix and missing evidence, and publish the independent audit outcome. Preserve failed audits/launch receipts. No retry, source/seed replacement, tolerated mismatch, extra update, new reference, new architecture, new acquisition rule or automatic coupling is allowed. Before running either block, validate clean-export CLI/report/inspection and synthetic row/checkpoint joins, same-stream continuation, source restoration, unchanged-parent controls, fall accounting, budget stops and missing-evidence behavior; preserve all existing tests and historical manifests.

## What the result could establish

A positive V result shows reference closed-loop behavior helped on the declared noninitial learner-state roster; it does not prove all its one-step labels are useful to the student. A negative V result makes this reference-as-recovery-label assumption unsupported here; stop/version without a successful-only corpus. A positive L contrast supports this specific fixed-label visitation intervention at its declared budget, not a unique coverage cause or architecture sufficiency. A negative L contrast rejects that version without refuting all distribution-aware imitation.

Uniform sampling frequency within each selected corpus stays fixed. Consequence weighting is absent, and trajectory structure is not optimized. The primary contrast is the constructed visitation distributions after this common selector, not unaltered time occupancy or pure coverage alone. State frequency, trajectory duration, phase, geometry and prior actions can still differ by controller; those are properties of the intervention, not separately identified causes. Only a confirmed useful learning result permits preparing a distinct MPPI teaching/coupling protocol for review. No outcome automatically launches such a protocol or a second round.
