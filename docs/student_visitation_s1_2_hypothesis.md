# S1.2 — fixed-label visitation contrast, draft for review

**Untested and not launchable.** This is one proposed next learning hypothesis after [S1.1](student_source_s1_1_results.md) and [K1.1](trajectory_review_k1_1_results.md). There is no executable recipe, newly collected data or authorized follow-up. Exact incoming checkpoints, source/array bindings, fresh seeds, resets, native counters and time/storage budgets must be frozen before launch. No representation, loss weighting, teacher algorithm, critic or environment change is proposed. This diagnoses the learner using the existing reference; success would not establish MPPI teaching or coupling, and the reference is not assumed better than MPPI at every state/action.

## Hypothesis and discriminating control

**H-S1.2:** with the label function fixed to the existing qualified deterministic SAC reference, supervised examples at student-visited states produce more useful fixed-update torque-policy learning than equal-size examples at reference-visited states.

Let the complete incoming learner state be \(\ell=(\theta,\mathrm{Adam},\mathrm{RNG},\mathrm{normalization},\mathrm{anchor})\). Fix the label function \(a^*(x)=\pi_{\rm ref}(o(x))\). The only contrast is the state distribution:

\[
D_T=\{(o(x),a^*(x)):x\sim d_{\rm ref}\},\qquad
D_S=\{(o(x),a^*(x)):x\sim d_{\rm incoming}\}.
\]
\[
\ell_T^+=\mathcal A_{2000}(\ell,D_T),\qquad
\ell_S^+=\mathcal A_{2000}(\ell,D_S).
\]

Both use ordinary six-action MSE, the existing width128 tanh torque actor, clipped native17 features, the fixed W2 normalization and current optimizer settings. Each contrast forks identical complete incoming state and private minibatch-index stream, uses 4,000 unique training rows and exactly 2,000 Adam updates, and executes the child directly without teacher prefix/fallback. Keep three independent paired training streams. Incoming-state construction and exact Adam/RNG continuation must be explicit in the eventual recipe; resetting an inherited optimizer or silently reseeding continuation is not admissible. Data collection and evaluation starts must be disjoint, and held-trajectory error is reported separately from training fit and execution.

This differs from S1's MPPI-versus-SAC source contrast because **both arms share the same label function**. It also differs from the previous 980-row corrective MPPI experiment, which reduced local fitting error without walking: changing coverage is not an established remedy. A negative result would reject this particular fixed-label visitation intervention at its frozen budget, not show that coverage never matters. A positive result would support this contrast, not prove distribution shift was the unique original cause.

The reference's walking on its own states does not certify its labels on student states or its recovery ability. Before any learning launch, a separately specified data-validity check must retain actual visited states, early falls and complete context and establish what those labels represent. Do not discard hard states until a desired result appears, fabricate post-fall rows, or silently condition the experiment on reference recovery. If the necessary input binding or label-validity interpretation cannot be established, stop/version the proposal before training.

## Outcomes and stopping

Use the existing three-stream, 16-primary plus 16-disjoint-confirmation reset convention. Report reconstruction, held-trajectory error, walking, full survival, executed controls and complete cost separately, relative to the unchanged incoming actor and the matched control. The proposed walking-gain criterion is at least two additional successes in at least two streams, with no walking/full-survival regression in any stream, repeated on confirmation. Individual competence remains at least 15/16 walking on both panels. Retention-only improvement remains partial. No selected checkpoint, seed replacement, additional updates or automatic round follows failure or ambiguity.

The eventual finite census must cover acquisition attempts and failed controls, all reference labels, network forwards, gradients, Adam steps, diagnostics, fresh native evaluation, snapshots, transfers, audits, storage and engineering overhead. Equal row/update budgets do not imply equal acquisition cost. This draft deliberately does not certify a missing census or input roster. Its first implementation step would be a reviewed binding/budget protocol, not launching the experiment from this prose.

## How this serves teacher adaptation

The target remains competent search that is softly constrained by the learner, followed by a fixed update and fresh standalone feedback. Present-output agreement is a coupling proxy, not a certificate of teachability. K1.1's bin cancellation does not prescribe weights, trajectory alignment or a new training objective. Actual saved motions must be inspected before invoking phase, repetition or matching subsequences; nonperiodic solutions and different successful gaits remain valid.

Only confirmed useful learning permits preparing a new coupling experiment. A substantive future mechanism could put

\[
Q_r(U;x_0)=J_H(U;x_0)+\frac{\lambda}{2}\sum_h m_h
 \|u_h-\pi_{\theta_r}(o(x_h))\|^2,\quad x_{h+1}=f(x_h,u_h)
\]

inside a guided branch's proposal/nominal updates, while leaving a separate independent MPPI producer unchanged. This would change subsequent search proposals, unlike the present selector's reranking of a completed independent bank. Keep \(\lambda\) fixed initially to test one change; no coefficient schedule is implied. The protected independent candidate remains available. Native cost guards and complete teacher outcomes must be specified; finite-horizon guards cannot guarantee whole-episode competence, and a failed teacher cannot be admitted solely because its agreement is good.

After a fixed learning update, child admission would depend on fresh standalone walking/nonregression and confirmation. Only an admitted actor, with exact Adam/RNG continuation, would change the policy field in the next actual trajectory optimization. Independent search, selector-only guidance and a fixed-initial-guide control would distinguish search influence from feedback by an updated actor. H4.1's negative child results must remain the parent-relative baseline; H4.2 remains an untested extension of bank reweighting. No benefit or novelty is assumed here.

That proposal is **not implemented or selected for launch**. It requires its own frozen objective, coefficient, teacher/admission guards, equal full budgets, tie rules, controls and stopping rule. Two useful actual acquisition–learning rounds would provide finite loop evidence; neither a diagnostic score nor a single useful fit would show convergence. The current direct torque formulation also remains distinct from CIO's desired state-to-state controller and analytic feedback targets.
