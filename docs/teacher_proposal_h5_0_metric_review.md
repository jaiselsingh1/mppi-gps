# H5.0 metric and interface review

This is a source review for protocol preparation, not a new empirical result. No policy, simulator or original motion array was evaluated. It separates three quantities that must not be treated as interchangeable.

| Quantity | Inputs and purpose | What it cannot establish |
|---|---|---|
| K1.2 physical motion distance | Fixed-scale ordered state fragments, excluding clock and global translation; descriptive comparison of recorded motion. | Teacher usefulness, recovery, stable phase, necessary recurrence or the cause of cloning failure. |
| Existing candidate agreement D | The current actor's action at each candidate's own pre-action observation, compared with that candidate's actuator controls. Encourages locally compatible action choices under the current actor field. | What a fixed learning update will learn, standalone walking, or which consequences matter most. |
| Post-update usefulness | Fresh paired standalone outcomes of a child trained for the frozen number of updates, relative to its parent and contemporary controls. | A convergence theorem or a population guarantee from three streams. |

K1.1/K1.2 show information lost by episode averaging and physical differences behind close episode scores. Their findings do not establish that D caused cloning failure, or that a spectral/recurrence/transport distance would improve teaching. H5.0 holds D fixed and tests where it enters optimization. A changed metric would be a separate versioned intervention.

## Exact existing agreement contract

For each candidate k, the current [selector](../src/gps/mppi_gps/search.py) evaluates

\[
D_k=\sum_{h=0}^{47}m_{kh}\sum_{j=1}^{6}
\left(u_{khj}-\pi_{\theta,j}(o(x_{kh}))\right)^2.
\]

The sum has no division by six, horizon, supported-row count or elapsed time. No discount, temporal weight, phase alignment, action whitening, motor-gear/Jacobian scaling or fitted metric is present. The six values are the native actuator controls in `[-1,1]`, the same coordinates produced by the policy's final tanh. Calling the actor a torque policy does not make this loss an error measured in physical joint-torque units.

The [Walker observation adapter](../src/envs/walker2d.py) excludes clock and global root-x, retains eight remaining positions, and clips all nine velocity inputs to `[-10,10]`, yielding 17 values. The [actor](../src/gps/mppi_gps/policy.py) then applies its checkpoint-owned mean and standard deviation in float32 before the width128 tanh network. The H5.0 normalization must be bound to the fixed W1/W2 training-corpus rule and never recomputed on a candidate bank, newly acquired child data, held data or visited states. D remains an action-space error; normalization only changes the actor's inputs.

Native batch rollout arrays contain post-action states `s_1,...,s_H`. The [alignment helper](../src/utils/rollout_data.py) constructs `s_0,...,s_(H-1)` by prepending the actual planning state and dropping the final post state. Scoring post-action states directly would be a different, misaligned objective. This reconstruction is required in **every** guided optimization pass, including both initial refinement passes from the same actual state and solver warmstart.

The mask is the conjunction of:

1. The executable healthy prefix. The initial live state is assumed healthy; action h is retained when all preceding post-action states are healthy. Thus the fall-causing action is retained, and every subsequent action is excluded.
2. Finite pre-action observation values.
3. The caller's support predicate. A scalar, per-candidate or per-row predicate is broadcast by the existing selector contract. The historical Walker recipe sets its provisional support radius to `null`, so it introduces no nearest-support distance threshold.
4. For bank selection, a finite native candidate cost. Corrupt executable-prefix native rollouts receive infinite native cost and cannot become viable through a small D.

The actor is called only for retained rows. Unsupported rows contribute zero, rather than a fabricated prediction. A completely unsupported candidate can therefore have D=0; that is absence of agreement evidence, not proof of perfect agreement or learnability. A shorter healthy prefix also contributes fewer terms. These duration/support discontinuities remain part of the fixed metric and must be reported with row counts; H5.0 does not remove them by normalizing or reweighting.

The current final score is native task cost plus `0.5*eta_u*sum((U-reference_U)^2)` plus `0.5*alpha*D`. The actor-independent reference plan is the protected producer's bounded nominal before optimization. H5.0 fixes `alpha=0.1199341983507337`, `eta_u=0.01` and the separate MPPI softmin temperature `lam=3`. Proximity is not newly inserted into the second stream's inner objective.

## Proposal update, control and accounting hazards

The existing [MPPI hook](../src/mppi/mppi.py) accepts `prior_cost(post_states, sampled_controls)` on every optimization pass. Its result changes softmin weights, the weighted bounded-perturbation update, the next refinement pass and the shifted future nominal. This is the intended point of change. A final selector alone only reweights the completed bank. In H5.0, no actor, guided plan or agreement score may be passed to the protected producer.

Each producer owns its own nominal, private RNG, initial-plan flag and final-pass caches. Read-only snapshots of those must compare identically before/after actor observers and final selection at the **same planning input**. Different selected live actions can cause different subsequent physical states across arms; that expected environmental feedback does not permit asserting that their later banks should be byte-identical. Matched seeds match innovations, not actual paths.

Both protected and second-stream optimized means require native verification before being included as measured candidates. The final selector also verifies synthesized baseline and guided means. Those calls are additional paid model rollouts, not arithmetic on an existing bank. The common native-cost feasibility guard must apply consistently to candidates and synthesized means; an infeasible mean falls back to the separately protected plan. A horizon guard does not guarantee whole-episode competence.

With K=256, H=48, two initial iterations and one pass thereafter, a full 1,000-control episode has 1,001 sampled batches per stream. Two streams, two producer means per decision, and up to two final-selector means reserve

\[
2(1001\cdot256\cdot48+1000\cdot48)+2\cdot1000\cdot48
=24,792,576
\]

model controls per episode. All native batch suffixes are actually simulated and paid even after their scoring masks turn false. Frame skip four multiplies model controls by four for physics substeps; live execution is additional. This expression is an interface upper bound, not a complete first-round census: observer matching, live controls, warm-up/child forwards, metrics, snapshots, evidence copies, failures, audit and storage still require explicit reservations. Unsupported or early-falling cases can reduce realized work without authorizing replacements or larger later budgets.

The planner caches only its final pass. If initial-pass scores/actions/predictions are needed to show that guidance changed the proposal, the callback must retain/count both passes. The instrumented [runtime](../src/gps/mppi_gps/runtime.py) records model input controls, native output states/costs/sensors, plus additional selector evidence. Copying entire banks again can dominate storage; compression ratios must not be treated as a guaranteed finite bound. Observer matching is a declared control overhead, not an efficiency gain or hidden guidance into the independent arm.

## Learner continuation and eligibility

The [fit function](../src/gps/mppi_gps/policy.py) has two distinct continuation contracts. Passing a complete optimizer state continues Adam moments/steps and private PCG64 minibatch RNG. `resume_fit=True` additionally requires the identical data/configuration and resumes the old active proximal anchor. Because H5.0 changes from parent corpus to newly acquired child rows, it must start a new fitting block with `resume_fit=False` while passing the full incoming optimizer state. This preserves Adam/RNG continuation while binding the new data and incoming-parent anchor. The fixed zero proximal coefficient and zero anchor refresh leave that anchor inactive; they do not justify dropping its ownership evidence.

The checkpoint owns policy tensors, normalization, the private Torch generator and optimizer/PCG64/anchor state. It does **not** itself contain global `torch.random.get_rng_state()`. The deterministic tanh/Adam path draws no global Torch randomness; policy construction uses `fork_rng` to restore the caller's global state. A protocol requiring complete global Torch ownership must bind/copy/assert that state separately, without changing old checkpoint keys or claiming an old checkpoint already contains it. Shared minibatch indices remain paired even though their arm-specific data rows differ.

The old useful-learning prerequisite is unmet. H5.0 would be a controlled attempt to obtain useful fixed-update learning **through** teacher adaptation with an initially weak learner. Preparation is eligible under the current authorization; native acquisition or fitting is not. The review decision must explicitly authorize or reject that first controlled attempt under the unmet prerequisite. A positive decision would not claim the prerequisite was already satisfied, and would not authorize round two. Agreement, reconstruction or recurrence improvements alone cannot pass child admission.

Launch readiness requires executable data/source/native bindings, enforceable call and storage reservations, exact controls and continuation checks, and a separately recorded eligibility/launch decision. An unbenchmarked wall envelope or a ceiling with unenumerated work remains a blocker, rather than a license to spend until a favorable result appears. No native fitting, teacher acquisition or follow-up was launched by this review.
