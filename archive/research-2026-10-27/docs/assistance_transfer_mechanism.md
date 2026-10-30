# From independent MPPI exploration to executable policy behavior

Mechanism review, 2026-10-25. This review does not change the running
[walking trial](expert_withdrawal_walking_trial.md), its source, configuration,
stopping rule or evaluation. New diagnostic output has its own directory.

**Recommendation:** use independent MPPI to discover reachable training states;
from a selected handover state, let the learner execute and optimize its entire
remaining episode. Move handover earlier only after checking learner execution.
Retain the existing parameter-space search initially, using measured learner
returns. Do not use an expert finish as the learner's terminal value. This
changes the source of exploration while keeping ownership of the outcome clear.
It is a guide-state curriculum, with substantial prior precedent, not a new
general learning algorithm.

The current learner-prefix/expert-suffix objective is correctly implemented as
an assisted objective. Its improvement does **not** generally improve the
standalone policy, even with exact values, full state, a sufficient policy class
and successful optimization. The existing two-action example already proves it.

## 1. Start with the original objectives

The original [objective notes](objective.md) define a simulator rollout cost

\[
 S_x(U)=\sum_{t=0}^{H-1}\ell(x_t,u_t)+\phi(x_H),\qquad
 q^*=\arg\min_q\{\mathbb E_q S_x+\lambda D_{KL}(q\Vert p)\},
 \quad q^*(U)\propto p(U)e^{-S_x(U)/\lambda}.
\]

The population inequality applies to distributions over those scored plans.
It does not establish improvement of a sampled weighted mean, its first action
with another controller continuing, or a fitted neural policy. With a
policy-centered base measure, the search also depends on the weak student.
Preserving independent MPPI requires its own nominal plan, sampling, task cost
and random stream, with no student tail, value or matching penalty. Removing
that coupling also removes any claim that its distributional update is directly
a policy improvement step.

The present [withdrawal formulation](expert_withdrawal.md) instead optimizes

\[
 A_L(\theta)=\mathbb E\left[
  \sum_{t<L}\gamma^t c(z_t,\pi_\theta(o_t))
  +\gamma^L V^{E_r}_{N-L}(z_L)\right], \tag{1}
\]

where the learner executes the first \(L\) controls and a restarted independent
teacher \(E_r\) executes the rest. Use cost throughout: lower is better.
The deployment objective is

\[
 J(\theta)=\mathbb E_{z_0\sim\rho}V^{\pi_\theta}_N(z_0). \tag{2}
\]

These are different controllers' returns. Expert outcomes can legitimately
supervise (1); they do not become observed learner outcomes in (2).

## 2. State, memory, horizons and actual execution

Let \(z=(x,w)\) include physical simulator state and numerical solver warmstart.
For the current Walker implementation this is FULLPHYSICS state (19 numbers)
and `qacc_warmstart` (9). The policy observes \(o=h(z)\): 17 features, omitting
horizontal position, time and warmstart, with clipped velocities. The current
actor is reactive and has neither remaining time nor recurrent memory.
Translation invariance may make horizontal position irrelevant; it does not
establish sufficiency of all the other omissions.

An MPPI controller additionally has planning state
\(m=(U,\text{initial-refinement flag},\text{RNG state})\). Its action and next
planning state depend on \((z,m)\). Define

\[
 V^E_n(z,m)=\mathbb E[C^E_n\mid z,m],\qquad
 V^{E_r}_n(z)=\mathbb E_{m\sim R_0}[V^E_n(z,m)], \tag{3}
\]

where \(R_0\) means zero nominal plan, initial refinements and a fresh private
seed. This definition keeps a restarted teacher separate from one carrying a
historical plan. A fixed seed gives a conditional realized return, not an
expectation over seeds. RNG may instead be modeled as fresh exogenous draws;
the same randomness convention must be used on both sides of a comparison.

In `independent_teacher_suffix`, a new planner is constructed **once per
handover**, then retains its own shifted plan for all subsequent suffix
decisions. It does not restart at every control. It takes the saved physical
state and solver warmstart, but no historical MPPI plan and no learner input.
The teacher's internal \(H=48\) planning score is distinct from its realized
\(N-L\)-control receding-horizon suffix return.

Fresh-once values generally do not satisfy a Bellman recursion obtained by
restarting again at the next state: the actual second decision retains the
shifted plan, whereas the newly queried oracle starts at zero. Complete-return
supervision is valid. A policy-improvement theorem using one consistent expert
value requires augmented planning state or explicit analysis of this switch
oracle; a state-only expert-value theorem cannot be imported unchanged.

For any controller that falls after \(\tau<N\) executed controls, define

\[
 C_N=\sum_{t=0}^{\tau-1}\gamma^t c_t
       +\gamma^\tau c_\dagger\sum_{j=0}^{N-\tau-1}\gamma^j,
 \qquad c_\dagger=5. \tag{4}
\]

The falling transition already includes its terminal charge, once. Afterward
there are no actions or entropy terms. Equivalently, extend the trajectory in
an absorbing state for algebra involving a fixed \(L\). A healthy finite
\(N\)-control cutoff has zero further cost. A prefix's administrative cutoff
at \(L<N\) is a handover, not a fall. Do not reuse the continuing SAC adapter's
infinite failure tail in (4). At \(L=N\), (1) equals (2) and no teacher or
expert-value call is allowed. For a recurrent policy, its own hidden state at
handover must also be specified; setting it arbitrarily changes the controller.

The current critic is
\(\widehat V^{E_r}(o,n/N)\), a supervised predictor of realized complete
teacher suffix costs. It is **not** \(V^\pi\), \(Q^\pi\), or a Bellman target
for the learner. With squared error and insufficient observation, its ideal
population target is
\(\mathbb E[C^{E_r}_n(z,\eta)\mid h(z)=o,n,\text{collection protocol}]\),
including its planner-seed rule. Identifying this with a conditional average
of \(V^{E_r}_n(z)\) additionally requires the specified seed law independent
of the endpoint; the runner reuses root-linked seeds across candidates.
Changing the policy can change that conditional distribution. Thus even an
unlimited observation-only regressor need not give a single correct value for
every endpoint. Full-state diagnostic evaluation can be privileged without
giving privileged inputs to the deployed actor.

The [runner](../scripts/training/train_expert_withdrawal.py) perturbs complete
policy parameters, executes each candidate's full prefix, scores endpoints
with a frozen expert critic (or actual takeovers when fitting is unavailable),
and proposes a bounded antithetic ES step. It verifies actual old/proposed
takeover means on training and separate verification roots with paired teacher
seeds. Only strict improvement on both accepts an update. This protects those
sampled assisted comparisons from critic hallucination; it does not test
standalone transfer. Calibration endpoints are from the old policy, whereas
candidate endpoints also enter fitting. Stage advancement compares current
assisted cost with pure teacher cost and can happen without accepting an update.

Legitimate learning information is therefore: observed prefix costs and
endpoints, independent realized teacher suffix labels, their provenance and a
clearly labeled assisted objective. Teacher actions need not be targets.
For a learner-return critic, a teacher transition may supervise the actual
teacher action's immediate reward/dynamics, but its bootstrap must follow the
declared learner continuation. A whole teacher return cannot be assigned to
an unexecuted learner action or continuation.

## 3. Exactly when does assisted improvement transfer?

Write the discounted completion gap as

\[
 G_L(\theta)=\gamma^L\mathbb E_{z_L\sim d_L^{\pi_\theta}}
 \left[V^{\pi_\theta}_{N-L}(z_L)-V^{E_r}_{N-L}(z_L)\right].
\]

Both continuations use exactly the same endpoint, horizon and absorbing
convention. Conditioning on the learner prefix gives the exact identity

\[
 \boxed{J(\theta)=A_L(\theta)+G_L(\theta).} \tag{5}
\]

For an update \(\theta\to\theta'\), let
\(\delta=A_L(\theta)-A_L(\theta')>0\). Then

\[
 J(\theta)-J(\theta')=
 \delta-[G_L(\theta')-G_L(\theta)]. \tag{6}
\]

Assisted improvement transfers **if and only if** growth of dependence on the
teacher is smaller than the assisted gain. Sufficient conditions include
nonincreasing \(G_L\), or an independently established bound
\(|V^\pi_n-V^{E_r}_n|\le\eta\) for both policies on all their endpoint support,
with \(\delta>2\gamma^L\eta\). Teacher optimality alone supplies neither.
The learner must execute the promised suffix on the states its own prefix
reaches. Accuracy of \(V^{E_r}\) says nothing about that ability.

If expert-value prediction error is uniformly at most \(\epsilon\) on those
same endpoint distributions, a predicted assisted gain \(\widehat\delta\)
only gives

\[
 J(\theta)-J(\theta')\ge
 \widehat\delta-2\gamma^L\epsilon-[G_L(\theta')-G_L(\theta)]. \tag{7}
\]

Held-out average RMSE is not this uniform bound. The current actual-takeover
check removes the need to trust predicted assisted gain on its finite panel;
the gap term remains.

Increasing \(L\) eventually eliminates the gap at \(N\), but does not make
successive objectives monotone in standalone performance or ensure arrival at
that stage. A bounded-cost finite-horizon bound shrinks with the remaining
discounted horizon; it can be vacuous until near \(N\). The curriculum can
stall, and full-horizon search can fail. Any final success might come entirely
from the last standalone stage. To attribute exploration transfer requires
evidence that earlier teacher information helped acquire a continuation the
learner subsequently executes.

Stronger dynamic-programming conditions can recover policy guarantees:
sufficient state/time representation, relevant state coverage, compatible
continuations and simultaneous statewise optimization, with controlled value
error. Optimizing a single reset-distribution assisted prefix is much weaker.

## 4. Smallest executable counterexample and what it separates

The existing [two-step regression](../tests/test_expert_withdrawal_counterexample.py)
uses

\[
 C(a_0,a_1)=\tfrac14a_0^2+(a_0-1)^2+(a_1-a_0)^2. \tag{8}
\]

The weak policy \((0,0)\) costs 1; independent expert \((1,1)\) costs .25.
Expert first action followed by the weak learner, \((1,0)\), costs 1.25.
That negative \(Q^\pi\) advantage is correct: it diagnoses incompatible
continuation, rather than a mistaken critic. Planning discovers a coordinated
sequence that a one-action intervention does not transfer.

After any learner \(a_0\), the expert executes \(a_1=a_0\). With
\(N=2,L=1,\gamma=1\), its exact suffix value is \((a_0-1)^2\). Consider a
small correct descent step changing only the learner's first action:

| Same update | Before \((.5,0)\) | After \((.55,0)\) | Difference after−before |
| --- | ---: | ---: | ---: |
| Exact assisted \(A_1\) | .3125 | .278125 | −.034375 |
| Exact standalone \(J\) | .5625 | .580625 | +.018125 |
| Completion gap \(G_1\) | .25 | .3025 | +.0525 |

The assisted gradient at .5 is −.75; a step of size \(1/15\) gives .55.
There is no critic, observation aliasing, randomness or failed optimization in
this counterexample. The two-parameter class contains the standalone optimum
\((.8,.8)\), cost .2. Even exact assisted optimization can set \(a_0=.8\)
while leaving \(a_1=0\): assisted cost .2, standalone cost .84. The original
success test adds 64 full-standalone search iterations after 28 assisted ones;
its eventual .2 result does not prove transfer of the assisted updates.

This convex example establishes ownership and a failed implication. It does
not establish an exploration advantage over root-only search. The current
walking trial could still succeed empirically; that would not make assisted
improvement a general guarantee.

Distinguish four explanations:

| Explanation | Isolating observation / falsifier |
| --- | --- |
| Dependence on assistance | Exact \(A_L\) falls while exact \(J\) rises; (5) attributes the difference to the completion gap. This already falsifies a critic-only explanation. |
| Critic approximation or aliasing | Frozen learned versus exact values/rankings differ on identical queries; replacing learned values removes an update failure. Exact-value failure rules out critic error as the necessary cause of that failure. |
| Policy information or expressiveness | An exact best-in-class learner suffix still fails while a sufficient-state/time diagnostic class succeeds. Failure to match teacher actions is insufficient evidence: another behavior may achieve equal return. |
| Optimization failure | A directly executed successful policy exists in the same class, but the finite update procedure cannot find it under exact scoring. An exhaustive tiny search or analytic optimum separates this from impossibility. |

Representation and optimization remain real Walker concerns. They are not
needed to explain the logical failure of (1).

## 5. What the raw experiments actually establish

Read directly from the completed smoke's
[mixed metrics](../runs/expert_withdrawal_validation_20261025_001/mixed_prefix_001/metrics.jsonl)
and [full-horizon metrics](../runs/expert_withdrawal_validation_20261025_001/full_withdrawal_001/metrics.jsonl):

| Check | Actual result |
| --- | --- |
| Mixed smoke, assisted \(L=1\) | Training cost 215.569→215.117, verification 255.029→255.383: rejected; curriculum nevertheless advances. |
| Mixed smoke's accepted update | Only at \(L=N=64\): training 225.401→223.754, verification 252.792→251.591. There are **zero accepted assisted updates**. |
| Mixed smoke standalone initial→final | Finite cost 224.011640→222.566459 over just 64 controls; this gain follows full-policy search. |
| Separate \(N=1000\) standalone smoke | Cost 3100.140259→3098.560050; 0/16 eight-second survival both before and after; two of three full-policy proposals accepted. |

I rechecked the frozen critic on its exact saved queries, using
`mixed_prefix_001/expert_data_0000.npz` and `expert_value_0000.pt`. Remaining
horizon is 63 for every row:

| Query | Saved realized teacher cost | Frozen prediction |
| --- | ---: | ---: |
| Train 0 | 159.342143 | 169.746490 |
| Train 1 | 197.215186 | 195.750381 |
| Train 2 | 158.771275 | 149.344803 |
| Train 3 | 198.321504 | 197.438324 |
| Holdout 0 | 177.716617 | 169.562363 |
| Holdout 1 | 220.422742 | 168.950607 |

Train RMSE is 7.071677; holdout RMSE is 36.850187, worse than the initial
constant predictor's 29.709788. Holdout ordering reverses. These are matched
single-realization labels, **not exact expected values**. Two holdout rows do
not isolate approximation, seed variance or observation aliasing. In particular,
subtracting an estimated base score from a separate acceptance score with
different planner seeds would not be a matched critic-error calculation.

The independent-teacher [raw probe](../runs/2026-10-25_independent_teacher_probe_001/analysis_summary.json)
replayed exactly at 11/11 anchors after changing externally loaded learners.
Teacher-first then initial-learner continuation survived 0/11; teacher-first
then final Selective survived 5/11 versus 6/11 for that learner's own first
action. These are correlated anchors from three trajectories, not reset-level
reliability estimates. The observed short/long rank reversals favored teacher
actions *more* at long horizon, so they do not demonstrate harmful teacher
myopia. Independence is supported; successful transfer is not established.

The historical [critic continuation probe](experiment_evidence/2026-10-25_independent_teacher_diagnostics/README.md)
evaluates a different \(Q^\pi\), with gamma .995 and stochastic SAC
continuation. Its 7/8 ranking-sign agreement is unresolved: all eight paired
Monte Carlo intervals include zero. It cannot validate the present
\(V^{E_r}\), gamma .999 or a new learner update.

The [memory ablation's raw report](../runs/2026-10-24_memoryless_001/report.json)
shows retained-plan K32 completion 39/50 versus 0/50 when resetting before
**every** decision. This motivates a handover restart test, but is not that
test: the current oracle restarts only once. No matched retained-plan versus
single-handover-restart value comparison is available for the current oracle.

The running walking trial is left unchanged. Its local mirror is incomplete
and may lag the server; it supplies no final standalone result for this review.
Intermediate stage advancement is not mechanism evidence. The frozen trial
should finish and be reported under its existing protocol regardless of this
recommendation.

## 6. Smallest repair and the learning signal

The literal correction to (1) is to add a measured learner completion gap:

\[
 A_L(\theta)+\gamma^L\mathbb E
 [V^{\pi_\theta}_{N-L}(z_L)-V^{E_r}_{N-L}(z_L)]=J(\theta). \tag{9}
\]

This repairs outcome ownership. In the exact case the teacher tail cancels,
revealing that an expert terminal value alone has supplied no extra executable
exploration. Estimating only the expert half accurately cannot fix this. A
learner-tail value must belong to each candidate learner, or the score is a
different frozen-continuation surrogate. At this mechanism stage it is safer
to execute the complete learner suffix than add another learned critic.

To retain a useful role for independent planning, move its contribution to the
**distribution of starting states**. An independent MPPI trajectory can reach a
state after coordinated preparation that the weak learner rarely visits.
Train the learner's completion there before asking it to discover that
preparation itself. This is the smallest useful role change: reverse the
handoff, keeping the teacher, simulator, task cost and policy-search operator.

For archived independent executed teacher states \(d_b^E\), define

\[
 \boxed{K_b(\theta)=\mathbb E_{z_b\sim d_b^E}
       V^{\pi_\theta}_{N-b}(z_b),\qquad b\downarrow0.} \tag{10}
\]

MPPI executes the prefix \([0,b)\), or a saved state from that actual prefix is
restored. The learner executes **every** remaining control \([b,N)\), including
all recovery and failure. There is no MPPI rescue after handover. Teacher
prefix costs are constant with respect to the candidate and excluded from its
learning signal. The whole assisted controller has cost

\[
 B_b(\theta)=\mathbb E[C^E_{0:b}+\gamma^b
                         V^{\pi_\theta}_{N-b}(z_b)], \tag{10a}
\]

where prefix cost is discounted from the original root. At fixed \(b\), its
positive scaling and additive constant preserve the minimizer of \(K_b\)
(though scaling changes a gradient step's size). At \(b=0\),
\(K_0=J\) on the deployment reset distribution.

The teacher retains its own plan throughout collection, independent of the
learner. Archive only actually reached states for the primary mechanism;
unexecuted planning candidates are not automatically valid handover data.
Save complete restore state, warmstart, original time index, failure status
and provenance. Save teacher plan/RNG separately for reproducibility and
restart diagnosis; they are not actor inputs or imitation targets. An absorbing
teacher trajectory contributes its declared absorbing result at later handover
times, rather than disappearing from the denominator: \(C^E_{0:b}\) includes
absorbing costs through \(b-1\), and its remaining value is
\(5\sum_{j<N-b}\gamma^j\). A simulator snapshot
reset is a model intervention, not evidence that a physical learner reached it.

In the coordinated example, use a sufficient feedback class
\(a_0=p,\ a_1=k a_0\). From the teacher-discovered state \(a_0=1\),
\(K_1=(k-1)^2\). Outcome optimization learns \(k=1\) without observing an
action target; at \(p=0\), standalone cost remains 1. Then
\(K_0=\tfrac14p^2+(p-1)^2\), whose minimum is \(p=.8\), cost .2.
The acquired continuation turns a preparation action from harmful to useful.
This is the precise proposed exploration transfer.

It is still conditional. With a phase-only class \((p,q)\), training
\(q:0\to1\) at the same teacher state improves \(K_1:1\to0\) but worsens
the root learner \(J:1\to2\). That class can represent the optimum; the
problem is interference across state distributions. Reverse handover repairs
expert-return ownership, not all distribution shift or neural interference.
If the update preserves decisions before \(b\), then

\[
 J(\theta')-J(\theta)=\gamma^b\mathbb E_{z\sim d_b^{\pi_\theta}}
 [V^{\pi_{\theta'}}_{N-b}(z)-V^{\pi_\theta}_{N-b}(z)]. \tag{10b}
\]

Thus improvement on that learner boundary distribution, or pointwise on its
support, suffices. Improvement averaged over \(d_b^E\) alone does not; coverage
without control of signs and weights is insufficient. A shared neural network
may also change decisions before \(b\). A finite root admission panel gives a
narrower measured safeguard. These safeguards can fail to yield progress;
the mechanism must be allowed to stall. A coordinated multi-parameter proposal
can pass where isolated suffix changes cannot.

## 7. One implementable update loop

Use the existing antithetic parameter search, without a critic in the first
mechanism experiment. Let \(\epsilon_i\sim\mathcal N(0,I)\) and
\(\theta_i^\pm=\theta\pm\sigma\epsilon_i\). Execute their full learner
suffixes from common saved states, then propose

\[
 \widehat g_b=\frac1{2M\sigma}\sum_{i=1}^M
 [\widehat K_b(\theta_i^+)-\widehat K_b(\theta_i^-)]\epsilon_i,
 \quad
 \theta'=\theta-\alpha\widehat g_b, \tag{11}
\]

with the existing bounded step norm. This estimates a gradient of the
parameter-smoothed objective, not a guarantee for the unperturbed candidate.

```text
Collect fixed independent MPPI episodes with retained own plan/private RNG.
Archive executed snapshots with original time index; retain all failures.
For b in a predeclared decreasing handover schedule ending at 0:
    Freeze the current learner and common training/verification snapshots.
    Evaluate antithetic candidate learners through the entire remaining N-b.
    Propose one bounded parameter step using only their own finite costs.
    Execute old and proposed learners on the same handover verification set.
    Also execute both from the same ordinary root reset panel through N.
    Log K_b, B_b = prefix_cost + gamma^b*K_b, and J before/after.
    Admit only measured K_b improvement and no root-cost degradation.
    Advance only after the declared learner-suffix criterion; otherwise stall.
At b=0, train/evaluate learner-only episodes; no planner/value calls in scoring.
```

The root panel used for admission is development data; a separate locked panel
measures the final result. Root ties must be allowed during suffix learning.
Empirical admission is not population monotonicity, and repeated use can
overfit the panel. It may also reject useful preparations and stall. Do not
interpret rejection as proof that teacher states have no information.

If learning a value later becomes necessary, the target is explicitly
\(V^{\pi_k}(z,n)\) from **learner-executed** complete suffixes, frozen for a
declared policy version, with full-rollout verification of new candidates.
This first test deliberately avoids that extra approximation.

## 8. Decisive diagnostics, with failure criteria fixed first

| Test | Matched measurement | Outcome that falsifies the named explanation |
| --- | --- | --- |
| Assistance ownership | Same old/new parameters; exact \(A_L,J,G_L\), identical roots/horizons | \(A_L\downarrow,J\uparrow\) falsifies universal assisted→standalone improvement and critic error as its sole explanation. Equation (5) must close numerically. |
| Critic contribution | Freeze a current critic, fixed candidate policies, endpoints, horizons, full restore states and private seed panel. Compare learned values with repeated realized restarted-teacher costs on those **same** queries; include learner suffix values separately. Score the same update with direct finite-seed returns and learned values; compare both proposals from the same perturbations. Analytic toys additionally provide exact expectations. | If failure remains with exact values, fixing the critic alone cannot repair it. If learned values reverse resolved candidate rankings and direct scoring repairs the update, critic error contributes. Unresolved MC differences decide neither. |
| Restart contribution | At already archived **teacher** anchors, cross retained historical \(U\) and fresh-zero \(U\), holding physical state, solver warmstart and future random draws fixed. Use current first-call refinement rules and log their extra work; a separately matched-refinement comparison isolates the plan itself. Replan thereafter in both arms. | If mean costs/rankings stay within a predeclared practically small range, restart is not a material explanation on that panel. A large resolved effect falsifies interchangeability of retained and restarted teacher values. It does not fix the gap in (5). |
| Class versus search | Exact tiny problem: analytic best-in-class value, known executable policy, actual optimizer output; optional sufficient-state diagnostic actor | No successful policy in the deployed class identifies a representation limit. A successful same-class witness plus failed exact search identifies an optimization limit. Teacher action disagreement alone identifies neither. |
| Exploration transfer | Reverse handover: same-update learner suffix and root outcomes; then compare final root behavior with identical search lacking guide-state resets | Better handover outcomes with no final root gain falsify the transfer claim for this budget. Final guide-state benefit disappearing under learner-only evaluation identifies remaining dependence on changed starts. |

Small fixed panels suffice for the next diagnostic distinction; they need not
be learning studies. For critic variance, reuse all six saved smoke queries
above, horizon 63, with eight private seeds chosen before simulation. Compare
the frozen predictions to direct seed-conditional returns and their means,
including paired candidate rankings only where queries share a root. There is
no available exact expected MPPI value: analytic toys give exact expectations,
and deterministic simulator replay gives a direct conditional outcome for a
fixed random stream. Replication estimates the expectation with uncertainty.

For a restart probe, use existing independent-teacher snapshot indices
`[1,3,5,8]`, four predeclared future streams, both retained-plan and
fresh-plan arms: 32 suffixes, each through original time 1000 or termination.
Record costs at 64 controls too. Before execution, define a material effect as
1% of the remaining finite absorbing-cost scale; equivalence on this panel
requires the paired uncertainty interval inside that band. An interval
overlapping both material improvement and harm is unresolved. Snapshot 0,
where the archived planner is already fresh, supplies a same-controller replay
guard outside this comparison. Original first-call refinement rules form the
primary comparison; equal-refinement plan-only controls are needed only if
interpreting a primary difference specifically as a plan-memory effect.

Retained-plan comparison is meaningful at teacher anchors with a real
historical planner snapshot. A policy-reached endpoint has no canonical
historical MPPI plan. Do not invent one and call it retained memory. Common
random numbers reduce variance but do not make a finite-seed mean exact; save
all replicates and account for unequal stopping times. No new restart test
requires changing the teacher in the active walking trial.

The companion [analytic diagnostic](../scripts/diagnostics/assistance_transfer_check.py)
records paired same-update results, the gap identity, the feedback repair,
the phase-only interference case and finite absorbing arithmetic. This is a
mechanism check, not Walker evidence. The existing executable takeover/ES
regressions remain a separate check of current code paths.

Executed result: [results.json](../runs/2026-10-25_assistance_transfer_check_001/results.json)
passes every analytic assertion, including both adverse same-update cases.
The existing takeover, coordinated-action, expert-value, withdrawal and
parameter-search tests pass **73/73** with:

```sh
.venv/bin/python -m pytest -q tests/test_expert_withdrawal_counterexample.py tests/test_expert_continuation.py tests/test_expert_withdrawal.py tests/test_expert_value.py tests/test_prefix_search.py
```

## 9. Closest primary work: established versus open

**Expert outcome information without cloning is established.** AggreVaTe uses
learner roll-in and expert cost-to-go. Its standalone guarantees require
no-regret learning and suitable class/coverage/error conditions, not monotonic
improvement of each assisted update. Section 2.5 explicitly warns that an
expert can favor continuations beyond the learner's capabilities.
[Ross & Bagnell, 2014, §§2.2–2.5](https://arxiv.org/pdf/1406.5979).

AggreVaTeD gives differentiable expert-cost updates and discusses the gap
between no-regret theory and nonconvex deep-policy optimization. Its favorable
exploration separations assume access to optimal cost-to-go in particular
settings, not finite-budget MPPI.
[Sun et al., 2017, §§3–5](https://proceedings.mlr.press/v70/sun17d/sun17d.pdf).

**Expert terminal values and longer learner horizons are established.**
THOR analyzes truncated learner cost plus an oracle terminal value. Theorem
3.2 requires simultaneous statewise minimization and uniform oracle accuracy
relative to optimal value. A reset-only ES improvement with a restarted MPPI
value is not that theorem's premise.
[Sun, Bagnell & Boots, 2018, §§2–3](https://www.ri.cmu.edu/app/uploads/2018/01/THOR.pdf).

LOLS gives a closely related negative result: expert-only rollout loss can
have vanishing surrogate regret while missing improvements relative to the
learned policy's own continuation. Its expert/learner rollout mixture has a
tradeoff guarantee, not per-update standalone monotonicity. This supports
measuring whose continuation is actually improved.
[Chang et al., 2015, §3](https://proceedings.mlr.press/v37/changb15.pdf).

**Guide prefix followed by learner suffix is also established.** Jump-Start RL
uses a guide-induced curriculum of learner starting states. Its exploration
bound needs sufficient representation and a coverage/density-ratio condition;
an arbitrary useful guide is not enough. Its simplified provable backward
scheme is explicitly connected to Policy Search by Dynamic Programming.
[Uchendu et al., 2023, §§4.1–4.3](https://proceedings.mlr.press/v202/uchendu23a/uchendu23a.pdf).

Potentially new here is a narrow empirical result: whether independent
finite-budget receding-horizon MPPI, with retained private plan memory, finds
states from which this observation-limited learner can acquire a transferable
coordinated continuation under outcome-only policy search. A new guarantee,
general exploration advantage or sample-efficiency claim is not established.
The memory/state qualifications above are deductions about this implementation,
not additional guarantees supplied by those papers.

## 10. One falsifiable claim and one bounded experiment

**Claim:** under the fixed budget below, independent MPPI handover states
allow the fixed reactive policy/search procedure to acquire a learner-executed
continuation that improves final standalone finite-horizon cost more than the
same procedure trained only from ordinary resets. Teacher rescue, teacher
actions as targets and an expert-value critic are unnecessary for that gain.

The next experiment should be a separate, complete-return **handover feasibility
assay**, after the analytic checks, with the active walking trial unchanged:

1. Freeze one weak initial network, task, gamma .999, \(N=1000\), the existing
   independent K32/H48 teacher and ES settings before execution. Predeclare six
   disjoint ordinary development reset/planner seeds after auditing existing
   ledgers; do not use reserved qualification seeds. Collect six teacher
   episodes once, retaining their plans and all failures. Four episodes provide
   training handovers, two provide verification handovers. No replacement for
   failed teachers. At each positive handover time, require at least one
   healthy training episode and one healthy verification episode; otherwise
   record an uninformative-handover stall. Keep absorbing rows in all means.
2. Test two uses of the **same search procedure**, initialized identically:
   guide-state training with handovers \(b=900,500,0\), versus ordinary-root
   training throughout. This is one mechanism ablation, not an algorithm
   benchmark. Cap at ten proposals per stage, eight antithetic pairs per
   proposal, four training and two verification starts. Run the guide arm
   first and give the root control exactly the same number of proposals
   actually attempted (at most 30), with matched perturbation seeds, even if
   the guide advances early or stalls. Record differing simulated-control
   costs without an efficiency claim. No critic, cloning, tuning, extra stages
   or rescue. Fix sigma .01, learning rate .01, maximum step norm .05, and
   the current cost-normalization rule in both arms.
3. Score every candidate with its actual complete learner continuation. For
   each proposed update record assisted whole-controller cost, learner-only
   handover cost and ordinary-root standalone cost **before and after that
   same update**, whether admitted or rejected. Use the six ordinary roots
   underlying the teacher episodes as the fixed root admission panel, separate
   from final evaluation. Allow exact root-mean ties but reject degradation;
   keep rejected proposals in the record. Apply the same root admission check
   to both arms.
4. Advance from \(b>0\) only if verification suffix means satisfy
   \(\overline K_b(\theta)\le\overline K_b(E)+.05\max(1,|\overline K_b(E)|)\),
   using the archived teacher suffix as reference; this is a declared
   curriculum heuristic, not proof of root competence. Require a strict mean
   improvement relative to the stage-entry policy on the informative healthy
   verification subset as well as the
   full-panel 5% check, so all-absorbing rows cannot advance a stage. At a stage
   cap, stop and report a stall. At \(b=0\), run its ten proposals. Evaluate
   both configured final networks once on the same 16 separately locked
   development resets, learner alone for 1000 controls, retaining failures.
   Report finite cost, survival and velocity MAE. This panel does not qualify
   reliable walking or training-seed robustness.
5. **Continue this mechanism only if** guide-state training reaches \(b=0\),
   has lower mean final standalone finite cost than both its own initial policy
   and the root-only control, and improves over the control on at least 12/16
   matched final resets. Also require at least one accepted earlier suffix
   update whose acquired continuation is still improved at the final network.
   These are fixed feasibility thresholds, not a power calculation or a
   significance claim. A stall, suffix-only benefit, or a final gain appearing
   only in the root-only stage without retained suffix improvement does not
   support the proposed transfer mechanism. Do not respond by expanding the
   budget or adding algorithms within this assay.

The analytic counterexample settles the invalid universal implication now.
This bounded assay asks the remaining constructive question: can MPPI's state
discovery train a continuation the learner can actually carry back to ordinary
starts? Only a positive answer warrants a larger exploration study.
