# Joint MPPI–policy goal and implementation boundary

Status: v31 accepts two guided-data cost/risk steps with64/64 fresh captures, but its .00006411 complete-cost gain misses the .001 qualification requirement. No changed child qualifies, no new search authority is granted, and a beneficial repeated loop remains unestablished. See [results](results.md) for the separate retained outcomes, including v28's numerical fragility and v30's negative Adam comparison.

## Current deployment-cost/risk admission v31

This [frozen two-attempt test](../configs/coupled_ending_deployment_cost_risk_v31.json) uses the same qualified incoming actor and original MPPI teacher packet as v30. It measures the controller that would actually deploy: deterministic native batch-one inference. Four freshly executed ordinary starts supply its original complete-cost gradient gD. Sixty-four current-actor stochastic executions supply gR, the discrete detour/ending score gradient with the existing teacher-state envelopes. These are different owned tapes and different objectives; a stochastic pathwise gradient is never called a deterministic deployment gradient.

For each alpha in {.01,.003,.001,.0003}, the update minimizes `gD·delta + ||delta||²/(2 alpha)` within radius .01 and the local constraint `gR·delta <= 0`. This is a closed-form projected step. Incoming log standard deviation stays fixed. No optimizer is constructed, and the full original optimizer/history remains unchanged in checkpoints. Every candidate pays for four complete ordinary batch-one executions and another64 stochastic executions with matched contexts and random seeds. Acceptance requires actual ordinary-cost improvement above1e-8, ordinary and stochastic capture nonregression, measured equal-context discrete-risk nonregression, realized parameter radius and linear-risk checks. All four candidates run; the first eligible one is selected. No new ending margin or absolute risk budget is introduced. The paired sample-risk gate does not prove population-risk nonregression.

Ordinary stochastic contexts have beta0 and no teacher budget; eight acquired teacher states at clocks4/8 have beta1 and real executed budgets. All twelve contexts have equal weight. Successful teachers here have the same discrete budgets, so the new risk constraint uses their state coverage; it does not exploit their different continuous component quality. No teacher control becomes an action label. Initial hybrid-credit and native-deployment continuous-cost finite differences are paid. Discrete quotient jumps remain recorded and are not relabeled as smooth derivatives.

Separate fresh sixteen-start **batch-one** qualification requires capture nonregression, at least90% capture and cost improvement above .001. Fresh64-start panels evaluate both retained controllers and unqualified frontiers; they never select a checkpoint. Batched panels are numerical diagnostics with distinct inference contracts. The [closed run](../runs/2026-10-28_deployment_cost_risk_001/result.json) passes derivative checks and paid replay: RAW accepts zero steps, GUIDE two, and both frontiers capture64/64. Guided frontier cost1.6370572267 versus parent1.6371213375 is a gain of6.41e-5, below .001; neither child qualifies. [Saved-array reporting](../runs/2026-10-28_deployment_cost_risk_report_001/receipt.json) verifies actual candidate gates and preserves this limited result.

This does not identify teacher-data leverage. The saved-step audit finds that the smaller RAW candidates fail only the realized linear-risk gate: float32 parameter rounding changes the ideal near-zero risk dot product to about7.6e-8 and1.18e-7, above the declared1e-8 tolerance. The first GUIDE candidate at iteration2 receives the same veto. The recorded rejections remain valid under the frozen rule, but accepted-update counts alone cannot distinguish useful teacher-state information from this numerical gate asymmetry.

Producer and verifier together charge94,304 scalar controls,125,680 neural rows,54 reverse passes,24 closed-form projections and1,026 queries, with zero Adam calls, failed queries or native physics. Entry wall time is21.931686625s; read-only reporting adds9.870s. Core validation and prior discovery, search and development remain additional costs. There is no efficiency or complete-loop claim. A future qualified child must reenter fresh independent search with zero imported trust before any actor-to-search benefit can be tested.

## Stochastic complete-outcome credit admission v30

This finite development test replaces the deterministic learner's missing discrete-event credit. Its input is the unchanged qualification-selected v14 actor and the original v25 executed teaching packet, copied from v28's inputs. V28's unqualified frontiers never initialize learning. The native actor inference implementation, task thresholds and old verdicts stay fixed. The new learner starts fresh Adam explicitly; its complete incoming optimizer and history remain preserved as source evidence.

For a context c, let C be the four measured complete component totals of the current stochastic actor. A successful executed teacher supplies its exact-state remaining-episode budget B. Define each component loss as C_k/25 plus beta_c times squared positive exceedance of B_k, scaled by the original component scales. Ordinary training starts have beta_c=0 and no teacher budget. Acquired starts have beta_c=1 and a real teacher budget. Split this loss into continuous effort/greed S and discrete detour/ending D. The critic-free update estimates

\[
 g_c=\frac1R\sum_r\left[\nabla_\theta S(\tau_{r,\epsilon})+
 (D_r-\overline D_{-r})\sum_{t<\mathrm{gate}}
 \partial_\theta\log\pi_\theta(u_{r,t}\mid o_{r,t},z)\right].
\]

The first term follows the sampled actor's complete continuous feedback consequences at fixed noise. The second holds observed states and sampled latents fixed and assigns measured ending-event credit through their generating actor's conditional likelihood. Only the discrete loss receives score credit; adding total-loss score credit would double-count the continuous derivative. Repeated executions are independent within each identical context. All twelve contexts receive equal weight, with 1/R weighting within each context. Teacher controls supply neither action labels nor actor gradients.

Each update uses 32 ordinary executions (four fresh training roots, eight repeats each) and 32 acquired executions (eight teacher states at clocks 4/8, four repeats each). Thus ordinary contexts have one third of the equal-context objective weight despite half of the execution count. Every return belongs to its exact generating actor. One fresh hybrid gradient produces one clipped Adam step at learning rate 3e-4 and gradient bound .5; changed actors require new complete executions. Float32 rounding is not differentiated. Same-shape recomputation must reproduce recorded actions and densities exactly, and prospective complete-feedback and conditional-density finite differences must pass before optimization.

The [frozen configuration](../configs/coupled_ending_stochastic_outcome_admission_v30.json) permits two updates per arm. Each producer/verification phase is capped at 30,000 scalar controls, 100,000 neural rows, 32 reverse passes and eight Adam calls. RAW/GUIDE share their ordinary roots and random-draw rules. They differ in their original acquired teaching states and measured budgets. The beta-zero contrast and all derivative probes are charged. A full paid repeat compares arrays and complete checkpoints; the original eight teacher episodes receive another 200-control transition replay.

Fresh sixteen-start qualification uses the actual native batch-one deployment controller. It requires capture nonregression, at least 90% capture and complete-cost gain greater than .001. Native batched panels are separate numerical diagnostics. Fresh sixty-four-start evaluations report retained actors and every rejected frontier; they select no checkpoint. This admission establishes neither policy-to-search improvement nor a completed bidirectional algorithm. A changed policy still needs fresh, zero-trust, executed search qualification and a matched next-policy test. Physical Acrobot and Walker transfer remain separate work.

Mixed pathwise/score estimators are established prior art; see [Schulman et al., 2015](https://proceedings.neurips.cc/paper/2015/file/de03beffeed9da5f3639a621bcab5dd4-Paper.pdf). This experiment tests their use inside the measured MPPI teaching loop, rather than claiming the estimator itself as new.

## Joint feedback learner v28 and return-to-search v29

V28 changes the update rule while holding v27's incoming full actor, original v25 trial/confirmation teacher packet, task, architecture, component-envelope objective and fixed0/4/8 contexts unchanged. All initial current-actor state/action/component/budget arrays reproduce v27. The earlier negative search and learning results remain negative. This is a development test of the optimization explanation, not a new task or a relaxed benefit criterion.

For each arm, measure the incoming complete ordinary-start cost J0 once. At every current actor, execute its uninterrupted suffixes from all12 teacher-defined contexts and differentiate those exact owned outcomes. Teacher controls are not labels. Let F be the existing mean complete cost plus one-sided teacher component-envelope penalty, h the12 measured ending residuals, G their current gradients, and c the current mean ordinary-cost gradient. When all contexts capture, propose a joint step minimizing normalized(gradient F) dot delta + norm(delta)^2/(2alpha), with h+Gdelta<=0, c dot delta<=J0+1e-8-Jordinary(current), and norm(delta)<=.01. This replaces a cost step followed by separate ending repairs.

If some acquired contexts fail, propose a bounded common-descent ray for the first-worst ending residual and ordinary cost, bounded by ordinary-ending rooms and the same fixed cost anchor. This ray is not a minimax-optimality claim. Complete executions must improve the worst positive residual by more than1e-6 or newly capture all12. Both phases preserve ordinary capture and the fixed incoming cost window. Improvement steps additionally require all12 actual captures and F improvement>1e-8. Float32 realized steps, numerical certificates and all rejected complete outcomes remain archived. The full certificate appears once per proposal; compact NPZ references avoid copying its full vectors into every candidate row.

The interface permits two total attempts per arm and requires two actual accepted updates in an arm, including at least one joint cost update, plus paid replay. It is bounded per phase by50,000 controls,200,000 neural rows,256 reverse passes,16 QPs,4 rays and72 parameter constructions. Only a passed interface admits the32-attempt full run with500,000 controls,one million neural rows,4,096 reverse passes,256 QPs,64 rays and1,032 constructions per phase. Both phases share each attempt cap. There is no budget extension when an arm stalls.

Fresh16 qualification still requires source12 capture, capture nonregression/floor90%, and cost gain>.001. Fresh64 evaluation separately compares guided-data and independent-data actors with the incoming actor. A qualified changed child must return through v29 to fresh zero-trust suffix tests and executed search comparisons with the exact incoming actor and unchanged inherited360-fragment RAW memory. An actor-only gain does not finish the feedback test. The [predeclaration](../artifacts/2026-10-28_joint_feedback_plan_001/predeclaration.json), [zero-query preflight](../artifacts/2026-10-28_joint_feedback_preflight_001/receipt.json) and [210-test core run](../runs/2026-10-28_joint_feedback_core_001/summary.json) precede the new learning queries.

## Qualified-child feedback continuation v25–v27

This continuation closes the next executable feedback round using a child that already passed the original complete ordinary-start qualification. It does not retune or erase v14's negative guided-data comparison. The v17–v23 event-state experiments added harder reset contexts and did not produce a qualified replacement; those diagnostics are not a universal deployment prerequisite.

V25 imports the v14 fresh-sixteen-selected full child checkpoint, its exact incoming actor, the unchanged 280-fragment independent RAW memory, and four executed support traces. Optimizer, RNG and learning-history state remain bound. No old trust is imported. The task, protected search proposals, agreement ladder, suffix tests, trial/confirmation/evaluation sizes and cost criteria equal v20. Only the explicitly declared source profile and disjoint root namespace change. Each producer/verification phase is bounded by six million scalar controls, two million neural rows, zero reverse passes and zero optimizer calls. Any return-to-search claim requires fresh earned authority, actual changed controls, lower complete executed cost without capture regression against independent memory search and the incoming actor, plus exact forced revocation.

V26 imports only the resulting newly qualified executed teacher pairs and uses the **unchanged v14 learner**, not the event-state restoration curriculum. Teacher states and measured complete component budgets constrain the actor; teacher actions are never labels. Current-actor uninterrupted outcomes supply credit. Both children share parent, optimizer work, learning method and ordinary qualification/evaluation roots. The two-attempt interface is bounded per producer/replay phase by 50,000 controls, 200,000 neural rows and 256 reverse passes. It must actually accept two updates in an arm and pass paid replay before the full run. The full run has 32 total attempts per arm, 500,000 controls, one million neural rows, 4,096 reverse passes and zero Adam calls per phase. All rejected candidates and verification costs remain charged.

The next-policy benefit criterion remains guided-data child cost at least .001 below both its independent-data child and incoming parent, with capture nonregression and at least 90% capture on fresh 64-start ordinary evaluations. Fresh sixteen-start panels select checkpoints; sixty-four-start evaluation does not select them. A negative result completes a test of the structural loop but does not establish a beneficial bidirectional algorithm, efficiency or physical transfer. Commands are in the README. V20/v21 and all older frozen sources/results remain unchanged.

V25 completed with paid audit PASS but search feedback FAIL: current guidance improved over independent search while losing to the incoming-policy guidance. Original v26 therefore does not start. **V27 is a new, prospectively frozen conditional experiment**, not a changed v25 success criterion. Its source gate requires the same paid audit, actual promotion, four useful original teacher pairs, held-out gain over independent search, changed executed controls, exact revocation and committed RAW memory, while explicitly retaining the failed incoming-policy comparison. Held-out outcomes may admit this follow-up but never become its training contexts or targets. V27 uses the same v14 learner, caps, two-accepted-update interface and next-policy benefit criteria stated above, with disjoint72M/74M root namespaces. Its [predeclaration](../artifacts/2026-10-28_conditional_feedback_plan_001/predeclaration.json) retains the old negative comparison. Neither a positive conditional result nor structural handoff can turn v14/v25 into an all-positive two-round result.

## Relationship to CIO-based guided policy search

Mordatch and Todorov's [2014 joint trajectory/policy method](https://www.roboticsproceedings.org/rss10/p52.pdf), sections II–III and equations 3–7, jointly optimizes task performance and policy reconstruction. The policy shapes trajectory optimization, while optimized trajectories constrain the policy. Replacing its local optimizer with MPPI does not reverse these arrows. Both are needed.

The professional next step is to derive and test this coupled optimization before resuming training. The raw material is simulator transitions, actual pre-action states/actions, complete outcomes, the independent planner's RNG/nominal/memory, and the raw query ledger. Previous reports and imitation losses cannot establish the coupling.

The old implementation supplied acquired states and outcome qualification but lacked an explicit teacher-dependent constraint in the actor update. The goal is to repair that structural omission, then test whether it improves outcomes. Similarity to CIO-GPS is not itself evidence of benefit or a novelty claim.

## Outcome-envelope formulation v0

Let `q` denote the teacher and `πθ` a feedback action policy. Let `C(τ)` be the vector of complete measured cost components. On Ending these are effort, greed, detour and ending failure; the dynamics, costs and final event are unchanged. A successful executed teacher supplies a context/family-specific budget `b=C(τq)`. Hypothetical plans cannot supply that budget. Exact context includes the physical state and remaining time.

Define the one-sided discrepancy

`D(a,b) = sum_j max((a_j-b_j)/s_j, 0)^2`.

It permits lower costs and different controls; it penalizes exceeding an achieved component budget. The actor minimizes

`E_actor[J(τ)/return_scale + β D(C(τ), b)]`.

All `τ` in that expectation are complete executions of the exact behavior actor being updated. The teacher contributes the envelope, never action labels or actor-return labels. This is cost-constrained outcome transfer; it is not a claim that all forms of imitation have disappeared. Ending feedback was already in the task. Search adds evidence of achievable, context-dependent cost tradeoffs, not new knowledge of what the goal means.

The structural test uses an exact score-function gradient with a leave-one-trajectory-out baseline among 64 repeated executions of one context. Because `D` is nonlinear in complete component totals, its whole-episode penalty multiplies the whole trajectory score. It is not assigned a fictitious per-state critic target. A clipped update is then qualified through fresh executions. A score-function adapter can later update the family MLP; the test's three-parameter linear Gaussian actor is explicitly a diagnostic fixture.

For policy-to-teacher coupling, keep independent MPPI's entire sampled bank and private state. Copy its candidate prefixes, simulate each prefix, then require its remaining suffix to use the frozen policy. This is a controller constraint, not action regression. Denote that complete hybrid forecast by `τ(U,π)`. Also execute a complete policy-only forecast from the same initial state to obtain `Cπ`. The candidate score is

`S(U) = S_H(U) + λ [J_tail(U,π) + α D(Cπ, C(τ(U,π)))]`.

In this scalar task, `S_H` is the actual prefix cost with no intermediate terminal heuristic. At `λ=1`, this is complete hybrid task cost plus agreement. At `λ=0`, it is exactly the independent score and policy evidence is unused. The reverse discrepancy discourages teacher solutions with component requirements very far ahead of the student's achieved behavior. Task quality and ending remain separate gates.

Both directions therefore use the same component envelope: the teacher adapts its achievable requirements to the student; the student attempts to satisfy requirements supplied by an actually successful teacher. This resembles the two-way optimization structure of CIO-GPS while replacing its action reconstruction objective. It is an empirical alternating surrogate, not literal CIO-GPS or exact coordinate descent: a hybrid forecast is not the executed replanning teacher, and complete measured teacher outcomes must replace it before the next actor constraint is formed.

The two coupling strengths need not be equal: verified successful teachers may constrain learning before the actor has earned influence over search. Actor-to-teacher influence starts at zero and requires support/version-bound actor tests followed by paired **executed replanning** improvement. Changed policy versions and failed qualification revoke it. This is a constrained alternating research formulation, not an inherited ADMM convergence guarantee.

Protected proposal generation remains actor-free at a restored identical physical/planner state. The coupled selector works on copies. It verifies any synthesized plan and retains independent fallback; learned selection cannot overwrite the private independent nominal, RNG or memory scores. State-distribution changes caused by executed guidance are reported separately.

The current mechanism command computes hypothetical teacher updates only; it deploys no guidance and creates no new guided teaching dataset. It checks zero influence and invalidation, but does not claim to have exercised revocation from an empirically promoted setting. That and paired executed replanning are required in the next stage.

## Frozen structural experiment

Run through the sole research entry point:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps mechanism \
  --config configs/coupled_ending_mechanism_v0.json \
  --out-dir runs/coupled_ending_mechanism_001
```

The command freezes active source and configuration before querying the task, then executes from that snapshot. Eight complete independent long-search episodes discover teacher evidence. Two successful executions from the same physical context must exist. Their different budgets constrain matched updates using the exact same 64 actor-owned trajectories, parent, step settings and compute. Coupling-off updates must match exactly. Changed actors are compared on one restored short-search bank, then that bank is regenerated to test independence. Three matched one-step fixture children receive 16 fresh ordinary-start executions each.

The predeclared structural gate requires both actor gradient and teacher objective dependence, a changed counterfactual teacher first action, exact off-switch and proposal behavior, ownership rejection and zero unqualified/invalidated influence. One-step child outcomes are reported separately; this is not a neural training or efficiency result. The producer cap is 150,000 scalar controls; a separate 150,000-control verification cap pays for transition replay of every query. All fixture inference and three diagnostic gradient applications are counted. Native physics and neural runtimes are blocked. A failed test retains its source, budget and verdict.

The replay auditor checks raw transitions, component costs, fixture actions/scores using the original RNG, independent proposal generation, mean verification, selected plans, private state, teacher budget ownership and actor-gradient arithmetic. It uses a separate counted executor. Numerical source identity is not a second physical-model validation.

Version v0 closed with a failed teacher-gradient contrast: the first two same-context successful teachers had exactly identical trajectories/budgets. The query replay passed and the actor changed the counterfactual teacher decision, but identical teacher inputs cannot test teacher dependence. The failure is retained in `runs/2026-10-27_coupled_ending_mechanism_001`.

Prospective diagnostic v1 uses the same eight discovery executions, settings, roots, actor data amount, thresholds and 150,000+150,000 control caps. It scans contexts in the declared order and requires two successful budgets with component-vector distance greater than `1e-8`; otherwise it stops for absent contrast. Its configuration is `configs/coupled_ending_mechanism_v1.json`. Context selection depends on contrast availability, not child evaluation outcomes. This repairs the intervention design, not an observed learning failure. The new run pays for repeated discovery and discloses the v0 development cost.

V1 passed its structural gate and paid raw replay. The canonical command now defaults to its configuration. See [results](results.md) for the small fixture-child difference and the distinction between changed hypothetical teacher decisions and untested executed guidance.

## Prior art and proposed addition

Jointly adapting trajectory optimization and policy reconstruction is established by the CIO-based GPS work linked above. Score-function policy gradients are established [REINFORCE](https://doi.org/10.1007/BF00992696). Completing candidate decisions with a base policy belongs to established [rollout methods](https://www.mit.edu/~jnt/Papers/J066-97-rollout.pdf). No rollout policy-improvement guarantee is inherited here: finite sampled candidates, the extra discrepancy, policy generalization and executed replanning must be tested.

The proposed research combination uses protected independent proposals, complete executed teacher component envelopes, actor-owned return learning and empirically revocable policy-constrained search. The cost-envelope objective is a deliberate change from CIO-GPS's action reconstruction. This structural test establishes implementation dependence in both directions, not novelty or a better learning algorithm. A complete literature/ablation assessment and the beneficial next-policy loop remain outstanding.

## Frozen neural development v0

The next command uses the same shared objective with the existing 128×128 family Gaussian MLP. It is one seed of open development on unchanged Ending, not the physical or three-seed scientific comparison.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_neural_interface_v0.json \
  --out-dir runs/coupled_neural_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_neural_v0.json \
  --out-dir runs/coupled_neural_development_001
```

The interface protocol runs two neural updates, a same-data constraint-gradient contrast, stale-owner rejection, checkpoint restoration, proposal replay and actual zero-influence fallback. Its caps are 400,000 scalar controls, 100,000 neural rows, 16 optimizer calls and 20 backward calls, separately for production and verification. Task success is not its software pass criterion. Full raw and optimizer replay must pass before the larger run starts.

The development protocol first acquires eight actual independent complete episodes from four roots. Successful executed traces supply full and acquired-state budgets. Each update collects exactly 64 actor-owned trajectories: four ordinary contexts repeated eight times, and eight acquired contexts (times 4 and 8) repeated four times. Teacher actions are never regression labels. Adam uses `3e-4`, gradient clipping `.5`, four clipped policy-gradient epochs at ratio `.2`, initial standard deviation `.8`, return scale 25 and envelope weight 1. All parameters, including log standard deviation, are trainable. The complete envelope penalty multiplies the whole effective trajectory score, while raw task cost uses measured Monte Carlo cost-to-go. Controls after the known gate cannot affect state or future cost; their gradient credit is masked, while their execution and inference remain charged.

There are 96 initial updates and 24 updates per matched child, two feedback rounds. Candidate deterministic policies are compared with their retained parent on paired fresh 16-root qualifications every eight updates. Promotion requires at least 90% ordinary capture, capture nonregression and improved raw complete cost. Unqualified training versions may continue learning but get no planning influence. Initial deployment must additionally pass 64 fresh ordinary starts. Failure closes this finite run without extra updates.

Independent control uses short H4 MPPI with 32 Gaussian, eight smooth and 16 memory proposals. It resimulates full nearest-neighbor continuations of every prefix, 16 direct recorded continuations and its own carried complete plan. All scores, RNG, nominal and carried plan updates remain independent. Learned selection cannot overwrite this state. This is a stronger reference than truncated fragment retrieval alone.

For each policy version, suffix tests use teacher states at times 0, 4 and 8 with position perturbations −.01, 0 and +.01. Empirical eligibility is limited to that clock interval and radius `.15`. Eligibility permits only explicit trials at `{.1,1,10}`. A setting needs changed controls, mean complete cost gain above `.001` and no capture regression in eight paired actual replanning episodes; a separate fresh eight-root confirmation is required for deployment. No setting is promoted from hypothetical plans. Unsupported states, changed versions and failed qualification have zero influence. Candidate completions must predict capture and strictly lower complete task cost than the independent fallback before selection; actual replanning remains the promotion test.

Each round collects four raw and four actual guided episodes from paired ordinary resets. It then trains matched children from the same parent and optimizer, equal update counts and common sampling seeds, using each dataset's measured budgets and acquired states. Complete fresh actor-only qualification selects teaching data for the next parent; separate 64-root evaluations compare both children and the parent. Sixteen fresh paired controller episodes test control benefit. Raw independent memory is updated only from raw executed search, even when guided data teaches the actor. A promoted version is deliberately invalidated and checked against actual independent fallback. Every new version must requalify in the next round.

The predeclared loop outcome requires two rounds with improved executed control, genuinely changed acquisition controls, a guided-data child beating both the raw-data child and its parent, successful promoted-then-revoked fallback, and at least 90% final standalone capture. Failing links remain separate outcomes. Caps are 10,000,000 scalar controls, 5,000,000 neural forward rows, 800 Adam calls and 810 backward calls, separately for producer and verifier. They are conservative bounds, not entitlements to expand a failed run. Scalar controls, neural work, optimizer work, failed reservations and complete wall time are reported separately. Native engines remain blocked. No efficiency claim follows without a later equal-cost comparison.

Complete query arrays are stored once in NPZ files, with durable per-transition input/output digests and slot counts in the ledger. This avoids multiplying small evidence files. Failed queries retain inputs, policy checkpoints/RNG, known completed slots and unknown remainder. The verifier independently replays every completed transition and neural sample, reconstructs independent proposals and private state, checks teacher budgets against actual complete suffixes, replays every optimizer update and binds selections to actual executed actions. It is software consistency evidence on this simulator, not independent physical-model validation.

The v0 run is closed negative, not extended. Its first failed trial supplied a precise counterexample: the scored four-control prefix plus actor would capture, while applying only its first control and replanning failed. The next implemented diagnostic freezes the final retained actor, uses the same independent bank/memory and agreement score, and compares independent control, the old replanning execution and prefix-then-actor execution on eight fresh paired roots at weight 1. It has no optimizer updates and a 2,000,000-control/1,000,000-neural-row cap separately for producer and verifier. Configuration and all input checkpoint, memory and support hashes are frozen before launch.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps diagnose \
  --source-run runs/2026-10-27_coupled_neural_development_001 \
  --out-dir runs/coupled_execution_diagnostic_001
```

The independent proposal process runs at every actual state in all conditions, including as a charged observer while the committed controller follows its prefix/actor continuation. It keeps its own nominal, RNG, full-plan carry and raw-only memory. At an eligible initial decision, the composite path is saved separately; subsequent prefix controls are executed, then the frozen actor is evaluated on actual states. This continuation support is the explicitly model-tested composite path, with state tolerance `1e-5`; it is not new state generalization or a blanket extension of the actor-only support radius. Changed policy/task versions, revoked authority, expired paths or state mismatch clear the commitment and use the current independent result. Actor-tail step queries have explicit composite-controller semantics and can never be treated as complete ordinary actor-owned returns.

The diagnostic requires identical initial independent banks, unchanged actor hash, changed executed controls, at least 90% complete capture for both the strong reference and commitment controller, mean complete-cost gains over `.001` against both the reference and the old replanning rule, and complete committed execution matching the scored continuation within `1e-5` in cost/state. It remains an unpromoted trial; positive results would require fresh full-controller confirmation before guided acquisition could teach the next policy. All ancestor training/replay costs remain disclosed. This is a limited controller diagnostic, not a replacement for the required next-policy learning test or a predetermined successful repair.

## Round-boundary continuation v1

The execution diagnostic passed on its eight declared roots; see [results](results.md). The next finite run changes only the controller execution rule to `commit_prefix_then_actor`. It resumes the closed v0 final actor, full Adam state, update index and both saved sampling/RNG states, with bitwise state checks. The independent memory and four raw support trajectories are copied with hashes. Those inherited trajectories qualify handoffs only; their old query IDs are never treated as new-run actor targets. All child budgets come from newly executed complete teacher traces in the new ledger.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_neural_commitment_v1.json \
  --source-run runs/2026-10-27_coupled_neural_development_001 \
  --out-dir runs/coupled_neural_commitment_001
```

There are zero repeated initial learning updates, two feedback rounds and 24 updates per raw/guided child. The original learning objective, four-epoch update, 64 actor-owned trajectories per update, qualification frequency, agreement ladder, cost/capture thresholds and strong independent search remain fixed. All reset-root blocks and actor sampling seed bases receive offset 2,000,000, so no previous qualification/evaluation root is reused. The raw and guided children share the same parent/optimizer, update work, sampling seeds and fresh comparison roots. Selecting teaching data still uses actor-only qualification, never child evaluation roots. If the guided child is selected, its measured trajectories also provide the next round's handoff support; independent memory still receives only raw-search executions.

The diagnostic's trial authority is not imported. Every resumed or changed policy starts at zero, must pass fresh suffix tests, paired complete trials and separate confirmation before deployment. Actual composite teacher suffixes retain the composite controller identity. A partial actor-tail step cannot be relabeled as a complete actor-owned learning return. Forecast-path support and commitment aborts are audited along with each independent/private decision.

The predeclared two-round success conjunction is unchanged: both rounds must improve held-out executed control; acquire changed guided experience; produce a guided-data standalone child beating both matched raw-data child and parent by more than `.001` with no capture regression; and exercise promoted-then-revoked exact fallback. Final actor capture must be at least 90% on 64 fresh starts. Failure of any link remains a negative loop result. Each of producer and full verifier has a fixed cap of 8,000,000 scalar controls, 4,000,000 neural rows, 400 Adam calls and 400 backward calls. Parent training, failed earlier formulations and the execution diagnostic remain additional charged development costs. The frozen run is not extended if a link fails.

## Qualified teaching acquisition v2: one-round causal gate

V1's four ordinary acquisition episodes contained no candidate that passed the complete-cost quality gate, even though trial/confirmation execution found actual useful improvements. V2 changes acquisition only. It keeps the learner, model, reward, independent search, execution buffer, agreement ladder and promotion thresholds unchanged. This test resumes the same closed v0 actor/Adam/RNG/memory, with all new roots and actor sampling seed bases offset by 4,000,000. Inherited costs and the negative v1 development remain charged separately.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_qualified_acquisition_v2.json \
  --source-run runs/2026-10-27_coupled_neural_development_001 \
  --out-dir runs/coupled_qualified_acquisition_001
```

First earn a fresh promoted setting through complete paired trials and a separate fresh confirmation. Only raw/guided execution pairs belonging to that confirmed weight/controller may teach. Consider trial pairs, then confirmation pairs, each in increasing declared root order. Keep the first distinct initial contexts in which both complete controllers captured, guided complete cost improved by more than `1e-6`, and actual physical states changed by more than `1e-8`. Let `N` be the largest of `{1,2,4}` no greater than the available distinct pairs; select the first `N`. No confirmation, no improving pair or numerical/ownership failure stops before child training. Failed weights, ties, losses and unselected successes remain archived; the pool is never expanded after observing results.

The selected executions keep their original query IDs/controller identities. New teaching aliases refer to those executions; they are not new episodes or free simulated controls. No additional four-root acquisition panel runs. Private independent memory ingests every distinct observed raw trial/confirmation trace at round end, without filtering by guided teaching success. Each arm retains its own states at times 0, 4 and 8 and its own measured remaining component budgets. A better complete guided episode can have a more costly suffix at its different acquired state; no same-state value superiority is inferred. Each update still has 64 uninterrupted actor-owned executions: `32/N` ordinary replicas per root, and `16/N` replicas at each of the two acquired times. Same-context leave-one-trajectory-out baselines remain valid. Teacher actions never enter a regression loss.

Train raw-data and guided-data children for exactly 24 updates each from identical full parent state and common sampling seeds, with the existing four-epoch outcome-envelope update. Use fresh paired 16-root actor qualifications at updates 8, 16 and 24, separate child selection roots, and 64-root actor evaluations. Fresh 16-root executed controller evaluation and promoted-then-revoked fallback remain required. No child or held-out control outcome selects the teaching pairs. This is **conditional curriculum acquisition**, not an unbiased estimate of population acquisition success or sample efficiency.

The one-round transfer gate requires actual changed teaching data, guided child cost more than `.001` below both matched raw child and parent with capture nonregression, improved held-out executed control above `.001`, successful revoked fallback and at least 90% final ordinary capture. It has its own `acquisition_transfer_pass` verdict. Even a pass leaves the full two-round beneficial-loop goal incomplete; `beneficial_bidirectional_loop` cannot pass with one round. Producer and full verifier each have caps of 4,000,000 scalar controls, 2,000,000 neural rows, 200 Adam calls and 200 backward calls. No extra optimizer updates or pool growth follows failure.

Before launch, pure core checks and a saved-array selection preflight exercise context deduplication, fixed-order selection, failed/hypothetical/actor-owned evidence rejection, state-change quality and zero-contrast stopping. The preflight uses old qualified trial evidence only as a retrospective software test; it supplies no new-run teaching data or fresh-performance claim. A separate paid auditor reconstructs source selection, original executed-query ancestry, own-state suffix budgets, exact 32/16/16 replication, every update and all independent/controller decisions.

## Persistent teacher evidence and ownership

V2 stopped without a newly qualified controller, so it ran no learning. That does not invalidate old measured successful traces. Current permission to influence planning and the validity of past outcome evidence are different objects. `experience.py` now implements `freeze_qualified_teachers(source_run, destination, round_index)` and `import_qualified_teachers(archive, input_directory, task_hash)` for this unchanged Ending task. These utilities execute no controller or optimizer and import no trust records.

Freeze pins the original source/configuration, complete producer ledger, audit, study, launch and every confirmed-weight trial/confirmation candidate's episode and control packets. Original bytes stay unchanged. Selection keeps the declared first 1/2/4 useful distinct context pairs. Import references use `inherited:<original producer-ledger SHA>:<original query ID>`, retain original controller/task/family/context and query IDs, and append explicit inherited-evidence events. They are never emitted as newly executed `query_requested`/`query_complete` events, or as current actor-owned returns. Returned records retain both original IDs and mapped budget references.

The [source-frozen binding preflight](../runs/2026-10-27_archived_teaching_binding_001/receipt.json) passed 3,967 checks on four context pairs and 200 inherited controls, including independent packet/episode comparisons, wrong-task rejection and duplicate-import rejection. Model and neural calls were guarded off. This took 5.548 s of saved-data processing with zero new control, forward, backward or optimizer calls. It is a provenance/ownership test, not an archive-to-policy learning result.

## Archived teacher-to-learner transfer v3

This stage is now wired into the same `develop` command and shared matched-child learner. It pins the closed v1 parent/Adam/RNG and its round-0 qualified trial/confirmation archive. The original producer-ledger SHA, source manifest and configuration are frozen before launch. It imports the first four improving contexts selected by the same fixed-order rule, keeping each arm's own state and measured component suffix at times 0, 4 and 8. No current planner, trust qualification, or guidance runs in this learning-only intervention. Importing old teacher evidence does not grant present planning authority.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_archive_interface_v3.json \
  --source-run runs/2026-10-27_coupled_neural_commitment_001 \
  --out-dir runs/coupled_archive_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_archive_transfer_v3.json \
  --source-run runs/2026-10-27_coupled_neural_commitment_001 \
  --out-dir runs/coupled_archive_transfer_001
```

Run the one-update-per-arm interface before the 24-update-per-arm experiment. The interface checks changed teaching data, identical ordinary actor-owned samples from the common parent, changed updates, no planning authority, and paid replay of every imported and local control plus every optimizer update. It has four-root actor panels and caps of 20,000 scalar controls, 100,000 neural rows, 16 Adam calls and 16 backward calls separately for producer and verifier. Passing it is a software result, not a learning benefit.

The full experiment keeps the outcome-envelope objective, Adam settings and four clipped policy-gradient epochs unchanged. Each update still executes 64 actor-owned trajectories: 32 ordinary and 32 acquired, with balanced exact-context repeats. Each arm starts from identical full parent state and common sampling seeds. Proposed deterministic actors qualify every eight updates on 16 fresh paired ordinary roots. Fresh child-selection panels choose a retained next parent; separate 64-root panels compare both retained children and the parent. The frozen v3 root blocks and sampling seed bases have offset 6,000,000. Both interface and main inadvertently used that offset, overlapping four roots per shared panel; the executed main therefore does not have a fully untouched 64-start panel. This limitation is retained. Future interface and scientific versions require disjoint blocks. Evaluation outcomes never select teacher data or policy checkpoints.

`archive_transfer_pass` requires a changed teaching intervention and a guided-data retained child beating both the matched raw-data child and parent by more than `.001` in complete cost, with capture nonregression; the guided child must be the qualification-selected next parent and finish at least 90% of a separate final 64-start panel. The original frozen v3 predicate omitted that selected-guided requirement; the actual negative result is unaffected. The full beneficial-loop flag remains false in this stage regardless of that result. Caps are 200,000 scalar controls, 1,000,000 neural rows, 200 Adam calls and 200 backward calls, separately for producer and verifier. No extension follows failure. Expected local control work is 81,584; verification also pays for 200 historical imported controls. Those imported rows remain past experiences rather than newly executed production controls. Their original acquisition/training/replay costs remain additional and are counted once in the cumulative development ledger.

The verifier resolves imported IDs only in the teacher-budget lookup. It checks exact origin ledger records and unchanged copied arrays, reconstructs the selected contexts and balanced batches, and replays selected historical actions once through its counted model. New actor Monte Carlo returns must come from new local uninterrupted queries of their generating actor. A positive conditional archive-transfer result would still require fresh policy-to-search qualification and the complete two-round loop; a negative result would isolate learner refinement without the empty-data intervention that stopped v2.

## Deterministic complete-feedback credit v4: bounded diagnostic

V3's saved-array diagnosis found that the guided late states leave very little slack before the gate: their time-8 positions are `.692294–.697544` against the derived feasibility boundary `.7`. The noisy generating actor finishes 2/16 there, versus 14/16 on independent-data acquired states, despite successful deterministic-parent teacher suffixes. This motivates testing the controller actually deployed, rather than assuming stochastic return refinement improves its mean. The diagnosis is observational; state-distribution and budget changes remain confounded.

V4 keeps the task, parent network, independent proposals and measured teacher component envelopes fixed. It computes a **smooth branch-local derivative of complete deterministic feedback cost**, using already executed actor states/actions/components and analytic local model Jacobians. No teacher actions are learner labels. For component totals `C`, executed-teacher budget `b`, fixed scales `s`, the actor objective remains

`F = sum(C)/25 + beta * sum(max((C-b)/s, 0)^2)`.

The component cotangent is `w_j = 1/25 + 2*beta*max(C_j-b_j,0)/s_j^2`. With local dynamics Jacobians `A=f_y`, `B=f_u`, local weighted cost derivatives `l_y,l_u` and bounded actor sensitivity `h=partial pi/partial y`, the complete feedback adjoint runs backward:

`r_t = l_u + B*lambda_(t+1)`;
`lambda_t = l_y + A*lambda_(t+1) + h*r_t`;
`grad_theta F = sum_t r_t * partial pi(o_t)/partial theta`.

This includes how changing early actions changes later actor actions. Differentiating actions only at fixed saved states omits that feedback. `credit.py` recomputes the live per-time neural batch shapes and performs two metered VJPs: input sensitivities and all named parameter sensitivities. `sensitivity.py` owns the finite experiment. Unused deterministic standard-deviation parameters have zero direction; no output-head-only repair or critic is used.

The ending/detour indicators have zero local derivative away from the lane threshold. The method therefore does not provide discovery credit across that discontinuity and is not a replacement for independent search. Exact clip/ending boundaries fail smooth-branch qualification. Float32 rounding is not differentiable in the classical sense: the returned smooth-arithmetic derivative must agree with **newly executed complete perturbed feedback controllers**, at declared epsilons `.001` and `.0005`, within absolute `.0001` plus relative `.02`. The predicted quotient uses realized float32 parameter changes. Candidate action replay is insufficient for this check.

Before any new repeated training, run this one diagnostic:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_deterministic_credit_v4.json \
  --source-run runs/2026-10-27_coupled_neural_commitment_001 \
  --out-dir runs/coupled_deterministic_credit_001
```

The same four frozen historical context pairs supply one new deterministic actor execution per own context at times 0/4/8. Seven declared gradients check the full raw/guided objectives, coupling off, and teacher-budget contrasts on identical ordinary actor tapes. All deterministic base contexts must finish, and all full-feedback directional finite differences must pass with stable active branches; teacher budgets must change the same-data gradient, and removing coupling must eliminate teacher dependence exactly. Each arm then tests **all four** fixed unit-gradient line lengths `{.01,.003,.001,.0003}` from the same parent, selecting the first training-objective improvement with capture nonregression. This is one full-network step with a paid line search, not four learning iterations. Fresh paired 16-start qualifications require `.001` cost improvement and capture nonregression before retaining a child; independent 64-start panels report the retained raw/guided children and parent. A separate final 64-start panel evaluates the qualification-selected next parent. Roots use offset **8,000,000**, disjoint from both v3 runs; evaluations never choose a line length, gradient or checkpoint.

Per producer and verifier, caps are **20,000 scalar controls, 100,000 neural rows, 32 backward calls and zero Adam calls**. Expected production work is 12,136 controls, 13,292 neural rows, fourteen VJP calls and eight trial parameter assignments. Verification additionally charges 200 historical controls, repeats all derivative work, and reconstructs/checks the eight trial assignments without installing additional gradient steps. Every finite-difference policy has its own checkpoint and uninterrupted measured return. No current planner, guidance qualification or deployment authority runs. The original acquisition/training costs remain additional.

`credit_diagnostic_pass` concerns ownership, teacher dependence and derivative validity only. The separately reported `guided_next_policy_advantage` requires the guided step to be qualification-selected and beat both the raw step and parent by `.001` on complete ordinary outcomes without capture regression. Neither flag can establish the full bidirectional loop; that still requires fresh policy-to-search qualification, changed new experience and matched next-policy learning over two rounds. A failed diagnostic receives no extra line lengths, epsilons or updates.

Analytic policy gradients through known dynamics are established prior art ([Wiedemann et al., 2023](https://arxiv.org/abs/2209.13052)); the stochastic-to-deterministic objective distinction is also studied ([Montenegro et al., 2024](https://arxiv.org/html/2405.02235v2)). Neither citation validates this implementation or establishes novelty for this combined mechanism. The new experiment tests their relevant mechanism inside the protected-proposal, executed-outcome-envelope coupling.

## Necessary coverage and representation assumptions

Independent search must actually find successful complete behavior in the contexts used for budgets. Actor-owned exploration must produce informative outcome variation: if every sample has the same ending failure, a constant failure penalty supplies no likelihood-ratio direction after the baseline. The policy class must be able to satisfy compatible family-specific envelopes across contexts. Matching four costs does not imply matching robust feedback or all task-relevant context. Exact-context budgets do not establish generalization; fresh ordinary and perturbed acquired-state executions must test it. These assumptions are neither supplied by a critic nor certified by low constraint loss.

## Required cheap falsification tests

1. Restoring physical and private planner state reproduces the candidate bank exactly under a deliberately changed/bad actor.
2. With that bank fixed, changing an eligible actor changes the coupled teacher's optimization; disabling coupling removes the effect.
3. Holding actor-owned rollout data, initial parameters and optimizer work fixed, changing verified teacher evidence changes the actor update through the declared constraint. Disabling that constraint removes the effect. Teacher action labels are absent.
4. Complete outcomes belong to the behavior controller that generated them, true terminal cost is charged once, and pre-action states align with executed actions.
5. Unsupported/degraded/version-changed guidance returns to zero and reproduces the independent fallback.
6. Matched guided-data and independent-data child policies are executed alone on fresh roots. A different gradient or selected plan is not a successful next policy.

Before launching these tests, freeze the exact behavior object, objective, finite work cap, random roots and pass/fail thresholds in a new run. Retain a negative result. A revised formulation receives a new version and budget; it does not change the old verdict.

## Evidence needed for the research claim

The result must survive independent replay of raw transitions and accounting, paired held-out complete controller executions, coupling-off interventions and matched next-policy comparisons. Report acquisition, representation/coverage, optimization, trust and executed-control failures separately. Count failed queries, verification, qualification, evaluations, inference and optimizer work with wall time. Do not infer efficiency from fewer training episodes.

Start on the existing cheap Ending task as a mechanism test. Return to Acrobot only after the coupled interfaces work; preserve its original task contract and the unresolved discovery result. Walker is a later port. Neither a cheap scalar result nor a value-guided MPPI pilot establishes a standalone physical policy or the full GPS thesis.

## Ending-constrained actor step v5

V4's retained negative result separates two requirements: local cost credit is numerically valid, while one feasible teacher context remains infeasible for the actor. The implemented v5 hypothesis is explicit **ending feasibility**, measured on the actor's complete feedback execution, rather than another stochastic-policy refinement or another reward for the planner. For Ending, the known task constraint is `g(theta)=y_T(theta)-ending_y <= 0`; its continuous residual is observed even when the discrete terminal penalty has zero derivative. Successful executed MPPI evidence certifies that the exact teacher context is feasible, without supplying an action label.

A minimal proposed actor step starts from the complete cost/envelope gradient, computes the terminal-state feedback sensitivity of the worst measured constraint, and projects a small cost step onto its linearized feasible half-space. For cost step `d0`, terminal gradient `a`, measured residual `g` and declared guard `m`, the proposal is `d=d0-max(g+m+a dot d0,0)*a/(a dot a)`. This is a local proposal, not a feasibility certificate. The changed actor must then execute complete continuations from every tested teaching context, with the original ending condition and complete cost used for retention. Multiple simultaneously active constraints, vanishing saturated-action sensitivity, model error or nonlinear boundary crossing may invalidate this one-constraint projection and must be reported.

A new version must freeze its guard, finite step ladder, derivative tests, roots and total budget before launch. It should test terminal-state sensitivities against full-feedback finite differences as well as the existing cost gradient. It must preserve v4 unchanged and require actual capture restoration rather than remove the failed prerequisite. Full-network outcome-constrained learning remains separate from teacher action imitation, and a one-step result still requires fresh search qualification and a subsequent matched-policy test.

V5 is implemented through the same `develop` entry point, shared actor/context collector and complete-feedback auditor:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_projection_v5.json \
  --source-run runs/2026-10-27_coupled_neural_commitment_001 \
  --out-dir runs/coupled_ending_projection_001
```

Before launch, freeze these additional decisions: first maximum terminal position across the twelve measured own contexts chooses one ending constraint per arm (ordered clock then source row); guard `m=.0002`; cost steps `{.01,.003,.001,.0003}`; endpoint directional FD epsilons `.001,.0005`, tolerance absolute `.00001` plus relative `.02`; cost FD criteria unchanged. Finite differences perturb only the cost direction and compare both complete cost/envelope and selected terminal-position quotients against realized float32 parameter displacements. Endpoint smoothness requires stable clipping, while cost smoothness still requires stable lane indicators. A positive directional test is not verification of every possible direction or a feasibility certificate.

The terminal adjoint starts at one for the selected actual trajectory, zero for every other trajectory, with no stage cost and **no trajectory averaging**. Two additional full-group neural gradient calls preserve the original per-time batch shapes and add 504 forward rows/four VJP calls. The goal inequality is already part of the unchanged task; this adds a constrained optimization proposal for the actor rather than changing the task rewards or granting the planner goal information withheld from a control. Teacher evidence still certifies complete successful continuations from each exact context and provides component envelopes; it supplies no action labels.

Unlike v4, v5 explicitly attempts to restore failed actor contexts: each eligible proposed step must achieve **12/12 actual captures** and lower the complete measured training objective. It does not waive the failure or accept capture nonregression at 11/12. Select the first eligible line length, then apply all derivative/evidence checks and the same fresh 16-start cost/capture qualification. Use distinct offset **10,000,000** roots for qualification, 64-start child/parent comparisons and a separate final panel. Original thresholds, costs, independent search and source traces remain unchanged.

Caps per producer/verifier remain 20,000 scalar controls, 100,000 neural rows, 32 backward calls and zero Adam calls. Expected producer work is 61 queries, 12,136 controls, 13,796 neural rows, eighteen VJPs and eight explicit trial assignments; verifier adds 200 inherited transition replays, repeats derivative work and verifies assignment bytes without installing gradient steps. No further lines, roots or updates follow a negative result.

Report three separate flags: `credit_diagnostic_pass` for derivative/ownership/teacher dependence; `ending_restoration_pass` only if a previously failing guided context set reaches 12/12 and that changed actor passes ordinary qualification; `ending_projection_pass` additionally requires the guided child to be the qualification-selected next parent and beat both the matched independent-data child and parent by more than `.001` complete cost without capture regression on their paired 64-start evaluations. The run's overall pass is this last conjunction, together with the independent replay audit. `beneficial_bidirectional_loop` remains false: this one-step archive intervention cannot establish fresh policy-to-search influence, two feedback rounds or efficiency.

## Ordinary-performance preservation v6

V5's independent saved-array diagnosis supports a specific next hypothesis: the endpoint/ordinary gradients have cosine `-.94785`, and the single-constraint correction predicts ordinary cost `+.010809` versus actual `+.010798`. The two-normal Gram condition is about `45.7`; a linear joint step is feasible, but needs larger parameter displacement. That algebra is not a tested remedy.

V6 changes only the actor-step optimization. Let `x=alpha*d0`, `a=grad(y_T)` for the worst measured teaching context, and `c=grad(J_ordinary/25)` from the already executed four ordinary contexts, without the teacher envelope. Those ordinary traces and gradients are identical across arms. Solve

`min_delta 0.5*||delta-x||^2`

subject to `a dot delta <= -g-.0002` and `c dot delta <= min(c dot x,0)`.

The first constraint repairs the measured ending, and the second preserves the cost step's predicted ordinary improvement. It forbids predicted ordinary worsening when the original step is uphill. The two-halfspace active-set solver enumerates zero/one/two active constraints, checks primal and dual feasibility/stationarity, and returns a certificate. Zero, redundant and opposing normals are handled explicitly; condition above `1e10` cannot be repaired by a hidden ridge. Actual source ordinary cost must also not regress, alongside 12/12 capture and decreased complete envelope objective, before a line is eligible. Fresh sixteen-start qualification and paired 64-start evaluation remain mandatory.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_preserved_v6.json \
  --source-run runs/2026-10-27_coupled_neural_commitment_001 \
  --out-dir runs/coupled_ending_preserved_001
```

Before launch, fix source/parent/teacher archive, all four step lengths, guard, derivative tolerances and every v5 pass gate unchanged. Reserve disjoint root/seed offset **12,000,000**. Projection numerical feasibility tolerance is `1e-10`; actual ordinary-training tolerance is zero. The method needs no new actor data or gradient calls compared with v5: expected 61 producer queries, 12,136 controls, 13,796 neural rows, eighteen VJPs, eight full-network trial assignments, zero Adam. Each producer/verifier keeps the same cap of 20,000 controls, 100,000 neural rows and 32 backwards; verification additionally charges 200 inherited controls and checks trial bytes. The independent auditor reconstructs the joint minimum-distance/KKT solution and then checks exact realized float32 parameters. No extension follows failure.

Each side also makes eight two-halfspace projection calls. Durable intent/completion/error records count actual candidate checks, condition estimates, and linear-solve requests/completions separately; failed partial work remains explicitly unknown. These are arithmetic optimization calls, additional to neural backwards and parameter assignments. Core validation exercises the solver without physics or neural execution. An initial validation counter-wrapper closure bug was corrected in a new invocation; the failed receipt remains preserved rather than counted as a scientific outcome.

This is still one matched constrained actor step. Overall success requires qualified guided ending restoration and the qualification-selected guided child beating both raw-data child and parent by more than `.001` complete cost without capture regression on fresh paired evaluations. The full-loop flag stays false until the changed policy improves freshly executed search and that search improves a subsequent standalone learner over two feedback rounds. Larger linear projections can cross nonlinear boundaries; a valid KKT certificate alone does not satisfy these outcomes.

Gradient projection to protect previous performance is established component prior art ([Lopez-Paz and Ranzato, 2017](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)), as is constrained policy optimization ([Achiam et al., 2017](https://proceedings.mlr.press/v70/achiam17a)). Neither their assumptions nor guarantees transfer automatically to this deterministic known-model actor step. The test is the executed outcome-envelope/independent-MPPI coupling, not a novelty claim about the QP.

## Fresh improved-policy search v7

V6 qualified a useful standalone policy, but did not establish a guided-data advantage. The next required link is fresh policy-to-search competence, rather than more fitting of the same archived data. V7 freezes the **qualification-selected v6 raw child**, with its full inherited optimizer/RNG metadata, and its hash-bound inherited independent memory/support. Memory is search state, not a teacher action dataset for the actor. The new policy starts with zero authority; no v1 trust record is imported.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_fresh_search_v7.json \
  --source-run runs/2026-10-27_coupled_ending_preserved_001 \
  --out-dir runs/coupled_fresh_search_001
```

This is one search/acquisition stage of a prospective staged feedback loop. It reuses suffix eligibility, the agreement ladder `{.1,1,10}`, eight paired executed trial roots and eight disjoint confirmation roots, with the unchanged `.001` complete-cost/capture gate. A promoted setting is tested on sixteen separate control roots and deliberately invalidated to check exact fallback. Trial/confirmation traces can supply the first four distinct successful, genuinely changed/improved teaching pairs by the existing fixed selector. After successful acquisition, every observed raw trial/confirmation trajectory enters independent memory, including traces not selected for teaching; guided choices do not overwrite it. The observed early failure exits before memory ingestion and leaves memory unchanged. No actor update runs in this stage.

All ordinary/qualification/evaluation roots use new offset **14,000,000**. Per producer/verifier caps are **4,000,000 scalar controls, 2,000,000 neural rows, zero Adam calls and zero backwards**. Eight/four-step search settings, task rewards, full trajectory-memory strength and all promotion criteria remain unchanged. Exact incoming canonical learner fields are checked and the additional deterministic-step metadata is preserved in the full ancestor checkpoint. Neither an overall-negative v6 result nor a different new evaluation score changes which v6 parent was selected.

`search_feedback_pass` requires fresh promotion, four selected useful teaching pairs, changed actual controls/states, held-out control improvement without capture regression, and exact revoked fallback. Lack of promotion or four-pair coverage is a retained failed link; there is no extra trial panel or relaxed threshold. A passing stage would feed its newly executed, hash-bound teaching outcomes into matched constrained child updates, then require fresh qualification again after the policy changes, for at most two feedback rounds. Search-stage success alone never sets `beneficial_bidirectional_loop` or a next-policy/efficiency claim. Staging pays each producer/auditor once and avoids duplicating the learning or verification implementation.

## Pointwise experience without deployment authority v8

V7 failed the declared confirmation effect size and remains failed. This exposes a separate acquisition choice: the implementation discards all trial teaching pairs when the controller setting is not promoted. A complete successful executed trajectory can still supply a valid exact-context outcome budget even if the controller's average advantage does not qualify future deployment. V8 tests that distinction; it does not relax `.001` search promotion or claim that v7 qualified.

Freeze the v7 actor, complete executed trial/confirmation pool, original query identities and unchanged task. Select the first originally passing trial weight in the original ladder, then the existing first four distinct pointwise useful pairs (both complete/capturing, guided cost gain greater than `1e-6`, state change greater than `1e-8`). Retain all sixteen candidate pairs, including unselected pairs. This is a **conditional curriculum** chosen on observed teacher outcomes. It is not an unbiased estimate of search improvement or evidence of broader coverage. The new packet explicitly labels the origin as deployment-unqualified, binds the failed confirmation and imports zero authority.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_pointwise_transfer_v8.json \
  --source-run runs/2026-10-27_coupled_fresh_search_001 \
  --out-dir runs/coupled_pointwise_transfer_001
```

From the same full v7 parent, each arm executes its own contexts at times 0/4/8 and uses the unchanged v6 full-feedback gradients, ending/ordinary projection, FD tolerances and four line lengths. Teacher controls remain absent from the actor loss. Only new uninterrupted actor executions measure actor performance; past controller returns are component envelopes, never actor-return labels. Incoming deterministic-step provenance is preserved alongside unchanged optimizer/RNG state.

Offset **16,000,000** reserves disjoint ordinary qualification/evaluation panels. Per producer/verifier caps remain **20,000 controls, 100,000 neural rows, 32 backwards and zero Adam**; the exact local schedule remains 61 queries/12,136 controls/13,796 neural rows/eighteen VJPs/eight trial assignments/eight projection calls, with 200 imported controls additionally replayed by the verifier. All previous discovery, trials and learning costs remain additional. No new search or deployment authority runs here.

`pointwise_transfer_pass` requires all ownership/credit checks and a qualification-selected guided child beating both matched raw child and parent by more than `.001` on fresh paired 64-start ordinary outcomes, with capture nonregression. Unlike v5/v6's diagnostic restoration flag, this criterion does not require the starting actor to have a failed teaching ending: learning from an already successful context is allowed. Existing ending-restoration flags retain their original meaning. A failed criterion adds no steps, roots or budget. Even a pass would establish only conditional next-policy learning; fresh executed search qualification and the full repeated feedback loop would remain outstanding.

## Measured nonlinear feasibility restoration v9

V8's saved-array audit found failure of the selected endpoint's affine prediction, rather than an omitted constraint. Shortening the unprojected cost step does not shrink the necessary feasibility correction to zero, so the four fixed cost-step lengths do not provide a restoration line search. No v8 guard, result or budget changes.

The next proposed update keeps the cost step and measured outcome envelopes, then permits a finite restoration phase. After each changed actor executes its complete continuations, recompute the worst actual ending residual and its complete-feedback sensitivity **at that generating actor**. Propose a minimum correction satisfying that new endpoint halfspace and a non-worsening ordinary-cost halfspace. Execute the corrected actor, check all twelve actual endings and the original ordinary-cost/source-objective gates, and repeat only up to a frozen restoration cap. The existing two-halfspace solver can propose the correction with a zero cost direction; it remains a proposal, never a feasibility certificate.

This changes the optimization method rather than the task, neural architecture, outcome thresholds or deployment trust. Every new sensitivity/parameter assignment binds its own executed actor and query IDs; past returns cannot be relabelled as corrected-policy measurements. Raw/guided arms use the same procedure and initial parent. Fresh standalone qualification, the matched next-policy criterion and subsequent fresh search qualification remain required. A cheap bounded nonlinear counterexample passes: one affine projection misses, while remeasuring and relinearizing restores the ending in two corrections with the same guard and ordinary-cost constraint. This is pure arithmetic, not a neural restoration result.

The implemented v9 freezes the same v7 parent and sixteen pointwise candidate pairs as v8, with the same first four selected teaching pairs. All initial cost/ending gradients, FD settings and four cost lines are unchanged. Only a line with fewer than twelve actual captures enters restoration. Each correction recomputes the first maximum terminal position on the last complete actor execution, then computes its endpoint gradient and the beta-zero ordinary-cost gradient at that exact generating actor. The existing two-halfspace projection receives zero cost direction and unit step, so it minimizes correction norm subject to the new measured endpoint bound and predicted non-worsening ordinary cost. Guard remains `.0002`. Execute all twelve contexts again, and stop on twelve captures or **three corrections**, whichever comes first. No correction is spent rescuing an ordinary-cost-only regression. Final line eligibility still requires all twelve actual captures, ordinary-source cost nonregression against the original parent, and original complete source-objective improvement. Select the first eligible initial line, using its final corrected actor.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_nonlinear_restoration_v9.json \
  --source-run runs/2026-10-27_coupled_fresh_search_001 \
  --out-dir runs/coupled_nonlinear_restoration_001
```

Use disjoint root/seed offset **18,000,000**. Caps per producer/verifier are **20,000 scalar controls, 100,000 neural rows, 128 backwards and zero Adam**. At most twenty-four corrections across eight initial lines add 3 queries, 252 controls, 604 neural rows, four VJPs, one projection and one parameter assignment each. Thus the exact dynamic maximum is **133 local queries / 18,184 controls / 28,292 neural rows / 114 VJPs / 32 projections / 32 trial-or-restoration assignments**. Verification additionally pays for the same 200 inherited controls and verifies assignments without installing new gradient steps. Initial FD assignments remain eight additional parameter constructions, separately recorded. Every correction has durable before/after actor hashes, query ownership, endpoint selection, gradient packets, projection counts, parameter bytes and complete measured outcomes. Independent replay reconstructs the dynamic stop rule, all current-actor VJPs and final line selection. Restoration gradients receive local branch/replay checks; no new finite-difference qualification at those actors is claimed.

`pointwise_transfer_pass` retains v8's criterion: all credit/ownership checks, a fresh-qualified guided next parent, and guided superiority over both matched raw child and parent by more than `.001` on the paired 64-start panel without capture regression. Source repair alone is insufficient; past v7 search qualification remains failed and planning influence remains zero in this stage. The full-loop flag stays false until fresh search and subsequent learning succeed over the required rounds. A negative v9 receives no extra corrections, guard changes or roots.

## Fresh restored-policy search v10

V9 passed its conditional transfer criterion. V10 tests the changed policy's effect on fresh executed search, with zero inherited authority. It imports the full qualification-selected v9 guided checkpoint, not a checkpoint selected on the 64-start evaluation. Its four selected guided teacher episodes provide handoff support; their old controller authority remains zero.

Strengthen independent memory before comparing controllers: append **all sixteen** already executed independent v7 trial/confirmation episodes to the inherited independent base memory, in original trial-then-confirmation order with stride five. Include unsuccessful or unselected raw episodes. The resulting 200 fragments comprise 120 base fragments plus 80 additions. Guided episodes never enter independent memory. Hash-bind the original episode/query packets, base memory, insertion order and exact derived bytes; retain the original failed v7 confirmation. This materialization uses saved evidence only. Its wall time is recorded separately and all original discovery costs remain additional.

The same derived memory, independent planner settings and paired roots are used for independent, current-policy-guided and previous-policy-guided controllers. The previous actor is the exact v9 incoming full checkpoint. It receives fresh suffix eligibility on the same support states. If eligible, it is tested at the current actor's newly qualified setting; if ineligible, its counterfactual controller executes the zero-influence fallback. This comparison grants the old actor no deployment promotion. It asks whether the new policy improves the resulting controller under the same candidate mechanism, rather than comparing two stages with different roots or memories.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_restored_policy_search_v10.json \
  --source-run runs/2026-10-27_coupled_nonlinear_restoration_001 \
  --out-dir runs/coupled_restored_policy_search_001
```

Root/seed offset **20,000,000** is disjoint from earlier panels. Each producer/verifier has caps of **6,000,000 scalar controls, 2,000,000 neural rows, zero backwards and zero Adam calls**. The larger control cap is declared before launch to cover the previous-actor controller condition. The `.9` suffix-eligibility gate, agreement ladder `{.1,1,10}`, eight paired trial roots, eight disjoint confirmation roots, sixteen held-out control roots, `.001` complete-cost gain and capture nonregression gates remain unchanged. Stop at the original failed gate; do not add roots or loosen thresholds.

`search_feedback_pass` requires current-policy promotion, four genuinely changed/improved executed teaching pairs, held-out current-guidance improvement over independent search, exact deliberate-revocation fallback, and current-guidance improvement over the previous-actor controller by more than `.001` without capture regression. A failure at promotion stops before the held-out old/new comparison. No learner update runs here. Even a pass is one newly qualified search/acquisition stage: matched next-policy learning and a second feedback round remain required. The v9 extra optimization work precludes an efficiency or data-only causal claim from this stage.

## Repeated outcome-constrained learning v11

V10 rejects the inference that one successful outcome-constrained actor step makes its continuations competitive with strong independent replanning. First isolate optimization depth before changing coverage, task or selector. Start both learning arms from the same full **incoming v9** checkpoint, keep the exact same v9 teacher packet and twelve contexts per arm, and apply a finite sequence of measured constrained cost steps. The first cost iteration should reproduce v9's candidate actor hashes; the separate one-step v9 selected policy is the later search counterfactual. Starting both arms from that selected policy instead would test additional learning conditional on its prior guided update and must be labeled differently. The target remains each executed teacher's component budget; every actor return and sensitivity comes from a newly completed execution of the actor at that step. No teacher control is a policy label.

Each arm advances its own training actor. Recompute its full outcome-envelope gradient, worst measured ending sensitivity and ordinary-start cost gradient at that actor. Retain the four fixed initial cost lines and at most three remeasured ending corrections per line. Accept the first line satisfying all twelve actual endings, ordinary-source cost nonregression and complete-source-objective improvement. Stop that arm when every line fails or the predeclared iteration limit is reached. Reusing the initial raw arm's ordinary gradient for a later guided actor would be invalid. Same-actor teacher-dependence/off-switch checks belong at the common initial actor; zero later contrast can simply mean that the envelopes are inactive.

The implemented protocol permits **32 cost iterations per arm**. A zero/nonfinite cost direction, invalid current branch or exhausted four-line grid stops that arm; reaching 32 is finite-grid stopping, not evidence of convergence. Initial same-actor contrast and the two finite-difference scales run once. Later current-actor derivatives receive owner/branch/formula replay checks and actual complete-step qualification. Initial finite-difference checks do not certify later actors or require a finite perturbation crossing the ending indicator to have a smooth derivative. A solver exception is retained as an incomplete run with durable work, not relabeled as a successful stopped arm.

Keep training frontiers separate from qualified incumbents. Every four iterations, at the final iteration, or when both arms stop, compare each frontier and its own incumbent on the same fresh sixteen ordinary starts. Roots are `offset + 200000 + 1000*iteration + [0,16)`. Promotion requires at least 90% capture, capture nonregression and complete-cost gain greater than `.001`. Each `(arm, frontier hash)` receives one promotion attempt; repeated scheduled panels for an unchanged frontier are diagnostic only and cannot rescue an earlier rejection. Qualification never changes the training contexts, loss, grid or frontier.

Final selection uses sixteen fresh paired starts at `offset + 710000` for the incoming parent, fixed one-step v9 comparator and both qualified incumbents. It uses the same capture/cost gates. Evaluate all four on sixty-four separate paired starts at `offset + 740000`; these evaluations select no checkpoint and update no actor. Evaluate the qualification-selected controller on a final separate sixty-four starts at `offset + 950000`. Full checkpoints preserve optimizer/RNG state and prior credit history; accepted cost steps, trial assignments and ending restorations are recorded separately. A parent/one-step fallback retains its exact original full learner state.

Before the longer test, run a **two-iteration interface** that must reproduce all eight v9 first-iteration candidate hashes and exercise own-actor credit after the arms diverge:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_learning_depth_interface_v11.json \
  --source-run runs/2026-10-27_coupled_fresh_search_001 \
  --out-dir runs/coupled_learning_depth_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_learning_depth_v11.json \
  --source-run runs/2026-10-27_coupled_fresh_search_001 \
  --out-dir runs/coupled_learning_depth_001
```

Interface offset is **24,000,000**, with caps **50,000 controls / 200,000 neural rows / 256 backwards / zero Adam calls** per producer and verifier. Its `depth_interface_pass` is a software/ownership contract; scientific depth and guided-data advantage flags stay false. Start the full test only after paid interface replay passes and at least one arm reaches two accepted steps without an early stop. An already exhausted two-step method is not restarted on new roots merely to seek a pass.

Full-test offset is **22,000,000**, with caps **500,000 controls / 1,000,000 neural rows / 4,096 backwards / zero Adam calls** per phase. At most 1,024 line/restoration projections and assignments can occur, plus eight initial finite-difference parameter constructions. Both stages freeze source/config and the exact v9 full comparator and metadata before querying the task. Import the sixteen original teacher candidates once per run; replay the selected 200 historical controls once in verification. All past discovery, one-step learning and replay costs remain additional, counted once by distinct-run accounting. Actual repair expenditure may differ between arms; equal procedure/caps do not establish efficiency.

`depth_training_pass` requires all credit/ownership checks, a qualification-selected deeper child, paired sixty-four-start cost gain greater than `.001` over the one-step v9 actor without capture regression, and at least 90% capture on both paired and final separate panels. `guided_next_policy_advantage` additionally requires the guided child to beat the matched independent-data child by `.001`. A raw-child depth win is valid evidence for additional learning but is not guided-data superiority. All ending/source gates retain the original task and thresholds; no new contexts, lines or roots follow a failure.

Both branches use the same finite update procedure and retain their actual restoration costs. A later fresh-search stage must compare the selected new policy against the one-step v9 policy on common roots and the same all-raw memory. The existing control/capture thresholds stay unchanged. If repeated learning still gives poor continuations, the next distinct test should address exact search-endpoint coverage; changing data freshness and learning depth together would not identify which limitation mattered. V11 alone supplies no fresh-control or completed-loop result.

## Fresh deep-policy search v12

Use the full, audited, non-interface v11 qualification-selected raw/guided child. Recompute its four-role selection from the fresh sixteen-start qualification only; neither the sixty-four-start evaluation nor `depth_training_pass` selects a policy for this stage. Reject parent/one-step fallbacks and unchanged actor hashes. The previous-policy control is the exact v9 one-step checkpoint copied into v11, **not** the older incoming v7 parent. Every policy starts with zero new search authority.

The independent memory remains exactly v10's 200 fragments: the same 120 inherited fragments plus all sixteen raw v7 trials. No guided trace or v11 actor rollout enters this memory. The selected teaching arm supplies four originally executed support episodes; current and previous policies receive fresh suffix tests on those same support contexts. Both controllers, and the independent reference, use common reset roots, identical private planner seed rules and memory. A previous actor that fails suffix eligibility executes zero-influence fallback; an eligible previous actor receives only an experimental trial at the new policy's setting. Its past authority never transfers.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_depth_policy_search_v12.json \
  --source-run runs/2026-10-27_coupled_learning_depth_001 \
  --out-dir runs/coupled_depth_policy_search_001
```

Offset **26,000,000**, caps **6,000,000 scalar controls / 2,000,000 neural rows / zero backwards / zero Adam** per producer and verifier. Search, selector, execution rule, support radius, `.9` eligibility, agreement ladder, eight paired trials/eight fresh confirmations/sixteen held-out controller roots, `.001` cost effect and capture nonregression remain v10's exact settings. No learning runs. Stop at failed qualification; do not compensate with more roots or a weaker reference. After successful acquisition, all newly observed independent trials enter memory without a guided-quality filter.

The stage criterion is unchanged: new promotion, four actually useful teaching pairs, changed physical behavior, fresh complete-control improvement over independent memory, improvement over the one-step-policy controller, and exact deliberate-revocation fallback. Keep the known hypothetical-fallback versus future-replanning limitation visible: v12 does not repair its comparator or inherit a dominance guarantee. Only actually executed comparisons can pass. A positive search stage must still teach matched next actors and repeat the feedback loop; it is not a bidirectional-learning or efficiency result by itself.

## Matched learning from newly acquired teachers v13

V12's fresh search qualifies; now test the missing teacher-to-next-policy link. Start both children from the exact full v12 deployment parent (the qualification-selected v11 raw actor). Import v12's complete **executed** independent and guided teaching episodes, using the existing confirmed-promotion binding and the same first-distinct-useful-context rule. Keep all sixteen paired discovery candidates and failures in the binding; train on the four selected pairs. These are new measured component budgets and acquired physical states, not action targets. No search authority transfers to a changed actor.

Use v11's unchanged 32-iteration procedure: twelve contexts per arm at clocks 0/4/8, current-actor complete outcomes and feedback derivatives, the same four cost lines, at most three measured ending repairs per line, source-ending/ordinary-cost acceptance, and fresh sixteen-start incumbent qualification every four iterations. Each candidate hash receives one promotion attempt. Preserve the full inherited optimizer/RNG and credit history. Stop an exhausted arm; do not add lines, contexts or updates. The two arms have equal procedures and caps; charge their actual differing repair work. The starting parent already contains the earlier raw-data training, a common condition for both children rather than a fresh-start data-only result.

There is no one-step v9 comparator in this learning test. Final selection uses only fresh sixteen-start qualification among the current parent and the two qualified children, at `offset + 710000`. A separate sixty-four paired-root panel at `offset + 740000` compares all three, and the selected actor receives another sixty-four starts at `offset + 950000`. Neither sixty-four-start panel selects or trains a model. **Feedback learning succeeds only if the selected guided-data child beats both the current parent and matched independent-data child by more than `.001` complete cost without capture regression, with at least 90% capture in paired and final evaluation.** Both children improving the parent is useful standalone learning but does not establish the guided-data advantage.

Recheck initial derivative/ownership/branch arithmetic on this new parent. Teacher-gradient contrast and active-envelope contrast are reported as diagnostics, not required to be positive: successful previous learning may have made a teacher envelope inactive. Their exact zero/positive values are independently checked; this does not remove source acceptance, full-return feedback or new-actor qualification gates. No old finite-difference test certifies later actors; current-owner replay and actual complete executions remain mandatory.

First run a two-sweep software interface with fresh root/seed offset **30,000,000**, caps **50,000 controls / 200,000 neural rows / 256 backwards / zero Adam calls** per producer and verifier. Launch the full test only after paid replay passes and an arm reaches two accepted steps; the interface's scientific benefit flag remains false. The full test uses offset **28,000,000**, caps **500,000 controls / 1,000,000 neural rows / 4,096 backwards / zero Adam calls** per phase and 32 cost iterations per arm. These limits and the learning-success criterion are fixed before querying. A failure is retained; it does not authorize extending this run. Reading inherited evidence adds no simulated experience, while its original discovery/learning costs and the paid 200-control import verification remain charged separately.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_fresh_feedback_interface_v13.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/coupled_fresh_feedback_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_fresh_feedback_learning_v13.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/coupled_fresh_feedback_learning_001
```

This isolates one fresh teaching round. Even a pass needs another freshly qualified search/acquisition and matched learning round before a repeated bidirectional-loop claim; it establishes no physical-task transfer, multi-seed result, or all-cost efficiency advantage.

## Prospective hybrid-credit qualification v14

The v13 interface stops before learning: its finite parameter probes cross the original detour/ending switches, so their full cost quotient cannot validate a local derivative. The saved-evidence diagnosis separates these jumps exactly and retains that negative result. V14 changes **credit qualification**, not the teacher packet, actor objective, task reward or actual-step acceptance. It starts both arms from the same v12 full parent and uses the same newly acquired v12 teachers, twelve contexts, finite perturbations, four cost lines, three-repair cap and 32-iteration limit as v13. No initial-condition retries or smaller perturbations are introduced to seek a pass.

Write the existing per-context outcome objective as `F(C,b) = sum(C)/25 + beta * sum(max((C-b)/s,0)^2)`, with measured component totals `C = (effort, greed, detour, ending)` and executed-teacher budget `b`. The network VJP is local credit on the base trajectory's discrete modes. For each already required plus/minus finite probe, retain its **actual** full objective quotient, mode changes and failed full-objective derivative check. Separately form the arithmetic diagnostic `Ctilde± = (Ceffort±, Cgreed±, Cdetour_base, Cending_base)`. Its central objective quotient tests the smooth extension of the base modes against the same VJP dotted into the realized float32 parameter displacement.

Compute the discontinuous contribution directly from measured perturbed detour and ending totals, including both nonlinear teacher-envelope terms. Require `full_quotient = smooth_quotient + jump_quotient` and agreement between the smooth quotient and the base-mode-frozen objective quotient, each within **1e-9** arithmetic tolerance. This identity may not use base objective weights multiplied by a component jump. Preserve the original `.001` and `.0005` parameter probes and derivative tolerance `.0001 + .02*abs(predicted)`. Require unchanged clipping regions across base and both probes, finite arithmetic and a base tape strictly away from the discrete boundary. The continuous squared envelope hinges are differentiable at zero: record their activity changes as diagnostics, while the unchanged finite-difference error gate tests their curvature. The separate terminal-position derivative check remains unchanged.

In v14, `all_hybrid_credit_checks_pass` is the initial derivative gate. The actual full-objective derivative result stays visible as a diagnostic and is never relabeled passed. This supports local constrained credit, not differentiability across a capture switch. Every proposed cost line and restoration still executes the **original** simulator and must capture all twelve source endings, avoid ordinary-source cost regression and improve the original full outcome objective. Fresh incumbent qualification, one promotion attempt per actor hash, parent/raw/guided selection, both disjoint sixty-four-start evaluations and the `.001` guided-child advantage over **both** matched raw child and incoming parent stay exactly v13's procedure. No step is accepted on the frozen-mode diagnostic objective.

Use a two-iteration interface at root/seed offset **34,000,000** before full learning. Caps remain **50,000 controls / 200,000 neural rows / 256 backwards / zero Adam** per producer and verifier. It must pass paid replay and accept two steps in at least one arm; a stopped/exhausted interface does not launch the long run. The full test uses offset **32,000,000**, **500,000 controls / 1,000,000 neural rows / 4,096 backwards / zero Adam** per phase and the unchanged 32 cost iterations per arm. These fresh roots supply independent qualification/evaluation; the teacher contexts and all initial credit probes remain the same. V13's stopped full protocol is not executed or overwritten.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_hybrid_feedback_interface_v14.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/coupled_hybrid_feedback_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_hybrid_feedback_learning_v14.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/coupled_hybrid_feedback_learning_001
```

A pass remains one fresh matched learning round. It neither completes a repeated bidirectional loop nor proves efficiency; the source/search/replay costs and every rejected probe, line and repair remain charged.

## Next formulation under development: teacher-performance feasibility

V14 produces nearly identical qualified children despite different useful teacher outcomes. Its saved interface geometry shows small teacher-gradient contrast dominated by shared ending/ordinary constraints and large projected corrections. The next candidate should change the learning coupling, rather than rerun deeper actor refinement or claim that the credit-check repair completed the algorithm. The bounded admission protocol below tests this update before another learning run; a full learner using it is not yet implemented.

For exact matched contexts, retain executed teacher total returns `B_i` and measure the common old actor's complete uninterrupted returns `J_old_i`. Freeze performance targets `T_i = J_old_i - rho * max(J_old_i - B_i, 0)`, with `rho=1` for a full-attainment test. Each teacher target evaluates that recorded teacher controller's continuation. It is an empirical achievability threshold for the student, not a measured value of a new actor. Student returns and per-context sensitivities must be obtained from that current generating actor's complete feedback executions; averaged gradients cannot substitute for individual context derivatives. No teacher action is a training target.

Use a fixed-work feasibility phase followed by original task-cost improvement. For current deficits `p_i = J_i - T_i` and current own cost gradients `q_i`, a candidate Phase-I subproblem minimizes `t + ||delta||^2/(2*alpha)` subject to `t>=0`, `p_i + q_i*delta <= t`, measured ending linearizations, ordinary-mean nonregression and a displacement cap. Its teacher dual weights are nonnegative and sum at most one; they sum to one when the predicted positive violation is nonzero. A tighter measured teacher target can therefore change the active context and credit direction, rather than merely perturb a common cost gradient by a small fixed penalty. At a currently feasible actor, the ending/ordinary tangent set must contain zero. Do not demand a new deeper ending margin or the entire unconstrained predicted cost decrease at every step.

Every actual candidate still executes the original task, preserves actual endings and ordinary-mean cost, and must reduce measured positive teacher-target violation during Phase I. Exhausting a predeclared finite grid is a retained optimization failure. Once targets hold, spend **all remaining work in the same fixed schedule** on original task-cost improvement while retaining teacher/ending constraints. The independent-data arm must not be artificially stopped at its weaker target. Qualification and held-out actor-only outcomes remain necessary; target satisfaction is not standalone policy improvement. Changed policies still have zero search influence until fresh executed qualification.

Before neural training, test the information limit and the update mechanism. For fully known linear costs on identical states, teacher constraints cannot beat the exact unconstrained optimum of that same scalar objective; a toy that reports such a win is invalid. Scalar returns supply measured achievable performance and coverage, not a new gradient or evidence of unseen success. A tied-target negative control must use the same actual teacher table, parent state, contexts and solver schedule and produce identical first assignments. On the real saved contexts, the raw and guided tables identify different worst remaining deficits; if their feasible-tangent current-actor gradients are parallel or vanish, report that lack of leverage. A subsequent bounded comparison may claim benefit only from improved fresh standalone outcomes over the equally budgeted independent-data child and parent, with capture protection. New source, budget, solver checks and stopping criteria must be frozen before that experiment queries the model.


### Frozen arithmetic falsifiers and v15 admission

Before any new actor training, run the canonical `validate` roster, now including thirteen performance-feasibility fixtures and twenty-two solver requests (twelve normal returns, ten deliberate invalid/backend-error cases). The pure solver cap is 32. Analytic optima, tied-data identity, target-dependent direction, zero-sensitivity/blocked-safety outcomes, Phase-II mean improvement and false-backend-success rejection are predeclared. No real physics or neural execution is permitted. Exact source snapshots and exposed solver work accompany the result.

The next model-query experiment is `configs/coupled_ending_performance_admission_v15.json`, through the same public `develop` command with the closed v14 full learning run as `--source-run`. Select the current full parent only from that run's fresh sixteen-start qualification; pin checkpoint SHA `42155d0ece01171059ebaa32f06d9472040bae33f866f84c51a7a93613058b9a`. Preserve full optimizer/RNG/history. Rebind the same v12 promoted teacher packet; do not choose teachers from the new actor's results or import authority. Use its four identical ordinary starts in both arms, with raw/guided measured complete returns as the only intervention. Acquired-state mixtures are deliberately absent in this single-step mechanism test.

One complete current-actor batch supplies four original cost gradients and four endpoint gradients, each retaining the original paid query and full forward batch. Freeze `T=min(J_old,B)` per arm. Validate each selected cost gradient and endpoint gradient along its normalized cost-gradient direction at epsilons `.001` and `.0005`, retaining raw cost quotients and separately checking smooth effort/greed credit plus exact measured detour/ending jumps. The derivative tolerances remain `.0001 + .02*abs(predicted)` and endpoint `.00001 + .02*abs(predicted)`, with `1e-9` decomposition arithmetic and stable clipping/base away from discontinuities. These eight pairs consume sixteen actual complete batches; they do not authorize a derivative through a discrete jump.

For each arm independently, solve Phase I from the same parent at alphas `{.01,.003,.001,.0003}`, radius `.01`, at most 200 SLSQP iterations, and the stated primal/dual certificate settings. Endpoint and ordinary-mean tangents contain zero. Execute every proposed actor completely on all four contexts. Admission requires certification, actual norm at most radius plus `1e-7` float32 tolerance, all four captures, actual mean-cost nonregression, and reduction of maximum positive target violation greater than `1e-6`. Reject uncertified/nonlinear-infeasible candidates; do not add lines or endpoint repairs. Two additional tied-RAW controls at first/last alpha execute the same target table again and must have identical deltas, actions, states, components and returns.

The formulation is admitted for a later learning test only if the credit and tied controls pass and at least one common alpha gives admissible candidates in both arms whose guided-specific maximum target violation is smaller for the guided update by more than `1e-6`. This is measurable teacher-specific update leverage, not a generalization or next-policy-benefit claim. The equally budgeted RAW branch is essential; parameter differences or target attainment alone cannot pass the scientific learning criterion.

Predeclared work per producer: 2,700 scalar controls, 3,500 neural forward rows, sixteen VJPs, ten solver calls, and 26 parameter constructions (sixteen finite-probe constructions and ten candidate assignments). The verifier repeats this work and additionally replays all 200 selected historical teacher controls. Hard per-phase caps are 10,000 controls, 20,000 neural rows, 32 backwards, ten solver calls, 26 constructions and zero Adam calls. Retain every exposed solver callback/kernel and failed intent; SciPy-internal factorization/operation counts remain unobserved and their wall time is charged. Source/config/input bytes freeze before querying. Deterministic replay compares complete query/gradient arrays and full candidate checkpoints, with no third evaluation panel. No new actor gains planning influence or deployment qualification here. A negative admission stops this formulation before full training and remains a negative result.


### V16: nonlinear globalization of the same performance update

V15 is closed and negative. Its certified linear subproblems predict the endpoint essentially at the exact capture boundary; all actual endpoints cross it by `1.31e-7`–`2.50e-7`. The next distinct experiment adds standard feasible-step backtracking, preserving the teacher-performance objective, model, parent, target tables, row derivatives, four alphas, solver, tolerances and actual capture threshold. It does not demand a new fixed ending margin or repair the actor independently of its teacher objective.

V16 executes both fractions `{1, .5}` of every certified proposal in declared order, including both tied controls. Each arm/alpha retains the first fraction passing all original actual gates plus realized-parameter linear safety. The full proposal's optimizer certificate is not called a certificate of optimality for the half step. Both fractions are paid even if the first passes, keeping fixed work across arms. Fresh model executions, rather than a hypothetical margin or mode-frozen cost, decide feasibility. The same predeclared teacher-specific leverage criterion applies. No full training is launched if that criterion fails.

Per producer, the finite schedule is 3,700 controls, 4,500 neural rows, sixteen VJPs, ten QP calls and 36 parameter constructions; verification adds the same work plus 200 historical replay controls. Hard caps remain 10,000 controls, 20,000 neural rows, 32 backwards, ten QP calls and zero Adam calls per phase; the construction cap is explicitly 36. V15's budget is not extended or its result edited. New v16 source/config/inputs freeze before any query, and all outcomes remain separate from fresh standalone-policy or planning qualification.


Reproduction uses the one public command:

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_performance_backtracking_admission_v16.json \
  --source-run runs/2026-10-27_coupled_hybrid_feedback_learning_001 \
  --out-dir runs/performance_backtracking_admission_001
.venv/bin/python -m scripts.training.train_bidirectional_gps report \
  --run-dir runs/performance_backtracking_admission_001 \
  --out-dir runs/performance_backtracking_admission_report_001
```


## Event-aligned outcome replay v17

V15/v16's return-target-only update is stopped. Saved executed trajectories show why a different information channel is worth testing: the selected guided teachers save `.0058853` mean complete cost, chiefly through effort/greed scheduling at controls 4–7. A single position target is insufficient (the current actor already reaches the same peak). The previous twelve-context learner already used teacher states at 0/4/8; coverage was not absent. However several fixed-clock raw/guided contexts coincide, and some are the original actor's own uninterrupted tail. That observation does not make handover states invalid; it motivates a prospective timing intervention.

Keep the exact v14 incoming full parent from v12 and the same four selected complete independent/guided teacher pairs. Change only the two acquired replay clocks per pair. From the guided trace, choose the first recorded changed PREFIX control whose saved post-position is strictly interior, with pre-action clock in `[1, gate-1]`. The prior paid source audit binds `changed` to the independent champion at that same physical state; the interior transition makes this an informative control intervention. This is not a claim to identify the globally first physical difference under clipping. If no such event exists, choose the first actual prefix pre-action state in range and disclose the fallback. Choose the last causally active prefix pre-action state before the **same actual verified handover**, strictly before `gate`. If events coincide, use the nearest distinct actual prefix clock with deterministic earlier-clock tie-break; reject a trace without two valid recorded prefix states. Bind selection to the original live query, controller, commitment and phase records. Do not select with student/evaluation outcomes. On the retained v12 data the expected pairs are `[2,4]`, `[4,7]`, `[1,4]`, `[1,4]`; these are checks of the rule, not hardcoded targets.

Both arms use those same two clock indices for each pair, but retain their own recorded physical states and their own measured teacher suffix-component budgets. With ordinary clock zero, there remain twelve equally weighted contexts per arm. Variable clock groups retain their original full forward batch shapes and paid query ownership. These are new actor execution **starts**, not target states, desired actions, a value function or relabeled returns. The student executes its own policy uninterrupted from each start and learns from its own complete outcomes. Teacher suffix budgets evaluate the recorded teacher continuation. Any raw/guided advantage combines their acquired-state and outcome-budget differences; it cannot be attributed to either alone.

The actor architecture, learning objective, complete-feedback derivatives, hybrid finite-probe qualification, four line steps, at most three measured ending corrections, source-ending and ordinary-cost gates, incumbent qualification every four updates, and one promotion attempt per actor remain v14. A two-sweep interface at offset **40,000,000** must pass paid replay and accept two steps in an arm before the 32-update full comparison at offset **38,000,000**. The interface requires no immediate guided superiority. Full scientific success still requires the qualification-selected guided child to beat both the independent-data child and incoming parent by **more than `.001`** on fresh paired 64-root complete actor-only outcomes with no capture regression and at least 90% capture, plus a separate selected-policy 64-root panel. Sixteen fresh qualification starts alone select incumbents/parent; neither 64-root panel selects or trains a policy.

Caps remain **50,000 controls / 200,000 neural rows / 256 backwards / zero Adam per interface phase**, and **500,000 controls / 1,000,000 neural rows / 4,096 backwards / zero Adam per full phase**. The 32-step cap and finite line/correction grid are not extended after exhaustion. All actual suffix lengths, grouping overhead, rejected lines, corrections, qualification, evaluation and the 200 historical teacher-control replays are charged. Source/config/input bytes freeze before queries; old runs remain unchanged. Changed actors have zero planning authority until fresh controller qualification. A learning pass still requires renewed search/acquisition and another matched child comparison to demonstrate the repeated loop.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_event_replay_interface_v17.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/event_replay_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_event_replay_learning_v17.json \
  --interface-run runs/event_replay_interface_001 \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/event_replay_learning_001
```

Trajectory-optimization-guided policy learning is established prior art ([GPS, 2013](https://proceedings.mlr.press/v28/levine13.html)); policy/teacher coupling through constrained optimization and supervised projection is developed in [MDGPS](https://proceedings.neurips.cc/paper_files/paper/2016/file/a00e5eb0973d24649a4a920fc53d9564-Paper.pdf). State-only imitation also exists ([SMODICE](https://arxiv.org/abs/2202.02433)). Here recorded states are replay starts and the actor is updated from complete self-owned outcomes, rather than trained to match teacher actions or state occupancy. That difference does not establish novelty of the whole algorithm. The tested contribution is the combined protected-search, outcome-coupled teaching, event-aware replay and earned-feedback mechanism; a positive causal result, broad generalization and an efficiency benefit remain empirical questions.


## Bounded feasibility curriculum v18 (prospective)

The retained v17 event replay fails before one accepted update. At the new clock-seven state the incoming actor misses capture by about .015. The one-shot endpoint/ordinary-cost projection makes parameter moves of .48–.61, followed by corrections about 2.0–2.2, and destroys capture. This is an optimization failure, not missing teacher success. One full v17 protocol was mistakenly launched from a software-pass flag despite zero accepted interface updates; that inadmissible run is retained and charged separately. New full event runs require an independently bound, paid-replayed interface with two actual accepted updates, checked before run creation.

Keep the same incoming full v12 actor, executed teacher pairs, event-derived starts, task, objective and architecture. Give each arm **32 total attempted updates across both phases**, not 32 per phase. Phase I minimizes the maximum positive terminal capture residual plus a quadratic step penalty, using all 12 current actor endpoint derivatives. A norm bound **.01** and linear protection of the four ordinary-start endings replace the forced large projection. Execute all four fractions **1,.5,.25,.125** of the proposal, even after an admissible candidate is found. Take the first certified finite step that preserves all four ordinary endings and reduces measured maximum positive residual by more than **1e-6**, or achieves all 12 captures. Ordinary cost may rise during this training-only feasibility phase. The incumbent remains untouched. No relaxed deployment threshold is introduced.

After all 12 source starts capture, Phase II minimizes the normalized complete outcome-envelope gradient with step alphas **.01,.003,.001,.0003**, hard linear ending constraints, norm bound .01, and linear ordinary mean-cost nonregression. Execute all16 alpha/fraction candidates; accept the first with verified original endings, measured objective improvement above **1e-8**, ordinary complete-cost nonregression, bounded actual float32 displacement and a valid solver certificate. No fixed extra terminal margin or corrective projection is added. Each finite candidate grid may terminate an arm; its budget is never enlarged. Pure custom-objective solver fixtures precede neural queries.

Initial four hybrid smooth-credit/endpoint finite probes are retained; failed raw full-cost derivatives are reported, not reclassified. Each later actor also retains current-owned branch qualification: endpoint derivatives require nonboundary clipping; Phase II cost derivatives additionally require clear task-indicator boundaries. Invalid local credit stops before a solver call without relaxing any measured outcome gate. Every later gradient belongs to its current actor's complete uninterrupted execution. Teacher actions never become labels. Qualification every four attempts requires all source starts capture and improved complete outcomes on 16 fresh ordinary starts. Only qualification promotes deployment. Final selection uses16 fresh starts; separate paired 64 and selected-policy 64 panels measure the same predeclared guided-next-policy advantage **>.001**, capture nonregression and 90% minimum. Neither panel selects or trains a checkpoint.

A two-attempt interface uses root/seed offset **44,000,000** and must accept two actual updates in an arm, including possible feasibility steps, plus paid deterministic replay. This is no deployment claim. Full comparison uses offset **42,000,000**. Per producer/verifier caps remain **50,000 controls /200,000 neural rows /256 backwards /zero Adam** for interface and **500,000 controls /1,000,000 neural rows /4,096 backwards /zero Adam** for full. Additional caps are16/256 dual solves and72/1,032 parameter constructions respectively, including eight initial FD constructions. SLSQP cap200, ftol1e-12, feasibility/certificate tolerances1e-8; exposed kernels and wall time are charged, library internals disclosed as unobserved. Verification repeats algorithm work and additionally replays200 historical teacher controls. A curriculum-only pass is not a completed bidirectional-loop result.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_feasibility_curriculum_interface_v18.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/2026-10-27_feasibility_curriculum_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_feasibility_curriculum_learning_v18.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --interface-run runs/2026-10-27_feasibility_curriculum_interface_001 \
  --out-dir runs/2026-10-27_feasibility_curriculum_learning_001
```


## Numerically conditioned feasibility v19 (prospective)

V18 is retained as a failed optimization stage: both arms accept nine feasibility steps and then stop at positive residual9.0294422e-8, with identical frontiers, no cost updates and no promoted child. Its final solver returns zero duals/delta and a falsely reassuring backend-success message, but its original-unit duality-gap/complementarity checks correctly reject the proposal. The achievable dual improvement is below the backend absolute stopping scale. No teacher-learning advantage was tested after that stall.

V19 changes only Phase I numerical conditioning. For positive current maximum ending residual s=max(p), pass the mathematically identical dual objective f/s and Jacobian grad(f)/s to the backend. Bounds, simplex, variables, radius, alpha, physical primal proposal, task success definition and original-unit certificate stay unchanged. Record s and its reciprocal; reject nonfinite conditioning. Phase II and default legacy calls keep scale1. There is no added success margin, relaxed threshold or fabricated outcome label. Every candidate still executes in float32 policy parameters and must satisfy the same actual capture/progress gates; a continuous certified move alone cannot qualify deployment. This does not promise to resolve arbitrarily small numerical residuals.

Before neural calls, the26-request pure roster includes analytic tiny-residual and ordinary-scale conditioned optima, unchanged original-unit certification, invalid scaling cases and a zero-cost float32 realization counterexample. The tiny one-row target9.0294422e-8 with unit gradient,alpha1,radius.1 has exact step minus that target and zero epigraph. The ordinary target.2,alpha.1 has step-.1 and epigraph.1. Backend/certificate tolerances remain unchanged. Two new invalid requests reject nonfinite scaling and nondefault Phase II scaling; total14 finite helper returns and12 expected errors under the existing32-request cap.

The two-attempt interface has disjoint root/seed offset48,000,000; the full32-attempt comparison uses46,000,000. All v18 controls, inference, gradient, solve/construction caps and fresh selection/evaluation criteria remain fixed. An arm must actually accept two interface updates and paid replay must pass before full launch. Source/task/inputs freeze before any queries. V18 expenditure remains charged; no run is extended or overwritten.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_normalized_feasibility_interface_v19.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/2026-10-27_normalized_feasibility_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_normalized_feasibility_learning_v19.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --interface-run runs/2026-10-27_normalized_feasibility_interface_001 \
  --out-dir runs/2026-10-27_normalized_feasibility_learning_001
```


## Repeatable feedback handoff v20/v21 (prospective)

A completed v18/v19 learning stage may feed fresh search only if its ordinary16-start qualification chooses a changed RAW or GUIDED incumbent. Reconstruct every incumbent promotion and final selection from qualification, requiring all12 source endings, capture nonregression,90% minimum and complete-cost gain>.001. Sixty-four-start outcomes and scientific benefit flags never choose the parent. A parent-retained result rejects before source freezing or any model query. Bind final full checkpoint to the selected qualified alias by complete-state equivalence; preserve inherited optimizer, global/private RNG and history. No action labels or planning authority travel with that checkpoint.

The new search inherits the280-fragment memory already committed by v12 **byte for byte**. It does not add those same RAW trials again. Both independent baseline and guided/current/previous-policy conditions use that identical memory. Handoff support retains the selected arm's four actual original teacher episodes, source controllers, original query identities, task and complete suffix budgets. The previous-policy counterfactual is the exact incoming learning actor, not an older convenient reference. Hash-bound copied evidence and a separate input audit enforce these roles.

V20 uses root/seed offset50,000,000 and otherwise the same strong independent search, suffix eligibility/support, agreement ladder.1/1/10, eight trial plus eight confirmation starts, held-out16 executed-controller comparison, acquired-teacher quality and forced revocation gates as v12. Per producer/verifier caps remain6,000,000 controls,2,000,000 neural rows and zero learner updates/backwards. Original positive search, first learning, all failed development and qualification/replay costs remain charged. Search must improve independent complete executed cost with no capture regression, change decisions, acquire four useful complete teacher pairs and beat the incoming-policy counterfactual before it supplies the next paired learning stage. All RAW trials, including unsuccessful ones, feed independent memory once.

V21 uses the same conditioned current-owner learner, task and target definitions on this newly acquired data. Its interface/full root offsets are54,000,000/52,000,000; two interface attempts and32 total full attempts, all caps and fresh qualification/evaluation gates stay v19. No successful-search assumption is manufactured if v20 fails. This supplies the second matched standalone-policy comparison. Each policy version starts at zero planning influence and must qualify again; full-loop benefit requires both directions and the next-policy advantage, not merely a completed handoff.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_feedback_policy_search_v20.json \
  --source-run runs/2026-10-27_normalized_feasibility_learning_001 \
  --out-dir runs/2026-10-27_feedback_policy_search_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_second_feedback_interface_v21.json \
  --source-run runs/2026-10-27_feedback_policy_search_001 \
  --out-dir runs/2026-10-27_second_feedback_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_second_feedback_learning_v21.json \
  --source-run runs/2026-10-27_feedback_policy_search_001 \
  --interface-run runs/2026-10-27_second_feedback_interface_001 \
  --out-dir runs/2026-10-27_second_feedback_learning_001
```

These are executable interface contracts pending qualification of an actual changed child; prospective stages are not reported as completed evidence.


## Measured-descent acceptance v22 (prospective)

V19's eighth proposal has original-unit primal feasibility, bounded norm, finite arithmetic and backend success, and its complete actual executions improve the ending while preserving ordinary capture. Its approximate dual optimality gap9.08e-8 exceeds the preset2e-8 tolerance; v19 therefore rejects it and stops. Retain that numerical negative. Requiring an almost optimal solution of the inner proposal problem is stronger than this learner needs.

V22 keeps the v19 proposal program, conditioning, task, complete actor-owned targets and all measured/fresh qualification gates. Accept a proposal only if the backend succeeds, its finite original-unit primal feasibility/norm checks pass, and the actual float32 candidate passes every existing ending/progress/cost/ordinary safety gate. The complete duality-gap/stationarity certificate remains saved and its **certified=False** status is never relabeled. A feasible suboptimal proposal can generate a measured policy improvement; a zero, infeasible, nonfinite, regressing or failed-backend candidate cannot. This is a change to step acceptance, not a relaxed task ending, invented reward or permission to deploy an unqualified frontier.

The core suite adds seven pure acceptance cases with no additional solver/model calls: allow a feasible nonoptimal certificate as a proposal; reject original-unit primal violation, nonfinite norm, failed backend, missing/malformed checks or an invalid negative violation. A float32 unchanged-step counterexample confirms that proposal admissibility alone cannot imply progress. The26-request solver roster remains unchanged. Source-only peer review and this finite suite precede any neural queries.

Use fresh offsets58,000,000 for the two-attempt interface and56,000,000 for the full32-attempt learner. All v19 per-phase caps, teacher packets, independent data, attempt counts and fresh actor-only success criteria are unchanged and frozen before launch. The interface still must accept two actual updates and replay; full qualification remains all12 source captures plus fresh ordinary capture/cost improvement. All failures and prior costs remain charged. The v20 resolver explicitly accepts a qualified v22 child, and v21 uses the same measured-descent learner for the next matched round.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_measured_descent_interface_v22.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/2026-10-27_measured_descent_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_measured_descent_learning_v22.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --interface-run runs/2026-10-27_measured_descent_interface_001 \
  --out-dir runs/2026-10-27_measured_descent_learning_001
```


## Bounded ending-gradient restoration v23 (prospective)

V22 retains another optimizer failure: its eighth accepted step reduces the residual to1.9641e-5, but the next scaled inner solver returns exactly zero. All fractions fail actual progress, so no child is promoted and cost learning never starts. Removing an optimality requirement alone does not cure a zero proposal.

V23 replaces **only Phase I** with a direct feasible gradient ray. Select the first maximum current measured ending residual p_w and its exact current-owned endpoint derivative g_w. Set d=-g_w/||g_w||. Choose the largest length no greater than .01 whose ordinary-ending linear safety rows satisfy A(length*d)<=b. For each row with positive A_j*d this imposes length<=b_j/(A_j*d); other rows impose no additional length bound because b>=0. A zero gradient or zero permitted length is a disclosed blocked proposal, not inferred evidence of success. The update deliberately seeks a bounded decrease rather than solving exactly to the capture boundary. It adds no target margin or task reward.

Pay for all four fractions and accept the first that passes finite realized parameter norm/linear safety, all four ordinary endings and measured max-ending improvement>1e-6 or all12 exact captures. The helper's geometry record states primal admissibility and **no optimizer-optimality claim**. Actual rollout outcomes remain authoritative. Phase II is exactly v22's complete own-outcome cost learning, including teacher budgets, original ending constraints, finite feasible proposals, actual cost/ordinary nonregression and separate fresh deployment qualification.

Four new pure helper requests test an unrestricted full-radius direction, an ordinary-ending safety cap, a zero-gradient blocked step and nonfinite input rejection. They have their own fixed cap8, three finite returns and one expected error; the26-request SLSQP roster is unchanged. All arithmetic work and partial errors are retained. Source-only review plus this finite core suite precede neural queries.

Interface/full offsets are62,000,000/60,000,000. The same two interface attempts,32 total full attempts, teacher packets, configurations, controls/NN/backwards/construction caps and scientific criteria remain fixed. Additional analytic-ray caps are4/64 per interface/full producer/verifier; they are counted separately from SLSQP calls. Every policy still begins with zero planning authority. V20 may load a genuinely qualified v23 child; the prospective v21 second-round profiles use this same learner. No prior run is extended or modified.

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_feasibility_ray_interface_v23.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --out-dir runs/2026-10-27_feasibility_ray_interface_001
.venv/bin/python -m scripts.training.train_bidirectional_gps develop \
  --config configs/coupled_ending_feasibility_ray_learning_v23.json \
  --source-run runs/2026-10-27_coupled_depth_policy_search_001 \
  --interface-run runs/2026-10-27_feasibility_ray_interface_001 \
  --out-dir runs/2026-10-27_feasibility_ray_learning_001
```
