# Current outcome-learning mechanism

The research goal is a standalone action policy and an independent MPPI teacher that measurably improve one another. This remains an **Ending-task continuation experiment from an already competent theta14 policy**, not learning from random initialization or a new Walker deployment. The latest completed full run, v43, is scientifically negative: RAW accepted six updates and GUIDE nine, but no child ever qualified under the unchanged.001 gate. Theta14 was retained and zero changed-policy rounds ran. Paid replay and the dedicated report pass; the beneficial loop remains unestablished. The earlier v42 intermediate qualification and negative final result remain preserved separately.

Use [the current configuration](../configs/bidirectional_ending.json) and the single public entry point described in [README](../README.md). [Results](results.md) retain versioned outcomes and all costs. The complete preceding formulation history and prior-work discussion are preserved in [before.md](../artifacts/2026-10-28_algorithm_document_consolidation_001/before.md), with [exact-byte hash proof](../artifacts/2026-10-28_algorithm_document_consolidation_001/receipt.json). No novelty or convergence claim is established.

V43 uses a training reserve and an ordinary-return objective with teacher constraints. Define the signed training residual literally as `(y_T-ending_y)+rho`, with `rho=1e-4`; physical capture remains `y_T<=ending_y`. PhaseI reduces the worst positive training residual while preserving previous physical captures and the same-context ordinary reference budget. Once all72 training endings meet the reserve, PhaseII uses the unit ordinary32 complete-cost gradient, with all training-ending rows and eight separate teacher rows `C_i @ delta<=max(B_i-J_i,0)`. Actual admission requires ordinary improvement and each teacher excess not to increase. Mixed64 F is diagnostic, not a descent gate. Proposal independence, fresh qualification and feedback criteria remain unchanged. The bounded component restores the training gap only partially and worsens cost relative to the current actor; its full-run admission grants no policy authority.

This [constraint rule](../src/gps/bidirectional/global_learning.py) has an identification limit: above a teacher budget (`J_i>B_i`), its bound is zero and actual admission requires only that the actor's own cost not worsen; it supplies no attraction toward `B_i`. Below budget, `B_i-J_i` supplies headroom. The reserve rho is task-defined. MPPI therefore contributes acquired states/clocks and conditional performance ceilings, while complete actor-owned outcomes provide long-term credit. A benefit from those acquired contexts would establish that limited link; it would not by itself establish learning the teacher's return quality or completion of the full bidirectional research goal.

## The ownership chain

`independent proposals → executed RAW/GUIDE teachers → current-actor outcomes and credit → complete candidate trials → fresh standalone qualification → changed-policy guidance → matched next children → revocation`

Independent MPPI owns its proposal RNG, nominal plan, carried full plan and trajectory memory. It has no actor access. Neural selection uses copies: restoring the same state and private planner snapshot must reproduce independent proposals regardless of the actor. The strong independent control includes resimulated memory, not just a weak reactive baseline.

A teacher outcome belongs to the controller that actually executed it. The declared prefix/actor program executes its scored prefix and recorded handover; its return is not relabeled as a different replanning controller. Teacher actions are never regression labels. Teacher states supply reset contexts, and complete measured costs supply scalar bounds. Every student cost, endpoint and gradient belongs to that student's own uninterrupted complete execution.

The immutable stage-incoming actor supplies a separately owned ordinary-cost reference. It provides no gradient or relabeled current-actor return. Diagnostic stopping actors and old authority never become full-run parents by implication.

## Acquisition and actor-owned objective

At each stage, execute eight matched independent-data RAW and policy-guided GUIDE teachers. Choose the first seed in each equal position bin of a fixed512-reset pool before observing outcomes. Retain failures and ties. Missing successful complete paired witnesses or actual handovers stop the finite packet.

Each arm starts a child from the same qualified stage-incoming actor and uses the same learner budget. Its acquired resets come from its own actual teacher traces. An update pays for:

| Outcomes | Contexts and ownership |
|---|---|
| Ordinary32 | Fresh ordinary starts, shared training roots between arms; current actor owns all returns |
| Acquired32 | Eight actual teacher states at each pre-action clock2,4,6,8; current actor executes each remaining suffix |
| Auxiliary8 | Original teacher starts; current actor executes complete episodes against eight fixed complete-cost bounds |
| Reference32 | The frozen incoming actor executes the same ordinary32 contexts separately |

Duplicate source slots retain their objective multiplicity and paid work. Exact physical deduplication is used only for endpoint derivatives after complete trajectory arrays agree bitwise.

Let `J(theta,c)` be the original sum of effort, target-distance, detour and ending costs, and `B(a)` the measured complete teacher cost. The preserved v42 objective, retained as a diagnostic in v43, is

`F(theta) = mean_64 J(theta,c)/s_J + beta * mean_8 [max(J(theta,a)-B(a),0)/s_B]^2`.

In that diagnostic F, the auxiliary mean return is not added again. The first64 outcomes have no invented teacher budget. Shared stage-frozen scales are `s_J=mean(all paired teacher costs)` and `s_B=max(RMS(positive incoming-actor deficits against those bounds),1e-6)`; beta is1. Diagnostic auxiliary gradients are deficit-weighted. V43's actual PhaseII proposal instead uses the ordinary32 mean gradient and eight individual, unaveraged teacher-cost rows. Means weight trajectories equally across original query groups, not groups or frames equally.

The mixed-reset F is not ordinary deployment performance. Ordinary cohorts change between attempts: report paired before/after changes on each cohort rather than interpreting different cohort means as learning gains.

## Complete feedback credit

The actor is executed deterministically through the same native single-row numerical path used in forecasts and credit. Before differentiation, bind the actor checkpoint, task, family, complete pre-action states/clock, actions, costs and original query/context descriptors. Ending uses horizon25 and the unchanged terminal threshold `y_T <= -.1`.

Known-model branch-local feedback derivatives carry long-term credit through the current actor's own complete trajectories. Discrete ending/detour costs are measured, not differentiated. The first attempt per arm/stage uses two complete finite-difference actors with realized float32 parameter changes; V43 checks the ordinary32 objective and selected physical endpoint on the declared clipping/discrete branch. Derivative validity does not replace actual candidate execution.

Endpoint packets replay one unchanged baseline query group in its original shape, with the original row selected inside that group. Owned64 groups are budget-free; auxiliary groups retain their actual teacher provenance. Neither candidate-owned outcomes, sliced groups nor the incoming-reference query can supply a baseline endpoint gradient.

## Finite endpoint constraints and proposals

For each distinct baseline physical context, retain physical residual `r_i=y_T-ending_y` and its current-actor derivative `e_i`. V43's constraint allocation and failed/protected classifications use the signed training residual `r_i+rho`; physical capture labels remain unchanged. Context identity includes the full physical state and clock, task and family. Seed the worst row of each of the five owned clock groups and the auxiliary group, break ties by first original slot, then deduplicate physical identities: at most six initial endpoint packets.

Keep at most nine endpoint packets for the whole attempt. After both fractions of each of the first three proposals execute, add at most one unrepresented failed context, ranked by maximum actual pair residual and then first slot. Its derivative still uses the original unchanged baseline group. Do not spend an unused addition after the final pair. This is finite constraint coverage, not a guarantee for unrepresented contexts.

Ordinary safety uses freshly measured same-context costs:

`limit=max(mean J_current,mean J_reference)`, `slack=max(0,mean J_reference-mean J_current)`.

The linear row is `g_ordinary @ delta <= slack`; the actual candidate ordinary mean must be at most `limit+1e-10`. Headroom is not banked across cohorts, and a mean bound is not a per-state guarantee or a promised improvement.

For the current v43 learner, write the training residual as `r_train=r+rho`.

| Phase | Local proposal and complete actual admission |
|---|---|
| I: some training residual is positive | Failed active training rows enter `p,Q`; feasible active training rows enter `A @ delta <= b` with `b_i=-r_train_i`. Minimize `t+||delta||²/(2 alpha)` subject to `p+Q@delta<=t`, `t>=0`, ordinary safety and the radius. Actual admission requires the worst72 positive training residual to fall by more than1e-10, every previous physical capture to survive and the ordinary budget to hold. Teacher-cost regression is permitted; F is diagnostic. |
| II: all72 training residuals are nonpositive | All active ending rows use `p_i=r_train_i,Q_i=e_i` and require `p+Q@delta<=0`. Minimize the unit ordinary32 cost-gradient linear term plus `||delta||²/(2 alpha)`, with ordinary safety, eight individual teacher rows and the radius. Actual admission requires all72 training and physical captures, ordinary cost improvement greater than1e-10 and nonincrease of every teacher excess. F is diagnostic. |

PhaseI divides every failed training residual and derivative by the same positive norm of the original worst baseline gradient; a zero norm uses scale1. This changes proximal units while preserving the training feasible set. Feasible-row margins and the ordinary row stay in original units. PhaseII keeps the training residuals in endpoint-position units.

The completed run exposed an omission in the PhaseI proposal: a represented context with `r<=0<r+rho` has only a soft training row, although actual admission must preserve its physical capture. The separate [contract fixture](../artifacts/2026-10-28_physical_capture_contract_probe_001/result.json) shows that adding `e_i @ delta<=-r_i` rejects all eight known unsafe steps while retaining zero as a feasible hard-constraint point. It does not establish a useful restoring step. At the distinct PhaseII stopping actor, a [one-LP diagnostic](../artifacts/2026-10-28_primal_feasible_direction_probe_001/result.json) finds a direction satisfying the original local constraints after float32 realization. The [paid full/half actor execution](../artifacts/2026-10-28_primal_direction_execution_probe_001/result.json) then improves ordinary cost and all eight teacher excesses with72/72 physical captures, but attains only0/72 and8/72 training-reserve targets. Both fail the unchanged reserve gate. Local constraint feasibility does not establish actual reserve preservation; these subsequent diagnostics neither change active v43 nor grant qualification, authority or a successful learning result.

A separately frozen [one-row normal-repair diagnostic](../artifacts/2026-10-28_normal_ending_repair_probe_001/result.json) uses a fresh endpoint derivative at each fixed trial actor and executes four fixed corrections. All remain ineligible. Half-step κ1 passes every actual outcome gate, but its realized normal prediction is positive while the actual reserve residual is negative; the frozen local certificate therefore rejects it. Full-step κ1 misses the reserve, and both κ2 trials regress teacher excesses. These local-model and measured-outcome results remain distinct. Any future admission change must be versioned and tested through a fresh complete pipeline; this probe grants no qualification or authority and does not change active v43.

The grid is four alphas `[.1,.03,.01,.003]`, each with fractions `[1,.5]`, and parameter radius.01. Each trial executes all64 owned and eight auxiliary continuations. Retain all eight results and choose the first eligible candidate. Zero gradients, failed credit, grid exhaustion and qualification failure are finite negative outcomes; unused work is not reassigned.

## Primal admission at the executed parameters

For each actual candidate, use `delta_real=parameters(candidate).astype(float64)-parameters(baseline).astype(float64)`, including float32 parameter realization. Recompute the same proposal's `p+Q@delta_real`, `A@delta_real-b` and parameter norm at the unchanged1e-8 linear feasibility tolerance. PhaseII uses epigraph0; PhaseI recomputes `t=max(0,max(p+Q@delta_real))` and checks its epigraph residuals.

Require finite arithmetic and successful finite original backend provenance with matching phase, radius and tolerance. The full QP certificate remains diagnostic. No duality, stationarity, complementarity or optimizer optimality certificate transfers to a fraction. Local feasibility does not certify true ending attainment: the actual ending threshold has no numerical capture allowance. The actual norm allowance remains1e-8; complete-outcome gates above remain authoritative.

Save each proposal's immutable matrices, active identities, original gradients, full delta/certificate, each actual delta and generating actor/query owners. Point arithmetic requests, completions, errors, partial work, matvec/norm counts and wall time are charged separately. A helper failure preserves already paid candidate work and grants no admission.

## Fresh qualification and both feedback directions

Training acceptance grants no planning authority. On the scheduled fresh16 ordinary actor-only executions, a child must improve original mean cost by more than.001 while preserving incoming capture and the configured.9 capture floor. Held64 panels do not select checkpoints and evaluate qualified incumbents; unqualified frontiers remain explicitly unqualified checkpoints. Rechecking a stopped frontier is not another learning update or automatic renewed promotion attempt.

After a child qualifies, clear authority and qualify its suffix support and executed guidance afresh. Agreement selects permitted guidance strength; paired actual trials and fresh confirmation must improve control cost without reducing capture. Compare against strong independent memory search and the actual incoming-policy controller on matched contexts, roots and planner state. Guidance trial/confirmation/held root counts are8/8/16. Changed versions, unsupported states and failed requalification revoke influence.

Acquire the next fixed eight-context paired packet only through actual controllers. Train matched next children under the same learner budget. Ingest distinct RAW trial, confirmation and acquisition traces into independent memory after held comparisons; do not append guided choices to private independent memory. Clear authority again after learning and retain exact policy, planner, memory, archive and RNG state.

For **policy → search**, a newly qualified policy must improve actual guidance over both independent search and incoming-policy guidance; bootstrap success with the old parent is insufficient. For **search → policy**, changed executed acquisition must yield a guided-data child that beats both its matched RAW-data child and incoming policy on fresh ordinary executions. A beneficial-loop result requires this complete chain, changed acquisition controls, successful revocation and two actual changed-policy feedback rounds. Teacher-score changes, F descent or an isolated qualified child do not complete it.

## Frozen bounds and current execution scope

| Limit | Current declaration |
|---|---|
| Learning | At most16 attempts per arm/stage, three stages total: initial stage plus two requested feedback rounds |
| One attempt | At most9 endpoint packets,4 QPs,8 complete candidate actors; first-attempt finite differences are separate and paid |
| Hard cap, each producer/replay phase | 15,000,000 scalar controls;6,000,000 neural rows;4,000 reverse passes;0 Adam steps |
| Conservative full roster, per phase | 14,169,307 controls;5,168,536 neural rows;3,648 reverse passes;384 learner QPs |
| Fixed-case preflight, per phase | 15,560 controls; at most26,600 neural rows and38 reverse passes;4 QPs;8 parameter constructions;55 queries;8 point checks |

Paid exact repeat receives its own equal cap. The executor records requested/completed/failed controls, complete arrays, neural work, backwards, solver work, arithmetic, reset events, failures and wall time; planner/RNG state is preserved in its checkpoints. Source, configuration and inputs freeze before queries. Pure core validation blocks neural/physics execution; saved-evidence reports and paid replay have separate costs and scopes.

Only the v43 preflight replays the rejected v42 GUIDE stage0/attempt13 actor and its original210000000 cohort under `inputs/update_origin`, using the distinct qualified stage-parent reference and exact six-query baseline checks. Its useful half-step grants no deployment qualification. Full v43 starts the preserved original theta14 parent, with fresh220000000 root/seed offsets and no supplier-frontier authority. Its copied admission proof checks the component's actual progress and critical learner source before any full-run queries. [Results](results.md) preserve v42's negative final qualification, the failed full-proposal arithmetic handoff and the earlier teacher-constraint diagnostic; none is relabeled as a pass.

The mechanism assumes successful independent witnesses on the declared support, useful policy representability and locally usable known-model derivatives. Finite context coverage and branch-local credit do not establish global generalization or teacher-bound representability. All prior discovery, failed launches, training, replay, reporting and audits remain additional; no efficiency claim follows from omitted costs.

The subsequent [native branch-credit diagnostic](../runs/2026-10-28_native_branch_credit_probe_002/result.json) remains inconclusive: one reversed point preserves branches but misses tolerance; the other crosses the unchanged ending threshold and adds19 cost. V43 does not reinterpret that test as derivative qualification. Its [actual component](../runs/2026-10-28_interior_ending_update_001/result.json) reduces the worst training gap from1e-4 to7.056722641e-5 with72 physical captures, while current ordinary cost increases by.000330261185 within the incoming reference budget. The completed full v43 run failed fresh qualification and retained the original parent; all final16/held64 roles evaluate that same parent, and no changed-policy round ran. All earlier negatives and costs remain additional; no subsequent formulation has been launched.
