# Deployment-aligned adjoint learning: prospective development test

The retained curriculum003 negative is unchanged. Raw diagnosis001 found that
the capturing mean actor reaches the goal around control7.56, while strong
search reaches it at12. Its mean ending is-.346 instead of-.104. Greed cost is
1.274 versus.668, effort.017 versus.110; detour1 and ending0 match. Recorded
stochastic episodes of that exact actor sometimes fail while its deployed mean
captures; late action-invariant times also receive substantial MC advantage
weights. These facts motivate an objective/credit change but do not establish
the cause of optimization regression.

The new global object remains the same bounded128x128 actor. Initialize from
the exact audited, action-label-free curriculum003 round1 parent; keep all
ancestor and unsuccessful development costs. Independent MPPI, memory, task
rewards, actual capture and evaluation criteria remain unchanged.

## New learning formulation

Optimize the deterministic deployment mean's complete ordinary-start running
cost, subject to explicit ending constraints on ordinary and acquired starts.
Acquired-state suffix cost is no longer averaged into the ordinary deployment
objective: those contexts supply ending constraints. No planner actions become
targets. The actor executes every complete suffix itself. Its measured states,
controls, component costs, owner and raw queries supply the linearization.

For Ending, use its qualified analytic state/control Jacobian at each actual
active pre-action state. Re-evaluate the same frozen actor on those states with
a state-gradient bridge. Backpropagate the closed-loop tangent, including later
policy responses, through all remaining active controls. Time>=12 is physically
action-invariant and supplies zero control derivative. The actual binary ending
cost is not assigned a fictitious derivative. An explicit ending constraint
provides additional geometric feedback:

slack = target_y - .001 - measured_final_y.

Use the relaxed log barrier b(s)=-.001*log(s) for s>=.01 and its quadratic C2
extension below. The ordinary objective is measured effort+greed plus b;
acquired contexts contribute.25*b only. This is a declared TRAINING SURROGATE;
actual task cost/capture remain unchanged. It assumes known differentiable
local dynamics; this adapter is not an unqualified Acrobot derivative method.

An Adam proposal uses lr.0003, clipping.5. Try fractions1,.5,.25,.125 in order.
Every trial executes the changed mean controller completely on the same paid
training contexts. Accept only ordinary cost improvement>=1e-6, all ordinary
captures with capture nonregression, and nonincreasing worst acquired ending
violation. Otherwise restore exact actor/optimizer state. Keep all rejected work.
Changed actors receive new own traces before every gradient; no stale actor
return or tangent is labeled as a changed actor measurement. Qualification and
final outcomes are fresh and cannot select fractions. Gradient-based/model-based
policy learning is established prior art, e.g.
[Heess et al.,2015](https://proceedings.neurips.cc/paper/2015/hash/148510031349642de5ca0c544f31b2ef-Abstract.html).
The research claim concerns the full protected-search/actor/data feedback loop,
not novelty of adjoints or a barrier.

## First falsifier before the full loop

Run the accounted tests: the relaxed barrier's C2 join, refusal of foreign or
stochastic trajectories, and an actual125-control neural-actor finite-difference
test. Compare the closed-loop adjoint against central parameter perturbations
of output bias and a state-dependent output weight on five complete simulator
episodes. All125 controls,137 neural rows and seven autograd calls are charged;
no optimizer or native engine is used. Tolerances are .004relative/.003absolute.
Failure blocks training; repairs use a new test packet and preserve failure.

The finite full-loop configuration and exact expenditure are frozen separately
before its launch. It must include the unchanged MC learning control, strong
independent memory/search, paired actually executed teachers, matched children,
two feedback rounds, fresh16-root qualification and64-root final evaluation.
Do not skip matched children because a teacher gate fails. No outcome-directed
reward change, hidden retry, budget enlargement or efficiency claim is allowed.

## Frozen full-loop schedule

Configuration: `configs/bidirectional_deployment_adjoint_dev0.json`. Pin the
curriculum003 round1 parent actor04d2ad6173f5ec8e360b8262b8c3af2faa5678d228249625efd6259c0b9b5cd9,
checkpoint SHA7e4915db2a0f03ed36ce46d9d89a6f06dadc5a41d33007e5bd9df217e22dee26,
and PASS audit SHA750f17769742d1881dad143cf60efeea4ffd35aa9a45b0c32e90cbfebb907e11.
Load its exact actor and private streams, then reset Adam identically for both
initial learning conditions. This is a repair of a competent pure-return actor,
not a fresh-discovery experiment. Every changed version still starts untrusted.

The initial control is the old complete-MC curriculum (32updates, four epochs
each); the treatment uses64single-adjoint proposals. Both start from the same
full common state and use the same eight successful source trajectories. They
do different optimizer/physics work, all charged separately; no efficiency or
single-component attribution follows. Retain best successful eight-root
development phases, then evaluate parent, MC and adjoint on16fresh ordinary
roots. The repair must beat both parent and MC by>=.001 mean complete cost
without capture regression and with>=90%capture. The algorithm continues with
the adjoint actor if it beats the parent; otherwise retain the parent. The MC
arm is an external control and does not silently rescue the algorithm.

For every adjoint step, use four fresh ordinary starts and four actual acquired
t4 states, alternating+/-.025 perturbations, cycling the eight source episodes.
Acquire actual complete current-mean trajectories before differentiation. The
worst acquired violation max(0,y_final+.101) cannot increase after acceptance.
No ordinary cost gain can be purchased by losing training capture. Try only
the four predeclared fractions; rejected gradients and all extra executions
remain charged. At step16/32/48/64 (or8/16/24/32 for children), evaluate eight
development roots. Save every proposal, optimizer/RNG state and rollback.

Each feedback round uses16new uninterrupted t4 suffixes from the actor's own
actual qualification traces, with two perturbations per source. The same
90%ordinary/suffix and radius.15 rules admit a trial, not promotion. Compare
eight actual acquisition and16fresh qualification RAW/hybrid/actor triples.
The selector, independent bank,50prefix/controller forecasts, paid RAW full
replanning preview and explicit prefix4-then-frozen-actor21 execution are
unchanged from curriculum003. Teacher promotion needs>=.001 complete-cost
gain against RAW AND actor alone with capture nonregression and>=90%capture.
Only actual RAW successes update independent memory.

Regardless of teacher success, train matched RAW/guided-data children with
32adjoint proposals each, exact same common parent/optimizer/RNG, ordinary
roots and source-selection rule. Their only input difference is actual executed
state experience. Require the guided-data child to beat parent AND matched RAW
child by>=.001 with capture nonregression. Choose next parent/data using these
complete outcomes, then revoke influence and repeat the whole round. Final64
untouched roots evaluate the last parent, both children, RAW, qualified hybrid
and its guiding actor. They select no settings. Full success requires learning
repair, both teacher AND next-student links, and fresh final confirmation.

New namespaces: baseline4100000, learning4110000, qualification4200000,
feedback4300000/4500000, final4800000--4800063. Curriculum003 and every earlier
negative are unchanged. This one-parent development test is not a three-seed
physical comparison or a standalone-policy Acrobot/Walker result.

Hard caps:3,222,784scalar controls =3,000,816model+221,968actual;464,960NN rows;
320Adam and320training backward attempts;32MC diagnostic backward attempts;
15,360analytic active-transition Jacobian rows. The bound includes192adjoint
steps with ALL four fraction trials rejected, the MC control, all curves,
suffixes, teacher panels, matched children and final evaluations. Rejections
cannot fund more steps or fractions. Errors stop and remain retained.

Qualification fixtures036(25controls before its caught validation error),
037(125controls/137NN rows/seven autograd calls, PASS), and038(pure integration
roster/acceptance checks) remain separate paid work. Freeze source, config,
this protocol and exact source inputs before training. Independently audit
raw transitions, actual acceptances, derivative-surrogate ownership, proposal
lineage, full parent equality, trust and all causal verdicts. Numerical
parameter-gradient re-execution is not implied by a raw arithmetic PASS.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m scripts.diagnostics.run_deployment_adjoint_loop \
  --geometry-run runs/2026-10-27_bidirectional_ending_geometry_001 \
  --parent-run runs/2026-10-27_bidirectional_return_curriculum_003 \
  --config configs/bidirectional_deployment_adjoint_dev0.json \
  --out runs/2026-10-27_bidirectional_deployment_adjoint_001
```
