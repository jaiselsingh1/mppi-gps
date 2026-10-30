# Code map

There is one public research entry point: `scripts/training/train_bidirectional_gps.py`. Its help lists only implemented commands. Historical orchestrators are in the archived checkout, outside the active Python package and default test collection.

| Location | Responsibility |
|---|---|
| `src/envs/` | Existing dynamics, costs, simulator snapshots and batched rollouts. No learner orchestration. |
| `src/mppi/mppi.py` | Original general-purpose MPPI baseline; optional historical interfaces remain for baseline compatibility. |
| `src/utils/` | Numerical weighting, configs, action/state alignment, replay, gait metrics and statistics. |
| `src/gps/bidirectional/task.py` | Acrobot capture task contract, periodic observations, full snapshots, terminal-once accounting and family classification. |
| `src/gps/bidirectional/backend.py` | Lazy CPU/Warp factory and numerical qualification utilities. Keeping this code does not qualify GPU physics. |
| `src/gps/bidirectional/search.py` | Actor-free proposal bank, private nominal/RNG, executed-trajectory memory, mean verification and separate agreement selector. |
| `src/gps/bidirectional/policy.py` | Bounded family-conditioned actor, distribution and checkpoint identities; not a learning algorithm by itself. |
| `src/gps/bidirectional/trust.py` | Support/version-bound evidence, suffix eligibility, complete executed-controller promotion and revocation records. |
| `src/gps/bidirectional/evidence.py` | Append-only query archive, actual episode traces, measured outcomes and query accounting. |
| `src/gps/bidirectional/experience.py` | Hash-bound reuse of complete teacher traces, with distinct promoted and tested-but-unpromoted evidence modes. Explicit original query namespaces and zero imported trust; only exact-state outcome budgets reach the learner. |
| `src/gps/bidirectional/meter.py` | Actor inference accounting hook and projection-work totals, including failed intents. |
| `src/gps/bidirectional/coupling.py` | Shared outcome discrepancy, teacher-derived budgets, actor-owned score-function update, policy-constrained teacher scores and small constrained-step solvers. No action labels. |
| `src/gps/bidirectional/credit.py` | Gradient primitives read complete owned evidence and perform metered native-shape VJPs. Qualification helpers execute metered changed-actor probes; the closed-form cost/risk projection is arithmetic only. No optimizer updates. |
| `src/gps/bidirectional/hybrid.py` | Current v31 native batch-one cost descent constrained by measured stochastic discrete risk, with fixed variance and no Adam. Also retains v30. Owns frozen inputs, actual qualification, paid repeat and read-only curves; grants no search authority. |
| `src/gps/bidirectional/performance.py` | Single-step teacher-performance admission: original complete-return row gradients, matched measured targets, certified proposals and actual outcome gates. No policy authority or training-benefit claim. |
| `src/gps/bidirectional/curriculum.py` | Bounded ending-feasibility training followed by own-outcome cost learning; separate qualified deployment, paid replay and saved-array curves. V18/v19 numerical failures are retained; v23 uses bounded ending-gradient restoration and measured cost descent with the same outcome and deployment gates. |
| `src/gps/bidirectional/sensitivity.py` | Complete-feedback learning: bounded ending restoration, own-actor repeated cost updates, prospective smooth-mode/jump credit checks, separate training frontiers/qualified incumbents and matched-child comparisons. No planner authority. |
| `src/gps/bidirectional/learning.py` | Neural execution adapter, complete outcome-envelope advantages, clipped policy-gradient updates, owned-data and optimizer checkpoints. |
| `src/gps/bidirectional/loop.py` | Strong independent MPPI with full continuation memory/private plan carry; separate neural selector and revocable prefix/actor execution buffer. |
| `src/gps/bidirectional/study.py` | Finite neural workflow: discovery, exact learner continuation or archived teacher intervention, shared matched-child learning, executed qualification and standalone evaluation. |
| `src/gps/bidirectional/execution.py` | Hash-bound parent/support/comparator loading and raw-only memory derivation; also the frozen-policy prefix/actor execution diagnostic. No actor updates. |
| `src/gps/bidirectional/audit.py` | Paid neural/scalar replay, complete-return ownership, independent proposals, every optimizer update and actual selected/executed action binding. |
| `src/gps/bidirectional/reporting.py` | Read-only outcome/promotion checks, actual learning curves, paired child results, raw/smooth/jump arithmetic and forecast-versus-execution diagnosis. |
| `src/gps/bidirectional/ending.py` | Unchanged cheap task dynamics, scalar query boundary, snapshots and durable accounting. |
| `src/gps/bidirectional/mechanism.py` | Frozen finite structural experiment and paid raw replay. Its linear actor is a diagnostic fixture. |
| `src/gps/bidirectional/validation.py` | Cheap core test roster, engine guard and synthetic fixture-work accounting. |
| `scripts/training/train_bidirectional_gps.py` | Current command routing, status, checks and explicit archive recovery. |
| `scripts/runners/`, `configs/` | Existing baseline controls/settings and explicitly versioned frozen research protocols; each is run through the same public entry point. |
| `tests/` | Core, baseline and new mechanism contracts. The canonical cheap check selects an explicit roster. |
| `archive/research-2026-10-27/` | Complete historical checkout, original documentation, negative studies and old launchers. |
| `runs/`, `artifacts/` | Existing raw evidence and checkpoints, kept in place; new invocations require new directories. |

Read `hybrid.py:RISK_CONSTRAINED` and `produce_risk` for the current finite learner, then its gradient primitives in `credit.py`. `loop.py` handles independent search and actual execution. `sensitivity.py` retains earlier complete-feedback learners; `learning.py` supplies the execution adapter, checkpoint utilities and v30 Adam learner. `mechanism.py` is the earlier structural test. `ending.py` exposes every scalar query. For physical tasks, read `task.py` → `search.py` → `evidence.py`.

The current path is public `develop` → `hybrid.py:produce_risk` → `credit.py:deterministic_deployment_gradient` and `hybrid_feedback_gradient` → `projected_cost_risk_step` → actual candidate executions. Four native batch-one ordinary trajectories supply the deployment-cost gradient. Complete stochastic outcomes supply discrete-only risk credit on twelve contexts; this constraint uses teacher states, without continuous teacher-quality or action targets. All four proposed steps are executed, and the first admissible one is retained. Fresh batch-one qualification remains separate from stochastic training and batched diagnostics. V31 accepts RAW0/GUIDE2 updates, but its guided64-start gain of .00006411 misses .001: no child qualifies and no return search runs. Float32 risk-gate asymmetry prevents treating the update-count contrast as identified teaching leverage. `hybrid.py:report` verifies saved candidate gates, outcomes and selection. The original teaching selector in `coupling.py` remains byte-identical to its frozen binding.

The preserved v30 path uses `HybridOutcomeLearner`: one fresh mixed pathwise/score gradient followed by one Adam step. Its closed negative result is distinct from v31; neither establishes a beneficial complete loop.

The prior v28 workflow is `study.py:JOINT_FEEDBACK` → `preflight_feedback_source` → `curriculum.py:produce` → `curriculum.py:report`. It uses the original v25 qualified teacher packet and exact incoming actor used by v27. `coupling.py` supplies its joint QP and bounded common-descent proposal. Actual complete outcomes and the fixed incoming ordinary-cost anchor determine acceptance. Its failed fresh qualification is retained; training frontiers have no deployment authority.

`study.py:JOINT_CHILD_SEARCH` → `execution.py:_qualified_feedback_ancestor` → `loop.py` → `audit.py` returns a changed, qualification-selected v28 child to fresh search with its exact incoming actor comparator, inherited360-fragment independent RAW memory and zero imported trust. The adapter requires all12 source captures plus fresh ordinary qualification; actor-only improvements cannot bypass this return test.

The earlier v25→v27 workflow is a completed negative comparison: v25 improved over independent search but lost to its incoming actor; v27 admitted two guided-data updates but retained its incoming policy. V18–v23 retain finishing diagnostics and frozen reproduction paths. Historical mechanisms remain explicit configuration profiles, not default training. One public command selects and freezes each finite profile. `docs/results.md` distinguishes executed results from prospective paths.

Memory here stores observed physical states and the actual controls executed afterward. Retrieval proposes recorded continuations and resimulates them from the current state. It supplies search candidates, not policy action labels or a measured return for a different controller. Hypothetical plans and executed traces remain separate.
