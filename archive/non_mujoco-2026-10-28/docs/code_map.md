# Current code path

One entry point: `scripts/training/train_bidirectional_gps.py`, with one active configuration, `configs/bidirectional_ending.json` (v43). Start with `global_learning.global_step` for one policy update, `loop.CoupledSelector` for policy-to-search influence, then `study.produce_temporal_cycle` for their integration. V43 first restores a training ending reserve; its second phase optimizes ordinary return under separate measured teacher constraints. Physical task capture and fresh qualification remain separate.

The current controller scores independent MPPI prefixes using measured model rollouts of the actor's suffix. When qualified, it commits the selected prefix and hands control to that same actor; the saved execution records the handover. The independent reference keeps replanning with its own trajectory memory. Qualification compares these actual complete controllers. An improved hypothetical suffix alone cannot promote guidance.

For the current complete-outcome protocol:

```text
CLI → teaching.run
        → verified qualified input parent + useful actual-component proof
        → study.produce_temporal_cycle
            → fresh bootstrap search/acquisition
            → matched global_learning.global_step
            → complete policy qualification
            → qualify_guidance / acquire_qualified_teachers
            → matched next policies → repeat or retain failure
```

| Read this file | What it owns |
|---|---|
| `src/gps/bidirectional/global_learning.py` | PhaseI training-reserve restoration; PhaseII ordinary32 complete-return objective and eight teacher-excess constraints; baseline-owned endpoint coverage over64+8 actual outcomes, acquired-state provenance and paid component runner. Mixed F is diagnostic. |
| `src/gps/bidirectional/endpoint.py` | Bound endpoint-coverage diagnostic and one actual-update preflight on an explicitly selected stopped actor. The current diagnostic actor remains separate from the qualified parent used for full training. |
| `src/gps/bidirectional/teaching.py` | Finite configurations, scales, input binding and paid repeat. `bind_interior_interface` / `verify_interior_interface` require useful paid component evidence and matching learner source while retaining the qualified input parent. |
| `src/gps/bidirectional/study.py` | Complete cycle orchestration, matched children, executed search qualification and RAW-only memory ingestion. Older protocol profiles remain compatibility code. |
| `src/gps/bidirectional/loop.py` | Strong independent search plus resimulated full trajectory memory; separate guided selection and explicit prefix/actor execution. |
| `src/gps/bidirectional/search.py` | Actor-free MPPI proposal generation, private nominal/RNG and trajectory-memory types. |
| `src/gps/bidirectional/policy.py`, `learning.py` | Family-conditioned action network, checkpoint identity, counted execution, explicit inference contract and shared learning records. |
| `src/gps/bidirectional/credit.py` | Complete actor-owned credit, separate physical/training ending metrics, terminal sensitivities and `realized_primal_certificate` at the actual changed parameters. Derivatives are qualified through measured probes; no teacher action target. |
| `src/gps/bidirectional/coupling.py` | Outcome-envelope arithmetic, independent-prefix scoring and bounded local feasibility/improvement solves. Original full-proposal certificates remain separate from checks of the executed policy displacement. |
| `src/gps/bidirectional/trust.py` | Eligibility, measured executed-controller promotion and revocation. |
| `src/gps/bidirectional/ending.py`, `evidence.py`, `meter.py` | Transitions, one query boundary, complete traces, append-only evidence and cost accounting. |
| `src/gps/bidirectional/reporting.py` | Read-only saved-outcome checks and figures. |
| `src/gps/bidirectional/validation.py`, `tests/` | Bounded software/adjoint/independence/revocation checks. Core validation does not train policies. |

`task.py` and `backend.py` contain the physical Acrobot wrapper and backend interfaces; their presence is not a trained physical-policy result. The original baseline controllers remain in `src/mppi/` and `scripts/runners/`.

`mechanism.py` also owns the bounded global-credit interface used before current training. `performance.py` provides shared solver/admission work accounting, and `sensitivity.py` provides counted parameter constructions. Their earlier experimental runners, plus `curriculum.py`, `hybrid.py` and `execution.py`, remain compatibility paths rather than additional current training recommendations. Detailed historical maps and all former protocol configurations are preserved in the dated archive.

Each run's `source/`, `config.json` and `inputs/` define its executable protocol. `producer/ledger.jsonl` records requested/completed work; `study.json` records scientific decisions; `audit.json` records the separately paid repeat. A report checks saved evidence and creates plots without executing a model. Use the latest completed full run in `docs/results.md` to assess the loop; an isolated `mechanism` pass does not establish the full result.

The saved [physical-capture contract check](../artifacts/2026-10-28_physical_capture_contract_probe_001/result.json) and [feasible-direction LP probe](../artifacts/2026-10-28_primal_feasible_direction_probe_001/result.json) are arithmetic diagnostics, not new training paths or controllers. They respectively reject known bad RAW7 steps and find a locally admissible GUIDE10 direction; neither executes a new actor or changes active v43.

The [separate fixed-direction execution probe](../artifacts/2026-10-28_primal_direction_execution_probe_001/result.json) executes and repeats both saved actors on the original64+8 cohort. Both improve ordinary cost and teacher excesses with72 physical captures, but fail the unchanged training reserve. It supplies no qualified child, controller authority or active-v43 change; its costs and negative result are recorded in [results](results.md).

The [normal-repair diagnostic](../artifacts/2026-10-28_normal_ending_repair_probe_001/result.json) adds two fresh trial-owned endpoint gradients and four fixed corrected actors. All four remain ineligible: half-step κ1 passes actual outcomes but fails its declared local prediction certificate; the other trials fail reserve or teacher constraints. Its paid costs are separate from training, and it grants no authority or active-v43 change.
