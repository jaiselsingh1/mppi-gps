# MPPI–GPS: what exists and what has been demonstrated

**Current focus: one outcome-trained bidirectional loop on real MuJoCo Walker. It has not yet been trained or demonstrated.** The Walker task adapter and actor-update helper have been written; the training orchestrator and physical validation are unfinished. No new Walker policy checkpoint, learning curve or beneficial feedback result exists for this formulation.

I previously described several different prototypes as “the loop.” The following separates them.

| Approach | Learning and coupling | Evidence and status |
|---|---|---|
| Earlier value-guided MPC diagnostics | A learned value/continuation estimate helps planning; these do not produce a standalone action policy by themselves. | Archived diagnostics. They do not establish the requested two-way standalone-policy loop. |
| CIO-GPS on the other branch | Policy-action agreement changes MPPI's optimization scores; the policy regresses to executed/re-solved MPPI actions. | A different implemented algorithm. The pinned read-only comparison records negative earlier Walker runs; it is not a verdict on later branch changes. |
| Our retired Ending experiments | Actor-owned complete outcomes and measured teacher constraints, followed by attempted policy-guided acquisition. | Non-MuJoCo scalar environment. Last full run qualified no changed child and completed zero changed-policy feedback rounds. Retired. |
| MuJoCo Point Mass baseline | Independent MPPI only; no actor learning. | Physics/planner replay passed;2/2 seeded episodes succeeded with matching paid repeat. This verifies the physical baseline, not MPPI–GPS. |
| Current MuJoCo Walker formulation | Independent MPPI discovers states/behavior; a global action policy learns from its own complete returns; qualified agreement selects among a copied independent bank. | Under implementation. No Walker training or feedback benefit has been demonstrated yet. |

Older action-label-trained Walker policies also exist in the preserved history. Some walk; they are references, not checkpoints trained by the current outcome-learning method.

## The one current loop

1. **Discover:** independent MPPI plus resimulated trajectory memory executes real Walker trajectories. Its proposals, RNG, nominal updates and memory scores have no policy input.
2. **Learn:** a bounded neural action policy runs its own complete continuations from ordinary and acquired states. It learns from their measured costs and endings, with scalar teacher-return bounds at the same states. MPPI actions are never regression labels.
3. **Qualify:** independently measure the changed actor and its supported suffixes. Training improvement alone grants no search influence.
4. **Feed back:** only qualified guidance can change selection among copied candidate plans. Paired complete replanning executions must improve outcomes before influence is promoted.
5. **Test the second direction:** acquire guided experience, train matched next actors from independent versus guided data, and compare them alone on fresh starts. Revoke influence when the policy/version/support changes or a qualification fails.

The current critic-free learning recipe uses paired parameter perturbations and complete actor-owned returns. It has no action-matching loss and no learned critic. It is an experimental component, not a proven learning solution or a novelty claim.

The [algorithm page](docs/algorithm.md) is the single current formulation. [Results](docs/results.md) contain measured evidence; [the code map](docs/code_map.md) identifies implemented versus missing pieces. The [other-branch comparison](artifacts/2026-10-29_ciogps_readonly_comparison_001/comparison.md) is pinned and read-only. Nothing was changed on that branch.

## Runnable commands today

```sh
.venv/bin/python -m scripts.training.train_bidirectional_gps status
.venv/bin/python -m scripts.training.train_bidirectional_gps verify-archive
```

The existing `validate` and `qualify` commands run the frozen **Point Mass** baseline. They do not train Walker. A Walker training command will be documented when its orchestrator is implemented and tested.

All active environments use MuJoCo. The [retirement archive](archive/non_mujoco-2026-10-28/ARCHIVE.md) preserves previous source and results. Existing runs remain unchanged; no old policy or toy result is relabeled as current progress.
