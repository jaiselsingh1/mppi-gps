# Historical Walker snapshots: static compatibility and availability

This is a **data-availability gate**, completed without constructing a
simulator, stepping physics, training an actor or evaluating a policy. It
does not qualify the current planner or establish current actor performance.
No historical run or phase-0 frozen file was changed.

The machine-readable record is
[historical_snapshot_compatibility.json](../runs/2026-10-26_walker_actor_credit_diagnostic_001/reused_evidence/historical_snapshot_compatibility.json),
SHA-256 `67429117e37d08f333df721bf7cb2ac37b8f36bbd133a47e988f95c08f3ff78e`.
It contains the complete source/package identities, normalized AST hashes,
raw-input hashes, fixed catalog, array indices and state/warmstart byte hashes.
All nine static checks passed. This note supplies their interpretation.

## Raw corpus and fixed catalog

The independent raw-MPPI inputs are preserved in
[/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/runs/walker2d_teacher](/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/runs/walker2d_teacher).
Their matching JSON reports record K=256, H=48, temperature 3, noise sigma .3,
the frozen lowpass settings, two initial refinements and the CPU backend. No
actor supplies their nominal sequence, action labels or continuation.

| Input | Episodes and recorded controls | Raw NPZ SHA-256 |
|---|---|---|
| `initial2_qualification_a.npz` | Resets 30–39; planner seeds 10030–10039; ten complete 1,000-control trajectories | `ba1fefe5623118152032f2209285761a3a13b0d9b0f31b8ca3115a9541be83a2` |
| `initial2_qualification_b.npz` | Same resets; planner seeds 20030–20039; nine complete trajectories and reset 31's fall at control 542 | `7c153af134f555267f3b58aad4bf3398b20a1b0e4fde469475945b776a26b09f` |

The fixed catalog uses **all 20 trajectories** at times **0, 500 and 900**,
with original root horizon 1,000. Resets 30–34 are training roots and 35–39
are heldout roots, including both planner streams. This creates 60 declared
entries: 59 healthy finite state/warmstart pairs and one retained inactive
entry, stream B / reset 31 / time 900. The latter has `active=false`; no healthy
state is fabricated after its actual fall. Its later treatment must preserve
the declared failed/absorbing outcome rather than silently removing it.

Training has 30 entries, 29 active; heldout has 30 entries, all active. There
are **10 reset clusters**, not 20 independent reset replications or 60
independent samples. Planner streams and guide times remain correlated within
a reset. These historical development roots are not a fresh final panel.

For an available entry at guide time b, restore
`states[episode_state_offsets[e] + b]` and the matching `warmstarts` row. The
arrays contain 19-element full-physics states and nine-element solver
warmstarts. Action/cost arrays use separate `episode_step_offsets`. A pure
actor suffix does not require MPPI's old nominal or random stream. Resuming
the original teacher would require them and is a different operation.

## Exact source and AST comparisons

The archived source at
`/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/src/envs/mujoco_env.py`
has SHA-256
`ef5730b35cdcc2cf284e3751538c3b8879850ebfce2ee20f6d266f16f45b3bba`, exactly
the source hash recorded by the original reports. The current
[MuJoCo environment](../src/envs/mujoco_env.py) has SHA-256
`ebc5833d2cf0dd7242e2e4674cc1099dda964cfd4ad8bcc3f59c1c319d61e502`.
The whole files therefore are **not** byte-identical.

Parsing both files with Python `ast`, selecting methods of `MuJoCoEnv`, and
hashing `ast.dump(node, include_attributes=False).encode()` gives identical
ASTs for `reset`, `step`, `get_state`, `set_state`, `get_warmstart`,
`state_qpos` and `state_qvel`. Docstrings are retained; comments and source
locations are absent. Both versions' `step` AST hash, for example, is
`0388abc295f82f459c7e12a4956cb6f3928524e8b675a9cbfb14ead2a7867fb5`.
The JSON records the complete paired hashes for every listed method.

For `batch_rollout`, remove only its top-level `if self._use_warp` dispatch
and hash the remaining statements as `ast.Module(body=..., type_ignores=[])`.
Both CPU-body hashes are
`a59e72e02746ce1f97ae7468c8a010bbd1aba8a50b3dcf61e83fa53d5f1c8e5e`.
The full function ASTs differ because the Warp dispatch changed. The source
diff adds `set_warmstart`; its remaining changes concern Warp initialization,
warmstarts and sensor-free handling. Those changes do not enter the original
recorded CPU path. This is a static code comparison, not a new execution test.

The following current files are byte-identical to their recorded historical
identities:

| File / identity | SHA-256 |
|---|---|
| `src/envs/walker2d.py` | `477c3d2d8735171825392c9fe011941bc26e386f2113b4dd54df8a8c1e03ebca` |
| `assets/walker2d.xml` | `4f33bf5069a0bf65056871ab4579f09c62696e7f46c3af6936efedacc572d590` |
| `configs/walker2d_task.json` / historical `task_vertical05.json` | `546f6902405daaddb37fb9c458d77fb5159234d1218ec73abf9a625ec9f8f27a` |
| `configs/walker2d_best.json` / historical `mppi_initial2.json` | `16435775c846c8037cc17be218cdf1255fa9abb971174ff04f4df363449a3ad6` |

Both original reports record Python 3.11.15, MuJoCo 3.6.0, NumPy 2.4.3 and
SciPy 1.17.1; installed package metadata currently reports the same versions.
Current metadata additionally reports Torch 2.10.0, Gymnasium 1.2.3 and
mujoco-warp 3.7.0.1. These additional packages were not recorded in the original
physics reports and are not retrospectively asserted equal. Metadata lookup
does not initialize or validate their runtime execution.

## Valid reuse and limits

The matching model, task, recorded CPU code and package identities support
using the saved states as historical reset interventions. The catalog requires
**zero new simulated or executed controls**. It does not establish cross-platform
bitwise replay, equivalence of current MPPI proposals, current planner
qualification, learner reachability of these states, or generalization to
ordinary deployment roots. Fresh qualification, if required, remains a
separate result with its own source and cost records.

A later actor must execute **its own complete remaining suffix under the
current frozen task and simulator**, including its own falls and declared
failure absorption. Archived teacher actions and teacher suffix costs are
provenance, not actor action targets or learner-continuation values. No
successful actor checkpoint, weights or search direction appears in this
catalog. Historical acquisition is imported evidence, not free new
exploration; retain its provenance and distinguish its cost attribution from
newly executed learner queries. Future execution, verification and learning
work must be charged separately.
