# Original Walker runner recovered without new physics

The original runner for the historical 20-trajectory K256 corpus has been
recovered with **the exact SHA-256 recorded by both qualification reports**.
This closes a missing source-identity link. The recovery and static call-path
audit do not execute that runner, replay all 20 trajectories or qualify the
current planner.

The preserved files are
[recovered_original_runner.py](../runs/2026-10-26_walker_actor_credit_diagnostic_001/reused_evidence/recovered_original_runner.py)
and [runner_recovery.json](../runs/2026-10-26_walker_actor_credit_diagnostic_001/reused_evidence/runner_recovery.json).
The latter's SHA-256 is
`51d08cd5c5890f87843d6ae5b20498e21dd465afd0684971ca1238730946c677`.
Both files were created separately; the archived working tree and the earlier
snapshot-compatibility JSON remain unchanged.

## Recovery and byte identity

The archive working-tree runner hashes to
`51d50059b219caa125b8b3a3c61e77feb5c9f286ef2d26de7fd428219d3675c5`, which differs
from the reports' expected runner. Searching the currently available runner
copies and reachable file histories did not find the expected bytes. A
read-only scan of the archive Git object database then found them:

- Archive Git root:
  `/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004`.
- Blob object ID: `71678cb0a1be302da12aa33d8aee79492965829f`; size 16,596 bytes.
- Content SHA-256:
  `b74470397520ebe748332a88fad2912443b5f044e2dc9fe4a0967f5b4f10272f`.
- Retrieval: `git -C /Users/jaisel/Documents/workbench/mppi-gps-archive-20261004 cat-file blob 71678cb0a1be302da12aa33d8aee79492965829f`.

The bounded search hashed 1,049 main-repository blobs and 655 archive blobs,
each between 1,000 and 131,072 bytes. The archive scan read 7,087,189 bytes.
`git log --all --reflog --find-object=...` found no referencing commit. The
exact content was therefore copied into the new evidence directory rather
than relying on continued availability of an otherwise unlocated Git object.

For both `initial2_qualification_a.json` and `_b.json`, all six reported source
identities are now byte-closed: the recovered runner, `Walker2d`, the historical
MuJoCo environment, MPPI, its configuration class and the XML. The historical
task and tuned-controller JSON files also match their current counterparts
byte for byte. The proof records each expected/actual hash and source path.
This source closure supplements the separate
[current CPU snapshot-compatibility audit](walker_historical_snapshot_reuse.md).

## Why the recovered acquisition path is actor independent

The AST audit checks the actual calls and assignments, rather than treating a
matching file hash as sufficient semantic evidence:

| Recovered runner lines | Checked behavior |
|---|---|
| Imports | Only standard library, NumPy, MuJoCo, `Walker2d`, `MPPI` and `MPPIConfig`; no actor, policy, Torch or GPS import |
| 138 | `cfg = MPPIConfig(**json.loads(args.config.read_text()))`; the original input hash binds the frozen tuned configuration |
| 155 | Exactly one `MPPI(env, cfg)` construction, with no additional argument |
| 179–184 | Each episode gets its recorded reset seed and a separate planner seed; `controller.reset(seed=planner_seed)` initializes its own plan/random stream |
| 200–203 | `controller.plan_step(state, initial_warmstart=env.get_warmstart())`; no nominal, first-action replacement, prior cost or coupling supplied |
| 205 | `env.step(action)` executes the returned action; no intervening action assignment or transformation |

The runner has no assignment to a controller attribute. The proof preserves
the exact constructor/planning/execution ASTs and source spans, and checks that
the planning assignment captures `(action, plan_info)` before the execution
call. The intervening statement only records elapsed planning time.

The historical MPPI file itself matches its recorded hash
`fa2c08904c36657c04292082bcdf34cb81985c54c8ee1f7d348d72b2c06e715e`.
Its `plan_step` defaults for `nominal`, `nominal_first`, `prior_cost` and
`coupling` are all `None`. With those defaults, it perturbs its own retained
nominal, computes the normalized cost weights, updates the bounded nominal
with weighted effective perturbations, copies `self.U[0]` as the action and
only then shifts the horizon. The runner executes that copied weighted-mean
action. Available optional policy hooks are not activated by this acquisition
path. The exact historical tuned file supplies K256/H48, temperature 3, noise
sigma .3, the documented lowpass parameters and two initial refinements.

## Scope of the completed link

The recorded source identities and inspected call path support the claim that
this historical corpus was acquired by the independent MPPI configuration.
They are not cryptographic execution attestation and do not newly verify every
recorded state transition. Earlier instrumentation separately reproduced the
reset-30 and reset-31 trajectories; it is not silently extended into a new
20-episode replay claim.

The fixed historical catalog still contains 60 entries at times 0/500/900,
59 active and one retained failed entry, grouped into 10 reset clusters. Its
states are historical interventions, not learner-reached states. Subsequent
learner costs must come from that learner's own complete current-simulator
suffixes. No recovered teacher action or teacher suffix cost becomes an actor
target. Fresh current-planner qualification remains a separate pending gate;
no new actor performance, exploration or efficiency result follows here.

Recovery and proof generation used **zero physics calls, zero simulated or
executed controls, and zero actor updates**.
