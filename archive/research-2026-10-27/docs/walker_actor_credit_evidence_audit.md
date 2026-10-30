# Walker acquisition evidence audit

Read-only audit on 2026-10-26 for the actor-learning goal. No simulator calls,
training, evaluation, remote launches, or changes to existing runs were made.
The new note records what can be checked from local raw evidence and what still
requires remote evidence or a separately charged acquisition.

The independent acquisition controller must load the complete tuned
`configs/walker2d_best.json`: K256, H48, temperature 3, sigma .3, second-order
low-pass noise at 23.4375 Hz with 125 Hz sampling, no importance correction,
and two first-decision refinements. The task is
`configs/walker2d_task.json`: 1.5 m/s target, velocity weight 2, vertical
velocity weight .5, angle weight .1, unhealthy cost 5, control weight .001,
one terminal fall charge 20, frame skip 4 and .008 s per executed control.
MPPI executes its own weighted-mean first action and retains its own plan.
No actor, learned tail, archive fragment or learned scorer belongs in this
acquisition controller.

## Locally available raw evidence

The historical source is
`/Users/jaisel/Documents/workbench/mppi-gps-archive-20261004/runs/walker2d_teacher/`:

- `initial2_qualification_a.npz` and `initial2_qualification_b.npz` contain
  `states`, `warmstarts`, `actions`, `contacts`, `contact_normal_forces`,
  `costs`, `episode_step_offsets`, `episode_state_offsets`, `reset_seeds` and
  `planner_seeds`. Matching JSON files contain the task, MPPI configuration,
  per-episode outcomes, run wall time and source provenance.
- `initial2_protocol.md` fixes nominal resets 30–39, each with planner seeds
  10030–10039 and 20030–20039, 1,000 controls, stopping at first unhealthy
  transition. Its gates are at least 18/20 healthy completions, equal-episode
  speed MAE at most .25 m/s, mean speed in [1.2,1.8], finite traces and bounded
  actions. These resets are now historical development data.
- `initial2_qualification_summary.json` records the original source and
  artifact hashes. It is useful provenance, not a substitute for raw arrays.

This audit loaded both NPZ files with `allow_pickle=False` and recomputed only
saved-state and aggregation arithmetic. Full state is
`[time, qpos(9), qvel(9)]`; post-action forward velocity is column 10,
torso height column 2 and torso angle column 3. Per-episode offsets preserve
the initial state separately from its post-action states.

| Raw-derived quantity | Result |
| --- | ---: |
| Episodes / full healthy 1,000-control completions | 20 / 19 |
| Equal-episode velocity MAE | .1480476042617674 m/s |
| Equal-episode mean forward velocity | 1.4609819118021394 m/s |
| Executed controls, including failure | 19,542 |
| Recorded elapsed time, sum of the two batches | 1,213.05716725 s |
| Maximum saved clock increment error from .008 s | 8.82e-16 s |
| Nonfinite or out-of-bounds traces | 0 |
| Maximum discrepancy from recorded per-episode MAE | 0 |

The retained failure is reset 31 / planner 20031 after 542 controls, with MAE
.18945403883401862. Every failed control remains in the aggregates.
Under the recorded K256/H48 and two initial refinements, inferred sampled
planning capacity is `(19,542 + 20) * 256 * 48 = 240,377,856` control slots.
This is configuration-derived accounting, not a raw per-query planning ledger.
It includes the failed episode; it excludes earlier tuning, rejected
qualifications, rescues and other historical acquisitions.

The current Walker task source and XML hashes match this historical record.
Current `src/envs/mujoco_env.py`, `src/mppi/mppi.py` and
`src/utils/config.py` hashes differ. Therefore 19/20 is evidence of a competent
historical independent teacher, not fresh qualification of the current
checkout/backend or perturbed reset distribution.

## Fresh remote qualification and its artifact contract

The local directory
`/Users/jaisel/Documents/workbench/walker_walking_diagnosis_20261026/diag/`
contains `step31.py`, `step32.py`, `run_validation.sh`, `env.sh` and regression
checks. Its local inventory has no mirrored Step31/Step32 outcomes or raw
traces. Documentation's Step31 K256 result, 61/64 with MAE .151, was not
independently recomputed here from raw local trajectories.

The existing frozen remote project is
`robopil5090:/home/jaisel/mppi-gps-walking-diagnosis-20261026`.
`env.sh` selects the server CUDA venv on CPU and the frozen replanning study
configuration. Step32 imports Step31's raw controller and changes K to 256;
the regression check asserts equality with the tuned configuration file.
The study reset generator adds angle/velocity perturbations to odd reset IDs.

Step32's predeclared repaired arm C uses resets 55000000–55000063 and planner
seeds 885000000+i for i=0..63. Its acceptance rule is at least 58/64 healthy
eight-second episodes and equal-episode MAE at most .30 m/s. The other arms
are A, deployed K32 plus archive; B1, the same selection rule without archive;
and B2, raw K32. These controls diagnose the repair, not actor learning.

`run_validation.sh` waits until `results/step31/finished_utc.txt` exists,
checking for a `run_panel.sh` process every 30 seconds. An empty Step32 log
while Step31 D2 is active is consequently compatible with a waiting process.
If that wrapper disappears without its completion marker, the waiter exits
with `step31 runner ended without finishing`. Remote liveness must be checked
read-only before any duplicate launch. A root-agent SSH snapshot reported
Step31 D2 PID399278 alive and the Step32 results directory/log empty; this
note does not promote that snapshot to a claim of continuing liveness.

Once Step32 actually starts, its expected output directory is
`results/step32/`, containing:

- `runner_sha256.txt` for `diag/step31.py` and `diag/step32.py`;
- `started_utc.txt` before the Python driver;
- `validation.pkl`, an append stream of separately pickled completed rows;
- `finished_utc.txt` only after successful driver completion.

The complete stream has 256 distinct job IDs: C:0..63 and A/B1/B2:0..63.
For raw C and B2, row keys are `job_id`, `reset_profile`, `ledger`, `steps`,
`survived`, `terminated`, `cost`, `velocity_mae` and `seconds`.
Reset/planner seeds and K must be reconstructed from the frozen job recipe;
they are not explicit in those raw-controller outcome rows.

For each completed fresh C row with n executed controls, validate
`ledger.executed_controls == ledger.planner_calls == steps == n` and
`ledger.candidate_controls == 256*48*(n+1)`.
The extra pass comes from the first decision's two refinements. Short
episodes retain the falling transition. Finite discounted outcome cost adds
the remaining absorbing cost; it is distinct from the executed cost prefix.
Sum every row's candidate and executed controls, including failures. Sum
worker `seconds` for work timing; parallel wall time comes from the start/end
timestamps. Do not confuse their ratio with real-time control capability.

`run_jobs` appends and flushes completed futures, then skips those job IDs on
resume. It does not durably record attempted queries before launch. An
exception or interruption can therefore leave failed/in-flight work outside
the committed ledger. Such accounting is a lower bound, and repeated work
must not become free when a job is resumed.

Most importantly, `raw_episode` saves no states, actions, solver warmstarts,
raw costs, MPPI nominal or RNG history. Thus completed Step32 outcomes can
support its declared competence panel, but `validation.pkl` is not an
independently replayable raw acquisition corpus or a source of guide states
for subsequent actor learning.

## K32 default hazards and the minimal missing work

`src/gps/expert_continuation.py:raw_teacher_config` unconditionally replaces
the tuned file's K with 32. Both existing actor drivers,
`train_expert_withdrawal.py` and `train_mppi_experience.py`, default
`teacher_K` to 32 and replace the helper configuration again. Reusing those
helpers without an explicit frozen K256 packet silently defeats the
competent-acquisition premise.

First inspect the existing remote Step32 process and append stream; preserve
it and avoid duplicate qualification. If completed, verify all 64 C outcome
IDs, source/configuration hashes, reset profiles, gates and full work. If it
failed or never started, retain the reason and every available spent cost.

Before an actor learner consumes data, a separately named and charged raw
acquisition must preserve every executed state, solver warmstart, action,
raw cost, termination flag, reset/planner seed, predecision MPPI nominal,
first-pass flag and RNG state, plus source/configuration identity. Include
unsuccessful acquisitions. Exact regeneration on the original machine may
recover omitted traces but is additional verification work, not another
independent observation. The existing current `run_walker2d.py` supports many
of these raw fields, but its defaults are only five 500-control episodes and
it lacks a durable per-attempt failure ledger; neither default is an
eight-second qualification protocol.

The smallest outstanding Gate1 action is consequently evidence recovery or
the existing preregistered qualification, followed by a raw acquisition
corpus under the actually qualified controller. It is not a critic change,
new teacher tuning sweep or actor training launch. Nothing in this note
establishes that a global learner can absorb or outperform the teacher.
