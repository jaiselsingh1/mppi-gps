# Walker repair: reusable acquisition, learning still unproven

This checkpoint supplements the frozen stage-0 findings. It does not replace
their negative results or describe the code repair as successful walking.
The research goal remains active, with new simulation stopped at the fresh
independent-MPPI qualification gate.

The historical independent K256/H48 qualification already contains exact
physical states and solver warm starts at useful reached states. The new
state-only catalog retains all 60 preselected slots: 59 usable snapshots and
one inactive slot whose ancestor fell before the requested time. The 20
source episodes contain 19 complete survivals and one fall. Roots 30–34 are
development training ancestry; roots 35–39 are held out, including every
planner stream and time from each root. These are ten correlated reset
clusters, not twenty independent task instances or training seeds.

The original runner's bytes were recovered from an otherwise unreferenced
Git object and match both historical reports. Its call path constructs plain
MPPI, retains its own plan and random stream, and executes the returned action
without actor or critic inputs. The task and model match the current task;
static CPU state-transition and restoration checks support state-only reuse.
This closes a source provenance gap. It does not replay every episode or
qualify the current planner. The old compatibility report is preserved, with
the recovery recorded separately.

The imported catalog exposes states, warm starts, time remaining and ancestry.
It supplies no teacher actions or returns as actor-training labels. A future
candidate must produce its own complete measured continuation from a restored
snapshot. Exact restart assumes the saved full physical state and warm start
are sufficient under the declared CPU backend and absence of external forces,
activation or mocap state. Generalization beyond these narrow development
roots remains an empirical question.

The source episodes cost 240,397,398 reconstructed control slots, including
19,542 executed controls, and about 1,213.057 seconds of recorded wall time.
Their separate instrumentation replays cost another 18,974,214 slots; those
replays add no independent episodes. Historical attempt-level accounting is
incomplete. Reuse costs no new simulation, but does not make this earlier
acquisition free. Static arithmetic independently checked all 19,542 saved
cost rows; saved instrumentation arrays agree for 1,542 transitions and 1,544
state boundaries. These checks are not newly simulated transitions.

The repaired scorer requires complete candidate-owned returns for every
ranked candidate. A new durable query journal charges an attempted control
before the simulator call and retains returned transitions, errors and unknown
partial work. A composite acquisition gate additionally binds outcome rows to
archived source, the fully expanded K256 configuration and observed server
runtime metadata. All 78 distinct synthetic checks pass; the last combined
invocation took 0.41 seconds. Neither the scorer nor the journal has yet been
used to demonstrate an actual Walker learning signal. No actor inference,
physical controls, actor updates or critic fits were added in this stage.

At 2026-10-26 22:27:05 UTC, the existing Step31 process and qualification
waiter were both alive. Step31 had durably completed 248 of its 1,454 D2
continuations, up from 210 at 22:25:08 UTC. Worker CPU counters also advance.
The fresh Step32 qualification had not started. Existing completed counters
are lower bounds while work is in flight; no duplicate run was launched.
The source/configuration/runtime portions of the qualification gate pass,
while the outcome gate correctly remains **PENDING**.

Standalone walking, learning transfer to ordinary starts, useful actor
feedback into independent replanning, and exploration efficiency remain
unproven. The next action is to preserve and audit the already queued fresh
qualification when it finishes. If it fails, stop before actor learning. If
it passes, freeze a bounded candidate-owned Walker signal diagnostic before
any new execution. Do not automatically launch a learning campaign.

Evidence is in
`runs/2026-10-26_walker_actor_credit_diagnostic_001/checkpoint_stage1.json`,
the state-only `historical_corpus_001/`, `stage1_manifest.json`,
`reused_evidence/runner_recovery.json`, and `observations/remote_004.json`.
Known per-operation timings are retained; complete development wall time was
not exhaustively instrumented, so no all-project efficiency claim follows.
